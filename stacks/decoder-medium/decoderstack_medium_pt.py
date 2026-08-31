# decoderstack_medium_pt.py
#
# nanochat-based pre-training pipeline, with model code implemented as a single
# forward_backward function, with no nn.Module or autograd.
#
# Style:
# - Helpers and classes are kept to a minimum, trading reuse for fewer
#   redirects and abstractions.
# - The config and model objects are globals, eliminating the need for argument
#   passing (which assigns multiple names and requires documentation).
#
# Design:
# - Model tensors are "banked" (they include a layer dimension), and their
#   optimizer state is directly attached.
# - Tensors are created directly on the device, with explicit dtypes. dtypes
#   are never inferred by matching another tensor's dtype.
#
# Grep for "§" to retrieve the document outline.

# ==============================================================================
# § Configuration
# ==============================================================================
# Imports, the distributed init, the run knobs, and every model
# and training hyperparameter.

# ------------------------------------------------------------------------------
# § Setup
# ------------------------------------------------------------------------------

# ======== Imports ========

import os
import sys
import time as _time
run_wall_t0 = _time.perf_counter()
del _time

with open(sys.argv[0], 'r') as f:
    code = f.read()   # the run section logs the script source to wandb

import datetime
import gc
import glob
import json
import math
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

dynamo.config.recompile_limit = 64

# ======== Distributed Setup ========
rank = int(os.environ["RANK"])
master_process = (rank == 0)
world_size = int(os.environ["WORLD_SIZE"])

assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)

props = torch.cuda.get_device_properties(device)
assert props.major >= 8, f"needs Ampere or newer (got sm{props.major}{props.minor}); FlashAttention-3 has no Turing build"

dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()


# ------------------------------------------------------------------------------
# § FlashAttention
# ------------------------------------------------------------------------------

from kernels import get_kernel

_cc_major, _ = torch.cuda.get_device_capability()
if _cc_major == 9:   # Hopper: the varunneal build gets better H100 results
    fa3 = get_kernel("varunneal/flash-attention-3").flash_attn_interface
    RAW_BWD_TAKES_BUFFERS = False   # raw backward allocates and RETURNS dq/dk/dv
else:                # Ampere sm80/86 / Ada sm89: community FA3 build
    assert _cc_major == 8, f"FA3 required (sm8x or sm90); got sm{_cc_major}x"
    _k = get_kernel("kernels-community/flash-attn3")
    # The raw ops live in flash_attn_interface; the top level only re-exports
    # the varlen/kvcache wrappers.
    fa3 = getattr(_k, "flash_attn_interface", _k)
    RAW_BWD_TAKES_BUFFERS = True    # raw backward takes pre-allocated dq/dk/dv buffers


def flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, max_seqlen, window_size):
    """Attention forward that also returns the softmax LSE (H, T) fp32."""
    out, softmax_lse, *_ = fa3._flash_attn_forward(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=q.shape[-1] ** -0.5, causal=True,
        window_size_left=window_size[0], window_size_right=window_size[1])
    return out, softmax_lse


def flash_attn_varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen, window_size):
    """Attention backward for flash_attn_varlen_fwd_lse: returns (dq, dk, dv).
    The two FA3 builds' raw backward ops differ in calling convention -- the
    sm80 community build's schema takes pre-allocated dq/dk/dv buffers (grads
    come back through them), the sm90 varunneal build's allocates and returns
    them -- hence the branch on the module-level flag."""
    softmax_scale = q.shape[-1] ** -0.5
    if RAW_BWD_TAKES_BUFFERS:
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        fa3._flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            cu_seqlens, cu_seqlens,     # cu_seqlens_q, cu_seqlens_k
            None, None,                 # seqused_q, seqused_k
            max_seqlen, max_seqlen,
            dq, dk, dv,
            softmax_scale,
            True,                       # is_causal
            window_size[0], window_size[1],
            0.0,                        # softcap
            False,                      # deterministic
            0,                          # sm_margin
        )
    else:
        dq, dk, dv, _ = fa3._flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            cu_seqlens, cu_seqlens,     # cu_seqlens_q, cu_seqlens_k
            None, None,                 # seqused_q, seqused_k
            max_seqlen, max_seqlen,
            softmax_scale,
            True,                       # is_causal
            window_size[0], window_size[1],
            0.0,                        # softcap
            False,                      # deterministic
            0,                          # sm_margin
        )
    return dq, dk, dv

# ------------------------------------------------------------------------------
# § Model Config
# ------------------------------------------------------------------------------

# Set this to run training under the normal schedule and have it abort
# part way. Great way to test things out without changing num_steps.
ABORT_STEP = None
#ABORT_STEP = 250

# Torch profiler: trace steps 12-13 (post-compile, post step-10 log hooks),
# then abort at 14. Rank 0 writes logs/<run_stamp>_<run_name>_trace.json.gz -- view at
# ui.perfetto.dev.
PROFILE = False

class StackConfig:

    # ---- Architecture ----

    # Model
    n_layers:   int = 24
    d_model:    int = 1536

    backout_layer: int = 12 # nanochat: n_layers // 2

    # Input
    d_vocab:    int = 32768
    d_smr_gate: int = 24    # Gate input is first 24-dims of input embed.

    # Attention
    n_qo_heads: int = 12
    n_kv_heads: int = 12    # n_qo == n_kv means full multihead attention.
    d_qk:       int = 128   # Attention head size.
    d_vo:       int = 128   # Note: FA2 requires d_qk == d_vo, FA3 does not.

    # Context and Sliding Window Attention
    seq_len:          int = 2048
    short_win_size:   int = 768
    full_ctxt_layers: list[int] = [   3,    7,    11,    15,    19,    23] # "SSSL" tiled, last layer always full
    window_sizes:     list[tuple[int, int]]  # Derived below.

    # Attention - Value Embeddings
    d_ve_gate: int = 12  # Gate input is first 12-dims of the layer's residual stream.
                         # Each head has its own gate, all with same input.
    ve_layers: list[int] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    ve_index:  list[int] # Derived from ve_layers.
    num_ves:   int

    # MLP
    d_mlp:      int = 4 * 1536 # 6144

    # Model stats, for MFU and logging
    num_params:          int = 1_384_122_122   # every trained weight (§ Weight Init & Schedule)
    num_flops_per_token: int = 4_860_160_128   # 6 * 729,810,624 matmul params + attention

    # ---- Training ----

    # Batch Size
    micro_batch_tokens: int = 65536   # 64K tokens per rank, per micro-batch
    total_batch_size:   int = 2**20   # 1M tokens per step -- nanochat's Power
                                      # Lines auto-compute for d24.
    world_size:         int           # From torchrun; the rest are derived below.
    grad_accum_steps:   int
    max_num_docs:       int           # Entries in the fixed-size cu_seqlens buffer.

    # Training
    num_steps: int = 5568

    # Evaluation and logging
    val_loss_every:  int = 250
    val_tokens:      int = 10485760   # 10M tokens per val-bpb pass
    eval_buffer_tokens: int = 65536   # tokens per rank per eval micro-batch, val AND CORE
    val_steps:       int              # Derived: val micro-batches per rank per pass.

    # Logging
    wandb_project:   str = "decoderstack"
    run_name:        str = "80GB-A100_d24_baseline"  # names the wandb run
    use_wandb:       bool = True
    run_stamp:       str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                                    # prepended to saved files so a rerun
                                    # under the same run_name can't clobber
                                    # the previous one's logs/checkpoints

    save_checkpoint: bool = True
    save_steps:      tuple = (1950,) # Last resumable checkpoint before warmdown.

    # CORE
    core_eval:       bool = True    # post-training CORE metric

    seed:            int  = 42      # For model initialization

cfg = StackConfig() # Make config a global, don't pass it around.

# Sanity: the constraints the axes above must satisfy.
assert cfg.d_vocab % 64 == 0, "vocab must arrive padded to 64 (no auto-padding here)"
assert cfg.n_qo_heads % cfg.n_kv_heads == 0, "GQA needs query heads to tile over kv heads"
assert cfg.d_qk % 2 == 0, "rotary splits the qk head dim in half"
assert cfg.full_ctxt_layers[-1] == cfg.n_layers - 1, "final layer recommended to have full context"

assert cfg.total_batch_size % (cfg.micro_batch_tokens * world_size) == 0, \
    "total batch must divide evenly into per-rank micro-batches"
assert cfg.val_tokens % (cfg.eval_buffer_tokens * world_size) == 0, \
    "val_tokens must divide evenly into per-rank eval buffers"

# Map layers to VE bank slots.
cfg.ve_index = [cfg.ve_layers.index(i) if i in cfg.ve_layers else -1 for i in range(cfg.n_layers)]
cfg.num_ves = len(cfg.ve_layers)

# Per-layer window sizes for sliding window attention, defined as (left, right)
# tuples. Left means number of tokens to attend to to the left of current
# position, and right is 0 for causal.
cfg.window_sizes = [(cfg.short_win_size, 0)] * cfg.n_layers  # All short, ...
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (cfg.seq_len, 0)                   # ... then overwrite with full.

cfg.world_size = world_size   # from torchrun, stored so the logged config carries it

cfg.grad_accum_steps = cfg.total_batch_size // (cfg.micro_batch_tokens * world_size)
cfg.val_steps =        cfg.val_tokens       // (cfg.eval_buffer_tokens * world_size)

# This is to set the fixed size of 'cu_seqlens' for varlen.
# Estimating 192 docs per 64K tokens.
cfg.max_num_docs = 192 * max(1, math.ceil(max(cfg.micro_batch_tokens, cfg.eval_buffer_tokens) / 65536))

gpu_device_name = torch.cuda.get_device_name(0)   # e.g. "NVIDIA H100 80GB HBM3"
# BF16 peak FLOPS for the MFU denominator (an unrecognized GPU reads mfu = 0).
gpu_peak_flops = (989e12 if any(x in gpu_device_name for x in ("H100", "GH200")) else
                  312e12 if "A100" in gpu_device_name else float("inf"))

DATASET_DIR = os.path.join(os.environ.get("DATA_PATH", "."), "data/climbmix_32k_8_170")
train_files = os.path.join(DATASET_DIR, "climbmix/train_*.bin")
val_files   = os.path.join(DATASET_DIR, "climbmix/val_*.bin")

# How many of the hub's 91 train shards (100M raw tokens each, numbered from
# 1) this horizon needs. Count against 85M usable per shard and round up.
num_train_shards = math.ceil(cfg.num_steps * cfg.total_batch_size / (0.85 * 100_000_000))

if master_process:
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"=== Downloading dataset files ===")
    for fname in HfApi().list_repo_files(repo_id="ChrisMcCormick/climbmix_32k_8_170", repo_type="dataset"):
        # Skip over excess training shards.
        if fname.startswith("climbmix/train_") and int(fname[len("climbmix/train_"):].split(".")[0]) > num_train_shards:
            continue
        # Download everything else.
        if not os.path.exists(os.path.join(DATASET_DIR, fname)):
            hf_hub_download(repo_id="ChrisMcCormick/climbmix_32k_8_170", filename=fname,
                            repo_type="dataset", local_dir=DATASET_DIR)
    print("  Done.")

dist.barrier()

# Read the BOS id the .bin shards were packed with; reject a vocab mismatch.
with open(os.path.join(DATASET_DIR, "config.json")) as f:
    _vocab_config = json.load(f)
BOS_ID = _vocab_config["bos_id"]
assert _vocab_config["vocab_size"] == cfg.d_vocab, "dataset vocab != model d_vocab"

# token_bytes: per-token-id byte lengths (0 for special tokens), for the
# vocab-size-independent bits-per-byte validation metric.
with open(os.path.join(DATASET_DIR, "tokenizer/token_bytes.pt"), "rb") as f:
    token_bytes = torch.load(f, map_location=device)


# ==============================================================================
# § The Math
# ==============================================================================


# ------------------------------------------------------------------------------
# § Data Structures
# ------------------------------------------------------------------------------

# NamedTuples can be passed to compiled functions.
class Param(NamedTuple):
    """Model parameter bundled with everything needed for training it."""

    name:         str
    w:            Tensor    # The actual weight

    # Optimizer State
    mantissa:     Tensor    # Larry Dial's trick for storing an fp32 master
    grad:         Tensor    # Matches full weight size
    gbank:        list      # Banked weights unbound into a list
    shard_size:   slice     # Shape of one GPU's share of optimizer work
    first_mntm:   Tensor    # shard_size, not full
    scnd_mntm:    Tensor
    residual_dim: int       # NorMuon only, dim that touches the res stream.

    # Schedules (Per-Step Coefficients)
    lr_bc_t:      Tensor    # bias-corrected learning rate
    wd_t:         Tensor    # weight decay * non-corrected lr; AdamW stores 1 - that
    mntm_b1_t:    Tensor    # Beta1
    grad_b1_t:    Tensor    # 1 - Beta1
    mntm_b2_t:    Tensor    # Beta2
    grad_b2_t:    Tensor    # 1 - Beta2
    eps_t:        Tensor    # AdamW only

class Model:
    """Container for the model's weights, plus RoPE buffers"""

    # Input
    input_embeds: Param = None
    smear_gate:   Param
    smear_lambda: Param

    # Attention
    W_Q: Param
    W_K: Param
    W_V: Param
    W_O: Param
    value_embeds: Param
    ve_gate:      Param

    # MLP
    W_in:  Param
    W_out: Param

    # Cross-Layer
    x0_lambdas:     Param   # Per-layer coefficient for reading the input embedding.
    resid_lambdas:  Param   # Per-layer gain on the residual stream.
    backout_lambda: Param

    # Output
    lm_head: Param

    # Rotary Cache
    cos: Tensor
    sin: Tensor

    # `for p in m` yields every trained weight, in config-table order.
    def __iter__(self):
        return (v for v in vars(self).values() if isinstance(v, Param))

# NamedTuples are torch.compile-friendly.
class LayerStash(NamedTuple):
    """One layer's forward activations, held for the backward pass.
    Commented-out rows are what we recompute rather than hold.
    For T=64K  -->  Held: 49.7GB,  Recomputed: 24.8GB
    """
    #                                                    Stash (Tiny) Recompute
    x_in:       Tensor    # (L,  T,    D)                4.5GB
    xb_norm:    Tensor    # (L,  T,    D)                4.5GB
    xb_inv_rms: Tensor    # (L,  T,    1)         fp32          (6MB)
    q_norm:     Tensor    # (L,  T, n_qo, d_qk)          4.5GB
    k_norm:     Tensor    # (L,  T, n_kv, d_qk)          4.5GB
    q_inv_rms:  Tensor    # (L,  T, n_qo,    1)   fp32         (72MB)
    k_inv_rms:  Tensor    # (L,  T, n_kv,    1)   fp32         (72MB)
    #ve:        Tensor    # (Lv, T, n_kv, d_vo)                         2.25GB
    #ve_gate_a: Tensor    # (Lv, T, n_kv)                                (18MB)
    v:          Tensor    # (L,  T, n_kv, d_vo)          4.5GB
    y:          Tensor    # (L,  T, n_qo, d_vo)          4.5GB
    lse:        Tensor    # (L,  n_qo,  T)        fp32         (72MB)
    xm:         Tensor    # (L,  T,     D)               4.5GB
    #xm_norm:   Tensor    # (L,  T,     D)                               4.5GB
    mlp_za:     Tensor    # (L,  T, d_mlp)              18.0GB
    #mlp_a:     Tensor    # (L,  T, d_mlp)                              18.0GB
    #                                                   ------          ------
    #                                           TOTAL:  49.7GB          24.8GB

# Model-level activations held as locals:
#   x0         (T, D)         192MB    layer-blend + smear backward
#   xe_norm    (T, D)         192MB    smear + embedding-norm backward
#   xe_inv_rms (T, 1)   fp32   (0.3MB)
#   x_backout  (T, D)         192MB    backout backward
#   xf_norm    (T, D)         192MB    lm_head grad + final-norm backward
#   xf_inv_rms (T, 1)   fp32   (0.3MB)

# Cast shorthands for the bodies below: the fp32 scalars/gates need explicit
# bf16 casts at their use sites (see forward_backward's docstring), and the
# scalar-parameter grad sums accumulate in fp32.
bf16  = lambda x: x.to(torch.bfloat16)
sum32 = lambda x: x.sum(dtype=torch.float32)


# ------------------------------------------------------------------------------
# § Train Fwd+Bwd
# ------------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def forward_backward(idx, targets, cu_seqlens, loss_scale):
    """One micro-batch: forward, stash, explicit backward into `.grad`.

    Returns the detached mean CE loss (unscaled; grads carry loss_scale).

    Activations are bf16 throughout. The live weights are already bf16, so no
    per-use casts; the fp32 scalars need care: indexing a 1-D fp32 bank gives a
    0-dim tensor, which does NOT promote a bf16 tensor (resid/x0 lambdas ride
    as-is), but the (1,)-shaped smear/backout scalars and the smear_gate matrix
    WOULD promote to fp32, so those are cast explicitly."""

    # Residual stream naming:
    # xe - The input 'e'mbedding
    # xb - Layer input, 'b'iased by resid_lambda and x0
    # xm - Post-attention stream, the 'm'lp's input
    # xf - 'f'inal stream, post-backout, feeding the lm_head
    #
    # RMS naming:
    # *_inv_rms  - 1/rms, used by RMS norm fwd and bwd.
    # *_norm     - RMS-normed


    assert idx.ndim == 1
    T = idx.size(0)
    half = cfg.d_qk // 2

    assert T > 1, "Training forward pass should have T > 1"
    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"

    cos, sin = m.cos[0, :T], m.sin[0, :T]  # (T, 1, half)
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)
    x_backout = None

    # -----------------------------
    #           Forward
    # -----------------------------

    # Input embeddings
    xe = F.embedding(idx, m.input_embeds.w)        # bf16

    # RMS normalize the input embeds
    xe_inv_rms = (xe.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xe_norm = bf16(xe.float() * xe_inv_rms)      # post-norm embedding, pre-smear

    # Smear: mix the previous token's embedding into the current position.
    # The sigmoid's argument is unnamed here; it appears as gate_logit_grad in bwd.
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(
        xe_norm[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)
    x_out = torch.cat([xe_norm[:1], xe_norm[1:] + gate * xe_norm[:-1]], dim=0)  # appears as smeared_grad in bwd

    # Smeared input, added back to the stream at every layer.
    x0 = x_out

    # One LayerStash of forward activations per layer
    stash = []

    # For each layer,
    for i in range(cfg.n_layers):
        x_in = x_out

        # Scale residual stream, add input embedding
        xb = m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
        xb_inv_rms = (xb.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xb_norm = bf16(xb.float() * xb_inv_rms)

        # QKV Projections
        q = (xb_norm @ m.W_Q.w[i].mT).view(T, cfg.n_qo_heads, cfg.d_qk)
        k = (xb_norm @ m.W_K.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (xb_norm @ m.W_V.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)

        # Value Embeddings
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_za = torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            ve_gate_a = 3 * ve_gate_za           # (T, n_kv_heads), in [0, 3]
            v = v + ve_gate_a.unsqueeze(-1) * ve # ve and the gate are both recomputed in bwd, not stashed

        # RoPE
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)

        # QK-Norm and Sharpening
        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q_norm = bf16(q.float() * q_inv_rms * 1.2) # 1.2^2 - Similar to temperature of 0.7
        k_norm = bf16(k.float() * k_inv_rms * 1.2)

        # Read V from past residual streams by matching their K.
        y, lse = flash_attn_varlen_fwd_lse(q_norm, k_norm, v, cu_seqlens, cfg.seq_len, cfg.window_sizes[i])
        y = y.contiguous()

        # Project value heads onto their output heads.
        attn_out = y.view(T, -1) @ m.W_O.w[i].mT

        # Write back to the stream.
        xm = xb + attn_out

        # MLP input norm
        # Recomputed in backward pass to save memory.
        xm_norm = bf16(xm.float() * (xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())

        # MLP
        mlp_za = F.relu(xm_norm @ m.W_in.w[i].mT)
        mlp_a = mlp_za.square()    # (T, d_mlp) - (64K, 6K) - 0.75GB, recompute rather than stash
        mlp_out = mlp_a @ m.W_out.w[i].mT

        # Write back to the stream.
        x_out = xm + mlp_out                     # the residual stream; appears as stream_grad in bwd

        # Stash the backout layer's output, to subtract it off before LM head.
        if i == cfg.backout_layer:
            x_backout = x_out

        # Stash activations for backward pass.
        stash.append(LayerStash(x_in=x_in, xb_norm=xb_norm, xb_inv_rms=xb_inv_rms,
                                q_norm=q_norm, k_norm=k_norm, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
                                v=v, y=y, lse=lse, xm=xm, mlp_za=mlp_za))

    xf = x_out - bf16(m.backout_lambda.w) * x_backout # TODO - backout lambda could be bf16 instead.

    # Final output norm
    xf_inv_rms = (xf.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xf_norm = bf16(xf.float() * xf_inv_rms)

    # -----------------------------
    #           LM Head
    # -----------------------------
    tgt = targets.unsqueeze(1)  # (T, 1)  TODO - Have the caller pass the right shape.

    # ==== Forward ====
    logits = xf_norm @ m.lm_head.w.mT  # (T, d_vocab) bf16, 4GB at (64K, 32K)

    # "softcap" logits to the range -15 to 15
    logits = 15.0 * torch.tanh(logits.float() / 15.0) # (T, d_vocab)

    # Typically, subtract off the highest logit per token first:
    #    max_logit = logits.amax(dim=1, keepdim=True) # (T, 1)
    #    e = (logits - max_logit).exp()  # (T, d_vocab) = ((T, d_vocab) - (T, 1)).exp()
    # Softcap bounds to [-15, 15], so exp() is [3.06e-7, 3.3e6], so fp32 is ok.
    e = logits.exp()  # (T, d_vocab)

    # Softmax denominator
    ssum = e.sum(dim=1, keepdim=True) # (T, 1)

    # ==== CE Loss for Logging ====
    # Just to track training loss--costs a few seconds over the run.
    # Convert back to logit space
    lse_ce = ssum.log().squeeze(1) # (T, 1)
    # Select prediction logits for target token (one random access per row)
    tgt_logit = logits.gather(1, tgt).squeeze(1) # (T,)
    # Average delta
    loss = (lse_ce - tgt_logit).mean()

    # ==== Backward ====
    onehot = torch.arange(cfg.d_vocab, device=device).unsqueeze(0) == tgt # (T, d_vocab)

    # Predicted probs = e / ssum; (T, d_vocab)
    logits_grad = bf16((e / ssum - onehot.float()) * (1.0 - logits/15.0 * logits/15.0) * loss_scale)

    # Every token updates every vocab entry.
    m.lm_head.grad.add_((logits_grad.mT @ xf_norm).float()) # (d_vocab, T) @ (T, d_model) --> (d_vocab, d_model)

    # For each residual stream, take a weighted sum of the lm head rows it
    # (meaningfully) predicted, and that's its gradient for backward.
    xf_norm_grad = logits_grad @ m.lm_head.w # (T, d_vocab) @ (d_vocab, d_model)
    del logits_grad

    # -----------------------------
    #           Backward
    # -----------------------------

    # Scalar grads are collected, grad tensors updated at the end.
    g_resid = []; g_x0 = []

    # Each stream's cosine similarity to the vocab signal, divided by d_model.
    # (T, 1) = ((T, d_model) * (T, d_model)).mean()
    res_ms = (xf_norm.float() * xf_norm_grad.float()).mean(dim=-1, keepdim=True)

    # The component of xf_norm_grad which is perpendicular to xf_norm?
    xf_grad = bf16(xf_inv_rms * (xf_norm_grad.float() - (xf_norm.float() * res_ms)))

    # The similarity between the backout layer's stream and the predicted tokens,
    # i.e., did adding the backout layer make the final stream look more or less
    # like the right and wrong tokens.
    m.backout_lambda.grad.add_(-sum32(xf_grad * x_backout))  # (T, d_model)

    # stream_grad updates every layer, keep xf_grad for backout layer.
    stream_grad = xf_grad           # grad wrt layer num_layers-1's output
    x0_grad = torch.zeros_like(x0)  # accumulates over layers

    # For each layer in reverse order,
    for i in reversed(range(cfg.n_layers)):

        # Stashed forward activations for this layer.
        st = stash[i]

        if i == cfg.backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes num_layers//2
            stream_grad = stream_grad - bf16(m.backout_lambda.w) * xf_grad

        # --- MLP backward (relu^2: dh = 2*a*du, self-masking since a = relu(h)) ---

        # Grad w.r.t. W_out
        mlp_a = st.mlp_za.square()  # (T, d_mlp) - 768MB at (64K, 6K)
        m.W_out.gbank[i].add_(stream_grad.mT @ mlp_a) # (d_model, T) @ (T, d_mlp) - Same flops as fwd (1.5K x 64K x 6K = 576 GFlops)

        # Grad w.r.t. activation
        mlp_a_grad = 2.0 * st.mlp_za * (stream_grad @ m.W_out.w[i]) # Same flops as fwd, plus a (T, d_mlp) elementwise op

        # Recompute the MLP input norm
        xm_inv_rms = (st.xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xm_norm = bf16(st.xm.float() * xm_inv_rms)

        # Grad w.r.t. W_in
        m.W_in.gbank[i].add_(mlp_a_grad.mT @ xm_norm) # (d_mlp, T) @ (T, d_model) - Same flops as fwd (6K x 64K x 1.5K = 576 GFlops)

        xm_norm_grad = mlp_a_grad @ m.W_in.w[i] # (T, d_mlp) @ (d_mlp, d_model) --> (T, d_model)

        xm_grad = stream_grad + bf16(xm_inv_rms * (xm_norm_grad.float() - (xm_norm.float() * (xm_norm.float() * xm_norm_grad.float()).mean(dim=-1, keepdim=True))))

        # Attention backward
        xb_norm = st.xb_norm
        m.W_O.gbank[i].add_(xm_grad.mT @ st.y.view(T, -1))
        y_grad = (xm_grad @ m.W_O.w[i]).view(T, cfg.n_qo_heads, cfg.d_vo)

        q_norm_grad, k_norm_grad, v_grad = flash_attn_varlen_bwd(
            y_grad, st.q_norm, st.k_norm, st.v, st.y, st.lse, cu_seqlens, cfg.seq_len,
            cfg.window_sizes[i])

        # per-(token, head) norm backward
        qr_grad = bf16(st.q_inv_rms * (1.2 * q_norm_grad.float() - st.q_norm.float() * ((st.q_norm.float() * q_norm_grad.float()).mean(dim=-1, keepdim=True) / 1.2)))
        kr_grad = bf16(st.k_inv_rms * (1.2 * k_norm_grad.float() - st.k_norm.float() * ((st.k_norm.float() * k_norm_grad.float()).mean(dim=-1, keepdim=True) / 1.2)))

        # rotary backward = rotation by -theta (transpose of the forward rotation)
        q1_grad, q2_grad = qr_grad[..., :half], qr_grad[..., half:]
        k1_grad, k2_grad = kr_grad[..., :half], kr_grad[..., half:]
        q_grad = torch.cat([q1_grad * cos - q2_grad * sin, q1_grad * sin + q2_grad * cos], dim=-1)
        k_grad = torch.cat([k1_grad * cos - k2_grad * sin, k1_grad * sin + k2_grad * cos], dim=-1)

        # --- VE gate backward (ve and ve_gate_za recomputed) ---
        j = cfg.ve_index[i]
        d_xn_ve = None
        if j >= 0:
            # Retrieve the value embeddings
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)

            # Recompute gate forward
            ve_gate_za = torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)

            # ve_gate_a broadcasts over d_vo in v = v0 + a*ve, so its grad sums
            # that axis back out; then d/dz[3*sigmoid(z)] = 3*za*(1 - za).
            ve_gate_a_grad = (v_grad * ve).sum(dim=-1)   # (T, n_kv_heads)
            ve_gate_logit_grad = ve_gate_a_grad * (3 * ve_gate_za * (1 - ve_gate_za))

            m.ve_gate.gbank[j].add_(ve_gate_logit_grad.mT @ xb_norm[..., :cfg.d_ve_gate])

            # Embedding gradient
            ve_grad = (v_grad * (3 * ve_gate_za).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            # embedding_dense_backward beat raw index_add_ atomics ~2x
            m.value_embeds.gbank[j].add_(
                torch.ops.aten.embedding_dense_backward(ve_grad, idx, cfg.d_vocab, -1, False))

            d_xn_ve = ve_gate_logit_grad @ m.ve_gate.w[j]

        # v_grad passes through the VE add unchanged: v = v0 + ve_gate_a*ve
        q_grad =    q_grad.view(T, cfg.n_qo_heads * cfg.d_qk)
        k_grad =    k_grad.view(T, cfg.n_kv_heads * cfg.d_qk)
        v_grad = v_grad.reshape(T, cfg.n_kv_heads * cfg.d_vo)

        m.W_Q.gbank[i].add_(q_grad.mT @ xb_norm)
        m.W_K.gbank[i].add_(k_grad.mT @ xb_norm)
        m.W_V.gbank[i].add_(v_grad.mT @ xb_norm)

        xb_norm_grad = q_grad @ m.W_Q.w[i] + k_grad @ m.W_K.w[i] + v_grad @ m.W_V.w[i]
        if d_xn_ve is not None:
            xb_norm_grad[:, :cfg.d_ve_gate] += d_xn_ve
        xb_grad = xm_grad + bf16(st.xb_inv_rms * (xb_norm_grad.float() - (xb_norm.float() * (xb_norm.float() * xb_norm_grad.float()).mean(dim=-1, keepdim=True))))
        # --- blend backward: xb = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        g_resid.append(sum32(xb_grad * st.x_in))
        g_x0.append(sum32(xb_grad * x0))
        x0_grad = x0_grad + m.x0_lambdas.w[i] * xb_grad  # TRAP: x0 feeds every layer, accumulate
        stream_grad = m.resid_lambdas.w[i] * xb_grad
        stash[i] = None                          # free this layer's stash as we go

    # Land the per-layer resid/x0 scalar sums (collected in REVERSED layer
    # order) as one stacked add each.
    m.resid_lambdas.grad.add_(torch.stack(g_resid[::-1]))
    m.x0_lambdas.grad.add_(torch.stack(g_x0[::-1]))

    # stream_grad is now the grad through layer 0's input, which IS x0 (same tensor)
    smeared_grad = x0_grad + stream_grad         # grad wrt the smeared embedding

    # --- smear backward: x = cat([xe_norm[:1], xe_norm[1:] + gate*xe_norm[:-1]]) ---
    sg = torch.sigmoid(xe_norm[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)  # (T-1, 1), recomputed
    gate = bf16(m.smear_lambda.w) * sg
    xe_norm_grad = smeared_grad.clone()
    xe_norm_grad[:-1] += gate * smeared_grad[1:]  # TRAP: shifted scatter -- p's grad reaches p-1
    gate_grad = (smeared_grad[1:] * xe_norm[:-1]).sum(dim=-1, keepdim=True)   # (T-1, 1)
    m.smear_lambda.grad.add_(sum32(gate_grad * sg))
    gate_logit_grad = gate_grad * bf16(m.smear_lambda.w) * sg * (1 - sg)
    m.smear_gate.grad.add_((gate_logit_grad.mT @ xe_norm[1:, :cfg.d_smr_gate]).float())
    xe_norm_grad[1:, :cfg.d_smr_gate] += gate_logit_grad @ bf16(m.smear_gate.w)

    # --- embedding norm + token embedding scatter ---
    xe_grad = bf16(xe_inv_rms * (xe_norm_grad.float() - (xe_norm.float() * (xe_norm.float() * xe_norm_grad.float()).mean(dim=-1, keepdim=True))))
    m.input_embeds.grad.add_(
        torch.ops.aten.embedding_dense_backward(xe_grad, idx, cfg.d_vocab, -1, False))

    return loss


# ------------------------------------------------------------------------------
# § Eval Fwd
# ------------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def forward(idx, cu_seqlens, targets=None, loss_reduction='mean'):
    """Forward for validation loss and CORE eval. Returns the loss if
    targets are given, else the (softcapped, fp32) logits (T, d_vocab)."""

    assert idx.ndim == 1
    T = idx.size(0)
    D = cfg.d_model
    half = cfg.d_qk // 2

    assert T > 1, "Scoring forward pass should have T > 1 (smear needs a previous token)"
    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"
    cos, sin = m.cos[0, :T], m.sin[0, :T]  # (T, 1, half)
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)

    # Embed the tokens
    x = F.embedding(idx, m.input_embeds.w)         # bf16
    x = F.rms_norm(x, (D,))

    # Smear: mix the previous token's embedding into the current position.
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(
        x[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)
    x = torch.cat([x[:1], x[1:] + gate * x[:-1]], dim=0)

    # Forward the trunk of the Transformer
    x0 = x
    x_backout = None
    for i in range(cfg.n_layers):
        x = m.resid_lambdas.w[i] * x + m.x0_lambdas.w[i] * x0
        # --- attention ---
        xn = F.rms_norm(x, (D,))
        # (T, H, D) - the varlen kernel's native layout, no transpose needed
        q = (xn @ m.W_Q.w[i].mT).view(T, cfg.n_qo_heads,  cfg.d_qk)
        k = (xn @ m.W_K.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (xn @ m.W_V.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)
        # Value residual (ResFormer): value embedding mixed in via an
        # input-dependent per-head gate, range (0, 3)
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_a = 3 * torch.sigmoid(xn[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            v = v + ve_gate_a.unsqueeze(-1) * ve
        # Rotary embeddings (relative positional encoding)
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        # QK norm, then sharper attention (the 1.2 splits the scale between Q and K)
        q = F.rms_norm(q, (cfg.d_qk,)) * 1.2
        k = F.rms_norm(k, (cfg.d_qk,)) * 1.2
        y, _ = flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, cfg.seq_len, cfg.window_sizes[i])
        x = x + y.contiguous().view(T, -1) @ m.W_O.w[i].mT
        # --- MLP (relu^2) ---
        x = x + F.relu(F.rms_norm(x, (D,)) @ m.W_in.w[i].mT).square() @ m.W_out.w[i].mT
        if i == cfg.backout_layer:
            x_backout = x
    # Subtract mid-layer residual to remove low-level features before logit projection
    x = x - bf16(m.backout_lambda.w) * x_backout
    x = F.rms_norm(x, (D,))

    # lm_head + softcap
    logits = (x @ m.lm_head.w.mT).float()          # (T, d_vocab)
    logits = 15.0 * torch.tanh(logits / 15.0)    # smoothly cap to [-15, 15]

    if targets is not None:
        # No ignore_index: targets here only ever come from the training/val
        # loader, which never emits pad (see forward_backward's CE note).
        return F.cross_entropy(logits, targets, reduction=loss_reduction)
    return logits


# ------------------------------------------------------------------------------
# § AdamW
# ------------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Param,
    grad: Tensor,
    t: Tensor,      # (1,) Current step for schedules
) -> None:
    """AdamW update of this rank's shard of `p`."""

    # ==== Buffer Update ====
    grad = grad.float() # Some grads are bf16, EMAs are fp32.

    # Update EMAs. Mix a large portion of the tracked value with a small portion
    # of the current gradient.
    p.first_mntm.mul_(p.mntm_b1_t[t]).add_(grad * p.grad_b1_t[t]) # m = beta1*m + (1 - beta)*g
    p.scnd_mntm.mul_(p.mntm_b2_t[t]).add_(grad.square() * p.grad_b2_t[t]) # v = beta2*m + (2 - beta)*g^2

    # ==== Parameter Update ====
    if p.mantissa is not None:
        # Rebuild the raw 32-bit integer representation
        bits = ((p.w[p.shard_size].view(torch.int16).to(torch.int32) << 16)
                | (p.mantissa.view(torch.int16).to(torch.int32) & 0xFFFF))
        master = bits.view(torch.float32) # Same memory, viewed as fp32
    else:
        master = p.w[p.shard_size].float()

    # Apply weight decay inplace
    master.mul_(p.wd_t[t])

    # Apply AdamW's update inplace, w = w - lr * (m / (sqrt(v) + eps))
    master.sub_(p.lr_bc_t[t] * (p.first_mntm / (p.scnd_mntm.sqrt() + p.eps_t[t])))

    # Re-split the weight.
    if p.mantissa is not None:
        bits = master.view(torch.int32)
        # p.w = upper 16-bits of master
        p.w[p.shard_size].view(torch.int16).copy_((bits >> 16).to(torch.int16))
        p.mantissa.view(torch.int16).copy_(bits.to(torch.int16)) # Truncate upper 16, store mantissa
    else:
        p.w[p.shard_size].copy_(master)


# ------------------------------------------------------------------------------
# § Muon
# ------------------------------------------------------------------------------

def rebuild_master(live: Tensor, mantissa: Tensor) -> Tensor:
    """Reconstruct the fp32 master from bf16 live bits + stashed mantissa."""
    bits = ((live.view(torch.int16).to(torch.int32) << 16)
            | (mantissa.view(torch.int16).to(torch.int32) & 0xFFFF))
    return bits.view(torch.float32)

def writeback_master(master: Tensor, live: Tensor, mantissa: Tensor) -> None:
    """Truncation split of the updated master back into live + mantissa."""
    bits = master.view(torch.int32)
    live.view(torch.int16).copy_((bits >> 16).to(torch.int16))
    mantissa.view(torch.int16).copy_(bits.to(torch.int16))

# Polar Express orthogonalization coefficients, 5 iterations.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    p: Param,       # the (K, out, in) weight-bank bundle: live tensor, state, schedule tables
    grad: Tensor,   # (K, out, in) fp32 gradient shard -- MUTATED (nesterov lerp)
    t: Tensor,      # (1,) int64 device tensor - the schedule row to read
) -> None:
    """Fused Muon step on this rank's shard of `p`: momentum -> polar_express
    -> variance_reduction -> cautious update on the reconstructed master."""

    # Nesterov momentum
    p.first_mntm.mul_(p.mntm_b1_t[t]).add_(grad * p.grad_b1_t[t])
    g = grad.lerp_(p.first_mntm, p.mntm_b1_t[t])

    # Polar express (orthogonalization), in bf16
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction (NorMuon), in fp32.
    v_mean = g.float().square().mean(dim=p.residual_dim, keepdim=True)
    residual_dim_size = g.size(p.residual_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * residual_dim_size
    v_norm = v_norm_sq.sqrt()
    p.scnd_mntm.mul_(p.mntm_b2_t[t]).add_(v_mean * p.grad_b2_t[t])
    step_size = p.scnd_mntm.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * residual_dim_size) * step_size.square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale

    # Cautious weight decay + master update + truncation split back to live
    live = p.w[p.shard_size]
    master = rebuild_master(live, p.mantissa)
    mask = (g * master) >= 0
    master.sub_(p.lr_bc_t[t] * g + p.wd_t[t] * master * mask)
    writeback_master(master, live, p.mantissa)


# ==============================================================================
# § Weight Init & Schedule
# ==============================================================================


# ------------------------------------------------------------------------------
# § LR Schedule
# ------------------------------------------------------------------------------

# Learning rate schedule as a per-step multiplier.
# Shared by Muon and AdamW.
lr_mult_t = np.ones(cfg.num_steps)

steps_0idx = np.arange(cfg.num_steps, dtype=np.float64)  # 0-based, the way the loop counts
steps_1idx = steps_0idx + 1.0                            # 1-based, the way bias corrections count

# Warmup 40 steps, starting at 0.5 of peak.
lr_mult_t[:40] = 0.5 + 0.5 * steps_1idx[:40] / 40

# Warmdown for 65% of the run.
warmdown_len  = round(0.65 * cfg.num_steps)
warmdown      = slice(cfg.num_steps - warmdown_len + 1, cfg.num_steps)   # the hold covers everything before
warmdown_frac = (cfg.num_steps - steps_0idx[warmdown]) / warmdown_len  # ~1 -> ~0 across the warmdown

lr_mult_t[warmdown] = 0.05 + (1.0 - 0.05) * warmdown_frac

m = Model()

# All ranks use the same seed so they all train the same initial model.
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

fp32_empty   = lambda *shape: torch.empty(*shape, dtype=torch.float32, device=device)
bf16_empty   = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=device)
fp32_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)

# Uniform init bound. Var(Uniform(-a, a)) = a^2/3, so std = a/sqrt(3): to hit
# a target std of 1/sqrt(d_model), the bound must be sqrt(3) times it.
matrix_init_s = (3 ** 0.5) * (cfg.d_model ** -0.5)

upper_bf16   = lambda w: (w.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
lower_uint16 = lambda w: (w.contiguous().view(torch.int32)      ).to(torch.int16).view(torch.uint16)

# Create an fp32 tensor on the device.
dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------
# § Scalars
# ------------------------------------------------------------------------------

resid_lambdas  = torch.linspace(1.15, 1.05, cfg.n_layers, dtype=torch.float32, device=device)
x0_lambdas     = torch.linspace(0.20, 0.05, cfg.n_layers, dtype=torch.float32, device=device)
smear_lambda   = fp32_zeros(1)
backout_lambda = fp32_empty(1).fill_(0.2)

smear_gate     = fp32_empty(1, cfg.d_smr_gate).uniform_(-cfg.d_smr_gate ** -0.5, cfg.d_smr_gate ** -0.5)

# Define the grad mix-in ramp the same way we do the global learning rate
# schedule, as a multiplier to apply to each scalar's individual coefficient.
scalar_grad_mult_t = np.ones(cfg.num_steps)

# Warmup 45 steps, starting at 0.00184 of peak.
scalar_grad_mult_t[:45] = 0.00184**np.clip(1.0 - steps_0idx[:45] / 45, 0.0, 1.0)


scalar_configs = [
#   name,               weights,       peak lr,  b1_grad,   b2_grad,    wd,
    ("resid_lambdas",   resid_lambdas,   0.0071,   0.2,        0.05,   0.05),
    ("x0_lambdas",      x0_lambdas,      0.71,     0.04,       0.05,   0.0),
    ("smear_gate",      smear_gate,      0.2,      0.2,        0.05,   0.0),
    ("smear_lambda",    smear_lambda,    0.2,      0.2,        0.05,   0.0),
    ("backout_lambda",  backout_lambda,  0.2,      0.2,        0.05,   0.0)
]

# For each of the scalar parameters...
for (name, w, peak_lr, b1_grad, b2_grad, wd) in scalar_configs:

    # Derive the momentum buffers' decays.
    b1_mntm = 1 - b1_grad # i.e., Beta1 = 1 - (1-Beta1)
    b2_mntm = 1 - b2_grad # i.e., Beta2 = 1 - (1-Beta2)

    # Build the Param for the scalar.
    p = Param(
        # Weight
        name         = name,
        w            = w,          # Live weights
        mantissa     = None,       # Scalars are fp32 live

        # Gradients
        grad         = fp32_zeros(w.shape),
        gbank        = None,       # Scalars don't need banks

        # Momentum buffers
        shard_size   = slice(None), # Scalars aren't sharded.
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(w.shape),

        residual_dim = None, # Muon only

        # Schedules
        # Fold bias correction into the learning rate.
        lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

        # Weight decay schedule
        wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

        # Beta schedule, scalars use warmup on b1/2_grads
        mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),   # first_mntm decay (Beta1)
        grad_b1_t    = dev(scalar_grad_mult_t * b1_grad),      # first_mntm grad mix-in (1-Beta1)

        mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
        grad_b2_t    = dev(scalar_grad_mult_t * b2_grad),      # scnd_mntm grad mix-in (1-Beta2)

        eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5), # May not be necessary
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)


# ------------------------------------------------------------------------------
# § Embeddings
# ------------------------------------------------------------------------------

input_embeds =     bf16_empty(cfg.d_vocab, cfg.d_model)
input_embeds.copy_(fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.8))
value_embeds =     bf16_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
value_embeds.copy_(fp32_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
                   .uniform_(-matrix_init_s, matrix_init_s))

ve_rows = cfg.num_ves * cfg.d_vocab
vocab_shard = slice(rank*(cfg.d_vocab  // world_size), (rank+1)*(cfg.d_vocab  // world_size))
ve_shard    = slice(rank*(ve_rows      // world_size), (rank+1)*(ve_rows      // world_size))

assert cfg.n_layers % world_size == 0 and cfg.d_vocab % world_size == 0 and ve_rows % world_size == 0, \
    "Invalid shape for sharding"

embed_configs = [
#   name,             weights,       peak lr,  b1_grad,  b2_grad,  wd,     shard,       slots
    ("input_embeds",  input_embeds,  0.3,      0.2,      0.005,    0.001,  vocab_shard, 1),
    ("value_embeds",  value_embeds,  0.15,     0.2,      0.005,    0.01,   ve_shard,    cfg.num_ves)
]

# For each of the embedding tables...
for (name, w, peak_lr, b1_grad, b2_grad, wd, shard, slots) in embed_configs:

    # Derive the momentum buffers' decays.
    b1_mntm = 1 - b1_grad
    b2_mntm = 1 - b2_grad

    # bf16 grad accumulators: the two embedding tables handle bf16
    # accumulation fine (and are the largest tensors).
    grad = torch.zeros(w.shape, dtype=torch.bfloat16, device=device)

    p = Param(
        # Weight
        name         = name,
        w            = w,          # bf16 live, no fp32 master
        mantissa     = None,

        # Gradients
        grad         = grad,
        gbank        = list(grad.view(slots, cfg.d_vocab, -1).unbind(0)) if slots > 1 else None,

        # Momentum buffers
        shard_size   = shard,
        first_mntm   = fp32_zeros(w[shard].shape),
        scnd_mntm    = fp32_zeros(w[shard].shape),

        residual_dim = None, # Muon only

        # Schedules
        # Fold bias correction into the learning rate.
        lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

        # Weight decay schedule
        wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

        # Beta schedule, constant (no warmup ramp)
        mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),   # first_mntm decay (Beta1)
        grad_b1_t    = dev(np.full(cfg.num_steps, b1_grad)),   # first_mntm grad mix-in (1-Beta1)

        mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
        grad_b2_t    = dev(np.full(cfg.num_steps, b2_grad)),   # scnd_mntm grad mix-in (1-Beta2)

        eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)


# ------------------------------------------------------------------------------
# § LM Head
# ------------------------------------------------------------------------------

lm_head = fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.001)

# Define the grad mix-in ramp the same way we do the global learning rate
# schedule, as a multiplier to apply to the head's individual coefficients.
lm_grad_mult_t = np.ones(cfg.num_steps)

# Warmup 20 steps, starting at 0.1185 of peak.
lm_grad_mult_t[:20] = 0.1185**np.clip(1.0 - steps_0idx[:20] / 20, 0.0, 1.0)

peak_lr = 0.008
b1_grad = 0.2    # (1-Beta1)
b2_grad = 0.04   # (1-Beta2)
wd      = 0.01

# Derive the momentum buffers' decays.
b1_mntm = 1 - b1_grad
b2_mntm = 1 - b2_grad

# Split the fp32 draw into bf16 live + this rank's stashed low bits.
live = upper_bf16(lm_head)

m.lm_head = Param(
    # Weight
    name         = "lm_head",
    w            = live,
    mantissa     = lower_uint16(lm_head[vocab_shard]),

    # Gradients
    grad         = fp32_zeros(live.shape),
    gbank        = None,

    # Momentum buffers
    shard_size   = vocab_shard,
    first_mntm   = fp32_zeros(live[vocab_shard].shape),
    scnd_mntm    = fp32_zeros(live[vocab_shard].shape),

    residual_dim = None, # Muon only

    # Schedules
    # Fold bias correction into the learning rate.
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

    # Weight decay schedule
    wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

    # Beta schedule, warmup on the grad mix-ins
    mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),   # first_mntm decay (Beta1)
    grad_b1_t    = dev(lm_grad_mult_t * b1_grad),          # first_mntm grad mix-in (1-Beta1)

    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(lm_grad_mult_t * b2_grad),          # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)


# ------------------------------------------------------------------------------
# § Attention & MLPs
# ------------------------------------------------------------------------------

W_Q =   fp32_empty(cfg.n_layers, cfg.n_qo_heads * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_K =   fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_V =   fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_vo, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_O =   fp32_zeros(cfg.n_layers,               cfg.d_model, cfg.n_qo_heads * cfg.d_vo)  # projections start at zero

ve_gate = fp32_empty(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate).uniform_(0.0, 0.02)

W_in  = fp32_empty(cfg.n_layers, cfg.d_mlp,   cfg.d_model).uniform_(-matrix_init_s * 0.4, matrix_init_s * 0.4)
W_out = fp32_zeros(cfg.n_layers, cfg.d_model, cfg.d_mlp)             # projections start at zero

# Muon momentum warmup 0.85 -> 0.97 over 400 steps
momentum_warmup = 400
momentum = np.full(cfg.num_steps, 0.97)
momentum[:momentum_warmup] = (0.85 + (0.97 - 0.85)
                              * steps_1idx[:momentum_warmup] / momentum_warmup)

# Muon momentum warmdown to 0.90.
momentum[warmdown] = 0.90 + (0.97 - 0.90) * warmdown_frac

# Muon weight decay. "half-cosine from its peak to zero over the whole
# run, with step 0 sitting at the peak"
muon_wd = np.empty(cfg.num_steps)
muon_wd[0] = 0.06
run_frac = (cfg.num_steps - steps_0idx[1:]) / cfg.num_steps
muon_wd[1:] = 0.06 * (0.5 * (1.0 + np.cos(math.pi * (1.0 - run_frac))))

# Muon peak lr is nanochat's tuned 0.02 on batch_lr_scale, scaled up further for
# tall matrices by their sqrt(fan_out/fan_in) aspect ratio -- only W_in, whose
# 4x MLP expansion gives 2.0 at every depth. rdim is the axis facing the
# residual stream: W_O and W_out live transposed -> -2; the ve_gate rows read a
# d_ve_gate slice of the stream -> -1.

layer_shard = slice(rank*(cfg.n_layers // world_size), (rank+1)*(cfg.n_layers // world_size))

muon_configs = [
#    name,      weights,  peak lr,  shard,        rdim
    ("W_Q",     W_Q,      0.028,    layer_shard,  -1),
    ("W_K",     W_K,      0.028,    layer_shard,  -1),
    ("W_V",     W_V,      0.028,    layer_shard,  -1),
    ("W_O",     W_O,      0.028,    layer_shard,  -2),
    ("W_in",    W_in,     0.056,    layer_shard,  -1),
    ("W_out",   W_out,    0.028,    layer_shard,  -2),
    ("ve_gate", ve_gate,  0.028,    slice(None),   -1)
]

# For each of the Muon-trained weight banks...
for (name, w, peak_lr, shard, rdim) in muon_configs:

    # Split the fp32 draw into bf16 live + this rank's stashed low bits.
    live = upper_bf16(w)

    grad = fp32_zeros(live.shape)

    # The NorMuon second moment holds each neuron's mean-square update -- the
    # shard's shape with the residual-facing dim `rdim` collapsed to 1 (after
    # orthogonalization only the smaller dim can carry variance).
    shard_shape = list(live[shard].shape)
    first_mntm = fp32_zeros(shard_shape)
    shard_shape[rdim] = 1

    p = Param(
        # Weight
        name         = name,
        w            = live,
        mantissa     = lower_uint16(w[shard]),

        # Gradients
        grad         = grad,
        gbank        = list(grad.unbind(0)),

        # Momentum buffers
        shard_size   = shard,
        first_mntm   = first_mntm,
        scnd_mntm    = fp32_zeros(shard_shape),

        residual_dim = rdim,

        # Schedules
        # The second moment is self-normalizing (the v_norm/v_norm_new
        # rescale), so lr_bc_t has no bias correction and there is no eps.
        lr_bc_t      = dev(lr_mult_t * peak_lr),
        wd_t         = dev(lr_mult_t * peak_lr * muon_wd),

        # b1 is the nesterov momentum (warmed up/down above); b2 is the
        # variance-reduction EMA, a constant 0.9.
        mntm_b1_t    = dev(momentum),
        grad_b1_t    = dev(1.0 - momentum),
        mntm_b2_t    = dev(np.full(cfg.num_steps, 0.9)),
        grad_b2_t    = dev(np.full(cfg.num_steps, 0.1)),

        eps_t        = None,
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)

# The Params own everything now: free the fp32 draws, drop the adopted names.
del lm_head, input_embeds, value_embeds, resid_lambdas, x0_lambdas, smear_gate, smear_lambda, backout_lambda, W_Q, W_K, W_V, W_O, ve_gate, W_in, W_out


# ------------------------------------------------------------------------------
# § Rotary Cache
# ------------------------------------------------------------------------------

rotary_seq_len = max(cfg.micro_batch_tokens, cfg.eval_buffer_tokens)
channel_range = torch.arange(0, cfg.d_qk, 2, dtype=torch.float32, device=device)  # stride the channels
inv_freq = 1.0 / (100000 ** (channel_range / cfg.d_qk))
t_pos = torch.arange(rotary_seq_len, dtype=torch.float32, device=device)          # stride the time steps
freqs = torch.outer(t_pos, inv_freq)   # rotation frequency at each (time, channel) pair

m.cos = freqs.cos().to(torch.bfloat16)[None, :, None, :]  # add batch and head dims
m.sin = freqs.sin().to(torch.bfloat16)[None, :, None, :]  # for later broadcasting

del channel_range, inv_freq, t_pos, freqs


# ==============================================================================
# § Training Harness
# ==============================================================================


# ------------------------------------------------------------------------------
# § Distributed Data Loader
# ------------------------------------------------------------------------------

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        """Returns (starts, ends) per rank, or None if this shard is exhausted."""
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    return None
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def distributed_data_generator(filename_pattern: str, num_tokens_local: int, max_seq_len: int):
    """
    Generator (i.e., yields rather than returns) of one micro-batch per call:
    `num_tokens_local` tokens for this rank, as (inputs, targets, cu_seqlens)
    device tensors -- the packed varlen layout the forward passes consume.
    Sequences are BOS-aligned and only returned from their beginning; tokens
    past max_seq_len are discarded (the next sequence starts at the next BOS).
    Single-epoch: the generator ends when the shards run out.
    Serves training (micro_batch_tokens per rank) and validation
    (eval_buffer_tokens per rank).
    """
    # Get the list of shard files and wrap in an iterator.
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
    file_iter = iter(files)

    # Load the first shard.
    tokens = _load_data_shard(next(file_iter))

    shard = Shard(tokens, world_size)
    remaining_files = list(file_iter)
    next_shard_idx = 0
    next_shard_getter = Shard.load_async(remaining_files[0], world_size) if remaining_files else None

    while True:
        # Get the start and end indices (within `tokens`) of the sequences to use for
        # the current micro-batch.
        result = shard.next_batch(num_tokens_local, max_seq_len)

        # If this shard is exhausted,
        if result is None:
            # If there are no more shards, kill the dataloader.
            if next_shard_getter is None:
                return

            # Load the next shard.
            shard = next_shard_getter()
            tokens = shard.tokens
            next_shard_idx += 1
            next_shard_getter = Shard.load_async(remaining_files[next_shard_idx], world_size) if next_shard_idx < len(remaining_files) else None

            # Re-start the loop.
            continue

        # Locations of the documents in `tokens`. Only specifies the
        # number of documents needed, not max.
        start_idxs = torch.tensor(result[0][rank])
        end_idxs = torch.tensor(result[1][rank])

        # `tokens` contains the entire shard. The sequences defined by the starts and ends
        # may or may not be contiguous within `tokens`, due to some sequences being
        # truncated, so we slice them and then re-concatenate into a single tensor.
        buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])

        # `buf` contains `num_tokens_local + 1` tokens to allow for the inputs vs.
        # targets offset.
        _inputs = buf[:-1] # All tokens minus the last
        _targets = buf[1:] # Shift the tokens to the left, so that targets contains the
                           # next token for each input token.

        # The final document includes an extra token that is the target of the last
        # token in the last document. Now that we have our `_targets`, we can remove it.
        end_idxs[-1] -= 1

        # Calculate the start indices of the documents within `_inputs`. (flashattention
        # start_idxs are relative to the `tokens` buffer, so we convert them by
        # accumulating the document lengths.
        # cum_lengths starts with the second document, so we'll shift
        cum_lengths = (end_idxs - start_idxs).cumsum(0)

        # One entry per doc plus the leading 0 must fit the fixed buffer.
        assert len(cum_lengths) < cfg.max_num_docs, \
            f"micro-batch packed {len(cum_lengths)} docs; cu_seqlens holds only {cfg.max_num_docs}"

        # The actual cu_seqlens array always needs to contain `max_num_docs` elements so we
        # the compiler can build a single graph.
        # We allocate that buffer here and fill it with "empty documents", i.e., setting their start index
        # to one past the end of the `_inputs` buffer.
        _cum_lengths = torch.full((cfg.max_num_docs,), num_tokens_local)

        # Then copy in the lengths, inserting the first document (index 0).
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 / int64 on the CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
        )
        # Execution resumes here on the next call.


# ------------------------------------------------------------------------------
# § Distributed Optimizer
# ------------------------------------------------------------------------------

# THE schedule position: one (1,) int64 device tensor, advanced on-device at
# the end of optimizer_step -- the host never syncs on it.
t_step = torch.zeros(1, dtype=torch.int64, device=device)

@torch.no_grad() # Required for in-place leaf updates
def optimizer_step():
    """One explicit optimizer step. Each weight's Param carries its state and
    schedule tables, so the fused steps take just (param, grad, t)."""

    # ---- Phase 1: launch every async grad reduction --------------------------
    reduced = {}   # param -> (async work handle, shard-size reduced grad)
    if world_size > 1:
        for p in (m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out):
            g_shard = torch.empty_like(p.first_mntm)
            reduced[p] = (dist.reduce_scatter_tensor(g_shard, p.grad, op=dist.ReduceOp.AVG, async_op=True), g_shard)
        for p in (m.lm_head, m.input_embeds, m.value_embeds):
            g_shard = torch.empty_like(p.first_mntm, dtype=p.grad.dtype)   # (row shard, cols) in the grad's dtype
            reduced[p] = (dist.reduce_scatter_tensor(g_shard, p.grad, op=dist.ReduceOp.AVG, async_op=True), g_shard)

    # ---- Phase 2: wait -> owned-shard update -> gather the live shard --------
    gathers = []

    # Muon banks, sharded over layers
    for p in (m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out):
        if world_size > 1:
            work, grad = reduced[p]
            work.wait()
        else:
            grad = p.grad
        muon_step_fused(p, grad, t_step)
        p.grad.zero_()
        if world_size > 1:
            gathers.append(dist.all_gather_into_tensor(p.w, p.w[p.shard_size], async_op=True))

    # Muon replicated: ve_gate is tiny, every rank updates all of it
    if world_size > 1:
        dist.all_reduce(m.ve_gate.grad, op=dist.ReduceOp.AVG)
    muon_step_fused(m.ve_gate, m.ve_gate.grad, t_step)
    m.ve_gate.grad.zero_()

    # AdamW, sharded over vocab rows.
    for p in (m.lm_head, m.input_embeds, m.value_embeds):
        if world_size > 1:
            work, grad = reduced[p]
            work.wait()
        else:
            grad = p.grad
        # Run AdamW
        adamw_step_fused(p, grad, t_step)
        p.grad.zero_()
        if world_size > 1:
            gathers.append(dist.all_gather_into_tensor(p.w, p.w[p.shard_size], async_op=True))

    # AdamW replicated scalars (fp32-live, no mantissa).
    if world_size > 1:
        for p in (m.resid_lambdas, m.x0_lambdas, m.smear_gate, m.smear_lambda, m.backout_lambda):
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    for p in (m.resid_lambdas, m.x0_lambdas, m.smear_gate, m.smear_lambda, m.backout_lambda):
        adamw_step_fused(p, p.grad, t_step)
        p.grad.zero_()

    # ---- Phase 3: wait out the live all-gathers ------------------------------
    for work in gathers:
        work.wait()

    t_step.add_(1)  # advance the schedule on-device


# ------------------------------------------------------------------------------
# § Logging
# ------------------------------------------------------------------------------

def gather_full(t):
    """For gathering optimizer state onto rank 0."""
    if world_size == 1:
        return t
    comm = t.view(torch.bfloat16) if t.dtype == torch.uint16 else t
    full = torch.empty(t.shape[0] * world_size, *t.shape[1:], dtype=comm.dtype, device=device)
    dist.all_gather_into_tensor(full, comm)
    return full.view(torch.uint16) if t.dtype == torch.uint16 else full

def write_checkpoint(step):
    state = {}

    for p in m:
        for attr in ("mantissa", "first_mntm", "scnd_mntm"):
            if getattr(p, attr) is not None:
                if p.shard_size != slice(None):   # replicated state needs no gather
                    full = gather_full(getattr(p, attr))
                else:
                    full = getattr(p, attr)
                if master_process:
                    state[f"{p.name}.{attr}"] = full.cpu()

    if not master_process:
        return

    os.makedirs(f"logs/{cfg.run_stamp}_{cfg.run_name}", exist_ok=True)
    torch.save(dict(step=step, code=code,
                    weights={p.name: p.w.cpu() for p in m}),
               f"logs/{cfg.run_stamp}_{cfg.run_name}/model_step{step:06d}.pt")
    torch.save(dict(step=step, t_step=int(t_step.item()), state=state),
               f"logs/{cfg.run_stamp}_{cfg.run_name}/optim_step{step:06d}.pt")

logfile = None
if master_process:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{cfg.run_stamp}_{cfg.run_name}.txt"
    print(logfile)

def print0(s="", console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# The notebook analog of the script logging its own source: `In` is IPython's
# input history, so this is every cell executed so far -- the whole file except
# the training-loop and eval cells below. Also lands in the checkpoints.


print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

print0(f"Model parameters: {cfg.num_params:,} | FLOPs/token: {cfg.num_flops_per_token:e}", console=True)
print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}", console=True)
print0(f"Total batch size: {cfg.total_batch_size:,} tokens = {cfg.micro_batch_tokens:,} tokens/micro "
       f"x {world_size} ranks x {cfg.grad_accum_steps} grad accum", console=True)

gc_t0 = 0.0
def gc_logging_hook(phase, info):
    """Registered at step 10: after setup, any collector run is a surprise
    worth flagging (cycle scans cost ~500ms at random steps)."""
    global gc_t0
    if phase == "start":
        gc_t0 = time.perf_counter()
    else:
        print(f"[rank {rank}] gc gen{info['generation']}: collected {info['collected']} "
              f"({(time.perf_counter() - gc_t0) * 1000:.0f}ms)", flush=True)

if not cfg.use_wandb or not master_process:
    class DummyWandb:
        """No-op wandb replacement when logging is disabled."""
        def log(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
        def finish(self): pass

    wandb_run = DummyWandb()
else:
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        # The config, verbatim: every StackConfig field, defaults and derived.
        config={name: getattr(cfg, name) for name in StackConfig.__annotations__},
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

profiler = None
if PROFILE:
    ABORT_STEP = 14   # wait out compile and the step-10 hooks, trace 12-13, stop
    if master_process:
        from torch.profiler import ProfilerActivity, profile as torch_profile
        profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,
            schedule=torch.profiler.schedule(wait=11, warmup=1, active=2, repeat=1))
        profiler.__enter__()


# ==============================================================================
# § Training Loop
# ==============================================================================


# ------------------------------------------------------------------------------
# § Run!
# ------------------------------------------------------------------------------

val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_val_time = 0.0
timed = []  # Length of each step in seconds, excluding first 10.
            # Total training time = np.sum(timed).

train_loader = distributed_data_generator(train_files, cfg.micro_batch_tokens, cfg.seq_len)

inputs, targets, cu_seqlens = next(train_loader)   # kick off the first batch

# Training loop
for step in range(cfg.num_steps + 1):
    last_step = step == (ABORT_STEP or cfg.num_steps)   # an abort cuts the loop, never the schedules

    # --------------- Validation Loop -----------------
    if last_step or (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0):
        torch.cuda.synchronize()
        val_t0 = time.perf_counter()
        val_loader = distributed_data_generator(val_files, cfg.eval_buffer_tokens, cfg.seq_len)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        for _ in range(cfg.val_steps):
            v_inputs, v_targets, v_cu_seqlens = next(val_loader)
            loss_flat = forward(v_inputs, v_cu_seqlens, v_targets, loss_reduction='none')
            num_bytes_flat = token_bytes[v_targets]
            total_nats += (loss_flat * (num_bytes_flat > 0)).sum()
            total_bytes += num_bytes_flat.sum()
        del val_loader
        if world_size > 1:
            dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        val_bpb = total_nats.item() / (math.log(2) * total_bytes.item())
        min_val_bpb = min(min_val_bpb, val_bpb)
        val_elapsed = time.perf_counter() - val_t0
        total_val_time += val_elapsed
        print0(f"step:{step}/{cfg.num_steps} val_bpb:{val_bpb:.6f} val_time:{val_elapsed:.2f}s", console=True)
        wandb_run.log({"step": step, "val/bpb": val_bpb, "val/eval_seconds": val_elapsed,
                       "total_training_time": np.sum(timed),
                       "time/wall_seconds": time.perf_counter() - run_wall_t0})

    # --------------- Checkpoint -----------------
    if cfg.save_checkpoint and (last_step or step in cfg.save_steps):
        ckpt_t0 = time.perf_counter()
        write_checkpoint(step)
        print0(f"checkpoint captured at step {step} ({time.perf_counter() - ckpt_t0:.1f}s)", console=True)

    # Exit final step after validation and checkpoint
    if last_step:
        break

    # --------------- Training Step -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()

    # Gradient Accumulation Loop
    for micro in range(cfg.grad_accum_steps):

        # Forward and Backward pass
        loss = forward_backward(inputs, targets, cu_seqlens,
                                loss_scale=1.0 / (cfg.grad_accum_steps * inputs.size(0)))

        # Next training batch
        inputs, targets, cu_seqlens = next(train_loader)

    # Smooth gradients, update weights, zero the grads
    optimizer_step()

    train_loss = loss.item()
    torch.cuda.synchronize()
    dt = time.perf_counter() - step_t0

    # --------------- Timing and Logging -----------------
    pct_done = 100 * step / cfg.num_steps

    # EMA the loss for readability.
    smooth_train_loss = 0.9*smooth_train_loss + 0.1*train_loss
    debiased_smooth_loss = smooth_train_loss / (1 - 0.9**(step + 1))

    # Track time and ETA after first 10 steps to exclude compile time.
    if step > 10:
        timed.append(dt)
        remaining_time = (cfg.num_steps - step - 1) * np.mean(timed) / 60
        eta_str = f" | eta: {remaining_time:.1f}m"
    else:
        eta_str = ""

    tok_per_sec = int(cfg.total_batch_size / dt)
    mfu = 100 * cfg.num_flops_per_token * cfg.total_batch_size / dt / (gpu_peak_flops * world_size)

    print0(f"step {step:05d}/{cfg.num_steps:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lr_mult_t: {lr_mult_t[step]:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | total time: {np.sum(timed)/60:.2f}m{eta_str}", console=True)

    wandb_run.log({
        "step": step,
        "train/loss": debiased_smooth_loss,
        "train/lr_mult_t": float(lr_mult_t[step]),
        "train/dt": dt,
        "time/wall_seconds": time.perf_counter() - run_wall_t0,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "total_training_time": np.sum(timed),
    })

    if profiler is not None:
        profiler.step()

    # Keep garbage collection out of timed portion.
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    # Garbage collection and compile are done by step 10, flag anything after.
    elif step == 10:
        torch._logging.set_logs(recompiles=True)
        gc.callbacks.append(gc_logging_hook)
    elif step % 5000 == 0:
        gc.collect()


# ------------------------------------------------------------------------------
# § Results
# ------------------------------------------------------------------------------

if profiler is not None:
    profiler.__exit__(None, None, None)
    trace_path = f"logs/{cfg.run_stamp}_{cfg.run_name}_trace.json.gz"
    profiler.export_chrome_trace(trace_path)
    print0(f"chrome trace -> {trace_path} (ui.perfetto.dev)", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {np.sum(timed)/60:.2f}m | val {total_val_time/60:.2f}m | "
       f"wall {(time.perf_counter() - run_wall_t0)/60:.2f}m", console=True)

# Step duration summary, steps 0-10 excluded (compile lives there).
if timed:
    print0(f"  {cfg.total_batch_size:>9,} tokens/step: {len(timed):5d} steps  mean {np.mean(timed):.3f}s  "
           f"median {np.median(timed):.3f}s  {int(cfg.total_batch_size / np.mean(timed)):,} tok/s", console=True)

wandb_run.log({"step": cfg.num_steps, "time/train_seconds": np.sum(timed),
               "time/val_seconds": total_val_time,
               "time/wall_seconds": time.perf_counter() - run_wall_t0,
               "time/step_seconds_mean": float(np.mean(timed)) if timed else 0.0})

if val_bpb is not None:
    print0(f"minimum validation bpb: {min_val_bpb:.6f}", console=True)


# ------------------------------------------------------------------------------
# § CORE Evaluation
# ------------------------------------------------------------------------------

# Off by default. The script exits here; a notebook just stops running cells,
# so this is a guard the reader steps over rather than a sys.exit().
if not cfg.core_eval:
    wandb_run.save(logfile, policy="now")
    wandb_run.finish()
    dist.destroy_process_group()
    raise SystemExit("core_eval is off -- nothing further to run")

core_eval_dir = os.path.join(DATASET_DIR, "core_eval")
with open(os.path.join(core_eval_dir, "config.json")) as f:
    core_tasks = json.load(f)["tasks"]

core_eval_t0 = time.perf_counter()
core_accuracy, core_centered = {}, {}
for task_info in core_tasks:
    task_t0 = time.perf_counter()
    task = torch.load(os.path.join(core_eval_dir, task_info["file"]), weights_only=False)
    label, task_type, num_examples = task["label"], task["task_type"], task["num_examples"]
    print0(f"Evaluating: {label} ({task_type}, {num_examples} examples)... ", console=True)

    # Each rank scores its own share of the examples (all of an example's
    # sequences stay on one rank); per-example correctness is summed at the
    # end. Sequences pack greedily, in order, into eval_buffer_tokens.
    buffers, cur_seqs, cur_len = [], [], 0
    for s in task["sequences"]:
        if s["example_idx"] % world_size != rank:
            continue
        assert len(s["tokens"]) <= cfg.eval_buffer_tokens
        if cur_len + len(s["tokens"]) > cfg.eval_buffer_tokens:
            buffers.append(cur_seqs)
            cur_seqs, cur_len = [], 0
        cur_seqs.append(s)
        cur_len += len(s["tokens"])
    if cur_seqs:
        buffers.append(cur_seqs)

    # Score every sequence: the mean loss over its answer span for multiple
    # choice / schema, exact match of the span for language modeling. Answer
    # tokens [start, end) are predicted by the logits at [start-1, end-1).
    seq_score = {}
    for seqs in buffers:
        # BOS pads the tail. cu_seqlens has a FIXED length so the compiled
        # forward sees one shape; its unused entries all point at the buffer
        # end, the first of them closing the pad as a ghost sequence.
        input_ids = torch.full((cfg.eval_buffer_tokens,), BOS_ID, dtype=torch.int32)
        cu_seqlens = torch.full((cfg.eval_buffer_tokens // 8,), cfg.eval_buffer_tokens, dtype=torch.int32)
        assert len(seqs) + 1 < cu_seqlens.numel()
        cu_seqlens[0] = 0
        spans, pos = [], 0
        for i, s in enumerate(seqs):
            input_ids[pos:pos + len(s["tokens"])] = torch.tensor(s["tokens"], dtype=torch.int32)
            spans.append((s["example_idx"], s["seq_idx"], pos + s["start_idx"], pos + s["end_idx"]))
            pos += len(s["tokens"])
            cu_seqlens[i + 1] = pos
        input_ids, cu_seqlens = input_ids.to(device), cu_seqlens.to(device)

        logits = forward(input_ids, cu_seqlens)                # (T, d_vocab) fp32, softcapped
        targets = torch.roll(input_ids.long(), shifts=-1)      # loss[j] = -log p(token j+1 | tokens ..j)
        losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = logits.argmax(dim=-1)
        for example_idx, seq_idx, start, end in spans:
            if task_type == "language_modeling":
                seq_score[example_idx, seq_idx] = bool((predictions[start - 1:end - 1] == input_ids[start:end]).all())
            else:
                seq_score[example_idx, seq_idx] = losses[start - 1:end - 1].mean().item()

    # Per-example correctness: language modeling needs the whole span right;
    # multiple choice / schema picks the lowest-loss continuation.
    correct = torch.zeros(num_examples, dtype=torch.float32, device=device)
    for idx in range(rank, num_examples, world_size):
        if task_type == "language_modeling":
            correct[idx] = seq_score[idx, 0]
        else:
            assert task_type in ("multiple_choice", "schema"), task_type
            choice_losses = [seq_score[idx, j] for j in range(task["num_seqs_per_example"][idx])]
            correct[idx] = choice_losses.index(min(choice_losses)) == task["gold_labels"][idx]
    if world_size > 1:
        dist.all_reduce(correct)
    accuracy = correct.mean().item()
    chance = task["random_baseline"] / 100                     # stored in percent
    core_accuracy[label] = accuracy
    core_centered[label] = (accuracy - chance) / (1 - chance)
    print0(f"accuracy: {accuracy:.4f} | centered: {core_centered[label]:.4f} | "
           f"time: {time.perf_counter() - task_t0:.2f}s", console=True)

core_metric = sum(core_centered.values()) / len(core_centered)
core_eval_elapsed = time.perf_counter() - core_eval_t0
print0(f"CORE metric: {core_metric:.4f} | total CORE eval time: {core_eval_elapsed:.2f}s", console=True)
wandb_run.log({
    "step": cfg.num_steps,
    "core_metric": core_metric,
    **{f"core/{label}/accuracy": a for label, a in core_accuracy.items()},
    **{f"core/{label}/centered": c for label, c in core_centered.items()},
    "timing/core_eval_seconds": core_eval_elapsed,
})

wandb_run.save(logfile, policy="now")
wandb_run.finish()
dist.destroy_process_group()
