# train_stack.py
#
# nanochat-based pre-training pipeline, with model code implemented as a single
# forward_backward function, no nn.Module or autograd.
#
# Downloads the required number of pre-tokenized Climbmix dataset shards if not
# already present.
#
# Style:
# - Minimal helpers and classes to consolidate math and reduce redirects. 
# - Config `cfg` and model tensor container `m` are globals.
# - Optimizer state and learning schedules are attached to their parameters.
# - Initialization of weight values, schedules, and optimizer state is done
#   together. Everything is created directly on device.
#
# Config:
# - "WANDB_API_KEY" is the one environment variable you need to set.
#   The final script must be self-contained, no environment variable passing
#   or command line arguments in the commited baseline.
# - Recommend setting cfg.run_name (for both wandb and log files) on every run.
# - For shorter tests, see these flags:
#     ABORT_STEP
#     cfg.use_wandb
#
# Grep for "§" to retrieve the document outline.

# ==============================================================================
# § Setup
# ==============================================================================

import os
import sys
import time as _time
run_wall_t0 = _time.perf_counter()
del _time

with open(sys.argv[0], 'r') as f:
    code = f.read()   # the run section logs the script source to wandb
# utils.py holds the data loader; log its source too.
with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "utils.py"), 'r') as f:
    code += "\n\n# " + "=" * 78 + "\n# utils.py\n# " + "=" * 78 + "\n\n" + f.read()

import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import Tensor

from utils import (DATASET_DIR, EVAL_BUFFER_TOKENS, data_generator, download_dataset,
                   flash_attn_varlen_fwd_lse, flash_attn_varlen_bwd)

dynamo.config.recompile_limit = 64

# Confirm Ampere or newer
assert torch.cuda.is_available(), "no GPU -- Runtime > Change runtime type > A100"

props = torch.cuda.get_device_properties(0)
print(f"{props.name} | {props.total_memory / 2**30:.1f} GiB | sm{props.major}{props.minor}")
assert props.major >= 8, f"needs Ampere or newer (got sm{props.major}{props.minor})"

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# ==============================================================================
# § Configuration
# ==============================================================================

class StackConfig:
    """Model architecture constants and training configuration. Weight 
    initialization, scheduling, and optimizer parameters are defined directly in
    their code instead. This codebase generally only includes the codepath 
    executed during the run, without togglable features; the "config" is for
    referencing constants and documenting key settings."""

    # ---- Architecture ----

    # Model
    n_layers:   int = 11
    d_model:    int = 768

    backout_layer: int = 6 # nanochat: n_layers // 2

    # Input
    d_vocab:    int = 32768
    d_smr_gate: int = 24    # Gate input is first 24-dims of input embed.

    # Attention
    n_qo_heads: int = 6
    n_kv_heads: int = 6     # n_qo == n_kv means full multihead attention.
    d_qk:       int = 128   # Attention head size.
    d_vo:       int = 128   # Note: FA2 requires d_qk == d_vo, FA3 does not.

    # Context and Sliding Window Attention
    seq_len:          int = 2048
    short_win_size:   int = 768
    full_ctxt_layers: list[int] = [   3,    6,    10]
    window_sizes:     list[tuple[int, int]]  # Derived below.

    # Attention - Value Embeddings
    d_ve_gate: int = 12  # Gate input is first 12-dims of the layer's residual stream.
                         # Each head has its own gate, all with same input.
    ve_layers: list[int] = [1, 2, 8, 9, 10]
    ve_index:  list[int] # Derived from ve_layers.
    num_ves:   int

    # MLP
    d_mlp:      int = 4 * 768 # 3072

    # Model stats, for MFU and logging
    num_params:          int = 286_261_730     # every trained weight (§ Weight Init & Schedule)
    num_flops_per_token: int = 780_929_568     # 6 * 110,100,912 matmul params + attention

    # ---- Training ----

    # Batch Size
    micro_batch_tokens: int = 2**18   # 256K tokens per micro-batch
    total_batch_size:   int = 2**19   # 512K tokens per step
    grad_accum_steps:   int

    # Training
    num_steps: int = 1035

    # Evaluation and logging
    val_loss_every:  int = 125
    val_tokens:      int = 10485760   # 10M tokens per val-bpb pass
    val_steps:       int              # Derived: val micro-batches per pass.

    # Logging
    wandb_project:   str = "decoderstack_rtx"  # baselines only
    run_name:        str = "baseline"  # both wandb and log files
    use_wandb:       bool = True

    save_checkpoint: bool = False
    save_steps:      tuple = ()

    seed:            int  = 42      # For model initialization

cfg = StackConfig() # Make config a global, don't pass it around.

# ==============================================================================
# § Derived Configs
# ==============================================================================

# Set this to run training under the normal schedule and have it abort
# part way. Great way to test things out without changing num_steps.
# None = run the full num_steps; 0 = validate the init and exit.
ABORT_STEP = None

# Torch profiler: trace steps 12-13 (post-compile, post step-10 log hooks),
# then abort at 14. Rank 0 writes logs/<run_name>_trace.json.gz -- view at
# ui.perfetto.dev.
PROFILE = False

if cfg.use_wandb:
    assert "WANDB_API_KEY" in os.environ, "cfg.use_wandb=True but WANDB_API_KEY not set"
    wandb.login(key=os.environ["WANDB_API_KEY"])

# Map layers to VE bank slots.
cfg.ve_index = [cfg.ve_layers.index(i) if i in cfg.ve_layers else -1 for i in range(cfg.n_layers)]
cfg.num_ves = len(cfg.ve_layers)

# Per-layer window sizes for sliding window attention, defined as (left, right)
# tuples. Left means number of tokens to attend to to the left of current
# position, and right is 0 for causal.
cfg.window_sizes = [(cfg.short_win_size, 0)] * cfg.n_layers  # All short, ...
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (cfg.seq_len, 0)                   # ... then overwrite with full.

cfg.grad_accum_steps = cfg.total_batch_size // cfg.micro_batch_tokens
cfg.val_steps =        cfg.val_tokens       // EVAL_BUFFER_TOKENS

VAL_BPB_TARGET = 0.900

gpu_device_name = torch.cuda.get_device_name(0)   # "NVIDIA RTX PRO 6000 Blackwell Server Edition"
# Dense BF16 peak FLOPS of the RTX PRO 6000, the MFU denominator.
gpu_peak_flops = 503.8e12

download_dataset()


# Reject a vocab mismatch between the .bin shards and the model.
with open(os.path.join(DATASET_DIR, "config.json")) as f:
    assert json.load(f)["vocab_size"] == cfg.d_vocab, "dataset vocab != model d_vocab"

# token_bytes: per-token-id byte lengths (0 for special tokens), for the
# vocab-size-independent bits-per-byte validation metric.
with open(os.path.join(DATASET_DIR, "tokenizer/token_bytes.pt"), "rb") as f:
    token_bytes = torch.load(f, map_location=device)


# ==============================================================================
# § Data Structures
# ==============================================================================

# NamedTuples can be passed to compiled functions.
class Param(NamedTuple):
    """Model parameter bundled with everything needed for training it."""

    name:         str
    w:            Tensor    # The actual weight

    # Optimizer State
    mantissa:     Tensor    # Larry Dial's trick for storing an fp32 master
    grad:         Tensor    # Matches full weight size
    gbank:        list      # Banked weights unbound into a list
    first_mntm:   Tensor
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
    For T=256K  -->  Held: 49.5GB,  Recomputed:  24.75GB
    """
    #                                                            Stash (Tiny) Recompute
    x_in:               Tensor    # (L,  T,    D)              4.5GB
    x_biased_hat:       Tensor    # (L,  T,    D)              4.5GB
    x_biased_inv_rms:   Tensor    # (L,  T,    1)         fp32          (3MB)
    q_hat:              Tensor    # (L,  T, n_qo, d_qk)        4.5GB
    k_hat:              Tensor    # (L,  T, n_kv, d_qk)        4.5GB
    q_inv_rms:          Tensor    # (L,  T, n_qo,    1)   fp32         (18MB)
    k_inv_rms:          Tensor    # (L,  T, n_kv,    1)   fp32         (18MB)
    #ve:                Tensor    # (Lv, T, n_kv, d_vo)                        2.25GB
    #ve_gate_a:         Tensor    # (Lv, T, n_kv)                               (18MB)
    v:                  Tensor    # (L,  T, n_kv, d_vo)        4.5GB
    y:                  Tensor    # (L,  T, n_qo, d_vo)        4.5GB
    lse:                Tensor    # (L,  n_qo,  T)        fp32         (18MB)
    x_attn_out:         Tensor    # (L,  T,     D)             4.5GB
    #x_attn_out_hat:    Tensor    # (L,  T,     D)                              4.5GB
    mlp_relu:           Tensor    # (L,  T, d_mlp)              18GB
    #mlp_a:             Tensor    # (L,  T, d_mlp)                               18GB
    #                                                         ------           ------
    #                                                 TOTAL:  49.5GB          24.75GB

# Model-level activations held as locals:
#   x0                 (T, D)          384MB    layer-blend + smear backward
#   x_embed_hat        (T, D)          384MB    smear + embedding-norm backward
#   x_embed_inv_rms    (T, 1)   fp32   (1.2MB)
#   x_backout          (T, D)          384MB    backout backward
#   x_final_hat        (T, D)          384MB    lm_head grad + final-norm backward
#   x_final_inv_rms    (T, 1)   fp32   (1.2MB)
#                               TOTAL: 1.5GB

# Cast shorthands for the bodies below: the fp32 scalars/gates need explicit
# bf16 casts at their use sites (see forward_backward's docstring), and the
# scalar-parameter grad sums accumulate in fp32.
bf16  = lambda x: x.to(torch.bfloat16)
sum32 = lambda x: x.sum(dtype=torch.float32)


# ==============================================================================
# § Train Forward + Backward
# ==============================================================================

@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def forward_backward(idx, targets, cu_seqlens, micro_step, loss_scale=1.0, backward=True):
    """One micro-batch through the model. Use backward=False for validation."""

    # Direction naming:
    # A bare name is the forward pass; a 'b' on its leading symbol is the
    # backward pass -- the gradient w.r.t. that exact tensor. Any tags after
    # the symbol carry over unchanged:
    #   x -> xb,  q -> qb,  q_hat -> qb_hat,  x_final_hat -> xb_final_hat.
    # The leading symbol can be more than one word -- 've_gate' is the core,
    # so its grads are ve_gateb_*, not veb_gate_* ('veb' is the value embeds).
    #
    # Stream stages (forward name; each has a matching xb_* in bwd). Each is
    # named for what was just done to it:
    # x_embed    - The input embedding
    # x0         - The smeared embedding, re-read by every layer
    # x          - The running residual stream (x_in is this layer's input)
    # x_biased   - Layer input scaled by resid_lambda, biased by x0
    # x_attn_out - Post-attention stream, the MLP's input
    # x_final    - Post-backout stream, feeding the lm_head
    #
    # RMS naming:
    # *_inv_rms  - 1/rms, used by RMS norm fwd and bwd.
    # *_hat      - RMS-normed  (x_hat = x * inv_rms)
    #
    # Activation naming -- the MLP and both gates share three stages:
    # *_z        - Pre-nonlinearity, the raw matmul output. Only ever named in
    #              bwd (mlpb_z, ve_gateb_z, gateb_z); inlined in fwd.
    #              ('logit' is reserved for the lm_head's output.)
    # *_relu     - Post-relu, pre-square (MLP only; the stashed half of relu^2).
    # *_sig      - Post-sigmoid, in [0, 1] (gates only; recomputed in bwd).
    # *_a        - The final activation handed onward: mlp_a = mlp_relu^2,
    #              ve_gate_a = 3*ve_gate_sig. The smear gate's is just 'gate'.

    assert idx.ndim == 1
    T = idx.size(0)
    half = cfg.d_qk // 2

    assert T > 1, "Forward pass should have T > 1 (smear needs a previous token)"
    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"

    cos, sin = m.cos[0, :T], m.sin[0, :T]  # (T, 1, half)
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)
    x_backout = None

    # -----------------------------
    #           Forward
    # -----------------------------

    # Input embeddings
    x_embed = F.embedding(idx, m.input_embeds.w)        # bf16

    # RMS normalize the input embeds
    x_embed_inv_rms = (x_embed.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x_embed_hat = bf16(x_embed.float() * x_embed_inv_rms)      # post-norm embedding, pre-smear

    # Smear: mix the previous token's embedding into the current position.
    # The sigmoid and its argument are unnamed here; bwd recomputes them as
    # gate_sig and (as a grad) gateb_z.
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(
        x_embed_hat[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)
    # The smeared input embedding; also added back to the stream at every layer.
    x0 = torch.cat([x_embed_hat[:1], x_embed_hat[1:] + gate * x_embed_hat[:-1]], dim=0)  # appears as x0b in bwd

    # The residual stream starts as the smeared embedding.
    x = x0

    # One LayerStash of forward activations per layer
    stash = []

    # For each layer,
    for i in range(cfg.n_layers):
        x_in = x

        # Scale residual stream, add input embedding
        x_biased = m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
        x_biased_inv_rms = (x_biased.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        x_biased_hat = bf16(x_biased.float() * x_biased_inv_rms)

        # QKV Projections
        q = (x_biased_hat @ m.W_Q.w[i].mT).view(T, cfg.n_qo_heads, cfg.d_qk)
        k = (x_biased_hat @ m.W_K.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (x_biased_hat @ m.W_V.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)

        # Value Embeddings
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            ve_gate_a = 3 * ve_gate_sig          # (T, n_kv_heads), in [0, 3]
            v = v + ve_gate_a.unsqueeze(-1) * ve # ve and the gate are both recomputed in bwd, not stashed

        # RoPE
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)

        # QK-Norm and Sharpening
        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q_hat = bf16(q.float() * q_inv_rms * 1.2) # 1.2^2 - Similar to temperature of 0.7
        k_hat = bf16(k.float() * k_inv_rms * 1.2)

        # Read V from past residual streams by matching their K.
        y, lse = flash_attn_varlen_fwd_lse(q_hat, k_hat, v, cu_seqlens, cfg.seq_len, cfg.window_sizes[i])
        y = y.contiguous()

        # Project value heads onto their output heads.
        attn_out = y.view(T, -1) @ m.W_O.w[i].mT

        # Write back to the stream.
        x_attn_out = x_biased + attn_out

        # MLP input norm
        # Recomputed in backward pass to save memory.
        x_attn_out_hat = bf16(x_attn_out.float() * (x_attn_out.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())

        # MLP
        mlp_relu = F.relu(x_attn_out_hat @ m.W_in.w[i].mT)
        mlp_a = mlp_relu.square()    # (T, d_mlp) - (64K, 3K) recomputed
        mlp_out = mlp_a @ m.W_out.w[i].mT

        # Write back to the stream.
        x = x_attn_out + mlp_out            # the residual stream; appears as xb in bwd

        # Stash the backout layer's output, to subtract it off before LM head.
        if i == cfg.backout_layer:
            x_backout = x

        # Stash activations for backward pass. Skipped entirely under eval.
        if backward:
            stash.append(LayerStash(x_in=x_in, x_biased_hat=x_biased_hat, x_biased_inv_rms=x_biased_inv_rms,
                                    q_hat=q_hat, k_hat=k_hat, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
                                    v=v, y=y, lse=lse, x_attn_out=x_attn_out, mlp_relu=mlp_relu))

    x_final = x - bf16(m.backout_lambda.w) * x_backout # TODO - backout lambda could be bf16 instead.

    # Final output norm
    x_final_inv_rms = (x_final.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x_final_hat = bf16(x_final.float() * x_final_inv_rms)

    # -----------------------------
    #           LM Head
    # -----------------------------
    tgt = targets.unsqueeze(1)  # (T, 1)  TODO - Have the caller pass the right shape.

    # ==== Forward ====
    logits_raw = x_final_hat @ m.lm_head.w.mT  # (T, d_vocab) bf16, 4GB at (64K, 32K)

    # "softcap" logits to the range -15 to 15
    logits = 15.0 * torch.tanh(logits_raw.float() / 15.0) # (T, d_vocab)

    # Typically, subtract off the highest logit per token first:
    #    max_logit = logits.amax(dim=1, keepdim=True) # (T, 1)
    #    e = (logits - max_logit).exp()  # (T, d_vocab) = ((T, d_vocab) - (T, 1)).exp()
    # Softcap bounds to [-15, 15], so exp() is [3.06e-7, 3.3e6], so fp32 is ok.
    e = logits.exp()  # (T, d_vocab)

    # Softmax denominator
    ssum = e.sum(dim=1, keepdim=True) # (T, 1)

    # ==== Cross Entropy ====
    # Convert back to logit space
    lse_ce = ssum.log().squeeze(1) # (T, 1)
    # Select prediction logits for target token (one random access per row)
    tgt_logit = logits.gather(1, tgt).squeeze(1) # (T,)
    # If validation pass, return CE in nats.
    if not backward:
        return (lse_ce - tgt_logit) # (T,)
    # Return training loss for logging.
    loss = (lse_ce - tgt_logit).mean()   # Return for tracking training loss

    # ==== Backward ====
    onehot = torch.arange(cfg.d_vocab, device=device).unsqueeze(0) == tgt # (T, d_vocab)

    # Predicted probs = e / ssum; (T, d_vocab)
    # Standard approach to backprop is (p - 1), with optimizer update w = w - grad.
    # We flip this to (1 - p) and w = w + grad, so that:
    # - grads and optim state move in the same direction as their param.
    # - target token contributes positively to stream and gradients.
    logitsb = bf16((onehot.float() - (e / ssum)) * (1.0 - logits/15.0 * logits/15.0) * loss_scale)

    # Every token updates every vocab entry.
    m.lm_head.grad.add_((logitsb.mT @ x_final_hat).float()) # (d_vocab, T) @ (T, d_model) --> (d_vocab, d_model)

    # The backward streams start as weighted sums of the head embeddings
    # that they (meaningfully) predicted.
    xb_final_hat = logitsb @ m.lm_head.w # (T, d_vocab) @ (d_vocab, d_model)
    del logitsb

    # -----------------------------
    #           Backward
    # -----------------------------
    # Only compute some grads on the last micro batch.
    is_last_micro = (micro_step == cfg.grad_accum_steps - 1)

    # Scalar grads are collected, grad tensors updated at the end.
    g_resid = []; g_x0 = []

    # Each stream's cosine similarity to the vocab signal, divided by d_model.
    # (T, 1) = ((T, d_model) * (T, d_model)).mean()
    res_ms = (x_final_hat.float() * xb_final_hat.float()).mean(dim=-1, keepdim=True)

    # The component of xb_final_hat which is perpendicular to x_final_hat?
    xb_final = bf16(x_final_inv_rms * (xb_final_hat.float() - (x_final_hat.float() * res_ms)))

    # Dot product between final vs. backout streams.
    if is_last_micro:
        m.backout_lambda.grad.add_(-sum32(xb_final * x_backout))  # (T, d_model)

    # xb updates every layer, keep xb_final for backout layer.
    xb = xb_final               # grad wrt layer num_layers-1's output
    x0b = torch.zeros_like(x0)  # accumulates over layers

    # For each layer in reverse order,
    for i in reversed(range(cfg.n_layers)):

        # Stashed forward activations for this layer.
        st = stash[i]

        if i == cfg.backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes num_layers//2
            xb = xb - bf16(m.backout_lambda.w) * xb_final

        # --- MLP backward (relu^2: mlpb_z = 2*mlp_relu*mlpb_a, self-masking
        #     since mlp_relu is already 0 where z < 0) ---

        # Grad w.r.t. W_out
        mlp_a = st.mlp_relu.square()  # (T, d_mlp)
        m.W_out.gbank[i].add_(xb.mT @ mlp_a) # (d_model, T) @ (T, d_mlp)

        # Grad w.r.t. the pre-relu matmul output (mlpb_a = xb @ W_out is inline)
        mlpb_z = 2.0 * st.mlp_relu * (xb @ m.W_out.w[i])

        # Recompute the MLP input norm
        x_attn_out_inv_rms = (st.x_attn_out.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        x_attn_out_hat = bf16(st.x_attn_out.float() * x_attn_out_inv_rms)

        # Grad w.r.t. W_in
        m.W_in.gbank[i].add_(mlpb_z.mT @ x_attn_out_hat) # (d_mlp, T) @ (T, d_model)

        xb_attn_out_hat = mlpb_z @ m.W_in.w[i] # (T, d_mlp) @ (d_mlp, d_model) --> (T, d_model)

        xb_attn_out = xb + bf16(x_attn_out_inv_rms * (xb_attn_out_hat.float() - (x_attn_out_hat.float() * (x_attn_out_hat.float() * xb_attn_out_hat.float()).mean(dim=-1, keepdim=True))))

        # Attention backward
        x_biased_hat = st.x_biased_hat
        m.W_O.gbank[i].add_(xb_attn_out.mT @ st.y.view(T, -1))
        yb = (xb_attn_out @ m.W_O.w[i]).view(T, cfg.n_qo_heads, cfg.d_vo)

        qb_hat, kb_hat, vb = flash_attn_varlen_bwd(
            yb, st.q_hat, st.k_hat, st.v, st.y, st.lse, cu_seqlens, cfg.seq_len,
            cfg.window_sizes[i])

        # per-(token, head) norm backward
        qb = bf16(st.q_inv_rms * (1.2 * qb_hat.float() - st.q_hat.float() * ((st.q_hat.float() * qb_hat.float()).mean(dim=-1, keepdim=True) / 1.2)))
        kb = bf16(st.k_inv_rms * (1.2 * kb_hat.float() - st.k_hat.float() * ((st.k_hat.float() * kb_hat.float()).mean(dim=-1, keepdim=True) / 1.2)))

        # rotary backward = rotation by -theta (transpose of the forward rotation)
        qb1, qb2 = qb[..., :half], qb[..., half:]
        kb1, kb2 = kb[..., :half], kb[..., half:]
        qb = torch.cat([qb1 * cos - qb2 * sin, qb1 * sin + qb2 * cos], dim=-1)
        kb = torch.cat([kb1 * cos - kb2 * sin, kb1 * sin + kb2 * cos], dim=-1)

        # --- VE gate backward (ve and ve_gate_sig recomputed) ---
        j = cfg.ve_index[i]
        xb_biased_hat_ve = None
        if j >= 0:
            # Retrieve the value embeddings
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)

            # Recompute gate forward
            ve_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)

            # ve_gate_a broadcasts over d_vo in v = v0 + a*ve, so its grad sums
            # that axis back out; then d/dz[3*sigmoid(z)] = 3*sig*(1 - sig).
            ve_gateb_a = (vb * ve).sum(dim=-1)   # (T, n_kv_heads)
            ve_gateb_z = ve_gateb_a * (3 * ve_gate_sig * (1 - ve_gate_sig))

            m.ve_gate.gbank[j].add_(ve_gateb_z.mT @ x_biased_hat[..., :cfg.d_ve_gate])

            # Embedding gradient
            veb = (vb * (3 * ve_gate_sig).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            # embedding_dense_backward beat raw index_add_ atomics ~2x
            m.value_embeds.gbank[j].add_(
                torch.ops.aten.embedding_dense_backward(veb, idx, cfg.d_vocab, -1, False))

            xb_biased_hat_ve = ve_gateb_z @ m.ve_gate.w[j]

        # vb passes through the VE add unchanged: v = v0 + ve_gate_a*ve
        qb = qb.view(T, cfg.n_qo_heads * cfg.d_qk)
        kb = kb.view(T, cfg.n_kv_heads * cfg.d_qk)
        vb = vb.reshape(T, cfg.n_kv_heads * cfg.d_vo)

        m.W_Q.gbank[i].add_(qb.mT @ x_biased_hat)
        m.W_K.gbank[i].add_(kb.mT @ x_biased_hat)
        m.W_V.gbank[i].add_(vb.mT @ x_biased_hat)

        xb_biased_hat = qb @ m.W_Q.w[i] + kb @ m.W_K.w[i] + vb @ m.W_V.w[i]
        if xb_biased_hat_ve is not None:
            xb_biased_hat[:, :cfg.d_ve_gate] += xb_biased_hat_ve
        xb_biased = xb_attn_out + bf16(st.x_biased_inv_rms * (xb_biased_hat.float() - (x_biased_hat.float() * (x_biased_hat.float() * xb_biased_hat.float()).mean(dim=-1, keepdim=True))))
        # --- blend backward: x_biased = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        if is_last_micro:
            g_resid.append(sum32(xb_biased * st.x_in))
            g_x0.append(sum32(xb_biased * x0))
        x0b = x0b + m.x0_lambdas.w[i] * xb_biased  # TRAP: x0 feeds every layer, accumulate
        xb = m.resid_lambdas.w[i] * xb_biased
        stash[i] = None                          # free this layer's stash as we go

    # Land the per-layer resid/x0 scalar sums (collected in REVERSED layer
    # order) as one stacked add each.
    if is_last_micro:
        m.resid_lambdas.grad.add_(torch.stack(g_resid[::-1]))
        m.x0_lambdas.grad.add_(torch.stack(g_x0[::-1]))

    # xb is now the grad through layer 0's input, which IS x0 (same tensor), so
    # it folds into x0b to give the full grad wrt the smeared embedding.
    x0b = x0b + xb

    # --- smear backward: x0 = cat([x_embed_hat[:1], x_embed_hat[1:] + gate*x_embed_hat[:-1]]) ---
    gate_sig = torch.sigmoid(x_embed_hat[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)  # (T-1, 1), recomputed
    gate = bf16(m.smear_lambda.w) * gate_sig
    xb_embed_hat = x0b.clone()
    xb_embed_hat[:-1] += gate * x0b[1:]  # TRAP: shifted scatter -- p's grad reaches p-1
    gateb = (x0b[1:] * x_embed_hat[:-1]).sum(dim=-1, keepdim=True)   # (T-1, 1)
    m.smear_lambda.grad.add_(sum32(gateb * gate_sig))
    gateb_z = gateb * bf16(m.smear_lambda.w) * gate_sig * (1 - gate_sig)
    m.smear_gate.grad.add_((gateb_z.mT @ x_embed_hat[1:, :cfg.d_smr_gate]).float())
    xb_embed_hat[1:, :cfg.d_smr_gate] += gateb_z @ bf16(m.smear_gate.w)

    # --- embedding norm + token embedding scatter ---
    xb_embed = bf16(x_embed_inv_rms * (xb_embed_hat.float() - (x_embed_hat.float() * (x_embed_hat.float() * xb_embed_hat.float()).mean(dim=-1, keepdim=True))))
    m.input_embeds.grad.add_(
        torch.ops.aten.embedding_dense_backward(xb_embed, idx, cfg.d_vocab, -1, False))

    return loss

# ==============================================================================
# § Optimizer Math
# ==============================================================================

# Helpers for Master vs. Live via Mantissa
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

# ------------------------------------------------------------------------------
# AdamW
# ------------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Param,
    grad: Tensor,
    t: Tensor,      # (1,) Current step for schedules
) -> None:
    """AdamW update of `p`."""

    # ==== Buffer Update ====
    grad = grad.float() # Some grads are bf16, EMAs are fp32.

    # Update EMAs. Mix a large portion of the tracked value with a small portion
    # of the current gradient.
    p.first_mntm.mul_(p.mntm_b1_t[t]).add_(grad * p.grad_b1_t[t])         # m = beta1*m + (1 - beta1)*g
    p.scnd_mntm.mul_(p.mntm_b2_t[t]).add_(grad.square() * p.grad_b2_t[t]) # v = beta2*v + (1 - beta2)*g^2

    # ==== Parameter Update ====
    if p.mantissa is not None:
        master = rebuild_master(p.w, p.mantissa)
    else:
        master = p.w.float()

    # Apply weight decay inplace
    master.mul_(p.wd_t[t])

    # Apply AdamW's update inplace, w = w + lr * (m / (sqrt(v) + eps))
    master.add_(p.lr_bc_t[t] * (p.first_mntm / (p.scnd_mntm.sqrt() + p.eps_t[t])))

    # Re-split the weight.
    if p.mantissa is not None:
        writeback_master(master, p.w, p.mantissa)
    else:
        p.w.copy_(master)


# ------------------------------------------------------------------------------
# Muon
# ------------------------------------------------------------------------------

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
    grad: Tensor,   # (K, out, in) fp32 gradient -- MUTATED (nesterov lerp)
    t: Tensor,      # (1,) int64 device tensor - the schedule row to read
) -> None:
    """Fused Muon step on `p`: momentum -> polar_express -> variance_reduction
    -> cautious update on the reconstructed master."""

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
    live = p.w
    master = rebuild_master(live, p.mantissa)
    # Decay still has to shrink the weight, so it stays negative under the
    # flipped update; the gate is on `g` pointing at zero, which is now `<= 0`.
    mask = (g * master) <= 0
    master.add_(p.lr_bc_t[t] * g - p.wd_t[t] * master * mask)
    writeback_master(master, live, p.mantissa)


# ==============================================================================
# § Weight Init & Schedule
# ==============================================================================

# ------------------------------------------------------------------------------
# Bigram Prior
# ------------------------------------------------------------------------------

# Corpus bigram counts (all 91 train shards): bigram_counts[i, j] = how often token j follows token i.
bigram = np.load(os.path.join(DATASET_DIR, "tokenizer/bigram_counts.npz"))
bigram_counts = torch.zeros(cfg.d_vocab, cfg.d_vocab, dtype=torch.float32, device=device)
bigram_counts[torch.from_numpy(np.repeat(np.arange(cfg.d_vocab), np.diff(bigram["train_indptr"]))).to(device),
              torch.from_numpy(bigram["train_indices"].astype(np.int64)).to(device)] = \
    torch.from_numpy(bigram["train_data"].astype(np.float32)).to(device)
del bigram

context_counts = bigram_counts.sum(dim=1, keepdim=True)                                            # (V, 1)
next_unigram   = (bigram_counts.sum(dim=0) + 0.5) / (bigram_counts.sum() + 0.5 * cfg.d_vocab)     # (V,)

# Smoothed next-token distribution of every context: 3,000 pseudo-counts of the unigram.
log_bigram = ((bigram_counts + 3000.0 * next_unigram) / (context_counts + 3000.0)).log()          # (V, V)
del bigram_counts

# The softcapped logits the direct path should produce: each context's best next token at +10.
log_bigram -= log_bigram.max(dim=1, keepdim=True).values - 10.0
log_bigram.clamp_(-14.25, 14.25)
raw_target = 15.0 * torch.atanh(log_bigram / 15.0)                                                 # pre-softcap
del log_bigram

# Rank-768 factorization, every context weighted by how often it occurs.
context_weight = (context_counts / context_counts.sum() + 1e-9).sqrt()                            # (V, 1)
torch.manual_seed(cfg.seed + 1)   # the randomized SVD's test matrix; the weights re-seed below
U, S, _ = torch.svd_lowrank(context_weight * raw_target, q=cfg.d_model + 64, niter=4)
embed_prior = U[:, :cfg.d_model] * S[:cfg.d_model].sqrt() / context_weight                          # (V, D)
del U, S

# The RMS norm keeps only each embedding row's direction: put every row at the stock norm
# 0.8 * sqrt(D), then solve the head by weighted least squares against the normed rows.
embed_prior *= 0.8 * cfg.d_model ** 0.5 / embed_prior.norm(dim=1, keepdim=True)
xe_prior   = embed_prior / 0.8                                                                     # normed rows
head_prior = torch.linalg.solve((xe_prior.T @ (context_weight ** 2 * xe_prior)).double(),
                                ((context_weight ** 2 * xe_prior).T @ raw_target).double()).float().T   # (V, D)
del raw_target, xe_prior, context_counts, context_weight, next_unigram

# Cosine of each normed embedding with its bigram direction; the rest is the stock random draw.
prior_cos = 1.0
embed_prior *= prior_cos
head_prior  /= prior_cos


# ------------------------------------------------------------------------------
# LR Schedule
# ------------------------------------------------------------------------------

# Learning rate schedule as a per-step multiplier.
# Shared by Muon and AdamW. No warmup: peak from step 0.
lr_mult_t = np.ones(cfg.num_steps)

steps_0idx = np.arange(cfg.num_steps, dtype=np.float64)  # 0-based, the way the loop counts
steps_1idx = steps_0idx + 1.0                            # 1-based, the way bias corrections count

# Warmdown for 65% of the run.
warmdown_len  = round(0.65 * cfg.num_steps)
warmdown      = slice(cfg.num_steps - warmdown_len + 1, cfg.num_steps)   # the hold covers everything before
warmdown_frac = (cfg.num_steps - steps_0idx[warmdown]) / warmdown_len  # ~1 -> ~0 across the warmdown

lr_mult_t[warmdown] = 0.05 + (1.0 - 0.05) * warmdown_frac


# ------------------------------------------------------------------------------
# Common
# ------------------------------------------------------------------------------

m = Model()

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

fp32_empty   = lambda *shape: torch.empty(*shape, dtype=torch.float32, device=device)
bf16_empty   = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=device)
fp32_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
bf16_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16, device=device)

# Uniform init bound. Var(Uniform(-a, a)) = a^2/3, so std = a/sqrt(3): to hit
# a target std of 1/sqrt(d_model), the bound must be sqrt(3) times it.
matrix_init_s = (3 ** 0.5) * (cfg.d_model ** -0.5)

upper_bf16   = lambda w: (w.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
lower_uint16 = lambda w: (w.contiguous().view(torch.int32)      ).to(torch.int16).view(torch.uint16)

# Create an fp32 tensor on the device.
dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------
# Scalars
# ------------------------------------------------------------------------------

resid_lambdas  = torch.linspace(1.15, 1.05, cfg.n_layers, dtype=torch.float32, device=device)
x0_lambdas     = torch.linspace(0.20, 0.05, cfg.n_layers, dtype=torch.float32, device=device)
smear_lambda   = fp32_zeros(1)
backout_lambda = fp32_empty(1).fill_(0.2)

smear_gate     = fp32_empty(1, cfg.d_smr_gate).uniform_(-cfg.d_smr_gate ** -0.5, cfg.d_smr_gate ** -0.5)

# Grad mix-in coefficient (1-beta), currently fixed, can be scheduled here.
scalar_grad_mult_t = np.ones(cfg.num_steps)


scalar_configs = [
#   name,               weights,       peak lr,  b1_grad,   b2_grad,    wd,
    ("resid_lambdas",   resid_lambdas,   0.005,    0.2,        0.05,   0.05),
    ("x0_lambdas",      x0_lambdas,      0.5,      0.04,       0.05,   0.0),
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
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(w.shape),

        residual_dim = None, # Muon only

        # Schedules
        # Fold bias correction into the learning rate.
        lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

        # Weight decay schedule
        wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

        # Beta schedule, constant (the mix-in multiplier is all ones)
        mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),   # first_mntm decay (Beta1)
        grad_b1_t    = dev(scalar_grad_mult_t * b1_grad),      # first_mntm grad mix-in (1-Beta1)

        mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
        grad_b2_t    = dev(scalar_grad_mult_t * b2_grad),      # scnd_mntm grad mix-in (1-Beta2)

        eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5), # May not be necessary
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)

# ------------------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------------------

input_embeds =     bf16_empty(cfg.d_vocab, cfg.d_model)
input_embeds.copy_(fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.8 * (1.0 - prior_cos ** 2) ** 0.5).add_(embed_prior))
value_embeds =     bf16_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
value_embeds.copy_(fp32_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
                   .uniform_(-matrix_init_s, matrix_init_s))

ve_rows = cfg.num_ves * cfg.d_vocab

embed_configs = [
#   name,             weights,       peak lr,  b1_grad,  b2_grad,  wd,       slots
    ("input_embeds",  input_embeds,  0.3,      0.2,      0.005,    0.001,   1),
    ("value_embeds",  value_embeds,  0.15,     0.2,      0.005,    0.01,   cfg.num_ves)
]

# For each of the embedding tables...
for (name, w, peak_lr, b1_grad, b2_grad, wd, slots) in embed_configs:

    # Derive the momentum buffers' decays.
    b1_mntm = 1 - b1_grad
    b2_mntm = 1 - b2_grad

    grad = bf16_zeros(w.shape) # Embeddings can handle bf16 accumulation fine

    p = Param(
        # Weight
        name         = name,
        w            = w,          # bf16 live, no fp32 master
        mantissa     = None,

        # Gradients
        grad         = grad, 
        gbank        = list(grad.view(slots, cfg.d_vocab, -1).unbind(0)) if slots > 1 else None,

        # Momentum buffers
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(w.shape),

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
# LM Head
# ------------------------------------------------------------------------------

lm_head = fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.001).add_(head_prior)

# Per-step multiplier on the head's grad mix-ins (1-beta). Constant: no warmup ramp.
lm_grad_mult_t = np.ones(cfg.num_steps)

peak_lr = 0.008
b1_grad = 0.2    # (1-Beta1)
b2_grad = 0.04   # (1-Beta2)
wd      = 0.01

# Derive the momentum buffers' decays.
b1_mntm = 1 - b1_grad
b2_mntm = 1 - b2_grad

# Split the fp32 draw into bf16 live + stashed low bits.
live = upper_bf16(lm_head)

m.lm_head = Param(
    # Weight
    name         = "lm_head",
    w            = live,
    mantissa     = lower_uint16(lm_head),

    # Gradients
    grad         = fp32_zeros(live.shape),
    gbank        = None,

    # Momentum buffers
    first_mntm   = fp32_zeros(live.shape),
    scnd_mntm    = fp32_zeros(live.shape),

    residual_dim = None, # Muon only

    # Schedules
    # Fold bias correction into the learning rate.
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

    # Weight decay schedule
    wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

    # Beta schedule, constant (the mix-in multiplier is all ones)
    mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),   # first_mntm decay (Beta1)
    grad_b1_t    = dev(lm_grad_mult_t * b1_grad),          # first_mntm grad mix-in (1-Beta1)

    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(lm_grad_mult_t * b2_grad),          # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)


# ------------------------------------------------------------------------------
# Attention & MLPs
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
muon_wd[0] = 0.28
run_frac = (cfg.num_steps - steps_0idx[1:]) / cfg.num_steps
muon_wd[1:] = 0.28 * (0.5 * (1.0 + np.cos(math.pi * (1.0 - run_frac))))

# Muon peak lr is 0.02, scaled up for tall matrices by their sqrt(fan_out/fan_in)
# aspect ratio -- at d12 only W_in (the 4x MLP expansion -> 2.0). rdim is the
# axis facing the residual stream: W_O and W_out live transposed -> -2; the
# ve_gate rows read a d_ve_gate slice of the stream -> -1.
muon_configs = [
#    name,      weights,  peak lr,  rdim
    ("W_Q",     W_Q,      0.02,      -1),
    ("W_K",     W_K,      0.02,      -1),
    ("W_V",     W_V,      0.02,      -1),
    ("W_O",     W_O,      0.02,      -2),
    ("W_in",    W_in,     0.04,      -1),
    ("W_out",   W_out,    0.02,      -2),
    ("ve_gate", ve_gate,  0.02,      -1)
]

# For each of the Muon-trained weight banks...
for (name, w, peak_lr, rdim) in muon_configs:

    # Split the fp32 draw into bf16 live + stashed low bits.
    live = upper_bf16(w)

    grad = fp32_zeros(live.shape)

    # The NorMuon second moment holds each neuron's mean-square update -- the
    # weight's shape with the residual-facing dim `rdim` collapsed to 1 (after
    # orthogonalization only the smaller dim can carry variance).
    scnd_shape = list(live.shape)
    scnd_shape[rdim] = 1

    p = Param(
        # Weight
        name         = name,
        w            = live,
        mantissa     = lower_uint16(w),

        # Gradients
        grad         = grad,
        gbank        = list(grad.unbind(0)),

        # Momentum buffers
        first_mntm   = fp32_zeros(live.shape),
        scnd_mntm    = fp32_zeros(scnd_shape),

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

# The Params own everything now: free the fp32 draws and the prior tables, drop the adopted names.
del lm_head, input_embeds, value_embeds, resid_lambdas, x0_lambdas, smear_gate, smear_lambda, backout_lambda, W_Q, W_K, W_V, W_O, ve_gate, W_in, W_out
del embed_prior, head_prior


# ------------------------------------------------------------------------------
# Rotary Cache
# ------------------------------------------------------------------------------

rotary_seq_len = cfg.micro_batch_tokens
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
# Stats
# ------------------------------------------------------------------------------
# One row per step. The CSV header is the field list, the wandb row is the same
# fields under their panel prefixes. A field nobody wrote stays None, and both
# sinks skip it.

@dataclass
class StepStats:

    step:         int          = 0

    # Training
    loss:         float | None = None
    lr_mult_t:    float | None = None
    dt:           float | None = None   # seconds; unset for steps 0-10
    tok_per_sec:  int   | None = None   # "
    mfu:          float | None = None   # "

    # Validation, on val steps only
    bpb:          float | None = None
    eval_seconds: float | None = None
    slack:        int   | None = None

    # Run clocks, in minutes; stamped by log_step
    train_total:  float | None = None
    wall_total:   float | None = None
    eta:          float | None = None


# Field -> wandb panel. A field not named here lands under `train/`.
WANDB_GROUPS = {
    "val":  ("bpb", "eval_seconds", "slack"),
    "time": ("train_total", "wall_total", "eta"),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(StepStats)}, \
    "WANDB_GROUPS names a field StepStats does not have"


@dataclass
class RunResult:
    """The run in one row: the results panel, the result JSON, the runs table."""

    # Identity, for the JSON; wandb has these from the run name and config
    run_name:         str          = ""
    num_steps:        int          = 0
    total_batch_size: int          = 0

    # The result
    val_bpb:          float | None = None
    min_val_bpb:      float | None = None
    slack:            int   | None = None   # micro-bpb under VAL_BPB_TARGET

    # The cost, in minutes
    train_time:       float        = 0.0
    val_time:         float        = 0.0
    compile_time:     float        = 0.0
    wall_time:        float        = 0.0

    # The rate
    avg_step_time:    float        = 0.0    # seconds
    avg_mfu:          float        = 0.0    # percent of BF16 peak
    peak_mem_gb:      float        = 0.0


FINAL_IDENTITY = ("run_name", "num_steps", "total_batch_size")


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def write_checkpoint(step):
    state = {}
    for p in m:
        for attr in ("mantissa", "first_mntm", "scnd_mntm"):
            buf = getattr(p, attr)
            if buf is not None:
                state[f"{p.name}.{attr}"] = buf.cpu()

    os.makedirs(f"logs/{cfg.run_name}", exist_ok=True)
    torch.save(dict(step=step, code=code,
                    weights={p.name: p.w.cpu() for p in m}),
               f"logs/{cfg.run_name}/model_step{step:06d}.pt")
    torch.save(dict(step=step, t_step=int(t_step.item()), state=state),
               f"logs/{cfg.run_name}/optim_step{step:06d}.pt")

logfile = None
os.makedirs("logs", exist_ok=True)
logfile = f"logs/{cfg.run_name}.txt"
print(logfile)

def print0(s="", console=False):
    with open(logfile, "a") as f:
        if console:
            print(s)
        print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

print0(f"Model parameters: {cfg.num_params:,} | FLOPs/token: {cfg.num_flops_per_token:e}", console=True)
print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}", console=True)
print0(f"Total batch size: {cfg.total_batch_size:,} tokens = {cfg.micro_batch_tokens:,} tokens/micro "
       f"x {cfg.grad_accum_steps} grad accum", console=True)

# A flat row per step and the result, for analysis without wandb.
metrics_path = f"logs/{cfg.run_name}_metrics.csv"
result_path  = f"logs/{cfg.run_name}_result.json"
metrics_file = open(metrics_path, "w", newline="")
metrics_csv = csv.DictWriter(metrics_file, fieldnames=[f.name for f in fields(StepStats)])
metrics_csv.writeheader()


def log_step(stats):
    """One step's row to the CSV and to wandb, stamped with the run clocks."""
    stats.train_total = np.sum(timed) / 60
    stats.wall_total = (time.perf_counter() - run_wall_t0) / 60
    row = asdict(stats)
    metrics_csv.writerow(row)
    metrics_file.flush()
    wandb_run.log({"step": stats.step,
                   **{f"{_WANDB_PREFIX.get(k, 'train')}/{k}": v
                      for k, v in row.items() if k != "step" and v is not None}})

gc_t0 = 0.0
def gc_logging_hook(phase, info):
    """Registered at step 10: after setup, any collector run is a surprise
    worth flagging (cycle scans cost ~500ms at random steps)."""
    global gc_t0
    if phase == "start":
        gc_t0 = time.perf_counter()
    else:
        print(f"gc gen{info['generation']}: collected {info['collected']} "
              f"({(time.perf_counter() - gc_t0) * 1000:.0f}ms)", flush=True)

if not cfg.use_wandb:
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
    wandb.define_metric("step", hidden=True)   # the x-axis, not a series
    wandb.define_metric("*", step_metric="step")

profiler = None
if PROFILE:
    ABORT_STEP = 14   # wait out compile and the step-10 hooks, trace 12-13, stop
    from torch.profiler import ProfilerActivity, profile as torch_profile
    profiler = torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,
        schedule=torch.profiler.schedule(wait=11, warmup=1, active=2, repeat=1))
    profiler.__enter__()


# ==============================================================================
# § Training Loop
# ==============================================================================

# Schedule position: one (1,) int64 device tensor, advanced on-device
t_step = torch.zeros(1, dtype=torch.int64, device=device)

val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_val_time = 0.0
timed = []   # Length of each step in seconds, steps 0-10 excluded.
             # Total training time = np.sum(timed).
warmup = []  # Those first 11 steps, where the compile lives.

train_loader = data_generator("train", cfg.seq_len, cfg.micro_batch_tokens,
                              cfg.num_steps * cfg.grad_accum_steps)

inputs, targets, cu_seqlens = next(train_loader)   # kick off the first batch

# Training loop
for step in range(cfg.num_steps + 1):
    # An abort cuts the loop but not the schedules.
    # Set ABORT_STEP=0 to run validation only.
    last_step = step == (cfg.num_steps if ABORT_STEP is None else ABORT_STEP)

    stats = StepStats(step=step)

    # --------------- Validation Loop -----------------
    if last_step or (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0):
        torch.cuda.synchronize()
        val_t0 = time.perf_counter()
        val_loader = data_generator("val", cfg.seq_len, cfg.micro_batch_tokens)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        for micro_i in range(cfg.val_steps):
            v_inputs, v_targets, v_cu_seqlens = next(val_loader)
            loss_flat = forward_backward(v_inputs, v_targets, v_cu_seqlens, micro_step=micro_i, backward=False)
            num_bytes_flat = token_bytes[v_targets]
            total_nats += (loss_flat * (num_bytes_flat > 0)).sum()
            total_bytes += num_bytes_flat.sum()
        del val_loader
        val_bpb = total_nats.item() / (math.log(2) * total_bytes.item())
        min_val_bpb = min(min_val_bpb, val_bpb)
        val_elapsed = time.perf_counter() - val_t0
        total_val_time += val_elapsed
        print0(f"step:{step}/{cfg.num_steps} val_bpb:{val_bpb:.6f} val_time:{val_elapsed:.2f}s", console=True)
        stats.bpb, stats.eval_seconds = val_bpb, val_elapsed
        stats.slack = round((VAL_BPB_TARGET - val_bpb) * 1e6)

    # --------------- Checkpoint -----------------
    if cfg.save_checkpoint and (last_step or step in cfg.save_steps):
        ckpt_t0 = time.perf_counter()
        write_checkpoint(step)
        print0(f"checkpoint captured at step {step} ({time.perf_counter() - ckpt_t0:.1f}s)", console=True)

    # Exit final step after validation and checkpoint
    if last_step:
        log_step(stats)
        break

    # --------------- Training Step -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()

    # Gradient Accumulation Loop
    for micro_i in range(cfg.grad_accum_steps):

        # Forward and Backward pass
        loss = forward_backward(inputs, targets, cu_seqlens,
                                micro_step=micro_i,
                                loss_scale=1.0 / (cfg.grad_accum_steps * inputs.size(0)))

        # Next training batch
        inputs, targets, cu_seqlens = next(train_loader)

    # Smooth gradients, update weights, zero the grads

    # Muon 
    for p in (m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out, m.ve_gate):
        muon_step_fused(p, p.grad, t_step)
        p.grad.zero_()

    # AdamW
    for p in (m.lm_head, m.input_embeds, m.value_embeds, m.resid_lambdas, m.x0_lambdas, \
              m.smear_gate, m.smear_lambda, m.backout_lambda):
        # Run AdamW
        adamw_step_fused(p, p.grad, t_step)
        p.grad.zero_()

    t_step.add_(1)  # advance the schedule on-device

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
        warmup.append(dt)
        eta_str = ""

    tok_per_sec = int(cfg.total_batch_size / dt)
    mfu = 100 * cfg.num_flops_per_token * cfg.total_batch_size / dt / gpu_peak_flops

    print0(f"step {step:05d}/{cfg.num_steps:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lr_mult_t: {lr_mult_t[step]:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | total time: {np.sum(timed)/60:.2f}m{eta_str}", console=True)

    stats.loss = train_loss
    stats.lr_mult_t = float(lr_mult_t[step])
    if step > 10:   # the compile and warm-up rates are not the run's rates
        stats.dt, stats.tok_per_sec, stats.mfu = dt, tok_per_sec, mfu
        stats.eta = remaining_time
    log_step(stats)

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
# Results
# ------------------------------------------------------------------------------

if profiler is not None:
    profiler.__exit__(None, None, None)
    trace_path = f"logs/{cfg.run_name}_trace.json.gz"
    profiler.export_chrome_trace(trace_path)
    print0(f"chrome trace -> {trace_path} (ui.perfetto.dev)", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

metrics_file.close()

# The warm-up steps cost the compile plus whatever they were going to cost anyway.
avg_step_time = float(np.mean(timed)) if timed else 0.0
compile_time = max(0.0, np.sum(warmup) - len(warmup) * avg_step_time)

result = RunResult(
    run_name         = cfg.run_name,
    num_steps        = cfg.num_steps,
    total_batch_size = cfg.total_batch_size,

    train_time       = float(np.sum(timed)) / 60,
    val_time         = total_val_time / 60,
    compile_time     = compile_time / 60,
    wall_time        = (time.perf_counter() - run_wall_t0) / 60,

    avg_step_time    = avg_step_time,
    avg_mfu          = (100 * cfg.num_flops_per_token * cfg.total_batch_size
                        / avg_step_time / gpu_peak_flops) if avg_step_time else 0.0,
    peak_mem_gb      = torch.cuda.max_memory_reserved() / 2**30,
)
if val_bpb is not None:
    result.val_bpb, result.min_val_bpb = val_bpb, min_val_bpb
    result.slack = round((VAL_BPB_TARGET - val_bpb) * 1e6)

row = asdict(result)
with open(result_path, "w") as f:
    json.dump(row, f, indent=1)
wandb_run.log({"step": cfg.num_steps,
               **{f"final/{k}": v for k, v in row.items()
                  if k not in FINAL_IDENTITY and v is not None}})

print0(f"== {result.run_name} ==", console=True)
if result.val_bpb is not None:
    print0(f"  val_bpb {result.val_bpb:.6f} | min {result.min_val_bpb:.6f} | slack {result.slack:+,} vs {VAL_BPB_TARGET:.3f}", console=True)
print0(f"  train {result.train_time:.2f}m | val {result.val_time:.2f}m | compile {result.compile_time:.2f}m | wall {result.wall_time:.2f}m", console=True)
if timed:
    print0(f"  {cfg.total_batch_size:,} tokens/step over {len(timed):,} timed steps: mean {result.avg_step_time:.3f}s | median {np.median(timed):.3f}s | {int(cfg.total_batch_size / result.avg_step_time):,} tok/s | mfu {result.avg_mfu:.2f}%", console=True)
print0(f"  rows -> {metrics_path} | result -> {result_path}", console=True)

for _f in (logfile, metrics_path, result_path):
    wandb_run.save(_f, policy="now")
wandb_run.finish()
