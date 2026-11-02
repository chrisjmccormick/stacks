# train_decision.py
#
# Qwen3.5-9B fine-tuned into a decision model with LoRA on one RTX PRO 6000, with the
# model code implemented as hand-written forward and backward functions, no nn.Module or
# autograd. The weights are Qwen's, banked by data/prepare_model.py; the data is
# data/prepare_decision_data.py's: d3-flash's prompt format in Qwen's tokens, every
# question one row, its options listed under letter codes and all of them read at the
# row's last token against a trained readout that starts from the LM head's code rows.
#
# Style:
# - Minimal helpers and classes to consolidate math and reduce redirects.
# - Config `cfg` and model tensor container `m` are globals.
# - Optimizer state is attached to its parameters; the LR schedule is a table.
# - Everything on the GPU runs under torch.compile(dynamic=False, fullgraph=True): every
#   pack has the same shape (pack_tokens tokens, max_rows segments, padded), and the two
#   library kernels (FlashAttention-2 and FLA's gated delta rule) are custom ops the
#   graph sees as opaque. Compiled code does not reproduce eager or transformers logits
#   bit for bit; what is validated is behavior, on the evaluation panel.
#
# Config:
# - "WANDB_API_KEY" is the one environment variable you need to set.
#   The script is self-contained: no environment variable passing or command line
#   arguments in the committed baseline.
# - Recommend setting cfg.run_name (for both wandb and log files) on every run.
# - For shorter tests, see these flags:
#     ABORT_STEP
#     cfg.use_wandb
#     cfg.val_splits
#
# Grep for "§" to retrieve the document outline
# ("§ " marks a chapter, "§§ " a subsection).

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

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import Tensor
from safetensors.torch import load_file

dynamo.config.recompile_limit = 64

assert torch.cuda.is_available(), "no GPU"

props = torch.cuda.get_device_properties(0)
print(f"{props.name} | {props.total_memory / 2**30:.1f} GiB | sm{props.major}{props.minor}")
assert props.major >= 8, f"needs Ampere or newer (got sm{props.major}{props.minor})"

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# FlashAttention-2 from the HF kernels hub (FA3 is not for Blackwell); the raw varlen ops
# live in flash_attn_interface.
from kernels import get_kernel
_k = get_kernel("kernels-community/flash-attn2", version=1)
fa2 = getattr(_k, "flash_attn_interface", _k)

# FLA's chunked gated delta rule, for the DeltaNet layers.
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# ==============================================================================
# § Configuration
# ==============================================================================

class DecisionConfig:
    """Model architecture constants and training configuration. The LR schedule and
    optimizer parameters are defined directly in their code instead. This codebase only
    includes the codepath executed during the run, without togglable features; the
    "config" is for referencing constants and documenting key settings."""

    # ---- Architecture: Qwen3.5-9B's text stack, as data/prepare_model.py banked it ----

    n_layers:   int = 32
    d_model:    int = 4096
    d_mlp:      int = 12288
    d_embed:    int = 248320   # embedding rows (more than the tokenizer's ids)
    rms_eps:    float = 1e-6

    # Gated DeltaNet layers: every layer but each fourth
    n_deltanet: int = 24
    n_qk_heads: int = 16
    d_qk:       int = 128
    n_v_heads:  int = 32    # each q / k head serves n_v_heads / n_qk_heads value heads
    d_v:        int = 128

    # Softmax attention layers: each fourth
    n_attention: int = 8
    n_qo_heads:  int = 16
    n_kv_heads:  int = 4
    d_head:      int = 256
    d_rotary:    int = 64   # RoPE on the first quarter of each head
    rope_theta:  float = 1e7

    # Readout: one row per answer code, scoring a question's options at its last token
    n_codes:    int = 255

    # ---- LoRA, on every projection (the fused mixer input, the mixer output, gate|up, down) ----
    rank:       int = 16
    alpha:      float = 32.0   # the update is (alpha / rank) B A

    # Model stats, for logging
    num_params:       int   # Derived: every trained value
    num_frozen_params: int  # Derived: the text stack

    # ---- Data ----

    train_splits: tuple = ("train",)   # Kev's 30,540 records; ("train", "phase1_train") adds Recast NLP's
    val_splits:   tuple = ("decision_dev", "transfer_dev", "phase1_holdout", "phase1_jevbench", "cookbook_dev")
    epochs:       float = 1.0
    soft_weight:  float = 1.0   # target = soft_weight x a teacher's soft target (where one exists) + the rest on the label

    # Packs: rows end to end, every pack the same shape
    pack_tokens:    int = 16384   # tokens per pack; a step is one pack (the optimizer runs per pack)
    max_row_tokens: int = 8192    # a row (one question) must fit this; also every segment's length cap
    max_rows:       int = 256     # segments per pack: the rows, then the padding in max_row_tokens pieces

    # ---- Training ----

    lr:        float = 2e-4   # peak; warmup 5%, cosine to 10% (§ Weight Init & Schedule)
    num_steps: int            # Derived: the packs one pass of the train splits makes, x epochs

    # Evaluation and logging
    val_every:  int = 400      # steps between panels; 0 = the end only (step 0 always)

    # Logging
    wandb_project:   str = "qwen_decision_rtx_dev"   # qwen_decision_rtx is for baselines
    run_name:        str = "rewrite-kev-1pass"      # both wandb and log files
    use_wandb:       bool = True

    save_checkpoint: bool = True   # the trained tensors, to ~/.cache/qwen-decision-rtx/runs/<run_name>/

    seed:            int  = 42     # LoRA's A, the packs' order and the options' shuffles

cfg = DecisionConfig() # Make config a global, don't pass it around.

# ==============================================================================
# § Derived Configs
# ==============================================================================

# Set this to run training under the normal schedule and have it abort
# part way. Great way to test things out without changing the data.
# None = run every step; 0 = evaluate the base model and exit.
ABORT_STEP = None

# The hand-written backward against autograd through the same forward (run eagerly), on
# the first pack, then exit. Same kernels either way, so the gap is bf16 rounding order.
CHECK_GRADS = False
if CHECK_GRADS:   # a small pack of short rows, so autograd's eager pass fits in memory
    cfg.train_splits, cfg.val_splits, cfg.pack_tokens, cfg.max_rows, cfg.use_wandb = ("decision_dev",), (), 1024, 32, False

if cfg.use_wandb:
    assert "WANDB_API_KEY" in os.environ, "cfg.use_wandb=True but WANDB_API_KEY not set"
    wandb.login(key=os.environ["WANDB_API_KEY"])

# The fused projections' widths, in bank order.
cfg.d_qk_all = cfg.n_qk_heads * cfg.d_qk                      # 2048, q or k of a DeltaNet layer
cfg.d_v_all = cfg.n_v_heads * cfg.d_v                         # 4096, v or z
cfg.d_conv_channels = 2 * cfg.d_qk_all + cfg.d_v_all          # 8192, the convolution's channels: [q | k | v]
cfg.d_qo_all = cfg.n_qo_heads * cfg.d_head                    # 4096, q or its gate of an attention layer
cfg.d_kv_all = cfg.n_kv_heads * cfg.d_head                    # 1024, k or v

# Every segment is at most max_row_tokens long, so a pack's padding takes this many segments at most.
cfg.max_pad_segments = math.ceil(cfg.pack_tokens / cfg.max_row_tokens)

CACHE = Path.home() / ".cache" / "qwen-decision-rtx"
DATA_DIR = CACHE / "decision"                                 # data/prepare_decision_data.py's output
MODEL_PATH = CACHE / "model" / "qwen3.5-9b-decision.safetensors"   # data/prepare_model.py's
RUN_DIR = CACHE / "runs" / cfg.run_name
assert MODEL_PATH.is_file(), f"{MODEL_PATH} missing: run data/prepare_model.py"
assert (DATA_DIR / "meta.json").is_file(), f"{DATA_DIR} missing: run data/prepare_decision_data.py"

# The data prep's answer codes and the prompt's tail after the last option; the banks were
# cut for the same codes.
data_meta = json.loads((DATA_DIR / "meta.json").read_text())
model_meta = json.loads(MODEL_PATH.with_suffix(".json").read_text())
CODE_IDS, TAIL_IDS = data_meta["code_ids"], data_meta["tail_ids"]
assert len(CODE_IDS) == cfg.n_codes and model_meta["code_ids"] == CODE_IDS, "the banks' answer codes differ from the data's"
for name in ("n_layers", "d_model", "d_mlp", "d_embed", "n_qk_heads", "d_qk", "n_v_heads", "d_v",
             "n_qo_heads", "n_kv_heads", "d_head", "d_rotary", "rope_theta"):
    assert model_meta["arch"][name] == getattr(cfg, name), f"banks' {name} = {model_meta['arch'][name]}"

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

# ==============================================================================
# § Data Structures
# ==============================================================================

# NamedTuples can be passed to compiled functions.
class Param(NamedTuple):
    """A trained tensor bundled with everything needed for training it."""

    name:        str
    w:           Tensor    # fp32

    # Optimizer State
    grad:        Tensor    # Matches full weight size
    first_mntm:  Tensor
    scnd_mntm:   Tensor

class Lora(NamedTuple):
    """One projection's LoRA in one layer: its rows of the A and B banks, and their gradient slots."""
    A:  Tensor    # (rank, in)  fp32
    B:  Tensor    # (out, rank) fp32
    Ab: Tensor
    Bb: Tensor

class DeltaNetLayer(NamedTuple):
    """One Gated DeltaNet layer: its slice of every frozen bank, and its LoRAs."""
    in_norm:     Tensor    # (d_model,)            fp32, 1 + the zero-centred weight
    in_proj:     Tensor    # (12352, d_model)      bf16, rows [q | k | v | z | b | a]
    conv:        Tensor    # (4, 8192)             fp32, tap j reads x[t - 3 + j]
    decay:       Tensor    # (n_v_heads,)          fp32, -exp(A_log)
    dt_bias:     Tensor    # (n_v_heads,)          fp32
    gate_norm:   Tensor    # (d_v,)                fp32
    out_proj:    Tensor    # (d_model, d_model)    bf16
    post_norm:   Tensor    # (d_model,)            fp32
    mlp_in:      Tensor    # (2 d_mlp, d_model)    bf16, rows [gate | up]
    mlp_out:     Tensor    # (d_model, d_mlp)      bf16
    lora_in:     Lora
    lora_out:    Lora
    lora_mlp_in: Lora
    lora_mlp_out: Lora

class AttentionLayer(NamedTuple):
    """One softmax attention layer: its slice of every frozen bank, and its LoRAs."""
    in_norm:     Tensor    # (d_model,)            fp32
    in_proj:     Tensor    # (10240, d_model)      bf16, rows [q | gate | k | v]
    q_norm:      Tensor    # (d_head,)             fp32
    k_norm:      Tensor    # (d_head,)             fp32
    out_proj:    Tensor    # (d_model, d_model)    bf16
    post_norm:   Tensor    # (d_model,)            fp32
    mlp_in:      Tensor    # (2 d_mlp, d_model)    bf16
    mlp_out:     Tensor    # (d_model, d_mlp)      bf16
    lora_in:     Lora
    lora_out:    Lora
    lora_mlp_in: Lora
    lora_mlp_out: Lora

class Model:
    """Container for the model's weights: the frozen banks, the trained Params, and the
    32 layer records that slice both."""

    # Frozen banks (data/prepare_model.py)
    embed:         Tensor   # (d_embed, d_model) bf16
    final_norm:    Tensor   # (d_model,) fp32
    inv_freq:      Tensor   # (d_rotary / 2,) fp32, the rotary frequencies

    # Trained: LoRA's A and B per projection, banked over each kind's layers
    deltanet_A_in: Param;    deltanet_B_in: Param       # (24, rank, d_model), (24, 12352, rank)
    deltanet_A_out: Param;   deltanet_B_out: Param
    deltanet_A_mlp_in: Param;  deltanet_B_mlp_in: Param
    deltanet_A_mlp_out: Param; deltanet_B_mlp_out: Param
    attention_A_in: Param;   attention_B_in: Param      # (8, rank, d_model), (8, 10240, rank)
    attention_A_out: Param;  attention_B_out: Param
    attention_A_mlp_in: Param;  attention_B_mlp_in: Param
    attention_A_mlp_out: Param; attention_B_mlp_out: Param

    # Trained: the readout, one row per answer code
    readout:       Param    # (n_codes, d_model)

    # The layers in checkpoint order, each a DeltaNetLayer or an AttentionLayer
    layers:        list

    # `for p in m` yields every trained Param, in the order above.
    def __iter__(self):
        return (v for v in vars(self).values() if isinstance(v, Param))

# One layer's forward activations, held for the backward pass. Everything element-wise
# between them is recomputed there (the norms' outputs, silu, the kernels' inputs).
class DeltaNetStash(NamedTuple):
    x_in:          Tensor    # (T, d_model)          bf16
    x_in_inv_rms:  Tensor    # (T, 1)                fp32
    u_in:          Tensor    # (T, rank)             bf16   x_in_hat A^T
    conv:          Tensor    # (T, 8192)             bf16   the convolution's output, before silu
    z:             Tensor    # (T, d_v_all)          bf16   the gate branch
    a:             Tensor    # (T, n_v_heads)        bf16   the decay branch, before softplus
    beta:          Tensor    # (T, n_v_heads)        bf16   sigmoid(b)
    core:          Tensor    # (T, n_v_heads, d_v)   bf16   the delta rule's output
    core_inv_rms:  Tensor    # (T, n_v_heads, 1)     fp32
    u_out:         Tensor    # (T, rank)             bf16
    x_mid:         Tensor    # (T, d_model)          bf16   the stream after the mixer
    x_mid_inv_rms: Tensor    # (T, 1)                fp32
    gu:            Tensor    # (T, 2 d_mlp)          bf16   the MLP's [gate | up]
    u_mlp_in:      Tensor    # (T, rank)             bf16
    u_mlp_out:     Tensor    # (T, rank)             bf16

class AttentionStash(NamedTuple):
    x_in:          Tensor    # (T, d_model)             bf16
    x_in_inv_rms:  Tensor    # (T, 1)                   fp32
    u_in:          Tensor    # (T, rank)                bf16
    q_raw:         Tensor    # (T, n_qo_heads, d_head)  bf16   before the q norm
    k_raw:         Tensor    # (T, n_kv_heads, d_head)  bf16
    v:             Tensor    # (T, n_kv_heads, d_head)  bf16
    q_inv_rms:     Tensor    # (T, n_qo_heads, 1)       fp32
    k_inv_rms:     Tensor    # (T, n_kv_heads, 1)       fp32
    gate:          Tensor    # (T, d_qo_all)            bf16   the output gate, before sigmoid
    y:             Tensor    # (T, n_qo_heads, d_head)  bf16   attention's output
    lse:           Tensor    # (n_qo_heads, T)          fp32   its softmax log-sum-exp
    u_out:         Tensor    # (T, rank)                bf16
    x_mid:         Tensor    # (T, d_model)             bf16
    x_mid_inv_rms: Tensor    # (T, 1)                   fp32
    gu:            Tensor    # (T, 2 d_mlp)             bf16
    u_mlp_in:      Tensor    # (T, rank)                bf16
    u_mlp_out:     Tensor    # (T, rank)                bf16

class Pack(NamedTuple):
    """One pack on the device: pack_tokens tokens in max_rows segments (the rows, then the
    padding, then empty segments), and per row its last token, option count and target."""
    tokens:    Tensor    # (pack_tokens,)             int32
    pos:       Tensor    # (pack_tokens,)             int32   position within the segment
    cu:        Tensor    # (max_rows + 1,)            int32   segment boundaries, cu_seqlens
    cu_cpu:    Tensor    # (max_rows + 1,)            int64, on the host: FLA's chunk indices without a sync
    last_pos:  Tensor    # (max_rows,)                int64   each row's readout token (a padding row: the pack's last)
    n_options: Tensor    # (max_rows,)                int32   options per row (a padding row: 1)
    target:    Tensor    # (max_rows, n_codes)        fp32    the target over the options (a padding row: option 0)
    n_rows:    Tensor    # ()                         int32
    rows:      list      # the Row records, for the evaluation's scoring

class Row(NamedTuple):
    """One question of one record, from the data prep's parquet."""
    split:   str
    record:  int
    source:  str
    clean:   bool
    kind:    str       # "choice" | "noul" | "score"
    prefix:  list      # token ids up to and including "Options:\n" (the state is inside)
    options: list      # each option's ": <text>\n" ids, in record order
    label:   int
    target:  list      # a teacher's probability per option, or None
    length:  int       # the row's tokens: prefix, a code per option, the options, the tail

bf16 = lambda x: x.to(torch.bfloat16)


# ==============================================================================
# § Kernels
# ==============================================================================
# The two segment-aware ops, as custom ops: the graph sees each as one opaque call with
# fixed output shapes, whatever happens inside (FLA builds data-dependent chunk indices;
# FA2's backward writes into preallocated buffers). Their autograd registrations serve
# CHECK_GRADS only; training calls the *_bwd ops by hand.

@torch.library.custom_op("decision::attention", mutates_args=())
def attention(q: Tensor, k: Tensor, v: Tensor, cu: Tensor) -> tuple[Tensor, Tensor]:
    """Causal softmax attention within each segment: q (T, n_qo_heads, d_head), k and v
    (T, n_kv_heads, d_head), grouped. -> (y, the softmax LSE (n_qo_heads, T) fp32)."""
    y, lse, *_ = fa2._flash_attn_varlen_forward(
        q, k, v, cu, cu, cfg.max_row_tokens, cfg.max_row_tokens,
        0.0,                        # dropout_p
        cfg.d_head ** -0.5,         # softmax_scale
        True,                       # causal
        window_size_left=cfg.max_row_tokens, window_size_right=0)
    return y, lse

@attention.register_fake
def _(q, k, v, cu):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)

@torch.library.custom_op("decision::attention_bwd", mutates_args=())
def attention_bwd(yb: Tensor, q: Tensor, k: Tensor, v: Tensor, y: Tensor, lse: Tensor, cu: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """-> (qb, kb, vb). FA2's varlen backward writes the grads into preallocated buffers."""
    qb, kb, vb = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    fa2._flash_attn_varlen_backward(
        yb, q, k, v, y, lse,
        qb, kb, vb,
        cu, cu,                     # cu_seqlens_q, cu_seqlens_k
        cfg.max_row_tokens, cfg.max_row_tokens,
        0.0,                        # dropout_p
        cfg.d_head ** -0.5,         # softmax_scale
        True,                       # is_causal
        cfg.max_row_tokens, 0,      # window_size_left, window_size_right
        0.0,                        # softcap
        None,                       # alibi_slopes
        False,                      # deterministic
        None,                       # rng_state
    )
    return qb, kb, vb

@attention_bwd.register_fake
def _(yb, q, k, v, y, lse, cu):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

def _attention_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3], *output, inputs[3])

def _attention_autograd(ctx, yb, lse_b):
    q, k, v, y, lse, cu = ctx.saved_tensors
    return *attention_bwd(yb.contiguous(), q, k, v, y, lse, cu), None

attention.register_autograd(_attention_autograd, setup_context=_attention_setup)


@torch.library.custom_op("decision::gated_delta_rule", mutates_args=())
def gated_delta_rule(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, cu: Tensor, cu_cpu: Tensor) -> Tensor:
    """Gated DeltaNet within each segment, the state reset at each boundary: q, k
    (T, n_qk_heads, d_qk), v (T, n_v_heads, d_v), the log decay g fp32 and beta (T, n_v_heads).
    q and k are L2-normalized inside; each q / k head serves n_v_heads / n_qk_heads value heads."""
    out, _ = chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None], cu_seqlens=cu.long(),
                                    cu_seqlens_cpu=cu_cpu, use_qk_l2norm_in_kernel=True)
    return out[0]

@gated_delta_rule.register_fake
def _(q, k, v, g, beta, cu, cu_cpu):
    return torch.empty_like(v)

@torch.library.custom_op("decision::gated_delta_rule_bwd", mutates_args=())
def gated_delta_rule_bwd(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, outb: Tensor, cu: Tensor, cu_cpu: Tensor) -> list[Tensor]:
    """-> [qb, kb, vb, gb, betab], through FLA's own backward: the forward is rerun under a
    local autograd graph (its kernels are a few percent of a step) and differentiated."""
    leaves = [x.detach().requires_grad_(True) for x in (q, k, v, g, beta)]
    with torch.enable_grad():
        out, _ = chunk_gated_delta_rule(*(x[None] for x in leaves), cu_seqlens=cu.long(),
                                        cu_seqlens_cpu=cu_cpu, use_qk_l2norm_in_kernel=True)
        return list(torch.autograd.grad(out[0], leaves, outb))

@gated_delta_rule_bwd.register_fake
def _(q, k, v, g, beta, outb, cu, cu_cpu):
    return [torch.empty_like(x) for x in (q, k, v, g, beta)]

def _gdn_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)

def _gdn_autograd(ctx, outb):
    q, k, v, g, beta, cu, cu_cpu = ctx.saved_tensors
    return *gated_delta_rule_bwd(q, k, v, g, beta, outb.contiguous(), cu, cu_cpu), None, None

gated_delta_rule.register_autograd(_gdn_autograd, setup_context=_gdn_setup)


# ==============================================================================
# § Train Forward + Backward
# ==============================================================================
# forward_backward (at the end of the chapter) drives the compiled pieces: the embedding,
# one layer_forward per layer (stashing), the readout's loss and its gradient into the
# stream, then one layer_backward per layer in reverse. The LoRA gradients land in their
# Param.grad banks; § Optimizer Math steps them after the pack.
#
# Direction naming (decoder-rtx's):
# A bare name is the forward pass; a 'b' on its leading symbol is the backward pass, the
# gradient w.r.t. that exact tensor: x -> xb, q_hat -> qb_hat, mixed -> mixedb.
#
# Stream stages:
# x_in    - the layer's input
# x_mid   - after the mixer's residual add, the MLP's input
# x       - after the MLP's residual add, the next layer's input
# *_hat   - RMS-normed and weighted, bf16, what the GEMMs read
# *_inv_rms - 1/rms, fp32, kept for the norm's backward

def lora(x, W, l):
    """x (T, in) -> x W^T + (alpha / rank) (x A^T) B^T, and u = x A^T for the backward."""
    u = x @ bf16(l.A).mT                                                           # (T, rank)
    return torch.addmm(x @ W.mT, u, bf16(l.B).mT, alpha=cfg.alpha / cfg.rank), u

def lora_bwd(yb, x, u, W, l):
    """-> xb (T, in); A's and B's gradients land in their slots."""
    yB = yb @ bf16(l.B)                                                            # (T, rank)
    l.Ab.copy_((yB.mT @ x).float() * (cfg.alpha / cfg.rank))                       # (rank, in)
    l.Bb.copy_((yb.mT @ u).float() * (cfg.alpha / cfg.rank))                       # (out, rank)
    return torch.addmm(yb @ W, yB, bf16(l.A), alpha=cfg.alpha / cfg.rank)

def rope(x, cos, sin):
    """x (T, H, d_head) fp32: the first d_rotary dims rotated, the rotate-half form. The
    transpose of the rotation is the same call with -sin."""
    r, rest = x[..., :cfg.d_rotary], x[..., cfg.d_rotary:]
    half = cfg.d_rotary // 2
    return torch.cat((r * cos + torch.cat((-r[..., half:], r[..., :half]), dim=-1) * sin, rest), dim=-1)


@torch.compile(dynamic=False, fullgraph=True)
def embed_forward(tokens, pos):
    """The token embeddings, the rotary tables and the convolution's row-start mask, for one pack."""
    x = F.embedding(tokens, m.embed)                                               # (T, d_model) bf16
    freqs = pos.float()[:, None] * m.inv_freq[None]                                # (T, d_rotary / 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]                              # (T, 1, d_rotary) fp32
    # Tap j reads x[t - 3 + j], inside the segment iff pos >= 3 - j.
    tap_keep = (pos[:, None] >= torch.arange(3, -1, -1, device=pos.device)).float()   # (T, 4)
    return x, cos, sin, tap_keep


@torch.compile(dynamic=False, fullgraph=True)
def layer_forward(x, L, cos, sin, tap_keep, cu, cu_cpu, backward):
    """One layer: x -> x, and its stash when backward=True (None otherwise)."""
    T = x.shape[0]
    deltanet = isinstance(L, DeltaNetLayer)
    x_in = x

    # Mixer input norm
    x_in_inv_rms = (x_in.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    x_in_hat = bf16(x_in.float() * x_in_inv_rms * L.in_norm)

    # The fused input projection
    proj, u_in = lora(x_in_hat, L.in_proj, L.lora_in)

    if deltanet:
        qkv, z, b, a = proj.split([cfg.d_conv_channels, cfg.d_v_all, cfg.n_v_heads, cfg.n_v_heads], dim=-1)

        # Causal depthwise convolution over the segment: tap j reads x[t - 3 + j], zero before the segment.
        qkv_pad = F.pad(qkv, (0, 0, 3, 0))
        conv = bf16(sum([qkv_pad[j:j + T].float() * tap_keep[:, j:j + 1] * L.conv[j] for j in range(4)]))   # (T, 8192)
        q, k, v = F.silu(conv).split([cfg.d_qk_all, cfg.d_qk_all, cfg.d_v_all], dim=-1)

        # The delta rule's gates: beta in (0, 1), the log decay g <= 0
        beta = torch.sigmoid(b)                                                    # (T, n_v_heads) bf16
        g = L.decay * F.softplus(a.float() + L.dt_bias)                            # (T, n_v_heads) fp32

        core = gated_delta_rule(q.reshape(T, cfg.n_qk_heads, cfg.d_qk), k.reshape(T, cfg.n_qk_heads, cfg.d_qk),
                                v.reshape(T, cfg.n_v_heads, cfg.d_v), g, beta, cu, cu_cpu)   # (T, n_v_heads, d_v)

        # Gated RMSNorm: the normed core, weighted per value dim, gated by silu(z)
        core_inv_rms = (core.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
        mixed = bf16(core.float() * core_inv_rms * L.gate_norm
                     * F.silu(z.float().view(T, cfg.n_v_heads, cfg.d_v))).view(T, cfg.d_model)
    else:
        q_raw, gate, k_raw, v = proj.split([cfg.d_qo_all, cfg.d_qo_all, cfg.d_kv_all, cfg.d_kv_all], dim=-1)
        q_raw = q_raw.reshape(T, cfg.n_qo_heads, cfg.d_head)
        k_raw = k_raw.reshape(T, cfg.n_kv_heads, cfg.d_head)
        v = v.reshape(T, cfg.n_kv_heads, cfg.d_head)

        # QK-norm, then RoPE
        q_inv_rms = (q_raw.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
        k_inv_rms = (k_raw.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
        q_hat = bf16(rope(q_raw.float() * q_inv_rms * L.q_norm, cos, sin))
        k_hat = bf16(rope(k_raw.float() * k_inv_rms * L.k_norm, cos, sin))

        # Read V from the segment's past by matching its K.
        y, lse = attention(q_hat, k_hat, v, cu)                                    # (T, n_qo_heads, d_head)

        # The sigmoid output gate
        mixed = bf16(y.view(T, cfg.d_model).float() * torch.sigmoid(gate.float()))

    # Project back onto the stream.
    attn_out, u_out = lora(mixed, L.out_proj, L.lora_out)
    x_mid = x_in + attn_out

    # MLP input norm
    x_mid_inv_rms = (x_mid.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    x_mid_hat = bf16(x_mid.float() * x_mid_inv_rms * L.post_norm)

    # SwiGLU
    gu, u_mlp_in = lora(x_mid_hat, L.mlp_in, L.lora_mlp_in)                        # (T, 2 d_mlp)
    act = bf16(F.silu(gu[:, :cfg.d_mlp].float()) * gu[:, cfg.d_mlp:].float())
    mlp_out, u_mlp_out = lora(act, L.mlp_out, L.lora_mlp_out)

    # Write back to the stream.
    x = x_mid + mlp_out

    if not backward:
        return x, None
    if deltanet:
        return x, DeltaNetStash(x_in=x_in, x_in_inv_rms=x_in_inv_rms, u_in=u_in, conv=conv, z=z.contiguous(),
                                a=a.contiguous(), beta=beta, core=core, core_inv_rms=core_inv_rms, u_out=u_out,
                                x_mid=x_mid, x_mid_inv_rms=x_mid_inv_rms, gu=gu, u_mlp_in=u_mlp_in, u_mlp_out=u_mlp_out)
    return x, AttentionStash(x_in=x_in, x_in_inv_rms=x_in_inv_rms, u_in=u_in, q_raw=q_raw, k_raw=k_raw, v=v,
                             q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms, gate=gate.contiguous(), y=y, lse=lse, u_out=u_out,
                             x_mid=x_mid, x_mid_inv_rms=x_mid_inv_rms, gu=gu, u_mlp_in=u_mlp_in, u_mlp_out=u_mlp_out)


@torch.compile(dynamic=False, fullgraph=True)
def readout_forward_backward(x, last_pos, n_options, target, n_rows, backward):
    """The final norm at each row's last token, the logits over the answer codes (masked to
    the row's options) and the cross-entropy against the target, averaged over the pack's
    rows. With backward=True also the readout's gradient and the loss's gradient w.r.t.
    the stream, xb (T, d_model) fp32, zero everywhere but the readout tokens."""
    rows = x[last_pos]                                                             # (R, d_model) bf16
    inv_rms = (rows.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    rows_hat = bf16(rows.float() * inv_rms * m.final_norm)

    # ==== Forward ====
    valid = torch.arange(cfg.n_codes, device=x.device)[None] < n_options[:, None]  # (R, n_codes)
    logits = (rows_hat @ bf16(m.readout.w).mT).float().masked_fill(~valid, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()
    if not backward:
        return probs
    loss = -(target * logp.masked_fill(~valid, 0.0)).sum() / n_rows

    # ==== Backward ====
    # A padding row's target is option 0 of 1, so its gradient is exactly zero; the mask
    # keeps it so whatever the target holds.
    real = (n_options > 1).float()[:, None]
    logitsb = (probs - target) * real / n_rows                                      # (R, n_codes) fp32
    m.readout.grad.copy_(logitsb.mT @ rows_hat.float())                             # (n_codes, d_model)
    rows_hatb = logitsb @ m.readout.w                                               # (R, d_model) fp32

    # Final norm backward
    gh = rows_hatb * m.final_norm
    xhat = rows.float() * inv_rms
    rowsb = inv_rms * (gh - xhat * (gh * xhat).mean(dim=-1, keepdim=True))
    xb = torch.zeros(x.shape, dtype=torch.float32, device=x.device).index_add_(0, last_pos, rowsb)
    return loss, probs, xb


@torch.compile(dynamic=False, fullgraph=True)
def layer_backward(xb, L, st, cos, sin, tap_keep, cu, cu_cpu):
    """One layer's backward: xb (T, d_model) fp32, the gradient w.r.t. the layer's output,
    -> the gradient w.r.t. its input. The LoRA gradients land in L's slots."""
    T = xb.shape[0]
    deltanet = isinstance(L, DeltaNetLayer)

    # --- MLP backward: x = x_mid + mlp_out, mlp_out = lora(silu(gate) * up) ---
    x_mid_hat = bf16(st.x_mid.float() * st.x_mid_inv_rms * L.post_norm)
    gate_f, up = st.gu[:, :cfg.d_mlp].float(), st.gu[:, cfg.d_mlp:].float()
    sig = torch.sigmoid(gate_f)
    silu_gate = gate_f * sig
    act = bf16(silu_gate * up)
    actb = lora_bwd(bf16(xb), act, st.u_mlp_out, L.mlp_out, L.lora_mlp_out)              # (T, d_mlp) bf16
    gub = torch.cat((bf16(actb.float() * up * (sig * (1 + gate_f * (1 - sig)))),         # d silu(gate) / d gate
                     bf16(actb.float() * silu_gate)), dim=-1)                            # (T, 2 d_mlp)
    x_mid_hatb = lora_bwd(gub, x_mid_hat, st.u_mlp_in, L.mlp_in, L.lora_mlp_in)

    # MLP input norm backward; the residual passes xb through.
    gh = x_mid_hatb.float() * L.post_norm
    xhat = st.x_mid.float() * st.x_mid_inv_rms
    xb_mid = xb + st.x_mid_inv_rms * (gh - xhat * (gh * xhat).mean(dim=-1, keepdim=True))

    # --- Mixer backward: x_mid = x_in + attn_out, attn_out = lora(mixed) ---
    if deltanet:
        # The gated norm, recomputed: mixed = n * gate_norm * silu(z), n the normed core
        z_f = st.z.float().view(T, cfg.n_v_heads, cfg.d_v)
        sig_z = torch.sigmoid(z_f)
        silu_z = z_f * sig_z
        n = st.core.float() * st.core_inv_rms
        mixed = bf16(n * L.gate_norm * silu_z).view(T, cfg.d_model)
        mixedb = lora_bwd(bf16(xb_mid), mixed, st.u_out, L.out_proj, L.lora_out)
        mb = mixedb.float().view(T, cfg.n_v_heads, cfg.d_v)
        zb = mb * n * L.gate_norm * (sig_z * (1 + z_f * (1 - sig_z)))
        nb = mb * L.gate_norm * silu_z
        coreb = bf16(st.core_inv_rms * (nb - n * (nb * n).mean(dim=-1, keepdim=True)))

        # The kernel's inputs, recomputed from the convolution's output, then FLA's backward
        q, k, v = F.silu(st.conv).split([cfg.d_qk_all, cfg.d_qk_all, cfg.d_v_all], dim=-1)
        g = L.decay * F.softplus(st.a.float() + L.dt_bias)
        qb, kb, vb, gb, betab = gated_delta_rule_bwd(q.reshape(T, cfg.n_qk_heads, cfg.d_qk), k.reshape(T, cfg.n_qk_heads, cfg.d_qk),
                                                     v.reshape(T, cfg.n_v_heads, cfg.d_v), g, st.beta, coreb, cu, cu_cpu)

        # The gates' backward: beta = sigmoid(b), g = decay * softplus(a + dt_bias)
        beta_f = st.beta.float()
        bb = betab.float() * beta_f * (1 - beta_f)
        ab = gb * L.decay * torch.sigmoid(st.a.float() + L.dt_bias)

        # silu, then the convolution: conv[t] = sum_j qkv[t - 3 + j] keep[t, j] w[j], so
        # qkvb[t] = sum_j (convb keep w[j])[t + 3 - j].
        conv_f = st.conv.float()
        sig_c = torch.sigmoid(conv_f)
        convb = torch.cat((qb.reshape(T, cfg.d_qk_all), kb.reshape(T, cfg.d_qk_all), vb.reshape(T, cfg.d_v_all)), dim=-1).float() \
                * (sig_c * (1 + conv_f * (1 - sig_c)))                                   # (T, 8192) fp32
        qkvb = sum([F.pad(convb * tap_keep[:, j:j + 1] * L.conv[j], (0, 0, 0, 3 - j))[3 - j:] for j in range(4)])
        projb = bf16(torch.cat((qkvb, zb.reshape(T, cfg.d_v_all), bb, ab), dim=-1))      # (T, 12352)
    else:
        # The output gate, recomputed
        sig = torch.sigmoid(st.gate.float())
        mixed = bf16(st.y.view(T, cfg.d_model).float() * sig)
        mixedb = lora_bwd(bf16(xb_mid), mixed, st.u_out, L.out_proj, L.lora_out)
        mb = mixedb.float()
        yb = bf16(mb * sig).view(T, cfg.n_qo_heads, cfg.d_head)
        gateb = mb * st.y.view(T, cfg.d_model).float() * sig * (1 - sig)                 # (T, d_qo_all)

        # q_hat and k_hat recomputed, then FlashAttention's backward
        q_hat = bf16(rope(st.q_raw.float() * st.q_inv_rms * L.q_norm, cos, sin))
        k_hat = bf16(rope(st.k_raw.float() * st.k_inv_rms * L.k_norm, cos, sin))
        qb_hat, kb_hat, vb = attention_bwd(yb, q_hat, k_hat, st.v, st.y, st.lse, cu)

        # RoPE's transpose is the rotation by -theta; then the per-(token, head) norms' backward
        gh = rope(qb_hat.float(), cos, -sin) * L.q_norm
        xhat = st.q_raw.float() * st.q_inv_rms
        q_rawb = st.q_inv_rms * (gh - xhat * (gh * xhat).mean(dim=-1, keepdim=True))
        gh = rope(kb_hat.float(), cos, -sin) * L.k_norm
        xhat = st.k_raw.float() * st.k_inv_rms
        k_rawb = st.k_inv_rms * (gh - xhat * (gh * xhat).mean(dim=-1, keepdim=True))
        projb = bf16(torch.cat((q_rawb.reshape(T, cfg.d_qo_all), gateb, k_rawb.reshape(T, cfg.d_kv_all),
                                vb.reshape(T, cfg.d_kv_all).float()), dim=-1))           # (T, 10240)

    # The fused input projection's backward, then the mixer input norm's; the residual passes xb_mid through.
    x_in_hat = bf16(st.x_in.float() * st.x_in_inv_rms * L.in_norm)
    x_in_hatb = lora_bwd(projb, x_in_hat, st.u_in, L.in_proj, L.lora_in)
    gh = x_in_hatb.float() * L.in_norm
    xhat = st.x_in.float() * st.x_in_inv_rms
    return xb_mid + st.x_in_inv_rms * (gh - xhat * (gh * xhat).mean(dim=-1, keepdim=True))


@torch.no_grad()
def forward_backward(pack, backward=True):
    """One pack through the model. Returns the loss; with backward=False (evaluation) the
    rows' probabilities over the answer codes instead, and nothing is kept."""
    x, cos, sin, tap_keep = embed_forward(pack.tokens, pack.pos)

    # -----------------------------
    #           Forward
    # -----------------------------
    stash = []
    for L in m.layers:
        x, st = layer_forward(x, L, cos, sin, tap_keep, pack.cu, pack.cu_cpu, backward)
        stash.append(st)
    if not backward:
        return readout_forward_backward(x, pack.last_pos, pack.n_options, pack.target, pack.n_rows, False)
    loss, probs, xb = readout_forward_backward(x, pack.last_pos, pack.n_options, pack.target, pack.n_rows, True)

    # -----------------------------
    #           Backward
    # -----------------------------
    for i in reversed(range(cfg.n_layers)):
        # ← each layer's backward, its LoRA gradients into their banks
        xb = layer_backward(xb, m.layers[i], stash[i], cos, sin, tap_keep, pack.cu, pack.cu_cpu)
        stash[i] = None                          # free this layer's stash as we go
    return loss


# ==============================================================================
# § Optimizer Math
# ==============================================================================

# ------------------------------------------------------------------------------
# §§ AdamW
# ------------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step(params, t):
    """One AdamW step on every Param: betas 0.9 / 0.999, no weight decay, the bias
    corrections folded into lr_bc_t and eps_t. t is the (,) int64 device step."""
    for p in params:
        p.first_mntm.lerp_(p.grad, 0.1)                                  # m = 0.9 m + 0.1 g
        p.scnd_mntm.mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)   # v = 0.999 v + 0.001 g^2
        p.w.sub_(lr_bc_t[t] * p.first_mntm / (p.scnd_mntm.sqrt() + eps_t[t]))


# ==============================================================================
# § Data
# ==============================================================================
# The data prep's parquets -> Row records -> packs of at most pack_tokens tokens and
# max_rows - max_pad_segments rows -> every pack's tensors staged in pinned memory, in
# one fixed shape. Training packs are planned for the whole run up front (shuffled, filled
# greedily, a choice question's options shuffled each time its row is built); evaluation
# packs are planned per split (first-fit decreasing).

def load_rows(split):
    t = pq.read_table(DATA_DIR / f"{split}.parquet").to_pydict()
    rows = []
    for r in range(len(t["id"])):
        for j in range(len(t["q_name"][r])):
            prefix, options = t["q_prefix"][r][j], t["q_options"][r][j]
            rows.append(Row(split=split, record=r, source=t["source"][r], clean=t["variant"][r] == "clean",
                            kind=t["q_type"][r][j], prefix=prefix, options=options, label=t["q_label"][r][j],
                            target=t["q_target"][r][j] or None,
                            length=len(prefix) + sum(len(o) + 1 for o in options) + len(TAIL_IDS)))
    longest = max(r.length for r in rows)
    assert longest <= min(cfg.max_row_tokens, cfg.pack_tokens), f"{split}: a {longest}-token row is over max_row_tokens"
    assert max(len(r.options) for r in rows) <= cfg.n_codes
    return rows


def plan_packs(rows, shuffle):
    """Rows into packs: in random order, filled greedily (training), or first-fit decreasing (evaluation)."""
    max_rows = cfg.max_rows - cfg.max_pad_segments
    if shuffle:
        order = list(range(len(rows)))
        random.shuffle(order)
        packs, cur, used = [], [], 0
        for i in order:
            if used + rows[i].length > cfg.pack_tokens or len(cur) == max_rows:
                packs.append(cur)
                cur, used = [], 0
            cur.append(rows[i])
            used += rows[i].length
        return packs + [cur]
    packs, room = [], []
    for i in sorted(range(len(rows)), key=lambda i: -rows[i].length):
        for p in range(len(packs)):
            if rows[i].length <= room[p] and len(packs[p]) < max_rows:
                packs[p].append(rows[i])
                room[p] -= rows[i].length
                break
        else:
            packs.append([rows[i]])
            room.append(cfg.pack_tokens - rows[i].length)
    return packs


class Staged(NamedTuple):
    """Every pack of a plan, in pinned memory, one fixed shape."""
    tokens:    Tensor    # (N, pack_tokens)         int32
    pos:       Tensor    # (N, pack_tokens)         int32
    cu:        Tensor    # (N, max_rows + 1)        int32
    last_pos:  Tensor    # (N, max_rows)            int64
    n_options: Tensor    # (N, max_rows)            int32
    target:    Tensor    # (N, max_rows, n_codes)   fp32
    n_rows:    Tensor    # (N,)                     int32
    rows:      list      # per pack, its Row records (in pack order)


def stage_packs(packs, train):
    """-> Staged. A row is its prefix, then a code and the option for each option (in the
    row's order), then the tail; every option is read at the row's last token. The padding
    after the rows is cut into segments of at most max_row_tokens (position 0 at each cut),
    and the remaining segments are empty."""
    N, P, R = len(packs), cfg.pack_tokens, cfg.max_rows
    pinned = lambda *shape, dtype: torch.empty(*shape, dtype=dtype, pin_memory=True)
    st = Staged(tokens=pinned(N, P, dtype=torch.int32), pos=pinned(N, P, dtype=torch.int32),
                cu=pinned(N, R + 1, dtype=torch.int32), last_pos=pinned(N, R, dtype=torch.int64),
                n_options=pinned(N, R, dtype=torch.int32), target=pinned(N, R, cfg.n_codes, dtype=torch.float32),
                n_rows=pinned(N, dtype=torch.int32), rows=[])
    tokens, pos, cu, last_pos, n_options, target = (x.numpy() for x in st[:6])
    tokens.fill(0); pos.fill(0); cu.fill(P); cu[:, 0] = 0; last_pos.fill(P - 1); n_options.fill(1); target.fill(0.0); target[:, :, 0] = 1.0
    for n, pack_rows in enumerate(packs):
        end = 0
        for j, row in enumerate(pack_rows):
            order = list(range(len(row.options)))
            if train and row.kind == "choice":
                random.shuffle(order)
            ids = list(row.prefix)
            for k, o in enumerate(order):
                ids.append(CODE_IDS[k])
                ids += row.options[o]
            ids += TAIL_IDS
            assert len(ids) == row.length
            start, end = end, end + row.length
            tokens[n, start:end] = ids
            pos[n, start:end] = np.arange(row.length)
            cu[n, j + 1] = end
            last_pos[n, j] = end - 1
            n_options[n, j] = len(order)
            hard = np.zeros(len(order), np.float32)
            hard[order.index(row.label)] = 1.0
            if train and row.target is not None:
                soft = np.asarray([row.target[o] for o in order], np.float32)
                target[n, j, :len(order)] = cfg.soft_weight * soft + (1 - cfg.soft_weight) * hard
            else:
                target[n, j, :len(order)] = hard
        j = len(pack_rows)
        while end < P:                                   # the padding, in segments of at most max_row_tokens
            start, end = end, min(end + cfg.max_row_tokens, P)
            pos[n, start:end] = np.arange(end - start)
            cu[n, j + 1] = end
            j += 1
        st.n_rows[n] = len(pack_rows)
        st.rows.append(pack_rows)
    return st


def pack_on_device(st, n):
    """Pack n of a Staged, moved to the device."""
    to = lambda x: x[n].to(device, non_blocking=True)
    return Pack(tokens=to(st.tokens), pos=to(st.pos), cu=to(st.cu), cu_cpu=st.cu[n].long(),
                last_pos=to(st.last_pos), n_options=to(st.n_options), target=to(st.target),
                n_rows=to(st.n_rows), rows=st.rows[n])


t0 = time.perf_counter()
train_rows = [r for split in cfg.train_splits for r in load_rows(split)]
train_tokens = sum(r.length for r in train_rows)
epochs = [plan_packs(train_rows, shuffle=True) for _ in range(math.ceil(cfg.epochs))]
train_packs = [p for ep in epochs for p in ep][:round(cfg.epochs * len(epochs[0]))]
train_staged = stage_packs(train_packs, train=True)
cfg.num_steps = len(train_packs)
val_rows = {split: load_rows(split) for split in cfg.val_splits}
print(f"train: {len(train_rows):,} questions, {train_tokens / 1e6:.1f}M tokens a pass, {len(epochs[0])} packs a pass; "
      f"{cfg.num_steps} steps; staged in {time.perf_counter() - t0:.1f}s "
      f"({sum(x.numel() * x.element_size() for x in train_staged[:7]) / 2**30:.2f} GiB pinned)")
rows_per_pack = train_staged.n_rows.numpy()
print(f"  rows per pack: mean {rows_per_pack.mean():.1f}, max {rows_per_pack.max()} (of {cfg.max_rows - cfg.max_pad_segments}); "
      f"fill {100 * train_tokens * cfg.epochs / (cfg.num_steps * cfg.pack_tokens):.1f}%")


# ==============================================================================
# § Weight Init & Schedule
# ==============================================================================

# ------------------------------------------------------------------------------
# §§ LR Schedule
# ------------------------------------------------------------------------------

# Learning rate as a per-step table: linear warmup over the first 5% of steps, then
# cosine to 10% of peak. The bias corrections are folded in: lr_bc_t and eps_t are what
# the update reads.
steps_0idx = np.arange(cfg.num_steps, dtype=np.float64)
steps_1idx = steps_0idx + 1.0
warmup_steps = max(1, int(0.05 * cfg.num_steps))
cosine_frac = np.clip((steps_0idx - warmup_steps) / max(1, cfg.num_steps - warmup_steps), 0.0, 1.0)
lr_mult_t = np.where(steps_0idx < warmup_steps, steps_1idx / warmup_steps,
                     0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * cosine_frac)))

dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
lr_bc_t = dev(cfg.lr * lr_mult_t * (1.0 - 0.999 ** steps_1idx) ** 0.5 / (1.0 - 0.9 ** steps_1idx))
eps_t   = dev(1e-8 * (1.0 - 0.999 ** steps_1idx) ** 0.5)

# ------------------------------------------------------------------------------
# §§ Frozen Banks
# ------------------------------------------------------------------------------

m = Model()

t0 = time.perf_counter()
banks = load_file(str(MODEL_PATH), device="cuda")
cfg.num_frozen_params = sum(t.numel() for name, t in banks.items() if name != "lm_head_codes")
m.embed = banks["embed"]
m.final_norm = banks["final_norm"]
m.inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.d_rotary, 2, device=device).float() / cfg.d_rotary)
print(f"banks: {cfg.num_frozen_params / 1e9:.2f}B parameters in {time.perf_counter() - t0:.0f}s; "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")

# ------------------------------------------------------------------------------
# §§ LoRA
# ------------------------------------------------------------------------------

fp32_zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)

def param(name, w):
    """A Param around w, with zero gradient and optimizer state."""
    return Param(name=name, w=w, grad=fp32_zeros(w.shape), first_mntm=fp32_zeros(w.shape), scnd_mntm=fp32_zeros(w.shape))

# Per kind and projection: A (L, rank, in) Kaiming-uniform as nn.Linear's, B (L, out, rank)
# zero, so training starts from the base model.
for kind, n_kind in (("deltanet", cfg.n_deltanet), ("attention", cfg.n_attention)):
    for proj in ("in", "out", "mlp_in", "mlp_out"):
        out_dim, in_dim = banks[f"{kind}_{proj}_proj" if proj in ("in", "out") else f"{kind}_{proj}"].shape[1:]
        A = torch.empty(n_kind, cfg.rank, in_dim, dtype=torch.float32, device=device)
        torch.nn.init.kaiming_uniform_(A.view(-1, in_dim), a=math.sqrt(5))
        setattr(m, f"{kind}_A_{proj}", param(f"{kind}_A_{proj}", A))
        setattr(m, f"{kind}_B_{proj}", param(f"{kind}_B_{proj}", fp32_zeros(n_kind, out_dim, cfg.rank)))

# ------------------------------------------------------------------------------
# §§ Readout
# ------------------------------------------------------------------------------

# The LM head's rows for the answer codes: before any training the model answers as Qwen does.
m.readout = param("readout", banks["lm_head_codes"].float())

cfg.num_params = sum(p.w.numel() for p in m)
print(f"trained: {cfg.num_params / 1e6:.1f}M parameters (LoRA rank {cfg.rank}, alpha {cfg.alpha:g}, and the readout)")

# ------------------------------------------------------------------------------
# §§ Layers
# ------------------------------------------------------------------------------

def build_layers():
    """The 32 layer records in checkpoint order, each slicing its kind's banks and Params."""
    layers = []
    for i in range(cfg.n_layers):
        kind, j = ("attention", i // 4) if i % 4 == 3 else ("deltanet", i - i // 4)
        bank = lambda name: banks[f"{kind}_{name}"][j]
        loras = {f"lora_{proj}": Lora(A=getattr(m, f"{kind}_A_{proj}").w[j], B=getattr(m, f"{kind}_B_{proj}").w[j],
                                      Ab=getattr(m, f"{kind}_A_{proj}").grad[j], Bb=getattr(m, f"{kind}_B_{proj}").grad[j])
                 for proj in ("in", "out", "mlp_in", "mlp_out")}
        if kind == "deltanet":
            layers.append(DeltaNetLayer(in_norm=bank("in_norm"), in_proj=bank("in_proj"), conv=bank("conv"), decay=bank("decay"),
                                        dt_bias=bank("dt_bias"), gate_norm=bank("gate_norm"), out_proj=bank("out_proj"),
                                        post_norm=bank("post_norm"), mlp_in=bank("mlp_in"), mlp_out=bank("mlp_out"), **loras))
        else:
            layers.append(AttentionLayer(in_norm=bank("in_norm"), in_proj=bank("in_proj"), q_norm=bank("q_norm"), k_norm=bank("k_norm"),
                                         out_proj=bank("out_proj"), post_norm=bank("post_norm"), mlp_in=bank("mlp_in"),
                                         mlp_out=bank("mlp_out"), **loras))
    return layers

m.layers = build_layers()
trained = list(m)


# ==============================================================================
# § Gradient Check
# ==============================================================================
# CHECK_GRADS: the hand-written backward (grads in the Param banks) against autograd
# through the same forward run eagerly, on the first training pack, with B random so A's
# gradients are not zero. The same kernels on both sides, so the gap is rounding order.

if CHECK_GRADS:
    pack = pack_on_device(train_staged, 0)
    for p in trained:
        if "_B_" in p.name:
            p.w.normal_(0, 0.05)
    forward_backward(pack)
    mine = {p.name: p.grad.clone() for p in trained}
    with torch.compiler.set_stance("force_eager"), torch.enable_grad():
        for p in trained:
            p.w.requires_grad_(True)
        m.layers = build_layers()                # views of the leaves, so autograd sees them
        x, cos, sin, tap_keep = embed_forward(pack.tokens, pack.pos)
        for L in m.layers:
            x, _ = layer_forward(x, L, cos, sin, tap_keep, pack.cu, pack.cu_cpu, False)
        probs = readout_forward_backward(x, pack.last_pos, pack.n_options, pack.target, pack.n_rows, False)
        loss = -(pack.target * probs.clamp_min(1e-30).log()).sum() / pack.n_rows
        ref = torch.autograd.grad(loss, [p.w for p in trained])
        for p in trained:
            p.w.requires_grad_(False)
    worst = 0.0
    for p, g in zip(trained, ref):
        rel = ((mine[p.name] - g).norm() / g.norm()).item()
        cos_sim = F.cosine_similarity(mine[p.name].flatten(), g.flatten(), dim=0).item()
        worst = max(worst, rel)
        print(f"  {p.name:24s} |ref| {g.norm().item():.3e}  rel err {rel:.3e}  cos {cos_sim:.6f}")
    print(f"check: hand-written gradients vs autograd, {len(trained)} tensors, worst relative error {worst:.3e}")
    assert worst < 0.05
    sys.exit(0)


# ==============================================================================
# § Evaluation
# ==============================================================================
# Accuracy, Brier score and log loss against the label, and KL from d3-flash's distribution
# where the data prep joined one, on each split's clean rows; Cookbook-dev also by source.

def evaluate(split):
    """-> (the split's summary, one dict per question)."""
    staged = stage_packs(plan_packs(val_rows[split], shuffle=False), train=False)
    questions = []
    for n in range(len(staged.rows)):
        pack = pack_on_device(staged, n)
        probs = forward_backward(pack, backward=False).cpu().numpy()
        for q, row in enumerate(pack.rows):
            p = probs[q, :len(row.options)]
            onehot = np.eye(len(p), dtype=np.float32)[row.label]
            kl = None
            if row.target is not None:
                t = np.asarray(row.target)
                kl = float((t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(p, 1e-12)))).sum())
            questions.append(dict(clean=row.clean, source=row.source, record=row.record, label=row.label,
                                  acc=float(p.argmax() == row.label), brier=float(((p - onehot) ** 2).sum()),
                                  logloss=float(-np.log(max(p[row.label], 1e-12))), kl=kl, p=p.round(4).tolist()))
    clean = [s for s in questions if s["clean"]]
    kls = [s["kl"] for s in clean if s["kl"] is not None]
    by_source = {}
    for s in clean:
        by_source.setdefault(s["source"], []).append(s["acc"])
    summary = dict(n=len(clean), acc=float(np.mean([s["acc"] for s in clean])), brier=float(np.mean([s["brier"] for s in clean])),
                   logloss=float(np.mean([s["logloss"] for s in clean])), kl_d3=float(np.mean(kls)) if kls else None,
                   by_source={k: (int(sum(v)), len(v)) for k, v in sorted(by_source.items())})
    return summary, questions


# ==============================================================================
# § Training Harness
# ==============================================================================

# ------------------------------------------------------------------------------
# §§ Stats
# ------------------------------------------------------------------------------
# One row per step. The CSV header is the field list, the wandb row is the same
# fields under their panel prefixes. A field nobody wrote stays None, and both
# sinks skip it.

@dataclass
class StepStats:

    step:         int          = 0

    # Training
    loss:         float | None = None
    lr:           float | None = None
    dt:           float | None = None   # seconds; unset for steps 0-10
    tok_per_sec:  int   | None = None   # "
    peak_mem_gb:  float | None = None

    # Validation, on val steps only: each split's accuracy, and the panel's time
    decision_dev:    float | None = None
    transfer_dev:    float | None = None
    phase1_holdout:  float | None = None
    phase1_jevbench: float | None = None
    cookbook_dev:    float | None = None
    eval_seconds:    float | None = None

    # Run clocks, in minutes; stamped by log_step
    train_total:  float | None = None
    wall_total:   float | None = None
    eta:          float | None = None


# Field -> wandb panel. A field not named here lands under `train/`.
WANDB_GROUPS = {
    "val":  ("decision_dev", "transfer_dev", "phase1_holdout", "phase1_jevbench", "cookbook_dev", "eval_seconds"),
    "time": ("train_total", "wall_total", "eta"),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(StepStats)}, \
    "WANDB_GROUPS names a field StepStats does not have"


@dataclass
class RunResult:
    """The run in one row: the results panel, the result JSON."""

    # Identity, for the JSON; wandb has these from the run name and config
    run_name:      str          = ""
    num_steps:     int          = 0
    pack_tokens:   int          = 0

    # The result: the final panel, accuracy per split
    panel:         dict | None  = None

    # The cost, in minutes
    train_time:    float        = 0.0
    val_time:      float        = 0.0
    compile_time:  float        = 0.0
    wall_time:     float        = 0.0

    # The rate
    avg_step_time: float        = 0.0    # seconds
    tok_per_sec:   int          = 0
    peak_mem_gb:   float        = 0.0


# ------------------------------------------------------------------------------
# §§ Logging
# ------------------------------------------------------------------------------

def write_checkpoint(step):
    """The trained tensors (LoRA's banks and the readout; ~150 MB) and the run's config."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=step, config={name: getattr(cfg, name) for name in DecisionConfig.__annotations__},
                    weights={p.name: p.w.cpu() for p in m}),
               RUN_DIR / "trained.pt")

os.makedirs("logs", exist_ok=True)
logfile = f"logs/{cfg.run_name}.txt"
print(logfile)

def print0(s="", console=False):
    with open(logfile, "a") as f:
        if console:
            print(s, flush=True)
        print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

print0(f"Frozen parameters: {cfg.num_frozen_params:,} | trained: {cfg.num_params:,}", console=True)
print0(f"GPU: {props.name}", console=True)
print0(f"Pack: {cfg.pack_tokens:,} tokens, {cfg.max_rows} segments; {cfg.num_steps} steps", console=True)

# A flat row per step and the result, for analysis without wandb.
metrics_path = f"logs/{cfg.run_name}_metrics.csv"
result_path  = f"logs/{cfg.run_name}_result.json"
questions_path = f"logs/{cfg.run_name}_questions.json"
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
        # The config, verbatim: every DecisionConfig field, defaults and derived.
        config={name: getattr(cfg, name) for name in DecisionConfig.__annotations__},
    )
    wandb.define_metric("step", hidden=True)   # the x-axis, not a series
    wandb.define_metric("*", step_metric="step")


# ==============================================================================
# § Training Loop
# ==============================================================================

# Schedule position: one () int64 device tensor, advanced on-device
t_step = torch.zeros((), dtype=torch.int64, device=device)

panel = None
panel_questions = None
total_val_time = 0.0
timed = []   # Length of each step in seconds, steps 0-10 excluded.
             # Total training time = np.sum(timed).
warmup = []  # Those first 11 steps, where the compile lives.

# Training loop
for step in range(cfg.num_steps + 1):
    # An abort cuts the loop but not the schedule.
    # Set ABORT_STEP=0 to evaluate the base model only.
    last_step = step == (cfg.num_steps if ABORT_STEP is None else ABORT_STEP)

    stats = StepStats(step=step)

    # --------------- Evaluation Panel -----------------
    if cfg.val_splits and (last_step or (cfg.val_every > 0 and step % cfg.val_every == 0)):
        torch.cuda.synchronize()
        val_t0 = time.perf_counter()
        panel, panel_questions = {}, {}
        for split in cfg.val_splits:
            panel[split], panel_questions[split] = evaluate(split)
            res = panel[split]
            kl_text = f" kl_d3 {res['kl_d3']:.3f}" if res["kl_d3"] is not None else ""
            print0(f"step:{step}/{cfg.num_steps} val {split}: n {res['n']} acc {res['acc']:.4f} brier {res['brier']:.4f} "
                   f"logloss {res['logloss']:.4f}{kl_text}", console=True)
            if split == "cookbook_dev":
                print0("    by source: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res["by_source"].items()), console=True)
            setattr(stats, split, res["acc"])
        val_elapsed = time.perf_counter() - val_t0
        total_val_time += val_elapsed
        print0(f"step:{step}/{cfg.num_steps} val_time:{val_elapsed:.1f}s", console=True)
        stats.eval_seconds = val_elapsed

    # --------------- Checkpoint -----------------
    if cfg.save_checkpoint and last_step:
        ckpt_t0 = time.perf_counter()
        write_checkpoint(step)
        print0(f"checkpoint captured at step {step} ({time.perf_counter() - ckpt_t0:.1f}s) -> {RUN_DIR}", console=True)

    # Exit final step after validation and checkpoint
    if last_step:
        log_step(stats)
        break

    # --------------- Training Step -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()

    pack = pack_on_device(train_staged, step)

    # Forward and Backward pass: the LoRA gradients land in their banks
    loss = forward_backward(pack)

    # Update weights
    adamw_step(trained, t_step)
    t_step.add_(1)  # advance the schedule on-device

    train_loss = loss.item()
    torch.cuda.synchronize()
    dt = time.perf_counter() - step_t0

    # --------------- Timing and Logging -----------------
    pct_done = 100 * step / cfg.num_steps

    # Track time and ETA after first 10 steps to exclude compile time.
    if step > 10:
        timed.append(dt)
        remaining_time = (cfg.num_steps - step - 1) * np.mean(timed) / 60
        eta_str = f" | eta: {remaining_time:.1f}m"
    else:
        warmup.append(dt)
        eta_str = ""

    tok_per_sec = int(cfg.pack_tokens / dt)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 2**30

    print0(f"step {step:05d}/{cfg.num_steps:05d} ({pct_done:.2f}%) | loss: {train_loss:.4f} | rows: {int(pack.n_rows):3d} | "
           f"lr: {cfg.lr * lr_mult_t[step]:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
           f"peak: {peak_mem_gb:.1f} GiB | total time: {np.sum(timed)/60:.2f}m{eta_str}", console=True)

    stats.loss = train_loss
    stats.lr = float(cfg.lr * lr_mult_t[step])
    stats.peak_mem_gb = peak_mem_gb
    if step > 10:   # the compile and warm-up rates are not the run's rates
        stats.dt, stats.tok_per_sec = dt, tok_per_sec
        stats.eta = remaining_time
    log_step(stats)

    # Every graph compiles in step 0 (and the first panel): one per kind and direction. Flag
    # any recompile after step 10.
    if step == 0:
        print0(f"compiled graphs after step 0: {dynamo.utils.counters['stats']['unique_graphs']}", console=True)
    elif step == 10:
        torch._logging.set_logs(recompiles=True)


# ------------------------------------------------------------------------------
# §§ Results
# ------------------------------------------------------------------------------

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

metrics_file.close()

# The warm-up steps cost the compile plus whatever they were going to cost anyway.
avg_step_time = float(np.mean(timed)) if timed else 0.0
compile_time = max(0.0, np.sum(warmup) - len(warmup) * avg_step_time)

result = RunResult(
    run_name      = cfg.run_name,
    num_steps     = cfg.num_steps,
    pack_tokens   = cfg.pack_tokens,

    panel         = {split: res["acc"] for split, res in panel.items()} if panel else None,

    train_time    = float(np.sum(timed)) / 60,
    val_time      = total_val_time / 60,
    compile_time  = compile_time / 60,
    wall_time     = (time.perf_counter() - run_wall_t0) / 60,

    avg_step_time = avg_step_time,
    tok_per_sec   = int(cfg.pack_tokens / avg_step_time) if avg_step_time else 0,
    peak_mem_gb   = torch.cuda.max_memory_reserved() / 2**30,
)

row = asdict(result)
with open(result_path, "w") as f:
    json.dump(dict(row, final_panel=panel), f, indent=1)
if panel_questions is not None:
    with open(questions_path, "w") as f:
        json.dump(panel_questions, f)
wandb_run.log({"step": cfg.num_steps,
               **{f"final/{k}": v for k, v in row.items() if isinstance(v, (int, float)) and k != "num_steps"}})

print0(f"== {result.run_name} ==", console=True)
if result.panel:
    print0("  panel: " + " | ".join(f"{s} {a:.4f}" for s, a in result.panel.items()), console=True)
print0(f"  train {result.train_time:.2f}m | val {result.val_time:.2f}m | compile {result.compile_time:.2f}m | wall {result.wall_time:.2f}m", console=True)
if timed:
    print0(f"  {cfg.pack_tokens:,} tokens/step over {len(timed):,} timed steps: mean {result.avg_step_time:.3f}s | "
           f"median {np.median(timed):.3f}s | {result.tok_per_sec:,} tok/s", console=True)
print0(f"  rows -> {metrics_path} | result -> {result_path}", console=True)

for _f in (logfile, metrics_path, result_path):
    wandb_run.save(_f, policy="now")
wandb_run.finish()
