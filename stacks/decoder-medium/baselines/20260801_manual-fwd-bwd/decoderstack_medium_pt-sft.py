# decoderstack_medium_pt-sft-fable.py
#
# Single-file d24 pre-training pipeline with a handwritten forward/backward and a
# written-out optimizer: no autograd, no torch.optim, no param groups, no nn.Module.
#
# Core design decisions:
# - No nn.Module, no m.to, no state_dict / load_state_dict.
#   - Every tensor is created directly on the device, at its final dtype.
#   - No accommodations for "prior checkpoints", we're starting from scratch.
# - No torch.optim or autograd, we're doing everything manually.
# - Use globals -- global cfg, global m -- don't pass things around.
#   - The model is a plain class used as a namespace of plain torch.Tensors.
#     nn.Parameter does nothing for us: Parameter exists for autograd leaf
#     bookkeeping and Module registration, neither of which we use. Plain
#     tensors are directly usable in the math (m.W_in, not m.W_in.weight),
#     accept attached state (.grad32, .mantissa, ...) just like Parameters,
#     and default to requires_grad=False -- which is what we want everywhere,
#     because we implement grad.
# - Dtypes are hardcoded everywhere -- stated at creation, never inferred by
#   matching another tensor's dtype. (No fp64 parity tier in this file.)
# - Hardcoded to the d24 config; none of nanochat's auto-scaling by model size.
# - Multi-GPU shards the optimizer, not the model (nanochat's scheme): every
#   rank holds the full bf16 live weights and full grad accumulators, optimizer
#   state is allocated at shard sizes, and optimizer_step wraps the same update
#   kernels in reduce-scatter -> owned-shard update -> live all-gather.
# - FP8 is currently a separate file.
#
# The "§" technique defines the code sections in here.
#
# The model/training code comes from the nanochat repo, branch fwd-bwd
# (nanochat/train_step.py, nanochat/gpt.py). That branch's d24 run is the
# reference implementation we want to match -- we're refactoring and dropping
# baggage, not changing the math:
# C:\Users\chris\Documents\GitHub\agent-ops\nanochat\2026-07-29_0833am_d24-throughput-gap\NOTES.md
#
# The code below the seam (marked near the bottom) comes from the 'stacks' repo,
# pulled mainly for the pre-tokenized data + distributed loader and CORE eval.
#
# One-off derived quantities (parameter counts, flops/token, the training
# horizon, the LR/WD batch corrections, cu_seqlens sizing) are HARDCODED in
# this script; `scaling.py` (kept alongside it) recomputes and documents them.


# --------------------------------------------------------------------------------
# § Setup
# --------------------------------------------------------------------------------

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
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kernels import get_kernel

dynamo.config.recompile_limit = 64

# ==== Distributed setup ====
# dist is always initialized (launch under torchrun, even for one process) --
# the data pipeline below the seam uses dist.barrier() and the loader shards
# by rank.
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0)

def print0(*args, console=False, **kwargs):
    if master_process:
        print(*args, **kwargs)

# ==== Wandb helpers ====

class DummyWandb:
    """No-op wandb replacement when logging is disabled."""
    def log(self, *args, **kwargs): pass
    def finish(self): pass
        

# BF16 dense peak FLOPS by GPU, for the MFU denominator. Just the GPUs this
# pipeline actually runs on; the full many-vendor table (and sources) lives in
# scaling.py. GH200 carries the same H100-class SXM die: 989 TFLOPS.
PEAK_FLOPS = {"GH200": 989e12, "H100": 989e12, "A100": 312e12}

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


# --------------------------------------------------------------------------------
# § Flash Attention (raw FA3 forward/backward)
# --------------------------------------------------------------------------------
# The handwritten backward calls FA3's raw _flash_attn_forward/_flash_attn_backward
# torch.library ops directly -- no autograd Function in between. The forward
# returns the softmax LSE, which the backward consumes alongside the stashed
# output. FA3 only; there is no SDPA/naive fallback in this file.

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
    """Attention forward that also returns what the handwritten backward needs:
    (out, softmax_lse), with lse (H, T) fp32."""
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


# --------------------------------------------------------------------------------
# § Model Config
# --------------------------------------------------------------------------------

# Value embeddings (ResFormer-style) live on alternating layers, last always
# included. Banked over just the VE layers; ve_index maps layer -> bank slot
# (-1 = no VE on this layer) and is read by every forward body.
#
# Note: Deriving head size or count from d_model is a bad habit that has
#       propagated through ~everyone's model code.
#       There are only three real constraints--these values must match:
#         1. Number of key and value heads
#         2. Query-key head sizes
#         3. Value-output head sizes
#
# Recommended short window size:
# -(-seq_len // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
class StackConfig:

    # Model
    n_layers:   int = 24
    d_model:    int = 1536

    # Input
    d_vocab:    int = 32768 # Must arrive padded (tensor cores, sharding) -- no
                            # auto-padding in this file; asserted below.
    d_smr_gate: int = 24    # Input to smear gate is first 'd' positions of the
                            # normed input embedding.
    # Attention
    n_q_heads:  int = 12
    n_kv_heads: int = 12
    n_o_heads:  int = 12
    d_qk:       int = 128 # Note: FA2 requires d_qk == d_vo, FA3 does not.
    d_vo:       int = 128

    # Context and Sliding Window Attention
    seq_len:          int = 2048
    short_win_size:   int = 768
    full_ctxt_layers: list[int] = [   3,    7,    11,     15,     19,     23] # "sssL" pattern

    window_sizes:     list[tuple[int, int]]  # Derived below.

    # Attention - Value Embeddings
    d_ve_gate: int = 12  # First 'd' positions of residual stream (after x0
                         # blending and norm) are the gate input.
                         # ve gates exist per head, per layer.
    ve_layers: list[int] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] # 0-indexed.
    ve_index:  list[int] # Derived from ve_layers.
    num_ves:   int

    # MLP
    d_mlp:      int = 4 * 1536

    # Training batch (nanochat d24 speedrun spec). Tokens, not sequences: with
    # varlen packing a micro-batch is one packed 1-D stream, so the token count
    # is the real quantity (= 16 seqs x 2048 in nanochat's batched terms).
    # Total batch 2^20 tokens/step is nanochat's Power Lines auto-compute for
    # d24.
    micro_batch_tokens: int = 32768   # per rank, per micro-batch
    total_batch_size:   int = 2**20   # tokens per optimizer step

    # Training horizon: the d24 speedrun spec (data:param ratio 8) --
    # 8 x 729,810,624 scaling params = 5,838,484,992 tokens // 2^20 per step
    # = 5,568 steps. Derivation: scaling.py.
    num_iterations: int = 5568

    # Evaluation and logging
    val_tokens:      int = 10485760   # per val-bpb pass: 320 training-shaped micro-batches
    val_loss_every:  int = 250
    eval_buffer_tokens: int = 65536   # CORE/chat eval packing buffer. Eval is
                                      # forward-only (no stash, no grads), so a
                                      # buffer well past the training micro-batch
                                      # fits easily; the rotary cache is sized to
                                      # cover it.
    save_checkpoint: bool = True
    # Mid-run checkpoint capture, in COMPLETED optimizer steps: state is
    # written on entering these loop steps (the final state always saves).
    # 1950 = the first LR/momentum-cooldown step at the 5568-step horizon
    # (the hold ends after update 1949 = N - round(0.65*N)): the last
    # uncooled state -- the one to resume from to train the horizon longer.
    save_steps:      tuple = (1950,)
    run_id:          str = f"{str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))}-d24"
    wandb_run:       str = "dummy"    # "dummy" disables wandb
    wandb_project:   str = "decoderstack"

cfg = StackConfig() # Make config a global, don't pass it around.

# Sanity: the constraints the axes above must satisfy.
assert cfg.d_vocab % 64 == 0, "vocab must arrive padded to 64 (no auto-padding here)"
assert cfg.n_o_heads == cfg.n_q_heads, "attention output consumes one slot per query head"
assert cfg.n_q_heads % cfg.n_kv_heads == 0, "GQA needs query heads to tile over kv heads"
assert cfg.d_qk % 2 == 0, "rotary splits the qk head dim in half"
assert cfg.full_ctxt_layers[-1] == cfg.n_layers - 1, "final layer recommended to have full context"

# Derived quantities:
# Map layers to VE bank slots.
cfg.ve_index = [cfg.ve_layers.index(i) if i in cfg.ve_layers else -1 for i in range(cfg.n_layers)]
cfg.num_ves = len(cfg.ve_layers)

# Per-layer window sizes for sliding window attention.
# List of (left, right) tuples for FA3's window_size parameter:
# - left: how many tokens before current position to attend to
# - right: how many tokens after current position to attend to (0 for causal)
# "Full context" is (seq_len, 0): documents are at most seq_len tokens and
# varlen attention is doc-isolated, so a seq_len window is unlimited in effect.
cfg.window_sizes = [(cfg.short_win_size, 0)] * cfg.n_layers  # All short, ...
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (cfg.seq_len, 0)                   # ... then overwrite with full.

# Derived batch quantities. Fixed total => grad accum scales down as GPUs are
# added: 32 at world=1, 4 at world=8. grad_scale rides into forward_backward
# as loss_scale, replacing the loss division of an autograd loop; at world>1
# it composes with ReduceOp.AVG grad comm to give the global batch mean.
assert cfg.total_batch_size % (cfg.micro_batch_tokens * world_size) == 0, \
    "total batch must divide evenly into per-rank micro-batches"
grad_accum_steps = cfg.total_batch_size // (cfg.micro_batch_tokens * world_size)
grad_scale = 1 / grad_accum_steps


# --------------------------------------------------------------------------------
# § Shard Assignment
# --------------------------------------------------------------------------------
# Each GPU is responsible for a "shard" of the optimizer work:
# - Muon banks shard over their layer axis (dim 0).
# - AdamW params shard over the row axis of their (rows, cols) view -- vocab
#   rows for input_embeds/lm_head, flattened (ve_slot * vocab) rows for
#   value_embeds. (ve_slot alone is too small to divide across a world, and the
#   rows are interchangeable for AdamW's elementwise update.)
# - ve_gate is NOT sharded: it is tiny (~thousands of floats) and ragged
#   against world sizes, so every rank runs the full-size update instead.
# - grad32 always stays FULL size on every rank -- it is the source buffer for
#   the reduce-scatter, not a shard.
#
# No zero-padding support: every sharded axis must divide evenly (asserted
# below). d24's axes -- 24 layers, 32768 vocab rows, 393,216 ve rows -- all
# divide by the world sizes we'd run (1, 2, 4, 8).
#
# At world_size == 1 every shard IS the whole tensor: the slices below span
# their full axes and optimizer_step's collectives short-circuit. One code
# path, degenerate comm.

assert cfg.n_layers % world_size == 0, \
    f"Muon layer-sharding needs n_layers % world == 0 ({cfg.n_layers} % {world_size})"
layer_shard_size  = cfg.n_layers // world_size
layer_shard_start = rank * layer_shard_size
layer_shard_slice = slice(layer_shard_start, layer_shard_start + layer_shard_size)

assert cfg.d_vocab % world_size == 0, \
    f"AdamW row-sharding needs vocab % world == 0 ({cfg.d_vocab} % {world_size})"
vocab_shard_size  = cfg.d_vocab // world_size
vocab_shard_start = rank * vocab_shard_size
vocab_shard_slice = slice(vocab_shard_start, vocab_shard_start + vocab_shard_size)

ve_rows = cfg.num_ves * cfg.d_vocab
assert ve_rows % world_size == 0, \
    f"AdamW row-sharding needs ve_slot*vocab % world == 0 ({ve_rows} % {world_size})"
ve_row_shard_size  = ve_rows // world_size
ve_row_shard_start = rank * ve_row_shard_size
ve_row_shard_slice = slice(ve_row_shard_start, ve_row_shard_start + ve_row_shard_size)


# --------------------------------------------------------------------------------
# § Model Initialization
# --------------------------------------------------------------------------------

class Model:
    """Namespace of plain tensors -- the live weights. Each weight also carries
    its training state as attached attributes, allocated alongside it below:

      .grad32        full-size gradient accumulator (fp32; bf16 for the two
                     embedding tables), explicitly zeroed between steps
      .grad32_slices per-layer views of grad32 for the 3-D banks (see below)
      .mantissa      lower 16 bits of the fp32 master (uint16, shard-size)
      .frst_mntm     Muon first moment (fp32, shard-size)
      .scnd_mntm     Muon factored second moment (fp32, shard-size)
      .residual_dim  the weight axis that faces the residual stream (-1 or -2);
                     NorMuon's per-neuron mean-square is taken along it
      .exp_avg       AdamW first moment (fp32, shard-size)
      .exp_avg_sq    AdamW second moment (fp32, shard-size)
    """

    # Input
    input_embeds: Tensor
    smear_gate:   Tensor
    smear_lambda: Tensor

    # Attention
    W_Q: Tensor
    W_K: Tensor
    W_V: Tensor
    W_O: Tensor
    value_embeds: Tensor
    ve_gate:      Tensor

    # MLP
    W_in:  Tensor
    W_out: Tensor

    # Cross-Layer
    resid_lambdas:  Tensor  # Per-layer gain on the residual stream.
    x0_lambdas:     Tensor  # Per-layer coefficient for reading the input embedding.
    backout_lambda: Tensor  # How much of layer 16's output to remove from the stream
                            # prior to the lm head.

    # Output
    lm_head: Tensor

    # Buffers (rotary cache; not trained, not checkpointed)
    cos: Tensor
    sin: Tensor

    # The trained weights, in declaration order -- this tuple defines "every
    # trained weight". __iter__ walks them so call sites can just say
    # `for p in m` (grad zeroing); the names key the checkpoint dicts.
    weight_names = ("input_embeds", "smear_gate", "smear_lambda",
                    "W_Q", "W_K", "W_V", "W_O", "value_embeds", "ve_gate",
                    "W_in", "W_out", "resid_lambdas", "x0_lambdas",
                    "backout_lambda", "lm_head")

    def __iter__(self):
        return (getattr(self, n) for n in self.weight_names)


# ==== Tensor Creation Idioms ====
# Reduce the boilerplate for defining weights and buffers.

fp32_empty   = lambda *shape: torch.empty(*shape, dtype=torch.float32, device=device)
bf16_empty   = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16, device=device)
fp32_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
bf16_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.bfloat16, device=device)
uint16_zeros = lambda *shape: torch.zeros(*shape, dtype=torch.uint16, device=device)

# We use fp32 for the "master" weights, which are what we store on disk, and for 
# avoiding rounding off small optimizer updates. 
# All forward and backward computation is done on bf16 matrices (the "live" weights). 
# Note that bf16 is just fp32 with the lower 16-bits of mantissa dropped;
# rather than hold 16-bit and 32-bit copies at once, we stash those lower
# 16 mantissa bits, and reconstruct the full 32-bit precision to update then
# resplit.
upper_bf16   = lambda w: (w.contiguous().view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
lower_uint16 = lambda w: (w.contiguous().view(torch.int32)      ).to(torch.int16).view(torch.uint16)

# Set the seed so that every rank gets the same initialization -- no broadcast
# from a master rank needed.
torch.manual_seed(42)
torch.cuda.manual_seed(42)

m = Model()

# Written out one tensor per line, deliberately: the shape, the dtype, and
# therefore the memory cost of every weight and every piece of optimizer state
# is readable in one place, and the axis names say which dimension is sharded.
#
# Dtype scheme (hardcoded, stated per tensor below):
# - Matrix banks + lm_head: bf16 live + uint16 mantissa (fp32 master via the
#   mantissa trick), fp32 gradients, fp32 moments.
# - Embedding tables (input_embeds, value_embeds): bf16 live + uint16 mantissa
#   (fp32 master via the mantissa trick). This deviates from nanochat, which
#   kept its embeddings plain bf16 and let AdamW update them in place -- we
#   pair them with a mantissa so the one AdamW kernel serves everything,
#   rather than carrying a second bf16-live variant. Overall our code
#   ~matches the validation loss of the original.
#   Gradients are bf16 -- these are the two biggest tensors in the model,
#   fp32 grads would double their scatter traffic and (at world>1) comm bytes,
#   and bf16 matches the autograd baseline's numerics (bf16 params -> bf16
#   .grad). Everything else accumulates gradients in fp32.
# - Scalars (resid/x0 lambdas, smear, backout): fp32 live, no mantissa, same
#   as they've always been. (Rounding them to bf16 was tried during the port
#   and cost +0.016 val bpb, so they stay fp32.)
#
# Initialization values:
#   input_embeds:   normal,  std=0.8
#   lm_head:        normal,  std=0.001
#   W_Q, W_K, W_V:  uniform, bound=sqrt(3)/sqrt(d_model)     -> std = 1/sqrt(d_model)
#   W_O:            zeros
#   W_in:           uniform, bound=0.4*sqrt(3)/sqrt(d_model)  -> std = 0.4/sqrt(d_model)
#   W_out:          zeros
#   value_embeds:   uniform, bound=sqrt(3)/sqrt(d_model)      (same as W_V)
#   ve_gate:        uniform in [0, 0.02] (slightly above neutral)
#   resid_lambdas:  1.15 -> 1.05 linear decay over depth
#   x0_lambdas:     0.20 -> 0.05 linear decay over depth
#   smear_gate:     zeros
#   smear_lambda:   zeros (smear disabled at init)
#   backout_lambda: zeros (backout disabled at init)
#   (Zeros for smear/backout is what nanochat's baselines actually trained
#   with: it intended backout_lambda=0.2 and a kaiming smear_gate, but its
#   meta-device init never ran those. Details at the Scalars block below.)

# Uniform init bound. Var(Uniform(-a, a)) = a^2/3, so std = a/sqrt(3): to hit
# a target std of 1/sqrt(d_model), the bound must be sqrt(3) times it.
matrix_init_s = (3 ** 0.5) * (cfg.d_model ** -0.5)

# ==== Input Embeddings ====
# bf16 live; draw in fp32 and let copy_ round -- drawing straight into bf16
# would quantize the distribution rather than the samples. The master upcast of
# a bf16 live is lossless, so the mantissa starts at zero.
m.input_embeds = bf16_empty(cfg.d_vocab, cfg.d_model)
m.input_embeds.copy_(fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.8))

m.input_embeds.grad32     = bf16_zeros(cfg.d_vocab, cfg.d_model)
m.input_embeds.mantissa   = uint16_zeros(vocab_shard_size, cfg.d_model)
m.input_embeds.exp_avg    = fp32_zeros(vocab_shard_size, cfg.d_model)
m.input_embeds.exp_avg_sq = fp32_zeros(vocab_shard_size, cfg.d_model)

# ==== Value Embeddings ====
# Same init std as W_V; same bf16-live / zero-mantissa path as input_embeds.
# AdamW state is shaped over the FLATTENED (ve_slot * vocab) row axis;
# optimizer_step passes matching 2-D views of the live bank and its grad.
# Flattening (vs a 3-D state mirroring the bank) is what lets ONE
# reduce-scatter/all-gather over dim-0 rows shard the whole bank evenly --
# per-slot vocab sharding on the 3-D layout would need a collective per VE
# slot. At world=1 a 3-D state would also work, but would have to reallocate
# the moment we go multi-GPU.
m.value_embeds = bf16_empty(cfg.num_ves, cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
m.value_embeds.copy_(fp32_empty(cfg.num_ves, cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
                     .uniform_(-matrix_init_s, matrix_init_s))

m.value_embeds.grad32     = bf16_zeros(cfg.num_ves, cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
m.value_embeds.mantissa   = uint16_zeros(ve_row_shard_size, cfg.n_kv_heads * cfg.d_vo)
m.value_embeds.exp_avg    = fp32_zeros(ve_row_shard_size, cfg.n_kv_heads * cfg.d_vo)
m.value_embeds.exp_avg_sq = fp32_zeros(ve_row_shard_size, cfg.n_kv_heads * cfg.d_vo)

# ==== LM Head ====
# Drawn in fp32 and split -- unlike the embeddings, its mantissa is real from
# step zero.
lm_head_fp32 = fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.001)

m.lm_head          = upper_bf16(lm_head_fp32)                       # Live weights - bf16
m.lm_head.mantissa = lower_uint16(lm_head_fp32[vocab_shard_slice])  # Lower 16 bits for optimizer

del lm_head_fp32

m.lm_head.grad32     = fp32_zeros(cfg.d_vocab, cfg.d_model)
m.lm_head.exp_avg    = fp32_zeros(vocab_shard_size, cfg.d_model)
m.lm_head.exp_avg_sq = fp32_zeros(vocab_shard_size, cfg.d_model)

# ==== Attention ====
# Parameter banks: the layer index is dim 0. Each slice uses F.linear's
# (out_features, in_features) convention and is consumed as `x @ w.mT`.
# Initialize in fp32 and split into bf16 live + uint16 mantissa.
W_Q_fp32 = fp32_empty(cfg.n_layers, cfg.n_q_heads  * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_K_fp32 = fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_V_fp32 = fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_vo, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_O_fp32 = fp32_zeros(cfg.n_layers,               cfg.d_model, cfg.n_o_heads * cfg.d_vo)  # projections start at zero

m.W_Q = upper_bf16(W_Q_fp32) # Live weights - bf16
m.W_K = upper_bf16(W_K_fp32)
m.W_V = upper_bf16(W_V_fp32)
m.W_O = upper_bf16(W_O_fp32)

# For the mantissa, we only need to hold our shard of the weights.
m.W_Q.mantissa = lower_uint16(W_Q_fp32[layer_shard_slice]) # Lower 16 bits for optimizer
m.W_K.mantissa = lower_uint16(W_K_fp32[layer_shard_slice])
m.W_V.mantissa = lower_uint16(W_V_fp32[layer_shard_slice])
m.W_O.mantissa = lower_uint16(W_O_fp32[layer_shard_slice])

del W_Q_fp32, W_K_fp32, W_V_fp32, W_O_fp32

# Gradients (full size -- the reduce-scatter source, never sharded)
m.W_Q.grad32 = fp32_zeros(cfg.n_layers, cfg.n_q_heads  * cfg.d_qk, cfg.d_model)
m.W_K.grad32 = fp32_zeros(cfg.n_layers, cfg.n_kv_heads * cfg.d_qk, cfg.d_model)
m.W_V.grad32 = fp32_zeros(cfg.n_layers, cfg.n_kv_heads * cfg.d_vo, cfg.d_model)
m.W_O.grad32 = fp32_zeros(cfg.n_layers,               cfg.d_model, cfg.n_o_heads * cfg.d_vo)

# First-momentum buffers for Muon (sharded)
m.W_Q.frst_mntm = fp32_zeros(layer_shard_size, cfg.n_q_heads  * cfg.d_qk, cfg.d_model)
m.W_K.frst_mntm = fp32_zeros(layer_shard_size, cfg.n_kv_heads * cfg.d_qk, cfg.d_model)
m.W_V.frst_mntm = fp32_zeros(layer_shard_size, cfg.n_kv_heads * cfg.d_vo, cfg.d_model)
m.W_O.frst_mntm = fp32_zeros(layer_shard_size,               cfg.d_model, cfg.n_o_heads * cfg.d_vo)

# Second momentum (NorMuon variance reduction) holds a running average of each
# neuron's mean-square update, so it is a vector (per layer) rather than a
# matrix mirroring the weights. (The neuron's rms is the square root of what's
# stored; the kernel applies it as an rsqrt.)
# NorMuon is a ~no-op for square matrices: polar express produces a
# ~orthonormal matrix, so the neuron norms are already ~uniform and there is
# nothing to normalize (confirmed with experiments). It only affects attention
# when the number of heads times the head size differs from d_model.
# The original code uses a heuristic to infer the neuron dimension by assuming
# that it is the smaller of the two. While typical, it's not certain. Instead,
# we specify it directly.
# Neurons can be identified directly by their interaction with the residual
# stream--they read from it and write to it and match it in length, so the
# mean-square is taken along the residual dimension.
# Note that the attention output projection consists of heads as well, and
# they are stored transposed relative to QKV, so we calculate the mean-square
# along dim -2.
m.W_Q.residual_dim = -1
m.W_K.residual_dim = -1
m.W_V.residual_dim = -1
m.W_O.residual_dim = -2
m.W_Q.scnd_mntm = fp32_zeros(layer_shard_size, cfg.n_q_heads  * cfg.d_qk, 1)
m.W_K.scnd_mntm = fp32_zeros(layer_shard_size, cfg.n_kv_heads * cfg.d_qk, 1)
m.W_V.scnd_mntm = fp32_zeros(layer_shard_size, cfg.n_kv_heads * cfg.d_vo, 1)
m.W_O.scnd_mntm = fp32_zeros(layer_shard_size, 1, cfg.n_o_heads * cfg.d_vo)

# ==== MLPs ====
# For a transformer, 'MLP' is something of a misnomer. It's closer to a
# lookup table, containing pairs of vectors, both of length d_m.
# For a given pair (w_in, w_out), if the residual stream is positively
# aligned with w_in, then w_out is written back to it.
# But unlike a look up table, where a read-write operation is captured
# by a single row, here the model composes the operation across many
# vector pairs.

W_in_fp32  = fp32_empty(cfg.n_layers, cfg.d_mlp,   cfg.d_model).uniform_(-matrix_init_s * 0.4, matrix_init_s * 0.4)
W_out_fp32 = fp32_zeros(cfg.n_layers, cfg.d_model, cfg.d_mlp)             # projections start at zero

m.W_in  = upper_bf16(W_in_fp32) # Live weights - bf16
m.W_out = upper_bf16(W_out_fp32)

m.W_in.mantissa  = lower_uint16(W_in_fp32[layer_shard_slice]) # Lower 16 bits for optimizer
m.W_out.mantissa = lower_uint16(W_out_fp32[layer_shard_slice])

del W_in_fp32, W_out_fp32

# Gradients (full size)
m.W_in.grad32  = fp32_zeros(cfg.n_layers, cfg.d_mlp,   cfg.d_model)
m.W_out.grad32 = fp32_zeros(cfg.n_layers, cfg.d_model, cfg.d_mlp)

# First-momentum buffers for Muon (sharded)
m.W_in.frst_mntm  = fp32_zeros(layer_shard_size, cfg.d_mlp,   cfg.d_model)
m.W_out.frst_mntm = fp32_zeros(layer_shard_size, cfg.d_model, cfg.d_mlp)

# Residual dimension: W_in rows read from the residual stream, W_out columns
# write to it.
m.W_in.residual_dim  = -1
m.W_out.residual_dim = -2
m.W_in.scnd_mntm  = fp32_zeros(layer_shard_size, cfg.d_mlp, 1)
m.W_out.scnd_mntm = fp32_zeros(layer_shard_size, 1, cfg.d_mlp)

# ==== VE Gates ====
# Muon, REPLICATED: tiny and ragged against world sizes, so every rank runs the
# full-size update rather than paying comm to shard a few thousand floats.
ve_gate_fp32 = fp32_empty(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate).uniform_(0.0, 0.02)

m.ve_gate          = upper_bf16(ve_gate_fp32)
m.ve_gate.mantissa = lower_uint16(ve_gate_fp32)  # replicated: full-size mantissa

del ve_gate_fp32

m.ve_gate.grad32     = fp32_zeros(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate)
m.ve_gate.frst_mntm  = fp32_zeros(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate)
m.ve_gate.residual_dim = -1  # gate rows read a d_ve_gate slice of the residual stream
m.ve_gate.scnd_mntm  = fp32_zeros(cfg.num_ves, cfg.n_kv_heads, 1)

# ==== Scalars ====
# fp32-LIVE with no mantissa pair (see the dtype scheme note above). AdamW,
# replicated.
# These serve separate purposes:
# - resid_lambdas: Directly scales the residual stream at the start of each layer.
# - x0_lambdas: How strongly the input embedding is added to the residual stream.
# Per-layer scalars: linear decay over depth. Stronger residual and more
# input-embedding blending at early layers, both tapering with depth.
m.resid_lambdas = torch.linspace(1.15, 1.05, cfg.n_layers, dtype=torch.float32, device=device)
m.x0_lambdas    = torch.linspace(0.20, 0.05, cfg.n_layers, dtype=torch.float32, device=device)

# Smear/backout start disabled, zeros everywhere.
# Note: nanochat pre-flattening had a bug here--it intended backout_lambda=0.2
# and a kaiming smear_gate, but under meta-device init those never executed and
# to_empty() left zeroed storage. Zeros is what every tuned baseline actually
# trained with, so now it's explicit rather than luck.
m.smear_gate     = fp32_zeros(1, cfg.d_smr_gate)
m.smear_lambda   = fp32_zeros(1)
m.backout_lambda = fp32_zeros(1)

m.resid_lambdas.grad32  = fp32_zeros(cfg.n_layers)
m.x0_lambdas.grad32     = fp32_zeros(cfg.n_layers)
m.smear_gate.grad32     = fp32_zeros(1, cfg.d_smr_gate)
m.smear_lambda.grad32   = fp32_zeros(1)
m.backout_lambda.grad32 = fp32_zeros(1)

m.resid_lambdas.exp_avg     = fp32_zeros(cfg.n_layers)
m.resid_lambdas.exp_avg_sq  = fp32_zeros(cfg.n_layers)
m.x0_lambdas.exp_avg        = fp32_zeros(cfg.n_layers)
m.x0_lambdas.exp_avg_sq     = fp32_zeros(cfg.n_layers)
m.smear_gate.exp_avg        = fp32_zeros(1, cfg.d_smr_gate)
m.smear_gate.exp_avg_sq     = fp32_zeros(1, cfg.d_smr_gate)
m.smear_lambda.exp_avg      = fp32_zeros(1)
m.smear_lambda.exp_avg_sq   = fp32_zeros(1)
m.backout_lambda.exp_avg    = fp32_zeros(1)
m.backout_lambda.exp_avg_sq = fp32_zeros(1)

# ==== Rotary Cache ====
# Without an nn.Module these are just attributes on m -- register_buffer only
# existed for state_dict/.to() plumbing we no longer have (and these were
# persistent=False anyway). With varlen training the whole micro-batch is one
# packed sequence, so the cache spans the largest T any forward sees: the
# training micro-batch (val micro-batches match it) or the CORE/chat eval
# packing buffer, whichever is bigger. The assert in the forward bodies
# catches it if we ever exceed.
rotary_seq_len = max(cfg.micro_batch_tokens, cfg.eval_buffer_tokens)
channel_range = torch.arange(0, cfg.d_qk, 2, dtype=torch.float32, device=device)  # stride the channels
inv_freq = 1.0 / (100000 ** (channel_range / cfg.d_qk))
t_pos = torch.arange(rotary_seq_len, dtype=torch.float32, device=device)          # stride the time steps
freqs = torch.outer(t_pos, inv_freq)   # rotation frequency at each (time, channel) pair
m.cos = freqs.cos().to(torch.bfloat16)[None, :, None, :]  # add batch and head dims
m.sin = freqs.sin().to(torch.bfloat16)[None, :, None, :]  # for later broadcasting
del channel_range, inv_freq, t_pos, freqs

# ==== Bank Gradient Slice Views ====
# The 3-D banks get `grad32_slices`: per-slice VIEWS built OUTSIDE any compiled
# graph. The forward/backward bodies accumulate through these, never through
# `grad32[i]` -- an in-graph bank slice functionalizes into a whole-bank
# select_scatter copy (10-20x the cost of the slice add at these bank sizes),
# while a view created out of graph arrives as an input and mutates genuinely
# in place.
m.W_Q.grad32_slices = list(m.W_Q.grad32.unbind(0))
m.W_K.grad32_slices = list(m.W_K.grad32.unbind(0))
m.W_V.grad32_slices = list(m.W_V.grad32.unbind(0))
m.W_O.grad32_slices = list(m.W_O.grad32.unbind(0))

m.W_in.grad32_slices  = list(m.W_in.grad32.unbind(0))
m.W_out.grad32_slices = list(m.W_out.grad32.unbind(0))

m.ve_gate.grad32_slices      = list(m.ve_gate.grad32.unbind(0))
m.value_embeds.grad32_slices = list(m.value_embeds.grad32.unbind(0))


# (Grad zeroing happens as a plain loop at the training-loop call site --
# every .grad32 is zeroed after each optimizer_step, since gradients
# accumulate across a step's micro-batches AND Muon's nesterov lerp mutates
# grad32 in place.)


# --------------------------------------------------------------------------------
# § Schedules
# --------------------------------------------------------------------------------
# A run's optimizer is defined up front: every learning rate, beta and weight
# decay for every step is computed here, before training starts, into per-step
# tables of *update coefficients* -- the numbers the fused kernels actually
# multiply by. The optimizer then holds no hyperparameters of its own and the
# training loop has nothing to set per step; the kernels just gather row
# `t_step` of each table. Folding all the way down to coefficients buys:
# - The bias corrections leave the kernel (betas are per-role constants, so
#   the closed `1 - beta^t` form is exact).
# - Nothing about the schedule is left for the loop to do per step. Tables are
#   device-resident and the step counter is a device tensor, so a step involves
#   the host for nothing at all.

class AdamWTabs(NamedTuple):
    """What an AdamW step multiplies by, one (N,) table per field. eps is never
    scheduled, so it rides as a plain kernel argument instead of a table."""
    wd_mul: Tensor           # 1 - lr*wd             decoupled weight decay
    one_minus_beta1: Tensor  # 1 - beta1             exp_avg lerp weight
    one_minus_beta2: Tensor  # 1 - beta2             exp_avg_sq lerp weight
    rsqrt_bias2: Tensor      # 1/sqrt(bias2)         second-moment bias correction
    step_size: Tensor        # lr / bias1            lr schedule x first-moment bias correction


class MuonCoeffs(NamedTuple):
    """What a Muon step multiplies by. Muon's second moment is self-normalizing
    (the v_norm/v_norm_new rescale), so it needs no bias correction."""
    momentum: Tensor            # nesterov momentum
    one_minus_momentum: Tensor  # 1 - momentum        frst_mntm lerp weight
    one_minus_beta2: Tensor     # 1 - beta2           variance-reduction lerp weight
    lr: Tensor                  # lr (the per-bank aspect scale arrives separately, via lr_mul)
    lr_wd: Tensor               # lr * weight_decay   cautious decay


def build_schedules(num_iterations, batch_lr_scale=1.0, weight_decay=0.28,
                    warmup_steps=40, warmdown_ratio=0.65, final_lr_frac=0.05):
    """Named table sets with the tuned nanochat base_train hyperparameters,
    written out flat. Baked assumptions (a Ramp class used to support more):
    exactly three shaped schedules exist -- the shared LR multiplier, Muon
    momentum, and Muon weight decay; every Adam beta is a per-role CONSTANT;
    windows are warmup_steps + round(warmdown_ratio * N). Verified
    bitwise-identical to the Ramp implementation it replaced
    (sched_parity_test.py in the session folder).

    `weight_decay` arrives already batch/horizon-scaled. Returns a namespace:
    .matrix (MuonCoeffs) + one AdamWTabs per AdamW role, .adamw_eps, and
    .num_steps. The trainer binds the result to the global `sched`."""
    N = num_iterations
    C = round(warmdown_ratio * N)               # LR warmdown length
    assert warmup_steps + C <= N, f"warmup ({warmup_steps}) + warmdown ({C}) exceed the run ({N})"
    i = np.arange(N, dtype=np.float64)
    cool = slice(N - C + 1, N)                  # the hold covers i <= N - C
    f = (N - i[cool]) / C                       # ~1 -> ~0 across the warmdown

    # The one LR shape for the whole run: linear warmup from 0 (reaching the
    # peak on the warmup window's last step), hold at 1, linear warmdown to
    # final_lr_frac (arriving one step past the run's end -- nanochat's
    # convention). Each role scales it to its own peak below.
    lrm = np.ones(N)
    lrm[:warmup_steps] = (i[:warmup_steps] + 1.0) / warmup_steps
    lrm[cool] = final_lr_frac + (1.0 - final_lr_frac) * f

    # Muon momentum: 0.85 -> 0.97 over 400 steps (the clamp only lets short
    # smoke/debug runs build a valid schedule; identical for N >= ~1150),
    # hold, then cool to 0.90 across the LR warmdown.
    mW = min(400, int(N * (1 - warmdown_ratio)))
    momentum = np.full(N, 0.97)
    momentum[:mW] = 0.85 + (0.97 - 0.85) * (i[:mW] + 1.0) / mW
    momentum[cool] = 0.90 + (0.97 - 0.90) * f

    # Muon weight decay: half-cosine from the peak to zero over the whole run
    # (step 0 sits at the peak; the decay begins at step 1).
    muon_wd = np.empty(N)
    muon_wd[0] = weight_decay
    fw = (N - i[1:]) / N
    muon_wd[1:] = weight_decay * (0.5 * (1.0 + np.cos(math.pi * (1.0 - fw))))

    # Numpy arrays -> fp32 device tables: a step reads its coefficients with
    # an on-device gather, never a host-to-device copy.
    dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    t1 = np.arange(1, N + 1, dtype=np.float64)

    # Fold one AdamW role's schedule down to the kernel's update coefficients.
    # The folding is ONE policy shared by every role (repeating it 6x would
    # obscure edits); the per-role peaks/betas/wd stay visible at the call
    # sites below.
    def adamw(peak, beta1, beta2, wd):
        lr = lrm * peak
        return AdamWTabs(
            wd_mul          = dev(1.0 - lr * wd),
            one_minus_beta1 = dev(np.full(N, 1.0 - beta1)),
            one_minus_beta2 = dev(np.full(N, 1.0 - beta2)),
            rsqrt_bias2     = dev(1.0 / ((1.0 - beta2 ** t1) ** 0.5)),
            step_size       = dev(lr / (1.0 - beta1 ** t1)),
        )

    # Muon's coefficients fold directly from the three shaped schedules.
    # Canonical lr: NO per-bank aspect fold (see § Optimizer Step).
    matrix_lr = lrm * (0.02 * batch_lr_scale)
    matrix = MuonCoeffs(
        momentum           = dev(momentum),
        one_minus_momentum = dev(1.0 - momentum),
        one_minus_beta2    = dev(np.full(N, 1.0 - 0.9)),   # variance-reduction beta2 = 0.9
        lr                 = dev(matrix_lr),
        lr_wd              = dev(matrix_lr * muon_wd),
    )

    # Per-role peak LRs (tuned values). The AdamW peaks were tuned at d12's
    # width, so they carry the 1/sqrt(width ratio) correction to d24.
    adamw_lr_scale = batch_lr_scale * (cfg.d_model / 768) ** -0.5
    return SimpleNamespace(
        matrix       = matrix,
        lm_head      = adamw(0.008 * adamw_lr_scale,        0.8,  0.96,  0.01),
        input_embeds = adamw(0.3 * adamw_lr_scale,          0.8,  0.995, 0.001),
        value_embeds = adamw(0.3 * adamw_lr_scale * 0.5,    0.8,  0.995, 0.01),
        resid        = adamw(0.5 * batch_lr_scale * 0.01,   0.8,  0.95,  0.05),
        x0           = adamw(0.5 * batch_lr_scale,          0.96, 0.95,  0.0),
        smear        = adamw(0.2,                           0.8,  0.95,  0.0),
        adamw_eps    = 1e-10,
        lrm_table    = lrm,   # host-side copy, for logging only
        num_steps    = N,
        batch_lr_scale = batch_lr_scale,   # echoed into the wandb config
        weight_decay   = weight_decay,
    )


# --------------------------------------------------------------------------------
# § Optimizer Code
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Mantissa Trick
# Masters use the mantissa trick (Larry Dial via modded-nanogpt train_gpt.py):
# the fp32 master's bit pattern is (live_bf16_bits << 16) | mantissa_uint16.
# Update math runs in fp32 on the reconstructed master; the split back is a
# TRUNCATION (load-bearing: round-to-nearest could carry into the top bits and
# break the lossless live/mantissa pairing).
#
# The bit arithmetic runs in int32 (CUDA has no uint32 shifts as of torch 2.9);
# int32's truncating .to(int16) and the <<16 discard of sign-extension bits
# make it equivalent. Mantissa tensors are STORED uint16, viewed int16 for the
# math.

def fp32_master(live: Tensor, mantissa: Tensor) -> Tensor:
    """Reconstruct the fp32 master from bf16 live bits + stashed mantissa."""
    bits = (live.view(torch.int16).to(torch.int32) << 16) | \
           (mantissa.view(torch.int16).to(torch.int32) & 0xFFFF)
    return bits.view(torch.float32)


def writeback_master(master: Tensor, live: Tensor, mantissa: Tensor) -> None:
    """Truncation split of the updated master back into live + mantissa."""
    bits = master.view(torch.int32)
    live.view(torch.int16).copy_((bits >> 16).to(torch.int16))
    mantissa.view(torch.int16).copy_(bits.to(torch.int16))


# -----------------------------------------------------------------------------
# Fused update kernels. The schedule row is gathered ON DEVICE by `t` -- no
# host involvement per step.

# We use the first five, remainder are just for completeness.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_fp32(
    p: Tensor,           # fp32 param, updated IN PLACE (live == master)
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    c: AdamWTabs,
    t: Tensor,           # (1,) int64 device tensor - the schedule row to read
    eps: float,
) -> None:
    """AdamW for the fp32-LIVE scalar params (resid/x0 lambdas, smear, backout
    -- ~30 floats). They are exempt from the bf16-live/mantissa scheme: see the
    dtype scheme note in § Model Initialization."""
    grad = grad.to(exp_avg.dtype)
    p.mul_(c.wd_mul[t])
    exp_avg.lerp_(grad, c.one_minus_beta1[t])
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[t])
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
    p.sub_(c.step_size[t] * (exp_avg / denom))


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    live: Tensor,        # bf16 live shard
    mantissa: Tensor,    # uint16, same shape
    grad: Tensor,        # gradient shard (fp32, or bf16 for the embeddings)
    exp_avg: Tensor,     # fp32 first moment
    exp_avg_sq: Tensor,  # fp32 second moment
    c: AdamWTabs,        # per-step coefficient tables, device-resident
    t: Tensor,           # (1,) int64 device tensor - the schedule row to read
    eps: float,
) -> None:
    """Fused AdamW step on the reconstructed master."""
    p = fp32_master(live, mantissa)
    grad = grad.to(exp_avg.dtype)  # embeddings hand in bf16 grads; moment math stays fp32
    p.mul_(c.wd_mul[t])
    exp_avg.lerp_(grad, c.one_minus_beta1[t])
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[t])
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
    p.sub_(c.step_size[t] * (exp_avg / denom))
    writeback_master(p, live, mantissa)

# The update kernels take explicit per-tensor arguments rather than the model
# object, twice over: (1) at world>1 the SAME kernels run on shard views
# (p[layer_shard_slice] with the shard-size state) rather than on m.X -- an
# object-reading kernel would need a different body per world size; (2) under
# fullgraph compile,
# attribute access on an ad-hoc Python object turns into dynamo guards on
# object identity/attributes -- fragile and recompile-prone next to plain
# tensor arguments.
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    grad: Tensor,        # (K, out, in) fp32 gradient shard -- MUTATED (nesterov lerp)
    live: Tensor,        # (K, out, in) bf16 live shard
    mantissa: Tensor,    # (K, out, in) uint16
    frst_mntm: Tensor,   # (K, out, in) fp32
    scnd_mntm: Tensor,   # (K, out, 1) or (K, 1, in) fp32 - factored second moment
    c: MuonCoeffs,       # per-step coefficient tables, device-resident (UNfolded lr)
    t: Tensor,           # (1,) int64 device tensor - the schedule row to read
    ns_steps: int,       # 5 - number of Polar Express iterations
    residual_dim: int,   # -1 or -2 - residual-facing axis; per-neuron mean-square is taken along it
    lr_mul: Tensor,      # (K, 1, 1) fp32 per-slice LR multiplier (aspect scale today)
    wd_mul: Tensor,      # (K, 1, 1) fp32 per-slice WD multiplier
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction ->
    cautious update on the reconstructed master. The sqrt(fan_out/fan_in)
    aspect scale is NOT in `c` -- it arrives through lr_mul/wd_mul, per slice,
    so the one coefficient table stays valid for every bank."""
    dtype = grad.dtype

    # Nesterov momentum
    frst_mntm.lerp_(grad, c.one_minus_momentum[t].to(dtype))
    g = grad.lerp_(frst_mntm, c.momentum[t].to(dtype))

    # Polar express (orthogonalization)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c_ns in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c_ns * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c_ns in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c_ns * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction (NorMuon). The lerp weight stays fp32.
    v_mean = g.float().square().mean(dim=residual_dim, keepdim=True)
    residual_dim_size = g.size(residual_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * residual_dim_size
    v_norm = v_norm_sq.sqrt()
    scnd_mntm.lerp_(v_mean.to(dtype=scnd_mntm.dtype),
                    c.one_minus_beta2[t].to(scnd_mntm.dtype))
    step_size = scnd_mntm.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * residual_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + master update + truncation split back to live
    p = fp32_master(live, mantissa)
    mask = (g * p) >= 0
    lr = (c.lr[t] * lr_mul).to(g.dtype)
    lr_wd = (c.lr_wd[t] * wd_mul).to(g.dtype)
    p.sub_(lr * g + lr_wd * p * mask)
    writeback_master(p, live, mantissa)


# --------------------------------------------------------------------------------
# § Model Code (Forward/Backward)
# --------------------------------------------------------------------------------
# Handwritten training step: explicit forward + backward (no autograd),
# accumulating into the fp32/bf16 `.grad32` buffers.
#
# Design notes:
# - Attention runs through the raw FA3 ops above, stashing out + LSE.
# - rms_norms: we stash the norm OUTPUT plus the per-vector 1/rms `r`. In
#   output space the backward is dx = r*(dy - y*mean(y*dy)) for ANY eps, so the
#   pre-norm input is never needed. Cheap norms (the MLP-side xm) are
#   recomputed from the stashed pre-norm x1 instead of stashed.
# - Weight-grad matmuls run in bf16, then accumulate upcast into grad32 -- the
#   same numerics autograd produces for a bf16 matmul.
# - loss_scale (1/grad_accum_steps) replaces the loss division of an autograd
#   loop; the returned loss is the plain (unscaled) mean CE for logging.

# Cast shorthands for the bodies below: the fp32 scalars/gates need explicit
# bf16 casts at their use sites (see forward_backward's docstring), and the
# scalar-parameter grad sums accumulate in fp32.
bf16  = lambda x: x.to(torch.bfloat16)
sum32 = lambda x: x.sum(dtype=torch.float32)

# -----------------------------------------------------------------------------
# rms_norm forward/backward in output space
def _rms_fwd(x):
    """rms_norm over the last dim plus the per-vector 1/rms its backward
    needs, sharing one mean-square. r is fp32 with eps = 2^-23 (fp32 machine
    eps -- the same number compiled F.rms_norm's decomposition uses); y is
    x * r cast back to bf16. Verified bitwise-identical to the F.rms_norm
    form under torch.compile, and the same speed (bench_rms.log; eager ATen
    differs in last-ulp on ~6/1M elements, but every call site is compiled)."""
    r = (x.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    y = bf16(x.float() * r)
    return y, r

def _rms_bwd(dy, y, r):
    """dx = r*(dy - y*mean(y*dy)): exact for any eps because r is the forward's
    actual 1/rms and y the actual output (substitute x = y/r in the usual
    form). Math in fp32, result back to bf16."""
    yf, dyf = y.float(), dy.float()
    dx = r * (dyf - yf * (yf * dyf).mean(dim=-1, keepdim=True))
    return bf16(dx)

def _rms_bwd_scaled(dy, ys, r, s):
    """Backward through ys = s * rms_norm(x), given the SCALED output ys --
    which is exactly what the attention kernel consumed, so it stashes directly
    with no recompute pass. Substituting y = ys/s into _rms_bwd's form:
    dx = r*(s*dy - ys*mean(ys*dy)/s). Exact algebra."""
    yf, dyf = ys.float(), dy.float()
    dx = r * (s * dyf - yf * ((yf * dyf).mean(dim=-1, keepdim=True) / s))
    return bf16(dx)


# -----------------------------------------------------------------------------
# forward_backward

@torch.no_grad()
def forward_backward(idx, targets, cu_seqlens, loss_scale=1.0):
    """One micro-batch: forward, stash, explicit backward into `.grad32`.
    Returns the detached mean CE loss (unscaled; grads carry loss_scale).

    Wrap in torch.compile -- the CE block below is written for inductor's
    fusion; run eager it materializes full (T, d_vocab) fp32 temporaries.

    Activations are bf16 throughout. The live weights are already bf16, so no
    per-use casts; the fp32 scalars need care: indexing a 1-D fp32 bank gives a
    0-dim tensor, which does NOT promote a bf16 tensor (resid/x0 lambdas ride
    as-is), but the (1,)-shaped smear/backout scalars and the smear_gate matrix
    WOULD promote to fp32, so those are cast explicitly."""

    assert idx.ndim == 1
    T = idx.size(0)
    nl = cfg.n_layers
    nh, nkv = cfg.n_q_heads, cfg.n_kv_heads
    dqk, dvo = cfg.d_qk, cfg.d_vo
    half = dqk // 2
    gch = cfg.d_ve_gate

    assert T > 1, "Training forward pass should have T > 1"
    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"
    cos, sin = m.cos[0, :T], m.sin[0, :T]  # (T, 1, half)

    # ==== forward half (mirrors forward() -- keep the two visibly line-parallel) ====
    x = F.embedding(idx, m.input_embeds)         # bf16
    xe, r_e = _rms_fwd(x)                     # post-norm embedding, pre-smear

    # Smear: mix the previous token's embedding into the current position.
    gate = bf16(m.smear_lambda) * torch.sigmoid(
        xe[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate).mT)
    x = torch.cat([xe[:1], xe[1:] + gate * xe[:-1]], dim=0)

    x0 = x
    backout_layer = nl // 2
    x_backout = None
    stash = []
    for i in range(nl):
        x_in = x
        # Scale residual stream, add input embedding
        b = m.resid_lambdas[i] * x_in + m.x0_lambdas[i] * x0
        xn, r_xn = _rms_fwd(b)
        # QKV Projections
        q = (xn @ m.W_Q[i].mT).view(T, nh, dqk)
        k = (xn @ m.W_K[i].mT).view(T, nkv, dqk)
        v = (xn @ m.W_V[i].mT).view(T, nkv, dvo)
        
        # Value Embeddings
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, m.value_embeds[j]).view(T, nkv, dvo)
            g = 3 * torch.sigmoid(xn[..., :gch] @ m.ve_gate[j].mT)
            v = v + g.unsqueeze(-1) * ve         # ve/g recomputed in backward, not stashed
        
        # RoPE
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        # QK-Norm 
        qn, r_q = _rms_fwd(q)
        kn, r_k = _rms_fwd(k)

        # Attention Temperature
        qf = qn * 1.2                            # stash the SCALED q/k (the kernel's inputs);
        kf = kn * 1.2                            # backward folds the 1.2 via _rms_bwd_scaled
        # Read V from past residual streams by matching their K.

        y, lse = flash_attn_varlen_fwd_lse(qf, kf, v, cu_seqlens, cfg.seq_len, cfg.window_sizes[i])
        y = y.contiguous()
        # Write V back to residual stream via W_O.
        x1 = b + y.view(T, -1) @ m.W_O[i].mT
                
        # MLP input norm

        xm, _ = _rms_fwd(x1)                  # xm recomputed in backward from stashed x1
        a = F.relu(xm @ m.W_in[i].mT)
        x = x1 + a.square() @ m.W_out[i].mT
        if i == backout_layer:
            x_backout = x
        
        # Build stash
        stash.append(dict(x_in=x_in, xn=xn, r_xn=r_xn, qf=qf, kf=kf, r_q=r_q, r_k=r_k,
                          v=v, y=y, lse=lse, x1=x1, a=a))

    x_pre = x - bf16(m.backout_lambda) * x_backout
    xf, r_f = _rms_fwd(x_pre)

    # lm_head + softcap + CE loss + dlogits, written for inductor's fusion:
    # tcap is an explicit CSE target (materialize once, no tanh recompute in
    # the dz pass), and the onehot is a broadcast compare (a scatter_add here
    # forces an extra full pass over the buffer). Vocab is unpadded by
    # construction, so there is no [:V] cropping anywhere. No pad/ignore
    # machinery either: every target is a real token by construction (the
    # loader packs whole documents; at a doc seam the target is the next
    # doc's BOS), so the mean runs over all T positions and the dz scale is
    # the compile-time constant loss_scale/T rather than a device n_valid.
    softcap = 15.0
    logits = xf @ m.lm_head.mT                   # (T, d_vocab) bf16
    tcap = torch.tanh(logits.float() / softcap)
    cap = softcap * tcap
    tgt = targets.unsqueeze(1)
    cap_y = cap.gather(1, tgt).squeeze(1)
    cmax = cap.amax(dim=1, keepdim=True)
    e = (cap - cmax).exp()
    ssum = e.sum(dim=1, keepdim=True)
    lse_ce = (ssum.log() + cmax).squeeze(1)
    loss = (lse_ce - cap_y).mean()
    onehot = torch.arange(cfg.d_vocab, device=targets.device).unsqueeze(0) == tgt
    dz = bf16((e / ssum - onehot.float()) * (1.0 - tcap * tcap) * (loss_scale / T))
    del logits
    m.lm_head.grad32.add_((dz.mT @ xf).float())
    dxf = dz @ m.lm_head
    del dz

    # ==== backward half ====
    # Bank wgrads add directly into grad32_slices views; only the per-layer
    # scalar sums are collected and landed stacked at the end.
    g_resid = []; g_x0 = []

    d_pre = _rms_bwd(dxf, xf, r_f)
    m.backout_lambda.grad32.add_(-sum32(d_pre * x_backout))
    d_stream = d_pre                             # grad wrt layer nl-1's output
    d_x0 = torch.zeros_like(x0)
    for i in reversed(range(nl)):
        st = stash[i]
        if i == backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes nl//2
            d_stream = d_stream - bf16(m.backout_lambda) * d_pre
        # --- MLP backward (relu^2: dh = 2*a*du, self-masking since a = relu(h)) ---
        x1, a = st["x1"], st["a"]
        d_u = d_stream @ m.W_out[i]
        m.W_out.grad32_slices[i].add_(d_stream.mT @ a.square())
        d_h = 2.0 * a * d_u
        xm, r_xm = _rms_fwd(x1)               # cheap recompute (bitwise: same input)
        m.W_in.grad32_slices[i].add_(d_h.mT @ xm)
        d_xm = d_h @ m.W_in[i]
        d_x1 = d_stream + _rms_bwd(d_xm, xm, r_xm)
        # --- attention backward ---
        xn, y = st["xn"], st["y"]
        m.W_O.grad32_slices[i].add_(d_x1.mT @ y.view(T, -1))
        d_y = (d_x1 @ m.W_O[i]).view(T, nh, dvo)
        dqf, dkf, dv = flash_attn_varlen_bwd(
            d_y, st["qf"], st["kf"], st["v"], y, st["lse"], cu_seqlens, cfg.seq_len,
            cfg.window_sizes[i])
        # per-(token, head) norm backward with the 1.2 scale folded in
        d_qr = _rms_bwd_scaled(dqf, st["qf"], st["r_q"], 1.2)
        d_kr = _rms_bwd_scaled(dkf, st["kf"], st["r_k"], 1.2)
        # rotary backward = rotation by -theta (transpose of the forward rotation)
        dq1, dq2 = d_qr[..., :half], d_qr[..., half:]
        d_q0 = torch.cat([dq1 * cos - dq2 * sin, dq1 * sin + dq2 * cos], dim=-1)
        dk1, dk2 = d_kr[..., :half], d_kr[..., half:]
        d_k0 = torch.cat([dk1 * cos - dk2 * sin, dk1 * sin + dk2 * cos], dim=-1)
        # --- VE gate backward (ve/g recomputed) ---
        j = cfg.ve_index[i]
        d_xn_ve = None
        if j >= 0:
            ve = F.embedding(idx, m.value_embeds[j]).view(T, nkv, dvo)
            sg = torch.sigmoid(xn[..., :gch] @ m.ve_gate[j].mT)
            d_g = (dv * ve).sum(dim=-1)          # (T, n_kv_heads)
            d_zg = d_g * (3 * sg * (1 - sg))
            m.ve_gate.grad32_slices[j].add_(d_zg.mT @ xn[..., :gch])
            d_ve = (dv * (3 * sg).unsqueeze(-1)).reshape(T, nkv * dvo)
            # embedding_dense_backward (autograd's own lowering) beats raw
            # index_add_ atomics ~2x at these shapes -- see the GH200 trace hunt
            m.value_embeds.grad32_slices[j].add_(
                torch.ops.aten.embedding_dense_backward(d_ve, idx, cfg.d_vocab, -1, False))
            d_xn_ve = d_zg @ m.ve_gate[j]
        # dv passes through the VE add unchanged: v = v0 + g*ve
        d_q0 = d_q0.view(T, nh * dqk)
        d_k0 = d_k0.view(T, nkv * dqk)
        d_v0 = dv.reshape(T, nkv * dvo)
        m.W_Q.grad32_slices[i].add_(d_q0.mT @ xn)
        m.W_K.grad32_slices[i].add_(d_k0.mT @ xn)
        m.W_V.grad32_slices[i].add_(d_v0.mT @ xn)
        d_xn = d_q0 @ m.W_Q[i] + d_k0 @ m.W_K[i] + d_v0 @ m.W_V[i]
        if d_xn_ve is not None:
            d_xn[:, :gch] += d_xn_ve
        d_b = d_x1 + _rms_bwd(d_xn, xn, st["r_xn"])
        # --- blend backward: b = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 ---
        g_resid.append(sum32(d_b * st["x_in"]))
        g_x0.append(sum32(d_b * x0))
        d_x0 = d_x0 + m.x0_lambdas[i] * d_b  # TRAP: x0 feeds every layer, accumulate
        d_stream = m.resid_lambdas[i] * d_b
        stash[i] = None                          # free this layer's stash as we go

    # Land the per-layer resid/x0 scalar sums (collected in REVERSED layer
    # order) as one stacked add each.
    m.resid_lambdas.grad32.add_(torch.stack(g_resid[::-1]))
    m.x0_lambdas.grad32.add_(torch.stack(g_x0[::-1]))

    # d_stream is now the grad through layer 0's input, which IS x0 (same tensor)
    d_xs = d_x0 + d_stream                       # grad wrt the smeared embedding
    # --- smear backward: xs = cat([xe[:1], xe[1:] + gate*xe[:-1]]) ---
    sg = torch.sigmoid(xe[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate).mT)  # (T-1, 1), recomputed
    gate = bf16(m.smear_lambda) * sg
    d_xe = d_xs.clone()
    d_xe[:-1] += gate * d_xs[1:]                 # TRAP: shifted scatter -- p's grad reaches p-1
    d_gate = (d_xs[1:] * xe[:-1]).sum(dim=-1, keepdim=True)        # (T-1, 1)
    m.smear_lambda.grad32.add_(sum32(d_gate * sg))
    d_zs = d_gate * bf16(m.smear_lambda) * sg * (1 - sg)
    m.smear_gate.grad32.add_((d_zs.mT @ xe[1:, :cfg.d_smr_gate]).float())
    d_xe[1:, :cfg.d_smr_gate] += d_zs @ bf16(m.smear_gate)
    # --- embedding norm + token embedding scatter ---
    d_emb = _rms_bwd(d_xe, xe, r_e)
    m.input_embeds.grad32.add_(
        torch.ops.aten.embedding_dense_backward(d_emb, idx, cfg.d_vocab, -1, False))

    return loss


# --------------------------------------------------------------------------------
# § Forward-Only
# --------------------------------------------------------------------------------

# Compiled by the trainer: § Main Loop rebinds this name through torch.compile
# (one specialization per shape/targets combination -- val loss and CORE logits).
@torch.no_grad()
def forward(idx, cu_seqlens, targets=None, loss_reduction='mean'):
    """Scoring forward for validation loss and CORE eval: one packed 1D
    sequence of documents with per-document attention isolation via varlen
    flash attention. idx/targets are (T,) and activations stay (T, ...)
    throughout -- the layout the varlen kernel wants. Returns the loss if
    targets are given, else the (softcapped, fp32) logits (T, d_vocab).

    Mirrors forward_backward's forward half line for line -- keep them that
    way; diff them when either changes."""
    assert idx.ndim == 1
    T = idx.size(0)
    D = cfg.d_model
    half = cfg.d_qk // 2

    assert T > 1, "Scoring forward pass should have T > 1 (smear needs a previous token)"
    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"
    cos, sin = m.cos[0, :T], m.sin[0, :T]  # (T, 1, half)

    # Embed the tokens
    x = F.embedding(idx, m.input_embeds)         # bf16
    x = F.rms_norm(x, (D,))

    # Smear: mix the previous token's embedding into the current position.
    gate = bf16(m.smear_lambda) * torch.sigmoid(
        x[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate).mT)
    x = torch.cat([x[:1], x[1:] + gate * x[:-1]], dim=0)

    # Forward the trunk of the Transformer
    x0 = x
    backout_layer = cfg.n_layers // 2
    x_backout = None
    for i in range(cfg.n_layers):
        x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0
        # --- attention ---
        xn = F.rms_norm(x, (D,))
        # (T, H, D) - the varlen kernel's native layout, no transpose needed
        q = (xn @ m.W_Q[i].mT).view(T, cfg.n_q_heads,  cfg.d_qk)
        k = (xn @ m.W_K[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (xn @ m.W_V[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)
        # Value residual (ResFormer): value embedding mixed in via an
        # input-dependent per-head gate, range (0, 3)
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, m.value_embeds[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            g = 3 * torch.sigmoid(xn[..., :cfg.d_ve_gate] @ m.ve_gate[j].mT)
            v = v + g.unsqueeze(-1) * ve
        # Rotary embeddings (relative positional encoding)
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        # QK norm, then sharper attention (the 1.2 splits the scale between Q and K)
        q = F.rms_norm(q, (cfg.d_qk,)) * 1.2
        k = F.rms_norm(k, (cfg.d_qk,)) * 1.2
        y, _ = flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, cfg.seq_len, cfg.window_sizes[i])
        x = x + y.contiguous().view(T, -1) @ m.W_O[i].mT
        # --- MLP (relu^2) ---
        x = x + F.relu(F.rms_norm(x, (D,)) @ m.W_in[i].mT).square() @ m.W_out[i].mT
        if i == backout_layer:
            x_backout = x
    # Subtract mid-layer residual to remove low-level features before logit projection
    x = x - bf16(m.backout_lambda) * x_backout
    x = F.rms_norm(x, (D,))

    # lm_head + softcap
    logits = (x @ m.lm_head.mT).float()          # (T, d_vocab)
    logits = 15.0 * torch.tanh(logits / 15.0)    # smoothly cap to [-15, 15]

    if targets is not None:
        # No ignore_index: targets here only ever come from the training/val
        # loader, which never emits pad (see forward_backward's CE note).
        return F.cross_entropy(logits, targets, reduction=loss_reduction)
    return logits


# --------------------------------------------------------------------------------
# § Optimizer Step
# --------------------------------------------------------------------------------
# The written-out step: one fused-kernel call per named tensor, policy at the
# call site, wrapped in the 3-phase comm flow (nanochat train_step.py):
#
#   1. Launch an async grad reduction for every sharded tensor: the full
#      grad32 reduce-scatters into a fresh shard-size buffer, in the grad's
#      dtype (bf16 for the two embedding tables, fp32 for everything else).
#      ReduceOp.AVG across ranks composes with loss_scale=1/grad_accum_steps
#      to make every reduced grad the global-batch mean.
#   2. In launch order (the comm stream completes reduces in that order):
#      wait for the tensor's reduced grad, run its update kernel on the owned
#      shard, then launch the async all-gather that writes the updated bf16
#      live shard back into every rank's full tensor. The gather is IN PLACE
#      -- our slice of the live tensor is the gather source, NCCL's
#      sanctioned in-place form; even divisibility (§ Shard Assignment) means
#      no padded staging buffer and no crop afterwards. Each gather overlaps
#      the updates that follow it. Replicated params (ve_gate, the fp32
#      scalars) ride along inline: plain all_reduce, then the identical
#      full-size update on every rank.
#   3. Wait out the gathers.
#
# Waits are stream waits, not host syncs -- the whole step stays async on the
# host, and t_step still advances on-device. At world_size == 1 every
# collective short-circuits and every shard view is the whole tensor: one
# code path, degenerate comm, numerics identical to the validated single-GPU
# step.
#
# NOTE: the world>1 path has not run yet (the reference's comm code never ran
# at world>1 either) -- it awaits an 8-GPU validation pass.

ns_steps = 5  # Polar Express iterations per Muon step

# Per-slice Muon LR/WD multipliers: each bank's sqrt(max(1, fan_out/fan_in))
# aspect scale -- Muon's tall-matrix correction -- kept OUT of the shared
# matrix table so that table stays one set of numbers valid for every bank.
# At d24 only W_in is non-square, so only it gets a real multiplier (2.0).
mul_unit    = torch.full((cfg.n_layers, 1, 1), 1.0, dtype=torch.float32, device=device)  # W_Q/W_K/W_V/W_O (square), W_out (wide -> clamped)
mul_W_in    = torch.full((cfg.n_layers, 1, 1), (cfg.d_mlp / cfg.d_model) ** 0.5,
                         dtype=torch.float32, device=device)                             # 2.0 (4x expansion, tall)
mul_ve_unit = torch.full((cfg.num_ves,  1, 1), 1.0, dtype=torch.float32, device=device)  # ve_gate (square)

# THE schedule position: one (1,) int64 device tensor, advanced on-device at
# the end of optimizer_step -- the host never syncs on it.
t_step = torch.zeros(1, dtype=torch.int64, device=device)


@torch.no_grad()
def optimizer_step():
    """One explicit optimizer step, written out per named tensor. Reads the
    global `sched` (bind build_schedules' result to `sched` before training).
    Muon MUTATES the grad it is handed (nesterov lerp) -- grad32 itself at
    world=1, the reduce-scattered shard at world>1 -- so zero every grad32
    afterwards either way (the loop in § Main Loop does).
    no_grad is load-bearing for the fp32 scalar kernel's in-place leaf updates
    (the mantissa kernels only dodge autograd's leaf check via their int
    views)."""
    eps = sched.adamw_eps

    # ---- Phase 1: launch every async grad reduction --------------------------
    # Fresh shard buffers each step (the caching allocator makes this free);
    # the state tensors already carry the shard geometry, so empty_like is the
    # whole allocation story.
    reduced = {}   # tensor -> (async work handle, shard-size reduced grad)
    if world_size > 1:
        for p in (m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out):
            g_shard = torch.empty_like(p.frst_mntm)                        # (layer shard, out, in) fp32
            reduced[p] = (dist.reduce_scatter_tensor(g_shard, p.grad32, op=dist.ReduceOp.AVG, async_op=True), g_shard)
        for p in (m.lm_head, m.input_embeds, m.value_embeds):
            g_shard = torch.empty_like(p.exp_avg, dtype=p.grad32.dtype)    # (row shard, cols) in the grad's dtype
            reduced[p] = (dist.reduce_scatter_tensor(g_shard, p.grad32.view(-1, p.shape[-1]), op=dist.ReduceOp.AVG, async_op=True), g_shard)

    # ---- Phase 2: wait -> owned-shard update -> gather the live shard --------
    gathers = []

    # Muon banks, sharded over layers
    for p, mul in ((m.W_Q, mul_unit), (m.W_K, mul_unit), (m.W_V, mul_unit),
                   (m.W_O, mul_unit), (m.W_in, mul_W_in), (m.W_out, mul_unit)):
        if world_size > 1:
            work, grad = reduced[p]
            work.wait()
        else:
            grad = p.grad32
        muon_step_fused(grad, p[layer_shard_slice], p.mantissa, p.frst_mntm, p.scnd_mntm,
                        sched.matrix, t_step, ns_steps, p.residual_dim,
                        mul[layer_shard_slice], mul[layer_shard_slice])
        if world_size > 1:
            gathers.append(dist.all_gather_into_tensor(p, p[layer_shard_slice], async_op=True))

    # Muon replicated: ve_gate is tiny, every rank updates all of it
    if world_size > 1:
        dist.all_reduce(m.ve_gate.grad32, op=dist.ReduceOp.AVG)
    muon_step_fused(m.ve_gate.grad32, m.ve_gate, m.ve_gate.mantissa, m.ve_gate.frst_mntm, m.ve_gate.scnd_mntm, sched.matrix, t_step, ns_steps, m.ve_gate.residual_dim, mul_ve_unit, mul_ve_unit)

    # AdamW, sharded over vocab rows. value_embeds' state is shaped over the
    # flattened (ve_slot * vocab) row axis, so live/grad pass 2-D views
    # throughout (a no-op reshape for the two already-2-D tables).
    # The roles differ only in their tables (peaks/betas: build_schedules):
    # lm_head runs the coolest peak (~40x below the embeddings); input_embeds
    # the hottest, with the heaviest second-moment smoothing (beta2 .995);
    # value_embeds rides the embedding schedule at half peak and 10x the decay.
    for p, table, row_shard in ((m.lm_head,      sched.lm_head,      vocab_shard_slice),
                                (m.input_embeds, sched.input_embeds, vocab_shard_slice),
                                (m.value_embeds, sched.value_embeds, ve_row_shard_slice)):
        rows = p.view(-1, p.shape[-1])
        if world_size > 1:
            work, grad = reduced[p]
            work.wait()
        else:
            grad = p.grad32.view(-1, p.shape[-1])
        adamw_step_fused(rows[row_shard], p.mantissa, grad, p.exp_avg, p.exp_avg_sq, table, t_step, eps)
        if world_size > 1:
            gathers.append(dist.all_gather_into_tensor(rows, rows[row_shard], async_op=True))

    # AdamW replicated scalars (fp32-live, no mantissa). Three schedule
    # flavors: resid -- the gentlest peak and the only decayed scalars (wd
    # .05); x0 -- the hottest peak with a slow first moment (beta1 .96);
    # smear -- one flat middling peak shared by all three smear/backout
    # scalars, no decay. (Peaks/betas: build_schedules.)
    if world_size > 1:
        for p in (m.resid_lambdas, m.x0_lambdas, m.smear_gate, m.smear_lambda, m.backout_lambda):
            dist.all_reduce(p.grad32, op=dist.ReduceOp.AVG)
    
    adamw_step_fused_fp32(m.resid_lambdas,  m.resid_lambdas.grad32,  m.resid_lambdas.exp_avg,  m.resid_lambdas.exp_avg_sq,  sched.resid, t_step, eps)
    
    adamw_step_fused_fp32(m.x0_lambdas,     m.x0_lambdas.grad32,     m.x0_lambdas.exp_avg,     m.x0_lambdas.exp_avg_sq,     sched.x0,    t_step, eps)
    
    adamw_step_fused_fp32(m.smear_gate,     m.smear_gate.grad32,     m.smear_gate.exp_avg,     m.smear_gate.exp_avg_sq,     sched.smear, t_step, eps)
    adamw_step_fused_fp32(m.smear_lambda,   m.smear_lambda.grad32,   m.smear_lambda.exp_avg,   m.smear_lambda.exp_avg_sq,   sched.smear, t_step, eps)
    adamw_step_fused_fp32(m.backout_lambda, m.backout_lambda.grad32, m.backout_lambda.exp_avg, m.backout_lambda.exp_avg_sq, sched.smear, t_step, eps)

    # ---- Phase 3: wait out the live all-gathers ------------------------------
    for work in gathers:
        work.wait()

    t_step.add_(1)  # advance the schedule on-device

# Model + optimizer state is CAPTURED to disk at cfg.save_steps and at the
# final step (write_checkpoint, below the seam): live weights, masters via
# mantissa, both optimizers' moments, and the step counter -- world-agnostic.
# There is still deliberately no LOAD path (runs start from scratch, see the
# design decisions at the top); resume arrives with the load half when first
# needed.

##########################################################################################
#  Code below comes from the 'stacks' repo
#  I pulled it mainly for:
#    - Pre-tokenized data, and the distributed data loader
#    - Simplified (maybe?) CORE eval code
# 
##########################################################################################

# --------------------------------------------------------------------------------
# § Dataset Download
# --------------------------------------------------------------------------------

NUM_TRAIN_SHARDS = 20   # full 5,568-step horizon: 70 (downloads shards 1-69,
                        # 6.9B raw ~= 6.1B usable after seq_len truncation --
                        # see the token-floor assert below the seam; 91 shards
                        # of 100M raw tokens are on the hub)
#DATASET_NAME = "fineweb_edu_32k_8_370"
DATASET_NAME = "climbmix_32k_8_170"
# Subdir for PT train/val .bin shards
#PT_DATA_SUBDIR = "fineweb_edu" 
PT_DATA_SUBDIR = "climbmix" 
HF_REPO_ID = f"ChrisMcCormick/{DATASET_NAME}"
_data_path = os.environ.get("DATA_PATH", ".")
DATASET_DIR = os.path.join(_data_path, f"data/{DATASET_NAME}")
_config_path = os.path.join(DATASET_DIR, "config.json")
train_files = os.path.join(DATASET_DIR, f"{PT_DATA_SUBDIR}/train_*.bin")
val_files   = os.path.join(DATASET_DIR, f"{PT_DATA_SUBDIR}/val_*.bin")

if master_process:
    from huggingface_hub import HfApi, hf_hub_download, login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    os.makedirs(DATASET_DIR, exist_ok=True)
    api = HfApi()
    train_prefix = f"{PT_DATA_SUBDIR}/train_"
    to_download = []
    for fname in api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"):
        if fname.startswith(train_prefix) and int(fname[len(train_prefix):].split(".")[0]) >= NUM_TRAIN_SHARDS:
            continue
        if not os.path.exists(os.path.join(DATASET_DIR, fname)):
            to_download.append(fname)
    if to_download:
        print(f"=== Downloading {len(to_download)} files from {HF_REPO_ID} ===")
        for fname in to_download:
            hf_hub_download(repo_id=HF_REPO_ID, filename=fname, repo_type="dataset", local_dir=DATASET_DIR)
        print("  Done.")
dist.barrier()

# Load vocab config
with open(_config_path) as f:
    _vocab_config = json.load(f)
VOCAB_SIZE = _vocab_config["vocab_size"]
BOS_ID = _vocab_config["bos_id"]
assert VOCAB_SIZE == cfg.d_vocab, \
    f"dataset vocab ({VOCAB_SIZE}) != model d_vocab ({cfg.d_vocab}) -- wrong dataset for this hardcoded model"

# --------------------------------------------------------------------------------
# § Distributed Data Loader
# --------------------------------------------------------------------------------
# Based on the dataloader from modded-nanogpt.
# - Designed for use with flashattention_varlen_func, meaning it returns a packed token
#   buffer of sequences and their lengths via cu_seqlens.
# - Hardcoded for single-epoch training.
# - Compared to `modded`, it does not support changing batch size mid-training.

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

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1):
    """
    Generator (i.e., yields rather than returns) of the token ids for a
    micro-batch: num_tokens / grad_accum_steps / world_size tokens per yield
    (32,768 for the d24 spec: total batch 2^20, grad accum 32 at world=1).
    Provides both the input and target ids.
    Sequences are BOS-aligned and only returned from their beginning; tokens
    past max_seq_len are discarded (the next sequence starts at the next BOS).
    Also used for validation batches.
    Args:
        filename_pattern: pattern to match the dataset .bin shard files
        num_tokens:       tokens per full batch (2^20 for training)
        max_seq_len:      2048
        grad_accum_steps: micro-batches per full batch
    """
    # This GPU's rank and total GPU count.
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Confirm it all divides evenly, then calculate the per-GPU micro-batch size.
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"    
    num_tokens_local = num_tokens // grad_accum_steps // world_size

    # cu_seqlens is FIXED SIZE (the compiled graph needs one shape), and ghost
    # entries cost real FA3 varlen overhead, so it is sized to the DATA rather
    # than a rounded guess: the densest run of climbmix docs packs 82 into one
    # 32,768-token micro-batch (measured -- scan_max_docs.py; an upper bound,
    # since batches can only start where the previous one ended). 96 gives
    # ~17% headroom (nanochat's own estimate for these shapes also lands on
    # 96), and the overflow assert below fails loudly rather than corrupt if
    # the data ever changes.
    max_num_docs = 96
    
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
        assert len(cum_lengths) < max_num_docs, \
            f"micro-batch packed {len(cum_lengths)} docs; cu_seqlens holds only {max_num_docs}"

        # The actual cu_seqlens array always needs to contain `max_num_docs` elements so we
        # the compiler can build a single graph.
        # We allocate that buffer here and fill it with "empty documents", i.e., setting their start index
        # to one past the end of the `_inputs` buffer.
        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        
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

# --------------------------------------------------------------------------------
# § CORE Evaluation
# --------------------------------------------------------------------------------

"""
CORE evaluation using pre-tokenized benchmark data.

The CORE metric (from the DCLM paper, https://arxiv.org/abs/2406.11794) evaluates
a base model on in-context learning tasks using logit-based scoring (no generation).

Pre-tokenized .pt files are produced by data/core_dataset.py and loaded at eval time.
Sequences are packed into fixed-size 1D buffers with cu_seqlens marking boundaries,
enabling batched evaluation through the compiled varlen flash attention m.
"""

# -----------------------------------------------------------------------------
# Packed CORE evaluation: batch multiple examples into fixed-length 1D buffers

def pack_for_eval(sequences, buffer_size):
    """
    Pack pre-tokenized sequences into fixed-size 1D buffers for batched evaluation.

    Args:
        sequences: list of (tokens, start_idx, end_idx, example_idx, seq_idx_within_example)
        buffer_size: fixed buffer size (must be multiple of 16)

    Returns:
        list of dicts with keys: input_ids, cu_seqlens, metadata
    """
    assert buffer_size % 16 == 0
    # CORE eval sequences can be short (~50-200 tokens), so allow many more per buffer
    # than training's //300 estimate. Use //8 for generous headroom (memory is negligible).
    max_num_seqs = next_multiple_of_n(buffer_size // 8, n=128)

    buffers = []
    cur_tokens = []
    cur_cu = [0]
    cur_meta = []
    cur_pos = 0

    for tokens, start_idx, end_idx, example_idx, seq_idx in sequences:
        seq_len = len(tokens)
        if seq_len > buffer_size:
            continue  # should not happen after truncation

        if cur_pos + seq_len > buffer_size:
            # Finalize current buffer
            _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                                  buffer_size, max_num_seqs)
            cur_tokens, cur_cu, cur_meta, cur_pos = [], [0], [], 0

        # Track answer span in global buffer coordinates
        global_start = cur_pos + start_idx
        global_end = cur_pos + end_idx
        cur_meta.append((example_idx, seq_idx, global_start, global_end))
        cur_tokens.extend(tokens)
        cur_pos += seq_len
        cur_cu.append(cur_pos)

    if cur_tokens:
        _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                              buffer_size, max_num_seqs)

    return buffers


def _finalize_eval_buffer(buffers, cur_tokens, cur_cu, cur_meta,
                          buffer_size, max_num_seqs):
    """Pad and finalize a packed eval buffer."""
    total_packed = len(cur_tokens)
    pad_count = buffer_size - total_packed

    # Input tokens: packed sequences + BOS padding
    input_ids = torch.full((buffer_size,), BOS_ID, dtype=torch.int32)
    input_ids[:total_packed] = torch.tensor(cur_tokens, dtype=torch.int32)

    # cu_seqlens: [0, end1, end2, ..., total_packed, buffer_size, buffer_size, ...]
    if pad_count > 0:
        cur_cu.append(buffer_size)  # ghost sequence for padding region
    cu_seqlens = torch.full((max_num_seqs,), buffer_size, dtype=torch.int32)
    cu_seqlens[:len(cur_cu)] = torch.tensor(cur_cu, dtype=torch.int32)

    buffers.append({
        'input_ids': input_ids,
        'cu_seqlens': cu_seqlens,
        'metadata': cur_meta,
    })

@torch.no_grad()
def forward_eval_packed(input_ids, cu_seqlens):
    """
    Forward a packed 1D eval buffer through the model's scoring forward.
    Returns (softcapped, fp32) logits of shape (buffer_size, vocab_size).
    """
    return forward(input_ids, cu_seqlens)


@torch.no_grad()
def evaluate_task_packed(task_data, buffer_size=cfg.eval_buffer_tokens):
    """Evaluate one task using pre-tokenized sequences and packed batched evaluation."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    task_type = task_data['task_type']
    num_examples = task_data['num_examples']
    all_sequences = task_data['sequences']
    num_seqs_per_example = task_data['num_seqs_per_example']
    gold_labels = task_data['gold_labels']

    # Step 1: Select this rank's share of pre-tokenized sequences
    rank_examples = set(range(rank, num_examples, world_size))
    sequences = [
        (s['tokens'], s['start_idx'], s['end_idx'], s['example_idx'], s['seq_idx'])
        for s in all_sequences if s['example_idx'] in rank_examples
    ]

    # Step 2: Pack into fixed-size buffers
    packed_buffers = pack_for_eval(sequences, buffer_size)

    # Step 3: Forward pass each buffer and collect per-sequence results
    seq_results = {}

    for buf in packed_buffers:
        input_ids = buf['input_ids'].to(device)
        cu_seqlens = buf['cu_seqlens'].to(device)

        logits = forward_eval_packed(input_ids, cu_seqlens)

        # Per-position losses: loss[j] = -log p(input_ids[j+1] | context up to j)
        target_ids = torch.roll(input_ids.long(), shifts=-1)
        all_losses = F.cross_entropy(logits.float(), target_ids, reduction='none')
        all_predictions = logits.argmax(dim=-1)

        for example_idx, seq_idx, gs, ge in buf['metadata']:
            # Answer span [gs, ge): logits at [gs-1, ge-1) predict tokens at [gs, ge)
            seq_results[(example_idx, seq_idx)] = {
                'losses': all_losses[gs - 1 : ge - 1],
                'predictions': all_predictions[gs - 1 : ge - 1],
                'input_ids': input_ids[gs : ge].long(),
            }

    # Step 4: Evaluate per-example correctness
    correct = torch.zeros(num_examples, dtype=torch.float32, device=device)

    for idx in range(rank, num_examples, world_size):
        if task_type == 'language_modeling':
            r = seq_results[(idx, 0)]
            is_correct = torch.all(r['predictions'] == r['input_ids']).item()
        elif task_type in ['multiple_choice', 'schema']:
            mean_losses = []
            for seq_j in range(num_seqs_per_example[idx]):
                r = seq_results[(idx, seq_j)]
                mean_losses.append(r['losses'].mean().item())
            pred_idx = mean_losses.index(min(mean_losses))
            is_correct = pred_idx == gold_labels[idx]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        correct[idx] = float(is_correct)

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()


@torch.no_grad()
def evaluate_chat_task_packed(task_data, buffer_size=cfg.eval_buffer_tokens):
    """Evaluate one chat categorical task using packed batched evaluation.

    Unlike CORE eval (which compares losses across multiple sequences per example),
    chat eval checks single-token logits at the answer position against letter choices.
    Each sequence ends with the prompt (including <|assistant_start|>), and we check
    what the model predicts as the next token, restricted to the valid answer letters.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    all_sequences = task_data['sequences']
    num_examples = task_data['num_examples']

    # Step 1: Select this rank's share and convert to pack_for_eval format.
    # We store answer_pos as start_idx (end_idx = start_idx + 1 for tuple compat)
    # and keep letter_token_ids / gold in a side table.
    sequences = []
    example_meta = {}  # example_idx -> (letter_token_ids, gold)
    for s in all_sequences:
        idx = s['example_idx']
        if idx % world_size != rank:
            continue
        answer_pos = s['answer_pos']
        sequences.append((s['tokens'], answer_pos, answer_pos + 1, idx, 0))
        example_meta[idx] = (s['letter_token_ids'], s['gold'])

    # Step 2: Pack into fixed-size buffers (reuse CORE eval packing infrastructure)
    packed_buffers = pack_for_eval(sequences, buffer_size)

    # Step 3: Forward pass each buffer and score
    correct = 0
    total = 0

    for buf in packed_buffers:
        input_ids = buf['input_ids'].to(device)
        cu_seqlens = buf['cu_seqlens'].to(device)

        logits = forward_eval_packed(input_ids, cu_seqlens)

        for example_idx, seq_idx, gs, ge in buf['metadata']:
            # gs = global position of answer_pos in the buffer.
            # logits[gs] predicts the token AFTER position gs — i.e. the assistant's answer.
            # (This differs from CORE's logits[gs-1:ge-1] convention because here the
            # answer token is NOT in the sequence — we want what the model predicts next.)
            answer_logits = logits[gs]  # (vocab_size,)
            letter_ids, gold = example_meta[example_idx]
            focus_logits = answer_logits[letter_ids]  # (num_choices,)
            pred = focus_logits.argmax().item()
            correct += int(pred == gold)
            total += 1

    # Step 4: Aggregate across ranks
    if world_size > 1:
        correct_t = torch.tensor([correct], dtype=torch.long, device=device)
        total_t = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        correct = correct_t.item()
        total = total_t.item()

    return correct / total if total > 0 else 0.0


def evaluate_chat_categorical():
    """
    Evaluate a chat model on categorical benchmarks (MMLU, ARC-Easy, ARC-Challenge)
    using pre-tokenized data from chat_eval_dataset.py.
    Returns dict with results, centered_results, and chatcore_metric.
    """
    chat_eval_dir = os.path.join(DATASET_DIR, "chat_eval")
    config_path = os.path.join(chat_eval_dir, "config.json")

    assert os.path.exists(config_path), f"Chat eval config not found: {config_path}"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task_info in config['tasks']:
        torch.cuda.synchronize()
        start_time = time.time()
        label = task_info['label']

        pt_path = os.path.join(chat_eval_dir, task_info['file'])
        assert os.path.exists(pt_path), f"Chat eval data not found: {pt_path}"
        task_data = torch.load(pt_path, weights_only=False)
        print0(f"Chat eval: {label} ({task_data['num_examples']} examples)... ", console=True)

        accuracy = evaluate_chat_task_packed(task_data)
        torch.cuda.synchronize()
        results[label] = accuracy
        random_baseline = task_data['random_baseline']
        centered_result = (accuracy - random_baseline) / (1.0 - random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s", console=True)

    chatcore_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "chatcore_metric": chatcore_metric,
    }
    return out


def evaluate_core():
    """
    Evaluate a base model on the CORE benchmark using pre-tokenized data.
    Returns dict with results, centered_results, and core_metric.
    """
    core_eval_dir = os.path.join(DATASET_DIR, "core_eval")
    config_path = os.path.join(core_eval_dir, "config.json")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task_info in config['tasks']:
        torch.cuda.synchronize()
        start_time = time.time()
        label = task_info['label']

        task_data = torch.load(os.path.join(core_eval_dir, task_info['file']),
                               weights_only=False)
        print0(f"Evaluating: {label} ({task_data['task_type']}, "
               f"{task_data['num_examples']} examples)... ", console=True)

        accuracy = evaluate_task_packed(task_data)
        torch.cuda.synchronize()
        results[label] = accuracy
        random_baseline = task_data['random_baseline']
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s", console=True)

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# --------------------------------------------------------------------------------
# § Main Loop
# --------------------------------------------------------------------------------
# Modeled on nanochat base_train (branch fwd-bwd) -- the flat trainer over the
# same forward_backward / optimizer_step API. No warmup-and-reset phase (that
# trick needs the state_dict save/restore this file deliberately lacks):
# compilation happens during the first real steps, and the time totals simply
# exclude the first 10 steps (the nanochat convention).

# begin logging
logfile = None
if master_process:
    run_id = cfg.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s="", console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

# -----------------------------------------------------------------------------
# Model stats, for MFU and the wandb config -- CONSTANTS; scaling.py recomputes
# them from the d24 shapes (params by group, 6 FLOPs per matmul-weight param
# plus the windowed attention term).

num_params          = 1_384_122_122   # every trained weight (Model.weight_names)
num_flops_per_token = 4_860_160_128   # 6 * 729,810,624 matmul params + attention

gpu_device_name = torch.cuda.get_device_name(0)
gpu_peak_flops = next((v for k, v in PEAK_FLOPS.items() if k in gpu_device_name.upper()),
                      float("inf"))
print0(f"Model parameters: {num_params:,} | FLOPs/token: {num_flops_per_token:e}", console=True)
print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}", console=True)
print0(f"Total batch size: {cfg.total_batch_size:,} tokens = {cfg.micro_batch_tokens:,} tokens/micro "
       f"x {world_size} ranks x {grad_accum_steps} grad accum", console=True)

# -----------------------------------------------------------------------------
# Schedules: every LR/beta/WD coefficient for the whole run, materialized up
# front. The two batch/horizon corrections are hardcoded (derivations in
# scaling.py):
#   batch_lr_scale = sqrt(2^20 / 2^19) = 1.4142...  -- eta ∝ sqrt(B/B_ref),
#     B_ref = 2^19 where the d12 LRs were tuned; build_schedules applies it to
#     the per-role peaks itself (do NOT also fold it into the LRs).
#   weight_decay = 0.28 * sqrt(2) * (d12/d24 scaling params) = 0.059738 -- the
#     T_epoch framework; matches nanochat's d24 printout exactly.
sched = build_schedules(cfg.num_iterations, batch_lr_scale=1.4142135623730951,
                        weight_decay=0.059738)

# -----------------------------------------------------------------------------
# Compile the training step. REQUIRED, not an optimization: the CE block in
# forward_backward is written for inductor's fusion -- run eager it
# materializes full (T, d_vocab) fp32 temporaries. fullgraph so any graph
# break errors loudly instead of silently fragmenting fusion (the FA3 raw ops
# have fake impls, so a full trace is achievable).
fb = torch.compile(forward_backward, dynamic=False, fullgraph=True)

# The eval forward is compiled too -- eager it materializes the full
# (T, d_vocab) fp32 logits chain (~13 GB of temporaries per val micro-batch).
# Rebinding the name routes every consumer (the val-loss section and
# forward_eval_packed) through it; it specializes once per shape/targets
# combination: the val path at the training micro-batch shape, the CORE
# logits path at the eval buffer shape.
forward = torch.compile(forward, dynamic=False, fullgraph=True)

# token_bytes: per-token-id byte lengths (0 for special tokens), for the
# vocab-size-independent bits-per-byte validation metric.
with open(os.path.join(DATASET_DIR, "tokenizer/token_bytes.pt"), "rb") as f:
    token_bytes = torch.load(f, map_location=device)

# Enough data for the horizon? The loader is single-epoch and TRUNCATES long
# documents at seq_len, discarding the tails: measured ~11-12% of climbmix's
# raw tokens (doc-length scan, 2026-07-31 session NOTES). 0.85 is that
# discard with margin -- a raw-token floor alone would pass configs that run
# dry ~11% before the horizon.
_shard_tokens = sum((os.path.getsize(f) - 256 * 4) // 2 for f in glob.glob(train_files))
assert _shard_tokens * 0.85 >= (cfg.num_iterations + 1) * cfg.total_batch_size, \
    f"train shards hold {_shard_tokens:,} raw tokens (~{int(_shard_tokens * 0.85):,} usable " \
    f"after seq_len truncation) < {(cfg.num_iterations + 1) * cfg.total_batch_size:,} needed " \
    f"-- raise NUM_TRAIN_SHARDS"

# --- wandb logging init ---
use_dummy_wandb = cfg.wandb_run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project=cfg.wandb_project, name=cfg.wandb_run,
    config={
        "num_params": num_params,
        "num_flops_per_token": num_flops_per_token,
        "n_layers": cfg.n_layers, "n_q_heads": cfg.n_q_heads, "d_model": cfg.d_model,
        "train_steps": cfg.num_iterations,
        "total_batch_size": cfg.total_batch_size,
        "micro_batch_tokens": cfg.micro_batch_tokens,
        "val_loss_every": cfg.val_loss_every,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "batch_lr_scale": sched.batch_lr_scale,
        "weight_decay": sched.weight_decay,
    },
)
if not use_dummy_wandb:
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

# -----------------------------------------------------------------------------
# Checkpoint capture (write only -- there is deliberately no load/resume path
# yet). Two files per capture point in logs/{run_id}/:
#   model_stepNNNNNN.pt -- {step, code, weights: {name: tensor}} -- the bf16
#     live weights + fp32 scalars, the payload the final save has always held.
#   optim_stepNNNNNN.pt -- {step, t_step, state: {"name.attr": tensor}} over
#     the five optimizer-state attrs; together with the live weights this is
#     the full fp32 masters and both optimizers' moments.
# World-agnostic: sharded state all-gathers to full size before writing, so a
# capture from an 8-GPU run loads at any world size (at world=1 the gathers
# short-circuit and this is a plain copy-out). Every rank participates in the
# gathers; only master materializes CPU copies and writes -- tensors are saved
# on CPU so the files open anywhere.

state_attrs = ("mantissa", "frst_mntm", "scnd_mntm", "exp_avg", "exp_avg_sq")

# The sharded weights -- their state gathers over dim 0; everything else is
# replicated, already full-size on every rank. Mirrors § Shard Assignment.
# A set, not a tuple: tuple membership falls through identity to elementwise
# tensor ==, while set membership stays on the identity hash.
sharded_weights = {m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out,
                   m.lm_head, m.input_embeds, m.value_embeds}

def gather_full(t):
    """All-gather a shard-size state tensor to full size over dim 0. uint16
    (mantissa) rides as a bf16 bitcast: NCCL has no 16-bit int type, and a
    gather only moves bytes."""
    if world_size == 1:
        return t
    comm = t.view(torch.bfloat16) if t.dtype == torch.uint16 else t
    full = torch.empty(t.shape[0] * world_size, *t.shape[1:], dtype=comm.dtype, device=device)
    dist.all_gather_into_tensor(full, comm)
    return full.view(torch.uint16) if t.dtype == torch.uint16 else full

def write_checkpoint(step):
    state = {}
    for n in m.weight_names:
        p = getattr(m, n)
        for attr in state_attrs:
            if hasattr(p, attr):
                full = gather_full(getattr(p, attr)) if p in sharded_weights else getattr(p, attr)
                if master_process:
                    state[f"{n}.{attr}"] = full.cpu()
    if not master_process:
        return
    os.makedirs(f"logs/{run_id}", exist_ok=True)
    torch.save(dict(step=step, code=code,
                    weights={n: getattr(m, n).cpu() for n in m.weight_names}),
               f"logs/{run_id}/model_step{step:06d}.pt")
    torch.save(dict(step=step, t_step=int(t_step.item()), state=state),
               f"logs/{run_id}/optim_step{step:06d}.pt")

# -----------------------------------------------------------------------------
# Training and validation

train_steps = cfg.num_iterations
train_loader = distributed_data_generator(train_files, cfg.total_batch_size, cfg.seq_len, grad_accum_steps)
inputs, targets, cu_seqlens = next(train_loader)   # kick off the first batch

# Each val pass draws val_tokens through micro-batches shaped exactly like
# training's (so the rotary-cache bound holds), scored with the eager forward.
micro_world_tokens = cfg.total_batch_size // grad_accum_steps   # tokens per micro-batch across ranks
assert cfg.val_tokens % micro_world_tokens == 0
val_steps = cfg.val_tokens // micro_world_tokens

val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0.0
total_training_time = 0.0   # seconds; excludes the first 10 steps (compile lives there)

for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0):
        torch.cuda.synchronize()
        val_t0 = time.perf_counter()
        val_loader = distributed_data_generator(val_files, cfg.total_batch_size, cfg.seq_len, grad_accum_steps)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        for _ in range(val_steps):
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
        print0(f"step:{step}/{train_steps} val_bpb:{val_bpb:.6f} val_time:{val_elapsed:.2f}s", console=True)
        wandb_run.log({"step": step, "val/bpb": val_bpb, "val/eval_seconds": val_elapsed,
                       "total_training_time": total_training_time})

    # --------------- CHECKPOINT CAPTURE -----------------
    # State on entering step `step` = after `step` completed updates. Every
    # rank enters (the gathers are collectives); only master writes.
    if cfg.save_checkpoint and (last_step or step in cfg.save_steps):
        ckpt_t0 = time.perf_counter()
        write_checkpoint(step)
        print0(f"checkpoint captured at step {step} ({time.perf_counter() - ckpt_t0:.1f}s)", console=True)

    if last_step:
        # --------------- CORE EVALUATION -----------------
        if os.path.exists(os.path.join(DATASET_DIR, "core_eval/config.json")):
            core_eval_t0 = time.perf_counter()
            core_out = evaluate_core()
            core_eval_elapsed = time.perf_counter() - core_eval_t0
            print0(f"CORE metric: {core_out['core_metric']:.4f} | total CORE eval time: {core_eval_elapsed:.2f}s", console=True)
            for label, acc in core_out['results'].items():
                print0(f"  {label}: accuracy={acc:.4f} centered={core_out['centered_results'][label]:.4f}", console=True)
            wandb_run.log({
                "step": step,
                "core_metric": core_out["core_metric"],
                **{f"core/{label}/accuracy": acc for label, acc in core_out["results"].items()},
                **{f"core/{label}/centered": c for label, c in core_out["centered_results"].items()},
                "timing/core_eval_seconds": core_eval_elapsed,
            })
        else:
            print0("No core_eval/ in the dataset dir; skipping the CORE metric.", console=True)
        break

    # --------------- TRAINING SECTION -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()
    for micro in range(grad_accum_steps):
        # loss_scale replaces the loss/grad_accum division of an autograd loop
        loss = fb(inputs, targets, cu_seqlens, loss_scale=grad_scale)
        inputs, targets, cu_seqlens = next(train_loader)  # prefetch while the GPU is busy
    optimizer_step()   # schedules pre-computed; advances t_step on-device
    # Zero every grad buffer: gradients accumulate across the next step's
    # micro-batches, and at world=1 Muon's nesterov lerp just MUTATED grad32
    # (at world>1 it mutates the reduce-scattered shard instead) -- this is
    # correctness, not hygiene. (`for p in m` = every trained weight, in
    # Model.weight_names order.)
    for p in m:
        p.grad32.zero_()
    train_loss = loss.item()   # the step's one host sync point
    torch.cuda.synchronize()
    dt = time.perf_counter() - step_t0

    # logging (CPU only). EMA the loss for readability; time totals exclude the
    # first 10 steps, where compilation dominates.
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    if step > 10:
        total_training_time += dt
    tok_per_sec = int(cfg.total_batch_size / dt)
    mfu = 100 * num_flops_per_token * cfg.total_batch_size / dt / (gpu_peak_flops * world_size)
    steps_timed = step - 10
    if steps_timed > 0:
        eta_seconds = (train_steps - step - 1) * (total_training_time / steps_timed)
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    pct_done = 100 * step / train_steps
    print0(f"step {step:05d}/{train_steps:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {sched.lrm_table[step]:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m{eta_str}", console=True)
    wandb_run.log({
        "step": step,
        "train/loss": debiased_smooth_loss,
        "train/lrm": float(sched.lrm_table[step]),
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
        "total_training_time": total_training_time,
    })

    # GC management: the collector's cycle scans cost ~500ms at random steps,
    # so collect the setup garbage once, then freeze survivors and disable.
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"total training time: {total_training_time/60:.2f}m", console=True)
if val_bpb is not None:
    print0(f"minimum validation bpb: {min_val_bpb:.6f}", console=True)

wandb_run.finish()
dist.destroy_process_group()
