# probe_table3.py
#
# Trying: the count-derived table init under run 3's learning rates (the run-4 family)
# on held-out batches, and which parameter group's step-0 update carries probe 2's
# step-1 blow-up (§ Probe 3 at the end). Above it: the branch's script (stacks@bigram-embeds
# d75f72d) up to its weight init, wandb and checkpoints off. Assembled by make_table_probe3.py.
#
# The sep-14 baseline plus hashed bigram embeddings, instrumented for studying them:
# every step it logs the value, grad and bias-corrected m/sqrt_v of a few weights in
# each AdamW parameter (§ Parameter Traces) to the decoder_rtx_bi_embed wandb project,
# and it checkpoints at steps 250, 500, 750 and the end. (The attention output gates
# this branch also carried are set aside for now; e8e6870 is the last commit with them.)
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
    d_bigram:   int = 5 * 32768 # 163,840 hashed [prev, curr] token pairs.
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
    num_params:          int = 412_090_861     # every trained weight (§ Weight Init & Schedule)
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
    wandb_project:   str = "decoder_rtx_bi_embed"
    run_name:        str = "bi_embed_01_no_gates"  # both wandb and log files
    use_wandb:       bool = False

    save_checkpoint: bool = False
    save_steps:      tuple = (250, 500, 750)  # and the last step

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
    input_embeds:  Param = None
    bigram_embeds: Param
    smear_gate:    Param
    smear_lambda:  Param

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
    bigram_lambdas: Param   # Per-layer coefficient for reading the bigram embedding.
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

    # Hash each [prev, curr] token pair into a bigram slot. Position 0 takes the
    # reserved slot; a doc's first token hashes against the previous doc's last.
    bigram_ids = idx.to(torch.int32).clone()
    bigram_ids[0] = cfg.d_bigram - 1
    bigram_ids[1:] = torch.bitwise_xor(36313 * bigram_ids[1:],
                                       27191 * bigram_ids[:-1]) % (cfg.d_bigram - 1)
    # Re-read by every layer alongside x0; appears as xb_bigram in bwd.
    x_bigram = F.embedding(bigram_ids, m.bigram_embeds.w)

    # The residual stream starts as the smeared embedding.
    x = x0

    # One LayerStash of forward activations per layer
    stash = []

    # For each layer,
    for i in range(cfg.n_layers):
        x_in = x

        # Scale residual stream, add input and bigram embeddings
        x_biased = (m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
                    + m.bigram_lambdas.w[i] * x_bigram)
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
    g_resid = []; g_x0 = []; g_bigram = []

    # Each stream's cosine similarity to the vocab signal, divided by d_model.
    # (T, 1) = ((T, d_model) * (T, d_model)).mean()
    res_ms = (x_final_hat.float() * xb_final_hat.float()).mean(dim=-1, keepdim=True)

    # The component of xb_final_hat which is perpendicular to x_final_hat?
    xb_final = bf16(x_final_inv_rms * (xb_final_hat.float() - (x_final_hat.float() * res_ms)))

    # Dot product between final vs. backout streams.
    if is_last_micro:
        m.backout_lambda.grad.add_(-sum32(xb_final * x_backout))  # (T, d_model)

    # xb updates every layer, keep xb_final for backout layer.
    xb = xb_final                       # grad wrt layer num_layers-1's output
    x0b = torch.zeros_like(x0)          # accumulates over layers
    xb_bigram = torch.zeros_like(x0)    # accumulates over layers

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
        # --- blend backward: x_biased = resid_lambdas[i]*x_in + x0_lambdas[i]*x0 + bigram_lambdas[i]*x_bigram ---
        if is_last_micro:
            g_resid.append(sum32(xb_biased * st.x_in))
            g_x0.append(sum32(xb_biased * x0))
            g_bigram.append(sum32(xb_biased * x_bigram))
        x0b = x0b + m.x0_lambdas.w[i] * xb_biased  # TRAP: x0 feeds every layer, accumulate
        xb_bigram = xb_bigram + m.bigram_lambdas.w[i] * xb_biased  # TRAP: same, x_bigram feeds every layer
        xb = m.resid_lambdas.w[i] * xb_biased
        stash[i] = None                          # free this layer's stash as we go

    # Land the per-layer resid/x0/bigram scalar sums (collected in REVERSED
    # layer order) as one stacked add each.
    if is_last_micro:
        m.resid_lambdas.grad.add_(torch.stack(g_resid[::-1]))
        m.x0_lambdas.grad.add_(torch.stack(g_x0[::-1]))
        m.bigram_lambdas.grad.add_(torch.stack(g_bigram[::-1]))

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
    m.bigram_embeds.grad.add_(
        torch.ops.aten.embedding_dense_backward(xb_bigram, bigram_ids, cfg.d_bigram, -1, False))

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
bigram_lambdas = fp32_empty(cfg.n_layers).fill_(0.1)
smear_lambda   = fp32_zeros(1)
backout_lambda = fp32_empty(1).fill_(0.2)

smear_gate     = fp32_empty(1, cfg.d_smr_gate).uniform_(-cfg.d_smr_gate ** -0.5, cfg.d_smr_gate ** -0.5)

# Grad mix-in coefficient (1-beta), currently fixed, can be scheduled here.
scalar_grad_mult_t = np.ones(cfg.num_steps)


scalar_configs = [
#   name,               weights,       peak lr,  b1_grad,   b2_grad,    wd,
    ("resid_lambdas",   resid_lambdas,   0.005,    0.2,        0.05,   0.05),
    ("x0_lambdas",      x0_lambdas,      0.5,      0.04,       0.05,   0.0),
    ("bigram_lambdas",  bigram_lambdas,  0.5,      0.04,       0.05,   0.0),
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

bigram_embeds = bf16_zeros(cfg.d_bigram, cfg.d_model)

ve_rows = cfg.num_ves * cfg.d_vocab

embed_configs = [
#   name,             weights,       peak lr,  b1_grad,  b2_grad,  wd,       slots
    ("input_embeds",  input_embeds,  0.3,      0.2,      0.005,    0.001,   1),
    ("value_embeds",  value_embeds,  0.15,     0.2,      0.005,    0.01,   cfg.num_ves),
    ("bigram_embeds", bigram_embeds, 0.9,      0.2,      0.005,    0.001,   1)
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

# Halve the head's log-frequency component.
token_counts = torch.tensor(np.load(os.path.join(DATASET_DIR, "tokenizer/token_counts.npz"))["train"],
                            dtype=torch.float32, device=device)
occurrences_per_step = token_counts / token_counts.sum() * cfg.total_batch_size
freq_weight = occurrences_per_step.clamp_min(1e-3)
log_freq = (occurrences_per_step + 1e-3).log()
log_freq = (log_freq - (freq_weight * log_freq).sum() / freq_weight.sum()).unsqueeze(1)           # centred, (V, 1)
head_mean = (freq_weight.unsqueeze(1) * head_prior).sum(0) / freq_weight.sum()
freq_direction = ((freq_weight.unsqueeze(1) * log_freq * (head_prior - head_mean)).sum(0)
                  / (freq_weight * log_freq.squeeze(1).square()).sum())                           # (D,)
head_prior -= 0.5 * log_freq * freq_direction
del token_counts, occurrences_per_step, freq_weight, log_freq, head_mean, freq_direction

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
del lm_head, input_embeds, value_embeds, bigram_embeds, resid_lambdas, x0_lambdas, bigram_lambdas, smear_gate, smear_lambda, backout_lambda, W_Q, W_K, W_V, W_O, ve_gate, W_in, W_out
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
# § Probe 3: the count-derived init under run 3's learning rates, and the step-1 blow-up
# ==============================================================================
# probe_table2.py: the trained slots follow the gradient-shaped direction
# g = H0^T (p_hat(.|slot) - p_b(.|curr)) (cos 0.35-0.38 for slots hit >= 10 times a step,
# against out-of-sample counts), not the least-squares fit (0.02); seeding the slots with
# g at a mean norm of 10 gave a head start on held-out batches (6.686 / 6.439 at steps 0-1
# against 6.833 / 6.502) that was level by step 5. But the least-squares inits exploded at
# step 1 (loss 19-25) at every norm, and so did g at norm 5 while 10 and 25 did not.
# Here, with the same out-of-sample statistics (steps 26-275):
#   1. the run-4 family: g on every slot with >= 16 hits, table lr 0.1 and bigram_lambdas
#      lr 0.1 (run 3's), at mean norms 5 / 10 / 25, plus table lr 0.3 at norm 10, against
#      the zero table under the same lrs;
#   2. the blow-up: g at norm 5 under the probe-2 lrs (table 0.9, lambda 0.1), with one
#      parameter group's step-0 update withheld at a time (its lr table zeroed for the
#      candidate), scored over 3 steps -- the group whose absence removes the step-1 loss
#      of ~19 is the one that carries it. The lambdas after step 0 are printed for each.

V, D, S = cfg.d_vocab, cfg.d_model, cfg.d_bigram
torch.cuda.empty_cache()
MUON = ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "ve_gate")

def hash_slots(idx):
    ids = idx.to(torch.int32)
    return (torch.bitwise_xor(36313 * ids[1:], 27191 * ids[:-1]) % (S - 1)).to(torch.int64)

t_step = torch.zeros(1, dtype=torch.int64, device=device)
train_loader = data_generator("train", cfg.seq_len, cfg.micro_batch_tokens, cfg.num_steps * cfg.grad_accum_steps)
batches = [next(train_loader) for _ in range(26 * cfg.grad_accum_steps)]   # steps 0-19 (unused here) and 20-25 held out

hits = torch.zeros(S, dtype=torch.int64, device=device)
pair_keys, next_keys = [], []
for i in range(250 * cfg.grad_accum_steps):
    inputs, targets, _ = next(train_loader)
    slots = hash_slots(inputs)
    hits += torch.bincount(slots, minlength=S)
    idx64 = inputs.to(torch.int64)
    pair_keys.append(idx64[:-1] * V + idx64[1:])
    next_keys.append(slots * V + targets[1:].to(torch.int64))
train_loader.close()
hits[S - 1] = 0
pk, pc = torch.unique(torch.cat(pair_keys), return_counts=True); del pair_keys
nk, nc = torch.unique(torch.cat(next_keys), return_counts=True); del next_keys
prev_p, curr_p = pk // V, pk % V
slot_p = (torch.bitwise_xor(36313 * curr_p.to(torch.int32), 27191 * prev_p.to(torch.int32)) % (S - 1)).to(torch.int64)
slot_n, next_n = nk // V, nk % V
hF = hits.float()
print(f"statistics from steps 26-275: {int(hits.sum().item()):,} positions, {pk.numel():,} distinct pairs, {nk.numel():,} distinct (slot, next)", flush=True)

H0 = rebuild_master(m.lm_head.w, m.lm_head.mantissa).clone()
bg = np.load(os.path.join(DATASET_DIR, "tokenizer/bigram_counts.npz"))
indptr = torch.from_numpy(bg["train_indptr"]).to(device)
bcols  = torch.from_numpy(bg["train_indices"].astype(np.int64)).to(device)
bvals  = torch.from_numpy(bg["train_data"].astype(np.float32)).to(device)
brows  = torch.repeat_interleave(torch.arange(V, device=device), indptr[1:] - indptr[:-1])
del bg
row_tot = torch.zeros(V, device=device).index_add_(0, brows, bvals)
col_tot = torch.zeros(V, device=device).index_add_(0, bcols, bvals)
unigram = (col_tot + 0.5) / (col_tot.sum() + 0.5 * V)
CH = torch.zeros(V, D, device=device)
for s in range(0, bvals.numel(), 2_000_000):
    e = s + 2_000_000
    CH.index_add_(0, brows[s:e], bvals[s:e, None] * H0[bcols[s:e]])
Hbar_b = (CH + 50.0 * (unigram @ H0)) / (row_tot + 50.0)[:, None]
del CH, brows, bcols, bvals, indptr
g = torch.zeros(S, D, device=device)
for s in range(0, nc.numel(), 2_000_000):
    e = s + 2_000_000
    g.index_add_(0, slot_n[s:e], (nc[s:e].float() / hits[slot_n[s:e]].clamp_min(1).float())[:, None] * H0[next_n[s:e]])
for s in range(0, pc.numel(), 2_000_000):
    e = s + 2_000_000
    g.index_add_(0, slot_p[s:e], -(pc[s:e].float() / hits[slot_p[s:e]].clamp_min(1).float())[:, None] * Hbar_b[curr_p[s:e]])
del Hbar_b

def scaled(rows, target_norm):
    T = torch.zeros(S, D, device=device)
    n = g[rows].norm(dim=1)
    T[rows] = g[rows] * (target_norm / ((n * hF[rows]).sum() / hF[rows].sum()))
    return T.to(torch.bfloat16).cpu()

wide = (hits >= 16).nonzero().squeeze(1)
inits = {nu: scaled(wide, nu) for nu in (5.0, 10.0, 25.0)}
print(f"seeded slots (>= 16 hits in steps 26-275): {wide.numel():,}, {hF[wide].sum().item() / hF.sum().item():.1%} of positions", flush=True)
del g, H0, pk, pc, nk, nc, prev_p, curr_p, slot_p, slot_n, next_n
torch.cuda.empty_cache()

snap = {p.name: {attr: getattr(p, attr).cpu() for attr in ("w", "mantissa", "first_mntm", "scnd_mntm")
                 if getattr(p, attr) is not None} for p in m}
lr_tables = {p.name: p.lr_bc_t.clone() for p in m}

def restore():
    for p in m:
        for attr, saved in snap[p.name].items():
            getattr(p, attr).copy_(saved)
        p.lr_bc_t.copy_(lr_tables[p.name])
        p.grad.zero_()
    t_step.zero_()

def train_steps(n, start=20):
    losses, lam = [], []
    for step in range(n):
        for micro_i in range(cfg.grad_accum_steps):
            inputs, targets, cu_seqlens = batches[cfg.grad_accum_steps * (start + step) + micro_i]
            loss = forward_backward(inputs, targets, cu_seqlens, micro_step=micro_i,
                                    loss_scale=1.0 / (cfg.grad_accum_steps * inputs.size(0)))
        losses.append(loss.item())
        for p in m:
            (muon_step_fused if p.name in MUON else adamw_step_fused)(p, p.grad, t_step)
            p.grad.zero_()
        t_step.add_(1)
        lam.append((m.bigram_lambdas.w[0].item(), m.bigram_lambdas.w[10].item(), m.x0_lambdas.w[0].item(), m.x0_lambdas.w[10].item()))
    return losses, lam

def run3_lrs():
    m.bigram_embeds.lr_bc_t.mul_(1 / 9); m.bigram_lambdas.lr_bc_t.mul_(0.2)

print("\n## 1. the run-4 family on held-out batches (steps 20-25; step 0 = the init on that batch)")
print("| configuration | step 0 | 1 | 2 | 3 | 4 | 5 | lambdas after step 0 (bigram L0, L10; x0 L0, L10) |")
print("|---|---|---|---|---|---|---|---|", flush=True)

def candidate(label, setup=lambda: None, n=6):
    restore()
    setup()
    ls, lam = train_steps(n)
    print(f"| {label} | " + " | ".join(f"{l:.3f}" for l in ls) + " |" * (6 - n + 1) + " ".join(f"{v:+.2f}" for v in lam[0]) + " |", flush=True)

candidate("table off (bigram_lambdas 0)", lambda: m.bigram_lambdas.w.zero_())
candidate("zero table, table lr 0.1, lambda lr 0.1 (run 3)", run3_lrs)
for nu in (5.0, 10.0, 25.0):
    candidate(f"g on the seeded slots, mean norm {nu:g}, table lr 0.1, lambda lr 0.1", lambda nu=nu: (m.bigram_embeds.w.copy_(inits[nu].to(device)), run3_lrs()))
candidate("g, mean norm 10, table lr 0.3, lambda lr 0.1", lambda: (m.bigram_embeds.w.copy_(inits[10.0].to(device)), m.bigram_embeds.lr_bc_t.mul_(1 / 3), m.bigram_lambdas.lr_bc_t.mul_(0.2)))
candidate("g, mean norm 10, table lr 0.1, lambda lr 0.1, bigram_lambdas init 0.3", lambda: (m.bigram_embeds.w.copy_(inits[10.0].to(device)), run3_lrs(), m.bigram_lambdas.w.fill_(0.3)))
candidate("g, mean norm 10, table lr 0.9, lambda lr 0.1 (probe 2's setting)", lambda: (m.bigram_embeds.w.copy_(inits[10.0].to(device)), m.bigram_lambdas.lr_bc_t.mul_(0.2)))
candidate("g, mean norm 5, table lr 0.9, lambda lr 0.1 (probe 2's blow-up)", lambda: (m.bigram_embeds.w.copy_(inits[5.0].to(device)), m.bigram_lambdas.lr_bc_t.mul_(0.2)))

print("\n## 2. the blow-up: g at norm 5, table lr 0.9, lambda lr 0.1, one group's step-0 update withheld (3 steps)")
print("| withheld | step 0 | 1 | 2 | lambdas after step 0 |")
print("|---|---|---|---|---|", flush=True)
groups = {
    "nothing": (),
    "bigram_embeds": ("bigram_embeds",),
    "bigram_lambdas": ("bigram_lambdas",),
    "x0_lambdas": ("x0_lambdas",),
    "input_embeds": ("input_embeds",),
    "lm_head": ("lm_head",),
    "value_embeds": ("value_embeds",),
    "the Muon banks": MUON,
    "the other scalars (resid, smear, backout)": ("resid_lambdas", "smear_gate", "smear_lambda", "backout_lambda"),
}
for label, names in groups.items():
    restore()
    m.bigram_embeds.w.copy_(inits[5.0].to(device)); m.bigram_lambdas.lr_bc_t.mul_(0.2)
    for name in names:
        getattr(m, name).lr_bc_t.zero_()
        if name in MUON:
            getattr(m, name).wd_t.zero_()
    ls, lam = train_steps(3)
    print(f"| {label} | " + " | ".join(f"{l:.3f}" for l in ls) + " | " + " ".join(f"{v:+.2f}" for v in lam[0]) + " |", flush=True)
