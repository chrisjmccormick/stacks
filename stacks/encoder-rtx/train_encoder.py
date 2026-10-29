# train_encoder.py
#
# A small encoder trained from scratch to be a decision model, jointly on MLM
# over ClimbMix and on Kev's decision records: forked from decoder-rtx's
# train_stack.py (sep-28-baseline), with a handwritten forward and backward.
#
# What changed from decoder-rtx (see agent-ops-stacks/encoder-rtx/.../PROPOSAL.md):
# - Attention is bidirectional. A pack holds decision records and ClimbMix
#   filler; every state, question and filler document is its own segment, and
#   each record's questions also attend to its state in a second FA2 call,
#   merged by the softmax LSEs (§§ Attention in forward_backward).
# - The head trains MLM on the filler, and a single readout vector scores each
#   option's readout token, with a softmax over each question's options.
# - The vocabulary is decoder-rtx's 32,768 BPE plus the decision special tokens.
# - The CLM-derived inits (token frequency priors, pair code) are kept, with
#   [MASK]'s row solved so the direct path emits the unigram at masked positions
#   (agent-ops-stacks/encoder-rtx/2026-10-28_0902pm_phase0-first-runs: 0.022 nats
#   better at the end of a run than a random row).
# - Two stages. Stage 1 is MLM alone (DECISION_FRAC=0) and writes a checkpoint;
#   stage 2 starts from it (INIT_FROM) with fresh optimizer state, its own step
#   count and schedules, a decision share up to 0.9 and LR_SCALE on every rate.
#
# Downloads the required number of pre-tokenized Climbmix dataset shards if not
# already present. The decision records come from data/prepare_kev_suites.py.
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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,pinned_use_cuda_host_register:True,pinned_num_register_threads:16"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import Tensor

from torch._inductor.codecache import CacheBase

from utils import (DATASET_DIR, KEV_DIR, Pack, doc_stream, download_dataset, load_records,
                   materialize_packs, pack_caps, pack_loader, plan_packs,
                   flash_attn_varlen_fwd_lse, flash_attn_varlen_bwd)

dynamo.config.recompile_limit = 64

# Matmul backend chosen per shape, read from the cache shipped beside this script.
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.autotune_in_subproc = False

# Confirm Ampere or newer
assert torch.cuda.is_available(), "no GPU -- Runtime > Change runtime type > A100"

props = torch.cuda.get_device_properties(0)
print(f"{props.name} | {props.total_memory / 2**30:.1f} GiB | sm{props.major}{props.minor}")
assert props.major >= 8, f"needs Ampere or newer (got sm{props.major}{props.minor})"

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# Install the shipped autotune cache, if it was measured on this GPU, triton and CUDA.
autotune_cache = Path(CacheBase.get_local_cache_path())
shipped_cache = json.loads((Path(sys.argv[0]).resolve().parent / "autotune_cache.json").read_text())
if not autotune_cache.is_file() and shipped_cache["system"] == CacheBase.get_system():
    autotune_cache.parent.mkdir(parents=True, exist_ok=True)
    autotune_cache.write_text(json.dumps(shipped_cache))

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
    n_layers:   int = 8
    d_model:    int = 768

    backout_layer: int = 4 # nanochat: n_layers // 2

    # Input
    d_vocab:         int = 32832     # decoder-rtx's BPE + the decision special tokens, padded to a multiple of 64
    bpe_vocab:       int = 32768     # the BPE alone: the shards, the corpus statistics and the table keys
    d_bigram:        int = 64 * 32768 # 2,097,152 hashed [prev, curr] pairs, read into the residual stream.
    d_pair_values:   int = 32 * 32768 # 1,048,576 hashed [prev, curr] pairs, read into the attention values.
    d_pair_code:     int = 64 * 32768 # 2,097,152 frequent [prev, curr] pairs with a frozen corpus code, likewise.
    touched_rows:    int = 120_000    # Bound on the distinct rows of one table a batch touches (measured max: 114,895).
    d_smr_gate:      int = 24    # Gate input is first 24-dims of input embed.

    # Attention
    n_qo_heads: int = 6
    n_kv_heads: int = 6     # n_qo == n_kv means full multihead attention.
    d_qk:       int = 128   # Attention head size.
    d_vo:       int = 128   # Note: FA2 requires d_qk == d_vo, FA3 does not.

    # Context and Sliding Window Attention
    seq_len:          int = 2048    # filler documents are cut to their first seq_len tokens
    short_win_size:   int = 768     # each way: the window is 2 x 768 + 1 wide
    full_ctxt_layers: list[int] = [   3,    4,     7]
    window_sizes:     list[tuple[int, int]]  # Derived below.

    # Attention - Value Embeddings
    d_ve_gate: int = 12  # Gate input is first 12-dims of the layer's residual stream.
                         # Each head has its own gate, all with same input.
    ve_layers: list[int] = [1, 2, 5, 6, 7]
    ve_index:  list[int] # Derived from ve_layers.
    num_ves:   int

    # MLP
    d_mlp:      list[int] = [768 * r for r in (2, 2, 3, 3, 5, 5, 6, 6)]  # per layer; mean 4x = 3072

    # Model stats, for MFU and logging
    num_params:          int    # every trained weight, counted after init (§ Weight Init & Schedule)
    num_flops_per_token: int    # estimated once the pack sizes are known (§ Data)

    # ---- Training ----

    # Batch Size
    train_batch_tokens: int = 96 * 2048   # 192K tokens per step, one pack

    # Training: decoder-rtx's token budget, 2,079 x 192K = 409M tokens
    num_steps: int = 2079

    # Packs (§ Data). The static sizes are derived from the planned packs.
    decision_frac:   float = 0.10   # share of each pack given to decision records; the rest is MLM filler
    decision_weight: float = 1.0    # decision loss (nats per question) against MLM loss (nats per masked token)
    mask_rate:       float = 0.30   # of the filler tokens, in the pack with the least filler
    cu_width:        int            # cu_self entries
    rec_width:       int            # cu_q / cu_k entries
    ro_cap:          int            # readout positions per pack
    q_cap:           int            # questions per pack
    mlm_per_pack:    int            # masked filler positions per training pack (the evaluation packs' count is fixed: eval_mlm_per_pack)
    max_seg_len:     int            # FA2 max_seqlen bounds: any segment,
    max_qblock_len:  int            # one record's questions,
    max_state_len:   int            # one state

    # Two stages (§§ Init From a Checkpoint)
    init_from:       str = ""       # stage 2: a model_step*.pt written by stage 1
    lr_scale:        float = 1.0    # stage 2: every peak rate times this

    # Evaluation and logging
    val_loss_every:  int = 333

    # Logging
    wandb_project:   str = "encoderstack_rtx_dev"  # session work; baselines go to encoderstack_rtx
    run_name:        str = "phase0-kev"  # both wandb and log files
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

# Temporary knobs for the first runs, logged below; stripped once the values settle.
cfg.decision_frac   = float(os.environ.get("DECISION_FRAC", cfg.decision_frac))
cfg.decision_weight = float(os.environ.get("DECISION_WEIGHT", cfg.decision_weight))
cfg.mask_rate       = float(os.environ.get("MASK_RATE", cfg.mask_rate))
cfg.num_steps       = int(os.environ.get("NUM_STEPS", cfg.num_steps))
cfg.run_name        = os.environ.get("RUN_NAME", cfg.run_name)
cfg.init_from       = os.environ.get("INIT_FROM", cfg.init_from)
cfg.lr_scale        = float(os.environ.get("LR_SCALE", cfg.lr_scale))
cfg.val_loss_every  = int(os.environ.get("VAL_EVERY", cfg.val_loss_every))
cfg.save_checkpoint = bool(int(os.environ.get("SAVE_CHECKPOINT", int(cfg.save_checkpoint))))
assert 0.0 <= cfg.decision_frac <= 0.9, "decision_frac above 0.9 leaves too little filler for the MLM validation"
print(f"knobs: decision_frac {cfg.decision_frac} | decision_weight {cfg.decision_weight} | "
      f"mask_rate {cfg.mask_rate} | num_steps {cfg.num_steps} | run_name {cfg.run_name} | "
      f"init_from {cfg.init_from or '-'} | lr_scale {cfg.lr_scale} | val_every {cfg.val_loss_every} | "
      f"save_checkpoint {cfg.save_checkpoint}")

# Per-layer window sizes for sliding window attention in the self call, defined
# as (left, right) tuples: tokens attended to on each side, -1 for the whole
# segment. The question-to-state call always reads the whole state.
cfg.window_sizes = [(cfg.short_win_size, cfg.short_win_size)] * cfg.n_layers  # All short, ...
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (-1, -1)                                           # ... then overwrite with full.

gpu_device_name = torch.cuda.get_device_name(0)   # "NVIDIA RTX PRO 6000 Blackwell Server Edition"
# Dense BF16 peak FLOPS of the RTX PRO 6000, the MFU denominator.
gpu_peak_flops = 503.8e12

download_dataset()


# Reject a vocab mismatch between the .bin shards, the decision records and the model.
with open(os.path.join(DATASET_DIR, "config.json")) as f:
    climbmix_config = json.load(f)
assert climbmix_config["vocab_size"] == cfg.bpe_vocab, "dataset vocab != model bpe_vocab"
with open(os.path.join(KEV_DIR, "kev_meta.json")) as f:
    kev_meta = json.load(f)
assert (kev_meta["bpe_vocab"], kev_meta["d_vocab"]) == (cfg.bpe_vocab, cfg.d_vocab), "decision records' vocab != model's"
MASK_ID = kev_meta["special_tokens"]["<|mask|>"]
BOS_ID = climbmix_config["bos_id"]


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
    wd_t:         Tensor    # weight decay * non-corrected lr; AdamW/RMSProp store 1 - that
    mntm_b1_t:    Tensor    # Beta1
    grad_b1_t:    Tensor    # 1 - Beta1
    mntm_b2_t:    Tensor    # Beta2
    grad_b2_t:    Tensor    # 1 - Beta2
    eps_t:        Tensor    # AdamW and RMSProp

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
    value_embeds:  Param
    ve_gate:       Param
    pair_values:   Param  # [prev, curr]-indexed value memory, one table per VE layer.
    pair_gate:     Param
    pair_code:     Tensor # Frozen corpus code per frequent [prev, curr] pair; last row is zeros.
    pair_decoder:  Param  # Reads a pair's code into the value space, once for all VE layers.

    # MLP -- one bank per width, W_in_<d> / W_out_<d> (see mlp_banks)

    # Cross-Layer
    x0_lambdas:     Param   # Per-layer coefficient for reading the input embedding.
    x0_gates:       Param   # Per-layer scale on the channel-mean gate of that read.
    bigram_lambdas: Param   # Per-layer coefficient for reading the bigram embedding.
    resid_lambdas:  Param   # Per-layer gain on the residual stream.
    backout_lambda: Param

    # Output
    lm_head: Param   # MLM, on the masked filler positions
    readout: Param   # (1, D): each option's logit is this dotted with its readout token's final stream

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
    For T=256K  -->  Held: 49.5GB,  Recomputed:  20.25GB
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
    x_attn_out_hat:     Tensor    # (L,  T,     D)             4.5GB
    x_attn_out_inv_rms: Tensor    # (L,  T,     1)        fp32          (3MB)
    mlp_a:              Tensor    # (L,  T, d_mlp[i])           18GB
    #mlp_relu:          Tensor    # (L,  T, d_mlp)                               18GB
    #                                                         ------           ------
    #                                                 TOTAL:  49.5GB          20.25GB

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
def forward_backward(pk: Pack, backward=True):
    """One pack through the model: returns (MLM loss, decision loss) for training,
    or each masked position's MLM nats and each readout's probability for
    validation (backward=False)."""

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
    # Activation naming -- the MLP and the gates share three stages:
    # *_z        - Pre-nonlinearity, the raw matmul output. Only ever named in
    #              bwd (mlpb_z, ve_gateb_z, gateb_z, x0_gateb_z); inlined in fwd.
    #              ('logit' is reserved for the lm_head's output.)
    # *_relu     - Post-relu, pre-square (MLP only; recomputed in bwd).
    # *_sig      - Post-sigmoid, in [0, 1] (gates only; recomputed in bwd).
    # *_a        - The final activation handed onward: mlp_a = mlp_relu^2,
    #              ve_gate_a = 3*ve_gate_sig. The smear gate's is just 'gate'.

    idx = pk.idx
    assert idx.ndim == 1
    T = idx.size(0)
    half = cfg.d_qk // 2

    assert T > 1, "Forward pass should have T > 1 (smear needs a previous token)"

    # Rotary angles at each token's position within its segment; a question's
    # positions continue its state's.
    cos, sin = m.cos[0, pk.pos], m.sin[0, pk.pos]  # (T, 1, half)
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)
    bigram_ids, pair_ids, pair_code_ids = pk.rows[:3]   # each table's rows, computed on the host (see table_rows)
    # The smear reads the previous token only within a segment, so no question or
    # record sees another through it.
    keep = bf16(~pk.seg_start[1:]).unsqueeze(1)   # (T-1, 1)
    pair_code = F.embedding(pair_code_ids, m.pair_code)
    pair_code_v = pair_code @ m.pair_decoder.w[0].mT
    x_backout = None

    # -----------------------------
    #           Forward
    # -----------------------------

    # Input embeddings
    x_embed = F.embedding(idx, m.input_embeds.w)        # bf16

    # RMS normalize the input embeds
    x_embed_inv_rms = (x_embed.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x_embed_hat = bf16(x_embed.float() * x_embed_inv_rms)      # post-norm embedding, pre-smear

    # Smear: mix the previous token's embedding into the current position, inside its segment.
    # The sigmoid and its argument are unnamed here; bwd recomputes them as
    # gate_sig and (as a grad) gateb_z.
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(
        x_embed_hat[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)
    # The smeared input embedding; also added back to the stream at every layer.
    x0 = torch.cat([x_embed_hat[:1], x_embed_hat[1:] + gate * keep * x_embed_hat[:-1]], dim=0)  # appears as x0b in bwd

    # Bigram embeddings, one row per [prev, curr] token pair (see table_rows)
    x_bigram = bf16(F.embedding(bigram_ids, m.bigram_embeds.w) * bigram_w_scale)  # appears as xb_bigram in bwd

    # The residual stream starts as the smeared embedding.
    x = x0

    # One LayerStash of forward activations per layer
    stash = []

    # For each layer,
    for i in range(cfg.n_layers):
        x_in = x

        # Scale residual stream, add gated input and bigram embeddings
        x0_gate = 2.0 * torch.sigmoid(m.x0_gates.w[i] * x_in.float().mean(dim=-1, keepdim=True))   # (T, 1)
        x_biased = (m.resid_lambdas.w[i] * x_in + bf16(m.x0_lambdas.w[i] * x0_gate) * x0
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

            # Pair value memory, plus the pair's decoded code: gated like ve, from the next slice of the stream
            pair_v = (bf16(F.embedding(pair_ids, pair_wbank[j]) * pair_w_scale) + pair_code_v).view(T, cfg.n_kv_heads, cfg.d_vo)
            pair_gate_sig = torch.sigmoid(x_biased_hat[..., cfg.d_ve_gate:2 * cfg.d_ve_gate] @ m.pair_gate.w[j].mT)
            v = v + (3 * pair_gate_sig).unsqueeze(-1) * pair_v

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

        # Read V from the streams that match K. Every segment -- state, question,
        # filler document -- attends to itself, both ways.
        y, lse = flash_attn_varlen_fwd_lse(q_hat, k_hat, v, pk.cu_self, pk.cu_self,
                                           cfg.max_seg_len, cfg.max_seg_len, cfg.window_sizes[i])

        # Each record's questions also read its whole state. The two calls are
        # softmaxes over disjoint keys, so their union is their outputs weighted by
        # each one's share of the merged normalizer, exp(lse - lse_merged). The
        # cross call leaves rows outside its segments unwritten: merge on question
        # rows only.
        y_x, lse_x = flash_attn_varlen_fwd_lse(q_hat, k_hat, v, pk.cu_q, pk.cu_k,
                                               cfg.max_qblock_len, cfg.max_state_len, (-1, -1))
        lse_merged = torch.logaddexp(lse, lse_x)    # (H, T)
        y_merged = (y.float() * (lse - lse_merged).exp().T.unsqueeze(-1)
                    + y_x.float() * (lse_x - lse_merged).exp().T.unsqueeze(-1))
        y = torch.where(pk.is_q[:, None, None], bf16(y_merged), y).contiguous()
        lse = torch.where(pk.is_q, lse_merged, lse)

        # Project value heads onto their output heads.
        attn_out = y.view(T, -1) @ m.W_O.w[i].mT

        # Write back to the stream.
        x_attn_out = x_biased + attn_out

        # MLP input norm. Stashed, not recomputed.
        x_attn_out_inv_rms = (x_attn_out.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        x_attn_out_hat = bf16(x_attn_out.float() * x_attn_out_inv_rms)

        # MLP -- this layer's width has its own bank; k is the layer's slot in it
        W_in, W_out, k = mlp_banks[i]
        mlp_a = F.relu(x_attn_out_hat @ W_in.w[k].mT).square()   # (T, d_mlp[i]) stashed
        mlp_out = mlp_a @ W_out.w[k].mT

        # Write back to the stream.
        x = x_attn_out + mlp_out            # the residual stream; appears as xb in bwd

        # Stash the backout layer's output, to subtract it off before LM head.
        if i == cfg.backout_layer:
            x_backout = x

        # Stash activations for backward pass. Skipped entirely under eval.
        if backward:
            stash.append(LayerStash(x_in=x_in, x_biased_hat=x_biased_hat, x_biased_inv_rms=x_biased_inv_rms,
                                    q_hat=q_hat, k_hat=k_hat, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
                                    v=v, y=y, lse=lse, x_attn_out_hat=x_attn_out_hat,
                                    x_attn_out_inv_rms=x_attn_out_inv_rms, mlp_a=mlp_a))

    x_final = x - bf16(m.backout_lambda.w) * x_backout # TODO - backout lambda could be bf16 instead.

    # Final output norm
    x_final_inv_rms = (x_final.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x_final_hat = bf16(x_final.float() * x_final_inv_rms)

    # -----------------------------
    #           MLM Head
    # -----------------------------
    # Only the masked filler positions are read out, M of them.
    x_mlm_hat = x_final_hat[pk.mlm_pos]         # (M, D)
    tgt = pk.mlm_tgt.unsqueeze(1)               # (M, 1)

    # ==== Forward ====
    logits_raw = x_mlm_hat @ m.lm_head.w.mT     # (M, d_vocab) bf16

    # "softcap" logits to the range -15 to 15
    logits = 15.0 * torch.tanh(logits_raw.float() / 15.0) # (M, d_vocab)

    # Typically, subtract off the highest logit per token first:
    #    max_logit = logits.amax(dim=1, keepdim=True) # (M, 1)
    #    e = (logits - max_logit).exp()  # (M, d_vocab) = ((M, d_vocab) - (M, 1)).exp()
    # Softcap bounds to [-15, 15], so exp() is [3.06e-7, 3.3e6], so fp32 is ok.
    e = logits.exp()  # (M, d_vocab)

    # Softmax denominator
    ssum = e.sum(dim=1, keepdim=True) # (M, 1)

    # ==== Cross Entropy ====
    # Convert back to logit space
    lse_ce = ssum.log().squeeze(1) # (M,)
    # Select prediction logits for target token (one random access per row)
    tgt_logit = logits.gather(1, tgt).squeeze(1) # (M,)
    mlm_nats = lse_ce - tgt_logit               # (M,)

    # -----------------------------
    #        Decision Readout
    # -----------------------------
    # Each option's logit is the readout vector dotted with its readout token's
    # stream; a softmax runs over each question's options. Padding readouts
    # belong to a dummy question, q_cap, and carry no target and no gradient.
    x_ro_hat = x_final_hat[pk.ro_pos]                               # (Rc, D)
    ro_logit = x_ro_hat.float() @ m.readout.w[0]                    # (Rc,)
    ro_max = torch.full((cfg.q_cap + 1,), -math.inf, device=device).scatter_reduce(
        0, pk.ro_qid, ro_logit, "amax")                             # (Qc + 1,)
    ro_e = (ro_logit - ro_max[pk.ro_qid]).exp()
    ro_ssum = torch.zeros(cfg.q_cap + 1, device=device).scatter_add(0, pk.ro_qid, ro_e)
    ro_p = ro_e / ro_ssum[pk.ro_qid]                                # (Rc,) each option's probability

    # If validation pass, return MLM nats and the options' probabilities.
    if not backward:
        return mlm_nats, ro_p
    # Return training losses for logging: nats per masked token, nats per question.
    loss = torch.stack([mlm_nats.mean(),
                        -(pk.ro_tgt * ro_p.clamp_min(1e-30).log()).sum() / pk.n_q[0].clamp_min(1.0)])

    # ==== Backward ====
    onehot = torch.arange(cfg.d_vocab, device=device).unsqueeze(0) == tgt # (M, d_vocab)

    # Predicted probs = e / ssum; (M, d_vocab)
    # Standard approach to backprop is (p - 1), with optimizer update w = w - grad.
    # We flip this to (1 - p) and w = w + grad, so that:
    # - grads and optim state move in the same direction as their param.
    # - target token contributes positively to stream and gradients.
    logitsb = bf16((onehot.float() - (e / ssum)) * (1.0 - logits/15.0 * logits/15.0) / pk.mlm_pos.size(0))

    # Every masked token updates every vocab entry.
    m.lm_head.grad.copy_((logitsb.mT @ x_mlm_hat).float()) # (d_vocab, M) @ (M, d_model) --> (d_vocab, d_model)

    # The decision readout, flipped the same way: target - p, scaled to the mean
    # over the pack's questions.
    ro_valid = (pk.ro_qid < cfg.q_cap).float()
    ro_logitb = (pk.ro_tgt - ro_p) * ro_valid * (cfg.decision_weight / pk.n_q[0].clamp_min(1.0))   # (Rc,); no questions in an MLM-only pack
    m.readout.grad.copy_((ro_logitb.unsqueeze(1) * x_ro_hat.float()).sum(dim=0, keepdim=True))

    # The backward streams start as weighted sums of the head embeddings that
    # they (meaningfully) predicted, at the masked positions, and as the readout
    # vector at the readout positions. Padding readouts add zeros at row 0.
    xb_final_hat = torch.zeros_like(x_final_hat)
    xb_final_hat.index_put_((pk.mlm_pos,), logitsb @ m.lm_head.w)   # (M, d_vocab) @ (d_vocab, d_model)
    xb_final_hat.index_put_((pk.ro_pos,), bf16(ro_logitb.unsqueeze(1) * m.readout.w), accumulate=True)
    del logitsb

    # -----------------------------
    #           Backward
    # -----------------------------
    # The trained tables' rows, compacted for the sparse updates (see table_rows).
    bigram_inverse, bigram_rows, pair_inverse, pair_rows = pk.rows[3:]
    bigram_rows, pair_rows = bigram_rows[:cfg.touched_rows], pair_rows[:cfg.touched_rows]

    pair_w_scale_next = pair_w_scale * m.pair_values.wd_t[t_step]
    bigram_w_scale_next = bigram_w_scale * m.bigram_embeds.wd_t[t_step]

    # Scalar grads are collected, grad tensors updated at the end.
    g_resid = []; g_x0 = []; g_bigram = []; g_x0_gate = []

    # Each stream's cosine similarity to the vocab signal, divided by d_model.
    # (T, 1) = ((T, d_model) * (T, d_model)).mean()
    res_ms = (x_final_hat.float() * xb_final_hat.float()).mean(dim=-1, keepdim=True)

    # The component of xb_final_hat which is perpendicular to x_final_hat?
    xb_final = bf16(x_final_inv_rms * (xb_final_hat.float() - (x_final_hat.float() * res_ms)))

    # Dot product between final vs. backout streams.
    m.backout_lambda.grad.copy_(-sum32(xb_final * x_backout))  # (T, d_model)

    # xb updates every layer, keep xb_final for backout layer.
    xb = xb_final                       # grad wrt layer num_layers-1's output
    x0b = torch.zeros_like(x0)          # accumulates over layers
    xb_bigram = torch.zeros_like(x0)    # accumulates over layers
    pair_code_vb = torch.zeros_like(pair_code_v)  # accumulates over VE layers

    # For each layer in reverse order,
    for i in reversed(range(cfg.n_layers)):

        # Stashed forward activations for this layer.
        st = stash[i]

        if i == cfg.backout_layer:
            # TRAP: x_backout gets an EXTRA contribution when the sweep passes num_layers//2
            xb = xb - bf16(m.backout_lambda.w) * xb_final

        # --- MLP backward (relu^2: mlpb_z = 2*mlp_relu*mlpb_a, self-masking
        #     since mlp_relu is already 0 where z < 0) ---

        W_in, W_out, k = mlp_banks[i]

        # Grad w.r.t. W_out
        W_out.gbank[k].copy_(xb.mT @ st.mlp_a) # (d_model, T) @ (T, d_mlp[i])

        # Grad w.r.t. the pre-relu matmul output (mlpb_a = xb @ W_out is inline)
        mlp_relu = st.mlp_a * (st.mlp_a + 1e-30).rsqrt()   # = mlp_a.sqrt()
        mlpb_z = 2.0 * mlp_relu * (xb @ W_out.w[k])

        x_attn_out_inv_rms, x_attn_out_hat = st.x_attn_out_inv_rms, st.x_attn_out_hat

        # Grad w.r.t. W_in
        W_in.gbank[k].copy_(mlpb_z.mT @ x_attn_out_hat) # (d_mlp[i], T) @ (T, d_model)

        xb_attn_out_hat = mlpb_z @ W_in.w[k] # (T, d_mlp[i]) @ (d_mlp[i], d_model) --> (T, d_model)

        xb_attn_out = xb + bf16(x_attn_out_inv_rms * (xb_attn_out_hat.float() - (x_attn_out_hat.float() * (x_attn_out_hat.float() * xb_attn_out_hat.float()).mean(dim=-1, keepdim=True))))

        # Attention backward
        x_biased_hat = st.x_biased_hat
        m.W_O.gbank[i].copy_(xb_attn_out.mT @ st.y.view(T, -1))
        yb = (xb_attn_out @ m.W_O.w[i]).view(T, cfg.n_qo_heads, cfg.d_vo)

        # Each call's backward, given the MERGED output and LSE: its softmax weights
        # exp(s - lse) are then its share of the merged softmax, and FA2's row term
        # rowsum(yb * y) is the merged one, so the two calls' grads sum to the exact
        # grad of the union. The cross call's grads are zero outside its rows.
        qb_hat, kb_hat, vb = flash_attn_varlen_bwd(
            yb, st.q_hat, st.k_hat, st.v, st.y, st.lse, pk.cu_self, pk.cu_self,
            cfg.max_seg_len, cfg.max_seg_len, cfg.window_sizes[i])
        qb_x, kb_x, vb_x = flash_attn_varlen_bwd(
            yb, st.q_hat, st.k_hat, st.v, st.y, st.lse, pk.cu_q, pk.cu_k,
            cfg.max_qblock_len, cfg.max_state_len, (-1, -1))
        qb_hat, kb_hat, vb = qb_hat + qb_x, kb_hat + kb_x, vb + vb_x

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
        xb_biased_hat_pair = None
        if j >= 0:
            # Retrieve the value embeddings
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)

            # Recompute gate forward
            ve_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)

            # ve_gate_a broadcasts over d_vo in v = v0 + a*ve, so its grad sums
            # that axis back out; then d/dz[3*sigmoid(z)] = 3*sig*(1 - sig).
            ve_gateb_a = (vb * ve).sum(dim=-1)   # (T, n_kv_heads)
            ve_gateb_z = ve_gateb_a * (3 * ve_gate_sig * (1 - ve_gate_sig))

            m.ve_gate.gbank[j].copy_(ve_gateb_z.mT @ x_biased_hat[..., :cfg.d_ve_gate])

            # Embedding gradient
            veb = (vb * (3 * ve_gate_sig).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            # embedding_dense_backward beat raw index_add_ atomics ~2x
            m.value_embeds.gbank[j].copy_(
                torch.ops.aten.embedding_dense_backward(veb, idx, cfg.d_vocab, -1, False))

            xb_biased_hat_ve = ve_gateb_z @ m.ve_gate.w[j]

            # --- pair value memory backward (pair_v and pair_gate_sig recomputed) ---
            pair_v = (bf16(F.embedding(pair_ids, pair_wbank[j]) * pair_w_scale) + pair_code_v).view(T, cfg.n_kv_heads, cfg.d_vo)
            pair_gate_sig = torch.sigmoid(x_biased_hat[..., cfg.d_ve_gate:2 * cfg.d_ve_gate] @ m.pair_gate.w[j].mT)
            pair_gateb_z = (vb * pair_v).sum(dim=-1) * (3 * pair_gate_sig * (1 - pair_gate_sig))   # (T, n_kv_heads)
            m.pair_gate.gbank[j].copy_(pair_gateb_z.mT @ x_biased_hat[..., cfg.d_ve_gate:2 * cfg.d_ve_gate])
            pair_vb = (vb * (3 * pair_gate_sig).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            pair_code_vb = pair_code_vb + pair_vb

            # Sparse update: row-wise RMSProp on this slot's touched rows (pair_rows), fused in at its gradient.
            pair_grad = torch.ops.aten.embedding_dense_backward(pair_vb.float(), pair_inverse, cfg.touched_rows, -1, False)
            pair_vbank[j].mul_(m.pair_values.mntm_b2_t[t_step])   # (rows, 1) fp32
            pair_slot_mntm = (pair_vbank[j].index_select(0, pair_rows)
                              + pair_grad.square().mean(dim=-1, keepdim=True) * m.pair_values.grad_b2_t[t_step])
            pair_vbank[j].index_copy_(0, pair_rows, pair_slot_mntm)
            pair_wbank[j].index_copy_(0, pair_rows, bf16(
                pair_wbank[j].index_select(0, pair_rows).float()
                + (m.pair_values.lr_bc_t[t_step] / pair_w_scale_next)
                  * (pair_grad / (pair_slot_mntm.sqrt() + m.pair_values.eps_t[t_step]))))
            xb_biased_hat_pair = pair_gateb_z @ m.pair_gate.w[j]

        # vb passes through the VE add unchanged: v = v0 + ve_gate_a*ve
        qb = qb.view(T, cfg.n_qo_heads * cfg.d_qk)
        kb = kb.view(T, cfg.n_kv_heads * cfg.d_qk)
        vb = vb.reshape(T, cfg.n_kv_heads * cfg.d_vo)

        m.W_Q.gbank[i].copy_(qb.mT @ x_biased_hat)
        m.W_K.gbank[i].copy_(kb.mT @ x_biased_hat)
        m.W_V.gbank[i].copy_(vb.mT @ x_biased_hat)

        xb_biased_hat = qb @ m.W_Q.w[i] + kb @ m.W_K.w[i] + vb @ m.W_V.w[i]
        if xb_biased_hat_ve is not None:
            xb_biased_hat[:, :cfg.d_ve_gate] += xb_biased_hat_ve
            xb_biased_hat[:, cfg.d_ve_gate:2 * cfg.d_ve_gate] += xb_biased_hat_pair
        xb_biased = xb_attn_out + bf16(st.x_biased_inv_rms * (xb_biased_hat.float() - (x_biased_hat.float() * (x_biased_hat.float() * xb_biased_hat.float()).mean(dim=-1, keepdim=True))))
        # --- blend backward: x_biased = resid_lambdas[i]*x_in + x0_lambdas[i]*x0_gate*x0 + bigram_lambdas[i]*x_bigram,
        #     x0_gate = 2*sigmoid(x0_gates[i] * mean_c(x_in)) ---
        x_in_mean  = st.x_in.float().mean(dim=-1, keepdim=True)                         # (T, 1)
        x0_gate    = 2.0 * torch.sigmoid(m.x0_gates.w[i] * x_in_mean)                   # (T, 1)
        x0_dot     = (xb_biased * x0).sum(dim=-1, keepdim=True, dtype=torch.float32)    # (T, 1)
        x0_gateb_z = x0_dot * m.x0_lambdas.w[i] * x0_gate * (1.0 - 0.5 * x0_gate)       # (T, 1)
        g_resid.append(sum32(xb_biased * st.x_in))
        g_x0.append((x0_dot * x0_gate).sum())
        g_bigram.append(sum32(xb_biased * x_bigram))
        g_x0_gate.append((x0_gateb_z * x_in_mean).sum())
        x0b = x0b + bf16(m.x0_lambdas.w[i] * x0_gate) * xb_biased  # TRAP: x0 feeds every layer, accumulate
        xb_bigram = xb_bigram + m.bigram_lambdas.w[i] * xb_biased  # TRAP: x_bigram feeds every layer, accumulate
        xb = m.resid_lambdas.w[i] * xb_biased + bf16(x0_gateb_z * (m.x0_gates.w[i] / cfg.d_model))
        stash[i] = None                          # free this layer's stash as we go

    m.pair_decoder.gbank[0].copy_(pair_code_vb.mT @ pair_code)

    # Land the per-layer resid/x0/bigram/x0-gate scalar sums (collected in REVERSED
    # layer order) as one stacked copy each.
    m.resid_lambdas.grad.copy_(torch.stack(g_resid[::-1]))
    m.x0_lambdas.grad.copy_(torch.stack(g_x0[::-1]))
    m.bigram_lambdas.grad.copy_(torch.stack(g_bigram[::-1]))
    m.x0_gates.grad.copy_(torch.stack(g_x0_gate[::-1]))

    # xb is now the grad through layer 0's input, which IS x0 (same tensor), so
    # it folds into x0b to give the full grad wrt the smeared embedding.
    x0b = x0b + xb

    # --- smear backward: x0 = cat([x_embed_hat[:1], x_embed_hat[1:] + gate*keep*x_embed_hat[:-1]]) ---
    gate_sig = torch.sigmoid(x_embed_hat[1:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)  # (T-1, 1), recomputed
    gate = bf16(m.smear_lambda.w) * gate_sig
    xb_embed_hat = x0b.clone()
    xb_embed_hat[:-1] += gate * keep * x0b[1:]  # TRAP: shifted scatter -- p's grad reaches p-1
    gateb = keep * (x0b[1:] * x_embed_hat[:-1]).sum(dim=-1, keepdim=True)   # (T-1, 1)
    m.smear_lambda.grad.copy_(sum32(gateb * gate_sig))
    gateb_z = gateb * bf16(m.smear_lambda.w) * gate_sig * (1 - gate_sig)
    m.smear_gate.grad.copy_((gateb_z.mT @ x_embed_hat[1:, :cfg.d_smr_gate]).float())
    xb_embed_hat[1:, :cfg.d_smr_gate] += gateb_z @ bf16(m.smear_gate.w)

    # --- embedding norm + token embedding scatter ---
    xb_embed = bf16(x_embed_inv_rms * (xb_embed_hat.float() - (x_embed_hat.float() * (x_embed_hat.float() * xb_embed_hat.float()).mean(dim=-1, keepdim=True))))
    m.input_embeds.grad.copy_(
        torch.ops.aten.embedding_dense_backward(xb_embed, idx, cfg.d_vocab, -1, False))

    # Sparse update: row-wise RMSProp on the bigram table's touched rows (bigram_rows).
    bigram_grad = torch.ops.aten.embedding_dense_backward(xb_bigram.float(), bigram_inverse, cfg.touched_rows, -1, False)
    m.bigram_embeds.scnd_mntm.mul_(m.bigram_embeds.mntm_b2_t[t_step])
    bigram_mntm = (m.bigram_embeds.scnd_mntm.index_select(0, bigram_rows)
                   + bigram_grad.square().mean(dim=-1, keepdim=True) * m.bigram_embeds.grad_b2_t[t_step])
    m.bigram_embeds.scnd_mntm.index_copy_(0, bigram_rows, bigram_mntm)
    m.bigram_embeds.w.index_copy_(0, bigram_rows, bf16(
        m.bigram_embeds.w.index_select(0, bigram_rows).float()
        + (m.bigram_embeds.lr_bc_t[t_step] / bigram_w_scale_next)
          * (bigram_grad / (bigram_mntm.sqrt() + m.bigram_embeds.eps_t[t_step]))))

    pair_w_scale.copy_(pair_w_scale_next)
    bigram_w_scale.copy_(bigram_w_scale_next)

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
# §§ AdamW
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
# §§ Muon
# ------------------------------------------------------------------------------

# Polar Express orthogonalization coefficients, 5 iterations.
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Not autotuned: autotuning a square 768x768 bank's bmm at a batch size the shipped
# cache lacks raises inductor's "launcher() got multiple values for argument 'stream'"
# (torch 2.10 / triton 3.6). Costs ~0.3% of the step.
@torch.compile(dynamic=False, fullgraph=True, options={"max_autotune_gemm": False})
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
# §§ LR Schedule
# ------------------------------------------------------------------------------

# Learning rate schedule as a per-step multiplier.
# Shared by Muon, AdamW and RMSProp. No warmup: peak from step 0.
lr_mult_t = np.ones(cfg.num_steps)

steps_0idx = np.arange(cfg.num_steps, dtype=np.float64)  # 0-based, the way the loop counts
steps_1idx = steps_0idx + 1.0                            # 1-based, the way bias corrections count

# Warmdown for 65% of the run.
warmdown_len  = round(0.65 * cfg.num_steps)
warmdown      = slice(cfg.num_steps - warmdown_len + 1, cfg.num_steps)   # the hold covers everything before
warmdown_frac = (cfg.num_steps - steps_0idx[warmdown]) / warmdown_len  # ~1 -> ~0 across the warmdown

lr_mult_t[warmdown] = 0.05 + (1.0 - 0.05) * warmdown_frac

# Muon warms down on its own schedule: the last 95% of the run, to 0.025.
muon_warmdown_len = round(0.95 * cfg.num_steps)
muon_warmdown     = slice(cfg.num_steps - muon_warmdown_len + 1, cfg.num_steps)
muon_lr_mult_t    = np.ones(cfg.num_steps)
muon_lr_mult_t[muon_warmdown] = 0.025 + (1.0 - 0.025) * (cfg.num_steps - steps_0idx[muon_warmdown]) / muon_warmdown_len

# Stage 2 trains from a checkpoint at a fraction of the pretraining rates (weight decay
# follows, being lr-coupled); 1.0 in stage 1.
lr_mult_t *= cfg.lr_scale
muon_lr_mult_t *= cfg.lr_scale


# ------------------------------------------------------------------------------
# §§ Common
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
# §§ Scalars
# ------------------------------------------------------------------------------

resid_lambdas  = torch.linspace(1.15, 1.05, cfg.n_layers, dtype=torch.float32, device=device)
x0_lambdas     = torch.linspace(0.20, 0.05, cfg.n_layers, dtype=torch.float32, device=device)
x0_gates       = fp32_zeros(cfg.n_layers)
bigram_lambdas = fp32_empty(cfg.n_layers).fill_(0.1)
smear_lambda   = fp32_zeros(1)
backout_lambda = fp32_empty(1).fill_(0.2)

smear_gate     = fp32_empty(1, cfg.d_smr_gate).uniform_(-cfg.d_smr_gate ** -0.5, cfg.d_smr_gate ** -0.5)

# Grad mix-in coefficient (1-beta), currently fixed, can be scheduled here.
scalar_grad_mult_t = np.ones(cfg.num_steps)


scalar_configs = [
#   name,               weights,       peak lr,  b1_grad,   b2_grad,    wd,
    ("resid_lambdas",   resid_lambdas,   0.005,    0.2,        0.05,   0.05),
    ("x0_lambdas",      x0_lambdas,      0.1,      0.04,       0.05,   0.0),
    ("x0_gates",        x0_gates,        0.1,      0.04,       0.05,   0.0),
    ("bigram_lambdas",  bigram_lambdas,  0.1,      0.04,       0.05,   0.0),
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
# §§ Embeddings
# ------------------------------------------------------------------------------

input_embeds =     bf16_empty(cfg.d_vocab, cfg.d_model)  # + the token frequency prior, below
input_embeds.copy_(fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.0))
value_embeds =     bf16_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
value_embeds.copy_(fp32_empty(cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo)
                   .uniform_(-matrix_init_s, matrix_init_s))

bigram_embeds = bf16_zeros(cfg.d_bigram + 1, cfg.d_model)                                     # + a scratch row
pair_values   = bf16_zeros(cfg.num_ves * (cfg.d_pair_values + 1), cfg.n_kv_heads * cfg.d_vo)  # + one per slot

ve_rows = cfg.num_ves * cfg.d_vocab

embed_configs = [
#   name,             weights,       peak lr,  b1_grad,  b2_grad,  wd,       slots
    ("input_embeds",  input_embeds,  0.1837,   0.2,      0.005,    0.001,   1),
    ("value_embeds",  value_embeds,  0.0919,   0.2,      0.005,    0.01,   cfg.num_ves)
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
# §§ Bigram Table
# ------------------------------------------------------------------------------

peak_lr = 0.0184
b2_grad = 0.005   # (1-Beta2)
wd      = 0.001

# Derive the momentum buffer's decay.
b2_mntm = 1 - b2_grad

m.bigram_embeds = Param(
    # Weight
    name         = "bigram_embeds",
    w            = bigram_embeds,      # bf16 live, no fp32 master
    mantissa     = None,

    # Gradients -- none: the update is fused into forward_backward
    grad         = None,
    gbank        = None,

    # Momentum buffers
    first_mntm   = None,
    scnd_mntm    = fp32_zeros(cfg.d_bigram + 1, 1),

    residual_dim = None, # Muon only

    # Schedules
    # Fold bias correction into the learning rate.
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5),

    # Weight decay schedule
    wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

    mntm_b1_t    = None,
    grad_b1_t    = None,

    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(np.full(cfg.num_steps, b2_grad)),   # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)

# The weight decay the rows have not had applied: the live weight is the row times this.
bigram_w_scale = fp32_empty(1).fill_(1.0)


# ------------------------------------------------------------------------------
# §§ Pair Value Memory
# ------------------------------------------------------------------------------

peak_lr = 0.0736
b2_grad = 0.005   # (1-Beta2)
wd      = 0.001

b2_mntm = 1 - b2_grad

m.pair_values = Param(
    # Weight
    name         = "pair_values",
    w            = pair_values,      # bf16 live, no fp32 master
    mantissa     = None,

    # Gradients -- none: the update is fused into forward_backward
    grad         = None,
    gbank        = None,

    # Momentum buffers
    first_mntm   = None,
    scnd_mntm    = fp32_zeros(cfg.num_ves * (cfg.d_pair_values + 1), 1),

    residual_dim = None, # Muon only

    # Schedules
    # Fold bias correction into the learning rate.
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5),

    # Weight decay schedule
    wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

    mntm_b1_t    = None,
    grad_b1_t    = None,

    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(np.full(cfg.num_steps, b2_grad)),   # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)

# The table's five slots as standalone tensors, so each update writes a whole one.
pair_wbank = list(pair_values.view(cfg.num_ves, cfg.d_pair_values + 1, -1).unbind(0))
pair_vbank = list(m.pair_values.scnd_mntm.view(cfg.num_ves, cfg.d_pair_values + 1, 1).unbind(0))

# The weight decay the rows have not had applied: the live weight is the row times this.
pair_w_scale = fp32_empty(1).fill_(1.0)


# ------------------------------------------------------------------------------
# §§ LM Head
# ------------------------------------------------------------------------------

lm_head = fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.001)  # + the token frequency prior, below

# Per-step multiplier on the head's grad mix-ins (1-beta). Constant: no warmup ramp.
lm_grad_mult_t = np.ones(cfg.num_steps)

peak_lr = 0.0049
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
# §§ Decision Readout
# ------------------------------------------------------------------------------
# One vector, dotted with each option's readout token: fp32 live and AdamW, with
# the head's betas and no weight decay. Starts small, like the head.

peak_lr = 0.0049
b1_grad = 0.2    # (1-Beta1)
b2_grad = 0.04   # (1-Beta2)
b1_mntm = 1 - b1_grad
b2_mntm = 1 - b2_grad

readout = fp32_empty(1, cfg.d_model).normal_(mean=0.0, std=0.001)
m.readout = Param(
    name         = "readout",
    w            = readout,      # fp32 live
    mantissa     = None,
    grad         = fp32_zeros(readout.shape),
    gbank        = None,
    first_mntm   = fp32_zeros(readout.shape),
    scnd_mntm    = fp32_zeros(readout.shape),
    residual_dim = None, # Muon only
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),
    wd_t         = dev(np.ones(cfg.num_steps)),
    mntm_b1_t    = dev(np.full(cfg.num_steps, b1_mntm)),
    grad_b1_t    = dev(np.full(cfg.num_steps, b1_grad)),
    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),
    grad_b2_t    = dev(np.full(cfg.num_steps, b2_grad)),
    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)
del readout


# ------------------------------------------------------------------------------
# §§ Attention & MLPs
# ------------------------------------------------------------------------------

W_Q =   fp32_empty(cfg.n_layers, cfg.n_qo_heads * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_K =   fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_qk, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_V =   fp32_empty(cfg.n_layers, cfg.n_kv_heads * cfg.d_vo, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s)
W_O =   fp32_zeros(cfg.n_layers,               cfg.d_model, cfg.n_qo_heads * cfg.d_vo)  # projections start at zero

ve_gate = fp32_empty(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate).uniform_(0.0, 0.02)
pair_gate = fp32_empty(cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate).uniform_(
    0.0, 0.02, generator=torch.Generator(device=device).manual_seed(cfg.seed + 2))
pair_decoder = fp32_zeros(1, cfg.n_kv_heads * cfg.d_vo, cfg.d_model)  # starts at zero

# One bank per distinct MLP width, holding that width's layers in order.
mlp_widths = sorted(set(cfg.d_mlp))
W_in  = {d: fp32_empty(cfg.d_mlp.count(d), d, cfg.d_model).uniform_(-matrix_init_s * 0.4, matrix_init_s * 0.4) for d in mlp_widths}
W_out = {d: fp32_zeros(cfg.d_mlp.count(d), cfg.d_model, d) for d in mlp_widths}   # projections start at zero

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

# Muon peak lr is 0.02, and 0.04 on W_in at every width. rdim is the
# axis facing the residual stream: W_O and W_out live transposed -> -2; the
# ve_gate rows read a d_ve_gate slice of the stream -> -1, pair_decoder the pair code -> -1.
muon_configs = [
#    name,           weights,       peak lr,  rdim
    ("W_Q",          W_Q,           0.02,      -1),
    ("W_K",          W_K,           0.02,      -1),
    ("W_V",          W_V,           0.02,      -1),
    ("W_O",          W_O,           0.02,      -2),
    *[(f"W_in_{d}",  W_in[d],       0.04,      -1) for d in mlp_widths],
    *[(f"W_out_{d}", W_out[d],      0.02,      -2) for d in mlp_widths],
    ("ve_gate",      ve_gate,       0.02,      -1),
    ("pair_gate",    pair_gate,     0.02,      -1),
    ("pair_decoder", pair_decoder,  0.02,      -1)
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
        lr_bc_t      = dev(muon_lr_mult_t * peak_lr),
        wd_t         = dev(muon_lr_mult_t * peak_lr * muon_wd),

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
del lm_head, input_embeds, value_embeds, bigram_embeds, pair_values, resid_lambdas, x0_lambdas, x0_gates, bigram_lambdas, smear_gate, smear_lambda, backout_lambda, W_Q, W_K, W_V, W_O, ve_gate, pair_gate, pair_decoder, W_in, W_out

# Each layer's MLP banks and its slot in them.
mlp_banks = [(getattr(m, f"W_in_{d}"), getattr(m, f"W_out_{d}"), cfg.d_mlp[:i].count(d)) for i, d in enumerate(cfg.d_mlp)]
mlp_params = [getattr(m, f"W_{io}_{d}") for io in ("in", "out") for d in mlp_widths]


# ------------------------------------------------------------------------------
# §§ Token Frequency Priors
# ------------------------------------------------------------------------------

torch.cuda.synchronize()
priors_t0 = time.perf_counter()

# Corpus bigram counts (all 91 train shards): bigram_counts[i, j] = how often token j follows token i.
bigram = np.load(os.path.join(DATASET_DIR, "tokenizer/bigram_counts.npz"))
# The priors cover the BPE vocabulary; the decision special tokens get their own rows below.
bigram_counts = torch.zeros(cfg.bpe_vocab, cfg.bpe_vocab, dtype=torch.float32, device=device)
bigram_counts[torch.from_numpy(np.repeat(np.arange(cfg.bpe_vocab), np.diff(bigram["train_indptr"]))).to(device),
              torch.from_numpy(bigram["train_indices"].astype(np.int64)).to(device)] = \
    torch.from_numpy(bigram["train_data"].astype(np.float32)).to(device)
del bigram

context_counts = bigram_counts.sum(dim=1, keepdim=True)                                            # (V, 1)
next_unigram   = (bigram_counts.sum(dim=0) + 0.5) / (bigram_counts.sum() + 0.5 * cfg.bpe_vocab)   # (V,)

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
torch.manual_seed(cfg.seed + 1)   # the randomized SVD's test matrix
U, S, _ = torch.svd_lowrank(context_weight * raw_target, q=cfg.d_model + 64, niter=4)
embed_prior = U[:, :cfg.d_model] * S[:cfg.d_model].sqrt()                                          # (V, D)
del U, S

# The RMS norm keeps only each embedding row's direction: put every row at the stock norm
# 0.8 * sqrt(D), then solve the head by weighted least squares against the normed rows.
embed_prior *= 0.8 * cfg.d_model ** 0.5 / embed_prior.norm(dim=1, keepdim=True)
xe_prior   = embed_prior / 0.8                                                                     # normed rows
head_prior = torch.linalg.solve((xe_prior.T @ (context_weight ** 2 * xe_prior)).double(),
                                ((context_weight ** 2 * xe_prior).T @ raw_target).double()).float().T   # (V, D)
del raw_target, xe_prior, context_weight

# Halve the head's log-frequency component.
occurrences_per_step = context_counts / context_counts.sum() * cfg.train_batch_tokens             # (V, 1)
freq_weight = occurrences_per_step.clamp_min(1e-3)
log_freq = (occurrences_per_step + 1e-3).log()
log_freq -= (freq_weight * log_freq).sum() / freq_weight.sum()                                    # centred
head_mean = (freq_weight * head_prior).sum(0) / freq_weight.sum()
freq_direction = ((freq_weight * log_freq * (head_prior - head_mean)).sum(0)
                  / (freq_weight * log_freq.square()).sum())                                      # (D,)
head_prior -= 0.5 * log_freq * freq_direction
del context_counts, occurrences_per_step, freq_weight, log_freq, head_mean, freq_direction

# [MASK] reads the unigram through the solved head: the normed row x (norm sqrt(D), what the
# RMS norm hands the head) with head_prior @ x closest to the centred log unigram, pre-softcap,
# each vocab row weighted by its probability (the cross-entropy's own weighting). A random row
# here starts MLM above uniform (the head reads bigram-scale logits off a random direction).
mask_target = next_unigram.log()
mask_target = 15.0 * torch.atanh((mask_target - mask_target.mean()).clamp(-14.25, 14.25) / 15.0)   # (V_bpe,)
mask_w = next_unigram.sqrt().unsqueeze(1)                                                         # (V_bpe, 1)
mask_row = torch.linalg.lstsq((mask_w * head_prior).double(), (mask_w * mask_target.unsqueeze(1)).double()).solution.float()[:, 0]
mask_row *= cfg.d_model ** 0.5 / mask_row.norm()
mask_logits = 15.0 * torch.tanh(head_prior @ mask_row / 15.0)
mask_ce = -(next_unigram * mask_logits.log_softmax(0)).sum()
print(f"[MASK] row: direct-path CE against the unigram {mask_ce:.4f} nats "
      f"(unigram entropy {-(next_unigram * next_unigram.log()).sum():.4f}; uniform {math.log(cfg.d_vocab):.4f})")

# Add the priors to the initialized weights' BPE rows.
m.input_embeds.w[:cfg.bpe_vocab].add_(embed_prior)
head_master = rebuild_master(m.lm_head.w, m.lm_head.mantissa)
head_master[:cfg.bpe_vocab].add_(head_prior)
writeback_master(head_master, m.lm_head.w, m.lm_head.mantissa)
del embed_prior, head_prior, head_master

# The decision special tokens have no corpus statistics: random directions, at
# the prior rows' norm 0.8 * sqrt(D). (Their lm_head rows keep the small random
# init; MLM never targets them.)
special_embeds = fp32_empty(cfg.d_vocab - cfg.bpe_vocab, cfg.d_model).normal_()
m.input_embeds.w[cfg.bpe_vocab:].copy_(special_embeds * (0.8 * cfg.d_model ** 0.5 / special_embeds.norm(dim=1, keepdim=True)))
m.input_embeds.w[MASK_ID].copy_(0.8 * mask_row)   # the solved direction, at the prior rows' norm
del special_embeds, mask_target, mask_w, mask_row, mask_logits, mask_ce, next_unigram

torch.cuda.synchronize()
priors_time = time.perf_counter() - priors_t0


# ------------------------------------------------------------------------------
# §§ Pair Code
# ------------------------------------------------------------------------------

torch.cuda.synchronize()
pair_code_t0 = time.perf_counter()

# Corpus trigram statistics (all 91 train shards): each frequent [prev, curr] pair's 64 next tokens with the largest
# log1p(count / (1,000 * bigram probability)), and those log-ratios.
pair_next = np.load(os.path.join(DATASET_DIR, "tokenizer/pair_next_tokens.npz"))
pair_code_keys = pair_next["keys"]  # prev * d_vocab + curr, ascending
pair_next_ids, pair_log_ratios, pair_counts = pair_next["next_ids"], pair_next["log_ratios"], pair_next["counts"]
del pair_next

# Each pair's code: the log-ratio-weighted sum of the head's centred rows.
centred_head = rebuild_master(m.lm_head.w, m.lm_head.mantissa)[:cfg.bpe_vocab]
centred_head -= centred_head.mean(dim=0, keepdim=True)
pair_code = fp32_zeros(cfg.d_pair_code + 1, cfg.d_model)
for start in range(0, cfg.d_pair_code, 16384):
    next_ids   = torch.from_numpy(pair_next_ids[start:start + 16384].astype(np.int64)).to(device)
    log_ratios = torch.from_numpy(pair_log_ratios[start:start + 16384].astype(np.float32)).to(device)
    pair_code[start:start + 16384] = (log_ratios.unsqueeze(-1) * centred_head[next_ids]).sum(dim=1)

# Centre and whiten, weighted by each pair's share of positions.
position_share = torch.from_numpy(pair_counts / pair_counts.sum()).float().unsqueeze(1).to(device)
pair_code[:-1] -= (pair_code[:-1] * position_share).sum(dim=0, keepdim=True)
covariance = torch.zeros(cfg.d_model, cfg.d_model, dtype=torch.float64, device=device)
for start in range(0, cfg.d_pair_code, 65536):
    code_rows = pair_code[start:start + 65536].double()
    covariance += code_rows.T @ (code_rows * position_share[start:start + 65536].double())
eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
whitening = (eigenvectors * (eigenvalues.clamp_min(0.0) + 1e-3 * eigenvalues.mean()).rsqrt()) @ eigenvectors.T
pair_code[:-1] = pair_code[:-1] @ whitening.float()
m.pair_code = pair_code.bfloat16()
del pair_next_ids, pair_log_ratios, pair_counts, centred_head, pair_code, next_ids, log_ratios, position_share
del covariance, code_rows, eigenvalues, eigenvectors, whitening

torch.cuda.synchronize()
pair_code_time = time.perf_counter() - pair_code_t0


# ------------------------------------------------------------------------------
# §§ Rotary Cache
# ------------------------------------------------------------------------------

rotary_seq_len = cfg.train_batch_tokens
channel_range = torch.arange(0, cfg.d_qk, 2, dtype=torch.float32, device=device)  # stride the channels
inv_freq = 1.0 / (100000 ** (channel_range / cfg.d_qk))
t_pos = torch.arange(rotary_seq_len, dtype=torch.float32, device=device)          # stride the time steps
freqs = torch.outer(t_pos, inv_freq)   # rotation frequency at each (time, channel) pair

m.cos = freqs.cos().to(torch.bfloat16)[None, :, None, :]  # add batch and head dims
m.sin = freqs.sin().to(torch.bfloat16)[None, :, None, :]  # for later broadcasting

del channel_range, inv_freq, t_pos, freqs

cfg.num_params = sum(p.w.numel() for p in m)


# ------------------------------------------------------------------------------
# §§ Init From a Checkpoint
# ------------------------------------------------------------------------------
# Stage 2 starts from stage 1's weights (write_checkpoint, §§ Logging): every trained
# weight and the frozen pair code are overwritten; the optimizer state stays fresh and the
# schedules are this run's. The hashed tables were saved with their pending weight decay
# folded in, so their scales start at 1 here as at init.

if cfg.init_from:
    saved = torch.load(os.path.expanduser(cfg.init_from), map_location=device)
    for p in m:
        w = saved["weights"][p.name]
        assert w.shape == p.w.shape, (p.name, tuple(w.shape), tuple(p.w.shape))
        if p.mantissa is not None:
            writeback_master(w.contiguous(), p.w, p.mantissa)   # the fp32 master -> bf16 live + low bits
        else:
            p.w.copy_(w)
    m.pair_code.copy_(saved["weights"]["pair_code"])
    print(f"init from {cfg.init_from}: run {saved['run_name']} at step {saved['step']}")
    del saved, w


# ==============================================================================
# § Data
# ==============================================================================
# Packs (utils.py § Packs): decision records in a decision_frac share of each
# pack, ClimbMix filler trained on MLM in the rest. Every pack is planned and laid
# out up front, in pinned memory. The static sizes forward_backward compiles
# against come from the plans, training's and evaluation's together.

def table_rows(tokens, seg_start):
    """Each position's [prev, curr] pair as a row of the three pair-indexed tables, plus the two
    trained tables' compacted rows, on the host (see materialize_packs). A segment's first token
    has no previous token: its pair is (BOS, curr). Only pairs of BPE tokens have a corpus code."""
    prev = tokens[:-1].copy()
    prev[seg_start[1:]] = BOS_ID
    curr = tokens[1:]
    pair     = np.bitwise_xor(36313 * curr, 27191 * prev)
    pair_key = cfg.bpe_vocab * prev + curr
    code_row = np.minimum(np.searchsorted(pair_code_keys, pair_key), cfg.d_pair_code - 1)
    coded    = (pair_code_keys[code_row] == pair_key) & (prev < cfg.bpe_vocab) & (curr < cfg.bpe_vocab)
    rows = np.empty((7, tokens.size), dtype=tokens.dtype)
    rows[0, :1], rows[0, 1:] = cfg.d_bigram - 1,      pair % (cfg.d_bigram - 1)
    rows[1, :1], rows[1, 1:] = cfg.d_pair_values - 1, pair % (cfg.d_pair_values - 1)
    rows[2, :1], rows[2, 1:] = cfg.d_pair_code,       np.where(coded, code_row, cfg.d_pair_code)
    # Compact each trained table's rows; padding points at the table's scratch row.
    bigram_touched, rows[3] = np.unique(rows[0], return_inverse=True)
    pair_touched,   rows[5] = np.unique(rows[1], return_inverse=True)
    assert max(bigram_touched.size, pair_touched.size) <= cfg.touched_rows
    rows[4], rows[4, :bigram_touched.size] = cfg.d_bigram,      bigram_touched
    rows[6], rows[6, :pair_touched.size]   = cfg.d_pair_values, pair_touched
    return rows

packs_t0 = time.perf_counter()
rng = np.random.default_rng(cfg.seed)

# Evaluation: every record of the two development splits once, as stored, with
# filler from the val shard. Training: shuffled epochs of Kev's training records,
# a choice question's options permuted each time.
val_docs = doc_stream("val", BOS_ID)
# The evaluation packs take a fixed 10% decision share whatever the training share, so a
# checkpoint scores the same in the run that wrote it and in the one that starts from it.
eval_plans = {split: plan_packs(load_records(split), val_docs, cfg.train_batch_tokens,
                                0.10, cfg.seq_len, None, rng)
              for split in ("decision_dev", "transfer_dev")}
train_records = load_records("train")
train_plans = plan_packs(train_records, doc_stream("train", BOS_ID), cfg.train_batch_tokens,
                         cfg.decision_frac, cfg.seq_len, cfg.num_steps + 1, rng)

caps = pack_caps(train_plans + eval_plans["decision_dev"] + eval_plans["transfer_dev"])
round_up = lambda x, k: k * math.ceil(x / k)
# Masked positions per pack, from the pack with the least filler -- of the training packs
# for training, and of the evaluation packs for evaluation, so the validation masks are the
# same in every run whatever its training share (53,056 at the 0.30 rate).
cfg.mlm_per_pack   = 64 * int(cfg.mask_rate * pack_caps(train_plans)["min_filler"] / 64)
eval_mlm_per_pack  = 64 * int(cfg.mask_rate * pack_caps(eval_plans["decision_dev"] + eval_plans["transfer_dev"])["min_filler"] / 64)
cfg.cu_width       = round_up(caps["segments"] + 2, 64)
cfg.rec_width      = round_up(caps["records"] + 1, 64)
cfg.ro_cap         = round_up(caps["readouts"], 64)
cfg.q_cap          = round_up(caps["questions"], 64)
cfg.max_seg_len    = round_up(caps["seg_len"], 128)
cfg.max_qblock_len = round_up(caps["qblock_len"], 128)
cfg.max_state_len  = round_up(caps["state_len"], 128)

# The evaluation masks are the same every run.
eval_packs = {split: materialize_packs(plans, cfg.train_batch_tokens, cfg, table_rows, MASK_ID,
                                       np.random.default_rng(cfg.seed + 3), eval_mlm_per_pack)
              for split, plans in eval_plans.items()}
train_packs, _ = materialize_packs(train_plans, cfg.train_batch_tokens, cfg, table_rows, MASK_ID, rng, cfg.mlm_per_pack)

placed = sum(len(chosen) for chosen, _ in train_plans)
dec_tokens = cfg.train_batch_tokens * len(train_plans) - sum(d.size for _, filler in train_plans for d in filler)
data_report = (
    f"Packs: {len(train_plans):,} train ({placed:,} records = {placed / len(train_records):.2f} epochs of "
    f"{len(train_records):,}; decision tokens {dec_tokens / 1e6:.1f}M = "
    f"{100 * dec_tokens / (cfg.train_batch_tokens * len(train_plans)):.1f}%), "
    f"eval {', '.join(f'{s} {len(p)}' for s, p in eval_plans.items())}\n"
    f"  per pack: {cfg.mlm_per_pack:,} masked ({eval_mlm_per_pack:,} in evaluation) | caps: segments {cfg.cu_width}, records {cfg.rec_width}, "
    f"readouts {cfg.ro_cap}, questions {cfg.q_cap} | max_seqlen: segment {cfg.max_seg_len}, "
    f"question block {cfg.max_qblock_len}, state {cfg.max_state_len}")
del eval_plans, train_plans, train_records, val_docs
packs_time = time.perf_counter() - packs_t0

# A rough MFU denominator: 6 x matmul params per token in the body, the head on the
# masked positions only, and decoder-rtx's causal attention term doubled for
# bidirectional windows.
body_params = sum(p.w.numel() for p in (m.W_Q, m.W_K, m.W_V, m.W_O, m.pair_decoder, *mlp_params))
cfg.num_flops_per_token = int(6 * body_params + 2 * 92_012_544
                              + 6 * cfg.d_vocab * cfg.d_model * cfg.mlm_per_pack / cfg.train_batch_tokens)


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
    loss:         float | None = None   # MLM, nats per masked token
    dec_loss:     float | None = None   # decision, nats per question
    lr_mult_t:    float | None = None
    dt:           float | None = None   # seconds; unset for steps 0-10
    tok_per_sec:  int   | None = None   # "
    mfu:          float | None = None   # "

    # Validation, on val steps only: MLM nats on the val filler, and accuracy and
    # Brier score (sum over options) on the clean rows of each decision split
    mlm:          float | None = None
    dec_acc:      float | None = None   # decision-v7 development
    dec_brier:    float | None = None
    xfer_acc:     float | None = None   # transfer-v4 development (Kev-0.8B: 0.648)
    xfer_brier:   float | None = None
    eval_seconds: float | None = None

    # Run clocks, in minutes; stamped by log_step
    train_total:  float | None = None
    wall_total:   float | None = None
    eta:          float | None = None


# Field -> wandb panel. A field not named here lands under `train/`.
WANDB_GROUPS = {
    "val":  ("mlm", "dec_acc", "dec_brier", "xfer_acc", "xfer_brier", "eval_seconds"),
    "time": ("train_total", "wall_total", "eta"),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(StepStats)}, \
    "WANDB_GROUPS names a field StepStats does not have"


@dataclass
class RunResult:
    """The run in one row: the results panel, the result JSON, the runs table."""

    # Identity, for the JSON; wandb has these from the run name and config
    run_name:           str          = ""
    num_steps:          int          = 0
    train_batch_tokens: int          = 0

    # The result, from the final validation
    val_mlm:            float | None = None
    dec_acc:            float | None = None
    dec_brier:          float | None = None
    xfer_acc:           float | None = None
    xfer_brier:         float | None = None

    # The cost, in minutes
    train_time:         float        = 0.0
    val_time:           float        = 0.0
    compile_time:       float        = 0.0
    init_time:          float        = 0.0
    wall_time:          float        = 0.0

    # The rate
    avg_step_time:      float        = 0.0    # seconds
    avg_mfu:            float        = 0.0    # percent of BF16 peak
    peak_mem_gb:        float        = 0.0


FINAL_IDENTITY = ("run_name", "num_steps", "train_batch_tokens")


# ------------------------------------------------------------------------------
# §§ Logging
# ------------------------------------------------------------------------------

# Checkpoints go to the cache; a run worth keeping is pushed to the hub on purpose.
runs_dir = os.path.expanduser("~/.cache/encoder-rtx/runs")

def write_checkpoint(step):
    """The model as a successor run starts from it (§§ Init From a Checkpoint): every
    trained weight -- the fp32 master where one exists, the hashed tables with their pending
    weight decay folded in -- plus the frozen pair code and this script, in model_step*.pt;
    the optimizer moments in optim_step*.pt."""
    fold = lambda w, scale: torch.cat([(rows.float() * scale).bfloat16() for rows in w.split(1 << 18)]).cpu()
    weights = {}
    for p in m:
        if p.mantissa is not None:
            weights[p.name] = rebuild_master(p.w, p.mantissa).cpu()
        elif p.name == "bigram_embeds":
            weights[p.name] = fold(p.w, bigram_w_scale)
        elif p.name == "pair_values":
            weights[p.name] = fold(p.w, pair_w_scale)
        else:
            weights[p.name] = p.w.cpu()
    weights["pair_code"] = m.pair_code.cpu()
    state = {}
    for p in m:
        for attr in ("first_mntm", "scnd_mntm"):
            buf = getattr(p, attr)
            if buf is not None:
                state[f"{p.name}.{attr}"] = buf.cpu()
    run_dir = os.path.join(runs_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(dict(step=step, run_name=cfg.run_name, code=code, weights=weights), f"{run_dir}/model_step{step:06d}.pt")
    torch.save(dict(step=step, t_step=int(t_step.item()), state=state), f"{run_dir}/optim_step{step:06d}.pt")
    return run_dir

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

print0(f"Model parameters: {cfg.num_params:,} | FLOPs/token (estimate): {cfg.num_flops_per_token:e}", console=True)
print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}", console=True)
print0(f"Batch size: {cfg.train_batch_tokens:,} tokens per step", console=True)
print0(f"Knobs: decision_frac {cfg.decision_frac} | decision_weight {cfg.decision_weight} | mask_rate {cfg.mask_rate}", console=True)
print0(f"Init: token frequency priors {priors_time:.1f}s | pair code {pair_code_time:.1f}s | packs {packs_time:.1f}s", console=True)
print0(data_report, console=True)

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

smooth_train_loss = np.zeros(2)   # (MLM, decision)
final_val = None  # the last validation's StepStats
total_val_time = 0.0
timed = []   # Length of each step in seconds, steps 0-10 excluded.
             # Total training time = np.sum(timed).
warmup = []  # Those first 11 steps, where the compile lives.


def validate():
    """MLM nats per masked token on the val filler, and, on each decision split's
    clean rows, (accuracy, Brier score summed over options, questions, accuracy
    by source), from the eval packs."""
    mlm_nats, mlm_n, result = 0.0, 0, {}
    for split, (packs, scored) in eval_packs.items():
        right = total = 0
        brier = 0.0
        by_source = {}
        for i, pk in enumerate(pack_loader(packs)):
            nats, ro_p = forward_backward(pk, backward=False)
            mlm_nats += nats.sum().item()
            mlm_n += nats.numel()
            ro_p = ro_p.cpu().numpy()
            k = 0
            for r, q, perm in scored[i]:   # the pack's questions, in readout order
                n = len(perm)
                probs, k = ro_p[k:k + n], k + n
                if r["variant"] != "clean":
                    continue
                label = int(np.flatnonzero(perm == q["label"])[0])
                hit = int(probs.argmax() == label)
                right += hit
                total += 1
                brier += float(((probs - np.eye(n)[label]) ** 2).sum())
                source = by_source.setdefault(r["source"], [0, 0])
                source[0] += hit
                source[1] += 1
        result[split] = (right / total, brier / total, total, by_source)
    return mlm_nats / max(mlm_n, 1), result


train_loader = pack_loader(train_packs)
pk = next(train_loader)   # kick off the first batch

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
        val_mlm, dec = validate()
        val_elapsed = time.perf_counter() - val_t0
        total_val_time += val_elapsed
        stats.mlm, stats.eval_seconds = val_mlm, val_elapsed
        stats.dec_acc, stats.dec_brier, n_dec, _ = dec["decision_dev"]
        stats.xfer_acc, stats.xfer_brier, n_xfer, _ = dec["transfer_dev"]
        print0(f"step:{step}/{cfg.num_steps} val_mlm:{val_mlm:.4f} | decision_dev acc:{stats.dec_acc:.4f} "
               f"brier:{stats.dec_brier:.4f} (n={n_dec}) | transfer_dev acc:{stats.xfer_acc:.4f} "
               f"brier:{stats.xfer_brier:.4f} (n={n_xfer}) | val_time:{val_elapsed:.2f}s", console=True)
        if last_step:
            for split, (*_, by_source) in dec.items():
                print0(f"  {split} accuracy by source: " + ", ".join(
                    f"{s} {h}/{n}" for s, (h, n) in sorted(by_source.items())), console=True)
        final_val = stats

    # --------------- Checkpoint -----------------
    if cfg.save_checkpoint and (last_step or step in cfg.save_steps):
        ckpt_t0 = time.perf_counter()
        run_dir = write_checkpoint(step)
        print0(f"checkpoint captured at step {step} -> {run_dir} ({time.perf_counter() - ckpt_t0:.1f}s)", console=True)

    # Exit final step after validation and checkpoint
    if last_step:
        log_step(stats)
        break

    # --------------- Training Step -----------------
    torch.cuda.synchronize()
    step_t0 = time.perf_counter()

    # Forward and Backward pass
    loss = forward_backward(pk)

    # Next training batch
    pk = next(train_loader)

    # Smooth gradients, update weights

    # Muon 
    for p in (m.W_Q, m.W_K, m.W_V, m.W_O, *mlp_params, m.ve_gate, m.pair_gate, m.pair_decoder):
        muon_step_fused(p, p.grad, t_step)

    # AdamW
    for p in (m.lm_head, m.readout, m.input_embeds, m.value_embeds, m.resid_lambdas, m.x0_lambdas, \
              m.x0_gates, m.bigram_lambdas, m.smear_gate, m.smear_lambda, m.backout_lambda):
        # Run AdamW
        adamw_step_fused(p, p.grad, t_step)

    t_step.add_(1)  # advance the schedule on-device

    train_loss = loss.cpu().numpy()   # (MLM, decision)
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

    tok_per_sec = int(cfg.train_batch_tokens / dt)
    mfu = 100 * cfg.num_flops_per_token * cfg.train_batch_tokens / dt / gpu_peak_flops

    print0(f"step {step:05d}/{cfg.num_steps:05d} ({pct_done:.2f}%) | mlm: {debiased_smooth_loss[0]:.4f} | decision: {debiased_smooth_loss[1]:.4f} | lr_mult_t: {lr_mult_t[step]:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | total time: {np.sum(timed)/60:.2f}m{eta_str}", console=True)

    stats.loss, stats.dec_loss = float(train_loss[0]), float(train_loss[1])
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
# §§ Results
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
    run_name           = cfg.run_name,
    num_steps          = cfg.num_steps,
    train_batch_tokens = cfg.train_batch_tokens,

    train_time         = float(np.sum(timed)) / 60,
    val_time           = total_val_time / 60,
    compile_time       = compile_time / 60,
    init_time          = (priors_time + pair_code_time + packs_time) / 60,
    wall_time          = (time.perf_counter() - run_wall_t0) / 60,

    avg_step_time      = avg_step_time,
    avg_mfu            = (100 * cfg.num_flops_per_token * cfg.train_batch_tokens
                          / avg_step_time / gpu_peak_flops) if avg_step_time else 0.0,
    peak_mem_gb        = torch.cuda.max_memory_reserved() / 2**30,
)
if final_val is not None:
    result.val_mlm, result.dec_acc, result.dec_brier = final_val.mlm, final_val.dec_acc, final_val.dec_brier
    result.xfer_acc, result.xfer_brier = final_val.xfer_acc, final_val.xfer_brier

row = asdict(result)
with open(result_path, "w") as f:
    json.dump(row, f, indent=1)
wandb_run.log({"step": cfg.num_steps,
               **{f"final/{k}": v for k, v in row.items()
                  if k not in FINAL_IDENTITY and v is not None}})

print0(f"== {result.run_name} ==", console=True)
if result.xfer_acc is not None:
    print0(f"  val_mlm {result.val_mlm:.4f} | decision_dev acc {result.dec_acc:.4f} brier {result.dec_brier:.4f} | "
           f"transfer_dev acc {result.xfer_acc:.4f} brier {result.xfer_brier:.4f} (Kev-0.8B: 0.648)", console=True)
print0(f"  train {result.train_time:.2f}m | val {result.val_time:.2f}m | compile {result.compile_time:.2f}m | init {result.init_time:.2f}m | wall {result.wall_time:.2f}m", console=True)
if timed:
    print0(f"  {cfg.train_batch_tokens:,} tokens/step over {len(timed):,} timed steps: mean {result.avg_step_time:.3f}s | median {np.median(timed):.3f}s | {int(cfg.train_batch_tokens / result.avg_step_time):,} tok/s | mfu {result.avg_mfu:.2f}%", console=True)
print0(f"  rows -> {metrics_path} | result -> {result_path}", console=True)

for _f in (logfile, metrics_path, result_path):
    wandb_run.save(_f, policy="now")
wandb_run.finish()
