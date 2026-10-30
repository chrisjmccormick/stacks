# train_decision.py
#
# The decision stage of encoder-rtx: the MLM base (an encoder pre-trained by masked LM,
# decoder-rtx's baseline-6 made bidirectional) trained on decision records, with MLM on
# ClimbMix filler alongside, as a single forward_backward with a handwritten backward.
#
# What it adds to the MLM base's script (agent-ops-stacks/encoder-rtx/
# 2026-10-29_0740am_b6-mlm-speedrun/train_mlm-28-base4gb.py), from the Phase-0 fork
# (train_encoder.py):
# - Packs of decision records plus ClimbMix filler (utils_decision.py § Packs). Every
#   state, question and filler document is its own segment, and each record's questions
#   also attend to its whole state in a second FA2 call, merged with the first by the
#   softmax LSEs before the per-head output norm (§ Train Forward + Backward).
# - A single readout vector scores each option's readout token, with a softmax over each
#   question's options, trained by cross-entropy against a target distribution: the hard
#   label, or a teacher's soft target mixed with it (soft_weight).
# - The 16K vocabulary plus the decision format's 18 special tokens: the input-side
#   tables grow to 16,448 rows, the LM head stays at 16,384.
# - init_from: the MLM base's checkpoint (weights; fresh optimizer state, this run's step
#   count and schedules), lr_scale on every pre-trained weight's rate.
# - Validation: the base's MLM validation (the same 3.30M corrupted positions), plus
#   accuracy and Brier score on the clean rows of decision-v7 and transfer-v4 development.
#
# Downloads the required number of pre-tokenized Climbmix dataset shards if not
# already present. The decision records come from data/prepare_decision_data.py.
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
# utils_decision.py holds the data loader; log its source too.
with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "utils_decision.py"), 'r') as f:
    code += "\n\n# " + "=" * 78 + "\n# utils_decision.py\n# " + "=" * 78 + "\n\n" + f.read()

import csv
import functools
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

from utils_decision import (DATASET_DIR, DECISION_DIR, EVAL_BUFFER_TOKENS, Pack, doc_stream, download_dataset,
                            load_records, materialize_packs, pack_caps, pack_loader, plan_packs, val_generator,
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

    pool_layer: int = 6   # Its output is added, scaled, onto the final stream.

    # Input
    d_vocab:         int = 16384   # the tokenizer's ids: the shards, the LM head's rows
    d_input_vocab:   int = 16448   # the input-side tables' rows: + the decision format's 18 special tokens at 16,384-16,401
    mask_id:         int = 16369   # the first unused id after BOS (16368); never a target

    # Attention
    n_qo_heads: int = 6
    n_kv_heads: int = 6     # n_qo == n_kv means full multihead attention.
    d_qk:       int = 128   # Attention head size.
    d_vo:       int = 128   # Note: FA2 requires d_qk == d_vo, FA3 does not.

    # Context and Sliding Window Attention
    seq_len:          int = 2048
    short_win_size:   int = 256     # each way: the window is 2 x 256 + 1 wide
    full_ctxt_layers: list[int] = [3, 7]
    window_sizes:     list[tuple[int, int]]  # Derived below.

    # Attention - Input Reuse
    source_layer:     int = 4
    src_layers:       list[int] = [5, 6, 7]

    # Attention - Gates
    d_gate:    int = 16  # Gate input is a 16-dim slice of the layer's attention input.
                         # Each head has its own gate, all with same input.

    # Attention - Value Embeddings
    ve_layers: list[int] = [1, 3, 5, 7]
    ve_index:  list[int] # Derived from ve_layers.
    num_ves:   int

    # Attention - N-gram Value Memories
    # Each memory sits on one layer and has two hash slots, each keyed on the tokens at
    # its offsets from the position (0 is the position itself, -1 the token before it,
    # +1 the one after). The keys hash the corrupted input, so a slot keyed on offset 0
    # reads [MASK] at a masked position.
    d_ngram:        int = 36 * 16384   # rows per hash slot
    ngram_specs:    list = [(1, (-1, 0),     (-1, 0)),
                            (3, (-1, 0),     (-1, 0)),
                            (5, (-1, 0),     (-1, 0)),
                            (7, (-1, 0),     (-1, 0))]   # (layer, slot 0 offsets, slot 1 offsets)
    num_ngrams:     int
    ngram_memories: list[list[tuple[int, int]]]  # Derived: per layer, its (memory, gate slice) pairs.
    touched_rows:   int = 180_000      # Bound on the distinct rows of one hash a batch touches (checked at startup).

    # MLP
    d_mlp:      list[int] = [768 * r for r in (2, 2, 3, 3, 5, 5, 6, 6)]  # per layer; mean 4x = 3072

    # Model stats, for MFU and logging
    num_params:          int                   # every trained weight, counted after init (§ Weight Init & Schedule)
    num_flops_per_token: int = 442_000_000     # 6 * (56.6M trunk + 0.3 * 12.6M head) matmul params + attention both ways

    # ---- Training ----

    # Batch Size
    train_batch_tokens: int = 96 * 2048   # 192K tokens per step, one pack

    # MLM: the corrupted share of each pack's filler (BOS excluded), 80% [MASK],
    # 10% a random token, 10% left as is. The count is fixed so the shapes are static.
    mask_rate:        float = 0.50   # the validation's rate stays 0.30
    train_targets:    int           # Derived: corrupted positions per training pack
    val_targets:      int           # Derived: per validation micro-batch
    eval_targets:     int = 64      # per decision evaluation pack (their MLM goes unread)

    # Packs (§ Data). The static sizes are derived from the planned packs.
    decision_frac:   float = 0.90   # share of each pack given to decision records; the rest is MLM filler
    decision_weight: float = 1.0    # decision loss (nats per question) against MLM loss (nats per masked token)
    soft_weight:     float = 1.0    # where a question has a soft target: its weight against the hard label's
    epochs:          float = 1.0    # passes over the training records; sets num_steps
    train_splits:    tuple = ("train",)  # data/prepare_decision_data.py splits trained on
    state_cap:       int = 2048     # a state's first tokens kept: the MLM base never saw a document past 2,048
    filler_shard:    int = 8        # the filler starts at this train shard: the MLM base read 1-7
    cu_width:        int            # cu_self entries
    rec_width:       int            # cu_q / cu_k entries
    ro_cap:          int            # readout positions per pack
    q_cap:           int            # questions per pack
    max_seg_len:     int            # FA2 max_seqlen bounds: any segment,
    max_qblock_len:  int            # one record's questions,
    max_state_len:   int            # one state

    # Training
    num_steps: int                  # Derived from epochs (§ Data)

    # Two stages: this one starts from the MLM base
    init_from:       str = "~/.cache/encoder-rtx/hub/runs/mlm-base4gb/model_step002605.pt"
    lr_scale:        float = 0.3    # every pre-trained weight's peak rate, times the MLM recipe's

    # Evaluation and logging
    val_loss_every:  int = 25
    val_tokens:      int = 168 * 65536   # 11M tokens per validation pass
    val_steps:       int                 # Derived: val micro-batches per pass.

    # Logging
    wandb_project:   str = "encoderstack_rtx_dev"
    run_name:        str = "decision-base4gb"  # both wandb and log files
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

# Session knobs, from the environment (values logged below). Stripped once settled.
cfg.run_name        = os.environ.get("RUN_NAME", cfg.run_name)
cfg.mask_rate       = float(os.environ.get("MASK_RATE", cfg.mask_rate))
cfg.use_wandb       = bool(int(os.environ.get("USE_WANDB", int(cfg.use_wandb))))
cfg.save_checkpoint = bool(int(os.environ.get("SAVE_CHECKPOINT", int(cfg.save_checkpoint))))
cfg.decision_frac   = float(os.environ.get("DECISION_FRAC", cfg.decision_frac))
cfg.decision_weight = float(os.environ.get("DECISION_WEIGHT", cfg.decision_weight))
cfg.soft_weight     = float(os.environ.get("SOFT_WEIGHT", cfg.soft_weight))
cfg.epochs          = float(os.environ.get("EPOCHS", cfg.epochs))
cfg.train_splits    = tuple(os.environ.get("TRAIN_SPLITS", ",".join(cfg.train_splits)).split(","))
cfg.state_cap       = int(os.environ.get("STATE_CAP", cfg.state_cap))
cfg.init_from       = os.environ.get("INIT_FROM", cfg.init_from)
cfg.lr_scale        = float(os.environ.get("LR_SCALE", cfg.lr_scale))
cfg.val_loss_every  = int(os.environ.get("VAL_EVERY", cfg.val_loss_every))
ABORT_STEP          = int(os.environ["ABORT_STEP"]) if "ABORT_STEP" in os.environ else ABORT_STEP
assert 0.0 < cfg.decision_frac <= 0.9, "a decision share above 0.9 leaves too little filler for the MLM"
print(f"Knobs: run_name {cfg.run_name} | decision_frac {cfg.decision_frac} | decision_weight {cfg.decision_weight} | "
      f"soft_weight {cfg.soft_weight} | epochs {cfg.epochs} | train_splits {','.join(cfg.train_splits)} | "
      f"state_cap {cfg.state_cap} | init_from {cfg.init_from or '-'} | lr_scale {cfg.lr_scale} | "
      f"mask_rate {cfg.mask_rate} | val_every {cfg.val_loss_every} | use_wandb {cfg.use_wandb} | "
      f"ABORT_STEP {ABORT_STEP} | save_checkpoint {cfg.save_checkpoint}")

# Every pre-trained weight's rate: the MLM recipe's (its rates at 0.7x) times this stage's
# lr_scale. The readout, new in this stage, keeps its own (§§ Decision Readout).
LR_SCALE = 0.7 * cfg.lr_scale

if cfg.use_wandb:
    assert "WANDB_API_KEY" in os.environ, "cfg.use_wandb=True but WANDB_API_KEY not set"
    wandb.login(key=os.environ["WANDB_API_KEY"])

# Map layers to VE and n-gram bank slots.
cfg.ve_index = [cfg.ve_layers.index(i) if i in cfg.ve_layers else -1 for i in range(cfg.n_layers)]
cfg.num_ves = len(cfg.ve_layers)
cfg.num_ngrams = len(cfg.ngram_specs)
# A memory's gate reads the gate slice of its order (bigram 1, trigram 2); slice 0 is the head and VE gates'.
cfg.ngram_memories = [[(n, max(len(s0), len(s1)) - 1) for n, (layer, s0, s1) in enumerate(cfg.ngram_specs) if layer == i]
                      for i in range(cfg.n_layers)]

# Per-layer window sizes for sliding window attention, defined as (left, right)
# tuples: tokens attended to each side of the position, within the document.
cfg.window_sizes = [(cfg.short_win_size, cfg.short_win_size)] * cfg.n_layers  # All short, ...
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (-1, -1)                                             # ... then overwrite with full.

cfg.val_steps = cfg.val_tokens // EVAL_BUFFER_TOKENS

# Corrupted positions per validation micro-batch, a multiple of 64 (the training packs' count
# comes from their filler, § Data).
cfg.val_targets   = 64 * int(0.30 * EVAL_BUFFER_TOKENS / 64)   # the validation metric is at 30% whatever the training rate

gpu_device_name = torch.cuda.get_device_name(0)   # "NVIDIA RTX PRO 6000 Blackwell Server Edition"
# Dense BF16 peak FLOPS of the RTX PRO 6000, the MFU denominator.
gpu_peak_flops = 503.8e12

download_dataset()


# Reject a vocab mismatch between the .bin shards and the model.
with open(os.path.join(DATASET_DIR, "config.json")) as f:
    dataset_config = json.load(f)
    assert dataset_config["vocab_size"] == cfg.d_vocab, "dataset vocab != model d_vocab"
    bos_id = dataset_config["bos_id"]
    assert bos_id < cfg.mask_id < cfg.d_vocab
with open(os.path.join(DECISION_DIR, "meta.json")) as f:
    decision_meta = json.load(f)
assert (decision_meta["bos_id"], decision_meta["d_tokenizer"], decision_meta["d_input_vocab"]) == (bos_id, cfg.d_vocab, cfg.d_input_vocab), \
    "decision records' vocabulary != model's"
assert decision_meta["special_tokens"]["<|mask|>"] == cfg.mask_id

# token_bytes: per-token-id byte lengths (0 for special tokens), for the
# vocab-size-independent bits-per-byte validation metric.
with open(os.path.join(DATASET_DIR, "tokenizer/token_bytes.pt"), "rb") as f:
    token_bytes = torch.load(f, map_location=device)
assert token_bytes[cfg.mask_id] == 0 and token_bytes[bos_id] == 0, "mask_id / bos_id are not special tokens"


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

    # Attention
    W_Q: Param
    W_K: Param
    W_V: Param
    W_O: Param
    head_gate:     Param  # Per-head gate on the normed attention output, every layer.
    value_embeds:  Param
    ve_gate:       Param
    ngram_values:  Param  # The n-gram value memories: two hash slots per memory.
    ngram_gate:    Param
    ngram_mults:   list   # [memory][slot][offset] odd hash multipliers

    # MLP -- one bank per width, W_in_<d> / W_out_<d> (see mlp_banks)

    # Cross-Layer
    x0_lambdas:     Param   # Per-layer coefficient for reading the input embedding.
    x0_gates:       Param   # Per-layer scale on the channel-mean gate of that read.
    resid_lambdas:  Param   # Per-layer gain on the residual stream.
    pool_lambda:    Param   # Weight of the pool layer's output in the final stream.

    # Output
    lm_head: Param   # MLM, on the corrupted filler positions
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
    For T=256K  -->  Held: 54GB,  Recomputed:  20.25GB
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
    y_inv_rms:          Tensor    # (L,  T, n_qo,    1)   fp32         (18MB)
    lse:                Tensor    # (L,  n_qo,  T)        fp32         (18MB)
    x_attn_out_hat:     Tensor    # (L,  T,     D)             4.5GB
    x_attn_out_inv_rms: Tensor    # (L,  T,     1)        fp32          (3MB)
    mlp_a:              Tensor    # (L,  T, d_mlp[i])           18GB
    #mlp_relu:          Tensor    # (L,  T, d_mlp)                               18GB
    mlp_out_hat:        Tensor    # (L,  T,     D)             4.5GB
    mlp_out_inv_rms:    Tensor    # (L,  T,     1)        fp32          (3MB)
    #                                                         ------           ------
    #                                                 TOTAL:    54GB          20.25GB

# Model-level activations held as locals:
#   x0                 (T, D)          384MB    layer-blend + embedding-norm backward
#   x_embed_inv_rms    (T, 1)   fp32   (1.2MB)
#   x_pool             (T, D)          384MB    pooling backward
#   x_final_hat        (T, D)          384MB    final-norm backward
#   x_mlm_hat          (M, D)          115MB    lm_head grad
#   ngram_ids          (7, 2, T) int64  22MB    the n-gram tables' rows
#                               TOTAL: 1.2GB

# Cast shorthands for the bodies below: the fp32 scalars/gates need explicit
# bf16 casts at their use sites (see forward_backward's docstring), and the
# scalar-parameter grad sums accumulate in fp32.
bf16  = lambda x: x.to(torch.bfloat16)
sum32 = lambda x: x.sum(dtype=torch.float32)


# ==============================================================================
# § MLM Corruption
# ==============================================================================

def corrupt(tokens, num_targets, gen, filler_start=0):
    """num_targets random non-BOS positions of `tokens` from filler_start on (a pack's
    decision records come before it), 80% of them replaced by [MASK], 10% by a random
    token, 10% left as they are: returns (idx, pos, labels), the corrupted input, the
    positions (sorted) and the tokens that were there."""
    T = tokens.numel()
    score = (torch.rand(T, device=device, generator=gen) + (tokens == bos_id).float()   # BOS and the records sort last
             + (torch.arange(T, device=device) < filler_start).float())
    pos = score.argsort()[:num_targets].sort().values
    labels = tokens[pos]
    draw = torch.rand(num_targets, device=device, generator=gen)
    random_tokens = torch.randint(0, bos_id, (num_targets,), device=device, generator=gen, dtype=torch.int32)
    replacement = torch.where(draw < 0.8, torch.full_like(labels, cfg.mask_id),
                              torch.where(draw < 0.9, random_tokens, labels))
    idx = tokens.clone()
    idx[pos] = replacement
    return idx, pos, labels.long()


# ==============================================================================
# § Train Forward + Backward
# ==============================================================================

@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def forward_backward(pk: Pack, idx, mlm_pos, labels, backward=True):
    """One pack through the model: the corrupted tokens `idx`, the MLM head read out
    at the M positions `mlm_pos` against `labels`, and the decision readout at the
    pack's readout positions. Returns (MLM loss, decision loss) for training, or each
    corrupted position's MLM nats and each readout's probability for validation
    (backward=False)."""

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
    # x0         - The normed embedding, re-read by every layer
    # x          - The running residual stream (x_in is this layer's input)
    # x_biased   - Layer input scaled by resid_lambda, biased by x0
    # x_src      - The source_layer's output, the src_layers' attention input
    # x_attn_out - Post-attention stream, the MLP's input
    # x_final    - Post-pooling stream, feeding the lm_head and the readout
    # x_mlm      - The final stream at the corrupted positions
    # x_ro       - The final stream at the readout positions
    #
    # RMS naming:
    # *_inv_rms  - 1/rms, used by RMS norm fwd and bwd.
    # *_hat      - RMS-normed  (x_hat = x * inv_rms)
    #
    # Activation naming -- the MLP and the gates share three stages:
    # *_z        - Pre-nonlinearity, the raw matmul output. Only ever named in
    #              bwd (mlpb_z, ve_gateb_z, ngram_gateb_z, x0_gateb_z); inlined in fwd.
    #              ('logit' is reserved for the lm_head's output.)
    # *_relu     - Post-relu, pre-square (MLP only; recomputed in bwd).
    # *_sig      - Post-sigmoid, in [0, 1] (gates only; recomputed in bwd).
    # *_a        - The final activation handed onward: mlp_a = mlp_relu^2,
    #              ve_gate_a = 2*ve_gate_sig.

    assert idx.ndim == 1
    T = idx.size(0)
    M = mlm_pos.size(0)
    half = cfg.d_qk // 2

    assert T <= m.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {m.cos.size(1)}"

    # Rotary angles at each token's position within its segment; a question's
    # positions continue its state's. (RoPE is relative, so a filler document reads
    # as it did in the MLM base, where positions ran across the micro-batch.)
    cos, sin = m.cos[0, pk.pos], m.sin[0, pk.pos]  # (T, 1, d_qk): [cos, cos] and [sin, -sin]
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_input_vocab, -1)
    x_pool = None
    x_src_hat = None

    # The n-gram memories' rows: each slot hashes the (corrupted) tokens at its offsets
    # from the position, the micro-batch's edges padded with their own token.
    curr = idx.long()
    at = lambda off: (curr if off == 0 else
                      torch.cat([curr[:1].expand(-off), curr[:off]]) if off < 0 else
                      torch.cat([curr[off:], curr[-1:].expand(off)]))
    ngram_ids = torch.stack([torch.stack([
        functools.reduce(torch.bitwise_xor, [at(off) * mult for off, mult in zip(offsets, mults)]) % cfg.d_ngram
        for offsets, mults in zip((s0, s1), m.ngram_mults[n])])
        for n, (_, s0, s1) in enumerate(cfg.ngram_specs)])   # (num_ngrams, 2, T)

    # -----------------------------
    #           Forward
    # -----------------------------

    # Input embeddings
    x_embed = F.embedding(idx, m.input_embeds.w)        # bf16

    # RMS normalize the input embeds; also added back to the stream at every layer.
    x_embed_inv_rms = (x_embed.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x0 = bf16(x_embed.float() * x_embed_inv_rms)      # appears as x0b in bwd

    # The residual stream starts as the normed embedding.
    x = x0

    # One LayerStash of forward activations per layer
    stash = []

    # For each layer,
    for i in range(cfg.n_layers):
        x_in = x

        # Scale residual stream, add gated input embedding
        x0_gate = 2.0 * torch.sigmoid(m.x0_gates.w[i] * x_in.float().mean(dim=-1, keepdim=True))   # (T, 1)
        x_biased = m.resid_lambdas.w[i] * x_in + bf16(m.x0_lambdas.w[i] * x0_gate) * x0

        # Attention input norm. The src_layers attend from the source stream instead.
        if i in cfg.src_layers:
            x_biased_hat, x_biased_inv_rms = x_src_hat, x_src_inv_rms
        else:
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
            ve_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_gate] @ m.ve_gate.w[j].mT)
            ve_gate_a = 2 * ve_gate_sig          # (T, n_kv_heads), in [0, 2]
            v = v + ve_gate_a.unsqueeze(-1) * ve # ve and the gate are both recomputed in bwd, not stashed

        # N-gram value memories, gated like ve, each from its own slice of the input
        for n, s in cfg.ngram_memories[i]:
            ngram_v = torch.cat([F.embedding(ngram_ids[n, h], ngram_wbank[2 * n + h]) for h in (0, 1)],
                                dim=-1).view(T, cfg.n_kv_heads, cfg.d_vo)
            ngram_gate_sig = torch.sigmoid(x_biased_hat[..., s * cfg.d_gate:(s + 1) * cfg.d_gate] @ m.ngram_gate.w[n].mT)
            v = v + (2 * ngram_gate_sig).unsqueeze(-1) * ngram_v

        # QK-Norm, then RoPE
        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q, k = q.float() * q_inv_rms, k.float() * k_inv_rms
        q_hat = bf16(q * cos + torch.roll(q, half, dims=-1) * sin)
        k_hat = bf16(k * cos + torch.roll(k, half, dims=-1) * sin)

        # Read V from the residual streams both ways by matching their K. Every
        # segment -- state, question, filler document -- attends to itself.
        y, lse = flash_attn_varlen_fwd_lse(q_hat, k_hat, v, pk.cu_self, pk.cu_self,
                                           cfg.max_seg_len, cfg.max_seg_len, cfg.window_sizes[i])

        # Each record's questions also read its whole state. The two calls are
        # softmaxes over disjoint keys, so their union is their outputs weighted by
        # each one's share of the merged normalizer, exp(lse - lse_merged). The
        # cross call leaves rows outside its segments unwritten: merge on question
        # rows only. The merged output is what the per-head norm below reads.
        y_x, lse_x = flash_attn_varlen_fwd_lse(q_hat, k_hat, v, pk.cu_q, pk.cu_k,
                                               cfg.max_qblock_len, cfg.max_state_len, (-1, -1))
        lse_merged = torch.logaddexp(lse, lse_x)    # (H, T)
        y_merged = (y.float() * (lse - lse_merged).exp().T.unsqueeze(-1)
                    + y_x.float() * (lse_x - lse_merged).exp().T.unsqueeze(-1))
        y = torch.where(pk.is_q[:, None, None], bf16(y_merged), y).contiguous()
        lse = torch.where(pk.is_q, lse_merged, lse)

        # Per-head output norm and gate
        y_inv_rms = (y.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        head_gate_a = 2 * torch.sigmoid(x_biased_hat[..., :cfg.d_gate] @ m.head_gate.w[i].mT)   # (T, n_qo_heads)
        y_gated = bf16(y.float() * y_inv_rms * head_gate_a.unsqueeze(-1))

        # Project value heads onto their output heads.
        attn_out = y_gated.view(T, -1) @ m.W_O.w[i].mT

        # Write back to the stream.
        x_attn_out = x_biased + attn_out

        # MLP input norm. Stashed, not recomputed.
        x_attn_out_inv_rms = (x_attn_out.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        x_attn_out_hat = bf16(x_attn_out.float() * x_attn_out_inv_rms)

        # MLP -- this layer's width has its own bank; k is the layer's slot in it
        W_in, W_out, k = mlp_banks[i]
        mlp_a = F.relu(x_attn_out_hat @ W_in.w[k].mT - 0.75).square()   # (T, d_mlp[i]) stashed
        mlp_out = mlp_a @ W_out.w[k].mT
        mlp_out_inv_rms = (mlp_out.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        mlp_out_hat = bf16(mlp_out.float() * mlp_out_inv_rms)

        # Write back to the stream.
        x = x_attn_out + mlp_out_hat        # the residual stream; appears as xb in bwd

        # The source layer's output is the src_layers' attention input.
        if i == cfg.source_layer:
            x_src_inv_rms = (x.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
            x_src_hat = bf16(x.float() * x_src_inv_rms)

        # Stash the pool layer's output, to add it onto the final stream.
        if i == cfg.pool_layer:
            x_pool = x

        # Stash activations for backward pass. Skipped entirely under eval.
        if backward:
            stash.append(LayerStash(x_in=x_in, x_biased_hat=x_biased_hat, x_biased_inv_rms=x_biased_inv_rms,
                                    q_hat=q_hat, k_hat=k_hat, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
                                    v=v, y=y, y_inv_rms=y_inv_rms, lse=lse, x_attn_out_hat=x_attn_out_hat,
                                    x_attn_out_inv_rms=x_attn_out_inv_rms, mlp_a=mlp_a,
                                    mlp_out_hat=mlp_out_hat, mlp_out_inv_rms=mlp_out_inv_rms))

    x_final = x + bf16(m.pool_lambda.w) * x_pool

    # Final output norm
    x_final_inv_rms = (x_final.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    x_final_hat = bf16(x_final.float() * x_final_inv_rms)

    # -----------------------------
    #           MLM Head
    # -----------------------------
    # Only the corrupted positions are read out, M of them.
    x_mlm_hat = x_final_hat[mlm_pos]    # (M, D)
    tgt = labels.unsqueeze(1)           # (M, 1)

    # ==== Forward ====
    logits_raw = x_mlm_hat @ m.lm_head.w.mT  # (M, d_vocab) bf16

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
    mlm_nats = lse_ce - tgt_logit                # (M,)

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
    # Return training losses for logging: nats per corrupted position, nats per question
    # (the cross-entropy against each question's target distribution).
    loss = torch.stack([mlm_nats.mean(),
                        -(pk.ro_tgt * ro_p.clamp_min(1e-30).log()).sum() / pk.n_q[0].clamp_min(1.0)])

    # ==== Backward ====
    onehot = torch.arange(cfg.d_vocab, device=device).unsqueeze(0) == tgt # (M, d_vocab)

    # Predicted probs = e / ssum; (M, d_vocab)
    # Standard approach to backprop is (p - 1), with optimizer update w = w - grad.
    # We flip this to (1 - p) and w = w + grad, so that:
    # - grads and optim state move in the same direction as their param.
    # - target token contributes positively to stream and gradients.
    logitsb = bf16((onehot.float() - (e / ssum)) * (1.0 - logits/15.0 * logits/15.0) / M)

    # Every corrupted position updates every vocab entry.
    m.lm_head.grad.copy_((logitsb.mT @ x_mlm_hat).float()) # (d_vocab, M) @ (M, d_model) --> (d_vocab, d_model)

    # The decision readout, flipped the same way: target - p (soft targets included:
    # the cross-entropy's gradient is linear in the target), scaled to the mean over
    # the pack's questions.
    ro_valid = (pk.ro_qid < cfg.q_cap).float()
    ro_logitb = (pk.ro_tgt - ro_p) * ro_valid * (cfg.decision_weight / pk.n_q[0].clamp_min(1.0))   # (Rc,)
    m.readout.grad.copy_((ro_logitb.unsqueeze(1) * x_ro_hat.float()).sum(dim=0, keepdim=True))

    # The backward streams start as weighted sums of the head embeddings that they
    # (meaningfully) predicted, at the corrupted positions, and as the readout vector
    # at the readout positions; zero elsewhere. Padding readouts add zeros at row 0.
    xb_final_hat = torch.zeros_like(x_final_hat).index_copy_(0, mlm_pos, logitsb @ m.lm_head.w) # (T, d_model)
    xb_final_hat.index_put_((pk.ro_pos,), bf16(ro_logitb.unsqueeze(1) * m.readout.w), accumulate=True)
    del logitsb

    # -----------------------------
    #           Backward
    # -----------------------------
    # Each n-gram hash's rows, compacted for the sparse updates.
    ngram_sorted, ngram_order = ngram_ids.sort(dim=-1)                                # (num_ngrams, 2, T)
    ngram_segment = torch.cat([torch.ones_like(ngram_sorted[..., :1]),
                               (ngram_sorted[..., 1:] != ngram_sorted[..., :-1]).long()], dim=-1).cumsum(dim=-1) - 1
    ngram_inverse = torch.empty_like(ngram_segment).scatter_(-1, ngram_order, ngram_segment)
    ngram_touched = torch.full_like(ngram_sorted, cfg.d_ngram).scatter_(-1, ngram_segment, ngram_sorted)[..., :cfg.touched_rows]

    # Scalar grads are collected, grad tensors updated at the end.
    g_resid = []; g_x0 = []; g_x0_gate = []

    # Each stream's cosine similarity to the vocab signal, divided by d_model.
    # (T, 1) = ((T, d_model) * (T, d_model)).mean()
    res_ms = (x_final_hat.float() * xb_final_hat.float()).mean(dim=-1, keepdim=True)

    # The component of xb_final_hat which is perpendicular to x_final_hat?
    xb_final = bf16(x_final_inv_rms * (xb_final_hat.float() - (x_final_hat.float() * res_ms)))

    # Dot product between final vs. pooled streams.
    m.pool_lambda.grad.copy_(sum32(xb_final * x_pool))  # (T, d_model)

    # xb updates every layer, keep xb_final for the pool layer.
    xb = xb_final                       # grad wrt layer num_layers-1's output
    x0b = torch.zeros_like(x0)          # accumulates over layers
    xb_src_hat = torch.zeros_like(x0)   # accumulates over the src_layers

    # For each layer in reverse order,
    for i in reversed(range(cfg.n_layers)):

        # Stashed forward activations for this layer.
        st = stash[i]

        if i == cfg.pool_layer:
            # TRAP: x_pool gets an EXTRA contribution from the final stream
            xb = xb + bf16(m.pool_lambda.w) * xb_final

        # --- MLP output norm backward: xb is the grad w.r.t. mlp_out_hat ---
        mlp_out_hat = st.mlp_out_hat.float()
        mlpb_out = bf16(st.mlp_out_inv_rms * (xb.float() - mlp_out_hat * (mlp_out_hat * xb.float()).mean(dim=-1, keepdim=True)))

        # --- MLP backward (relu(z - 0.75)^2: mlpb_z = 2*mlp_relu*mlpb_a, self-masking
        #     since mlp_relu is already 0 where z < 0.75) ---

        W_in, W_out, k = mlp_banks[i]

        # Grad w.r.t. W_out
        W_out.gbank[k].copy_(mlpb_out.mT @ st.mlp_a) # (d_model, T) @ (T, d_mlp[i])

        # Grad w.r.t. the pre-relu matmul output (mlpb_a = mlpb_out @ W_out is inline)
        mlp_relu = st.mlp_a * (st.mlp_a + 1e-30).rsqrt()   # = mlp_a.sqrt()
        mlpb_z = 2.0 * mlp_relu * (mlpb_out @ W_out.w[k])

        x_attn_out_inv_rms, x_attn_out_hat = st.x_attn_out_inv_rms, st.x_attn_out_hat

        # Grad w.r.t. W_in
        W_in.gbank[k].copy_(mlpb_z.mT @ x_attn_out_hat) # (d_mlp[i], T) @ (T, d_model)

        xb_attn_out_hat = mlpb_z @ W_in.w[k] # (T, d_mlp[i]) @ (d_mlp[i], d_model) --> (T, d_model)

        xb_attn_out = xb + bf16(x_attn_out_inv_rms * (xb_attn_out_hat.float() - (x_attn_out_hat.float() * (x_attn_out_hat.float() * xb_attn_out_hat.float()).mean(dim=-1, keepdim=True))))

        # Attention backward
        x_biased_hat = st.x_biased_hat
        y_hat = st.y.float() * st.y_inv_rms
        head_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_gate] @ m.head_gate.w[i].mT)   # (T, n_qo_heads)
        m.W_O.gbank[i].copy_(xb_attn_out.mT @ bf16(y_hat * (2 * head_gate_sig).unsqueeze(-1)).view(T, -1))
        yb_gated = (xb_attn_out @ m.W_O.w[i]).view(T, cfg.n_qo_heads, cfg.d_vo).float()

        # Head gate backward
        head_gateb_z = bf16((yb_gated * y_hat).sum(dim=-1) * (2 * head_gate_sig * (1 - head_gate_sig)))   # (T, n_qo_heads)
        m.head_gate.gbank[i].copy_(head_gateb_z.mT @ x_biased_hat[..., :cfg.d_gate])
        yb_hat = yb_gated * (2 * head_gate_sig).unsqueeze(-1)
        yb = bf16(st.y_inv_rms * (yb_hat - y_hat * (y_hat * yb_hat).mean(dim=-1, keepdim=True)))

        # Grads w.r.t. the gate slices of x_biased_hat
        xb_gates = torch.zeros(T, 3 * cfg.d_gate, dtype=torch.bfloat16, device=device)
        xb_gates[:, :cfg.d_gate] += head_gateb_z @ m.head_gate.w[i]

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
        qb = bf16(st.q_inv_rms * (qb_hat.float() - st.q_hat.float() * (st.q_hat.float() * qb_hat.float()).mean(dim=-1, keepdim=True)))
        kb = bf16(st.k_inv_rms * (kb_hat.float() - st.k_hat.float() * (st.k_hat.float() * kb_hat.float()).mean(dim=-1, keepdim=True)))

        # rotary backward = rotation by -theta (transpose of the forward rotation)
        qb = qb * cos - torch.roll(qb, half, dims=-1) * sin
        kb = kb * cos - torch.roll(kb, half, dims=-1) * sin

        # --- VE gate backward (ve and ve_gate_sig recomputed) ---
        j = cfg.ve_index[i]
        if j >= 0:
            # Retrieve the value embeddings
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)

            # Recompute gate forward
            ve_gate_sig = torch.sigmoid(x_biased_hat[..., :cfg.d_gate] @ m.ve_gate.w[j].mT)

            # ve_gate_a broadcasts over d_vo in v = v0 + a*ve, so its grad sums
            # that axis back out; then d/dz[2*sigmoid(z)] = 2*sig*(1 - sig).
            ve_gateb_a = (vb * ve).sum(dim=-1)   # (T, n_kv_heads)
            ve_gateb_z = ve_gateb_a * (2 * ve_gate_sig * (1 - ve_gate_sig))

            m.ve_gate.gbank[j].copy_(ve_gateb_z.mT @ x_biased_hat[..., :cfg.d_gate])

            # Embedding gradient
            veb = (vb * (2 * ve_gate_sig).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            # embedding_dense_backward beat raw index_add_ atomics ~2x
            m.value_embeds.gbank[j].copy_(
                torch.ops.aten.embedding_dense_backward(veb, idx, cfg.d_input_vocab, -1, False))

            xb_gates[:, :cfg.d_gate] += ve_gateb_z @ m.ve_gate.w[j]

        # --- n-gram memory backward (ngram_v and ngram_gate_sig recomputed) ---
        for n, s in cfg.ngram_memories[i]:
            gate_in = x_biased_hat[..., s * cfg.d_gate:(s + 1) * cfg.d_gate]
            ngram_v = torch.cat([F.embedding(ngram_ids[n, h], ngram_wbank[2 * n + h]) for h in (0, 1)],
                                dim=-1).view(T, cfg.n_kv_heads, cfg.d_vo)
            ngram_gate_sig = torch.sigmoid(gate_in @ m.ngram_gate.w[n].mT)
            ngram_gateb_z = (vb * ngram_v).sum(dim=-1) * (2 * ngram_gate_sig * (1 - ngram_gate_sig))   # (T, n_kv_heads)
            m.ngram_gate.gbank[n].copy_(ngram_gateb_z.mT @ gate_in)
            xb_gates[:, s * cfg.d_gate:(s + 1) * cfg.d_gate] += ngram_gateb_z @ m.ngram_gate.w[n]
            ngram_vb = (vb * (2 * ngram_gate_sig).unsqueeze(-1)).reshape(T, 2, -1)   # each hash's half of the values

            # Sparse update: row-wise RMSProp on each hash's touched rows, fused in at its gradient.
            for h in (0, 1):
                slot, rows = 2 * n + h, ngram_touched[n, h]
                ngram_grad = torch.ops.aten.embedding_dense_backward(
                    ngram_vb[:, h].float(), ngram_inverse[n, h], cfg.touched_rows, -1, False)   # (touched_rows, 384)
                ngram_vbank[slot].mul_(m.ngram_values.mntm_b2_t[t_step])   # (rows, 1) fp32
                ngram_mntm = (ngram_vbank[slot].index_select(0, rows)
                              + ngram_grad.square().mean(dim=-1, keepdim=True) * m.ngram_values.grad_b2_t[t_step])
                ngram_vbank[slot].index_copy_(0, rows, ngram_mntm)
                ngram_wbank[slot].index_copy_(0, rows, bf16(
                    ngram_wbank[slot].index_select(0, rows).float()
                    + m.ngram_values.lr_bc_t[t_step] * (ngram_grad / (ngram_mntm.sqrt() + m.ngram_values.eps_t[t_step]))))

        # vb passes through the value adds unchanged: v = v0 + ve_gate_a*ve + ...
        qb = qb.view(T, cfg.n_qo_heads * cfg.d_qk)
        kb = kb.view(T, cfg.n_kv_heads * cfg.d_qk)
        vb = vb.reshape(T, cfg.n_kv_heads * cfg.d_vo)

        m.W_Q.gbank[i].copy_(qb.mT @ x_biased_hat)
        m.W_K.gbank[i].copy_(kb.mT @ x_biased_hat)
        m.W_V.gbank[i].copy_(vb.mT @ x_biased_hat)

        xb_biased_hat = qb @ m.W_Q.w[i] + kb @ m.W_K.w[i] + vb @ m.W_V.w[i]
        xb_biased_hat[:, :3 * cfg.d_gate] += xb_gates

        # Attention input norm backward
        if i in cfg.src_layers:
            xb_src_hat = xb_src_hat + xb_biased_hat
            xb_biased = xb_attn_out
        else:
            xb_biased = xb_attn_out + bf16(st.x_biased_inv_rms * (xb_biased_hat.float() - (x_biased_hat.float() * (x_biased_hat.float() * xb_biased_hat.float()).mean(dim=-1, keepdim=True))))

        # --- blend backward: x_biased = resid_lambdas[i]*x_in + x0_lambdas[i]*x0_gate*x0,
        #     x0_gate = 2*sigmoid(x0_gates[i] * mean_c(x_in)) ---
        x_in_mean  = st.x_in.float().mean(dim=-1, keepdim=True)                         # (T, 1)
        x0_gate    = 2.0 * torch.sigmoid(m.x0_gates.w[i] * x_in_mean)                   # (T, 1)
        x0_dot     = (xb_biased * x0).sum(dim=-1, keepdim=True, dtype=torch.float32)    # (T, 1)
        x0_gateb_z = x0_dot * m.x0_lambdas.w[i] * x0_gate * (1.0 - 0.5 * x0_gate)       # (T, 1)
        g_resid.append(sum32(xb_biased * st.x_in))
        g_x0.append((x0_dot * x0_gate).sum())
        g_x0_gate.append((x0_gateb_z * x_in_mean).sum())
        x0b = x0b + bf16(m.x0_lambdas.w[i] * x0_gate) * xb_biased  # TRAP: x0 feeds every layer, accumulate
        xb = m.resid_lambdas.w[i] * xb_biased + bf16(x0_gateb_z * (m.x0_gates.w[i] / cfg.d_model))

        # TRAP: the source layer's output (this layer's input) also fed the src_layers' attention.
        if i == cfg.source_layer + 1:
            xb = xb + bf16(x_src_inv_rms * (xb_src_hat.float() - (x_src_hat.float() * (x_src_hat.float() * xb_src_hat.float()).mean(dim=-1, keepdim=True))))
        stash[i] = None                          # free this layer's stash as we go

    # Land the per-layer resid/x0/x0-gate scalar sums (collected in REVERSED
    # layer order) as one stacked copy each.
    m.resid_lambdas.grad.copy_(torch.stack(g_resid[::-1]))
    m.x0_lambdas.grad.copy_(torch.stack(g_x0[::-1]))
    m.x0_gates.grad.copy_(torch.stack(g_x0_gate[::-1]))

    # xb is now the grad through layer 0's input, which IS x0 (same tensor), so
    # it folds into x0b to give the full grad wrt the normed embedding.
    x0b = x0b + xb

    # --- embedding norm + token embedding scatter ---
    xb_embed = bf16(x_embed_inv_rms * (x0b.float() - (x0.float() * (x0.float() * x0b.float()).mean(dim=-1, keepdim=True))))
    m.input_embeds.grad.copy_(
        torch.ops.aten.embedding_dense_backward(xb_embed, idx, cfg.d_input_vocab, -1, False))

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
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
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
# § Data
# ==============================================================================
# Packs (utils_decision.py § Packs): decision records in a decision_frac share of each
# pack, ClimbMix filler trained on MLM in the rest. Every pack is planned and laid out
# up front, in pinned memory. The run's step count is the training packs' count, and the
# static sizes forward_backward compiles against come from all the plans together.

packs_t0 = time.perf_counter()
rng = np.random.default_rng(cfg.seed)

# Evaluation: every record of the two development splits once, as stored, at a fixed 0.9
# share whatever the training share (so a checkpoint scores the same in any run), with
# filler from the val shard whose MLM goes unread. Training: shuffled epochs of the
# training records, a choice question's options permuted each time, with filler the MLM
# base never read.
val_docs = doc_stream("val", bos_id)
eval_records = {split: load_records(split, cfg.state_cap, cfg.soft_weight) for split in ("decision_dev", "transfer_dev")}
eval_plans = {split: plan_packs(records, val_docs, cfg.train_batch_tokens, 0.9, cfg.seq_len, rng)
              for split, records in eval_records.items()}
train_records = [r for split in cfg.train_splits for r in load_records(split, cfg.state_cap, cfg.soft_weight)]
train_plans = plan_packs(train_records, doc_stream("train", bos_id, cfg.filler_shard), cfg.train_batch_tokens,
                         cfg.decision_frac, cfg.seq_len, rng, cfg.epochs)
cfg.num_steps = len(train_plans) - 1   # the last pack is the loop's prefetch

caps = pack_caps(train_plans + eval_plans["decision_dev"] + eval_plans["transfer_dev"])
round_up = lambda x, k: k * math.ceil(x / k)
# Corrupted positions per training pack, from the pack with the least filler.
cfg.train_targets  = 64 * int(cfg.mask_rate * pack_caps(train_plans)["min_filler"] / 64)
assert pack_caps(eval_plans["decision_dev"] + eval_plans["transfer_dev"])["min_filler"] >= cfg.eval_targets
cfg.cu_width       = round_up(caps["segments"] + 2, 64)
cfg.rec_width      = round_up(caps["records"] + 1, 64)
cfg.ro_cap         = round_up(caps["readouts"], 64)
cfg.q_cap          = round_up(caps["questions"], 64)
cfg.max_seg_len    = round_up(max(caps["seg_len"], cfg.seq_len), 128)   # the MLM validation's documents too
cfg.max_qblock_len = round_up(caps["qblock_len"], 128)
cfg.max_state_len  = round_up(caps["state_len"], 128)

eval_packs = {split: materialize_packs(plans, cfg.train_batch_tokens, cfg) for split, plans in eval_plans.items()}
train_packs, _ = materialize_packs(train_plans, cfg.train_batch_tokens, cfg)

placed = sum(len(chosen) for chosen, _ in train_plans[:-1])
dec_tokens = cfg.train_batch_tokens * cfg.num_steps - sum(d.size for _, filler in train_plans[:-1] for d in filler)
capped = [r for r in train_records if r["state_len"] > cfg.state_cap]
data_report = (
    f"Packs: {cfg.num_steps:,} train ({placed:,} records = {placed / len(train_records):.2f} epochs of "
    f"{len(train_records):,} from {', '.join(cfg.train_splits)}; decision tokens {dec_tokens / 1e6:.1f}M = "
    f"{100 * dec_tokens / (cfg.train_batch_tokens * cfg.num_steps):.1f}%), "
    f"eval {', '.join(f'{s} {len(p)}' for s, p in eval_plans.items())}\n"
    f"  states capped at {cfg.state_cap}: {len(capped):,} training records "
    f"({sum(r['state_len'] - cfg.state_cap for r in capped) / 1e6:.2f}M state tokens cut)\n"
    f"  per pack: {cfg.train_targets:,} corrupted filler positions ({cfg.eval_targets} in evaluation) | "
    f"caps: segments {cfg.cu_width}, records {cfg.rec_width}, readouts {cfg.ro_cap}, questions {cfg.q_cap} | "
    f"max_seqlen: segment {cfg.max_seg_len}, question block {cfg.max_qblock_len}, state {cfg.max_state_len}")
# Uniform guessing on each split's clean questions, the floor the accuracies read against.
chance = {split: np.mean([1 / len(q["options"]) for r in records if r["variant"] == "clean" for q in r["questions"]])
          for split, records in eval_records.items()}
del eval_plans, train_plans, train_records, eval_records, val_docs, capped
packs_time = time.perf_counter() - packs_t0


# ==============================================================================
# § Weight Init & Schedule
# ==============================================================================

# ------------------------------------------------------------------------------
# §§ LR Schedule
# ------------------------------------------------------------------------------

# Learning rate schedules as per-step multipliers.
# No warmup: peak from step 0.
steps_0idx = np.arange(cfg.num_steps, dtype=np.float64)  # 0-based, the way the loop counts
steps_1idx = steps_0idx + 1.0                            # 1-based, the way bias corrections count
run_frac   = steps_0idx / cfg.num_steps

# Warmdown to 0.025: Muon over the last 95% of the run, AdamW over the last 60%.
muon_frac = np.maximum(0.0, (run_frac - 0.05) / 0.95)
adam_frac = np.maximum(0.0, (run_frac - 0.40) / 0.60)

muon_lr_mult_t    = (1.0 - (1.0 - 0.025) * muon_frac) * LR_SCALE
lr_mult_t         = (1.0 - (1.0 - 0.025) * adam_frac) * LR_SCALE   # AdamW's
readout_lr_mult_t = 1.0 - (1.0 - 0.025) * adam_frac                # the readout's: AdamW's shape, no LR_SCALE


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

resid_lambdas  = fp32_empty(cfg.n_layers).fill_(1.0)
x0_lambdas     = fp32_empty(cfg.n_layers).fill_(0.1)
x0_gates       = fp32_zeros(cfg.n_layers)
pool_lambda    = fp32_zeros(1)

# Grad mix-in coefficient (1-beta), currently fixed, can be scheduled here.
scalar_grad_mult_t = np.ones(cfg.num_steps)


scalar_configs = [
#   name,               weights,        lr schedule,     peak lr,  b1_grad,   b2_grad,    wd,
    ("resid_lambdas",   resid_lambdas,  lr_mult_t,       0.008,    0.2,        0.05,   0.0),
    ("x0_lambdas",      x0_lambdas,     muon_lr_mult_t,  1.2,      0.04,       0.05,   0.002),
    ("x0_gates",        x0_gates,       muon_lr_mult_t,  1.2,      0.04,       0.05,   0.002),
    ("pool_lambda",     pool_lambda,    lr_mult_t,       0.06,     0.04,       0.05,   0.0),
]

# For each of the scalar parameters...
for (name, w, sched_t, peak_lr, b1_grad, b2_grad, wd) in scalar_configs:

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
        lr_bc_t      = dev(sched_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),

        # Weight decay schedule
        wd_t         = dev(1.0 - sched_t * peak_lr * wd),

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

input_embeds =     bf16_empty(cfg.d_input_vocab, cfg.d_model)
input_embeds.copy_(fp32_empty(cfg.d_input_vocab, cfg.d_model).normal_(mean=0.0, std=1.0))
value_embeds =     bf16_empty(cfg.num_ves * cfg.d_input_vocab, cfg.n_kv_heads * cfg.d_vo)
value_embeds.copy_(fp32_empty(cfg.num_ves * cfg.d_input_vocab, cfg.n_kv_heads * cfg.d_vo)
                   .uniform_(-matrix_init_s, matrix_init_s))

# Beta1 anneals 0.8 -> 0.4 across AdamW's warmdown.
embed_b1_mntm_t = 0.8 + (0.4 - 0.8) * adam_frac

embed_configs = [
#   name,             weights,       peak lr,  b2_grad,  wd,     slots
    ("input_embeds",  input_embeds,  0.6,      0.05,     0.0,    1),
    ("value_embeds",  value_embeds,  0.6,      0.05,     0.0,    cfg.num_ves)
]

# For each of the embedding tables...
for (name, w, peak_lr, b2_grad, wd, slots) in embed_configs:

    # Derive the momentum buffer's decay.
    b2_mntm = 1 - b2_grad

    grad = bf16_zeros(w.shape) # Embeddings can handle bf16 accumulation fine

    p = Param(
        # Weight
        name         = name,
        w            = w,          # bf16 live, no fp32 master
        mantissa     = None,

        # Gradients
        grad         = grad, 
        gbank        = list(grad.view(slots, cfg.d_input_vocab, -1).unbind(0)) if slots > 1 else None,

        # Momentum buffers
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(w.shape),

        residual_dim = None, # Muon only

        # Schedules
        # Fold bias correction into the learning rate.
        lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - embed_b1_mntm_t ** steps_1idx)),

        # Weight decay schedule
        wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

        # Beta1 annealed, Beta2 constant
        mntm_b1_t    = dev(embed_b1_mntm_t),                   # first_mntm decay (Beta1)
        grad_b1_t    = dev(1.0 - embed_b1_mntm_t),             # first_mntm grad mix-in (1-Beta1)

        mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),   # scnd_mntm decay (Beta2)
        grad_b2_t    = dev(np.full(cfg.num_steps, b2_grad)),   # scnd_mntm grad mix-in (1-Beta2)

        eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)

# ------------------------------------------------------------------------------
# §§ N-gram Value Memories
# ------------------------------------------------------------------------------

ngram_values = bf16_zeros(cfg.num_ngrams * 2 * (cfg.d_ngram + 1), cfg.n_kv_heads * cfg.d_vo // 2)  # + a scratch row per slot

# Beta2 anneals 0.999 -> 0.99995 across Muon's warmdown.
peak_lr = 0.6
ngram_b2_mntm_t = 0.999 + (0.99995 - 0.999) * muon_frac
ngram_bc2_t     = 1.0 - ngram_b2_mntm_t ** steps_1idx

m.ngram_values = Param(
    # Weight
    name         = "ngram_values",
    w            = ngram_values,      # bf16 live, no fp32 master
    mantissa     = None,

    # Gradients -- none: the update is fused into forward_backward
    grad         = None,
    gbank        = None,

    # Momentum buffers
    first_mntm   = None,
    scnd_mntm    = fp32_zeros(cfg.num_ngrams * 2 * (cfg.d_ngram + 1), 1),

    residual_dim = None, # Muon only

    # Schedules
    # Fold bias correction into the learning rate.
    lr_bc_t      = dev(np.full(cfg.num_steps, peak_lr * LR_SCALE) * ngram_bc2_t ** 0.5),

    wd_t         = None,

    mntm_b1_t    = None,
    grad_b1_t    = None,

    mntm_b2_t    = dev(ngram_b2_mntm_t),                   # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(1.0 - ngram_b2_mntm_t),             # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * ngram_bc2_t ** 0.5),
)

# The slots as standalone tensors, so each update writes a whole one.
ngram_wbank = list(ngram_values.view(cfg.num_ngrams * 2, cfg.d_ngram + 1, -1).unbind(0))
ngram_vbank = list(m.ngram_values.scnd_mntm.view(cfg.num_ngrams * 2, cfg.d_ngram + 1, 1).unbind(0))

# Each slot's odd multipliers, one per offset of its key.
hash_rng = torch.Generator(device="cpu").manual_seed(0)
m.ngram_mults = [[[2 * int(torch.randint(16384 * 64 // 2, (), generator=hash_rng)) + 1 for _ in offsets]
                  for offsets in (s0, s1)] for _, s0, s1 in cfg.ngram_specs]


# ------------------------------------------------------------------------------
# §§ LM Head
# ------------------------------------------------------------------------------

lm_head = fp32_empty(cfg.d_vocab, cfg.d_model).normal_(mean=0.0, std=0.001)

# Per-step multiplier on the head's grad mix-ins (1-beta). Constant: no warmup ramp.
lm_grad_mult_t = np.ones(cfg.num_steps)

peak_lr = 0.004
b2_grad = 0.05   # (1-Beta2)
wd      = 0.0

# Derive the momentum buffer's decay.
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
    lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - embed_b1_mntm_t ** steps_1idx)),

    # Weight decay schedule
    wd_t         = dev(1.0 - lr_mult_t * peak_lr * wd),

    # Beta1 annealed (the mix-in multiplier is all ones), Beta2 constant
    mntm_b1_t    = dev(embed_b1_mntm_t),                           # first_mntm decay (Beta1)
    grad_b1_t    = dev(lm_grad_mult_t * (1.0 - embed_b1_mntm_t)),  # first_mntm grad mix-in (1-Beta1)

    mntm_b2_t    = dev(np.full(cfg.num_steps, b2_mntm)),           # scnd_mntm decay (Beta2)
    grad_b2_t    = dev(lm_grad_mult_t * b2_grad),                  # scnd_mntm grad mix-in (1-Beta2)

    eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
)


# ------------------------------------------------------------------------------
# §§ Decision Readout
# ------------------------------------------------------------------------------
# One vector, dotted with each option's readout token: fp32 live and AdamW, no weight
# decay. Starts small, like the head, so every question starts uniform over its options.
# New in this stage, so its rate is the Phase-0 fork's, without lr_scale.

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
    lr_bc_t      = dev(readout_lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),
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

# Gates start at zero.
head_gate  = fp32_zeros(cfg.n_layers,   cfg.n_qo_heads, cfg.d_gate)
ve_gate    = fp32_zeros(cfg.num_ves,    cfg.n_kv_heads, cfg.d_gate)
ngram_gate = fp32_zeros(cfg.num_ngrams, cfg.n_kv_heads, cfg.d_gate)

# One bank per distinct MLP width, holding that width's layers in order.
mlp_widths = sorted(set(cfg.d_mlp))
W_in  = {d: fp32_empty(cfg.d_mlp.count(d), d, cfg.d_model).uniform_(-matrix_init_s, matrix_init_s) for d in mlp_widths}
W_out = {d: fp32_zeros(cfg.d_mlp.count(d), cfg.d_model, d) for d in mlp_widths}   # projections start at zero

# Muon momentum: warmup from 0.85, then warmdown 0.95 -> 0.85 from 5% of the run.
momentum = 0.85 + (0.95 - 0.85) * np.minimum(steps_0idx / 300, 1.0)
momentum[muon_frac > 0] = 0.95 + (0.85 - 0.95) * muon_frac[muon_frac > 0] ** 2

# NorMuon's second-moment decay anneals 0.95 -> 0.98 across the warmdown.
muon_b2 = 0.95 + (0.98 - 0.95) * muon_frac

# Muon weight decay: linear to zero, with three pulses.
muon_wd = 0.1 * (1.0 - run_frac)
pulse = np.abs(run_frac - 0.015) < 0.005
muon_wd[pulse] *= 3.0
pulse = np.abs(run_frac - 0.03) < 0.01
muon_wd[pulse] *= 5.0
pulse = np.abs(run_frac - 0.8) < 0.025
muon_wd[pulse] *= 1.0 + (6.0 - 1.0) * (1.0 - np.abs(run_frac[pulse] - 0.8) / 0.025)

# Muon peak lr is 0.04; 0.1 on W_in and 0.05 on W_out at every width.
# rdim is the smaller dim of each matrix (-1 on the square ones).
muon_configs = [
#    name,           weights,       peak lr,  rdim
    ("W_Q",          W_Q,           0.04,      -1),
    ("W_K",          W_K,           0.04,      -1),
    ("W_V",          W_V,           0.04,      -1),
    ("W_O",          W_O,           0.04,      -1),
    *[(f"W_in_{d}",  W_in[d],       0.1,       -1) for d in mlp_widths],
    *[(f"W_out_{d}", W_out[d],      0.05,      -2) for d in mlp_widths],
    ("head_gate",    head_gate,     0.04,      -2),
    ("ve_gate",      ve_gate,       0.04,      -2),
    ("ngram_gate",   ngram_gate,    0.04,      -2)
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

        # b1 is the nesterov momentum; b2 is the variance-reduction EMA (both scheduled above).
        mntm_b1_t    = dev(momentum),
        grad_b1_t    = dev(1.0 - momentum),
        mntm_b2_t    = dev(muon_b2),
        grad_b2_t    = dev(1.0 - muon_b2),

        eps_t        = None,
    )

    # Add the parameter to the "model" container.
    setattr(m, name, p)

# The Params own everything now: free the fp32 draws, drop the adopted names.
del lm_head, input_embeds, value_embeds, ngram_values, resid_lambdas, x0_lambdas, x0_gates, pool_lambda, W_Q, W_K, W_V, W_O, head_gate, ve_gate, ngram_gate, W_in, W_out

# Each layer's MLP banks and its slot in them.
mlp_banks = [(getattr(m, f"W_in_{d}"), getattr(m, f"W_out_{d}"), cfg.d_mlp[:i].count(d)) for i, d in enumerate(cfg.d_mlp)]
mlp_params = [getattr(m, f"W_{io}_{d}") for io in ("in", "out") for d in mlp_widths]


# ------------------------------------------------------------------------------
# §§ Rotary Cache
# ------------------------------------------------------------------------------

rotary_seq_len = cfg.train_batch_tokens
channel_range = torch.arange(0, cfg.d_qk, 2, dtype=torch.float32, device=device)  # stride the channels
inv_freq = 1.0 / (3_000_000 ** (channel_range / cfg.d_qk))
t_pos = torch.arange(rotary_seq_len, dtype=torch.float32, device=device)          # stride the time steps
freqs = torch.outer(t_pos, inv_freq)   # rotation frequency at each (time, channel) pair

m.cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(torch.bfloat16)[None, :, None, :]   # add batch and head dims
m.sin = torch.cat([freqs.sin(), -freqs.sin()], dim=-1).to(torch.bfloat16)[None, :, None, :]  # for later broadcasting

del channel_range, inv_freq, t_pos, freqs


# ------------------------------------------------------------------------------
# §§ Init From the MLM Base
# ------------------------------------------------------------------------------
# Every pre-trained weight from the base's checkpoint (its write_checkpoint, as in §§
# Logging: the live weights in model_step*.pt, their fp32 masters' low bits in
# optim_step*.pt); the optimizer state stays fresh and the schedules are this run's. The
# readout is new. The input-side tables grow by the special tokens' rows, each started
# from the mean of the base's rows for the word that names it (a level from its digit,
# <|yes|> from "yes", ...), in the input embeddings and in each value embedding table.

special_init_text = {"<|state|>": "state", "<|noul|>": "yes or no", "<|score|>": "score",
                     "<|choice|>": "choice", '<|"|>': '"', "<|yes|>": "yes", "<|no|>": "no",
                     "<|option|>": "option", **{f"<|level_{i}|>": str(i) for i in range(10)}}

if cfg.init_from:
    from tokenizers import Tokenizer
    model_path = os.path.expanduser(cfg.init_from)
    saved = torch.load(model_path, map_location=device)
    saved_state = torch.load(model_path.replace("model_step", "optim_step"), map_location="cpu")["state"]
    assert [tuple(s) for s in saved["ngram_specs"]] == [tuple(s) for s in cfg.ngram_specs], "n-gram keys differ from the base's"
    assert saved["ngram_mults"] == m.ngram_mults, "n-gram hash multipliers differ from the base's"
    tokenizer = Tokenizer.from_file(os.path.join(DATASET_DIR, "tokenizer/tokenizer.json"))
    for p in m:
        if p.name == "readout":
            continue
        w = saved["weights"][p.name]
        if p.name in ("input_embeds", "value_embeds"):
            # (slots x d_vocab, width) into (slots x d_input_vocab, width), then the specials' rows.
            base_rows = w.view(-1, cfg.d_vocab, w.size(-1))
            rows = p.w.view(base_rows.size(0), cfg.d_input_vocab, -1)
            rows[:, :cfg.d_vocab].copy_(base_rows)
            for name, text in special_init_text.items():
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                rows[:, decision_meta["special_tokens"][name]] = bf16(base_rows[:, ids].float().mean(dim=1))
        else:
            assert w.shape == p.w.shape, (p.name, tuple(w.shape), tuple(p.w.shape))
            p.w.copy_(w)
        if p.mantissa is not None:
            p.mantissa.copy_(saved_state[f"{p.name}.mantissa"])
    print(f"init from {cfg.init_from} (step {saved['step']})")
    del saved, saved_state, w, base_rows, rows

cfg.num_params = sum(p.w.numel() for p in m)


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
    loss:         float | None = None   # MLM, nats per corrupted position
    dec_loss:     float | None = None   # decision, nats per question
    lr_mult_t:    float | None = None
    dt:           float | None = None   # seconds; unset for steps 0-10
    tok_per_sec:  int   | None = None   # "
    mfu:          float | None = None   # "

    # Validation, on val steps only: the MLM base's MLM validation, and accuracy and
    # Brier score (summed over options) on the clean rows of each decision split
    mlm:          float | None = None   # nats per corrupted position
    mbpb:         float | None = None   # bits per byte of the corrupted positions' tokens
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
    "val":  ("mlm", "mbpb", "dec_acc", "dec_brier", "xfer_acc", "xfer_brier", "eval_seconds"),
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
    val_mbpb:           float | None = None
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

def write_checkpoint(step):
    state = {}
    for p in m:
        for attr in ("mantissa", "first_mntm", "scnd_mntm"):
            buf = getattr(p, attr)
            if buf is not None:
                state[f"{p.name}.{attr}"] = buf.cpu()

    ckpt_dir = os.path.expanduser(f"~/.cache/encoder-rtx/runs/{cfg.run_name}")   # outside both repos; nothing outside the cache survives the box
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(dict(step=step, code=code, ngram_specs=cfg.ngram_specs, ngram_mults=m.ngram_mults,
                    weights={p.name: p.w.cpu() for p in m}),
               f"{ckpt_dir}/model_step{step:06d}.pt")
    torch.save(dict(step=step, t_step=int(t_step.item()), state=state),
               f"{ckpt_dir}/optim_step{step:06d}.pt")

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
print0(f"Batch size: {cfg.train_batch_tokens:,} tokens per step, {cfg.train_targets:,} corrupted filler positions "
       f"(mask_rate {cfg.mask_rate}); MLM validation {cfg.val_steps} x {EVAL_BUFFER_TOKENS:,} tokens, {cfg.val_targets:,} corrupted each", console=True)
print0(f"N-gram memories: {cfg.ngram_specs} | {cfg.d_ngram:,} rows x 2 slots each | touched_rows {cfg.touched_rows:,}", console=True)
print0(data_report + f" | packs {packs_time:.1f}s", console=True)
print0("Uniform guessing on the clean questions: " + ", ".join(f"{s} {c:.4f}" for s, c in chance.items()), console=True)

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

# The corruption draws: the training stream from the seed, the validation re-seeded
# per micro-batch so every validation (and every run) corrupts the same positions.
train_gen = torch.Generator(device=device).manual_seed(cfg.seed)
val_gen = torch.Generator(device=device)

# The MLM validation's micro-batches are packs of filler alone: no records, so the cross
# call runs on ghost segments and the merge keeps the self call's rows as they are.
val_rest = dict(pos=torch.arange(EVAL_BUFFER_TOKENS, dtype=torch.int32, device=device),
                is_q=torch.zeros(EVAL_BUFFER_TOKENS, dtype=torch.bool, device=device),
                cu_q=torch.zeros(cfg.rec_width, dtype=torch.int32, device=device),
                cu_k=torch.zeros(cfg.rec_width, dtype=torch.int32, device=device),
                filler_start=torch.zeros(1, dtype=torch.int64, device=device),
                ro_pos=torch.zeros(cfg.ro_cap, dtype=torch.int64, device=device),
                ro_qid=torch.full((cfg.ro_cap,), cfg.q_cap, dtype=torch.int64, device=device),
                ro_tgt=torch.zeros(cfg.ro_cap, dtype=torch.float32, device=device),
                n_q=torch.zeros(1, dtype=torch.float32, device=device))


def validate():
    """The MLM base's validation -- nats per corrupted position and bits per corrupted
    byte, on the same 3.30M positions as the base's own run -- and, on each decision
    split's clean rows, (accuracy, Brier score summed over options, questions, accuracy
    by source, accuracy by variant over every row) from the evaluation packs."""
    val_loader = val_generator(cfg.seq_len, cfg.train_batch_tokens)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    for i in range(cfg.val_steps):
        v_tokens, v_cu_seqlens = next(val_loader)
        v_idx, v_pos, v_labels = corrupt(v_tokens, cfg.val_targets, val_gen.manual_seed(1000 + i))
        nats, _ = forward_backward(Pack(tokens=v_tokens, cu_self=v_cu_seqlens, **val_rest), v_idx, v_pos, v_labels, backward=False)
        total_nats += nats.sum()
        total_bytes += token_bytes[v_labels].sum()
    del val_loader
    val_mlm = total_nats.item() / (cfg.val_steps * cfg.val_targets)
    val_mbpb = total_nats.item() / (math.log(2) * total_bytes.item())

    result = {}
    for split, (packs, scored) in eval_packs.items():
        right = total = 0
        brier = 0.0
        by_source, by_variant = {}, {}
        for i, pk in enumerate(pack_loader(packs)):
            e_idx, e_pos, e_labels = corrupt(pk.tokens, cfg.eval_targets, val_gen.manual_seed(2000 + i), pk.filler_start)
            _, ro_p = forward_backward(pk, e_idx, e_pos, e_labels, backward=False)
            ro_p = ro_p.cpu().numpy()
            k = 0
            for r, q, perm in scored[i]:   # the pack's questions, in readout order
                n = len(perm)
                probs, k = ro_p[k:k + n], k + n
                label = int(np.flatnonzero(perm == q["label"])[0])
                hit = int(probs.argmax() == label)
                variant = by_variant.setdefault(r["variant"], [0, 0])
                variant[0] += hit
                variant[1] += 1
                if r["variant"] != "clean":
                    continue
                right += hit
                total += 1
                brier += float(((probs - np.eye(n)[label]) ** 2).sum())
                source = by_source.setdefault(r["source"], [0, 0])
                source[0] += hit
                source[1] += 1
        result[split] = (right / total, brier / total, total, by_source, by_variant)
    return val_mlm, val_mbpb, result


train_loader = pack_loader(train_packs)
pk = next(train_loader)   # kick off the first batch
idx, pos, labels = corrupt(pk.tokens, cfg.train_targets, train_gen, pk.filler_start)

# The distinct rows one hash touches must fit the sparse update's scratch (rows past
# the bound get no update, silently). Checked on the first batch, with its corruption.
curr = idx.long()
at = lambda off: (curr if off == 0 else
                  torch.cat([curr[:1].expand(-off), curr[:off]]) if off < 0 else
                  torch.cat([curr[off:], curr[-1:].expand(off)]))
ngram_rows_touched = max(
    int(torch.unique(functools.reduce(torch.bitwise_xor, [at(off) * mult for off, mult in zip(offsets, mults)]) % cfg.d_ngram).numel())
    for n, (_, s0, s1) in enumerate(cfg.ngram_specs) for offsets, mults in zip((s0, s1), m.ngram_mults[n]))
del curr, at
print0(f"n-gram rows touched by the first batch, max over slots: {ngram_rows_touched:,} (bound {cfg.touched_rows:,})", console=True)
assert ngram_rows_touched <= cfg.touched_rows, "raise cfg.touched_rows"

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
        val_mlm, val_mbpb, dec = validate()
        val_elapsed = time.perf_counter() - val_t0
        total_val_time += val_elapsed
        stats.mlm, stats.mbpb, stats.eval_seconds = val_mlm, val_mbpb, val_elapsed
        stats.dec_acc, stats.dec_brier, n_dec, *_ = dec["decision_dev"]
        stats.xfer_acc, stats.xfer_brier, n_xfer, *_ = dec["transfer_dev"]
        print0(f"step:{step}/{cfg.num_steps} val_mlm:{val_mlm:.4f} val_mbpb:{val_mbpb:.4f} | decision_dev acc:{stats.dec_acc:.4f} "
               f"brier:{stats.dec_brier:.4f} (n={n_dec}) | transfer_dev acc:{stats.xfer_acc:.4f} "
               f"brier:{stats.xfer_brier:.4f} (n={n_xfer}) | val_time:{val_elapsed:.2f}s", console=True)
        if last_step:
            for split, (*_, by_source, by_variant) in dec.items():
                print0(f"  {split} accuracy by source: " + ", ".join(
                    f"{s} {h}/{n}" for s, (h, n) in sorted(by_source.items())), console=True)
                print0(f"  {split} accuracy by variant (every row): " + ", ".join(
                    f"{v} {h}/{n}" for v, (h, n) in sorted(by_variant.items())), console=True)
        final_val = stats

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

    # Forward and Backward pass
    loss = forward_backward(pk, idx, pos, labels)

    # Next training batch
    pk = next(train_loader)
    idx, pos, labels = corrupt(pk.tokens, cfg.train_targets, train_gen, pk.filler_start)

    # Smooth gradients, update weights

    # Muon
    for p in (m.W_Q, m.W_K, m.W_V, m.W_O, *mlp_params, m.head_gate, m.ve_gate, m.ngram_gate):
        muon_step_fused(p, p.grad, t_step)

    # AdamW
    for p in (m.lm_head, m.readout, m.input_embeds, m.value_embeds, m.resid_lambdas, m.x0_lambdas, \
              m.x0_gates, m.pool_lambda):
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
    wall_time          = (time.perf_counter() - run_wall_t0) / 60,

    avg_step_time      = avg_step_time,
    avg_mfu            = (100 * cfg.num_flops_per_token * cfg.train_batch_tokens
                          / avg_step_time / gpu_peak_flops) if avg_step_time else 0.0,
    peak_mem_gb        = torch.cuda.max_memory_reserved() / 2**30,
)
result.init_time = packs_time / 60
if final_val is not None:
    result.val_mlm, result.val_mbpb = final_val.mlm, final_val.mbpb
    result.dec_acc, result.dec_brier = final_val.dec_acc, final_val.dec_brier
    result.xfer_acc, result.xfer_brier = final_val.xfer_acc, final_val.xfer_brier

row = asdict(result)
with open(result_path, "w") as f:
    json.dump(row, f, indent=1)
wandb_run.log({"step": cfg.num_steps,
               **{f"final/{k}": v for k, v in row.items()
                  if k not in FINAL_IDENTITY and v is not None}})

print0(f"== {result.run_name} ==", console=True)
if result.val_mlm is not None:
    print0(f"  val_mlm {result.val_mlm:.4f} nats per corrupted position | val_mbpb {result.val_mbpb:.4f} | "
           f"decision_dev acc {result.dec_acc:.4f} brier {result.dec_brier:.4f} | "
           f"transfer_dev acc {result.xfer_acc:.4f} brier {result.xfer_brier:.4f} (Kev-0.8B: 0.648)", console=True)
print0(f"  train {result.train_time:.2f}m | val {result.val_time:.2f}m | compile {result.compile_time:.2f}m | init {result.init_time:.2f}m | wall {result.wall_time:.2f}m", console=True)
if timed:
    print0(f"  {cfg.train_batch_tokens:,} tokens/step over {len(timed):,} timed steps: mean {result.avg_step_time:.3f}s | median {np.median(timed):.3f}s | {int(cfg.train_batch_tokens / result.avg_step_time):,} tok/s | mfu {result.avg_mfu:.2f}%", console=True)
print0(f"  rows -> {metrics_path} | result -> {result_path}", console=True)

for _f in (logfile, metrics_path, result_path):
    wandb_run.save(_f, policy="now")
wandb_run.finish()
