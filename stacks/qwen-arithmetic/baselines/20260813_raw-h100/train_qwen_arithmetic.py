# train_qwen_arithmetic.py
#
# Single-file GRPO speedrun: Qwen2.5-0.5B-Instruct on basic arithmetic
# (ChrisMcCormick/basic-arithmetic), 1 GPU (H100 target, A100 supported).
# Handwritten forward/backward (no autograd, no torch.optim, no nn.Module) and
# a CUDA-graph decode engine fused into the training process — one model
# instance, in-place bf16 updates with fp32 masters, so the captured graphs
# never re-capture. At 0.5B the decode step is kernel-count bound, not
# bandwidth bound, so the engine is built around launch-count: compiled decode
# bodies under CUDA graphs, fused QKV and gate/up GEMMs, macro-window replays
# with ONE pinned D2H per window, a coarse row-count bucket ladder, and a
# Gumbel-max sampler (one elementwise pass + argmax). KV is a single static
# allocation — each row permanently owns its pages — with prefix sharing as
# COMPUTE only: every context prefills once (packed varlen) and broadcasts
# into its K sibling rows.
#
# This is qwen-gsm8k's train_qwen_gsm8k.py retargeted at the arithmetic task;
# the model, engine, optimizer and pack machinery are carried over verbatim.
# What changed is the task layer: the reward (last number in the completion,
# plus a method-vocabulary bonus gated on correctness — no \boxed{}, no
# trim-to-answer surgery), the data (a real val split plus in-distribution and
# out-of-distribution test splits, pre-tokenized by data/prepare_arithmetic.py),
# and the eval (GREEDY accuracy, K=1, so the numbers compare directly against
# the reference below).
#
# THE REFERENCE — train_qwen_arithmetic-hf-vllm.py (HF transformers training
# fwd/bwd + in-process vLLM generation + LoRA), same task/model, H100 PCIe:
#   68 rounds of 64 problems x K=16 @ budget 640, LoRA lr 1e-5, temp 0.8
#   val 89.0% @ step 180 | test ID 87.2% | test OOD 86.2% | ~7.5 min wall
# THIS SCRIPT, first run of the defaults below (run raw-v1-lr1e-6, H100 PCIe):
#   val best 94.0% @ round 270 | test ID 89.0% | test OOD 86.75% | 3.7 min loop
#   (median round 0.61s = 0.32 gen + 0.28 train | peak 28.7 GB | val was still
#   climbing at the horizon, so the budget, not the method, set the ceiling)
# This script keeps the qwen-gsm8k ALGORITHM rather than copying the
# reference's: on-policy, ONE optimizer step per round, no PPO ratio/clip, no
# KL, no reference model, token-mean loss, full-parameter AdamW at a constant
# lr. Same data budget as the reference (rounds_cap 272 x 256 rollouts =
# 69,632 rollouts = 68 x 1,024), so the accuracy-vs-wall-clock comparison is
# like for like.
#   - sampler: temperature 1.0, no top-k, no top-p  ->  Gumbel-max over logits
#     (the reference sampled at 0.8 with top-p 0.9; the raw sampler is
#     deliberately sort-free, and temp 1.0 is the qwen-gsm8k/verl heritage)
#   - GRPO advantage: per-group (r - mean) / (std_{ddof=1} + 1e-6), z-scored
#   - loss: token-mean over ALL response tokens in the round
#   - AdamW lr 1e-6 constant, betas (0.9, 0.999), eps 1e-8, wd 0.01, one
#     optimizer step per round, ALL params trained. UNSWEPT for this task —
#     the reference trains LoRA at 1e-5; full-param wants smaller, and this
#     is the first knob worth sweeping.
#   - reward: 1.0 * correct + 1.0 * uses_method (see § Reward). Rewards are
#     {0, 1, 2}-valued; the group z-score doesn't care.
#   - ONE optimizer-machinery departure from qwen-gsm8k: the fp32 masters
#     start MID-BIN (mantissa 0x8000) instead of on the bf16 bin edge. The
#     all-zero init turns the run's first update into a ~2^-9-relative signSGD
#     kick that this task does not survive (val 61.5% -> 0.5% in 3 rounds,
#     measured) — TECHNIQUES.md § The mantissa first-step kick.
#
# Qwen2.5-0.5B architecture (config.json, asserted at load):
#   24 layers, d_model 896, 14 Q heads / 2 KV heads, head_dim 64, MLP 4864
#   (SwiGLU), vocab 151,936, rope_theta 1e6, rms eps 1e-6, QKV biases (o/mlp
#   none), TIED embeddings (embed table == lm_head), no qk-norm.
#
# Run (after clone + cd into this folder):
#   bash setup.sh                        # uv env + prepare_model + prepare_arithmetic
#   python train_qwen_arithmetic.py      # reads ~/.cache/qwen-arithmetic/data/
# There is no command line and no config env: every knob is a field of
# ArithConfig in § Config, edited in place. Useful modes: `host_test`
# (host-only self-tests, no GPU), `fixed_problems` + `rounds_cap`
# (single-problem overfit smoke), `eval_every = 0` + `final_eval = False`
# (train-only).
# Telemetry: one console line per round, full rows to wandb (`wandb = False`
# disables) and metrics_<tag>.csv / evals_<tag>.csv / evals_detail_<tag>.csv /
# result_<tag>.json.
#
# Tricky invariants live in TECHNIQUES.md (this folder: § The mantissa
# first-step kick; pack layout: qwen-gsm8k's § Padded varlen); the code points
# at the section by name.
#
# Env needs: torch + kernels (FA3/FA2 via the HF kernels hub — no wheel
# builds), tokenizers, safetensors, pyarrow, numpy, wandb — see pyproject.toml.

# --------------------------------------------------------------------------------
# § Setup (host-safe: everything above the host_test gate runs without a GPU)
# --------------------------------------------------------------------------------

import os
import sys
import time as _time
run_wall_t0 = _time.perf_counter()
del _time

with open(sys.argv[0], "r") as f:
    code = f.read()   # logged into checkpoints for provenance

import csv
import gc
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------------
# § Config
# --------------------------------------------------------------------------------
# Defaults keep qwen-gsm8k's algorithm at the reference's data budget; change
# them only as labelled arms. Every knob lives here — there is no command line
# and no env override, so a run is defined by the source, and the source is
# archived into every checkpoint (`code`) and the wandb config.

class ArithConfig:

    # Run identity
    tag:      str = "run"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Data — pre-tokenized by data/prepare_arithmetic.py, which owns the prompt
    # rendering (the dataset's own ChatML text, re-verified against a local
    # render) and the split layout: train 10,000 / val 200 / test_id 400 /
    # test_ood 400
    data_dir: Path = Path.home() / ".cache" / "qwen-arithmetic" / "data"

    # Model (Qwen2.5-0.5B — asserted against the checkpoint's config.json at load)
    n_layers:   int = 24
    d_model:    int = 896
    n_qo_heads: int = 14
    n_kv_heads: int = 2
    d_head:     int = 64
    d_mlp:      int = 4864
    d_vocab:    int = 151936          # 2374 x 64 — arrives padded
    rope_theta: float = 1_000_000.0
    rms_eps:    float = 1e-6

    d_q:   int   # Derived below: the fused QKV column split, [Q | K | V].
    d_kv:  int
    d_qkv: int
    half:  int   # Derived below: rotary splits the head dim in half.

    # Rollouts / rounds
    k_draws:            int   = 16    # rollouts per problem per round
    problems_per_round: int   = 16
    epochs:             int   = 1
    rounds_cap:         int   = 272   # = the reference's 68 x 1,024 rollouts at
                                      # this round shape; 0 = the full `epochs`
                                      # horizon (625 rounds)
    max_tokens:         int   = 640   # sized to clear the UNTRAINED model's p99
                                      # (~512): a truncated rollout loses its
                                      # final number, scores 0, and teaches
                                      # 'shorter is safer' exactly while the
                                      # model is still learning what to write.
                                      # Trained, p99 falls to ~75 and truncation
                                      # sits at 0-1%.
    temperature:        float = 1.0
    seed:               int   = 1337  # round schedule + sampler RNG

    # Reward weights (§ Reward). uses_method is gated on correct, so rewards
    # take values in {0, w_correct, w_correct + w_method}.
    w_correct: float = 1.0
    w_method:  float = 1.0

    # Optimizer — qwen-gsm8k's verl-style actor defaults (constant LR)
    lr:           float = 1e-6
    weight_decay: float = 0.01
    beta1:        float = 0.9
    beta2:        float = 0.999
    adam_eps:     float = 1e-8
    lr_schedule:  str   = "const"     # const | linear (->0)

    # Training packs (packed varlen, one compiled shape)
    train_t:  int = 16384             # tokens per pack (fixed compile shape)
    sel_cap:  int = 12288             # completion positions per pack, lm_head
                                      # runs ONLY on these (prompt tokens have
                                      # zero grad at the head; ~27% of step FLOPs
                                      # x the skipped fraction is free)
    ce_chunk: int = 2048              # lm_head/CE row chunk (152k vocab: a full
                                      # (SEL, V) fp32 chain would be ~7 GB/temp)
    max_docs: int = 160               # cu_seqlens fixed size (ghost-padded).
                                      # Arithmetic docs are SHORT (~100-170 tok
                                      # incl. prompt), so this is sized so the
                                      # token/sel caps bind before the doc count
                                      # does; gsm8k's 96 would cap a pack first.

    # Engine
    max_seqs:  int   = 256            # decode rows (= top bucket)
    macro_n:   int   = 8              # decode steps per window (one D2H each)
    buckets:   tuple = (32, 64, 128, 192, 256)
    prefill_t: int   = 0              # 0 = auto-size (see § Data)
    max_ctxs:  int   = 64             # prefill cu fixed size. 64 (vs gsm8k's 48)
                                      # because eval is K=1: a wave carries one
                                      # row per context, so max_ctxs is what
                                      # sets eval batch width.
    page:      int   = 256            # FA2 paged KV page size (multiple of 256 required)

    max_prompt:    int   # Derived in § Data: the longest rendered prompt.
    gen_steps:     int   # Derived in § Data: max decode steps a row can run.
    t_row:         int   # Derived in § Data: decode row capacity, whole pages.
    pages_per_row: int
    rope_t:        int   # Derived in § Model Load: rotary cache length.

    # Eval — GREEDY (K=1) accuracy through the graphs, directly comparable to
    # the hf-vllm reference's greedy eval. In-loop: the 200-problem val split.
    # Final: test_id + test_ood (400 each).
    eval_every: int  = 30             # rounds between val evals; 0 = off
    final_eval: bool = True

    # Checkpoints / output
    save_every:    int  = 0           # rounds; 0 = final only
    run_root:      Path = Path.home() / ".cache" / "qwen-arithmetic" / "runs"
    run_dir:       Path               # Derived below: run_root / tag.
    wandb:         bool = True        # per-round rows + evals + final summary
    wandb_run:     str  = ""          # "" = use tag
    wandb_project: str  = "qwen-arithmetic"
    mem_every:     int  = 50          # rounds between cudaMemGetInfo samples — it
                                      # costs real ms against a busy process

    # Smokes
    fixed_problems: list[int] | None = None   # overfit this fixed problem set instead
    profile:        bool = False      # chrome trace (ui.perfetto.dev)
    prof_wait:      int = 3
    prof_active:    int = 1
    host_test:      bool = False      # host-only self-tests, then exit (no GPU)

cfg = ArithConfig() # Make config a global, don't pass it around.

# Sanity: the constraints the knobs above must satisfy.
assert cfg.lr_schedule in ("const", "linear"), f"bad lr_schedule {cfg.lr_schedule!r}"
assert cfg.problems_per_round * cfg.k_draws <= cfg.max_seqs, "round rows exceed max_seqs"
assert cfg.max_seqs == max(cfg.buckets), "top bucket must equal max_seqs"
assert cfg.sel_cap % cfg.ce_chunk == 0, "sel_cap must divide into CE chunks"
assert cfg.sel_cap <= cfg.train_t

# Derived quantities:
# QKV fuses into ONE (d_qkv, d_model) GEMM per layer, gate/up into one
# (2*d_mlp, d_model) — ~72 fewer kernels per decode step, and the GEMMs sit
# below the launch floor so the fusion is ~free throughput.
cfg.d_q   = cfg.n_qo_heads * cfg.d_head    # 896
cfg.d_kv  = cfg.n_kv_heads * cfg.d_head    # 128
cfg.d_qkv = cfg.d_q + 2 * cfg.d_kv         # 1152 — fused QKV rows: [Q | K | V]
cfg.half  = cfg.d_head // 2

# Derived run paths.
cfg.run_dir   = cfg.run_root / cfg.tag
cfg.wandb_run = cfg.wandb_run or cfg.tag


def config_dict() -> dict:
    """Every cfg field — class defaults and derived alike — flattened for the
    wandb config. The annotations carry declaration order."""
    vals = ((k, getattr(cfg, k, None)) for k in ArithConfig.__annotations__)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vals}


# --------------------------------------------------------------------------------
# § Stats — the one row every sink reports
# --------------------------------------------------------------------------------
# Generation, grading and the training step each write their slice of the
# round's row into this single global; the CSV writer, the wandb row and the
# console line then all read the same object, so the metric set is defined
# once (the CSV header IS the field list) and no sink can drift from another.
# Values land already rounded for display. The loop rebinds `stats` to a fresh
# instance per round, so a field nobody wrote reports its default rather than
# last round's value.

@dataclass
class RoundStats:

    round: int = 0

    # Rollouts + grading (§ Main Loop)
    n_rollouts: int   = 0
    n_correct:  int   = 0
    solve_rate: float = 0.0
    n_eos:      int   = 0
    n_trunc:    int   = 0
    method_pct: float = 0.0  # uses_method share — the shaped half of the
                             # reward, and where reward hacking would show first

    # Generation (Engine.run_round)
    gen_s:         float = 0.0
    gen_tok:       int   = 0
    gen_tok_per_s: float = 0.0
    occ:           float = 0.0  # % of paid decode slots that kept their token
    t50:           float = 0.0  # seconds to retire half the rows ...
    t90:           float = 0.0  # ... and 90% of them (the tail's cost)

    # Training step (train_step)
    train_s:       float = 0.0
    n_groups_used: int   = 0
    n_groups_sat:  int   = 0
    n_groups_dead: int   = 0
    n_docs:        int   = 0
    n_loss_tokens: int   = 0
    n_packs:       int   = 0
    pad_pct:       float = 0.0
    loss_total:    float = 0.0
    grad_norm:     float = 0.0

    # Round bookkeeping (§ Main Loop)
    lr:      float = 0.0
    mem_gb:  float = 0.0
    round_s: float = 0.0

stats = RoundStats() # Make stats a global, don't pass it around.


# --------------------------------------------------------------------------------
# § Reward — last number + method vocabulary, both from the hf-vllm reference
# --------------------------------------------------------------------------------
# The answer channel: the LAST number anywhere in the completion is the model's
# answer, compared as an integer against gold. No format demand — this is the
# scorer the whole experiment line (and its baselines) used, so the accuracy
# numbers stay comparable.
#
# The method channel: correctness alone pays for NOT working the problem — a
# short answer has fewer places to go wrong — so a second reward names the
# vocabulary of a written method and hard-zeros the phrasings that dodge one.
# Gated on correctness on purpose: rewarding method words on a wrong answer
# would pay for reciting the ritual without doing the arithmetic. Measured in
# the reference line: method reward at weight 1.0 beat 0.5 beat absent
# (val 90.5 / 89.0 / 85.5 at matched budgets).
#
# NOT carried over from qwen-gsm8k: trim-to-answer surgery. Its anchor was the
# \boxed{} span; "the last number" anchors nothing (every number is
# potentially the answer), and at budget 640 truncation is 0-1% — the
# attractor the surgery drains doesn't form here. Overlong rollouts are simply
# scored like any other, which the reference measured BETTER than masking them
# (test ID 87.2 vs 76.5).

_ALL_NUMS = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def last_number(text):
    """The final number in the text as an int, or None.

    The regex accepts scientific notation, so a rollout that ends in "1e999"
    parses to inf and int(inf) raises OverflowError — which killed a reference
    run at step 176. A number we cannot turn into an int is simply not the
    answer."""
    matches = _ALL_NUMS.findall(text)
    if not matches:
        return None
    try:
        return int(float(matches[-1]))
    except (OverflowError, ValueError):
        return None


def reward_correct(text, gold: int) -> float:
    """1.0 if the completion's final number matches the gold integer."""
    return 1.0 if last_number(text) == gold else 0.0


_METHOD_GOOD = re.compile(r"\b(?:method|place|align|long|distributive)\b", re.IGNORECASE)
_METHOD_BAD = re.compile(r"\b(?:simply|calculator|directly)\b", re.IGNORECASE)


def reward_method(text, correct: float) -> float:
    """1.0 for a correct answer that shows a method; 0.0 if it dodges one."""
    if _METHOD_BAD.search(text):
        return 0.0
    return 1.0 if (correct and _METHOD_GOOD.search(text)) else 0.0


def grade(text, gold: int) -> tuple[float, float, float]:
    """(total reward, correct, method) for one completion."""
    r_c = reward_correct(text, gold)
    r_m = reward_method(text, r_c)
    return cfg.w_correct * r_c + cfg.w_method * r_m, r_c, r_m


# --------------------------------------------------------------------------------
# § Data — pre-tokenized arithmetic + balanced round schedule
# --------------------------------------------------------------------------------
# Prompts arrive already rendered and tokenized. Everything tokenizer-shaped —
# the ChatML render, the gold integer — is decided by data/prepare_arithmetic.py
# and frozen into a parquet, so a prompt change is a dataset change (new file,
# new ids hash) rather than a silent edit inside the training loop.

print(f"[{cfg.tag}] loading pre-tokenized arithmetic ...", flush=True)
import pyarrow.parquet as pq
from tokenizers import Tokenizer as _RustTokenizer

# IM_END ends the assistant turn; a sampled ENDOFTEXT ends the document — both
# retire a decode row, and ENDOFTEXT doubles as the pack pad.
IM_END = 151645       # <|im_end|>
ENDOFTEXT = 151643    # <|endoftext|>
TERMINALS = (IM_END, ENDOFTEXT)
PAD_ID = ENDOFTEXT

_BUILD_IT = "    python data/prepare_arithmetic.py"
_BUILD_MODEL = "    python data/prepare_model.py"
_data = cfg.data_dir / "arithmetic.parquet"
assert _data.exists(), f"{_data} not found — build it with:\n{_BUILD_IT}"
_t = pq.read_table(_data).to_pydict()
SPLITS = ("train", "val", "test_id", "test_ood")
_rows = {s: {} for s in SPLITS}
for _s, _i, _gold, _ids in zip(_t["split"], _t["idx"], _t["gold"], _t["prompt_ids"]):
    _rows[_s][_i] = (int(_gold), _ids)
assert all(_rows[s] for s in SPLITS), f"{_data} is missing a split — rebuild it with:\n{_BUILD_IT}"

def _cols(split):
    rows = [_rows[split][i] for i in range(len(_rows[split]))]
    return [r[0] for r in rows], [r[1] for r in rows]

train_gold, train_prompts = _cols("train")
val_gold, val_prompts = _cols("val")
test_id_gold, test_id_prompts = _cols("test_id")
test_ood_gold, test_ood_prompts = _cols("test_ood")

# Decode only: the prompts arrive tokenized, so this reads completions back for
# the reward. It is the copy data/prepare_model.py left beside the weights —
# the same file that produced the prompt ids.
_tok_file = cfg.data_dir / "tokenizer.json"
assert _tok_file.exists(), f"{_tok_file} not found — build it with:\n{_BUILD_MODEL}"
tokenizer = _RustTokenizer.from_file(str(_tok_file))
assert (tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")) \
    == (IM_END, ENDOFTEXT), "tokenizer disagrees on the Qwen special ids"


def decode(ids: list[int]) -> str:
    """`tokenizers` defaults skip_special_tokens=True; keep it False so a
    decode is a faithful readback of exactly the ids the engine produced."""
    return tokenizer.decode(ids, skip_special_tokens=False)


def encode(text: str) -> list[int]:
    """Host-test helper. add_special_tokens=False: these are raw completion
    fragments, not turns — nothing may be prepended."""
    return tokenizer.encode(text, add_special_tokens=False).ids


_prompt_lens = [len(p) for p in train_prompts + val_prompts
                + test_id_prompts + test_ood_prompts]
cfg.max_prompt = max(_prompt_lens)
assert cfg.max_prompt <= 256, f"prompt of {cfg.max_prompt} tokens — arithmetic prompts are short"
assert min(_prompt_lens) >= 2, "prompt too short for the forced-last-token split"


def assemble_rounds(n_problems: int, ppr: int, epochs: int, rng: random.Random) -> list[list[int]]:
    """Balanced round schedule: per epoch, sort problems by context length and
    snake-deal into bins of `ppr` — every bin's context sum lands near the mean
    (so one prefill_t covers every round with little padding) AND every bin
    draws one problem from each length stratum (variety). Bin order shuffled.
    The remainder problems (n % ppr) are dropped each epoch."""
    rounds = []
    for _ in range(epochs):
        idx = list(range(n_problems))
        rng.shuffle(idx)
        n_bins = len(idx) // ppr
        idx = idx[:n_bins * ppr]
        idx.sort(key=lambda i: len(train_prompts[i]), reverse=True)
        bins = [[] for _ in range(n_bins)]
        for j, i in enumerate(idx):
            lap, off = divmod(j, n_bins)
            bins[off if lap % 2 == 0 else n_bins - 1 - off].append(i)
        rng.shuffle(bins)
        rounds.extend(bins)
    return rounds


_rng = random.Random(cfg.seed)
if cfg.fixed_problems is not None:
    round_schedule = None
    num_rounds = cfg.rounds_cap or 100
    round_ctx_max = sum(len(train_prompts[i]) - 1 for i in cfg.fixed_problems)
else:
    round_schedule = assemble_rounds(len(train_prompts), cfg.problems_per_round, cfg.epochs, _rng)
    num_rounds = len(round_schedule)
    if cfg.rounds_cap:
        num_rounds = min(num_rounds, cfg.rounds_cap)
        round_schedule = round_schedule[:num_rounds]
    round_ctx_max = max(sum(len(train_prompts[i]) - 1 for i in r) for r in round_schedule)
rounds_per_epoch = max(1, (len(train_prompts) // cfg.problems_per_round) if cfg.fixed_problems is None else num_rounds)

# Prefill pack: one static compiled shape covers both the train rounds and the
# eval waves (which pack themselves under the same cap).
if not cfg.prefill_t:
    cfg.prefill_t = -(-max(round_ctx_max, 4096) // 64) * 64
assert cfg.prefill_t >= round_ctx_max, f"cfg.prefill_t={cfg.prefill_t} < longest round ctx {round_ctx_max}"

# Decode row capacity, rounded up to whole KV pages. A row's K lands at
# positions [ctx, ctx + steps): the first decode step writes the FORCED token,
# and a window always replays macro_n times, so a row runs a macro-ALIGNED
# budget before the host can retire it. That rounding is the real bound.
cfg.gen_steps = -(-cfg.max_tokens // cfg.macro_n) * cfg.macro_n
cfg.t_row = -(-(cfg.max_prompt - 1 + cfg.gen_steps) // cfg.page) * cfg.page
cfg.pages_per_row = cfg.t_row // cfg.page

print(f"[{cfg.tag}] {cfg.problems_per_round} problems x K={cfg.k_draws} = {cfg.problems_per_round * cfg.k_draws} rollouts/round "
      f"x {num_rounds} rounds @ budget {cfg.max_tokens} | max prompt {cfg.max_prompt} | "
      f"prefill T {cfg.prefill_t} (longest round {round_ctx_max}) | "
      f"row {cfg.t_row} tok = {cfg.pages_per_row} pages", flush=True)


# --------------------------------------------------------------------------------
# § Advantage + pack planning (host)
# --------------------------------------------------------------------------------

def group_advantage(rewards) -> np.ndarray | None:
    """GRPO advantage: (r - mean) / (std + 1e-6), std with ddof=1 (the verl
    convention qwen-gsm8k matched; torch.std is Bessel-corrected). None for an
    all-equal group — the advantage is exactly zero everywhere, so the docs
    carry no gradient and are skipped outright (their tokens still count in
    the token-mean normalizer, which includes every response token in the
    batch)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2 or (r == r[0]).all():
        return None
    return (r - r.mean()) / (r.std(ddof=1) + 1e-6)


def plan_packs(docs: list[tuple[list[int], list[int], float]]):
    """docs: (prompt_ids, gen_ids, weight) — weight is the per-token loss
    coefficient (advantage / round-total response tokens), applied to every
    completion target of the doc. Returns (packs, pack_stats); each pack is a
    dict of numpy arrays at the ONE compiled shape:

      idx (train_t,) int32     packed inputs (per doc: seq[:-1])
      pos (train_t,) int64     rotary positions, restarting at each doc
      cu  (max_docs+2,) int32  doc boundaries, ghost-padded with train_t
                               (zero-length trailing segments)
      sel (sel_cap,) int64     positions of completion targets (lm_head runs
                               only here); padded with 0
      tgt (sel_cap,) int64     targets at sel; padded with 0
      w   (sel_cap,) fp32      per-token loss weight at sel; padded with 0 (a
                               zero weight zeroes the padded entries' gradient,
                               so the duplicate position-0 entries are inert)

    First-fit-decreasing over three caps: train_t tokens, sel_cap completion
    positions, max_docs docs. The pad tail is a real attended segment carrying
    zero weight — see qwen-gsm8k's TECHNIQUES.md § Padded varlen."""
    order = sorted(range(len(docs)), key=lambda i: -(len(docs[i][0]) + len(docs[i][1])))
    packs_docs, packs_tok, packs_selc = [], [], []
    for i in order:
        p, g, _ = docs[i]
        n_tok, n_sel = len(p) + len(g) - 1, len(g)
        assert n_tok <= cfg.train_t and n_sel <= cfg.sel_cap, "doc exceeds pack caps"
        for j in range(len(packs_docs)):
            if (packs_tok[j] + n_tok <= cfg.train_t and packs_selc[j] + n_sel <= cfg.sel_cap
                    and len(packs_docs[j]) < cfg.max_docs):
                packs_docs[j].append(i)
                packs_tok[j] += n_tok
                packs_selc[j] += n_sel
                break
        else:
            packs_docs.append([i])
            packs_tok.append(n_tok)
            packs_selc.append(n_sel)

    packs = []
    pad_tokens = 0
    for members in packs_docs:
        idx = np.full(cfg.train_t, PAD_ID, dtype=np.int32)
        pos = np.zeros(cfg.train_t, dtype=np.int64)
        cu = np.full(cfg.max_docs + 2, cfg.train_t, dtype=np.int32)
        cu[0] = 0
        sel = np.zeros(cfg.sel_cap, dtype=np.int64)
        tgt = np.zeros(cfg.sel_cap, dtype=np.int64)
        w = np.zeros(cfg.sel_cap, dtype=np.float32)
        o = s = 0
        for n_doc, i in enumerate(members):
            p, g, wt = docs[i]
            seq = np.asarray(p + g, dtype=np.int64)
            n = len(seq) - 1
            idx[o:o + n] = seq[:-1]
            pos[o:o + n] = np.arange(n)
            ng = len(g)
            sel[s:s + ng] = o + n - ng + np.arange(ng)   # targets seq[1:]; the
            tgt[s:s + ng] = seq[-ng:]                    # last ng are the completion
            w[s:s + ng] = wt
            s += ng
            o += n
            cu[n_doc + 1] = o
        if o < cfg.train_t:
            # The pad tail is its OWN attended segment, and its ids/positions
            # must VARY — either one wrong NaNs every weight grad while the
            # loss stays finite (TECHNIQUES.md § Padded varlen).
            n_pad = cfg.train_t - o
            idx[o:] = 1 + (np.arange(n_pad) % 4096)
            pos[o:] = np.arange(n_pad) % cfg.t_row
            cu[len(members) + 1] = cfg.train_t
            pad_tokens += n_pad
        assert int(pos.max()) < cfg.t_row, "doc position exceeds the rotary cache"
        packs.append(dict(idx=idx, pos=pos, cu=cu, sel=sel, tgt=tgt, w=w,
                          n_tok=o, n_sel=s, n_docs=len(members)))
    return packs, dict(n_packs=len(packs), pad_tokens=pad_tokens,
                       cap_tokens=cfg.train_t * max(1, len(packs_docs)))


# --------------------------------------------------------------------------------
# § Window-event math (host) — the retirement scan's per-window decision
# --------------------------------------------------------------------------------

def window_events(t_live: np.ndarray, allows: np.ndarray, base: int, terminals):
    """One macro-window's retirements, vectorized. t_live (n, N) sampled tokens
    for the live rows; allows (n,) per-row budgets; base = tokens generated
    before this window. Returns (done, eos, n_take): rows retiring this window,
    whether by terminal (vs budget), and how many of the window's tokens they
    keep. A terminal AT the budget position still counts as eos (the terminal
    check runs first)."""
    N = t_live.shape[1]
    hit = np.isin(t_live, terminals)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), N)   # in-window terminal idx
    bidx = allows - base - 1                                   # in-window budget idx
    eos = first <= np.minimum(bidx, N - 1)
    trunc = (~eos) & (bidx <= N - 1)
    done = eos | trunc
    n_take = np.where(eos, first + 1, np.minimum(bidx + 1, N)).astype(np.int64)
    return done, eos, n_take


# --------------------------------------------------------------------------------
# § host_test gate — everything below requires a GPU
# --------------------------------------------------------------------------------

if cfg.host_test:
    print("[host-test] reward ...")
    assert last_number("12 + 5 gives us 17.") == 17
    assert last_number("So 170 / 2 = 85.0") == 85
    assert last_number("the result is -42") == -42
    assert last_number("no numerals here") is None
    assert last_number("overflowing 1e999") is None    # inf is not an answer
    assert reward_correct("The answer is 85.", 85) == 1.0
    assert reward_correct("I get 84. No wait, 85.", 85) == 1.0   # last number wins
    assert reward_correct("85 is close but I'll say 84.", 85) == 0.0
    assert reward_method("Using long division, 85.", 1.0) == 1.0
    assert reward_method("Aligning by place value: 85.", 1.0) == 1.0
    assert reward_method("Simply, it's 85.", 1.0) == 0.0         # dodge word zeroes it
    assert reward_method("Using long division, 84.", 0.0) == 0.0 # gated on correct
    assert grade("By the standard method: 85.", 85) == (2.0, 1.0, 1.0)
    assert grade("85", 85) == (1.0, 1.0, 0.0)
    assert grade("simply 85", 85) == (1.0, 1.0, 0.0)
    assert grade("simply 84", 85) == (0.0, 0.0, 0.0)

    print("[host-test] advantage ...")
    assert group_advantage([1.0] * 16) is None and group_advantage([0.0] * 16) is None
    assert group_advantage([2.0] * 16) is None       # all correct+method: saturated too
    a = group_advantage([1, 0, 0, 0])
    ref = (np.array([1, 0, 0, 0]) - 0.25) / (np.array([1., 0, 0, 0]).std(ddof=1) + 1e-6)
    assert np.allclose(a, ref)
    assert group_advantage([2, 1, 0, 0]) is not None  # three-valued rewards work

    print("[host-test] pack planning ...")
    rng = np.random.default_rng(0)
    docs = [(list(rng.integers(1, 1000, rng.integers(30, 80))),
             list(rng.integers(1, 1000, rng.integers(5, cfg.max_tokens))) + [IM_END],
             float(rng.normal())) for _ in range(256)]
    packs, st = plan_packs(docs)
    tot_sel = sum(p["n_sel"] for p in packs)
    assert tot_sel == sum(len(g) for _, g, _ in docs)
    for p in packs:
        assert p["n_tok"] <= cfg.train_t and p["n_sel"] <= cfg.sel_cap and p["n_docs"] <= cfg.max_docs
        ends = p["cu"][1:1 + p["n_docs"] + (1 if p["n_tok"] < cfg.train_t else 0)]
        assert (np.diff(p["cu"].astype(np.int64)) >= 0).all() and p["cu"][-1] == cfg.train_t
        # every selected target equals the packed stream shifted by one
        live = p["w"] != 0
        s_idx = p["sel"][live]
        interior = s_idx[(s_idx + 1) % cfg.train_t != 0]
        nxt = p["idx"][interior + 1]
        d_end = np.isin(interior + 1, ends)      # at doc seams the next input is
        assert (p["tgt"][live][(s_idx + 1) % cfg.train_t != 0][~d_end]
                == nxt[~d_end]).all()            # the next doc — not a target

    print("[host-test] window events ...")
    TERM = np.array(TERMINALS)
    t = np.array([[5, 5, IM_END, 5, 5, 5, 5, 5],      # eos at idx 2
                  [5, 5, 5, 5, 5, 5, 5, 5],           # runs on
                  [5, 5, 5, 5, 5, 5, 5, IM_END]])     # eos at last slot
    done, eos, n_take = window_events(t, np.array([512, 24, 24]), 16, TERM)
    assert done.tolist() == [True, True, True]
    assert eos.tolist() == [True, False, True]        # row 1 truncates at budget 24
    assert n_take.tolist() == [3, 8, 8]               # row 2: terminal AT budget = eos
    done, eos, n_take = window_events(t[[1]], np.array([512]), 16, TERM)
    assert not done[0]

    print("[host-test] schedule + data shape ...")
    assert round_schedule is None or all(len(r) == cfg.problems_per_round for r in round_schedule)
    lens = sorted(len(p) for p in train_prompts)
    print(f"  prompt tokens: min {lens[0]} med {lens[len(lens)//2]} max {lens[-1]} | "
          f"rounds {num_rounds} | prefill_t {cfg.prefill_t} | t_row {cfg.t_row}")
    print(f"  val {len(val_prompts)} | test_id {len(test_id_prompts)} | "
          f"test_ood {len(test_ood_prompts)} | gold sample: {train_gold[0]}")
    print("[host-test] ALL PASS")
    sys.exit(0)


# --------------------------------------------------------------------------------
# § CUDA init + Flash Attention kernels
# --------------------------------------------------------------------------------
# Training fwd/bwd uses FA3's RAW varlen ops — they return the softmax LSE the
# handwritten backward needs. Generation uses FA2 (varlen prefill + paged
# kvcache decode): FA2's paged single-query decode outruns the sm80 FA3
# build's, and at a step this launch-bound sm90 gains nothing from FA3 either.
# Both come off the HF kernels hub — no wheel builds. (Never let a flash-attn
# WHEEL import into this process: its op registrations collide with the
# kernels-hub build and the next varlen call dies.)

assert torch.cuda.is_available(), "CUDA required (set cfg.host_test for the host-only checks)"
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

import torch._dynamo as dynamo
dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, len(cfg.buckets) + 16)
for _attr in ("cache_size_limit", "accumulated_cache_size_limit"):
    if hasattr(dynamo.config, _attr):
        setattr(dynamo.config, _attr, max(getattr(dynamo.config, _attr), len(cfg.buckets) + 16))

from kernels import get_kernel

# kernels>=0.15 requires an explicit major version (or revision=...).
_cc_major, _ = torch.cuda.get_device_capability()
if _cc_major == 9:   # Hopper: prefer varunneal (better H100); fall back to community
    try:
        fa3 = get_kernel("varunneal/flash-attention-3", version=1).flash_attn_interface
        RAW_BWD_TAKES_BUFFERS = False   # raw backward allocates and RETURNS dq/dk/dv
    except Exception:
        _k = get_kernel("kernels-community/flash-attn3", version=1)
        fa3 = getattr(_k, "flash_attn_interface", _k)
        RAW_BWD_TAKES_BUFFERS = True    # community raw bwd takes pre-allocated buffers
else:                # Ampere sm80/86 / Ada sm89: community FA3 build
    assert _cc_major == 8, f"FA3 required (sm8x or sm90); got sm{_cc_major}x"
    _k = get_kernel("kernels-community/flash-attn3", version=1)
    fa3 = getattr(_k, "flash_attn_interface", _k)
    RAW_BWD_TAKES_BUFFERS = True    # raw backward takes pre-allocated dq/dk/dv buffers

_fa2 = get_kernel("kernels-community/flash-attn2", version=1)
_fa2i = _fa2 if hasattr(_fa2, "flash_attn_varlen_func") else _fa2.flash_attn_interface
flash_attn_varlen_func = _fa2i.flash_attn_varlen_func
_fa_kvcache_raw = _fa2i.flash_attn_with_kvcache


def flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, max_seqlen):
    """FA3 varlen forward returning (out, softmax_lse) for the handwritten
    backward. Full causal — window (max_seqlen, 0) is unlimited in effect since
    varlen attention is doc-isolated and docs are <= max_seqlen."""
    out, softmax_lse, *_ = fa3._flash_attn_forward(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=q.shape[-1] ** -0.5, causal=True,
        window_size_left=max_seqlen, window_size_right=0)
    return out, softmax_lse


def flash_attn_varlen_bwd(dout, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen):
    """Backward for flash_attn_varlen_fwd_lse -> (dq, dk, dv). The two FA3
    builds' raw ops differ in calling convention — the sm80 build takes
    pre-allocated dq/dk/dv buffers, the sm90 build allocates and returns."""
    softmax_scale = q.shape[-1] ** -0.5
    if RAW_BWD_TAKES_BUFFERS:
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        fa3._flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            cu_seqlens, cu_seqlens, None, None,
            max_seqlen, max_seqlen,
            dq, dk, dv,
            softmax_scale, True, max_seqlen, 0, 0.0, False, 0)
    else:
        dq, dk, dv, _ = fa3._flash_attn_backward(
            dout, q, k, v, out, softmax_lse,
            cu_seqlens, cu_seqlens, None, None,
            max_seqlen, max_seqlen,
            softmax_scale, True, max_seqlen, 0, 0.0, False, 0)
    return dq, dk, dv


# fa_kvcache wrapper: the FA2 build registers no fake impl for its kvcache op,
# so a direct call dies under fake-tensor tracing (varlen IS fake-safe, hence
# prefill calls it directly).
@torch.library.custom_op("qwen_arithmetic::fa_kvcache_paged", mutates_args=("k_cache", "v_cache"))
def fa_kvcache_paged(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                     k: torch.Tensor, v: torch.Tensor, cache_seqlens: torch.Tensor,
                     block_table: torch.Tensor) -> torch.Tensor:
    return _fa_kvcache_raw(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                           block_table=block_table, causal=True)


@fa_kvcache_paged.register_fake
def _(q, k_cache, v_cache, k, v, cache_seqlens, block_table):
    return torch.empty_like(q)


# --------------------------------------------------------------------------------
# § Model Load — banked weights -> live tensors (+ optimizer state)
# --------------------------------------------------------------------------------
# The banks arrive already assembled: data/prepare_model.py did the hub
# round-trip, the config.json audit, the per-layer stacking and the QKV /
# gate-up concatenation, and wrote one safetensors file whose tensor names,
# shapes and dtype are the ones below. So this section opens a file and attaches
# run state — the reshape is a property of the checkpoint, not of the run.
#
# Every trained tensor is a plain bf16 CUDA tensor (the LIVE weights, seen in
# place by every captured graph and compiled fn) carrying its state as
# attached attributes:
#   .grad32      fp32 gradient accumulator, zeroed after each optimizer step
#   .mantissa    lower 16 bits of the fp32 master (uint16) — master = live<<16|mantissa
#   .exp_avg / .exp_avg_sq   fp32 AdamW moments
#   .grad32_slices  out-of-graph per-layer views for the 3-D banks (an in-graph
#                   bank slice functionalizes into a whole-bank select_scatter)
# The live weights ARE the bf16 checkpoint, bit-exact. The mantissa starts
# MID-BIN (0x8000), not all-zero — see the init loop below for why; it is the
# one deliberate change to qwen-gsm8k's optimizer machinery in this file.
# The fused QKV (1152, 896) and gate/up (9728, 896) GEMMs are why: ~72 fewer
# kernels per decode step, and they sit below the launch floor, so the fusion is
# ~free throughput.

class Model:
    embed:      Tensor   # (V, D) bf16 — tied: input table AND lm_head
    W_QKV:      Tensor   # (L, 1152, 896)
    b_QKV:      Tensor   # (L, 1152)       Qwen2.5 QKV biases
    W_O:        Tensor   # (L, 896, 896)
    W_GU:       Tensor   # (L, 9728, 896)  [gate | up]
    W_down:     Tensor   # (L, 896, 4864)
    attn_norm:  Tensor   # (L, 896)  input_layernorm weights
    mlp_norm:   Tensor   # (L, 896)  post_attention_layernorm weights
    final_norm: Tensor   # (896,)
    cos: Tensor          # rotary caches (not trained)
    sin: Tensor

    weight_names = ("embed", "W_QKV", "b_QKV", "W_O", "W_GU", "W_down",
                    "attn_norm", "mlp_norm", "final_norm")

    def __iter__(self):
        return (getattr(self, n) for n in self.weight_names)


print(f"[{cfg.tag}] loading {cfg.model_id} banks ...", flush=True)
t = time.perf_counter()
torch.cuda.reset_peak_memory_stats()

from safetensors.torch import load_file

_banks = cfg.data_dir / f"banks_{cfg.model_id.split('/')[-1]}.safetensors"
assert _banks.exists(), f"{_banks} not found — build it with:\n{_BUILD_MODEL}"
# The arch is hardcoded on both sides of the file: prepare_model.py asserted it
# against the checkpoint's config.json, and the sidecar carries what it wrote.
_bmeta = json.loads(_banks.with_suffix(".json").read_text())
assert _bmeta["model_id"] == cfg.model_id, \
    f"banks are {_bmeta['model_id']}, cfg.model_id is {cfg.model_id} — rebuild:\n{_BUILD_MODEL}"
for _k in ("n_layers", "d_model", "n_qo_heads", "n_kv_heads", "d_head", "d_mlp",
           "d_vocab", "rope_theta", "rms_eps"):
    assert _bmeta[_k] == getattr(cfg, _k), f"banks {_k}={_bmeta[_k]} != cfg.{_k}={getattr(cfg, _k)}"

m = Model()
_sd = load_file(str(_banks), device=str(device))   # straight to device, bf16
for _n in Model.weight_names:
    setattr(m, _n, _sd[_n])
del _sd

assert m.embed.shape == (cfg.d_vocab, cfg.d_model)
assert m.W_QKV.shape == (cfg.n_layers, cfg.d_qkv, cfg.d_model)
assert m.W_GU.shape == (cfg.n_layers, 2 * cfg.d_mlp, cfg.d_model)

fp32_zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)

# MID-BIN mantissa init — the one deliberate departure from qwen-gsm8k's
# optimizer, and load-bearing. All-zero mantissa parks every master ON its
# bf16 bin edge, and the truncating writeback then turns the run's FIRST
# update — at any lr — into a ~2^-9-relative signSGD kick that this task does
# not survive (val 61.5% -> 0.5% in 3 rounds; gsm8k takes the same kick and
# happens to live). Mid-bin, small updates accumulate honestly and the live
# flips only after a genuine half-ULP of movement; the pairing stays lossless
# and live == checkpoint at init either way.
# TECHNIQUES.md § The mantissa first-step kick has the full account.
for p in m:
    p.grad32     = fp32_zeros(*p.shape)
    p.mantissa   = torch.full(p.shape, 0x8000, dtype=torch.uint16, device=device)
    p.exp_avg    = fp32_zeros(*p.shape)
    p.exp_avg_sq = fp32_zeros(*p.shape)
for p in (m.W_QKV, m.b_QKV, m.W_O, m.W_GU, m.W_down, m.attn_norm, m.mlp_norm):
    p.grad32_slices = list(p.grad32.unbind(0))

# ==== Rotary cache ====
# HF/Qwen convention (rotate_half, non-interleaved): channel j pairs with
# j + head_dim/2; cos/sin are (T, head_dim/2) and broadcast over both halves.
# Forward rotation: y1 = q1*cos - q2*sin ; y2 = q2*cos + q1*sin.
cfg.rope_t = cfg.t_row
_inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, dtype=torch.float32, device=device) / cfg.d_head))
_freqs = torch.outer(torch.arange(cfg.rope_t, dtype=torch.float32, device=device), _inv_freq)
m.cos = _freqs.cos().to(torch.bfloat16)    # (cfg.rope_t, 32)
m.sin = _freqs.sin().to(torch.bfloat16)
del _inv_freq, _freqs

_n_params = sum(p.numel() for p in m)
print(f"[{cfg.tag}] loaded: {_n_params:,} params "
      f"({(_n_params - m.embed.numel()):,} non-embedding) in {time.perf_counter() - t:.0f}s",
      flush=True)


# --------------------------------------------------------------------------------
# § Schedules — AdamW update coefficients, precomputed on device
# --------------------------------------------------------------------------------
# Every number the fused kernel multiplies by is folded into per-step tables up
# front; the kernels gather row `t_step` on device and the loop sets nothing
# per step. Bias corrections use the closed 1-beta^t form. The default is a
# CONSTANT lr, so the tables are flat — the mechanism costs nothing and the
# cfg.lr_schedule arm slots straight in.

class AdamWTabs(NamedTuple):
    wd_mul: Tensor           # 1 - lr*wd            decoupled weight decay
    one_minus_beta1: Tensor  # exp_avg lerp weight
    one_minus_beta2: Tensor  # exp_avg_sq lerp weight
    rsqrt_bias2: Tensor      # 1/sqrt(1 - beta2^t)
    step_size: Tensor        # lr / (1 - beta1^t)


def build_schedules(n_steps: int):
    N = max(1, n_steps)
    t1 = np.arange(1, N + 1, dtype=np.float64)
    lr = np.full(N, cfg.lr)
    if cfg.lr_schedule == "linear":
        lr *= 1.0 - np.arange(N) / N
    dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return SimpleNamespace(
        tabs=AdamWTabs(
            wd_mul          = dev(1.0 - lr * cfg.weight_decay),
            one_minus_beta1 = dev(np.full(N, 1.0 - cfg.beta1)),
            one_minus_beta2 = dev(np.full(N, 1.0 - cfg.beta2)),
            rsqrt_bias2     = dev(1.0 / np.sqrt(1.0 - cfg.beta2 ** t1)),
            step_size       = dev(lr / (1.0 - cfg.beta1 ** t1)),
        ),
        lr_host=lr, num_steps=N)


sched = build_schedules(num_rounds)
t_step = torch.zeros(1, dtype=torch.int64, device=device)   # advanced on-device


# --------------------------------------------------------------------------------
# § Optimizer — mantissa-trick AdamW, one fused kernel for every tensor
# --------------------------------------------------------------------------------
# Rather than holding bf16 live weights AND an fp32 master copy, the master's
# lower 16 mantissa bits are stashed separately: master bits = (live_bf16 <<
# 16) | mantissa. Update math runs in fp32 on the reconstructed master; the
# split back is a TRUNCATION (round-to-nearest could carry into the top bits
# and break the lossless pairing). In-place on the live tensor's storage —
# captured graphs and compiled fns see every update with no re-capture.

def fp32_master(live: Tensor, mantissa: Tensor) -> Tensor:
    bits = (live.view(torch.int16).to(torch.int32) << 16) | \
           (mantissa.view(torch.int16).to(torch.int32) & 0xFFFF)
    return bits.view(torch.float32)


def writeback_master(master: Tensor, live: Tensor, mantissa: Tensor) -> None:
    bits = master.view(torch.int32)
    live.view(torch.int16).copy_((bits >> 16).to(torch.int16))
    mantissa.view(torch.int16).copy_(bits.to(torch.int16))


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(live: Tensor, mantissa: Tensor, grad: Tensor,
                     exp_avg: Tensor, exp_avg_sq: Tensor,
                     c: AdamWTabs, t: Tensor, eps: float) -> None:
    p = fp32_master(live, mantissa)
    p.mul_(c.wd_mul[t])
    exp_avg.lerp_(grad, c.one_minus_beta1[t])
    exp_avg_sq.lerp_(grad.square(), c.one_minus_beta2[t])
    denom = exp_avg_sq.sqrt() * c.rsqrt_bias2[t] + eps
    p.sub_(c.step_size[t] * (exp_avg / denom))
    writeback_master(p, live, mantissa)


@torch.no_grad()
def optimizer_step() -> None:
    """One AdamW step over every trained tensor (a single param group — same
    lr/wd for matrices, biases and norms), then advance t_step on-device.
    Caller zeroes grad32 afterwards (gradients accumulate across a round's
    packs)."""
    for p in m:
        adamw_step_fused(p, p.mantissa, p.grad32, p.exp_avg, p.exp_avg_sq,
                         sched.tabs, t_step, cfg.adam_eps)
    t_step.add_(1)


# --------------------------------------------------------------------------------
# § Training Forward/Backward — handwritten, packed varlen, advantage-weighted CE
# --------------------------------------------------------------------------------
# One micro-batch = one pack: a 1-D stream of (prompt+completion) docs with
# per-doc attention isolation via FA3 varlen. No autograd: forward stashes,
# backward accumulates into .grad32. The RL loss is a per-token WEIGHTED CE
# (weight = advantage / round response tokens, zero on prompt/pad targets), so
# the only change from a pretraining CE backward is that the compile-time
# constant loss_scale/T becomes the per-token vector w — nothing else moves.
#
# The lm_head runs ONLY on the cfg.sel_cap gathered completion positions and is
# row-chunked (ce_chunk) so the 152k-vocab logits never materialize past
# (ce_chunk, V): at (16384, 151936) a full bf16 logits tensor alone is 5 GB
# and each fp32 temp 10 GB. The body backward still covers every position —
# prompt K/V receive gradient through completion queries.
#
# rms_norm handling: stash the UNWEIGHTED norm output x_hat plus 1/rms. In
# output space the backward needs no pre-norm input and is exact for any eps:
#   dw    = sum_T(dy * x_hat)
#   dx_hat = dy * w
#   dx    = r * (dx_hat - x_hat * mean(x_hat * dx_hat))

bf16 = lambda x: x.to(torch.bfloat16)


def _rms_fwd(x):
    """Unweighted rms_norm + 1/rms (fp32), Qwen eps."""
    r = (x.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    return bf16(x.float() * r), r


def _rms_bwd(d_hat, x_hat, r):
    xf, df = x_hat.float(), d_hat.float()
    return bf16(r * (df - xf * (xf * df).mean(dim=-1, keepdim=True)))


class LayerStash(NamedTuple):
    """One layer's forward activations held for backward. Sizes at T=16,384
    bf16, totals across 24 layers: x_hat pairs 1.4GB, qkv-post-rope+y 1.5GB,
    xm 0.7GB, gu 7.6GB — ~11.4GB, the price of not recomputing the wide GEMM."""
    xb_hat:     Tensor   # (T, D)        attn-norm output, unweighted
    xb_inv_rms: Tensor   # (T, 1) fp32
    q:          Tensor   # (T, 14, 64)   post-rope (what FA consumed)
    k:          Tensor   # (T, 2, 64)    post-rope
    v:          Tensor   # (T, 2, 64)
    y:          Tensor   # (T, 14, 64)   attn out
    lse:        Tensor   #               softmax lse (fp32)
    xm:         Tensor   # (T, D)        post-attn residual (mlp norm recomputed)
    gu:         Tensor   # (T, 9728)     fused gate|up pre-activation


@torch.no_grad()
def forward_backward(idx, pos, cu_seqlens, sel, tgt_sel, w_sel):
    """One pack: forward, stash, explicit backward into .grad32. Returns the
    summed weighted CE (the round's token-mean pg-loss contribution — the
    normalizer already rode in on w). Wrap in torch.compile: the CE chunk block
    is written for inductor's fusion."""
    T = idx.size(0)
    Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
    cos = m.cos[pos].unsqueeze(1)   # (T, 1, 32) — broadcasts over heads
    sin = m.sin[pos].unsqueeze(1)

    # -----------------------------
    #           Forward
    # -----------------------------
    x = F.embedding(idx, m.embed)                # (T, D) bf16 — no input norm in Qwen
    stash = []
    for i in range(cfg.n_layers):
        xb_hat, xb_r = _rms_fwd(x)
        xbn = xb_hat * m.attn_norm[i]
        qkv = xbn @ m.W_QKV[i].mT + m.b_QKV[i]
        q = qkv[:, :cfg.d_q].view(T, Hq, Dh)
        k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
        v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh)
        q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
        k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        # max_seqlen is train_t — the pad tail is the pack's longest segment,
        # not the longest doc (TECHNIQUES.md § Padded varlen).
        y, lse = flash_attn_varlen_fwd_lse(q, k, v, cu_seqlens, cfg.train_t)
        y = y.contiguous()
        xm = x + y.view(T, -1) @ m.W_O[i].mT
        xm_hat, xm_r = _rms_fwd(xm)              # recomputed in bwd, not stashed
        xmn = xm_hat * m.mlp_norm[i]
        gu = xmn @ m.W_GU[i].mT                  # (T, 9728)
        g, u = gu[:, :cfg.d_mlp], gu[:, cfg.d_mlp:]
        x = xm + (F.silu(g) * u) @ m.W_down[i].mT
        stash.append(LayerStash(xb_hat=xb_hat, xb_inv_rms=xb_r, q=q, k=k, v=v,
                                y=y, lse=lse, xm=xm, gu=gu))

    xf_hat, xf_r = _rms_fwd(x)
    xfn = xf_hat * m.final_norm

    # -----------------------------
    #     LM head + weighted CE  (chunked over the gathered completion rows)
    # -----------------------------
    xfn_sel = xfn.index_select(0, sel)           # (S, D)
    loss = torch.zeros((), dtype=torch.float32, device=idx.device)
    sel_grads = []
    for c0 in range(0, cfg.sel_cap, cfg.ce_chunk):
        xs = xfn_sel[c0:c0 + cfg.ce_chunk]           # (c, D)
        tg = tgt_sel[c0:c0 + cfg.ce_chunk]
        wc = w_sel[c0:c0 + cfg.ce_chunk]
        logits = (xs @ m.embed.mT).float()       # (c, V) — tied head
        cmax = logits.amax(dim=1, keepdim=True)
        e = (logits - cmax).exp()
        ssum = e.sum(dim=1, keepdim=True)
        lse_c = (ssum.log() + cmax).squeeze(1)
        ly = logits.gather(1, tg.unsqueeze(1)).squeeze(1)
        loss += (wc * (lse_c - ly)).sum()
        onehot = torch.arange(cfg.d_vocab, device=idx.device).unsqueeze(0) == tg.unsqueeze(1)
        logits_grad = bf16((e / ssum - onehot.float()) * wc.unsqueeze(1))
        m.embed.grad32.add_((logits_grad.mT @ xs).float())
        sel_grads.append(logits_grad @ m.embed)
    xfn_grad = torch.zeros_like(xfn)
    xfn_grad.index_add_(0, sel, torch.cat(sel_grads))   # w=0 pads land as zeros at row 0

    # -----------------------------
    #           Backward
    # -----------------------------
    m.final_norm.grad32.add_((xfn_grad.float() * xf_hat.float()).sum(dim=0))
    stream_grad = _rms_bwd(xfn_grad * m.final_norm, xf_hat, xf_r)

    for i in reversed(range(cfg.n_layers)):
        st = stash[i]
        # --- MLP backward (SwiGLU) ---
        xm_hat, xm_r = _rms_fwd(st.xm)
        xmn = xm_hat * m.mlp_norm[i]
        g, u = st.gu[:, :cfg.d_mlp], st.gu[:, cfg.d_mlp:]
        sg = torch.sigmoid(g)
        silu_g = g * sg
        a = silu_g * u
        m.W_down.grad32_slices[i].add_(stream_grad.mT @ a)
        a_grad = stream_grad @ m.W_down[i]
        u_grad = a_grad * silu_g
        g_grad = a_grad * u * (sg * (1 + g * (1 - sg)))   # d silu / dg
        gu_grad = torch.cat([g_grad, u_grad], dim=1)
        m.W_GU.grad32_slices[i].add_(gu_grad.mT @ xmn)
        xmn_grad = gu_grad @ m.W_GU[i]
        m.mlp_norm.grad32_slices[i].add_((xmn_grad.float() * xm_hat.float()).sum(dim=0))
        xm_grad = stream_grad + _rms_bwd(xmn_grad * m.mlp_norm[i], xm_hat, xm_r)

        # --- Attention backward ---
        xbn = st.xb_hat * m.attn_norm[i]
        m.W_O.grad32_slices[i].add_(xm_grad.mT @ st.y.view(T, -1))
        y_grad = (xm_grad @ m.W_O[i]).view(T, Hq, Dh)
        q_grad, k_grad, v_grad = flash_attn_varlen_bwd(
            y_grad, st.q, st.k, st.v, st.y, st.lse, cu_seqlens, cfg.train_t)
        # rotary backward = rotation by -theta (Jacobian transpose of forward)
        q1g, q2g = q_grad[..., :cfg.half], q_grad[..., cfg.half:]
        k1g, k2g = k_grad[..., :cfg.half], k_grad[..., cfg.half:]
        q_grad = torch.cat([q1g * cos + q2g * sin, q2g * cos - q1g * sin], dim=-1)
        k_grad = torch.cat([k1g * cos + k2g * sin, k2g * cos - k1g * sin], dim=-1)
        qkv_grad = torch.cat([q_grad.reshape(T, cfg.d_q), k_grad.reshape(T, cfg.d_kv),
                              v_grad.reshape(T, cfg.d_kv)], dim=1)
        m.b_QKV.grad32_slices[i].add_(qkv_grad.sum(dim=0, dtype=torch.float32))
        m.W_QKV.grad32_slices[i].add_(qkv_grad.mT @ xbn)
        xbn_grad = qkv_grad @ m.W_QKV[i]
        m.attn_norm.grad32_slices[i].add_((xbn_grad.float() * st.xb_hat.float()).sum(dim=0))
        stream_grad = xm_grad + _rms_bwd(xbn_grad * m.attn_norm[i], st.xb_hat, st.xb_inv_rms)
        stash[i] = None                          # free as we go

    # --- token embedding scatter (the tied table's second gradient path) ---
    m.embed.grad32.add_(
        torch.ops.aten.embedding_dense_backward(stream_grad, idx, cfg.d_vocab, -1, False))
    return loss


fb = torch.compile(forward_backward, dynamic=False, fullgraph=True)


# --------------------------------------------------------------------------------
# § Generation — dense-page KV cache + bucketed CUDA-graph decode
# --------------------------------------------------------------------------------
# The whole round (256 rows x a 768-token row ~ 2.3 GB of KV) fits resident at
# once, so KV management collapses to a STATIC page assignment: one (L, NB,
# page, H_kv, Dh) allocation where row r permanently owns pages [r*P, (r+1)*P)
# — the same tensor viewed (as_strided) as a dense (L, max_seqs, t_row, H_kv,
# Dh) cache for prefill's broadcast write. The block table earns its keep as
# pure INDIRECTION: a bucket drop compacts survivors to the front by permuting
# (B,P) int32 rows, so no KV ever moves. Every paging DYNAMIC — refcounts,
# free lists, COW, growth — is gone. One extra NULL page backs parked
# (dead/padded) rows: a parked row keeps replaying at bucket cost, writes its
# sampled K into the null page and attends over ~nothing (cache_seqlens 0).
# Parking MUST redirect the block table — after a compaction the tail
# positions' stale tables can alias survivors' pages.
#
# Decode state per row is just (input_id, cache_seqlen, block-table row): the
# graph itself carries all three between windows (each replay writes the
# sampled token into input_ids and advances cache_seqlens), so a steady-state
# window uploads NOTHING and downloads one pinned (bucket, macro_n) token
# block. Rows move only at bucket drops (compact); prefix sharing is COMPUTE
# only — each context prefills once (varlen) and broadcasts to its K siblings.

def decode_body(input_ids, cache_seqlens, block_table, k_pool, v_pool):
    """One decode step for B rows. input_ids (B,1) long | cache_seqlens (B,)
    int32 | block_table (B,P) int32 | k/v_pool (L, NB, page, H_kv, Dh).
    Returns fp32 logits (B, V)."""
    B = input_ids.shape[0]
    Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
    x = F.embedding(input_ids, m.embed)          # (B, 1, D)
    posn = cache_seqlens.to(torch.long)
    cos = m.cos[posn].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 32)
    sin = m.sin[posn].unsqueeze(1).unsqueeze(1)
    for i in range(cfg.n_layers):
        xn = F.rms_norm(x, (cfg.d_model,), m.attn_norm[i], cfg.rms_eps)
        qkv = xn @ m.W_QKV[i].mT + m.b_QKV[i]
        q = qkv[..., :cfg.d_q].view(B, 1, Hq, Dh)
        k = qkv[..., cfg.d_q:cfg.d_q + cfg.d_kv].view(B, 1, Hkv, Dh)
        v = qkv[..., cfg.d_q + cfg.d_kv:].view(B, 1, Hkv, Dh)
        q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
        k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        y = fa_kvcache_paged(q, k_pool[i], v_pool[i], k, v, cache_seqlens, block_table)
        x = x + y.view(B, 1, -1) @ m.W_O[i].mT
        xn2 = F.rms_norm(x, (cfg.d_model,), m.mlp_norm[i], cfg.rms_eps)
        gu = xn2 @ m.W_GU[i].mT
        x = x + (F.silu(gu[..., :cfg.d_mlp]) * gu[..., cfg.d_mlp:]) @ m.W_down[i].mT
    x = F.rms_norm(x, (cfg.d_model,), m.final_norm, cfg.rms_eps)
    return (x[:, 0] @ m.embed.mT).float()        # (B, V)


def prefill_body(ids, pos, cu_seqlens):
    """Packed varlen prefill of node contexts. Returns the per-layer K and V
    (post-rope, exactly what attention consumed) stacked (L, T, H_kv, Dh); the
    caller broadcasts them into the dense cache EAGERLY, outside the compiled
    region — an in-graph cache store can fuse into the producing kernels and
    pick up different bf16 rounding than the packed k/v the prefill attention
    consumed. No lm_head: the forced-last-token split means prefill's only
    product is KV."""
    T = ids.shape[0]
    Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
    cos = m.cos[pos].unsqueeze(1)                # (T, 1, 32)
    sin = m.sin[pos].unsqueeze(1)
    x = F.embedding(ids, m.embed)
    ks, vs = [], []
    for i in range(cfg.n_layers):
        xn = F.rms_norm(x, (cfg.d_model,), m.attn_norm[i], cfg.rms_eps)
        qkv = xn @ m.W_QKV[i].mT + m.b_QKV[i]
        q = qkv[:, :cfg.d_q].view(T, Hq, Dh)
        k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
        v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh)
        q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
        k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        ks.append(k)
        vs.append(v)
        y = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                                   max_seqlen_q=cfg.prefill_t, max_seqlen_k=cfg.prefill_t, causal=True)
        x = x + y.reshape(T, -1) @ m.W_O[i].mT
        xn2 = F.rms_norm(x, (cfg.d_model,), m.mlp_norm[i], cfg.rms_eps)
        gu = xn2 @ m.W_GU[i].mT
        x = x + (F.silu(gu[:, :cfg.d_mlp]) * gu[:, cfg.d_mlp:]) @ m.W_down[i].mT
    return torch.stack(ks), torch.stack(vs)      # (L, T, H_kv, Dh) x2


def sample(logits: Tensor, inv_temp: Tensor) -> Tensor:
    """Gumbel-max draw == exact softmax sampling at temperature 1/inv_temp —
    no top-k, no top-p, no sort, no cumsum: one elementwise pass + one argmax
    over the fp32 logits. inv_temp lives in a 0-D CUDA buffer so eval can
    retune without re-capturing (see GREEDY_INV_TEMP)."""
    e = torch.empty_like(logits).exponential_()
    return (logits * inv_temp - e.log()).argmax(dim=-1)


# Greedy through the SAME captured graphs: at inv_temp 1e4 the Gumbel noise
# (O(1)) is negligible against the scaled logits, so the draw is argmax —
# greedy — without a second sampler or a re-capture. Ties closer than ~1e-4
# in logit space still fall to the noise; that is measurement-noise level.
GREEDY_INV_TEMP = 1e4


class Engine:
    """Bucketed CUDA-graph decoder + compiled varlen prefill over the live
    banks. One graph per row-count bucket; each replay = one decode step; the
    driver replays cfg.macro_n times per window and reads back one pinned block."""

    def __init__(self):
        L, Hkv, Dh = cfg.n_layers, cfg.n_kv_heads, cfg.d_head
        P = cfg.pages_per_row
        NB = cfg.max_seqs * P + 1                       # + the null page (id cfg.max_seqs*P)
        self.NULL_PAGE = cfg.max_seqs * P
        self.k_pool = torch.zeros(L, NB, cfg.page, Hkv, Dh, dtype=torch.bfloat16, device=device)
        self.v_pool = torch.zeros(L, NB, cfg.page, Hkv, Dh, dtype=torch.bfloat16, device=device)
        # Dense view of the row-owned pages (a row's P pages are consecutive,
        # so (page, page-size) merges into t_row; dim L keeps its own stride).
        el = cfg.page * Hkv * Dh
        self.k_dense = self.k_pool.as_strided((L, cfg.max_seqs, cfg.t_row, Hkv, Dh),
                                              (NB * el, P * el, Hkv * Dh, Dh, 1))
        self.v_dense = self.v_pool.as_strided((L, cfg.max_seqs, cfg.t_row, Hkv, Dh),
                                              (NB * el, P * el, Hkv * Dh, Dh, 1))
        self.bt_identity = torch.arange(cfg.max_seqs * P, dtype=torch.int32,
                                        device=device).view(cfg.max_seqs, P)
        # Static graph buffers
        self.input_ids = torch.zeros(cfg.max_seqs, 1, dtype=torch.long, device=device)
        self.cache_seqlens = torch.zeros(cfg.max_seqs, dtype=torch.int32, device=device)
        self.block_table = torch.full((cfg.max_seqs, P), self.NULL_PAGE,
                                      dtype=torch.int32, device=device)
        self.tok_buf = torch.zeros(cfg.max_seqs, dtype=torch.long, device=device)
        self.token_record = torch.zeros(cfg.max_seqs, cfg.macro_n, dtype=torch.long, device=device)
        self.tok_host = torch.empty(cfg.max_seqs, cfg.macro_n, dtype=torch.long, pin_memory=True)
        self.tok_host_np = self.tok_host.numpy()
        self.inv_temp = torch.tensor(1.0 / cfg.temperature, dtype=torch.float32, device=device)
        # Prefill static buffers (compiled, not captured: ~1-4k tokens once a
        # round is a few ms of compute; a graph would save ~1 ms of a ~2 s round)
        self.pf_ids = torch.zeros(cfg.prefill_t, dtype=torch.int32, device=device)
        self.pf_pos = torch.zeros(cfg.prefill_t, dtype=torch.int64, device=device)
        self.pf_cu = torch.zeros(cfg.max_ctxs + 2, dtype=torch.int32, device=device)
        self.decode_fn = torch.compile(decode_body, dynamic=False)
        self.prefill_fn = torch.compile(prefill_body, dynamic=False, fullgraph=True)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mempool = None
        self.TERM = np.array(TERMINALS, dtype=np.int64)

    def _macro_body(self, b: int) -> None:
        logits = self.decode_fn(self.input_ids[:b], self.cache_seqlens[:b],
                                self.block_table[:b], self.k_pool, self.v_pool)
        tok = sample(logits, self.inv_temp)
        self.tok_buf[:b] = tok
        self.input_ids[:b, 0] = tok
        self.cache_seqlens[:b] += 1

    @torch.no_grad()
    def capture(self) -> None:
        print(f"  engine: {cfg.max_seqs} rows x {cfg.t_row} tok "
              f"({(self.k_pool.numel() + self.v_pool.numel()) * 2 / 2**30:.1f} GB KV) | "
              f"buckets {cfg.buckets} | macro_n {cfg.macro_n} | prefill T={cfg.prefill_t} | "
              f"temp {cfg.temperature:g} (gumbel-argmax)", flush=True)
        print("  capture+compile decode buckets:", flush=True)
        for b in sorted(cfg.buckets, reverse=True):
            t0 = time.perf_counter()
            self.input_ids[:] = 0
            self.block_table[:] = self.bt_identity   # valid pages during warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.cache_seqlens[:] = 0
                    self._macro_body(b)
            torch.cuda.current_stream().wait_stream(s)
            self.cache_seqlens[:] = 0
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, **({"pool": self._mempool} if self._mempool else {})):
                self._macro_body(b)
            self._mempool = self._mempool or g.pool()
            self.graphs[b] = g
            print(f"    bucket {b:3d}: {time.perf_counter() - t0:5.1f}s", flush=True)
        # Prefill compile warmup (one padded dummy call)
        t0 = time.perf_counter()
        self.pf_ids[:] = PAD_ID
        self.pf_pos[:] = 0
        self.pf_cu[:] = cfg.prefill_t
        self.pf_cu[0] = 0
        self.prefill_fn(self.pf_ids, self.pf_pos, self.pf_cu)
        torch.cuda.synchronize()
        print(f"    prefill compile: {time.perf_counter() - t0:5.1f}s", flush=True)

    @torch.no_grad()
    def run_round(self, specs: list[tuple]) -> list[dict]:
        """specs: (meta, prompt_ids, k, allow). Context = prompt[:-1]; forced
        first decode input = prompt[-1] — so the first SAMPLED token already
        comes out of the decode graph. Prefill every context in ONE compiled
        call, broadcast KV to the K sibling rows, decode all rows to
        completion. Returns the rows, carrying the sampled ids INCLUDING the
        terminal (the pack trains on it; text decode strips it), and writes the
        generation slice of the global `stats`. An eval wave overwrites those
        fields, which is harmless: eval runs at the TOP of a round, before the
        round's own generation refills them."""
        t0 = time.perf_counter()
        ctx_tok = sum(len(p) - 1 for _, p, _, _ in specs)
        n_rows = sum(k for _, _, k, _ in specs)
        assert len(specs) <= cfg.max_ctxs and ctx_tok <= cfg.prefill_t and n_rows <= cfg.max_seqs

        # -- prefill pack (host-side assembly, one H2D per buffer) -------------
        ids, pos, cu = [], [], [0]
        for _, p, _, _ in specs:
            ctx = p[:-1]
            ids.extend(ctx)
            pos.extend(range(len(ctx)))
            cu.append(cu[-1] + len(ctx))
        pad = cfg.prefill_t - len(ids)
        cu = cu + [cfg.prefill_t] * (cfg.max_ctxs + 2 - len(cu))   # pad tail + ghost segments
        self.pf_ids.copy_(torch.tensor(ids + [PAD_ID] * pad, dtype=torch.int32), non_blocking=True)
        self.pf_pos.copy_(torch.tensor(pos + [0] * pad, dtype=torch.int64), non_blocking=True)
        self.pf_cu.copy_(torch.tensor(cu, dtype=torch.int32), non_blocking=True)
        k_all, v_all = self.prefill_fn(self.pf_ids, self.pf_pos, self.pf_cu)

        # -- broadcast each context's KV into its K sibling rows ---------------
        r0, o = 0, 0
        metas, plens_l, forced, allows_l = [], [], [], []
        for meta, p, k, allow in specs:
            plen = len(p) - 1
            self.k_dense[:, r0:r0 + k, :plen] = k_all[:, o:o + plen].unsqueeze(1)
            self.v_dense[:, r0:r0 + k, :plen] = v_all[:, o:o + plen].unsqueeze(1)
            metas.extend([meta] * k)
            plens_l.extend([plen] * k)
            forced.extend([p[-1]] * k)
            allows_l.extend([allow] * k)
            r0 += k
            o += plen

        # -- seed the graph state (the round's one full upload) ----------------
        B0 = n_rows
        plens = np.asarray(plens_l, dtype=np.int64)
        allows = np.asarray(allows_l, dtype=np.int64)
        bucket = next(x for x in cfg.buckets if x >= B0)
        self.input_ids[:B0, 0] = torch.tensor(forced, dtype=torch.long, device=device)
        self.cache_seqlens[:B0] = torch.tensor(plens, dtype=torch.int32, device=device)
        self.block_table[:B0] = self.bt_identity[:B0]
        if bucket > B0:                          # park the padded tail
            self.cache_seqlens[B0:bucket] = 0
            self.block_table[B0:bucket] = self.NULL_PAGE

        # -- decode windows ----------------------------------------------------
        orig = np.arange(B0)
        live = np.ones(B0, dtype=bool)
        gen_buf = np.empty((B0, (-(-int(allows.max()) // cfg.macro_n)) * cfg.macro_n), dtype=np.int64)
        # A row past its capacity would index past its P block-table entries and
        # write K into a NEIGHBOR row's page — silent corruption, not a crash.
        assert int(plens.max()) + gen_buf.shape[1] <= cfg.t_row, \
            f"row overflow: ctx {int(plens.max())} + {gen_buf.shape[1]} steps > t_row {cfg.t_row}"
        rows: list[dict] = [None] * B0
        rolls_done = tok_total = paid_slots = 0
        t50 = t90 = None
        n_half, n_ninety = (B0 + 1) // 2, (B0 * 9 + 9) // 10
        park_dirty = False
        w = 0
        while True:
            lp = np.flatnonzero(live)
            if lp.size == 0:
                break
            nb = next(x for x in cfg.buckets if x >= lp.size)
            if nb < bucket:                      # bucket drop: compact survivors
                idxs = torch.from_numpy(lp).to(device, non_blocking=True)
                for buf in (self.input_ids, self.cache_seqlens, self.block_table):
                    buf[:lp.size].copy_(buf.index_select(0, idxs))
                orig, plens, allows = orig[lp], plens[lp], allows[lp]
                live = np.ones(lp.size, dtype=bool)
                bucket = nb
                self.cache_seqlens[lp.size:bucket] = 0
                self.block_table[lp.size:bucket] = self.NULL_PAGE
                park_dirty = False
                lp = np.arange(lp.size)
            elif park_dirty:                     # park mid-bucket retirees in place
                dead = np.flatnonzero(~live)
                di = torch.from_numpy(dead).to(device, non_blocking=True)
                self.cache_seqlens.index_fill_(0, di, 0)
                self.block_table.index_fill_(0, di, self.NULL_PAGE)
                park_dirty = False
            g = self.graphs[bucket]
            for j in range(cfg.macro_n):
                g.replay()
                self.token_record[:bucket, j] = self.tok_buf[:bucket]
            self.tok_host[:bucket].copy_(self.token_record[:bucket], non_blocking=True)
            torch.cuda.synchronize()             # the window's single host sync
            toks = self.tok_host_np[:bucket]
            paid_slots += bucket * cfg.macro_n
            base = w * cfg.macro_n
            t_live = toks[lp]
            gen_buf[orig[lp], base:base + cfg.macro_n] = t_live
            done, eos, n_take = window_events(t_live, allows[lp], base, self.TERM)
            for ri in np.flatnonzero(done):
                p_ = lp[ri]
                o_ = orig[p_]
                n_ = base + int(n_take[ri])
                rows[o_] = dict(meta=metas[o_], ids=gen_buf[o_, :n_].tolist(),
                                eos=bool(eos[ri]))
                live[p_] = False
                rolls_done += 1
                tok_total += int(n_take[ri])
            tok_total += cfg.macro_n * int((~done).sum())
            if done.any():
                park_dirty = True
            w += 1
            el = time.perf_counter() - t0
            if t90 is None and rolls_done >= n_ninety:
                t90 = el
                t50 = t50 if t50 is not None else el
            elif t50 is None and rolls_done >= n_half:
                t50 = el
        gen_s = time.perf_counter() - t0
        stats.gen_s = round(gen_s, 2)
        stats.gen_tok = tok_total
        stats.gen_tok_per_s = round(tok_total / gen_s, 0)
        stats.occ = round(100 * tok_total / max(1, paid_slots), 1)
        stats.t50 = round(t50 if t50 is not None else gen_s, 2)
        stats.t90 = round(t90 if t90 is not None else gen_s, 2)
        return rows


# --------------------------------------------------------------------------------
# § Trainer — round groups -> advantages -> packs -> fb -> optimizer step
# --------------------------------------------------------------------------------

def train_step(groups: list[dict]) -> None:
    """One optimizer step over the round's problem groups — GRPO/token-mean,
    into the training slice of the global `stats`. The normalizer is the
    round's TOTAL response-token count (all 256 rollouts, saturated groups
    included — their tokens dilute the mean exactly as a masked mean would);
    it folds into the per-token weight, so the packed forward just sums."""
    t0 = time.perf_counter()
    n_total_tok = sum(len(c) for g in groups for c in g["completions"])
    docs = []
    n_used = n_sat = n_dead = 0
    for g in groups:
        adv = group_advantage(g["rewards"])
        if adv is None:                          # zero-signal group: zero gradient
            # All-equal rewards: >= w_correct means every rollout was correct
            # (all 1s or all 2s — solved, ± method); 0 means none was.
            n_sat += int(np.mean(g["rewards"]) >= cfg.w_correct)
            n_dead += int(np.mean(g["rewards"]) <= 0.0)
            continue
        n_used += 1
        for comp, a in zip(g["completions"], adv):
            docs.append((g["prompt_ids"], comp, float(a) / max(1, n_total_tok)))
    loss_total = 0.0
    n_loss_tok = pstats = 0
    if docs:
        packs, pstats = plan_packs(docs)
        for pk in packs:
            args = [torch.from_numpy(pk[k]).to(device, non_blocking=True)
                    for k in ("idx", "pos", "cu", "sel", "tgt", "w")]
            loss = fb(*args)
            loss_total += float(loss)
            n_loss_tok += pk["n_sel"]
    gnorm = math.sqrt(sum(float((p.grad32.float() ** 2).sum()) for p in m))
    optimizer_step()                             # unconditional, one step per round
    for p in m:
        p.grad32.zero_()
    torch.cuda.synchronize()
    stats.train_s = round(time.perf_counter() - t0, 2)
    stats.n_groups_used, stats.n_groups_sat, stats.n_groups_dead = n_used, n_sat, n_dead
    stats.n_docs = len(docs)
    stats.n_loss_tokens = n_loss_tok
    stats.n_packs = pstats["n_packs"] if docs else 0
    stats.pad_pct = round(100.0 * pstats["pad_tokens"] / pstats["cap_tokens"], 1) if docs else 0.0
    stats.loss_total = round(loss_total, 6)
    stats.grad_norm = round(gnorm, 6)


# --------------------------------------------------------------------------------
# § Eval — greedy K=1 accuracy through the graphs (val in-loop, tests at the end)
# --------------------------------------------------------------------------------
# GREEDY, deliberately: the hf-vllm reference evaluates greedy accuracy, so
# these numbers land on the same scale (val 89.0 / ID 87.2 / OOD 86.2). K=1
# also means an eval wave carries one row per context, which is why max_ctxs
# is 64 here — it, not max_seqs, sets the eval batch width.

def make_eval_waves(prompts: list[list[int]], k: int) -> list[list[int]]:
    """Greedy wave assembly under the engine's static caps."""
    waves, cur, cur_tok = [], [], 0
    for i in range(len(prompts)):
        ctx = len(prompts[i]) - 1
        if cur and (len(cur) + 1 > cfg.max_ctxs or cur_tok + ctx > cfg.prefill_t
                    or (len(cur) + 1) * k > cfg.max_seqs):
            waves.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += ctx
    if cur:
        waves.append(cur)
    return waves


def run_eval(prompts: list[list[int]], golds: list[int], label: str) -> dict:
    """Greedy accuracy over one split. Sampler RNG is saved and restored so
    the training rollout stream is bit-identical to an eval-off run; inv_temp
    is flipped to GREEDY_INV_TEMP for the duration (same graphs, no capture)."""
    rng_state = torch.cuda.get_rng_state()
    self_inv = 1.0 / cfg.temperature
    engine.inv_temp.fill_(GREEDY_INV_TEMP)
    t0 = time.perf_counter()
    ok, lens = {}, []
    n_method = n_trunc = n_roll = 0
    try:
        for wave in make_eval_waves(prompts, 1):
            rows = engine.run_round([(i, prompts[i], 1, cfg.max_tokens) for i in wave])
            for r in rows:
                i = r["meta"]
                text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
                r_c = reward_correct(text, golds[i])
                ok[i] = int(r_c)
                n_method += reward_method(text, r_c) == 1.0
                n_trunc += not r["eos"]
                lens.append(len(r["ids"]))
                n_roll += 1
    finally:
        engine.inv_temp.fill_(self_inv)
        torch.cuda.set_rng_state(rng_state)
    n_prob = len(prompts)
    out = dict(label=label, n_problems=n_prob,
               accuracy=round(100 * sum(ok.values()) / max(1, n_prob), 2),
               n_correct=sum(ok.values()),
               method_pct=round(100 * n_method / max(1, n_roll), 1),
               trunc_pct=round(100 * n_trunc / max(1, n_roll), 1),
               mean_len=round(float(np.mean(lens)) if lens else 0.0, 1),
               eval_s=round(time.perf_counter() - t0, 1))
    # Per-problem correctness, so two evals can be compared PAIRED — per
    # problem, the difficulty cancels.
    out["per_problem"] = [(i, ok[i]) for i in range(n_prob)]
    return out


# --------------------------------------------------------------------------------
# § Warmup — capture graphs, compile prefill + training step
# --------------------------------------------------------------------------------

engine = Engine()
build_s = time.perf_counter() - run_wall_t0
t = time.perf_counter()
engine.capture()

# Compile fb on one dummy pack (weights untouched; w=0 so every grad lands as
# an exact zero — zeroed again after anyway).
_t = time.perf_counter()
_dummy_docs = [([3 + (j % 97) for j in range(50)],
                [5 + (j % 89) for j in range(110)] + [IM_END], 0.0)
               for _ in range(cfg.train_t // 160 + 1)]
_packs, _ = plan_packs(_dummy_docs)
_pk = _packs[0]
fb(*[torch.from_numpy(_pk[k]).to(device) for k in ("idx", "pos", "cu", "sel", "tgt", "w")])
for p in m:
    p.grad32.zero_()
del _dummy_docs, _packs, _pk
torch.cuda.synchronize()
print(f"    train fwd+bwd compile: {time.perf_counter() - _t:5.1f}s", flush=True)

# Compile the AdamW kernel per tensor shape NOW, with NEUTRAL coefficients
# (wd_mul 1, lerp weights 0, step_size 0): the update math runs on the real
# tensors and writes back bit-identical values — moments untouched, t_step
# untouched — so round 0 pays no compile stall.
_t = time.perf_counter()
_one = lambda: torch.ones(1, dtype=torch.float32, device=device)
_zero = lambda: torch.zeros(1, dtype=torch.float32, device=device)
_neutral = AdamWTabs(wd_mul=_one(), one_minus_beta1=_zero(),
                     one_minus_beta2=_zero(), rsqrt_bias2=_one(), step_size=_zero())
with torch.no_grad():
    for p in m:
        adamw_step_fused(p, p.mantissa, p.grad32, p.exp_avg, p.exp_avg_sq,
                         _neutral, torch.zeros(1, dtype=torch.int64, device=device),
                         cfg.adam_eps)
torch.cuda.synchronize()
print(f"    adamw kernels ({len(m.weight_names)} shapes): "
      f"{time.perf_counter() - _t:5.1f}s", flush=True)
warm_s = time.perf_counter() - t
print(f"  build {build_s:.0f}s + capture/compile {warm_s:.0f}s | "
      f"peak mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
assert int(t_step.item()) == 0


# --------------------------------------------------------------------------------
# § Main Loop
# --------------------------------------------------------------------------------

cfg.run_dir.mkdir(parents=True, exist_ok=True)

use_wandb = cfg.wandb
if use_wandb:
    try:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run,
                               config=config_dict() | dict(num_rounds=num_rounds))
    except Exception as e:
        print(f"[{cfg.tag}] wandb unavailable ({e}) — CSV/JSON only", flush=True)
        use_wandb = False


def save_ckpt(step: int) -> None:
    path = cfg.run_dir / f"model_step{step:06d}.pt"
    torch.save(dict(step=step, code=code, model_id=cfg.model_id,
                    weights={n: getattr(m, n).cpu() for n in m.weight_names}), path)
    print(f"  saved checkpoint -> {path}", flush=True)


METRIC_COLS = [f.name for f in fields(RoundStats)]   # the stats dataclass IS the schema
mf = open(Path.cwd() / f"metrics_{cfg.tag}.csv", "w", newline="")
mw = csv.DictWriter(mf, fieldnames=METRIC_COLS)
mw.writeheader()
eval_rows: list[dict] = []

# Per-problem eval detail — one row per (eval, problem). Small, and the only
# way to compare two horizons paired.
edf = open(Path.cwd() / f"evals_detail_{cfg.tag}.csv", "w", newline="")
edw = csv.DictWriter(edf, fieldnames=["round", "label", "idx", "ok"])
edw.writeheader()


def log_eval(rnd: int, res: dict) -> None:
    per_problem = res.pop("per_problem")
    edw.writerows(dict(round=rnd, label=res["label"], idx=i, ok=ok)
                  for i, ok in per_problem)
    edf.flush()
    res = dict(round=rnd, **res)
    eval_rows.append(res)
    print(f"  [eval {rnd:4d}] {res['label']}: acc {res['accuracy']:5.2f} "
          f"({res['n_correct']}/{res['n_problems']}) | method {res['method_pct']:4.1f}% | "
          f"len {res['mean_len']:.0f} | trunc {res['trunc_pct']:4.1f}% | {res['eval_s']}s",
          flush=True)
    with open(Path.cwd() / f"evals_{cfg.tag}.csv", "w", newline="") as ef:
        wtr = csv.DictWriter(ef, fieldnames=list(eval_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(eval_rows)
    if use_wandb:
        wandb_run.log({"round": rnd, f"eval/{res['label']}_accuracy": res["accuracy"],
                       f"eval/{res['label']}_method_pct": res["method_pct"],
                       f"eval/{res['label']}_trunc_pct": res["trunc_pct"],
                       f"eval/{res['label']}_mean_len": res["mean_len"]})


_mem_last = 0.0


def _device_mem_gb(rnd: int) -> float:
    global _mem_last
    if rnd == 0 or rnd % cfg.mem_every == 0:
        free, total = torch.cuda.mem_get_info()
        _mem_last = round((total - free) / 2**30, 1)
    return _mem_last


profiler = None
if cfg.profile:
    from torch.profiler import ProfilerActivity, profile as torch_profile
    num_rounds = min(num_rounds, cfg.prof_wait + 1 + cfg.prof_active)
    profiler = torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,
        schedule=torch.profiler.schedule(wait=cfg.prof_wait, warmup=1, active=cfg.prof_active, repeat=1))
    profiler.__enter__()

curve: list[dict] = []
run_error = None
loop_t0 = time.perf_counter()
try:
    for rnd in range(num_rounds):
        if cfg.eval_every and rnd % cfg.eval_every == 0:
            log_eval(rnd, run_eval(val_prompts, val_gold, "val"))
        r_t0 = time.perf_counter()
        stats = RoundStats(round=rnd)

        # -- generation ------------------------------------------------------
        idxs = cfg.fixed_problems if cfg.fixed_problems is not None else round_schedule[rnd]
        specs = [(i, train_prompts[i], cfg.k_draws, cfg.max_tokens) for i in idxs]
        rows = engine.run_round(specs)

        # -- grade + group ---------------------------------------------------
        by_pid: dict[int, dict] = {}
        n_roll, n_method = len(rows), 0
        for r in rows:
            pid = r["meta"]
            text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
            rw, r_c, r_m = grade(text, train_gold[pid])
            stats.n_correct += r_c == 1.0
            stats.n_eos += r["eos"]
            n_method += r_m == 1.0
            g = by_pid.setdefault(pid, dict(prompt_ids=train_prompts[pid],
                                            completions=[], rewards=[]))
            g["completions"].append(r["ids"])
            g["rewards"].append(rw)
        groups = list(by_pid.values())

        # -- train -----------------------------------------------------------
        train_step(groups)

        # -- telemetry -------------------------------------------------------
        stats.n_rollouts = n_roll
        stats.solve_rate = round(stats.n_correct / max(1, n_roll), 4)
        stats.n_trunc = n_roll - stats.n_eos
        stats.method_pct = round(100 * n_method / max(1, n_roll), 1)
        stats.lr = float(sched.lr_host[min(rnd, sched.num_steps - 1)])
        stats.mem_gb = _device_mem_gb(rnd)
        stats.round_s = round(time.perf_counter() - r_t0, 2)
        row = asdict(stats)
        curve.append(row)
        mw.writerow(row)
        mf.flush()
        el = time.perf_counter() - loop_t0
        eta = el / (rnd + 1) * (num_rounds - rnd - 1)
        print(f"  [{rnd:3d}/{num_rounds}] ({100 * (rnd + 1) / num_rounds:5.1f}%) "
              f"{stats.round_s:5.2f}s ({stats.gen_s:.2f} gen / {stats.train_s:.2f} trn) | "
              f"solve {100 * stats.solve_rate:5.1f}% | method {stats.method_pct:4.1f}% | "
              f"trunc {stats.n_trunc:3d} | dead {stats.n_groups_dead:2d}/{len(idxs)} | "
              f"total {el / 60:5.1f}m | eta {eta / 60:4.1f}m", flush=True)
        if use_wandb:
            wandb_run.log({"round": rnd, **{f"train/{k}": v for k, v in row.items()
                                            if k != "round"}})

        if (rnd + 1) % rounds_per_epoch == 0 or rnd + 1 == num_rounds:
            ep = curve[-((rnd % rounds_per_epoch) + 1):]
            ep_cor = sum(c["n_correct"] for c in ep)
            ep_roll = sum(c["n_rollouts"] for c in ep)
            print(f"  == epoch {rnd // rounds_per_epoch + 1}/{cfg.epochs} | solve "
                  f"{ep_cor:,}/{ep_roll:,} ({100 * ep_cor / ep_roll:5.2f}%) | avg round "
                  f"{sum(c['round_s'] for c in ep) / len(ep):.2f}s "
                  f"(gen {sum(c['gen_s'] for c in ep) / len(ep):.2f} + "
                  f"train {sum(c['train_s'] for c in ep) / len(ep):.2f}) ==", flush=True)

        if cfg.save_every and rnd > 0 and rnd % cfg.save_every == 0:
            save_ckpt(rnd)
        if profiler is not None:
            profiler.step()
        if rnd == 0:                             # collect setup garbage once, then freeze
            gc.collect()
            gc.freeze()
            gc.disable()

    if cfg.eval_every:
        log_eval(num_rounds, run_eval(val_prompts, val_gold, "val"))
    if cfg.final_eval:
        res_id = run_eval(test_id_prompts, test_id_gold, "test_id")
        log_eval(num_rounds, res_id)
        res_ood = run_eval(test_ood_prompts, test_ood_gold, "test_ood")
        log_eval(num_rounds, res_ood)
        print(f"\n  == FINAL vs hf-vllm reference: test ID {res_id['accuracy']} "
              f"(ref 87.2) | test OOD {res_ood['accuracy']} (ref 86.2) | "
              f"ref val best 89.0, ~7.5 min wall ==", flush=True)
except BaseException as e:
    run_error = f"{type(e).__name__}: {e}"
    raise
finally:
    if profiler is not None:
        profiler.__exit__(None, None, None)
        trace_path = Path.cwd() / f"trace_{cfg.tag}.json.gz"
        try:
            profiler.export_chrome_trace(str(trace_path))
            print(f"  chrome trace -> {trace_path} (ui.perfetto.dev)", flush=True)
        except Exception as pe:
            print(f"  !! trace export failed: {pe}", flush=True)
    mf.close()
    edf.close()
    total_s = time.perf_counter() - run_wall_t0
    if curve:
        save_ckpt(len(curve))
    result = dict(
        tag=cfg.tag, model=cfg.model_id, k=cfg.k_draws, problems_per_round=cfg.problems_per_round,
        rounds_run=len(curve), budget=cfg.max_tokens, lr=cfg.lr, temperature=cfg.temperature,
        seed=cfg.seed, error=run_error,
        solve_rate_first=(curve[0]["solve_rate"] if curve else None),
        solve_rate_last=(curve[-1]["solve_rate"] if curve else None),
        gen_tok_per_s_med=(sorted(c["gen_tok_per_s"] for c in curve)[len(curve) // 2]
                           if curve else None),
        gen_s_med=(sorted(c["gen_s"] for c in curve)[len(curve) // 2] if curve else None),
        train_s_med=(sorted(c["train_s"] for c in curve)[len(curve) // 2] if curve else None),
        round_s_med=(sorted(c["round_s"] for c in curve)[len(curve) // 2] if curve else None),
        loop_s=round(time.perf_counter() - loop_t0, 1) if curve else None,
        total_s=round(total_s, 1),
        peak_mem_gb=round(torch.cuda.max_memory_reserved() / 2**30, 1),
        evals=eval_rows,
        reference=dict(script="train_qwen_arithmetic-hf-vllm.py",
                       val_best="89.0 @ step 180", test_id=87.2, test_ood=86.2,
                       wall="~7.5 min / 68 rounds of 64x16 on H100 PCIe"))
    (Path.cwd() / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    (cfg.run_dir / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    print(f"\n== train_qwen_arithmetic [{cfg.tag}] ==", flush=True)
    if curve:
        print(f"  rounds {len(curve)} | solve {result['solve_rate_first']} -> "
              f"{result['solve_rate_last']} | round_s med {result['round_s_med']} "
              f"(gen {result['gen_s_med']} + train {result['train_s_med']}) | "
              f"loop {result['loop_s'] / 60:.1f} min (hf-vllm ref: ~7.5) | "
              f"total {total_s / 60:.1f} min | peak {result['peak_mem_gb']} GB", flush=True)
    print(f"  results -> result_{cfg.tag}.json / metrics_{cfg.tag}.csv / "
          f"evals_{cfg.tag}.csv / evals_detail_{cfg.tag}.csv "
          f"| ckpt -> {cfg.run_dir}", flush=True)
    if use_wandb:
        wandb_run.summary.update({k: v for k, v in result.items()
                                  if not isinstance(v, (list, dict))})
        wandb_run.finish()
