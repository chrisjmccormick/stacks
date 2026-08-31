# train_qwen_gsm8k.py

# --------------------------------------------------------------------------------
# § Setup 
# --------------------------------------------------------------------------------
# host-safe: everything above the host_test gate runs without a GPU

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
# Defaults are the verl reference config; change them only as labelled arms.
# Every knob lives here — there is no command line and no env override, so a
# run is defined by the source, and the source is archived into every
# checkpoint (`code`) and the wandb config.

class GSM8KConfig:

    # Run identity
    tag:      str = "run"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Data — pre-tokenized by data/prepare_gsm8k.py, which owns the prompt and
    # the validation set
    prompt:   str  = "boxed_qwen"   # names gsm8k_<prompt>.parquet; see that
                                    # script. boxed_qwen is the phrasing
                                    # Qwen2.5-Math was post-trained on; it and
                                    # `boxed` measured within noise of each
                                    # other and both beat `hash` by ~6.5pp.
    data_dir: Path = Path.home() / ".cache" / "qwen-gsm8k" / "data"

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
    rounds_cap:         int   = 0     # 0 = the full `epochs` horizon
    max_tokens:         int   = 512   # verl's response budget (mean resp 272, ~2% trunc)
    temperature:        float = 1.0
    seed:               int   = 1337  # round schedule + sampler RNG

    # Optimizer — verl actor defaults (constant LR; verl's warmup ratio is 0).
    lr:           float = 1e-6
    weight_decay: float = 0.01
    adam_eps:     float = 1e-8        # seeds the eps_bias_corr schedule
    beta1:        float = 0.9         # § Schedules folds the bias corrections
    beta2:        float = 0.999       # from these, but the COMPILED update in
                                      # § Optimizer hard-codes 1-beta as lerp_
                                      # literals — change both, or the assert
                                      # below stops the run.
    lr_schedule:  str   = "const"     # const (verl) | linear (->0)

    # Training packs (packed varlen, one compiled shape)
    train_t:  int = 16384             # tokens per pack (fixed compile shape)
    sel_cap:  int = 12288             # completion positions per pack, lm_head
                                      # runs ONLY on these (prompt tokens have
                                      # zero grad at the head; ~27% of step FLOPs
                                      # x the skipped fraction is free)
    ce_chunk: int = 2048              # lm_head/CE row chunk (152k vocab: a full
                                      # (SEL, V) fp32 chain would be ~7 GB/temp)
    max_docs: int = 96                # cu_seqlens fixed size (ghost-padded)

    # Engine
    max_seqs:  int   = 256            # decode rows (= top bucket)
    macro_n:   int   = 8              # decode steps per window (one D2H each)
    buckets:   tuple = (32, 64, 128, 192, 256)
    prefill_t: int   = 0              # 0 = auto-size (see § Data)
    max_ctxs:  int   = 48             # prefill cu fixed size
    page:      int   = 256            # FA2 paged KV page size (multiple of 256 required)

    max_prompt:    int   # Derived in § Data: the longest rendered prompt.
    gen_steps:     int   # Derived in § Data: max decode steps a row can run.
    t_row:         int   # Derived in § Data: decode row capacity, whole pages.
    pages_per_row: int
    rope_t:        int   # Derived in § Model Load: rotary cache length.

    # Eval
    eval_every: int  = 30             # 0 = off; val-subset eval through the graphs
    eval_k:     int  = 8
    final_eval: bool = True           # full 1,319-problem test x K=8 at the end
    full_eval_every: int = 0          # 0 = off. Rounds between FULL-test evals
                                      # during the run — ~48 s each, so this is
                                      # an instrument for horizon questions
                                      # ("what did the last 300 rounds buy?"),
                                      # not something a speedrun leaves on.

    # Checkpoints / output
    save_every:    int  = 0           # rounds; 0 = final only
    run_root:      Path = Path.home() / ".cache" / "qwen-gsm8k" / "runs"
    run_dir:       Path               # Derived below: run_root / tag.
    wandb:         bool = True        # per-round rows + evals + final summary
    wandb_run:     str  = ""          # "" = use tag
    wandb_project: str  = "qwen-gsm8k"
    grad_norm:     bool = False       # log the round's gradient norm — an fp32
                                      # reduction over all 494M gradient
                                      # elements, so it is a diagnostic, not
                                      # something a speedrun leaves on

    # Smokes
    fixed_problems: list[int] | None = None   # overfit this fixed problem set instead
    profile:        bool = False      # chrome trace (ui.perfetto.dev)
    prof_wait:      int  = 3
    prof_active:    int  = 1
    host_test:      bool = False      # host-only self-tests, then exit (no GPU)

cfg = GSM8KConfig() # Make config a global, don't pass it around.

# Sanity: the constraints the knobs above must satisfy.
assert cfg.lr_schedule in ("const", "linear"), f"bad lr_schedule {cfg.lr_schedule!r}"
# The betas live in two places: the schedules below fold them, and the
# compiled update carries 1-beta as lerp_ literals it cannot read from cfg.
assert math.isclose(1.0 - cfg.beta1, 0.1) and math.isclose(1.0 - cfg.beta2, 0.001), \
    "betas changed — update the lerp_ literals in adamw_step_fused (§ Optimizer)"
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
    vals = ((k, getattr(cfg, k, None)) for k in GSM8KConfig.__annotations__)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vals}


# --------------------------------------------------------------------------------
# § Stats
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
    n_trimmed:  int   = 0    # truncated-but-correct rollouts cut at their answer
                             # ("clip" is reserved for its PPO meaning)
    fmt_pct:    float = 0.0  # `\boxed{}` share — where format drift shows first
    mean_len:   float = 0.0  # mean completion length (tokens, incl. terminal)

    # Generation (Engine.run_round)
    gen_s:         float = 0.0
    gen_tok:       int   = 0
    gen_tok_per_s: float = 0.0
    occ:           float = 0.0  # % of paid decode slots that kept their token
    t50:           float = 0.0  # seconds to retire half the rows ...
    t90:           float = 0.0  # ... and 90% of them (the tail's cost)

    # Training step (§ Main Loop)
    train_s:       float = 0.0
    n_groups_used: int   = 0
    n_groups_sat:  int   = 0
    n_groups_dead: int   = 0
    reward_std:    float = 0.0  # mean within-group reward std over the LIVE
                                # groups — the advantage denominator, i.e. the
                                # signal a live group carries
    n_docs:        int   = 0
    n_loss_tokens: int   = 0
    n_packs:       int   = 0
    pad_pct:       float = 0.0
    loss_total:    float = 0.0
    grad_norm:     float = 0.0

    # AdamW schedules (§ Main Loop reads the row host-side; see § Schedules)
    lr_nominal:    float = 0.0  # cfg.lr's own schedule — flat on a const run
    lr_bias_corr:  float = 0.0  # what the weights actually saw: 0.32x -> 0.15x
                                # by step 10 -> 0.61x at the last step, never 1x
    eps_bias_corr: float = 0.0
    wd_mul:        float = 0.0

    # Round bookkeeping (§ Main Loop)
    round_s:   float = 0.0
    elapsed_s: float = 0.0   # cumulative loop wall clock, eval time included

stats = RoundStats() # Make stats a global, don't pass it around.


# --------------------------------------------------------------------------------
# § Reward
# --------------------------------------------------------------------------------
# The reward is 1.0 when the LAST parseable \boxed{} holds the gold answer and
# 0.0 otherwise. Nothing else counts — not `####`, not a trailing bare number.
#
# NOT required: that the box be the last thing said, or that the turn ended.
# Qwen opens a LaTeX environment and must close it (`\(\boxed{18}\).`,
# `\[\boxed{18}\n\]`), so the box is almost never final — demanding it left 97%
# of groups all-zero. TECHNIQUES.md § Boxed answers carries the probe that
# settled boxed-only over the looser channels.
#
# fmt% measures \boxed{} presence — the channel where format drift shows first.

_BOXED = re.compile(r"\\boxed\s*\{\s*(-?[0-9][0-9,.]*)\s*\}")
# Whitespace then a LaTeX/math closer — the display-math tail Qwen writes as
# `\boxed{18}\n\]`. Consumed when trimming so the cut cannot orphan a `\[`.
_CLOSER = re.compile(r"\s*(?:\\\)|\\\]|\$\$|\$)")


def norm_answer(s):
    """'65,000' / '$65000.00' / '65000.' -> '65000'. None if not a number."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    # float() does NOT raise on a long digit run — it overflows to inf, and
    # int(inf) raises. That crashed a verl run inside a ray worker. A non-
    # finite value is never a valid GSM8K answer; keep the cleaned digits so
    # they compare unequal to any gold (scoring wrong, the right outcome).
    if not math.isfinite(v):
        return s or None
    return str(int(v)) if v == int(v) else repr(v)


def extract_answer(text):
    """The LAST parseable \\boxed{} — the model's answer, and nothing else."""
    for m in reversed(list(_BOXED.finditer(text))):
        got = norm_answer(m.group(1))
        if got is not None:
            return got
    return None


def has_answer_format(text):
    """The \\boxed{} channel (fmt%) — where format drift shows first."""
    return extract_answer(text) is not None


def anchored_answer_end(text):
    """Char index just past the last parseable \\boxed{}, extended so a cut
    there leaves no dangling delimiters: first through any immediately-
    following non-alphanumeric run (`\\)`, `.`), then through ONE optional
    whitespace-and-closer group.

    That second step matters. Qwen writes display math as `\\[\\n\\boxed{18}\\n\\]`,
    and an extension that halts at whitespace orphans the `\\]` — measured on
    the probe's rollouts, cutting there left UNBALANCED LaTeX in 34% of trims
    (647 of 1,882 in the trim-every-correct counterfactual), teaching the
    policy to open a math environment and stop inside it. Swallowing the
    closer removes ~84% of those."""
    ms = [m for m in _BOXED.finditer(text) if norm_answer(m.group(1)) is not None]
    if not ms:
        return None
    end = ms[-1].end()
    while end < len(text) and not (text[end].isalnum() or text[end].isspace()):
        end += 1
    if (m := _CLOSER.match(text, end)) is not None:
        end = m.end()
        while end < len(text) and not (text[end].isalnum() or text[end].isspace()):
            end += 1
    return end


def trim_to_answer(comp_ids: list[int], text: str) -> list[int]:
    """Training surgery for a correct TRUNCATED completion with an ANCHORED
    answer: token-granular cut at the first token whose decode covers the
    anchor's end, with <|im_end|> attached — positive advantage then
    reinforces answer -> stop instead of the rambling that hit the budget
    (and the trimmed doc trains on fewer tokens). This is the ONE deliberate
    departure from the verl config, which trains full-length responses:
    without it, running to the budget is an attractor for any rollout whose
    text happens to contain the right box, and the policy's eos rate
    collapses. Applied ONLY to truncated
    rollouts — it removes exactly the attractor's fuel; trimming healthy
    naturally-stopped completions as well was measured to shock the policy
    short (solve 23% -> 9% in one update). Returns comp_ids ITSELF when
    there is nothing to cut (callers identity-check). `text` is the
    terminal-stripped decode of comp_ids."""
    end = anchored_answer_end(text)
    if end is None:
        return comp_ids
    body = comp_ids[:-1] if comp_ids and comp_ids[-1] in TERMINALS else comp_ids
    if not text[end:].strip() and len(body) < len(comp_ids):
        return comp_ids                        # already answer + terminal
    lo, hi = 1, len(body)                      # smallest prefix covering the answer
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode(body[:mid])) >= end:
            hi = mid
        else:
            lo = mid + 1
    return body[:lo] + [IM_END]


# --------------------------------------------------------------------------------
# § Data
# --------------------------------------------------------------------------------
# Prompts arrive already rendered. Everything tokenizer-shaped — chat template,
# prompt suffix, gold normalization, the end-anchored answer suffix — plus the
# choice of validation problems is decided by data/prepare_gsm8k.py and frozen
# into a parquet, so a prompt or eval-set change is a dataset change (new file,
# new ids hash) rather than a silent edit inside the training loop.
# `cfg.prompt` picks which one to train on.

print(f"[{cfg.tag}] loading pre-tokenized GSM8K ...", flush=True)
import pyarrow.parquet as pq
from tokenizers import Tokenizer as _RustTokenizer

# IM_END ends the assistant turn; a sampled ENDOFTEXT ends the document — both
# retire a decode row, and ENDOFTEXT doubles as the pack pad.
IM_END = 151645       # <|im_end|>
ENDOFTEXT = 151643    # <|endoftext|>
TERMINALS = (IM_END, ENDOFTEXT)
PAD_ID = ENDOFTEXT

_BUILD_IT = "    python data/prepare_gsm8k.py"
_BUILD_MODEL = "    python data/prepare_model.py"
_data = cfg.data_dir / f"gsm8k_{cfg.prompt}.parquet"
assert _data.exists(), f"{_data} not found — build it with:\n{_BUILD_IT}"
_t = pq.read_table(_data).to_pydict()
assert "is_val" in _t, f"{_data} predates the is_val column — rebuild it with:\n{_BUILD_IT}"
_rows = {"train": {}, "test": {}}
for _s, _i, _gold, _ids, _val in zip(_t["split"], _t["idx"], _t["gold"],
                                     _t["prompt_ids"], _t["is_val"]):
    _rows[_s][_i] = (_gold, _ids, _val)
_train = [_rows["train"][i] for i in range(len(_rows["train"]))]
_test = [_rows["test"][i] for i in range(len(_rows["test"]))]

train_gold, train_prompts = [r[0] for r in _train], [r[1] for r in _train]
test_gold, test_prompts = [r[0] for r in _test], [r[1] for r in _test]
# The reward compares extract_answer(text) — which is None on a completion
# with no parseable box — straight against gold, so a missing gold would
# score every such completion CORRECT. prepare_gsm8k.py's gold_of() rules
# that out; this is the one line that says so where the reward can see it.
assert all(train_gold) and all(test_gold), "a gold answer is empty"

# Decode only: the prompts arrive tokenized, so this reads completions back for
# the reward and the format telemetry. It is the copy data/prepare_model.py left
# beside the weights — the same file that produced the prompt ids.
_tok_file = cfg.data_dir / "tokenizer.json"
assert _tok_file.exists(), f"{_tok_file} not found — build it with:\n{_BUILD_MODEL}"
tokenizer = _RustTokenizer.from_file(str(_tok_file))
assert (tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")) \
    == (IM_END, ENDOFTEXT), "tokenizer disagrees on the Qwen special ids"


def decode(ids: list[int]) -> str:
    """`tokenizers` defaults skip_special_tokens=True; this must be False.
    trim_to_answer compares character offsets between two decodes, so both have
    to agree on whether the specials occupy characters."""
    return tokenizer.decode(ids, skip_special_tokens=False)


def encode(text: str) -> list[int]:
    """Host-test helper. add_special_tokens=False: these are raw completion
    fragments, not turns — nothing may be prepended."""
    return tokenizer.encode(text, add_special_tokens=False).ids


_prompt_lens = [len(p) for p in train_prompts + test_prompts]
cfg.max_prompt = max(_prompt_lens)
assert cfg.max_prompt <= 512, f"prompt of {cfg.max_prompt} tokens exceeds verl's 512 filter"
assert min(_prompt_lens) >= 2, "prompt too short for the forced-last-token split"

# In-loop validation tracker: the test problems data/prepare_gsm8k.py flagged
# `is_val` — 256 MOVERS of a reference trajectory, deliberately NOT
# representative of the full test. It over-reports GAINS (full ~= 10.5 + 0.74 x
# subset), so the gap widens as the score rises — a constant offset is the wrong
# correction. That script owns the choice, the qid list and the measured
# caveats; this is the lookup.
VAL_SUBSET = [i for i, r in enumerate(_test) if r[2]]
assert VAL_SUBSET, f"{_data} flags no is_val rows — rebuild it with:\n{_BUILD_IT}"


def assemble_rounds(n_problems: int, ppr: int, epochs: int, rng: random.Random) -> list[list[int]]:
    """Balanced round schedule: per epoch, sort problems by context length and
    snake-deal into bins of `ppr` — every bin's context sum lands near the mean
    (so one prefill_t covers every round with little padding) AND every bin
    draws one problem from each length stratum (variety). Bin order shuffled.
    The remainder problems (n % ppr) are dropped each epoch, as verl does."""
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
# § Pack planning (host)
# --------------------------------------------------------------------------------

def plan_packs(docs: list[tuple[list[int], list[int], float]]):
    """docs: (prompt_ids, gen_ids, weight) — weight is the per-token loss
    coefficient (advantage / round-total response tokens), applied to every
    completion target of the doc. Returns (packs, pack_stats); each pack is a
    dict of DEVICE tensors at the ONE compiled shape:

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

    The buffers are filled in PINNED host memory and uploaded here, so the
    copies genuinely overlap — non_blocking on pageable memory is a no-op, and
    the caching host allocator both reuses these fixed-size blocks and holds
    them until their copy's event retires.

    First-fit-decreasing over three caps: train_t tokens, sel_cap completion
    positions, max_docs docs. The pad tail is a real attended segment carrying
    zero weight — see TECHNIQUES.md § Padded varlen."""
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
    upload = lambda t: t.to(device, non_blocking=True)
    for members in packs_docs:
        idx = torch.full((cfg.train_t,), PAD_ID, dtype=torch.int32, pin_memory=True)
        pos = torch.zeros(cfg.train_t, dtype=torch.int64, pin_memory=True)
        cu = torch.full((cfg.max_docs + 2,), cfg.train_t, dtype=torch.int32, pin_memory=True)
        cu[0] = 0
        sel = torch.zeros(cfg.sel_cap, dtype=torch.int64, pin_memory=True)
        tgt = torch.zeros(cfg.sel_cap, dtype=torch.int64, pin_memory=True)
        w = torch.zeros(cfg.sel_cap, dtype=torch.float32, pin_memory=True)
        o = s = 0
        for n_doc, i in enumerate(members):
            p, g, wt = docs[i]
            seq = torch.tensor(p + g, dtype=torch.int64)
            n = len(seq) - 1
            idx[o:o + n] = seq[:-1]
            pos[o:o + n] = torch.arange(n)
            ng = len(g)
            sel[s:s + ng] = o + n - ng + torch.arange(ng)  # targets seq[1:]; the
            tgt[s:s + ng] = seq[-ng:]                      # last ng are the completion
            w[s:s + ng] = wt
            s += ng
            o += n
            cu[n_doc + 1] = o
        if o < cfg.train_t:
            # The pad tail is its OWN attended segment, and its ids/positions
            # must VARY — either one wrong NaNs every weight grad while the
            # loss stays finite (TECHNIQUES.md § Padded varlen).
            n_pad = cfg.train_t - o
            idx[o:] = 1 + (torch.arange(n_pad) % 4096)
            pos[o:] = torch.arange(n_pad) % cfg.t_row
            cu[len(members) + 1] = cfg.train_t
            pad_tokens += n_pad
        assert int(pos.max()) < cfg.t_row, "doc position exceeds the rotary cache"
        packs.append(dict(idx=upload(idx), pos=upload(pos), cu=upload(cu),
                          sel=upload(sel), tgt=upload(tgt), w=upload(w),
                          n_tok=o, n_sel=s, n_docs=len(members)))
    return packs, dict(n_packs=len(packs), pad_tokens=pad_tokens,
                       cap_tokens=cfg.train_t * max(1, len(packs_docs)))


# --------------------------------------------------------------------------------
# § Window-event math (host)
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
# § host_test gate
# --------------------------------------------------------------------------------
# Everything below requires a GPU

if cfg.host_test:
    print("[host-test] reward ...")
    assert norm_answer("$65,000.00") == "65000" and norm_answer("9" * 400) == "9" * 400
    assert extract_answer(r"Thus the profit is \(\boxed{65,000}\).") == "65000"
    assert extract_answer(r"\boxed{70,000}") == "70000"
    assert extract_answer(r"first \boxed{70000}, corrected to \boxed{65000}") == "65000"
    # boxed-ONLY: the loose channels no longer score (see § Reward)
    assert extract_answer("The answer is\n#### 65000") is None
    assert extract_answer("I think it's 64999. No wait, 65000.") is None
    assert has_answer_format(r"x \boxed{12}") and not has_answer_format("x\n#### 12")

    print("[host-test] trim-to-answer ...")
    t1 = r"We add 5 and 7 to get \boxed{12}. Now let me double-check this by..."
    c1 = encode(t1)
    k1 = trim_to_answer(c1, t1)
    assert k1[-1] == IM_END and decode(k1[:-1]).rstrip().endswith(r"\boxed{12}.")
    assert "double-check" not in decode(k1)
    t2 = r"The answer is \boxed{12}"
    c2 = encode(t2) + [IM_END]
    assert trim_to_answer(c2, t2) is c2               # already answer + terminal
    t3 = "so the total is 12 dollars and then some words"   # no box: NO trim
    c3 = encode(t3)
    assert trim_to_answer(c3, t3) is c3
    t4 = r"Thus the profit is \(\boxed{65,000}\). And furthermore we can see..."
    k4 = trim_to_answer(encode(t4), t4)
    d4 = decode(k4[:-1])
    assert k4[-1] == IM_END and d4.rstrip().endswith(r"\boxed{65,000}\).")
    assert "furthermore" not in d4
    # display math: the `\]` closer sits past a newline. Cutting before it would
    # orphan the `\[` — the 34%-of-trims defect the probe measured.
    t5 = "Total:\n\\[\n\\boxed{440}\n\\]\n\n### Answer:\nCharlie has 440."
    d5 = decode(trim_to_answer(encode(t5), t5)[:-1])
    assert d5.count("\\[") == d5.count("\\]") == 1, f"unbalanced LaTeX: {d5!r}"
    assert "Charlie" not in d5

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
    print(f"  val subset {len(VAL_SUBSET)} problems | gold sample: {train_gold[0]}")
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
@torch.library.custom_op("qwen_gsm8k::fa_kvcache_paged", mutates_args=("k_cache", "v_cache"))
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
#   .gacc        fp32 gradient accumulator, zeroed by the AdamW kernel as it steps
#   .mantissa    lower 16 bits of the fp32 master (uint16) — master = live<<16|mantissa
#   .exp_avg / .exp_avg_sq   fp32 AdamW moments
#   .gacc_lyr    out-of-graph per-layer views for the 3-D banks (an in-graph
#                bank slice functionalizes into a whole-bank select_scatter)
# The HF checkpoint is bf16, so the initial mantissa is all-zeros (the master
# upcast of a bf16 live is lossless) and live == checkpoint bit-exact.
# The fused QKV (1152, 896) and gate/in (9728, 896) GEMMs are why: ~72 fewer
# kernels per decode step, and they sit below the launch floor, so the fusion is
# ~free throughput.

class Model:
    embed:      Tensor   # (V, D) bf16 — tied: input table AND lm_head
    W_QKV:      Tensor   # (L, 1152, 896)
    b_QKV:      Tensor   # (L, 1152)       Qwen2.5 QKV biases
    W_O:        Tensor   # (L, 896, 896)
    W_gin:      Tensor   # (L, 9728, 896)  [gate | in]  SwiGLU gate + value branches
    W_out:      Tensor   # (L, 896, 4864)
    attn_norm:  Tensor   # (L, 896)  input_layernorm weights
    mlp_norm:   Tensor   # (L, 896)  post_attention_layernorm weights
    final_norm: Tensor   # (896,)
    cos: Tensor          # rotary caches (not trained)
    sin: Tensor

    weight_names = ("embed", "W_QKV", "b_QKV", "W_O", "W_gin", "W_out",
                    "attn_norm", "mlp_norm", "final_norm")

    def __iter__(self):
        return (getattr(self, n) for n in self.weight_names)


print(f"[{cfg.tag}] loading {cfg.model_id} banks ...", flush=True)
t = time.perf_counter()
torch.cuda.reset_peak_memory_stats()

from safetensors.torch import load_file

_BUILD_MODEL = "    python data/prepare_model.py"
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
assert m.W_gin.shape == (cfg.n_layers, 2 * cfg.d_mlp, cfg.d_model)

fp32_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
uint16_zeros = lambda *shape: torch.zeros(*shape, dtype=torch.uint16, device=device)

for p in m:
    p.gacc       = fp32_zeros(*p.shape)
    p.mantissa   = uint16_zeros(*p.shape)   # zeros: master == bf16 checkpoint exactly
    p.exp_avg    = fp32_zeros(*p.shape)
    p.exp_avg_sq = fp32_zeros(*p.shape)
for p in (m.W_QKV, m.b_QKV, m.W_O, m.W_gin, m.W_out, m.attn_norm, m.mlp_norm):
    p.gacc_lyr = list(p.gacc.unbind(0))

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
# per step. The Adam bias corrections (closed 1-beta^t form) fold INTO the
# tables:
#
#   lr * m_hat / (sqrt(v_hat) + eps)
#     = [lr * sqrt(1-b2^t) / (1-b1^t)] * m / (sqrt(v) + eps * sqrt(1-b2^t))
#     = lr_bias_corr * m / (sqrt(v) + eps_bias_corr)
#
# so the kernel reads RAW moments and two schedules. The price of folding the
# denominator's correction is that eps must follow it — eps becomes a schedule
# too.
#
# The verl reference runs a CONSTANT lr (warmup ratio 0), so `lr_nominal` is
# flat at cfg.lr — but the rate the weights SEE is not. Over 467 steps at
# betas (0.9, 0.999), lr_bias_corr runs 0.32x of nominal at step 1, dips to
# 0.15x by step 10, and is still only 0.61x at the last step: this run never
# reaches its own learning rate. That is why both go to the optim panel.

class AdamWSchedule(NamedTuple):
    wd_mul:        Tensor  # 1 - lr*wd    fraction of each weight KEPT per step;
                           #              follows the lr schedule
    lr_bias_corr:  Tensor  # lr * sqrt(1-b2^t) / (1-b1^t)    the step size the
                           #              weights actually see
    eps_bias_corr: Tensor  # eps * sqrt(1-b2^t)    eps, in raw-sqrt(v) units


# Built once, flat: one optimizer step per round, so the tables are `num_rounds`
# long and row `rnd` is the row round `rnd` gathers. The host copies keep the
# field names — a bare name is the schedule as numbers (§ Main Loop logs from
# these), `adamw_sched.<name>` is the same schedule on device.
n_steps = max(1, num_rounds)
steps = np.arange(1, n_steps + 1, dtype=np.float64)
lr_nominal = np.full(n_steps, cfg.lr)
if cfg.lr_schedule == "linear":
    lr_nominal *= 1.0 - np.arange(n_steps) / n_steps
bias_corr2 = np.sqrt(1.0 - cfg.beta2 ** steps)

wd_mul        = 1.0 - lr_nominal * cfg.weight_decay
lr_bias_corr  = lr_nominal * bias_corr2 / (1.0 - cfg.beta1 ** steps)
eps_bias_corr = cfg.adam_eps * bias_corr2

to_device = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
adamw_sched = AdamWSchedule(to_device(wd_mul), to_device(lr_bias_corr),
                            to_device(eps_bias_corr))
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


# The gradient accumulator is zeroed by the same kernel that consumes it — one
# fewer launch per tensor per round, and no way to step without clearing.
@torch.no_grad()
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(live: Tensor, mantissa: Tensor, grad: Tensor,
                     exp_avg: Tensor, exp_avg_sq: Tensor,
                     sched: AdamWSchedule, step: Tensor) -> None:
    p = fp32_master(live, mantissa)
    p.mul_(sched.wd_mul[step])
    exp_avg.lerp_(grad, 0.1)                  # 1 - beta1  (cfg.beta1 = 0.9)
    exp_avg_sq.lerp_(grad.square(), 0.001)    # 1 - beta2  (cfg.beta2 = 0.999)
    p.sub_(sched.lr_bias_corr[step]
           * (exp_avg / (exp_avg_sq.sqrt() + sched.eps_bias_corr[step])))
    writeback_master(p, live, mantissa)
    grad.zero_()                              # gradients accumulate across a
                                              # round's packs, so clear per step


# --------------------------------------------------------------------------------
# § Training Forward/Backward — handwritten, packed varlen, advantage-weighted CE
# --------------------------------------------------------------------------------
# One micro-batch = one pack: a 1-D stream of (prompt+completion) docs with
# per-doc attention isolation via FA3 varlen. No autograd: forward stashes,
# backward accumulates into .gacc. The RL loss is a per-token WEIGHTED CE
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
    xm 0.7GB, h_gin 7.6GB — ~11.4GB, the price of not recomputing the wide GEMM."""
    xb_hat:     Tensor   # (T, D)        attn-norm output, unweighted
    xb_inv_rms: Tensor   # (T, 1) fp32
    q:          Tensor   # (T, 14, 64)   post-rope (what FA consumed)
    k:          Tensor   # (T, 2, 64)    post-rope
    v:          Tensor   # (T, 2, 64)
    y:          Tensor   # (T, 14, 64)   attn out
    lse:        Tensor   #               softmax lse (fp32)
    xm:         Tensor   # (T, D)        post-attn residual (mlp norm recomputed)
    h_gin:      Tensor   # (T, 9728)     fused [gate | in] pre-activation


@torch.no_grad()
@torch.compile(dynamic=False, fullgraph=True)
def forward_backward(idx, pos, cu_seqlens, sel, tgt_sel, w_sel):
    """One pack: forward, stash, explicit backward into .gacc. Returns the
    summed weighted CE (the round's token-mean pg-loss contribution — the
    normalizer already rode in on w). Compiled: the CE chunk block is written
    for inductor's fusion."""
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
        h_gin = xmn @ m.W_gin[i].mT              # (T, 9728)
        h_gate, h_in = h_gin[:, :cfg.d_mlp], h_gin[:, cfg.d_mlp:]
        x = xm + (F.silu(h_gate) * h_in) @ m.W_out[i].mT
        stash.append(LayerStash(xb_hat=xb_hat, xb_inv_rms=xb_r, q=q, k=k, v=v,
                                y=y, lse=lse, xm=xm, h_gin=h_gin))

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
        m.embed.gacc.add_((logits_grad.mT @ xs).float())
        sel_grads.append(logits_grad @ m.embed)
    xfn_grad = torch.zeros_like(xfn)
    xfn_grad.index_add_(0, sel, torch.cat(sel_grads))   # w=0 pads land as zeros at row 0

    # -----------------------------
    #           Backward
    # -----------------------------
    m.final_norm.gacc.add_((xfn_grad.float() * xf_hat.float()).sum(dim=0))
    stream_grad = _rms_bwd(xfn_grad * m.final_norm, xf_hat, xf_r)

    for i in reversed(range(cfg.n_layers)):
        st = stash[i]
        # --- MLP backward (SwiGLU) ---
        xm_hat, xm_r = _rms_fwd(st.xm)
        xmn = xm_hat * m.mlp_norm[i]
        h_gate, h_in = st.h_gin[:, :cfg.d_mlp], st.h_gin[:, cfg.d_mlp:]
        gate_sig = torch.sigmoid(h_gate)
        silu_gate = h_gate * gate_sig
        act = silu_gate * h_in
        m.W_out.gacc_lyr[i].add_(stream_grad.mT @ act)
        act_grad = stream_grad @ m.W_out[i]
        h_in_grad = act_grad * silu_gate
        h_gate_grad = act_grad * h_in * (gate_sig * (1 + h_gate * (1 - gate_sig)))  # d silu / dh
        h_gin_grad = torch.cat([h_gate_grad, h_in_grad], dim=1)
        m.W_gin.gacc_lyr[i].add_(h_gin_grad.mT @ xmn)
        xmn_grad = h_gin_grad @ m.W_gin[i]
        m.mlp_norm.gacc_lyr[i].add_((xmn_grad.float() * xm_hat.float()).sum(dim=0))
        xm_grad = stream_grad + _rms_bwd(xmn_grad * m.mlp_norm[i], xm_hat, xm_r)

        # --- Attention backward ---
        xbn = st.xb_hat * m.attn_norm[i]
        m.W_O.gacc_lyr[i].add_(xm_grad.mT @ st.y.view(T, -1))
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
        m.b_QKV.gacc_lyr[i].add_(qkv_grad.sum(dim=0, dtype=torch.float32))
        m.W_QKV.gacc_lyr[i].add_(qkv_grad.mT @ xbn)
        xbn_grad = qkv_grad @ m.W_QKV[i]
        m.attn_norm.gacc_lyr[i].add_((xbn_grad.float() * st.xb_hat.float()).sum(dim=0))
        stream_grad = xm_grad + _rms_bwd(xbn_grad * m.attn_norm[i], st.xb_hat, st.xb_inv_rms)
        stash[i] = None                          # free as we go

    # --- token embedding scatter (the tied table's second gradient path) ---
    m.embed.gacc.add_(
        torch.ops.aten.embedding_dense_backward(stream_grad, idx, cfg.d_vocab, -1, False))
    return loss


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
        h_gin = xn2 @ m.W_gin[i].mT
        x = x + (F.silu(h_gin[..., :cfg.d_mlp]) * h_gin[..., cfg.d_mlp:]) @ m.W_out[i].mT
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
        h_gin = xn2 @ m.W_gin[i].mT
        x = x + (F.silu(h_gin[:, :cfg.d_mlp]) * h_gin[:, cfg.d_mlp:]) @ m.W_out[i].mT
    return torch.stack(ks), torch.stack(vs)      # (L, T, H_kv, Dh) x2


def sample(logits: Tensor, inv_temp: Tensor) -> Tensor:
    """Gumbel-max draw == exact softmax sampling at temperature 1/inv_temp —
    the verl sampler (temp 1.0, no top-k, top-p 1.0) with no sort, no cumsum:
    one elementwise pass + one argmax over the fp32 logits. inv_temp lives in
    a 0-D CUDA buffer so eval could retune without re-capturing."""
    e = torch.empty_like(logits).exponential_()
    return (logits * inv_temp - e.log()).argmax(dim=-1)


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
        # Prefill static buffers (compiled, not captured: ~2-4k tokens once a
        # round is ~3 ms of compute; a graph would save ~1 ms of a ~2 s round)
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
        terminal (verl trains on it; text decode strips it), and writes the
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
# § Eval — val-subset tracker in-loop, full test at the end (both through the graphs)
# --------------------------------------------------------------------------------

def make_eval_waves(problem_idxs: list[int], k: int) -> list[list[int]]:
    """Greedy wave assembly under the engine's static caps."""
    waves, cur, cur_tok = [], [], 0
    for i in problem_idxs:
        ctx = len(test_prompts[i]) - 1
        if cur and (len(cur) + 1 > cfg.max_ctxs or cur_tok + ctx > cfg.prefill_t
                    or (len(cur) + 1) * k > cfg.max_seqs):
            waves.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += ctx
    if cur:
        waves.append(cur)
    return waves


def run_eval(problem_idxs: list[int], k: int, label: str) -> dict:
    """mean@k / pass@k at temp 1.0 over the given test problems, scored by the
    training objective: the last \\boxed{} holds gold. pass@k is the exact
    any-correct fraction. Sampler RNG is saved and restored so the training
    rollout stream is bit-identical to an eval-off run."""
    rng_state = torch.cuda.get_rng_state()
    t0 = time.perf_counter()
    n_ok, n_fmt, n_trunc, n_roll = {}, 0, 0, 0
    for wave in make_eval_waves(problem_idxs, k):
        rows = engine.run_round([(i, test_prompts[i], k, cfg.max_tokens) for i in wave])
        for r in rows:
            i = r["meta"]
            text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
            n_ok[i] = n_ok.get(i, 0) + int(extract_answer(text) == test_gold[i])
            n_fmt += has_answer_format(text)
            n_trunc += not r["eos"]
            n_roll += 1
    torch.cuda.set_rng_state(rng_state)
    n_prob = len(problem_idxs)
    out = dict(label=label, n_problems=n_prob, k=k,
               mean_at_k=round(100 * sum(n_ok.values()) / max(1, n_roll), 2),
               pass_at_k=round(100 * sum(v > 0 for v in n_ok.values()) / max(1, n_prob), 2),
               fmt_pct=round(100 * n_fmt / max(1, n_roll), 1),
               trunc_pct=round(100 * n_trunc / max(1, n_roll), 1),
               eval_s=round(time.perf_counter() - t0, 1))
    # Per-problem counts, so two evals can be compared PAIRED. At 1,319 problems
    # an unpaired mean@8 carries ~1.1pp of standard error — wide enough to hide
    # the difference between two horizons; per problem, the difficulty cancels.
    out["per_problem"] = [(i, n_ok[i]) for i in problem_idxs]
    return out


# --------------------------------------------------------------------------------
# § Warmup — capture graphs, compile prefill + training step
# --------------------------------------------------------------------------------

engine = Engine()
build_s = time.perf_counter() - run_wall_t0
t = time.perf_counter()
engine.capture()

# Compile forward_backward on one dummy pack (weights untouched; w=0 so every
# grad lands as an exact zero — and cleared here regardless, so the AdamW warmup
# below lerps its moments toward an exact zero).
_t = time.perf_counter()
_dummy_docs = [([3 + (j % 97) for j in range(80)],
                [5 + (j % 89) for j in range(200)] + [IM_END], 0.0)
               for _ in range(cfg.train_t // 280 + 1)]
_packs, _ = plan_packs(_dummy_docs)
_pk = _packs[0]
forward_backward(_pk["idx"], _pk["pos"], _pk["cu"], _pk["sel"], _pk["tgt"], _pk["w"])
for p in m:
    p.gacc.zero_()
del _dummy_docs, _packs, _pk
torch.cuda.synchronize()
print(f"    train fwd+bwd compile: {time.perf_counter() - _t:5.1f}s", flush=True)

# Compile the AdamW kernel per tensor shape NOW, against the zeroed gradients
# left above, with the REAL tables at a throwaway step-0 index — so round 0
# pays no compile stall, and nothing changes: with g = 0 the moments lerp toward
# 0 from 0 (untouched), the update term is exactly 0/eps_bias_corr = 0, and
# wd_mul's 1-1e-8 nudge on the master is under half an fp32 ULP, so the multiply
# rounds back to the same bits and the writeback truncation stores exactly
# what it read. t_step itself is untouched — round 0 still runs row 0.
# (The neutral-coefficient tables this replaces were shape (1,) against the real
# tables' (n_steps,): dynamic=False specializes on shape, so the "warmed" kernels
# silently recompiled at round 0 anyway — measured 11.7 s of round-0 train_s
# that warming with adamw_sched itself removes.)
_t = time.perf_counter()
for p in m:
    adamw_step_fused(p, p.mantissa, p.gacc, p.exp_avg, p.exp_avg_sq,
                     adamw_sched, torch.zeros(1, dtype=torch.int64, device=device))
torch.cuda.synchronize()
print(f"    adamw kernels ({len(m.weight_names)} shapes): "
      f"{time.perf_counter() - _t:5.1f}s", flush=True)
warm_s = time.perf_counter() - t
print(f"  build {build_s:.0f}s + capture/compile {warm_s:.0f}s | "
      f"peak mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
assert int(t_step.item()) == 0


# --------------------------------------------------------------------------------
# § Logging — the round CSV, the eval CSVs, wandb, checkpoints
# --------------------------------------------------------------------------------
# Two consumers of the same rounds, with different needs. The CSV is FLAT and
# the RoundStats dataclass IS its schema. wandb wants those fields grouped into
# panels, and its step axis pinned to the round number.

cfg.run_dir.mkdir(parents=True, exist_ok=True)

# A run page with 28 ungrouped charts is unreadable, so the four questions
# actually asked of a round get their own panel. A field not named here lands
# under `train/`.
WANDB_GROUPS = {
    "time":   ("round_s", "gen_s", "train_s", "t50", "t90", "gen_tok_per_s", "occ",
               "elapsed_s"),
    "length": ("mean_len", "gen_tok", "n_eos", "n_trunc", "n_trimmed"),
    "reward": ("solve_rate", "n_correct", "fmt_pct"),
    "groups": ("n_groups_used", "n_groups_sat", "n_groups_dead", "reward_std",
               "n_docs"),
    "optim":  ("lr_nominal", "lr_bias_corr", "eps_bias_corr", "wd_mul"),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(RoundStats)}, \
    "WANDB_GROUPS names a field RoundStats does not have"


# Checked before wandb.init rather than catching its failure: an unattended run
# with no credentials should say so immediately, not stall inside init.
def _wandb_available() -> bool:
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return True
    for name in (".netrc", "_netrc"):
        try:
            if "api.wandb.ai" in (Path.home() / name).read_text():
                return True
        except Exception:
            pass
    return False


use_wandb = cfg.wandb and _wandb_available()
if use_wandb:
    try:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run,
                               config=config_dict() | dict(num_rounds=num_rounds))
        # Every .log() pins step=rnd. An eval is an extra .log() INSIDE a round,
        # so on wandb's own autoincrementing counter each later round would sit
        # further ahead of its own number; pinning also merges an eval into the
        # row of the round it belongs to. define_metric names that axis `round`.
        wandb.define_metric("round")
        wandb.define_metric("*", step_metric="round")
    except Exception as e:
        print(f"[{cfg.tag}] wandb unavailable ({e}) — CSV/JSON only", flush=True)
        use_wandb = False
elif cfg.wandb:
    print(f"[{cfg.tag}] no wandb credentials — CSV/JSON only", flush=True)


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

# Per-problem eval detail — one row per (eval, problem). Small (a full-test eval
# is 1,319 rows) and the only way to compare two horizons paired.
edf = open(Path.cwd() / f"evals_detail_{cfg.tag}.csv", "w", newline="")
edw = csv.DictWriter(edf, fieldnames=["round", "label", "idx", "n_ok", "k"])
edw.writeheader()


def log_eval(rnd: int, res: dict) -> None:
    per_problem = res.pop("per_problem")
    edw.writerows(dict(round=rnd, label=res["label"], idx=i, n_ok=ok, k=res["k"])
                  for i, ok in per_problem)
    edf.flush()
    res = dict(round=rnd, **res)
    eval_rows.append(res)
    print(f"  [eval {rnd:4d}] {res['label']}({res['n_problems']}): "
          f"mean@{res['k']} {res['mean_at_k']:5.2f} | "
          f"pass@{res['k']} {res['pass_at_k']:5.2f} | fmt {res['fmt_pct']:4.1f}% | "
          f"trunc {res['trunc_pct']:4.1f}% | {res['eval_s']}s", flush=True)
    with open(Path.cwd() / f"evals_{cfg.tag}.csv", "w", newline="") as ef:
        wtr = csv.DictWriter(ef, fieldnames=list(eval_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(eval_rows)
    if use_wandb:
        # The label goes IN the key: both evals sample the GSM8K test split, so
        # unlabelled they interleave into one series — and the subset is drawn
        # from the movers only, so it is not an unbiased estimate of the full
        # number. The CSV keeps them apart with its `label` column.
        lbl, k = res["label"], res["k"]
        wandb_run.log({"round": rnd,
                       f"eval/{lbl}_mean_at_{k}": res["mean_at_k"],
                       f"eval/{lbl}_pass_at_{k}": res["pass_at_k"],
                       f"eval/{lbl}_fmt_pct": res["fmt_pct"],
                       f"eval/{lbl}_trunc_pct": res["trunc_pct"]}, step=rnd)


# Optional pytorch profiling
profiler = None
if cfg.profile:
    from torch.profiler import ProfilerActivity, profile as torch_profile
    num_rounds = min(num_rounds, cfg.prof_wait + 1 + cfg.prof_active)
    profiler = torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,
        schedule=torch.profiler.schedule(wait=cfg.prof_wait, warmup=1, active=cfg.prof_active, repeat=1))
    profiler.__enter__()

# --------------------------------------------------------------------------------
# § Main Loop
# --------------------------------------------------------------------------------

curve: list[dict] = []
run_error = None
loop_t0 = time.perf_counter()

try:
    for rnd in range(num_rounds):
        
        # -- evaluation ------------------------------------------------------
        if cfg.eval_every and rnd % cfg.eval_every == 0:
            log_eval(rnd, run_eval(VAL_SUBSET, cfg.eval_k, "subset"))
        if cfg.full_eval_every and rnd % cfg.full_eval_every == 0:
            log_eval(rnd, run_eval(list(range(len(test_prompts))), cfg.eval_k, "full"))
        
        r_t0 = time.perf_counter()
        stats = RoundStats(round=rnd)

        # -- generation ------------------------------------------------------
        idxs = cfg.fixed_problems if cfg.fixed_problems is not None else round_schedule[rnd]
        specs = [(i, train_prompts[i], cfg.k_draws, cfg.max_tokens) for i in idxs]
        rows = engine.run_round(specs)

        # -- grade, group, advantage -> the round's training docs -------------
        # Rewards are per ROLLOUT, advantages are per GROUP and the token-mean
        # normalizer is per ROUND, so this is two passes and cannot be one:
        # grade every rollout (accumulating the round's response tokens), then
        # turn each group's rewards into the per-token weights the packs carry.
        by_pid: dict[int, tuple[list, list]] = {}
        n_roll, n_fmt, n_total_tok, len_sum = len(rows), 0, 0, 0
        for r in rows:
            pid = r["meta"]
            text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
            rw = float(extract_answer(text) == train_gold[pid])
            comp = r["ids"]
            if rw == 1.0 and not r["eos"]:       # truncated-but-correct only
                trimmed = trim_to_answer(comp, text)
                if trimmed is not comp:          # surgery happened: now ends in EOS
                    comp = trimmed
                    stats.n_trimmed += 1
            stats.n_correct += rw == 1.0
            stats.n_eos += r["eos"]
            n_fmt += has_answer_format(text)
            n_total_tok += len(comp)             # EVERY response token in the
                                                 # round, dead groups included:
                                                 # their tokens dilute the mean
                                                 # exactly as verl's masked_mean
            len_sum += len(r["ids"])             # what the model actually
                                                 # generated, before the trim
            comps, rews = by_pid.setdefault(pid, ([], []))
            comps.append(comp)
            rews.append(rw)

        # verl's GRPO advantage: (r - mean) / (std + 1e-6), std with ddof=1
        # (verl uses torch.std, which is Bessel-corrected). An all-equal group
        # is exactly zero everywhere, so it carries no gradient and is skipped
        # outright — its tokens still count in the normalizer above. Dividing by
        # that normalizer here is what lets the packed forward just sum.
        docs = []
        std_sum = 0.0
        for pid, (comps, rews) in by_pid.items():
            r = np.asarray(rews, dtype=np.float64)
            if r.size < 2 or (r == r[0]).all():
                stats.n_groups_sat += int(r[0] >= 1.0)
                stats.n_groups_dead += int(r[0] <= 0.0)
                continue
            stats.n_groups_used += 1
            std = r.std(ddof=1)
            std_sum += std
            adv = (r - r.mean()) / (std + 1e-6)
            docs += [(train_prompts[pid], c, float(a) / max(1, n_total_tok))
                     for c, a in zip(comps, adv)]
        stats.n_docs = len(docs)
        stats.reward_std = round(std_sum / max(1, stats.n_groups_used), 4)

        # -- train -----------------------------------------------------------
        t_train0 = time.perf_counter()
        loss_total = 0.0
        if docs:
            packs, pstats = plan_packs(docs)
            for pk in packs:
                loss_total += float(forward_backward(
                    pk["idx"], pk["pos"], pk["cu"], pk["sel"], pk["tgt"], pk["w"]))
                stats.n_loss_tokens += pk["n_sel"]
            stats.n_packs = pstats["n_packs"]
            stats.pad_pct = round(100.0 * pstats["pad_tokens"] / pstats["cap_tokens"], 1)
        if cfg.grad_norm:
            stats.grad_norm = round(math.sqrt(
                sum(float((p.gacc.float() ** 2).sum()) for p in m)), 6)

        # -- AdamW step ------------------------------------------------------
        # Step optimizer for each parameter, zeroing grads on the way out.
        for p in m:
            adamw_step_fused(p, p.mantissa, p.gacc, p.exp_avg, p.exp_avg_sq,
                             adamw_sched, t_step)
        t_step.add_(1)
        
        torch.cuda.synchronize()
        stats.loss_total = round(loss_total, 6)
        stats.train_s = round(time.perf_counter() - t_train0, 2)

        # -- telemetry -------------------------------------------------------
        stats.n_rollouts = n_roll
        stats.solve_rate = round(stats.n_correct / max(1, n_roll), 4)
        stats.n_trunc = n_roll - stats.n_eos
        stats.fmt_pct = round(100 * n_fmt / max(1, n_roll), 1)
        stats.mean_len = round(len_sum / max(1, n_roll), 1)
        step = min(rnd, n_steps - 1)         # the row the kernel just gathered,
        stats.lr_nominal    = float(lr_nominal[step])      # read from the host
        stats.lr_bias_corr  = float(lr_bias_corr[step])    # copies — syncing
        stats.eps_bias_corr = float(eps_bias_corr[step])   # t_step would stall
        stats.wd_mul        = float(wd_mul[step])          # the round
        stats.round_s = round(time.perf_counter() - r_t0, 2)
        el = time.perf_counter() - loop_t0
        stats.elapsed_s = round(el, 2)
        row = asdict(stats)
        curve.append(row)
        mw.writerow(row)
        mf.flush()
        eta = el / (rnd + 1) * (num_rounds - rnd - 1)
        print(f"  [{rnd:3d}/{num_rounds}] ({100 * (rnd + 1) / num_rounds:5.1f}%) "
              f"{stats.round_s:5.2f}s ({stats.gen_s:.2f} gen / {stats.train_s:.2f} trn) | "
              f"solve {100 * stats.solve_rate:5.1f}% | len {stats.mean_len:5.1f} | "
              f"trunc {stats.n_trunc:3d} | "
              f"dead {stats.n_groups_dead:2d}/{len(idxs)} rstd {stats.reward_std:.2f} | "
              f"total {el / 60:5.1f}m | eta {eta / 60:4.1f}m", flush=True)
        if use_wandb:
            wandb_run.log({"round": rnd,
                           **{f"{_WANDB_PREFIX.get(k, 'train')}/{k}": v
                              for k, v in row.items() if k != "round"}}, step=rnd)

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
        log_eval(num_rounds, run_eval(VAL_SUBSET, cfg.eval_k, "subset"))
    if cfg.final_eval:
        log_eval(num_rounds, run_eval(list(range(len(test_prompts))), cfg.eval_k, "full"))
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
        evals=eval_rows)
    (Path.cwd() / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    (cfg.run_dir / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    print(f"\n== train_qwen_gsm8k [{cfg.tag}] ==", flush=True)
    if curve:
        print(f"  rounds {len(curve)} | solve {result['solve_rate_first']} -> "
              f"{result['solve_rate_last']} | round_s med {result['round_s_med']} "
              f"(gen {result['gen_s_med']} + train {result['train_s_med']}) | "
              f"loop {result['loop_s'] / 60:.1f} min | "
              f"total {total_s / 60:.1f} min | peak {result['peak_mem_gb']} GB", flush=True)
    print(f"  results -> result_{cfg.tag}.json / metrics_{cfg.tag}.csv / "
          f"evals_{cfg.tag}.csv / evals_detail_{cfg.tag}.csv "
          f"| ckpt -> {cfg.run_dir}", flush=True)
    if use_wandb:
        wandb_run.summary.update({k: v for k, v in result.items()
                                  if not isinstance(v, (list, dict))})
        wandb_run.finish()
