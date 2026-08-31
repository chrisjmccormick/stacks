# train_qwen_arithmetic.py
#
# Single-file GRPO speedrun of Qwen2.5-0.5B-Instruct on basic arithmetic
# (add, subtract, multiply, divide; all integers) which runs on a free
# Colab T4 GPU.
#
# See README.md for details, and TECHNIQUES.md for deeper dives on the
# nuanced decisions. 
#
# Sections are marked with '# @', view the outline with `grep -n '# @'`
# 
# Keep the file ASCII-only to avoid issues with the Colab CLI.

# --------------------------------------------------------------------------------
# @ Setup
# --------------------------------------------------------------------------------
# Host-safe: everything down to @ Host-Test Gate runs without a GPU.

import os
import sys
import time as _time
run_wall_t0 = _time.perf_counter()
del _time

# Provenance: the source is archived into checkpoints. Under `colab run` /
# a notebook cell there is no file to read (sys.argv[0] is a basename that
# does not exist on the VM), so this is best-effort.
try:
    with open(sys.argv[0], "r", encoding="utf-8") as _f:
        code = _f.read()
except Exception:
    code = ""

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
# @ Config
# --------------------------------------------------------------------------------
# Every knob lives here -- there is no command line and no env override, so a
# run is defined by the source, and the source is archived into every
# checkpoint (`code`) and the wandb config.

class T4Config:

    # Run identity
    tag:        str = "t4"
    model_id:   str = "Qwen/Qwen2.5-0.5B-Instruct"   # identity only: the weights
                                      # this script downloads are pre-banked in
                                      # fp16, off a repo of ours (see @ Prepare)

    # Data -- downloaded pre-tokenized (see @ Prepare): train 10,000 / val 200 /
    # test_id 400 / test_ood 400. Same cache the H100 script uses.
    data_dir: Path = Path.home() / ".cache" / "qwen-arithmetic" / "data"

    # Model (Qwen2.5-0.5B -- asserted against the banks' sidecar at load)
    n_layers:   int = 24
    d_model:    int = 896
    n_qo_heads: int = 14
    n_kv_heads: int = 2
    d_head:     int = 64
    d_mlp:      int = 4864
    d_vocab:    int = 151936          # 2374 x 64 -- arrives padded
    rope_theta: float = 1_000_000.0
    rms_eps:    float = 1e-6

    d_q:   int   # Derived below: the fused QKV column split, [Q | K | V].
    d_kv:  int
    d_qkv: int
    half:  int   # Derived below: rotary splits the head dim in half.
    group: int   # Derived below: query heads per KV head (GQA).

    # Rollouts / rounds
    k_draws:            int   = 16    # rollouts per problem per round
    problems_per_round: int   = 16
    epochs:             int   = 1
    rounds_cap:         int   = 200   # 0 = the full `epochs` horizon (625 rounds)
    max_tokens:         int   = 320   # the H100 script runs 640 to clear the
                                      # UNTRAINED model's p99 (~512). On a T4
                                      # a decode step floors at ~7 ms (1 GB of
                                      # weights per step at 270 GB/s), so the
                                      # last few rows of a round running to
                                      # the budget cost ~5 s at 640 -- more
                                      # than the other 250 rows together. 320
                                      # halves that tail (and the KV cache);
                                      # trained, p99 is ~75, so it truncates
                                      # nothing past the first rounds.
    tail_windows:       int   = 8     # Stragglers get 8x8=64 tokens to wrap-up.

    temperature:        float = 1.0
    seed:               int   = 1337  # round schedule + sampler RNG

    # Reward weights (@ Reward). uses_method is gated on correct, so rewards
    # take values in {0, w_correct, w_correct + w_method}.
    w_correct: float = 1.0
    w_method:  float = 1.0

    # Optimizer -- AdamW on fp32 masters, constant LR (H100-validated 1e-6).
    # The betas (0.9 / 0.999) are not config: they are literals in @ Schedules
    # and @ Optimizer, written into the compiled update itself.
    lr:           float = 1e-6
    weight_decay: float = 0.01
    adam_eps:     float = 1e-8        # seeds the eps_t schedule (@ Schedules)
    lr_schedule:  str   = "const"     # const | linear (->0)

    # Precision (@ Model Load, @ Optimizer, @ Trainer). Live weights, grads and
    # activations are fp16; masters and every reduction are fp32.
    loss_scale: float = 4096.0        # static, with backoff: a non-finite
                                      # gradient skips the step and halves it
                                      # (divided out inside the fused AdamW,
                                      # from a device scalar)

    # Training packs (packed varlen; a pack is one fwd/bwd)
    train_t:      int = 2048          # tokens per pack (cap)
    sel_cap:      int = 1536          # completion positions per pack (cap);
                                      # lm_head runs ONLY on these
    ce_chunk:     int = 512           # lm_head/CE row chunk (152k vocab: each
                                      # (chunk, V) fp32 temp is 0.6 MB/row)
    max_docs:     int = 64            # docs per pack (cap); arithmetic docs are
                                      # ~70-200 tokens incl. prompt
    pack_quantum: int = 256           # a pack's T and S are trimmed to a
                                      # multiple of this (>= its content), so
                                      # a round's last, half-empty pack does
                                      # not pay for the whole cap

    # Engine
    macro_n:    int   = 8             # decode steps per window (one D2H each)
    min_bucket: int   = 32            # smallest decode-graph row bucket; the
                                      # ladder doubles up to max_seqs
    max_seqs:   int   # Derived below: problems_per_round * k_draws (= the
                      # decode row count and the eval wave width, K=1)
    max_prompt: int   # Derived in @ Data: the longest rendered prompt.
    gen_steps:  int   # Derived in @ Data: max decode steps a row can run.
    t_row:      int   # Derived in @ Data: decode row capacity (KV positions).
    rope_t:     int   # Derived in @ Model Load: rotary cache length.

    # Eval -- GREEDY (K=1) accuracy through the graphs. In-loop: the
    # 200-problem val split. Final: test_id + test_ood (400 each).
    eval_every: int  = 20             # rounds between val evals; 0 = off
    final_eval: bool = True

    # Checkpoints / output
    save_every:    int  = 0           # rounds; 0 = final only
    save_final:    bool = False       # the final ckpt is 1 GB; off by default
                                      # on a throwaway Colab disk
    run_root:      Path = Path.home() / ".cache" / "qwen-arithmetic" / "runs"
    run_dir:       Path               # Derived below: run_root / tag.
    wandb:         bool = True        # per-round rows + evals + final summary
                                      # (silently off without an API key)
    wandb_run:     str  = ""          # "" = use tag
    wandb_project: str  = "qwen-arithmetic-t4"

    # Smokes
    fixed_problems: list[int] | None = None   # overfit this fixed problem set instead
    profile:        bool = False      # chrome trace (ui.perfetto.dev)
    prof_wait:      int = 3
    prof_active:    int = 1
    host_test:      bool = False      # host-only self-tests, then exit (no GPU)

cfg = T4Config()   # Make config a global, don't pass it around.

# Sanity: the constraints the knobs above must satisfy.
assert cfg.lr_schedule in ("const", "linear"), f"bad lr_schedule {cfg.lr_schedule!r}"
assert cfg.sel_cap % cfg.ce_chunk == 0, "sel_cap must divide into CE chunks"
assert cfg.sel_cap <= cfg.train_t
assert cfg.train_t % cfg.pack_quantum == 0 and cfg.pack_quantum % 8 == 0
assert cfg.min_bucket % 8 == 0 and cfg.min_bucket >= 8

# Derived quantities:
cfg.d_q   = cfg.n_qo_heads * cfg.d_head    # 896
cfg.d_kv  = cfg.n_kv_heads * cfg.d_head    # 128
cfg.d_qkv = cfg.d_q + 2 * cfg.d_kv         # 1152 -- fused QKV rows: [Q | K | V]
cfg.half  = cfg.d_head // 2
assert cfg.n_qo_heads % cfg.n_kv_heads == 0
cfg.group = cfg.n_qo_heads // cfg.n_kv_heads   # 7
cfg.max_seqs = cfg.problems_per_round * cfg.k_draws
assert cfg.max_seqs % 8 == 0 and cfg.max_seqs >= cfg.min_bucket

# Derived run paths.
cfg.run_dir   = cfg.run_root / cfg.tag
cfg.wandb_run = cfg.wandb_run or cfg.tag


def config_dict() -> dict:
    """Every cfg field -- class defaults and derived alike -- flattened for the
    wandb config. The annotations carry declaration order."""
    vals = ((k, getattr(cfg, k, None)) for k in T4Config.__annotations__)
    return {k: (str(v) if isinstance(v, (Path, torch.dtype)) else v) for k, v in vals}


# --------------------------------------------------------------------------------
# @ Stats
# --------------------------------------------------------------------------------
# Generation, grading and the training step each write their slice of the
# round's row into this single global; the CSV writer, the wandb row and the
# console line then all read the same object, so the metric set is defined
# once (the CSV header IS the field list) and no sink can drift from another.

@dataclass
class RoundStats:

    round: int = 0

    # Rollouts + grading (@ Main Loop)
    n_rollouts: int   = 0
    n_correct:  int   = 0
    solve_rate: float = 0.0
    n_eos:      int   = 0
    n_trunc:    int   = 0    # budget truncations + tail cuts
    n_tail:     int   = 0    # of which tail cuts (cfg.tail_windows)
    method_pct: float = 0.0  # uses_method share -- the shaped half of the
                             # reward, and where reward hacking would show first
    mean_len:   float = 0.0  # mean completion length (tokens, incl. terminal)

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
    loss_tokens:   int   = 0     # ... and the running total over the run: the
                                 # only honest x-axis for "how much training
                                 # has actually happened", since the per-round
                                 # count decays as groups saturate
    n_packs:       int   = 0
    pad_pct:       float = 0.0
    loss_total:    float = 0.0   # unscaled
    grad_norm:     float = 0.0   # unscaled
    loss_scale:    float = 0.0   # the scale this round's backward ran at
    step_ok:       int   = 0     # 1 = the optimizer stepped; 0 = non-finite
                                 # grads, step skipped, scale halved

    # Round bookkeeping (@ Main Loop)
    lr:        float = 0.0
    round_s:   float = 0.0
    elapsed_s: float = 0.0   # cumulative loop wall clock, eval time included

stats = RoundStats()   # Make stats a global, don't pass it around.

# The wandb panel grouping. The CSV stays flat -- the dataclass IS its schema --
# but a wandb run page with 28 ungrouped line charts is unreadable, so the three
# questions actually asked of a round get their own section: how long did it
# take, how long were the completions, and how well did they score. Everything
# else (the optimizer step, the pack shapes, the group accounting) stays under
# `train/`. A field missing here is not an error; it just lands in `train/`.
WANDB_GROUPS = {
    "time":   ("round_s", "gen_s", "train_s", "t50", "t90", "gen_tok_per_s", "occ",
               "elapsed_s"),
    "length": ("mean_len", "gen_tok", "n_eos", "n_trunc", "n_tail"),
    "reward": ("solve_rate", "n_correct", "method_pct"),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(RoundStats)}, \
    "WANDB_GROUPS names a field RoundStats does not have"


# --------------------------------------------------------------------------------
# @ Reward
# --------------------------------------------------------------------------------
# The answer channel: the LAST number anywhere in the completion is the model's
# answer, compared as an integer against gold. No format demand -- this is the
# scorer the whole experiment line (and its baselines) used, so the accuracy
# numbers stay comparable.
#
# The method channel: correctness alone pays for NOT working the problem -- a
# short answer has fewer places to go wrong -- so a second reward names the
# vocabulary of a written method and hard-zeros the phrasings that dodge one.
# Gated on correctness on purpose. Measured in the reference line: method
# reward at weight 1.0 beat 0.5 beat absent (val 90.5 / 89.0 / 85.5).

_ALL_NUMS = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def last_number(text):
    """The final number in the text as an int, or None. A number we cannot
    turn into an int (1e999 -> inf) is simply not the answer."""
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
# @ Prepare
# --------------------------------------------------------------------------------
# Runs once per box (the hub skips a file already in cfg.data_dir). Nothing
# here builds anything: this script downloads three artifacts of ours and
# starts, so their repo ids are baked in here rather than configured.
#
# The weights arrive already banked and already fp16: one ~940 MB download
# instead of the checkpoint plus a bank-and-cast pass on Colab's two vCPUs, and
# no bf16 transient on a 16 GB card at load. What built them is
# data/prepare_model.py, and the tensors are bit-for-bit the ones this
# script used to produce by casting the raw-H100 baseline's bf16 banks
# (baselines/20260813_raw-h100/data/prepare_model.py) --
# bf16's mantissa fits inside fp16's, so the cast just happened earlier.
#
# The data arrives already tokenized, under `pretokenized/` in the same repo
# the raw splits live in, published by data/prepare_arithmetic.py. So this
# script no longer owns a copy of the prompt template, and cannot drift from
# the H100 line's: it reads token ids and never renders a prompt. A prompt
# change is a dataset change. That script also pins the properties this one
# then assumes -- splits written in idx order, integer golds, prompt lengths
# in range -- so nothing is re-checked here.
#
# The attention kernel arrives the same way: flash-attention-turing, prebuilt
# for the pinned Colab runtime (agent-ops built it, ~12 min of nvcc that nobody
# has to spend again). It is a 4.9 MB wheel, but pip never runs -- the wheel
# holds one .so, which is extracted into the cache and put on sys.path. A
# PyTorch C++ extension is welded to the Python ABI tag, the torch version,
# torch's C++ ABI flag and the CUDA toolchain, so the sidecar records what it
# was built against and the assert below fails on a mismatch rather than
# letting it surface as an `undefined symbol` at import.

IM_END = 151645       # <|im_end|>    ends the assistant turn
ENDOFTEXT = 151643    # <|endoftext|> ends the document; doubles as the pack pad
TERMINALS = (IM_END, ENDOFTEXT)
PAD_ID = ENDOFTEXT
SPLITS = ("train", "val", "test_id", "test_ood")

BANKS_REPO = "ChrisMcCormick/qwen-arithmetic-t4"   # fp16 banks + tokenizer
DATA_REPO = "ChrisMcCormick/basic-arithmetic"      # raw splits + the pre-tokenized copy

# The banks are fp16 and named apart from the H100 line's bf16
# `banks_<model>.safetensors`, so both can sit in this one shared cache dir.
_BANKS_PATH = cfg.data_dir / "banks_fp16_Qwen2.5-0.5B-Instruct.safetensors"
_TOK_PATH = cfg.data_dir / "tokenizer.json"
# Flat in the cache, under the name data/prepare_arithmetic.py writes locally,
# so a box that ran the H100 prep and a box that downloaded share one file.
_DATA_PATH = cfg.data_dir / "arithmetic.parquet"
# The extension lands in its own directory because that directory goes on
# sys.path -- the cache dir itself must not.
_FA_DIR = cfg.data_dir / "fa_turing"
_FA_SO = f"flash_attn_turing.cpython-{sys.version_info[0]}{sys.version_info[1]}-x86_64-linux-gnu.so"

import shutil
from huggingface_hub import hf_hub_download, snapshot_download

print(f"[{cfg.tag}] fetching {BANKS_REPO} + {DATA_REPO} ...", flush=True)
cfg.data_dir.mkdir(parents=True, exist_ok=True)
# The 940 MB half is the banks; the host-only tests need no weights, so they
# take the tokenizer alone. The hub verifies each file's hash on arrival, and
# the banks' shapes are asserted against cfg in @ Model Load.
_want = [_TOK_PATH.name] if cfg.host_test else [_TOK_PATH.name, _BANKS_PATH.name]
snapshot_download(BANKS_REPO, local_dir=str(cfg.data_dir), allow_patterns=_want)
shutil.copyfile(hf_hub_download(DATA_REPO, "pretokenized/qwen2.5.parquet",
                                repo_type="dataset"), _DATA_PATH)
# The one check: a pattern that matched nothing is the only way the hub returns
# quietly empty-handed. Everything past here is allowed to crash.
assert cfg.host_test or _BANKS_PATH.exists(), f"{BANKS_REPO} is missing {_BANKS_PATH.name}"

if not cfg.host_test:                     # the extension needs a GPU to be worth having
    _fa = json.loads(Path(hf_hub_download(BANKS_REPO, "fa_turing/flash_attn_turing.json")).read_text())
    _built = _fa["built_for"]
    _have = (f"cp{sys.version_info[0]}{sys.version_info[1]}", torch.__version__,
             torch._C._GLIBCXX_USE_CXX11_ABI)
    _want = (_built["abi_tag"], _built["torch"], _built["cxx11abi"])
    assert _have == _want, (
        f"the prebuilt flash-attention-turing extension is for {_want} "
        f"(Colab runtime {_built['colab_runtime']}), this box is {_have}. "
        f"Rebuild it with agent-ops/stacks/2026-08-16_0131pm_t4-turing-fa-wheel/"
        f"build_wheel.py and republish, or pin the runtime.")
    if not (_FA_DIR / _FA_SO).exists():
        import zipfile
        _FA_DIR.mkdir(parents=True, exist_ok=True)
        _whl = hf_hub_download(BANKS_REPO, f"fa_turing/{_fa['wheel']}")
        with zipfile.ZipFile(_whl) as _z:
            _z.extract(_FA_SO, str(_FA_DIR))
    sys.path.insert(0, str(_FA_DIR))


# --------------------------------------------------------------------------------
# @ Data
# --------------------------------------------------------------------------------
# The pre-tokenized splits, and the balanced round schedule dealt from them.

import pyarrow.parquet as pq
from tokenizers import Tokenizer as _RustTokenizer

# Split-major, idx-ordered as prepare_arithmetic.py wrote it, so a problem's
# position within its split IS its index.
_t = pq.read_table(_DATA_PATH).to_pydict()
_gold = {s: [] for s in SPLITS}
_prompts = {s: [] for s in SPLITS}
for _s, _g, _ids in zip(_t["split"], _t["gold"], _t["prompt_ids"]):
    _gold[_s].append(int(_g))
    _prompts[_s].append(list(_ids))
del _t

train_gold, train_prompts = _gold["train"], _prompts["train"]
val_gold, val_prompts = _gold["val"], _prompts["val"]
test_id_gold, test_id_prompts = _gold["test_id"], _prompts["test_id"]
test_ood_gold, test_ood_prompts = _gold["test_ood"], _prompts["test_ood"]
print(f"[{cfg.tag}] pre-tokenized arithmetic: "
      + " | ".join(f"{s} {len(_gold[s]):,}" for s in SPLITS), flush=True)

tokenizer = _RustTokenizer.from_file(str(_TOK_PATH))


def decode(ids: list[int]) -> str:
    """`tokenizers` defaults skip_special_tokens=True; keep it False so a
    decode is a faithful readback of exactly the ids the engine produced."""
    return tokenizer.decode(ids, skip_special_tokens=False)


cfg.max_prompt = max(len(p) for p in train_prompts + val_prompts
                     + test_id_prompts + test_ood_prompts)


def assemble_rounds(n_problems: int, ppr: int, epochs: int, rng: random.Random) -> list[list[int]]:
    """Balanced round schedule: per epoch, sort problems by context length and
    snake-deal into bins of `ppr` -- every bin draws one problem from each
    length stratum (variety). Bin order shuffled. The remainder problems
    (n % ppr) are dropped each epoch."""
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
    assert len(cfg.fixed_problems) * cfg.k_draws <= cfg.max_seqs
else:
    round_schedule = assemble_rounds(len(train_prompts), cfg.problems_per_round, cfg.epochs, _rng)
    num_rounds = len(round_schedule)
    if cfg.rounds_cap:
        num_rounds = min(num_rounds, cfg.rounds_cap)
        round_schedule = round_schedule[:num_rounds]
rounds_per_epoch = max(1, (len(train_prompts) // cfg.problems_per_round)
                       if cfg.fixed_problems is None else num_rounds)

# Decode row capacity. A row's K lands at positions [ctx, ctx + steps): the
# first decode step writes the FORCED token, and a window always replays
# macro_n times, so a row runs a macro-ALIGNED budget before the host can
# retire it. Rounded up to a multiple of 64.
cfg.gen_steps = -(-cfg.max_tokens // cfg.macro_n) * cfg.macro_n
cfg.t_row = -(-(cfg.max_prompt - 1 + cfg.gen_steps) // 64) * 64
# A parked (retired) row keeps replaying at bucket cost from position 0 and
# must not run off its slot before the round ends.
assert cfg.gen_steps <= cfg.t_row

print(f"[{cfg.tag}] {cfg.problems_per_round} problems x K={cfg.k_draws} = {cfg.max_seqs} rollouts/round "
      f"x {num_rounds} rounds @ budget {cfg.max_tokens} | max prompt {cfg.max_prompt} | "
      f"row {cfg.t_row} tok | val {len(val_prompts)} / test {len(test_id_prompts)}+{len(test_ood_prompts)}",
      flush=True)


# --------------------------------------------------------------------------------
# @ Advantage + Packing
# --------------------------------------------------------------------------------
# Host-side, between generation and the training step.

def group_advantage(rewards) -> np.ndarray | None:
    """GRPO advantage: (r - mean) / (std + 1e-6), std with ddof=1. None for an
    all-equal group -- the advantage is exactly zero everywhere, so the docs
    carry no gradient and are skipped outright (their tokens still count in
    the token-mean normalizer, which includes every response token in the
    round)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2 or (r == r[0]).all():
        return None
    return (r - r.mean()) / (r.std(ddof=1) + 1e-6)


def _roundup(n: int, q: int) -> int:
    return -(-n // q) * q


def plan_packs(docs: list[tuple[list[int], list[int], float]]):
    """docs: (prompt_ids, gen_ids, weight) -- weight is the per-token loss
    coefficient (advantage / round-total response tokens, times the loss
    scale), applied to every completion target of the doc. Returns (packs,
    pack_stats); each pack is a dict of numpy arrays:

      idx (T,) int32       packed inputs (per doc: seq[:-1])
      pos (T,) int64       rotary positions, restarting at each doc
      cu  (n_seg+1,) int32 segment boundaries: the docs, then the pad tail
      sel (S,) int64       positions of completion targets (lm_head runs
                           only here); padded with 0
      tgt (S,) int64       targets at sel; padded with 0
      w   (S,) fp32        per-token loss weight at sel; padded with 0 (a
                           zero weight zeroes the padded entries' gradient,
                           so the duplicate position-0 entries are inert)
      max_seg              the longest segment (for the attention kernel)

    First-fit-decreasing over three caps: train_t tokens, sel_cap completion
    positions, max_docs docs. T and S are then trimmed to a multiple of
    pack_quantum / ce_chunk that holds the content. The pad tail is a real
    attended segment carrying zero weight, with VARYING ids and positions --
    a constant pad segment can NaN the attention backward (qwen-gsm8k
    TECHNIQUES.md, Padded varlen)."""
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
    pad_tokens = cap_tokens = 0
    for members, n_tok, n_sel in zip(packs_docs, packs_tok, packs_selc):
        T = min(cfg.train_t, _roundup(n_tok, cfg.pack_quantum))
        S = cfg.sel_cap                   # NOT trimmed like T: the CE chunk
                                          # loop is a python `range` over S, so
                                          # a second value of it would compile a
                                          # second graph. The padded rows carry
                                          # w=0 and cost ~2% of the pack.
        idx = np.full(T, PAD_ID, dtype=np.int32)
        pos = np.zeros(T, dtype=np.int64)
        cu = [0]
        sel = np.zeros(S, dtype=np.int64)
        tgt = np.zeros(S, dtype=np.int64)
        w = np.zeros(S, dtype=np.float32)
        o = s = 0
        max_seg = 0
        for i in members:
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
            cu.append(o)
            max_seg = max(max_seg, n)
        if o < T:
            n_pad = T - o
            idx[o:] = 1 + (np.arange(n_pad) % 4096)
            pos[o:] = np.arange(n_pad) % cfg.t_row
            cu.append(T)
            max_seg = max(max_seg, n_pad)
            pad_tokens += n_pad
        cap_tokens += T
        assert int(pos.max()) < cfg.t_row, "doc position exceeds the rotary cache"
        packs.append(dict(idx=idx, pos=pos, cu=np.asarray(cu, dtype=np.int32),
                          sel=sel, tgt=tgt, w=w, max_seg=max_seg,
                          n_tok=o, n_sel=s, n_docs=len(members)))
    return packs, dict(n_packs=len(packs), pad_tokens=pad_tokens,
                       cap_tokens=max(1, cap_tokens))


# What those caps actually cost, run through the planner itself rather than
# derived on paper. A round's docs are (prompt + completion) and the planner
# packs them under three caps at once, so the one that BINDS -- and therefore
# the one worth moving -- changes with completion length: short completions run
# out of CE rows (sel_cap) long before they run out of tokens, long ones run out
# of tokens. The preview below is the worst case, every group alive; saturated
# groups drop out and shrink the real round.

def _pack_preview(glen: int) -> str:
    """One line: what a full max_seqs-rollout round looks like at completion
    length `glen`, and which cap stopped the packs getting fuller."""
    p = [0] * _med_prompt
    packs, st = plan_packs([(p, [0] * glen, 1.0)] * cfg.max_seqs)
    per_pack = max(pk["n_docs"] for pk in packs)
    caps = ((cfg.train_t // (_med_prompt + glen - 1), "tokens"),
            (cfg.sel_cap // glen, "CE rows"),
            (cfg.max_docs, "doc count"))
    tight = min(c for c, _ in caps)          # a tie means BOTH have to move
    binds = " + ".join(n for c, n in caps if c == tight)
    return (f"    gen {glen:4d} tok: {st['n_packs']:3d} packs x <= {per_pack:2d} docs"
            f" | {100 * st['pad_tokens'] / st['cap_tokens']:4.1f}% pad"
            f" | {binds} bind")


_med_prompt = int(np.median([len(p) for p in train_prompts]))
print(f"[{cfg.tag}] packs: <= {cfg.train_t} tok x <= {cfg.sel_cap} CE rows x "
      f"<= {cfg.max_docs} docs | T trimmed to a multiple of {cfg.pack_quantum}, S "
      f"pinned at the cap ({cfg.sel_cap // cfg.ce_chunk} CE chunks of {cfg.ce_chunk})",
      flush=True)
print(f"[{cfg.tag}] a full {cfg.max_seqs}-rollout round at the median {_med_prompt}-token "
      f"prompt, by completion length:", flush=True)
for _gl in (32, 64, 128, cfg.max_tokens):
    print(_pack_preview(_gl), flush=True)


# --------------------------------------------------------------------------------
# @ Window Events
# --------------------------------------------------------------------------------
# Host-side: the decode loop's per-window retirement decision.

def window_events(t_live: np.ndarray, allows: np.ndarray, base: int, terminals):
    """One macro-window's retirements, vectorized. t_live (n, N) sampled tokens
    for the live rows; allows (n,) per-row budgets; base = tokens generated
    before this window. Returns (done, eos, n_take): rows retiring this window,
    whether by terminal (vs budget), and how many of the window's tokens they
    keep. A terminal AT the budget position still counts as eos."""
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
# @ Host-Test Gate
# --------------------------------------------------------------------------------
# Everything below this point requires a GPU.

if cfg.host_test:
    print("[host-test] reward ...")
    assert last_number("12 + 5 gives us 17.") == 17
    assert last_number("So 170 / 2 = 85.0") == 85
    assert last_number("the result is -42") == -42
    assert last_number("no numerals here") is None
    assert last_number("overflowing 1e999") is None
    assert reward_correct("The answer is 85.", 85) == 1.0
    assert reward_correct("I get 84. No wait, 85.", 85) == 1.0
    assert reward_correct("85 is close but I'll say 84.", 85) == 0.0
    assert reward_method("Using long division, 85.", 1.0) == 1.0
    assert reward_method("Simply, it's 85.", 1.0) == 0.0
    assert reward_method("Using long division, 84.", 0.0) == 0.0
    assert grade("By the standard method: 85.", 85) == (2.0, 1.0, 1.0)
    assert grade("85", 85) == (1.0, 1.0, 0.0)
    assert grade("simply 84", 85) == (0.0, 0.0, 0.0)

    print("[host-test] advantage ...")
    assert group_advantage([1.0] * 16) is None and group_advantage([0.0] * 16) is None
    a = group_advantage([1, 0, 0, 0])
    ref = (np.array([1, 0, 0, 0]) - 0.25) / (np.array([1., 0, 0, 0]).std(ddof=1) + 1e-6)
    assert np.allclose(a, ref)
    assert group_advantage([2, 1, 0, 0]) is not None

    print("[host-test] pack planning ...")
    rng = np.random.default_rng(0)
    docs = [(list(rng.integers(1, 1000, rng.integers(30, 60))),
             list(rng.integers(1, 1000, rng.integers(5, cfg.max_tokens))) + [IM_END],
             float(rng.normal())) for _ in range(256)]
    packs, st = plan_packs(docs)
    tot_sel = sum(p["n_sel"] for p in packs)
    assert tot_sel == sum(len(g) for _, g, _ in docs)
    for p in packs:
        T, S = len(p["idx"]), len(p["sel"])
        assert T % cfg.pack_quantum == 0 and S % cfg.ce_chunk == 0
        assert p["n_tok"] <= T <= cfg.train_t and p["n_sel"] <= S <= cfg.sel_cap
        assert p["n_docs"] <= cfg.max_docs
        cu = p["cu"].astype(np.int64)
        assert (np.diff(cu) > 0).all() and cu[0] == 0 and cu[-1] == T
        assert p["max_seg"] == int(np.diff(cu).max())
        # every selected target equals the packed stream shifted by one
        live = p["w"] != 0
        s_idx = p["sel"][live]
        interior = s_idx[(s_idx + 1) % T != 0]
        nxt = p["idx"][interior + 1]
        d_end = np.isin(interior + 1, cu[1:])     # at doc seams the next input is
        assert (p["tgt"][live][(s_idx + 1) % T != 0][~d_end]
                == nxt[~d_end]).all()             # the next doc -- not a target
    # A tiny round trims to the quantum instead of paying the whole cap.
    small = [(list(range(3, 53)), list(range(5, 45)) + [IM_END], 0.5) for _ in range(3)]
    packs, st = plan_packs(small)
    assert len(packs) == 1 and len(packs[0]["idx"]) == cfg.pack_quantum * 2
    assert len(packs[0]["sel"]) == cfg.sel_cap and packs[0]["n_sel"] == 3 * 41

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
          f"rounds {num_rounds} | t_row {cfg.t_row} | max_seqs {cfg.max_seqs}")
    print("[host-test] ALL PASS")
    sys.exit(0)


# --------------------------------------------------------------------------------
# @ CUDA Init
# --------------------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA required (set cfg.host_test for the host-only checks)"
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
try:   # the env var above only counts if torch was not already imported (Colab)
    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
except Exception:
    pass
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
torch.set_grad_enabled(False)          # nothing here uses autograd
# fp16 GEMMs accumulate in fp32 (cuBLAS default) -- never let split-K reduce
# in fp16.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
_cc = torch.cuda.get_device_capability()
_gpu_name = torch.cuda.get_device_name(0)
_total_gb = torch.cuda.mem_get_info()[1] / 2**30
print(f"[{cfg.tag}] {_gpu_name} (sm_{_cc[0]}{_cc[1]}, {_total_gb:.1f} GB) | "
      f"torch {torch.__version__} | cuda {torch.version.cuda}", flush=True)
assert _cc >= (7, 0), "fp16 tensor cores required (sm70+)"


# --------------------------------------------------------------------------------
# @ Attention
# --------------------------------------------------------------------------------
# Two kernels, because no single one covers a T4.
#
# VARLEN -- the training pack's forward and backward, and generation's prefill:
# flash-attention-turing (github.com/ssiu/flash-attention-turing), a
# FlashAttention-2 written for sm75, arriving prebuilt (see @ Prepare). fp16,
# head_dim 64, causal, and both varlen and GQA NATIVE -- so K/V stay at 2 heads,
# dK/dV come back at 2 heads, and three pieces of scaffolding the CUTLASS path
# needed are simply gone: the expansion of K/V to 14 query heads, the dK/dV
# group-sum, and the uninitialized-LSE-padding fix.
#
# DECODE -- PyTorch's vendored CUTLASS FMHA (the op behind SDPA's
# 'mem-efficient' backend), in its padded-KV mode. It has no choice:
# flash-attention-turing has no KV-cache path.
#
# Layout for both: (seq, heads, head_dim), packed varlen via int32 cu_seqlens;
# the CUTLASS op additionally wants a leading batch of 1.
#
# The two were measured against each other on this card at these pack shapes:
# flash-attention-turing runs the attention call 1.03-1.34x, but attention is
# only ~6% of the pack's fwd/bwd, so end to end it is inside the +-2% run-to-run
# spread. It is here for the simpler kernel, not for a speed record; the win
# grows with segment length (1.31x at 2048 tokens, against 84-224 here).
# Measurements: agent-ops/stacks/2026-08-16_0131pm_t4-turing-fa-wheel/.

import flash_attn_turing

_EA_FWD = torch.ops.aten._efficient_attention_forward
_ATTN_SCALE = cfg.d_head ** -0.5
_NO_MASK = 0


# Both halves are wrapped as custom ops with fake implementations, because the
# pack fwd/bwd is compiled fullgraph and a raw pybind11 extension function is
# not traceable -- dynamo would break the graph on it. (The CUTLASS backward
# this replaces needed the same wrapper for a different reason: torch 2.11's
# meta function for it mis-binds `scale`.) The .contiguous() calls are no-ops
# on everything the training path passes -- they are here for prefill, whose V
# is a stride-1152 view of the fused QKV output.

@torch.library.custom_op("t4::fa_fwd", mutates_args=())
def _fa_fwd_op(q: Tensor, k: Tensor, v: Tensor, cu: Tensor,
               max_seg: int) -> tuple[Tensor, Tensor]:
    return flash_attn_turing.varlen_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), cu, cu,
        max_seg, max_seg, _ATTN_SCALE, True)


@_fa_fwd_op.register_fake
def _(q, k, v, cu, max_seg):
    return (torch.empty_like(q),
            q.new_empty((cu.shape[0] - 1, q.shape[1], max_seg), dtype=torch.float32))


@torch.library.custom_op("t4::fa_bwd", mutates_args=())
def _fa_bwd_op(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor,
               lse: Tensor, cu: Tensor,
               max_seg: int) -> tuple[Tensor, Tensor, Tensor]:
    return flash_attn_turing.varlen_bwd(
        q, k, v, out, lse, dout.contiguous(), cu, cu,
        max_seg, max_seg, _ATTN_SCALE, True)


@_fa_bwd_op.register_fake
def _(dout, q, k, v, out, lse, cu, max_seg):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def attn_varlen_fwd(q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_seg: int):
    """Packed causal attention. q (T, H_q, Dh), k/v (T, H_kv, Dh) fp16;
    cu (n_seg+1,) int32 on device. -> (out (T, H_q, Dh),
    lse (n_seg, H_q, max_seg) fp32), the lse passed straight back to
    attn_varlen_bwd. Prefill discards it."""
    return _fa_fwd_op(q, k, v, cu, max_seg)


def attn_varlen_bwd(dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor,
                    lse: Tensor, cu: Tensor, max_seg: int):
    """Backward for attn_varlen_fwd -> (dq (T,H_q,Dh), dk (T,H_kv,Dh), dv).
    dK/dV arrive at the KV head count already: GQA is inside the kernel."""
    return _fa_bwd_op(dout, q, k, v, out, lse, cu, max_seg)


# --------------------------------------------------------------------------------
# @ Model Load
# --------------------------------------------------------------------------------
# The banks arrive already assembled (see @ Prepare): per-layer matrices
# STACKED into (L, ...) banks, QKV and gate/up CONCATENATED into one GEMM each.
# Every trained tensor is a plain fp16 CUDA tensor (the LIVE weights, seen in
# place by every captured graph) carrying its state as attached attributes:
#   .master      fp32 shadow; the optimizer's tensor of record
#   .gacc        gradient accumulator (fp16 for the 5 big matrices, fp32 for
#                norms/biases), zeroed after each optimizer step
#   .exp_avg / .exp_avg_sq   AdamW moments (fp32)
#   .grad_slices out-of-graph per-layer views for the 3-D banks

class Model:
    embed:      Tensor   # (V, D) fp16 -- tied: input table AND lm_head
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
    big_names = ("embed", "W_QKV", "W_O", "W_GU", "W_down")

    def __iter__(self):
        return (getattr(self, n) for n in self.weight_names)


print(f"[{cfg.tag}] loading {cfg.model_id} banks ...", flush=True)
t = time.perf_counter()
torch.cuda.reset_peak_memory_stats()

from safetensors.torch import load_file

m = Model()
_sd = load_file(str(_BANKS_PATH), device=str(device))   # fp16, straight to device
for _n in Model.weight_names:
    # Already the live dtype and already contiguous, so both calls are no-ops
    # returning the loaded tensor itself -- the live weights ARE the file, and
    # nothing here costs a second copy of the 1 GB.
    setattr(m, _n, _sd[_n].to(torch.float16).contiguous())
del _sd
torch.cuda.empty_cache()

assert m.embed.shape == (cfg.d_vocab, cfg.d_model)
assert m.W_QKV.shape == (cfg.n_layers, cfg.d_qkv, cfg.d_model)
assert m.W_GU.shape == (cfg.n_layers, 2 * cfg.d_mlp, cfg.d_model)

for _n in Model.weight_names:
    p = getattr(m, _n)
    p.pname = _n                               # (Tensor.name is a read-only property)
    p.master = p.float()                       # == live, exactly, at init
    gd = torch.float16 if _n in Model.big_names else torch.float32
    p.gacc = torch.zeros(p.shape, dtype=gd, device=device)
    p.exp_avg = torch.zeros(p.shape, dtype=torch.float32, device=device)
    p.exp_avg_sq = torch.zeros(p.shape, dtype=torch.float32, device=device)
    if p.dim() >= 2 and _n != "embed":
        p.grad_slices = list(p.gacc.unbind(0))
del p

# ==== Rotary cache ====
# HF/Qwen convention (rotate_half, non-interleaved): channel j pairs with
# j + head_dim/2; cos/sin are (T, head_dim/2) and broadcast over both halves.
# Forward rotation: y1 = q1*cos - q2*sin ; y2 = q2*cos + q1*sin.
cfg.rope_t = cfg.t_row
_inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.d_head, 2, dtype=torch.float32, device=device) / cfg.d_head))
_freqs = torch.outer(torch.arange(cfg.rope_t, dtype=torch.float32, device=device), _inv_freq)
m.cos = _freqs.cos().to(torch.float16)    # (cfg.rope_t, 32)
m.sin = _freqs.sin().to(torch.float16)
del _inv_freq, _freqs

_n_params = sum(p.numel() for p in m)
# live fp16 (2) + master fp32 (4) + two fp32 moments (8) + grad (2 big / 4 small)
_state_gb = sum(p.numel() * (14 + (2 if p.pname in Model.big_names else 4))
                for p in m) / 2**30
print(f"[{cfg.tag}] loaded: {_n_params:,} params in "
      f"{time.perf_counter() - t:.0f}s | live fp16 + master fp32 + fp32 moments "
      f"+ fp16 grads = {_state_gb:.1f} GB", flush=True)


# --------------------------------------------------------------------------------
# @ Schedules
# --------------------------------------------------------------------------------
# Every number the fused kernel multiplies by is folded into a per-step table
# up front; the kernel gathers row `t_step` on device, so the step itself sets
# nothing per round, syncs nothing, and never recompiles as the step count
# advances. The Adam bias corrections (closed 1-beta^t form, with t = steps
# actually TAKEN -- a skipped step does not advance it) fold INTO the tables:
#
#   lr * m_hat / (sqrt(v_hat) + eps)
#     = [lr * sqrt(1-b2^t) / (1-b1^t)] * m / (sqrt(v) + eps * sqrt(1-b2^t))
#     = lr_t * m / (sqrt(v) + eps_t)
#
# so the kernel reads RAW moments and two schedules. The price of folding the
# denominator's correction is that eps must follow it -- eps becomes a
# schedule too. The betas (0.9 / 0.999) appear only as literals, here and in
# the compiled update.

class AdamWTabs(NamedTuple):
    wd_mul: Tensor  # 1 - lr*wd                        decoupled weight decay
    lr_t:   Tensor  # lr * sqrt(1-b2^t) / (1-b1^t)     bias-corrected step size
    eps_t:  Tensor  # eps * sqrt(1-b2^t)               eps, in raw-sqrt(v) units


def build_schedules(n_steps: int):
    N = max(1, n_steps)
    t1 = np.arange(1, N + 1, dtype=np.float64)
    lr = np.full(N, cfg.lr)
    if cfg.lr_schedule == "linear":
        lr *= 1.0 - np.arange(N) / N
    bc2 = np.sqrt(1.0 - 0.999 ** t1)
    dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return SimpleNamespace(
        tabs=AdamWTabs(
            wd_mul = dev(1.0 - lr * cfg.weight_decay),
            lr_t   = dev(lr * bc2 / (1.0 - 0.9 ** t1)),
            eps_t  = dev(cfg.adam_eps * bc2),
        ),
        lr_host=lr, num_steps=N)


sched = build_schedules(num_rounds)
# Shape (1,), NOT 0-D: `tab[t]` with a 0-D tensor index is a data-dependent
# `select` that dynamo cannot specialize ("could not extract specialized
# integer from u0"), while a 1-element index tensor is plain advanced indexing
# and broadcasts against the parameter just the same.
t_step = torch.zeros(1, dtype=torch.int64, device=device)   # advanced on-device
inv_scale = torch.tensor(1.0 / cfg.loss_scale, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------------
# @ Optimizer
# --------------------------------------------------------------------------------
# ONE compiled graph, one kernel per trained tensor, no chunking: inductor
# fuses the whole update so the only traffic is the state itself (master 4 B +
# two fp32 moments 8 B + grad 2 B read, master/moments/live/grad written -- about
# 15 GB a round on 494M params, i.e. bandwidth-bound at ~55 ms on a T4). The
# eager chunked version this replaces ran ~9 kernels per 16M-element chunk over
# ~40 chunks, plus separate passes for the grad norm and the zeroing.

opt_t = 0   # steps taken (host mirror of t_step, for logging)


@torch.compile(dynamic=False, fullgraph=True)
def _adamw_all(c: AdamWTabs, t: Tensor, scale: Tensor) -> None:
    """AdamW over every trained tensor: fp32 master of record, fp16 live copy
    re-derived by round-to-nearest, and the gradient zeroed while it is still
    in registers. One compiled graph, one kernel per tensor. The moments are
    raw EMAs -- bias correction lives in the lr_t / eps_t schedules."""
    for p in m:
        g = p.gacc.float() * scale
        p.master.mul_(c.wd_mul[t])
        p.exp_avg.lerp_(g, 0.1)                  # 1 - beta1  (beta1 = 0.9)
        p.exp_avg_sq.lerp_(g.square(), 0.001)    # 1 - beta2  (beta2 = 0.999)
        p.master.sub_(c.lr_t[t] * (p.exp_avg / (p.exp_avg_sq.sqrt() + c.eps_t[t])))
        p.copy_(p.master)        # fp32 -> fp16, round-to-nearest
        p.gacc.zero_()           # fused: the grad is already resident here


@torch.no_grad()
def optimizer_step() -> None:
    """One step, then advance t on device. Leaves the gradients zeroed."""
    global opt_t
    _adamw_all(sched.tabs, t_step, inv_scale)
    t_step.add_(1)
    opt_t += 1


@torch.compile(dynamic=False, fullgraph=True)
def _grad_sq_sum() -> Tensor:
    """sum g^2 over every trained tensor, in fp32. Compiled so the fp32 cast
    fuses into the reduction -- eager it materialized a full fp32 copy of each
    grad buffer and cost 41 ms a round instead of 5."""
    tot = torch.zeros((), dtype=torch.float32, device=device)
    for p in m:
        tot += p.gacc.float().square().sum()
    return tot


@torch.no_grad()
def grad_global_norm() -> float:
    """nan/inf if any grad overflowed (fp16 inf squares to inf; nan
    propagates). The one host sync of the training step."""
    return float(_grad_sq_sum())


@torch.no_grad()
def zero_grads() -> None:
    """Only the skipped-step path needs this -- a taken step zeroes as it
    goes -- plus warmup, which must not leave the dummy pack's grads behind."""
    for p in m:
        p.gacc.zero_()


# --------------------------------------------------------------------------------
# @ Training Forward/Backward
# --------------------------------------------------------------------------------
# One micro-batch = one pack: a 1-D stream of (prompt+completion) docs with
# per-doc attention isolation via cu_seqlens. No autograd: forward stashes,
# backward accumulates into .gacc. The RL loss is a per-token WEIGHTED CE
# (weight = advantage / round response tokens x loss scale, zero on
# prompt/pad targets), so the only change from a pretraining CE backward is
# that the compile-time constant loss_scale/T becomes the per-token vector w.
#
# fp16 discipline: GEMM in/out fp16 (fp32 accumulate inside cuBLAS); every
# reduction in fp32 (rms stats, the CE block, norm/bias grads); weight grads
# land straight into the fp16 grad buffers via addmm_ (beta=1 -- cuBLAS adds
# the running sum in fp32 before rounding once), no (V, D) temporaries.
#
# rms_norm handling: stash the UNWEIGHTED norm output x_hat plus 1/rms. In
# output space the backward needs no pre-norm input and is exact for any eps:
#   dw     = sum_T(dy * x_hat)
#   dx_hat = dy * w
#   dx     = r * (dx_hat - x_hat * mean(x_hat * dx_hat))

fp16 = lambda x: x.to(torch.float16)


def _rms_fwd(x):
    """Unweighted rms_norm + 1/rms (fp32), Qwen eps."""
    r = (x.float().square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    return fp16(x.float() * r), r


def _rms_bwd(d_hat, x_hat, r):
    xf, df = x_hat.float(), d_hat.float()
    return fp16(r * (df - xf * (xf * df).mean(dim=-1, keepdim=True)))


class LayerStash(NamedTuple):
    """One layer's forward activations held for backward. Sizes at T=2048
    fp16, totals across 24 layers: ~1.3 GB."""
    xb_hat:     Tensor   # (T, D)        attn-norm output, unweighted
    xb_inv_rms: Tensor   # (T, 1) fp32
    q:          Tensor   # (T, 14, 64)   post-rope (what attention consumed)
    k:          Tensor   # (T, 2, 64)    post-rope
    v:          Tensor   # (T, 2, 64)
    y:          Tensor   # (T, 14, 64)   attn out
    lse:        Tensor   # (n_seg, 14, max_seg) fp32   softmax lse
    xm:         Tensor   # (T, D)        post-attn residual (mlp norm recomputed)
    gu:         Tensor   # (T, 9728)     fused gate|up pre-activation


@torch.no_grad()
def forward_backward(idx, pos, cu, sel, tgt_sel, w_sel, max_seg: int):
    """One pack: forward, stash, explicit backward into .gacc. Returns the
    summed weighted CE (scaled by the loss scale like everything else)."""
    T = idx.size(0)
    Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
    cos = m.cos[pos].unsqueeze(1)   # (T, 1, 32) -- broadcasts over heads
    sin = m.sin[pos].unsqueeze(1)

    # -----------------------------
    #           Forward
    # -----------------------------
    
    # Input embeddings
    x = F.embedding(idx, m.embed) 

    stash = []
    
    # Layers
    for i in range(cfg.n_layers):
        
        xb_hat, xb_r = _rms_fwd(x)
        xbn = xb_hat * m.attn_norm[i]
        
        # ---- Attention Forward ----
        qkv = torch.addmm(m.b_QKV[i], xbn, m.W_QKV[i].mT)
        q = qkv[:, :cfg.d_q].view(T, Hq, Dh)
        k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
        v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh).contiguous()
        
        # RoPE
        q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
        k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        
        # FlashAttention
        y, lse = attn_varlen_fwd(q, k, v, cu, max_seg)
        y = y.contiguous()
        
        xm = torch.addmm(x, y.view(T, -1), m.W_O[i].mT) # 

        xm_hat, xm_r = _rms_fwd(xm)   # Recompute in backward pass
        xmn = xm_hat * m.mlp_norm[i]
        
        # ---- MLP Forward ----
        gu = xmn @ m.W_GU[i].mT                  # (T, 9728)
        g, u = gu[:, :cfg.d_mlp], gu[:, cfg.d_mlp:]
        x = torch.addmm(xm, F.silu(g) * u, m.W_down[i].mT)
        
        # Stash activations
        stash.append(LayerStash(xb_hat=xb_hat, xb_inv_rms=xb_r, q=q, k=k, v=v,
                                y=y, lse=lse, xm=xm, gu=gu))

    xf_hat, xf_r = _rms_fwd(x)
    xfn = xf_hat * m.final_norm

    # -----------------------------
    #     LM head + weighted CE  (chunked over the gathered completion rows)
    # -----------------------------
    xfn_sel = xfn.index_select(0, sel)           # (S, D)
    S = sel.size(0)
    loss = torch.zeros((), dtype=torch.float32, device=idx.device)
    sel_grads = []
    ar = torch.arange(cfg.ce_chunk, device=idx.device)
    for c0 in range(0, S, cfg.ce_chunk):
        xs = xfn_sel[c0:c0 + cfg.ce_chunk]           # (c, D)
        tg = tgt_sel[c0:c0 + cfg.ce_chunk]
        wc = w_sel[c0:c0 + cfg.ce_chunk]
        c = xs.size(0)
        lg = (xs @ m.embed.mT).float()               # (c, V) fp32 -- tied head
        cmax = lg.amax(dim=1, keepdim=True)
        lg.sub_(cmax)
        ly = lg.gather(1, tg.unsqueeze(1)).squeeze(1)    # logit_y - cmax
        lg.exp_()                                        # e
        ssum = lg.sum(dim=1, keepdim=True)
        loss += (wc * (ssum.log().squeeze(1) - ly)).sum()   # lse - logit_y
        lg.div_(ssum)                                    # p
        lg[ar[:c], tg] -= 1.0                            # p - onehot
        lg.mul_(wc.unsqueeze(1))                         # weighted
        lgh = fp16(lg)                                   # (c, V) fp16
        del lg
        m.embed.gacc.addmm_(lgh.mT, xs)                  # (V, c) @ (c, D)
        sel_grads.append(lgh @ m.embed)                  # (c, D)
    xfn_grad = torch.zeros_like(xfn)
    xfn_grad.index_add_(0, sel, torch.cat(sel_grads))   # w=0 pads land as zeros at row 0
    del sel_grads, xfn_sel

    # -----------------------------
    #           Backward
    # -----------------------------
    m.final_norm.gacc.add_((xfn_grad.float() * xf_hat.float()).sum(dim=0))
    stream_grad = _rms_bwd(xfn_grad * m.final_norm, xf_hat, xf_r)
    del xfn_grad, xfn, xf_hat

    # Layers
    for i in reversed(range(cfg.n_layers)):
        st = stash[i]
        
        # --- MLP backward (SwiGLU) ---
        xm_hat, xm_r = _rms_fwd(st.xm)
        xmn = xm_hat * m.mlp_norm[i]
        g, u = st.gu[:, :cfg.d_mlp], st.gu[:, cfg.d_mlp:]
        sg = torch.sigmoid(g)
        silu_g = g * sg
        a = silu_g * u
        m.W_down.grad_slices[i].addmm_(stream_grad.mT, a)
        a_grad = stream_grad @ m.W_down[i]
        del a
        u_grad = a_grad * silu_g
        g_grad = a_grad * u * (sg * (1 + g * (1 - sg)))   # d silu / dg
        del a_grad, sg, silu_g
        gu_grad = torch.cat([g_grad, u_grad], dim=1)
        del g_grad, u_grad
        m.W_GU.grad_slices[i].addmm_(gu_grad.mT, xmn)
        xmn_grad = gu_grad @ m.W_GU[i]
        del gu_grad
        m.mlp_norm.grad_slices[i].add_((xmn_grad.float() * xm_hat.float()).sum(dim=0))
        xm_grad = stream_grad + _rms_bwd(xmn_grad * m.mlp_norm[i], xm_hat, xm_r)
        del xmn_grad, xm_hat, xmn

        # --- Attention backward ---
        xbn = st.xb_hat * m.attn_norm[i]
        m.W_O.grad_slices[i].addmm_(xm_grad.mT, st.y.view(T, -1))
        y_grad = (xm_grad @ m.W_O[i]).view(T, Hq, Dh)
        q_grad, k_grad, v_grad = attn_varlen_bwd(
            y_grad, st.q, st.k, st.v, st.y, st.lse, cu, max_seg)
        del y_grad
        
        # RoPE backward
        q1g, q2g = q_grad[..., :cfg.half], q_grad[..., cfg.half:]
        k1g, k2g = k_grad[..., :cfg.half], k_grad[..., cfg.half:]
        q_grad = torch.cat([q1g * cos + q2g * sin, q2g * cos - q1g * sin], dim=-1)
        k_grad = torch.cat([k1g * cos + k2g * sin, k2g * cos - k1g * sin], dim=-1)
        
        qkv_grad = torch.cat([q_grad.reshape(T, cfg.d_q), k_grad.reshape(T, cfg.d_kv),
                              v_grad.reshape(T, cfg.d_kv)], dim=1)
        del q_grad, k_grad, v_grad
        m.b_QKV.grad_slices[i].add_(qkv_grad.sum(dim=0, dtype=torch.float32))
        m.W_QKV.grad_slices[i].addmm_(qkv_grad.mT, xbn)
        xbn_grad = qkv_grad @ m.W_QKV[i]
        del qkv_grad, xbn
        
        m.attn_norm.grad_slices[i].add_((xbn_grad.float() * st.xb_hat.float()).sum(dim=0))
        stream_grad = xm_grad + _rms_bwd(xbn_grad * m.attn_norm[i], st.xb_hat, st.xb_inv_rms)
        del xbn_grad, xm_grad
        
        stash[i] = None                          # free as we go

    # --- token embedding scatter (the tied table's second gradient path) ---
    m.embed.gacc.add_(torch.ops.aten.embedding_dense_backward(
        stream_grad, idx, cfg.d_vocab, -1, False))
    return loss


# Compiled: measured 1.35x on the pack fwd/bwd against the eager version on a
# T4 (see README). dynamic=True holds ONE graph across the pack's varying token
# count -- the compile is ~4 minutes on Colab's 2 vCPUs and happens once, in
# warmup, which is why the pack's CE row count is pinned rather than trimmed.
fb = torch.compile(forward_backward, dynamic=True, fullgraph=True)


# --------------------------------------------------------------------------------
# @ Generation
# --------------------------------------------------------------------------------
# One static allocation (L, max_seqs, t_row, H_kv, Dh) x2: row r owns slot r.
# Prefix sharing is COMPUTE only -- every context prefills once (packed
# varlen) and its K/V are broadcast into its K sibling rows. Decode state per
# row is (input_id, cache_seqlen): the graph itself carries both between
# windows (each replay writes the sampled token into input_ids and advances
# cache_seqlens), so a steady-state window uploads NOTHING and downloads one
# pinned (bucket, macro_n) token block. Rows move only at bucket drops
# (survivors compacted to the front, KV rows included).
#
# Decode attention: the CUTLASS op, in its padded-KV mode -- the whole
# cache viewed as ONE packed key stream of B rows x t_row slots, cu_seqlens_k
# marking the row starts and seqlen_k the live length per row, so the kernel
# reads only [0, seqlen) of each row. GQA becomes 7 query ROWS per KV head
# (cu_seqlens_q steps of 7): no K/V expansion, no mask. The cache is laid out
# (L, B, t_row, H_kv, Dh) -- heads inner, so a row IS a packed key stream.


def _rms(x: Tensor, w: Tensor) -> Tensor:
    """fp16 rms_norm with fp32 statistics (torch's rms_norm upcasts
    internally; spelled out so the numerics are not version-dependent)."""
    xf = x.float()
    r = (xf.square().mean(dim=-1, keepdim=True) + cfg.rms_eps).rsqrt()
    return fp16(xf * r) * w


def attn_decode(q, k_cache, v_cache, seqlen_k, cu_q, cu_k):
    """q (B, H_q, Dh); k/v_cache (B, t_row, H_kv, Dh); seqlen_k (B,) int32
    live keys per row; cu_q/cu_k static int32 (B+1,) row starts. -> (B, D)."""
    B = q.shape[0]
    qg = q.view(B, cfg.n_kv_heads, cfg.group, cfg.d_head).transpose(1, 2) \
          .reshape(1, B * cfg.group, cfg.n_kv_heads, cfg.d_head)
    out, *_ = _EA_FWD(
        qg, k_cache.view(1, -1, cfg.n_kv_heads, cfg.d_head),
        v_cache.view(1, -1, cfg.n_kv_heads, cfg.d_head), None,
        cu_q, cu_k, cfg.group, cfg.t_row, 0.0, _NO_MASK, False,
        scale=_ATTN_SCALE, seqlen_k=seqlen_k)
    return out.view(B, cfg.group, cfg.n_kv_heads, cfg.d_head).transpose(1, 2) \
              .reshape(B, cfg.d_q)


def sample_eager(logits: Tensor, inv_temp: Tensor) -> Tensor:
    """Gumbel-max draw == exact softmax sampling at temperature 1/inv_temp --
    no top-k, no top-p, no sort, no cumsum: one elementwise pass + one argmax
    in fp32. inv_temp lives in a 0-D CUDA buffer so eval can retune without
    re-capturing (see GREEDY_INV_TEMP). logits (B, V) fp16."""
    lf = logits.float()
    e = torch.empty_like(lf).exponential_()
    return (lf * inv_temp - e.log()).argmax(dim=-1)


def sample_fused(logits: Tensor, inv_temp: Tensor) -> Tensor:
    """The same draw written for inductor: G = -log(-log(U)), U ~ Uniform[0,1)
    (U = 0 gives G = -inf -- that token can never win, which IS the limit),
    so the noise is generated inside the reduction kernel and the fp16 logits
    are read once. Compiled with dynamic=True: one kernel for every bucket."""
    u = torch.rand_like(logits, dtype=torch.float32)
    g = -torch.log(-torch.log(u))
    return (logits.float() * inv_temp + g).argmax(dim=-1)


# Greedy through the SAME captured graphs: at inv_temp 1e4 the Gumbel noise
# (O(1)) is negligible against the scaled logits, so the draw is argmax.
GREEDY_INV_TEMP = 1e4


class Engine:
    """Bucketed CUDA-graph decoder + eager varlen prefill over the live
    weights. One graph per row-count bucket; each replay = one decode step;
    the driver replays cfg.macro_n times per window and reads back one pinned
    block."""

    def __init__(self):
        L, Hkv, Dh = cfg.n_layers, cfg.n_kv_heads, cfg.d_head
        B, TR = cfg.max_seqs, cfg.t_row
        shape = (L, B, TR, Hkv, Dh)
        self.k_cache = torch.zeros(shape, dtype=torch.float16, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.float16, device=device)
        self.buckets = []
        b = cfg.min_bucket
        while b < B:
            self.buckets.append(b)
            b *= 2
        self.buckets.append(B)
        # Static graph buffers
        self.input_ids = torch.zeros(B, dtype=torch.long, device=device)
        self.cache_seqlens = torch.zeros(B, dtype=torch.int32, device=device)
        self.tok_buf = torch.zeros(B, dtype=torch.long, device=device)
        self.token_record = torch.zeros(B, cfg.macro_n, dtype=torch.long, device=device)
        self.tok_host = torch.empty(B, cfg.macro_n, dtype=torch.long, pin_memory=True)
        self.tok_host_np = self.tok_host.numpy()
        self.inv_temp = torch.tensor(1.0 / cfg.temperature, dtype=torch.float32, device=device)
        self.row_idx = torch.arange(B, dtype=torch.long, device=device)
        self.cu_q = torch.arange(0, (B + 1) * cfg.group, cfg.group, dtype=torch.int32, device=device)
        self.cu_k = torch.arange(0, (B + 1) * TR, TR, dtype=torch.int32, device=device)
        self.sample = sample_eager     # only if the compile below fails
        try:
            t0 = time.perf_counter()
            fn = torch.compile(sample_fused, dynamic=True)
            probe = torch.randn(B, cfg.d_vocab, dtype=torch.float16, device=device)
            for b in (B, cfg.min_bucket):
                tok = fn(probe[:b], self.inv_temp)
                assert tok.shape == (b,) and int(tok.max()) < cfg.d_vocab
            torch.cuda.synchronize()
            self.sample = fn
            print(f"  sampler compiled in {time.perf_counter() - t0:.0f}s", flush=True)
            del probe
        except Exception as e:
            print(f"  !! sampler compile failed ({type(e).__name__}: {str(e)[:120]}) "
                  f"-- eager sampler", flush=True)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mempool = None
        self.TERM = np.array(TERMINALS, dtype=np.int64)

    # ---- decode step (eager body, captured per bucket) ----------------------
    def _decode_step(self, b: int) -> None:
        ids = self.input_ids[:b]
        csl = self.cache_seqlens[:b]
        posn = csl.long()
        rows = self.row_idx[:b]
        x = F.embedding(ids, m.embed)                       # (b, D)
        cos = m.cos[posn].unsqueeze(1)                      # (b, 1, 32)
        sin = m.sin[posn].unsqueeze(1)
        seqlen_k = csl + 1
        cu_q, cu_k = self.cu_q[:b + 1], self.cu_k[:b + 1]
        for i in range(cfg.n_layers):
            xn = _rms(x, m.attn_norm[i])
            qkv = torch.addmm(m.b_QKV[i], xn, m.W_QKV[i].mT)
            q = qkv[:, :cfg.d_q].view(b, cfg.n_qo_heads, cfg.d_head)
            k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(b, cfg.n_kv_heads, cfg.d_head)
            v = qkv[:, cfg.d_q + cfg.d_kv:].view(b, cfg.n_kv_heads, cfg.d_head)
            q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
            k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
            kc, vc = self.k_cache[i, :b], self.v_cache[i, :b]
            kc[rows, posn] = k
            vc[rows, posn] = v
            y = attn_decode(q, kc, vc, seqlen_k, cu_q, cu_k)
            x = torch.addmm(x, y, m.W_O[i].mT)
            xn2 = _rms(x, m.mlp_norm[i])
            gu = xn2 @ m.W_GU[i].mT
            x = torch.addmm(x, F.silu(gu[:, :cfg.d_mlp]) * gu[:, cfg.d_mlp:], m.W_down[i].mT)
        x = _rms(x, m.final_norm)
        logits = x @ m.embed.mT                              # (b, V) fp16
        tok = self.sample(logits, self.inv_temp)
        self.tok_buf[:b] = tok
        self.input_ids[:b] = tok
        self.cache_seqlens[:b] += 1

    # ---- prefill (eager, packed varlen) --------------------------------------
    @torch.no_grad()
    def prefill(self, ids: Tensor, pos: Tensor, cu: Tensor, max_seg: int):
        """ids (T,) int32 | pos (T,) int64 | cu (n+1,) int32. Returns per-layer
        post-rope K and V, each (L, T, H_kv, Dh) fp16 -- exactly what the
        attention consumed. No lm_head: the forced-last-token split means
        prefill's only product is KV."""
        T = ids.shape[0]
        Hq, Hkv, Dh = cfg.n_qo_heads, cfg.n_kv_heads, cfg.d_head
        cos = m.cos[pos].unsqueeze(1)
        sin = m.sin[pos].unsqueeze(1)
        x = F.embedding(ids, m.embed)
        ks = torch.empty(cfg.n_layers, T, Hkv, Dh, dtype=torch.float16, device=device)
        vs = torch.empty_like(ks)
        for i in range(cfg.n_layers):
            xn = _rms(x, m.attn_norm[i])
            qkv = torch.addmm(m.b_QKV[i], xn, m.W_QKV[i].mT)
            q = qkv[:, :cfg.d_q].view(T, Hq, Dh)
            k = qkv[:, cfg.d_q:cfg.d_q + cfg.d_kv].view(T, Hkv, Dh)
            v = qkv[:, cfg.d_q + cfg.d_kv:].view(T, Hkv, Dh)
            q1, q2 = q[..., :cfg.half], q[..., cfg.half:]
            k1, k2 = k[..., :cfg.half], k[..., cfg.half:]
            q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
            k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
            ks[i] = k
            vs[i] = v
            y, _ = attn_varlen_fwd(q, k, v, cu, max_seg)
            x = torch.addmm(x, y.reshape(T, -1), m.W_O[i].mT)
            xn2 = _rms(x, m.mlp_norm[i])
            gu = xn2 @ m.W_GU[i].mT
            x = torch.addmm(x, F.silu(gu[:, :cfg.d_mlp]) * gu[:, cfg.d_mlp:], m.W_down[i].mT)
        return ks, vs

    def _write_kv(self, r0: int, k: int, plen: int, k_ctx: Tensor, v_ctx: Tensor) -> None:
        """Broadcast one context's KV (L, plen, H_kv, Dh) into rows [r0, r0+k)."""
        self.k_cache[:, r0:r0 + k, :plen] = k_ctx.unsqueeze(1)
        self.v_cache[:, r0:r0 + k, :plen] = v_ctx.unsqueeze(1)

    def _compact_kv(self, lp: np.ndarray) -> None:
        """Move survivors' KV rows to the front (layer by layer: bounded temp).
        Only ever called at a bucket drop, a few times a round."""
        idxs = torch.from_numpy(lp).to(device, non_blocking=True)
        n = lp.size
        for i in range(cfg.n_layers):
            for cache in (self.k_cache, self.v_cache):
                cache[i, :n].copy_(cache[i].index_select(0, idxs))

    @torch.no_grad()
    def capture(self) -> None:
        print(f"  engine: {cfg.max_seqs} rows x {cfg.t_row} tok "
              f"({(self.k_cache.numel() + self.v_cache.numel()) * 2 / 2**30:.2f} GB KV) | "
              f"buckets {tuple(self.buckets)} | macro_n {cfg.macro_n} | "
              f"temp {cfg.temperature:g} (gumbel-argmax)", flush=True)
        print("  capture decode buckets:", flush=True)
        for b in sorted(self.buckets, reverse=True):
            t0 = time.perf_counter()
            self.input_ids[:] = 0
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):                # warmup: cuBLAS handles, workspaces
                    self.cache_seqlens[:] = 0
                    self._decode_step(b)
            torch.cuda.current_stream().wait_stream(s)
            self.cache_seqlens[:] = 0
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, **({"pool": self._mempool} if self._mempool else {})):
                self._decode_step(b)
            self._mempool = self._mempool or g.pool()
            self.graphs[b] = g
            torch.cuda.synchronize()
            print(f"    bucket {b:3d}: {time.perf_counter() - t0:5.1f}s", flush=True)
        self.cache_seqlens[:] = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


    @torch.no_grad()
    def run_round(self, specs: list[tuple], tail_windows: int = 0) -> list[dict]:
        """specs: (meta, prompt_ids, k, allow). Context = prompt[:-1]; forced
        first decode input = prompt[-1] -- so the first SAMPLED token already
        comes out of the decode graph. Prefill every context in ONE eager
        call, broadcast KV to the K sibling rows, decode all rows to
        completion. Returns the rows, carrying the sampled ids INCLUDING the
        terminal (the pack trains on it; text decode strips it), and writes the
        generation slice of the global `stats`. tail_windows > 0: once 90% of
        the rows have retired, the rest get that many more windows and are
        then retired as truncated (rows carry tail=True)."""
        t0 = time.perf_counter()
        n_rows = sum(k for _, _, k, _ in specs)
        assert n_rows <= cfg.max_seqs

        # -- prefill pack (host-side assembly, one H2D per buffer) -------------
        ids, pos, cu = [], [], [0]
        for _, p, _, _ in specs:
            ctx = p[:-1]
            ids.extend(ctx)
            pos.extend(range(len(ctx)))
            cu.append(cu[-1] + len(ctx))
        max_seg = max(len(p) - 1 for _, p, _, _ in specs)
        k_all, v_all = self.prefill(
            torch.tensor(ids, dtype=torch.int32, device=device),
            torch.tensor(pos, dtype=torch.int64, device=device),
            torch.tensor(cu, dtype=torch.int32, device=device), max_seg)

        # -- broadcast each context's KV into its K sibling rows ---------------
        r0, o = 0, 0
        metas, plens_l, forced, allows_l = [], [], [], []
        for meta, p, k, allow in specs:
            plen = len(p) - 1
            self._write_kv(r0, k, plen, k_all[:, o:o + plen], v_all[:, o:o + plen])
            metas.extend([meta] * k)
            plens_l.extend([plen] * k)
            forced.extend([p[-1]] * k)
            allows_l.extend([allow] * k)
            r0 += k
            o += plen
        del k_all, v_all

        # -- seed the graph state (the round's one full upload) ----------------
        B0 = n_rows
        plens = np.asarray(plens_l, dtype=np.int64)
        allows = np.asarray(allows_l, dtype=np.int64)
        bucket = next(x for x in self.buckets if x >= B0)
        self.input_ids[:B0] = torch.tensor(forced, dtype=torch.long, device=device)
        self.cache_seqlens[:B0] = torch.tensor(plens, dtype=torch.int32, device=device)
        if bucket > B0:                          # park the padded tail
            self.cache_seqlens[B0:bucket] = 0

        # -- decode windows ----------------------------------------------------
        orig = np.arange(B0)
        live = np.ones(B0, dtype=bool)
        gen_buf = np.empty((B0, (-(-int(allows.max()) // cfg.macro_n)) * cfg.macro_n), dtype=np.int64)
        # A row past its capacity would write K into a position that does not
        # exist -- an out-of-bounds scatter, not a crash.
        assert int(plens.max()) + gen_buf.shape[1] <= cfg.t_row, \
            f"row overflow: ctx {int(plens.max())} + {gen_buf.shape[1]} steps > t_row {cfg.t_row}"
        rows: list[dict] = [None] * B0
        rolls_done = tok_total = paid_slots = n_tail = 0
        t50 = t90 = None
        n_half, n_ninety = (B0 + 1) // 2, (B0 * 9 + 9) // 10
        w_ninety = None                          # window index at which 90% had retired
        park_dirty = False
        w = 0
        while True:
            lp = np.flatnonzero(live)
            if lp.size == 0:
                break
            nb = next(x for x in self.buckets if x >= lp.size)
            if nb < bucket:                      # bucket drop: compact survivors
                idxs = torch.from_numpy(lp).to(device, non_blocking=True)
                for buf in (self.input_ids, self.cache_seqlens):
                    buf[:lp.size].copy_(buf.index_select(0, idxs))
                self._compact_kv(lp)
                orig, plens, allows = orig[lp], plens[lp], allows[lp]
                live = np.ones(lp.size, dtype=bool)
                bucket = nb
                self.cache_seqlens[lp.size:bucket] = 0
                park_dirty = False
                lp = np.arange(lp.size)
            elif park_dirty:                     # park mid-bucket retirees in place
                dead = np.flatnonzero(~live)
                di = torch.from_numpy(dead).to(device, non_blocking=True)
                self.cache_seqlens.index_fill_(0, di, 0)
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
                                eos=bool(eos[ri]), tail=False)
                live[p_] = False
                rolls_done += 1
                tok_total += int(n_take[ri])
            tok_total += cfg.macro_n * int((~done).sum())
            if done.any():
                park_dirty = True
            if w_ninety is None and rolls_done >= n_ninety:
                w_ninety = w
            if tail_windows and w_ninety is not None and w - w_ninety >= tail_windows:
                # Straggler cut: retire every live row with what it has so far.
                for p_ in np.flatnonzero(live):
                    o_ = orig[p_]
                    rows[o_] = dict(meta=metas[o_], ids=gen_buf[o_, :base + cfg.macro_n].tolist(),
                                    eos=False, tail=True)
                    live[p_] = False
                    rolls_done += 1
                    n_tail += 1
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
        stats.n_tail = n_tail
        return rows


# --------------------------------------------------------------------------------
# @ Trainer
# --------------------------------------------------------------------------------
# One round, end to end: groups -> advantages -> packs -> fwd/bwd -> step.

loss_scale = cfg.loss_scale


def train_step(groups: list[dict]) -> None:
    """One optimizer step over the round's problem groups -- GRPO/token-mean,
    into the training slice of the global `stats`. The normalizer is the
    round's TOTAL response-token count (all rollouts, saturated groups
    included -- their tokens dilute the mean exactly as a masked mean would);
    it folds into the per-token weight together with the loss scale, so the
    packed forward just sums. A non-finite gradient (fp16 overflow somewhere
    in the backward) skips the step and halves the scale."""
    global loss_scale
    t0 = time.perf_counter()
    n_total_tok = sum(len(c) for g in groups for c in g["completions"])
    docs = []
    n_used = n_sat = n_dead = 0
    for g in groups:
        adv = group_advantage(g["rewards"])
        if adv is None:                          # zero-signal group: zero gradient
            n_sat += int(np.mean(g["rewards"]) >= cfg.w_correct)
            n_dead += int(np.mean(g["rewards"]) <= 0.0)
            continue
        n_used += 1
        for comp, a in zip(g["completions"], adv):
            docs.append((g["prompt_ids"], comp,
                         float(a) / max(1, n_total_tok) * loss_scale))
    loss_total = 0.0
    n_loss_tok = 0
    pstats = None
    if docs:
        packs, pstats = plan_packs(docs)
        for pk in packs:
            args = [torch.from_numpy(pk[k]).to(device, non_blocking=True)
                    for k in ("idx", "pos", "cu", "sel", "tgt", "w")]
            loss = fb(*args, pk["max_seg"])
            loss_total += float(loss)
            n_loss_tok += pk["n_sel"]
    gsq = grad_global_norm()
    if math.isfinite(gsq):
        optimizer_step()                     # zeroes the grads as it steps
        stats.step_ok = 1
    else:
        stats.step_ok = 0
        print(f"  !! non-finite gradient at loss scale {loss_scale:g} -- step skipped, "
              f"scale -> {loss_scale / 2:g}", flush=True)
        loss_scale /= 2.0
        inv_scale.fill_(1.0 / loss_scale)    # device-side: no recompile
        zero_grads()
    torch.cuda.synchronize()
    stats.train_s = round(time.perf_counter() - t0, 2)
    stats.n_groups_used, stats.n_groups_sat, stats.n_groups_dead = n_used, n_sat, n_dead
    stats.n_docs = len(docs)
    stats.n_loss_tokens = n_loss_tok
    stats.n_packs = pstats["n_packs"] if docs else 0
    stats.pad_pct = round(100.0 * pstats["pad_tokens"] / pstats["cap_tokens"], 1) if docs else 0.0
    stats.loss_total = round(loss_total / (stats.loss_scale or 1.0), 6)
    stats.grad_norm = round(math.sqrt(gsq) / (stats.loss_scale or 1.0), 6) if math.isfinite(gsq) else float("nan")


# --------------------------------------------------------------------------------
# @ Eval
# --------------------------------------------------------------------------------
# Greedy (K=1) accuracy through the same graphs: val in-loop, tests at the end.

def make_eval_waves(prompts: list[list[int]], k: int) -> list[list[int]]:
    """Greedy wave assembly under the engine's static row cap."""
    waves, cur = [], []
    for i in range(len(prompts)):
        if cur and (len(cur) + 1) * k > cfg.max_seqs:
            waves.append(cur)
            cur = []
        cur.append(i)
    if cur:
        waves.append(cur)
    return waves


def run_eval(prompts: list[list[int]], golds: list[int], label: str) -> dict:
    """Greedy accuracy over one split. Sampler RNG is saved and restored so
    the training rollout stream is identical to an eval-off run; inv_temp is
    flipped to GREEDY_INV_TEMP for the duration (same graphs, no capture)."""
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
    out["per_problem"] = [(i, ok[i]) for i in range(n_prob)]
    return out


# --------------------------------------------------------------------------------
# @ Warmup
# --------------------------------------------------------------------------------
# Capture the decode graphs, then warm the training step.

engine = Engine()
build_s = time.perf_counter() - run_wall_t0
t = time.perf_counter()
engine.capture()

# One dummy pack through fb (weights untouched; w=0 so every grad lands as an
# exact zero -- zeroed again after anyway): pays the cuBLAS/attention
# first-call costs before the first timed round.
_t = time.perf_counter()
_dummy_docs = [([3 + (j % 97) for j in range(50)],
                [5 + (j % 89) for j in range(110)] + [IM_END], 0.0)
               for _ in range(cfg.train_t // 160 + 1)]
_packs, _ = plan_packs(_dummy_docs)
_pk = _packs[0]
fb(*[torch.from_numpy(_pk[k]).to(device) for k in ("idx", "pos", "cu", "sel", "tgt", "w")],
   _pk["max_seg"])
zero_grads()
del _dummy_docs, _packs, _pk
torch.cuda.synchronize()
print(f"    train fwd+bwd compile: {time.perf_counter() - _t:5.1f}s", flush=True)

# The optimizer and grad-norm graphs compile on their FIRST CALL, which would
# otherwise land inside round 0's train_step (measured: +22 s on that round).
# Warm them here against the zeroed gradients left above. With g = 0 the step
# is a no-op on the moments (lerp toward 0 from 0) and applies only decoupled
# weight decay to the fp32 masters -- a 1-1e-8 relative nudge, far under fp16's
# resolution, so the live weights round to the same bits and round 0 still
# reproduces the reference exactly. t is rewound afterwards.
_t = time.perf_counter()
grad_global_norm()
optimizer_step()
t_step.zero_()
opt_t = 0
zero_grads()
torch.cuda.synchronize()
print(f"    optimizer compile: {time.perf_counter() - _t:5.1f}s", flush=True)
warm_s = time.perf_counter() - t
print(f"  build {build_s:.0f}s + capture/warmup {warm_s:.0f}s | "
      f"peak mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)


# --------------------------------------------------------------------------------
# @ Main Loop
# --------------------------------------------------------------------------------

cfg.run_dir.mkdir(parents=True, exist_ok=True)


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
                               config=config_dict() | dict(num_rounds=num_rounds, gpu=_gpu_name))
    except Exception as e:
        print(f"[{cfg.tag}] wandb unavailable ({e}) -- CSV/JSON only", flush=True)
        use_wandb = False
elif cfg.wandb:
    print(f"[{cfg.tag}] no wandb credentials -- CSV/JSON only", flush=True)


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
        # `test/` holds all three held-out splits. The redundant `test_` comes
        # off the label -- the group already says it -- so the panel reads
        # val / id / ood.
        s = res["label"].removeprefix("test_")
        # step=rnd, NOT wandb's implicit counter: an eval is an extra .log()
        # inside a round, so letting wandb autoincrement pushes every later
        # round's step one further ahead of its round number. Pinning the step
        # also merges the eval into the round it belongs to.
        wandb_run.log({"round": rnd, f"test/{s}_accuracy": res["accuracy"],
                       f"test/{s}_method_pct": res["method_pct"],
                       f"test/{s}_trunc_pct": res["trunc_pct"],
                       f"test/{s}_mean_len": res["mean_len"]}, step=rnd)


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
tokens_seen = 0                # running total behind stats.loss_tokens
try:
    for rnd in range(num_rounds):
        if cfg.eval_every and rnd % cfg.eval_every == 0:
            log_eval(rnd, run_eval(val_prompts, val_gold, "val"))
        r_t0 = time.perf_counter()
        stats = RoundStats(round=rnd)
        stats.loss_scale = loss_scale

        # -- generation ------------------------------------------------------
        idxs = cfg.fixed_problems if cfg.fixed_problems is not None else round_schedule[rnd]
        specs = [(i, train_prompts[i], cfg.k_draws, cfg.max_tokens) for i in idxs]
        rows = engine.run_round(specs, tail_windows=cfg.tail_windows)

        # -- grade + group ---------------------------------------------------
        by_pid: dict[int, dict] = {}
        n_roll, n_method, len_sum = len(rows), 0, 0
        for r in rows:
            pid = r["meta"]
            text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
            rw, r_c, r_m = grade(text, train_gold[pid])
            stats.n_correct += r_c == 1.0
            stats.n_eos += r["eos"]
            n_method += r_m == 1.0
            len_sum += len(r["ids"])
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
        stats.mean_len = round(len_sum / max(1, n_roll), 1)
        # opt_t has already advanced past the step this round took.
        stats.lr = float(sched.lr_host[min(max(0, opt_t - 1), sched.num_steps - 1)])
        stats.round_s = round(time.perf_counter() - r_t0, 2)
        el = time.perf_counter() - loop_t0
        stats.elapsed_s = round(el, 2)
        tokens_seen += stats.n_loss_tokens
        stats.loss_tokens = tokens_seen
        row = asdict(stats)
        curve.append(row)
        mw.writerow(row)
        mf.flush()
        eta = el / (rnd + 1) * (num_rounds - rnd - 1)
        print(f"  [{rnd:3d}/{num_rounds}] ({100 * (rnd + 1) / num_rounds:5.1f}%) "
              f"{stats.round_s:5.2f}s ({stats.gen_s:.2f} gen / {stats.train_s:.2f} trn) | "
              f"solve {100 * stats.solve_rate:5.1f}% | len {stats.mean_len:5.1f} | "
              f"trunc {stats.n_trunc:3d} (tail {stats.n_tail:2d}) | dead {stats.n_groups_dead:2d} sat {stats.n_groups_sat:2d} /{len(idxs)} | "
              f"pk {stats.n_packs:2d} ({stats.n_docs:3d}d {stats.n_loss_tokens / 1000:4.1f}k tok, pad {stats.pad_pct:4.1f}%) | "
              f"gn {stats.grad_norm:6.3f} | total {el / 60:5.1f}m | eta {eta / 60:4.1f}m",
              flush=True)
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
        log_eval(num_rounds, run_eval(val_prompts, val_gold, "val"))
    if cfg.final_eval:
        res_id = run_eval(test_id_prompts, test_id_gold, "test_id")
        log_eval(num_rounds, res_id)
        res_ood = run_eval(test_ood_prompts, test_ood_gold, "test_ood")
        log_eval(num_rounds, res_ood)
        print(f"\n  == FINAL vs H100 raw reference: test ID {res_id['accuracy']} "
              f"(ref 89.0) | test OOD {res_ood['accuracy']} (ref 86.75) | "
              f"ref val best 94.0 @ 272 rounds ==", flush=True)
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
    if curve and cfg.save_final:
        save_ckpt(len(curve))
    result = dict(
        tag=cfg.tag, model=cfg.model_id, gpu=_gpu_name, k=cfg.k_draws,
        problems_per_round=cfg.problems_per_round, rounds_run=len(curve),
        budget=cfg.max_tokens, lr=cfg.lr, temperature=cfg.temperature, seed=cfg.seed,
        loss_scale_final=loss_scale, steps_skipped=sum(1 - c["step_ok"] for c in curve),
        error=run_error,
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
        reference=dict(script="baselines/20260813_raw-h100 (H100 PCIe, raw-v1-lr1e-6)",
                       val="61.5 -> 80.5 @30 / 84.0 @60 / 87.0 @90 / 90.5 @150 / 94.0 @270",
                       test_id=89.0, test_ood=86.75, wall="3.7 min loop / 272 rounds"))
    (Path.cwd() / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    (cfg.run_dir / f"result_{cfg.tag}.json").write_text(json.dumps(result, indent=1))
    print(f"\n== train_qwen_arithmetic [{cfg.tag}] on {_gpu_name} ==", flush=True)
    if curve:
        print(f"  rounds {len(curve)} | solve {result['solve_rate_first']} -> "
              f"{result['solve_rate_last']} | round_s med {result['round_s_med']} "
              f"(gen {result['gen_s_med']} + train {result['train_s_med']}) | "
              f"loop {result['loop_s'] / 60:.1f} min | total {total_s / 60:.1f} min | "
              f"peak {result['peak_mem_gb']} GB | skipped steps {result['steps_skipped']}",
              flush=True)
    print(f"  results -> result_{cfg.tag}.json / metrics_{cfg.tag}.csv / "
          f"evals_{cfg.tag}.csv / evals_detail_{cfg.tag}.csv", flush=True)
    if use_wandb:
        wandb_run.summary.update({k: v for k, v in result.items()
                                  if not isinstance(v, (list, dict))})
        wandb_run.finish()
