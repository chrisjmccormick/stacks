# train_gsm8k.py
#
# GSM8K post-training for the DecoderStack d12 that train_stack.py pretrains:
# supervised fine-tuning on Qwen3-8B teacher traces, then on-policy RL
# (REINFORCE with a per-problem mean baseline) -- one process, one set of live
# weights, the same handwritten forward/backward as pretraining (no torch.nn,
# no autograd), plus a CUDA-graph decode engine over those weights.
#
# Why this file exists: the pretraining speedrun is the contest. This is its
# sanity check for "is the architecture still cheap to decode" -- a change to
# train_stack.py's forward has to be carried into decode_body / prefill_body
# here without wrecking generation throughput. Secondary objective: time to a
# GSM8K score.
#
# Style, config, logging follow train_stack.py: globals `cfg` and `m`,
# optimizer state and per-step schedule tables attached to each Param, no
# command line, the source archived to wandb and into the checkpoints.
#
# Data is downloaded, never built here. ChrisMcCormick/decoderstack-gsm8k
# carries the prompts (nanochat chat template, ClimbMix 32k ids) and the
# teacher traces; ChrisMcCormick/decoderstack-d12 the pretrained weights.
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
    code = f.read()   # logged to wandb and into every checkpoint

import gc
import json
import math
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from typing import NamedTuple

import numpy as np
import wandb

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch import Tensor

from utils import fa2, flash_attn_varlen_fwd_lse, flash_attn_varlen_bwd

dynamo.config.recompile_limit = 64

assert torch.cuda.is_available(), "no GPU"
props = torch.cuda.get_device_properties(0)
print(f"{props.name} | {props.total_memory / 2**30:.1f} GiB | sm{props.major}{props.minor}")
assert props.major >= 8, f"needs Ampere or newer (got sm{props.major}{props.minor})"

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# ==============================================================================
# § Configuration
# ==============================================================================

class GSM8KConfig:

    # ---- Architecture (train_stack.py's StackConfig; asserted against the checkpoint) ----

    n_layers:   int = 12
    d_model:    int = 768
    backout_layer: int = 6

    d_vocab:    int = 32768
    d_smr_gate: int = 24

    n_qo_heads: int = 6
    n_kv_heads: int = 6
    d_qk:       int = 128
    d_vo:       int = 128

    seq_len:          int = 2048
    short_win_size:   int = 768
    full_ctxt_layers: list[int] = [   3,    7,    11]
    window_sizes:     list[tuple[int, int]]

    d_ve_gate: int = 12
    ve_layers: list[int] = [1, 3, 5, 7, 9, 11]
    ve_index:  list[int]
    num_ves:   int

    d_mlp:      int = 4 * 768

    num_params:          int = 286_261_730
    num_flops_per_token: int = 780_929_568

    # ---- Inputs ----

    model_repo: str = "ChrisMcCormick/decoderstack-d12"
    model_file: str = "checkpoints/rtx-newbase-nr-1000-r3/model_step001000.pt"
    init_ckpt:  str = ""      # a local .pt (e.g. logs/<run>/model_sft.pt) instead of the hub file
    data_repo:  str = "ChrisMcCormick/decoderstack-gsm8k"

    # ---- Phases ----

    run_sft:      bool = True
    run_rl:       bool = True
    engine_check: bool = True   # teacher-forced decode vs the training forward, before capture

    # ---- SFT ----

    sft_epochs:         float = 2.0
    sft_packs_per_step: int   = 8      # x train_t = 256K tokens per optimizer step
    sft_lr_frac:        float = 0.8    # nanochat chat_sft: 0.8 x the pretraining peaks ...
    sft_warmdown_frac:  float = 0.5    # ... linear to 0 over the last half
    sft_steps_cap:      int   = 0      # 0 = the full epochs horizon

    # ---- RL (nanochat chat_rl + the fmt05 recipe: token-mean loss, 0.02 x peaks, partial format credit) ----

    k_draws:            int   = 16     # rollouts per problem per round
    problems_per_round: int   = 32
    rl_epochs:          int   = 1
    rounds_cap:         int   = 0      # 0 = the full epochs horizon
    max_tokens:         int   = 512    # generation budget, rollouts and eval (teacher traces: p99 469)
    temperature:        float = 1.0
    top_k:              int   = 50     # 0 = full-vocab sampling
    rl_lr_frac:         float = 0.02   # x the pretraining peaks, linear to 0
    fmt_reward:         float = 0.5    # a wrong but `#### n`-formatted answer
    seed:               int   = 1337   # round schedule, SFT shuffle, sampler RNG

    # ---- Training packs (packed varlen, one compiled shape) ----

    train_t:  int = 32768              # tokens per pack
    max_docs: int = 192                # cu_seqlens fixed size (ghost-padded)
    rope_t:   int = 2048               # rotary cache; every doc and decode row must fit

    # ---- Engine ----

    max_seqs:  int   = 512             # decode rows (= top bucket)
    macro_n:   int   = 8               # decode steps per window (one D2H each)
    buckets:   tuple = (64, 128, 256, 384, 512)
    prefill_t: int   = 8192            # prefill pack (one compiled shape; covers train rounds and eval waves)
    max_ctxs:  int   = 96              # prefill cu fixed size
    page:      int   = 256             # FA2 paged KV page size (multiple of 256 required)

    max_prompt:    int   # Derived in § Data: the longest rendered prompt.
    gen_steps:     int   # Derived: max decode steps a row can run (macro-aligned budget).
    t_row:         int   # Derived: decode row capacity, whole pages.
    pages_per_row: int

    # ---- Eval (val subset through the graphs; full test after SFT and at the end) ----

    sft_eval_every: int  = 16          # SFT steps (0 = off)
    rl_eval_every:  int  = 30          # RL rounds (0 = off)
    eval_k:         int  = 8
    full_eval:      bool = True

    # ---- Logging ----

    wandb_project:   str  = "decoderstack_rtx_gsm8k"   # test runs -> decoderstack_rtx_gsm8k_dev
    run_name:        str  = "gsm8k"
    use_wandb:       bool = True
    save_checkpoint: bool = True       # logs/<run_name>/model_{sft,final}.pt

cfg = GSM8KConfig() # Make config a global, don't pass it around.

# ==============================================================================
# § Derived Configs
# ==============================================================================

if cfg.use_wandb:
    assert "WANDB_API_KEY" in os.environ, "cfg.use_wandb=True but WANDB_API_KEY not set"
    wandb.login(key=os.environ["WANDB_API_KEY"])

cfg.ve_index = [cfg.ve_layers.index(i) if i in cfg.ve_layers else -1 for i in range(cfg.n_layers)]
cfg.num_ves = len(cfg.ve_layers)

cfg.window_sizes = [(cfg.short_win_size, 0)] * cfg.n_layers
for i in cfg.full_ctxt_layers:
    cfg.window_sizes[i] = (cfg.seq_len, 0)

assert cfg.problems_per_round * cfg.k_draws <= cfg.max_seqs, "round rows exceed max_seqs"
assert cfg.max_seqs == max(cfg.buckets) and list(cfg.buckets) == sorted(cfg.buckets)
assert cfg.page % 256 == 0
assert cfg.run_sft or cfg.run_rl, "nothing to run"

# A row's K lands at positions [ctx, ctx + steps): the first decode step writes
# the FORCED token, and a window always replays macro_n times, so a row runs a
# macro-ALIGNED budget before the host can retire it.
cfg.gen_steps = -(-cfg.max_tokens // cfg.macro_n) * cfg.macro_n

gpu_device_name = torch.cuda.get_device_name(0)
gpu_peak_flops = 503.8e12   # dense bf16, RTX PRO 6000 (the MFU denominator)


# ==============================================================================
# § Data
# ==============================================================================
# Prompts arrive rendered and the traces tokenized. Everything tokenizer-shaped
# (chat template, gold normalization, which test problems are the in-loop
# validation subset, which traces are held out) was decided by the dataset
# builder and frozen into the parquets; this section only loads ids.

from huggingface_hub import hf_hub_download

DATA_DIR  = os.path.join("./data", cfg.data_repo.split("/")[-1])
MODEL_DIR = os.path.join("./data", cfg.model_repo.split("/")[-1])
print("=== Downloading dataset + checkpoint ===")
for fname in ("prompts.parquet", "sft.parquet", "tokenizer/tokenizer.pkl", "config.json"):
    if not os.path.exists(os.path.join(DATA_DIR, fname)):
        hf_hub_download(repo_id=cfg.data_repo, filename=fname, repo_type="dataset", local_dir=DATA_DIR)
if cfg.init_ckpt:
    ckpt_path = cfg.init_ckpt
    assert os.path.exists(ckpt_path), f"cfg.init_ckpt {ckpt_path} not found"
else:
    ckpt_path = os.path.join(MODEL_DIR, cfg.model_file)
    if not os.path.exists(ckpt_path):
        hf_hub_download(repo_id=cfg.model_repo, filename=cfg.model_file, repo_type="model", local_dir=MODEL_DIR)
print("  Done.")

with open(os.path.join(DATA_DIR, "config.json")) as f:
    _dcfg = json.load(f)
assert _dcfg["vocab_size"] == cfg.d_vocab, "dataset vocab != model d_vocab"
BOS_ID          = _dcfg["bos_id"]
ASSISTANT_START = _dcfg["assistant_start"]
ASSISTANT_END   = _dcfg["assistant_end"]
# <|assistant_end|> ends the turn; a sampled <|bos|> ends the document. Both
# retire a decode row, and BOS doubles as the pack pad id.
TERMINALS = (ASSISTANT_END, BOS_ID)
PAD_ID = BOS_ID

# Decode only (rewards, telemetry, the engine check): the tiktoken Encoding the
# dataset's ids came from. Nothing here ever encodes text.
with open(os.path.join(DATA_DIR, "tokenizer/tokenizer.pkl"), "rb") as f:
    _enc = pickle.load(f)
assert _enc.n_vocab == cfg.d_vocab
assert _enc.encode_single_token("<|assistant_end|>") == ASSISTANT_END

def decode(ids: list[int]) -> str:
    return _enc.decode(ids)

import pyarrow.parquet as pq

_t = pq.read_table(os.path.join(DATA_DIR, "prompts.parquet")).to_pydict()
_rows = {"train": {}, "test": {}}
for _s, _i, _gold, _ids, _val in zip(_t["split"], _t["idx"], _t["gold"], _t["prompt_ids"], _t["is_val"]):
    _rows[_s][_i] = (_gold, _ids, _val)
_train = [_rows["train"][i] for i in range(len(_rows["train"]))]
_test  = [_rows["test"][i]  for i in range(len(_rows["test"]))]
train_gold, train_prompts = [r[0] for r in _train], [r[1] for r in _train]
test_gold,  test_prompts  = [r[0] for r in _test],  [r[1] for r in _test]
assert all(train_gold) and all(test_gold), "a gold answer is empty"
assert all(p[-1] == ASSISTANT_START for p in train_prompts + test_prompts), "prompts must end in <|assistant_start|>"
VAL_SUBSET = [i for i, r in enumerate(_test) if r[2]]
assert VAL_SUBSET, "prompts.parquet flags no is_val rows"

_t = pq.read_table(os.path.join(DATA_DIR, "sft.parquet"), columns=["ids", "prompt_len", "is_val"]).to_pydict()
sft_train = [(ids, pl) for ids, pl, v in zip(_t["ids"], _t["prompt_len"], _t["is_val"]) if not v]
sft_val   = [(ids, pl) for ids, pl, v in zip(_t["ids"], _t["prompt_len"], _t["is_val"]) if v]
del _t, _rows, _train, _test

_prompt_lens = [len(p) for p in train_prompts + test_prompts]
cfg.max_prompt = max(_prompt_lens)
assert min(_prompt_lens) >= 2, "prompt too short for the forced-last-token split"
cfg.t_row = -(-(cfg.max_prompt - 1 + cfg.gen_steps) // cfg.page) * cfg.page
cfg.pages_per_row = cfg.t_row // cfg.page
assert cfg.t_row <= cfg.rope_t, "decode row exceeds the rotary cache"
_max_doc = max(len(ids) for ids, _ in sft_train + sft_val)
assert _max_doc <= cfg.rope_t and _max_doc - 1 <= cfg.train_t, f"an SFT doc of {_max_doc} tokens does not fit"

print(f"data: {len(train_prompts)} train / {len(test_prompts)} test problems (val subset {len(VAL_SUBSET)}) | "
      f"SFT traces {len(sft_train)} train / {len(sft_val)} val (doc max {_max_doc}) | "
      f"max prompt {cfg.max_prompt} | row {cfg.t_row} tok = {cfg.pages_per_row} pages")


# ==============================================================================
# § Reward
# ==============================================================================
# nanochat's GSM8K reward: the number after the FIRST `#### `, commas stripped,
# compared to the gold digits as strings. The teacher traces end that way, so
# it is also what SFT teaches. A formatted-but-wrong answer earns
# cfg.fmt_reward (the fix-ladder recipe): binary reward scores a wrong `####`
# the same as no `####`, and the format erodes under the negative advantages.

GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(text: str):
    m = GSM_RE.search(text)
    return m.group(1).strip().replace(",", "") if m else None

def reward_of(text: str, gold: str) -> float:
    pred = extract_answer(text)
    return 1.0 if pred == gold else (cfg.fmt_reward if pred is not None else 0.0)


# ==============================================================================
# § Pack Planning (host)
# ==============================================================================
# A training doc is (ids, n_prompt, weight): inputs ids[:-1], targets ids[1:],
# and the per-token loss weight `weight` on every target from ids[n_prompt]
# on (the assistant tokens) -- SFT passes 1/N, RL passes advantage/N. One pack
# is cfg.train_t tokens at one compiled shape; the pad tail is a real attended
# segment with varying ids and positions (qwen-gsm8k/TECHNIQUES.md § Padded
# varlen: miss any of the three rules and every weight grad NaNs while the
# loss stays finite).

def pack_ffd(lengths: list[int]) -> list[list[int]]:
    """First-fit-decreasing under the token and doc caps. Returns the packs as
    lists of doc indices."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    packs, tok = [], []
    for i in order:
        n = lengths[i]
        assert n <= cfg.train_t, f"doc of {n} tokens exceeds train_t"
        for j in range(len(packs)):
            if tok[j] + n <= cfg.train_t and len(packs[j]) < cfg.max_docs:
                packs[j].append(i)
                tok[j] += n
                break
        else:
            packs.append([i])
            tok.append(n)
    return packs

def pack_sequential(lengths: list[int], order: list[int]) -> list[list[int]]:
    """Fill packs in the given (shuffled) order, so every pack is a random draw
    of docs. ~0.5% tail waste at these doc sizes."""
    packs, cur, tok = [], [], 0
    for i in order:
        n = lengths[i]
        assert n <= cfg.train_t
        if cur and (tok + n > cfg.train_t or len(cur) >= cfg.max_docs):
            packs.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += n
    if cur:
        packs.append(cur)
    return packs

def build_pack(docs: list[tuple[list[int], int, float]]) -> dict:
    """The docs of ONE pack -> device tensors at the compiled shape:
      idx (train_t,) int32   packed inputs
      pos (train_t,) int64   rotary positions, restarting at each doc
      cu  (max_docs+2,) int32  doc boundaries, ghost-padded with train_t
      tgt (train_t,) int64   next-token targets (0 on the pad tail)
      w   (train_t,) fp32    per-target loss weight (0 on prompt targets and the pad tail)
    Filled in pinned memory, uploaded non_blocking."""
    idx = torch.full((cfg.train_t,), PAD_ID, dtype=torch.int32, pin_memory=True)
    pos = torch.zeros(cfg.train_t, dtype=torch.int64, pin_memory=True)
    cu  = torch.full((cfg.max_docs + 2,), cfg.train_t, dtype=torch.int32, pin_memory=True)
    cu[0] = 0
    tgt = torch.zeros(cfg.train_t, dtype=torch.int64, pin_memory=True)
    w   = torch.zeros(cfg.train_t, dtype=torch.float32, pin_memory=True)
    assert len(docs) <= cfg.max_docs
    o = n_sup = 0
    for d, (ids, n_prompt, wt) in enumerate(docs):
        seq = torch.tensor(ids, dtype=torch.int64)
        n = len(ids) - 1
        assert 1 <= n_prompt <= n
        idx[o:o + n] = seq[:-1]
        tgt[o:o + n] = seq[1:]
        pos[o:o + n] = torch.arange(n)
        w[o + n_prompt - 1:o + n] = wt       # targets ids[n_prompt:] -- the assistant tokens
        n_sup += n - (n_prompt - 1)
        o += n
        cu[d + 1] = o
    assert o <= cfg.train_t
    if o < cfg.train_t:
        n_pad = cfg.train_t - o
        idx[o:] = 1 + (torch.arange(n_pad) % 4096)   # ids vary, inside the vocab
        pos[o:] = torch.arange(n_pad) % cfg.rope_t    # positions vary, inside the rotary cache
        cu[len(docs) + 1] = cfg.train_t               # the tail is its own attended segment
    up = lambda t: t.to(device, non_blocking=True)
    return dict(idx=up(idx), pos=up(pos), cu=up(cu), tgt=up(tgt), w=up(w),
                n_tok=o, n_sup=n_sup, n_docs=len(docs))

# ---- SFT plan: shuffled epochs -> packs -> steps of sft_packs_per_step packs ----
_rng = random.Random(cfg.seed)
_sft_lens = [len(ids) - 1 for ids, _ in sft_train]
sft_plan: list[list[int]] = []       # packs, each a list of sft_train indices
if cfg.run_sft:
    _n_full = int(cfg.sft_epochs)
    for ep in range(math.ceil(cfg.sft_epochs)):
        order = list(range(len(sft_train)))
        _rng.shuffle(order)
        if ep == _n_full:                  # the fractional last epoch
            order = order[:round((cfg.sft_epochs - _n_full) * len(order))]
        sft_plan += pack_sequential(_sft_lens, order)
n_sft = len(sft_plan) // cfg.sft_packs_per_step
if cfg.sft_steps_cap:
    n_sft = min(n_sft, cfg.sft_steps_cap)
sft_plan = sft_plan[:n_sft * cfg.sft_packs_per_step]

# ---- RL plan: balanced rounds (qwen-gsm8k's snake deal over context length) ----
def assemble_rounds(n_problems: int, ppr: int, epochs: int, rng: random.Random) -> list[list[int]]:
    """Per epoch, sort problems by context length and snake-deal into bins of
    `ppr` -- every bin's context sum lands near the mean (one prefill_t covers
    every round) AND every bin draws from each length stratum. Bin order
    shuffled; the remainder problems (n % ppr) dropped each epoch."""
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

round_schedule = assemble_rounds(len(train_prompts), cfg.problems_per_round, cfg.rl_epochs, _rng) if cfg.run_rl else []
if cfg.rounds_cap:
    round_schedule = round_schedule[:cfg.rounds_cap]
num_rounds = len(round_schedule)
rounds_per_epoch = max(1, len(train_prompts) // cfg.problems_per_round)
_round_ctx_max = max((sum(len(train_prompts[i]) - 1 for i in r) for r in round_schedule), default=0)
assert cfg.prefill_t >= _round_ctx_max, f"cfg.prefill_t={cfg.prefill_t} < longest round ctx {_round_ctx_max}"
assert cfg.max_ctxs >= cfg.problems_per_round

print(f"plan: SFT {n_sft} steps x {cfg.sft_packs_per_step} packs x {cfg.train_t} tok "
      f"({len(sft_plan)} packs over {cfg.sft_epochs:g} epochs) | "
      f"RL {num_rounds} rounds x {cfg.problems_per_round} problems x K={cfg.k_draws} @ budget {cfg.max_tokens} "
      f"(longest round ctx {_round_ctx_max}, prefill T {cfg.prefill_t})")


# ==============================================================================
# § Data Structures
# ==============================================================================

class Param(NamedTuple):
    """Model parameter bundled with everything needed for training it (train_stack.py)."""

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
    x0_lambdas:     Param
    resid_lambdas:  Param
    backout_lambda: Param

    # Output
    lm_head: Param

    # Rotary Cache, (rope_t, 1, d_qk/2): gathered by per-document position.
    cos: Tensor
    sin: Tensor

    def __iter__(self):
        return (v for v in vars(self).values() if isinstance(v, Param))

class LayerStash(NamedTuple):
    """One layer's forward activations, held for the backward pass (train_stack.py)."""
    x_in:       Tensor
    xb_norm:    Tensor
    xb_inv_rms: Tensor
    q_norm:     Tensor
    k_norm:     Tensor
    q_inv_rms:  Tensor
    k_inv_rms:  Tensor
    v:          Tensor
    y:          Tensor
    lse:        Tensor
    xm:         Tensor
    mlp_za:     Tensor

bf16  = lambda x: x.to(torch.bfloat16)
sum32 = lambda x: x.sum(dtype=torch.float32)


# ==============================================================================
# § Train Forward + Backward
# ==============================================================================
# train_stack.py's forward_backward, with three changes for packed
# post-training docs:
#   pos      per-document rotary positions (a decode row's cache position),
#            instead of the micro-batch offset,
#   notstart the smear never crosses a document boundary -- a decode row has
#            no previous token at its BOS, so training must not see one either,
#   w        a per-target loss weight in place of the constant loss_scale:
#            zero on prompt targets and the pad tail.
# mode: "train" (weighted CE -> backward into the grads, returns the weighted
# CE sum), "eval" (returns the per-token CE in nats).

@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def forward_backward(idx, pos, cu_seqlens, targets, w, mode="train"):
    backward = mode == "train"
    assert idx.ndim == 1
    T = idx.size(0)
    half = cfg.d_qk // 2

    cos, sin = m.cos[pos], m.sin[pos]   # (T, 1, half)
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)
    notstart = (pos != 0).to(torch.bfloat16).unsqueeze(1)   # (T, 1): 0 at every doc's first position
    x_backout = None

    # -----------------------------
    #           Forward
    # -----------------------------

    xe = F.embedding(idx, m.input_embeds.w)
    xe_inv_rms = (xe.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xe_norm = bf16(xe.float() * xe_inv_rms)      # post-norm embedding, pre-smear

    # Smear: mix the previous token's embedding into the current position.
    xe_prev = torch.cat([xe_norm[:1], xe_norm[:-1]], dim=0)
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(
        xe_norm[:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT) * notstart          # (T, 1)
    x_out = xe_norm + gate * xe_prev
    x0 = x_out

    stash = []
    for i in range(cfg.n_layers):
        x_in = x_out

        xb = m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
        xb_inv_rms = (xb.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xb_norm = bf16(xb.float() * xb_inv_rms)

        q = (xb_norm @ m.W_Q.w[i].mT).view(T, cfg.n_qo_heads, cfg.d_qk)
        k = (xb_norm @ m.W_K.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (xb_norm @ m.W_V.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)

        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_za = torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            ve_gate_a = 3 * ve_gate_za
            v = v + ve_gate_a.unsqueeze(-1) * ve

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)

        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q_norm = bf16(q.float() * q_inv_rms * 1.2)
        k_norm = bf16(k.float() * k_inv_rms * 1.2)

        # max_seqlen is train_t: the pad tail is the pack's longest segment.
        y, lse = flash_attn_varlen_fwd_lse(q_norm, k_norm, v, cu_seqlens, cfg.train_t, cfg.window_sizes[i])
        y = y.contiguous()

        attn_out = y.view(T, -1) @ m.W_O.w[i].mT
        xm = xb + attn_out

        xm_norm = bf16(xm.float() * (xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())
        mlp_za = F.relu(xm_norm @ m.W_in.w[i].mT)
        mlp_a = mlp_za.square()
        mlp_out = mlp_a @ m.W_out.w[i].mT
        x_out = xm + mlp_out

        if i == cfg.backout_layer:
            x_backout = x_out

        if backward:
            stash.append(LayerStash(x_in=x_in, xb_norm=xb_norm, xb_inv_rms=xb_inv_rms,
                                    q_norm=q_norm, k_norm=k_norm, q_inv_rms=q_inv_rms, k_inv_rms=k_inv_rms,
                                    v=v, y=y, lse=lse, xm=xm, mlp_za=mlp_za))

    xf = x_out - bf16(m.backout_lambda.w) * x_backout
    xf_inv_rms = (xf.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xf_norm = bf16(xf.float() * xf_inv_rms)

    # -----------------------------
    #           LM Head
    # -----------------------------
    tgt = targets.unsqueeze(1)                       # (T, 1)
    logits_raw = xf_norm @ m.lm_head.w.mT            # (T, d_vocab) bf16
    logits = 15.0 * torch.tanh(logits_raw.float() / 15.0)
    e = logits.exp()
    ssum = e.sum(dim=1, keepdim=True)
    lse_ce = ssum.log().squeeze(1)
    tgt_logit = logits.gather(1, tgt).squeeze(1)
    ce = lse_ce - tgt_logit                          # (T,) nats
    if not backward:
        return ce
    loss = (w * ce).sum()                            # the weighted CE this step minimizes

    # ==== Backward ====
    onehot = torch.arange(cfg.d_vocab, device=device).unsqueeze(0) == tgt
    # (1 - p), update direction (train_stack.py's sign convention), per-token weight.
    logits_grad = bf16((onehot.float() - (e / ssum)) * (1.0 - logits/15.0 * logits/15.0) * w.unsqueeze(1))
    m.lm_head.grad.add_((logits_grad.mT @ xf_norm).float())
    xf_norm_grad = logits_grad @ m.lm_head.w
    del logits_grad

    # -----------------------------
    #           Backward
    # -----------------------------
    g_resid = []; g_x0 = []

    res_ms = (xf_norm.float() * xf_norm_grad.float()).mean(dim=-1, keepdim=True)
    xf_grad = bf16(xf_inv_rms * (xf_norm_grad.float() - (xf_norm.float() * res_ms)))
    m.backout_lambda.grad.add_(-sum32(xf_grad * x_backout))

    stream_grad = xf_grad
    x0_grad = torch.zeros_like(x0)

    for i in reversed(range(cfg.n_layers)):
        st = stash[i]

        if i == cfg.backout_layer:
            stream_grad = stream_grad - bf16(m.backout_lambda.w) * xf_grad

        # --- MLP backward ---
        mlp_a = st.mlp_za.square()
        m.W_out.gbank[i].add_(stream_grad.mT @ mlp_a)
        mlp_a_grad = 2.0 * st.mlp_za * (stream_grad @ m.W_out.w[i])
        xm_inv_rms = (st.xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xm_norm = bf16(st.xm.float() * xm_inv_rms)
        m.W_in.gbank[i].add_(mlp_a_grad.mT @ xm_norm)
        xm_norm_grad = mlp_a_grad @ m.W_in.w[i]
        xm_grad = stream_grad + bf16(xm_inv_rms * (xm_norm_grad.float() - (xm_norm.float() * (xm_norm.float() * xm_norm_grad.float()).mean(dim=-1, keepdim=True))))

        # --- Attention backward ---
        xb_norm = st.xb_norm
        m.W_O.gbank[i].add_(xm_grad.mT @ st.y.view(T, -1))
        y_grad = (xm_grad @ m.W_O.w[i]).view(T, cfg.n_qo_heads, cfg.d_vo)

        q_norm_grad, k_norm_grad, v_grad = flash_attn_varlen_bwd(
            y_grad, st.q_norm, st.k_norm, st.v, st.y, st.lse, cu_seqlens, cfg.train_t,
            cfg.window_sizes[i])

        qr_grad = bf16(st.q_inv_rms * (1.2 * q_norm_grad.float() - st.q_norm.float() * ((st.q_norm.float() * q_norm_grad.float()).mean(dim=-1, keepdim=True) / 1.2)))
        kr_grad = bf16(st.k_inv_rms * (1.2 * k_norm_grad.float() - st.k_norm.float() * ((st.k_norm.float() * k_norm_grad.float()).mean(dim=-1, keepdim=True) / 1.2)))

        q1_grad, q2_grad = qr_grad[..., :half], qr_grad[..., half:]
        k1_grad, k2_grad = kr_grad[..., :half], kr_grad[..., half:]
        q_grad = torch.cat([q1_grad * cos - q2_grad * sin, q1_grad * sin + q2_grad * cos], dim=-1)
        k_grad = torch.cat([k1_grad * cos - k2_grad * sin, k1_grad * sin + k2_grad * cos], dim=-1)

        # --- VE gate backward (ve and ve_gate_za recomputed) ---
        j = cfg.ve_index[i]
        d_xn_ve = None
        if j >= 0:
            ve = F.embedding(idx, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_za = torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            ve_gate_a_grad = (v_grad * ve).sum(dim=-1)
            ve_gate_logit_grad = ve_gate_a_grad * (3 * ve_gate_za * (1 - ve_gate_za))
            m.ve_gate.gbank[j].add_(ve_gate_logit_grad.mT @ xb_norm[..., :cfg.d_ve_gate])
            ve_grad = (v_grad * (3 * ve_gate_za).unsqueeze(-1)).reshape(T, cfg.n_kv_heads * cfg.d_vo)
            m.value_embeds.gbank[j].add_(
                torch.ops.aten.embedding_dense_backward(ve_grad, idx, cfg.d_vocab, -1, False))
            d_xn_ve = ve_gate_logit_grad @ m.ve_gate.w[j]

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
        g_resid.append(sum32(xb_grad * st.x_in))
        g_x0.append(sum32(xb_grad * x0))
        x0_grad = x0_grad + m.x0_lambdas.w[i] * xb_grad
        stream_grad = m.resid_lambdas.w[i] * xb_grad
        stash[i] = None

    m.resid_lambdas.grad.add_(torch.stack(g_resid[::-1]))
    m.x0_lambdas.grad.add_(torch.stack(g_x0[::-1]))

    smeared_grad = x0_grad + stream_grad

    # --- smear backward: x = xe_norm + gate*notstart * xe_prev ---
    sg = torch.sigmoid(xe_norm[:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)   # (T, 1), recomputed
    gate = bf16(m.smear_lambda.w) * sg * notstart
    xe_norm_grad = smeared_grad.clone()
    xe_norm_grad[:-1] += gate[1:] * smeared_grad[1:]   # p's grad reaches p-1 through the smear
    gate_grad = (smeared_grad * xe_prev).sum(dim=-1, keepdim=True) * notstart     # (T, 1)
    m.smear_lambda.grad.add_(sum32(gate_grad * sg))
    gate_logit_grad = gate_grad * bf16(m.smear_lambda.w) * sg * (1 - sg)
    m.smear_gate.grad.add_((gate_logit_grad.mT @ xe_norm[:, :cfg.d_smr_gate]).float())
    xe_norm_grad[:, :cfg.d_smr_gate] += gate_logit_grad @ bf16(m.smear_gate.w)

    # --- embedding norm + token embedding scatter ---
    xe_grad = bf16(xe_inv_rms * (xe_norm_grad.float() - (xe_norm.float() * (xe_norm.float() * xe_norm_grad.float()).mean(dim=-1, keepdim=True))))
    m.input_embeds.grad.add_(
        torch.ops.aten.embedding_dense_backward(xe_grad, idx, cfg.d_vocab, -1, False))

    return loss


# ==============================================================================
# § Optimizer Math (train_stack.py, verbatim)
# ==============================================================================

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

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p: Param, grad: Tensor, t: Tensor) -> None:
    """AdamW update of `p`."""
    grad = grad.float()
    p.first_mntm.mul_(p.mntm_b1_t[t]).add_(grad * p.grad_b1_t[t])
    p.scnd_mntm.mul_(p.mntm_b2_t[t]).add_(grad.square() * p.grad_b2_t[t])
    if p.mantissa is not None:
        master = rebuild_master(p.w, p.mantissa)
    else:
        master = p.w.float()
    master.mul_(p.wd_t[t])
    master.add_(p.lr_bc_t[t] * (p.first_mntm / (p.scnd_mntm.sqrt() + p.eps_t[t])))
    if p.mantissa is not None:
        writeback_master(master, p.w, p.mantissa)
    else:
        p.w.copy_(master)

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(p: Param, grad: Tensor, t: Tensor) -> None:
    """Fused Muon step on `p`: momentum -> polar_express -> variance_reduction
    -> cautious update on the reconstructed master."""
    p.first_mntm.mul_(p.mntm_b1_t[t]).add_(grad * p.grad_b1_t[t])
    g = grad.lerp_(p.first_mntm, p.mntm_b1_t[t])

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

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

    live = p.w
    master = rebuild_master(live, p.mantissa)
    mask = (g * master) <= 0
    master.add_(p.lr_bc_t[t] * g - p.wd_t[t] * master * mask)
    writeback_master(master, live, p.mantissa)


# ==============================================================================
# § Model Load & Schedules
# ==============================================================================
# The pretrained weights arrive as bf16 (matrices, embeddings, head) and fp32
# (scalars); every bf16 weight gets a zero mantissa, so its fp32 master starts
# bit-exact and the small post-training steps do not round away on the bf16
# live copy (the pretraining embeddings had no master; at 0.02x the peak LR
# they would freeze). Grads accumulate in fp32.
#
# One schedule table covers both phases, SFT steps then RL rounds, read by one
# on-device counter `t_step`: each phase has its own LR curve and its own Adam
# bias correction (the moments are zeroed at the phase boundary).

print("=== Loading model ===")
_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
_wts = _ck["weights"]
print(f"  {ckpt_path} (step {_ck.get('step')})")

_expected = {
    "input_embeds":   (cfg.d_vocab, cfg.d_model),
    "value_embeds":   (cfg.num_ves * cfg.d_vocab, cfg.n_kv_heads * cfg.d_vo),
    "lm_head":        (cfg.d_vocab, cfg.d_model),
    "W_Q":            (cfg.n_layers, cfg.n_qo_heads * cfg.d_qk, cfg.d_model),
    "W_K":            (cfg.n_layers, cfg.n_kv_heads * cfg.d_qk, cfg.d_model),
    "W_V":            (cfg.n_layers, cfg.n_kv_heads * cfg.d_vo, cfg.d_model),
    "W_O":            (cfg.n_layers, cfg.d_model, cfg.n_qo_heads * cfg.d_vo),
    "W_in":           (cfg.n_layers, cfg.d_mlp, cfg.d_model),
    "W_out":          (cfg.n_layers, cfg.d_model, cfg.d_mlp),
    "ve_gate":        (cfg.num_ves, cfg.n_kv_heads, cfg.d_ve_gate),
    "resid_lambdas":  (cfg.n_layers,),
    "x0_lambdas":     (cfg.n_layers,),
    "smear_gate":     (1, cfg.d_smr_gate),
    "smear_lambda":   (1,),
    "backout_lambda": (1,),
}
for _n, _shape in _expected.items():
    assert _n in _wts, f"checkpoint lacks {_n}"
    assert tuple(_wts[_n].shape) == _shape, f"{_n}: checkpoint {tuple(_wts[_n].shape)} != cfg {_shape}"
assert sum(v.numel() for v in _wts.values()) == cfg.num_params

# ---- LR / momentum schedules, per phase ----
n_total = max(1, n_sft + num_rounds)
lr_mult_t  = np.zeros(n_total)
steps_1idx = np.ones(n_total)          # 1-based within its phase, for the bias corrections
muon_mom   = np.full(n_total, 0.95)
if n_sft:
    i = np.arange(n_sft, dtype=np.float64)
    wd_start = n_sft - round(cfg.sft_warmdown_frac * n_sft)
    mult = np.ones(n_sft)
    if wd_start < n_sft:
        mult[wd_start:] = (n_sft - i[wd_start:]) / (n_sft - wd_start)
    lr_mult_t[:n_sft] = cfg.sft_lr_frac * mult
    steps_1idx[:n_sft] = i + 1.0
    muon_mom[:n_sft] = 0.85 + 0.10 * np.minimum(i / 300.0, 1.0)   # nanochat chat_sft's momentum warmup
if num_rounds:
    i = np.arange(num_rounds, dtype=np.float64)
    lr_mult_t[n_sft:n_sft + num_rounds] = cfg.rl_lr_frac * (1.0 - i / num_rounds)
    steps_1idx[n_sft:n_sft + num_rounds] = i + 1.0

m = Model()
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

fp32_zeros   = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
uint16_zeros = lambda *shape: torch.zeros(*shape, dtype=torch.uint16, device=device)
dev = lambda a: torch.tensor(a, dtype=torch.float32, device=device)

adamw_configs = [
#   name,             peak lr,  b1_grad,  b2_grad,  slots      (the pretraining table; wd = 0 here)
    ("input_embeds",   0.3,      0.2,      0.005,    1),
    ("value_embeds",   0.15,     0.2,      0.005,    cfg.num_ves),
    ("lm_head",        0.008,    0.2,      0.04,     1),
    ("resid_lambdas",  0.005,    0.2,      0.05,     1),
    ("x0_lambdas",     0.5,      0.04,     0.05,     1),
    ("smear_gate",     0.2,      0.2,      0.05,     1),
    ("smear_lambda",   0.2,      0.2,      0.05,     1),
    ("backout_lambda", 0.2,      0.2,      0.05,     1),
]
for (name, peak_lr, b1_grad, b2_grad, slots) in adamw_configs:
    b1_mntm = 1 - b1_grad
    b2_mntm = 1 - b2_grad
    w = _wts[name].to(device)
    grad = fp32_zeros(w.shape)
    setattr(m, name, Param(
        name         = name,
        w            = w,
        mantissa     = uint16_zeros(w.shape) if w.dtype == torch.bfloat16 else None,
        grad         = grad,
        gbank        = list(grad.view(slots, cfg.d_vocab, -1).unbind(0)) if slots > 1 else None,
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(w.shape),
        residual_dim = None,
        lr_bc_t      = dev(lr_mult_t * peak_lr * (1.0 - b2_mntm ** steps_1idx) ** 0.5 / (1.0 - b1_mntm ** steps_1idx)),
        wd_t         = dev(np.ones(n_total)),
        mntm_b1_t    = dev(np.full(n_total, b1_mntm)),
        grad_b1_t    = dev(np.full(n_total, b1_grad)),
        mntm_b2_t    = dev(np.full(n_total, b2_mntm)),
        grad_b2_t    = dev(np.full(n_total, b2_grad)),
        eps_t        = dev(1e-10 * (1.0 - b2_mntm ** steps_1idx) ** 0.5),
    ))

muon_configs = [
#    name,      peak lr,  rdim
    ("W_Q",     0.02,      -1),
    ("W_K",     0.02,      -1),
    ("W_V",     0.02,      -1),
    ("W_O",     0.02,      -2),
    ("W_in",    0.04,      -1),
    ("W_out",   0.02,      -2),
    ("ve_gate", 0.02,      -1),
]
for (name, peak_lr, rdim) in muon_configs:
    w = _wts[name].to(device)
    assert w.dtype == torch.bfloat16
    grad = fp32_zeros(w.shape)
    scnd_shape = list(w.shape)
    scnd_shape[rdim] = 1
    setattr(m, name, Param(
        name         = name,
        w            = w,
        mantissa     = uint16_zeros(w.shape),
        grad         = grad,
        gbank        = list(grad.unbind(0)),
        first_mntm   = fp32_zeros(w.shape),
        scnd_mntm    = fp32_zeros(scnd_shape),
        residual_dim = rdim,
        lr_bc_t      = dev(lr_mult_t * peak_lr),
        wd_t         = dev(np.zeros(n_total)),
        mntm_b1_t    = dev(muon_mom),
        grad_b1_t    = dev(1.0 - muon_mom),
        mntm_b2_t    = dev(np.full(n_total, 0.9)),
        grad_b2_t    = dev(np.full(n_total, 0.1)),
        eps_t        = None,
    ))
del _wts, _ck

MUON_PARAMS  = (m.W_Q, m.W_K, m.W_V, m.W_O, m.W_in, m.W_out, m.ve_gate)
ADAMW_PARAMS = (m.lm_head, m.input_embeds, m.value_embeds, m.resid_lambdas, m.x0_lambdas,
                m.smear_gate, m.smear_lambda, m.backout_lambda)

# Schedule position: one (1,) int64 device tensor, advanced on-device
t_step = torch.zeros(1, dtype=torch.int64, device=device)

def optimizer_step():
    for p in MUON_PARAMS:
        muon_step_fused(p, p.grad, t_step)
        p.grad.zero_()
    for p in ADAMW_PARAMS:
        adamw_step_fused(p, p.grad, t_step)
        p.grad.zero_()
    t_step.add_(1)

def reset_optimizer_state():
    """Phase boundary: fresh moments (their bias corrections restart in the tables)."""
    for p in m:
        p.first_mntm.zero_()
        p.scnd_mntm.zero_()

# Rotary cache, gathered by per-document position.
channel_range = torch.arange(0, cfg.d_qk, 2, dtype=torch.float32, device=device)
inv_freq = 1.0 / (100000 ** (channel_range / cfg.d_qk))
t_pos = torch.arange(cfg.rope_t, dtype=torch.float32, device=device)
freqs = torch.outer(t_pos, inv_freq)
m.cos = freqs.cos().to(torch.bfloat16)[:, None, :]   # (rope_t, 1, half)
m.sin = freqs.sin().to(torch.bfloat16)[:, None, :]
del channel_range, inv_freq, t_pos, freqs

print(f"  {cfg.num_params:,} params on device | schedule rows {n_total} "
      f"(SFT {n_sft} @ {cfg.sft_lr_frac:g}x, RL {num_rounds} @ {cfg.rl_lr_frac:g}x)")


# ==============================================================================
# § Generation -- static-page KV cache + bucketed CUDA-graph decode
# ==============================================================================
# The whole round fits resident (512 rows x t_row tokens of K and V), so KV
# management is a STATIC page assignment: one (L, NB, page, H_kv, Dh) pool
# where row r permanently owns pages [r*P, (r+1)*P), viewed (as_strided) as a
# dense (L, max_seqs, t_row, H_kv, Dh) cache for prefill's broadcast write. The
# block table is pure indirection: a bucket drop compacts survivors to the front
# by permuting (B, P) int32 rows, so no KV ever moves. One extra NULL page backs
# parked (retired / padded) rows. (qwen-gsm8k's engine; the paging dynamics of
# nanochat's fast_engine -- refcounts, COW, VMM -- are not needed here.)
#
# Decode state per row: (input_id, cache_seqlen, block-table row, prev_emb).
# prev_emb is the smear's cross-step state, the previous token's post-norm
# PRE-smear embedding; prefill seeds it from each context's last position and
# every sibling row inherits the seed. The graph carries all four between
# windows, so a steady-state window uploads nothing and downloads one pinned
# (bucket, macro_n) token block.

# The FA2 build registers no fake impl for its kvcache op, so a direct call
# dies under fake-tensor tracing (varlen IS fake-safe; prefill calls it directly).
@torch.library.custom_op("decoder_rtx::fa_kvcache_paged", mutates_args=("k_cache", "v_cache"))
def fa_kvcache_paged(q: Tensor, k_cache: Tensor, v_cache: Tensor, k: Tensor, v: Tensor,
                     cache_seqlens: Tensor, block_table: Tensor,
                     window_left: int, window_right: int) -> Tensor:
    return fa2.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                                       block_table=block_table, causal=True,
                                       window_size=(window_left, window_right))

@fa_kvcache_paged.register_fake
def _(q, k_cache, v_cache, k, v, cache_seqlens, block_table, window_left, window_right):
    return torch.empty_like(q)


def decode_body(input_ids, cache_seqlens, block_table, k_pool, v_pool, prev_emb):
    """One decode step for B rows -- forward_backward's forward, one token per
    row, K/V from the paged cache. input_ids (B,) long | cache_seqlens (B,)
    int32 | block_table (B, P) int32 | k/v_pool (L, NB, page, H_kv, Dh) |
    prev_emb (B, D) bf16. Returns (softcapped fp32 logits (B, V), this token's
    post-norm pre-smear embedding (B, D) -- the caller writes it back into
    prev_emb)."""
    B = input_ids.shape[0]
    half = cfg.d_qk // 2
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)

    xe = F.embedding(input_ids, m.input_embeds.w)                   # (B, D)
    xe_inv_rms = (xe.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xe_norm = bf16(xe.float() * xe_inv_rms)
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(xe_norm[:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT)
    x_out = xe_norm + gate * prev_emb
    x0 = x_out

    posn = cache_seqlens.to(torch.long)
    cos, sin = m.cos[posn], m.sin[posn]                              # (B, 1, half)
    x_backout = None
    for i in range(cfg.n_layers):
        x_in = x_out
        xb = m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
        xb_inv_rms = (xb.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xb_norm = bf16(xb.float() * xb_inv_rms)

        q = (xb_norm @ m.W_Q.w[i].mT).view(B, cfg.n_qo_heads, cfg.d_qk)
        k = (xb_norm @ m.W_K.w[i].mT).view(B, cfg.n_kv_heads, cfg.d_qk)
        v = (xb_norm @ m.W_V.w[i].mT).view(B, cfg.n_kv_heads, cfg.d_vo)
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(input_ids, ve_table[j]).view(B, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_a = 3 * torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            v = v + ve_gate_a.unsqueeze(-1) * ve

        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q_norm = bf16(q.float() * q_inv_rms * 1.2)
        k_norm = bf16(k.float() * k_inv_rms * 1.2)

        wl, wr = cfg.window_sizes[i]
        y = fa_kvcache_paged(q_norm.unsqueeze(1), k_pool[i], v_pool[i], k_norm.unsqueeze(1), v.unsqueeze(1),
                             cache_seqlens, block_table, wl, wr)         # (B, 1, H, Dh)
        xm = xb + y.view(B, -1) @ m.W_O.w[i].mT

        xm_norm = bf16(xm.float() * (xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())
        mlp_a = F.relu(xm_norm @ m.W_in.w[i].mT).square()
        x_out = xm + mlp_a @ m.W_out.w[i].mT
        if i == cfg.backout_layer:
            x_backout = x_out

    xf = x_out - bf16(m.backout_lambda.w) * x_backout
    xf_norm = bf16(xf.float() * (xf.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())
    logits = 15.0 * torch.tanh((xf_norm @ m.lm_head.w.mT).float() / 15.0)
    return logits, xe_norm


def prefill_body(ids, pos, cu_seqlens, gather_idx):
    """Packed varlen prefill of the round's contexts. Returns the per-layer K
    and V exactly as attention consumed them (post-rope, normed K; VE-added V)
    stacked (L, T, H_kv, Dh), plus each context's last-position post-norm
    pre-smear embedding (max_ctxs, D) -- the smear seed for its rows. The
    caller broadcasts the K/V into the dense cache EAGERLY: an in-graph cache
    store can fuse into the producing kernels and pick up different bf16
    rounding than the packed k/v the prefill attention consumed. No lm_head:
    the forced-last-token split means prefill's only product is state."""
    T = ids.shape[0]
    half = cfg.d_qk // 2
    ve_table = m.value_embeds.w.view(cfg.num_ves, cfg.d_vocab, -1)
    cos, sin = m.cos[pos], m.sin[pos]
    notstart = (pos != 0).to(torch.bfloat16).unsqueeze(1)

    xe = F.embedding(ids, m.input_embeds.w)
    xe_inv_rms = (xe.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
    xe_norm = bf16(xe.float() * xe_inv_rms)
    xe_prev = torch.cat([xe_norm[:1], xe_norm[:-1]], dim=0)
    gate = bf16(m.smear_lambda.w) * torch.sigmoid(xe_norm[:, :cfg.d_smr_gate] @ bf16(m.smear_gate.w).mT) * notstart
    x_out = xe_norm + gate * xe_prev
    x0 = x_out

    ks, vs = [], []
    for i in range(cfg.n_layers):
        x_in = x_out
        xb = m.resid_lambdas.w[i] * x_in + m.x0_lambdas.w[i] * x0
        xb_inv_rms = (xb.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        xb_norm = bf16(xb.float() * xb_inv_rms)
        q = (xb_norm @ m.W_Q.w[i].mT).view(T, cfg.n_qo_heads, cfg.d_qk)
        k = (xb_norm @ m.W_K.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_qk)
        v = (xb_norm @ m.W_V.w[i].mT).view(T, cfg.n_kv_heads, cfg.d_vo)
        j = cfg.ve_index[i]
        if j >= 0:
            ve = F.embedding(ids, ve_table[j]).view(T, cfg.n_kv_heads, cfg.d_vo)
            ve_gate_a = 3 * torch.sigmoid(xb_norm[..., :cfg.d_ve_gate] @ m.ve_gate.w[j].mT)
            v = v + ve_gate_a.unsqueeze(-1) * ve
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)
        q_inv_rms = (q.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        k_inv_rms = (k.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt()
        q_norm = bf16(q.float() * q_inv_rms * 1.2)
        k_norm = bf16(k.float() * k_inv_rms * 1.2)
        ks.append(k_norm)
        vs.append(v)
        y, _ = flash_attn_varlen_fwd_lse(q_norm, k_norm, v, cu_seqlens, cfg.prefill_t, cfg.window_sizes[i])
        xm = xb + y.reshape(T, -1) @ m.W_O.w[i].mT
        xm_norm = bf16(xm.float() * (xm.float().square().mean(dim=-1, keepdim=True) + 2.0 ** -23).rsqrt())
        mlp_a = F.relu(xm_norm @ m.W_in.w[i].mT).square()
        x_out = xm + mlp_a @ m.W_out.w[i].mT
    return torch.stack(ks), torch.stack(vs), xe_norm[gather_idx]


def sample(logits: Tensor, inv_temp: Tensor, top_k: int) -> Tensor:
    """Gumbel-max draw == exact softmax sampling at temperature 1/inv_temp; with
    top_k > 0, over the k largest logits (nanochat chat_rl: temp 1.0, top-k 50)."""
    if top_k > 0:
        vals, idx = logits.topk(top_k, dim=-1)
        e = torch.empty_like(vals).exponential_()
        j = (vals * inv_temp - e.log()).argmax(dim=-1, keepdim=True)
        return idx.gather(1, j).squeeze(1)
    e = torch.empty_like(logits).exponential_()
    return (logits * inv_temp - e.log()).argmax(dim=-1)


def window_events(t_live: np.ndarray, allows: np.ndarray, base: int, terminals):
    """One macro-window's retirements, vectorized. t_live (n, N) sampled tokens
    for the live rows; allows (n,) per-row budgets; base = tokens generated
    before this window. Returns (done, eos, n_take): rows retiring this window,
    whether by terminal (vs budget), and how many of the window's tokens they
    keep. A terminal AT the budget position still counts as eos."""
    N = t_live.shape[1]
    hit = np.isin(t_live, terminals)
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), N)
    bidx = allows - base - 1
    eos = first <= np.minimum(bidx, N - 1)
    trunc = (~eos) & (bidx <= N - 1)
    done = eos | trunc
    n_take = np.where(eos, first + 1, np.minimum(bidx + 1, N)).astype(np.int64)
    return done, eos, n_take


class Engine:
    """Bucketed CUDA-graph decoder + compiled varlen prefill over the live
    weights. One graph per row-count bucket; each replay = one decode step; the
    driver replays cfg.macro_n times per window and reads back one pinned block."""

    def __init__(self):
        L, Hkv, Dh = cfg.n_layers, cfg.n_kv_heads, cfg.d_vo
        P = cfg.pages_per_row
        NB = cfg.max_seqs * P + 1
        self.NULL_PAGE = cfg.max_seqs * P
        self.k_pool = torch.zeros(L, NB, cfg.page, Hkv, Dh, dtype=torch.bfloat16, device=device)
        self.v_pool = torch.zeros(L, NB, cfg.page, Hkv, Dh, dtype=torch.bfloat16, device=device)
        el = cfg.page * Hkv * Dh
        self.k_dense = self.k_pool.as_strided((L, cfg.max_seqs, cfg.t_row, Hkv, Dh),
                                              (NB * el, P * el, Hkv * Dh, Dh, 1))
        self.v_dense = self.v_pool.as_strided((L, cfg.max_seqs, cfg.t_row, Hkv, Dh),
                                              (NB * el, P * el, Hkv * Dh, Dh, 1))
        self.bt_identity = torch.arange(cfg.max_seqs * P, dtype=torch.int32,
                                        device=device).view(cfg.max_seqs, P)
        # Static graph state
        self.input_ids = torch.zeros(cfg.max_seqs, dtype=torch.long, device=device)
        self.cache_seqlens = torch.zeros(cfg.max_seqs, dtype=torch.int32, device=device)
        self.block_table = torch.full((cfg.max_seqs, P), self.NULL_PAGE, dtype=torch.int32, device=device)
        self.prev_emb = torch.zeros(cfg.max_seqs, cfg.d_model, dtype=torch.bfloat16, device=device)
        self.tok_buf = torch.zeros(cfg.max_seqs, dtype=torch.long, device=device)
        self.token_record = torch.zeros(cfg.max_seqs, cfg.macro_n, dtype=torch.long, device=device)
        self.tok_host = torch.empty(cfg.max_seqs, cfg.macro_n, dtype=torch.long, pin_memory=True)
        self.tok_host_np = self.tok_host.numpy()
        self.inv_temp = torch.tensor(1.0 / cfg.temperature, dtype=torch.float32, device=device)
        # Prefill static buffers (compiled, not captured: a few ms per round)
        self.pf_ids = torch.zeros(cfg.prefill_t, dtype=torch.int32, device=device)
        self.pf_pos = torch.zeros(cfg.prefill_t, dtype=torch.int64, device=device)
        self.pf_cu = torch.zeros(cfg.max_ctxs + 2, dtype=torch.int32, device=device)
        self.pf_gather = torch.zeros(cfg.max_ctxs, dtype=torch.int64, device=device)
        self.decode_fn = torch.compile(decode_body, dynamic=False)
        self.prefill_fn = torch.compile(prefill_body, dynamic=False, fullgraph=True)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._mempool = None
        self.TERM = np.array(TERMINALS, dtype=np.int64)

    def _macro_body(self, b: int) -> None:
        logits, x_pre = self.decode_fn(self.input_ids[:b], self.cache_seqlens[:b], self.block_table[:b],
                                       self.k_pool, self.v_pool, self.prev_emb[:b])
        tok = sample(logits, self.inv_temp, cfg.top_k)
        self.tok_buf[:b] = tok
        self.input_ids[:b] = tok
        self.cache_seqlens[:b] += 1
        self.prev_emb[:b].copy_(x_pre)

    @torch.no_grad()
    def capture(self) -> None:
        print(f"  engine: {cfg.max_seqs} rows x {cfg.t_row} tok "
              f"({(self.k_pool.numel() + self.v_pool.numel()) * 2 / 2**30:.1f} GB KV) | "
              f"buckets {cfg.buckets} | macro_n {cfg.macro_n} | prefill T={cfg.prefill_t} | "
              f"temp {cfg.temperature:g} top-k {cfg.top_k} (gumbel-argmax)", flush=True)
        print("  capture+compile decode buckets:", flush=True)
        for b in sorted(cfg.buckets, reverse=True):
            t0 = time.perf_counter()
            self.input_ids[:] = 0
            self.block_table[:] = self.bt_identity
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
        t0 = time.perf_counter()
        self.pf_ids[:] = PAD_ID
        self.pf_pos[:] = 0
        self.pf_cu[:] = cfg.prefill_t
        self.pf_cu[0] = 0
        self.pf_gather[:] = 0
        self.prefill_fn(self.pf_ids, self.pf_pos, self.pf_cu, self.pf_gather)
        torch.cuda.synchronize()
        print(f"    prefill compile: {time.perf_counter() - t0:5.1f}s", flush=True)

    @torch.no_grad()
    def _prefill(self, specs: list[tuple]):
        """specs: (meta, prompt_ids, k, allow). Context = prompt[:-1]; the forced
        first decode input = prompt[-1], so the first SAMPLED token already comes
        out of the decode graph. Prefills every context in ONE compiled call,
        broadcasts K/V and the smear seed into the K sibling rows, and seeds the
        graph state. Returns (n_rows, ctx lens, budgets, metas)."""
        ctx_tok = sum(len(p) - 1 for _, p, _, _ in specs)
        n_rows = sum(k for _, _, k, _ in specs)
        assert len(specs) <= cfg.max_ctxs and ctx_tok <= cfg.prefill_t and n_rows <= cfg.max_seqs

        ids, pos, cu, gather = [], [], [0], []
        for _, p, _, _ in specs:
            ctx = p[:-1]
            ids.extend(ctx)
            pos.extend(range(len(ctx)))
            cu.append(cu[-1] + len(ctx))
            gather.append(cu[-1] - 1)
        pad = cfg.prefill_t - len(ids)
        cu = cu + [cfg.prefill_t] * (cfg.max_ctxs + 2 - len(cu))       # pad tail + ghost segments
        gather = gather + [0] * (cfg.max_ctxs - len(gather))
        self.pf_ids.copy_(torch.tensor(ids + [PAD_ID] * pad, dtype=torch.int32), non_blocking=True)
        self.pf_pos.copy_(torch.tensor(pos + [0] * pad, dtype=torch.int64), non_blocking=True)
        self.pf_cu.copy_(torch.tensor(cu, dtype=torch.int32), non_blocking=True)
        self.pf_gather.copy_(torch.tensor(gather, dtype=torch.int64), non_blocking=True)
        k_all, v_all, seeds = self.prefill_fn(self.pf_ids, self.pf_pos, self.pf_cu, self.pf_gather)

        r0, o = 0, 0
        metas, plens, forced, allows = [], [], [], []
        for ci, (meta, p, k, allow) in enumerate(specs):
            plen = len(p) - 1
            self.k_dense[:, r0:r0 + k, :plen] = k_all[:, o:o + plen].unsqueeze(1)
            self.v_dense[:, r0:r0 + k, :plen] = v_all[:, o:o + plen].unsqueeze(1)
            self.prev_emb[r0:r0 + k] = seeds[ci]
            metas.extend([meta] * k)
            plens.extend([plen] * k)
            forced.extend([p[-1]] * k)
            allows.extend([allow] * k)
            r0 += k
            o += plen

        plens = np.asarray(plens, dtype=np.int64)
        allows = np.asarray(allows, dtype=np.int64)
        bucket = next(x for x in cfg.buckets if x >= n_rows)
        self.input_ids[:n_rows] = torch.tensor(forced, dtype=torch.long, device=device)
        self.cache_seqlens[:n_rows] = torch.tensor(plens, dtype=torch.int32, device=device)
        self.block_table[:n_rows] = self.bt_identity[:n_rows]
        if bucket > n_rows:                          # park the padded tail
            self.cache_seqlens[n_rows:bucket] = 0
            self.block_table[n_rows:bucket] = self.NULL_PAGE
        return n_rows, plens, allows, metas

    @torch.no_grad()
    def run_round(self, specs: list[tuple]) -> tuple[list[dict], dict]:
        """Prefill, then decode every row to completion. Returns the rows
        (sampled ids INCLUDING the terminal when one was hit; `eos` says so)
        and the generation telemetry."""
        t0 = time.perf_counter()
        B0, plens, allows, metas = self._prefill(specs)
        bucket = next(x for x in cfg.buckets if x >= B0)

        orig = np.arange(B0)
        live = np.ones(B0, dtype=bool)
        gen_buf = np.empty((B0, (-(-int(allows.max()) // cfg.macro_n)) * cfg.macro_n), dtype=np.int64)
        # A row past its capacity would index past its P block-table entries and
        # write K into a NEIGHBOR row's page -- silent corruption, not a crash.
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
            if nb < bucket:                          # bucket drop: compact survivors
                idxs = torch.from_numpy(lp).to(device, non_blocking=True)
                for buf in (self.input_ids, self.cache_seqlens, self.block_table, self.prev_emb):
                    buf[:lp.size].copy_(buf.index_select(0, idxs))
                orig, plens, allows = orig[lp], plens[lp], allows[lp]
                live = np.ones(lp.size, dtype=bool)
                bucket = nb
                self.cache_seqlens[lp.size:bucket] = 0
                self.block_table[lp.size:bucket] = self.NULL_PAGE
                park_dirty = False
                lp = np.arange(lp.size)
            elif park_dirty:                         # park mid-bucket retirees in place
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
            torch.cuda.synchronize()                 # the window's single host sync
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
                rows[o_] = dict(meta=metas[o_], ids=gen_buf[o_, :n_].tolist(), eos=bool(eos[ri]))
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
        gen = dict(gen_s=round(gen_s, 2), gen_tok=tok_total, gen_tok_per_s=round(tok_total / gen_s, 0),
                   occ=round(100 * tok_total / max(1, paid_slots), 1),
                   t50=round(t50 if t50 is not None else gen_s, 2),
                   t90=round(t90 if t90 is not None else gen_s, 2))
        return rows, gen

    @torch.no_grad()
    def teacher_forced(self, prompt_ids: list[int], cont_ids: list[int]) -> Tensor:
        """The engine check: prefill the prompt, then step the compiled decode
        body through cont_ids, feeding the KNOWN next token instead of sampling.
        Returns the fp32 logits (len(cont_ids), V) that predicted each of them."""
        b = cfg.buckets[0]
        self.block_table[:] = self.NULL_PAGE
        self._prefill([(0, prompt_ids, 1, len(cont_ids))])
        out = []
        for nxt in cont_ids:
            logits, x_pre = self.decode_fn(self.input_ids[:b], self.cache_seqlens[:b], self.block_table[:b],
                                           self.k_pool, self.v_pool, self.prev_emb[:b])
            out.append(logits[0].clone())
            self.input_ids[0] = nxt
            self.cache_seqlens[:b] += 1
            self.prev_emb[:b].copy_(x_pre)
        return torch.stack(out)


# ==============================================================================
# § Eval -- val subset in-loop, full test after SFT and at the end (both through the graphs)
# ==============================================================================

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
    """mean@k / pass@k at the training sampler (temp, top-k) over the given test
    problems, scored by the reward's `#### n`. The sampler RNG is saved and
    restored, so the training rollout stream is identical to an eval-off run."""
    rng_state = torch.cuda.get_rng_state()
    t0 = time.perf_counter()
    n_ok, n_fmt, n_trunc, n_roll, len_sum = {}, 0, 0, 0, 0
    for wave in make_eval_waves(problem_idxs, k):
        rows, _ = engine.run_round([(i, test_prompts[i], k, cfg.max_tokens) for i in wave])
        for r in rows:
            i = r["meta"]
            text = decode(r["ids"][:-1] if r["eos"] else r["ids"])
            pred = extract_answer(text)
            n_ok[i] = n_ok.get(i, 0) + int(pred == test_gold[i])
            n_fmt += pred is not None
            n_trunc += not r["eos"]
            n_roll += 1
            len_sum += len(r["ids"])
    torch.cuda.set_rng_state(rng_state)
    n_prob = len(problem_idxs)
    return dict(label=label, n_problems=n_prob, k=k,
                mean_at_k=round(100 * sum(n_ok.values()) / max(1, n_roll), 2),
                pass_at_k=round(100 * sum(v > 0 for v in n_ok.values()) / max(1, n_prob), 2),
                fmt_pct=round(100 * n_fmt / max(1, n_roll), 1),
                trunc_pct=round(100 * n_trunc / max(1, n_roll), 1),
                mean_len=round(len_sum / max(1, n_roll), 1),
                eval_s=round(time.perf_counter() - t0, 1))


# ==============================================================================
# § Logging
# ==============================================================================

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
print0(f"GPU: {gpu_device_name}", console=True)

if not cfg.use_wandb:
    class DummyWandb:
        summary = {}
        def log(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
        def finish(self): pass
    wandb_run = DummyWandb()
else:
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config={name: getattr(cfg, name, None) for name in GSM8KConfig.__annotations__}
               | dict(n_sft=n_sft, num_rounds=num_rounds, checkpoint=ckpt_path),
    )
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

eval_rows: list[dict] = []
eval_time = 0.0

def log_eval(step: int, phase: str, res: dict) -> None:
    global eval_time
    eval_time += res["eval_s"]
    res = dict(step=step, phase=phase, **res)
    eval_rows.append(res)
    print0(f"  [eval {phase} {step:4d}] {res['label']}({res['n_problems']}): "
           f"mean@{res['k']} {res['mean_at_k']:5.2f} | pass@{res['k']} {res['pass_at_k']:5.2f} | "
           f"fmt {res['fmt_pct']:4.1f}% | trunc {res['trunc_pct']:4.1f}% | len {res['mean_len']:5.1f} | "
           f"{res['eval_s']}s", console=True)
    lbl, k = res["label"], res["k"]
    wandb_run.log({"step": step,
                   f"eval/{lbl}_mean_at_{k}": res["mean_at_k"],
                   f"eval/{lbl}_pass_at_{k}": res["pass_at_k"],
                   f"eval/{lbl}_fmt_pct": res["fmt_pct"],
                   f"eval/{lbl}_trunc_pct": res["trunc_pct"],
                   f"eval/{lbl}_mean_len": res["mean_len"],
                   "time/wall_seconds": time.perf_counter() - run_wall_t0})

def write_checkpoint(tag: str) -> None:
    os.makedirs(f"logs/{cfg.run_name}", exist_ok=True)
    path = f"logs/{cfg.run_name}/model_{tag}.pt"
    torch.save(dict(step=int(t_step.item()), code=code, weights={p.name: p.w.cpu() for p in m}), path)
    print0(f"  checkpoint -> {path}", console=True)


# ==============================================================================
# § Warmup -- capture graphs, engine check, compile the training step
# ==============================================================================

setup_t0 = time.perf_counter()
engine = Engine()
engine.capture()

if cfg.engine_check:
    # Teacher-forced decode vs the training forward on real SFT docs: the same
    # ids through prefill + the paged decode body must give the same next-token
    # CE as forward_backward's eval mode. Not bitwise (paged decode vs packed
    # varlen, different GEMM shapes), but a wrong smear seed, rotary position,
    # window, VE gate or backout shows up as a CE gap of O(1).
    t0 = time.perf_counter()
    _docs = sorted(sft_train, key=lambda d: len(d[0]))
    _fit = [d for d in _docs if len(d[0]) <= cfg.t_row]
    worst = 0.0
    for ids, P in (_fit[len(_fit) // 2], _fit[-1]):
        pk = build_pack([(ids, P, 1.0)])
        ce_train = forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="eval")[P - 1:len(ids) - 1]
        logits = engine.teacher_forced(ids[:P], ids[P:])
        tgt = torch.tensor(ids[P:], device=device)
        ce_dec = torch.logsumexp(logits, dim=-1) - logits.gather(1, tgt.unsqueeze(1)).squeeze(1)
        d = (ce_dec - ce_train).abs()
        worst = max(worst, float(d.max()))
        print0(f"  engine check: doc {len(ids)} tok (prompt {P}) | CE train {float(ce_train.mean()):.4f} "
               f"decode {float(ce_dec.mean()):.4f} | |dCE| max {float(d.max()):.4f} mean {float(d.mean()):.4f} | "
               f"tf-acc {float((logits.argmax(-1) == tgt).float().mean()):.3f}", console=True)
    assert worst < 0.5, f"decode disagrees with the training forward (max |dCE| {worst:.3f})"
    for p in m:
        p.grad.zero_()
    print0(f"    engine check: {time.perf_counter() - t0:5.1f}s", console=True)

# Compile the training step on one dummy pack (w=0: every grad an exact zero,
# cleared regardless), and the optimizer kernels at a throwaway step-0 index
# against those zeros (no state changes: moments stay 0, the update is 0, and
# wd is 1 / 0).
t0 = time.perf_counter()
_dummy = [([3 + (j % 97) for j in range(80)] + [5 + (j % 89) for j in range(200)] + [ASSISTANT_END], 80, 0.0)
          for _ in range(cfg.train_t // 280 + 1)]
pk = build_pack(_dummy[:len(pack_ffd([len(d[0]) - 1 for d in _dummy])[0])])
forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="train")
for p in m:
    p.grad.zero_()
if sft_val or cfg.engine_check:
    forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="eval")
_t0 = torch.zeros(1, dtype=torch.int64, device=device)
for p in MUON_PARAMS:
    muon_step_fused(p, p.grad, _t0)
    p.grad.zero_()
for p in ADAMW_PARAMS:
    adamw_step_fused(p, p.grad, _t0)
    p.grad.zero_()
del _dummy, pk, _t0
torch.cuda.synchronize()
print0(f"    train fwd+bwd + optimizer compile: {time.perf_counter() - t0:5.1f}s", console=True)
setup_s = time.perf_counter() - setup_t0
print0(f"  setup {setup_s:.0f}s (wall so far {time.perf_counter() - run_wall_t0:.0f}s) | "
       f"peak mem {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", console=True)
assert int(t_step.item()) == 0

sft_val_packs = []
if sft_val:
    _lens = [len(ids) - 1 for ids, _ in sft_val]
    sft_val_packs = [build_pack([(sft_val[i][0], sft_val[i][1], 1.0) for i in members]) for members in pack_ffd(_lens)]

def sft_val_loss() -> float:
    """Mean CE per held-out assistant token."""
    tot = torch.zeros((), dtype=torch.float32, device=device)
    n = 0
    for pk in sft_val_packs:
        ce = forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="eval")
        tot += (ce * (pk["w"] > 0)).sum()
        n += pk["n_sup"]
    return tot.item() / max(1, n)

gc.collect()
gc.freeze()
gc.disable()


# ==============================================================================
# § SFT Loop
# ==============================================================================
# One step = cfg.sft_packs_per_step packs; the loss is the mean CE over the
# step's assistant tokens (w = 1/N on every one of them, so the packed forward
# just sums).

sft_timed = []
smooth_loss = 0.0
if cfg.run_sft:
    print0(f"=== SFT: {n_sft} steps ===", console=True)
    for step in range(n_sft):
        if cfg.sft_eval_every and step % cfg.sft_eval_every == 0:
            if sft_val_packs:
                vl = sft_val_loss()
                print0(f"  [sft {step:4d}] val_loss {vl:.4f}", console=True)
                wandb_run.log({"step": step, "sft/val_loss": vl})
            log_eval(step, "sft", run_eval(VAL_SUBSET, cfg.eval_k, "subset"))

        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        members = sft_plan[step * cfg.sft_packs_per_step:(step + 1) * cfg.sft_packs_per_step]
        n_sup = sum(len(sft_train[i][0]) - sft_train[i][1] for pack in members for i in pack)
        w_tok = 1.0 / n_sup
        loss = torch.zeros((), dtype=torch.float32, device=device)
        n_tok = 0
        for pack in members:
            pk = build_pack([(sft_train[i][0], sft_train[i][1], w_tok) for i in pack])
            loss += forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="train")
            n_tok += pk["n_tok"]
        optimizer_step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        dt = time.perf_counter() - step_t0

        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        debiased = smooth_loss / (1 - 0.9 ** (step + 1))
        sft_timed.append(dt)                       # compile happened in § Warmup: every step counts
        tok_per_sec = int(n_tok / dt)
        mfu = 100 * cfg.num_flops_per_token * n_tok / dt / gpu_peak_flops
        print0(f"sft {step:04d}/{n_sft:04d} | loss: {debiased:.4f} | lr_mult: {lr_mult_t[step]:.3f} | "
               f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} (sup {n_sup:,}) | bf16_mfu: {mfu:.1f} | "
               f"total: {np.sum(sft_timed)/60:.2f}m", console=True)
        wandb_run.log({"step": step, "sft/loss": debiased, "sft/loss_raw": train_loss,
                       "sft/lr_mult": float(lr_mult_t[step]), "sft/dt": dt, "sft/tok_per_sec": tok_per_sec,
                       "sft/mfu": mfu, "sft/n_sup": n_sup, "time/wall_seconds": time.perf_counter() - run_wall_t0})

    sft_s = float(np.sum(sft_timed))
    print0(f"=== SFT done: {n_sft} steps in {sft_s/60:.2f}m (mean {np.mean(sft_timed) if sft_timed else 0:.3f}s/step) ===",
           console=True)
    if sft_val_packs:
        vl = sft_val_loss()
        print0(f"  [sft {n_sft:4d}] val_loss {vl:.4f}", console=True)
        wandb_run.log({"step": n_sft, "sft/val_loss": vl})
    if cfg.save_checkpoint:
        write_checkpoint("sft")
    if cfg.sft_eval_every:
        log_eval(n_sft, "sft", run_eval(VAL_SUBSET, cfg.eval_k, "subset"))
    if cfg.full_eval:
        log_eval(n_sft, "sft", run_eval(list(range(len(test_prompts))), cfg.eval_k, "full"))
    reset_optimizer_state()
else:
    sft_s = 0.0
assert int(t_step.item()) == n_sft


# ==============================================================================
# § RL Loop
# ==============================================================================
# Per round: K rollouts for each of problems_per_round train problems through
# the graphs -> `#### n` reward (partial credit for a formatted wrong answer)
# -> per-problem advantage r - mean(r) (nanochat chat_rl; an all-equal group
# carries no gradient and is skipped) -> ONE optimizer step on every
# completion token except the terminal, token-mean over the round's trained
# tokens (the fix ladder's LOSS_NORM=token_mean).

@dataclass
class RoundStats:
    round: int = 0
    n_rollouts: int = 0
    n_correct:  int = 0
    solve_rate: float = 0.0
    n_eos:      int = 0
    n_trunc:    int = 0
    fmt_pct:    float = 0.0
    mean_len:   float = 0.0
    gen_s:         float = 0.0
    gen_tok:       int   = 0
    gen_tok_per_s: float = 0.0
    occ:           float = 0.0
    t50:           float = 0.0
    t90:           float = 0.0
    train_s:       float = 0.0
    n_groups_used: int   = 0
    n_groups_sat:  int   = 0
    n_groups_dead: int   = 0
    n_docs:        int   = 0
    n_loss_tokens: int   = 0
    n_packs:       int   = 0
    pad_pct:       float = 0.0
    loss_total:    float = 0.0
    lr_mult:       float = 0.0
    round_s:       float = 0.0
    elapsed_s:     float = 0.0

WANDB_GROUPS = {
    "time":   ("round_s", "gen_s", "train_s", "t50", "t90", "gen_tok_per_s", "occ", "elapsed_s"),
    "length": ("mean_len", "gen_tok", "n_eos", "n_trunc"),
    "reward": ("solve_rate", "n_correct", "fmt_pct", "loss_total"),
    "groups": ("n_groups_used", "n_groups_sat", "n_groups_dead", "n_docs", "n_loss_tokens", "n_packs", "pad_pct"),
    "optim":  ("lr_mult",),
}
_WANDB_PREFIX = {f: g for g, fs in WANDB_GROUPS.items() for f in fs}
assert set(_WANDB_PREFIX) <= {f.name for f in fields(RoundStats)}

curve: list[dict] = []
rl_t0 = time.perf_counter()
if cfg.run_rl:
    print0(f"=== RL: {num_rounds} rounds ===", console=True)
    for rnd in range(num_rounds):
        step = n_sft + rnd
        if cfg.rl_eval_every and rnd % cfg.rl_eval_every == 0:
            log_eval(step, "rl", run_eval(VAL_SUBSET, cfg.eval_k, "subset"))

        r_t0 = time.perf_counter()
        stats = RoundStats(round=rnd)

        # -- generation ------------------------------------------------------
        idxs = round_schedule[rnd]
        rows, gen = engine.run_round([(i, train_prompts[i], cfg.k_draws, cfg.max_tokens) for i in idxs])
        for k_, v_ in gen.items():
            setattr(stats, k_, v_)

        # -- grade, group, advantage -> the round's training docs -------------
        by_pid: dict[int, tuple[list, list]] = {}
        n_fmt = len_sum = 0
        for r in rows:
            pid = r["meta"]
            body = r["ids"][:-1] if r["eos"] else r["ids"]     # the terminal is never trained
            text = decode(body)
            pred = extract_answer(text)
            rw = 1.0 if pred == train_gold[pid] else (cfg.fmt_reward if pred is not None else 0.0)
            stats.n_correct += rw == 1.0
            stats.n_eos += r["eos"]
            n_fmt += pred is not None
            len_sum += len(r["ids"])
            bodies, rews = by_pid.setdefault(pid, ([], []))
            bodies.append(body)
            rews.append(rw)

        docs = []
        for pid, (bodies, rews) in by_pid.items():
            r = np.asarray(rews, dtype=np.float64)
            if r.size < 2 or r.std() < 1e-6:
                stats.n_groups_sat += int(r[0] >= 1.0)
                stats.n_groups_dead += int(r[0] <= 0.0)
                continue
            stats.n_groups_used += 1
            adv = r - r.mean()
            for body, a in zip(bodies, adv):
                if body:
                    docs.append((train_prompts[pid] + body, len(train_prompts[pid]), float(a)))
        n_loss_tok = sum(len(d[0]) - d[1] for d in docs)
        docs = [(ids, P, a / max(1, n_loss_tok)) for ids, P, a in docs]
        stats.n_docs = len(docs)
        stats.n_loss_tokens = n_loss_tok

        # -- train -----------------------------------------------------------
        t_train0 = time.perf_counter()
        loss = torch.zeros((), dtype=torch.float32, device=device)
        if docs:
            packs = pack_ffd([len(d[0]) - 1 for d in docs])
            n_tok = 0
            for members in packs:
                pk = build_pack([docs[i] for i in members])
                loss += forward_backward(pk["idx"], pk["pos"], pk["cu"], pk["tgt"], pk["w"], mode="train")
                n_tok += pk["n_tok"]
            stats.n_packs = len(packs)
            stats.pad_pct = round(100.0 * (1 - n_tok / (cfg.train_t * len(packs))), 1)
        optimizer_step()                             # unconditional (chat_rl steps every round)
        stats.loss_total = round(loss.item(), 6)
        torch.cuda.synchronize()
        stats.train_s = round(time.perf_counter() - t_train0, 2)

        # -- telemetry -------------------------------------------------------
        stats.n_rollouts = len(rows)
        stats.solve_rate = round(stats.n_correct / max(1, len(rows)), 4)
        stats.n_trunc = len(rows) - stats.n_eos
        stats.fmt_pct = round(100 * n_fmt / max(1, len(rows)), 1)
        stats.mean_len = round(len_sum / max(1, len(rows)), 1)
        stats.lr_mult = float(lr_mult_t[step])
        stats.round_s = round(time.perf_counter() - r_t0, 2)
        el = time.perf_counter() - rl_t0
        stats.elapsed_s = round(el, 2)
        row = asdict(stats)
        curve.append(row)
        eta = el / (rnd + 1) * (num_rounds - rnd - 1)
        print0(f"rl {rnd:04d}/{num_rounds:04d} | {stats.round_s:5.2f}s ({stats.gen_s:.2f} gen / {stats.train_s:.2f} trn) | "
               f"solve {100 * stats.solve_rate:5.1f}% | fmt {stats.fmt_pct:5.1f}% | len {stats.mean_len:5.1f} | "
               f"trunc {stats.n_trunc:3d} | dead {stats.n_groups_dead:2d} sat {stats.n_groups_sat:2d} /{len(idxs)} | "
               f"tok/s {stats.gen_tok_per_s:,.0f} | total {el / 60:5.1f}m | eta {eta / 60:4.1f}m", console=True)
        wandb_run.log({"step": step, "time/wall_seconds": time.perf_counter() - run_wall_t0,
                       **{f"rl_{_WANDB_PREFIX.get(k_, 'train')}/{k_}": v_ for k_, v_ in row.items() if k_ != "round"}})

        if (rnd + 1) % rounds_per_epoch == 0 or rnd + 1 == num_rounds:
            ep = curve[-((rnd % rounds_per_epoch) + 1):]
            ep_cor = sum(c["n_correct"] for c in ep)
            ep_roll = sum(c["n_rollouts"] for c in ep)
            print0(f"  == epoch {rnd // rounds_per_epoch + 1}/{cfg.rl_epochs} | solve {ep_cor:,}/{ep_roll:,} "
                   f"({100 * ep_cor / max(1, ep_roll):5.2f}%) | avg round {sum(c['round_s'] for c in ep) / len(ep):.2f}s "
                   f"(gen {sum(c['gen_s'] for c in ep) / len(ep):.2f} + train {sum(c['train_s'] for c in ep) / len(ep):.2f}) ==",
                   console=True)

    if cfg.rl_eval_every:
        log_eval(n_sft + num_rounds, "rl", run_eval(VAL_SUBSET, cfg.eval_k, "subset"))
rl_s = time.perf_counter() - rl_t0 if cfg.run_rl else 0.0


# ==============================================================================
# § Results
# ==============================================================================

final_step = n_sft + num_rounds
if cfg.full_eval and cfg.run_rl:
    log_eval(final_step, "final", run_eval(list(range(len(test_prompts))), cfg.eval_k, "full"))
if cfg.save_checkpoint:
    write_checkpoint("final")

total_wall = time.perf_counter() - run_wall_t0
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"setup {setup_s/60:.2f}m | SFT {sft_s/60:.2f}m | RL {rl_s/60:.2f}m (incl. in-loop evals) | "
       f"evals {eval_time/60:.2f}m | wall {total_wall/60:.2f}m", console=True)
if curve:
    med = lambda k_: sorted(c[k_] for c in curve)[len(curve) // 2]
    print0(f"  RL rounds {len(curve)} | solve {curve[0]['solve_rate']} -> {curve[-1]['solve_rate']} | "
           f"round_s med {med('round_s')} (gen {med('gen_s')} + train {med('train_s')}) | "
           f"gen tok/s med {med('gen_tok_per_s'):,.0f}", console=True)
for r in eval_rows:
    print0(f"  eval {r['phase']} {r['step']:4d} {r['label']}: mean@{r['k']} {r['mean_at_k']} pass@{r['k']} {r['pass_at_k']} "
           f"fmt {r['fmt_pct']} trunc {r['trunc_pct']}", console=True)

result = dict(run_name=cfg.run_name, checkpoint=ckpt_path, n_sft=n_sft, num_rounds=num_rounds,
              setup_s=round(setup_s, 1), sft_s=round(sft_s, 1), rl_s=round(rl_s, 1),
              eval_s=round(eval_time, 1), wall_s=round(total_wall, 1),
              peak_mem_gb=round(torch.cuda.max_memory_reserved() / 2**30, 1),
              solve_rate_first=(curve[0]["solve_rate"] if curve else None),
              solve_rate_last=(curve[-1]["solve_rate"] if curve else None),
              evals=eval_rows)
os.makedirs(f"logs/{cfg.run_name}", exist_ok=True)
with open(f"logs/{cfg.run_name}/result.json", "w") as f:
    json.dump(result, f, indent=1)
wandb_run.log({"step": final_step, "time/setup_seconds": setup_s, "time/sft_seconds": sft_s,
               "time/rl_seconds": rl_s, "time/eval_seconds": eval_time, "time/wall_seconds": total_wall})
if cfg.use_wandb:
    wandb_run.summary.update({k_: v_ for k_, v_ in result.items() if not isinstance(v_, (list, dict))})
wandb_run.save(logfile, policy="now")
wandb_run.finish()
