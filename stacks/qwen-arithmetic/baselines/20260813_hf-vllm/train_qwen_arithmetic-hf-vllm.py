# train_qwen_arithmetic-hf-vllm.py
#
# Single-file GRPO speedrun: Qwen2.5-0.5B-Instruct on basic arithmetic
# (ChrisMcCormick/basic-arithmetic), 1x H100 (A100 works too — everything here
# is plain bf16 + sdpa). This is the STACK-CONSTRAINED variant: HF
# `transformers` for the training model, vLLM (in-process) for generation, LoRA
# via peft. Its sibling train_qwen_arithmetic.py implements the same task with
# a handwritten model and engine; this file is the ceiling of what the
# off-the-shelf stack reaches, distilled from the grpo-course experiment line
# (exp/train_h100.py) with every losing arm removed.
#
# THE REFERENCE — this exact config, measured on an H100 PCIe (run
# `overlong-none`, wandb grpo-arithmetic, 2026-08-14):
#   68 generation rounds of 64 problems x K=16 @ budget 640, lr 1e-5 (LoRA)
#   val 89.0% @ step 180 | test ID 87.2% | test OOD 86.2% | 256 steps | ~7.5 min
# Baked-in winners, each measured against its alternative rather than assumed:
#   - compaction: live groups repacked into as few, full optimizer steps as
#     will hold them (vs. fixed slices: same gradient work, ~2.4x fuller steps,
#     no zero-live steps burning schedule)
#   - length bucketing: micro-batches sorted by completion length and padded to
#     their OWN max (gradient-identical; removes ~40% of forwarded padding)
#   - overlong handling: NONE. DAPO's mask scored 76.5/72.8 vs none's 87.2/86.2
#     at the same 68-round budget — the length collapse it targets happens with
#     truncation already at 0-1%, and on 2-operand arithmetic the rambling that
#     masking permits costs accuracy
#   - rewards: correct (1.0) + uses_method (1.0, gated on correct). has_words
#     is DELETED: once the method reward exists it is a flat +0.5 on nearly
#     every rollout — constant within a group, so GRPO's group-mean subtraction
#     erases it while it dilutes the two signals that vary
#   - attention: sdpa. FA3 measured within noise (val 83.5 vs 82.5, test
#     identical) — the training forward is not where this run's time goes
#   - budget 640: sized to clear the UNTRAINED model's p99 (~512). 256 truncated
#     ~20% of early rollouts, and a truncated rollout scores 0 — teaching
#     'shorter is safer' exactly while the model is still learning what to write
#   - lr 1e-5: swept 1e-4 -> 79.0%, 3e-5 -> 80.0%, 1e-5 -> 83.0% (val)
#   - compiled log-prob tail: 184 -> 123 ms on the full fwd+bwd (see § 3.3)
#
# Run (after `bash setup_hf_vllm.sh` or any env with vllm 0.19 + transformers):
#   python train_qwen_arithmetic-hf-vllm.py
# There is no command line and no config env: every knob is a field of Config
# in § 2, edited in place — a run is defined by the source. Telemetry: one
# console line per optimizer step, full rows to wandb (unset WANDB_API_KEY to
# disable), best-val markdowns + final LoRA checkpoint in the run dir.
#
# Env needs: vllm==0.19.0, transformers==4.57.6, peft, torchao>=0.16.0 (peft
# refuses to build LoRA against older), datasets, numpy, wandb, tabulate,
# huggingface_hub — see requirements-hf-vllm.txt. vLLM 0.19 is the last CUDA-12
# build; it pins torch==2.10.0.

# ==========================================================================
# 1 - Setup
# ==========================================================================

import os
import sys
import re
import time
import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset

# By default vLLM runs its engine in a separate process, which means we can't
# hand it GPU tensors directly -- and we need to, every time we update the
# weights. This makes it run in *our* process instead. It has to be set before
# vLLM is imported.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# vLLM logs an INFO line every time the prefix cache is reset -- which we do on
# every weight sync -- and it lands in the middle of the training log.
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
datasets.disable_progress_bar()


@contextmanager
def suppress_output():
    """Redirect stdout/stderr to devnull — silences vLLM's multi-screen engine
    init spew, which would otherwise bury the training log."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def fmt_time(seconds):
    """Format seconds as Xm YYs."""
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


# --------------------------------------------------------------------------
# 1.1. Weights & Biases
# --------------------------------------------------------------------------

import wandb

api_key = os.environ.get("WANDB_API_KEY", None)
if api_key:
    wandb.login(key=api_key)
    use_wandb = True
else:
    print("No WANDB_API_KEY found -- logging disabled.")
    os.environ["WANDB_MODE"] = "disabled"
    use_wandb = False


# --------------------------------------------------------------------------
# 1.2. Dataset
# --------------------------------------------------------------------------
# Difficulty-balanced 2-operand arithmetic (add/sub/mult/div), integer answers.
# train 10,000 | val 200 | test_id 400 | test_ood 400 (OOD = held-out
# phrasings of the same operations). Each row carries the chat `prompt` turns
# and the integer `answer`.

from huggingface_hub import snapshot_download

data_dir = Path.home() / ".cache" / "qwen-arithmetic" / "data"
data_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="ChrisMcCormick/basic-arithmetic",
    repo_type="dataset",
    local_dir=str(data_dir),
)

train_dataset = Dataset.from_parquet(str(data_dir / "train.parquet"))
val_dataset = Dataset.from_parquet(str(data_dir / "val.parquet"))
test_id_dataset = Dataset.from_parquet(str(data_dir / "test_id.parquet"))
test_ood_dataset = Dataset.from_parquet(str(data_dir / "test_ood.parquet"))

print("======== Dataset Sizes ========")
print(f"train:    {len(train_dataset)}")
print(f"val:      {len(val_dataset)}")
print(f"test_id:  {len(test_id_dataset)}")
print(f"test_ood: {len(test_ood_dataset)}")

example = train_dataset[0]
print("Prompt:")
for turn in example["prompt"]:
    print(f"  [{turn['role']}] {turn['content']}")
print(f"\nAnswer: {example['answer']}")


# --------------------------------------------------------------------------
# 1.3. Model and Tokenizer
# --------------------------------------------------------------------------

device = "cuda"
capability = torch.cuda.get_device_capability()
assert capability >= (8, 0), \
    f"sm_{capability[0]}{capability[1]}: this script assumes native bf16 (Ampere+)"

# The whole precision strategy. The frozen base stays bf16 for the tensor
# cores; the trained (LoRA) params live in fp32 so an lr-sized optimizer
# update has somewhere to land -- bf16 has only 7 mantissa bits, and a 1e-5
# update on a bf16 param quietly rounds away to nothing. The failure mode of
# getting this wrong is a silent accuracy problem, not an error.
dtype = torch.bfloat16          # the frozen base weights
compute_dtype = torch.bfloat16  # what autocast runs the matmuls in
param_dtype = torch.float32     # what the trainable (LoRA) params live in

# TF32 (19-bit) matmuls are a free speedup on Ampere+.
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"

print(f"Device:      {torch.cuda.get_device_name(0)} (sm_{capability[0]}{capability[1]})")
print(f"Training in: {dtype}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# sdpa, on the evidence: FA3 measured within noise on this workload (val 83.5
# vs 82.5, test identical) -- the training forward is not where the time goes,
# and vLLM generation uses its own bundled kernels either way.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=dtype, attn_implementation="sdpa")
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

# Attach the adapter. The base weights are frozen from here on, which is what
# lets them stay bf16 -- nothing ever applies a 1e-5 update to them.
from peft import LoraConfig, get_peft_model

model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0, # dropout would desync pi_old from pi_theta
    target_modules=[
     "q_proj", "k_proj", "v_proj", "o_proj",
     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
))

base_params = sum(p.numel() for p in model.parameters())

# The point of the split: the *trained* params are fp32, so AdamW's eps and
# its ~lr-sized updates have somewhere to land.
for name, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.to(param_dtype)

trainable = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)

print("Loaded Qwen/Qwen2.5-0.5B-Instruct")
print(f"Base parameters:      {base_params / 1e6:.0f}M  ({dtype})")
print(f"Trainable (LoRA):     {n_trainable / 1e6:.2f}M ({param_dtype})")
print(f"Trainable fraction:   {n_trainable / base_params:.2%}")


# ==========================================================================
# 2 - Configuration
# ==========================================================================
# The winning config, baked. There is no command line and no env override —
# every knob is edited here, in place, so a run is defined by the source.

@dataclass
class Config:

    # ── Batch structure ────────────────────────────────────────────────────
    group_size: int = 16              # rollouts per problem
    problems_per_mini_epoch: int = 64 # problems per generation round
    problems_per_step: int = 8        # problems per optimizer step
    micro_batch_size: int = 32        # rollouts per forward/backward

    # ── Sampling ───────────────────────────────────────────────────────────
    temperature: float = 0.8          # >0 so the group has variety to compare
    top_p: float = 0.9
    # Sized off the UNTRAINED model: its p99 is ~512, and a truncated rollout
    # loses its final number, scores 0, and teaches 'shorter is safer' exactly
    # when the model is still learning what to write. (Trained, p99 falls to
    # 62-75 tokens and truncation sits at 0-1%.)
    max_completion_length: int = 640
    max_prompt_length: int = 256      # headroom for the chat template + question

    # Eval is measurement, not compute budget: capping it below the step-0
    # model's own lengths would truncate the baseline and silently deflate the
    # number every later step is compared against. Greedy completions run
    # shorter than sampled ones, so 512 clears them.
    eval_max_tokens: int = 512

    # ── GRPO loss ──────────────────────────────────────────────────────────
    epsilon: float = 0.2              # clip range, symmetric
    adv_eps: float = 1e-4             # guards against divide-by-zero std

    # ── Rewards ────────────────────────────────────────────────────────────
    weight_correct: float = 1.0
    # Early on almost nothing collects it (the model must be correct AND name
    # a method), so at equal weight it is still the faintest thing in the
    # reward -- at 0.5 it was fainter and measured worse.
    weight_uses_method: float = 1.0

    # ── Budget ─────────────────────────────────────────────────────────────
    # In GENERATION ROUNDS, not optimizer steps: compaction makes steps/round
    # a function of saturation (~4 of 8 slots by mid-run), so rounds are the
    # unit of data the run actually consumes -- and a fully-saturated pool
    # yields zero steps, which a step budget would spin on forever.
    generation_rounds: int = 68

    # ── Optimizer ──────────────────────────────────────────────────────────
    learning_rate: float = 1e-5   # swept: 1e-4 -> 79.0%, 3e-5 -> 80.0%, 1e-5 -> 83.0%
    max_grad_norm: float = 1.0
    warmup_steps: int = 10

    # ── Evaluation & logging ───────────────────────────────────────────────
    eval_every: int = 20              # optimizer steps between val evals
    vllm_gpu_memory_utilization: float = 0.25
    wandb_project: str = "qwen-arithmetic"
    run_name: str = ""                # "" = describe the config

    # ── Reproducibility ────────────────────────────────────────────────────
    seed_phrase: str = "RAG Pack"

    # ── Derived ────────────────────────────────────────────────────────────

    @property
    def rollouts_per_mini_epoch(self):
        return self.problems_per_mini_epoch * self.group_size

    @property
    def rollouts_per_step(self):
        return self.problems_per_step * self.group_size

    @property
    def steps_per_mini_epoch(self):
        # An upper bound, not a count: it's what you get when nothing
        # saturates. The real number is ceil(live_groups / problems_per_step),
        # decided per pool at runtime.
        return self.problems_per_mini_epoch // self.problems_per_step

    @property
    def max_micro_batches_per_step(self):
        # Also an upper bound: saturated groups are dropped before the pool is
        # chunked. Ceiling division -- the last micro-batch is usually short.
        return -(-self.rollouts_per_step // self.micro_batch_size)


cfg = Config()

# Turn the seed phrase into a seed, so a memorable string picks the run.
seed = int(hashlib.sha256(cfg.seed_phrase.encode()).hexdigest(), 16) % (2**32)
torch.manual_seed(seed)
np.random.seed(seed)

# Terse descriptive run names -- a timestamp tells you nothing in a W&B run
# list. Set cfg.run_name to say what the run IS; the fallback describes the
# config.
run_name = cfg.run_name or (
    f"g{cfg.group_size}-p{cfg.problems_per_step}-mb{cfg.micro_batch_size}"
    f"-len{cfg.max_completion_length}"
)

# The name is deliberately reusable, so keep the output dir from colliding.
run_root = Path.home() / ".cache" / "qwen-arithmetic" / "runs"
run_dir = run_root / run_name
_n = 2
while run_dir.exists():
    run_dir = run_root / f"{run_name}-{_n}"
    _n += 1
run_dir.mkdir(parents=True)

print(f"Run:  {run_name}")
print(f"Seed: '{cfg.seed_phrase}' -> {seed}")


# ==========================================================================
# 3 - Loss Calculations
# ==========================================================================


# --------------------------------------------------------------------------
# 3.1. Rewards
# --------------------------------------------------------------------------

_ALL_NUMS_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")

def is_correct(completion, answer):
    """1.0 if the final number in the completion matches the answer."""
    matches = _ALL_NUMS_RE.findall(completion)
    if not matches:
        return 0.0
    # The regex accepts scientific notation, so a rollout that ends in "1e999"
    # parses to inf and int(inf) raises OverflowError -- which killed a run at
    # step 176. A number we cannot turn into an int is simply not the answer.
    try:
        last_number = int(float(matches[-1]))
    except (OverflowError, ValueError):
        return 0.0
    return 1.0 if last_number == answer else 0.0

# The method reward. Correctness alone pays for NOT working the problem -- a
# short answer has fewer places to go wrong -- so this names the vocabulary of
# a written method, and hard-zeros the phrasings that skip one.
_METHOD_GOOD_RE = re.compile(r'\b(?:method|place|align|long|distributive)\b', re.IGNORECASE)
_METHOD_BAD_RE = re.compile(r'\b(?:simply|calculator|directly)\b', re.IGNORECASE)

def uses_method(text, correct):
    """1.0 for a correct answer that shows a method; 0.0 if it dodges one.

    Gated on correctness on purpose: rewarding method words on a wrong answer
    would pay for reciting the ritual without doing the arithmetic.
    """
    if _METHOD_BAD_RE.search(text):
        return 0.0
    return 1.0 if (correct and _METHOD_GOOD_RE.search(text)) else 0.0

def compute_rewards(completions, answers):
    """Weighted reward per completion, plus the components for logging."""
    r_correct = np.array([is_correct(t, a) for t, a in zip(completions, answers)])
    r_method = np.array([uses_method(t, c) for t, c in zip(completions, r_correct)])
    total = cfg.weight_correct * r_correct + cfg.weight_uses_method * r_method
    return total, r_correct, r_method


# --------------------------------------------------------------------------
# 3.2. Advantages & Step Planning
# --------------------------------------------------------------------------

def group_advantages(rewards):
    """Group-relative, std-normalized advantages. rewards: [num_rollouts]"""
    grouped = rewards.view(-1, cfg.group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    return ((grouped - mean) / (std + cfg.adv_eps)).flatten()


def live_group_ids(group_live):
    """Indices of the groups that still carry a gradient."""
    # .tolist() once, rather than indexing a CUDA tensor element by element.
    return [g for g, alive in enumerate(group_live.tolist()) if alive]


def rows_for_groups(groups):
    """The rollout rows of `groups`. Liveness is a property of a whole group
    -- it shares one std -- so groups arrive already filtered."""
    return [g * cfg.group_size + i for g in groups for i in range(cfg.group_size)]


def plan_steps(group_live):
    """Assign a pool's live groups to optimizer steps (compacted).

    A saturated group -- all rollouts scoring the same -- has zero std, so
    every advantage in it is zero and it carries no gradient. Rather than
    carving the pool into fixed slices and letting the saturated groups leave
    holes (a fully-saturated slice would yield a step that trains on nothing),
    the survivors decide the schedule: as few steps as will hold them...
    """
    live = live_group_ids(group_live)
    if not live:
        return []

    n_steps = -(-len(live) // cfg.problems_per_step)   # ceiling division

    # ...dealt out round-robin rather than filling each step before starting
    # the next. Both give `n_steps` steps, but greedy packing leaves the
    # remainder alone in a stub step (26 groups at 8/step -> 8,8,8,2), while
    # dealing spreads it (-> 7,7,6,6). Since a step's gradient is scaled by
    # how full it is, dealing keeps every update about the same size -- which
    # is the point of compacting in the first place.
    return [live[i::n_steps] for i in range(n_steps)]


def micro_batches(rows, lengths):
    """Chunk rows into pieces of at most `micro_batch_size`, sorted by
    completion length first (length bucketing).

    Every rollout in a micro-batch is padded out to the longest one in it, so
    a single 500-token rollout drags 31 forty-token siblings up to 500 columns
    of mostly padding. Sorting puts the long ones together, and the cost of a
    chunk falls to about its own rollouts' length instead of the pool's worst
    case. Gradient-identical: a rollout's loss is a mean over its own tokens
    and the step sums those, so which chunk a row rides in is irrelevant.
    """
    rows = sorted(rows, key=lambda r: lengths[r])
    for lo in range(0, len(rows), cfg.micro_batch_size):
        yield rows[lo : lo + cfg.micro_batch_size]


def micro_batch_view(batch, rows):
    """Trim the pool's one big rectangle down to what `rows` actually need.

    `generate_mini_epoch` pads the whole pool into a single tensor laid out as

        [ left-pad | prompt | completion | right-pad ]

    -- prompts padded on the left, completions on the right -- and its width is
    set by the longest prompt and the longest completion *anywhere* in the
    pool. Every forward pass over any subset of it would then pay for that
    worst case (in a typical pool the mean completion is ~76 tokens and the
    longest ~335, so ~4/5 of the completion block is padding that still gets
    multiplied through the model and the 151k-wide lm_head).

    The layout is what makes this cheap to fix. Because prompts sit flush
    right against the boundary and completions flush left, the live tokens of
    *any* row subset occupy one contiguous band of columns: the last
    `n_prompt` of the prompt block, then the first `n_comp` of the completion
    block. So this is a single slice -- no re-padding, no copy, nothing to
    keep in sync.

    Returns the trimmed ids and mask plus `n_comp`, which is the
    `num_completion_tokens` the loss should use for this micro-batch.
    """
    n_prompt = max(batch["prompt_lengths"][r] for r in rows)
    n_comp = max(batch["completion_lengths"][r] for r in rows)
    p_pool = batch["n_prompt_tokens"]
    cols = slice(p_pool - n_prompt, p_pool + n_comp)
    return batch["input_ids"][rows][:, cols], batch["attention_mask"][rows][:, cols], n_comp


# --------------------------------------------------------------------------
# 3.3. The GRPO Loss
# --------------------------------------------------------------------------

# --- the log-prob tail, compiled on ONE fixed shape ------------------------
# Measured on an H100 at [64, 90, 151936]: this tail was 21.5 ms of a 46.7 ms
# forward, because `.float()` writes a 3.3 GiB fp32 logits copy and the
# per-row logsumexp loop launches one kernel per sequence. Compiled, Inductor
# fuses upcast + gather + reduction into a single pass that never materializes
# the fp32 tensor: 1.8 ms, and 184 -> 123 ms on the full fwd+bwd.
#
# The trick that makes a FIXED shape possible without padding the batch: this
# computation is independent per row, so [B, C, V] flattens to [B*C, V] and
# runs in blocks of exactly _LOGP_BLOCK rows. One graph, compiled once, no
# matter how many live groups a step has or how long the completions are.
# The leftover rows go through eager -- same math, no recompile, nothing
# dropped. (Compiling the 24-layer model body too was measured at a further
# 2% for 66 s of compile time, so it is deliberately left eager.)
_LOGP_BLOCK = 2048

def _logp_block(logits_block, target_block):
    """[N, V] logits + [N] targets -> [N] log-probs. One fused kernel."""
    lg = logits_block.float() / cfg.temperature
    selected = lg.gather(-1, target_block.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(lg, dim=-1)

_logp_block_compiled = torch.compile(_logp_block, dynamic=False)


def _token_logps(logits, completion_ids):
    """Same math as the eager version, in fixed-size compiled blocks."""
    rows, toks, vocab = logits.shape
    flat, tgt = logits.reshape(-1, vocab), completion_ids.reshape(-1)
    n_full = flat.shape[0] // _LOGP_BLOCK * _LOGP_BLOCK

    out = []
    for lo in range(0, n_full, _LOGP_BLOCK):
        out.append(_logp_block_compiled(flat[lo:lo + _LOGP_BLOCK], tgt[lo:lo + _LOGP_BLOCK]))
    if n_full < flat.shape[0]:                      # ragged tail, eager
        out.append(_logp_block(flat[n_full:], tgt[n_full:]))
    return torch.cat(out).view(rows, toks)


def forward_logps(input_ids, attention_mask, num_completion_tokens):
    """Log-prob of each completion token, under the current weights."""

    # The matmuls run in bf16 on the tensor cores. This is the speedup.
    with torch.autocast("cuda", dtype=compute_dtype):
        # `logits_to_keep` runs the lm_head on only the last N positions
        # instead of all of them. The 24 transformer layers still see the
        # whole sequence -- they have to, the completion attends to the prompt
        # -- but the lm_head is a [896, 151936] projection, and the fp32
        # logsumexp over that vocabulary is the single most expensive thing in
        # the pass. Every prompt position gets thrown away anyway (only
        # completion tokens have a target), so computing them is pure waste.
        # num_completion_tokens + 1 leaves exactly the positions the slice
        # below needs: position t predicts token t+1, so the completion's
        # first target is predicted by the last prompt position.
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
            logits_to_keep=num_completion_tokens + 1,
        ).logits

    # Position t predicts token t+1, so drop the final position and keep the
    # last `num_completion_tokens` -- those line up with the completion.
    logits = logits[:, :-1, :][:, -num_completion_tokens:, :]

    # The token the model actually generated at each of those positions.
    completion_ids = input_ids[:, -num_completion_tokens:]

    # The fp32 upcast, the gather of the generated token's logit, and the
    # logsumexp that turns it into a log-prob -- log_softmax(x)[i] is just
    # x[i] - logsumexp(x). fp32 because a reduction over a 151k vocabulary is
    # exactly what bf16 is bad at; in fused blocks because the fp32 copy would
    # otherwise be the largest tensor in the step and autograd would hold it
    # alive until the backward pass.
    return _token_logps(logits, completion_ids)

def grpo_loss(logps, old_logps, advantages, mask):
    """The GRPO objective for one micro-batch.

    logps, old_logps, mask: [num_rollouts, num_completion_tokens]
    advantages:             [num_rollouts]

    Returns the loss plus a couple of tensors we want for metrics.
    """
    advantages = advantages.unsqueeze(1)  # [B] -> [B, 1], broadcasts over tokens

    # The importance ratio. Log space, so a subtraction then exp.
    ratio = torch.exp(logps - old_logps)

    # `clamp` is *flat* outside [1-eps, 1+eps]: its gradient out there is
    # exactly zero.
    clipped = torch.clamp(ratio, 1 - cfg.epsilon, 1 + cfg.epsilon)

    # Take the pessimistic branch: whichever pushes the loss up. Together with
    # the clamp above, this is the entire clipping mechanism -- there is no
    # explicit mask anywhere. Whenever `min` selects the clamped branch, that
    # token is contributing a *constant* to the loss, so it still shows up in
    # the loss value but contributes nothing to the gradient.
    per_token_loss = -torch.min(ratio * advantages, clipped * advantages)

    # ...which makes "was this token clipped?" the question of whether the
    # clamped branch is the one `min` took -- not simply "the ratio left the
    # range".
    was_clipped = (clipped * advantages) < (ratio * advantages)

    # Average over the tokens of each sequence, then over sequences. Dividing
    # per-sequence keeps a long rollout from outvoting a short one.
    num_tokens = mask.sum(dim=1).clamp(min=1.0)
    per_sequence_loss = (per_token_loss * mask).sum(dim=1) / num_tokens
    loss = per_sequence_loss.mean()

    return loss, ratio, was_clipped


# ==========================================================================
# 4 - Generation
# ==========================================================================


# --------------------------------------------------------------------------
# 4.1. vLLM Engine
# --------------------------------------------------------------------------

with suppress_output():
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_num_seqs=cfg.rollouts_per_mini_epoch,
        # Must cover the EVAL budget too -- it could be the larger of the two.
        max_model_len=cfg.max_prompt_length + max(cfg.max_completion_length,
                                                  cfg.eval_max_tokens),
        enable_prefix_caching=True,
        seed=seed,
    )

print("vLLM ready.")

def sync_weights_to_vllm():
    """Copy the training model's weights into the vLLM engine.

    With LoRA the trainable delta lives in separate A/B matrices, but vLLM here
    holds a plain dense model -- so fold the adapter into the base weights,
    push, and immediately fold it back out. peft's merge/unmerge does this in
    place, so nothing extra is allocated.
    """
    model.merge_adapter()
    try:
        named = [
            (name.replace("base_model.model.", "").replace("base_layer.", ""), p.data)
            for name, p in model.named_parameters()
            if "lora_" not in name
        ]
        llm.apply_model(lambda m: m.load_weights(named))
    finally:
        model.unmerge_adapter()
    # The prefix cache holds keys/values computed under the old weights.
    llm.reset_prefix_cache()


# --------------------------------------------------------------------------
# 4.2. Generation Step
# --------------------------------------------------------------------------

def pad_sequences(sequences, pad_value, side="right"):
    """Pad a list of 1-D tensors into a rectangular batch, plus the mask."""
    max_len = max(s.size(0) for s in sequences)
    padded, masks = [], []
    for s in sequences:
        pad_len = max_len - s.size(0)
        pad_spec = (pad_len, 0) if side == "left" else (0, pad_len)
        padded.append(F.pad(s, pad_spec, value=pad_value))
        masks.append(F.pad(torch.ones_like(s), pad_spec, value=0))
    return torch.stack(padded), torch.stack(masks)

def generate_mini_epoch(prompt_cursor):
    """Generate and score one pool of rollouts. Returns (batch_dict, new_cursor)."""
    model.eval()

    # Next slice of problems, wrapping around the dataset.
    indices = [
        (prompt_cursor + i) % len(train_dataset)
        for i in range(cfg.problems_per_mini_epoch)
    ]
    prompt_cursor = (prompt_cursor + cfg.problems_per_mini_epoch) % len(train_dataset)

    prompts = [train_dataset[i]["prompt"] for i in indices]
    answers = [train_dataset[i]["answer"] for i in indices]

    # Repeat each prompt group_size times -- these become the groups.
    prompt_texts, expanded_answers = [], []
    for p, a in zip(prompts, answers):
        text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        prompt_texts.extend([text] * cfg.group_size)
        expanded_answers.extend([a] * cfg.group_size)

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_completion_length,
        n=1,
    )
    t_sample = time.perf_counter()

    # Generate!
    outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=False)
    sample_time = time.perf_counter() - t_sample

    completions = [o.outputs[0].text for o in outputs]
    completion_ids = [torch.tensor(list(o.outputs[0].token_ids), device=device) for o in outputs]
    prompt_ids = [torch.tensor(list(o.prompt_token_ids), device=device) for o in outputs]

    # Ask vLLM why each rollout stopped, rather than inferring it from the
    # length: finish_reason is "length" exactly when the sampler hit the cap.
    # (Telemetry only -- with budget 640 truncation sits at 0-1%, and scoring
    # truncated rollouts like any other measured BETTER than masking them.)
    truncated = np.array([o.outputs[0].finish_reason == "length" for o in outputs])

    # Score, then convert rewards to group-relative advantages.
    rewards, r_correct, r_method = compute_rewards(completions, expanded_answers)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = group_advantages(rewards_t)

    # A group is "live" if its rollouts didn't all score the same. A saturated
    # group -- all correct or all wrong -- has zero std, so every advantage in
    # it is zero and it contributes no gradient.
    group_live = rewards_t.view(-1, cfg.group_size).std(dim=1, unbiased=False) > 0

    # Prompts pad left, completions pad right, so the completion tokens all land
    # in the same trailing block of the sequence.
    prompt_padded, prompt_mask = pad_sequences(prompt_ids, tokenizer.pad_token_id, "left")
    completion_padded, completion_mask = pad_sequences(
        completion_ids, tokenizer.pad_token_id, "right"
    )

    input_ids = torch.cat([prompt_padded, completion_padded], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    num_completion_tokens = completion_padded.size(1)

    # The true (unpadded) lengths, so a micro-batch can be trimmed to its own
    # worst case rather than the pool's. See `micro_batch_view`.
    prompt_lengths = [int(p.size(0)) for p in prompt_ids]
    completion_lengths = [int(c.size(0)) for c in completion_ids]
    n_prompt_tokens = prompt_padded.size(1)

    # Freeze pi_old for this pool -- for the live groups only. A saturated
    # group's rows are never read by the loss, so they stay zeros and we skip
    # the forward pass over them.
    t_logps = time.perf_counter()
    old_logps = torch.zeros(
        input_ids.size(0), num_completion_tokens, dtype=param_dtype, device=device
    )
    live_rows = rows_for_groups(live_group_ids(group_live))

    # The pool's rows are still laid out in one rectangle, so a row's log-probs
    # go into columns [0, n_comp) of its slot -- completions pad right, so a
    # row's real tokens always start at column 0 and the rest stays zero.
    _view = {"input_ids": input_ids, "attention_mask": attention_mask,
             "prompt_lengths": prompt_lengths,
             "completion_lengths": completion_lengths,
             "n_prompt_tokens": n_prompt_tokens}
    with torch.no_grad():
        for rows in micro_batches(live_rows, completion_lengths):
            ids, attn, n_comp = micro_batch_view(_view, rows)
            old_logps[rows, :n_comp] = forward_logps(ids, attn, n_comp)
    logps_time = time.perf_counter() - t_logps

    model.train()

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "old_logps": old_logps,
        "num_completion_tokens": num_completion_tokens,
        "prompt_lengths": prompt_lengths,
        "completion_lengths": completion_lengths,
        "n_prompt_tokens": n_prompt_tokens,
        "group_live": group_live,
        # Metrics
        "mean_reward": float(rewards.mean()),
        "mean_correct": float(r_correct.mean()),
        "mean_uses_method": float(r_method.mean()),
        "mean_length": float(np.mean(completion_lengths)),
        "max_length": float(np.max(completion_lengths)),
        "frac_truncated": float(truncated.mean()),
        "sample_time": sample_time,
        "logps_time": logps_time,
    }
    return batch, prompt_cursor


# --------------------------------------------------------------------------
# 4.3. Evaluation
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    accuracy: float
    num_correct: int
    completions: list
    mean_length: float


def evaluate(dataset, max_tokens=None):
    """Greedy-decode the whole dataset and report accuracy."""
    max_tokens = max_tokens or cfg.eval_max_tokens
    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in dataset["prompt"]
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompt_texts, params, use_tqdm=False)

    completions = [o.outputs[0].text for o in outputs]
    mean_length = float(np.mean([len(o.outputs[0].token_ids) for o in outputs]))

    answers = list(dataset["answer"])
    num_correct = int(sum(is_correct(t, a) for t, a in zip(completions, answers)))

    return EvalResult(
        accuracy=num_correct / len(completions) * 100.0,
        num_correct=num_correct,
        completions=completions,
        mean_length=mean_length,
    )


_OP_SHORT = {"+": "Add", "-": "Sub", "*": "Mult", "/": "Div"}


def write_val_markdown(result, step):
    """Write one markdown file per validation pass, for before/after comparison."""
    path = run_dir / f"step{step}_val.md"
    answers = list(val_dataset["answer"])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Validation @ Step {step}\n\n")
        f.write(f"Accuracy: {result.accuracy:.1f}%\n\n---\n\n")
        for i, (comp, ans) in enumerate(zip(result.completions, answers)):
            correct = is_correct(comp, ans) > 0
            op = _OP_SHORT.get(val_dataset["op"][i], val_dataset["op"][i])
            tag = "PASS" if correct else "FAIL"
            f.write(f"**Q{i+1} - {op}** | {tag}\n")
            f.write(f"{val_dataset['question'][i]} (Answer: {int(ans)})\n\n")
            f.write(f"{comp.strip()}\n\n")
    return path


# ==========================================================================
# 5 - Run Training
# ==========================================================================


# --------------------------------------------------------------------------
# 5.1. Review & Setup
# --------------------------------------------------------------------------

import tabulate

rows = [
    ["Group size", cfg.group_size, "Rollouts per problem"],
    ["Problems / mini-epoch", cfg.problems_per_mini_epoch, "Problems per generation round"],
    ["Rollouts / mini-epoch", cfg.rollouts_per_mini_epoch, "Sequences vLLM generates at once"],
    ["Problems / step", cfg.problems_per_step, "Problems per optimizer step"],
    ["Rollouts / step", cfg.rollouts_per_step, "Sequences behind one weight update"],
    ["Steps / mini-epoch", cfg.steps_per_mini_epoch,
     "Optimizer steps before regenerating (max -- compacted)"],
    ["Micro-batch size", cfg.micro_batch_size, "Rollouts per forward/backward"],
    ["Micro-batches / step", cfg.max_micro_batches_per_step, "Grad accumulation depth (max)"],
    ["Generation rounds", cfg.generation_rounds, "The run's data budget"],
]
print(tabulate.tabulate(rows, headers=["Item", "Value", "Meaning"], tablefmt="github"))
print("")

rows = [
    ["Learning rate", cfg.learning_rate],
    ["Warmup steps", cfg.warmup_steps],
    ["Max completion length", cfg.max_completion_length],
    ["Temperature", cfg.temperature],
    ["Top p", cfg.top_p],
    ["Clip epsilon", cfg.epsilon],
    ["Reward weights", [cfg.weight_correct, cfg.weight_uses_method]],
]
print(tabulate.tabulate(rows, headers=["Item", "Value"], tablefmt="github"))

# A mini-epoch must divide evenly into optimizer steps.
assert cfg.problems_per_mini_epoch % cfg.problems_per_step == 0, \
    "problems_per_mini_epoch must be a multiple of problems_per_step"

# `micro_batch_size` is deliberately unconstrained -- it's a pure memory knob,
# and the last micro-batch of a step is allowed to be short. Turn it up until
# the card complains; turn it down if it OOMs.

optimizer = torch.optim.AdamW(
    trainable,
    lr=cfg.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)

def warmup_then_constant(step):
    """Linear warmup, then hold. Keeps early steps from wrecking the policy."""
    if step < cfg.warmup_steps:
        return step / max(cfg.warmup_steps, 1)
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_constant)

if use_wandb:
    wandb.init(project=cfg.wandb_project, name=run_name, config=vars(cfg))


# --------------------------------------------------------------------------
# 5.2. Training Loop
# --------------------------------------------------------------------------

global_step = 0
prompt_cursor = 0
best_accuracy = -1.0
best_step = -1
clip_by_substep = [[] for _ in range(cfg.steps_per_mini_epoch)]
live_history = []            # live groups per optimizer step
pool_live_history = []       # live groups per generated pool
steps_per_pool_history = []  # optimizer steps that pool turned into
pad_history = []             # (real, padded) completion positions per step

# Where the wall clock goes. Logged to W&B too, but tracked here so the summary
# works without it.
t_sample_total = t_logps_total = t_train_total = t_eval_total = 0.0

# Baseline before any training. This counts as the incumbent best, so if
# training makes things worse, the summary says so rather than hiding it.
result = evaluate(val_dataset)
write_val_markdown(result, 0)
best_accuracy, best_step = result.accuracy, 0
print(f"  Step {0:>4d} │ val {result.accuracy:5.1f}%  (len {result.mean_length:.0f})")

t0 = time.time()

for round_no in range(cfg.generation_rounds):

    # ── Generation ───────────────────────────────────────────────────────────
    mini_epoch, prompt_cursor = generate_mini_epoch(prompt_cursor)
    t_sample_total += mini_epoch["sample_time"]
    t_logps_total += mini_epoch["logps_time"]

    # ── Schedule the pool into optimizer steps ───────────────────────────────
    step_plan = plan_steps(mini_epoch["group_live"])
    pool_live_history.append(sum(len(s) for s in step_plan))
    steps_per_pool_history.append(len(step_plan))

    if not step_plan:
        print("  Step      │ pool fully saturated -- no live groups, regenerating")

    # ── Training over that pool ──────────────────────────────────────────────
    for step_in_epoch, step_groups in enumerate(step_plan):

        t_step = time.perf_counter()

        # Compacted, the groups are all live by construction and only the last
        # step or two are short, so this is "how full is the step".
        live_groups = len(step_groups)
        live_history.append(live_groups)

        rows_to_train = rows_for_groups(step_groups)

        # Accumulate weighted sums rather than averaging per-micro-batch means:
        # micro-batches hold unequal numbers of rollouts *and* unequal numbers
        # of tokens, so a mean of means would quietly mis-weight them.
        clipped_tokens, ratio_dev_sum, token_total = 0.0, 0.0, 0.0
        loss_sum, rows_trained = 0.0, 0
        # Completion positions the forward passes actually ran, vs. positions
        # carrying a real token. The gap is padding -- the thing bucketing is
        # for -- and it's the honest denominator for "how much of this step's
        # compute was wasted".
        padded_tokens, real_tokens = 0, 0

        for rows in micro_batches(rows_to_train, mini_epoch["completion_lengths"]):

            # Trim the pool rectangle to this micro-batch's own longest prompt
            # and completion, rather than the pool's. Everything downstream --
            # the mask, pi_old, the loss -- is sliced to the same width.
            ids, attn, n_comp = micro_batch_view(mini_epoch, rows)
            mask = mini_epoch["completion_mask"][rows][:, :n_comp]
            padded_tokens += n_comp * len(rows)
            real_tokens += sum(mini_epoch["completion_lengths"][r] for r in rows)

            # Score the completion tokens under the *current* weights. This one
            # line is the bulk of the step's cost.
            logps = forward_logps(ids, attn, n_comp)

            loss, ratio, was_clipped = grpo_loss(
                logps,
                mini_epoch["old_logps"][rows][:, :n_comp],
                mini_epoch["advantages"][rows],
                mask,
            )

            # `loss` is a mean over this micro-batch, so weight it by the share
            # of the step's rollouts it holds before accumulating -- ragged
            # chunks would otherwise be counted as if they were full ones.
            #
            # The denominator is rollouts_per_step, not the number of live
            # rollouts in this step: it keeps an update's size proportional to
            # how full its step is, so the one short step at the end of a pool
            # doesn't land as hard as the full ones. (With compaction that
            # ratio is ~1.0 on almost every step instead of ~0.4.)
            (loss * len(rows) / cfg.rollouts_per_step).backward()

            with torch.no_grad():
                clipped_tokens += (was_clipped.float() * mask).sum().item()
                ratio_dev_sum += ((ratio - 1).abs() * mask).sum().item()
                token_total += mask.sum().item()
                loss_sum += loss.item() * len(rows)
                rows_trained += len(rows)

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        global_step += 1
        step_time = time.perf_counter() - t_step
        t_train_total += step_time

        token_total = max(token_total, 1.0)
        clip_frac = clipped_tokens / token_total
        ratio_dev = ratio_dev_sum / token_total
        step_loss = loss_sum / rows_trained if rows_trained else 0.0
        clip_by_substep[step_in_epoch].append(clip_frac)
        if padded_tokens:
            pad_history.append((real_tokens, padded_tokens))

        # ── Logging ──────────────────────────────────────────────────────────
        print(
            f"  Step {global_step:>4d} │ "
            f"reward {mini_epoch['mean_reward']:.3f} │ "
            f"acc {mini_epoch['mean_correct']:.2f} │ "
            f"len {mini_epoch['mean_length']:.0f}/{mini_epoch['max_length']:.0f} │ "
            f"trunc {mini_epoch['frac_truncated']:.0%} │ "
            f"live {live_groups}/{cfg.problems_per_step} │ "
            f"clip {clip_frac:.1%} │ "
            f"{step_time:.2f}s"
        )

        if use_wandb:
            wandb.log({
                "train/reward": mini_epoch["mean_reward"],
                "train/reward_correct": mini_epoch["mean_correct"],
                "train/reward_uses_method": mini_epoch["mean_uses_method"],
                "train/loss": step_loss,
                "train/clip_frac": clip_frac,
                "train/ratio_deviation": ratio_dev,
                "train/step_in_epoch": step_in_epoch,
                "train/live_groups": live_groups,
                # Compacted, live_groups/problems_per_step is ~1.0 by
                # construction and no longer measures saturation -- these two
                # do. They describe the pool, so they're constant across its
                # steps.
                "train/pool_live_frac": (
                    pool_live_history[-1] / cfg.problems_per_mini_epoch),
                "train/steps_per_pool": steps_per_pool_history[-1],
                "length/pad_frac": (1 - real_tokens / padded_tokens) if padded_tokens else 0.0,
                "length/tokens_forwarded": padded_tokens,
                "train/grad_norm": float(grad_norm),
                "train/lr": scheduler.get_last_lr()[0],

                "length/completion_mean": mini_epoch["mean_length"],
                "length/completion_max": mini_epoch["max_length"],
                "length/frac_truncated": mini_epoch["frac_truncated"],

                # Generation happens once per mini-epoch, not once per step.
                # Report the real cost on the step that follows it and zero on
                # the rest -- repeating the value would imply every step paid
                # it. The spikes still sum to the correct total over a run.
                "time/train_step": step_time,
                "time/generate": mini_epoch["sample_time"] if step_in_epoch == 0 else 0.0,
                "time/old_logps": mini_epoch["logps_time"] if step_in_epoch == 0 else 0.0,
            }, step=global_step)

        # ── Validation ───────────────────────────────────────────────────────
        if global_step % cfg.eval_every == 0:
            t_eval = time.perf_counter()
            sync_weights_to_vllm()
            result = evaluate(val_dataset)
            t_eval_total += time.perf_counter() - t_eval
            is_best = result.accuracy > best_accuracy
            if is_best:
                best_accuracy, best_step = result.accuracy, global_step
                write_val_markdown(result, global_step)

            marker = " ★" if is_best else f"  (best {best_accuracy:.1f}% @ {best_step})"
            print(
                f"  Step {global_step:>4d} │ val {result.accuracy:5.1f}%  "
                f"(len {result.mean_length:.0f}){marker}"
            )
            if use_wandb:
                wandb.log({"eval/accuracy": result.accuracy}, step=global_step)

    # The pool is spent. Push the updated weights into vLLM before regenerating.
    sync_weights_to_vllm()

print(f"\nTraining finished in {fmt_time(time.time() - t0)}")


# --------------------------------------------------------------------------
# 5.3. Test Set
# --------------------------------------------------------------------------

sync_weights_to_vllm()

test_id_result = evaluate(test_id_dataset)
test_ood_result = evaluate(test_ood_dataset)

rows = [
    ["Validation (best)", f"{best_accuracy:.1f}%", f"step {best_step}"],
    ["Test (in-distribution)", f"{test_id_result.accuracy:.1f}%",
     f"{test_id_result.num_correct}/{len(test_id_dataset)}"],
    ["Test (out-of-distribution)", f"{test_ood_result.accuracy:.1f}%",
     f"{test_ood_result.num_correct}/{len(test_ood_dataset)}"],
]
print(tabulate.tabulate(rows, headers=["Split", "Accuracy", ""], tablefmt="github"))
print("Reference (this config, H100 PCIe): val 89.0 @ 180 | ID 87.2 | OOD 86.2 | ~7.5 min")

if use_wandb:
    wandb.log({
        "test/accuracy_id": test_id_result.accuracy,
        "test/accuracy_ood": test_ood_result.accuracy,
    })

final_dir = run_dir / "final_checkpoint"
model.save_pretrained(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
print(f"Saved to {final_dir}")
print(f"Best-val markdowns in {run_dir} (step0_val.md vs step{best_step}_val.md)")


# ==========================================================================
# 6 - Performance
# ==========================================================================


# --------------------------------------------------------------------------
# 6.1. Saturated Groups & Padding
# --------------------------------------------------------------------------

live = np.array(live_history)               # live groups per optimizer step
pool_live = np.array(pool_live_history)     # live groups per generated pool
pool_steps = np.array(steps_per_pool_history)

# Two different fractions that say different things: `survival` is how much of
# a generated pool carried a gradient at all -- the saturation rate, and the
# compute the skip saves. `fill` is how full the average optimizer step was
# once those survivors were scheduled -- ~1.0, which is compaction working.
survival = pool_live.mean() / cfg.problems_per_mini_epoch
fill = live.mean() / cfg.problems_per_step if len(live) else 0.0

rows = [
    ["Groups per pool", cfg.problems_per_mini_epoch, "Generated per round"],
    ["Live groups per pool", f"{pool_live.mean():.2f}", f"{survival:.0%} survived"],
    ["Fwd/bwd passes skipped", f"{1 - survival:.0%}", "of the training compute"],
    ["Steps per pool", f"{pool_steps.mean():.2f}",
     f"of {cfg.steps_per_mini_epoch} slots"],
    ["Groups per step", f"{live.mean():.2f}" if len(live) else "n/a",
     f"{fill:.0%} of a full {cfg.problems_per_step}-group step"],
    ["Optimizer steps", len(live), ""],
    ["Generation rounds", cfg.generation_rounds,
     f"{cfg.generation_rounds * cfg.rollouts_per_mini_epoch} rollouts"],
]
print(tabulate.tabulate(rows, headers=["Item", "Value", ""], tablefmt="github"))

# What the training forwards actually chewed through. `real` is completion
# positions holding a token; `padded` is positions the model ran anyway.
if pad_history:
    real_tot = sum(r for r, _ in pad_history)
    pad_tot = sum(p for _, p in pad_history)
    rows = [
        ["Completion positions forwarded", f"{pad_tot:,}", ""],
        ["...carrying a real token", f"{real_tot:,}", f"{real_tot / pad_tot:.0%}"],
        ["...padding", f"{pad_tot - real_tot:,}",
         f"{1 - real_tot / pad_tot:.0%} of the training forward compute"],
        ["Mean micro-batch width", f"{pad_tot / len(pad_history):,.0f}",
         "completion positions per step"],
    ]
    print(tabulate.tabulate(rows, headers=["Item", "Value", ""], tablefmt="github"))


# --------------------------------------------------------------------------
# 6.2. Timing Breakdown
# --------------------------------------------------------------------------

total = time.time() - t0
accounted = t_sample_total + t_logps_total + t_train_total + t_eval_total

rows = [
    ["Generation (vLLM)", fmt_time(t_sample_total), f"{t_sample_total / total:.0%}"],
    ["pi_old log-probs", fmt_time(t_logps_total), f"{t_logps_total / total:.0%}"],
    ["Training steps", fmt_time(t_train_total), f"{t_train_total / total:.0%}"],
    ["Validation", fmt_time(t_eval_total), f"{t_eval_total / total:.0%}"],
    ["Unaccounted", fmt_time(total - accounted), f"{(total - accounted) / total:.0%}"],
    ["Total", fmt_time(total), ""],
]
print(tabulate.tabulate(rows, headers=["Phase", "Time", "Share"], tablefmt="github"))

# Clip rates by position within the mini-epoch: pi_theta drifts from pi_old as
# a pool's steps go by, so the later substeps clip more. If step 0 clips, the
# LR is too hot.
rows = [
    [i, f"{np.mean(fracs):.2%}" if fracs else "n/a"]
    for i, fracs in enumerate(clip_by_substep)
]
print(tabulate.tabulate(
    rows, headers=["Step within mini-epoch", "Mean clipped fraction"], tablefmt="github"
))

if use_wandb:
    wandb.finish()
