# 20260813_raw-h100

**The raw speedrun** — `qwen-gsm8k`'s implementation retargeted at arithmetic.
Handwritten forward/backward (no autograd, no `torch.optim`, no `nn.Module`) and
a CUDA-graph decode engine fused into the training process: one model instance,
in-place bf16 updates with fp32 masters, so the captured graphs never re-capture.

At 0.5B the decode step is **kernel-count bound, not bandwidth bound**, so the
engine is built around launch count — compiled decode bodies under CUDA graphs,
fused QKV and gate/up GEMMs, macro-window replays with ONE pinned D2H per window,
a coarse row-count bucket ladder, and a Gumbel-max sampler (one elementwise pass
plus argmax). KV is a single static allocation, each row permanently owning its
pages, with prefix sharing as COMPUTE only: every context prefills once (packed
varlen) and broadcasts into its K sibling rows.

| | |
|---|---|
| Frozen at | `e5062c1` — *"qwen-arithmetic: new speedrun folder — hf-vllm distillation + raw port"*, Thu 2026-08-13 |
| Run of record | `raw-v1-lr1e-6`, H100 PCIe |
| Supersedes | [`20260813_hf-vllm`](../20260813_hf-vllm/) |
| Ported to a free Colab T4 as | the sub-repo's working copy, [`../../train_qwen_arithmetic.py`](../../train_qwen_arithmetic.py) |

## The run

272 rounds of 16 problems x K=16 @ budget 640, full-parameter AdamW lr 1e-6,
temp 1.0 — 69,632 rollouts, the **same data budget** as the hf-vllm baseline's
68 x 1,024, so the accuracy-vs-wall-clock comparison is like for like.

| | |
|---|---|
| val best | **94.0%** @ round 270 |
| test ID | **89.0%** |
| test OOD | **86.75%** |
| wall (loop) | **3.7 min** (median round 0.61 s = 0.32 gen + 0.28 train) |
| peak memory | 28.7 GB |

**Val was still climbing at the horizon** — the budget, not the method, set the
ceiling. Read this as one sample, not a measurement: see
[the spread caveat](../README.md#reading-the-accuracy-columns).

## What it keeps from qwen-gsm8k, and what it does not

It keeps the **algorithm**, not the reference's: on-policy, ONE optimizer step
per round, no PPO ratio or clip, no KL, no reference model, token-mean loss,
full-parameter AdamW at constant lr.

- **sampler**: temperature 1.0, no top-k, no top-p → Gumbel-max over logits. (The
  hf-vllm baseline sampled at 0.8 with top-p 0.9; the raw sampler is deliberately
  sort-free, and temp 1.0 is the qwen-gsm8k/verl heritage.)
- **GRPO advantage**: per-group `(r - mean) / (std_{ddof=1} + 1e-6)`.
- **loss**: token-mean over ALL response tokens in the round.
- **reward**: `1.0 * correct + 1.0 * uses_method`, gated on correctness. Rewards
  are {0, 1, 2}-valued; the group z-score does not care.
- **lr 1e-6, UNSWEPT** for this task — the hf-vllm baseline trains LoRA at 1e-5,
  and full-parameter wants smaller. This is the first knob worth sweeping.

**One optimizer-machinery departure from qwen-gsm8k**: the fp32 masters start
MID-BIN (mantissa `0x8000`) instead of on the bf16 bin edge. The all-zero init
turns the run's first update into a ~2^-9-relative signSGD kick that this task
does not survive — **val 61.5% → 0.5% in 3 rounds, measured**. See
[`TECHNIQUES.md`](../../TECHNIQUES.md) § The mantissa first-step kick.

## Files

| | |
|---|---|
| `train_qwen_arithmetic.py` | the speedrun, verbatim from `e5062c1` |
| `data/prepare_model.py` | banks the HF checkpoint in **bf16** + tokenizer, into `~/.cache/qwen-arithmetic/data/`. Unique to this baseline — the working copy's `data/prepare_model.py` banks **fp16** for the T4 |

The shared env and dataset prep stay at the sub-repo root rather than being
duplicated here; take them at `e5062c1` for an exact reproduction:

```bash
bash setup.sh                                          # uv env + data/prepare_arithmetic.py
python baselines/20260813_raw-h100/data/prepare_model.py    # bf16 banks + tokenizer
python baselines/20260813_raw-h100/train_qwen_arithmetic.py
```

There is no command line and no config env: every knob is a field of
`ArithConfig` in § Config, edited in place. Useful modes: `host_test` (host-only
self-tests, no GPU), `fixed_problems` + `rounds_cap` (single-problem overfit
smoke), `eval_every = 0` + `final_eval = False` (train-only).

Env needs torch + `kernels` (FA3/FA2 via the HF kernels hub — no wheel builds),
tokenizers, safetensors, pyarrow, numpy, wandb. `transformers` must NOT be
installed: a stray model load can pull a flash-attn wheel that collides with the
kernels-hub FA build.

## Model

Qwen2.5-0.5B (`config.json`, asserted against the banks' sidecar at load):
24 layers, d_model 896, 14 Q heads / 2 KV heads, head_dim 64, MLP 4864 (SwiGLU),
vocab 151,936, rope_theta 1e6, rms eps 1e-6, QKV biases (o/mlp none), TIED
embeddings (embed table == lm_head), no qk-norm.
