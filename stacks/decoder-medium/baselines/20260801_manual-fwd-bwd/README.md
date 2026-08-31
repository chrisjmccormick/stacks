# 20260801_manual-fwd-bwd

The current baseline, and the code that produced the public d24 weights. A
single-file pipeline with a **handwritten forward/backward and a written-out
optimizer**: no autograd, no `torch.optim`, no param groups, no `nn.Module`. The
model is a plain class used as a namespace of plain `torch.Tensor`s.

Pre-training + CORE eval only — SFT and generative eval were dropped relative to
[`20260315_pt-sft-gen`](../20260315_pt-sft-gen/).

The file here keeps its historical name, `decoderstack_medium_pt-sft.py`, because
that is what shipped — the `pt-sft` was inherited and already inaccurate by then.
The working copy has since been renamed to `decoderstack_medium_pt.py`.

| | |
|---|---|
| Frozen at | `7a9998a` — *"Add the d24 training stack that produced the public hub weights"*, Sat 2026-08-01 |
| Weights | [`ChrisMcCormick/decoderstack-d24`](https://huggingface.co/ChrisMcCormick/decoderstack-d24) |

> **Why this snapshot and not the working copy.** The live file at the sub-repo
> root has moved on — `7f31caa` renamed variables and reworked comments, and
> revision is ongoing. Those changes are **no longer numerically identical** to
> the run below, so the baseline is pinned to `7a9998a`, the commit that
> published the weights.

## Files

| | |
|---|---|
| `decoderstack_medium_pt-sft.py` | the stack, verbatim from `7a9998a` |
| `scaling.py` | recomputes and documents every hardcoded derived constant (param counts, flops/token, horizon, LR/WD batch corrections, `cu_seqlens` sizing). Pure python — `python scaling.py` |
| `run_medium.sh` | `torchrun --standalone --nproc_per_node=1 …` |

## The run

5,568 steps × 2²⁰ tokens = **5.84B tokens** (data:param ratio 8), bf16, varlen
FlashAttention, on an 8×H100.

| | |
|---|---|
| parameters | 1,384,122,122 |
| min val bpb | **0.719042** |
| CORE | 0.2517 |
| train time | 110.7 min (1,212 ms/step, 864,856 tok/sec) |

The HF repo carries both capture points (step 5568, and 1950 = the last uncooled
state), the full training log, the exact training script, and the tokenizer.
`utils/convert_ckpt_to_nanochat.py` converts a capture into a nanochat
checkpoint.

### Read those two numbers with the right error bars

Karpathy's own re-runs of an identical d24 config span **0.0153 in CORE** (5
runs, mean 0.261), so 0.2517 sits inside the noise band of his and any CORE gap
under ~0.015 is measuring the seed. Val bpb is the metric with the resolution to
rank runs, but ours is computed over 10.5M eval tokens against his 40M.

**Our val bpb is NOT comparable to nanochat's.** His d24 ClimbMix run bottoms out
near 0.7151; that is not a 0.004 gap — the two numbers come from different
measurements (nanochat's best-fit cropping discards ~35% of tokens vs. our 11–12%
`seq_len` truncation, and nanochat measures loss with cross-document attention
while our varlen path isolates documents). The direction of the bias is not
obvious, which is the point. Full write-up in the repo-root
[`README.md`](../../../../README.md) and
[`models/nanochat/reference/METRICS.md`](../../../../models/nanochat/reference/METRICS.md).

## Design notes

* Every tensor created directly on device at its final dtype; dtypes hardcoded,
  never inferred by matching another tensor
* Globals throughout (`cfg`, `m`) rather than passing state around
* Hardcoded to d24 — none of nanochat's auto-scaling by model size
* Multi-GPU shards the **optimizer**, not the model (nanochat's scheme): every
  rank holds full bf16 live weights and full grad accumulators; optimizer state
  is allocated at shard sizes, and `optimizer_step` wraps the update kernels in
  reduce-scatter → owned-shard update → live all-gather
* FP8 lives in a separate file

The model/training math came from the nanochat repo, branch `fwd-bwd`
(`nanochat/train_step.py`, `nanochat/gpt.py`) — that branch's d24 run is the
reference implementation this matches; the rewrite dropped baggage without
changing the math. Everything below the marked seam came from this repo
(pre-tokenized data + distributed loader, CORE eval).

## Provenance

* `agent-ops/stacks/2026-07-31_0822am_fable-rewrite-handoff/` — the rewrite, plus `scaling.py`
* `agent-ops/stacks/2026-07-31_0627pm_8xh100-d24-baseline-world8/` — the run + `baseline_report.md`
* `agent-ops/stacks/2026-08-01_0647am_d24-nanochat-converter/` — the release + converter
* `agent-ops/stacks/2026-08-01_1241pm_d24-val-bpb-recalibration/` — the comparability analysis
