# Baselines

The history of qwen-arithmetic, in the spirit of modded-nanogpt's `records/`
folder: one dated, named folder per baseline, holding a frozen copy of the code
that produced it plus everything it needed to run.

Both baselines were run on a 1x H100 PCIe. They ran the same task, the
same model and the same greedy eval on the same data budget — 69,632 rollouts —
so the rows compare like for like.

| Baseline | Date | Stack | val best | test ID | test OOD | wall (loop) |
|---|---|---|---|---|---|---|
| [`20260813_raw-h100`](20260813_raw-h100/) | 2026-08-13 | handwritten fwd/bwd + CUDA-graph decode, full-parameter | **94.0%** | **89.0%** | **86.75%** | **3.7 min** |
| [`20260813_hf-vllm`](20260813_hf-vllm/) | 2026-08-13 | HF `transformers` + in-process vLLM + LoRA | 89.0% | 87.2% | 86.2% | ~7.5 min |

Same date because both were frozen in the same commit — `e5062c1`, the one that
created this sub-repo. They are a progression in stack, not in time: the hf-vllm
run is the off-the-shelf ceiling, and the raw run is what replacing that stack
with a handwritten one bought.

The script at the sub-repo root is the **working copy** —
`train_qwen_arithmetic.py`, the Tesla T4 port, which is where the line is now.
When you need the code that produced a number in the table, take it from here,
not from there.

## Why the T4 script is not in this table

It targets different hardware, so its numbers do not belong in the same column.
It is also the only one of the three you can run for free. Its results — and the
four-run spread that says how to read them — are in the
[sub-repo README](../README.md).

## Reading the accuracy columns

`val best` is greedy K=1 on the 200-problem val split, at its best eval; test ID
and OOD are greedy K=1 at the end of the run. **`test_ood` is the same 400
problems as `test_id`**, at the same indices, re-worded from a disjoint template
set — so the ID/OOD gap measures sensitivity to phrasing with difficulty held
exactly fixed, not harder arithmetic.

These are single runs. The T4 line's four-run spread (val 84.0-92.0, test ID
79.0-86.5 on identical config and seed) is the closest thing to an error bar
this task has, and there is no reason the H100 line is tighter. Treat gaps under
a few points as unmeasured.

## Adding a baseline

1. `baselines/YYYYMMDD_name/` — the date is the commit that froze it, the name
   says what changed relative to the previous baseline.
2. Copy the training script **at that commit** and every file it imports, plus
   whatever prep is unique to it. Shared prep (`data/prepare_arithmetic.py`) and
   the shared env (`setup.sh`, `pyproject.toml`) stay at the sub-repo root — pin
   the commit in the baseline's README instead of duplicating them.
3. Write a `README.md`: the pinning commit, the files, the config, the measured
   result, and what distinguishes it from the last baseline.
4. Add a row to the table above.
