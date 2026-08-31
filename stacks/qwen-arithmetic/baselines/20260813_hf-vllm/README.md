# 20260813_hf-vllm

**The off-the-shelf-stack ceiling** — what the standard tooling reaches on this
task, so the handwritten line has something honest to beat. HF `transformers`
for the training forward/backward, in-process **vLLM** for generation, **LoRA**
via peft.

Distilled from the grpo-course experiment line (`exp/train_h100.py`) with every
losing arm removed, so the defaults here are measured winners rather than
plausible settings.

| | |
|---|---|
| Frozen at | `e5062c1` — *"qwen-arithmetic: new speedrun folder — hf-vllm distillation + raw port"*, Thu 2026-08-13 |
| Run of record | `overlong-none`, wandb `grpo-arithmetic`, 2026-08-14, H100 PCIe |
| Superseded by | [`20260813_raw-h100`](../20260813_raw-h100/) — 2x faster at +1.8 test ID |

## The run

68 generation rounds of 64 problems x K=16 @ budget 640, LoRA lr 1e-5, temp 0.8
with top-p 0.9 — 256 optimizer steps, 69,632 rollouts.

| | |
|---|---|
| val best | **89.0%** @ step 180 |
| test ID | **87.2%** |
| test OOD | **86.2%** |
| wall (loop) | ~7.5 min |

## Files

| | |
|---|---|
| `train_qwen_arithmetic-hf-vllm.py` | the stack, verbatim from `e5062c1` |
| `setup_hf_vllm.sh` | builds `.venv-hf-vllm` — a SEPARATE env from the sub-repo root's `.venv`, which forbids `transformers` |
| `requirements-hf-vllm.txt` | the pins |

```bash
bash setup_hf_vllm.sh
source .venv-hf-vllm/bin/activate
python train_qwen_arithmetic-hf-vllm.py    # downloads its own data on first run
```

No prep step and no dependency on the sub-repo root's `setup.sh` — this script
fetches the dataset and the checkpoint itself. `vllm==0.19.0` is the last CUDA-12
build and pins `torch==2.10.0`; `torchao>=0.16.0` is required because peft
refuses to build LoRA against older.

There is no command line and no config env: every knob is a field of `Config` in
§ 2, edited in place.

## The baked-in winners

Each was measured against its alternative rather than assumed — this is the part
worth reading even if you never run the script.

- **Compaction.** Live groups repacked into as few, full optimizer steps as will
  hold them. Versus fixed slices: same gradient work, ~2.4x fuller steps, and no
  zero-live steps burning schedule.
- **Length bucketing.** Micro-batches sorted by completion length and padded to
  their *own* max — gradient-identical, and it removes ~40% of forwarded padding.
- **Overlong handling: NONE.** DAPO's mask scored 76.5 / 72.8 against none's
  87.2 / 86.2 at the same 68-round budget. The length collapse it targets happens
  with truncation already at 0-1%, and on 2-operand arithmetic the rambling that
  masking permits costs accuracy.
- **Rewards: correct (1.0) + uses_method (1.0, gated on correct).** `has_words`
  is deleted — once the method reward exists it is a flat +0.5 on nearly every
  rollout, so it is constant within a group, and GRPO's group-mean subtraction
  erases it while it dilutes the two signals that do vary.
- **Attention: sdpa.** FA3 measured within noise (val 83.5 vs 82.5, test
  identical) — the training forward is not where this run's time goes.
- **Budget 640**, sized to clear the *untrained* model's p99 (~512). At 256, ~20%
  of early rollouts truncate, and a truncated rollout scores 0 — which teaches
  "shorter is safer" exactly while the model is still learning what to write.
- **lr 1e-5**, swept: 1e-4 → 79.0%, 3e-5 → 80.0%, 1e-5 → 83.0% val.
- **Compiled log-prob tail**: 184 → 123 ms on the full fwd+bwd.

## Provenance

The experiment line this was distilled from lives in
`agent-ops/grpo-arithmetic/` — in particular
`2026-08-13_0259pm_h100-speedrun-demo/` (the H100 rehearsal and retune) and
`2026-08-14_1201pm_difficulty-balancing/`.
