# 20260315_pt-sft-gen

The original DecoderStack-medium: a near-replica of nanochat's d24 built on
`nn.Module` + autograd + `torch.optim`, and the last version of the medium track
that still carried **SFT and generative evaluation**. Both were dropped in the
rewrite, which is what the name records.

| | |
|---|---|
| Frozen at | `04cb7d1` — *"Clean up and comment the batch generator"*, Sun 2026-03-15 |
| Retired by | `e206f73` (2026-08-01), superseded by [`20260801_manual-fwd-bwd`](../20260801_manual-fwd-bwd/) |
| Full-horizon run | **none** — `num_iterations` was still at the 50-step smoke value |
| Weights | none published |

## Files

| | |
|---|---|
| `decoderstack_medium_pt-sft.py` | the stack — pre-training, CORE eval, SFT, generative eval |
| `triton_kernels_medium.py` | Polar Express kernels (`XTX`, `XXT`, `ba_plus_cAA`), retuned for 1536-dim |
| `generation_medium.py` | KV-cache model + the generation-based benchmark harness |
| `run_medium.sh` | `torchrun --standalone --nproc_per_node=1 …` |

The `data/` and `utils/` folders at the sub-repo root are the ones this baseline
was written against and are not duplicated here.

## Configuration

d24, hardcoded: `n_layer=24`, `n_head=n_kv_head=12`, `n_embd=1536`,
`sequence_len=2048`, `window_pattern="SSSL"`, vocab 32,768.

* `total_batch_size = 2**20` tokens, `grad_accum_steps = 2**20 / (16 · 2048 · world_size)`
* `batch_lr_scale = sqrt(B / 2**19)` ≈ 1.4142; `weight_decay = 0.06`
  (nanochat's 0.28 scaled for d24 via its T_epoch framework)
* Dataset `climbmix_32k_8_170`, `val_tokens = 10,485,760`
* Bigram hash embeddings, `bigram_vocab_size = 5 × 32,768`
* `assert 8 % world_size == 0` — grad accum absorbs the difference, so the
  effective batch is world-size-invariant

## What was measured

No full-horizon run was ever made from this code, so there is no val bpb or CORE
number to report. The only recorded throughput is from the Polar Express kernel
tuning two days earlier (`cd4b8ba`, 2026-03-13), on a 50-step run:

| | |
|---|---|
| step_avg | 2126.29 ms (from 2131.86 ms pre-tuning, −0.26%) |
| MFU | ~39.6% |

Details, plus the abandoned fused-ReLU²-MLP investigation for this track, are in
[`dev/LOG.md`](../../../../dev/LOG.md) at the repo root.

## Distinguishing features vs. the current stack

* `nn.Module`, autograd, `torch.optim`-style optimizer objects, `state_dict`
* `torch.compile(model, dynamic=False)` over the whole model
* SFT training + generative eval (MMLU / ARC / chat) via `generation_medium.py`
* Hand-tuned Triton Polar Express kernels rather than the written-out optimizer
