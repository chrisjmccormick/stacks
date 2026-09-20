The starting point, mostly identical to nanochat-d12.

Compared to nanochat-d12:
- Uses FlashAttention varlen
- Includes bigram initialization of the embedding matrices
    - This also allowed removing the 40 step LR warmup.
- Doesn't include pad masking in the LM head.
- Recomputes some forward activations to save memory.
    - Runs with a 256K micro-batch size as a result.
- Targets a val bpb of `0.9` and runs 1000 steps vs. nanochat d12's 1680.

wandb run: https://wandb.ai/chrismccormick/decoderstack_rtx/runs/ykjt80he

### Logging brought forward (2026-09-19)

The original run (`ykjt80he`) predates the September 7th baseline's `StepStats` /
`RunResult` logging, so it carries none of the series the later baselines are plotted
on. `train_stack.py` here now writes those same series -- `val/bpb`, `val/slack`,
`time/train_total`, the `train/` panel and the `final/` summary, plus the metrics CSV
and result JSON -- so a re-run lands on the same panels as Baselines 2 and 3.

Only the logging changed: no weight, schedule, optimizer or RNG draw is touched, and
the recipe still finishes at the 0.899590 / 410 points of slack the README quotes.
`utils.py` pins `get_kernel(..., version=1)`, the flash-attn2 build the hub still
serves (the classic model-type repos were retired on 2026-09-14).

Produced by `agent-ops-stacks/decoder-rtx/2026-09-19_0514pm_baseline1-relog/`.
