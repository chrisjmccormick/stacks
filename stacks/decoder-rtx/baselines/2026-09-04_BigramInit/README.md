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
