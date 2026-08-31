# Baselines

Measured runs of DecoderStack-medium. Not a leaderboard, more like a history of variants.

| Baseline | Summary | Result |
|---|---|---|
| [`20260801_manual-fwd-bwd`](20260801_manual-fwd-bwd/) | Handwritten forward/backward and written-out optimizer — no autograd, no `torch.optim`, no `nn.Module`. SFT + generative eval dropped. | 0.719042 val bpb · CORE 0.2517 · 110.7 min on 8×H100 · [weights](https://huggingface.co/ChrisMcCormick/decoderstack-d24) |
| [`20260315_pt-sft-gen`](20260315_pt-sft-gen/) | A single file refactor of nanochat d24. Includes SFT and generative eval | never did a full run, only measured step times |
