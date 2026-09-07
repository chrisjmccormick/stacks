# RL pipeline (GSM8K)

`train_gsm8k.py` is the decode-engine sanity check mentioned in the rules of the
[pretraining speedrun](../README.md): it takes the pretrained d12 and runs SFT on
Qwen3-8B GSM8K teacher traces, then on-policy RL (REINFORCE, per-problem mean
baseline) with generation through a CUDA-graph decode engine over the same live
weights. Same style as [`../train_stack.py`](../train_stack.py): handwritten
forward/backward, no `nn`/autograd, config object, no CLI.

- Downloads what it needs: `ChrisMcCormick/decoderstack-gsm8k` (prompts in the
  nanochat chat template + 50K teacher traces, ClimbMix 32k ids) and the step-1000
  baseline checkpoint from `ChrisMcCormick/decoderstack-d12`.
- The forward lives three times in the file -- packed training forward/backward,
  varlen prefill, and the one-token-per-row decode body over a paged KV cache.
  A change to the architecture in `../train_stack.py` has to be carried into all
  three; the built-in engine check (teacher-forced decode CE vs the training
  forward's CE on real docs, asserted < 0.5 nats, measured ~0.01 mean) is what
  catches a mismatch.
- On the RTX PRO 6000: setup ~20 s (warm compile cache), an SFT step of 256K tokens
  1.07 s, an RL round of 512 rollouts at a 512-token budget ~2 s (1.3 s generation
  at ~90K tok/s + 0.7 s train), peak ~30 GB.

It shares the parent directory's `utils.py`, downloaded `data/`, and `logs/` tree
rather than keeping its own, so it puts `..` on the import path and cds there at
startup -- run it from anywhere:

```bash
python rl/train_gsm8k.py
```

*First run (September 4th, the step-1000 baseline weights):*

| stage | GSM8K test mean@8 | pass@8 | `#### n` format | phase time |
|---|---|---|---|---|
| base d12 (step 1000) | 0 | 0 | 0% | |
| + SFT, 2 epochs of the teacher traces (109 steps) | 4.08 | 20.55 | 98.4% | 1.9 min |
| + RL, 1 epoch (233 rounds x 32 problems x K=16) | 4.24 | 19.41 | 85.5% | 7.7 min |

11.8 minutes wall including the evals
([wandb run](https://wandb.ai/chrismccormick/decoderstack_rtx_gsm8k_dev/runs/y3oqsnpm)).
The RL phase is functional but not yet a gain: train-set solve climbs 33% -> 45%
(the SFT memorized those problems) while the test number holds and the format
rate erodes, the same channel the nanochat fix-ladder found. Tuning that recipe
is the open (secondary) objective; the primary one is that the whole pipeline
keeps working against the weekly baseline.
