# Backward Pass Speedrun

To encourage more exploration/hacking of the backward pass, I'm running a single-GPU pre-training speedrun for the month of September. 

The metric is the time to 0.90 val bpb on an RTX Pro 6000 (a 96GB Blackwell GPU), current baseline is 27 minutes.

The baseline 12 layer model (same scale as [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and pipeline come from [nanochat](https://github.com/karpathy/nanochat).

Final validation score is reported in terms of the amount of slack versus the target. Current baseline slack is (0.900000 - 0.899590) x 1e6 = **410** points (or 410 µbpb, if you prefer). 

- From identical runs, the current "noise floor" is ±35 
- At current step time, 40 points is worth ~1 second.
- One training step appears to cost / gain ~70 points.

### Competition

The leaderboard won't be the traditional PR ladder. I'll do a weekly baseline on Mondays that folds in whatever changes I liked and managed to validate, and credit everyone involved in that. 

(Generally, though, you'll be on your own to promote your successes. What makes it into the baseline is going to be very subjective and limited by my capacity, so I apologize in advance to those who don't get the recognition they deserve.)

Final baseline will be on Monday, September 28th.

The baseline approach means that we just need to assemble enough improvements that we clearly have something better than the previous week, so there are no rigorous requirements here around verifying timing or loss. 

**Rules & Guidelines**

- The 'weekly folded-baseline' approach means sharing one run is enough.
- The script logs itself and the run to wandb, so just share your result and the link to your run in the discussion section and/or on X.
  - Avoid passing env variables or command line arguments in your final run.
- No autograd, obviously :)
- A few valuable tools we're going to set aside--they're awesome but messy:
  - Triton and custom ops
  - FP8 (maybe another time!)
- I'm going to avoid adopting changes that are difficult / inefficient to port into a decoding engine. 
  - I'll be adding an RL pipeline to serve as a sanity check for that requirement.


### How to run

Run it from within this folder as `python ./train_stack.py`.

If you'd like help with setup: Point your coding agent at `agent-setup-env.md` or go through the steps
yourself to set up a fresh GPU instance. 

It references an '~/env.sh' file--I keep one of those locally 
(not committed to any repos!) to make it easy to set up my environment
variables like GitHub and wandb credentials on a fresh instance.

Then run

```bash
micromamba activate stacks
cd ~/stacks/stacks/decoder-rtx/
python train_stack.py
```


### RL pipeline (GSM8K)

`train_gsm8k.py` is the decode-engine sanity check mentioned in the rules: it
takes the pretrained d12 and runs SFT on Qwen3-8B GSM8K teacher traces, then
on-policy RL (REINFORCE, per-problem mean baseline) with generation through a
CUDA-graph decode engine over the same live weights. Same style as
`train_stack.py`: handwritten forward/backward, no `nn`/autograd, config object,
no CLI.

- Downloads what it needs: `ChrisMcCormick/decoderstack-gsm8k` (prompts in the
  nanochat chat template + 50K teacher traces, ClimbMix 32k ids) and the step-1000
  baseline checkpoint from `ChrisMcCormick/decoderstack-d12`.
- The forward lives three times in the file -- packed training forward/backward,
  varlen prefill, and the one-token-per-row decode body over a paged KV cache.
  A change to the architecture in `train_stack.py` has to be carried into all
  three; the built-in engine check (teacher-forced decode CE vs the training
  forward's CE on real docs, asserted < 0.5 nats, measured ~0.01 mean) is what
  catches a mismatch.
- On the RTX PRO 6000: setup ~20 s (warm compile cache), an SFT step of 256K tokens
  1.07 s, an RL round of 512 rollouts at a 512-token budget ~2 s (1.3 s generation
  at ~90K tok/s + 0.7 s train), peak ~30 GB.

```bash
python train_gsm8k.py
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

### Baselines

*September 4th Baseline*

- [wandb run](https://wandb.ai/chrismccormick/decoderstack_rtx/runs/ykjt80he)
- 1,627 seconds (~27 minutes)
- 410 ubpb of slack


### Acknowledgements

- `modded-nanogpt` and Larry Dial in particular, who is a saint for running that competition.
  - All of the architecture features in the initial baseline here came from that project.
- Adrej Karpathy for his beautifully clean repos, and for nanochat here in particular. 
  - The baseline model architecture and hyperparameters come from 12-layer nanochat ("d12"), and the overall pipeline is just a refactoring.

TODO - I'd like to try and carry forward crediting the individual algorithm and architecture improvements leveraged here that were first introduced by modded-nanogpt contributors, and any research paper behind them.

### Interesting Techniques

- The baseline includes initialization of the vocabulary with the bigram distribution measured from the training data.
- Selectively re-computing certain parts of the forward activation conserved enough memory to allow for a micro-batch size of 256K tokens.

