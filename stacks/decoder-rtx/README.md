# Backward Pass Speedrun

To encourage more exploration/hacking of the backward pass, I ran a single-GPU pre-training speedrun for the month of September 2026. 

The metric is the time to 0.90 val bpb on an RTX Pro 6000 (a 96GB Blackwell GPU). The final baseline, Baseline 6, trains in 12m 21s, down from 26m 51s.

Baseline 1's 12 layer model (same scale as [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and pipeline came from [nanochat](https://github.com/karpathy/nanochat).

Final validation score is reported in terms of the amount of slack versus the target. Baseline 1's slack was (0.900000 - 0.899729) x 1e6 = **271** points (or 271 µbpb, if you prefer). 

You can roughly estimate a conversion rate between slack and steps, and therefore time.
In the original baseline:
- 40 points was worth ~1 second.
- 1 training step was worth ~70 points. 

### Competition

For the leaderboard, instead of the traditional PR ladder, weekly baselines (on Mondays) fold in whatever changes I liked and managed to validate, and credit everyone involved in that. 

(Generally, though, you'll be on your own to promote your successes. What makes it into the baseline is going to be subjective and limited by my capacity, so I apologize in advance to those who don't get the recognition they deserve.)

The final baseline landed on Wednesday, September 30th (Baseline 6), in place of the planned Monday the 28th.

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
  - I had planned an RL pipeline as a sanity check for that requirement; it didn't land before the challenge ended.


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


### Baselines

| Name | Time | Δ from Baseline 1 | Slack (μbpb) | Write-up |
| --- | --- | --- | --- | --- |
| 2026-09-04 - Baseline 1 | 26m 51s | - | 271 | [#1](https://github.com/chrisjmccormick/stacks/pull/1) |
| 2026-09-07 - Baseline 2 | 25m 56s | -0m 55s | 837 | [#3](https://github.com/chrisjmccormick/stacks/pull/3) |
| 2026-09-14 - Baseline 3 | 24m 21s | -2m 30s | 837 | [#4](https://github.com/chrisjmccormick/stacks/pull/4) |
| 2026-09-21 - Baseline 4 | 15m 47s | -11m 04s | 920 | [#5](https://github.com/chrisjmccormick/stacks/pull/5) |
| 2026-09-28 - Baseline 5 | 14m 52s | -11m 59s | 1,531 | [#6](https://github.com/chrisjmccormick/stacks/pull/6) |
| 2026-09-30 - Baseline 6 | 12m 21s | -14m 30s | 917 | [#7](https://github.com/chrisjmccormick/stacks/pull/7) |

Each baseline's PR has the write-up of what it changed. The baseline runs can all be viewed on wandb, [here](https://wandb.ai/chrismccormick/decoderstack_rtx).

Baseline 6 switches to priml's 16K Unigram tokenizer, scored on the same validation documents as Baselines 1-5. Its tokens cover about 5.5% fewer bytes each, so compare step and token counts only within Baselines 1-5. The leaderboard also has "2026-09-30 - Baseline Priml" (13m 05s, slack 651): priml's recipe as first ported, before Baseline 6's RoPE changes.

### Acknowledgements

- `modded-nanogpt` and Larry Dial in particular, who is a saint for running that competition.
  - All of the architecture features in the initial baseline here came from that project.
- Adrej Karpathy for his beautifully clean repos, and for nanochat here in particular. 
  - The baseline model architecture and hyperparameters come from 12-layer nanochat ("d12"), and the overall pipeline is just a refactoring.

TODO - I'd like to try and carry forward crediting the individual algorithm and architecture improvements leveraged here that were first introduced by modded-nanogpt contributors, and any research paper behind them.

- The September 21st and 28th baselines include several innovations from the priml autoresearch run ([(auto)²-research](https://rekursiv.ai/blog/autoautoresearch/)), and Baseline 6 adopts its published `exp022` recipe whole ([#7](https://github.com/chrisjmccormick/stacks/pull/7)).

### Interesting Techniques

- A handwritten backward for the whole model, priml's recipe included: no autograd.
- Priml's per-layer hashed n-gram value memories (bigram on layers 1, 3, 5, 7; trigram on 1, 5, 7), 14 slots of 2,097,152 rows x 384, 22.5GB in bf16. They have no `.grad` buffers: their sparse RMSProp step is inlined into the backward, on the rows the batch touched. Skipping those buffers is what fits them on the 96GB card; priml's own layout adds ~45GB of fp32 gradient buffers for them. See [#7](https://github.com/chrisjmccormick/stacks/pull/7).
- Selectively re-computing parts of the forward activations (the `LayerStash` table in `train_stack.py`) keeps each 192K-token step a single micro-batch.
- RoPE written as `x * [cos, cos] + roll(x, 64) * [sin, -sin]`, so the per-head QK norm and the rotation fuse into one kernel, and nothing holds an fp32 copy of the rotated q and k for the backward. See [#7](https://github.com/chrisjmccormick/stacks/pull/7).
- Earlier baselines' own techniques (the bigram-distribution initialization, the residual bigram table, the frozen trigram "pair code", the document-length curriculum) are written up in their PRs. Baseline 6 set them aside for priml's recipe.
