# Backward Pass Speedrun

To encourage more exploration/hacking of the backward pass, I'm running a single-GPU pre-training speedrun for the month of September. 

The metric is the time to 0.90 val bpb on an RTX Pro 6000 (a 96GB Blackwell GPU), current baseline is 16 minutes.

The baseline 12 layer model (same scale as [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) and pipeline come from [nanochat](https://github.com/karpathy/nanochat).

Final validation score is reported in terms of the amount of slack versus the target. The original baseline slack was (0.900000 - 0.899590) x 1e6 = **410** points (or 410 µbpb, if you prefer). 

You can roughly estimate a conversion rate between slack and steps, and therefore time.
In the original baseline:
- 40 points was worth ~1 second.
- 1 training step was worth ~70 points. 

### Competition

For the leaderboard, instead of the traditional PR ladder, weekly baselines (on Mondays) fold in whatever changes I liked and managed to validate, and credit everyone involved in that. 

(Generally, though, you'll be on your own to promote your successes. What makes it into the baseline is going to be subjective and limited by my capacity, so I apologize in advance to those who don't get the recognition they deserve.)

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


### Baselines

| Name | Time | Δ from Baseline 1 | Slack (μbpb) | Write-up |
| --- | --- | --- | --- | --- |
| 2026-09-04 - Baseline 1 | 26m 51s | - | 271 | [#1](https://github.com/chrisjmccormick/stacks/pull/1) |
| 2026-09-07 - Baseline 2 | 25m 56s | -0m 55s | 837 | [#3](https://github.com/chrisjmccormick/stacks/pull/3) |
| 2026-09-14 - Baseline 3 | 24m 21s | -2m 30s | 837 | [#4](https://github.com/chrisjmccormick/stacks/pull/4) |
| 2026-09-21 - Baseline 4 | 15m 47s | -11m 04s | 920 | [#5](https://github.com/chrisjmccormick/stacks/pull/5) |

Each baseline's PR has the write-up of what it changed. The baseline runs can all be viewed on wandb, [here](https://wandb.ai/chrismccormick/decoderstack_rtx).

### Acknowledgements

- `modded-nanogpt` and Larry Dial in particular, who is a saint for running that competition.
  - All of the architecture features in the initial baseline here came from that project.
- Adrej Karpathy for his beautifully clean repos, and for nanochat here in particular. 
  - The baseline model architecture and hyperparameters come from 12-layer nanochat ("d12"), and the overall pipeline is just a refactoring.

TODO - I'd like to try and carry forward crediting the individual algorithm and architecture improvements leveraged here that were first introduced by modded-nanogpt contributors, and any research paper behind them.

- The September 21st baseline includes several innovations from the priml autoresearch run.

### Interesting Techniques

- Initialization of the vocabulary with the bigram distribution measured from the training data.
- Selectively re-computing certain parts of the forward activation conserved enough memory to allow for a micro-batch size of 256K tokens.
- Two huge bigram hash tables (2M rows added to x0, 1M rows added to the values), from rekursive.ai's autoresearch run. They have no `.grad` buffers: their optimizer (sparse, no first moment, one second moment per row) is inlined in their backward math. See [#5](https://github.com/chrisjmccormick/stacks/pull/5).
- A frozen 2M-row trigram ("pair code") stats table tallied from the dataset, projected and added to the values. See [#5](https://github.com/chrisjmccormick/stacks/pull/5).
