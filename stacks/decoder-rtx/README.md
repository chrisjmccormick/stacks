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

