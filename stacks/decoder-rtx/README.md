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
- The head's log-frequency (unigram) component is halved at init. The first ten steps of training did that on their own, and doing it up front removes the step-1 loss spike (6.05 -> 7.6 nats) and finished 270 ubpb ahead of the same run without it.
- The document-prefix cap ramp is rounded to multiples of 64 tokens: FA2 varlen's cost steps up each time a document crosses a 64-token boundary.
- Selectively re-computing certain parts of the forward activation conserved enough memory to allow for a micro-batch size of 256K tokens.
- A frozen "pair code" read into the attention values. `tokenizer/pair_next_tokens.npz` holds, for the 2.1M most frequent [prev, curr] token pairs of the training data (88% of positions), the 64 next tokens the pair makes most likely relative to the bigram: `log1p(count / (1,000 * bigram probability))`, counted over all 91 train shards. At init each pair's log-ratios are summed through the head's rows and whitened into a fixed 768-d vector, and one zero-initialized matrix learns to read it into the values of the five value-embedding layers. It is never trained, so it carries no gradient or optimizer state, and the pair is looked up on the host by the data loader. It finished 833 ubpb ahead of the same run without it, for +0.8% step time. Read straight into the residual stream instead, the same statistics gave a large early lead that was gone by step ~500.

- **The optimizer's gradient plumbing is gone from the step.** There is one micro-batch per step (`micro_batch_tokens == total_batch_size`), so every gradient buffer is written exactly once, read once and zeroed: the writes are `copy_` rather than `add_` and the zeroing pass does not exist. The two row-wise RMSProp tables (`bigram_embeds`, `pair_values`) keep no first momentum, so nothing about their gradient has to outlive the point where `embedding_dense_backward` produces it -- their update is inlined there and their 10.5 GiB of fp32 gradient buffer is deleted outright. They also touch only the rows the micro-batch looked up: the ids are sorted so the distinct ones can be numbered, the gradient accumulates over at most `micro_batch_tokens` rows instead of the table's 524,288 or 1,048,576, the `(rows, 1)` second moment is still decayed densely because that costs 2 MB, and the weight decay rides along as a running scalar (the live weight is the stored row times it) so an untouched row needs no write at all. Together: -7.7% of step time and -10.5 GB of peak memory, and the tables' cost is now proportional to tokens looked up rather than rows allocated.
  - The trap worth knowing: **the fused write has to go to a whole tensor.** Writing a slot through `pair_table[j]`, a `select()` on a mutated graph input, lowers to a scatter over the whole table -- 21.5 ms in a single kernel. The five slots are held as standalone tensors (`list(pair_values.view(...).unbind(0))`) instead.

- **The MLP's input norm is stashed, not recomputed.** `x_attn_out` was held for the backward and the backward's only use of it was to rebuild `x_attn_out_hat` and its `1/rms` -- and the hat is the same shape and dtype, so holding the hat instead costs the same bytes and removes the recompute: one reduction pass over a `(T, 768)` tensor and one write of the same size, per layer. -0.5% of step time, for +2.7 GB of peak (in the forward the hat now lives to the end of the layer where it used to be freed after the `W_in` matmul, so it overlaps `x_attn_out` rather than replacing it). `x_biased` already worked this way.

- **The MLP stashes `relu(z)^2`, not `relu(z)`.** The square is the second matmul's input, so the forward writes it either way; stashing that instead means the forward never writes `relu(z)` as well and the backward never rebuilds the square from it -- about 2.3 GiB per layer. `relu(z)` is then only needed inside `2*relu(z)*mlpb_a`, where `sqrt(mlp_a)` stands in for it and fuses into that kernel for free. -1.5% of step time for a stash of the same size.


