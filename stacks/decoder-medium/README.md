# DecoderStack-medium

`decoderstack_medium_pt.py` is the nanochat pre-training pipeline, hardcoded for the "d24" configuration (24-layers). Timed on an 8xH100, but supports Ampere as well.

Some cool features:
- Model code is a single handwritten forward_backward function.
    - No nn.Module, no Autograd, no torch.optim.
- Each model parameter carries its own everything--weights, gradients, 
  optimizer state, and learning schedules.
- Flat / linear coding style (minimal use of classes and helpers).
- Two primary global objects: the config and the model, make it easy
  to access values without defining function arguments.

Notable differences from nanochat:
- Vocabulary is fixed and dataset is downloaded pre-tokenized.
- Uses variable length attention (FlashAttention `varlen`).

To keep things simpler, I'm omitting / disallowing **triton kernels**
and **FP8**. Both are very cool and make a big difference, but lead to some
pretty nasty code. 

(I do have an FP8 variant that I intend to share as a one-off, since it's 
interesting to see its handwritten `forward_backward`)

**Colab Notebook**

I've also included a 12-layer configuration for running on a 40GB A100
in Colab, roughly 80 minutes end-to-end (include the CORE eval).

The notebook outline and cell divisions make it a nice way to interact
with the codebase if you're exploring it.

The 40GB A100s are great because they're only $0.53/hr. Just make sure to
unselect "high ram" when choosing the GPU, or you'll get a more expensive
80GB machine.

**Submissions**

I won't be accepting PRs for records or maintaining a formal leaderboard; but 
share the link to your wandb run on X/Twitter and tag me `@ChrisJMcCormick`, 
I'd love to see it. (The code attaches itself to the wandb run)

**How to run**

Point your coding agent at `agent-setup-env.md` or go through the steps
yourself. 

It references an '~/env.sh' file--I keep one of those locally 
(not committed to any repos!) to make it easy to set up my environment
variables like GitHub and wandb credentials on a fresh instance.

Run the 24-layer model from inside this directory with `./run_medium.sh`. 
Set the torchrun variable inside there to the number of GPUs you're running.

**Baselines**

The bpb and CORE scores need some error bars that I don't have yet. CORE has
pretty high variance; bpb is more consistent.

Varlen makes comparison to standard `nanochat` difficult, because it changes
truncation boundaries. e.g., for a 64K token batch, varlen truncates one doc,
while batched attention truncates 32. That means val bpb loss isn't measured
on the same tokens.

The most recent baseline I've run:

[`20260801_manual-fwd-bwd`](baselines/20260801_manual-fwd-bwd/) 

| min val bpb | 0.719042 |
| CORE | 0.2517 |
| train time | 110.7 min (1,212 ms/step, 864,856 tok/sec) |
| weights | [`ChrisMcCormick/decoderstack-d24`](https://huggingface.co/ChrisMcCormick/decoderstack-d24) |

Note that the current implementation (in `decoderstack_medium_pt.py`) doesn't have a timed 8xH100 run yet.
It includes a major refactoring / clean-up of the codebase, and I've only evaluated it on smaller setups 
so far.

**Current Enhancements**

- Managing the "stashed" forward activations allowed for bumping the
  micro-batch size from 32K tokens per device to 64K.
- Ramping AdamW's (1-beta) coefficients (which control how much of the gradient
  is added to the buffer in a step) tamed the gradient swings of the scalars
  in early steps. 

Ideas to add:

- Bigram embeddings; these have already been shown to work well at d12 and d24.
- The Muon+ changes in nanochat haven't made it in yet.
- There are a number of nanochat record PRs that weren't adopted.
- Your cool ideas.
