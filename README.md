# stacks
Self-contained training stacks

This repo might best be described as "modded-nanogpt wrapped in flattened nanochat infrastructure".

I considered calling it "modded-nanochat", but:
1. I like referring to modded-nanogpt affectionately as just "modded", so I don't want the name collision.
2. I think a project with that name should preserve the design and ethos of modded-nanogpt, but I'd like the freedom to diverge.

Instead, I'm going with DecoderStack--alluding both to it being a stack of transformer decoder layers, and an end-to-end training and evaluation stack. 
(It also sets us up nicely for EncoderStack, DiffusionStack, ...)

Things I pulled in that I love from `modded`:

* Self-contained / single file code.   
* Pre-tokenized data.  
* Different code for different model sizes.  
* More leeway for complicated ideas.

What I’m pillaging from nanochat:

* Customizable vocabulary and tokenizer 
* bits-per-byte for validation 
* Efficient inference via KV cache
* Evaluation harness (this is especially significant)
* Better logging and reporting  
* The RL and tool-use based GSM8K benchmark

Clearly, none of this would be possible without all of the incredible engineering work of Andrej Karpathy. This repo is starting out as just a refactoring of his work executed primarily by Claude.

I do plan to contribute more than just telling Claude to blend modded + nanochat.

I'm hoping to place a greater emphasis on accessibility, with:
* A single GPU speedrun that promotes collaboration
* As many tutorials as I can manage
* Include Colab Notebook implementations--they're easy to read and run, and a great entry point
* Make Linux-expertise less of a pre-req

# Contents

Here's what exists currently:

* `decoderstack_small_pt-eval-sft.py` - Training and evaluating the DecoderStack is the main focus of this repo.
* `baselines/` - Similar to modded's `records/` folder, this will hold the history of improvements to the DecoderStack.
* `data/*.py` - Train a tokenizer then pre-process + pre-tokenize all of the nanochat datasets.
* `models/` - One-off implementations of other projects and architectures, wrapped in the DecoderStack infrastructure.
    * `modded/train_gpt.py` - modded-nanogpt with minimal changes to compare performance on nanochat's benchmarks.

## DecoderStack

Started out as: flatten nanochat to a single file then swap in all of modded-nanogpt's optimizations.

The optimizer, "scheduler", dataloader, model code, and triton kernels all come straight from modded-nanogpt.

The training loop comes primarily from modded, but I've integrated the nicer printouts, the bpb metric, and wandb logging from nanochat.

Here's what currently sets the DecoderStack apart from either of its parent projects:

**Text Generation**
`modded` doesn't / can't support efficient text generation because it doesn't support a kv-cache. To add nanochat's kv-cache and text generation abilities, I removed a couple of modded's pre-training innovations--specifically, Paired Head Attention and Partial Key Offset.

It might be possible to integrate those ideas into the kv-cache, I'm not certain. For now I'm going off of Claude's recommendations--it thinks that the paired-head technique is fundamentally at odds with kv caching, and that the partial key offset would be very difficult to support.

**varlen FlashAttention**

`varlen` is a technique where you pack all of the documents together like they're a single training sample (a "batch size of 1"), and then provide FA with the document boundaries.

`modded` supports this and `nanochat` does not. It's definitely faster for pre-training in some regimes, but I'm curious to explore where the edges are. (TODO - I have a fun trace file illustration for this). Karpathy decided that it wasn't worth the complexity, so I suspect that means its benefits are less dramatic at larger model and/or batch sizes.

I'm definitely getting a huge speedup from it, though, for the post-training tasks. CORE eval and SFT training are much faster in DecoderStack than nanochat due to a combination of varlen (TODO - get some measurements!) and pre-tokenized data.

**Pre-Tokenized Shards**

One of my favorite things about modded-nanogpt is that you don't have to worry about whether your data pre-processing and tokenization are slowing things down. The dataset is just raw token IDs stored very efficiently in binary shards. 

nanochat includes the ability to define your own vocabulary / choose its size / train a tokenizer, which is just beautiful. The GPT-2 tokenizer feels ridiculous and it's always bugged me that we can't touch the vocabulary in `modded`. 

To get the best of both worlds, the `data/` folder includes scripts for training a tokenizer and creating those binary shards. I've uploaded the shards for the 32k vocabulary that nanochat currently uses. I'll probably play with this, and would also welcome any contributions that improve on the vocab.

I've also folded in all of nanochat's code for preparing the CORE and SFT training and evaluation data and then tokenizing those into hosted shards as well. Data processing turned out to be a major bottleneck in nanochat's post-training (CORE evaluation, SFT training and evaluation) which this removes.


## models/modded-nanogpt










