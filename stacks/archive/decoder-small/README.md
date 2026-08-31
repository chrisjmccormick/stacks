## DecoderStack-small

> Lives in its own sub-repo: **[`stacks/decoder-small/`](stacks/decoder-small/README.md)**, with its own
> `data/`, `utils/`, requirements, and [`baselines/`](stacks/decoder-small/baselines/README.md).

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

### How to Run

Run from `stacks/decoder-small/` (the script expects `data/`, `triton_kernels.py`, and `generation_small.py` as siblings):

```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 decoderstack_small_pt-sft.py

# Multi-GPU (e.g. 8xH100)
torchrun --standalone --nproc_per_node=8 decoderstack_small_pt-sft.py
```

The dataset is downloaded automatically from HuggingFace on first run. Set `HF_TOKEN` if needed for gated repos. To store data elsewhere, set `DATA_PATH`:

```bash
DATA_PATH=/mnt/data torchrun --standalone --nproc_per_node=1 decoderstack_small_pt-sft.py
```


# DecoderStack-small

nanochat flattened to a single file with modded-nanogpt's optimizations swapped
in: a ~12-layer model at `d_model` 768, trained from pre-tokenized binary shards
through pre-training → CORE eval → SFT → generative eval.

This folder is self-contained: it carries its own copies of the data pipeline,
utilities, and requirements, and is meant to be readable and runnable on its own.
The `medium` track lives at [`../decoder-medium/`](../decoder-medium/README.md)
and is allowed to diverge from this one.

## Contents

| | |
|---|---|
| `decoderstack_small_pt-sft.py` | **the working copy** — pre-training, CORE eval, SFT, generative eval |
| `triton_kernels.py` | Polar Express (`XTX`, `XXT`, `ba_plus_cAA`), fused ReLU² MLP, fused softcapped cross-entropy |
| `generation_small.py` | KV-cache model + the generation-based benchmark harness |
| `run_small.sh` | `torchrun --standalone --nproc_per_node=1 …` |
| `baselines/` | dated snapshots of the code behind each result — see [`baselines/README.md`](baselines/README.md) |
| `data/*.py` | train a tokenizer, then pre-process + pre-tokenize the nanochat datasets into binary shards |
| `utils/convert_ckpt_to_nanochat.py` | convert a capture into a nanochat checkpoint |
| `TODO/` | variants parked mid-integration — not wired up, not expected to run as-is |
| `agent-setup-env.md` | fresh-instance setup (micromamba, requirements, dataset download) |

## Configuration

`num_layers=11`, `num_heads=6` (no GQA), `model_dim=768`, vocab 32,768, dataset
`climbmix_32k_8_170`, `val_tokens = 10,485,760`.

3,960 scheduled steps + 40 extension steps at final LR and window size. Training
ramps through three equal-duration stages, each raising batch size and attention
window together, with the LR scaled to match:

| stage | batch (tokens) | window (short, long) | `lr_mul` |
|---|---|---|---|
| 1 | 8 × 2048 × 8 | (1, 3) | 1.0 |
| 2 | 16 × 2048 × 8 | (3, 7) | 1.52 — (16/8)^0.6 |
| 3 | 24 × 2048 × 8 | (5, 11) | 1.73 — (24/8)^0.5 |
| ext | 24 × 2048 × 8 | (6, 13) | — |

## What sets it apart from its parents

**Text generation.** `modded` can't generate efficiently because it has no
KV-cache. Adding nanochat's cache meant dropping two of modded's pre-training
innovations — Paired Head Attention and Partial Key Offset. Paired-head looks
fundamentally at odds with KV caching; partial key offset looks merely very hard.

**varlen FlashAttention.** Packing all documents as a single sample and handing
FA the document boundaries. `modded` supports it, nanochat doesn't — Karpathy
judged it not worth the complexity, which suggests the benefit shrinks at larger
model and batch sizes. The speedup here is large for the post-training tasks:
CORE eval and SFT are much faster than nanochat's, from varlen plus pre-tokenized
data.

**Pre-tokenized shards.** The dataset is raw token IDs in efficient binary
shards, so pre-processing can't silently become the bottleneck — but with
nanochat's ability to define your own vocabulary and train the tokenizer, which
`modded` gives up by pinning GPT-2's. `data/` builds both.

## How to run

Run from **this** directory — the script expects `data/`, `triton_kernels.py`,
and `generation_small.py` as siblings.

```bash
torchrun --standalone --nproc_per_node=1 decoderstack_small_pt-sft.py
```

```bash
torchrun --standalone --nproc_per_node=8 decoderstack_small_pt-sft.py
```

The dataset downloads from HuggingFace on first run; set `HF_TOKEN` if needed. To
store data elsewhere, set `DATA_PATH`:

```bash
DATA_PATH=/mnt/data torchrun --standalone --nproc_per_node=1 decoderstack_small_pt-sft.py
```
