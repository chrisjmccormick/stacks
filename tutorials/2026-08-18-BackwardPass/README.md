# The backward pass, by hand

A tutorial on the backward pass, built on the hand-written forward/backward of
[`stacks/qwen-arithmetic/train_qwen_arithmetic.py`](../../stacks/qwen-arithmetic/train_qwen_arithmetic.py)
(Qwen2.5-0.5B-Instruct, fp16, no autograd) and runnable on a free Colab T4.

**Status: runs end to end on a T4. The prose pass is in progress** — the notebook
is the copy being revised, and
[The Backward Pass, by Hand](https://colab.research.google.com/drive/1_iUvG8pilfqfem9RR0nEHUc8EcW1yV60)
(opens on a T4) is where that happens.

## What it does

Two parts (outline: `grep -nE '^# (@|={4,} )' backward_pass.py`):

- **Part 1** trains the model for 20 AdamW steps with the *production* code --
  the compiled `forward_backward` (packed varlen, flash attention, chunked
  lm_head/CE) and the compiled, table-driven AdamW -- on Qwen's own correct
  generations from the
  [`basic-arithmetic`](https://huggingface.co/datasets/ChrisMcCormick/basic-arithmetic)
  dataset (`baseline_eval/*_generations.parquet`, 569 docs after filtering).
  Compact and lightly commented; its job is to hand Part 2 a model and
  optimizer state that has genuinely been training. No decode engine, no
  eval, no wandb.
- **Part 2** takes that state and runs ONE pack -- built by Part 1's own
  `plan_packs`, so it is a real training pack (8 docs, T = 512, pad tail and
  all) -- through an *uncompiled, flat* version of the same math: the layer
  loop, the last layer's forward op by op (RMSNorm, fused QKV + RoPE,
  attention, SwiGLU), a plain unchunked lm_head + CE over every
  position, the backward through all of it in the same order reversed, the
  layer-loop backward, the input-embedding scatter, and a hand-written AdamW
  step on one weight tensor. Every intermediate is a named top-level tensor, so
  each section becomes a notebook cell.

  **Attention stays the varlen flash kernel** in both directions -- it is the
  one op left as a call, because spelling it out means materializing the
  `(T, T)` scores it exists to avoid, and its internals are their own subject.
  A parity check then runs Part 1's uncompiled `forward_backward` on the very
  same pack and compares gradients tensor by tensor: they agree to fp16
  rounding (relative L2 error 5.6e-3 to 1.2e-2, cosine >= 0.99994 on all nine).

## What ships here, and which copy is canon

**The Colab notebook on Drive is the working copy.** Both files in this folder
are generated from it, and both are worth shipping because they serve different
readers:

| file | for |
|---|---|
| `backward_pass.ipynb` | opening in Colab and running |
| `backward_pass.py` | handing the code to an agent, or reading it as one file |

`backward_pass.py` is `to-script` output: the code and its inline comments, with
section headings as `# ====` banners and the tutorial prose dropped. It is not
the source of truth -- edits belong in the notebook.

Regenerating both, from the [`colab-utils`](https://github.com/chrisjmccormick/colab-utils)
repo root:

```
python colab_utils.py download <colab_url> --md -o backward_pass.md
python colab_utils.py to-nb    backward_pass.md -o <this_dir>/backward_pass.ipynb
python colab_utils.py to-script backward_pass.md -o <this_dir>/backward_pass.py
```

The `.md` in the middle is the round-trippable editing format (real markdown
prose, code in fences); it is transient and is not committed.

## Running it

Any CUDA box with a **Turing** GPU works; the intended target is a Colab T4.

```
python backward_pass.py
```

It downloads ~1 GB (the fp16 weight banks, the tokenizer, the prebuilt
flash-attention-turing extension, three small parquets) into
`~/.cache/qwen-arithmetic/data/`, compiles for ~5 min on Colab's 2 vCPUs
(step 0 of Part 1), then trains at ~0.6 s/step; Part 2 is under a second. Peak
memory: 10.2 GB (Part 1) / 10.8 GB (Part 2) on the T4's 14.6 GB.

## The flash-attention extension

[`utils/fa_turing.py`](utils/fa_turing.py) handles it, and the notebook just
calls `ensure()`.

flash-attention-turing is a PyTorch C++/CUDA extension, so it is welded to the
Python ABI tag, torch version and C++ ABI flag of the box it was built on --
and building it takes ~12 minutes on Colab's 2 vCPUs, which is far too long to
spend before a tutorial reaches its first line. So prebuilt wheels live in the
model repo, one per runtime ABI, indexed by a sidecar:

```
fa_turing/flash_attn_turing.json     { "builds": { "cp312": {...}, "cp313": {...} } }
fa_turing/flash_attn_turing-0.0.0-cp313-cp313-linux_x86_64.whl
fa_turing/fa_turing.py               <- mirror of utils/fa_turing.py
```

`ensure()` picks the entry matching the box and extracts its `.so` (seconds,
nothing installed into site-packages). **If nothing matches, it builds one on
the spot** rather than stopping the reader -- ~12 min, and it says so loudly.

Colab's image moves without warning (it went python 3.12 -> 3.13 on 2026-08-20
with nothing else changing, which was enough to invalidate every existing
wheel), so when that happens, catch the repo up with:

```
python utils/fa_turing.py status     # what this box is vs what the repo has
python utils/fa_turing.py publish    # build here, smoke it, add a sidecar entry
```

`publish` **adds** an entry and keeps the existing ones, so a reader who
selected an older image in Colab's "Change runtime type" still gets a prebuilt
wheel. It needs a write token in `HF_TOKEN`.
