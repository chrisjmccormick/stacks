# qwen-decision-rtx

Fine-tuning Qwen3.5-9B into a decision (System One) model on one RTX PRO 6000: a state plus typed questions in,
a probability per allowed answer out. The forward and backward are written by hand, the input format is
encoder-rtx's special tokens, and nothing has to stay compatible with an inference engine, so the architecture,
training and inference are free to change.

## Running it

On an RTX PRO 6000, in the `stacks` env (plus `pip install "transformers>=5.17" flash-linear-attention` for
FLA's Gated DeltaNet kernels and the checks; FlashAttention-2 comes from the HF kernels hub):

```bash
python data/prepare_decision_data.py   # once: the decision splits in Qwen's tokens -> ~/.cache/qwen-decision-rtx/decision/
python train_decision.py               # Kev's + Recast NLP's records, one pass, then the evaluation panel
CHECK=1 python train_decision.py       # the forward against transformers, the backward against autograd (tiny model, fp64)
```

The weights download on first run: `Qwen/Qwen3.5-9B` @ `c2022362` (the post-trained model, also d3-flash's base).

## The model

Qwen3.5-9B's text stack, unchanged: 32 layers, every fourth one softmax attention (causal, a sigmoid output gate,
per-head q / k RMSNorms, RoPE on a quarter of each head) and the rest Gated DeltaNet (a causal depthwise
convolution, then the gated delta rule with a gated RMSNorm on its output); an RMSNorm before each mixer and each
MLP. The vision tower, the MTP layer and the LM head are dropped: one learned readout vector, dotted with the
final-norm hidden state at each option's readout token, scores the options, with a softmax over the question's
options. 14.8 GiB of bf16 weights on the GPU.

**LoRA** at rank 16 (`ALPHA` 32) on every projection: the fused mixer input (q|k|v|z|b|a in a DeltaNet layer,
q+gate|k|v in an attention layer), the mixer output, the fused gate|up and the down projection; 35.4M
parameters. A shared A per fused matrix, so it's one LoRA per GEMM. Also trained: the readout vector (from zero)
and the special tokens' 18 embedding rows (each from the mean embedding of the word that names it).

## Input format

encoder-rtx's (`data/prepare_decision_data.py`), one row per question:

```
<|state|>I was charged twice for order 1182. Please refund one of the charges.
<|choice|>Which team should handle this?
<|"|>billing<|"|> Charges and refunds<|option|><|"|>shipping<|"|> Deliveries<|option|>
```

`<|noul|>` questions end their two criteria in `<|yes|>` / `<|no|>`, `<|score|>` levels in `<|level_i|>`. The 18
special tokens take the spare rows of Qwen3.5's 248,320-row embedding table after the tokenizer's ids, so user
text can't produce them. States are cut at `STATE_CAP` (8,192) tokens; none of the current data reaches it. In
training a choice question's options are shuffled each time its row is built.

## Training

- **Packs:** rows end to end up to `TOKENS` (16,384), each its own segment: FlashAttention-2 varlen for the
  softmax layers, FLA's chunk kernel with `cu_seqlens` for the DeltaNet layers, the convolution's taps masked at
  row starts. A step is one pack.
- **The backward is hand-written**, a layer at a time in reverse, from activations kept in the forward (nothing
  recomputed but element-wise pieces). FlashAttention's and FLA's own backwards run under a local autograd graph
  that the forward kept for each call (it saves exactly what those backwards read); everything else (the LoRA
  GEMMs, the norms, gates, convolution, decay and RoPE) is differentiated by hand.
- **The optimizer steps inside the backward:** each layer's LoRA weights take their AdamW step as soon as their
  gradients exist, and the layer's activations are freed, so there are no gradient buffers and no gradient
  accumulation: the pack is the batch. LR 2e-4, warmup over 5% of steps, cosine to 10%; no weight decay.
- **Loss:** cross-entropy over each question's options against its target (the hard label, or `SOFT_WEIGHT` x a
  teacher's soft target + the rest on the label), averaged over the pack's questions.

## Evaluation

encoder-rtx's panel, on each split's clean rows: accuracy, Brier score and log loss against the label, and KL
from d3-flash's distribution: decision-dev (Kev's decision-v7 development), transfer-dev (Kev's transfer-v4
development), Phase 1 holdout (Recast NLP's held-out families), JevBench's 231 public items, Cookbook-dev
(TypeSafe's cookbooks as records; read by source). For reference, d3-flash (accuracy): 0.845 / 0.840 / 0.775 /
0.823 / 0.900; encoder-rtx's default decision stage: 0.741 / 0.509 / 0.569 / 0.468.

## Knobs

From the environment, each printed to the log: `TRAIN_SPLITS` (`train,phase1_train`), `VAL_SPLITS` (the panel;
empty skips it), `EPOCHS` (1), `TOKENS` (16384), `EVAL_TOKENS` (32768), `LR` (2e-4), `RANK` (16), `ALPHA` (32),
`SOFT_WEIGHT` (1.0), `STATE_CAP` (8192), `VAL_EVERY` (0: the end only), `ABORT_STEP`, `SEED` (42), `RUN_NAME`,
`USE_WANDB` (project `qwen_decision_rtx_dev`), `SAVE_CHECKPOINT` (the trained tensors, ~140 MB, to
`~/.cache/qwen-decision-rtx/runs/<run>/`), `CHECK`.

## Speed and memory

Eager PyTorch, no compile (agent-ops-stacks `qwen-decision-rtx/2026-11-01_0507pm_lora-fwd-bwd`, T1):

| `TOKENS` | step | tokens/s | peak memory |
|---|---|---|---|
| 8,192 | 1.8-2.0 s | 4.0-4.1k | 50.1 GiB |
| 16,384 | 3.4-4.2 s | 3.9-4.1k | 85.9 GiB |
| 24,576 | out of memory | | |

The kept activations cost 4.4 MiB a token; ~4.0k tokens/s is ~130 TFLOP/s of LoRA forward-plus-backward.

## Checks

`CHECK=1` builds a tiny random Qwen3.5 (4 layers) in fp64 with per-row reference kernels: with LoRA's B at zero
the forward's final hidden states match transformers' `Qwen3_5TextModel` to 1.4e-5 (transformers computes the
delta rule, gated norm, decay and RMSNorms in fp32 whatever the dtype), and with B random the hand-written
gradients match autograd's through the same forward to 1e-15 on all 34 trained tensors.
