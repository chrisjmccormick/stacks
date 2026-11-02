# qwen-decision-rtx

Fine-tuning Qwen3.5-9B into a decision (System One) model on one RTX PRO 6000: a state plus typed questions in,
a probability per allowed answer out. The forward and backward are written by hand in decoder-rtx's style and run
under `torch.compile(dynamic=False, fullgraph=True)`; nothing has to stay compatible with an inference engine, so the
architecture, training and inference are free to change.

## Running it

On an RTX PRO 6000, in the `stacks` env (`requirements-x86.txt` adds FLA's Gated DeltaNet kernels; FlashAttention-2
comes from the HF kernels hub):

```bash
python data/prepare_model.py           # once: Qwen3.5-9B's text stack, banked -> ~/.cache/qwen-decision-rtx/model/ (15 GB)
python data/prepare_decision_data.py   # once: the decision splits in d3's prompt format, Qwen's tokens -> .../decision/
python train_decision.py               # Kev's records, one pass, the panel at step 0, every 400 steps and at the end
```

No command line and no environment knobs: the configuration is `DecisionConfig` at the top of `train_decision.py`
(`cfg`), and `ABORT_STEP` / `CHECK_GRADS` under it are the two development switches. `WANDB_API_KEY` is the one
environment variable. Logs land in `logs/<run_name>.txt`, a metrics CSV and a result JSON beside it; the trained
tensors (LoRA and the readout, ~150 MB) in `~/.cache/qwen-decision-rtx/runs/<run_name>/trained.pt`.

## The model

Qwen3.5-9B's text stack, unchanged: 32 layers, every fourth one softmax attention (causal, a sigmoid output gate,
per-head q / k RMSNorms, RoPE on a quarter of each head) and the rest Gated DeltaNet (a causal depthwise
convolution, then the gated delta rule with a gated RMSNorm on its output); an RMSNorm before each mixer and each
MLP. The vision tower and the MTP layer are dropped, and the LM head is cut to its rows for the 255 answer codes,
which become the trained readout.

**`data/prepare_model.py` banks the weights the way the trainer reads them** (qwen-arithmetic's pattern): per-layer
matrices stacked into `(L, ...)` banks, one set for the 24 DeltaNet layers (`deltanet_*`) and one for the 8 attention
layers (`attention_*`); each layer's input projections fused into one GEMM ([q | k | v | z | b | a], [q | gate | k | v]
with q and gate de-interleaved, [gate | up]); the zero-centred norms stored as 1 + w in fp32; the convolution's taps
transposed to (4, channels); the decay as -exp(A_log). 7.94B parameters, 14.8 GiB of bf16 on the GPU. The sidecar
JSON carries the arch and the code ids, and the trainer asserts both against its config and the data.

**LoRA** at rank 16 (`alpha` 32) on every projection (the fused mixer input, the mixer output, gate|up, down), A and B
banked per kind like the frozen weights; 35.4M parameters. Plus the readout, (255, 4096), from the LM head's rows for
the codes: before any training the model answers as Qwen does.

## Input format

d3-flash's prompt (`data/prepare_decision_data.py`), one row per question: Qwen's chat template with d3's system
prompt, then

```
State:
{"request": "I was charged twice for order 1182. Please refund one of the charges."}

Question:
Which team should handle this?

Options:
A: billing: Charges and refunds
B: shipping: Deliveries

Reply with only the code of the best option.
```

and the assistant turn opened with thinking off. Every option is read at the row's last token: the logit of option k
is that token's final-norm hidden state dotted with the readout's k-th row, and the softmax runs over the row's
options. A noul question lists false then true; a score question one line per level. In training a choice question's
options are shuffled each time its row is built. The longest row in the current data is 7,447 tokens (Kev's train).

## Training

- **Packs, all one shape:** rows end to end up to `pack_tokens` (16,384), at most `max_rows` (256) segments: the rows,
  then the padding in `max_row_tokens` (8,192) pieces, then empty segments (`cu_seqlens` padded with the pack
  length, as decoder-rtx's loader does; FLA and FA2 both handle it). Every pack of the run is planned and staged
  in pinned memory up front. A step is one pack.
- **Compiled, static, full graphs:** `embed_forward`, `layer_forward`, `readout_forward_backward`, `layer_backward`
  and `adamw_step` each carry `@torch.compile(dynamic=False, fullgraph=True)`; `layer_forward` and `layer_backward`
  specialize once per layer kind. FlashAttention-2 and FLA's chunked gated delta rule are `torch.library.custom_op`s
  with fake shapes, so the graph sees them as opaque calls and stays whole. The backward recomputes every
  element-wise piece (the norms' outputs, silu, the kernels' inputs) from a lean stash: ~3 MiB a token.
- **The backward is hand-written**, a layer at a time in reverse, the LoRA gradients landing in their `(L, ...)`
  gradient banks; FA2's and FLA's own backwards are called by name (below their autograd functions). Then one
  compiled AdamW step over the 17 trained tensors: betas 0.9 / 0.999, peak LR 2e-4 with 5% warmup and cosine to
  10%, no weight decay, the bias corrections folded into per-step tables.
- **Loss:** cross-entropy over each row's options against its target (the hard label, or `soft_weight` x a teacher's
  soft target + the rest on the label), averaged over the pack's rows.

Compiled code does not reproduce eager or transformers logits bit for bit, and it isn't asked to: `CHECK_GRADS`
confirms the hand-written backward against autograd through the same forward (eager, 1.5e-2 relative in bf16), and
what is validated is behavior, on the panel below.

## Evaluation

encoder-rtx's panel, on each split's clean rows: accuracy, Brier score and log loss against the label, and KL
from d3-flash's distribution: decision-dev (Kev's decision-v7 development), transfer-dev (Kev's transfer-v4
development), Phase 1 holdout (Recast NLP's held-out families), JevBench's 231 public items, Cookbook-dev
(TypeSafe's cookbooks as records; read by source). For reference, d3-flash (accuracy): 0.845 / 0.840 / 0.775 /
0.823 / 0.900; encoder-rtx's default decision stage: 0.741 / 0.509 / 0.569 / 0.468.

## Speed and memory

_To be filled from agent-ops-stacks `qwen-decision-rtx/2026-11-01_0823pm_compiled-rewrite` (T1, R1)._

The eager predecessor (agent-ops-stacks `qwen-decision-rtx/2026-11-01_0507pm_lora-fwd-bwd`, T1) ran 16,384-token packs
at 3.4-4.2 s a step, ~4.0k tokens/s, 85.9 GiB peak, two thirds of the step in element-wise glue.
