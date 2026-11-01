# encoder-rtx

Pre-training an encoder to be a decision model: a model that takes a `state` and typed
questions about it, and returns a probability for every allowed answer (the shape of
TypeSafe's System One API). The starting idea is to train jointly on MLM and on
predictions distilled from an open decision model, beginning from
[`decoder-rtx`](../decoder-rtx/)'s code.

**Status (Nov 1, 2026):** two stages, each a single file with a handwritten backward, on
one RTX PRO 6000. `train_mlm.py` pre-trains the encoder on masked LM -- decoder-rtx's
Baseline 6 made bidirectional, reading documents of up to 8,192 tokens -- in 13.7 minutes of
training (val MLM 1.779 nats per corrupted position). Its checkpoint is `runs/mlm-base8k/` in
the private model repo `ChrisMcCormick/encoder-rtx`. `train_decision.py` starts from it and
trains on decision records: two passes over Kev's records and Recast NLP's (d3-flash's
labels, yes/no questions balanced per source) at a 0.5 decision share (616 steps, 3.4 min of
training at 0.328 s/step) reach decision-dev accuracy 0.70-0.73, transfer-dev 0.48-0.51 and
JevBench 0.45-0.50, against 0.33 / 0.39 / 0.32 for uniform guessing (Kev's records alone,
one pass: 0.64 / 0.50 / 0.38-0.44). `train_encoder.py`,
at the bottom, is Phase 0's
single-stage fork (joint MLM + decision from scratch on the 32K BPE), superseded by the two
stages and kept for reference. The proposal, the sessions and their logs are in
`agent-ops-stacks/encoder-rtx/` (private).

## Stage 1: masked-LM pre-training, `train_mlm.py`

On an RTX PRO 6000, in the `stacks` env:

```bash
SAVE_CHECKPOINT=1 RUN_NAME=mlm-base8k python train_mlm.py   # downloads the ClimbMix shards on first run
```

The model is decoder-rtx's Baseline 6 (8 layers, d768, priml's recipe on the 16K Unigram
tokenizer) made bidirectional and trained on MLM alone:

- **Attention** looks both ways within each document (FA2 varlen, `causal=False`): 256
  tokens each side on layers 0-2 and 4-6, the whole document on layers 3 and 7. Documents
  are cut at 8,192 tokens. RoPE (base 3,000,000) runs on positions across the micro-batch;
  it is relative, so a document reads the same wherever it sits.
- **Layer 0** attends from the token's normed embedding itself, so its q, k and v are
  projected once per vocabulary row and gathered per token.
- **Layers 5-7 attend from layer 4's output** (their Q, K, V and gates all read it), so past
  layer 4 only the rows the loss reads need a stream of their own: the corrupted
  positions. Those layers compute K and V at every row and everything else -- Q, W_O, the
  MLP, the pooled layer-6 output -- at the corrupted rows alone. Layer 7 (whole document)
  runs its queries at those rows only. Layers 5-6 (windowed) run theirs at every row and
  keep the corrupted rows' output, since FA2 places a window by row index. The change is
  exact (checked in fp64 against the unrestricted path), and the evaluation forward,
  which is the inference path, gets it too.
- **Value memories:** value-embedding tables on layers 1, 3, 5 and 7, and four bigram
  n-gram memories on the same layers (two hash slots of 589,824 rows each, keyed on
  (prev, curr) of the corrupted input): the ~4 GB model.
- **MLM:** each step corrupts 50% of the non-BOS positions (80% `[MASK]`, 10% a random
  token, 10% unchanged), and the LM head runs on the corrupted positions only. The n-gram
  memories hash the corrupted input, so no key sees a target.
- **Recipe:** Baseline 6's optimizers -- Muon on the matrices, AdamW on the embedding
  tables, head and scalars (the input embeddings stepped inside the backward), row-wise
  RMSProp on the touched n-gram rows -- with every peak rate at 0.7x, a 20-step warmup on
  the x0 scalars, and 2,605 steps of 196,608 tokens (512M).
- **Validation:** 168 x 65,536 tokens of the val shard, documents cut at 2,048, corrupted at
  30% / 80-10-10 by a generator re-seeded per micro-batch, so every run scores the same 3.30M
  positions: `val_mlm` in nats per corrupted position, `val_mbpb` the same over the
  corrupted tokens' bytes, in bits.
- Knobs from the environment, logged: `RUN_NAME`, `MASK_RATE` (0.50), `USE_WANDB`,
  `SAVE_CHECKPOINT` (the final weights and optimizer state to
  `~/.cache/encoder-rtx/runs/<run_name>/`), `ABORT_STEP`.

| run | documents cut at | val MLM | long documents, whole | s/step | training |
|---|---|---|---|---|---|
| `mlm-bigrams590k` / `mlm-base4gb` (the 4 GB base, before Baseline 6's last fold) | 2,048 | 1.7669 / 1.7756 | -- | 0.349 | 15.1 min |
| `mlm-b6fold-2k` (this script at 2,048) | 2,048 | 1.7649 | 1.8082 | 0.302 | 13.1 min |
| **`mlm-base8k`** (this script) | 8,192 | 1.7790 | **1.7637** | 0.316 | 13.7 min |

"Long documents" is the val shard's 2,709 documents over 2,048 tokens, cut at 8,192, each
corrupted once at 30% and read whole. By place in the document the 8K model leads the 2K
one by 0.019 nats at 0-2K, 0.050 at 2-4K and 0.107 at 4-8K. Read in 2,048-token chunks
instead, the two models are level (1.842 / 1.840), so the lead comes from the context. The
standard validation, whose documents are cut at 2,048, is 0.014 worse at 8K. Under the 16K
tokenizer 3.3% of ClimbMix's documents run past 2,048 tokens and hold 23% of its tokens;
0.3% run past 8,192 and hold 7.5%.

Same-seed runs of a config differ by up to ~0.009 val MLM (the kernels' nondeterminism),
and two seeds by ~0.025.

## Stage 2: decision training, `train_decision.py`

On an RTX PRO 6000, in the `stacks` env:

```bash
python data/prepare_decision_data.py   # once: Kev's suites (+ Phase 1's) -> ~/.cache/encoder-rtx/decision16k/
# the 8K MLM base, once (private repo; HF_TOKEN):
huggingface-cli download ChrisMcCormick/encoder-rtx --include 'runs/mlm-base8k/*' --local-dir ~/.cache/encoder-rtx/hub
RUN_NAME=decision python train_decision.py   # defaults: Kev + Recast NLP, two passes, share 0.5, LR_SCALE 0.3
```

It is stage 1's script with Phase 0's decision path brought in:

- **Packs** of decision records and ClimbMix filler, as in Phase 0 (below): every state,
  question and filler document is its own segment, and a question's RoPE positions
  continue its state's. The filler is trained on MLM at the base's 0.50 rate, cut at 8,192
  like the base's documents, and starts at train shard 8, past what the base read. `EPOCHS`
  passes over the records set the step count. Training packs and evaluation packs are each
  padded to their own static widths (segment slots, records, readout slots): FA2 launches
  blocks for every segment slot up to `max_seqlen`, so padding the training packs to the
  evaluation packs' ~4x larger widths cost ~20% of the step at 8K.
- **The question-to-state FA2 call**, merged with the self call by the softmax LSEs before
  Baseline 6's per-head output norm and gate. The self call keeps the base's windows
  (256 each way, the whole segment on layers 3 and 7), and the cross call reads the whole
  state. The n-gram memories hash as in the base, across segment boundaries: a segment's
  first token pairs with the previous segment's last. No label is ever an input, so no
  label crosses.
- **Layers 5-7 at the read rows**, as in stage 1: here the corrupted filler positions and
  the readouts. Padding readout slots sit outside every gathered query segment.
- **The readout** is Phase 0's single vector, trained by cross-entropy against each
  question's target. That target is the hard label, or `SOFT_WEIGHT` x a soft target
  (the record's own, or a teacher's joined in by the data prep) + the rest on the label.
- **Init from the base** (`INIT_FROM`): the weights, with exact fp32 masters. Optimizer
  state is fresh, and the step count and schedules are this run's (the x0 scalars' 20-step
  warmup included). `LR_SCALE` multiplies stage 1's rates on every pre-trained weight. The
  readout is new and keeps its own rate.
- **Evaluation:** the base's own MLM validation, on the same 3.30M corrupted positions (the
  loaded 8K base reads its run's 1.7790). Plus accuracy and Brier score on the clean rows
  of decision-v7 and transfer-v4 development, Phase 1's holdout families (clinc150,
  formality, health_fact), JevBench's 231 public items and Cookbook-dev (798 records shaped
  like TypeSafe's cookbooks: a 194-way skill choice, 10-Ks over 75 SIC groups, a same-product
  yes/no with a 0.849 majority, citation checks; eval only), with accuracy by source and by
  state length (over / under 2,048 tokens) at the end. Read Cookbook-dev by source: its
  overall figure follows entity_alignment's yes/no lean, which flips between runs.
- Knobs from the environment, logged: `EPOCHS` (2), `DECISION_FRAC` (0.5), `LR_SCALE`
  (0.3), `SOFT_WEIGHT` (1.0), `DECISION_WEIGHT` (1.0), `TRAIN_SPLITS` (`train,phase1_train`:
  any splits the data prep writes; `train` alone is Kev's records), `STATE_CAP` (8192), `INIT_FROM`,
  `VAL_EVERY` (25), `MASK_RATE` (0.50), `RUN_NAME`, `USE_WANDB`, `ABORT_STEP`,
  `SAVE_CHECKPOINT`.

**The default mix** (accuracy on the clean rows; Phase 1 holdout = Recast NLP's held-out
families at `bd5ae231`; agent-ops-stacks `encoder-rtx/2026-11-01_1203am_decision-data-arms`):

| training records | runs | val MLM | decision-dev | transfer-dev | Phase 1 holdout | JevBench | steps | s/step |
|---|---|---|---|---|---|---|---|---|
| Kev's, one pass (the previous default, `TRAIN_SPLITS=train EPOCHS=1`) | 3 seeds | 1.835 | 0.638 / 0.635 / 0.646 | 0.508 / 0.497 / 0.494 | 0.443 / 0.445 / 0.449 | 0.437 / 0.381 / 0.398 | 145 | 0.318 |
| **Kev's + Recast NLP's (d3 labels, nouls balanced), two passes** | 3 seeds (session) | 1.822 | 0.718 / 0.726 / 0.718 | 0.512 / 0.502 / 0.497 | 0.531 / 0.538 / 0.547 | 0.485 / 0.446 / 0.472 | 616 | 0.330 |
| the same, the committed data prep (seed 42: the defaults with no knobs; 43 / 44: the session's variant with a seed knob) | 3 seeds | 1.820 | 0.704 / 0.714 / 0.710 | 0.508 / 0.488 / 0.482 | 0.527 / 0.532 / 0.546 | 0.468 / 0.472 / 0.498 | 616 | 0.328 |

On Cookbook-dev the committed defaults read 0.535 overall: skill suggestion 33 / 280 (d3-flash
0.929; majority 0.029), the 10-K question 13 / 60 (0.650; 0.167), entity_alignment 378 / 450
(0.913; 0.849), citation_check 3 / 8; Kev-only runs read 27-30 and 3 on the first two.

The new mix lifts every in-distribution and Jev-shaped number at every seed and holds
transfer-dev (the committed data path against Kev-only, seed means: +0.07 decision-dev, +0.09
Phase 1 holdout, +0.07 JevBench, -0.007 transfer-dev); on Cookbook-dev (TypeSafe's cookbooks as decision records, eval only) its
many-option choices beat every Kev-only run. Transfer-dev doesn't move with any mix tried:
on most of its families the stage answers one constant per family, so its accuracy reads
priors, and the same config moves ~0.015 between seeds.

The earlier table, one epoch of Kev's records, `DECISION_FRAC=0.5 LR_SCALE=0.3`:

| base | states cut at | val MLM | decision-dev | transfer-dev | Phase 1 holdout | JevBench | s/step |
|---|---|---|---|---|---|---|---|
| `mlm-base4gb` (before Baseline 6's last fold, shared pack widths) | 2,048 | 1.830 | 0.619 | 0.514 | 0.438 | 0.424 | 0.399 |
| `mlm-base8k` | 2,048 | 1.834 | 0.634 / 0.634 | 0.485 / 0.480 | 0.433 / 0.439 | 0.463 / 0.437 | 0.312 |
| **`mlm-base8k`** | 8,192 | 1.835 | 0.645 / 0.638 | 0.499 / 0.511 | 0.442 / 0.439 | 0.442 / 0.433 | 0.318 |

Two runs each for the 8K base. The same config run twice differs by up to 0.03 on JevBench
(n = 231) and 0.01-0.02 elsewhere, so whole states and cut ones are level here: JevBench's 36
states over 2,048 scored 12 whole against 13 cut. Kev-0.8B scores 0.648 on transfer-dev,
d3-flash 0.823 on JevBench.

**Input format** under the base's 16K Unigram tokenizer: Phase 0's (below), with the
state opening with BOS. `<|mask|>` is the base's `[MASK]` (16,369). The other 18 special
tokens take ids 16,384-16,401, so the input embeddings and value embeddings grow to
16,448 rows. Each new row starts from the base's rows for the word that names its token.
The LM head stays at 16,384, since MLM never targets a special token. States are cut to
their first `STATE_CAP` tokens: at 8,192, 2 of Kev's 30,540 training records are cut, and
none in the evaluation splits (JevBench's longest state is 4,170 tokens).

**Data:** `data/prepare_decision_data.py` tokenizes anything in Kev's suite schema, one
JSON record per line, into `~/.cache/encoder-rtx/decision16k/<split>.parquet`. `SPLITS`
lists the sources: Kev's pinned Hub and GitHub files, a file in any other Hub dataset repo
(`hf:<repo>@<revision>`), or a local path; Phase 1's (Recast NLP's) three splits are listed,
pinned at `bd5ae231`. `LABELS` joins a teacher's soft labels on (record id, question name),
from a parquet at any of the same kinds of source, or d3-flash's labels for a split as
`("d3", <source>)`, joined on (record index, question name) with each record's line hash and
the source files checked; `phase1_train` carries d3-flash's labels by default (all 121,503 of
its questions). `BALANCE_NOULS` balances a split's yes/no questions to 0.5 true within each
source (`phase1_train`: Recast NLP's renderer makes a two-class family's yes/no questions
~0.75 true whatever the template, and that phrasing prior cost transfer-dev's yes/no families);
the majority side is dropped by a hash of (record id, question name): 85,627 records ->
81,953.

## Phase 0: `train_encoder.py`

Superseded by the two stages above and kept for reference; nothing below is maintained.

On an RTX Pro 6000, in the `stacks` env (decoder-rtx's `requirements-x86.txt` covers it):

```bash
python data/prepare_kev_suites.py    # once: Kev's suites -> ~/.cache/encoder-rtx/data/
python train_encoder.py              # downloads the ClimbMix shards on first run
```

Training runs in two stages. Stage 1 is MLM alone and writes a checkpoint to
`~/.cache/encoder-rtx/runs/<run>/`; stage 2 starts from it with fresh optimizer state, its
own step count and schedules, and trains the decision readout on Kev's records (with some
MLM filler kept, so the language doesn't drift):

```bash
DECISION_FRAC=0 NUM_STEPS=8316 SAVE_CHECKPOINT=1 RUN_NAME=stage1-mlm python train_encoder.py
INIT_FROM=~/.cache/encoder-rtx/runs/stage1-mlm/model_step008316.pt \
  DECISION_FRAC=0.5 LR_SCALE=0.3 NUM_STEPS=400 VAL_EVERY=50 RUN_NAME=stage2-frac50 python train_encoder.py
```

For the first runs, these environment variables override the config (the values are
logged): `DECISION_FRAC` (0.10; 0 for MLM only, at most 0.9 so the MLM validation keeps
enough filler), `DECISION_WEIGHT` (1.0), `MASK_RATE` (0.30), `NUM_STEPS` (2079),
`VAL_EVERY` (333), `LR_SCALE` (1.0), `SAVE_CHECKPOINT` (0), `INIT_FROM` and `RUN_NAME`.
At a 0.5 share one epoch of Kev's 30.5k records is about 130 steps; at 0.9, about 75.

### The model

decoder-rtx's 8-layer, d768 stack and its training recipe (Muon, the embedding tables
and their priors), with these changes:

- **Packs.** Each 192K-token micro-batch holds decision records, about 10% of it, and
  ClimbMix filler for the rest. Every state, question and filler document is its own
  segment.
- **Attention.**
  - It's bidirectional, using the same sliding windows each way (768) and full context on
    layers 3, 4 and 7.
  - Each record's questions also read their state in a second FA2 call, merged with the
    first by the two softmax LSEs.
  - The state is encoded once, and questions never see each other. The smear and the
    `[prev, curr]` tables stop at segment boundaries too.
- **Objectives.** MLM on 30% of the filler (80/10/10), plus cross-entropy over each
  question's options against its target.
- **Evaluation**, every 333 steps: MLM loss on the val shard, plus accuracy and Brier
  score on the clean rows of decision-v7 development and transfer-v4 development. On
  transfer-v4, Kev-0.8B scores 0.648.

### Input format

The vocabulary is decoder-rtx's 32,768-token BPE, unchanged, plus 19 special tokens at
ids 32,768–32,786. The state is encoded once. Each question attends to the state and to
itself, never to another question, and is read out at one token per option:

```
<|state|>I was charged twice for order 1182. Please refund one of the charges.
<|choice|>Which team should handle this?
<|"|>billing<|"|> Charges and refunds<|option|><|"|>shipping<|"|> Deliveries<|option|>
<|noul|>Does this need a reply today?
Must be handled today<|yes|>Can wait<|no|>
<|score|>How frustrated is the customer?
low<|level_0|>normal<|level_1|>high<|level_2|>
```

`<|"|>` wraps option names, and string values in JSON. Each option's readout logit is one
learned vector dotted with its readout token's hidden state, with a softmax over the
question's options.

### Data

`data/prepare_kev_suites.py` renders [Kev](https://github.com/jaredpalmer/kev)'s published
decision suites in this format. That's the 30,540 records Kev-4B trained on, plus the
decision-v7 and transfer-v4 development and test splits. It writes parquet files to
`~/.cache/encoder-rtx/data/`.
