# encoder-rtx

Pre-training an encoder to be a decision model: a model that takes a `state` and typed
questions about it, and returns a probability for every allowed answer (the shape of
TypeSafe's System One API). The starting idea is to train jointly on MLM and on
predictions distilled from an open decision model, beginning from
[`decoder-rtx`](../decoder-rtx/)'s code.

**Status (Oct 30, 2026):** the decision stage runs on the MLM base. `train_decision.py`
takes the 4 GB MLM base (decoder-rtx's `baseline-6` made bidirectional and trained on MLM
alone; its checkpoint is `runs/mlm-base4gb/` in the private model repo
`ChrisMcCormick/encoder-rtx`) and trains it on decision records. One epoch of Kev's
records (72 steps, 23 s on an RTX Pro 6000) reaches decision-dev accuracy 0.585 and
transfer-dev 0.514, against 0.33 / 0.39 for uniform guessing. `train_encoder.py`, below
it, is Phase 0: a fork of decoder-rtx's `sep-28-baseline` on the 32K BPE, trained jointly
from scratch (decision-dev 0.56, transfer-dev at chance). The proposal and the discussion
behind both are in `agent-ops-stacks/encoder-rtx/` (private).

## The decision stage: `train_decision.py`

On an RTX Pro 6000, in the `stacks` env:

```bash
python data/prepare_decision_data.py   # once: Kev's suites -> ~/.cache/encoder-rtx/decision16k/
# the MLM base, once (private repo; HF_TOKEN):
huggingface-cli download ChrisMcCormick/encoder-rtx --include 'runs/mlm-base4gb/*' --local-dir ~/.cache/encoder-rtx/hub
EPOCHS=1 DECISION_FRAC=0.9 LR_SCALE=0.3 RUN_NAME=dec-f90-e1 python train_decision.py
```

It is the MLM base's script (`train_mlm-28-base4gb.py` in the ops repo) with Phase 0's
decision path brought in:

- **Packs** of decision records and ClimbMix filler, as in Phase 0 (below). The filler is
  trained on MLM at the base's 0.50 rate and starts at train shard 8, past what the base
  read. `EPOCHS` passes over the records set the step count.
- **The question-to-state FA2 call**, merged with the self call by the softmax LSEs before
  baseline-6's per-head output norm and gate. The self call keeps the base's windows
  (256 each way, full on layers 3 and 7), and the cross call reads the whole state. The
  n-gram memories hash as in the base, across segment boundaries: a segment's first
  token pairs with the previous segment's last. No label is ever an input, so no label
  crosses.
- **The readout** is Phase 0's single vector, trained by cross-entropy against each
  question's target. That target is the hard label, or `SOFT_WEIGHT` x a soft target
  (the record's own, or a teacher's joined in by the data prep) + the rest on the label.
- **Init from the base** (`INIT_FROM`): the weights, with exact fp32 masters. Optimizer
  state is fresh, and the step count and schedules are this run's. `LR_SCALE` multiplies
  the recipe's rates on every pre-trained weight. The readout is new and keeps its own
  rate.
- **Evaluation:** the base's own MLM validation, on the same 3.30M corrupted positions
  (the loaded base scores its run's 1.7756). Plus accuracy and Brier score on the clean
  rows of decision-v7 and transfer-v4 development.
- Knobs from the environment, logged: `EPOCHS` (1), `DECISION_FRAC` (0.9), `LR_SCALE`
  (0.3), `SOFT_WEIGHT` (1.0), `DECISION_WEIGHT` (1.0), `TRAIN_SPLITS` (`train`; add
  splits the data prep writes), `STATE_CAP` (2048), `INIT_FROM`, `VAL_EVERY` (25),
  `MASK_RATE` (0.50), `RUN_NAME`, `USE_WANDB`, `ABORT_STEP`, `SAVE_CHECKPOINT`.

**Input format** under the base's 16K Unigram tokenizer: Phase 0's (below), with the
state opening with BOS. `<|mask|>` is the base's `[MASK]` (16,369). The other 18 special
tokens take ids 16,384-16,401, so the input embeddings and value embeddings grow to
16,448 rows. Each new row starts from the base's rows for the word that names its token.
The LM head stays at 16,384, since MLM never targets a special token. States are cut to
their first 2,048 tokens, since the base never saw a longer document. That cut affects
1,057 training records and none in the dev splits.

**Data:** `data/prepare_decision_data.py` tokenizes anything in Kev's suite schema, one
JSON record per line, into `~/.cache/encoder-rtx/decision16k/<split>.parquet`. `SPLITS`
lists the sources: Kev's pinned Hub and GitHub files, a file in any other Hub dataset repo
(`hf:<repo>@<revision>`), or a local path. `LABELS` joins a teacher's soft labels on
(record id, question name), from a parquet at any of the same kinds of source.

## Phase 0: `train_encoder.py`

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
