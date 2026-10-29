# encoder-rtx

Pre-training an encoder to be a decision model: a model that takes a `state` and typed
questions about it, and returns a probability for every allowed answer (the shape of
TypeSafe's System One API). The starting idea is to train jointly on MLM and on
predictions distilled from an open decision model, beginning from
[`decoder-rtx`](../decoder-rtx/)'s code.

**Status:** Phase 0 (October 2026). `train_encoder.py` is forked from decoder-rtx's
`sep-28-baseline`; its attention merge is verified against the real FA2 kernel, and the
first runs on an RTX Pro 6000 (17.6 min for 409M tokens, 53.5 GB peak) reach val MLM 2.04
nats, decision-dev accuracy 0.56 against 0.33 for chance, and chance on transfer-dev.
Training is now split into an MLM stage and a decision stage (below). The proposal and
the discussion behind it are in `agent-ops-stacks/encoder-rtx/` (private).

## How to run

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

## The model

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

## Input format

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

## Data

`data/prepare_kev_suites.py` renders [Kev](https://github.com/jaredpalmer/kev)'s published
decision suites in this format. That's the 30,540 records Kev-4B trained on, plus the
decision-v7 and transfer-v4 development and test splits. It writes parquet files to
`~/.cache/encoder-rtx/data/`.
