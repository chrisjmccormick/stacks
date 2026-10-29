# encoder-rtx

Pre-training an encoder to be a decision model: a model that takes a `state` and typed
questions about it, and returns a probability for every allowed answer (the shape of
TypeSafe's System One API). The starting idea is to train jointly on MLM and on
predictions distilled from an open decision model, beginning from
[`decoder-rtx`](../decoder-rtx/)'s code.

**Status:** Phase 0 (October 2026). The data prep for Kev's suites is written; the training
script isn't yet. The proposal and the discussion behind it are in
`agent-ops-stacks/encoder-rtx/` (private).

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
