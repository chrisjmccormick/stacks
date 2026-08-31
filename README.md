# stacks

Self-contained (~single file) training stacks.

## Pre-Training

### DecoderStack-medium

- Pretraining pipeline for a 24-layer nanochat model (simplified modded-nanogpt).
- Timed on 8xH100, but runnable on Ampere as well.
- [`stacks/decoder-medium/`](stacks/decoder-medium/README.md)
- Also includes a matching Colab Notebook version, reconfigured for a 12-layer
  model, which runs on a 40GB A100. Easier to read through with Colab's outline.

## Reinforcement Learning

### Qwen GSM8K

RL pipeline for training the 500M param Qwen 2.5 model on GSM8K.
- Timed on a 1xH100, runs in (20?) min.
- Much simpler model and optimizer code than pre-training.
- Adds a custom decoding engine--a new optimization target!

### Qwen Arithmetic

A variant of the Qwen GSM8K speedrun which:
- Runs on a **Colab T4**, in xx min.
- Trains on basic arithmetic with shorter responses than GSM8K.
- [`stacks/qwen-arithmetic/`](stacks/qwen-arithmetic/README.md)

### Archive

Speedrun code goes stale fast, so I've moved older projects to a subfolder to 
keep the focus on the more active / up-to-date ones.

## Tutorials

### Stepping Through Forward-Backward

A Notebook written for the Colab T4 which breaks apart Qwen's hand-written
backward, allowing us to step through it and inspect each of the Tensors
as we go.



