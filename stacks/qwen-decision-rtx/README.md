# qwen-decision-rtx

Fine-tuning Qwen3.5-9B into a decision (System One) model on one RTX PRO 6000: a state plus typed
questions in, a probability per allowed answer out. The forward and backward are hand-written, the
input format is our own (encoder-rtx's special tokens), and nothing has to stay compatible with an
inference engine, so the model is free to change.

Work in progress: the first session is building the LoRA forward and backward
(agent-ops-stacks `qwen-decision-rtx/`).
