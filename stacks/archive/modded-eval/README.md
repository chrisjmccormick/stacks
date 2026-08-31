This is setup to evaluate modded-nanogpt's performance against nanochat with minimal changes to modded's train_gpt.py script.

The changes are:
* Switch to nanochat's 32k vocab and ClimbMix training and validation sets (previously fineweb-edu).
* Switch to bits-per-byte as the validation metric.
* Evaluate the trained model on the CORE benchmark.

I don't necessarily plan to maintain this--it's more like an interesting experiment, and served as the starting point for the DecoderStack.

modded-nanogpt with stuff from nanochat:
- ClimbMix (previously fineweb-edu)
- custom tokenizer / vocab
- validation bpb
- core eval 

Changes from nanochat:
- Fast CORE evaluation with varlen.
- Pre-tokenized training data (ClimbMix).
    - https://huggingface.co/datasets/ChrisMcCormick/climbmix_32k_8_170
    - 170 parquet files --> ~93 training files + 1 validation file
        - Validation comes from the last parquet shard.
    - Tokenizer trained from first 8 files (7 used for training, 8th used for "evaluating")

