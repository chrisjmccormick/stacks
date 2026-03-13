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

