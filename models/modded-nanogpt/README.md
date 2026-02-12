modded-nanogpt with stuff from nanochat:
- fineweb-edu 
- custom tokenizer / vocab
- validation bpb
- core eval 

Changes from nanochat:
- Fast CORE evaluation with varlen.
- Pre-tokenized training fineweb-edu.
    - https://huggingface.co/datasets/ChrisMcCormick/fineweb_edu_32k_8_370
    - 370 parquet files --> 203 training files + 1 validation file
        - Validation comes from parquet number 370.
    - Tokenizer trained from first 8 files (7 used for training, 8th used for "evaluating")

