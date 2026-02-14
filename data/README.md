# data/

_NOTE: You do not need to run anything from this directory unless you want to train your own tokenizer / custom vocabulary. Otherwise, the training scripts in this repo automatically download the pre-processed files from my huggingface repo._

Scripts for training a custom tokenizer, then pre-processing and pre-tokenizing all datasets (pre-training, CORE eval, SFT, chat eval) using that vocabulary.

All scripts should be run from the **repo root** (`stacks/`), not from inside `data/`.

## Pipeline

### 1. `data/dataset_and_vocab.py` — Train tokenizer + pre-tokenize training data

This is the starting point. It downloads FineWeb-Edu, trains a BPE tokenizer on it, then pre-tokenizes the full dataset into `.bin` shards that the training script can load directly.

```bash
python data/dataset_and_vocab.py
```

Steps it performs:
1. Downloads FineWeb-Edu parquet shards from HuggingFace (cached in `~/.cache/nanochat/base_data/`)
2. Trains a BPE tokenizer on the first N shards
3. Evaluates the tokenizer against GPT-2/GPT-4 baselines
4. Pre-tokenizes all shards into `.bin` files
5. Optionally uploads everything to HuggingFace (pass `--skip-upload` to skip)

**Produces:** `data/<dataset_name>/` containing:
- `tokenizer/tokenizer.pkl` and `tokenizer/token_bytes.pt`
- `config.json`
- `train_000000.bin`, `train_000001.bin`, ... (pre-training shards)
- `val_000000.bin` (validation shard)

The default dataset name is `fineweb_edu_32k_8_370` (32k vocab, 8 shards for tokenizer training, 370 total shards). You can customize with `--vocab-size`, `--tok-train-shards`, `--num-shards`, etc.

---

### 2. `data/core_dataset.py` — CORE eval pre-tokenization

Pre-tokenizes the CORE evaluation benchmark (Karpathy's logit-based eval suite). Downloads the eval bundle, renders few-shot prompts, tokenizes them, and writes one `.pt` file per task.

```bash
python data/core_dataset.py
```

**Requires:** The tokenizer from step 1 (`data/<dataset_name>/tokenizer/`).

**Produces:** `data/<dataset_name>/core_eval/` containing per-task `.pt` files and a `config.json`.

---

### 3. `data/sft_dataset.py` — SFT training data pre-tokenization

Loads the SFT conversation mixture, formats and tokenizes conversations, and writes `.bin` shards (with parallel mask shards for supervision masking).

```bash
python data/sft_dataset.py
```

**Requires:** The tokenizer from step 1.

**Data sources (train):**
- **SmolTalk** (~460K conversations) from `HuggingFaceTB/smol-smoltalk`
- **MMLU** auxiliary_train (~100K) — multiple choice, assistant answers with letter
- **GSM8K** train x2 (~16K) — with `<<expr=result>>` tool calls parsed into python/output
- **CustomJSON** x2 (~2K) — identity conversations from local JSONL (optional)
- **SimpleSpelling** (200K) — synthetic spelling tasks
- **SpellingBee** (80K) — synthetic letter counting with verification

**Data sources (val):** SmolTalk test (24K), MMLU test (5.2K), GSM8K test (420)

**Produces:** `data/<dataset_name>/sft/` containing:
- `sft_train_000000.bin` + `sft_train_000000_mask.bin` (paired token + mask shards)
- `sft_val_000000.bin` + `sft_val_000000_mask.bin`
- `config.json`

---

### 4. `data/chat_eval_dataset.py` — Chat categorical eval pre-tokenization

Pre-tokenizes MMLU, ARC-Easy, and ARC-Challenge for single-token logit evaluation at inference time.

```bash
python data/chat_eval_dataset.py
```

**Requires:** The tokenizer from step 1.

Each problem is formatted as a chat conversation (user asks a multiple-choice question, no assistant reply), tokenized with `render_for_completion()`, and saved with metadata for answer-position logit scoring.

**Produces:** `data/<dataset_name>/chat_eval/` containing `MMLU.pt`, `ARC-Easy.pt`, `ARC-Challenge.pt`, and `config.json`.

---

## Script order

`dataset_and_vocab.py` must run first (it creates the tokenizer). The other three can run in any order after that.

## Other files

- `tokenizer.py` — BPE tokenizer library (not run directly; imported by the other scripts)
