"""
Pre-tokenize SFT conversation data into .bin shards with parallel mask files.

This script downloads all SFT training/validation datasets, formats them as
conversations, tokenizes with render_conversation(), and writes paired
token + mask .bin shards into data/<dataset>/sft/.

The output files can then be uploaded to HuggingFace alongside the training
data so that the decoderstack training script can load pre-tokenized sequences
directly, skipping all data loading, conversation formatting, and tokenization
at training time.

Prerequisites:
  - The tokenizer must already exist at data/<dataset>/tokenizer/
    (run download_dataset.py --tokenizer-only, or dataset_and_vocab.py first)

Usage:
  # Pre-tokenize and upload
  python data/sft_dataset.py

  # Pre-tokenize only (no upload)
  python data/sft_dataset.py --skip-upload

  # Use a different dataset name / HF user
  python data/sft_dataset.py --dataset-name my_dataset --hf-user MyUser

  # Customize shard size or sequence length
  python data/sft_dataset.py --max-seq-len 2048 --shard-size 10000000
"""

import os
import sys
import re
import json
import random
import time
import argparse
import urllib.request

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Pre-tokenize SFT conversation data into .bin shards with mask files"
)
parser.add_argument(
    "--dataset-name", type=str, default="fineweb_edu_32k_8_370",
    help="Name of the dataset directory containing the tokenizer (default: fineweb_edu_32k_8_370)"
)
parser.add_argument(
    "--hf-user", type=str, default="ChrisMcCormick",
    help="HuggingFace username/org for upload (default: ChrisMcCormick)"
)
parser.add_argument(
    "--skip-upload", action="store_true",
    help="Skip uploading to HuggingFace"
)
parser.add_argument(
    "--max-seq-len", type=int, default=2048,
    help="Maximum conversation length in tokens (default: 2048)"
)
parser.add_argument(
    "--shard-size", type=int, default=10_000_000,
    help="Number of tokens per shard (default: 10M)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, args.dataset_name)
TOKENIZER_DIR = os.path.join(DATASET_DIR, "tokenizer")
SFT_DIR = os.path.join(DATASET_DIR, "sft")

print(f"Dataset name:     {args.dataset_name}")
print(f"Tokenizer dir:    {TOKENIZER_DIR}")
print(f"Output dir:       {SFT_DIR}")
print(f"Max seq len:      {args.max_seq_len}")
print(f"Shard size:       {args.shard_size:,} tokens")
print()

# ---------------------------------------------------------------------------
# Phase 1: Load tokenizer
# ---------------------------------------------------------------------------

print("=== Phase 1: Load tokenizer ===\n")

if not os.path.exists(TOKENIZER_DIR):
    print(f"ERROR: Tokenizer not found at {TOKENIZER_DIR}")
    print("Run download_dataset.py --tokenizer-only first, or run dataset_and_vocab.py.")
    sys.exit(1)

from tokenizer import RustBPETokenizer
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
BOS_ID = tokenizer.get_bos_token_id()
print(f"  Loaded tokenizer (vocab_size={tokenizer.get_vocab_size()}, BOS_ID={BOS_ID})")
print()

# ---------------------------------------------------------------------------
# Phase 2: Load all datasets and build conversation lists
# ---------------------------------------------------------------------------

print("=== Phase 2: Load datasets ===\n")

from datasets import load_dataset

# ---------------------------------------------------------------------------
# render_mc: shared multiple-choice formatting (same as nanochat/tasks/common.py)
# ---------------------------------------------------------------------------

def render_mc(question, letters, choices):
    """
    Render a multiple-choice question.
    Letter *after* choice (better binding for small models).
    No whitespace before letter (critical for tokenizer consistency).
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query

# ---------------------------------------------------------------------------
# Data source loaders
# Each returns a list of conversation dicts with 'messages' key.
# ---------------------------------------------------------------------------

def load_smoltalk(split):
    """Load SmolTalk conversations from HuggingFace."""
    t0 = time.time()
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
    ds = ds.shuffle(seed=42)
    conversations = []
    for row in ds:
        messages = row["messages"]
        # Validate structure (same checks as nanochat SmolTalk task)
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role
            assert isinstance(message["content"], str)
        conversations.append({"messages": messages})
    elapsed = time.time() - t0
    print(f"  SmolTalk ({split}): {len(conversations):,} conversations ({elapsed:.1f}s)")
    return conversations


MMLU_LETTERS = ('A', 'B', 'C', 'D')

def load_mmlu_sft(subset, split):
    """Load MMLU as SFT conversations (user asks MC question, assistant answers with letter)."""
    t0 = time.time()
    ds = load_dataset("cais/mmlu", subset, split=split)
    if subset == "auxiliary_train":
        ds = ds.map(lambda row: row['train'], remove_columns=['train'])
    ds = ds.shuffle(seed=42)
    conversations = []
    for row in ds:
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]  # index 0,1,2,3
        assert len(choices) == 4
        user_message = render_mc(question, MMLU_LETTERS, choices)
        assistant_message = MMLU_LETTERS[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        conversations.append({"messages": messages})
    elapsed = time.time() - t0
    label = f"MMLU ({subset}/{split})"
    print(f"  {label}: {len(conversations):,} conversations ({elapsed:.1f}s)")
    return conversations


def load_gsm8k(subset, split):
    """Load GSM8K with tool-call parsing (<<expr=result>> -> python parts)."""
    t0 = time.time()
    ds = load_dataset("openai/gsm8k", subset, split=split)
    ds = ds.shuffle(seed=42)
    conversations = []
    for row in ds:
        question = row['question']
        answer = row['answer']
        # Parse tool calls from answer
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]
        conversations.append({"messages": messages})
    elapsed = time.time() - t0
    print(f"  GSM8K ({subset}/{split}): {len(conversations):,} conversations ({elapsed:.1f}s)")
    return conversations


def load_custom_json(filepath):
    """Load conversations from a JSONL file. Returns empty list if file missing."""
    t0 = time.time()
    conversations = []
    if not os.path.exists(filepath):
        print(f"  CustomJSON: file not found at {filepath}, skipping")
        return conversations
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)
            assert isinstance(messages, list)
            assert len(messages) >= 2
            for i, message in enumerate(messages):
                assert "role" in message
                assert "content" in message
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message["role"] == expected_role
                assert isinstance(message["content"], str)
            conversations.append({"messages": messages})
    elapsed = time.time() - t0
    print(f"  CustomJSON: {len(conversations):,} conversations from {filepath} ({elapsed:.1f}s)")
    return conversations


# ---------------------------------------------------------------------------
# Synthetic data: spelling tasks
# ---------------------------------------------------------------------------

# Constants matching nanochat/tasks/spellingbee.py
LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


def download_word_list():
    """Download the word list, caching locally."""
    filename = WORD_LIST_URL.split("/")[-1]
    cache_path = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(cache_path):
        print(f"  Word list already cached at {cache_path}")
    else:
        print(f"  Downloading word list from {WORD_LIST_URL}...")
        urllib.request.urlretrieve(WORD_LIST_URL, cache_path)
        print(f"  Saved to {cache_path}")
    with open(cache_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def generate_simple_spelling(words, size, split):
    """Generate SimpleSpelling conversations: 'Spell the word: X' -> 'X:a,p,p,l,e'."""
    t0 = time.time()
    # Match nanochat: shuffle words with seed 42 for SimpleSpelling
    words_shuffled = list(words)
    rng = random.Random(42)
    rng.shuffle(words_shuffled)

    TEST_RANDOM_SEED_OFFSET = 10_000_000
    conversations = []
    for index in range(size):
        seed = index if split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        word = rng.choice(words_shuffled)
        word_letters = ",".join(list(word))
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]
        conversations.append({"messages": messages})
    elapsed = time.time() - t0
    print(f"  SimpleSpelling ({split}, {size:,}): {len(conversations):,} conversations ({elapsed:.1f}s)")
    return conversations


def generate_spelling_bee(words, size, split):
    """Generate SpellingBee conversations: counting letter occurrences with manual + Python verification."""
    t0 = time.time()
    TEST_RANDOM_SEED_OFFSET = 10_000_000
    conversations = []
    for index in range(size):
        seed = index if split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # Pick a random word and letter
        word = rng.choice(words)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
        count = word.count(letter)

        # Create user message with data augmentation
        template = rng.choice(USER_MSG_TEMPLATES)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5:
            user_msg += "?"

        # Build assistant response with manual counting + Python verification
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"
        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        assistant_parts.append({"type": "python_output", "text": str(count)})
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts},
        ]
        conversations.append({"messages": messages})
    elapsed = time.time() - t0
    print(f"  SpellingBee ({split}, {size:,}): {len(conversations):,} conversations ({elapsed:.1f}s)")
    return conversations


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

# Download word list for synthetic tasks
words = download_word_list()

# Identity conversations file (optional)
identity_filepath = os.path.join(SCRIPT_DIR, "identity_conversations.jsonl")
# Also check in the dataset directory
if not os.path.exists(identity_filepath):
    identity_filepath_alt = os.path.join(DATASET_DIR, "identity_conversations.jsonl")
    if os.path.exists(identity_filepath_alt):
        identity_filepath = identity_filepath_alt

# --- Training data (matching nanochat chat_sft.py mixture) ---
print("\n  --- Training sources ---")
train_sources = {}
train_sources["SmolTalk"] = load_smoltalk("train")                                    # ~460K
train_sources["MMLU"] = load_mmlu_sft("auxiliary_train", "train")                      # ~100K
train_sources["GSM8K"] = load_gsm8k("main", "train")                                  # ~8K (x2 below)
train_sources["CustomJSON"] = load_custom_json(identity_filepath)                      # ~1K (x2 below)
train_sources["SimpleSpelling"] = generate_simple_spelling(words, 200000, "train")     # 200K
train_sources["SpellingBee"] = generate_spelling_bee(words, 80000, "train")            # 80K

# Build unified train conversation list with oversampling (matching nanochat)
# GSM8K x2, CustomJSON x2
train_conversations = []
train_conversations.extend(train_sources["SmolTalk"])
train_conversations.extend(train_sources["MMLU"])
train_conversations.extend(train_sources["GSM8K"])
train_conversations.extend(train_sources["GSM8K"])       # 2nd epoch
train_conversations.extend(train_sources["CustomJSON"])
train_conversations.extend(train_sources["CustomJSON"])   # 2nd epoch
train_conversations.extend(train_sources["SimpleSpelling"])
train_conversations.extend(train_sources["SpellingBee"])

# Deterministic shuffle (matching TaskMixture seed=42)
rng = random.Random(42)
rng.shuffle(train_conversations)
print(f"\n  Train total: {len(train_conversations):,} conversations")

# --- Validation data ---
print("\n  --- Validation sources ---")
val_sources = {}
val_sources["SmolTalk"] = load_smoltalk("test")                                       # ~24K
val_sources["MMLU"] = load_mmlu_sft("all", "test")                                    # ~14K (capped at 5.2K below)
val_sources["GSM8K"] = load_gsm8k("main", "test")                                     # ~1.3K (capped at 420 below)

# Cap MMLU and GSM8K val sizes (matching nanochat chat_sft.py)
val_sources["MMLU"] = val_sources["MMLU"][:5200]
val_sources["GSM8K"] = val_sources["GSM8K"][:420]

val_conversations = []
val_conversations.extend(val_sources["SmolTalk"])
val_conversations.extend(val_sources["MMLU"])
val_conversations.extend(val_sources["GSM8K"])

rng_val = random.Random(42)
rng_val.shuffle(val_conversations)
print(f"\n  Val total: {len(val_conversations):,} conversations")
print()

# ---------------------------------------------------------------------------
# Phase 3: Tokenize and write .bin shards
# ---------------------------------------------------------------------------

print("=== Phase 3: Tokenize and write shards ===\n")

os.makedirs(SFT_DIR, exist_ok=True)

# Reuse write_datafile from dataset_and_vocab.py
def write_datafile(filename, toks):
    """
    Saves token data as a .bin file.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of tokens
    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


SHARD_SIZE = args.shard_size

# Shard writing state
shard_index = 0
token_buffer = np.empty((SHARD_SIZE,), dtype=np.uint16)
mask_buffer = np.empty((SHARD_SIZE,), dtype=np.uint16)
token_count = 0
progress_bar = None
current_split = "val"

def pack_tokens_and_mask(ids_np, mask_np):
    """Pack one conversation's tokens + mask into shard buffers, writing full shards to disk."""
    global shard_index, token_count, progress_bar

    assert len(ids_np) == len(mask_np), "ids and mask must have same length"
    pos = 0
    while pos < len(ids_np):
        space = SHARD_SIZE - token_count
        n = min(len(ids_np) - pos, space)

        token_buffer[token_count:token_count + n] = ids_np[pos:pos + n]
        mask_buffer[token_count:token_count + n] = mask_np[pos:pos + n]
        token_count += n
        pos += n

        if progress_bar is None:
            progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(n)

        # Shard full — write it out
        if token_count == SHARD_SIZE:
            tok_filename = os.path.join(SFT_DIR, f"sft_{current_split}_{shard_index:06d}.bin")
            mask_filename = os.path.join(SFT_DIR, f"sft_{current_split}_{shard_index:06d}_mask.bin")
            write_datafile(tok_filename, token_buffer)
            write_datafile(mask_filename, mask_buffer)
            shard_index += 1
            token_count = 0
            progress_bar.close()
            progress_bar = None


def flush_shard():
    """Flush an incomplete shard to disk."""
    global shard_index, token_count, progress_bar
    if token_count > 0:
        tok_filename = os.path.join(SFT_DIR, f"sft_{current_split}_{shard_index:06d}.bin")
        mask_filename = os.path.join(SFT_DIR, f"sft_{current_split}_{shard_index:06d}_mask.bin")
        write_datafile(tok_filename, token_buffer[:token_count])
        write_datafile(mask_filename, mask_buffer[:token_count])
        shard_index += 1
        token_count = 0
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None


# Track stats
stats = {
    "max_seq_len": args.max_seq_len,
    "shard_size": SHARD_SIZE,
    "bos_id": BOS_ID,
    "vocab_size": tokenizer.get_vocab_size(),
}

# Process val first, then train (same pattern as dataset_and_vocab.py)
for split, conversations in [("val", val_conversations), ("train", train_conversations)]:
    current_split = split
    t0 = time.time()
    print(f"  --- Processing {split} split ({len(conversations):,} conversations) ---")

    total_tokens_split = 0
    num_truncated = 0
    shard_start = shard_index

    for conv in tqdm(conversations, desc=f"Tokenizing {split}", unit="conv"):
        ids, mask = tokenizer.render_conversation(conv, max_tokens=args.max_seq_len)

        if len(ids) == args.max_seq_len:
            num_truncated += 1

        total_tokens_split += len(ids)

        ids_np = np.array(ids, dtype=np.uint16)
        mask_np = np.array(mask, dtype=np.uint16)
        pack_tokens_and_mask(ids_np, mask_np)

    # Flush incomplete shard at split boundary
    flush_shard()

    num_shards = shard_index - shard_start
    elapsed = time.time() - t0
    print(f"  {split}: {len(conversations):,} conversations, {total_tokens_split:,} tokens, "
          f"{num_shards} shards, {num_truncated:,} truncated, {elapsed:.1f}s")
    print()

    stats[f"{split}_conversations"] = len(conversations)
    stats[f"{split}_tokens"] = total_tokens_split
    stats[f"{split}_shards"] = num_shards
    stats[f"{split}_truncated"] = num_truncated

# Record per-source conversation counts
stats["train_source_counts"] = {
    "SmolTalk": len(train_sources["SmolTalk"]),
    "MMLU": len(train_sources["MMLU"]),
    "GSM8K": len(train_sources["GSM8K"]) * 2,  # 2 epochs
    "CustomJSON": len(train_sources["CustomJSON"]) * 2,  # 2 epochs
    "SimpleSpelling": len(train_sources["SimpleSpelling"]),
    "SpellingBee": len(train_sources["SpellingBee"]),
}
stats["val_source_counts"] = {
    "SmolTalk": len(val_sources["SmolTalk"]),
    "MMLU": len(val_sources["MMLU"]),
    "GSM8K": len(val_sources["GSM8K"]),
}

# Write config.json
config_path = os.path.join(SFT_DIR, "config.json")
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)

print(f"  Config saved to {config_path}")
print(f"  Total shards: {shard_index}")
print()

# ---------------------------------------------------------------------------
# Phase 4: Upload to HuggingFace
# ---------------------------------------------------------------------------

if args.skip_upload:
    print("--skip-upload specified, skipping HuggingFace upload.")
else:
    from huggingface_hub import HfApi, create_repo, login

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set in environment. Upload may fail if not already logged in.")
    else:
        login(token=hf_token)

    HF_REPO_ID = f"{args.hf_user}/{args.dataset_name}"
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    api = HfApi()

    num_files = len(os.listdir(SFT_DIR))
    print(f"\n=== Phase 4: Upload sft/ ({num_files} files) to {HF_REPO_ID} ===\n")

    api.upload_folder(
        folder_path=SFT_DIR,
        path_in_repo="sft",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\n  Done! Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
