"""
Pre-tokenize chat-formatted categorical evaluation data into .pt files.

This script downloads MMLU, ARC-Easy, and ARC-Challenge test sets, formats
each problem as a chat conversation (user asks MC question, no assistant
response), tokenizes with render_for_completion(), and saves per-task .pt
files into data/<dataset>/chat_eval/.

Unlike the CORE eval (which creates N sequences per problem for log-likelihood
scoring), chat eval creates 1 sequence per problem and scores by comparing
single-token logits at the answer position. This matches nanochat's
`run_categorical_eval` approach.

Prerequisites:
  - The tokenizer must already exist at data/<dataset>/tokenizer/
    (run download_dataset.py --tokenizer-only, or dataset_and_vocab.py first)

Usage:
  # Pre-tokenize and upload
  python data/chat_eval_dataset.py

  # Pre-tokenize only (no upload)
  python data/chat_eval_dataset.py --skip-upload

  # Use a different dataset name / HF user
  python data/chat_eval_dataset.py --dataset-name my_dataset --hf-user MyUser
"""

import os
import sys
import json
import time
import argparse

import torch

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Pre-tokenize chat-formatted categorical evaluation data into .pt files"
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
    help="Maximum sequence length; longer sequences are skipped (default: 2048)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, args.dataset_name)
TOKENIZER_DIR = os.path.join(DATASET_DIR, "tokenizer")
CHAT_EVAL_DIR = os.path.join(DATASET_DIR, "chat_eval")

print(f"Dataset name:     {args.dataset_name}")
print(f"Tokenizer dir:    {TOKENIZER_DIR}")
print(f"Output dir:       {CHAT_EVAL_DIR}")
print(f"Max seq len:      {args.max_seq_len}")
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
# Phase 2: Pre-tokenize all tasks
# ---------------------------------------------------------------------------

print("=== Phase 2: Pre-tokenize chat eval tasks ===\n")

from datasets import load_dataset

os.makedirs(CHAT_EVAL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

MMLU_LETTERS = ('A', 'B', 'C', 'D')

TASKS = [
    {
        "label": "MMLU",
        "load_fn": lambda: load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42),
        "format_fn": "mmlu",
        "random_baseline": 0.25,
    },
    {
        "label": "ARC-Easy",
        "load_fn": lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=42),
        "format_fn": "arc",
        "random_baseline": None,  # computed from data (variable num choices)
    },
    {
        "label": "ARC-Challenge",
        "load_fn": lambda: load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=42),
        "format_fn": "arc",
        "random_baseline": None,  # computed from data
    },
]


def format_mmlu_problem(row):
    """Format a single MMLU problem as a chat conversation."""
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
    conversation = {"messages": messages, "letters": list(MMLU_LETTERS)}
    gold_index = answer
    return conversation, gold_index


def format_arc_problem(row):
    """Format a single ARC problem as a chat conversation."""
    question = row["question"]
    choices = row["choices"]["text"]
    letters = row["choices"]["label"]
    answer_string = row["answerKey"]
    assert answer_string in letters
    user_message = render_mc(question, letters, choices)
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer_string},
    ]
    conversation = {"messages": messages, "letters": list(letters)}
    gold_index = letters.index(answer_string)
    return conversation, gold_index


FORMAT_FNS = {
    "mmlu": format_mmlu_problem,
    "arc": format_arc_problem,
}

# ---------------------------------------------------------------------------
# Pre-tokenize each task
# ---------------------------------------------------------------------------

max_seq_len = args.max_seq_len
manifest_tasks = []
total_examples = 0
total_time = 0

# Cache: letter string -> token ID (validated to be single token)
letter_token_cache = {}

def get_letter_token_id(letter):
    """Get the token ID for a single answer letter, with caching and validation."""
    if letter not in letter_token_cache:
        encoded = tokenizer.encode(letter)
        assert len(encoded) == 1, f"Letter '{letter}' encodes to {len(encoded)} tokens, expected 1"
        letter_token_cache[letter] = encoded[0]
    return letter_token_cache[letter]


for task_spec in TASKS:
    t0 = time.time()
    label = task_spec["label"]
    format_fn = FORMAT_FNS[task_spec["format_fn"]]

    print(f"  {label}... ", end='', flush=True)

    # Load dataset
    ds = task_spec["load_fn"]()

    # Pre-tokenize all examples
    sequences = []
    num_skipped = 0
    num_choices_set = set()

    for idx in range(len(ds)):
        row = ds[idx]
        conversation, gold_index = format_fn(row)
        letters = conversation["letters"]
        num_choices_set.add(len(letters))

        # Tokenize: render_for_completion strips the assistant message and appends <|assistant_start|>
        ids = tokenizer.render_for_completion(conversation)

        # Skip if too long
        if len(ids) > max_seq_len:
            num_skipped += 1
            continue

        # The answer position is the last token (where the model predicts the answer letter)
        answer_pos = len(ids) - 1

        # Get token IDs for all answer letters
        letter_token_ids = [get_letter_token_id(letter) for letter in letters]

        sequences.append({
            'tokens': ids,
            'answer_pos': answer_pos,
            'letter_token_ids': letter_token_ids,
            'gold': gold_index,
            'example_idx': idx,
        })

    # Compute random baseline
    if task_spec["random_baseline"] is not None:
        random_baseline = task_spec["random_baseline"]
    else:
        # Average 1/num_choices across all examples
        if len(num_choices_set) == 1:
            random_baseline = 1.0 / num_choices_set.pop()
        else:
            # Variable number of choices - compute average
            total_inv = sum(1.0 / len(seq['letter_token_ids']) for seq in sequences)
            random_baseline = total_inv / len(sequences) if sequences else 0.25

    # Save per-task .pt file
    task_data = {
        'task_type': 'categorical',
        'label': label,
        'num_examples': len(sequences),
        'random_baseline': random_baseline,
        'sequences': sequences,
    }

    pt_filename = f"{label}.pt"
    pt_path = os.path.join(CHAT_EVAL_DIR, pt_filename)
    torch.save(task_data, pt_path)

    elapsed = time.time() - t0
    total_time += elapsed
    total_examples += len(sequences)
    file_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    skip_str = f", {num_skipped} skipped (too long)" if num_skipped > 0 else ""
    print(f"{len(ds)} problems, {len(sequences)} sequences, {file_size_mb:.1f} MB, {elapsed:.1f}s{skip_str}")

    manifest_tasks.append({
        'label': label,
        'file': pt_filename,
        'task_type': 'categorical',
        'num_examples': len(sequences),
        'random_baseline': random_baseline,
    })

# Write config.json manifest
manifest = {
    'max_seq_len': max_seq_len,
    'bos_id': BOS_ID,
    'tasks': manifest_tasks,
}
manifest_path = os.path.join(CHAT_EVAL_DIR, "config.json")
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print(f"\n  Total: {total_examples} sequences across {len(TASKS)} tasks in {total_time:.1f}s")
print(f"  Saved to {CHAT_EVAL_DIR}/")
print()

# ---------------------------------------------------------------------------
# Phase 3: Upload to HuggingFace
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

    num_files = len(os.listdir(CHAT_EVAL_DIR))
    print(f"\n=== Phase 3: Upload chat_eval/ ({num_files} files) to {HF_REPO_ID} ===\n")

    api.upload_folder(
        folder_path=CHAT_EVAL_DIR,
        path_in_repo="chat_eval",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\n  Done! Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
