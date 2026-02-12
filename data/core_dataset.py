"""
Pre-tokenize the CORE evaluation benchmark for modded-nanogpt.

This script downloads the raw eval bundle (JSONL + YAML + CSV), renders
prompts with fewshot examples, tokenizes everything using the project's
BPE tokenizer, and saves one .pt file per task into data/<dataset>/core_eval/.

The output files can then be uploaded to HuggingFace alongside the training
data so that train_gpt.py can load pre-tokenized sequences directly, skipping
all template rendering and tokenization at evaluation time.

Prerequisites:
  - The tokenizer must already exist at data/<dataset>/tokenizer/
    (run download_dataset.py --tokenizer-only, or dataset_and_vocab.py first)

Usage:
  # Pre-tokenize and upload
  python data/core_dataset.py

  # Pre-tokenize only (no upload)
  python data/core_dataset.py --skip-upload

  # Use a different dataset name / HF user
  python data/core_dataset.py --dataset-name my_dataset --hf-user MyUser
"""

import os
import sys
import csv
import json
import random
import time
import argparse
import shutil
import tempfile
import urllib.request
import zipfile

import torch
import yaml
from jinja2 import Template

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Pre-tokenize the CORE eval benchmark and optionally upload to HuggingFace"
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
    help="Maximum sequence length; longer sequences are truncated from the left (default: 2048)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_BUNDLE_DIR = os.path.join(SCRIPT_DIR, "eval_bundle")
DATASET_DIR = os.path.join(SCRIPT_DIR, args.dataset_name)
TOKENIZER_DIR = os.path.join(DATASET_DIR, "tokenizer")
CORE_EVAL_DIR = os.path.join(DATASET_DIR, "core_eval")

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

print(f"Dataset name:     {args.dataset_name}")
print(f"Tokenizer dir:    {TOKENIZER_DIR}")
print(f"Eval bundle dir:  {EVAL_BUNDLE_DIR}")
print(f"Output dir:       {CORE_EVAL_DIR}")
print(f"Max seq len:      {args.max_seq_len}")
print()

# ---------------------------------------------------------------------------
# Phase 1: Download eval bundle
# ---------------------------------------------------------------------------

print("=== Phase 1: Download eval bundle ===\n")

if os.path.exists(EVAL_BUNDLE_DIR):
    print(f"  Eval bundle already exists at {EVAL_BUNDLE_DIR}")
else:
    print(f"  Downloading eval bundle from {EVAL_BUNDLE_URL}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "eval_bundle.zip")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, EVAL_BUNDLE_DIR)
    print(f"  Eval bundle extracted to {EVAL_BUNDLE_DIR}")

print()

# ---------------------------------------------------------------------------
# Phase 2: Pre-tokenize
# ---------------------------------------------------------------------------

print("=== Phase 2: Pre-tokenize CORE benchmark ===\n")

# Load tokenizer
if not os.path.exists(TOKENIZER_DIR):
    print(f"ERROR: Tokenizer not found at {TOKENIZER_DIR}")
    print("Run download_dataset.py --tokenizer-only first, or run dataset_and_vocab.py.")
    sys.exit(1)

from tokenizer import RustBPETokenizer
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
BOS_ID = tokenizer.get_bos_token_id()
print(f"  Loaded tokenizer (BOS_ID={BOS_ID})")

# Load task config
config_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
tasks = config['icl_tasks']

# Load random baseline values
eval_meta_data_path = os.path.join(EVAL_BUNDLE_DIR, "eval_meta_data.csv")
random_baselines = {}
with open(eval_meta_data_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        random_baselines[row['Eval Task']] = float(row['Random baseline'])

data_base_path = os.path.join(EVAL_BUNDLE_DIR, "eval_data")

# ---------------------------------------------------------------------------
# Prompt rendering utilities (from train_gpt.py / nanochat core_eval.py)
# ---------------------------------------------------------------------------

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    return [template.render(choice=choice, **context) for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    return [template.render(context=context_option, **context)
            for context_option in item['context_options']]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompt for a language modeling task."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


# ---------------------------------------------------------------------------
# Sequence batching utilities (from train_gpt.py / nanochat core_eval.py)
# ---------------------------------------------------------------------------

def find_common_length(token_sequences, direction='left'):
    """Find the length of the common prefix or suffix across token sequences."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    return [tokens_with], [start_idx], [end_idx]


# ---------------------------------------------------------------------------
# Pre-tokenize all tasks
# ---------------------------------------------------------------------------

os.makedirs(CORE_EVAL_DIR, exist_ok=True)

max_seq_len = args.max_seq_len
manifest_tasks = []
total_sequences = 0
total_time = 0

for task in tasks:
    t0 = time.time()
    label = task['label']
    task_type = task['icl_task_type']
    num_fewshot = task['num_fewshot'][0]
    continuation_delimiter = task.get('continuation_delimiter', ' ')

    print(f"  {label} ({num_fewshot}-shot, {task_type})... ", end='', flush=True)

    # Load and shuffle data with the same seed as train_gpt.py
    data_path = os.path.join(data_base_path, task['dataset_uri'])
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)

    # Pre-tokenize all examples
    sequences = []
    num_seqs_per_example = {}
    gold_labels = {}

    for idx in range(len(data)):
        item = data[idx]

        # Sample fewshot examples (same seeding as train_gpt.py)
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_type == 'multiple_choice':
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            tokens_list, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
        elif task_type == 'schema':
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            tokens_list, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
        elif task_type == 'language_modeling':
            prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            tokens_list, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Truncate and record each sequence
        for seq_j, (t, s, e) in enumerate(zip(tokens_list, start_idxs, end_idxs)):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                t = t[-max_seq_len:]
                s = s - num_to_crop
                e = e - num_to_crop
                assert s >= 0 and e >= 0
            sequences.append({
                'tokens': t,
                'start_idx': s,
                'end_idx': e,
                'example_idx': idx,
                'seq_idx': seq_j,
            })

        num_seqs_per_example[idx] = len(tokens_list)
        if task_type in ['multiple_choice', 'schema']:
            gold_labels[idx] = item['gold']

    # Save per-task .pt file
    random_baseline = random_baselines[label]
    task_data = {
        'task_type': task_type,
        'label': label,
        'random_baseline': random_baseline,
        'num_examples': len(data),
        'sequences': sequences,
        'num_seqs_per_example': num_seqs_per_example,
        'gold_labels': gold_labels,
    }

    pt_filename = f"{label}.pt"
    pt_path = os.path.join(CORE_EVAL_DIR, pt_filename)
    torch.save(task_data, pt_path)

    elapsed = time.time() - t0
    total_time += elapsed
    total_sequences += len(sequences)
    file_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"{len(data)} examples, {len(sequences)} sequences, {file_size_mb:.1f} MB, {elapsed:.1f}s")

    manifest_tasks.append({
        'label': label,
        'file': pt_filename,
        'task_type': task_type,
        'num_examples': len(data),
        'num_sequences': len(sequences),
    })

# Write config.json manifest
manifest = {
    'max_seq_len': max_seq_len,
    'bos_id': BOS_ID,
    'tasks': manifest_tasks,
}
manifest_path = os.path.join(CORE_EVAL_DIR, "config.json")
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print(f"\n  Total: {total_sequences} sequences across {len(tasks)} tasks in {total_time:.1f}s")
print(f"  Saved to {CORE_EVAL_DIR}/")
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

    # Create the repo if it doesn't already exist (no-op if it does)
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    api = HfApi()

    num_files = len(os.listdir(CORE_EVAL_DIR))
    print(f"\n=== Phase 3: Upload core_eval/ ({num_files} files) to {HF_REPO_ID} ===\n")

    api.upload_folder(
        folder_path=CORE_EVAL_DIR,
        path_in_repo="core_eval",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\n  Done! Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
