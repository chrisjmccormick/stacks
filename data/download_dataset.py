"""
Download the pre-tokenized FineWeb-Edu dataset, custom vocabulary, and
pre-tokenized CORE eval benchmark needed for training and evaluation.

This downloads the outputs of dataset_and_vocab.py and core_dataset.py so you
don't have to re-run the full tokenizer training + dataset tokenization
pipeline (~hours of work).

Downloads into ./data/:
  - Tokenizer files       -> ./data/<dataset_name>/tokenizer/
  - Dataset shards        -> ./data/<dataset_name>/
  - CORE eval (pre-tok)   -> ./data/<dataset_name>/core_eval/

Usage:
  # Download everything (all training shards + validation + tokenizer + core eval)
  python stacks/modded/data/download_dataset.py

  # Download a specific dataset variant
  python stacks/modded/data/download_dataset.py --dataset-name fineweb_edu_16k

  # Download only the first N training shards (+ validation + tokenizer)
  python stacks/modded/data/download_dataset.py --num-train-shards 10

  # Download only the tokenizer (no dataset shards or core eval)
  python stacks/modded/data/download_dataset.py --tokenizer-only
"""

import os
import argparse
from huggingface_hub import HfApi, hf_hub_download, login


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(repo_id, filename, local_dir):
    """Download a single file from the HF repo, skipping if it already exists."""
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        print(f"  Skipping {filename} (already exists)")
        return
    print(f"  Downloading {filename}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-tokenized FineWeb-Edu dataset and vocabulary from HuggingFace"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="fineweb_edu_32k_8_370",
        help="Name of the dataset to download (default: fineweb_edu_32k_8_370)"
    )
    parser.add_argument(
        "--num-train-shards", type=int, default=-1,
        help="Number of training shards to download (-1 = all, default: -1)"
    )
    parser.add_argument(
        "--tokenizer-only", action="store_true",
        help="Only download the tokenizer files (no dataset shards)"
    )
    parser.add_argument(
        "--hf-user", type=str, default="ChrisMcCormick",
        help="HuggingFace username/org for the repo (default: ChrisMcCormick)"
    )
    args = parser.parse_args()

    # Log in using HF_TOKEN from the environment (needed for private repos)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    HF_REPO_ID = f"{args.hf_user}/{args.dataset_name}"

    # Everything goes into ./modded/<dataset_name>/ (co-located with the script)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(SCRIPT_DIR, args.dataset_name)
    TOKENIZER_DIR = os.path.join(DATASET_DIR, "tokenizer")

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print(f"HuggingFace repo: {HF_REPO_ID}")
    print(f"Dataset dir:      {DATASET_DIR}")
    print(f"Tokenizer dir:    {TOKENIZER_DIR}")

    # ------------------------------------------------------------------
    # List all files in the repo
    # ------------------------------------------------------------------
    print("\n  Listing repo files...")
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")

    # ------------------------------------------------------------------
    # 1) Download pre-tokenized CORE eval
    # ------------------------------------------------------------------
    CORE_EVAL_DIR = os.path.join(DATASET_DIR, "core_eval")
    os.makedirs(CORE_EVAL_DIR, exist_ok=True)

    core_eval_files = sorted([f for f in repo_files if f.startswith("core_eval/")])
    if core_eval_files:
        print(f"\n=== Downloading pre-tokenized CORE eval to {CORE_EVAL_DIR} ({len(core_eval_files)} files) ===")
        for fname in core_eval_files:
            download_file(HF_REPO_ID, fname, DATASET_DIR)
        print("  CORE eval download complete.")
    else:
        print("\n  No core_eval/ files found in repo, skipping CORE eval download.")

    # ------------------------------------------------------------------
    # 2) Download tokenizer files
    # ------------------------------------------------------------------
    print(f"\n=== Downloading tokenizer to {TOKENIZER_DIR} ===")

    tokenizer_repo_files = [f for f in repo_files if f.startswith("tokenizer/")]
    for fname in tokenizer_repo_files:
        download_file(HF_REPO_ID, fname, DATASET_DIR)

    print("  Tokenizer download complete.")

    if args.tokenizer_only:
        print("\n--tokenizer-only specified, skipping dataset shards.")
        return

    # ------------------------------------------------------------------
    # 3) Download dataset files
    # ------------------------------------------------------------------
    print(f"\n=== Downloading dataset to {DATASET_DIR} ===")

    # Download config.json
    if "config.json" in repo_files:
        download_file(HF_REPO_ID, "config.json", DATASET_DIR)

    # Separate val and train shard files
    val_files = sorted([f for f in repo_files if f.startswith("val_") and f.endswith(".bin")])
    train_files = sorted([f for f in repo_files if f.startswith("train_") and f.endswith(".bin")])

    # Download validation shard(s)
    print(f"\n  --- Validation shards ({len(val_files)} files) ---")
    for fname in val_files:
        download_file(HF_REPO_ID, fname, DATASET_DIR)

    # Download training shards
    if args.num_train_shards == -1:
        num_to_download = len(train_files)
    else:
        num_to_download = min(args.num_train_shards, len(train_files))

    print(f"\n  --- Training shards ({num_to_download}/{len(train_files)} files) ---")
    for fname in train_files[:num_to_download]:
        download_file(HF_REPO_ID, fname, DATASET_DIR)

    print(f"\nDone! Downloaded to {DATASET_DIR}")
    print(f"  Validation shards: {len(val_files)}")
    print(f"  Training shards:   {num_to_download}/{len(train_files)}")

if __name__ == "__main__":
    main()
