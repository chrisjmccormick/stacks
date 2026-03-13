"""
Download all pre-tokenized data needed for a training run from HuggingFace.

This downloads everything in the HF repo: tokenizer, pre-training shards,
validation shards, CORE eval, chat eval, and SFT data.

Usage:
  # Download with defaults (climbmix, 20 training shards)
  python data/download_dataset.py

  # Download a different dataset
  python data/download_dataset.py --dataset-name fineweb_edu_32k_8_370

  # Download all training shards
  python data/download_dataset.py --num-train-shards -1

  # Download only the tokenizer
  python data/download_dataset.py --tokenizer-only
"""

import os
import argparse
from huggingface_hub import HfApi, hf_hub_download, login

# Known training shard prefixes per dataset
TRAIN_PREFIXES = {
    "climbmix_32k_8_170": "climbmix/train_",
    "fineweb_edu_32k_8_370": "fineweb_edu/train_",
}

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-tokenized dataset from HuggingFace"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="climbmix_32k_8_170",
        help="Name of the dataset to download (default: climbmix_32k_8_170)"
    )
    parser.add_argument(
        "--num-train-shards", type=int, default=20,
        help="Max training shards to download (-1 = all, default: 20)"
    )
    parser.add_argument(
        "--tokenizer-only", action="store_true",
        help="Only download the tokenizer files"
    )
    parser.add_argument(
        "--hf-user", type=str, default="ChrisMcCormick",
        help="HuggingFace username/org (default: ChrisMcCormick)"
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    repo_id = f"{args.hf_user}/{args.dataset_name}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Repo:        {repo_id}")
    print(f"Dataset dir: {dataset_dir}")

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    train_prefix = TRAIN_PREFIXES.get(args.dataset_name)

    to_download = []
    for fname in repo_files:
        if args.tokenizer_only and not fname.startswith("tokenizer/"):
            continue
        if train_prefix and fname.startswith(train_prefix) and args.num_train_shards >= 0:
            try:
                shard_idx = int(fname[len(train_prefix):].split(".")[0])
                if shard_idx >= args.num_train_shards:
                    continue
            except ValueError:
                pass
        if not os.path.exists(os.path.join(dataset_dir, fname)):
            to_download.append(fname)

    if not to_download:
        print("All files already downloaded.")
        return

    print(f"Downloading {len(to_download)} files...")
    for fname in to_download:
        print(f"  {fname}")
        hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset", local_dir=dataset_dir)

    print(f"Done! Downloaded to {dataset_dir}")

if __name__ == "__main__":
    main()
