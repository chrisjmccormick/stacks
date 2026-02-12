# ▂▂▂▂▂▂▂▂▂▂▂▂

# Dataset


### Source

# `nanochat/dataset.py`

"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

#from nanochat.common import get_base_dir

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

# `list_parquet_files`

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

# `parquets_iter_batched`

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------

# `download_single_file`

def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False

# ### Download

# `main`

# Invoked with:

# ```python
# # Download the first ~2B characters of pretraining dataset
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# python -m nanochat.dataset -n 8
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # Approximately 350 shards are needed for 10B tokens of data for pretraining.
# # The maximum total number of shards available in the entire dataset is 1822.
# python -m nanochat.dataset -n 370 &
# DATASET_DOWNLOAD_PID=$!
# ```

#if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    # parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    # parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    # args = parser.parse_args()

# -----------------------------------------------------------------------------
# Command line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Download, tokenize, and prepare FineWeb-Edu dataset")
parser.add_argument("--vocab-size", type=int, default=32768,
                    help="Vocabulary size (default: 32768 = 2^15)")
parser.add_argument("--num-shards", type=int, default=370,
                    help="Total parquet shards to download. Last shard becomes val. (default: 370)")
parser.add_argument("--tok-train-shards", type=int, default=8,
                    help="Number of shards to download before tokenizer training (default: 8, matching nanochat)")
parser.add_argument("--dataset-name", type=str, default="fineweb_edu_32k_8_370",
                    help="Name for the output dataset directory (default: fineweb_edu_32k_8_370)")
parser.add_argument("--num-workers", type=int, default=4,
                    help="Number of parallel download workers (default: 4)")
parser.add_argument("--max-chars", type=int, default=2_000_000_000,
                    help="Max characters for tokenizer training (default: 2B)")
parser.add_argument("--doc-cap", type=int, default=10_000,
                    help="Max characters per document for tokenizer training (default: 10,000)")
parser.add_argument("--hf-user", type=str, default="ChrisMcCormick",
                    help="HuggingFace username/org for upload (default: ChrisMcCormick)")
parser.add_argument("--skip-upload", action="store_true",
                    help="Skip uploading to HuggingFace")
args = parser.parse_args()

# Output directories (co-located with this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(SCRIPT_DIR, args.dataset_name)
TOKENIZER_DIR = os.path.join(DATA_CACHE_DIR, "tokenizer")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print(f"Dataset name:       {args.dataset_name}")
print(f"Vocab size:         {args.vocab_size}")
print(f"Tok train shards:   {args.tok_train_shards}")
print(f"Total shards:       {args.num_shards}")
print(f"Output dir:         {DATA_CACHE_DIR}")
print(f"Tokenizer dir:      {TOKENIZER_DIR}")
print()

import time

def download_shards(n):
    """Download the first `n` parquet shards (skips any that already exist)."""
    n = MAX_SHARD + 1 if n == -1 else min(n, MAX_SHARD + 1)
    ids = list(range(n))
    t0 = time.time()
    print(f"Downloading up to {len(ids)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids)
    successful = sum(1 for r in results if r)
    print(f"Done! {successful}/{len(ids)} shards in {DATA_DIR} ({time.time() - t0:.1f}s)\n")

# -------------------------------------------------------------------------
# Phase 1: Download just enough shards for tokenizer training.
#
# This replicates nanochat's two-phase approach from speedrun.sh:
#   python -m nanochat.dataset -n 8    ← download 8 shards
#   python -m scripts.tok_train        ← train tokenizer (sees shards 0-6 as train, shard 7 as val)
#   python -m nanochat.dataset -n 370  ← download rest in background
#
# parquets_iter_batched(split="train") uses list_parquet_files()[:-1],
# so with 8 shards on disk it trains on shards 0-6 (~1.75B chars).
# If we downloaded all 370 first, it would include shard 7 in training
# data, producing a different tokenizer.
# -------------------------------------------------------------------------
print(f"=== Phase 1: Download {args.tok_train_shards} shards for tokenizer training ===\n")
download_shards(args.tok_train_shards)


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

from tokenizer import RustBPETokenizer

# `get_tokenizer`

def get_tokenizer(tokenizer_dir=None):
    if tokenizer_dir is None:
        tokenizer_dir = TOKENIZER_DIR
    return RustBPETokenizer.from_directory(tokenizer_dir)

# `get_token_bytes`

def get_token_bytes(tokenizer_dir=None, device="cpu"):
    import torch
    if tokenizer_dir is None:
        tokenizer_dir = TOKENIZER_DIR
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written during tokenizer training."
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes

## ⚙️ / ▶ Train


# `scripts/tok_train.py`

# From speedrun.sh:

# ```python
# # train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
# python -m scripts.tok_train
# ```

"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch

#from nanochat.tokenizer import RustBPETokenizer
# from nanochat.common import get_base_dir
#from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

print("\n======== Tokenizer Training ========\n")

# Tokenizer training args (from CLI)
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

# `text_iterator`

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return

# -----------------------------------------------------------------------------
# Check if tokenizer already exists (skip training if so)
tokenizer_pkl_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
token_bytes_pt_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

if os.path.exists(tokenizer_pkl_path) and os.path.exists(token_bytes_pt_path):
    print(f"\nTokenizer already exists at {TOKENIZER_DIR}, skipping training.")
    tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
else:
    text_iter = text_iterator()

    # -------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save the tokenizer to disk
    tokenizer.save(TOKENIZER_DIR)

    # -------------------------------------------------------------------------
    # Quick inline sanity check
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    # -----------------------------------------------------------------------------
    # One more thing: we wish to cache a mapping from token id to number of bytes of that token
    # for efficient evaluation of bits per byte. Unlike the typical mean loss, this
    # allows us to report a loss that is invariant to the vocab size of the tokenizer.
    # The bits per byte on the validation set is then one of the primary metrics we care about.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    with open(token_bytes_pt_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_pt_path}")

    # Log report
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    report_data = [
        {"train_time": train_time},
        {"num_special_tokens": len(special_set)},
        {
            "token_bytes_min": int(token_bytes_nonzero.min().item()),
            "token_bytes_max": int(token_bytes_nonzero.max().item()),
            "token_bytes_mean": token_bytes_nonzero.mean().item(),
            "token_bytes_std": token_bytes_nonzero.std().item(),
        }
    ]
    print(report_data)

## Evaluate

# `scripts/tok_eval.py`

# From speedrun.sh:

# ```python
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval
# ```

"""
Evaluate compression ratio of the tokenizer.
"""

#from nanochat.tokenizer import get_tokenizer, RustBPETokenizer
#from nanochat.dataset import parquets_iter_batched

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 ‘정보의 균형’을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 ‘무엇이 쟁점인지’, ‘누가 무엇을 주장하는지’, ‘사실은 무엇인지’**를 명확히 전달하는 데 집중합니다.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square’’—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere’s primary productivity.
""".strip()



# The tokenizer was trained on data from earlier shards, so it has seen this data
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)
val_docs = next(parquets_iter_batched(split="val"))
val_text = "\n".join(val_docs)

all_text = [
    ("news", news_text),
    ("korean", korean_text),
    ("code", code_text),
    ("math", math_text),
    ("science", science_text),
    ("fwe-train", train_text),
]
if val_text:
    all_text.append(("fwe-val", val_text))

# Try out current default compared to GPT-2 and GPT-4 tokenizers
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt2", "gpt4", "ours"]:

    if tokenizer_name == "gpt2":
        tokenizer = RustBPETokenizer.from_pretrained("gpt2") # gpt-2 base model tokenizer
    elif tokenizer_name == "gpt4":
        tokenizer = RustBPETokenizer.from_pretrained("cl100k_base") # gpt-4 base model tokenizer
    else:
        tokenizer = get_tokenizer()

    vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio
        }

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Print vocab sizes
print(f"\nVocab sizes:")
print(f"GPT-2: {vocab_sizes['gpt2']}")
print(f"GPT-4: {vocab_sizes['gpt4']}")
print(f"Ours: {vocab_sizes['ours']}")

#`print_comparison`

def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better, negative means worse)
        # Using tokens: fewer tokens is better, so we calculate (baseline_tokens - ours_tokens) / baseline_tokens
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")



# Print comparisons
print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)

print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)

# Log to report
#from nanochat.report import get_report

lines = []
for baseline_name in ["GPT-2", "GPT-4"]:
    baseline_key = baseline_name.lower().replace('-', '')
    baseline_results = tokenizer_results[baseline_key]
    ours_results = tokenizer_results['ours']
    lines.append(f"### Comparison with {baseline_name}")
    lines.append("")
    lines.append("| Text Type | Bytes | " + baseline_name + " Tokens | " + baseline_name + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |")
    lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100
        lines.append(f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |")
    lines.append("")
report_markdown = "\n".join(lines)
# get_report().log(section="Tokenizer evaluation", data=[
#     report_markdown,
# ])

print(report_markdown)


# -------------------------------------------------------------------------
# Phase 2: Download remaining shards for pre-tokenization.
# With all shards on disk, val becomes the last shard (e.g. shard 369)
# and train is everything else (shards 0-368). This matches how nanochat's
# base_train.py sees the data after the full download completes.
# -------------------------------------------------------------------------
print(f"\n=== Phase 2: Download remaining shards (up to {args.num_shards} total) ===\n")
download_shards(args.num_shards)

# ================================================================================
#    Pre-tokenize fineweb-edu parquets into .bin shards for modded-nanogpt
#
#    Reads from the already-downloaded fineweb-edu parquet shards (above) and
#    tokenizes with the custom-trained vocabulary (above) to produce .bin files
#    in the format expected by the modded-nanogpt dataloader.
# ================================================================================

import os
import json
import numpy as np
from tqdm import tqdm

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ------------------------------------------
# Configuration
# ------------------------------------------

SHARD_SIZE = 10**8  # 100M tokens per shard

# DATA_CACHE_DIR and TOKENIZER_DIR are already defined at the top from CLI args

# ------------------------------------------
# Load the custom-trained tokenizer
# ------------------------------------------

tokenizer = get_tokenizer(TOKENIZER_DIR)
bos_id = tokenizer.get_bos_token_id()
tok_vocab_size = tokenizer.get_vocab_size()

print(f"Tokenizer vocab size: {tok_vocab_size}")
print(f"BOS token id: {bos_id}")
assert tok_vocab_size < 2**16, f"vocab size {tok_vocab_size} too large for uint16"

# ------------------------------------------
# Save a small config file with vocab metadata for the training script
# This avoids needing to load the tokenizer pickle at training time.
# ------------------------------------------

config = {
    "vocab_size": tok_vocab_size,
    "bos_id": bos_id,
}
config_path = os.path.join(DATA_CACHE_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved vocab config to {config_path}")

# ------------------------------------------
# Iterate over parquets, tokenize, and write .bin shards
# ------------------------------------------
# Process validation split first (last parquet file), then training split.
# Shard 0 is always validation, shards 1+ are training.
#
# Key design choices:
#   - Batch-encode entire row groups at once for multi-threaded tokenization
#   - Flush val shard at the val/train boundary to keep splits cleanly separated
#   - Drop the last incomplete training shard (negligible data loss, avoids edge cases)

shard_index = 0
all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
token_count = 0
progress_bar = None

current_split = "val"  # updated by the outer loop

def pack_tokens(tokens_np):
    """Pack one document's tokens into the shard buffer, writing full shards to disk."""
    global shard_index, token_count, progress_bar

    pos = 0  # position within tokens_np
    while pos < len(tokens_np):
        # how many tokens can we fit in the current shard?
        space = SHARD_SIZE - token_count
        n = min(len(tokens_np) - pos, space)

        all_tokens_np[token_count:token_count + n] = tokens_np[pos:pos + n]
        token_count += n
        pos += n

        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(n)

        # shard is full — write it out
        if token_count == SHARD_SIZE:
            filename = os.path.join(DATA_CACHE_DIR, f"{current_split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            token_count = 0
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None

t0_tokenize = time.time()

for split in ["val", "train"]:
    current_split = split
    print(f"\n--- Processing {split} split ---")
    for batch in parquets_iter_batched(split=split):
        # Batch-encode all documents in this row group at once.
        # This uses tiktoken's encode_ordinary_batch with num_threads=8 internally,
        # giving a large speedup over encoding one document at a time.
        encoded_batch = tokenizer.encode(batch)
        for doc_tokens in encoded_batch:
            tokens_np = np.empty(len(doc_tokens) + 1, dtype=np.uint16)
            tokens_np[0] = bos_id
            tokens_np[1:] = doc_tokens
            pack_tokens(tokens_np)

    # After processing val, flush the (likely incomplete) val shard immediately.
    # This keeps val data cleanly separated from training data.
    if split == "val" and token_count > 0:
        filename = os.path.join(DATA_CACHE_DIR, f"val_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
        shard_index += 1
        token_count = 0
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None

# Drop the last incomplete training shard.
# The data loss is negligible (<100M tokens out of billions) and avoids
# edge cases in the training data loader with undersized shards.
num_complete_train_shards = shard_index - 1  # subtract 1 for the val shard
if token_count > 0:
    print(f"\nDropping incomplete last training shard "
          f"({token_count:,}/{SHARD_SIZE:,} tokens, "
          f"{token_count/SHARD_SIZE*100:.1f}% full)")
    if progress_bar is not None:
        progress_bar.close()

t1_tokenize = time.time()
print(f"\nDone! Wrote 1 val shard + {num_complete_train_shards} train shards to {DATA_CACHE_DIR}")
print(f"Tokenization took {t1_tokenize - t0_tokenize:.1f}s")
print(f"Config saved to {config_path}")

# ================================================================================
#    Upload processed dataset and vocabulary to HuggingFace
# ================================================================================

if args.skip_upload:
    print("\n--skip-upload specified, skipping HuggingFace upload.")
else:
    from huggingface_hub import HfApi, create_repo, login

    # Log in using HF_TOKEN from the environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set in environment. Upload may fail if not already logged in.")
    else:
        login(token=hf_token)

    HF_REPO_ID = f"{args.hf_user}/{args.dataset_name}"

    # Create the repo if it doesn't already exist (no-op if it does)
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    api = HfApi()

    # ------------------------------------------
    # 1) Upload tokenizer files (tokenizer.pkl and token_bytes.pt)
    # ------------------------------------------
    tokenizer_files = ["tokenizer.pkl", "token_bytes.pt"]

    print(f"\n--- Uploading tokenizer files from {TOKENIZER_DIR} ---")
    for fname in tokenizer_files:
        fpath = os.path.join(TOKENIZER_DIR, fname)
        if os.path.exists(fpath):
            print(f"Uploading {fname}...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"tokenizer/{fname}",
                repo_id=HF_REPO_ID,
                repo_type="dataset",
            )
            print(f"  -> uploaded tokenizer/{fname}")
        else:
            print(f"  WARNING: {fpath} not found, skipping.")

    # ------------------------------------------
    # 2) Upload dataset files (config.json + all .bin shards)
    # ------------------------------------------
    print(f"\n--- Uploading dataset files from {DATA_CACHE_DIR} ---")

    # Upload config.json
    config_file = os.path.join(DATA_CACHE_DIR, "config.json")
    if os.path.exists(config_file):
        print("Uploading config.json...")
        api.upload_file(
            path_or_fileobj=config_file,
            path_in_repo="config.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print("  -> uploaded config.json")

    # Upload all files in the data cache directory (skips already-uploaded files)
    print(f"Uploading folder: {DATA_CACHE_DIR}")
    api.upload_large_folder(
        folder_path=DATA_CACHE_DIR,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )

    print(f"\nDone! All files uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")