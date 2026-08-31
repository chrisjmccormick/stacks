#!/usr/bin/env bash
# One-shot env + data prep for qwen-arithmetic on a CUDA box. This builds the
# venv the data/ publishers need, and the one the RAW H100 BASELINE runs in
# (baselines/20260813_raw-h100/). transformers must NOT land in this env — the
# hf-vllm baseline has its own, baselines/20260813_hf-vllm/setup_hf_vllm.sh.
#
# The shipped speedrun, train_qwen_arithmetic.py, needs NONE of this: it runs on
# a free Colab T4 against what Colab already ships, and downloads its weights,
# data and attention kernel finished. Run this only to REBUILD or REPUBLISH
# those artifacts, or to run the H100 baseline.
#
# From a fresh clone:
#   git clone https://github.com/chrisjmccormick/stacks.git
#   cd stacks/stacks/qwen-arithmetic
#   bash setup.sh
#
# What it does:
#   1. Installs uv if needed, creates .venv, uv sync --extra gpu
#      (torch 2.10 cu128 + kernels hub FA — no flash-attn wheel, no transformers)
#   2. Runs data/prepare_model.py       (HF checkpoint -> fp16 banks + tokenizer,
#                                        published to ChrisMcCormick/qwen-arithmetic-t4)
#   3. Runs data/prepare_arithmetic.py  (basic-arithmetic -> pre-tokenized parquet,
#                                        published back to the dataset repo)
#   4. If a GPU is present, smokes kernels-hub FA2/FA3 (the H100 baseline's
#      attention; the T4 line uses a prebuilt sm75 flash-attention-turing)
#
# The bf16 banks the H100 baseline wants are a SEPARATE build — after this, run
#   python baselines/20260813_raw-h100/data/prepare_model.py
#
# Artifacts land in ~/.cache/qwen-arithmetic/data/. HF_TOKEN in the environment
# is used for Hub auth when set (anonymous works for these public assets, with
# lower rate limits).
set -eu
cd "$(dirname "$0")"

# MassedCompute (and some other images) ship ~/.config owned by root; uv needs
# to write ~/.config/uv. Reclaim when we can; ignore if we can't.
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi
$SUDO chown -R "$(id -u):$(id -g)" ~/.config 2>/dev/null || true
mkdir -p ~/.config

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv --version

[ -d .venv ] || uv venv --python 3.12
uv sync --extra gpu
# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import importlib.util
assert importlib.util.find_spec("flash_attn") is None, (
    "flash-attn wheel must NOT be installed — it collides with kernels-hub FA"
)
assert importlib.util.find_spec("transformers") is None, (
    "transformers must NOT be installed — tokenizer is tokenizers/rustbpe, "
    "and a model load can pull flash-attn"
)
print("flash_attn / transformers absent (good)")
PY

python data/prepare_model.py
python data/prepare_arithmetic.py

python - <<'PY'
import torch
from kernels import get_kernel

if not torch.cuda.is_available():
    print("CUDA not available — skipped FA smoke (data prep is host-safe; "
          "re-run setup.sh on the GPU box before training)")
    raise SystemExit(0)

print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
      "| device:", torch.cuda.get_device_name(0))
# kernels>=0.15 requires an explicit major version (or revision=...).
_fa2 = get_kernel("kernels-community/flash-attn2", version=1)
_fa2i = _fa2 if hasattr(_fa2, "flash_attn_varlen_func") else _fa2.flash_attn_interface
assert hasattr(_fa2i, "flash_attn_varlen_func")
assert hasattr(_fa2i, "flash_attn_with_kvcache")
print("kernels-community/flash-attn2 OK")
cc_major, _ = torch.cuda.get_device_capability()
if cc_major >= 9:
    # Prefer community FA3 — varunneal's kernel-type repo can 401 without access.
    _k = get_kernel("kernels-community/flash-attn3", version=1)
    fa3 = getattr(_k, "flash_attn_interface", _k)
    assert hasattr(fa3, "_flash_attn_forward")
    print("kernels-community/flash-attn3 OK (H100 path)")
else:
    print(f"FA3 train kernel not smoked here (cc_major={cc_major}); "
          "the H100 baseline loads community FA3 for sm8x itself")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok:", bool((x @ x).sum().isfinite()))
PY

echo "=== qwen-arithmetic ready ==="
echo "    source .venv/bin/activate"
echo "    python baselines/20260813_raw-h100/data/prepare_model.py   # bf16 banks"
echo "    python baselines/20260813_raw-h100/train_qwen_arithmetic.py"
echo ""
echo "The shipped T4 speedrun does not use this venv:"
echo "    colab run --gpu T4 train_qwen_arithmetic.py --timeout 1h"
