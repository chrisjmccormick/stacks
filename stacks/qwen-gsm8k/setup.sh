#!/usr/bin/env bash
# One-shot env + data prep for qwen-gsm8k.
#
# From a fresh clone:
#   git clone https://github.com/chrisjmccormick/stacks.git
#   cd stacks/stacks/qwen-gsm8k
#   bash setup.sh
#
# What it does:
#   1. Installs uv if needed, creates .venv, uv sync --extra gpu
#      (torch 2.10 cu128 + kernels hub FA — no flash-attn wheel, no transformers)
#   2. Runs data/prepare_model.py  (HF checkpoint -> banked safetensors + tokenizer)
#   3. Runs data/prepare_gsm8k.py  (GSM8K -> pre-tokenized parquet)
#   4. If a GPU is present, smokes kernels-hub FA2/FA3
#
# Artifacts land in ~/.cache/qwen-gsm8k/data/. HF_TOKEN in the environment is
# used for Hub auth when set (anonymous works for these public assets, with
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
python data/prepare_gsm8k.py

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
          "train_qwen_gsm8k.py loads community FA3 for sm8x itself")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok:", bool((x @ x).sum().isfinite()))
PY

echo "=== qwen-gsm8k ready ==="
echo "    source .venv/bin/activate"
echo "    python train_qwen_gsm8k.py"
