#!/usr/bin/env bash
# Env for train_qwen_arithmetic-hf-vllm.py — a SEPARATE venv from setup.sh's:
# this one needs vllm + transformers + peft, which the raw speedrun's env
# forbids (its setup asserts transformers is absent).
#
#   bash setup_hf_vllm.sh
#   source .venv-hf-vllm/bin/activate
#   python train_qwen_arithmetic-hf-vllm.py
#
# The script downloads its own data (HF dataset ChrisMcCormick/basic-arithmetic
# -> ~/.cache/qwen-arithmetic/data/) and model on first run — no prep step.
set -eu
cd "$(dirname "$0")"

command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv --version

[ -d .venv-hf-vllm ] || uv venv .venv-hf-vllm --python 3.12
VENV_PY=".venv-hf-vllm/bin/python"
uv pip install --python "$VENV_PY" -r requirements-hf-vllm.txt

"$VENV_PY" - <<'PY'
import torch, vllm, transformers, peft
print("torch:", torch.__version__, "| vllm:", vllm.__version__,
      "| transformers:", transformers.__version__, "| peft:", peft.__version__)
assert torch.version.cuda and torch.version.cuda.startswith("12"), \
    f"expected a cu12 torch, got {torch.version.cuda}"
PY

echo "=== qwen-arithmetic (hf-vllm variant) ready ==="
echo "    source .venv-hf-vllm/bin/activate"
echo "    python train_qwen_arithmetic-hf-vllm.py"
