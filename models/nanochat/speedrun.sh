#!/bin/bash

# [Chris] - This is what I currently use for nanochat speedruns. Currently 
#           it's configured for d12, bs32, fp8 enabled.

# Timestamped run name--update this as needed.
# Set this to "dummy" to skip wandb logging
WANDB_RUN="$(date +%Y%m%d-%H%M%S)-nanochat-d12-bs32-fp8"


# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# add uv to PATH (the installer puts it in ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8  # Blocking
python -m nanochat.dataset -n 370 & # Non-blocking
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Redirect all subsequent stdout+stderr to both terminal and log file
LOG_FILE="./logs/$WANDB_RUN.log"
exec > >(tee -a "$LOG_FILE") 2>&1

cat runs/speedrun.sh
echo -e "\n\n ========================== Training $WANDB_RUN ========================== \n\n"

torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=32 \
    --run=$WANDB_RUN \
    --model-tag=$WANDB_RUN \
    --device-batch-size=32 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=8.25 \
    --fp8

# evaluate the model: CORE metric, BPB on train/val, and draw samples
CORE_EVAL_START=$(date +%s)
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=32
CORE_EVAL_END=$(date +%s)
CORE_EVAL_TIME=$((CORE_EVAL_END - CORE_EVAL_START))
printf "CORE eval time: %ds (%.2fm)\n" "$CORE_EVAL_TIME" "$(awk "BEGIN{printf \"%.2f\", $CORE_EVAL_TIME/60}")"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
SFT_TRAIN_START=$(date +%s)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=32 --run=$WANDB_RUN
SFT_TRAIN_END=$(date +%s)
SFT_TRAIN_TIME=$((SFT_TRAIN_END - SFT_TRAIN_START))
printf "SFT training time: %ds (%.2fm)\n" "$SFT_TRAIN_TIME" "$(awk "BEGIN{printf \"%.2f\", $SFT_TRAIN_TIME/60}")"

SFT_EVAL_START=$(date +%s)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
SFT_EVAL_END=$(date +%s)
SFT_EVAL_TIME=$((SFT_EVAL_END - SFT_EVAL_START))
printf "SFT eval time: %ds (%.2fm)\n" "$SFT_EVAL_TIME" "$(awk "BEGIN{printf \"%.2f\", $SFT_EVAL_TIME/60}")"

# -----------------------------------------------------------------------------
# Log pipeline timings to wandb
echo ""
echo "============ Pipeline Timings ============"
printf "CORE eval time:    %ds (%.2fm)\n" "$CORE_EVAL_TIME" "$(awk "BEGIN{printf \"%.2f\", $CORE_EVAL_TIME/60}")"
printf "SFT training time: %ds (%.2fm)\n" "$SFT_TRAIN_TIME" "$(awk "BEGIN{printf \"%.2f\", $SFT_TRAIN_TIME/60}")"
printf "SFT eval time:     %ds (%.2fm)\n" "$SFT_EVAL_TIME" "$(awk "BEGIN{printf \"%.2f\", $SFT_EVAL_TIME/60}")"
echo "=========================================="
echo ""

python -c "
import wandb
run_name = '$WANDB_RUN'
if run_name != 'dummy':
    run = wandb.init(project='nanochat-timings', name=run_name, config={})
    run.log({
        'core_eval_time_s': $CORE_EVAL_TIME,
        'sft_train_time_s': $SFT_TRAIN_TIME,
        'sft_eval_time_s': $SFT_EVAL_TIME,
    })
    run.finish()
    print('Logged pipeline timings to wandb')
else:
    print('Skipping wandb logging (WANDB_RUN=dummy)')
"

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
