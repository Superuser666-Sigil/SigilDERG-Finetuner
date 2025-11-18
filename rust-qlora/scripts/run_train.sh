#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Try to activate virtual environment (check multiple common locations)
if [ -f ~/.venvs/qlora/bin/activate ]; then
    source ~/.venvs/qlora/bin/activate
elif [ -f ~/venv/qlora/bin/activate ]; then
    source ~/venv/qlora/bin/activate
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using existing virtual environment: $VIRTUAL_ENV"
elif command -v pyenv &> /dev/null; then
    # Try pyenv if available
    if pyenv versions --bare | grep -q qlora; then
        eval "$(pyenv init -)"
        pyenv activate qlora || true
    fi
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # Ensure unbuffered output for real-time metrics display

# Only enable hf_transfer if the package is installed
if python -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
    echo "hf_transfer available - fast downloads enabled"
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "hf_transfer not available - using standard downloads (install with: pip install hf_transfer)"
fi

detect_gpu_count() {
  python - <<'PY' || echo 0
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
}

ensure_accelerate_config() {
  local processes="$1"
  local cfg_dir="${HF_HOME:-$HOME/.cache/huggingface}/accelerate"
  if command -v accelerate >/dev/null 2>&1; then
    mkdir -p "$cfg_dir"
    if [ ! -f "$cfg_dir/default_config.yaml" ]; then
      echo "Creating Accelerate default config for ${processes} GPUs..."
      accelerate config default \
        --compute_environment LOCAL_MACHINE \
        --distributed_type MULTI_GPU \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes "$processes" \
        --use_cpu False || true
    fi
  else
    echo "Warning: accelerate CLI not installed; falling back to single-GPU launch."
    NUM_GPUS=1
  fi
}

NUM_GPUS=${NUM_GPUS:-$(detect_gpu_count)}
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "$NUM_GPUS" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  else
    CUDA_VISIBLE_DEVICES=0
  fi
  export CUDA_VISIBLE_DEVICES
fi

# Use FlashAttention if present
python -c "import flash_attn" >/dev/null 2>&1 && export FLASH_ATTENTION=1 || true

# Create output directory
mkdir -p out

TRAIN_CFG=${TRAIN_CFG:-configs/llama8b-phase1.yml}
EXTRA_ARGS=("$@")

if [ "$NUM_GPUS" -gt 1 ]; then
  ensure_accelerate_config "$NUM_GPUS"
  LAUNCH_CMD=(accelerate launch --num_machines 1 --num_processes "$NUM_GPUS")
  echo "Detected $NUM_GPUS GPUs -> launching with Accelerate."
else
  LAUNCH_CMD=()
  echo "Detected single GPU -> launching with standard Trainer."
fi

echo "Starting training at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

if [ "${#LAUNCH_CMD[@]}" -gt 0 ]; then
  # Use -- to separate accelerate args from the command
  "${LAUNCH_CMD[@]}" -- python -m rust_qlora.train --cfg "$TRAIN_CFG" "${EXTRA_ARGS[@]}" 2>&1 | tee -a out/train.log
else
  python -m rust_qlora.train --cfg "$TRAIN_CFG" "${EXTRA_ARGS[@]}" 2>&1 | tee -a out/train.log
fi
