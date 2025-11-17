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
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Use FlashAttention if present
python -c "import flash_attn" >/dev/null 2>&1 && export FLASH_ATTENTION=1 || true

# Create output directory
mkdir -p out

# Run training
echo "Starting training at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
sigilderg-train --cfg configs/llama8b.yml 2>&1 | tee -a out/train.log
