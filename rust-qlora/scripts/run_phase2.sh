#!/usr/bin/env bash
# Phase 2 training script - loads from Phase 1 checkpoint
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

# Only enable hf_transfer if the package is installed
if python -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
    echo "hf_transfer available - fast downloads enabled"
else
    export HF_HUB_ENABLE_HF_TRANSFER=0
    echo "hf_transfer not available - using standard downloads (install with: pip install hf_transfer)"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Use FlashAttention if present
python -c "import flash_attn" >/dev/null 2>&1 && export FLASH_ATTENTION=1 || true

# Check if Phase 1 checkpoint exists
PHASE1_CHECKPOINT="out/llama8b-rust-qlora-phase1/checkpoint-12000"
if [ ! -d "$PHASE1_CHECKPOINT" ]; then
    echo "Error: Phase 1 checkpoint not found at $PHASE1_CHECKPOINT"
    echo "Please run Phase 1 training first:"
    echo "  sigilderg-train --cfg configs/llama8b-phase1.yml"
    exit 1
fi

# Update Phase 2 config to load from Phase 1
# This is a simple approach - you can also manually edit the config
python -c "
import yaml
with open('configs/llama8b-phase2.yml', 'r') as f:
    cfg = yaml.safe_load(f)
if 'misc' not in cfg:
    cfg['misc'] = {}
cfg['misc']['load_from'] = '$PHASE1_CHECKPOINT'
with open('configs/llama8b-phase2.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

mkdir -p out
echo "Starting Phase 2 training (sharpening) from Phase 1 checkpoint..."
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
sigilderg-train --cfg configs/llama8b-phase2.yml 2>&1 | tee -a out/phase2_train.log

