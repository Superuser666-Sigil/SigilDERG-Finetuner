#!/usr/bin/env bash
# Phase 2 training script - loads from Phase 1 checkpoint
set -euo pipefail
source ~/.venvs/qlora/bin/activate
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export CUDA_VISIBLE_DEVICES=0

# Use FlashAttention if present
python -c "import flash_attn" >/dev/null 2>&1 && export FLASH_ATTENTION=1 || true

# Check if Phase 1 checkpoint exists
PHASE1_CHECKPOINT="out/llama8b-rust-qlora-phase1/checkpoint-12000"
if [ ! -d "$PHASE1_CHECKPOINT" ]; then
    echo "Error: Phase 1 checkpoint not found at $PHASE1_CHECKPOINT"
    echo "Please run Phase 1 training first:"
    echo "  python train.py --cfg configs/llama8b-phase1.yml"
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
python train.py --cfg configs/llama8b-phase2.yml 2>&1 | tee -a out/phase2_train.log

