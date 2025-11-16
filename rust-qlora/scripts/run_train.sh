#!/usr/bin/env bash
set -euo pipefail
source ~/.venvs/qlora/bin/activate
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export CUDA_VISIBLE_DEVICES=0
# Use FlashAttention if present
python -c "import flash_attn" >/dev/null 2>&1 && export FLASH_ATTENTION=1 || true

mkdir -p out
python train.py --cfg configs/llama8b.yml 2>&1 | tee -a out/train.log
