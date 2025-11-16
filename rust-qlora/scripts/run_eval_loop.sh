#!/usr/bin/env bash
set -euo pipefail
source ~/.venvs/qlora/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

mkdir -p eval_out

while true; do
  sleep 1800   # every 30 minutes
  echo "[eval] $(date)"
  python gen_eval_samples.py
  # Enhanced evaluation with functionality checking
  python eval_rust.py eval_out/samples.jsonl 16 true | tee -a eval_out/metrics.jsonl
done
