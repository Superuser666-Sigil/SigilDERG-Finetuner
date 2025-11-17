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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

# Create output directory
mkdir -p eval_out

# Evaluation loop
while true; do
  sleep 1800   # every 30 minutes
  echo "[eval] $(date)"
  echo "Working directory: $(pwd)"
  echo "Python: $(which python)"
  python gen_eval_samples.py
  # Enhanced evaluation with functionality checking and parallelization
  python eval_rust.py eval_out/samples.jsonl --sample-n 16 --check-func --seed 0 | tee -a eval_out/metrics.jsonl || \
    sigilderg-eval eval_out/samples.jsonl --sample-n 16 --check-func --seed 0 | tee -a eval_out/metrics.jsonl
done
