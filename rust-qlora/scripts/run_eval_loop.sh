#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Check if we're already running in tmux
if [ -n "${TMUX:-}" ]; then
    # Already in tmux, just run the loop
    RUN_IN_TMUX=true
else
    # Not in tmux, check if we should create a session
    RUN_IN_TMUX=false
    
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "Warning: tmux is not installed. Running without tmux session."
        echo "Install with: sudo apt-get install tmux"
        RUN_IN_TMUX=true  # Just run directly
    else
        # Check if session already exists
        if tmux has-session -t eval-loop 2>/dev/null; then
            echo "Eval loop session 'eval-loop' already exists. Attaching..."
            tmux attach -t eval-loop
            exit 0
        fi
        
        # Create new tmux session and run this script inside it
        echo "Creating tmux session 'eval-loop' for evaluation..."
        tmux new-session -d -s eval-loop -c "$PROJECT_DIR" "bash $SCRIPT_DIR/run_eval_loop.sh"
        echo "Eval loop started in tmux session 'eval-loop'"
        echo "Attaching to session..."
        sleep 1
        tmux attach -t eval-loop
        exit 0
    fi
fi

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
  
  # Generate evaluation samples
  python gen_eval_samples.py --model-path "${MODEL_PATH:-out/llama8b-rust-qlora-phase1}"
  
  # Enhanced evaluation with:
  # - Functionality checking
  # - Error type classification and tracking
  # - Detailed error logs for analysis
  # - Parallel evaluation (auto-detected)
  python eval_rust.py eval_out/samples.jsonl \
    --sample-n 16 \
    --check-func \
    --seed 0 \
    --save-errors eval_out/errors.jsonl \
    | tee -a eval_out/metrics.jsonl || \
  sigilderg-eval eval_out/samples.jsonl \
    --sample-n 16 \
    --check-func \
    --seed 0 \
    --save-errors eval_out/errors.jsonl \
    | tee -a eval_out/metrics.jsonl
done
