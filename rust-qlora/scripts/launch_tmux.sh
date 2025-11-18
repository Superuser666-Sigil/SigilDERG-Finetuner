#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Install it with: sudo apt-get install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t rustft 2>/dev/null; then
    echo "Session 'rustft' already exists. Attaching..."
    tmux attach -t rustft
    exit 0
fi

# Create new tmux session with training script
tmux new-session -d -s rustft -c "$PROJECT_DIR" "bash $SCRIPT_DIR/run_train.sh"

# Split window and run eval loop
tmux split-window -v -t rustft -c "$PROJECT_DIR" "bash $SCRIPT_DIR/run_eval_loop.sh"

# Attach to session
tmux attach -t rustft
