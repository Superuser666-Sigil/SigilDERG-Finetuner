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
if tmux has-session -t tensorboard 2>/dev/null; then
    echo "TensorBoard session 'tensorboard' already exists. Attaching..."
    tmux attach -t tensorboard
    exit 0
fi

# Set environment variables to suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2  # Suppress INFO and WARNING messages
export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN warnings

# Default log directory (can be overridden)
LOG_DIR="${1:-out/}"
PORT="${2:-6006}"

# Create new tmux session with TensorBoard
tmux new-session -d -s tensorboard -c "$PROJECT_DIR" \
    "tensorboard --logdir $LOG_DIR --port $PORT --host 0.0.0.0 2>&1 | grep -v -E '(oneDNN|cuFFT|cuDNN|cuBLAS|computation_placer|Unable to register|absl::InitializeLog|All log messages before)' || true"

echo "TensorBoard started in tmux session 'tensorboard'"
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "To attach: tmux attach -t tensorboard"
echo "To view: http://<server-ip>:$PORT"

