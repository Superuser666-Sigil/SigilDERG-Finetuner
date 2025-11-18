#!/bin/bash
# System setup script for SigilDERG-Finetuner
# This script installs system dependencies and sets up the Python environment

set -euo pipefail

echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
  git build-essential wget curl tmux htop pkg-config libssl-dev \
  libffi-dev unzip python3.12 python3.12-venv

# Create virtual environment
VENV_DIR="${VENV_DIR:-~/.venvs/qlora}"
echo "Creating virtual environment at $VENV_DIR..."
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 12.6 (required for PyTorch 2.9 / NVIDIA 570+ drivers):
echo "Installing PyTorch 2.9.0 with CUDA 12.6 support..."
pip install torch==2.9.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# Install Python dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Ensuring distributed training tooling is available..."
# hf_transfer avoids slow downloads when HF_HUB_ENABLE_HF_TRANSFER=1
pip install --upgrade accelerate hf_transfer || true

ACCEL_CFG_DIR="${HF_HOME:-$HOME/.cache/huggingface}/accelerate"
if command -v accelerate >/dev/null 2>&1; then
  mkdir -p "$ACCEL_CFG_DIR"
  if [ ! -f "$ACCEL_CFG_DIR/default_config.yaml" ]; then
    echo "Creating default Accelerate config (multi-GPU, bf16)..."
    accelerate config default \
      --compute_environment LOCAL_MACHINE \
      --distributed_type MULTI_GPU \
      --mixed_precision bf16 \
      --num_processes 4 \
      --num_machines 1 \
      --use_cpu False || true
  fi
fi

# Optional: FlashAttention 2 for H100 (nice speedup; skip if wheel mismatch)
echo "Attempting to install FlashAttention 2 (optional)..."
pip install "flash-attn>=2.5.6" --no-build-isolation || echo "FlashAttention 2 installation failed; continuing with SDPA."

# Install the package in editable mode
echo "Installing sigilderg-finetuner package..."
pip install -e .

# Rust toolchain for evaluation
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source "$HOME/.cargo/env"
    rustup default stable
    rustup component add clippy rustfmt
else
    echo "Rust toolchain already installed."
fi

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "You can now use the package:"
echo "  python -m rust_qlora.train --cfg rust-qlora/configs/llama8b-phase1.yml"
echo "  or"
echo "  sigilderg-train --cfg rust-qlora/configs/llama8b-phase1.yml"
