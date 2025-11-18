#!/bin/bash
# System setup script for SigilDERG-Finetuner
# This script installs system dependencies and sets up the Python environment

set -euo pipefail

echo "Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
  git build-essential wget curl tmux htop pkg-config libssl-dev \
  libffi-dev unzip make libbz2-dev libreadline-dev libsqlite3-dev zlib1g-dev liblzma-dev

# Install pyenv if not present
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    # Add to shell profile for persistence
    if ! grep -q 'pyenv init' ~/.bashrc 2>/dev/null; then
        echo '' >> ~/.bashrc
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    fi
else
    echo "pyenv already installed."
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install Python 3.12.11 via pyenv
echo "Installing Python 3.12.11 via pyenv..."
if ! pyenv versions --bare | grep -q "^3.12.11$"; then
    pyenv install 3.12.11
else
    echo "Python 3.12.11 already installed via pyenv."
fi

# Set Python 3.12.11 as local version
pyenv local 3.12.11

# Create virtual environment using pyenv's Python
# Use absolute path to avoid tilde expansion issues
VENV_DIR="${VENV_DIR:-$HOME/.venvs/qlora}"
echo "Creating virtual environment at $VENV_DIR..."
mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify Python version
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Upgrade pip
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
# For CUDA 12.8 (NVIDIA 570+ drivers): Use cu128 wheels
echo "Installing PyTorch 2.7.1 with CUDA 12.8 support..."
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

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
