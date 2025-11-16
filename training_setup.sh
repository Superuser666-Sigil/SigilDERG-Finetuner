# System base
sudo apt-get update && sudo apt-get install -y \
  git build-essential wget curl tmux htop pkg-config libssl-dev \
  libffi-dev unzip python3.10 python3.10-venv

python3.10 -m venv ~/.venvs/qlora && source ~/.venvs/qlora/bin/activate
python -m pip install --upgrade pip wheel

# PyTorch 2.4 + CUDA 12.1 wheels (good on H100)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Core libs
pip install "transformers>=4.44.0" "accelerate>=0.33.0" "datasets>=2.20.0" \
            "bitsandbytes>=0.43.1" "trl>=0.9.6" "peft>=0.12.0" \
            "evaluate>=0.4.2" "sentencepiece" "tqdm" "huggingface_hub>=0.24.0" \
            "protobuf<5" "numpy" "pandas" "scikit-learn" "jsonlines" "typer[all]" "rich"

# Optional: FlashAttention 2 for H100 (nice speedup; skip if wheel mismatch)
pip install "flash-attn>=2.5.6" --no-build-isolation || echo "FA2 optional install failed; continuing with SDPA."

# Rust toolchain for eval
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup component add clippy rustfmt
