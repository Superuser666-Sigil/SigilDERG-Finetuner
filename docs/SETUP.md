# Setup Guide

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16 GB | 24+ GB |
| System RAM | 32 GB | 64+ GB |
| Storage | 100 GB | 500+ GB SSD |
| CPU | 8 cores | 16+ cores |

### Software Requirements

- Python 3.12.10 or later
- CUDA 11.8+ (for GPU training)
- Rust toolchain (for evaluation)
- Git

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Superuser666-Sigil/SigilDERG-Finetuner.git
cd SigilDERG-Finetuner
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Package

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e .[dev]

# With evaluation dependencies
pip install -e .[evaluation]

# Full ecosystem installation
pip install -e .[ecosystem]

# With Flash Attention (Linux only)
pip install -e .[flash-attention]
```

### 4. Install Rust (for evaluation)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable
```

### 5. Configure HuggingFace Access

```bash
# Option 1: Environment variable
export HF_TOKEN="hf_your_token_here"

# Option 2: huggingface-cli login
pip install huggingface_hub
huggingface-cli login
```

## Verification

### Check Installation

```bash
python -c "import rust_qlora; print(f'Version: {rust_qlora.__version__}')"
```

### Check GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Check Rust

```bash
rustc --version
cargo --version
```

## Ecosystem Setup

### Install sigil-pipeline

```bash
pip install sigil-pipeline>=2.3.0
```

### Install human-eval-rust

```bash
pip install human-eval-rust>=2.3.0
```

### Verify Ecosystem

```python
from sigil_pipeline import run_pipeline
from human_eval import evaluate_functional_correctness
print("Ecosystem packages available")
```

## Docker Setup (Optional)

### Build Evaluation Image

```bash
cd rust-qlora
docker build -f Dockerfile.eval -t rust-eval .
```

### Verify Docker

```bash
docker run --rm rust-eval cargo --version
```

## Common Issues

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### bitsandbytes Linux Only

bitsandbytes is Linux-first. For Windows/macOS:

```bash
# Windows (experimental)
pip install bitsandbytes-windows

# macOS (CPU only)
pip install bitsandbytes --no-deps
```

### Flash Attention Compilation

```bash
# Requires CUDA toolkit
pip install flash-attn --no-build-isolation
```

## Next Steps

1. Review [Training Execution Runbook](runbooks/training-execution.md)
2. Configure your first training run
3. Run evaluation on trained model

