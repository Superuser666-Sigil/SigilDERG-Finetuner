# SigilDERG-Finetuner

Model finetuner for the SigilDERG Ecosystem. This project fine-tunes large language models on Rust code using QLoRA (Quantized Low-Rank Adaptation) for efficient training with reduced memory requirements.

## Overview

This repository provides a complete pipeline for fine-tuning LLaMA models on Rust code datasets. It uses 4-bit quantization combined with LoRA adapters to enable training on consumer and enterprise GPUs while maintaining model quality. The system includes automated evaluation that compiles generated Rust code and checks for compilation errors and clippy warnings.

## Features

- QLoRA fine-tuning with 4-bit quantization (BitsAndBytes)
- LoRA adapters for efficient parameter updates
- Streaming dataset support for memory-efficient training
- Multi-dataset support with intelligent filtering heuristics
- Comprehensive evaluation metrics (compilation, clippy, documentation, idiomatic patterns, functionality coverage)
- TensorBoard logging for training curve visualization
- Hyperparameter sweep script for systematic optimization
- Automatic evaluation loop with Rust compilation testing
- Model merging for deployment-ready exports
- FlashAttention 2 support for faster training (optional)
- Tmux-based training and evaluation orchestration

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on H100 80GB, but works on smaller GPUs)
- Rust toolchain (for evaluation)
- Linux environment (setup script targets Ubuntu/Debian)

## Installation

### Standard Python Installation

This project follows standard Python packaging conventions. Install using one of the following methods:

#### Option 1: Using pip (Recommended)

1. Create and activate a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 12.1 (H100):
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.4.0 torchvision==0.19.0
```

3. Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

4. (Optional) Install FlashAttention 2 for faster training:

```bash
pip install "flash-attn>=2.5.6" --no-build-isolation
```

#### Option 2: Automated Setup Script

For a complete system setup including system dependencies and Rust toolchain:

```bash
bash training_setup.sh
```

This script will:
- Install system dependencies (build tools, Python 3.10, etc.)
- Create a Python virtual environment
- Install PyTorch 2.4 with CUDA 12.1 support
- Install all Python dependencies from `requirements.txt`
- Install the package in editable mode
- Optionally install FlashAttention 2
- Install Rust toolchain with clippy and rustfmt (if not present)

#### Option 3: Manual Installation

1. Install system dependencies (Ubuntu/Debian):

```bash
sudo apt-get update && sudo apt-get install -y \
  git build-essential wget curl tmux htop pkg-config libssl-dev \
  libffi-dev unzip python3.10 python3.10-venv
```

2. Install Rust toolchain (for evaluation):

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup component add clippy rustfmt
```

3. Follow steps 1-4 from Option 1 above.

## Configuration

Training parameters are configured in YAML files under `rust-qlora/configs/`. The default configuration (`llama8b.yml`) includes:

- Model: Meta-Llama-3.1-8B-Instruct
- Dataset: ammarnasr/the-stack-rust-clean (supports multiple datasets)
- LoRA rank: 16
- Sequence length: 4096
- Batch size: 8 (with gradient accumulation of 6)
- Learning rate: 1.0e-4
- Training steps: 12000
- TensorBoard logging enabled

The configuration supports:
- Multiple datasets (list format)
- Dataset filtering options (exclude tests/benches, prefer idiomatic code, etc.)
- Dataset caching to avoid network bottlenecks
- Configurable logging backends (TensorBoard, WandB, or none)
- Additional training hyperparameters (scheduler, optimizer, etc.)

Modify the configuration file to adjust hyperparameters for your use case.

## Usage

### Training

Start training with the default configuration:

```bash
cd rust-qlora
bash scripts/run_train.sh
```

Or use the installed package:

```bash
# Using the command-line script
sigilderg-train --cfg rust-qlora/configs/llama8b.yml

# Or as a Python module
python -m rust_qlora.train --cfg rust-qlora/configs/llama8b.yml

# Or directly (if in rust-qlora directory)
python train.py --cfg configs/your_config.yml
```

Training logs are saved to `out/train.log`. TensorBoard logs are saved to `out/llama8b-rust-qlora/logs/`.

View training curves in TensorBoard:

```bash
tensorboard --logdir out/llama8b-rust-qlora/logs
```

### Training with Evaluation Loop

Launch both training and evaluation in a tmux session:

```bash
bash scripts/launch_tmux.sh
```

This starts:
- Training process in the top pane
- Evaluation loop in the bottom pane (runs every 30 minutes)

### Evaluation

Generate evaluation samples:

```bash
python gen_eval_samples.py
```

Evaluate generated Rust code:

```bash
# Basic evaluation
python eval_rust.py eval_out/samples.jsonl

# With custom sample count and functionality checking
python eval_rust.py eval_out/samples.jsonl 32 true
```

The enhanced evaluation script provides comprehensive metrics:
- Compilation success rate
- Average clippy warnings
- Documentation comment rate and count
- Idiomatic pattern detection (Result/Option handling, iterator chains, trait impls, etc.)
- Functionality coverage (functions, structs, impls, traits)
- Prompt keyword matching (when prompts are provided)

### Hyperparameter Sweep

Run systematic hyperparameter sweeps to find optimal settings:

```bash
cd rust-qlora
python hyperparameter_sweep.py --base-cfg configs/llama8b.yml
```

The sweep script:
- Tests multiple combinations of learning rate, LoRA rank/alpha, warmup steps, etc.
- Saves each configuration and results separately
- Logs all runs to TensorBoard for easy comparison
- Supports dry-run mode to preview sweep configurations

View sweep results:

```bash
tensorboard --logdir out/
```

### Model Export

Merge LoRA adapters into the base model for deployment:

```bash
python infer_export.py
```

The merged model will be saved to `out/merged/`.

## Project Structure

```
.
├── rust-qlora/                   # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── configs/
│   │   └── llama8b.yml          # Training configuration
│   ├── scripts/
│   │   ├── launch_tmux.sh        # Launch training + eval in tmux
│   │   ├── run_eval_loop.sh      # Continuous evaluation loop
│   │   └── run_train.sh          # Training script
│   ├── data_filters.py           # Enhanced dataset filtering with multi-dataset support
│   ├── eval_rust.py              # Comprehensive Rust code evaluation
│   ├── gen_eval_samples.py       # Generate evaluation samples
│   ├── hyperparameter_sweep.py  # Hyperparameter sweep script
│   ├── infer_export.py           # Merge and export model
│   └── train.py                  # Main training script with TensorBoard support
├── requirements.txt              # Python dependencies
├── requirements-optional.txt     # Optional dependencies (FlashAttention)
├── pyproject.toml                # Modern Python package configuration
├── setup.py                      # Fallback setup script
├── training_setup.sh             # Automated environment setup script
└── README.md
```

## Dataset

The default configuration uses `ammarnasr/the-stack-rust-clean`, a cleaned subset of Rust code from The Stack dataset. The enhanced data filter (`data_filters.py`) supports:

### Multi-Dataset Support

You can train on multiple datasets simultaneously by specifying a list in the configuration:

```yaml
dataset:
  names:
    - ammarnasr/the-stack-rust-clean
    - another/rust-dataset
```

### Enhanced Filtering

The filtering system includes:

- **Path-based exclusions**: Vendor directories, node_modules, lock files, test files, benchmarks, examples
- **Length filtering**: Configurable min/max code length
- **Quality heuristics**:
  - Idiomatic pattern detection (Result/Option handling, iterator chains, derive macros, trait implementations)
  - Documentation comment detection
  - Low-quality code markers (TODO, debug prints, unsafe blocks, suppressed warnings)
- **Dataset caching**: Optional local caching to avoid network bottlenecks
- **Shuffling**: Optional dataset shuffling (uses memory, use with caution)

### Filtering Options

Configure filtering in your YAML config:

```yaml
dataset:
  exclude_tests: true
  exclude_examples: false
  exclude_benches: true
  prefer_idiomatic: false  # Prioritize code with idiomatic patterns
  prefer_documented: false  # Prioritize code with documentation
  cache_dir: ~/.cache/huggingface/datasets  # Local cache location
```

## Training Details

### Quantization

Uses BitsAndBytes 4-bit quantization with:
- Quantization type: NF4
- Double quantization: enabled
- Compute dtype: bfloat16

### LoRA Configuration

LoRA adapters target attention and MLP layers:
- q_proj, k_proj, v_proj, o_proj (attention)
- up_proj, down_proj, gate_proj (MLP)

Default settings: rank=16, alpha=16, dropout=0.05

### Memory Optimization

- Gradient checkpointing enabled
- Streaming dataset (no full dataset in memory)
- 4-bit quantization reduces model memory footprint
- Sequence packing for efficient batching
- Optional dataset caching to reduce network I/O

### Logging and Monitoring

- **TensorBoard**: Default logging backend for training curves
- **WandB**: Optional support via configuration
- **Training metrics**: Loss, learning rate, gradient norms logged automatically
- **Evaluation metrics**: Comprehensive code quality metrics logged to JSONL

## Evaluation

The enhanced evaluation system provides comprehensive quality assessment:

### Metrics Collected

1. **Compilation metrics**:
   - Compilation success rate
   - Average clippy warnings per sample

2. **Code quality metrics**:
   - Documentation comment rate and average count
   - Idiomatic pattern score (detects Result/Option handling, iterator chains, trait impls, match expressions, pattern matching)

3. **Functionality coverage**:
   - Average number of functions, structs, impls, traits
   - Prompt keyword matching (when prompts provided)
   - Test presence detection

### Evaluation Workflow

1. Generate code samples from predefined prompts (saves prompts for analysis)
2. Compile each sample in a temporary Cargo project
3. Run clippy with all warnings enabled
4. Analyze code structure and patterns
5. Report comprehensive metrics

Evaluation metrics are logged to `eval_out/metrics.jsonl` in JSON format for easy analysis.

## Output

- Trained LoRA adapters: `out/llama8b-rust-qlora/`
- Training logs: `out/train.log`
- TensorBoard logs: `out/llama8b-rust-qlora/logs/`
- Evaluation samples: `eval_out/samples.jsonl` (includes prompts)
- Evaluation metrics: `eval_out/metrics.jsonl` (comprehensive metrics)
- Merged model: `out/merged/` (after running infer_export.py)
- Sweep configurations: `sweeps/` (when using hyperparameter sweep)

## Environment Variables

- `FLASH_ATTENTION`: Set to 1 to enable FlashAttention 2 (auto-detected if installed)
- `CUDA_VISIBLE_DEVICES`: Specify GPU device (default: 0)
- `TOKENIZERS_PARALLELISM`: Set to false to avoid warnings
- `HF_HUB_ENABLE_HF_TRANSFER`: Enable faster HuggingFace downloads

## Optimization Guide

For detailed instructions on achieving high compile rates (≥95%), low clippy warnings, and high idiomatic scores, see [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md).

The guide covers:
- Stricter dataset filtering for high-quality training
- Two-phase training strategy (broad → sharpening)
- RLAIF-lite synthetic reward training
- Hyperparameter tuning strategies
- Step-by-step workflow to reach target metrics

## License

MIT License

Copyright (c) 2025 Dave Tofflemire

