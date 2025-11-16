# SigilDERG-Finetuner

Model finetuner for the SigilDERG Ecosystem. This project fine-tunes large language models on Rust code using QLoRA (Quantized Low-Rank Adaptation) for efficient training with reduced memory requirements.

## Overview

This repository provides a complete pipeline for fine-tuning LLaMA models on Rust code datasets. It uses 4-bit quantization combined with LoRA adapters to enable training on consumer and enterprise GPUs while maintaining model quality. The system includes automated evaluation that compiles generated Rust code and checks for compilation errors and clippy warnings.

## Features

- QLoRA fine-tuning with 4-bit quantization (BitsAndBytes)
- LoRA adapters for efficient parameter updates
- Streaming dataset support for memory-efficient training
- Multi-dataset support with configurable interleaving (sequential, round-robin, weighted)
- Comprehensive evaluation metrics (compilation, clippy, documentation, idiomatic patterns, functionality coverage including traits, tests, prompt matching)
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
- Multiple datasets (list format) with interleaving modes (sequential, round-robin, weighted)
- Dataset filtering options (exclude tests/benches, prefer idiomatic code, etc.) with per-dataset filter reason tracking
- Dataset caching to avoid network bottlenecks (use_cache flag)
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
# Basic usage
python gen_eval_samples.py

# With custom model path and seed for reproducibility
python gen_eval_samples.py --model-path out/llama8b-rust-qlora --seed 42
```

Evaluate generated Rust code:

```bash
# Basic evaluation (uses parallel processing by default)
python eval_rust.py eval_out/samples.jsonl

# With custom options
python eval_rust.py eval_out/samples.jsonl \
    --sample-n 32 \
    --check-func \
    --num-workers 4 \
    --seed 0 \
    --output eval_out/metrics.jsonl

# Sequential evaluation (single worker)
python eval_rust.py eval_out/samples.jsonl --num-workers 1

# Write metrics to file (recommended for automation)
python eval_rust.py eval_out/samples.jsonl --output eval_out/metrics.jsonl
```

**Evaluation Features:**
- **Parallel processing**: Automatically uses multiple CPU cores for faster evaluation
- **Pre-filtering**: Skips invalid samples (no `fn main`, incomplete code, etc.) before compilation
- **Reproducibility**: `--seed` argument ensures consistent sample selection
- **Direct file output**: `--output` flag writes metrics directly to JSONL file (no shell piping required)
- **Comprehensive metrics**: Compilation, clippy, documentation, idiomatic patterns, functionality coverage

The enhanced evaluation script provides comprehensive metrics:
- Compilation success rate
- Average clippy warnings
- Documentation comment rate and count
- Idiomatic pattern detection (Result/Option handling, iterator chains, trait impls, etc.)
- Functionality coverage (functions, structs, impls, **traits**, **tests**)
- **Prompt keyword matching** (aggregated when prompts are provided)

### Hyperparameter Sweep

Run systematic hyperparameter sweeps to find optimal settings:

```bash
cd rust-qlora
python hyperparameter_sweep.py --base-cfg configs/llama8b.yml
```

The sweep script:
- Tests multiple combinations of learning rate, LoRA rank/alpha, warmup steps, etc.
- Saves each configuration and results separately with unique run IDs
- Logs all runs to TensorBoard for easy comparison
- Supports dry-run mode to preview sweep configurations
- Includes timeout handling and error recovery
- Generates a summary JSON file (`sweeps/sweep_summary.json`) with all run metadata

View sweep results:

```bash
# View all runs in TensorBoard
tensorboard --logdir out/

# Or check the summary file
cat sweeps/sweep_summary.json
```

### Model Export

Merge LoRA adapters into the base model for deployment:

```bash
# Basic usage (uses defaults)
python infer_export.py

# With custom paths and options
python infer_export.py \
    --checkpoint out/llama8b-rust-qlora \
    --output out/merged \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cpu

# Using CUDA for faster merging
python infer_export.py --device cuda
```

The script includes:
- Input validation and error handling
- Model shape verification
- Configurable device selection (CPU/CUDA/auto)
- Custom checkpoint and output paths

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
│   ├── eval_rust.py              # Comprehensive Rust code evaluation (parallel + template reuse)
│   ├── eval_template.py          # Template project reuse for faster evaluation
│   ├── gen_eval_samples.py       # Generate evaluation samples
│   ├── hyperparameter_sweep.py  # Hyperparameter sweep script (with deepcopy fix)
│   ├── infer_export.py           # Merge and export model
│   ├── rlaif_lite.py             # RLAIF-lite synthetic reward training
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

You can train on multiple datasets simultaneously by specifying a list in the configuration. The system supports three interleaving modes:

**Sequential (default)**: Process datasets one after another
```yaml
dataset:
  names:
    - ammarnasr/the-stack-rust-clean
    - another/rust-dataset
  interleave_mode: sequential
```

**Round-robin**: Alternate between datasets evenly
```yaml
dataset:
  names:
    - dataset1
    - dataset2
  interleave_mode: round_robin
```

**Weighted**: Sample datasets based on weights
```yaml
dataset:
  names:
    - large_dataset
    - small_high_quality_dataset
  interleave_mode: weighted
  dataset_weights:
    large_dataset: 0.3
    small_high_quality_dataset: 0.7
```

Note: Round-robin and weighted modes require loading all datasets into memory, so they work best with cached datasets (`use_cache: true`).

### Enhanced Filtering

The filtering system includes:

- **Path-based exclusions**: Vendor directories, node_modules, lock files, test files, benchmarks, examples
- **Length filtering**: Configurable min/max code length
- **Quality heuristics**:
  - Idiomatic pattern detection (Result/Option handling, iterator chains, derive macros, trait implementations)
- **Filter reason tracking**: Per-dataset statistics showing why samples were filtered (too_short, test_file, not_idiomatic, etc.)
  - Documentation comment detection
  - Low-quality code markers (TODO, debug prints, unsafe blocks, suppressed warnings)
- **Dataset caching**: Control streaming vs cached datasets via `use_cache` flag
  - `use_cache: true` → Non-streaming (cached) for better throughput
  - `use_cache: false` → Streaming for lower RAM usage
  - Note: `shuffle_seed` requires non-streaming mode regardless of `use_cache`
- **Shuffling**: Optional dataset shuffling (uses memory, use with caution)
- **Filter telemetry**: Automatic reporting of filter statistics during training

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
- Configurable dataset streaming (`use_cache` flag)
  - Streaming mode: Lower RAM usage, suitable for large datasets
  - Cached mode: Better throughput, requires more RAM
- 4-bit quantization reduces model memory footprint
- Sequence packing for efficient batching
- Optional dataset caching to reduce network I/O

### Reproducibility

- **Training**: Seed controlled via `misc.seed` in config files
  - Automatically sets PyTorch, CUDA, and CuDNN seeds
  - Enables deterministic CuDNN operations for consistent results
- **Evaluation**: `--seed` argument in all evaluation scripts
- **Generation**: `--seed` argument in `gen_eval_samples.py` and `rlaif_lite.py`
- **Hyperparameter sweeps**: `--seed` argument for reproducible sweep configurations

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

## Performance & Reproducibility

### Evaluation Performance

The evaluation system uses parallel processing to speed up compilation checks:

- **Automatic parallelization**: Uses `CPU_COUNT - 1` workers by default
- **Manual control**: Set `--num-workers` to control parallelism
  - `--num-workers 1`: Sequential evaluation (slower but uses less CPU)
  - `--num-workers 4`: Use 4 parallel workers
  - `--num-workers None`: Auto-detect (default)

The evaluation system also reuses a template Cargo project to avoid the overhead of running `cargo new` for every sample, further improving throughput.

This dramatically speeds up evaluation for large sample sets, making hyperparameter sweeps and RLAIF loops more practical.

### Reproducibility Guarantees

All scripts support deterministic execution:

- **Training**: Seed from `misc.seed` in config (default: 42)
- **Evaluation**: `--seed` argument (default: 0)
- **Generation**: `--seed` argument (default: 0 for eval, 42 for RLAIF)
- **Sweeps**: `--seed` argument (default: 42)

When seeds are set, repeated runs produce identical results (assuming same hardware/software versions).

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

