# SigilDERG-Finetuner

Model finetuner for the SigilDERG Ecosystem. This project fine-tunes large language models on Rust code using QLoRA (Quantized Low-Rank Adaptation) for efficient training with reduced memory requirements.

## Overview

This repository provides a complete pipeline for fine-tuning LLaMA models on Rust code datasets. It uses 4-bit quantization combined with LoRA adapters to enable training on consumer and enterprise GPUs while maintaining model quality. The system includes automated evaluation that compiles generated Rust code and checks for compilation errors and clippy warnings. You can preview or deploy the latest checkpoint on Hugging Face: https://huggingface.co/Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora.

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
- Multi-GPU scaling via Hugging Face Accelerate (2×/4×/8× H100 nodes)
- H100 GPU optimizations: pre-tokenization, parallel data loading, TF32 tensor cores
- Tmux-based training and evaluation orchestration

## Requirements

- Python 3.12.10+
- CUDA-capable GPU (tested on H100 80GB, but works on smaller GPUs)
- Rust toolchain (for evaluation)
- Linux environment (setup script targets Ubuntu/Debian)

## Installation

### Standard Python Installation

This project follows standard Python packaging conventions. Install using one of the following methods:

#### Option 1: Using pip (Recommended)

1. Create and activate a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
# For CUDA 12.8 (NVIDIA 570+ drivers):
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6 (if using older CUDA toolkit):
# pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 11.8 (if running on older GPUs):
# pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch==2.6.0 torchvision==0.21.0
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

**Note:** If you upgrade PyTorch after installing FlashAttention, you must reinstall FlashAttention so it rebuilds against the new CUDA runtime:
```bash
pip uninstall flash-attn
pip install "flash-attn>=2.5.6" --no-build-isolation
```

#### Option 2: Automated Setup Script

For a complete system setup including system dependencies and Rust toolchain:

```bash
bash training_setup.sh
```

This script will:
- Install system dependencies (build tools, pyenv dependencies, etc.)
- Install and configure pyenv
- Install Python 3.12.11 via pyenv
- Create a Python virtual environment using pyenv's Python 3.12.11
- Install PyTorch 2.7.1+cu128 (for CUDA 12.8 / NVIDIA 570+ drivers)
- Install all Python dependencies from `requirements.txt`
- Install the package in editable mode
- Optionally install FlashAttention 2
- Install Rust toolchain with clippy and rustfmt (if not present)

#### Option 3: Manual Installation

1. Install system dependencies (Ubuntu/Debian):

```bash
sudo apt-get update && sudo apt-get install -y \
  git build-essential wget curl tmux htop pkg-config libssl-dev \
  libffi-dev unzip python3.12 python3.12-venv
```

2. Install Rust toolchain (for evaluation):

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable
rustup component add clippy rustfmt
```

3. Follow steps 1-4 from Option 1 above (use the CUDA variant that matches your GPU/driver).

## SigilDERG Ecosystem Integration

This package is part of the **SigilDERG ecosystem** for Rust code model training. It integrates seamlessly with:

- **[sigil-pipeline](https://github.com/Superuser666-Sigil/SigilDERG-Data_Production)**: Dataset generation from Rust crates
- **[human-eval-rust](https://github.com/Superuser666-Sigil/human-eval-Rust)**: Evaluation harness for Rust code generation

### Install Full Ecosystem

```bash
pip install sigilderg-finetuner[ecosystem]
```

This installs all three packages with proper version constraints.

### Load Pipeline Datasets Directly

The finetuner can now load JSONL files directly from sigil-pipeline:

```yaml
dataset:
  names:
    - local:datasets/phase2_full.jsonl  # Load pipeline JSONL directly
  use_cache: true
```

Or mix HuggingFace and local datasets:

```yaml
dataset:
  names:
    - ammarnasr/the-stack-rust-clean  # HuggingFace dataset
    - local:datasets/phase2_full.jsonl  # Pipeline output
  interleave_mode: "weighted"
  dataset_weights:
    "ammarnasr/the-stack-rust-clean": 0.3
    "local:datasets/phase2_full.jsonl": 0.7
```

### Use HumanEval for Evaluation

Enable human-eval-rust integration in evaluation:

```bash
sigilderg-eval samples.jsonl --use-human-eval
```

This runs both standard Rust compilation/clippy evaluation and human-eval-rust functional correctness tests.

See the [Ecosystem Integration Guide](https://github.com/Superuser666-Sigil/SigilDERG-Data_Production/blob/main/docs/ECOSYSTEM_INTEGRATION.md) for complete workflow documentation.

## Configuration

Training parameters are configured in YAML files under `rust-qlora/configs/`. The default configuration (`llama8b-phase1.yml`) includes:

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
sigilderg-train --cfg rust-qlora/configs/llama8b-phase1.yml

# Or as a Python module
python -m rust_qlora.train --cfg rust-qlora/configs/llama8b-phase1.yml

# Or directly (if in rust-qlora directory)
python train.py --cfg configs/your_config.yml
```

#### Multi-GPU training (2×/4×/8× H100)

`training_setup.sh` installs [Hugging Face Accelerate](https://github.com/huggingface/accelerate), configures `hf_transfer` for faster downloads, and the helper scripts auto-detect GPU count.

```bash
# On a multi-GPU node
cd rust-qlora
# Optional: override config/micro-batch values per GPU
export TRAIN_CFG=configs/llama8b-phase1.yml
# Optional: pin the GPU count (defaults to all visible GPUs)
export NUM_GPUS=4

bash scripts/run_train.sh
```

Per-GPU micro-batch sizes for Phase 1:

| GPUs | micro_batch_size | gradient_accumulation | Effective batch |
|------|-----------------|-----------------------|-----------------|
| 1    | 16              | 4                     | 64              |
| 2    | 8               | 4                     | 64              |
| 4    | 4               | 4                     | 64              |
| 8    | 2               | 4                     | 64              |

Per-GPU micro-batch sizes for Phase 2:

| GPUs | micro_batch_size | gradient_accumulation | Effective batch |
|------|-----------------|-----------------------|-----------------|
| 1    | 16              | 4                     | 64              |
| 2    | 8               | 4                     | 64              |
| 4    | 4               | 4                     | 64              |
| 8    | 2               | 4                     | 64              |

**Note:** Phase 2 uses the same batch size scaling as Phase 1 to maintain consistent effective batch size across phases. The shorter sequence length (2048 vs 4096) allows for efficient multi-GPU scaling.

Both `scripts/run_train.sh` and `scripts/run_phase2.sh` automatically fall back to single-GPU launches when only one accelerator is visible.

**Cost guidance (based on on-demand pricing):**

| Instance           | Cost/hr | Est. wall time | Est. cost |
|--------------------|--------:|---------------:|----------:|
| 1× H100 SXM5       | $3.29   | ~40 h          | ~$132     |
| 2× H100 SXM5       | $6.38   | ~20 h          | ~$128     |
| 4× H100 SXM5       | $12.36  | ~10 h          | ~$124     |

Estimates assume `grad_checkpointing: false` and the table above for per-GPU batch size. Adjust as needed for your budget.

**Logging:**
- **TensorBoard logs**: Automatically saved to `out/llama8b-rust-qlora-phase1/logs/` (or path specified in config)
- **Training log file**: Optional, can be enabled with `--log-file` argument or `misc.log_file` in config
  - Uses Python's `logging` module (replaces previous Tee implementation)
  - Logs to both file and stdout simultaneously
  - The `scripts/run_train.sh` script automatically logs to `out/train.log` via `tee`
  - Direct CLI usage (`sigilderg-train` or `python -m rust_qlora.train`) only logs to stdout unless `--log-file` is specified

View training curves in TensorBoard:

**Option 1: Using the launch script (recommended):**
```bash
# Launch TensorBoard in a tmux session (suppresses warnings, shows all runs)
bash scripts/launch_tensorboard.sh

# Or with custom directory/port
bash scripts/launch_tensorboard.sh out/llama8b-rust-qlora-phase1/logs 6006
```

**Option 2: Manual launch:**
```bash
# View Phase 1 logs specifically
tensorboard --logdir out/llama8b-rust-qlora-phase1/logs

# Or view all runs (Phase 1, Phase 2, etc.)
tensorboard --logdir out/
```

**Note:** The launch script automatically:
- Creates a persistent tmux session
- Suppresses TensorFlow/CUDA warnings
- Points to the correct log directory
- Allows remote access via `--host 0.0.0.0`

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
# Basic usage (uses latest checkpoint from Phase 1)
python gen_eval_samples.py

# Specify a checkpoint directory (auto-finds latest checkpoint)
python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase1

# Specify a specific checkpoint
python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase1/checkpoint-1000

# Use base model for baseline comparison
python gen_eval_samples.py --model-path meta-llama/Meta-Llama-3.1-8B-Instruct

# With custom seed for reproducibility
python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase1 --seed 42

# Use custom prompts from YAML or JSON file
python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase1 --prompts-file prompts.yaml
```

**Custom Prompts:** You can provide prompts via a YAML or JSON file:
```yaml
# prompts.yaml
- "Write a Rust function that calculates fibonacci numbers"
- "Create a Rust struct with a Display implementation"
```

Or JSON format:
```json
{
  "prompts": [
    "Write a Rust function that calculates fibonacci numbers",
    "Create a Rust struct with a Display implementation"
  ]
}
```

**Note:** The script automatically:
- Detects PEFT (LoRA) checkpoints and loads them correctly
- Finds the latest checkpoint if a directory is provided
- Falls back to full model loading for merged checkpoints or base models
- Works with both Phase 1 and Phase 2 checkpoints

Evaluate generated Rust code:

```bash
# Basic evaluation (uses parallel processing by default)
python eval_rust.py eval_out/samples.jsonl

# Or using the installed package command
sigilderg-eval eval_out/samples.jsonl

# With custom options (including functionality coverage checks)
python eval_rust.py eval_out/samples.jsonl \
    --sample-n 32 \
    --check-func \
    --num-workers 4 \
    --seed 0 \
    --output eval_out/metrics.jsonl

# Or using the installed package
sigilderg-eval eval_out/samples.jsonl \
    --sample-n 32 \
    --check-func \
    --num-workers 4 \
    --seed 0 \
    --output eval_out/metrics.jsonl

# Without functionality checks (faster, compilation-only)
python eval_rust.py eval_out/samples.jsonl \
    --sample-n 32 \
    --num-workers 4

# Sequential evaluation (single worker)
python eval_rust.py eval_out/samples.jsonl --num-workers 1

# Write metrics to file (recommended for automation)
python eval_rust.py eval_out/samples.jsonl --output eval_out/metrics.jsonl

# Configure pre-filtering thresholds
python eval_rust.py eval_out/samples.jsonl \
    --pre-filter-min-length 50 \
    --pre-filter-min-lines 5 \
    --pre-filter-no-main-check  # Don't require fn main

# Configure timeouts for compilation and Clippy
python eval_rust.py eval_out/samples.jsonl \
    --compile-timeout 60 \
    --clippy-timeout 45
```

**Evaluation Features:**
- **Parallel processing**: Automatically uses multiple CPU cores for faster evaluation
- **Pre-filtering**: Skips invalid samples (no `fn main`, incomplete code, etc.) before compilation
  - Configurable thresholds: `--pre-filter-min-length`, `--pre-filter-min-lines`
  - Optional checks: `--pre-filter-no-main-check`, `--pre-filter-no-incomplete-check`
- **Reproducibility**: `--seed` argument ensures consistent sample selection
- **Direct file output**: `--output` flag writes metrics directly to JSONL file (no shell piping required)
- **Comprehensive metrics**: Compilation, clippy, documentation, idiomatic patterns
- **Functionality coverage**: Enable with `--check-func` flag (trait counts, test detection, prompt matching)
- **Configurable timeouts**: Separate timeouts for compilation (`--compile-timeout`) and Clippy (`--clippy-timeout`)
- **Security sandboxing**: Automatically sandboxes cargo commands using Docker (recommended) or Firejail

**Security: Sandboxed Evaluation**

By default, the evaluation system automatically sandboxes all cargo compilation commands to prevent malicious code execution. This is **critical** when evaluating LLM-generated code, as build scripts or macro expansions could execute arbitrary code.

**Sandbox modes:**
- **Docker** (recommended): Runs cargo commands in isolated Docker containers with resource limits
- **Firejail**: Alternative sandboxing using Firejail (if Docker unavailable)
- **Auto-detect**: Automatically uses Docker if available, falls back to Firejail, then warns if neither is available

```bash
# Explicitly specify sandbox mode
python eval_rust.py eval_out/samples.jsonl --sandbox-mode docker

# Disable sandboxing (UNSAFE - only for local dev with trusted code)
python eval_rust.py eval_out/samples.jsonl --no-sandbox
```

**Important:** Never disable sandboxing when evaluating untrusted LLM-generated code. The sandbox provides:
- Network isolation (no internet access)
- Memory limits (512MB per container)
- CPU limits (1 core per container)
- Read-only filesystem (except temp directories)
- Automatic container cleanup

If Docker is not installed, the evaluator will warn and fall back to unsandboxed execution. Install Docker for production use:
```bash
# Ubuntu/Debian
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # Add user to docker group (logout/login required)
```

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
python hyperparameter_sweep.py --base-cfg configs/llama8b-phase1.yml
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
# View all runs in TensorBoard (using launch script)
bash scripts/launch_tensorboard.sh out/

# Or manually
tensorboard --logdir out/

# Or check the summary file
cat sweeps/sweep_summary.json
```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

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
- Input validation and error handling with helpful error messages
- Base model validation (warns if checkpoint was trained with different base model)
- Disk space checking before export
- Model shape verification
- Configurable device selection (CPU/CUDA/auto)
- Custom checkpoint and output paths
- Comprehensive exception handling for disk space and permission issues

## Project Structure

```
.
├── rust-qlora/                   # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── configs/
│   │   ├── llama8b-phase1.yml    # Phase 1 training configuration
│   │   └── llama8b-phase2.yml    # Phase 2 ("sharpening") config
│   ├── scripts/
│   │   ├── checkpoint_eval_workflow.sh # Inspect/evaluate/push checkpoints
│   │   ├── launch_tensorboard.sh       # Launch TensorBoard in tmux (with warning suppression)
│   │   ├── launch_tmux.sh              # Launch training + eval loop in tmux
│   │   ├── run_eval_loop.sh            # Continuous evaluation loop
│   │   ├── run_phase2.sh               # Phase 2 training wrapper
│   │   └── run_train.sh                # Phase 1 training wrapper
│   ├── datasets/
│   │   └── loader.py             # Unified cached/streaming dataset loader
│   ├── data_filters.py           # Enhanced dataset filtering with multi-dataset support
│   ├── eval_rust.py              # Comprehensive Rust code evaluation (parallel + template reuse)
│   ├── eval_template.py          # Template project reuse for faster evaluation
│   ├── gen_eval_samples.py       # Generate evaluation samples
│   ├── hyperparameter_sweep.py   # Hyperparameter sweep script (with deepcopy fix)
│   ├── infer_export.py           # Merge and export model
│   ├── push_model_card.py        # Generate/push model card README
│   ├── expert_iteration.py       # Expert Iteration / Rejection Sampling Fine-Tuning (RSFT)
│   ├── train.py                  # Main training script with TensorBoard support
│   └── update_model_card_eval.py # Inject evaluation metrics into model card
├── requirements.txt              # Python dependencies
├── requirements-optional.txt     # Optional dependencies (FlashAttention)
├── pyproject.toml                # Modern Python package configuration
├── setup.py                      # Fallback setup script
├── training_setup.sh             # Automated environment setup script
└── README.md
```

## Dataset

The default configuration uses `ammarnasr/the-stack-rust-clean`, a cleaned subset of Rust code from The Stack dataset. The enhanced data filter (`data_filters.py`) supports:

`rust_qlora/datasets/loader.py` wraps all dataset loading logic (cached vs streaming) into a single module so training scripts only need to request a dataset object. It automatically applies filtering, pre-tokenization, worker overrides, and shuffling warnings to keep behaviour consistent across configs.

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
  - Configurable quality ratio (`idiomatic_quality_ratio` in config, default: 2.0) - controls how much idiomatic patterns must outnumber low-quality markers
  - Documentation comment detection
  - Low-quality code markers (TODO, debug prints, unsafe blocks, suppressed warnings)
- **Filter reason tracking**: Per-dataset statistics showing why samples were filtered (too_short, test_file, not_idiomatic, etc.)
- **Filter statistics export**: Save filter statistics to JSON/CSV files for analysis (via `save_filter_stats` parameter)
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

**Configuration Format:** `target_modules` is now specified as a list in YAML configs:
```yaml
lora:
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
    - gate_proj
```

The system maintains backward compatibility with semicolon-separated strings for existing configs.

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
  - **Note**: When using `flash_attention_2` (via `FLASH_ATTENTION` env var), some deterministic settings may be overridden due to kernel optimizations. For fully deterministic training, use `sdpa` instead.
- **Evaluation**: `--seed` argument in all evaluation scripts
- **Generation**: `--seed` argument in `gen_eval_samples.py` and `expert_iteration.py`
- **Hyperparameter sweeps**: `--seed` argument for reproducible sweep configurations

### Checkpoint Inspection

Inspect checkpoint directories to see files and configuration:

```bash
# Human-readable output
python inspect_checkpoint.py out/llama8b-rust-qlora-phase1/checkpoint-1000

# JSON output for automation/scripting
python inspect_checkpoint.py out/llama8b-rust-qlora-phase1/checkpoint-1000 --json
```

The inspection script shows:
- File sizes and purposes
- PEFT adapter configuration
- Training state (step, epoch, metrics)
- Model configuration

### Logging and Monitoring

- **TensorBoard**: Default logging backend for training curves
  - Use `bash scripts/launch_tensorboard.sh` for easy setup with warning suppression
  - Logs are saved to `out/llama8b-rust-qlora-phase1/logs/` (or path in config)
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

### Manual Checkpoint Workflow Script

After a save (e.g., checkpoint-1000) you can run the entire manual workflow—inspect → generate samples → evaluate → refresh README → (optional) push to HuggingFace—with one command:

```bash
bash scripts/checkpoint_eval_workflow.sh \
  --checkpoint out/llama8b-rust-qlora-phase1/checkpoint-1000 \
  --repo-id Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora
```

Key flags:

- `--checkpoint`: Specific checkpoint directory (defaults to the latest under `out/llama8b-rust-qlora-phase1`)
- `--sample-n`: Number of evaluation samples (default: 64)
- `--repo-id`: HuggingFace repo to push `README.md` (optional)
- `--hf-token`: Token to use when pushing (falls back to `HF_TOKEN` env var)
- `--config`: Override config path for model-card metadata

Outputs:

- `eval_out/samples.jsonl`, `eval_out/metrics.jsonl`, `eval_out/errors.jsonl`
- Updated `README.md` inside the checkpoint directory (with fresh eval metrics)
- Optional push to the specified HuggingFace repo

## Output

- **Phase 1 training:**
  - Trained LoRA adapters: `out/llama8b-rust-qlora-phase1/`
  - Checkpoints: `out/llama8b-rust-qlora-phase1/checkpoint-*/` (saved every 1000 steps)
  - TensorBoard logs: `out/llama8b-rust-qlora-phase1/logs/`
- **Phase 2 training:**
  - Trained LoRA adapters: `out/llama8b-rust-qlora-phase2/`
  - Checkpoints: `out/llama8b-rust-qlora-phase2/checkpoint-*/` (saved every 500 steps)
  - TensorBoard logs: `out/llama8b-rust-qlora-phase2/logs/`
- **Training logs:** `out/train.log` (from run_train.sh)
- **Evaluation samples:** `eval_out/samples.jsonl` (includes prompts)
- **Evaluation metrics:** `eval_out/metrics.jsonl` (comprehensive metrics)
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
- **Generation**: `--seed` argument (default: 0 for eval, 42 for Expert Iteration)
- **Sweeps**: `--seed` argument (default: 42)

When seeds are set, repeated runs produce identical results (assuming same hardware/software versions).

## Optimization Guide

For detailed instructions on achieving high compile rates (≥95%), low clippy warnings, and high idiomatic scores, see [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md).

The guide covers:
- Stricter dataset filtering for high-quality training
- Two-phase training strategy (broad → sharpening)
- Expert Iteration / Rejection Sampling Fine-Tuning (RSFT)
- Hyperparameter tuning strategies
- Step-by-step workflow to reach target metrics

## License

MIT License

Copyright (c) 2025 Dave Tofflemire

