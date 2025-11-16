# SigilDERG-Finetuner

Model finetuner for the SigilDERG Ecosystem. This project fine-tunes large language models on Rust code using QLoRA (Quantized Low-Rank Adaptation) for efficient training with reduced memory requirements.

## Overview

This repository provides a complete pipeline for fine-tuning LLaMA models on Rust code datasets. It uses 4-bit quantization combined with LoRA adapters to enable training on consumer and enterprise GPUs while maintaining model quality. The system includes automated evaluation that compiles generated Rust code and checks for compilation errors and clippy warnings.

## Features

- QLoRA fine-tuning with 4-bit quantization (BitsAndBytes)
- LoRA adapters for efficient parameter updates
- Streaming dataset support for memory-efficient training
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

Run the setup script to install all dependencies:

```bash
bash training_setup.sh
```

This script will:
- Install system dependencies (build tools, Python 3.10, etc.)
- Create a Python virtual environment
- Install PyTorch 2.4 with CUDA 12.1 support
- Install required Python packages (transformers, peft, trl, bitsandbytes, etc.)
- Optionally install FlashAttention 2
- Install Rust toolchain with clippy and rustfmt

## Configuration

Training parameters are configured in YAML files under `rust-qlora/configs/`. The default configuration (`llama8b.yml`) includes:

- Model: Meta-Llama-3.1-8B-Instruct
- Dataset: ammarnasr/the-stack-rust-clean
- LoRA rank: 16
- Sequence length: 4096
- Batch size: 8 (with gradient accumulation of 6)
- Learning rate: 1.0e-4
- Training steps: 12000

Modify the configuration file to adjust hyperparameters for your use case.

## Usage

### Training

Start training with the default configuration:

```bash
cd rust-qlora
bash scripts/run_train.sh
```

Or specify a custom configuration:

```bash
python train.py --cfg configs/your_config.yml
```

Training logs are saved to `out/train.log`.

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
python eval_rust.py eval_out/samples.jsonl
```

The evaluation script:
- Samples generated code snippets
- Compiles each snippet in a temporary Cargo project
- Counts compilation success rate
- Measures average clippy warnings

### Model Export

Merge LoRA adapters into the base model for deployment:

```bash
python infer_export.py
```

The merged model will be saved to `out/merged/`.

## Project Structure

```
.
├── rust-qlora/
│   ├── configs/
│   │   └── llama8b.yml          # Training configuration
│   ├── scripts/
│   │   ├── launch_tmux.sh        # Launch training + eval in tmux
│   │   ├── run_eval_loop.sh      # Continuous evaluation loop
│   │   └── run_train.sh          # Training script
│   ├── data_filters.py           # Dataset filtering logic
│   ├── eval_rust.py              # Rust code evaluation
│   ├── gen_eval_samples.py       # Generate evaluation samples
│   ├── infer_export.py           # Merge and export model
│   └── train.py                  # Main training script
├── training_setup.sh             # Environment setup script
└── README.md
```

## Dataset

The default configuration uses `ammarnasr/the-stack-rust-clean`, a cleaned subset of Rust code from The Stack dataset. The data filter (`data_filters.py`) automatically:

- Excludes vendor directories, node_modules, and lock files
- Filters code by length (64 to 200,000 characters)
- Streams data to minimize memory usage

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

## Evaluation

The evaluation system tests model output quality by:

1. Generating code samples from predefined prompts
2. Compiling each sample in a temporary Rust project
3. Running clippy to check code quality
4. Reporting compilation success rate and average warnings

Evaluation metrics are logged to `eval_out/metrics.jsonl`.

## Output

- Trained LoRA adapters: `out/llama8b-rust-qlora/`
- Training logs: `out/train.log`
- Evaluation samples: `eval_out/samples.jsonl`
- Evaluation metrics: `eval_out/metrics.jsonl`
- Merged model: `out/merged/` (after running infer_export.py)

## Environment Variables

- `FLASH_ATTENTION`: Set to 1 to enable FlashAttention 2 (auto-detected if installed)
- `CUDA_VISIBLE_DEVICES`: Specify GPU device (default: 0)
- `TOKENIZERS_PARALLELISM`: Set to false to avoid warnings
- `HF_HUB_ENABLE_HF_TRANSFER`: Enable faster HuggingFace downloads

## License

MIT License

Copyright (c) 2025 Dave Tofflemire

