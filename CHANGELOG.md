# Changelog

All notable changes to SigilDERG-Finetuner will be documented in this file.

## [Unreleased]

- Added official multi-GPU training support:
  - `training_setup.sh` now installs Hugging Face Accelerate + hf_transfer and pre-seeds an
    accelerate config
  - `requirements.txt` now pins `torch>=2.9.0`, `torchvision>=0.22.0`, `accelerate>=1.2.1`,
    and `hf_transfer>=0.1.5`
  - `scripts/run_train.sh` / `scripts/run_phase2.sh` auto-detect GPU count and launch via
    `accelerate` when more than one GPU is visible
  - README documents per-GPU batch sizing, cost guidance, and launch instructions
- Updated PyTorch baseline to 2.9.0 (CUDA 12.6) for better Hopper performance; setup scripts
  install the matching wheels and rebuild FlashAttention automatically

## [2.6.0] - 2025-11-18

### Added
- **H100 GPU Optimizations**:
  - Pre-tokenization with parallel processing using up to 25 CPU workers (reduces tokenization time from ~35 minutes to ~2-3 minutes)
  - Dataset caching enabled by default for faster data loading (uses Dataset.filter() for multi-worker support)
  - Increased default batch size from 8 to 16 for H100 (80GB VRAM)
  - Configurable dataloader workers (default: 12 for 25 vCPU systems)
  - Pin memory and prefetch factor optimizations for faster CPU-GPU transfers
  - Flash Attention 2 support via config (`use_flash_attention: true`)
  - TF32 tensor cores enabled for faster matmuls on H100
  - CuDNN benchmark mode enabled (can be disabled with `deterministic: true` for reproducibility)
  - Memory optimization callback with configurable cache clearing frequency
- **Dataset Loading Improvements**:
  - Smart dataset loading: uses Dataset.filter() for cached datasets (supports multiple workers) vs IterableDataset for streaming
  - Automatic worker adjustment based on dataset type (cached vs streaming)
  - Pre-tokenization detection: SFTTrainer automatically skips tokenization when dataset has `input_ids` column
- **Configuration Enhancements**:
  - `dataloader_num_workers` config option (default: 12 for H100)
  - `dataloader_pin_memory` config option (default: true)
  - `dataloader_prefetch_factor` config option (default: 4)
  - `clear_cache_every_n_steps` config option (default: 500)
  - `use_flash_attention` config option for Flash Attention 2
  - `deterministic` config option in misc section (default: false for performance, true for reproducibility)
  - `model_card_frequency` config option to control model card generation frequency

### Changed
- **Training Performance**:
  - Default batch size increased from 8 to 16 for H100 systems
  - Gradient accumulation reduced from 6 to 4 (effective batch size: 64 vs 48)
  - Dataset caching enabled by default (`use_cache: true`) for better throughput
  - Shuffling enabled by default when using cached datasets
  - Model card generation optimized with buffered I/O and configurable frequency
- **Memory Management**:
  - Less aggressive GPU cache clearing (every 500 steps vs every 100 steps) for H100
  - Memory optimization callback only clears cache periodically, not on every log

### Fixed
- **Dataset Loading**:
  - Fixed "Too many dataloader workers" warning by using Dataset.filter() instead of Dataset.from_list() for cached datasets
  - Proper multi-shard dataset support for parallel data loading
  - Streaming mode now correctly uses 0 workers (workers don't work well with IterableDataset generators)
- **Training Speed**:
  - Resolved training speed degradation by optimizing dataset loading and tokenization
  - Pre-tokenization eliminates bottleneck during SFTTrainer initialization

### Performance
- **Massive speedup for H100 systems**:
  - Tokenization: ~35 minutes → ~2-3 minutes (10-15x faster with parallel processing)
  - Data loading: 2-3x faster with cached datasets and multiple workers
  - Overall training: Expected ~30-50% faster training iterations
  - Better GPU utilization with larger batch sizes and optimized data pipeline

## [2.5.1] - 2025-11-18

### Fixed
- **Logging alignment**: Fixed `logging_steps` to align with `gradient_accumulation` in config files
  - Changed `logging_steps` from 10 to 12 (multiple of gradient_accumulation: 6)
  - Fixes "ghost gradient" issue where `grad_norm` showed 0.0 due to logging during micro-steps
  - Now logs at steps aligned with actual gradient updates (12, 24, 36...) for accurate gradient monitoring

## [2.5.0] - 2025-11-18

### Added
- **Configuration improvements**:
  - `target_modules` now uses list format in YAML configs (backward compatible with semicolon-separated strings)
  - Model name validation with warnings for unknown base models
  - Configurable `idiomatic_quality_ratio` parameter (default: 2.0) for dataset filtering
- **Evaluation improvements**:
  - Configurable pre-filtering thresholds (`--pre-filter-min-length`, `--pre-filter-min-lines`)
  - Optional pre-filtering checks (`--pre-filter-no-main-check`, `--pre-filter-no-incomplete-check`)
  - Separate timeouts for compilation (`--compile-timeout`) and Clippy (`--clippy-timeout`)
- **Generation improvements**:
  - `--prompts-file` argument to load prompts from YAML or JSON files
  - Improved code fence extraction (handles ```rust, ```rs, missing closing fences)
- **Filter statistics export**: Save filter statistics to JSON/CSV files for analysis
- **Checkpoint inspection**: `--json` flag for machine-readable checkpoint information
- **Model export improvements**:
  - Base model validation (warns if checkpoint was trained with different base model)
  - Disk space checking before export
  - Better error handling for disk space and permission issues

### Changed
- **Training script**:
  - Replaced custom `Tee` class with Python's `logging` module
  - Abstracted TRL version handling into `_create_sft_trainer()` helper function
  - Improved error messages with specific exceptions and helpful hints
  - Documented checkpoint resumption behavior (automatic optimizer state handling)
- **Configuration loading**: Better error messages for YAML parsing and validation failures
- **Hyperparameter sweep**: Added `update_nested_config()` helper for safe nested config updates

### Fixed
- Improved error handling throughout codebase with specific exception types
- Better code extraction in `gen_eval_samples.py` for various code fence formats
- Fixed indentation issues in data filtering code

## [2.0.0] - 2025-01-XX

### Added
- **Security**: Docker-based sandboxing for Rust code evaluation (`eval_sandbox.py`)
  - All cargo commands now run in isolated Docker containers by default
  - Network isolation, memory limits (512MB), CPU limits (1 core), read-only filesystem
  - Firejail fallback support for systems without Docker
  - Auto-detection of available sandboxing tools with warnings when unavailable
  - `--sandbox-mode` CLI argument for explicit control (docker/firejail/none/auto)
  - Automatic Docker image building (`rust-eval-sandbox`) on first use
  - Comprehensive security documentation in README with Docker installation instructions
- Parallel evaluation support in `eval_rust.py` for faster compilation checks
- Template project reuse (`eval_template.py`) to avoid `cargo new` overhead per sample
- Seed propagation across all scripts for reproducibility
- Filter telemetry showing pass/filter statistics during dataset loading
- Proper `use_cache` flag implementation controlling streaming vs cached datasets
- Deterministic training with CuDNN settings for reproducible results
- Expert Iteration metadata logging with seed information for dataset reproducibility
- `launch_tensorboard.sh` script for easy TensorBoard setup with warning suppression
- PEFT checkpoint auto-detection in `gen_eval_samples.py` (automatically finds latest checkpoint)

### Fixed
- `use_cache` flag now properly controls streaming mode with clear logic
- Hyperparameter sweep now uses `copy.deepcopy` to prevent config state leakage
- Evaluation scripts now support seed arguments for reproducible sample selection
- Training script now properly sets seeds from config for deterministic runs
- Expert Iteration seed is now logged in metadata.json for dataset reproducibility
- `gen_eval_samples.py` now uses proper chat template for instruct models (fixes baseline evaluation)

### Changed
- **Breaking**: Updated default PyTorch version from 2.4.0 to 2.6.0 with CUDA 12.4 support
  - Required for secure checkpoint loading (PyTorch 2.6+ enforces `weights_only=True` by default)
  - Updated `training_setup.sh` and README to use cu124 wheels (recommended for H100/Hopper)
  - FlashAttention must be reinstalled after PyTorch upgrade (documented in README)
- Model card generation now uses MIT License and updated citation format
- `eval_rust.py` CLI migrated from positional arguments to argparse for better ergonomics
- `eval_rust.py` and `expert_iteration.py` now require sandboxing by default (can be disabled with `--no-sandbox` for local dev)
- `gen_eval_samples.py` now accepts `--model-path` and `--seed` arguments
- `gen_eval_samples.py` now automatically detects and loads PEFT (LoRA) checkpoints
- `gen_eval_samples.py` now auto-finds latest checkpoint when given a directory path
- `gen_eval_samples.py` default model path changed to `out/llama8b-rust-qlora-phase1`
- `expert_iteration.py` now accepts `--seed` argument for reproducible generation
- `expert_iteration.py` now supports sandboxed evaluation via `--sandbox-mode` argument
- `hyperparameter_sweep.py` now accepts `--seed` argument
- Phase 1 (`llama8b-phase1.yml`) is now the default configuration
- Removed `llama8b.yml` config file (replaced by Phase 1/Phase 2 structure)

### Security
- **Critical**: Added Docker-based sandboxing for all Rust code evaluation
  - Prevents arbitrary code execution from malicious LLM-generated code (build.rs, macros, etc.)
  - All `cargo check` and `cargo clippy` commands now run in isolated containers by default
  - Addresses critical security vulnerability identified in code review (EVALUATION.txt)
  - Sandboxing enforced in both `eval_rust.py` and `expert_iteration.py`
  - Clear warnings when sandboxing is disabled or unavailable
  - Production-ready isolation suitable for evaluating untrusted LLM-generated code

### Performance
- Evaluation throughput significantly improved with parallel processing
- Dataset loading now reports filter statistics for debugging
- Better control over RAM vs throughput tradeoffs via `use_cache` flag

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- QLoRA fine-tuning pipeline
- Enhanced dataset filtering with idiomatic/documentation heuristics
- Comprehensive evaluation metrics
- TensorBoard logging
- Two-phase training support
- Expert Iteration / Rejection Sampling Fine-Tuning (RSFT)
- Hyperparameter sweep script
- Standard Python package structure

