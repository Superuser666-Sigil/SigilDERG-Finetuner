# Changelog

All notable changes to SigilDERG-Finetuner will be documented in this file.

## [Unreleased]

## [2.5.0] - 2025-01-XX

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

