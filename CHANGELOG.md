# Changelog

All notable changes to SigilDERG-Finetuner will be documented in this file.

## [Unreleased]

### Added
- Parallel evaluation support in `eval_rust.py` for faster compilation checks
- Seed propagation across all scripts for reproducibility
- Filter telemetry showing pass/filter statistics during dataset loading
- Proper `use_cache` flag implementation controlling streaming vs cached datasets
- Deterministic training with CuDNN settings for reproducible results

### Fixed
- `use_cache` flag now actually controls streaming mode (was previously ignored)
- Evaluation scripts now support seed arguments for reproducible sample selection
- Training script now properly sets seeds from config for deterministic runs

### Changed
- `eval_rust.py` CLI migrated from positional arguments to argparse for better ergonomics
- `gen_eval_samples.py` now accepts `--model-path` and `--seed` arguments
- `rlaif_lite.py` now accepts `--seed` argument for reproducible generation
- `hyperparameter_sweep.py` now accepts `--seed` argument

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
- RLAIF-lite synthetic reward training
- Hyperparameter sweep script
- Standard Python package structure

