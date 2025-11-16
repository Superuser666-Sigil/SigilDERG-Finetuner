# Changelog

All notable changes to SigilDERG-Finetuner will be documented in this file.

## [Unreleased]

### Added
- Parallel evaluation support in `eval_rust.py` for faster compilation checks
- Template project reuse (`eval_template.py`) to avoid `cargo new` overhead per sample
- Seed propagation across all scripts for reproducibility
- Filter telemetry showing pass/filter statistics during dataset loading
- Proper `use_cache` flag implementation controlling streaming vs cached datasets
- Deterministic training with CuDNN settings for reproducible results
- RLAIF metadata logging with seed information for dataset reproducibility

### Fixed
- `use_cache` flag now properly controls streaming mode with clear logic
- Hyperparameter sweep now uses `copy.deepcopy` to prevent config state leakage
- Evaluation scripts now support seed arguments for reproducible sample selection
- Training script now properly sets seeds from config for deterministic runs
- RLAIF seed is now logged in metadata.json for dataset reproducibility

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

