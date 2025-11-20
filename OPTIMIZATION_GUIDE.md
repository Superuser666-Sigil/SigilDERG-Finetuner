# Optimization Guide: Achieving 95% Compile Rate

This guide outlines the strategy for pushing the SigilDERG-Finetuner toward:
- **compile_rate ≥ 0.95**
- **avg_clippy_warnings** as low as possible
- **avg_idiomatic_score ≥ 0.7-0.8**

## Overview

The optimization strategy uses multiple techniques:
1. Stricter dataset filtering for high-quality training data
2. Improved evaluation prompts for better compile rates
3. Two-phase training (broad → sharpening)
4. Expert Iteration / Rejection Sampling Fine-Tuning (RSFT)
5. Hyperparameter tuning

## 1. Dataset Filtering

### Default Configuration

The main config (`configs/llama8b-phase1.yml`) is designed for broad Phase 1 training:

```yaml
dataset:
  min_length: 64
  max_length: 200000
  exclude_tests: true
  exclude_examples: false
  exclude_benches: true
  prefer_idiomatic: false
  prefer_documented: false
  shuffle_seed: 42
```

Phase 1 intentionally keeps idiomatic/documentation requirements relaxed so the model sees a wide distribution of Rust code. Turn those flags on (or tighten other thresholds) when you move to sharpening or Expert Iteration loops.

### What Gets Filtered

**Path-based exclusions:**
- Vendor directories, node_modules, lock files
- Test files (`/tests/`, `#[cfg(test)]`)
- Benchmark files (`/benches/`, `#[cfg(bench)]`)
- Example files (`/examples/`)

**Quality heuristics (when `prefer_idiomatic`/`prefer_documented` are enabled):**
- **Idiomatic patterns enforced:** Result/Option handling, iterator chains, derive macros, trait implementations, public API
- **Configurable quality ratio:** The `idiomatic_quality_ratio` parameter (default: 2.0) controls how much idiomatic patterns must outnumber low-quality markers. Higher values are more strict.
- **Documentation required:** Must have doc comments (`///`, `//!`, `/**`)
- **Low-quality markers down-weighted:** TODO/FIXME, debug prints, unsafe blocks, suppressed warnings
- **Filter statistics export:** Filter statistics can be saved to JSON/CSV files for analysis

### Tightening Filters Further

To be even more strict, you can modify `data_filters.py`:

```python
LOW_QUALITY_PATTERNS = [
    re.compile(r"TODO|FIXME|XXX|HACK", re.IGNORECASE),
    re.compile(r"println!\s*\(|dbg!\s*\("),
    re.compile(r"unsafe\s+\{"),
    re.compile(r"#\[allow\([^)]+\)\]"),
    # Add more patterns:
    # re.compile(r"^\s*use\s+.*;\s*$", re.MULTILINE),  # Excessive imports
    # re.compile(r"\bmut\s+\w+\s*="),  # Excessive mutability
]
```

### Dataset Loading Modes

The `use_cache` flag controls how datasets are loaded:

```yaml
dataset:
  use_cache: true  # true = non-streaming (better throughput), false = streaming (lower RAM)
  shuffle_seed: 42  # Enable shuffling (requires non-streaming mode)
```

**Dataset Loading Modes:**

- **`use_cache: true`** (default): Non-streaming mode
  - Better throughput and faster training
  - Uses Dataset.filter() for multi-worker support (much faster data loading)
  - Pre-tokenization with parallel processing (10-15x faster tokenization)
  - Requires more RAM (dataset loaded into memory)
  - Recommended for systems with sufficient RAM (200GB+ for H100)
  
- **`use_cache: false`**: Streaming mode
  - Lower RAM usage
  - Slower throughput (network I/O for each batch)
  - Uses IterableDataset with 0 workers (workers don't work well with streaming)
  - Recommended for very large datasets or memory-constrained systems

**Note:** Filter statistics are automatically printed during training, showing how many samples passed vs. were filtered. This helps verify filters aren't over-pruning your dataset. Filter statistics can also be exported to JSON/CSV files for detailed analysis.

### H100 GPU Optimizations

For H100 systems with 25 vCPUs, 200GB RAM, and 1TB SSD, the following optimizations are enabled by default:

```yaml
train:
  micro_batch_size: 16  # Increased from 8 for H100
  gradient_accumulation: 4  # Reduced from 6 (effective batch: 64)
  use_flash_attention: true  # Enable Flash Attention 2
  dataloader_num_workers: 12  # Use ~half of vCPUs for data loading
  dataloader_pin_memory: true  # Faster CPU-GPU transfers
  dataloader_prefetch_factor: 4  # Prefetch batches ahead
  clear_cache_every_n_steps: 500  # Less frequent cache clearing

misc:
  deterministic: false  # Enable CuDNN benchmark mode for speed
```

**Performance Improvements:**
- **Tokenization**: Pre-tokenization with 25 parallel workers reduces time from ~35 minutes to ~2-3 minutes
- **Data Loading**: 2-3x faster with cached datasets and multiple workers
- **Training Speed**: 30-50% faster iterations with optimized pipeline
- **GPU Utilization**: Better utilization with larger batch sizes and Flash Attention 2

## 2. Evaluation Improvements

### Better Prompts

The `gen_eval_samples.py` script:
- **Automatically detects PEFT (LoRA) checkpoints** and loads them correctly
- **Finds the latest checkpoint** if you provide a checkpoint directory (e.g., `out/llama8b-rust-qlora-phase1`)
- **Falls back to full model loading** for merged checkpoints or base models
- Uses prompts that explicitly request code-only output
- Ask for complete `fn main()` programs
- Request code wrapped in ```rust blocks
- Use system prompts to enforce code-only generation
- Supports `--seed` argument for reproducible generation

### Pre-filtering

The `eval_rust.py` script now pre-filters samples before compilation with configurable thresholds:
- Skips samples without `fn main` (can be disabled with `--pre-filter-no-main-check`)
- Skips samples that are mostly comments (configurable via `--pre-filter-min-lines`)
- Skips obviously incomplete code (can be disabled with `--pre-filter-no-incomplete-check`)
- Configurable minimum length (`--pre-filter-min-length`, default: 20)
- Only evaluates valid-looking samples

This improves compile_rate by not counting invalid samples as failures.

**Example with custom thresholds:**
```bash
python eval_rust.py eval_out/samples.jsonl \
    --pre-filter-min-length 50 \
    --pre-filter-min-lines 5 \
    --pre-filter-no-main-check  # Allow samples without fn main
```

### Parallel Evaluation

Evaluation now uses multiprocessing for faster compilation checks:

```bash
# Automatic parallelization (uses all but one CPU core)
python eval_rust.py eval_out/samples.jsonl

# Manual control
python eval_rust.py eval_out/samples.jsonl --num-workers 8

# Sequential (for debugging)
python eval_rust.py eval_out/samples.jsonl --num-workers 1
```

This dramatically speeds up evaluation, making it practical to:
- Evaluate larger sample sets (100+ samples)
- Run frequent evaluation during training
- Perform hyperparameter sweeps with comprehensive evaluation

## 3. Two-Phase Training

### Phase 1: Broad Rust Training

**Config:** `configs/llama8b-phase1.yml`

- Moderate filtering (idiomatic/documented optional)
- Longer sequences (4096 tokens)
- Standard learning rate (1e-4)
- Full training (12000 steps)

**Goal:** Get rich coverage of Rust patterns, not just the pretty ones.

```bash
python train.py --cfg configs/llama8b-phase1.yml
```

### Phase 2: High-Quality Sharpening

**Config:** `configs/llama8b-phase2.yml`

- Strict filtering (idiomatic/documented required)
- Shorter sequences (2048 tokens) for self-contained programs
- Lower learning rate (5e-5) for fine-tuning
- Fewer steps (4000) on high-quality data
- **Multi-GPU optimized**: Same batch size scaling as Phase 1 (effective batch: 64)
- **H100 optimizations**: Flash Attention 2, parallel data loading (48 workers), optimized prefetching

**Goal:** Sharpen behavior toward compilable, idiomatic code.

```bash
# Automatically loads from Phase 1 checkpoint
bash scripts/run_phase2.sh

# Or manually:
python train.py --cfg configs/llama8b-phase2.yml
```

**Multi-GPU Scaling for Phase 2:**

Phase 2 uses the same per-GPU batch size scaling as Phase 1 to maintain consistent effective batch size:

| GPUs | micro_batch_size | gradient_accumulation | Effective batch |
|------|-----------------|-----------------------|-----------------|
| 1    | 16              | 4                     | 64              |
| 2    | 8               | 4                     | 64              |
| 4    | 4               | 4                     | 64              |
| 8    | 2               | 4                     | 64              |

The shorter sequence length (2048 vs 4096) in Phase 2 allows for efficient multi-GPU scaling while maintaining the same effective batch size as Phase 1.

Make sure to set `misc.load_from` in the Phase 2 config to point to your Phase 1 checkpoint.

## 4. Expert Iteration / Rejection Sampling Fine-Tuning (RSFT)

The `expert_iteration.py` script implements Expert Iteration (also known as Rejection Sampling Fine-Tuning):

1. **Generate samples** from current model
2. **Evaluate** each sample (compile, clippy, idiomatic, doc)
3. **Filter** to keep only good samples:
   - Compiles successfully
   - Clippy warnings ≤ threshold (default: 2)
   - Idiomatic score ≥ threshold (default: 0.7)
   - Doc comment rate ≥ threshold (default: 0.5)
4. **Create training dataset** from good samples
5. **Fine-tune** on this self-generated, self-vetted data

### Usage

```bash
# Generate and filter high-quality samples
python expert_iteration.py \
    --model-path out/llama8b-rust-qlora-phase2 \
    --output-dir expert_iter_data \
    --num-samples 20 \
    --clippy-max 2.0 \
    --idiomatic-min 0.7 \
    --doc-min 0.5 \
    --seed 42  # For reproducible generation

# Fine-tune on the filtered data
# Create a config that uses expert_iter_data/instruction_data.jsonl or code_only.jsonl
# Use low LR (5e-5) and fewer steps (1000-2000)
```

**Performance Note:** The Expert Iteration script uses parallel evaluation internally, so filtering large sample sets (100+ samples) is much faster than before.

### Creating Training Config for Expert Iteration Data

You can create a custom config that loads from a JSONL file:

```yaml
# configs/llama8b-expert-iter.yml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
misc:
  load_from: out/llama8b-rust-qlora-phase2/checkpoint-4000
  output_dir: out/llama8b-rust-qlora-expert-iter
  # ... other settings ...
```

Then modify `train.py` or create a custom dataset loader for JSONL files.

## 5. Hyperparameter Tuning

### Key Hyperparameters

**Learning Rate:**
- Too high → unstable, weird syntax, hallucinations
- Too low → underfit
- Recommended range: 5e-5 to 5e-4
- For code: bias toward lower-mid range (1e-4 to 2e-4)
- For Phase 2/Expert Iteration: use lower (5e-5)

**Sequence Length:**
- For compile rate: keep modest (2048) to focus on complete programs
- Packing enabled: sees many short, complete programs per batch
- Giant sequences (10k tokens) less relevant for single-file compile

**LoRA Rank/Alpha:**
- More capacity: bump rank/alpha (e.g., r=32, alpha=32)
- Watch stability and VRAM
- Use `hyperparameter_sweep.py` to explore

**Training Steps:**
- Phase 1: Full training (12000 steps)
- Phase 2: Fewer steps on high-quality data (4000)
- Expert Iteration: Very few steps (1000-2000) to nudge behavior

### Running Sweeps

```bash
# Basic sweep
python hyperparameter_sweep.py \
    --base-cfg configs/llama8b-phase2.yml \
    --sweep-dir sweeps/phase2

# With seed for reproducibility
python hyperparameter_sweep.py \
    --base-cfg configs/llama8b-phase2.yml \
    --sweep-dir sweeps/phase2 \
    --seed 42

# Dry run to preview configurations
python hyperparameter_sweep.py \
    --base-cfg configs/llama8b-phase2.yml \
    --dry-run
```

View results in TensorBoard:
```bash
# Using the launch script (recommended - suppresses warnings)
bash scripts/launch_tensorboard.sh out/

# Or manually
tensorboard --logdir out/
```

**Tip:** With parallel evaluation enabled, you can evaluate each sweep run more quickly, making it practical to run comprehensive sweeps with larger sample sets.

## 6. Realistic Path to 95% Compile Rate

### Step-by-Step Workflow

1. **Phase 1 Training** (Broad coverage)
   ```bash
   python train.py --cfg configs/llama8b-phase1.yml
   ```
   Note: Training automatically uses seed from `misc.seed` in config for reproducibility.

2. **Evaluate Phase 1**
   ```bash
   python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase1 --seed 0
   python eval_rust.py eval_out/samples.jsonl --sample-n 32 --check-func --seed 0
   ```
   With parallel evaluation, this runs much faster than before.

3. **Phase 2 Training** (Sharpening)
   ```bash
   bash scripts/run_phase2.sh
   ```
   Or manually:
   ```bash
   python train.py --cfg configs/llama8b-phase2.yml
   ```

4. **Evaluate Phase 2**
   ```bash
   python gen_eval_samples.py --model-path out/llama8b-rust-qlora-phase2 --seed 0
   python eval_rust.py eval_out/samples.jsonl --sample-n 32 --check-func --seed 0
   ```

5. **Expert Iteration Loop** (if needed)
   ```bash
   python expert_iteration.py \
       --model-path out/llama8b-rust-qlora-phase2 \
       --num-samples 20 \
       --seed 42
   # Fine-tune on expert_iter_data/instruction_data.jsonl
   ```

6. **Hyperparameter Sweep** (if metrics plateau)
   ```bash
   python hyperparameter_sweep.py \
       --base-cfg configs/llama8b-phase2.yml \
       --seed 42
   ```
   With parallel evaluation, you can evaluate each sweep run quickly.

### Expected Progress

- **After Phase 1:** compile_rate ~0.70-0.80, idiomatic ~0.5-0.6
- **After Phase 2:** compile_rate ~0.85-0.90, idiomatic ~0.6-0.7
- **After Expert Iteration:** compile_rate ~0.90-0.95, idiomatic ~0.7-0.8

### Monitoring

Track metrics in `eval_out/metrics.jsonl`:
```bash
# View latest metrics
tail -1 eval_out/metrics.jsonl | python -m json.tool
```

Key metrics to watch:
- `compile_rate`: Target ≥ 0.95
- `avg_clippy_warnings`: Target ≤ 2.0
- `avg_idiomatic_score`: Target ≥ 0.7
- `doc_comment_rate`: Target ≥ 0.5
- `filtered_samples`: Number of samples pre-filtered (should be reasonable)
- `evaluated_samples`: Number of samples actually compiled

**Training Filter Telemetry:**

During training, you'll see filter statistics printed:
```
Dataset filter stats: 12345/50000 passed (24.7%), 37655 filtered
```

This helps verify your filters aren't too strict (very low pass rate) or too lenient (very high pass rate).

## 7. Generation Settings

For best results during inference:

```python
# Greedy decoding for stability
model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=False,  # Greedy
    temperature=None,
)

# Or very low temperature if sampling
model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,  # Very low
    top_p=0.9,
)
```

Use system-style prompts that force code-only outputs:
```
"You are a Rust code generator. Output only valid Rust code, wrapped in ```rust code blocks."
```

## Troubleshooting

### Compile Rate Stuck Below 0.90

1. Check prompts in `gen_eval_samples.py` - ensure they request code-only output
2. Verify pre-filtering is working - check `filter_reasons` in metrics
3. Tighten dataset filters - increase `min_length`, enable all quality heuristics
4. Try Phase 2 training with even stricter filters

### Idiomatic Score Low

1. Ensure `prefer_idiomatic: true` in dataset config
2. Check that training data actually has idiomatic patterns
3. Consider Expert Iteration to reinforce good patterns
4. Increase LoRA rank for more model capacity

### High Clippy Warnings

1. Filter training data more strictly (exclude `#[allow(...)]` patterns)
2. Use Expert Iteration with strict clippy threshold (e.g., `--clippy-max 1.0`)
3. Train longer on high-quality data

## Additional Resources

- Main README: See installation and basic usage
- Config files: `rust-qlora/configs/` for all configuration options
- Evaluation: `eval_rust.py` for comprehensive metrics
- Hyperparameter sweep: `hyperparameter_sweep.py` for systematic optimization

