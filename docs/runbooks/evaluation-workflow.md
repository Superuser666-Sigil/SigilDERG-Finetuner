# Evaluation Workflow Runbook

## Prerequisites

- Trained model checkpoint or merged model
- sigilderg-finetuner with evaluation dependencies: `pip install -e .[evaluation]`
- Rust toolchain installed: `rustup default stable`
- Docker or Firejail for sandboxed evaluation (recommended)

## Quick Evaluation

### 1. Generate Samples

```bash
python eval_rust.py \
  --model-path out/phase2/checkpoint-final \
  --num-samples 100 \
  --output samples.jsonl
```

### 2. Run Compile Check

```bash
python eval_rust.py \
  --input samples.jsonl \
  --check-compile \
  --output results.json
```

Expected output:
```json
{
  "total": 100,
  "compiled": 82,
  "compile_rate": 0.82,
  "clippy_pass_rate": 0.75,
  "avg_warnings": 1.2
}
```

## Full HumanEval Evaluation

### 1. Install human-eval-rust

```bash
pip install human-eval-rust>=2.3.0
```

### 2. Generate Completions

```bash
python scripts/generate_completions.py \
  --model-path out/merged-model \
  --problems-path data/HumanEval_rust.jsonl \
  --output completions.jsonl \
  --n-samples 10
```

### 3. Run Functional Correctness

```bash
evaluate_functional_correctness \
  --sample_file completions.jsonl \
  --n_workers 8 \
  --timeout 10
```

Expected output:
```
{'pass@1': 0.35, 'pass@10': 0.58}
```

## Sandbox Modes

### Docker (Recommended)

```bash
# Build sandbox image
docker build -f rust-qlora/Dockerfile.eval -t rust-eval .

# Run with Docker sandbox
python eval_rust.py --sandbox-mode docker ...
```

### Firejail (Linux)

```bash
# Install Firejail
sudo apt install firejail

# Run with Firejail sandbox
python eval_rust.py --sandbox-mode firejail ...
```

### No Sandbox (Development Only)

```bash
# WARNING: Only for trusted code
python eval_rust.py --sandbox-mode none ...
```

## Metrics Interpretation

| Metric | Description | Target |
|--------|-------------|--------|
| compile_rate | % of samples that compile | > 80% |
| clippy_pass_rate | % without Clippy warnings | > 70% |
| pass@1 | Single-attempt functional correctness | > 30% |
| pass@10 | Best-of-10 functional correctness | > 50% |
| doc_coverage | % with doc comments | > 50% |
| idiomatic_score | Average idiomaticity | > 0.6 |

## Comparing Models

### Generate Comparison Report

```bash
python scripts/compare_models.py \
  --baseline out/baseline-model \
  --candidate out/new-model \
  --output comparison.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Sandbox Not Available | Install Docker or Firejail |
| Timeout Errors | Increase timeout or reduce complexity |
| Low Compile Rate | Check for syntax issues in generation |
| Rust Not Found | Verify `rustc` is in PATH |

