# Training Execution Runbook

## Prerequisites

- Python 3.12+ installed
- CUDA-capable GPU with 24GB+ VRAM
- sigilderg-finetuner installed: `pip install -e .[dev]`
- Access to model weights (HuggingFace token if using gated models)

## Phase 1: Foundation Training

### 1. Configure Training

Create or modify `rust-qlora/configs/phase1.yml`:

```yaml
model_name: "meta-llama/Meta-Llama-3-8B"
max_seq_len: 4096
pack: true

dataset:
  names:
    - "ammarnasr/the-stack-rust-clean"
  use_cache: true
  min_length: 64
  max_length: 200000
  exclude_tests: true
  exclude_benches: true

lora:
  r: 16
  alpha: 16
  dropout: 0.05

train:
  micro_batch_size: 8
  gradient_accumulation: 6
  lr: 1.0e-4
  num_steps: 12000
  warmup_steps: 250
  grad_checkpointing: true
  bf16: true

misc:
  output_dir: "out/phase1"
  seed: 42
```

### 2. Set Environment Variables

```bash
export HF_TOKEN="your-huggingface-token"
export CUDA_VISIBLE_DEVICES=0  # or 0,1 for multi-GPU
```

### 3. Start Training

```bash
cd rust-qlora
python train.py --config configs/phase1.yml
```

### 4. Monitor Training

- **TensorBoard**: `tensorboard --logdir out/phase1/logs`
- **GPU Usage**: `nvidia-smi -l 1`
- **Loss Curve**: Check for smooth decrease

### Expected Output

```
Loading model meta-llama/Meta-Llama-3-8B...
Quantizing to 4-bit NF4...
Setting up LoRA adapters (r=16, alpha=16)...
Loading dataset...
Starting training for 12000 steps...
Step 100: loss=2.45, lr=1.0e-4
Step 200: loss=2.12, lr=1.0e-4
...
```

## Phase 2: Refinement Training

### 1. Update Configuration

Create `rust-qlora/configs/phase2.yml`:

```yaml
model_name: "meta-llama/Meta-Llama-3-8B"
max_seq_len: 4096
pack: true

dataset:
  names:
    - "ammarnasr/the-stack-rust-clean"
  use_cache: true
  prefer_idiomatic: true
  prefer_documented: true
  idiomatic_quality_ratio: 2.0

train:
  micro_batch_size: 8
  gradient_accumulation: 6
  lr: 5.0e-5  # Lower learning rate
  num_steps: 6000

misc:
  output_dir: "out/phase2"
  load_from: "out/phase1/checkpoint-final"
  seed: 42
```

### 2. Start Phase 2 Training

```bash
python train.py --config configs/phase2.yml
```

## Post-Training

### 1. Merge Adapters (Optional)

```bash
python scripts/merge_adapter.py \
  --base-model meta-llama/Meta-Llama-3-8B \
  --adapter-path out/phase2/checkpoint-final \
  --output-path out/merged-model
```

### 2. Run Evaluation

```bash
python eval_rust.py \
  --model-path out/phase2/checkpoint-final \
  --output eval_results.json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `micro_batch_size` or enable `grad_checkpointing` |
| Training Divergence | Reduce learning rate or increase warmup steps |
| Slow Training | Check GPU utilization; enable Flash Attention |
| Model Not Loading | Verify HF_TOKEN and model access permissions |

