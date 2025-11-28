# Troubleshooting Runbook

## Common Issues and Solutions

### Out of Memory (OOM) Errors

**Symptoms:**
- `CUDA out of memory` error
- Training crashes suddenly
- GPU memory utilization at 100%

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   train:
     micro_batch_size: 4  # Reduce from 8
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   train:
     grad_checkpointing: true
   ```

3. **Reduce sequence length:**
   ```yaml
   max_seq_len: 2048  # Reduce from 4096
   ```

4. **Use streaming mode:**
   ```yaml
   dataset:
     use_cache: false
   ```

### Training Divergence

**Symptoms:**
- Loss increases or oscillates wildly
- NaN values in loss
- Model outputs garbage

**Solutions:**

1. **Lower learning rate:**
   ```yaml
   train:
     lr: 5.0e-5  # Reduce from 1.0e-4
   ```

2. **Increase warmup:**
   ```yaml
   train:
     warmup_steps: 500  # Increase from 250
   ```

3. **Enable gradient clipping:**
   ```yaml
   train:
     max_grad_norm: 0.5  # Reduce from 1.0
   ```

### Slow Training

**Symptoms:**
- Training progress much slower than expected
- Low GPU utilization
- High CPU wait times

**Solutions:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1
   # Should show 90%+ utilization
   ```

2. **Increase dataloader workers:**
   ```yaml
   train:
     dataloader_num_workers: 12
   ```

3. **Enable Flash Attention:**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

4. **Use cached mode:**
   ```yaml
   dataset:
     use_cache: true
   ```

### Model Loading Errors

**Symptoms:**
- `HTTPError: 401 Unauthorized`
- `Cannot find model`
- `Token required`

**Solutions:**

1. **Set HuggingFace token:**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

2. **Login to HuggingFace:**
   ```bash
   huggingface-cli login
   ```

3. **Request model access:**
   - Visit model page on HuggingFace Hub
   - Accept license agreement

### Dataset Loading Errors

**Symptoms:**
- `FileNotFoundError: local:./path.jsonl`
- `Dataset not found on HuggingFace Hub`
- Empty dataset after filtering

**Solutions:**

1. **Check file path:**
   ```bash
   ls -la ./data/  # Verify file exists
   ```

2. **Verify dataset name:**
   ```python
   from datasets import load_dataset
   ds = load_dataset("dataset/name")
   ```

3. **Relax filters:**
   ```yaml
   dataset:
     min_length: 10  # Lower threshold
     prefer_idiomatic: false
   ```

### Evaluation Errors

**Symptoms:**
- `SandboxError: Docker not available`
- `Compilation timeout`
- Low pass rate

**Solutions:**

1. **Install sandbox:**
   ```bash
   # Docker
   docker --version
   
   # Or Firejail
   sudo apt install firejail
   ```

2. **Increase timeout:**
   ```bash
   python eval_rust.py --compile-timeout 60
   ```

3. **Check Rust installation:**
   ```bash
   rustc --version
   cargo --version
   ```

## Diagnostic Commands

### Check GPU Status
```bash
nvidia-smi
```

### Check CUDA Version
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Check Package Versions
```bash
pip list | grep -E "torch|transformers|bitsandbytes|peft|trl"
```

### Test Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print("Model loaded successfully")
```

### Test Dataset Loading
```python
from datasets import load_dataset
ds = load_dataset("ammarnasr/the-stack-rust-clean", split="train", streaming=True)
print(next(iter(ds)))
```

## Getting Help

1. **Check logs:** `cat out/*/train.log`
2. **Search issues:** [GitHub Issues](https://github.com/Superuser666-Sigil/SigilDERG-Finetuner/issues)
3. **Ask in discussions:** [GitHub Discussions](https://github.com/Superuser666-Sigil/SigilDERG-Finetuner/discussions)

