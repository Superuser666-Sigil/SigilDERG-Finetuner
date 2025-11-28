# ADR-004: H100 GPU Optimizations

## Status

Accepted

## Context

NVIDIA H100 GPUs represent the current state-of-the-art for LLM training. They offer:
- 80GB HBM3 memory
- 4th generation Tensor Cores
- Transformer Engine for automatic mixed precision
- NVLink for multi-GPU communication

To maximize training efficiency and reduce costs, we need to leverage H100-specific features while maintaining compatibility with other GPUs.

## Decision

### Memory Optimization

1. **Gradient Checkpointing**: Enable by default to trade compute for memory
   ```yaml
   train:
     grad_checkpointing: true
   ```

2. **8-bit Optimizer**: Use paged AdamW to reduce optimizer state memory
   ```yaml
   train:
     optimizer: paged_adamw_8bit
   ```

3. **Sequence Packing**: Pack multiple sequences per batch for efficiency
   ```yaml
   pack: true
   max_seq_len: 4096
   ```

### Compute Optimization

1. **BFloat16 Training**: Use BF16 for numerical stability on H100
   ```yaml
   train:
     bf16: true
   bnb_4bit:
     compute_dtype: bfloat16
   ```

2. **Flash Attention 2**: Enable when available for O(n) attention memory
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       attn_implementation="flash_attention_2",
   )
   ```

3. **Batch Size Tuning**: Maximize batch size for H100's large memory
   ```yaml
   train:
     micro_batch_size: 8
     gradient_accumulation: 6
     # Effective batch size: 48
   ```

### Multi-GPU Scaling

1. **Data Parallelism**: Use accelerate for multi-GPU training
2. **Gradient Accumulation**: Balance memory and throughput
3. **Worker Configuration**: Optimize dataloader workers for NVLink

## Consequences

### Positive

- **Training Speed**: 2-3x faster than A100 on equivalent workloads
- **Memory Efficiency**: 80GB allows larger batch sizes and sequences
- **Numerical Stability**: BF16 + Tensor Cores provide stable training
- **Cost Efficiency**: Faster training = lower cloud compute costs

### Negative

- **Hardware Lock-in**: Some optimizations are H100-specific
- **Complexity**: More configuration parameters to tune
- **Debugging**: Optimizations can mask underlying issues
- **Compatibility**: Flash Attention requires specific CUDA versions

## Alternatives Considered

### Option 1: Conservative Defaults

Use minimal optimization, prioritize compatibility.

Rejected because:
- Underutilizes expensive hardware
- Significantly slower training
- Higher cloud costs

### Option 2: Full FP32 Training

Avoid mixed precision entirely.

Rejected because:
- 2x memory usage
- No Tensor Core utilization
- Slower training speed

### Option 3: DeepSpeed ZeRO

Use DeepSpeed for memory optimization.

Considered but deferred because:
- Adds significant complexity
- QLoRA already provides sufficient memory savings
- Can be added later for multi-node training

## Related

- [ADR-001: QLoRA Architecture](ADR-001-qlora-architecture.md)
- NVIDIA H100 Datasheet
- Flash Attention Paper: https://arxiv.org/abs/2205.14135

