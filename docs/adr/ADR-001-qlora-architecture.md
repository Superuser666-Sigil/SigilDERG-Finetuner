# ADR-001: QLoRA Architecture for Rust LLM Fine-tuning

## Status

Accepted

## Context

Fine-tuning large language models (LLMs) for Rust code generation requires significant computational resources. A full fine-tune of an 8B parameter model requires 60+ GB of GPU memory, making it impractical for most development and experimentation scenarios.

We needed an approach that:
- Enables fine-tuning on consumer-grade GPUs (24GB VRAM)
- Maintains model quality comparable to full fine-tuning
- Supports the latest Llama 3 family of models
- Integrates with the SigilDERG ecosystem

## Decision

We adopt QLoRA (Quantized Low-Rank Adaptation) as our fine-tuning architecture:

1. **4-bit NF4 Quantization**: Use bitsandbytes NF4 quantization to reduce memory footprint by 4x
2. **Double Quantization**: Apply secondary quantization to the quantization constants for additional memory savings
3. **LoRA Adapters**: Train low-rank adapter matrices instead of full weight updates
4. **BFloat16 Compute**: Use BF16 for computation to maintain numerical stability

Configuration:
```yaml
bnb_4bit:
  quant_type: nf4
  compute_dtype: bfloat16
  use_double_quant: true

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
```

## Consequences

### Positive

- **Memory Efficiency**: Reduces memory requirements from 60GB to ~16GB for 8B models
- **Training Speed**: Faster training due to reduced gradient computation
- **Quality Preservation**: Research shows QLoRA maintains 99%+ of full fine-tune quality
- **Ecosystem Compatibility**: Works with Hugging Face transformers, TRL, and PEFT libraries
- **Portability**: Trained adapters are small (~100MB) and can be merged or swapped

### Negative

- **Quantization Overhead**: Initial model quantization adds startup time
- **Inference Overhead**: Merged models may have slightly higher inference latency
- **Compatibility Constraints**: Requires bitsandbytes library (Linux-first, CUDA-only)
- **Debugging Complexity**: Quantized weights are harder to inspect and debug

## Alternatives Considered

### Option 1: Full Fine-tuning

Full parameter updates without quantization.

Rejected because:
- Requires 60+ GB VRAM (multi-GPU setup)
- 10x longer training time
- Higher infrastructure costs
- Marginal quality improvement doesn't justify the cost

### Option 2: LoRA without Quantization

Standard LoRA on FP16/BF16 models.

Rejected because:
- Still requires 32+ GB VRAM for 8B models
- Excludes many development environments
- Double the memory cost of QLoRA for minimal benefit

### Option 3: Full Quantization (GPTQ/AWQ)

Post-training quantization only, no fine-tuning.

Rejected because:
- Cannot adapt model to Rust-specific patterns
- No domain-specific learning capability
- Insufficient for our use case

## Related

- [ADR-002: Two-Phase Training Strategy](ADR-002-two-phase-training.md)
- [ADR-004: H100 GPU Optimizations](ADR-004-h100-optimizations.md)
- QLoRA Paper: https://arxiv.org/abs/2305.14314

