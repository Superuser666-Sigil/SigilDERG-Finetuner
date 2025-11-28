# ADR-002: Two-Phase Training Strategy

## Status

Accepted

## Context

Training an LLM on Rust code requires balancing two objectives:

1. **Breadth**: Learning general Rust syntax, patterns, and idioms
2. **Depth**: Mastering specific coding tasks with high quality

A single-phase approach risks either:
- Overfitting to low-quality examples early in training
- Underfitting on high-quality examples due to noise

We needed a training strategy that:
- Prioritizes learning fundamentals first
- Refines behavior on high-quality examples
- Provides clear checkpoints for evaluation
- Supports curriculum learning principles

## Decision

Implement a two-phase training strategy:

### Phase 1: Foundation Training
```yaml
# phase1-config.yml
dataset:
  names: [ammarnasr/the-stack-rust-clean]
  prefer_idiomatic: false
  prefer_documented: false
train:
  num_steps: 12000
  lr: 1.0e-4
```

- Train on the full filtered dataset
- No quality preferences (idiomatic/documented)
- Higher learning rate for faster convergence
- Goal: Learn Rust syntax and common patterns

### Phase 2: Refinement Training
```yaml
# phase2-config.yml
dataset:
  names: [ammarnasr/the-stack-rust-clean]
  prefer_idiomatic: true
  prefer_documented: true
  idiomatic_quality_ratio: 2.0
train:
  num_steps: 6000
  lr: 5.0e-5
```

- Continue from Phase 1 checkpoint
- Filter for idiomatic, well-documented code
- Lower learning rate for stable refinement
- Goal: Polish code quality and style

## Consequences

### Positive

- **Curriculum Learning**: Model learns fundamentals before refinement
- **Quality Control**: Phase 2 filters ensure high-quality final behavior
- **Checkpoint Flexibility**: Can stop after Phase 1 for a "general" model
- **Debugging**: Clear separation helps identify training issues
- **Resource Efficiency**: Phase 2 uses smaller, higher-quality dataset

### Negative

- **Training Time**: Total training time is longer than single-phase
- **Hyperparameter Tuning**: Two sets of hyperparameters to optimize
- **Checkpoint Management**: Need to track Phase 1 → Phase 2 handoff
- **Complexity**: More complex training pipeline to maintain

## Alternatives Considered

### Option 1: Single-Phase Training

Train on the full dataset in one pass.

Rejected because:
- No curriculum structure
- Low-quality examples may dominate early training
- Harder to control final model behavior

### Option 2: Progressive Filtering

Gradually increase quality thresholds during training.

Rejected because:
- Complex implementation
- Difficult to tune filter schedules
- Less clear checkpoints for evaluation

### Option 3: Multi-Task Learning

Train on multiple objectives simultaneously.

Rejected because:
- Adds complexity without clear benefit
- Requires task-specific heads or prompts
- Our use case benefits from sequential learning

## Related

- [ADR-001: QLoRA Architecture](ADR-001-qlora-architecture.md)
- [ADR-003: Dataset Pipeline Integration](ADR-003-dataset-pipeline.md)
- Curriculum Learning: https://arxiv.org/abs/1904.03626

