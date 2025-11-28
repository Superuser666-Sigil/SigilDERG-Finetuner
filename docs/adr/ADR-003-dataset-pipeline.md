# ADR-003: Dataset Pipeline Integration

## Status

Accepted

## Context

The SigilDERG ecosystem consists of three main components:

1. **sigil-pipeline**: Data generation and processing
2. **sigilderg-finetuner**: Model training (this project)
3. **human-eval-rust**: Evaluation and benchmarking

Data flows from pipeline to finetuner to evaluator. We needed:
- Seamless data format compatibility
- Support for multiple dataset sources
- Efficient loading for large datasets
- Quality filtering at training time

## Decision

### Data Format

Adopt the `{prompt, gen}` format as the primary training data schema:

```json
{
  "prompt": "fn calculate_sum(values: &[i32]) -> i32 {",
  "gen": "    values.iter().sum()\n}\n",
  "_source": "synthetic",
  "_split": "train",
  "_task_type": "completion"
}
```

### Multi-Source Loading

Support multiple dataset sources via prefixes:

```yaml
dataset:
  names:
    - "ammarnasr/the-stack-rust-clean"    # HuggingFace
    - "local:./data/synthetic.jsonl"       # Local JSONL
    - "parquet:./data/output.parquet"      # Parquet files
```

### Streaming vs Cached Mode

Provide both modes based on hardware constraints:

- **Cached Mode** (`use_cache: true`): Pre-load and tokenize for multi-worker training
- **Streaming Mode** (`use_cache: false`): Stream from disk for memory-constrained environments

### Filtering Integration

Apply sigil-pipeline compatible filters at load time:

```yaml
dataset:
  exclude_tests: true
  exclude_benches: true
  prefer_idiomatic: true
  prefer_documented: true
  idiomatic_quality_ratio: 2.0
```

## Consequences

### Positive

- **Ecosystem Compatibility**: Seamless data flow between components
- **Flexibility**: Multiple source types and loading strategies
- **Quality Control**: Filtering integrated into training pipeline
- **Scalability**: Streaming mode handles arbitrarily large datasets

### Negative

- **Format Lock-in**: Changing the schema requires ecosystem-wide updates
- **Complexity**: Multiple loaders and modes to maintain
- **Performance Variance**: Streaming mode is slower than cached

## Alternatives Considered

### Option 1: HuggingFace Datasets Only

Use only HuggingFace Hub datasets.

Rejected because:
- Cannot use synthetic data from sigil-pipeline
- No support for private/local datasets
- Upload overhead for generated data

### Option 2: Custom Binary Format

Design a custom format optimized for training.

Rejected because:
- Ecosystem incompatibility
- Additional tooling required
- JSONL is widely supported and debuggable

### Option 3: Database Backend

Store training data in a database.

Rejected because:
- Overkill for sequential training access
- Added infrastructure complexity
- Not compatible with standard ML tooling

## Related

- [ADR-002: Two-Phase Training Strategy](ADR-002-two-phase-training.md)
- sigil-pipeline documentation
- human-eval-rust documentation

