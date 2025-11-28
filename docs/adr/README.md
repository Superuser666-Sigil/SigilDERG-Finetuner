# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the SigilDERG-Finetuner project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

Each ADR follows a consistent format:

- **Status**: Draft, Proposed, Accepted, Deprecated, Superseded
- **Context**: The issue motivating the decision
- **Decision**: The decision and its justification
- **Consequences**: The resulting effects, positive and negative
- **Alternatives Considered**: Other options that were evaluated
- **Related**: Links to related ADRs or documentation

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-qlora-architecture.md) | QLoRA Architecture for Rust LLM Fine-tuning | Accepted | 2025-01 |
| [ADR-002](ADR-002-two-phase-training.md) | Two-Phase Training Strategy | Accepted | 2025-01 |
| [ADR-003](ADR-003-dataset-pipeline.md) | Dataset Pipeline Integration | Accepted | 2025-01 |
| [ADR-004](ADR-004-h100-optimizations.md) | H100 GPU Optimizations | Accepted | 2025-01 |

## Creating a New ADR

1. Copy the [template](template.md)
2. Fill in the sections
3. Add entry to this index
4. Submit for review

