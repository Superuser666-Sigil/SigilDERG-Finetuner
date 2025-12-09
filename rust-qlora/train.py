"""
Main training module for QLoRA fine-tuning of Rust code generation models.

Handles model loading, dataset preparation, training loop, and checkpoint management.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

import importlib
import logging
import os
import warnings
from datetime import datetime

import torch
import yaml
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

# Note: PyTorch 2.9+ requires use_reentrant parameter in torch.utils.checkpoint.checkpoint
# We now explicitly pass use_reentrant=False via gradient_checkpointing_kwargs in TrainingArguments
# and in model.gradient_checkpointing_enable() calls. The warning filters below are kept as a
# fallback for any deep library calls that may not yet support the parameter.
warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=".*torch.utils.checkpoint.*use_reentrant.*",
    category=UserWarning,
)
# Suppress PyTorch deprecation warning for torch.cpu.amp.autocast (will be fixed in future PyTorch versions)
warnings.filterwarnings(
    "ignore",
    message=".*torch.cpu.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)


def _resolve_import(module_name, fallback_module_name=None):
    """
    Helper function to resolve imports with fallback support.

    NOTE: This dynamic import resolution is a workaround for flexible packaging.
    Since the package is installed in editable mode (pip install -e .), standard
    imports should work. This function provides backward compatibility and handles
    edge cases where the package structure might differ.

    TODO: Consider simplifying to direct imports once packaging is fully standardized.

    Args:
        module_name: Primary module name (e.g., '.data_filters' or 'rust_qlora.data_filters')
        fallback_module_name: Fallback module name (e.g., 'rust_qlora.data_filters')

    Returns:
        Imported module or None if both imports fail
    """
    is_relative = module_name.startswith(".")
    package_name = __package__
    if not package_name and "." in __name__:
        package_name = __name__.rsplit(".", 1)[0]

    # Try relative import first (works when running as module)
    if is_relative and package_name:
        try:
            return importlib.import_module(module_name, package=package_name)
        except (ImportError, AttributeError, ValueError):
            pass

    # Try direct import (only for absolute module names)
    if not is_relative:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            pass

    # Try fallback
    if fallback_module_name:
        try:
            return importlib.import_module(fallback_module_name)
        except ImportError:
            pass

    return None


# Import data_filters
_data_filters_module = _resolve_import(".data_filters", "rust_qlora.data_filters")
if _data_filters_module:
    stream_rust = _data_filters_module.stream_rust
else:
    raise ImportError("Could not import data_filters. Install package in editable mode: pip install -e .")

# Import dataset loader abstraction
# Note: Using dataset_utils instead of datasets to avoid conflict with HuggingFace datasets package
_dataset_loader_module = _resolve_import(".dataset_utils.loader", "rust_qlora.dataset_utils.loader")
if _dataset_loader_module:
    DatasetLoader = _dataset_loader_module.DatasetLoader
else:
    raise ImportError("Could not import DatasetLoader. Install package in editable mode: pip install -e .")

# Import Pydantic config models
_config_models_module = _resolve_import(".config_models", "rust_qlora.config_models")
if _config_models_module:
    TrainingConfig = _config_models_module.TrainingConfig
else:
    TrainingConfig = None
    warnings.warn(
        "Pydantic config models not available. Configuration validation disabled.",
        UserWarning,
    )


def load_yaml(p):
    """Legacy YAML loader - use TrainingConfig.from_yaml() instead."""
    with open(p) as f:
        return yaml.safe_load(f)


def _create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    training_args,
    peft_config=None,
    max_seq_length=None,
    packing=False,
    callbacks=None,
    logger=None,
):
    """
    Create SFTTrainer with TRL version compatibility handling.

    NOTE: This function acts as a compatibility shim across TRL versions.
    The progressive try/except pattern handles API changes, but adds complexity.
    Consider pinning a minimum TRL version (e.g., >=0.25.0) and removing fallbacks
    once all environments are standardized.

    TRL API has changed significantly across versions:
    - TRL 0.25+: processing_class, minimal parameters (no max_seq_length, no dataset_text_field, no packing)
    - TRL 0.12-0.24: processing_class, with dataset_text_field and max_seq_length
    - TRL < 0.12: tokenizer, with dataset_text_field and max_seq_length

    This function tries the newest API first, then falls back to older APIs.

    Args:
        model: The model to train
        tokenizer: The tokenizer/processor
        train_dataset: Training dataset
        training_args: TrainingArguments instance
        peft_config: Optional PEFT config (only for fresh training, not when resuming)
        max_seq_length: Maximum sequence length (for older TRL versions)
        packing: Whether to pack sequences (for older TRL versions)
        callbacks: List of TrainerCallback instances

    Returns:
        SFTTrainer instance
    """
    # Build base kwargs (peft_config only included if starting fresh, not when loading from checkpoint)
    base_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "args": training_args,
    }
    if peft_config is not None:
        base_kwargs["peft_config"] = peft_config
    if callbacks:
        base_kwargs["callbacks"] = callbacks

    # Check if dataset is pre-tokenized (has input_ids column)
    # If so, SFTTrainer will skip tokenization automatically
    # Note: IterableDataset (streaming) doesn't have column_names, handle gracefully
    is_pre_tokenized = False
    if train_dataset is not None:
        if hasattr(train_dataset, "column_names") and train_dataset.column_names is not None:
            is_pre_tokenized = "input_ids" in train_dataset.column_names
        else:
            # IterableDataset (streaming) - assume not pre-tokenized
            if logger:
                logger.info("Streaming dataset detected - assuming non-pre-tokenized")
            is_pre_tokenized = False

    try:
        # TRL 0.25+ API (minimal parameters - many moved to TrainingArguments)
        # For pre-tokenized datasets, don't set dataset_text_field (SFTTrainer auto-detects input_ids)
        if not is_pre_tokenized:
            # Only set dataset_text_field for non-tokenized datasets
            pass  # TRL 0.25+ handles this automatically
        return SFTTrainer(**base_kwargs)
    except TypeError:
        try:
            # TRL 0.12-0.24 API (with dataset_text_field and max_seq_length)
            # Note: Proper indentation is critical here - dataset_text_field must be indented under the if block
            if not is_pre_tokenized:
                base_kwargs["dataset_text_field"] = "text"
            if max_seq_length is not None:
                base_kwargs["max_seq_length"] = max_seq_length
            if packing is not None:
                base_kwargs["packing"] = packing
            return SFTTrainer(**base_kwargs)
        except TypeError:
            # TRL < 0.12 API (tokenizer instead of processing_class)
            # Note: Proper indentation is critical here - dataset_text_field must be indented under the if block
            kwargs_old = base_kwargs.copy()
            kwargs_old["tokenizer"] = kwargs_old.pop("processing_class")
            if not is_pre_tokenized:
                kwargs_old["dataset_text_field"] = "text"
            if max_seq_length is not None:
                kwargs_old["max_seq_length"] = max_seq_length
            if packing is not None:
                kwargs_old["packing"] = packing
            return SFTTrainer(**kwargs_old)


class ModelCardCallback(TrainerCallback):
    """Callback to generate a comprehensive model card README.md after each checkpoint save."""

    def __init__(self, cfg, model_id):
        self.cfg = cfg
        self.model_id = model_id
        self.training_start_time = datetime.now()
        # Only generate model card on final checkpoint or every N checkpoints to reduce I/O overhead
        self.generate_every_n = cfg.get("misc", {}).get("model_card_frequency", 1)  # 1 = every checkpoint, None = only final

    def on_save(self, args, state, control, model=None, **kwargs):
        """Generate model card README.md when checkpoint is saved."""
        # Skip if not time to generate yet (unless it's the final step)
        if self.generate_every_n and self.generate_every_n > 1:
            if state.global_step % (args.save_steps * self.generate_every_n) != 0:
                # Check if this is the final step
                if state.global_step < args.max_steps:
                    return

        checkpoint_dir = args.output_dir
        if state.global_step > 0:
            # Checkpoint directory is the output_dir, not a subdirectory
            # But we need to write to the latest checkpoint if it exists
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
            if not os.path.exists(checkpoint_path):
                # If checkpoint subdirectory doesn't exist, write to output_dir
                checkpoint_path = checkpoint_dir
        else:
            checkpoint_path = checkpoint_dir

        readme_path = os.path.join(checkpoint_path, "README.md")

        # Get training metrics from state
        latest_metrics = {}
        if state.log_history:
            latest_metrics = state.log_history[-1]

        # Extract dataset names
        dataset_config = self.cfg.get("dataset", {})
        dataset_names = dataset_config.get("names", self.cfg.get("dataset_name", []))
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        # Build model card content
        readme_content = self._generate_model_card(
            checkpoint_path=checkpoint_path,
            global_step=state.global_step,
            latest_metrics=latest_metrics,
            dataset_names=dataset_names,
        )

        # Write README.md (use buffered write for better performance)
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8", buffering=8192) as f:
            f.write(readme_content)

        print(f"Generated model card: {readme_path}")

    def _generate_model_card(self, checkpoint_path, global_step, latest_metrics, dataset_names):
        """Generate model card markdown content using comprehensive template."""
        lora_cfg = self.cfg.get("lora", {})
        train_cfg = self.cfg.get("train", {})
        dataset_cfg = self.cfg.get("dataset", {})
        total_steps = train_cfg.get("num_steps", 12000)
        model_name = self.cfg["misc"]["output_dir"].split("/")[-1]

        # Determine phase from model name or config
        phase = "1"
        if "phase2" in model_name.lower() or "phase-2" in model_name.lower():
            phase = "2"
        elif "phase1" in model_name.lower() or "phase-1" in model_name.lower():
            phase = "1"

        # Build comprehensive YAML metadata section
        yaml_metadata = {
            "base_model": self.model_id,
            "library_name": "transformers",
            "license": "other",  # Overall usage governed by Meta's Llama 3.1 Community License
            "adapter_license": "mit",
            "base_model_license": "Llama 3.1 Community License",
            "base_model_license_url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            "tags": [
                "rust",
                "rust-programming",
                "code-generation",
                "qlora",
                "lora",
                "peft",
                "llama",
                "meta-llama-3.1",
                "instruction-tuned",
                "text-generation",
                "sigilderg",
                "lora-adapter",
                "base-required",
            ],
            "datasets": dataset_names,
            "language": ["en"],
            "pipeline_tag": "text-generation",
        }

        # Add evaluation results structure (can be populated later)
        # This follows the model-index specification from Papers with Code
        model_index = {
            "name": f"{model_name}-step-{global_step}",
            "results": [
                {
                    "task": {"type": "text-generation"},
                    "dataset": {
                        "name": "rust-code-evaluation",
                        "type": "code-generation",
                    },
                    "metrics": [
                        {
                            "name": "Compilation Rate",
                            "type": "compilation_rate",
                            "value": None,  # Will be updated when evaluation results are available
                        },
                        {
                            "name": "Clippy Warnings (avg)",
                            "type": "clippy_warnings",
                            "value": None,
                        },
                        {
                            "name": "Idiomatic Score",
                            "type": "idiomatic_score",
                            "value": None,
                        },
                        {
                            "name": "Documentation Rate",
                            "type": "doc_comment_rate",
                            "value": None,
                        },
                        {
                            "name": "Avg Functions",
                            "type": "avg_functions",
                            "value": None,
                        },
                        {"name": "Avg Structs", "type": "avg_structs", "value": None},
                        {"name": "Avg Traits", "type": "avg_traits", "value": None},
                        {"name": "Test Rate", "type": "test_rate", "value": None},
                        {
                            "name": "Prompt Match Score",
                            "type": "prompt_match",
                            "value": None,
                        },
                    ],
                    "source": {
                        "name": "SigilDERG Evaluation",
                        "url": "https://github.com/Superuser666-Sigil/SigilDERG-Finetuner",
                    },
                }
            ],
        }

        yaml_metadata["model-index"] = [model_index]

        # Convert YAML metadata to string
        yaml_str = yaml.dump(yaml_metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Find nearest logged step for metrics display
        log_step = global_step
        if latest_metrics and "log_step" in latest_metrics:
            log_step = latest_metrics.get("log_step", global_step)
        elif latest_metrics and "step" in latest_metrics:
            log_step = latest_metrics.get("step", global_step)

        # Get learning rate (peak or current)
        lr_value = train_cfg.get("lr", "N/A")
        if latest_metrics and "learning_rate" in latest_metrics:
            lr_value = latest_metrics.get("learning_rate", lr_value)
            if isinstance(lr_value, float):
                lr_value = f"{lr_value:.2e}"

        # Determine phase description
        phase_desc = f"Phase {phase}"
        if phase == "1":
            phase_desc = "Phase 1: Broad Rust Training"
        elif phase == "2":
            phase_desc = "Phase 2: High-Quality Sharpening"

        # Calculate effective batch size
        effective_batch = train_cfg.get("micro_batch_size", 1) * train_cfg.get("gradient_accumulation", 1)

        # Build comprehensive markdown content
        md_content = f"""---
{yaml_str}---

# {model_name} (checkpoint {global_step} / {total_steps})

> This card describes **checkpoint {global_step}** of the {phase_desc} Rust QLoRA run.
> For the full training plan, governance details, and final recommended checkpoints, see the **root model card** in the repository.

> **Important:** This repository distributes **LoRA adapter weights only**, **not** the full `{self.model_id}` model.
> To use these adapters, you must separately obtain access to the base model from Meta under the **Llama 3.1 Community License** and comply with Meta's license and acceptable-use policy. The adapters alone are not useful without the base model.

## Model Description

This is a QLoRA fine-tuned **LoRA adapter on top of** `{self.model_id}` specifically trained on Rust code. The model uses 4-bit quantization with LoRA (Low-Rank Adaptation) adapters for efficient training and inference.

The primary modality is **Rust code with English comments and explanations**.

This checkpoint is part of the **SigilDERG** ecosystem and is intended as a building block for Rust-focused evaluation and governance tooling, not as a general-purpose all-domain assistant.

## Training Details

### Training Configuration

- **Base Model**: `{self.model_id}`
- **Checkpoint**: {phase_desc}, step {global_step:,} / {total_steps:,}
- **Effective Batch Size**: {train_cfg.get('micro_batch_size', 'N/A')} × {train_cfg.get('gradient_accumulation', 'N/A')} (effective {effective_batch} tokens-per-step equivalent)
- **Sequence Length**: {self.cfg.get('max_seq_len', 'N/A')}
- **Optimizer**: `{train_cfg.get('optimizer', 'paged_adamw_8bit')}`
- **LR Scheduler**: {train_cfg.get('lr_scheduler_type', 'cosine')}
- **Peak Learning Rate**: ~{lr_value} (around this checkpoint)
- **Warmup Steps**: {train_cfg.get('warmup_steps', 'N/A')}
- **Weight Decay**: {train_cfg.get('weight_decay', 'N/A')}
- **Gradient Checkpointing**: {train_cfg.get('grad_checkpointing', False)}
- **BF16**: {train_cfg.get('bf16', False)}
- **Quantization During Training**: 4-bit QLoRA (NF4) with LoRA adapters

### LoRA Configuration

- **Rank (r)**: {lora_cfg.get('r', 'N/A')}
- **Alpha**: {lora_cfg.get('alpha', 'N/A')}
- **Dropout**: {lora_cfg.get('dropout', 'N/A')}
- **Target Modules**: `{', '.join(lora_cfg.get('target_modules', []))}`

These adapters are intended to be loaded on top of the unmodified base weights.

### Quantization

- **Method**: 4-bit NF4 (BitsAndBytes)
- **Compute Dtype**: {self.cfg.get('bnb_4bit', {}).get('compute_dtype', 'bfloat16')}
- **Double Quantization**: {self.cfg.get('bnb_4bit', {}).get('use_double_quant', False)}

### Datasets

{phase_desc} was trained on:

{chr(10).join(f'- `{ds}`' for ds in dataset_names)}

**Dataset configuration for this phase:**

- **Min Length**: {dataset_cfg.get('min_length', 'N/A')}
- **Max Length**: {dataset_cfg.get('max_length', 'N/A')}
- **Exclude Tests**: {dataset_cfg.get('exclude_tests', 'N/A')}
- **Exclude Examples**: {dataset_cfg.get('exclude_examples', 'N/A')}
- **Exclude Benches**: {dataset_cfg.get('exclude_benches', 'N/A')}
- **Prefer Idiomatic**: {dataset_cfg.get('prefer_idiomatic', False)}
- **Prefer Documented**: {dataset_cfg.get('prefer_documented', False)}

{
        (
            "Phase 1 is a broad-inhale pass over cleaned Rust from The Stack. "
            "Later phases are designed to be more selective and incorporate "
            "explicit evaluation feedback."
            if phase == "1"
            else "Phase 2 focuses on high-quality, compilable, idiomatic code "
            "with stricter filtering."
        )
    }

## Training Metrics (around checkpoint {global_step})

Latest logged training metrics in the vicinity of this checkpoint:

{self._format_metrics_checkpoint(latest_metrics, log_step, global_step)}

> Note: Logging occurs every few steps, so `log_step` reflects the nearest logged step to the checkpoint.

## Evaluation Results

All evaluation here is based on **automatic Rust-focused checks** "
"(compile, `clippy`, idiomatic heuristics, doc comments, prompt adherence) "
"over a small but structured evaluation set.

### Aggregate Metrics (checkpoint {global_step}, evaluation pending)

- **Compilation Rate**: Evaluation pending
- **Average Clippy Warnings**: Evaluation pending
- **Idiomatic Score**: Evaluation pending
- **Documentation Rate**: Evaluation pending
- **Test Rate**: Evaluation pending

### Functionality Coverage (approximate averages)

- **Average Functions**: Evaluation pending
- **Average Structs**: Evaluation pending
- **Average Traits**: Evaluation pending
- **Average Impls**: Evaluation pending

### Evaluation Artifacts

- **Full metrics (JSONL)** – per-sample evaluation:
  - `metrics.jsonl` – compilation success, clippy warnings, idiomatic scores, doc detection, and structural stats
- **Error logs (JSONL)** – compiler and runtime errors:
  - `errors.jsonl` – rustc diagnostics, clippy output, and runtime error messages

(Replace these with your actual Hugging Face links as needed:)

- [Metrics (JSONL)](https://huggingface.co/Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora/blob/main/checkpoint-{global_step}/metrics.jsonl)
- [Error Logs (JSONL)](https://huggingface.co/Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora/blob/main/checkpoint-{global_step}/errors.jsonl)

*Evaluation results will be updated when available.*

## Governance and Intended Use

This checkpoint is part of the **SigilDERG** ecosystem and follows **Rule Zero** principles:

- **Primary Intended Use**
  - Rust code generation (functions, modules, small programs)
  - Rust code explanation, refactoring, and review
  - Tooling experiments for automated code evaluation, scoring, and self-improvement loops

- **Not Intended For**
  - Medical, legal, financial, or other high-stakes decision-making
  - Safety-critical or life-critical systems without extensive human review
  - Domains outside software engineering where the model hasn't been evaluated

Users remain responsible for:

- Reviewing and testing all generated code before use in production.
- Ensuring that their use of the **combined base model + adapters** complies with:
  - Meta's **Llama 3.1 Community License** and acceptable-use policy.
  - Any additional organizational or regulatory requirements.

This work is not affiliated with or endorsed by Meta.

## Usage

### Loading the Model (LoRA adapters on base)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (requires access from Meta under the Llama 3.1 Community License)
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.model_id}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter (this checkpoint)
model = PeftModel.from_pretrained(
    base_model,
    "{checkpoint_path}"  # or your local path
)

tokenizer = AutoTokenizer.from_pretrained("{self.model_id}")
```

### Generation Example

```python
# Format prompt for the instruct model
messages = [
    {{"role": "system", "content": "You are a helpful Rust programming assistant."}},
    {{"role": "user", "content": "Write a function that calculates Fibonacci numbers."}}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

> Note: You must load the Meta base model first and then apply this LoRA checkpoint.
> The base weights are not redistributed in this repository.

## Limitations

This adapter is tuned specifically for Rust code; performance on other programming languages or general natural language tasks may be degraded relative to the base model.

The model inherits any limitations, biases, and failure modes from:

- The base `{self.model_id}` model.
- The training data used for Rust fine-tuning ({', '.join(dataset_names)}).

Evaluation so far is focused on:

- Compilation success.
- Static analysis (clippy).
- Simple idiomatic and documentation heuristics.
- A small prompt suite.

It should not be treated as a fully benchmarked or certified Rust expert.

Generated code should always be reviewed, tested, and security-audited (where relevant) before use.

## Citation

If you use this model or its training pipeline, please cite:

```bibtex
@software{{sigilderg_finetuner,
  title  = {{SigilDERG Rust Code Fine-tuned Model}},
  author = {{Dave Tofflemire (Superuser666-Sigil)}},
  year   = {{2025}},
  url    = {{https://github.com/Superuser666-Sigil/SigilDERG-Finetuner}}
}}
```

You should also follow any citation or attribution requirements specified in the Llama 3.1 Community License when referencing the base model.

## License

This repository combines several components with different licenses:

**Base Model (not included here)**

- `{self.model_id}`
- Licensed under the Llama 3.1 Community License by Meta.
- See: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE

**LoRA Adapter Weights (this checkpoint)**

- The adapter weights in this repository are my original contribution and are provided under the MIT License,
only to the extent compatible with the Llama 3.1 Community License.

- You may not use the combined base model + adapters in ways that violate Meta's license or acceptable-use policy, even though the adapter deltas themselves are MIT.

**Training & Evaluation Code (SigilDERG-Finetuner, configs, scripts)**

- All original code in the SigilDERG ecosystem is released under the MIT License, unless otherwise noted in the specific repository.

**Practical summary:**

To actually run this model, you must:

1. Have legitimate access to `{self.model_id}` under Meta's terms.
2. Load these LoRA adapters on top of that base model.
3. Your use of the combined system (base + adapters) is governed primarily by Meta's Llama 3.1 Community License.

The MIT terms apply to the adapters and the SigilDERG code, but do not override or relax Meta's license.

This project is independent and not affiliated with or endorsed by Meta.
"""
        return md_content

    def _format_metrics_checkpoint(self, metrics, log_step, checkpoint_step):
        """Format training metrics for checkpoint display with step information."""
        if not metrics:
            return "No metrics available yet."

        # Format metrics in the new style
        formatted = []
        metric_order = [
            "loss",
            "grad_norm",
            "learning_rate",
            "entropy",
            "num_tokens",
            "mean_token_accuracy",
            "epoch",
            "log_step",
            "checkpoint_step",
        ]

        # Add log_step and checkpoint_step if not already in metrics
        display_metrics = metrics.copy()
        display_metrics["log_step"] = log_step
        display_metrics["checkpoint_step"] = checkpoint_step

        for key in metric_order:
            if key in display_metrics:
                value = display_metrics[key]
                if isinstance(value, float):
                    if key in ["loss", "grad_norm", "learning_rate", "entropy"]:
                        formatted.append(f"- **{key}**: {value:.6f}")
                    elif key == "mean_token_accuracy":
                        formatted.append(f"- **{key}**: {value:.6f}")
                    elif key == "num_tokens":
                        formatted.append(f"- **{key}**: {value:.0f}")
                    elif key == "epoch":
                        formatted.append(f"- **{key}**: {value:.6f}")
                    else:
                        formatted.append(f"- **{key}**: {value:.6f}")
                elif isinstance(value, int):
                    formatted.append(f"- **{key}**: {value:,}")
                else:
                    formatted.append(f"- **{key}**: {value}")

        # Add any remaining metrics not in the ordered list
        for key, value in display_metrics.items():
            if key not in metric_order:
                if isinstance(value, float):
                    formatted.append(f"- **{key}**: {value:.6f}")
                elif isinstance(value, int):
                    formatted.append(f"- **{key}**: {value:,}")
                else:
                    formatted.append(f"- **{key}**: {value}")

        result = "\n".join(formatted)
        result += "\n\n(Logging is done every few steps, so `log_step` reflects the nearest logged step to the checkpoint.)"
        return result

    def _format_metrics(self, metrics):
        """Format training metrics for display."""
        if not metrics:
            return "No metrics available yet."

        lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.6f}")
                else:
                    lines.append(f"- **{key}**: {value:,}")
            else:
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines) if lines else "No metrics available."


class MemoryOptimizationCallback(TrainerCallback):
    """Callback to optimize GPU memory usage and prevent fragmentation.

    Less aggressive version:

    - Logs memory every `log_every_n_steps`.
    - Only considers aggressive cache clears every `clear_cache_every_n_steps`.
    - Uses a higher fragmentation ratio and a percentage of total GPU memory.
    - If clear_cache_every_n_steps <= 0, aggressive clearing is disabled.
    """

    def __init__(
        self,
        clear_cache_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        min_fragmentation_ratio: float = 1.6,
        hard_limit_fraction: float = 0.90,
    ):
        self.clear_cache_every_n_steps = max(int(clear_cache_every_n_steps), 0)
        self.log_every_n_steps = max(int(log_every_n_steps), 1)
        self.min_fragmentation_ratio = float(min_fragmentation_ratio)
        self.hard_limit_fraction = float(hard_limit_fraction)

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.total_mem_gb = props.total_memory / 1024**3
        else:
            self.total_mem_gb = 0.0

    def _should_check_fragmentation(self, step: int) -> bool:
        if self.clear_cache_every_n_steps <= 0:
            return False  # disabled via config/env
        if step <= 0:
            return False
        return step % self.clear_cache_every_n_steps == 0

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB

        # Periodic logging only
        if state.global_step % self.log_every_n_steps == 0:
            print(
                f"Step {state.global_step}: "
                f"GPU Memory - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB"
            )

        # Throttle fragmentation checks
        if not self._should_check_fragmentation(state.global_step):
            return

        fragmentation_detected = False

        # 1) Reserved much larger than allocated
        if allocated > 0 and reserved > allocated * self.min_fragmentation_ratio:
            fragmentation_detected = True

        # 2) Reserved close to total device memory
        if self.total_mem_gb > 0 and reserved > self.total_mem_gb * self.hard_limit_fraction:
            fragmentation_detected = True

        if fragmentation_detected:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(
                f"Step {state.global_step}: "
                f"Aggressive cache clear (Alloc: {allocated:.2f}GB, "
                f"Res: {reserved:.2f}GB, "
                f"Total: {self.total_mem_gb:.2f}GB)"
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Keep this very light: optional gentle clear every N steps.
        if (
            torch.cuda.is_available()
            and self.clear_cache_every_n_steps > 0
            and state.global_step > 0
            and state.global_step % (self.clear_cache_every_n_steps * 2) == 0
        ):
            torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Tiny safety net: occasional cache clear after logging.
        if (
            torch.cuda.is_available()
            and self.clear_cache_every_n_steps > 0
            and state.global_step > 0
            and state.global_step % (self.clear_cache_every_n_steps * 4) == 0
        ):
            torch.cuda.empty_cache()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/llama8b-phase1.yml")
    ap.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (default: stdout only)",
    )
    args = ap.parse_args()

    # Load config with Pydantic validation if available, fallback to raw YAML
    if TrainingConfig is not None:
        try:
            cfg_obj = TrainingConfig.from_yaml(args.cfg)
            # Convert to dict for backward compatibility with existing code
            # This maintains dict-style access while getting validation benefits
            cfg = cfg_obj.to_dict()
            # Also store the object for type-safe access where needed
            cfg["_pydantic_obj"] = cfg_obj
            print("✓ Configuration validated with Pydantic")
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found: {args.cfg}. Please check the path.")
        except yaml.YAMLError as e:
            raise RuntimeError(
                f"Failed to parse YAML configuration file '{args.cfg}': {e}\n"
                f"Common issues: incorrect indentation, invalid YAML syntax, or missing quotes around strings."
            )
        except Exception as e:
            error_msg = str(e)
            if "validation" in error_msg.lower() or "field" in error_msg.lower():
                raise RuntimeError(
                    f"Configuration validation failed for '{args.cfg}': {e}\n"
                    f"Common issues: wrong field names, invalid types, or missing required fields.\n"
                    f"Check the config file against the expected schema."
                )
            print(f"Warning: Pydantic validation failed: {e}")
            print("Falling back to raw YAML loading (no validation)")
            cfg = load_yaml(args.cfg)
    else:
        cfg = load_yaml(args.cfg)

    # Set up logging to file if requested
    log_file = args.log_file or cfg.get("misc", {}).get("log_file")
    if log_file:
        log_path = os.path.join(cfg["misc"]["output_dir"], log_file) if not os.path.isabs(log_file) else log_file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Use Python's logging module instead of Tee class
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),  # Also log to stdout
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to {log_path}")
    else:
        # Set up basic logging even without file
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed = cfg.get("misc", {}).get("seed", 42)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # H100 optimizations: Enable CuDNN benchmark for better performance
        # (disable deterministic mode for speed - only use deterministic=True for reproducibility)
        use_deterministic = cfg.get("misc", {}).get("deterministic", False)
        torch.backends.cudnn.deterministic = use_deterministic
        torch.backends.cudnn.benchmark = not use_deterministic  # Benchmark mode is faster
        # Enable tensor cores and other H100 optimizations using new API
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        # Enable TF32 for faster matmuls
        torch.backends.cudnn.fp32_precision = "tf32"
        logger.info(f"CuDNN benchmark mode: {not use_deterministic}, " f"TF32 enabled: True")

        # Set CUDA memory management to reduce fragmentation
        # Use new PYTORCH_ALLOC_CONF instead of deprecated PYTORCH_CUDA_ALLOC_CONF
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        logger.info("Set PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 " "to reduce memory fragmentation")

        # Clear any cached memory before training starts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache and synchronized before training")
    logger.info(f"Set random seed to {seed} for reproducibility")

    # Enable Flash Attention 2 for H100 if available (check via env var or config)
    use_flash_attention = os.environ.get("FLASH_ATTENTION") or cfg.get("train", {}).get("use_flash_attention", False)
    if use_flash_attention:
        logger.info("Flash Attention 2 enabled for improved performance")
    # Note: When using flash_attention_2 (attn_implementation="flash_attention_2"),
    # some deterministic settings may be overridden due to kernel optimizations.
    # For fully deterministic training, consider using "sdpa" instead.

    model_id = cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token

    # Check if loading from a checkpoint (for Phase 2)
    load_from = cfg.get("misc", {}).get("load_from")

    # Detect if we're in a multi-GPU accelerate context
    # When using accelerate launch with BitsAndBytes, we need to explicitly set device_map
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_multi_gpu = world_size > 1 or local_rank >= 0

    # For multi-GPU with accelerate + BitsAndBytes, use device_map with explicit device
    # BitsAndBytes requires explicit device mapping for each process
    # Use LOCAL_RANK to determine which GPU this process should use
    if is_multi_gpu:
        # Ensure CUDA is available and set the device explicitly
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but multi-GPU training was requested")
        # Use LOCAL_RANK to set the device (accelerate sets this for each process)
        device_id = local_rank if local_rank >= 0 else 0
        torch.cuda.set_device(device_id)
        device_map_setting = {"": device_id}
        logger.info(f"Multi-GPU mode: Loading model on device {device_id} (LOCAL_RANK={local_rank})")
    else:
        device_map_setting = "auto"
        logger.info("Single-GPU mode: Using device_map='auto'")

    if load_from:
        print(f"Loading model from checkpoint: {load_from}")
        # Load model the same way it was during training: base model + quantization + PEFT adapter
        # This ensures optimizer state compatibility
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit"]["quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit"]["use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map_setting,
            dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation=("flash_attention_2" if use_flash_attention else "sdpa"),
            trust_remote_code=True,
            use_cache=not cfg["train"].get("grad_checkpointing", False),
        )

        # Prepare model for k-bit training
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["train"].get("grad_checkpointing", False),
        )

        # Load PEFT adapter from checkpoint
        model = PeftModel.from_pretrained(model, load_from, is_trainable=True)

        # Ensure model is in training mode and output hidden states
        model.train()
        model.config.use_cache = False
    else:
        # Fresh training from base model
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit"]["quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit"]["use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map_setting,
            dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        )

        # Prepare model for k-bit training (required for PEFT with quantization)
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing on model if configured
        if cfg["train"]["grad_checkpointing"]:
            if hasattr(model, "gradient_checkpointing_enable"):
                # PyTorch 2.9+ requires use_reentrant parameter
                # Pass it via kwargs if the method supports it
                try:
                    model.gradient_checkpointing_enable(use_reentrant=False)
                except TypeError:
                    # Fallback for older versions that don't accept kwargs
                    model.gradient_checkpointing_enable()

    # Only create PEFT config when starting fresh training
    # When loading from checkpoint, the adapter is already loaded
    if load_from:
        peft_cfg = None
    else:
        lora = cfg["lora"]
        # target_modules is now already a list (handled by config validator)
        peft_cfg = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Load dataset via DatasetLoader abstraction
    if _data_filters_module:
        create_filter_function = _data_filters_module.create_filter_function
    else:
        raise ImportError("Could not import data_filters. Install package in editable mode: pip install -e .")

    dataset_loader = DatasetLoader(
        cfg=cfg,
        tokenizer=tok,
        create_filter_function=create_filter_function,
        stream_rust_fn=stream_rust,
        logger=logger,
    )
    ds_iter, dataset_metadata = dataset_loader.load()
    logger.info(
        "Dataset loader summary | mode=%s | pre_tokenized=%s | datasets=%s",
        "streaming" if dataset_metadata.get("is_streaming") else "cached",
        dataset_metadata.get("pre_tokenized"),
        ", ".join(dataset_metadata.get("dataset_names", [])),
    )

    # Determine logging backend
    log_backend = cfg["train"].get("log_backend", "tensorboard")
    if log_backend == "tensorboard":
        report_to = ["tensorboard"]
    elif log_backend == "wandb":
        report_to = ["wandb"]
    else:
        report_to = []

    # Prepare gradient checkpointing kwargs for PyTorch 2.9+ compatibility
    grad_checkpointing_kwargs = None
    if cfg["train"]["grad_checkpointing"]:
        # PyTorch 2.9+ requires use_reentrant to be passed explicitly
        # use_reentrant=False is recommended for better performance and flexibility
        grad_checkpointing_kwargs = {"use_reentrant": False}

    args_tr = TrainingArguments(
        output_dir=cfg["misc"]["output_dir"],
        max_steps=cfg["train"]["num_steps"],
        per_device_train_batch_size=cfg["train"]["micro_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        lr_scheduler_type=cfg["train"].get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg["train"]["warmup_steps"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_every"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["grad_checkpointing"],
        gradient_checkpointing_kwargs=grad_checkpointing_kwargs,  # PyTorch 2.9+ compatibility
        optim=cfg["train"].get("optimizer", "paged_adamw_8bit"),
        report_to=report_to,
        logging_dir=cfg["misc"].get("logging_dir", os.path.join(cfg["misc"]["output_dir"], "logs")),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        save_total_limit=cfg["train"].get("save_total_limit", 3),
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", False),
        dataloader_num_workers=cfg["train"].get("dataloader_num_workers", 2),  # Enable workers for faster data loading
        dataloader_pin_memory=cfg["train"].get("dataloader_pin_memory", True),  # Pin memory for faster CPU-GPU transfers
        dataloader_prefetch_factor=(
            cfg["train"].get("dataloader_prefetch_factor", 2) if cfg["train"].get("dataloader_num_workers", 2) > 1 else None
        ),  # Prefetch batches ahead (only with workers)
        do_train=True,  # Explicitly enable training
        ddp_find_unused_parameters=(False if is_multi_gpu else None),  # LoRA has no unused params - disable for performance
    )

    # Create callbacks
    callbacks = [ModelCardCallback(cfg, model_id)]

    # Add memory optimization callback to prevent GPU memory fragmentation
    clear_cache_frequency = cfg.get("train", {}).get("clear_cache_every_n_steps", 100)
    callbacks.append(MemoryOptimizationCallback(clear_cache_every_n_steps=clear_cache_frequency))

    # Create trainer with TRL version compatibility handling
    trainer = _create_sft_trainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_iter,
        training_args=args_tr,
        peft_config=peft_cfg,
        max_seq_length=cfg["max_seq_len"],
        packing=cfg["pack"],
        callbacks=callbacks,
        logger=logger,
    )

    # Resume from checkpoint if loading from one
    # Handle optimizer state incompatibility by temporarily removing it
    # NOTE: When resuming from a checkpoint, if the optimizer state is incompatible
    # (e.g., due to different parameter groups or optimizer type changes), the script
    # will automatically backup the incompatible optimizer.pt and scheduler.pt files
    # and retry training with a fresh optimizer. The training step number is preserved
    # from trainer_state.json, so training continues from the correct step.
    # This behavior ensures training can resume even after configuration changes.
    # Skip full checkpoint resumption to avoid multi-GPU to single-GPU
    # compatibility issues. Instead, load only adapter weights (handled above)
    # and start fresh training state.
    if load_from:
        logger.info(f"Loaded adapter weights from {load_from}")
        logger.info("Starting fresh training state to avoid multi-GPU " "compatibility issues")
        logger.info("Note: Training step counter will restart from 0, " "but adapter weights are preserved")
        load_from = None  # Don't resume training state

    # Start training with fresh optimizer state
    # (adapter weights already loaded)
    try:
        trainer.train()
    except Exception as e:
        if "shift_logits" in str(e) and "logits" in str(e):
            error_msg = (
                "Error: Model output format doesn't match SFT trainer's "
                "expectations. This usually happens when the model's forward "
                f"method doesn't return logits in the expected format. "
                f"Full error: {str(e)}"
            )
            logger.error(error_msg)

            # Try to get more debug info
            logger.info("Model output keys:")
            with torch.no_grad():
                train_loader = trainer.get_train_dataloader()
                sample_input = next(iter(train_loader))
                sample_input = {k: v.to(model.device) for k, v in sample_input.items() if k not in ["labels", "input_ids"]}
                try:
                    outputs = model(**sample_input, output_hidden_states=True)
                    logger.info(f"Output type: {type(outputs)}")
                    if hasattr(outputs, "logits"):
                        logger.info(f"Logits shape: {outputs.logits.shape if hasattr(outputs.logits, 'shape') else 'N/A'}")
                    else:
                        logger.info("No logits in output")
                        logger.info(f"Output attributes: {[a for a in dir(outputs) if not a.startswith('_')]}")
                except Exception as debug_e:
                    logger.error(f"Error during debug output: {str(debug_e)}")
        raise


if __name__ == "__main__":
    main()
