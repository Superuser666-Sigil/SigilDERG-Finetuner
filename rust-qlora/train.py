import os
import yaml
import torch
import warnings
import logging
import importlib
from datetime import datetime
from pathlib import Path
from datasets import IterableDataset, Dataset, load_dataset

# Note: PyTorch 2.9+ requires use_reentrant parameter in torch.utils.checkpoint.checkpoint
# We now explicitly pass use_reentrant=False via gradient_checkpointing_kwargs in TrainingArguments
# and in model.gradient_checkpointing_enable() calls. The warning filters below are kept as a
# fallback for any deep library calls that may not yet support the parameter.
warnings.filterwarnings("ignore", message=".*use_reentrant.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*", category=UserWarning)
# Suppress PyTorch deprecation warning for torch.cpu.amp.autocast (will be fixed in future PyTorch versions)
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*is deprecated.*", category=FutureWarning)

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    set_seed, TrainerCallback
)
from trl import SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel


def _resolve_import(module_name, fallback_module_name=None):
    """
    Helper function to resolve imports with fallback support.
    
    Args:
        module_name: Primary module name (e.g., '.data_filters' or 'rust_qlora.data_filters')
        fallback_module_name: Fallback module name (e.g., 'rust_qlora.data_filters')
    
    Returns:
        Imported module or None if both imports fail
    """
    is_relative = module_name.startswith('.')
    package_name = __package__
    if not package_name and '.' in __name__:
        package_name = __name__.rsplit('.', 1)[0]
    
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
_data_filters_module = _resolve_import('.data_filters', 'rust_qlora.data_filters')
if _data_filters_module:
    stream_rust = _data_filters_module.stream_rust
else:
    raise ImportError("Could not import data_filters. Install package in editable mode: pip install -e .")

# Import Pydantic config models
_config_models_module = _resolve_import('.config_models', 'rust_qlora.config_models')
if _config_models_module:
    TrainingConfig = _config_models_module.TrainingConfig
else:
    TrainingConfig = None
    warnings.warn("Pydantic config models not available. Configuration validation disabled.", UserWarning)


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
    packing=None,
    callbacks=None
):
    """
    Create SFTTrainer with TRL version compatibility handling.
    
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
        "args": training_args
    }
    if peft_config is not None:
        base_kwargs["peft_config"] = peft_config
    if callbacks:
        base_kwargs["callbacks"] = callbacks
    
    # Check if dataset is pre-tokenized (has input_ids column)
    # If so, SFTTrainer will skip tokenization automatically
    is_pre_tokenized = hasattr(train_dataset, 'column_names') and "input_ids" in train_dataset.column_names
    
    try:
        # TRL 0.25+ API (minimal parameters - many moved to TrainingArguments or removed)
        # For pre-tokenized datasets, don't set dataset_text_field (SFTTrainer auto-detects input_ids)
        if not is_pre_tokenized:
            # Only set dataset_text_field for non-tokenized datasets
            pass  # TRL 0.25+ handles this automatically
        return SFTTrainer(**base_kwargs)
    except TypeError:
        try:
            # TRL 0.12-0.24 API (with dataset_text_field and max_seq_length)
            if not is_pre_tokenized:
                base_kwargs["dataset_text_field"] = "text"
            if max_seq_length is not None:
                base_kwargs["max_seq_length"] = max_seq_length
            if packing is not None:
                base_kwargs["packing"] = packing
            return SFTTrainer(**base_kwargs)
        except TypeError:
            # TRL < 0.12 API (tokenizer instead of processing_class)
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
            dataset_names=dataset_names
        )
        
        # Write README.md (use buffered write for better performance)
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8", buffering=8192) as f:
            f.write(readme_content)
        
        print(f"Generated model card: {readme_path}")
    
    def _generate_model_card(self, checkpoint_path, global_step, latest_metrics, dataset_names):
        """Generate model card markdown content."""
        lora_cfg = self.cfg.get("lora", {})
        train_cfg = self.cfg.get("train", {})
        dataset_cfg = self.cfg.get("dataset", {})
        
        # Build YAML metadata section
        yaml_metadata = {
            "base_model": self.model_id,
            "library_name": "transformers",
            "tags": [
                "rust",
                "code-generation",
                "qlora",
                "lora",
                "peft",
                "llama",
                "text-generation",
                "sigilderg"
            ],
            "datasets": dataset_names,
            "license": "mit",
            "language": ["en"],
            "pipeline_tag": "text-generation"
        }
        
        # Add evaluation results structure (can be populated later)
        # This follows the model-index specification from Papers with Code
        model_index = {
            "name": f"{self.cfg['misc']['output_dir'].split('/')[-1]}",
            "results": [
                {
                    "task": {
                        "type": "text-generation"
                    },
                    "dataset": {
                        "name": "rust-code-evaluation",
                        "type": "code-generation"
                    },
                    "metrics": [
                        {
                            "name": "Compilation Rate",
                            "type": "compilation_rate",
                            "value": None  # Will be updated when evaluation results are available
                        },
                        {
                            "name": "Clippy Warnings (avg)",
                            "type": "clippy_warnings",
                            "value": None
                        },
                        {
                            "name": "Idiomatic Score",
                            "type": "idiomatic_score",
                            "value": None
                        }
                    ]
                }
            ]
        }
        
        yaml_metadata["model-index"] = [model_index]
        
        # Convert YAML metadata to string
        yaml_str = yaml.dump(yaml_metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Build markdown content
        md_content = f"""---
{yaml_str}---

# {self.cfg['misc']['output_dir'].split('/')[-1]}

## Model Description

This is a QLoRA fine-tuned version of **{self.model_id}** specifically trained on Rust code. The model uses 4-bit quantization with LoRA (Low-Rank Adaptation) adapters for efficient training and inference.

## Training Details

### Training Configuration

- **Base Model**: `{self.model_id}`
- **Training Steps**: {global_step:,} / {train_cfg.get('num_steps', 'N/A'):,}
- **Learning Rate**: {train_cfg.get('lr', 'N/A')}
- **Batch Size**: {train_cfg.get('micro_batch_size', 'N/A')} × {train_cfg.get('gradient_accumulation', 'N/A')} (effective: {train_cfg.get('micro_batch_size', 1) * train_cfg.get('gradient_accumulation', 1)})
- **Sequence Length**: {self.cfg.get('max_seq_len', 'N/A')}
- **Optimizer**: {train_cfg.get('optimizer', 'paged_adamw_8bit')}
- **LR Scheduler**: {train_cfg.get('lr_scheduler_type', 'cosine')}
- **Warmup Steps**: {train_cfg.get('warmup_steps', 'N/A')}
- **Weight Decay**: {train_cfg.get('weight_decay', 'N/A')}
- **Gradient Checkpointing**: {train_cfg.get('grad_checkpointing', False)}
- **BF16**: {train_cfg.get('bf16', False)}

### LoRA Configuration

- **Rank (r)**: {lora_cfg.get('r', 'N/A')}
- **Alpha**: {lora_cfg.get('alpha', 'N/A')}
- **Dropout**: {lora_cfg.get('dropout', 'N/A')}
- **Target Modules**: {', '.join(lora_cfg.get('target_modules', []))}

### Quantization

- **Method**: 4-bit NF4 (BitsAndBytes)
- **Compute Dtype**: {self.cfg.get('bnb_4bit', {}).get('compute_dtype', 'bfloat16')}
- **Double Quantization**: {self.cfg.get('bnb_4bit', {}).get('use_double_quant', False)}

### Datasets

The model was trained on the following datasets:

{chr(10).join(f'- `{ds}`' for ds in dataset_names)}

**Dataset Configuration:**
- **Min Length**: {dataset_cfg.get('min_length', 'N/A')}
- **Max Length**: {dataset_cfg.get('max_length', 'N/A')}
- **Exclude Tests**: {dataset_cfg.get('exclude_tests', 'N/A')}
- **Exclude Examples**: {dataset_cfg.get('exclude_examples', 'N/A')}
- **Exclude Benches**: {dataset_cfg.get('exclude_benches', 'N/A')}
- **Prefer Idiomatic**: {dataset_cfg.get('prefer_idiomatic', False)}
- **Prefer Documented**: {dataset_cfg.get('prefer_documented', False)}

## Training Metrics

Latest training metrics (step {global_step:,}):

{self._format_metrics(latest_metrics)}

## Evaluation Results

Evaluation results will be populated here as they become available. The model is evaluated on:

- **Compilation Rate**: Percentage of generated Rust code that compiles successfully
- **Clippy Warnings**: Average number of clippy warnings per sample
- **Idiomatic Score**: Measure of idiomatic Rust patterns in generated code
- **Documentation Rate**: Percentage of code with documentation comments
- **Functionality Coverage**: Analysis of functions, structs, traits, and impls

## Usage

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{self.model_id}",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{checkpoint_path}")
tokenizer = AutoTokenizer.from_pretrained("{self.model_id}")
```

### Generation

```python
# Format prompt for instruct model
messages = [
    {{"role": "system", "content": "You are a helpful Rust programming assistant."}},
    {{"role": "user", "content": "Write a function that calculates fibonacci numbers"}}
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Limitations

- This model is fine-tuned specifically for Rust code generation and may not perform well on other programming languages or general text tasks.
- The model inherits any limitations and biases from the base model.
- Generated code should always be reviewed and tested before use in production.

## Citation

If you use this model, please cite:

```bibtex
@software{{sigilderg_finetuner,
  title = {{SigilDERG Rust Code Fine-tuned Model}},
  author = {{Superuser666-Sigil/Dave Tofflemire}},
  year = {{2025}},
  url = {{https://github.com/Superuser666-Sigil/SigilDERG-Finetuner}}
}}
```

## License

This model is released under the MIT License.
"""
        return md_content
    
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
    """Callback to optimize GPU memory usage and prevent fragmentation."""
    
    def __init__(self, clear_cache_every_n_steps=100):
        self.clear_cache_every_n_steps = clear_cache_every_n_steps
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Clear GPU cache periodically to prevent memory fragmentation."""
        if torch.cuda.is_available() and state.global_step % self.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Clear GPU cache after logging to free up memory (less aggressive for H100)."""
        # Only clear cache periodically, not on every log (H100 has plenty of memory)
        if torch.cuda.is_available() and state.global_step % self.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/llama8b-phase1.yml")
    ap.add_argument("--log-file", type=str, default=None, 
                   help="Optional log file path (default: stdout only)")
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
            print(f"✓ Configuration validated with Pydantic")
        except FileNotFoundError as e:
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
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()  # Also log to stdout
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to {log_path}")
    else:
        # Set up basic logging even without file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
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
        # Enable tensor cores and other H100 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmuls on H100
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"CuDNN benchmark mode: {not use_deterministic}, TF32 enabled: True")
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
    if load_from:
        print(f"Loading model from checkpoint: {load_from}")
        # Load model the same way it was during training: base model + quantization + PEFT adapter
        # This ensures optimizer state compatibility
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit"]["quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit"]["use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation="flash_attention_2" if use_flash_attention else "sdpa"
        )
        # Prepare model for k-bit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        # Load PEFT adapter from checkpoint (this attaches the adapter, doesn't merge it)
        model = PeftModel.from_pretrained(model, load_from)
        # Enable gradient checkpointing if configured
        if cfg["train"]["grad_checkpointing"]:
            if hasattr(model, "gradient_checkpointing_enable"):
                # PyTorch 2.9+ requires use_reentrant parameter
                # Pass it via kwargs if the method supports it
                try:
                    model.gradient_checkpointing_enable(use_reentrant=False)
                except TypeError:
                    # Fallback for older versions that don't accept kwargs
                    model.gradient_checkpointing_enable()
    else:
        # Fresh training from base model
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit"]["quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit"]["use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation="flash_attention_2" if use_flash_attention else "sdpa"
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
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none", task_type="CAUSAL_LM"
        )

    # Load dataset with enhanced filtering
    dataset_config = cfg.get("dataset", {})
    dataset_names = dataset_config.get("names", cfg.get("dataset_name", "ammarnasr/the-stack-rust-clean"))
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    use_cache = dataset_config.get("use_cache", True)
    cache_config = {"cache_dir": dataset_config.get("cache_dir")} if dataset_config.get("cache_dir") else {}
    
    # Import filter functions from data_filters (use existing module resolution)
    if _data_filters_module:
        create_filter_function = _data_filters_module.create_filter_function
    else:
        raise ImportError("Could not import data_filters. Install package in editable mode: pip install -e .")
    
    # When caching is enabled, use HuggingFace's Dataset.filter() for multi-worker support
    # When streaming, use IterableDataset but with 0 workers (streaming doesn't work well with workers)
    if use_cache:
        # Load dataset directly and filter using HuggingFace's optimized filter (supports multiple workers)
        logger.info("Loading dataset in cached mode - using Dataset.filter() for multi-worker efficiency")
        
        if len(dataset_names) == 1:
            # Single dataset - load and filter directly
            dataset_name = dataset_names[0]
            logger.info(f"Loading dataset: {dataset_name}")
            ds = load_dataset(
                dataset_name,
                split="train",
                streaming=False,
                **cache_config
            )
            
            # Create filter function
            filter_fn = create_filter_function(
                min_length=dataset_config.get("min_length", 64),
                max_length=dataset_config.get("max_length", 200_000),
                exclude_tests=dataset_config.get("exclude_tests", True),
                exclude_examples=dataset_config.get("exclude_examples", False),
                exclude_benches=dataset_config.get("exclude_benches", True),
                prefer_idiomatic=dataset_config.get("prefer_idiomatic", False),
                prefer_documented=dataset_config.get("prefer_documented", False),
                idiomatic_quality_ratio=dataset_config.get("idiomatic_quality_ratio", 2.0),
            )
            
            # Filter dataset (this creates a proper multi-shard dataset)
            logger.info("Filtering dataset...")
            ds_filtered = ds.filter(filter_fn, desc=f"Filtering {dataset_name}")
            
            # Pre-tokenize the dataset using parallel processing (much faster than letting SFTTrainer do it)
            # This uses all available CPU cores to tokenize in parallel
            logger.info("Pre-tokenizing dataset with parallel processing...")
            num_proc = min(25, os.cpu_count() or 1)  # Use up to 25 workers (H100 has 25 vCPUs)
            
            def tokenize_function(examples):
                """Tokenize text in batches - processes multiple examples at once for efficiency"""
                # Use the tokenizer that was loaded earlier (tok is defined above)
                return tok(
                    examples["content"],
                    truncation=True,
                    max_length=cfg["max_seq_len"],
                    padding=False,  # Don't pad here - packing will handle it if enabled
                    return_overflowing_tokens=False,
                )
            
            # Tokenize in parallel with batching (much faster than sequential tokenization)
            ds_tokenized = ds_filtered.map(
                tokenize_function,
                batched=True,
                batch_size=1000,  # Process 1000 examples at a time per worker
                remove_columns=[col for col in ds_filtered.column_names if col != "content"],
                num_proc=num_proc,
                desc="Tokenizing dataset (parallel)"
            )
            
            # Map to "text" format - keep input_ids and attention_mask, remove original content
            # SFTTrainer will detect input_ids and skip tokenization
            columns_to_remove = [col for col in ds_tokenized.column_names if col not in ["input_ids", "attention_mask"]]
            ds_iter = ds_tokenized.remove_columns(columns_to_remove) if columns_to_remove else ds_tokenized
            
            # Shuffle if requested
            shuffle_seed = dataset_config.get("shuffle_seed")
            if shuffle_seed is not None:
                logger.info(f"Shuffling dataset with seed {shuffle_seed}")
                ds_iter = ds_iter.shuffle(seed=shuffle_seed)
        else:
            # Multiple datasets - use stream_rust for interleaving logic
            logger.info("Multiple datasets detected - using stream_rust for interleaving")
            # For multiple datasets, fall back to generator approach but convert to Dataset
            all_items = list(stream_rust(
                dataset_names=dataset_names,
                cache_dir=dataset_config.get("cache_dir"),
                use_cache=True,
                min_length=dataset_config.get("min_length", 64),
                max_length=dataset_config.get("max_length", 200_000),
                exclude_tests=dataset_config.get("exclude_tests", True),
                exclude_examples=dataset_config.get("exclude_examples", False),
                exclude_benches=dataset_config.get("exclude_benches", True),
                prefer_idiomatic=dataset_config.get("prefer_idiomatic", False),
                prefer_documented=dataset_config.get("prefer_documented", False),
                idiomatic_quality_ratio=dataset_config.get("idiomatic_quality_ratio", 2.0),
                shuffle_seed=dataset_config.get("shuffle_seed"),
                interleave_mode=dataset_config.get("interleave_mode", "sequential"),
                dataset_weights=dataset_config.get("dataset_weights"),
            ))
            logger.info(f"Loaded {len(all_items)} filtered samples")
            ds_iter = Dataset.from_list(all_items)
            # Note: Dataset.from_list creates single-shard, so reduce workers
            if cfg["train"].get("dataloader_num_workers", 12) > 1:
                logger.warning("Multiple datasets with interleaving - reducing workers to 1 (single-shard dataset)")
                cfg["train"]["dataloader_num_workers"] = 1
    else:
        # Streaming mode: use IterableDataset with 0 workers (workers don't work well with streaming)
        logger.info("Loading dataset in streaming mode - using IterableDataset with 0 workers")
        ds_iter = IterableDataset.from_generator(
            lambda: stream_rust(
                dataset_names=dataset_names,
                cache_dir=dataset_config.get("cache_dir"),
                use_cache=False,
                min_length=dataset_config.get("min_length", 64),
                max_length=dataset_config.get("max_length", 200_000),
                exclude_tests=dataset_config.get("exclude_tests", True),
                exclude_examples=dataset_config.get("exclude_examples", False),
                exclude_benches=dataset_config.get("exclude_benches", True),
                prefer_idiomatic=dataset_config.get("prefer_idiomatic", False),
                prefer_documented=dataset_config.get("prefer_documented", False),
                idiomatic_quality_ratio=dataset_config.get("idiomatic_quality_ratio", 2.0),
                shuffle_seed=dataset_config.get("shuffle_seed"),
                interleave_mode=dataset_config.get("interleave_mode", "sequential"),
                dataset_weights=dataset_config.get("dataset_weights"),
            )
        )
        # Force 0 workers for streaming mode
        if cfg["train"].get("dataloader_num_workers", 2) > 0:
            logger.warning("Streaming mode detected - setting dataloader_num_workers to 0 (workers don't work well with streaming)")
            cfg["train"]["dataloader_num_workers"] = 0

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
        dataloader_prefetch_factor=cfg["train"].get("dataloader_prefetch_factor", 2),  # Prefetch batches ahead
        do_train=True,  # Explicitly enable training
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
        callbacks=callbacks
    )

    # Resume from checkpoint if loading from one
    # Handle optimizer state incompatibility by temporarily removing it
    # NOTE: When resuming from a checkpoint, if the optimizer state is incompatible
    # (e.g., due to different parameter groups or optimizer type changes), the script
    # will automatically backup the incompatible optimizer.pt and scheduler.pt files
    # and retry training with a fresh optimizer. The training step number is preserved
    # from trainer_state.json, so training continues from the correct step.
    # This behavior ensures training can resume even after configuration changes.
    if load_from:
        import json
        import shutil
        trainer_state_path = os.path.join(load_from, "trainer_state.json")
        optimizer_path = os.path.join(load_from, "optimizer.pt")
        scheduler_path = os.path.join(load_from, "scheduler.pt")
        
        # Read the step number from trainer state
        global_step = 0
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)
                global_step = trainer_state.get("global_step", 0)
                logger.info(f"Resuming from checkpoint at step {global_step}")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Could not read trainer_state.json: {e}. Starting from step 0.")
        
        # Try to resume normally first
        try:
            trainer.train(resume_from_checkpoint=load_from)
        except ValueError as e:
            if "parameter group" in str(e) or "optimizer" in str(e).lower():
                # Optimizer state incompatible - backup and remove optimizer files, then retry
                logger.warning(
                    f"Optimizer state incompatible (likely due to config changes). "
                    f"Backing up optimizer/scheduler and resuming from step {global_step} with fresh optimizer."
                )
                # Backup the incompatible files
                if os.path.exists(optimizer_path):
                    shutil.move(optimizer_path, optimizer_path + ".backup")
                    logger.info(f"Backed up optimizer to {optimizer_path}.backup")
                if os.path.exists(scheduler_path):
                    shutil.move(scheduler_path, scheduler_path + ".backup")
                    logger.info(f"Backed up scheduler to {scheduler_path}.backup")
                # Also backup training_args.bin to force use of current config's max_steps
                # This ensures the scheduler uses the current num_steps instead of the checkpoint's old value
                training_args_path = os.path.join(load_from, "training_args.bin")
                if os.path.exists(training_args_path):
                    shutil.move(training_args_path, training_args_path + ".backup")
                    logger.info(f"Backed up training_args to {training_args_path}.backup")
                    logger.info("Using current config's max_steps for scheduler initialization")
                # Retry without optimizer/scheduler/training_args (will use fresh optimizer and current config)
                # Note: Backups are kept for investigation; incompatible files won't be restored
                trainer.train(resume_from_checkpoint=load_from)
            else:
                raise
        except OSError as e:
            raise RuntimeError(f"Failed to load checkpoint from {load_from}: {e}. Check file permissions and disk space.")
    else:
        trainer.train()

if __name__ == "__main__":
    main()
