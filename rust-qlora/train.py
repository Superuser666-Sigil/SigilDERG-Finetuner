import os, yaml, torch
import warnings
from datetime import datetime
from datasets import IterableDataset

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
try:
    from .data_filters import stream_rust
except ImportError:
    # Allow running as script
    from data_filters import stream_rust

def load_yaml(p):
    """Legacy YAML loader - use TrainingConfig.from_yaml() instead."""
    with open(p) as f: return yaml.safe_load(f)

# Import Pydantic config models
try:
    from .config_models import TrainingConfig
except ImportError:
    try:
        from config_models import TrainingConfig
    except ImportError:
        # Fallback: Pydantic not available
        TrainingConfig = None


class ModelCardCallback(TrainerCallback):
    """Callback to generate a comprehensive model card README.md after each checkpoint save."""
    
    def __init__(self, cfg, model_id):
        self.cfg = cfg
        self.model_id = model_id
        self.training_start_time = datetime.now()
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Generate model card README.md when checkpoint is saved."""
        checkpoint_dir = args.output_dir
        if state.global_step > 0:
            # Checkpoint directory is the output_dir, not a subdirectory
            # But we need to write to the latest checkpoint if it exists
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{state.global_step}")
            if not os.path.exists(checkpoint_path):
                # If checkpoint subdirectory doesn't exist, write to output_dir
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
        
        # Write README.md
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
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

def main():
    import argparse
    import sys
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
        except Exception as e:
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
        # Create a tee-like wrapper that writes to both stdout and file
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        log_f = open(log_path, "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_f)
        sys.stderr = Tee(sys.stderr, log_f)
        print(f"Logging to {log_path}")

    # Set seed for reproducibility
    seed = cfg.get("misc", {}).get("seed", 42)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic CuDNN operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed} for reproducibility")

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
            attn_implementation="flash_attention_2" if os.environ.get("FLASH_ATTENTION") else "sdpa"
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
            attn_implementation="flash_attention_2" if os.environ.get("FLASH_ATTENTION") else "sdpa"
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
        targets = sum([t.split(";") for t in lora["target_modules"]], [])
        peft_cfg = LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=[t.strip() for t in targets if t.strip()],
            bias="none", task_type="CAUSAL_LM"
        )

    # Stream dataset with enhanced filtering
    dataset_config = cfg.get("dataset", {})
    dataset_names = dataset_config.get("names", cfg.get("dataset_name", "ammarnasr/the-stack-rust-clean"))
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    ds_iter = IterableDataset.from_generator(
        lambda: stream_rust(
            dataset_names=dataset_names,
            cache_dir=dataset_config.get("cache_dir"),
            use_cache=dataset_config.get("use_cache", True),
            min_length=dataset_config.get("min_length", 64),
            max_length=dataset_config.get("max_length", 200_000),
            exclude_tests=dataset_config.get("exclude_tests", True),
            exclude_examples=dataset_config.get("exclude_examples", False),
            exclude_benches=dataset_config.get("exclude_benches", True),
            prefer_idiomatic=dataset_config.get("prefer_idiomatic", False),
            prefer_documented=dataset_config.get("prefer_documented", False),
            shuffle_seed=dataset_config.get("shuffle_seed"),
            interleave_mode=dataset_config.get("interleave_mode", "sequential"),
            dataset_weights=dataset_config.get("dataset_weights"),
        )
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
        dataloader_num_workers=0,  # Disable multiprocessing to avoid dataset loading hangs
        do_train=True,  # Explicitly enable training
    )

    # TRL API has changed significantly across versions:
    # - TRL 0.25+: processing_class, minimal parameters (no max_seq_length, no dataset_text_field, no packing)
    # - TRL 0.12-0.24: processing_class, with dataset_text_field and max_seq_length
    # - TRL < 0.12: tokenizer, with dataset_text_field and max_seq_length
    # Try newest API first (minimal), then fall back to older APIs
    # Build base kwargs (peft_config only included if starting fresh, not when loading from checkpoint)
    base_kwargs = {
        "model": model,
        "processing_class": tok,
        "train_dataset": ds_iter,
        "args": args_tr
    }
    if peft_cfg is not None:
        base_kwargs["peft_config"] = peft_cfg
    
    # Create model card callback
    model_card_callback = ModelCardCallback(cfg, model_id)
    
    try:
        # TRL 0.25+ API (minimal parameters - many moved to TrainingArguments or removed)
        trainer = SFTTrainer(**base_kwargs, callbacks=[model_card_callback])
    except TypeError as e:
        try:
            # TRL 0.12-0.24 API (with dataset_text_field and max_seq_length)
            trainer = SFTTrainer(
                **base_kwargs,
                dataset_text_field="text",
                max_seq_length=cfg["max_seq_len"],
                packing=cfg["pack"],
                callbacks=[model_card_callback]
            )
        except TypeError:
            # TRL < 0.12 API (tokenizer instead of processing_class)
            # Remove processing_class and use tokenizer instead
            kwargs_old = base_kwargs.copy()
            kwargs_old["tokenizer"] = kwargs_old.pop("processing_class")
            trainer = SFTTrainer(
                **kwargs_old,
                dataset_text_field="text",
                max_seq_length=cfg["max_seq_len"],
                packing=cfg["pack"],
                callbacks=[model_card_callback]
            )

    # Resume from checkpoint if loading from one
    # Handle optimizer state incompatibility by temporarily removing it
    if load_from:
        import json
        import shutil
        trainer_state_path = os.path.join(load_from, "trainer_state.json")
        optimizer_path = os.path.join(load_from, "optimizer.pt")
        scheduler_path = os.path.join(load_from, "scheduler.pt")
        
        # Read the step number from trainer state
        global_step = 0
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            global_step = trainer_state.get("global_step", 0)
        
        # Try to resume normally first
        try:
            trainer.train(resume_from_checkpoint=load_from)
        except ValueError as e:
            if "parameter group" in str(e) or "optimizer" in str(e).lower():
                # Optimizer state incompatible - backup and remove optimizer files, then retry
                print(f"Warning: Optimizer state incompatible. Removing optimizer/scheduler state and resuming from step {global_step}.")
                # Backup the incompatible files
                if os.path.exists(optimizer_path):
                    shutil.move(optimizer_path, optimizer_path + ".backup")
                if os.path.exists(scheduler_path):
                    shutil.move(scheduler_path, scheduler_path + ".backup")
                # Retry without optimizer/scheduler (will use fresh optimizer but correct step)
                # Note: Backups are kept for investigation; incompatible optimizer/scheduler won't be restored
                trainer.train(resume_from_checkpoint=load_from)
            else:
                raise
    else:
        trainer.train()

if __name__ == "__main__":
    main()
