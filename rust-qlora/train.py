import os, yaml, torch
from datasets import IterableDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    set_seed
)
from trl import SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM
try:
    from .data_filters import stream_rust
except ImportError:
    # Allow running as script
    from data_filters import stream_rust

def load_yaml(p):
    with open(p) as f: return yaml.safe_load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/llama8b.yml")
    args = ap.parse_args()
    cfg = load_yaml(args.cfg)

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
        # Load PEFT adapter from checkpoint
        model = AutoPeftModelForCausalLM.from_pretrained(
            load_from,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if os.environ.get("FLASH_ATTENTION") else "sdpa"
        )
        # Note: When loading from checkpoint, quantization is already applied
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
            torch_dtype=torch.bfloat16,
            quantization_config=bnb,
            attn_implementation="flash_attention_2" if os.environ.get("FLASH_ATTENTION") else "sdpa"
        )
        
        # Prepare model for k-bit training (required for PEFT with quantization)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

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
        optim=cfg["train"].get("optimizer", "paged_adamw_8bit"),
        report_to=report_to,
        logging_dir=cfg["misc"].get("logging_dir", os.path.join(cfg["misc"]["output_dir"], "logs")),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        save_total_limit=cfg["train"].get("save_total_limit", 3),
        load_best_model_at_end=cfg["train"].get("load_best_model_at_end", False),
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=ds_iter,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
        packing=cfg["pack"],
        peft_config=peft_cfg,
        args=args_tr
    )

    trainer.train()

if __name__ == "__main__":
    main()
