#!/usr/bin/env python3
"""
Export merged model from LoRA checkpoint.

This script loads a PEFT LoRA checkpoint, merges the adapters into the base model,
and saves the merged model for deployment.
"""

import os
import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoConfig


def main():
    parser = argparse.ArgumentParser(description="Export merged model from LoRA checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out/llama8b-rust-qlora",
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/merged",
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model identifier (for tokenizer/config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to load model on"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint directory does not exist: {args.checkpoint}")
    
    # Check disk space before proceeding (rough estimate: model size ~2-4GB for 8B model)
    try:
        import shutil
        stat = shutil.disk_usage(args.output if os.path.isabs(args.output) else os.path.dirname(args.output) or ".")
        free_gb = stat.free / (1024**3)
        if free_gb < 5:
            print(f"Warning: Low disk space ({free_gb:.1f} GB free). Model export may require ~5-10 GB.")
    except Exception:
        pass  # Skip disk space check if it fails
    
    print(f"Loading LoRA checkpoint from {args.checkpoint}...")
    try:
        m = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map=args.device if args.device != "cpu" else None,
            dtype="auto" if args.device != "cpu" else None
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {e}\n"
            f"Check that the checkpoint directory exists and contains adapter files."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Validate that base model matches training base
    try:
        import json
        adapter_config_path = os.path.join(args.checkpoint, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            checkpoint_base_model = adapter_config.get("base_model_name_or_path", "")
            if checkpoint_base_model and checkpoint_base_model != args.base_model:
                print(f"Warning: Checkpoint was trained with base model '{checkpoint_base_model}', "
                      f"but you specified '{args.base_model}'. Merging with a different base model "
                      f"may produce incorrect weights.")
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return 1
    except Exception as e:
        print(f"Warning: Could not validate base model match: {e}")
    
    print("Merging adapters into base model...")
    try:
        m = m.merge_and_unload()
    except Exception as e:
        raise RuntimeError(f"Failed to merge adapters: {e}")
    
    print(f"Loading tokenizer from {args.base_model}...")
    try:
        t = AutoTokenizer.from_pretrained(args.base_model)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load tokenizer: {e}\n"
            f"Check that '{args.base_model}' is a valid model identifier or path."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # Validate model shape
    config = m.config
    print(f"Model config: {config.model_type}, vocab_size={config.vocab_size}, "
          f"hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    print(f"Saving merged model to {args.output}...")
    try:
    os.makedirs(args.output, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create output directory '{args.output}': {e}. Check permissions.")
    
    try:
        m.save_pretrained(args.output)
        t.save_pretrained(args.output)
    except OSError as e:
        if "No space left on device" in str(e) or "disk" in str(e).lower():
            raise RuntimeError(
                f"Failed to save model: {e}\n"
                f"Insufficient disk space. Free up space and try again."
            )
        raise RuntimeError(
            f"Failed to save model: {e}\n"
            f"Check disk space and write permissions for '{args.output}'."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")
    
    print(f"Successfully saved merged model to {args.output}")


if __name__ == "__main__":
    main()
