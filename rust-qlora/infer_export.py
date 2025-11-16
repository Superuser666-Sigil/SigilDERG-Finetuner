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
    
    print(f"Loading LoRA checkpoint from {args.checkpoint}...")
    try:
        m = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map=args.device if args.device != "cpu" else None,
            torch_dtype="auto" if args.device != "cpu" else None
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    print("Merging adapters into base model...")
    try:
        m = m.merge_and_unload()
    except Exception as e:
        raise RuntimeError(f"Failed to merge adapters: {e}")
    
    print(f"Loading tokenizer from {args.base_model}...")
    try:
        t = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # Validate model shape
    config = m.config
    print(f"Model config: {config.model_type}, vocab_size={config.vocab_size}, "
          f"hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    
    print(f"Saving merged model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    try:
        m.save_pretrained(args.output)
        t.save_pretrained(args.output)
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")
    
    print(f"Successfully saved merged model to {args.output}")


if __name__ == "__main__":
    main()
