#!/usr/bin/env python3
"""
Inspect checkpoint directory to see what files are generated.

This script lists all files in a checkpoint directory and provides
information about their purpose and size.
"""

import os
import json
import argparse
from pathlib import Path


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def inspect_checkpoint(checkpoint_dir):
    """Inspect a checkpoint directory and list all files."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    if not checkpoint_path.is_dir():
        print(f"Error: Path is not a directory: {checkpoint_dir}")
        return
    
    print(f"Checkpoint Directory: {checkpoint_path.absolute()}")
    print("=" * 80)
    
    # List all files
    files = sorted(checkpoint_path.iterdir())
    
    if not files:
        print("No files found in checkpoint directory.")
        return
    
    total_size = 0
    file_info = []
    
    for file_path in files:
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_info.append((file_path.name, size))
    
    # Sort by size (largest first)
    file_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFiles ({len(file_info)} total):\n")
    print(f"{'Filename':<40} {'Size':<15} {'Purpose'}")
    print("-" * 80)
    
    # Known file purposes
    purposes = {
        'adapter_model.bin': 'PEFT LoRA adapter weights (binary)',
        'adapter_model.safetensors': 'PEFT LoRA adapter weights (safetensors)',
        'adapter_config.json': 'PEFT adapter configuration',
        'trainer_state.json': 'Training state (step, epoch, metrics history)',
        'training_args.bin': 'Serialized TrainingArguments',
        'optimizer.pt': 'Optimizer state (AdamW, etc.)',
        'scheduler.pt': 'Learning rate scheduler state',
        'rng_state.pth': 'Random number generator state',
        'README.md': 'Auto-generated model card (HuggingFace)',
        'config.json': 'Model configuration (architecture, etc.)',
        'generation_config.json': 'Generation parameters (temperature, etc.)',
        'tokenizer_config.json': 'Tokenizer configuration',
        'tokenizer.json': 'Tokenizer model file',
        'special_tokens_map.json': 'Special tokens mapping',
        'vocab.json': 'Vocabulary file (if applicable)',
        'merges.txt': 'BPE merges (if applicable)',
    }
    
    for filename, size in file_info:
        purpose = purposes.get(filename, 'Unknown file')
        print(f"{filename:<40} {format_size(size):<15} {purpose}")
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {format_size(total_size):<15}")
    
    # Try to read and display key configuration files
    print("\n" + "=" * 80)
    print("Key Configuration Files:\n")
    
    # PEFT adapter config
    adapter_config = checkpoint_path / "adapter_config.json"
    if adapter_config.exists():
        print("PEFT Adapter Configuration:")
        try:
            with open(adapter_config, 'r') as f:
                config = json.load(f)
                print(json.dumps(config, indent=2))
        except Exception as e:
            print(f"  Error reading adapter_config.json: {e}")
        print()
    
    # Trainer state (show step info)
    trainer_state = checkpoint_path / "trainer_state.json"
    if trainer_state.exists():
        print("Training State (excerpt):")
        try:
            with open(trainer_state, 'r') as f:
                state = json.load(f)
                print(f"  Global Step: {state.get('global_step', 'N/A')}")
                print(f"  Epoch: {state.get('epoch', 'N/A')}")
                if 'log_history' in state and state['log_history']:
                    last_log = state['log_history'][-1]
                    print(f"  Last Log Entry: {last_log}")
        except Exception as e:
            print(f"  Error reading trainer_state.json: {e}")
        print()
    
    # Model config
    model_config = checkpoint_path / "config.json"
    if model_config.exists():
        print("Model Configuration (excerpt):")
        try:
            with open(model_config, 'r') as f:
                config = json.load(f)
                print(f"  Model Type: {config.get('model_type', 'N/A')}")
                print(f"  Hidden Size: {config.get('hidden_size', 'N/A')}")
                print(f"  Num Layers: {config.get('num_hidden_layers', 'N/A')}")
                print(f"  Vocab Size: {config.get('vocab_size', 'N/A')}")
        except Exception as e:
            print(f"  Error reading config.json: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint directory contents"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint directory (e.g., out/llama8b-rust-qlora-phase1/checkpoint-1000)"
    )
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()

