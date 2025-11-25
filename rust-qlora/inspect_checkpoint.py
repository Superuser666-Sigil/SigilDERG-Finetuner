#!/usr/bin/env python3
"""
Inspect checkpoint directory to see what files are generated.

This script lists all files in a checkpoint directory and provides
information about their purpose and size.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
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


def inspect_checkpoint(checkpoint_dir, json_output=False):
    """
    Inspect a checkpoint directory and list all files.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        json_output: If True, return dict instead of printing
    
    Returns:
        Dict with checkpoint information if json_output=True, else None
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        error_msg = f"Error: Checkpoint directory does not exist: {checkpoint_dir}"
        if json_output:
            return {"error": error_msg}
        print(error_msg)
        return
    
    if not checkpoint_path.is_dir():
        error_msg = f"Error: Path is not a directory: {checkpoint_dir}"
        if json_output:
            return {"error": error_msg}
        print(error_msg)
        return
    
    if json_output:
        result = {
            "checkpoint_dir": str(checkpoint_path.absolute()),
            "files": [],
            "total_size_bytes": 0,
            "configs": {}
        }
    else:
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
    
    if json_output:
        result["files"] = [
            {
                "filename": filename,
                "size_bytes": size,
                "size_human": format_size(size),
                "purpose": purposes.get(filename, 'Unknown file')
            }
            for filename, size in file_info
        ]
        result["total_size_bytes"] = total_size
        result["total_size_human"] = format_size(total_size)
    else:
        print(f"\nFiles ({len(file_info)} total):\n")
        print(f"{'Filename':<40} {'Size':<15} {'Purpose'}")
        print("-" * 80)
        
        for filename, size in file_info:
            purpose = purposes.get(filename, 'Unknown file')
            print(f"{filename:<40} {format_size(size):<15} {purpose}")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {format_size(total_size):<15}")
    
    # Try to read and display key configuration files
    if json_output:
        # PEFT adapter config
        adapter_config = checkpoint_path / "adapter_config.json"
        if adapter_config.exists():
            try:
                with open(adapter_config, 'r') as f:
                    result["configs"]["adapter_config"] = json.load(f)
            except Exception as e:
                result["configs"]["adapter_config_error"] = str(e)
        
        # Trainer state
        trainer_state = checkpoint_path / "trainer_state.json"
        if trainer_state.exists():
            try:
                with open(trainer_state, 'r') as f:
                    state = json.load(f)
                    result["configs"]["trainer_state"] = {
                        "global_step": state.get('global_step'),
                        "epoch": state.get('epoch'),
                        "last_log_entry": state.get('log_history', [])[-1] if state.get('log_history') else None
                    }
            except Exception as e:
                result["configs"]["trainer_state_error"] = str(e)
        
        # Model config
        model_config = checkpoint_path / "config.json"
        if model_config.exists():
            try:
                with open(model_config, 'r') as f:
                    config = json.load(f)
                    result["configs"]["model_config"] = {
                        "model_type": config.get('model_type'),
                        "hidden_size": config.get('hidden_size'),
                        "num_hidden_layers": config.get('num_hidden_layers'),
                        "vocab_size": config.get('vocab_size')
                    }
            except Exception as e:
                result["configs"]["model_config_error"] = str(e)
        
        return result
    else:
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output information in JSON format instead of human-readable text"
    )
    args = parser.parse_args()
    
    result = inspect_checkpoint(args.checkpoint, json_output=args.json)
    
    if args.json and result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

