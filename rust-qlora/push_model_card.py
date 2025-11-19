#!/usr/bin/env python3
"""
Generate and push model card README.md to HuggingFace Hub.

This script generates a comprehensive model card from a checkpoint and
optionally pushes it to your HuggingFace repository.
"""

import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

# Import the model card generation logic from train.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train import ModelCardCallback, load_yaml


def generate_model_card_from_checkpoint(checkpoint_dir, config_file=None):
    """Generate model card from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Try to find config file
    if config_file and os.path.exists(config_file):
        cfg = load_yaml(config_file)
    else:
        # Try to infer config from checkpoint path
        # Look for common config locations
        possible_configs = [
            "rust-qlora/configs/llama8b-phase1.yml",
            "rust-qlora/configs/llama8b-phase2.yml",
            "configs/llama8b-phase1.yml",
            "configs/llama8b-phase2.yml",
        ]
        cfg = None
        for config_path in possible_configs:
            if os.path.exists(config_path):
                cfg = load_yaml(config_path)
                break
        
        if not cfg:
            raise ValueError("Could not find config file. Please specify --config")
    
    # Get model ID from config
    model_id = cfg.get("model_name", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Try to extract checkpoint step from directory name (e.g., checkpoint-2000)
    checkpoint_name = checkpoint_path.name
    global_step = 0
    if checkpoint_name.startswith("checkpoint-"):
        try:
            global_step = int(checkpoint_name.split("-")[1])
        except (ValueError, IndexError):
            pass
    
    # Try to read trainer state for metrics (may override step if available)
    trainer_state_path = checkpoint_path / "trainer_state.json"
    latest_metrics = {}
    
    if trainer_state_path.exists():
        import json
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
            # Use step from trainer_state if available (more accurate)
            if trainer_state.get("global_step"):
                global_step = trainer_state.get("global_step", global_step)
            if trainer_state.get("log_history"):
                latest_metrics = trainer_state["log_history"][-1]
    
    # Extract dataset names
    dataset_config = cfg.get("dataset", {})
    dataset_names = dataset_config.get("names", cfg.get("dataset_name", []))
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Generate model card
    callback = ModelCardCallback(cfg, model_id)
    readme_content = callback._generate_model_card(
        checkpoint_path=str(checkpoint_path),
        global_step=global_step,
        latest_metrics=latest_metrics,
        dataset_names=dataset_names
    )
    
    return readme_content


def push_to_hub(readme_content, repo_id, token=None):
    """Push README.md to HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    import tempfile
    
    # Create temporary file with README content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(readme_content)
        tmp_path = tmp_file.name
    
    try:
        # Initialize API with token
        if token:
            api = HfApi(token=token)
        else:
            # Try to use token from environment or existing login
            api = HfApi(token=os.getenv("HF_TOKEN"))
            # Verify we can authenticate
            try:
                api.whoami()
            except Exception:
                print("Not logged in. Please run: huggingface-cli login")
                print("Or set HF_TOKEN environment variable, or provide --token")
                return False
        
        # Upload README
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card with comprehensive metadata and evaluation results"
        )
        print(f"✓ Successfully pushed README.md to {repo_id}")
        return True
    except Exception as e:
        print(f"✗ Failed to push to Hub: {e}")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate and push model card README.md to HuggingFace Hub"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (e.g., out/llama8b-rust-qlora-phase1/checkpoint-1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g., Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora). Required for --push"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub (requires --repo-id)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (optional, will use existing login if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: checkpoint_dir/README.md or stdout if not pushing)"
    )
    args = parser.parse_args()
    
    # Generate model card
    print(f"Generating model card from {args.checkpoint_dir}...")
    try:
        readme_content = generate_model_card_from_checkpoint(
            args.checkpoint_dir,
            args.config
        )
    except Exception as e:
        print(f"✗ Failed to generate model card: {e}")
        return 1
    
    # Determine output location
    if args.output:
        output_path = args.output
    elif args.push:
        # If pushing, save to checkpoint dir
        output_path = os.path.join(args.checkpoint_dir, "README.md")
    else:
        # If not pushing and no output specified, print to stdout
        print("\n" + "=" * 80)
        print("Generated Model Card:")
        print("=" * 80 + "\n")
        print(readme_content)
        return 0
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✓ Saved model card to {output_path}")
    
    # Push to Hub if requested
    if args.push:
        if not args.repo_id:
            print("✗ Error: --repo-id is required when using --push")
            return 1
        
        print(f"Pushing to HuggingFace Hub: {args.repo_id}...")
        if push_to_hub(readme_content, args.repo_id, args.token):
            return 0
        else:
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

