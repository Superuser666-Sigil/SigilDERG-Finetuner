#!/usr/bin/env python3
"""
Hyperparameter sweep script for QLoRA fine-tuning.

This script runs multiple training runs with different hyperparameter combinations
to find optimal settings. Results are logged to TensorBoard for comparison.
"""

import os
import yaml
import subprocess
import itertools
from pathlib import Path


def load_yaml(p):
    """Load YAML configuration file."""
    with open(p) as f:
        return yaml.safe_load(f)


def save_yaml(cfg, p):
    """Save YAML configuration file."""
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_training(cfg_path, output_suffix):
    """Run a single training job."""
    print(f"\n{'='*60}")
    print(f"Starting training with config: {cfg_path}")
    print(f"Output suffix: {output_suffix}")
    print(f"{'='*60}\n")
    
    cmd = ["python", "train.py", "--cfg", cfg_path]
    result = subprocess.run(cmd, cwd=os.path.dirname(cfg_path) or ".")
    
    if result.returncode != 0:
        print(f"Warning: Training failed for {output_suffix}")
    else:
        print(f"Training completed successfully for {output_suffix}")
    
    return result.returncode == 0


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run hyperparameter sweep for QLoRA fine-tuning"
    )
    ap.add_argument(
        "--base-cfg",
        default="configs/llama8b.yml",
        help="Base configuration file"
    )
    ap.add_argument(
        "--sweep-dir",
        default="sweeps",
        help="Directory to save sweep configurations and results"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sweep configurations without running training"
    )
    args = ap.parse_args()
    
    base_cfg = load_yaml(args.base_cfg)
    sweep_dir = Path(args.sweep_dir)
    sweep_dir.mkdir(exist_ok=True)
    
    # Define hyperparameter search space
    # Adjust these ranges based on your needs and compute budget
    search_space = {
        "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
        "lora_r": [8, 16, 32],
        "lora_alpha": [8, 16, 32],
        "warmup_steps": [100, 250, 500],
        # "max_seq_len": [2048, 4096],  # Uncomment to sweep sequence length
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"Total combinations: {len(combinations)}")
    print(f"Search space: {search_space}\n")
    
    if args.dry_run:
        print("DRY RUN - Configuration preview:")
        for i, combo in enumerate(combinations):
            print(f"\nRun {i+1}:")
            for key, val in zip(keys, combo):
                print(f"  {key}: {val}")
        return
    
    # Run each combination
    results = []
    for i, combo in enumerate(combinations):
        # Create modified config
        cfg = base_cfg.copy()
        cfg = {**cfg}  # Shallow copy
        
        # Update config with sweep values
        for key, val in zip(keys, combo):
            if key == "learning_rate":
                cfg["train"]["lr"] = val
            elif key == "lora_r":
                cfg["lora"]["r"] = val
            elif key == "lora_alpha":
                cfg["lora"]["alpha"] = val
            elif key == "warmup_steps":
                cfg["train"]["warmup_steps"] = val
            elif key == "max_seq_len":
                cfg["max_seq_len"] = val
        
        # Create unique output directory
        suffix_parts = [f"{k}_{v}" for k, v in zip(keys, combo)]
        suffix = "_".join(suffix_parts).replace(".", "p")
        cfg["misc"]["output_dir"] = f"{base_cfg['misc']['output_dir']}_{suffix}"
        cfg["misc"]["logging_dir"] = f"{cfg['misc']['output_dir']}/logs"
        
        # Save sweep config
        sweep_cfg_path = sweep_dir / f"config_{i+1:03d}_{suffix}.yml"
        save_yaml(cfg, sweep_cfg_path)
        
        # Run training
        success = run_training(str(sweep_cfg_path), suffix)
        results.append({
            "config": suffix,
            "params": dict(zip(keys, combo)),
            "success": success
        })
        
        print(f"\nCompleted {i+1}/{len(combinations)} runs")
    
    # Summary
    print(f"\n{'='*60}")
    print("Sweep Summary:")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["success"])
    print(f"Successful runs: {successful}/{len(results)}")
    print(f"\nView results in TensorBoard:")
    print(f"  tensorboard --logdir {base_cfg['misc']['output_dir']}")


if __name__ == "__main__":
    main()

