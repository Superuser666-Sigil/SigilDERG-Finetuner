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
import copy
from pathlib import Path

# Import Pydantic config models
try:
    from .config_models import TrainingConfig
except ImportError:
    try:
        from config_models import TrainingConfig
    except ImportError:
        TrainingConfig = None


def load_yaml(p):
    """Load YAML configuration file with Pydantic validation if available."""
    if TrainingConfig is not None:
        try:
            cfg_obj = TrainingConfig.from_yaml(p)
            # Convert to dict for manipulation in sweep
            return cfg_obj.to_dict()
        except Exception as e:
            print(f"Warning: Pydantic validation failed: {e}")
            print("Falling back to raw YAML loading")
    
    # Fallback to raw YAML
    with open(p) as f:
        return yaml.safe_load(f)


def save_yaml(cfg, p):
    """Save YAML configuration file."""
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def update_nested_config(cfg: dict, key_path: str, value):
    """
    Update a nested configuration value using dot-separated key path.
    
    Args:
        cfg: Configuration dictionary to update
        key_path: Dot-separated path (e.g., "train.lr" or "lora.r")
        value: Value to set
    
    Example:
        update_nested_config(cfg, "train.lr", 1e-4)
        update_nested_config(cfg, "lora.r", 16)
    """
    keys = key_path.split(".")
    current = cfg
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def run_training(cfg_path, output_suffix, timeout=None):
    """Run a single training job with timeout and error handling."""
    print(f"\n{'='*60}")
    print(f"Starting training with config: {cfg_path}")
    print(f"Output suffix: {output_suffix}")
    print(f"{'='*60}\n")
    
    cmd = ["python", "train.py", "--cfg", cfg_path]
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(cfg_path) or ".",
            timeout=timeout,
            capture_output=False  # Let output stream to console
        )
        
        if result.returncode != 0:
            print(f"Warning: Training failed for {output_suffix} (exit code: {result.returncode})")
            return False
        else:
            print(f"Training completed successfully for {output_suffix}")
            return True
    except subprocess.TimeoutExpired:
        print(f"Error: Training timed out for {output_suffix}")
        return False
    except Exception as e:
        print(f"Error: Training failed with exception for {output_suffix}: {e}")
        return False


def main():
    import argparse
    import random
    ap = argparse.ArgumentParser(
        description="Run hyperparameter sweep for QLoRA fine-tuning"
    )
    ap.add_argument(
        "--base-cfg",
        default="configs/llama8b-phase1.yml",
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
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sweep reproducibility"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout per training run in seconds (None = no timeout)"
    )
    args = ap.parse_args()
    
    # Set seed for reproducibility
    random.seed(args.seed)
    
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
        # Create deep copy of config to avoid mutating base config
        cfg = copy.deepcopy(base_cfg)
        
        # Map search space keys to config paths
        key_mapping = {
            "learning_rate": "train.lr",
            "lora_r": "lora.r",
            "lora_alpha": "lora.alpha",
            "warmup_steps": "train.warmup_steps",
            "max_seq_len": "max_seq_len"
        }
        
        # Update config with sweep values using helper function
        for key, val in zip(keys, combo):
            config_path = key_mapping.get(key, key)
            update_nested_config(cfg, config_path, val)
        
        # Create unique output directory with run ID
        suffix_parts = [f"{k}_{v}" for k, v in zip(keys, combo)]
        suffix = "_".join(suffix_parts).replace(".", "p")
        run_id = f"run_{i+1:03d}_{suffix}"
        cfg["misc"]["output_dir"] = f"{base_cfg['misc']['output_dir']}_{suffix}"
        cfg["misc"]["logging_dir"] = f"{cfg['misc']['output_dir']}/logs"
        
        # Add run metadata to config
        if "sweep_metadata" not in cfg["misc"]:
            cfg["misc"]["sweep_metadata"] = {}
        cfg["misc"]["sweep_metadata"]["run_id"] = run_id
        cfg["misc"]["sweep_metadata"]["run_index"] = i + 1
        cfg["misc"]["sweep_metadata"]["total_runs"] = len(combinations)
        cfg["misc"]["sweep_metadata"]["params"] = dict(zip(keys, combo))
        
        # Save sweep config
        sweep_cfg_path = sweep_dir / f"config_{run_id}.yml"
        save_yaml(cfg, sweep_cfg_path)
        
        # Run training
        success = run_training(str(sweep_cfg_path), suffix, timeout=args.timeout)
        results.append({
            "run_id": run_id,
            "run_index": i + 1,
            "config": suffix,
            "params": dict(zip(keys, combo)),
            "output_dir": cfg["misc"]["output_dir"],
            "logging_dir": cfg["misc"]["logging_dir"],
            "success": success
        })
        
        print(f"\nCompleted {i+1}/{len(combinations)} runs")
    
    # Save results summary
    import json
    summary_path = sweep_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if r["success"]),
            "failed_runs": sum(1 for r in results if not r["success"]),
            "results": results,
            "base_config": args.base_cfg,
            "seed": args.seed
        }, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("Sweep Summary:")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["success"])
    print(f"Successful runs: {successful}/{len(results)}")
    print(f"Failed runs: {len(results) - successful}/{len(results)}")
    print(f"\nResults summary saved to: {summary_path}")
    print(f"\nView results in TensorBoard:")
    print(f"  tensorboard --logdir {base_cfg['misc']['output_dir']}")
    print(f"\nOr view individual runs:")
    for r in results[:5]:  # Show first 5
        if r["success"]:
            print(f"  - {r['run_id']}: {r['logging_dir']}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more (see {summary_path})")


if __name__ == "__main__":
    main()

