#!/usr/bin/env python3
"""
RLAIF-lite: Synthetic reward training loop.

This script generates samples from the current model, evaluates them,
keeps only the high-quality ones, and creates a training dataset for
fine-tuning the model to produce better outputs.

Usage:
    python rlaif_lite.py --model-path out/llama8b-rust-qlora --output-dir rlaif_data
"""

import os
import json
import jsonlines
import argparse
import torch
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from .eval_rust import compile_and_clippy, is_valid_sample, evaluate_single_sample
    from .gen_eval_samples import PROMPTS
except ImportError:
    from eval_rust import compile_and_clippy, is_valid_sample, evaluate_single_sample
    from gen_eval_samples import PROMPTS


def extract_rust_code(text: str, prompt: str = "") -> str:
    """
    Extract Rust code from model output with robust parsing.
    
    Tries multiple strategies:
    1. Look for ```rust ... ``` code blocks
    2. Look for ``` ... ``` blocks (any language)
    3. Extract code after prompt
    4. Return cleaned text as fallback
    """
    # Strategy 1: Look for ```rust code blocks
    rust_block_pattern = r"```rust\s*\n(.*?)```"
    match = re.search(rust_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Strategy 2: Look for any ``` code blocks
    any_block_pattern = r"```\s*\w*\s*\n(.*?)```"
    match = re.search(any_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Strategy 3: Extract text after prompt (if prompt provided)
    if prompt and prompt in text:
        after_prompt = text.split(prompt, 1)[-1].strip()
        # Remove any leading markdown formatting
        after_prompt = re.sub(r"^```\w*\s*\n?", "", after_prompt, flags=re.MULTILINE)
        after_prompt = re.sub(r"```\s*$", "", after_prompt, flags=re.MULTILINE)
        if after_prompt:
            return after_prompt.strip()
    
    # Strategy 4: Return cleaned text (remove markdown artifacts)
    cleaned = text.strip()
    # Remove leading/trailing markdown code fences
    cleaned = re.sub(r"^```\w*\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def generate_samples(model_path: str, num_samples_per_prompt: int = 10, max_new_tokens: int = 512, 
                     seed: int = 42, tokenizer_path: str = None):
    """Generate samples from the model."""
    print(f"Loading model from {model_path}...")
    
    # Try to load tokenizer from model path first, fall back to tokenizer_path or default
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        print(f"Loaded tokenizer from model path: {model_path}")
    except Exception as e:
        if tokenizer_path:
            print(f"Could not load tokenizer from {model_path}, trying {tokenizer_path}...")
            tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        else:
            print(f"Could not load tokenizer from {model_path}, using default...")
            tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", use_fast=True)
    
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # Set seeds for reproducibility
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    all_samples = []
    system_prompt = "You are a Rust code generator. Output only valid Rust code, wrapped in ```rust code blocks. No explanations or comments outside code blocks."
    
    for prompt in PROMPTS:
        print(f"Generating {num_samples_per_prompt} samples for prompt: {prompt[:50]}...")
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        for _ in range(num_samples_per_prompt):
            x = tok(full_prompt, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                y = mdl.generate(
                    **x, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True,  # Use sampling for diversity
                    temperature=0.7,
                    top_p=0.9,
                )
            txt = tok.decode(y[0], skip_special_tokens=True)
            
            # Extract code using robust extraction
            code = extract_rust_code(txt, prompt)
            
            all_samples.append({"prompt": prompt, "gen": code})
    
    return all_samples


def filter_good_samples(samples, compile_threshold: float = 0.95, clippy_max: float = 2.0, 
                        idiomatic_min: float = 0.7, doc_min: float = 0.5, num_workers: int = None,
                        use_reward_weighting: bool = True):
    """
    Filter samples to keep only high-quality ones using parallel evaluation.
    
    Args:
        samples: List of sample dicts
        compile_threshold: Minimum overall compile rate - if actual rate is below this,
                          filtering thresholds are tightened dynamically
        clippy_max: Maximum average clippy warnings
        idiomatic_min: Minimum idiomatic score
        doc_min: Minimum doc comment rate
        num_workers: Number of parallel workers (None = auto)
        use_reward_weighting: If True, weight samples by quality scores (for weighted dataset)
    
    Returns:
        List of good samples (with optional 'weight' field if use_reward_weighting=True)
    """
    print(f"Evaluating {len(samples)} samples...")
    
    # Evaluate all samples in parallel using the existing evaluation infrastructure
    from multiprocessing import Pool, cpu_count
    from functools import partial
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Evaluate all samples in parallel
    if num_workers > 1 and len(samples) > 1:
        with Pool(processes=num_workers) as pool:
            eval_func = partial(evaluate_single_sample, check_functionality=True)
            results = pool.map(eval_func, samples)
    else:
        results = [evaluate_single_sample(s, check_functionality=True) for s in samples]
    
    # Aggregate overall metrics
    ok_compile = sum(1 for r in results if r["compiled"])
    actual_compile_rate = ok_compile / len(results) if results else 0.0
    clippy_warns = sum(r["clippy_warnings"] for r in results)
    idiomatic_scores = [r["idiomatic_score"] for r in results]
    has_doc_count = sum(1 for r in results if r["has_doc"])
    
    print(f"Overall metrics: compile_rate={actual_compile_rate:.2f}, "
          f"avg_clippy={clippy_warns/len(results):.2f}, "
          f"idiomatic={sum(idiomatic_scores)/len(idiomatic_scores):.2f}, "
          f"doc_rate={has_doc_count/len(results):.2f}")
    
    # Use compile_threshold to dynamically adjust filtering if needed
    if actual_compile_rate < compile_threshold:
        # Tighten thresholds if compile rate is below target
        adjustment_factor = actual_compile_rate / compile_threshold
        adjusted_clippy_max = clippy_max * adjustment_factor
        adjusted_idiomatic_min = idiomatic_min + (1.0 - idiomatic_min) * (1.0 - adjustment_factor)
        print(f"Compile rate {actual_compile_rate:.2f} < threshold {compile_threshold:.2f}, "
              f"tightening filters: clippy_max={adjusted_clippy_max:.2f}, "
              f"idiomatic_min={adjusted_idiomatic_min:.2f}")
    else:
        adjusted_clippy_max = clippy_max
        adjusted_idiomatic_min = idiomatic_min
    
    # Filter samples based on thresholds and compute quality scores
    good_samples = []
    for sample, result in zip(samples, results):
        if not result["compiled"]:
            continue
        if result["clippy_warnings"] > adjusted_clippy_max:
            continue
        if result["idiomatic_score"] < adjusted_idiomatic_min:
            continue
        if not result["has_doc"] and doc_min > 0:
            continue
        
        # Compute quality score for reward weighting
        if use_reward_weighting:
            # Normalize scores to [0, 1] range for weighting
            # Higher is better for all metrics
            clippy_score = max(0, 1.0 - (result["clippy_warnings"] / (adjusted_clippy_max + 1)))
            idiomatic_score = result["idiomatic_score"]  # Already [0, 1]
            doc_score = 1.0 if result["has_doc"] else 0.0
            
            # Weighted combination (can be tuned)
            quality_score = (
                0.4 * clippy_score +      # Compilation is binary, so weight clippy more
                0.4 * idiomatic_score +   # Idiomatic patterns are important
                0.2 * doc_score           # Documentation is nice but less critical
            )
            
            # Add weight to sample (for potential use in training)
            sample_with_weight = sample.copy()
            sample_with_weight["weight"] = quality_score
            sample_with_weight["quality_metrics"] = {
                "clippy_score": clippy_score,
                "idiomatic_score": idiomatic_score,
                "doc_score": doc_score,
                "overall_score": quality_score
            }
            good_samples.append(sample_with_weight)
        else:
            good_samples.append(sample)
    
    print(f"Filtered to {len(good_samples)} good samples ({len(good_samples)/len(samples)*100:.1f}%)")
    if use_reward_weighting and good_samples:
        avg_weight = sum(s.get("weight", 0) for s in good_samples) / len(good_samples)
        print(f"Average quality weight: {avg_weight:.3f}")
    
    return good_samples


def create_training_dataset(good_samples, output_dir: str, seed: int = None, use_weights: bool = False):
    """
    Create training dataset from good samples.
    
    Args:
        good_samples: List of samples (may include 'weight' and 'quality_metrics' fields)
        output_dir: Output directory
        seed: Random seed for reproducibility
        use_weights: If True, include weight information in metadata (for weighted sampling during training)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Prompt -> Code pairs (for instruction tuning)
    instruction_data = []
    for sample in good_samples:
        item = {
            "text": f"{sample['prompt']}\n\n```rust\n{sample['gen']}\n```"
        }
        # Include weight if available and requested
        if use_weights and "weight" in sample:
            item["weight"] = sample["weight"]
        instruction_data.append(item)
    
    # Option 2: Code-only (for continued pretraining)
    code_only = []
    for sample in good_samples:
        item = {"text": sample["gen"]}
        if use_weights and "weight" in sample:
            item["weight"] = sample["weight"]
        code_only.append(item)
    
    # Save both formats
    with jsonlines.open(os.path.join(output_dir, "instruction_data.jsonl"), "w") as w:
        for item in instruction_data:
            w.write(item)
    
    with jsonlines.open(os.path.join(output_dir, "code_only.jsonl"), "w") as w:
        for item in code_only:
            w.write(item)
    
    # Collect quality statistics
    has_weights = any("weight" in s for s in good_samples)
    quality_stats = {}
    if has_weights:
        weights = [s.get("weight", 0) for s in good_samples]
        quality_stats = {
            "avg_weight": sum(weights) / len(weights),
            "min_weight": min(weights),
            "max_weight": max(weights),
            "weighted_samples": len([w for w in weights if w > 0])
        }
    
    # Save metadata including seed for reproducibility
    metadata = {
        "num_samples": len(good_samples),
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "has_weights": has_weights,
    }
    if quality_stats:
        metadata["quality_stats"] = quality_stats
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(instruction_data)} instruction samples and {len(code_only)} code-only samples to {output_dir}")
    if seed is not None:
        print(f"Generation seed: {seed} (recorded in {output_dir}/metadata.json)")
    if has_weights:
        print(f"Quality weights included (avg: {quality_stats['avg_weight']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="RLAIF-lite: Generate and filter high-quality samples")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="rlaif_data", help="Output directory for training data")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per prompt")
    parser.add_argument("--compile-threshold", type=float, default=0.95, 
                       help="Target compile rate - if actual rate is below this, filtering thresholds are tightened dynamically")
    parser.add_argument("--clippy-max", type=float, default=2.0, help="Max clippy warnings")
    parser.add_argument("--idiomatic-min", type=float, default=0.7, help="Min idiomatic score")
    parser.add_argument("--doc-min", type=float, default=0.5, help="Min doc comment rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (None=auto)")
    parser.add_argument("--tokenizer-path", type=str, default=None, 
                       help="Tokenizer path (default: try model_path, then fallback to default)")
    parser.add_argument("--no-reward-weighting", action="store_true", 
                       help="Disable reward weighting (all samples treated equally)")
    parser.add_argument("--use-weights", action="store_true",
                       help="Include weight information in output dataset (for weighted sampling during training)")
    
    args = parser.parse_args()
    
    # Generate samples
    samples = generate_samples(
        args.model_path, 
        num_samples_per_prompt=args.num_samples, 
        seed=args.seed,
        tokenizer_path=args.tokenizer_path
    )
    
    # Filter good samples using parallel evaluation
    good_samples = filter_good_samples(
        samples,
        compile_threshold=args.compile_threshold,
        clippy_max=args.clippy_max,
        idiomatic_min=args.idiomatic_min,
        doc_min=args.doc_min,
        num_workers=args.num_workers,
        use_reward_weighting=not args.no_reward_weighting
    )
    
    # Create training dataset
    if good_samples:
        create_training_dataset(good_samples, args.output_dir, seed=args.seed, use_weights=args.use_weights)
        print(f"\nNext step: Fine-tune on {args.output_dir}/instruction_data.jsonl or code_only.jsonl")
        print(f"Use a low learning rate (e.g., 5e-5) and fewer steps (e.g., 1000-2000)")
        print(f"Dataset generated with seed {args.seed} (see {args.output_dir}/metadata.json)")
    else:
        print("No good samples found. Model may need more training before RLAIF.")


if __name__ == "__main__":
    main()

