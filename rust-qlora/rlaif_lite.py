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
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_rust import compile_and_clippy, is_valid_sample
from gen_eval_samples import PROMPTS


def generate_samples(model_path: str, num_samples_per_prompt: int = 10, max_new_tokens: int = 512):
    """Generate samples from the model."""
    print(f"Loading model from {model_path}...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
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
            
            # Extract code
            code = txt.split("```rust")
            if len(code) > 1:
                snip = code[1].split("```")[0].strip()
            else:
                snip = txt.split(prompt)[-1].strip() if prompt in txt else txt.strip()
            
            all_samples.append({"prompt": prompt, "gen": snip})
    
    return all_samples


def filter_good_samples(samples, compile_threshold: float = 0.95, clippy_max: float = 2.0, 
                        idiomatic_min: float = 0.7, doc_min: float = 0.5):
    """
    Filter samples to keep only high-quality ones.
    
    Args:
        samples: List of sample dicts
        compile_threshold: Minimum compile rate to consider (not used directly, but for reference)
        clippy_max: Maximum average clippy warnings
        idiomatic_min: Minimum idiomatic score
        doc_min: Minimum doc comment rate
    
    Returns:
        List of good samples
    """
    print(f"Evaluating {len(samples)} samples...")
    
    # Evaluate all samples
    metrics = compile_and_clippy(
        samples, 
        sample_n=len(samples), 
        check_functionality=True,
        pre_filter=True
    )
    
    print(f"Overall metrics: compile_rate={metrics.get('compile_rate', 0):.2f}, "
          f"avg_clippy={metrics.get('avg_clippy_warnings', 0):.2f}, "
          f"idiomatic={metrics.get('avg_idiomatic_score', 0):.2f}, "
          f"doc_rate={metrics.get('doc_comment_rate', 0):.2f}")
    
    # Evaluate each sample individually
    good_samples = []
    from eval_rust import has_doc_comments, has_idiomatic_patterns
    import subprocess
    import tempfile
    
    for sample in samples:
        code = sample.get("gen", "")
        prompt = sample.get("prompt", "")
        
        # Pre-filter
        is_valid, _ = is_valid_sample(code, prompt)
        if not is_valid:
            continue
        
        # Check idiomatic patterns
        idiomatic = has_idiomatic_patterns(code)
        idiomatic_score = sum(idiomatic.values()) / max(len(idiomatic), 1)
        if idiomatic_score < idiomatic_min:
            continue
        
        # Check documentation
        has_doc = has_doc_comments(code)
        if not has_doc and doc_min > 0:
            continue
        
        # Try to compile
        try:
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(
                    ["cargo", "new", "--quiet", "app"],
                    cwd=td,
                    check=True,
                    capture_output=True
                )
                proj = os.path.join(td, "app")
                with open(os.path.join(proj, "src", "main.rs"), "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Compile check
                c1 = subprocess.run(
                    ["cargo", "check", "-q"],
                    cwd=proj,
                    capture_output=True,
                    timeout=30
                )
                if c1.returncode != 0:
                    continue
                
                # Clippy check
                c2 = subprocess.run(
                    ["cargo", "clippy", "-q", "--", "-W", "clippy::all"],
                    cwd=proj,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                clippy_warns = c2.stdout.count(": warning:")
                if clippy_warns > clippy_max:
                    continue
                
                # Passed all checks!
                good_samples.append(sample)
        except Exception:
            continue
    
    print(f"Filtered to {len(good_samples)} good samples ({len(good_samples)/len(samples)*100:.1f}%)")
    return good_samples


def create_training_dataset(good_samples, output_dir: str):
    """Create training dataset from good samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Option 1: Prompt -> Code pairs (for instruction tuning)
    instruction_data = []
    for sample in good_samples:
        instruction_data.append({
            "text": f"{sample['prompt']}\n\n```rust\n{sample['gen']}\n```"
        })
    
    # Option 2: Code-only (for continued pretraining)
    code_only = [{"text": sample["gen"]} for sample in good_samples]
    
    # Save both formats
    with jsonlines.open(os.path.join(output_dir, "instruction_data.jsonl"), "w") as w:
        for item in instruction_data:
            w.write(item)
    
    with jsonlines.open(os.path.join(output_dir, "code_only.jsonl"), "w") as w:
        for item in code_only:
            w.write(item)
    
    print(f"Saved {len(instruction_data)} instruction samples and {len(code_only)} code-only samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="RLAIF-lite: Generate and filter high-quality samples")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="rlaif_data", help="Output directory for training data")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per prompt")
    parser.add_argument("--compile-threshold", type=float, default=0.95, help="Target compile rate (reference)")
    parser.add_argument("--clippy-max", type=float, default=2.0, help="Max clippy warnings")
    parser.add_argument("--idiomatic-min", type=float, default=0.7, help="Min idiomatic score")
    parser.add_argument("--doc-min", type=float, default=0.5, help="Min doc comment rate")
    
    args = parser.parse_args()
    
    # Generate samples
    samples = generate_samples(args.model_path, num_samples_per_prompt=args.num_samples)
    
    # Filter good samples
    good_samples = filter_good_samples(
        samples,
        compile_threshold=args.compile_threshold,
        clippy_max=args.clippy_max,
        idiomatic_min=args.idiomatic_min,
        doc_min=args.doc_min
    )
    
    # Create training dataset
    if good_samples:
        create_training_dataset(good_samples, args.output_dir)
        print(f"\nNext step: Fine-tune on {args.output_dir}/instruction_data.jsonl or code_only.jsonl")
        print(f"Use a low learning rate (e.g., 5e-5) and fewer steps (e.g., 1000-2000)")
    else:
        print("No good samples found. Model may need more training before RLAIF.")


if __name__ == "__main__":
    main()

