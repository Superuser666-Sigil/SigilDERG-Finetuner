import os
import random
import jsonlines
import torch
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

# Default prompts (used if no prompts file is provided)
DEFAULT_PROMPTS = [
  "Write only a single Rust source file (main.rs) with a `fn main() -> anyhow::Result<()>` that demonstrates basic anyhow error handling. Do not explain; output only Rust code wrapped in ```rust code blocks.",
  "Write only Rust code for a program that uses iterators with map/filter to compute a sum. Include `fn main()` and necessary imports. No explanations. Wrap code in ```rust code blocks.",
  "Define a newtype wrapper for [u8; 32] and implement FromStr and Display. Provide a small `fn main()` that parses and prints it. Output only Rust code in ```rust code blocks.",
  "Create a small example showing a thiserror enum, Display, a source cause, and a function using the ? operator. Include `fn main()` to drive it. Output only Rust code wrapped in ```rust code blocks.",
  "Write a complete Rust program with `fn main()` that demonstrates pattern matching with `match` and `if let`. Include necessary imports. Output only code in ```rust code blocks.",
  "Create a Rust program with a struct, an impl block, and a trait implementation. Include `fn main()` to demonstrate usage. Output only code in ```rust code blocks.",
  "Write a Rust program that uses serde to serialize and deserialize a struct to JSON. Include `fn main()` and use serde_json. Output only Rust code in ```rust code blocks.",
  "Create a Rust program that uses the regex crate to find and replace patterns in a string. Include `fn main()` with example text. Output only Rust code in ```rust code blocks.",
  "Write a Rust program that uses chrono to parse a date string and format it differently. Include `fn main()` with example dates. Output only Rust code in ```rust code blocks.",
  "Create a Rust program that generates UUIDs using the uuid crate. Include `fn main()` that generates and prints multiple UUIDs. Output only Rust code in ```rust code blocks.",
  "Write a Rust program that uses the rand crate to generate random numbers and select random items from a vector. Include `fn main()`. Output only Rust code in ```rust code blocks.",
]


def load_prompts(prompts_path: str | None) -> list[str]:
    """
    Load prompts from YAML or JSON file, or return default prompts.
    
    Args:
        prompts_path: Path to YAML or JSON file containing prompts.
                     File should contain a list of strings or a dict with 'prompts' key.
    
    Returns:
        List of prompt strings
    """
    if not prompts_path or not os.path.exists(prompts_path):
        return DEFAULT_PROMPTS
    
    import json
    import yaml
    
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            if prompts_path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            elif prompts_path.endswith(".json"):
                data = json.load(f)
            else:
                # Try to auto-detect format
                content = f.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = yaml.safe_load(content)
            
            # Handle different formats
            if isinstance(data, list):
                return [str(p) for p in data]
            elif isinstance(data, dict):
                if "prompts" in data:
                    return [str(p) for p in data["prompts"]]
                else:
                    raise ValueError(f"Expected 'prompts' key in {prompts_path}")
            else:
                raise ValueError(f"Invalid format in {prompts_path}: expected list or dict")
    except Exception as e:
        print(f"Warning: Failed to load prompts from {prompts_path}: {e}")
        print("Falling back to default prompts.")
        return DEFAULT_PROMPTS

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate evaluation samples from fine-tuned model")
    ap.add_argument("--model-path", default="out/llama8b-rust-qlora-phase1", help="Path to model checkpoint or checkpoint directory")
    ap.add_argument("--output-dir", default="eval_out", help="Output directory for samples")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--samples-per-prompt", type=int, default=None, help="Number of samples to generate per prompt (default: auto-calculated to ensure enough samples)")
    ap.add_argument("--min-total-samples", type=int, default=64, help="Minimum total samples to generate (default: 64)")
    ap.add_argument("--prompts-file", type=str, default=None, help="Path to YAML or JSON file containing prompts (default: use built-in prompts)")
    args = ap.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find the latest checkpoint if a directory is provided
    model_path = args.model_path
    if os.path.isdir(model_path):
        # Look for checkpoint directories (checkpoint-*)
        checkpoints = glob.glob(os.path.join(model_path, "checkpoint-*"))
        if checkpoints:
            # Sort by step number and use the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            model_path = checkpoints[-1]
            print(f"Using latest checkpoint: {model_path}")
    
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Try to load as PEFT adapter first (LoRA checkpoint), fall back to full model
    try:
        mdl = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16
        )
        print(f"Loaded PEFT adapter from {model_path}")
    except Exception as e:
        # Fall back to full model loading (merged checkpoint or base model)
        print(f"Could not load as PEFT adapter, trying as full model: {e}")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16
        )
        print(f"Loaded full model from {model_path}")
    
    mdl.eval()  # Set to eval mode for consistent generation
    
    # Ensure tokenizer has pad_token set
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Load prompts from file or use defaults
    prompts = load_prompts(args.prompts_file)
    if args.prompts_file:
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        print(f"Using {len(prompts)} default prompts")
    
    # Calculate samples per prompt if not specified
    num_prompts = len(prompts)
    if args.samples_per_prompt is None:
        # Generate enough samples to meet minimum requirement
        # Add 20% buffer to account for potential filtering
        samples_per_prompt = int((args.min_total_samples * 1.2) / num_prompts) + 1
    else:
        samples_per_prompt = args.samples_per_prompt
    
    print(f"Generating {samples_per_prompt} samples per prompt ({num_prompts} prompts = {samples_per_prompt * num_prompts} total)")
    
    outs = []
    for p in prompts:
        # Use chat template for instruct models (properly formats system/user messages)
        messages = [
            {"role": "system", "content": "You are a Rust code generator. Output only valid Rust code, wrapped in ```rust code blocks. No explanations or comments outside code blocks."},
            {"role": "user", "content": p}
        ]
        # apply_chat_template handles all the special tokens automatically for instruct models
        full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate multiple samples per prompt for diversity
        for sample_idx in range(samples_per_prompt):
            x = tok(full_prompt, return_tensors="pt").to(mdl.device)
            input_length = x["input_ids"].shape[1]
            
            with torch.no_grad():
                # Use sampling for diversity when generating multiple samples per prompt
                if samples_per_prompt > 1:
                    # Sampling for diversity
                    y = mdl.generate(
                        **x, 
                        max_new_tokens=512, 
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tok.eos_token_id
                    )
                else:
                    # Greedy decoding for single sample (deterministic)
                    y = mdl.generate(**x, max_new_tokens=512, do_sample=False, temperature=None, pad_token_id=tok.eos_token_id)
            
            # Extract only the newly generated tokens (not the input prompt)
            generated_ids = y[0][input_length:]
            txt = tok.decode(generated_ids, skip_special_tokens=True)
            
            # Try to extract code fence if present
            # Handle variations: ```rust, ```rs, ```, etc.
            code_blocks = []
            # Try rust-specific fence first
            if "```rust" in txt:
                parts = txt.split("```rust")
                for part in parts[1:]:  # Skip first part (before first fence)
                    code_block = part.split("```")[0].strip()
                    if code_block:
                        code_blocks.append(code_block)
            elif "```rs" in txt:
                parts = txt.split("```rs")
                for part in parts[1:]:
                    code_block = part.split("```")[0].strip()
                    if code_block:
                        code_blocks.append(code_block)
            elif "```" in txt:
                # Try generic code fence
                parts = txt.split("```")
                # Skip first part, then take every other part (code blocks)
                for i in range(1, len(parts), 2):
                    if i < len(parts):
                        code_block = parts[i].strip()
                        # Skip language identifier if present
                        if code_block.startswith(("rust", "rs", "\n")):
                            code_block = code_block.split("\n", 1)[-1] if "\n" in code_block else ""
                        if code_block:
                            code_blocks.append(code_block)
            
            # Use first code block found, or entire text if no fences
            if code_blocks:
                snip = code_blocks[0]
            else:
                # Fallback: use the entire response (might be code without fences)
                snip = txt.strip()
            
            outs.append({"prompt": p, "gen": snip})
    output_path = os.path.join(args.output_dir, "samples.jsonl")
    with jsonlines.open(output_path, "w") as w:
        for r in outs: w.write(r)
    print(f"Generated {len(outs)} samples, saved to {output_path}")

if __name__ == "__main__":
    main()
