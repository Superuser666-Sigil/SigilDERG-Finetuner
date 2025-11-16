import os, random, jsonlines, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
  "Write only a single Rust source file (main.rs) with a `fn main() -> anyhow::Result<()>` that demonstrates basic anyhow error handling. Do not explain; output only Rust code wrapped in ```rust code blocks.",
  "Write only Rust code for a program that uses iterators with map/filter to compute a sum. Include `fn main()` and necessary imports. No explanations. Wrap code in ```rust code blocks.",
  "Define a newtype wrapper for [u8; 32] and implement FromStr and Display. Provide a small `fn main()` that parses and prints it. Output only Rust code in ```rust code blocks.",
  "Create a small example showing a thiserror enum, Display, a source cause, and a function using the ? operator. Include `fn main()` to drive it. Output only Rust code wrapped in ```rust code blocks.",
  "Write a complete Rust program with `fn main()` that demonstrates pattern matching with `match` and `if let`. Include necessary imports. Output only code in ```rust code blocks.",
  "Create a Rust program with a struct, an impl block, and a trait implementation. Include `fn main()` to demonstrate usage. Output only code in ```rust code blocks.",
]

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate evaluation samples from fine-tuned model")
    ap.add_argument("--model-path", default="out/llama8b-rust-qlora", help="Path to model checkpoint")
    ap.add_argument("--output-dir", default="eval_out", help="Output directory for samples")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = ap.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    outs = []
    for p in PROMPTS:
        # Use system-style prompt to force code-only output
        system_prompt = "You are a Rust code generator. Output only valid Rust code, wrapped in ```rust code blocks. No explanations or comments outside code blocks."
        full_prompt = f"{system_prompt}\n\n{p}"
        x = tok(full_prompt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            # Greedy decoding for stability, reasonable token limit for single-file programs
            y = mdl.generate(**x, max_new_tokens=512, do_sample=False, temperature=None)
        txt = tok.decode(y[0], skip_special_tokens=True)
        # Try to extract code fence if present
        code = txt.split("```rust")
        if len(code) > 1:
            snip = code[1].split("```")[0].strip()
        else:
            # Fallback: try to find code after prompt
            snip = txt.split(p)[-1].strip() if p in txt else txt.strip()
        outs.append({"prompt": p, "gen": snip})
    output_path = os.path.join(args.output_dir, "samples.jsonl")
    with jsonlines.open(output_path, "w") as w:
        for r in outs: w.write(r)
    print(f"Generated {len(outs)} samples, saved to {output_path}")

if __name__ == "__main__":
    main()
