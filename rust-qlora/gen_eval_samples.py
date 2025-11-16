import os, random, jsonlines, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
  "Write a minimal Rust program that uses anyhow for error handling and returns Result<()> from main.",
  "Example: iterators with map/filter to compute a sum in Rust.",
  "Define a newtype for [u8;32] and implement FromStr and Display.",
  "Show a thiserror enum with Display and a source cause; use the ? operator in a function."
]

def main():
    random.seed(0)
    os.makedirs("eval_out", exist_ok=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained("out/llama8b-rust-qlora", device_map="auto", torch_dtype=torch.bfloat16)
    outs = []
    for p in PROMPTS:
        x = tok(p, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            y = mdl.generate(**x, max_new_tokens=220, do_sample=False)
        txt = tok.decode(y[0], skip_special_tokens=True)
        # Try to extract code fence if present
        code = txt.split("```rust")
        snip = code[1].split("```")[0] if len(code)>1 else txt
        outs.append({"gen": snip})
    with jsonlines.open("eval_out/samples.jsonl","w") as w:
        for r in outs: w.write(r)

if __name__ == "__main__":
    main()
