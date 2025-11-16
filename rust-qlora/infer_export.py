from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
print("Loading LoRA checkpoint...")
m = AutoPeftModelForCausalLM.from_pretrained("out/llama8b-rust-qlora", device_map="cpu")
print("Merging adapters into base...")
m = m.merge_and_unload()
t = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
m.save_pretrained("out/merged")
t.save_pretrained("out/merged")
print("Merged model saved to out/merged")
