from datasets import load_dataset
import re

VENDOR_PAT = re.compile(r"/(target|node_modules|vendor|__pycache__)/")
LOCK_PAT = re.compile(r"(Cargo\.lock)$")

def stream_rust(dataset_name: str):
    # Streaming keeps RAM low and starts instantly
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for ex in ds:
        path = (ex.get("path") or "")
        if VENDOR_PAT.search(path) or LOCK_PAT.search(path):
            continue
        code = ex.get("content") or ""
        if not (64 <= len(code) <= 200_000):
            continue
        yield {"text": code}
