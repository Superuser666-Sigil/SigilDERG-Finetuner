"""
JSONL loader for sigil-pipeline prompt/gen format.

Loads JSONL files with {"prompt": "...", "gen": "..."} format and converts
them to {"text": "..."} format for training.

Uses streaming approach similar to convert_jsonl_to_parquet.py to handle
large files efficiently.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prompt_gen_jsonl(
    jsonl_path: str,
    tokenizer,
    apply_chat_template: bool = True,
    remove_metadata: bool = True,
    task_weights: dict[str, float] | None = None,
) -> Iterator[dict[str, str]]:
    """
    Load JSONL files with prompt/gen format from sigil-pipeline.

    Formats data for training by combining prompt and gen into text field.
    If apply_chat_template is True, uses the tokenizer's chat template.
    Otherwise, simply concatenates prompt and gen.

    Args:
        jsonl_path: Path to JSONL file with prompt/gen format
        tokenizer: HuggingFace tokenizer (for chat template if needed)
        apply_chat_template: Whether to apply chat template formatting
        remove_metadata: Whether to remove metadata fields (starting with _)
        task_weights: Optional per `_task_type` multipliers for oversampling/undersampling

    Yields:
        Dict with "text" key containing formatted training text
    """
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    logger.info(f"Loading prompt/gen JSONL from {jsonl_path}")

    count = 0
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
                continue

            # Validate required fields
            if "prompt" not in sample or "gen" not in sample:
                logger.warning(f"Skipping sample on line {line_num} - missing prompt/gen")
                continue

            prompt = str(sample["prompt"])
            gen = str(sample["gen"])

            task_type = sample.get("_task_type")

            # Remove metadata if requested
            if remove_metadata:
                # Keep only prompt, gen, and split (if present)
                clean_sample = {"prompt": prompt, "gen": gen}
                if "split" in sample:
                    clean_sample["split"] = sample["split"]
                sample = clean_sample

            # Format for training
            if apply_chat_template and tokenizer is not None:
                try:
                    # Use tokenizer's chat template
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": gen},
                    ]
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply chat template on line {line_num}: {e}, using simple concatenation"
                    )
                    # Fallback to simple concatenation
                    text = f"{prompt}\n\n{gen}"
            else:
                # Simple concatenation
                text = f"{prompt}\n\n{gen}"

            weight = 1.0
            if task_weights and task_type:
                weight = task_weights.get(task_type, 1.0)
            if weight <= 0:
                continue

            repeats = int(weight)
            fractional = weight - repeats

            emitted = 0

            for _ in range(repeats):
                yield {"text": text}
                emitted += 1

            if fractional > 0 and random.random() < fractional:
                yield {"text": text}
                emitted += 1

            if weight < 1.0 and emitted == 0:
                continue

            count += emitted

            if count % 10000 == 0:
                logger.info(f"Loaded {count} samples from {jsonl_path}...")

    logger.info(f"Loaded {count} total samples from {jsonl_path}")


def load_prompt_gen_jsonl_streaming(
    jsonl_path: str,
    tokenizer,
    apply_chat_template: bool = True,
    remove_metadata: bool = True,
    task_weights: dict[str, float] | None = None,
) -> Iterator[dict[str, str]]:
    """
    Streaming version of load_prompt_gen_jsonl (alias for consistency).

    This is the same as load_prompt_gen_jsonl since it already streams.
    """
    return load_prompt_gen_jsonl(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        remove_metadata=remove_metadata,
        task_weights=task_weights,
    )
