from datasets import load_dataset, Dataset
import re
import os
from typing import List, Iterator, Dict, Any

VENDOR_PAT = re.compile(r"/(target|node_modules|vendor|__pycache__)/")
LOCK_PAT = re.compile(r"(Cargo\.lock)$")
TEST_PAT = re.compile(r"/tests?/|#[cfg\(test\)]")
BENCH_PAT = re.compile(r"/benches?/|#[cfg\(bench\)]")
EXAMPLE_PAT = re.compile(r"/examples?/")

# Idiomatic patterns to prioritize
IDIOMATIC_PATTERNS = [
    re.compile(r"\bResult<[^>]+>\s*\{"),  # Result handling
    re.compile(r"\bOption<[^>]+>\s*\{"),   # Option handling
    re.compile(r"\b\.unwrap_or\(|\.expect\(|\.map\(|\.and_then\("),  # Common methods
    re.compile(r"#\[derive\([^)]+\)\]"),  # Derive macros
    re.compile(r"impl\s+\w+\s+for\s+\w+"),  # Trait implementations
    re.compile(r"pub\s+(fn|struct|enum|trait|mod)\s+"),  # Public API
]

# Patterns indicating lower quality
LOW_QUALITY_PATTERNS = [
    re.compile(r"TODO|FIXME|XXX|HACK", re.IGNORECASE),
    re.compile(r"println!\s*\(|dbg!\s*\("),  # Debug prints
    re.compile(r"unsafe\s+\{"),  # Unsafe blocks (less idiomatic)
    re.compile(r"#\[allow\([^)]+\)\]"),  # Suppressed warnings
]

def is_idiomatic(code: str) -> bool:
    """Check if code contains idiomatic Rust patterns."""
    if not code:
        return False
    idiomatic_count = sum(1 for pat in IDIOMATIC_PATTERNS if pat.search(code))
    low_quality_count = sum(1 for pat in LOW_QUALITY_PATTERNS if pat.search(code))
    # Prefer code with more idiomatic patterns and fewer low-quality markers
    return idiomatic_count > 0 and (idiomatic_count >= low_quality_count * 2)


def has_doc_comments(code: str) -> bool:
    """Check if code has documentation comments."""
    return bool(re.search(r"///|//!|/\*\*", code))


def filter_rust_code(
    code: str,
    path: str = "",
    min_length: int = 64,
    max_length: int = 200_000,
    exclude_tests: bool = True,
    exclude_examples: bool = False,
    exclude_benches: bool = True,
    prefer_idiomatic: bool = False,
    prefer_documented: bool = False,
) -> bool:
    """
    Filter Rust code based on various criteria.
    
    Args:
        code: The code content to filter
        path: File path (for path-based filtering)
        min_length: Minimum code length in characters
        max_length: Maximum code length in characters
        exclude_tests: Exclude test files
        exclude_examples: Exclude example files
        exclude_benches: Exclude benchmark files
        prefer_idiomatic: Prefer code with idiomatic patterns
        prefer_documented: Prefer code with documentation comments
    
    Returns:
        True if code should be included, False otherwise
    """
    # Path-based exclusions
    if VENDOR_PAT.search(path) or LOCK_PAT.search(path):
        return False
    
    if exclude_tests and (TEST_PAT.search(path) or TEST_PAT.search(code)):
        return False
    
    if exclude_examples and EXAMPLE_PAT.search(path):
        return False
    
    if exclude_benches and BENCH_PAT.search(path):
        return False
    
    # Length filtering
    if not (min_length <= len(code) <= max_length):
        return False
    
    # Quality heuristics
    if prefer_idiomatic and not is_idiomatic(code):
        return False
    
    if prefer_documented and not has_doc_comments(code):
        return False
    
    return True


def stream_rust(
    dataset_names: str | List[str],
    cache_dir: str | None = None,
    use_cache: bool = True,
    min_length: int = 64,
    max_length: int = 200_000,
    exclude_tests: bool = True,
    exclude_examples: bool = False,
    exclude_benches: bool = True,
    prefer_idiomatic: bool = False,
    prefer_documented: bool = False,
    shuffle_seed: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream Rust code from one or more datasets with enhanced filtering.
    
    Args:
        dataset_names: Single dataset name or list of dataset names
        cache_dir: Directory to cache datasets (default: ~/.cache/huggingface/datasets)
        use_cache: Whether to use cached datasets
        min_length: Minimum code length
        max_length: Maximum code length
        exclude_tests: Exclude test files
        exclude_examples: Exclude example files
        exclude_benches: Exclude benchmark files
        prefer_idiomatic: Prefer idiomatic code patterns
        prefer_documented: Prefer code with documentation
        shuffle_seed: Random seed for shuffling (None = no shuffle)
    
    Yields:
        Dict with "text" key containing filtered Rust code
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    cache_config = {"cache_dir": cache_dir} if cache_dir else {}
    
    for dataset_name in dataset_names:
        try:
            # Use streaming for memory efficiency, but allow caching
            ds = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                **cache_config
            )
            
            # If shuffle requested, convert to non-streaming temporarily
            # Note: This loads into memory, use with caution
            if shuffle_seed is not None:
                # For large datasets, consider shuffling in batches
                import random
                random.seed(shuffle_seed)
                buffer = []
                for ex in ds:
                    buffer.append(ex)
                    if len(buffer) >= 10000:  # Shuffle in chunks
                        random.shuffle(buffer)
                        for item in buffer:
                            if filter_rust_code(
                                item.get("content", ""),
                                item.get("path", ""),
                                min_length, max_length,
                                exclude_tests, exclude_examples, exclude_benches,
                                prefer_idiomatic, prefer_documented
                            ):
                                yield {"text": item.get("content", "")}
                        buffer = []
                # Process remaining buffer
                random.shuffle(buffer)
                for item in buffer:
                    if filter_rust_code(
                        item.get("content", ""),
                        item.get("path", ""),
                        min_length, max_length,
                        exclude_tests, exclude_examples, exclude_benches,
                        prefer_idiomatic, prefer_documented
                    ):
                        yield {"text": item.get("content", "")}
            else:
                # Standard streaming
                for ex in ds:
                    if filter_rust_code(
                        ex.get("content", ""),
                        ex.get("path", ""),
                        min_length, max_length,
                        exclude_tests, exclude_examples, exclude_benches,
                        prefer_idiomatic, prefer_documented
                    ):
                        yield {"text": ex.get("content", "")}
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_name}: {e}")
            continue
