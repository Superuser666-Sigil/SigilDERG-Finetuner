from datasets import load_dataset, Dataset, interleave_datasets
import re
import os
from functools import partial
from typing import List, Iterator, Dict, Any, Callable

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


def create_filter_function(
    min_length: int = 64,
    max_length: int = 200_000,
    exclude_tests: bool = True,
    exclude_examples: bool = False,
    exclude_benches: bool = True,
    prefer_idiomatic: bool = False,
    prefer_documented: bool = False,
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a filter function for use with datasets.filter().
    
    Returns:
        A function that takes an example dict and returns True if it should be included
    """
    def filter_fn(example: Dict[str, Any]) -> bool:
        code = example.get("content", "")
        path = example.get("path", "")
        return filter_rust_code(
            code, path,
            min_length, max_length,
            exclude_tests, exclude_examples, exclude_benches,
            prefer_idiomatic, prefer_documented,
            return_reason=False
        )
    return filter_fn


def create_tracking_filter(
    filter_fn: Callable[[Dict[str, Any]], bool],
    dataset_name: str,
    dataset_stats: Dict[str, Dict[str, int]],
    filter_reasons: Dict[str, Dict[str, int]],
    min_length: int,
    max_length: int,
    exclude_tests: bool,
    exclude_examples: bool,
    exclude_benches: bool,
    prefer_idiomatic: bool,
    prefer_documented: bool,
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a filter function that tracks statistics during filtering.
    
    Returns:
        A function that filters examples and tracks stats
    """
    def tracking_filter(example: Dict[str, Any]) -> bool:
        dataset_stats[dataset_name]["total"] += 1
        passed = filter_fn(example)
        if not passed:
            dataset_stats[dataset_name]["filtered"] += 1
            # Get filter reason for telemetry
            code = example.get("content", "")
            path = example.get("path", "")
            _, reason = filter_rust_code(
                code, path,
                min_length, max_length,
                exclude_tests, exclude_examples, exclude_benches,
                prefer_idiomatic, prefer_documented,
                return_reason=True
            )
            filter_reasons[dataset_name][reason] = filter_reasons[dataset_name].get(reason, 0) + 1
        else:
            dataset_stats[dataset_name]["passed"] += 1
        return passed
    return tracking_filter


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
    return_reason: bool = False,
) -> bool | tuple[bool, str]:
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
        return_reason: If True, return (bool, reason) tuple instead of just bool
    
    Returns:
        True if code should be included, False otherwise.
        If return_reason=True, returns (bool, reason) tuple.
    """
    # Path-based exclusions
    if VENDOR_PAT.search(path) or LOCK_PAT.search(path):
        return (False, "vendor_path") if return_reason else False
    
    if exclude_tests and (TEST_PAT.search(path) or TEST_PAT.search(code)):
        return (False, "test_file") if return_reason else False
    
    if exclude_examples and EXAMPLE_PAT.search(path):
        return (False, "example_file") if return_reason else False
    
    if exclude_benches and BENCH_PAT.search(path):
        return (False, "bench_file") if return_reason else False
    
    # Length filtering
    if len(code) < min_length:
        return (False, "too_short") if return_reason else False
    if len(code) > max_length:
        return (False, "too_long") if return_reason else False
    
    # Quality heuristics
    if prefer_idiomatic and not is_idiomatic(code):
        return (False, "not_idiomatic") if return_reason else False
    
    if prefer_documented and not has_doc_comments(code):
        return (False, "no_documentation") if return_reason else False
    
    return (True, "passed") if return_reason else True


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
    interleave_mode: str = "sequential",
    dataset_weights: Dict[str, float] | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream Rust code from one or more datasets with enhanced filtering.
    
    Args:
        dataset_names: Single dataset name or list of dataset names
        cache_dir: Directory to cache datasets (default: ~/.cache/huggingface/datasets)
        use_cache: If True, use non-streaming (cached) datasets for better throughput.
                   If False, use streaming for lower RAM usage. Note: shuffle_seed
                   requires non-streaming mode regardless of this flag.
        min_length: Minimum code length
        max_length: Maximum code length
        exclude_tests: Exclude test files
        exclude_examples: Exclude example files
        exclude_benches: Exclude benchmark files
        prefer_idiomatic: Prefer idiomatic code patterns
        prefer_documented: Prefer code with documentation
        shuffle_seed: Random seed for shuffling (None = no shuffle)
        interleave_mode: How to interleave multiple datasets. Options:
            - "sequential": Process datasets one after another (default)
            - "round_robin": Alternate between datasets
            - "weighted": Sample based on dataset_weights
        dataset_weights: Dict mapping dataset names to weights (for weighted mode)
    
    Yields:
        Dict with "text" key containing filtered Rust code
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    cache_config = {"cache_dir": cache_dir} if cache_dir else {}
    
    # Control streaming based on use_cache flag
    # If use_cache=True: non-streaming mode (dataset cached to disk, better throughput)
    # If use_cache=False: streaming mode (lower RAM usage, no disk cache)
    # Note: shuffle_seed requires non-streaming mode regardless of use_cache
    if shuffle_seed is not None:
        # Shuffling requires non-streaming mode
        streaming_mode = False
    else:
        # Respect use_cache flag: True = non-streaming (cached), False = streaming
        streaming_mode = not use_cache
    
    # Track filter statistics for telemetry (per-dataset)
    dataset_stats = {}
    filter_reasons = {}  # Track reasons per dataset
    
    # Create filter function for use with datasets.filter()
    filter_fn = create_filter_function(
        min_length, max_length,
        exclude_tests, exclude_examples, exclude_benches,
        prefer_idiomatic, prefer_documented
    )
    
    # For interleaving modes (round_robin, weighted), use HuggingFace's interleave_datasets
    # This replaces custom Python interleaving loops with optimized C++ backend
    if interleave_mode in ("round_robin", "weighted") and len(dataset_names) > 1:
        # Interleaving requires non-streaming mode (datasets must be loaded)
        if streaming_mode:
            print("Warning: Interleaving requires non-streaming mode. Switching to cached datasets.")
            streaming_mode = False
        
        # Load and filter all datasets
        filtered_datasets = []
        dataset_name_list = []  # Track dataset names in same order as filtered_datasets
        
        for dataset_name in dataset_names:
            try:
                # Load dataset
                ds = load_dataset(
                    dataset_name,
                    split="train",
                    streaming=False,  # Interleaving requires non-streaming
                    **cache_config
                )
                
                # Initialize stats tracking
                dataset_stats[dataset_name] = {"total": 0, "passed": 0, "filtered": 0}
                filter_reasons[dataset_name] = {}
                
                # Create tracking filter that captures dataset_name correctly
                tracking_filter_fn = create_tracking_filter(
                    filter_fn,
                    dataset_name,
                    dataset_stats,
                    filter_reasons,
                    min_length,
                    max_length,
                    exclude_tests,
                    exclude_examples,
                    exclude_benches,
                    prefer_idiomatic,
                    prefer_documented
                )
                
                # Apply filtering with stats tracking
                filtered_ds = ds.filter(tracking_filter_fn, desc=f"Filtering {dataset_name}")
                filtered_datasets.append(filtered_ds)
                dataset_name_list.append(dataset_name)
                
            except Exception as e:
                print(f"Warning: Failed to load dataset {dataset_name}: {e}")
                continue
        
        if not filtered_datasets:
            print("Error: No datasets could be loaded for interleaving")
            return
        
        # Set up probabilities for weighted mode
        probabilities = None
        if interleave_mode == "weighted":
            weights = dataset_weights or {name: 1.0 for name in dataset_names}
            # Create probability list matching the order of filtered_datasets
            probabilities = [weights.get(ds_name, 1.0) for ds_name in dataset_name_list]
            
            # Normalize probabilities
            if probabilities:
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                else:
                    probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        # Use HuggingFace's interleave_datasets (optimized C++ backend)
        interleaved = interleave_datasets(
            filtered_datasets,
            probabilities=probabilities,  # None for round-robin (equal weights), list for weighted
            stopping_strategy="first_exhausted",  # Stop when first dataset is exhausted
            seed=shuffle_seed
        )
        
        # Yield from interleaved dataset
        for example in interleaved:
            code = example.get("content", "")
            yield {"text": code}
        
        # Print telemetry
        for ds_name, stats in dataset_stats.items():
            if stats["total"] > 0:
                pass_rate = stats["passed"] / stats["total"] * 100
                print(f"Dataset '{ds_name}': {stats['passed']}/{stats['total']} passed "
                      f"({pass_rate:.1f}%), {stats['filtered']} filtered")
                if filter_reasons.get(ds_name):
                    reasons_str = ", ".join(f"{k}: {v}" for k, v in filter_reasons[ds_name].items())
                    print(f"  Filter reasons: {reasons_str}")
        return
    
    # Sequential mode (original behavior)
    for dataset_name in dataset_names:
        # Initialize stats for this dataset
        dataset_stats[dataset_name] = {
            "total": 0,
            "passed": 0,
            "filtered": 0,
        }
        filter_reasons[dataset_name] = {}
        try:
            # Use streaming or cached based on use_cache flag
            ds = load_dataset(
                dataset_name,
                split="train",
                streaming=streaming_mode,
                **cache_config
            )
            
            # Note: streaming_mode is already set correctly above based on use_cache and shuffle_seed
            
            if shuffle_seed is not None:
                # For large datasets, consider shuffling in batches
                import random
                random.seed(shuffle_seed)
                buffer = []
                for ex in ds:
                    buffer.append(ex)
                    dataset_stats[dataset_name]["total"] += 1
                    if len(buffer) >= 10000:  # Shuffle in chunks
                        random.shuffle(buffer)
                        for item in buffer:
                            passed, reason = filter_rust_code(
                                item.get("content", ""),
                                item.get("path", ""),
                                min_length, max_length,
                                exclude_tests, exclude_examples, exclude_benches,
                                prefer_idiomatic, prefer_documented,
                                return_reason=True
                            )
                            if passed:
                                dataset_stats[dataset_name]["passed"] += 1
                                yield {"text": item.get("content", "")}
                            else:
                                dataset_stats[dataset_name]["filtered"] += 1
                                filter_reasons[dataset_name][reason] = filter_reasons[dataset_name].get(reason, 0) + 1
                        buffer = []
                # Process remaining buffer
                random.shuffle(buffer)
                for item in buffer:
                    passed, reason = filter_rust_code(
                        item.get("content", ""),
                        item.get("path", ""),
                        min_length, max_length,
                        exclude_tests, exclude_examples, exclude_benches,
                        prefer_idiomatic, prefer_documented,
                        return_reason=True
                    )
                    if passed:
                        dataset_stats[dataset_name]["passed"] += 1
                        yield {"text": item.get("content", "")}
                    else:
                        dataset_stats[dataset_name]["filtered"] += 1
                        filter_reasons[dataset_name][reason] = filter_reasons[dataset_name].get(reason, 0) + 1
            else:
                # Standard streaming or cached iteration
                for ex in ds:
                    dataset_stats[dataset_name]["total"] += 1
                    code = ex.get("content", "")
                    path = ex.get("path", "")
                    
                    # Track filter reasons
                    passed, reason = filter_rust_code(
                        code, path,
                        min_length, max_length,
                        exclude_tests, exclude_examples, exclude_benches,
                        prefer_idiomatic, prefer_documented,
                        return_reason=True
                    )
                    if not passed:
                        dataset_stats[dataset_name]["filtered"] += 1
                        filter_reasons[dataset_name][reason] = filter_reasons[dataset_name].get(reason, 0) + 1
                        continue
                    
                    dataset_stats[dataset_name]["passed"] += 1
                    yield {"text": code}
        
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_name}: {e}")
            continue
    
    # Print per-dataset telemetry
    for ds_name, stats in dataset_stats.items():
        if stats["total"] > 0:
            pass_rate = stats["passed"] / stats["total"] * 100
            print(f"Dataset '{ds_name}': {stats['passed']}/{stats['total']} passed "
                  f"({pass_rate:.1f}%), {stats['filtered']} filtered")
            if filter_reasons.get(ds_name):
                reasons_str = ", ".join(f"{k}: {v}" for k, v in filter_reasons[ds_name].items())
                print(f"  Filter reasons: {reasons_str}")
        else:
            print(f"Dataset '{ds_name}': No samples processed")
