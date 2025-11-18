import os, random, subprocess, tempfile, json, jsonlines, re, shutil
from typing import List, Dict, Any, Union, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# Note: On Windows, multiprocessing requires if __name__ == "__main__" guard
# This is already present, so parallel evaluation will work cross-platform

# Try to import template helper, fall back to creating new projects if not available
try:
    from .eval_template import create_eval_project
    USE_TEMPLATE = True
except ImportError:
    try:
        from eval_template import create_eval_project
        USE_TEMPLATE = True
    except ImportError:
        USE_TEMPLATE = False
        # Fallback: create projects manually
        def create_eval_project(code: str) -> str:
            td = tempfile.mkdtemp()
            subprocess.run(["cargo", "new", "--quiet", "app"], cwd=td, check=True, capture_output=True)
            proj = os.path.join(td, "app")
            with open(os.path.join(proj, "src", "main.rs"), "w", encoding="utf-8") as f:
                f.write(code)
            return proj

def has_doc_comments(code: str) -> bool:
    """Check if code has documentation comments."""
    return bool(re.search(r"///|//!|/\*\*", code))


def count_doc_comments(code: str) -> int:
    """Count the number of documentation comments."""
    return len(re.findall(r"///|//!|/\*\*", code))


def has_idiomatic_patterns(code: str) -> Dict[str, bool]:
    """Check for idiomatic Rust patterns."""
    patterns = {
        "result_handling": bool(re.search(r"\bResult<[^>]+>", code)),
        "option_handling": bool(re.search(r"\bOption<[^>]+>", code)),
        "error_propagation": bool(re.search(r"\?[^?]", code)),  # ? operator
        "iterator_chains": bool(re.search(r"\.(map|filter|fold|collect)\([^)]*\)", code)),
        "derive_macros": bool(re.search(r"#\[derive\([^)]+\)\]", code)),
        "trait_impls": bool(re.search(r"impl\s+\w+\s+for\s+\w+", code)),
        "match_expressions": bool(re.search(r"\bmatch\s+\w+\s*\{", code)),
        "pattern_matching": bool(re.search(r"if\s+let\s+|while\s+let\s+", code)),
    }
    return patterns


def check_functionality_coverage(code: str, prompt: str = "") -> Dict[str, Any]:
    """
    Check if generated code covers requested functionality.
    This is a heuristic-based check.
    """
    coverage = {
        "has_main": bool(re.search(r"fn\s+main\s*\(", code)),
        "has_functions": len(re.findall(r"fn\s+\w+\s*\(", code)),
        "has_structs": len(re.findall(r"struct\s+\w+", code)),
        "has_impls": len(re.findall(r"impl\s+", code)),
        "has_traits": len(re.findall(r"trait\s+\w+", code)),
        "has_tests": bool(re.search(r"#\[test\]|#\[cfg\(test\)\]", code)),
    }
    
    # Check if prompt keywords are present in code
    if prompt:
        prompt_keywords = set(re.findall(r"\b\w+\b", prompt.lower()))
        code_lower = code.lower()
        matched_keywords = sum(1 for kw in prompt_keywords if kw in code_lower and len(kw) > 3)
        coverage["prompt_keyword_match"] = matched_keywords / max(len(prompt_keywords), 1)
    
    return coverage


def is_valid_sample(code: str, prompt: str = "") -> tuple[bool, str]:
    """
    Pre-filter samples before compilation to skip obviously invalid ones.
    
    Returns:
        (is_valid, reason)
    """
    if not code or len(code.strip()) < 20:
        return False, "too_short"
    
    # Check if it's mostly comments
    lines = code.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]
    if len(code_lines) < 3:
        return False, "mostly_comments"
    
    # Must have fn main (for single-file programs)
    if "fn main" not in code:
        return False, "no_main"
    
    # Skip if it's clearly incomplete (ends mid-statement)
    if code.strip().endswith(('{', '(', '[', '::', '->', '=>')):
        return False, "incomplete"
    
    return True, "valid"


def evaluate_single_sample(sample: Dict[str, Any], check_functionality: bool = True) -> Dict[str, Any]:
    """
    Evaluate a single sample (designed for parallel execution).
    
    Returns:
        Dict with metrics for this sample
    """
    code = sample.get("gen", "")
    prompt = sample.get("prompt", "")
    
    result = {
        "compiled": False,
        "clippy_warnings": 0,
        "has_doc": False,
        "doc_count": 0,
        "idiomatic_score": 0.0,
        "functionality": {},
        "error": None,
    }
    
    # Documentation metrics
    result["has_doc"] = has_doc_comments(code)
    result["doc_count"] = count_doc_comments(code)
    
    # Idiomatic patterns
    idiomatic = has_idiomatic_patterns(code)
    result["idiomatic_score"] = sum(idiomatic.values()) / max(len(idiomatic), 1)
    
    # Functionality coverage
    if check_functionality:
        result["functionality"] = check_functionality_coverage(code, prompt)
    
    # Compilation and clippy
    try:
        # Use template project for faster evaluation (avoids cargo new overhead)
        proj = create_eval_project(code)
        proj_parent = os.path.dirname(proj)
        
        try:
            # Compile
            c1 = subprocess.run(
                ["cargo", "check", "-q"],
                cwd=proj,
                capture_output=True,
                text=True,
                timeout=30
            )
            result["compiled"] = (c1.returncode == 0)
            
            if not result["compiled"]:
                # Capture compilation error for analysis
                error_output = c1.stderr + c1.stdout
                result["compile_error"] = error_output[:500]  # Limit size
                
                # Extract error type patterns for aggregation
                error_lower = error_output.lower()
                if "cannot find" in error_lower and "in this scope" in error_lower:
                    result["error_type"] = "undefined_variable"
                elif "expected" in error_lower and "found" in error_lower:
                    result["error_type"] = "type_mismatch"
                elif "missing" in error_lower and ("import" in error_lower or "use" in error_lower):
                    result["error_type"] = "missing_import"
                elif "cannot borrow" in error_lower or "borrow checker" in error_lower:
                    result["error_type"] = "borrow_checker"
                elif "trait" in error_lower and ("not implemented" in error_lower or "not found" in error_lower):
                    result["error_type"] = "trait_error"
                elif "syntax error" in error_lower or "expected one of" in error_lower:
                    result["error_type"] = "syntax_error"
                else:
                    result["error_type"] = "other"
            else:
                # Clippy warnings count
                c2 = subprocess.run(
                    ["cargo", "clippy", "-q", "--", "-W", "clippy::all"],
                    cwd=proj,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                result["clippy_warnings"] = c2.stdout.count(": warning:")
        finally:
            # Clean up temporary project
            if USE_TEMPLATE:
                shutil.rmtree(proj_parent, ignore_errors=True)
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def compile_and_clippy(
    samples: List[Dict[str, Any]],
    sample_n: int = 16,
    check_functionality: bool = True,
    pre_filter: bool = True,
    num_workers: int = None,
    return_details: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Comprehensive evaluation of Rust code samples with optional parallelization.
    
    Args:
        samples: List of dicts with "gen" (code) and optionally "prompt"
        sample_n: Number of samples to evaluate
        check_functionality: Whether to check functionality coverage
        pre_filter: Whether to pre-filter samples before compilation
        num_workers: Number of parallel workers (None = auto, 1 = sequential)
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Seed should be set before calling this function
    picks = random.sample(samples, min(sample_n, len(samples)))
    
    # Pre-filter invalid samples
    filtered_count = 0
    filter_reasons = {}
    valid_samples = []
    
    for sample in picks:
        code = sample.get("gen", "")
        prompt = sample.get("prompt", "")
        
        if pre_filter:
            is_valid, reason = is_valid_sample(code, prompt)
            if not is_valid:
                filtered_count += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                continue
        
        valid_samples.append(sample)
    
    if not valid_samples:
        empty_metrics = {
            "compile_rate": 0.0,
            "avg_clippy_warnings": 0.0,
            "total_samples": len(picks),
            "filtered_samples": filtered_count,
            "evaluated_samples": 0,
        }
        if return_details:
            return empty_metrics, [], []
        return empty_metrics
    
    # Evaluate samples (parallel or sequential)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    if num_workers > 1 and len(valid_samples) > 1:
        # Parallel evaluation
        with Pool(processes=num_workers) as pool:
            eval_func = partial(evaluate_single_sample, check_functionality=check_functionality)
            results = pool.map(eval_func, valid_samples)
    else:
        # Sequential evaluation
        results = [evaluate_single_sample(s, check_functionality) for s in valid_samples]
    
    # Aggregate results
    ok_compile = sum(1 for r in results if r["compiled"])
    clippy_warns = sum(r["clippy_warnings"] for r in results)
    has_doc_count = sum(1 for r in results if r["has_doc"])
    doc_comment_count = sum(r["doc_count"] for r in results)
    idiomatic_scores = [r["idiomatic_score"] for r in results]
    functionality_scores = [r["functionality"] for r in results if r["functionality"]]
    
    # Aggregate error types for failed compilations
    error_types = {}
    for r in results:
        if not r["compiled"] and "error_type" in r:
            error_type = r.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    evaluated_count = len(picks) - filtered_count
    metrics = {
        "compile_rate": ok_compile / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_clippy_warnings": clippy_warns / evaluated_count if evaluated_count > 0 else 0.0,
        "doc_comment_rate": has_doc_count / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_doc_comments": doc_comment_count / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_idiomatic_score": sum(idiomatic_scores) / len(idiomatic_scores) if idiomatic_scores else 0.0,
        "total_samples": len(picks),
        "filtered_samples": filtered_count,
        "evaluated_samples": evaluated_count,
    }
    
    # Add error type breakdown if there were failures
    if error_types:
        metrics["error_types"] = error_types
    if filter_reasons:
        metrics["filter_reasons"] = filter_reasons
    
    if functionality_scores:
        metrics["avg_functions"] = sum(f["has_functions"] for f in functionality_scores) / len(functionality_scores)
        metrics["avg_structs"] = sum(f["has_structs"] for f in functionality_scores) / len(functionality_scores)
        metrics["avg_impls"] = sum(f["has_impls"] for f in functionality_scores) / len(functionality_scores)
        # Aggregate trait counts (not just boolean)
        metrics["avg_traits"] = sum(f.get("has_traits", 0) for f in functionality_scores) / len(functionality_scores)
        # Aggregate test detection (boolean -> rate)
        metrics["test_rate"] = sum(1 for f in functionality_scores if f.get("has_tests", False)) / len(functionality_scores)
        # Aggregate prompt keyword match scores
        if any("prompt_keyword_match" in f for f in functionality_scores):
            metrics["avg_prompt_match"] = sum(
                f.get("prompt_keyword_match", 0) for f in functionality_scores
            ) / len(functionality_scores)
        else:
            metrics["avg_prompt_match"] = 0.0
    
    if return_details:
        return metrics, valid_samples, results
    return metrics


def main() -> None:
    """CLI entry point for the Rust evaluation script."""
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate Rust code samples")
    ap.add_argument("path", help="Path to samples JSONL file")
    ap.add_argument("--sample-n", type=int, default=16, help="Number of samples to evaluate")
    ap.add_argument("--check-func", action="store_true", default=False, help="Enable functionality coverage checks")
    ap.add_argument("--no-check-func", dest="check_func", action="store_false", help="Disable functionality coverage checks")
    ap.add_argument("--no-pre-filter", action="store_true", help="Disable pre-filtering")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (None=auto)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    ap.add_argument("--output", type=str, default=None, help="Output JSONL file path (default: stdout)")
    ap.add_argument("--save-errors", type=str, default=None, help="Save detailed error logs to JSONL file (optional)")
    args = ap.parse_args()

    path = args.path
    sample_n = args.sample_n
    check_func = args.check_func
    pre_filter = not args.no_pre_filter
    num_workers = args.num_workers
    random.seed(args.seed)

    samples = []
    with jsonlines.open(path) as r:
        for rec in r:
            samples.append(rec)

    # Get metrics and optionally detailed results for error logging
    return_details = args.save_errors is not None
    result = compile_and_clippy(
        samples,
        sample_n=sample_n,
        check_functionality=check_func,
        pre_filter=pre_filter,
        num_workers=num_workers,
        return_details=return_details
    )
    
    if return_details:
        metrics, valid_samples, results = result
    else:
        metrics = result
        valid_samples, results = [], []

    # Add metadata
    metrics["seed"] = args.seed
    metrics["timestamp"] = __import__("datetime").datetime.now().isoformat()

    # Save detailed error logs if requested
    if args.save_errors and valid_samples and results:
        error_logs = []
        for i, (sample, result_item) in enumerate(zip(valid_samples, results)):
            if not result_item.get("compiled", False):
                error_logs.append({
                    "sample_index": i,
                    "prompt": sample.get("prompt", "")[:200],  # Truncate for readability
                    "code": sample.get("gen", "")[:500],  # Truncate code
                    "error_type": result_item.get("error_type", "unknown"),
                    "compile_error": result_item.get("compile_error", ""),
                })
        
        if error_logs:
            with jsonlines.open(args.save_errors, mode="w") as writer:
                for log in error_logs:
                    writer.write(log)
            print(f"Saved {len(error_logs)} error logs to {args.save_errors}")

    # Write to file if specified, otherwise stdout
    if args.output:
        with jsonlines.open(args.output, mode="a") as writer:
            writer.write(metrics)
        print(f"Metrics written to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
