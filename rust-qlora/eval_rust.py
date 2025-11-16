import os, random, subprocess, tempfile, json, jsonlines, re, shutil
from typing import List, Dict, Any
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
            
            if result["compiled"]:
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
) -> Dict[str, Any]:
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
        return {
            "compile_rate": 0.0,
            "avg_clippy_warnings": 0.0,
            "total_samples": len(picks),
            "filtered_samples": filtered_count,
            "evaluated_samples": 0,
        }
    
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
    if filter_reasons:
        metrics["filter_reasons"] = filter_reasons
    
    if functionality_scores:
        metrics["avg_functions"] = sum(f["has_functions"] for f in functionality_scores) / len(functionality_scores)
        metrics["avg_structs"] = sum(f["has_structs"] for f in functionality_scores) / len(functionality_scores)
        metrics["avg_impls"] = sum(f["has_impls"] for f in functionality_scores) / len(functionality_scores)
        if any("prompt_keyword_match" in f for f in functionality_scores):
            metrics["avg_prompt_match"] = sum(
                f.get("prompt_keyword_match", 0) for f in functionality_scores
            ) / len(functionality_scores)
    
    return metrics


if __name__ == "__main__":
    import sys
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate Rust code samples")
    ap.add_argument("path", help="Path to samples JSONL file")
    ap.add_argument("--sample-n", type=int, default=16, help="Number of samples to evaluate")
    ap.add_argument("--check-func", action="store_true", default=True, help="Check functionality coverage")
    ap.add_argument("--no-pre-filter", action="store_true", help="Disable pre-filtering")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (None=auto)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
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
    
    metrics = compile_and_clippy(
        samples, 
        sample_n=sample_n, 
        check_functionality=check_func,
        pre_filter=pre_filter,
        num_workers=num_workers
    )
    print(json.dumps(metrics, indent=2))
