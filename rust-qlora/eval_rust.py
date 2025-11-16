import os, random, subprocess, tempfile, json, jsonlines, re
from typing import List, Dict, Any

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


def compile_and_clippy(
    samples: List[Dict[str, Any]],
    sample_n: int = 16,
    check_functionality: bool = True,
    pre_filter: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of Rust code samples.
    
    Args:
        samples: List of dicts with "gen" (code) and optionally "prompt"
        sample_n: Number of samples to evaluate
        check_functionality: Whether to check functionality coverage
        pre_filter: Whether to pre-filter samples before compilation
    
    Returns:
        Dictionary of evaluation metrics
    """
    random.seed(0)
    picks = random.sample(samples, min(sample_n, len(samples)))
    
    ok_compile = 0
    clippy_warns = 0
    doc_comment_count = 0
    has_doc_count = 0
    idiomatic_scores = []
    functionality_scores = []
    filtered_count = 0
    filter_reasons = {}
    
    for sample in picks:
        code = sample.get("gen", "")
        prompt = sample.get("prompt", "")
        
        # Pre-filter invalid samples
        if pre_filter:
            is_valid, reason = is_valid_sample(code, prompt)
            if not is_valid:
                filtered_count += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                continue
        
        # Documentation metrics
        if has_doc_comments(code):
            has_doc_count += 1
        doc_comment_count += count_doc_comments(code)
        
        # Idiomatic patterns
        idiomatic = has_idiomatic_patterns(code)
        idiomatic_score = sum(idiomatic.values()) / max(len(idiomatic), 1)
        idiomatic_scores.append(idiomatic_score)
        
        # Functionality coverage
        if check_functionality:
            func_coverage = check_functionality_coverage(code, prompt)
            functionality_scores.append(func_coverage)
        
        # Compilation and clippy
        with tempfile.TemporaryDirectory() as td:
            try:
                # Minimal cargo project
                subprocess.run(
                    ["cargo", "new", "--quiet", "app"],
                    cwd=td,
                    check=True,
                    capture_output=True
                )
                proj = os.path.join(td, "app")
                
                # Write code
                with open(os.path.join(proj, "src", "main.rs"), "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Compile
                c1 = subprocess.run(
                    ["cargo", "check", "-q"],
                    cwd=proj,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                ok_compile += int(c1.returncode == 0)
                
                # Clippy warnings count
                c2 = subprocess.run(
                    ["cargo", "clippy", "-q", "--", "-W", "clippy::all"],
                    cwd=proj,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                clippy_warns += c2.stdout.count(": warning:")
            except (subprocess.TimeoutExpired, Exception) as e:
                # If compilation fails or times out, count as failure
                pass
    
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
    path = sys.argv[1]
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    check_func = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
    pre_filter = sys.argv[4].lower() != "false" if len(sys.argv) > 4 else True
    
    samples = []
    with jsonlines.open(path) as r:
        for rec in r:
            samples.append(rec)
    
    metrics = compile_and_clippy(
        samples, 
        sample_n=sample_n, 
        check_functionality=check_func,
        pre_filter=pre_filter
    )
    print(json.dumps(metrics, indent=2))
