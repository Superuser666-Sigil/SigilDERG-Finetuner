"""
Rust code evaluation module for fine-tuned models.

Evaluates generated Rust code using compilation, clippy, and functional correctness tests.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
"""

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any

import jsonlines

# Optional human-eval-rust integration
try:
    from human_eval.evaluate_functional_correctness import evaluate_functional_correctness

    HUMAN_EVAL_AVAILABLE = True
except ImportError:
    HUMAN_EVAL_AVAILABLE = False
    evaluate_functional_correctness = None

# Import sandbox wrapper
try:
    from .eval_sandbox import (
        SandboxError,
        check_docker_available,
        check_firejail_available,
        run_cargo_sandboxed,
    )
except ImportError:
    try:
        from eval_sandbox import (
            SandboxError,
            check_docker_available,
            check_firejail_available,
            run_cargo_sandboxed,
        )
    except ImportError:
        # Fallback: no sandboxing available
        def run_cargo_sandboxed(*args, **kwargs):
            raise SandboxError(
                "Sandbox module not available. Install Docker or Firejail for secure evaluation."
            )

        SandboxError = Exception

        def check_docker_available():
            return False

        def check_firejail_available():
            return False


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
            subprocess.run(
                ["cargo", "new", "--quiet", "app"], cwd=td, check=True, capture_output=True
            )
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


def has_idiomatic_patterns(code: str) -> dict[str, bool]:
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


def check_functionality_coverage(code: str, prompt: str = "") -> dict[str, Any]:
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


def is_valid_sample(
    code: str,
    prompt: str = "",
    min_length: int = 20,
    min_code_lines: int = 3,
    require_main: bool = True,
    check_incomplete: bool = True,
) -> tuple[bool, str]:
    """
    Pre-filter samples before compilation to skip obviously invalid ones.

    Args:
        code: Code content to validate
        prompt: Optional prompt (for future use)
        min_length: Minimum code length in characters (default: 20)
        min_code_lines: Minimum number of non-comment code lines (default: 3)
        require_main: Whether to require 'fn main' (default: True)
        check_incomplete: Whether to check for incomplete code patterns (default: True)

    Returns:
        (is_valid, reason)
    """
    if not code or len(code.strip()) < min_length:
        return False, "too_short"

    # Check if it's mostly comments
    lines = code.split("\n")
    code_lines = [line for line in lines if line.strip() and not line.strip().startswith("//")]
    if len(code_lines) < min_code_lines:
        return False, "mostly_comments"

    # Must have fn main (for single-file programs)
    if require_main and "fn main" not in code:
        return False, "no_main"

    # Skip if it's clearly incomplete (ends mid-statement)
    if check_incomplete:
        code_stripped = code.strip()
        # Check for incomplete patterns
        incomplete_endings = ("{", "(", "[", "::", "->", "=>", ",")
        if code_stripped.endswith(incomplete_endings):
            return False, "incomplete"
        # Check for incomplete string/char literals
        if code_stripped.count('"') % 2 != 0 and not code_stripped.endswith('\\"'):
            return False, "incomplete_string"
        if code_stripped.count("'") % 2 != 0 and not code_stripped.endswith("\\'"):
            return False, "incomplete_char"
        # Check for incomplete macro calls: macros use !(, ![, or !{ and close with ), ], or }
        # We need to check if there's a ! followed by an opening delimiter that isn't closed
        i = 0
        while i < len(code_stripped) - 1:
            if code_stripped[i] == "!" and code_stripped[i + 1] in "([{":
                # Found a macro invocation, check if it's closed
                opener = code_stripped[i + 1]
                closer = ")" if opener == "(" else ("]" if opener == "[" else "}")
                # Count opening and closing delimiters from this point
                depth = 0
                for j in range(i + 1, len(code_stripped)):
                    if code_stripped[j] == opener:
                        depth += 1
                    elif code_stripped[j] == closer:
                        depth -= 1
                        if depth == 0:
                            break
                # If depth > 0, the macro is incomplete
                if depth > 0:
                    return False, "incomplete_macro"
            i += 1

    return True, "valid"


def evaluate_single_sample(
    sample: dict[str, Any],
    check_functionality: bool = True,
    sandbox_mode: str | None = None,
    compile_timeout: int = 30,
    clippy_timeout: int = 30,
    allow_network_fallback: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a single sample (designed for parallel execution).

    Args:
        sample: Sample dict with "gen" (code) and optionally "prompt"
        check_functionality: Whether to check functionality coverage
        sandbox_mode: Sandbox mode ("docker", "firejail", "none", or None for auto-detect)

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
            # Compile using sandboxed cargo command
            try:
                c1 = run_cargo_sandboxed(
                    proj,
                    [
                        "cargo",
                        "check",
                        "-q",
                        "--frozen",
                    ],  # --frozen prevents Cargo.lock writes (requires pre-generated lock file)
                    timeout=compile_timeout,
                    capture_output=True,
                    sandbox_mode=sandbox_mode,
                    allow_network_fallback=allow_network_fallback,
                )
                result["compiled"] = c1.returncode == 0
            except SandboxError as e:
                # Sandbox infrastructure error - this is a Docker/system issue, not a code compilation error
                result["error"] = f"sandbox_error: {str(e)}"
                result["error_type"] = "infrastructure_error"
                result["compiled"] = False
                # Include full error message for debugging
                result["compile_error"] = str(e)[:1000]  # Full error for infrastructure issues
                return result

            if not result["compiled"]:
                # Capture compilation error for analysis
                error_output = ""
                if hasattr(c1, "stderr") and c1.stderr:
                    error_output += c1.stderr
                if hasattr(c1, "stdout") and c1.stdout:
                    error_output += c1.stdout
                if not error_output:
                    error_output = "Compilation failed but no error output captured"
                result["compile_error"] = error_output[
                    :1000
                ]  # Increased limit for better debugging

                # Extract error type patterns for aggregation
                error_lower = error_output.lower()

                # First check for Docker/infrastructure errors (should have been caught earlier, but double-check)
                docker_error_patterns = [
                    "docker: error response from daemon",
                    "failed to create task for container",
                    "read-only file system",
                    "error mounting",
                    # Network-related errors (when --network=none but code needs dependencies)
                    "couldn't resolve host name",
                    "could not resolve host",
                    "failed to query replaced source registry",
                    "failed to download from",
                    "network is unreachable",
                ]
                if any(pattern in error_lower for pattern in docker_error_patterns):
                    # This should have been caught as SandboxError, but if it wasn't, mark it clearly
                    result["error_type"] = "infrastructure_error"
                    result["error"] = "Docker infrastructure error detected in output"
                    return result

                # Now check for actual Rust compilation errors
                if "cannot find" in error_lower and "in this scope" in error_lower:
                    result["error_type"] = "undefined_variable"
                elif "expected" in error_lower and "found" in error_lower:
                    result["error_type"] = "type_mismatch"
                elif "missing" in error_lower and ("import" in error_lower or "use" in error_lower):
                    result["error_type"] = "missing_import"
                elif "cannot borrow" in error_lower or "borrow checker" in error_lower:
                    result["error_type"] = "borrow_checker"
                elif "trait" in error_lower and (
                    "not implemented" in error_lower or "not found" in error_lower
                ):
                    result["error_type"] = "trait_error"
                elif "syntax error" in error_lower or "expected one of" in error_lower:
                    result["error_type"] = "syntax_error"
                elif "timed out" in error_lower or "timeout" in error_lower:
                    result["error_type"] = "timeout"
                else:
                    result["error_type"] = "other"
            else:
                # Clippy warnings count using sandboxed cargo command
                try:
                    c2 = run_cargo_sandboxed(
                        proj,
                        ["cargo", "clippy", "-q", "--", "-W", "clippy::all"],
                        timeout=clippy_timeout,
                        capture_output=True,
                        sandbox_mode=sandbox_mode,
                        allow_network_fallback=allow_network_fallback,
                    )
                    result["clippy_warnings"] = c2.stdout.count(": warning:")
                except SandboxError:
                    # If clippy fails in sandbox, just skip clippy warnings
                    result["clippy_warnings"] = 0
        finally:
            # Clean up temporary project
            if USE_TEMPLATE:
                shutil.rmtree(proj_parent, ignore_errors=True)
    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
        result["error_type"] = "timeout"
        result["compile_error"] = "Evaluation timed out"
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = "other"
        result["compile_error"] = f"Unexpected error: {str(e)}"

    return result


def compile_and_clippy(
    samples: list[dict[str, Any]],
    sample_n: int = 16,
    check_functionality: bool = True,
    pre_filter: bool = True,
    pre_filter_min_length: int = 20,
    pre_filter_min_code_lines: int = 3,
    pre_filter_require_main: bool = True,
    pre_filter_check_incomplete: bool = True,
    num_workers: int = None,
    return_details: bool = False,
    sandbox_mode: str | None = None,
    compile_timeout: int = 30,
    clippy_timeout: int = 30,
    allow_network_fallback: bool = True,
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Comprehensive evaluation of Rust code samples with optional parallelization.

    Args:
        samples: List of dicts with "gen" (code) and optionally "prompt"
        sample_n: Number of samples to evaluate
        check_functionality: Whether to check functionality coverage
        pre_filter: Whether to pre-filter samples before compilation
        num_workers: Number of parallel workers (None = auto, 1 = sequential)
        sandbox_mode: Sandbox mode ("docker", "firejail", "none", or None for auto-detect)

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
            is_valid, reason = is_valid_sample(
                code,
                prompt,
                min_length=pre_filter_min_length,
                min_code_lines=pre_filter_min_code_lines,
                require_main=pre_filter_require_main,
                check_incomplete=pre_filter_check_incomplete,
            )
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
            eval_func = partial(
                evaluate_single_sample,
                check_functionality=check_functionality,
                sandbox_mode=sandbox_mode,
                compile_timeout=compile_timeout,
                clippy_timeout=clippy_timeout,
                allow_network_fallback=allow_network_fallback,
            )
            results = pool.map(eval_func, valid_samples)
    else:
        # Sequential evaluation
        results = [
            evaluate_single_sample(
                s,
                check_functionality,
                sandbox_mode,
                compile_timeout=compile_timeout,
                clippy_timeout=clippy_timeout,
                allow_network_fallback=allow_network_fallback,
            )
            for s in valid_samples
        ]

    # Aggregate results
    ok_compile = sum(1 for r in results if r["compiled"])
    clippy_warns = sum(r["clippy_warnings"] for r in results)
    has_doc_count = sum(1 for r in results if r["has_doc"])
    doc_comment_count = sum(r["doc_count"] for r in results)
    idiomatic_scores = [r["idiomatic_score"] for r in results]
    functionality_scores = [r["functionality"] for r in results if r["functionality"]]

    # Aggregate error types for failed compilations
    error_types = {}
    infrastructure_error_count = 0
    for r in results:
        if not r["compiled"] and "error_type" in r:
            error_type = r.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
            if error_type == "infrastructure_error":
                infrastructure_error_count += 1

    evaluated_count = len(picks) - filtered_count
    metrics = {
        "compile_rate": ok_compile / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_clippy_warnings": clippy_warns / evaluated_count if evaluated_count > 0 else 0.0,
        "doc_comment_rate": has_doc_count / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_doc_comments": doc_comment_count / evaluated_count if evaluated_count > 0 else 0.0,
        "avg_idiomatic_score": (
            sum(idiomatic_scores) / len(idiomatic_scores) if idiomatic_scores else 0.0
        ),
        "total_samples": len(picks),
        "filtered_samples": filtered_count,
        "evaluated_samples": evaluated_count,
    }

    # Add error type breakdown if there were failures
    if error_types:
        metrics["error_types"] = error_types

    # Warn if infrastructure errors detected
    if infrastructure_error_count > 0:
        print(
            f"\n⚠️  WARNING: {infrastructure_error_count} infrastructure error(s) detected!",
            file=__import__("sys").stderr,
        )
        print(
            "   These are Docker/system issues, not code compilation errors.",
            file=__import__("sys").stderr,
        )
        print(
            "   Check Docker daemon status, disk space, and permissions.",
            file=__import__("sys").stderr,
        )
        print(
            "   Infrastructure errors are tracked separately in error_types['infrastructure_error']",
            file=__import__("sys").stderr,
        )
    if filter_reasons:
        metrics["filter_reasons"] = filter_reasons

    if functionality_scores:
        metrics["avg_functions"] = sum(f["has_functions"] for f in functionality_scores) / len(
            functionality_scores
        )
        metrics["avg_structs"] = sum(f["has_structs"] for f in functionality_scores) / len(
            functionality_scores
        )
        metrics["avg_impls"] = sum(f["has_impls"] for f in functionality_scores) / len(
            functionality_scores
        )
        # Aggregate trait counts (not just boolean)
        metrics["avg_traits"] = sum(f.get("has_traits", 0) for f in functionality_scores) / len(
            functionality_scores
        )
        # Aggregate test detection (boolean -> rate)
        metrics["test_rate"] = sum(
            1 for f in functionality_scores if f.get("has_tests", False)
        ) / len(functionality_scores)
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
    ap.add_argument(
        "--check-func",
        action="store_true",
        default=False,
        help="Enable functionality coverage checks",
    )
    ap.add_argument(
        "--no-check-func",
        dest="check_func",
        action="store_false",
        help="Disable functionality coverage checks",
    )
    ap.add_argument("--no-pre-filter", action="store_true", help="Disable pre-filtering")
    ap.add_argument(
        "--pre-filter-min-length",
        type=int,
        default=20,
        help="Minimum code length for pre-filtering (default: 20)",
    )
    ap.add_argument(
        "--pre-filter-min-lines",
        type=int,
        default=3,
        help="Minimum non-comment code lines for pre-filtering (default: 3)",
    )
    ap.add_argument(
        "--pre-filter-no-main-check",
        action="store_true",
        help="Don't require 'fn main' in pre-filtering",
    )
    ap.add_argument(
        "--pre-filter-no-incomplete-check",
        action="store_true",
        help="Don't check for incomplete code patterns",
    )
    ap.add_argument(
        "--num-workers", type=int, default=None, help="Number of parallel workers (None=auto)"
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    ap.add_argument(
        "--compile-timeout",
        type=int,
        default=30,
        help="Timeout for compilation in seconds (default: 30)",
    )
    ap.add_argument(
        "--clippy-timeout", type=int, default=30, help="Timeout for Clippy in seconds (default: 30)"
    )
    ap.add_argument(
        "--output", type=str, default=None, help="Output JSONL file path (default: stdout)"
    )
    ap.add_argument(
        "--save-errors",
        type=str,
        default=None,
        help="Save detailed error logs to JSONL file (optional)",
    )
    ap.add_argument(
        "--sandbox-mode",
        type=str,
        choices=["docker", "firejail", "none", "auto"],
        default="auto",
        help="Sandbox mode: 'docker' (recommended), 'firejail', 'none' (unsafe, local dev only), or 'auto' (auto-detect)",
    )
    ap.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Disable sandboxing (UNSAFE: only for local development with trusted code)",
    )
    ap.add_argument(
        "--use-human-eval",
        action="store_true",
        help="Use human-eval-rust for functional correctness evaluation (requires human-eval-rust package)",
    )
    args = ap.parse_args()

    path = args.path
    sample_n = args.sample_n
    check_func = args.check_func
    pre_filter = not args.no_pre_filter
    num_workers = args.num_workers
    random.seed(args.seed)

    # Pre-filter configuration
    pre_filter_min_length = args.pre_filter_min_length
    pre_filter_min_lines = args.pre_filter_min_lines
    pre_filter_require_main = not args.pre_filter_no_main_check
    pre_filter_check_incomplete = not args.pre_filter_no_incomplete_check

    # Determine sandbox mode
    if args.no_sandbox:
        sandbox_mode = "none"
        print("WARNING: Sandboxing disabled. This is UNSAFE for untrusted LLM-generated code!")
        print("         Only use --no-sandbox for local development with trusted code.")
    elif args.sandbox_mode == "auto":
        # Auto-detect: prefer Docker, fallback to Firejail, then none
        if check_docker_available():
            sandbox_mode = "docker"
            print("Using Docker sandboxing (auto-detected)")
        elif check_firejail_available():
            sandbox_mode = "firejail"
            print("Using Firejail sandboxing (auto-detected)")
        else:
            sandbox_mode = "none"
            print("WARNING: No sandboxing available (Docker/Firejail not found)")
            print("         Evaluation will run unsandboxed. Install Docker for secure evaluation.")
    else:
        sandbox_mode = args.sandbox_mode
        if sandbox_mode == "none":
            print("WARNING: Sandboxing disabled via --sandbox-mode=none")
            print("         This is UNSAFE for untrusted LLM-generated code!")
        else:
            print(f"Using {sandbox_mode} sandboxing")

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
        pre_filter_min_length=pre_filter_min_length,
        pre_filter_min_code_lines=pre_filter_min_lines,
        pre_filter_require_main=pre_filter_require_main,
        pre_filter_check_incomplete=pre_filter_check_incomplete,
        num_workers=num_workers,
        return_details=return_details,
        sandbox_mode=sandbox_mode,
        compile_timeout=args.compile_timeout,
        clippy_timeout=args.clippy_timeout,
    )

    if return_details:
        metrics, valid_samples, results = result
    else:
        metrics = result
        valid_samples, results = [], []

    # Add metadata
    metrics["seed"] = args.seed
    metrics["timestamp"] = __import__("datetime").datetime.now().isoformat()

    # Check for infrastructure errors and warn
    if "error_types" in metrics and "infrastructure_error" in metrics["error_types"]:
        infra_count = metrics["error_types"]["infrastructure_error"]
        print(
            f"\n⚠️  WARNING: {infra_count} infrastructure error(s) detected!",
            file=__import__("sys").stderr,
        )
        print(
            "   These are Docker/system issues, not code compilation errors.",
            file=__import__("sys").stderr,
        )
        print(
            "   Check Docker daemon status, disk space, and permissions.",
            file=__import__("sys").stderr,
        )
        print(
            "   Review errors.jsonl for detailed error messages.\n", file=__import__("sys").stderr
        )

    # Save detailed error logs if requested
    if args.save_errors and valid_samples and results:
        error_logs = []
        for i, (sample, result_item) in enumerate(zip(valid_samples, results)):
            if not result_item.get("compiled", False):
                error_logs.append(
                    {
                        "sample_index": i,
                        "prompt": sample.get("prompt", "")[:200],  # Truncate for readability
                        "code": sample.get("gen", "")[:500],  # Truncate code
                        "error_type": result_item.get("error_type", "unknown"),
                        "compile_error": result_item.get("compile_error", ""),
                    }
                )

        if error_logs:
            with jsonlines.open(args.save_errors, mode="w") as writer:
                for log in error_logs:
                    writer.write(log)
            print(f"Saved {len(error_logs)} error logs to {args.save_errors}")

    # Run human-eval-rust if requested
    if args.use_human_eval:
        if not HUMAN_EVAL_AVAILABLE:
            print(
                "WARNING: --use-human-eval requested but human-eval-rust not available.",
                file=__import__("sys").stderr,
            )
            print(
                "         Install with: pip install human-eval-rust", file=__import__("sys").stderr
            )
        else:
            print("\nRunning human-eval-rust functional correctness evaluation...")
            # Convert samples to human-eval format
            # HumanEval expects {"task_id": "...", "completion": "..."}
            # Our samples have {"prompt": "...", "gen": "..."}
            human_eval_samples = []
            for i, sample in enumerate(samples[:sample_n]):
                # Use prompt as task_id (or generate one)
                task_id = sample.get("task_id", f"task_{i}")
                completion = sample.get("gen", sample.get("completion", ""))
                if completion:
                    human_eval_samples.append({"task_id": task_id, "completion": completion})

            if human_eval_samples:
                # Write temporary file for human-eval
                temp_samples_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jsonl", delete=False
                )
                with jsonlines.open(temp_samples_file.name, mode="w") as writer:
                    for sample in human_eval_samples:
                        writer.write(sample)

                try:
                    # Run human-eval evaluation
                    human_eval_results = evaluate_functional_correctness(
                        sample_file=temp_samples_file.name,
                        k=[1, 10, 100],
                        n_workers=num_workers or cpu_count(),
                        timeout=args.compile_timeout,
                    )

                    # Add human-eval metrics to output
                    metrics["human_eval"] = human_eval_results
                    print(f"HumanEval results: {human_eval_results}")
                except Exception as e:
                    print(
                        f"ERROR: human-eval-rust evaluation failed: {e}",
                        file=__import__("sys").stderr,
                    )
                    import traceback

                    traceback.print_exc()
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_samples_file.name)
                    except OSError:
                        pass

    # Write to file if specified, otherwise stdout
    if args.output:
        with jsonlines.open(args.output, mode="a") as writer:
            writer.write(metrics)
        print(f"Metrics written to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
