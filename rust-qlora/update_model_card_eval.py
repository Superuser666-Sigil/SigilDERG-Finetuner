#!/usr/bin/env python3
"""
Update model card README.md with evaluation results.

This script reads evaluation metrics from eval_out/metrics.jsonl and updates
the model-index section in the checkpoint README.md with actual evaluation results.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.7.2
"""

import os
import json
import yaml
import argparse
from pathlib import Path


def load_latest_metrics(metrics_file):
    """Load the latest evaluation metrics from JSONL file."""
    if not os.path.exists(metrics_file):
        return None
    
    metrics = None
    with open(metrics_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines
            try:
                entry = json.loads(line)
                metrics = entry  # Keep the last valid entry
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid metrics line ({e})")
                continue
    
    return metrics


def update_model_card_readme(readme_path, metrics, repo_id=None, checkpoint_name=None):
    """Update README.md with evaluation results."""
    if not os.path.exists(readme_path):
        print(f"Error: README.md not found at {readme_path}")
        return False
    
    # Read existing README
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract YAML front matter
    if not content.startswith("---"):
        print("Error: README.md doesn't have YAML front matter")
        return False
    
    # Split YAML and markdown
    parts = content.split("---", 2)
    if len(parts) < 3:
        print("Error: Invalid YAML front matter format")
        return False
    
    yaml_content = parts[1].strip()
    markdown_content = parts[2]
    
    # Parse YAML
    try:
        metadata = yaml.safe_load(yaml_content)
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        return False
    
    # Update model-index with evaluation results
    if "model-index" not in metadata:
        metadata["model-index"] = []
    
    if not metadata["model-index"]:
        # Create model-index structure if it doesn't exist
        metadata["model-index"] = [{
            "name": "rust-code-model",
            "results": []
        }]
    
    # Update or add evaluation results
    model_index = metadata["model-index"][0]
    if "results" not in model_index:
        model_index["results"] = []
    
    # Find or create rust-code-evaluation result entry
    rust_eval_result = None
    for result in model_index["results"]:
        if result.get("dataset", {}).get("name") == "rust-code-evaluation":
            rust_eval_result = result
            break
    
    if not rust_eval_result:
        rust_eval_result = {
            "task": {
                "type": "text-generation"
            },
            "dataset": {
                "name": "rust-code-evaluation",
                "type": "code-generation"
            },
            "metrics": []
        }
        model_index["results"].append(rust_eval_result)
    
    # Update metrics with actual values
    metrics_map = {
        "compile_rate": ("Compilation Rate", "compilation_rate"),
        "avg_clippy_warnings": ("Clippy Warnings (avg)", "clippy_warnings"),
        "avg_idiomatic_score": ("Idiomatic Score", "idiomatic_score"),
        "doc_comment_rate": ("Documentation Rate", "doc_comment_rate"),
        "avg_functions": ("Avg Functions", "avg_functions"),
        "avg_structs": ("Avg Structs", "avg_structs"),
        "avg_traits": ("Avg Traits", "avg_traits"),
        "test_rate": ("Test Rate", "test_rate"),
        "avg_prompt_match": ("Prompt Match Score", "prompt_match"),
    }
    
    # Update or add metrics
    existing_metrics = {m["type"]: m for m in rust_eval_result["metrics"]}
    
    for key, (name, metric_type) in metrics_map.items():
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            if isinstance(value, float):
                # Round to reasonable precision
                if value < 1.0:
                    value = round(value, 4)
                else:
                    value = round(value, 2)
            
            if metric_type in existing_metrics:
                existing_metrics[metric_type]["value"] = value
                existing_metrics[metric_type]["name"] = name
            else:
                rust_eval_result["metrics"].append({
                    "name": name,
                    "type": metric_type,
                    "value": value
                })
    
    # Update the metrics list
    rust_eval_result["metrics"] = list(existing_metrics.values()) + [
        m for m in rust_eval_result["metrics"]
        if m["type"] not in existing_metrics
    ]
    
    # Add source if available
    if "timestamp" in metrics:
        rust_eval_result["source"] = {
            "name": "SigilDERG Evaluation",
            "url": "https://github.com/Superuser666-Sigil/SigilDERG-Finetuner"
        }
    
    # Update markdown content with evaluation results and links
    updated_markdown = update_evaluation_section(markdown_content, metrics, repo_id, checkpoint_name)
    
    # Reconstruct README with updated YAML and markdown
    updated_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)
    updated_content = f"---\n{updated_yaml}---{updated_markdown}"
    
    # Write back
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    return True


def update_evaluation_section(markdown_content, metrics, repo_id=None, checkpoint_name=None):
    """Update the Evaluation Results section in markdown with actual results and links."""
    import re
    
    if not metrics:
        # No metrics available, keep placeholder
        return markdown_content
    
    # Extract checkpoint step from checkpoint_name if available
    checkpoint_step = "N/A"
    if checkpoint_name and checkpoint_name.startswith("checkpoint-"):
        try:
            checkpoint_step = checkpoint_name.split("-")[1]
        except (ValueError, IndexError):
            pass
    
    # Build evaluation results summary matching the new template format
    eval_summary = []
    eval_summary.append("## Evaluation Results\n")
    eval_summary.append("\nAll evaluation here is based on **automatic Rust-focused checks** (compile, `clippy`, idiomatic heuristics, doc comments, prompt adherence) over a small but structured evaluation set.\n")
    
    # Add aggregate metrics
    eval_summary.append(f"### Aggregate Metrics (checkpoint {checkpoint_step}, {metrics.get('evaluated_samples', metrics.get('total_samples', 'N/A'))} samples)\n")
    
    if "compile_rate" in metrics and metrics["compile_rate"] is not None:
        compile_rate = metrics['compile_rate']
        eval_summary.append(f"- **Compilation Rate**: {compile_rate:.2%}")
    else:
        eval_summary.append("- **Compilation Rate**: Evaluation pending")
    
    if "avg_clippy_warnings" in metrics and metrics["avg_clippy_warnings"] is not None:
        eval_summary.append(f"- **Average Clippy Warnings**: {metrics['avg_clippy_warnings']:.2f}")
    else:
        eval_summary.append("- **Average Clippy Warnings**: Evaluation pending")
    
    if "avg_idiomatic_score" in metrics and metrics["avg_idiomatic_score"] is not None:
        eval_summary.append(f"- **Idiomatic Score**: {metrics['avg_idiomatic_score']:.4f}")
    else:
        eval_summary.append("- **Idiomatic Score**: Evaluation pending")
    
    if "doc_comment_rate" in metrics and metrics["doc_comment_rate"] is not None:
        eval_summary.append(f"- **Documentation Rate**: {metrics['doc_comment_rate']:.2%}")
    else:
        eval_summary.append("- **Documentation Rate**: Evaluation pending")
    
    if "test_rate" in metrics and metrics["test_rate"] is not None:
        eval_summary.append(f"- **Test Rate**: {metrics['test_rate']:.2%}")
    else:
        eval_summary.append("- **Test Rate**: Evaluation pending")
    
    # Add functionality coverage
    eval_summary.append("\n### Functionality Coverage (approximate averages)\n")
    
    if "avg_functions" in metrics and metrics["avg_functions"] is not None:
        eval_summary.append(f"- **Average Functions**: {metrics['avg_functions']:.2f}")
    else:
        eval_summary.append("- **Average Functions**: Evaluation pending")
    
    if "avg_structs" in metrics and metrics["avg_structs"] is not None:
        eval_summary.append(f"- **Average Structs**: {metrics['avg_structs']:.2f}")
    else:
        eval_summary.append("- **Average Structs**: Evaluation pending")
    
    if "avg_traits" in metrics and metrics["avg_traits"] is not None:
        eval_summary.append(f"- **Average Traits**: {metrics['avg_traits']:.2f}")
    else:
        eval_summary.append("- **Average Traits**: Evaluation pending")
    
    if "avg_impls" in metrics and metrics["avg_impls"] is not None:
        eval_summary.append(f"- **Average Impls**: {metrics['avg_impls']:.2f}")
    else:
        eval_summary.append("- **Average Impls**: Evaluation pending")
    
    # Add evaluation artifacts section
    eval_summary.append("\n### Evaluation Artifacts\n")
    eval_summary.append("- **Full metrics (JSONL)** – per-sample evaluation:")
    eval_summary.append("  - `metrics.jsonl` – compilation success, clippy warnings, idiomatic scores, doc detection, and structural stats")
    eval_summary.append("- **Error logs (JSONL)** – compiler and runtime errors:")
    eval_summary.append("  - `errors.jsonl` – rustc diagnostics, clippy output, and runtime error messages")
    eval_summary.append("")
    
    # Add links to detailed evaluation files if repo_id and checkpoint_name are provided
    if repo_id and checkpoint_name:
        repo_url = f"https://huggingface.co/{repo_id}"
        eval_summary.append("(Replace these with your actual Hugging Face links as needed:)")
        eval_summary.append("")
        eval_summary.append(f"- [Metrics (JSONL)]({repo_url}/blob/main/{checkpoint_name}/metrics.jsonl)")
        eval_summary.append(f"- [Error Logs (JSONL)]({repo_url}/blob/main/{checkpoint_name}/errors.jsonl)")
    elif checkpoint_name:
        # Local links if no repo_id
        eval_summary.append("(Local evaluation files:)")
        eval_summary.append("")
        eval_summary.append(f"- Metrics: `{checkpoint_name}/metrics.jsonl`")
        eval_summary.append(f"- Errors: `{checkpoint_name}/errors.jsonl`")
    else:
        eval_summary.append("(Replace these with your actual Hugging Face links as needed:)")
        eval_summary.append("")
        eval_summary.append("- [Metrics (JSONL)](https://huggingface.co/Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora/blob/main/checkpoint-{checkpoint_step}/metrics.jsonl)")
        eval_summary.append("- [Error Logs (JSONL)](https://huggingface.co/Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora/blob/main/checkpoint-{checkpoint_step}/errors.jsonl)")
    
    eval_summary.append("")
    
    # Add timestamp if available
    if "timestamp" in metrics:
        eval_summary.append(f"*Evaluation completed: {metrics['timestamp']}*")
    else:
        eval_summary.append("*Evaluation results will be updated when available.*")
    
    eval_summary_text = "\n".join(eval_summary)
    
    # Replace the placeholder evaluation section
    # Pattern to match the "## Evaluation Results" section until the next ## or end of file
    pattern = r"## Evaluation Results.*?(?=\n## |\Z)"
    
    if re.search(pattern, markdown_content, re.DOTALL):
        # Replace existing section
        updated_markdown = re.sub(pattern, eval_summary_text, markdown_content, flags=re.DOTALL)
    else:
        # Append if we can't find the section
        updated_markdown = markdown_content + "\n\n" + eval_summary_text
    
    return updated_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Update model card README.md with evaluation results"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to checkpoint directory (e.g., out/llama8b-rust-qlora-phase1/checkpoint-1000)"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="eval_out/metrics.jsonl",
        help="Path to evaluation metrics JSONL file (default: eval_out/metrics.jsonl)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID for generating links to evaluation files (optional)"
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint name (e.g., checkpoint-1000) for generating links (optional, auto-detected if not provided)"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir)
    readme_path = checkpoint_path / "README.md"
    
    if not readme_path.exists():
        print(f"Error: README.md not found in {checkpoint_path}")
        return 1
    
    # Auto-detect checkpoint name if not provided
    checkpoint_name = args.checkpoint_name
    if not checkpoint_name:
        checkpoint_name = checkpoint_path.name
        if not checkpoint_name.startswith("checkpoint-"):
            # Try parent directory
            checkpoint_name = checkpoint_path.parent.name
    
    # Load latest metrics
    print(f"Loading metrics from {args.metrics_file}...")
    metrics = load_latest_metrics(args.metrics_file)
    
    if not metrics:
        print(f"Warning: No metrics found in {args.metrics_file}")
        print("Model card will be updated with empty evaluation results structure.")
        metrics = {}
    
    # Update README
    print(f"Updating {readme_path}...")
    if update_model_card_readme(str(readme_path), metrics, args.repo_id, checkpoint_name):
        print("✓ Successfully updated model card with evaluation results")
        
        # Show updated metrics
        if metrics:
            print("\nUpdated metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key in [
                    "compile_rate", "avg_clippy_warnings", "avg_idiomatic_score",
                    "doc_comment_rate", "test_rate", "avg_prompt_match"
                ]:
                    print(f"  {key}: {value}")
        return 0
    else:
        print("✗ Failed to update model card")
        return 1


if __name__ == "__main__":
    exit(main())

