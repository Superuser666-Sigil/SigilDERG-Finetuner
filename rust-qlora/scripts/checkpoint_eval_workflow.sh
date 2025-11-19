#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/checkpoint_eval_workflow.sh [options]

Automates the manual checkpoint workflow:
  1. Inspect checkpoint contents
  2. Generate evaluation samples
  3. Run eval_rust.py (with fallback to sigilderg-eval)
  4. Refresh checkpoint README (using new model card template)
  5. Inject latest evaluation metrics
  6. Upload checkpoints, model cards, and metrics to HuggingFace subdirectories
  7. Update root README with latest checkpoint (when using --all-checkpoints)

Options:
  --checkpoint PATH      Path to checkpoint directory (default: latest under --model-path)
  --model-path PATH      Training output directory to search for checkpoints
                         (default: out/llama8b-rust-qlora-phase1)
  --all-checkpoints      Process all checkpoints in --model-path sequentially
  --sample-n N           Number of samples to evaluate (default: 64)
  --repo-id ID           HuggingFace repo id (default: Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora)
  --hf-token TOKEN       HuggingFace token (defaults to $HF_TOKEN if set)
  --config FILE          Override training config used for model card metadata
  --metrics-file FILE    Metrics JSONL path (default: eval_out/metrics.jsonl)
                         When using --all-checkpoints, each checkpoint uses its own file
  --errors-file FILE     Error log path (default: eval_out/errors.jsonl)
                         When using --all-checkpoints, each checkpoint uses its own file
  --skip-inspect         Skip inspect_checkpoint step
  --reuse-samples        Skip generation step and reuse existing eval_out/samples.jsonl
  --no-upload            Skip HuggingFace uploads (useful for local-only evaluation)
  -h, --help             Show this help text
EOF
}

MODEL_PATH="out/llama8b-rust-qlora-phase1"
CHECKPOINT=""
ALL_CHECKPOINTS=false
SAMPLE_N=64
REPO_ID="Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora"
HF_TOKEN_ARG=""
CONFIG_PATH=""
METRICS_FILE="eval_out/metrics.jsonl"
ERRORS_FILE="eval_out/errors.jsonl"
RUN_INSPECT=true
RUN_GENERATE=true
SAMPLES_FILE="eval_out/samples.jsonl"
SKIP_UPLOAD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"; shift 2;;
    --model-path)
      MODEL_PATH="$2"; shift 2;;
    --all-checkpoints)
      ALL_CHECKPOINTS=true; shift;;
    --sample-n)
      SAMPLE_N="$2"; shift 2;;
    --repo-id)
      REPO_ID="$2"; shift 2;;
    --hf-token)
      HF_TOKEN_ARG="$2"; shift 2;;
    --config)
      CONFIG_PATH="$2"; shift 2;;
    --metrics-file)
      METRICS_FILE="$2"; shift 2;;
    --errors-file)
      ERRORS_FILE="$2"; shift 2;;
    --skip-inspect)
      RUN_INSPECT=false; shift;;
    --reuse-samples)
      RUN_GENERATE=false; shift;;
    --no-upload)
      SKIP_UPLOAD=true; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1;;
  esac
done

# Function to discover all checkpoints
discover_checkpoints() {
  local base_path="$1"
  if [[ ! -d "$base_path" ]]; then
    echo "Error: model path '$base_path' not found." >&2
    return 1
  fi
  
  local checkpoints=()
  # Use find with null-terminated output for safety with spaces in paths
  while IFS= read -r -d '' checkpoint; do
    checkpoints+=("$checkpoint")
  done < <(find "$base_path" -maxdepth 1 -type d -name "checkpoint-*" -print0 2>/dev/null | sort -zV)
  
  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found in '$base_path'" >&2
    return 1
  fi
  
  printf '%s\n' "${checkpoints[@]}"
}

# Function to process a single checkpoint
process_checkpoint() {
  local checkpoint="$1"
  local checkpoint_name=$(basename "$checkpoint")
  
  # Use checkpoint-specific output files when processing all checkpoints
  local checkpoint_metrics_file="$METRICS_FILE"
  local checkpoint_errors_file="$ERRORS_FILE"
  local checkpoint_samples_file="$SAMPLES_FILE"
  
  if [[ "$ALL_CHECKPOINTS" == true ]]; then
    # Create checkpoint-specific output directories
    local eval_dir="eval_out/${checkpoint_name}"
    mkdir -p "$eval_dir"
    checkpoint_metrics_file="${eval_dir}/metrics.jsonl"
    checkpoint_errors_file="${eval_dir}/errors.jsonl"
    checkpoint_samples_file="${eval_dir}/samples.jsonl"
  fi
  
  mkdir -p "$(dirname "$checkpoint_metrics_file")"
  mkdir -p "$(dirname "$checkpoint_errors_file")"
  
  echo ""
  echo "========================================"
  echo "Processing: $checkpoint_name"
  echo "========================================"
  echo "Project dir : $PROJECT_DIR"
  echo "Checkpoint  : $checkpoint"
  echo "Samples     : $SAMPLE_N"
  echo "Metrics file: $checkpoint_metrics_file"
  echo ""
  
  if $RUN_INSPECT; then
    echo "--- Inspecting checkpoint ---"
    python inspect_checkpoint.py "$checkpoint" || echo "Warning: inspect_checkpoint encountered an issue."
    echo
  fi
  
  if $RUN_GENERATE; then
    echo "--- Generating evaluation samples ---"
    python gen_eval_samples.py \
      --model-path "$checkpoint" \
      --min-total-samples "$SAMPLE_N" \
      --output-dir "$(dirname "$checkpoint_samples_file")"
    # Update samples file path to the generated one
    checkpoint_samples_file="$(dirname "$checkpoint_samples_file")/samples.jsonl"
    echo
  else
    if [[ ! -f "$checkpoint_samples_file" ]]; then
      echo "Error: --reuse-samples specified but '$checkpoint_samples_file' not found."
      return 1
    fi
    echo "--- Reusing existing samples at $checkpoint_samples_file ---"
    echo
  fi
  
  echo "--- Running evaluation ($SAMPLE_N samples) ---"
  if ! python eval_rust.py "$checkpoint_samples_file" \
        --sample-n "$SAMPLE_N" \
        --check-func \
        --seed 0 \
        --save-errors "$checkpoint_errors_file" \
        | tee -a "$checkpoint_metrics_file"; then
    echo "Primary evaluation failed. Attempting fallback (sigilderg-eval)..."
    sigilderg-eval "$checkpoint_samples_file" \
      --sample-n "$SAMPLE_N" \
      --check-func \
      --seed 0 \
      --save-errors "$checkpoint_errors_file" \
      | tee -a "$checkpoint_metrics_file"
  fi
  echo
  
  echo "--- Cleaning metrics log ---"
  python - <<PY
import json
from pathlib import Path

metrics_path = Path("${checkpoint_metrics_file}")
if metrics_path.exists():
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    cleaned = []
    buffer = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("Saved "):
            continue
        if not buffer and not stripped.startswith("{"):
            continue
        buffer.append(stripped)
        candidate = "".join(buffer)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        else:
            cleaned.append(json.dumps(obj))
            buffer = []
    if buffer:
        print("Warning: trailing partial JSON in metrics log was discarded.")
    metrics_path.write_text(
        ("\n".join(cleaned) + "\n") if cleaned else "",
        encoding="utf-8"
    )
else:
    print("Warning: metrics file not found, skipping clean step.")
PY
  echo
  
  # Generate model card using new template
  MODEL_CARD_CMD=(python push_model_card.py "$checkpoint" --output "$checkpoint/README.md")
  if [[ -n "$CONFIG_PATH" ]]; then
    MODEL_CARD_CMD+=(--config "$CONFIG_PATH")
  fi
  
  echo "--- Refreshing checkpoint README (using new model card template) ---"
  "${MODEL_CARD_CMD[@]}"
  echo
  
  echo "--- Updating evaluation results in README ---"
  python update_model_card_eval.py "$checkpoint" --metrics-file "$checkpoint_metrics_file"
  echo
  
  # Upload to HuggingFace if enabled
  if [[ "$SKIP_UPLOAD" != true && -n "$REPO_ID" ]]; then
    HF_TOKEN_FINAL="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"
    if [[ -z "$HF_TOKEN_FINAL" ]]; then
      echo "Warning: HF token not provided (set HF_TOKEN env var or use --hf-token)."
      echo "Skipping HuggingFace uploads for this checkpoint."
    else
      echo "--- Uploading checkpoint folder to HuggingFace subdirectory ($REPO_ID) ---"
      python - <<PY
import os
from huggingface_hub import HfApi

token = "${HF_TOKEN_FINAL}"
repo_id = "${REPO_ID}"
folder_path = "${checkpoint}"
checkpoint_name = "${checkpoint_name}"

api = HfApi(token=token)
# Upload checkpoint folder to subdirectory
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
    path_in_repo=checkpoint_name,  # Upload to checkpoint-XXXX/ subdirectory
    commit_message="Upload ${checkpoint_name} checkpoint with evaluation results"
)
print(f"✓ Uploaded ${checkpoint_name} folder to {checkpoint_name}/")
PY
      echo
      
      # Upload checkpoint-specific README to subdirectory
      echo "--- Uploading checkpoint README to subdirectory ---"
      python - <<PY
import os
from huggingface_hub import HfApi

token = "${HF_TOKEN_FINAL}"
repo_id = "${REPO_ID}"
readme_path = "${checkpoint}/README.md"
checkpoint_name = "${checkpoint_name}"

if os.path.exists(readme_path):
    api = HfApi(token=token)
    # Upload README to checkpoint subdirectory
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo=f"{checkpoint_name}/README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update ${checkpoint_name} model card with evaluation results"
    )
    print(f"✓ Uploaded README to {checkpoint_name}/README.md")
else:
    print(f"Warning: README.md not found at {readme_path}")
PY
      echo
      
      # Upload metrics files to checkpoint subdirectory
      if [[ -f "$checkpoint_metrics_file" ]]; then
        echo "--- Uploading metrics to checkpoint subdirectory ---"
        python - <<PY
import os
from huggingface_hub import HfApi

token = "${HF_TOKEN_FINAL}"
repo_id = "${REPO_ID}"
metrics_path = "${checkpoint_metrics_file}"
checkpoint_name = "${checkpoint_name}"

if os.path.exists(metrics_path):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=metrics_path,
        path_in_repo=f"{checkpoint_name}/metrics.jsonl",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload ${checkpoint_name} evaluation metrics"
    )
    print(f"✓ Uploaded metrics to {checkpoint_name}/metrics.jsonl")
else:
    print(f"Warning: metrics file not found at {metrics_path}")
PY
      fi
      
      if [[ -f "$checkpoint_errors_file" ]]; then
        echo "--- Uploading errors log to checkpoint subdirectory ---"
        python - <<PY
import os
from huggingface_hub import HfApi

token = "${HF_TOKEN_FINAL}"
repo_id = "${REPO_ID}"
errors_path = "${checkpoint_errors_file}"
checkpoint_name = "${checkpoint_name}"

if os.path.exists(errors_path):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=errors_path,
        path_in_repo=f"{checkpoint_name}/errors.jsonl",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload ${checkpoint_name} evaluation errors"
    )
    print(f"✓ Uploaded errors log to {checkpoint_name}/errors.jsonl")
else:
    print(f"Warning: errors file not found at {errors_path}")
PY
      fi
      echo
    fi
  fi
  
  echo "=== Completed: $checkpoint_name ==="
  echo "Metrics log : $checkpoint_metrics_file"
  echo "Error log   : $checkpoint_errors_file"
  echo "Model card  : $checkpoint/README.md"
  echo
}

# Main execution logic
if [[ "$ALL_CHECKPOINTS" == true ]]; then
  # Process all checkpoints
  if [[ -n "$CHECKPOINT" ]]; then
    echo "Warning: --checkpoint specified with --all-checkpoints. Ignoring --checkpoint."
  fi
  
  echo "=== Running checkpoint workflow for ALL checkpoints ==="
  echo "Model path: $MODEL_PATH"
  echo "Sample count: $SAMPLE_N"
  echo "Repo ID: $REPO_ID"
  echo ""
  
  checkpoints=($(discover_checkpoints "$MODEL_PATH"))
  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "Error: No checkpoints found in '$MODEL_PATH'"
    exit 1
  fi
  
  echo "Found ${#checkpoints[@]} checkpoint(s):"
  for cp in "${checkpoints[@]}"; do
    echo "  - $(basename "$cp")"
  done
  echo ""
  
  # Process each checkpoint sequentially
  results=()
  latest_checkpoint=""
  for checkpoint in "${checkpoints[@]}"; do
    if process_checkpoint "$checkpoint"; then
      results+=("$(basename "$checkpoint"):✅")
      latest_checkpoint="$checkpoint"  # Track latest for root README update
    else
      results+=("$(basename "$checkpoint"):❌")
      echo "Warning: Failed to process $(basename "$checkpoint")"
    fi
  done
  
  # Update root README with latest checkpoint's model card
  if [[ "$SKIP_UPLOAD" != true && -n "$REPO_ID" && -n "$latest_checkpoint" ]]; then
    HF_TOKEN_FINAL="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"
    if [[ -n "$HF_TOKEN_FINAL" ]]; then
      echo ""
      echo "--- Updating root README with latest checkpoint model card ---"
      PUSH_CMD=(python push_model_card.py "$latest_checkpoint" --repo-id "$REPO_ID" --push)
      if [[ -n "$CONFIG_PATH" ]]; then
        PUSH_CMD+=(--config "$CONFIG_PATH")
      fi
      PUSH_CMD+=(--token "$HF_TOKEN_FINAL")
      "${PUSH_CMD[@]}"
      echo "✓ Root README updated with latest checkpoint ($(basename "$latest_checkpoint")) model card"
      echo ""
    fi
  fi
  
  # Summary
  echo ""
  echo "========================================"
  echo "Workflow Summary"
  echo "========================================"
  for result in "${results[@]}"; do
    echo "  $result"
  done
  echo ""
  
else
  # Single checkpoint mode (original behavior)
  if [[ -z "$CHECKPOINT" ]]; then
    if [[ -d "$MODEL_PATH" ]]; then
      latest_checkpoint="$(ls -1d "$MODEL_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
      if [[ -n "$latest_checkpoint" ]]; then
        CHECKPOINT="$latest_checkpoint"
      else
        CHECKPOINT="$MODEL_PATH"
      fi
    else
      echo "Error: model path '$MODEL_PATH' not found."
      exit 1
    fi
  fi
  
  if [[ ! -d "$CHECKPOINT" ]]; then
    echo "Error: checkpoint directory '$CHECKPOINT' does not exist."
    exit 1
  fi
  
  process_checkpoint "$CHECKPOINT"
  
  echo "=== Workflow complete ==="
  echo "Metrics log : $METRICS_FILE"
  echo "Error log   : $ERRORS_FILE"
  echo "Model card  : $CHECKPOINT/README.md"
fi
