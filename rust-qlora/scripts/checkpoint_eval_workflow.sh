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
  4. Refresh checkpoint README
  5. Inject latest evaluation metrics
  6. Optionally push the model card to HuggingFace

Options:
  --checkpoint PATH      Path to checkpoint directory (default: latest under --model-path)
  --model-path PATH      Training output directory to search for checkpoints
                         (default: out/llama8b-rust-qlora-phase1)
  --sample-n N           Number of samples to evaluate (default: 64)
  --repo-id ID           HuggingFace repo id to push README (optional)
  --hf-token TOKEN       HuggingFace token (defaults to $HF_TOKEN if set)
  --config FILE          Override training config used for model card metadata
  --metrics-file FILE    Metrics JSONL path (default: eval_out/metrics.jsonl)
  --errors-file FILE     Error log path (default: eval_out/errors.jsonl)
  --skip-inspect         Skip inspect_checkpoint step
  --reuse-samples        Skip generation step and reuse existing eval_out/samples.jsonl
  -h, --help             Show this help text
EOF
}

MODEL_PATH="out/llama8b-rust-qlora-phase1"
CHECKPOINT=""
SAMPLE_N=64
REPO_ID=""
HF_TOKEN_ARG=""
CONFIG_PATH=""
METRICS_FILE="eval_out/metrics.jsonl"
ERRORS_FILE="eval_out/errors.jsonl"
RUN_INSPECT=true
RUN_GENERATE=true
SAMPLES_FILE="eval_out/samples.jsonl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"; shift 2;;
    --model-path)
      MODEL_PATH="$2"; shift 2;;
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
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1;;
  esac
done

# Resolve checkpoint path
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

mkdir -p "$(dirname "$METRICS_FILE")"
mkdir -p "$(dirname "$ERRORS_FILE")"

echo "=== Running checkpoint workflow ==="
echo "Project dir : $PROJECT_DIR"
echo "Checkpoint  : $CHECKPOINT"
echo "Samples     : $SAMPLE_N"
echo "Metrics file: $METRICS_FILE"
echo

if $RUN_INSPECT; then
  echo "--- Inspecting checkpoint ---"
  python inspect_checkpoint.py "$CHECKPOINT" || echo "Warning: inspect_checkpoint encountered an issue."
  echo
fi

if $RUN_GENERATE; then
  echo "--- Generating evaluation samples ---"
  python gen_eval_samples.py \
    --model-path "$CHECKPOINT" \
    --min-total-samples "$SAMPLE_N"
  echo
else
  if [[ ! -f "$SAMPLES_FILE" ]]; then
    echo "Error: --reuse-samples specified but '$SAMPLES_FILE' not found."
    exit 1
  fi
  echo "--- Reusing existing samples at $SAMPLES_FILE ---"
  echo
fi

echo "--- Running evaluation ($SAMPLE_N samples) ---"
if ! python eval_rust.py "$SAMPLES_FILE" \
      --sample-n "$SAMPLE_N" \
      --check-func \
      --seed 0 \
      --save-errors "$ERRORS_FILE" \
      | tee -a "$METRICS_FILE"; then
  echo "Primary evaluation failed. Attempting fallback (sigilderg-eval)..."
  sigilderg-eval "$SAMPLES_FILE" \
    --sample-n "$SAMPLE_N" \
    --check-func \
    --seed 0 \
    --save-errors "$ERRORS_FILE" \
    | tee -a "$METRICS_FILE"
fi
echo

echo "--- Cleaning metrics log ---"
python - <<PY
import json
from pathlib import Path

metrics_path = Path("${METRICS_FILE}")
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

MODEL_CARD_CMD=(python push_model_card.py "$CHECKPOINT" --output "$CHECKPOINT/README.md")
if [[ -n "$CONFIG_PATH" ]]; then
  MODEL_CARD_CMD+=(--config "$CONFIG_PATH")
fi

echo "--- Refreshing checkpoint README ---"
"${MODEL_CARD_CMD[@]}"
echo

echo "--- Updating evaluation results in README ---"
python update_model_card_eval.py "$CHECKPOINT" --metrics-file "$METRICS_FILE"
echo

if [[ -n "$REPO_ID" ]]; then
  echo "--- Uploading checkpoint to HuggingFace ($REPO_ID) ---"
  HF_TOKEN_FINAL="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"
  if [[ -z "$HF_TOKEN_FINAL" ]]; then
    echo "Error: HF token not provided (set HF_TOKEN env var or use --hf-token)."
    exit 1
  fi
  python - <<PY
import os
from huggingface_hub import HfApi

token = "${HF_TOKEN_FINAL}"
repo_id = "${REPO_ID}"
folder_path = "${CHECKPOINT}"

api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
    commit_message="Upload checkpoint from checkpoint_eval_workflow"
)
PY
  echo

  echo "--- Pushing model card to HuggingFace ($REPO_ID) ---"
  PUSH_CMD=(python push_model_card.py "$CHECKPOINT" --repo-id "$REPO_ID" --push)
  if [[ -n "$CONFIG_PATH" ]]; then
    PUSH_CMD+=(--config "$CONFIG_PATH")
  fi
  PUSH_CMD+=(--token "$HF_TOKEN_FINAL")
  "${PUSH_CMD[@]}"
  echo
else
  echo "Skipping HuggingFace upload/push (no --repo-id provided)."
fi

echo "=== Workflow complete ==="
echo "Metrics log : $METRICS_FILE"
echo "Error log   : $ERRORS_FILE"
echo "Model card  : $CHECKPOINT/README.md"

