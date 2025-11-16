#!/usr/bin/env bash
set -euo pipefail
tmux new -d -s rustft "bash scripts/run_train.sh"
tmux split-window -v -t rustft "bash scripts/run_eval_loop.sh"
tmux attach -t rustft
