#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 "$ROOT/scripts/run_prompt_message_grid.py" \
  --out-dir "$ROOT/results/prompt-message-grid"
