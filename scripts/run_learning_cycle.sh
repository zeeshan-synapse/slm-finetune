#!/usr/bin/env bash
set -euo pipefail

# Task 5 automation:
# - Check how many NEW guardrail failure logs were added
# - Build corrective pairs from logs
# - Trigger full train/export/eval cycle only if threshold is met
#
# Usage:
#   scripts/run_learning_cycle.sh [model_tag] [run_id] [min_new_failures]
#
# Example:
#   scripts/run_learning_cycle.sh synapse-qwen1.5b-v7 v7 30

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/logs/guardrail_failures.jsonl"
STATE_FILE="$PROJECT_DIR/logs/.learning_state"

MODEL_TAG="${1:-synapse-qwen1.5b-v7}"
RUN_ID="${2:-v7}"
MIN_NEW_FAILURES="${3:-30}"

if ! [[ "$MIN_NEW_FAILURES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: min_new_failures must be a non-negative integer."
  exit 1
fi

if [[ ! -f "$LOG_FILE" ]]; then
  echo "No failure log found at: $LOG_FILE"
  echo "Run guardrailed collection first:"
  echo "  python3 scripts/collect_failures_batch.py --repeat 3"
  exit 0
fi

mkdir -p "$PROJECT_DIR/logs"

LAST_PROCESSED=0
if [[ -f "$STATE_FILE" ]]; then
  LAST_PROCESSED="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
fi

if ! [[ "$LAST_PROCESSED" =~ ^[0-9]+$ ]]; then
  LAST_PROCESSED=0
fi

TOTAL_LINES="$(python3 - <<'PY'
import pathlib
p = pathlib.Path("logs/guardrail_failures.jsonl")
count = 0
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
print(count)
PY
)"

NEW_LINES=$((TOTAL_LINES - LAST_PROCESSED))
if (( NEW_LINES < 0 )); then
  NEW_LINES=TOTAL_LINES
fi

echo "Learning cycle precheck:"
echo "  total_failure_rows: $TOTAL_LINES"
echo "  last_processed:     $LAST_PROCESSED"
echo "  new_failure_rows:   $NEW_LINES"
echo "  threshold:          $MIN_NEW_FAILURES"

if (( NEW_LINES < MIN_NEW_FAILURES )); then
  echo "Not enough new failure rows yet. Skipping training cycle."
  exit 0
fi

echo "Building corrective pairs from logs..."
cd "$PROJECT_DIR"
python3 scripts/build_corrective_from_logs.py \
  --input "$LOG_FILE" \
  --output "$PROJECT_DIR/data/corrective_pairs_from_logs.jsonl"

echo "Starting full automation run..."
scripts/run_cycle.sh "$MODEL_TAG" "$RUN_ID"

echo "$TOTAL_LINES" > "$STATE_FILE"
echo "Learning cycle complete."
echo "Updated state file: $STATE_FILE"
