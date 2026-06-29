#!/bin/bash

set -euo pipefail

# Usage from /work/pi_jstauden_umass_edu/SBnovel_10s:
#   scripts/inspect_10s_data.sh raw
#   scripts/inspect_10s_data.sh day 65645
#   scripts/inspect_10s_data.sh raw 65645 2000-01-07
#   scripts/inspect_10s_data.sh all 65645
#
# Optional environment overrides:
#   PROJECT_ROOT, RAW_ROOT, SUMMARY_ROOT, INSPECTION_ROOT, POSTURE_ENV,
#   RAW_ROWS, FULL, SEED

VIEW="${1:-}"
PARTICIPANT_ID="${2:-}"
DAY="${3:-}"

if [[ ! "$VIEW" =~ ^(raw|day|participant|all)$ ]]; then
  echo "Usage: $0 {raw|day|participant|all} [participant_id] [YYYY-MM-DD]" >&2
  exit 1
fi

module load conda/latest || true

PROJECT_ROOT="${PROJECT_ROOT:-/work/pi_jstauden_umass_edu/SBnovel_10s}"
RAW_ROOT="${RAW_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s}"
SUMMARY_ROOT="${SUMMARY_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s_summaries}"
INSPECTION_ROOT="${INSPECTION_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/inspection}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
RAW_ROWS="${RAW_ROWS:-20}"
FULL="${FULL:-0}"
SEED="${SEED:-}"

cd "$PROJECT_ROOT"
mkdir -p "$INSPECTION_ROOT"

label="${PARTICIPANT_ID:-random}"
if [ -n "$DAY" ]; then
  label="${label}_${DAY}"
fi
OUTPUT_CSV="$INSPECTION_ROOT/inspect_${label}_${VIEW}.csv"

CMD=(
  conda run -n "$POSTURE_ENV" --no-capture-output python scripts/inspect_10s_summaries.py
  --view "$VIEW"
  --raw-root "$RAW_ROOT"
  --summary-root "$SUMMARY_ROOT"
  --raw-rows "$RAW_ROWS"
  --output-csv "$OUTPUT_CSV"
)

if [ -n "$PARTICIPANT_ID" ]; then
  CMD+=(--participant-id "$PARTICIPANT_ID")
fi
if [ -n "$DAY" ]; then
  CMD+=(--day "$DAY")
fi
if [ -n "$SEED" ]; then
  CMD+=(--seed "$SEED")
fi
if [ "$FULL" = "1" ]; then
  CMD+=(--full)
fi

"${CMD[@]}"

echo "[INFO] Inspection output directory: $INSPECTION_ROOT"
