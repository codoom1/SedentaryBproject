#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=24gb
#SBATCH -t 04:00:00
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH -o logs/epoch10s_summary_%j.log
#SBATCH --job-name=sb10s_summary

set -euo pipefail

# Submit from /work/pi_jstauden_umass_edu/SBnovel_10s:
#   sbatch cluster/run_epoch_10s_summary_refresh.sh
#
# This job can be submitted repeatedly while the 10-second production array runs.
# It processes only new or updated participant-day Parquet files.

module load conda/latest || true
export SHELL=/bin/bash
export PYTHONUNBUFFERED=1

REPO_DIR="${PROJECT_ROOT:-/work/pi_jstauden_umass_edu/SBnovel_10s}"
INPUT_ROOT="${EPOCH_10S_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s}"
SUMMARY_ROOT="${EPOCH_10S_SUMMARY_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s_summaries}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
WRITE_CSV_SNAPSHOTS="${WRITE_CSV_SNAPSHOTS:-0}"
OVERWRITE_DAYS="${OVERWRITE_DAYS:-0}"
REFRESH_ALL_PARTICIPANTS="${REFRESH_ALL_PARTICIPANTS:-0}"

cd "$REPO_DIR"
mkdir -p logs "$SUMMARY_ROOT"

if [ ! -f scripts/summarize_10s_dataset.py ]; then
  echo "[ERROR] Missing script: $REPO_DIR/scripts/summarize_10s_dataset.py" >&2
  exit 1
fi
if [ ! -d "$INPUT_ROOT" ]; then
  echo "[ERROR] Input dataset does not exist: $INPUT_ROOT" >&2
  exit 1
fi

# Prevent overlapping refresh jobs from writing the same summaries.
exec 9>"$SUMMARY_ROOT/.summary_refresh.lock"
if ! flock -n 9; then
  echo "[INFO] Another summary refresh is already running; exiting."
  exit 0
fi

CMD=(
  conda run -n "$POSTURE_ENV" --no-capture-output python scripts/summarize_10s_dataset.py
  --input-root "$INPUT_ROOT"
  --output-root "$SUMMARY_ROOT"
)

if [ "$WRITE_CSV_SNAPSHOTS" = "1" ]; then
  CMD+=(--write-csv-snapshots)
fi
if [ "$OVERWRITE_DAYS" = "1" ]; then
  CMD+=(--overwrite-days)
fi
if [ "$REFRESH_ALL_PARTICIPANTS" = "1" ]; then
  CMD+=(--refresh-all-participants)
fi

"${CMD[@]}"
