#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=16gb
#SBATCH -t 12:00:00
#SBATCH -o logs/epoch1min_%j.log
#SBATCH --job-name=epoch1min

# Incrementally aggregate completed 10-second partitions to one minute.
# Safe to rerun while the CHAP/SWaN array is still producing new days.

set -euo pipefail

module load conda/latest || true
export SHELL=/bin/bash

REPO_DIR="${PROJECT_ROOT:-/work/pi_jstauden_umass_edu/SBnovel_10s}"
INPUT_ROOT="${EPOCH_10S_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s}"
OUTPUT_ROOT="${EPOCH_1MIN_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_1min}"
PAXMIN_CACHE_ROOT="${PAXMIN_CACHE_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/paxmin_cache}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
DATASET="${DATASET:-2011-2012}"
MODEL="${MODEL:-CHAP}"

cd "$REPO_DIR"
mkdir -p logs

conda run -n "$POSTURE_ENV" --no-capture-output python \
  scripts/summarize_10s_to_1min.py summarize \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --paxmin-cache-root "$PAXMIN_CACHE_ROOT" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  "$@"
