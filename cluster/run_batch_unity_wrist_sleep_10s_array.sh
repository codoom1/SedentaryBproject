#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40gb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN
#SBATCH --ntasks=1
#SBATCH -o logs/epoch10s_%A_%a.log
#SBATCH --job-name=sb10s_array
#SBATCH --array=1-25%25

set -euo pipefail

# Produces durable 10-second Parquet files only; summary-epoch CSV output is disabled.
#
# Submit:
#   sbatch cluster/run_batch_unity_wrist_sleep_10s_array.sh
#
# Useful overrides:
#   sbatch --export=ALL,PROJECT_ROOT=/work/pi_jstauden_umass_edu/SBnovel_10s cluster/run_batch_unity_wrist_sleep_10s_array.sh
#   sbatch --export=ALL,BATCH_PREFIX=batch cluster/run_batch_unity_wrist_sleep_10s_array.sh
#   sbatch --export=ALL,INCLUDE_SLEEP_PROBABILITIES=1 cluster/run_batch_unity_wrist_sleep_10s_array.sh
#   sbatch --export=ALL,EPOCH_10S_ROOT=/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s cluster/run_batch_unity_wrist_sleep_10s_array.sh

module load conda/latest || true
export SHELL=/bin/bash

REPO_DIR="${PROJECT_ROOT:-/work/pi_jstauden_umass_edu/SBnovel_10s}"
if [ ! -d "$REPO_DIR" ]; then
  echo "[ERROR] Project root does not exist: $REPO_DIR" >&2
  exit 1
fi
cd "$REPO_DIR"

DATA_ROOT="${ARRAY_DATA_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_data_10s}"
OUTPUT_ROOT="${ARRAY_OUTPUT_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s}"
EPOCH_10S_ROOT="${EPOCH_10S_ROOT:-$OUTPUT_ROOT/epoch_10s}"

REQUIRED_FILES=(
  scripts/batch_pipeline.py
  scripts/run_participant_pipeline.py
  scripts/export_10s_dataset.py
  scripts/get_posture_predictions.py
  scripts/sleep_scripts/sleep_classify.py
)
for required_file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$required_file" ]; then
    echo "[ERROR] Required pipeline file is missing: $REPO_DIR/$required_file" >&2
    exit 1
  fi
done

export MASTER_ADDR=$(hostname)
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
MASTER_PORT_OFFSET="${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_PORT="${MASTER_PORT:-$((MASTER_PORT_BASE + MASTER_PORT_OFFSET))}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"
export RANK="${SLURM_PROCID:-0}"
export LOCAL_RANK="${SLURM_LOCALID:-0}"

mkdir -p logs "$DATA_ROOT" "$OUTPUT_ROOT" "$EPOCH_10S_ROOT"
mkdir -p \
  "$DATA_ROOT/raw" \
  "$DATA_ROOT/processed" \
  "$DATA_ROOT/preprocessed" \
  "$DATA_ROOT/predictions" \
  "$DATA_ROOT/sleep_predictions" \
  "$DATA_ROOT/summaries" \
  "$DATA_ROOT/tmp/sleep"

if [ -e data ] && [ ! -L data ]; then
  mv data "data_backup_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" || true
fi
ln -sfn "$DATA_ROOT" data

BATCH_PREFIX="${BATCH_PREFIX:-batch}"
BATCH_FILE="batches/${BATCH_PREFIX}_${SLURM_ARRAY_TASK_ID}.txt"

SLEEP_ENV="${SLEEP_ENV:-sklearn023}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
MODEL="${MODEL:-CHAP}"
WRIST_MODEL="${WRIST_MODEL:-CHAP}"
WRIST_DEVICE="${WRIST_DEVICE:-cuda}"
WRIST_CHECKPOINT="${WRIST_CHECKPOINT:-scripts/CHAP2/SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth}"
POSTURE_PADDING="${POSTURE_PADDING:-wrap}"
PARTICIPANT_TIMEOUT="${PARTICIPANT_TIMEOUT:-3500}"
SLEEP_DAY_CHUNKS="${SLEEP_DAY_CHUNKS:-6}"
SLEEP_CHUNK_OVERLAP="${SLEEP_CHUNK_OVERLAP:-30}"
SWAN_TIMEOUT="${SWAN_TIMEOUT:-300}"
SLEEP_SWAN_USE_WORKER="${SLEEP_SWAN_USE_WORKER:-1}"
SWAN_RETRIES="${SWAN_RETRIES:-1}"
SWAN_RETRY_TIMEOUT="${SWAN_RETRY_TIMEOUT:-120}"
SLEEP_MAX_SUBDIVISION_DEPTH="${SLEEP_MAX_SUBDIVISION_DEPTH:-4}"
SLEEP_MIN_CHUNK_MINUTES="${SLEEP_MIN_CHUNK_MINUTES:-15}"
INCLUDE_SLEEP_PROBABILITIES="${INCLUDE_SLEEP_PROBABILITIES:-1}"
INCLUDE_POSTURE_PROBABILITY="${INCLUDE_POSTURE_PROBABILITY:-1}"

SLEEP_TMP_BASE="$DATA_ROOT/tmp/sleep/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FAILED_DIR="$OUTPUT_ROOT/failed"
FAILED_OUT="$FAILED_DIR/epoch_10s_batch_${SLURM_ARRAY_TASK_ID}_failed.txt"
mkdir -p "$SLEEP_TMP_BASE" "$FAILED_DIR"

if [ ! -f "$BATCH_FILE" ]; then
  echo "[ERROR] Missing batch file: $BATCH_FILE" >&2
  exit 1
fi

hostname
nvidia-smi || true

echo "[INFO] Batch file: $BATCH_FILE"
echo "[INFO] Intermediate data root: $DATA_ROOT"
echo "[INFO] Durable 10-second output root: $EPOCH_10S_ROOT"
echo "[INFO] Include SWaN probabilities: $INCLUDE_SLEEP_PROBABILITIES"
echo "[INFO] Include posture sitting probability: $INCLUDE_POSTURE_PROBABILITY"
echo "[INFO] Posture padding: $POSTURE_PADDING"
echo "[INFO] Failed participants: $FAILED_OUT"

echo "[INFO] Checking posture CUDA and Parquet support in $POSTURE_ENV"
conda run -n "$POSTURE_ENV" --no-capture-output python - <<PY
import sys
import pyarrow
import torch

print("[CHECK] pyarrow_version=", pyarrow.__version__)
print("[CHECK] cuda_available=", torch.cuda.is_available())
if "${WRIST_DEVICE}" == "cuda" and not torch.cuda.is_available():
    sys.exit("Requested WRIST_DEVICE=cuda, but torch.cuda.is_available() is False")
PY

echo "[INFO] Checking SWaN availability in $SLEEP_ENV"
conda run -n "$SLEEP_ENV" --no-capture-output python - <<'PY'
import SWaN_accel
print("[CHECK] SWaN_accel import OK")
PY

CMD=(
  conda run -n "$POSTURE_ENV" --no-capture-output python scripts/batch_pipeline.py
  --batch-file "$BATCH_FILE"
  --model "$MODEL"
  --sleep-conda-env "$SLEEP_ENV"
  --posture-conda-env "$POSTURE_ENV"
  --posture-site wrist
  --posture-wrist-model "$WRIST_MODEL"
  --posture-wrist-device "$WRIST_DEVICE"
  --posture-wrist-checkpoint "$WRIST_CHECKPOINT"
  --posture-padding "$POSTURE_PADDING"
  --export-10s
  --skip-summary
  --export-10s-output-root "$EPOCH_10S_ROOT"
  --failed-out "$FAILED_OUT"
  --sleep-tmp-dir "$SLEEP_TMP_BASE"
  --sleep-day-chunks "$SLEEP_DAY_CHUNKS"
  --sleep-chunk-overlap "$SLEEP_CHUNK_OVERLAP"
  --sleep-swan-timeout "$SWAN_TIMEOUT"
  --sleep-swan-retries "$SWAN_RETRIES"
  --sleep-swan-retry-timeout "$SWAN_RETRY_TIMEOUT"
  --sleep-max-subdivision-depth "$SLEEP_MAX_SUBDIVISION_DEPTH"
  --sleep-min-chunk-minutes "$SLEEP_MIN_CHUNK_MINUTES"
  --participant-timeout "$PARTICIPANT_TIMEOUT"
  --download
)

if [ "$SLEEP_SWAN_USE_WORKER" = "1" ]; then
  CMD+=(--sleep-swan-use-worker)
fi

if [ "$INCLUDE_SLEEP_PROBABILITIES" = "1" ]; then
  CMD+=(--export-10s-include-sleep-probabilities)
fi

if [ "$INCLUDE_POSTURE_PROBABILITY" = "1" ]; then
  CMD+=(--posture-wrist-include-probability)
fi

"${CMD[@]}"
