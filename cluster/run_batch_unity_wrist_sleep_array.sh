#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40gb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN
#SBATCH --ntasks=1
#SBATCH -o logs/sed_%A_%a.log
#SBATCH --job-name=sbB_array
#SBATCH --array=1-25%25

set -euo pipefail

# Run from the repository root:
#   sbatch cluster/run_batch_unity_wrist_sleep_array.sh
#
# Optional overrides at submit time:
#   sbatch --export=ALL,SUMMARY_EPOCH=30s cluster/run_batch_unity_wrist_sleep_array.sh
#   sbatch --export=ALL,SUMMARY_EPOCH=30s,EPOCH_COLUMNS="20m 30m" cluster/run_batch_unity_wrist_sleep_array.sh
#   sbatch --export=ALL,SLEEP_DAY_CHUNKS=6,PARTICIPANT_TIMEOUT=900 cluster/run_batch_unity_wrist_sleep_array.sh
#   sbatch --export=ALL,ARRAY_DATA_ROOT=/work/pi_jstauden_umass_edu/SBpaper_data_30s cluster/run_batch_unity_wrist_sleep_array.sh

module load conda/latest || true
export SHELL=/bin/bash

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

# Heavy intermediates and final batch masters go on /work.
DATA_ROOT="${ARRAY_DATA_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_data}"
OUTPUT_ROOT="${ARRAY_OUTPUT_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"
export RANK="${SLURM_PROCID:-0}"
export LOCAL_RANK="${SLURM_LOCALID:-0}"

mkdir -p logs "$DATA_ROOT" "$OUTPUT_ROOT"
mkdir -p \
  "$DATA_ROOT/raw" \
  "$DATA_ROOT/processed" \
  "$DATA_ROOT/preprocessed" \
  "$DATA_ROOT/predictions" \
  "$DATA_ROOT/sleep_predictions" \
  "$DATA_ROOT/summaries" \
  "$DATA_ROOT/tmp/sleep"

# Redirect repo/data to DATA_ROOT so all large pipeline outputs land on /work.
if [ -e data ] && [ ! -L data ]; then
  mv data "data_backup_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" || true
fi
ln -sfn "$DATA_ROOT" data

BATCH_FILE="batches/batch_${SLURM_ARRAY_TASK_ID}.txt"

SLEEP_ENV="${SLEEP_ENV:-sklearn023}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
MODEL="${MODEL:-CHAP}"
WRIST_MODEL="${WRIST_MODEL:-CHAP}"
WRIST_DEVICE="${WRIST_DEVICE:-cuda}"
WRIST_CHECKPOINT="${WRIST_CHECKPOINT:-scripts/CHAP2/SUBMIT_RESULT/SOL_W/CHAP_FT/checkpoint-submit.pth}"
POSTURE_PADDING="${POSTURE_PADDING:-wrap}"
SUMMARY_EPOCH="${SUMMARY_EPOCH:-30m}"
EPOCH_COLUMNS="${EPOCH_COLUMNS:-}"
PARTICIPANT_TIMEOUT="${PARTICIPANT_TIMEOUT:-900}"
SLEEP_DAY_CHUNKS="${SLEEP_DAY_CHUNKS:-6}"
SLEEP_CHUNK_OVERLAP="${SLEEP_CHUNK_OVERLAP:-30}"
SWAN_TIMEOUT="${SWAN_TIMEOUT:-300}"
INCLUDE_POSTURE_PROBABILITY="${INCLUDE_POSTURE_PROBABILITY:-0}"

SLEEP_TMP_BASE="$DATA_ROOT/tmp/sleep/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MASTER_OUT="$OUTPUT_ROOT/wrist_sleep_${SUMMARY_EPOCH}_batch_${SLURM_ARRAY_TASK_ID}.csv.gz"
FAILED_DIR="$OUTPUT_ROOT/failed"
FAILED_OUT="$FAILED_DIR/wrist_sleep_${SUMMARY_EPOCH}_batch_${SLURM_ARRAY_TASK_ID}_failed.txt"

mkdir -p "$SLEEP_TMP_BASE" "$FAILED_DIR"

hostname
nvidia-smi || true

echo "[INFO] Repo: $REPO_DIR"
echo "[INFO] Batch file: $BATCH_FILE"
echo "[INFO] Data root: $DATA_ROOT"
echo "[INFO] Master output: $MASTER_OUT"
echo "[INFO] Failed output: $FAILED_OUT"
echo "[INFO] MASTER_ADDR: $MASTER_ADDR"
echo "[INFO] MASTER_PORT: $MASTER_PORT"
echo "[INFO] WORLD_SIZE: $WORLD_SIZE"
echo "[INFO] RANK: $RANK"
echo "[INFO] LOCAL_RANK: $LOCAL_RANK"
echo "[INFO] Sleep env: $SLEEP_ENV"
echo "[INFO] Posture env: $POSTURE_ENV"
echo "[INFO] Posture site: wrist"
echo "[INFO] Wrist model: $WRIST_MODEL"
echo "[INFO] Wrist device: $WRIST_DEVICE"
echo "[INFO] Wrist checkpoint: $WRIST_CHECKPOINT"
echo "[INFO] Posture padding: $POSTURE_PADDING"
echo "[INFO] Include posture sitting probability: $INCLUDE_POSTURE_PROBABILITY"
echo "[INFO] Summary epoch: $SUMMARY_EPOCH"
echo "[INFO] Extra epoch columns: ${EPOCH_COLUMNS:-none}"
echo "[INFO] Sleep temp: $SLEEP_TMP_BASE"
echo "[INFO] Participant timeout: $PARTICIPANT_TIMEOUT seconds"
echo "[INFO] Sleep chunks/day: ${SLEEP_DAY_CHUNKS:-sleep_classify default}"
echo "[INFO] Sleep chunk overlap: ${SLEEP_CHUNK_OVERLAP:-sleep_classify default}"
echo "[INFO] SWaN normal-run timeout: $SWAN_TIMEOUT seconds"
echo "[INFO] SWaN worker fallback: disabled"

echo "[INFO] Checking posture CUDA availability in $POSTURE_ENV"
conda run -n "$POSTURE_ENV" --no-capture-output python - <<PY
import sys
import torch

requested = "${WRIST_DEVICE}"
print("[CHECK] torch_version=", torch.__version__)
print("[CHECK] cuda_available=", torch.cuda.is_available())
print("[CHECK] cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[CHECK] cuda_device_0=", torch.cuda.get_device_name(0))

if requested == "cuda" and not torch.cuda.is_available():
    sys.exit("Requested WRIST_DEVICE=cuda, but torch.cuda.is_available() is False")
PY

echo "[INFO] Checking SWaN availability in $SLEEP_ENV"
conda run -n "$SLEEP_ENV" --no-capture-output python - <<'PY'
import SWaN_accel
print("[CHECK] SWaN_accel import OK")
print("[CHECK] SWaN runs in the sleep env; this pipeline has no SWaN CUDA/device option.")
PY

if [ ! -f "$BATCH_FILE" ]; then
  echo "[ERROR] Missing batch file: $BATCH_FILE" >&2
  exit 1
fi

# Sleep is included by default. Do not pass --skip-sleep.
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
  --summary-epoch "$SUMMARY_EPOCH"
  --master-out "$MASTER_OUT"
  --failed-out "$FAILED_OUT"
  --sleep-tmp-dir "$SLEEP_TMP_BASE"
  --sleep-swan-timeout "$SWAN_TIMEOUT"
  --participant-timeout "$PARTICIPANT_TIMEOUT"
  --download
)

if [ -n "$SLEEP_DAY_CHUNKS" ]; then
  CMD+=(--sleep-day-chunks "$SLEEP_DAY_CHUNKS")
fi

if [ -n "$SLEEP_CHUNK_OVERLAP" ]; then
  CMD+=(--sleep-chunk-overlap "$SLEEP_CHUNK_OVERLAP")
fi

if [ -n "$EPOCH_COLUMNS" ]; then
  read -r -a EXTRA_EPOCH_COLUMNS <<< "$EPOCH_COLUMNS"
  CMD+=(--epoch-columns "${EXTRA_EPOCH_COLUMNS[@]}")
fi

if [ "$INCLUDE_POSTURE_PROBABILITY" = "1" ]; then
  CMD+=(--posture-wrist-include-probability)
fi

"${CMD[@]}"
