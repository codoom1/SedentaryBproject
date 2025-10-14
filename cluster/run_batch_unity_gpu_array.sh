#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40gb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN
#SBATCH -o logs/sedentary_array_%A_%a.log
#SBATCH --job-name=sbnovel_batch_array
#SBATCH --array=1-25%25

set -euo pipefail

# Load conda and activate environments as needed (cluster-specific)
module load conda/latest || true
export SHELL=/bin/bash

# Paths
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

# Choose cluster storage roots
# Heavy intermediates (raw/processed/preprocessed/predictions/sleep_predictions) on fast/shared work
DATA_ROOT="/work/pi_bpachev_umass_edu/SBnovel_data"
# Persistent final outputs (per-task masters) on project
OUTPUT_ROOT="/project/pi_bpachev_umass_edu/SBnovel_outputs"

# Ensure directories exist
mkdir -p logs "$DATA_ROOT" "$OUTPUT_ROOT"
mkdir -p "$DATA_ROOT"/raw "$DATA_ROOT"/processed "$DATA_ROOT"/preprocessed "$DATA_ROOT"/predictions "$DATA_ROOT"/sleep_predictions "$DATA_ROOT"/summaries

# Redirect repo/data -> $DATA_ROOT so all pipeline writes land on /work
if [ -e data ] && [ ! -L data ]; then
  mv data "data_backup_${SLURM_ARRAY_TASK_ID}" || true
fi
ln -sfn "$DATA_ROOT" data

# Resolve batch file for this array index
BATCH_FILE="batches/batch_${SLURM_ARRAY_TASK_ID}.txt"

# Environment names (should match README/env YAMLs)
SLEEP_ENV="sklearn023"
POSTURE_ENV="deepposture-gpu"
MODEL="CHAP_ALL_ADULTS"

# Each task writes to its OWN master output to avoid concurrent append to the same gzip
MASTER_OUT="$OUTPUT_ROOT/master_batch_${SLURM_ARRAY_TASK_ID}.csv.gz"

# Sanity: show context
hostname
nvidia-smi || true

echo "[INFO] Using batch file: $BATCH_FILE"
echo "[INFO] Writing per-task master: $MASTER_OUT"

# Run: download raw, sleep in sklearn023, posture in deepposture-gpu; write compressed per-task master summary
python scripts/batch_pipeline.py \
  --batch-file "$BATCH_FILE" \
  --model "$MODEL" \
  --sleep-conda-env "$SLEEP_ENV" \
  --posture-conda-env "$POSTURE_ENV" \
  --master-out "$MASTER_OUT" \
  --download
