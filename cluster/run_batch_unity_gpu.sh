#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40gb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10-00:00:00
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN
#SBATCH -o logs/sedentary_%A_%a.log
#SBATCH --job-name=sbnovel_batch
#SBATCH --array=1-25%25

set -euo pipefail

# Load conda and activate environments as needed
module load conda/latest

# Paths
REPO_DIR="$SLURM_SUBMIT_DIR"
cd "$REPO_DIR"

# Ensure logs directory exists
mkdir -p logs

# Resolve batch file for this array index
BATCH_FILE="batches/batch_${SLURM_ARRAY_TASK_ID}.txt"

# Environment names (should match README/env YAMLs)
SLEEP_ENV="sklearn023"
POSTURE_ENV="deepposture-gpu"
MODEL="CHAP_ALL_ADULTS"

# Sanity: show context
hostname
nvidia-smi || true

# Run: download raw, sleep in sklearn023, posture in deepposture; write compressed master summary
python scripts/batch_pipeline.py \
  --batch-file "$BATCH_FILE" \
  --model "$MODEL" \
  --sleep-conda-env "$SLEEP_ENV" \
  --posture-conda-env "$POSTURE_ENV" \
  --download
