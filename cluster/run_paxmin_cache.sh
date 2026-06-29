#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=24gb
#SBATCH -t 2-00:00:00
#SBATCH -o logs/paxmin_cache_%j.log
#SBATCH --job-name=paxcache

# Stream one complete CDC PAXMIN XPT and cache compact participant Parquets.
# Submit one cycle at a time:
#   sbatch --export=ALL,DATASET=2011-2012 cluster/run_paxmin_cache.sh
#   sbatch --export=ALL,DATASET=2013-2014 cluster/run_paxmin_cache.sh

set -euo pipefail

module load conda/latest || true
export SHELL=/bin/bash

REPO_DIR="${PROJECT_ROOT:-/work/pi_jstauden_umass_edu/SBnovel_10s}"
PAXMIN_ROOT="${PAXMIN_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_data_10s/nhanes/paxmin}"
PAXMIN_CACHE_ROOT="${PAXMIN_CACHE_ROOT:-/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/paxmin_cache}"
POSTURE_ENV="${POSTURE_ENV:-deepposture-gpu}"
DATASET="${DATASET:-2011-2012}"

case "$DATASET" in
  2011-2012) XPT="$PAXMIN_ROOT/PAXMIN_G.xpt" ;;
  2013-2014) XPT="$PAXMIN_ROOT/PAXMIN_H.xpt" ;;
  *) echo "[ERROR] DATASET must be 2011-2012 or 2013-2014" >&2; exit 2 ;;
esac

cd "$REPO_DIR"
mkdir -p logs "$PAXMIN_CACHE_ROOT"

conda run -n "$POSTURE_ENV" --no-capture-output python \
  scripts/summarize_10s_to_1min.py cache-paxmin \
  --xpt "$XPT" \
  --dataset "$DATASET" \
  --cache-root "$PAXMIN_CACHE_ROOT"
