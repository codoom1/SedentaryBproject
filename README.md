# SBnovel — Time‑resolved sedentary behavior at national scale

This repository contains a fully reproducible pipeline to derive time‑resolved (10‑second, aggregated hourly) sedentary behavior from raw wrist accelerometers (NHANES 2011–2014) and to summarize diurnal and weekday–weekend patterns using survey‑weighted models. We combine:

- Sleep/non‑wear detection (SWaN) to identify valid wear windows
- Posture classification (DeepPostures/CHAP) in 10‑second windows
- Participant‑level hourly summaries and batch orchestration

Background and purpose (from the abstract): Sedentary behavior is a major public health concern, but most surveillance collapses raw signals into daily summaries that obscure within‑day structure and are hard to reproduce. Our goal is to produce objective, nationally representative, time‑resolved estimates of sedentary behavior in the U.S., quantifying diurnal patterns and weekday–weekend differences via a reproducible pipeline.

Methods overview:

- Data: raw triaxial wrist accelerometer (NHANES 2011–2014; 14,692 participants)
- Sleep/non‑wear: SWaN model on 30‑second windows (WEAR, SLEEP, NON‑WEAR)
- Posture: CHAP model (DeepPostures) to label 10‑second windows as sitting vs not‑sitting; aggregated to hourly
- Alignment: Sleep expanded to 10‑second resolution to mask posture during sleep/non‑wear; sitting/not‑sitting computed among WEAR minutes
- Analytic models: survey‑weighted mixed‑effects models for national estimates across hours (performed downstream of this repo’s data products)

## Environments

Preferred: use the included conda environment YAMLs for reproducibility. This pins Python and package versions so everyone gets the same binaries without solver surprises.

Create from YAML (recommended):

```bash
# Sleep / SWaN environment
conda env create -f environment-swan.yml

# Posture / DeepPostures environment
conda env create -f environment-posture.yml

# Quick checks
conda run -n sklearn023 python scripts/sleep_scripts/sleep_classify.py --help
conda run -n deepposture python scripts/get_posture_predictions.py --help
```

Update an existing env after YAML changes:

```bash
conda env update -n sklearn023 -f environment-swan.yml --prune
conda env update -n deepposture -f environment-posture.yml --prune
```

Why preferred:

- Reproducible and shareable (checked into the repo)
- Exact versions ensure compatible binary wheels
- Easier onboarding and CI usage

Alternative (manual creation): if you prefer to construct envs by hand, use the steps below.

### 1) Sleep environment (swan)

- Purpose: run `scripts/sleep_scripts/sleep_classify.py` which requires older binary wheels.
- Uses: Python 3.8, scikit-learn==0.23.2

Create it with conda (Miniforge/Miniconda recommended):

```bash
conda create -n sklearn023 python=3.8 -c conda-forge scikit-learn=0.23.2 pandas numpy scipy -y
```

Usage:

```bash
# activate
conda activate sklearn023
python scripts/sleep_scripts/sleep_classify.py --participant-id 62161 --data-dir data/raw --by-day

# or without activating
conda run -n sklearn023 python scripts/sleep_scripts/sleep_classify.py --participant-id 62161 --data-dir data/raw --by-day
```

### 2) Posture environment (posture)

- Purpose: run posture prediction (DeepPostures) which requires modern scikit-learn and PyTorch.
- Uses: Python 3.11, scikit-learn==1.5.2, torch

Create it with conda:

```bash
conda create -n deepposture python=3.11 -c conda-forge scikit-learn=1.5.2 pandas numpy pytorch=2.4.1 -y
```

Usage:

```bash
conda activate deepposture
python scripts/get_posture_predictions.py --participant-id 62161 --model CHAP_ALL_ADULTS --skip-incomplete-days

# or without activating
conda run -n deepposture python scripts/get_posture_predictions.py --participant-id 62161 --model CHAP_ALL_ADULTS --skip-incomplete-days
```

## Wrapper scripts (optional)

Create small wrappers so you don't need to remember env names. Add these to the repo and make them executable.

`run_sleep.sh`:

```bash
#!/usr/bin/env bash
conda run -n sklearn023 python scripts/sleep_scripts/sleep_classify.py "$@"
```

`run_posture.sh`:

```bash
#!/usr/bin/env bash
conda run -n deepposture python scripts/get_posture_predictions.py "$@"
```

Make them executable:

```bash
chmod +x run_sleep.sh run_posture.sh
```

## Notes

- You already have the `swan` (sklearn023) environment active; use it for sleep scripts.
- Prefer `conda` for the older scikit-learn to avoid building from source.
- For reproducibility you can export environments to `environment-swan.yml` and `environment-posture.yml` with `conda env export -n <env> > environment-<name>.yml`.

If you'd like, I can add the two `environment-*.yml` files and the wrapper scripts to this repository. Which would you like me to create now?

## Repository structure

Key folders and files you’ll interact with:

```text
SBnovel/
├─ batches/                       # Batch lists of participants (CSV/TXT: cycle,participant_id)
│  ├─ batch_1.txt
│  └─ ...
├─ data/
│  ├─ raw/                        # Extracted raw NHANES archives
│  │  ├─ 2011-12/<SEQN>/*.sensor.csv[.gz]
│  │  └─ 2013-14/<SEQN>/...
│  ├─ processed/<SEQN>/           # Day‑level ActiGraph‑format CSVs (from prepare_DeepPosture_format)
│  ├─ preprocessed/<SEQN>/        # DeepPostures preprocessed tensors/CSV per day
│  ├─ predictions/<SEQN>/<MODEL>/ # 10‑second posture predictions per day (e.g., CHAP_ALL_ADULTS)
│  ├─ sleep_predictions/<SEQN>/   # SWaN outputs; per‑day CSVs under predictions/
│  └─ summaries/                  # Participant and batch‑level hourly summaries
├─ scripts/
│  ├─ batch_pipeline.py           # Batch runner: download → sleep → posture → summarize → append
│  ├─ run_participant_pipeline.py # Single participant orchestrator (uses conda run if provided)
│  ├─ download_participantPAMdata.py  # Download/extract NHANES participant archives
│  ├─ prepare_DeepPosture_format.py   # Convert raw hourly .sensor.csv to day‑level ActiGraph CSVs
│  ├─ get_posture_predictions.py       # Preprocess + run DeepPostures predictions
│  ├─ summarize_participant.py         # Merge sleep/posture into hourly percent metrics
│  ├─ sleep_scripts/
│  │  └─ sleep_classify.py        # SWaN sleep/non‑wear classification by day
│  └─ posture_library/MSSE-2021/  # DeepPostures (CHAP) code & pre‑trained models
└─ README.md
```

## Data flow

1) Download raw archives per participant and extract into `data/raw/<cycle>/<SEQN>/`.
2) Optional: convert raw hourly `.sensor.csv` into day‑level ActiGraph CSVs at `data/processed/<SEQN>/`.
3) Preprocess to DeepPostures format at `data/preprocessed/<SEQN>/` and run CHAP predictions to `data/predictions/<SEQN>/<MODEL>/`.
4) Run SWaN sleep/non‑wear per day to `data/sleep_predictions/<SEQN>/predictions/`.
5) Summarize to hourly metrics per participant to `data/summaries/<SEQN>_sleep_posture_hourly.csv` and optionally append to a batch master CSV.

## Scripts guide

- `scripts/download_participantPAMdata.py`
  - Purpose: Download and (optionally) extract participant archives from NHANES FTP.
  - Key functions: `download_participant_archive_only`, `download_partfiles`, `batch_download_logs`.
  - Example: download and extract to `data/raw/2011-12/62161/` and remove archive after extract.
    - python scripts/download_participantPAMdata.py 62161 2011-12 data/raw/2011-12 --extract --remove-archive

- `scripts/prepare_DeepPosture_format.py`
  - Purpose: Convert raw hourly `.sensor.csv` files into day‑level ActiGraph CSVs used by DeepPostures.
  - Output: `data/processed/<SEQN>/<YYYY-MM-DD>.csv`
  - Example:
    - python scripts/prepare_DeepPosture_format.py 2011-12 62161 --dest-dir data/raw --processed-dir data/processed

- `scripts/sleep_scripts/sleep_classify.py`
  - Purpose: Run SWaN sleep/non‑wear classification by day for a participant.
  - Output: `data/sleep_predictions/<SEQN>/predictions/<YYYY-MM-DD>_sleep_predictions.csv`
  - Typical run inside the older sklearn env:
    - conda run -n sklearn023 python scripts/sleep_scripts/sleep_classify.py --participant-id 62161 --data-dir data/raw/2011-12 --output-dir data/sleep_predictions --by-day

- `scripts/get_posture_predictions.py`
  - Purpose: Preprocess and run DeepPostures/CHAP to get 10‑second sitting vs not‑sitting predictions.
  - Finds model code under `scripts/posture_library/MSSE-2021/` and writes per‑day predictions to `data/predictions/<SEQN>/<MODEL>/`.
  - Useful flags: `--model` (e.g., CHAP_ALL_ADULTS), `--skip-incomplete-days`, `--model-root` (override), `--preprocess-only`, `--predict-only`.
  - Example (modern env):
    - conda run -n deepposture python scripts/get_posture_predictions.py --participant-id 62161 --model CHAP_ALL_ADULTS --skip-incomplete-days

- `scripts/summarize_participant.py`
  - Purpose: Merge 10‑second posture with 30‑second sleep (expanded to 10‑second) and compute hourly percentages:
    - percent_sleep_nonwear, percent_wear, percent_sitting, percent_not_sitting
  - Output: `data/summaries/<SEQN>_sleep_posture_hourly.csv`
  - Example:
    - python scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS --out data/summaries/62161_sleep_posture_hourly.csv

- `scripts/run_participant_pipeline.py`
  - Purpose: Orchestrate the full participant pipeline via subprocess with optional conda env selection per step.
  - Steps: optional download → sleep (SWaN) → posture (DeepPostures)
  - Example (dry run):
    - python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --dry-run
  - Example (execute, use envs, skip first/last incomplete days for both steps):
    - python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --download --skip-incomplete-days-sleep --skip-incomplete-days-posture --sleep-conda-env sklearn023 --posture-conda-env deepposture --posture-model CHAP_ALL_ADULTS

- `scripts/batch_pipeline.py`
  - Purpose: Batch‑process a list of participants: run participant pipeline, then `summarize_participant.py`, and append to a master CSV.
  - Input batch file: CSV/TXT with rows like `2011-12,62161` (ignore blank lines and `#` comments).
  - Default master CSV (compressed): `data/summaries/batch_sleep_posture_hourly.csv.gz` (gzip). Use `--no-compress-master` to write plain CSV.
  - Disk‑saving default: per‑participant summaries are written to a temporary file and deleted after appending to the master CSV. Pass `--keep-participant-summaries` if you want to keep each `data/summaries/<SEQN>_sleep_posture_hourly.csv` for debugging.
  - Example:
    - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --download

## Quickstart

1) Create two conda environments (sleep and posture) as described above.
2) Prepare a batch file under `batches/` with lines like `2011-12,62161`.
3) Run the batch pipeline (downloads raw data as needed, runs sleep and posture, summarizes, and appends to a master file):

   - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --download

4) Find outputs under:

- Sleep: `data/sleep_predictions/<SEQN>/predictions/*.csv`
- Posture: `data/predictions/<SEQN>/<MODEL>/*.csv`
- Hourly summary (per participant): `data/summaries/<SEQN>_sleep_posture_hourly.csv` (only if `--keep-participant-summaries` is used in batch mode; otherwise summaries are appended directly to the master and the temp files are removed)
- Batch master summary: `data/summaries/batch_sleep_posture_hourly.csv.gz` (gzip by default; pass `--no-compress-master` for plain CSV)

## Cluster batch (Unity GPU) 💻

We include a SLURM script to run each `batches/batch_<N>.txt` as a job array on the Unity cluster GPU partition:

- File: `cluster/run_batch_unity_gpu.slurm`
- Requests GPU partition, 4 cores, 40GB RAM, 10 days, long QoS
- Array: 25 tasks with concurrency 25 (1–25%25)
- Uses conda environments from this repo’s YAMLs

Submit from repo root:

```bash
sbatch cluster/run_batch_unity_gpu.slurm
```

Customize by editing env names (`SLEEP_ENV`, `POSTURE_ENV`), model, or array range in the script.

### GPU posture environment on cluster

Create a CUDA-enabled posture environment using the provided YAML and validate GPU access:

```bash
# One-time: create envs on the cluster login node
conda env create -f environment-swan.yml          # sleep
conda env create -f environment-posture-gpu.yml   # posture (GPU)

# Sanity checks
conda run -n deepposture-gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n deepposture-gpu python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
  print('device_count=', torch.cuda.device_count())
  print('device=', torch.cuda.get_device_name(0))
PY
```

The SLURM script requests a GPU with `#SBATCH --gres=gpu:1` and sets `POSTURE_ENV=deepposture-gpu`. If your cluster uses different partition names or CUDA versions, adjust `-p` and the `pytorch-cuda` version in `environment-posture-gpu.yml` accordingly.

## Outputs and definitions

- Sleep CSV (per day): 30‑second windows with `START_TIME`, `STOP_TIME`, `STATE ∈ {WEAR, SLEEP, NON‑WEAR}`.
- Posture CSV (per day): 10‑second rows with `timestamp`, `prediction ∈ {sitting, not-sitting}`.
- Hourly summary columns (per participant):
  - `percent_sleep_nonwear`: percent of hour in SLEEP or NON‑WEAR
  - `percent_wear`: percent of hour in WEAR
  - `percent_sitting`: percent of hour in sitting during WEAR (scaled to full hour so `percent_sitting + percent_not_sitting = percent_wear`)
  - `percent_not_sitting`: complement of sitting during WEAR

## Notes on reproducibility

- Use `conda run -n <env>` to pin binary dependencies across steps.
- The YAML files (`environment-swan.yml`, `environment-posture.yml`) are included and preferred for setup. If you customize locally, you can export your updates:
  - conda env export -n sklearn023 > environment-swan.yml
  - conda env export -n deepposture > environment-posture.yml
- Batch runs will, by default, clean up participant directories after summarization to save disk; pass `--no-cleanup` to keep all intermediates.
