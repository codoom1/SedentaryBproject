# SBnovel — Time‑resolved sedentary behavior at national scale

This repository contains a fully reproducible upstream data-generation pipeline to derive time-resolved sedentary behavior summaries from raw wrist accelerometers (NHANES 2011-2014). It produces analysis-ready CHAP posture and SWaN sleep/non-wear summaries that can be consumed by a separate manuscript/statistical-analysis repository. We combine:

- Sleep/non‑wear detection (SWaN) to identify valid wear windows
- Posture classification (DeepPostures/CHAP) in 10‑second windows
- Participant-level summaries at a requested summary epoch, such as 1-hour, 1-minute, 30-second, or 10-second

Background and purpose (from the abstract): Sedentary behavior is a major public health concern, but most surveillance collapses raw signals into daily summaries that obscure within‑day structure and are hard to reproduce. Our goal is to produce objective, nationally representative, time‑resolved estimates of sedentary behavior in the U.S., quantifying diurnal patterns and weekday–weekend differences via a reproducible pipeline.

Methods overview:

- Data: raw triaxial wrist accelerometer (NHANES 2011–2014; 14,692 participants)
- Sleep/non‑wear: SWaN model on 30‑second windows (WEAR, SLEEP, NON‑WEAR)
- Posture: CHAP model (DeepPostures) to label 10-second windows as sitting vs not-sitting
- Summary stage: SWaN 30-second states are expanded to the 10-second CHAP grid, then both model outputs are summarized to the requested epoch
- Sleep removal and analytic models are performed downstream of this repo's data products

## Environments

Preferred: use the included conda environment YAMLs for reproducibility. This pins Python and package versions so everyone gets the same binaries without solver surprises.

Create from YAML (recommended):

```bash
# Sleep / SWaN environment
conda env create -f environments/swan.yml

# Posture / DeepPostures environment
conda env create -f environments/posture.yml

# Optional: Posture (GPU) environment
conda env create -f environments/posture-gpu.yml
```

Troubleshooting: if env creation fails during the pip stage with an error like "TypeError: sequence item 0: expected str instance, dict found", update your local copy of the YAMLs. We removed a malformed pip options block; the fixed files in this repo no longer use a pip section for posture envs (all packages come from conda channels).

Troubleshooting (SWaN/NumPy): if you see errors like "module 'numpy' has no attribute 'float'", your NumPy is too new (>= 1.24 removed np.float). The sleep env is pinned to avoid this, but if your cluster solved a newer NumPy, downgrade to < 1.24:

```bash
# Preferred (conda)
conda install -n sklearn023 -c conda-forge 'numpy<1.24' -y

# Alternative (pip inside the env)
conda run -n sklearn023 pip install 'numpy<1.24'

# Verify
conda run -n sklearn023 python - <<'PY'
import numpy as np
print('numpy=', np.__version__, 'has np.float?', hasattr(np, 'float'))
PY
```

Cluster note: if you see a rollback with "No compatible shell found!", set your shell explicitly and retry:

```bash
export SHELL=/bin/bash
conda --no-plugins env create -f environments/posture-gpu.yml
```

Solver note (cluster): if the above with `--no-plugins` errors with "non-default solver backend (libmamba) not recognized", force the classic solver for this command:

```bash
export CONDA_SOLVER=classic
conda --no-plugins env create -f environments/posture-gpu.yml
```

Alternatively, you can pass the flag or set user config once:

```bash
# One-off flag
conda env create -f environments/posture-gpu.yml --solver=classic

# Or persist to your ~/.condarc (you can switch back later)
conda config --set solver classic
```

### Quick checks

```bash
conda run -n sklearn023 python scripts/sleep_scripts/sleep_classify.py --help
conda run -n deepposture python scripts/get_posture_predictions.py --help
conda run -n deepposture-gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Update an existing env after YAML changes:

```bash
conda env update -n sklearn023 -f environments/swan.yml --prune
conda env update -n deepposture -f environments/posture.yml --prune
conda env update -n deepposture-gpu -f environments/posture-gpu.yml --prune
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

### Optional: Posture (GPU) — manual pip alternative

If the conda YAML for GPU fails on your system/cluster, you can create a clean env and install PyTorch CUDA 12.1 wheels directly via pip. Do not load system CUDA modules or add extra nvidia-* pip packages; the cu121 wheels bundle the CUDA runtime already.

Create a fresh env and install:

```bash
# Fresh env with Python 3.11
conda create -n deepposture-gpu python=3.11 pip -y
conda activate deepposture-gpu

# Ensure any CPU-only torch is removed (safe if not installed)
pip uninstall -y torch torchvision torchaudio || true

# Install PyTorch + CUDA 12.1 wheels from the official index
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

# Scientific stack
pip install scikit-learn==1.5.2 pandas numpy scipy h5py tqdm
```

Validate GPU is visible:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'cuda', getattr(torch.version,'cuda',None))
if torch.cuda.is_available():
  print('device_count=', torch.cuda.device_count(), 'name=', torch.cuda.get_device_name(0))
PY
```

Notes:

- Don’t mix with system CUDA modules (e.g., `module load cuda/...`); the cu121 wheels carry their own runtime.
- Don’t install extra pip packages like `nvidia-cudnn-cu12`; they’re not needed and can conflict.
- Prefer the conda YAML (`environments/posture-gpu.yml`) when it resolves; use this pip route as a pragmatic fallback.

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

- If you already have the `swan` (sklearn023) environment active; use it for sleep scripts.
- Prefer `conda` for the older scikit-learn to avoid building from source.
// For reproducibility you can export environments to `environments/swan.yml` and `environments/posture.yml` with `conda env export -n <env> > environments/<name>.yml`.

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
│  └─ summaries/                  # Participant and batch-level summaries by requested epoch
├─ constants/
│  └─ participants.csv            # All (cycle, participant_id) pairs. Use in data download
├─ scripts/
│  ├─ batch_pipeline.py           # Batch runner: download → sleep → posture → summarize → append
│  ├─ run_participant_pipeline.py # Single participant orchestrator (uses conda run if provided)
│  ├─ download_participantPAMdata.py  # Download/extract NHANES participant archives
│  ├─ prepare_DeepPosture_format.py   # Convert raw hourly .sensor.csv to day‑level ActiGraph CSVs
│  ├─ get_posture_predictions.py       # Preprocess + run DeepPostures predictions
│  ├─ summarize_participant.py         # Summarize sleep/posture into epoch-level percent metrics
│  ├─ make_batches_from_constants.py   # Generate deduped fixed-size batch files from constants
│  ├─ sleep_scripts/
│  │  └─ sleep_classify.py        # SWaN sleep/non‑wear classification by day
│  └─ posture_library/MSSE-2021/  # DeepPostures (CHAP) code & pre‑trained models
├─ cluster/
│  └─ run_batch_unity_gpu.sh  # SLURM array batch script for Unity GPU partition
├─ environments/
│  ├─ posture.yml
│  ├─ posture-gpu.yml
│  └─ swan.yml
└─ README.md
```

## Data flow

1) Download raw archives per participant and extract into `data/raw/<cycle>/<SEQN>/`.
2) Optional: convert raw hourly `.sensor.csv` into day‑level ActiGraph CSVs at `data/processed/<SEQN>/`.
3) Preprocess to DeepPostures format at `data/preprocessed/<SEQN>/` and run CHAP predictions to `data/predictions/<SEQN>/<MODEL>/`.
4) Run SWaN sleep/non‑wear per day to `data/sleep_predictions/<SEQN>/predictions/`.
5) Summarize to epoch-level metrics per participant, for example `data/summaries/<SEQN>_sleep_posture_1h_epoch.csv`, and optionally append to a batch master CSV.

### Participants list (constants)

We provide `constants/participants.csv` with columns:

- `cycle` in {`2011-12`, `2013-14`}
- `participant_id` (SEQN)

This can be used to:

- Drive downloads programmatically
- Generate new batch files (e.g., split by N per file)
- Quickly search for SEQNs by cycle

Example to create a new 25‑file batch set (100 IDs each):

```bash
mkdir -p batches
tail -n +2 constants/participants.csv | split -l 100 - batches/batch_ --additional-suffix=.txt --numeric-suffixes=1
sed -i '' 's/^/2011-12,/g' batches/batch_*.txt # if your split lost cycle column, keep both columns otherwise
```

### Generate batches via helper script

Use `scripts/make_batches_from_constants.py` to create deduped, fixed-size batch files from the canonical list:

```bash
# Batches of 100 rows each, written to batches/batch_<N>.txt
python scripts/make_batches_from_constants.py --batch-size 100

# Shuffle the order with a fixed seed, 250 per batch, start numbering at 10
python scripts/make_batches_from_constants.py --batch-size 250 --shuffle --seed 42 --start-index 10

# Custom paths and naming
python scripts/make_batches_from_constants.py \
  --participants constants/participants.csv \
  --out-dir batches \
  --prefix nhanes_ \
  --batch-size 200
```

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
  - Purpose: Expand 30-second SWaN states to the 10-second CHAP grid, then compute neutral epoch-level percentages:
    - percent_sleep_nonwear, percent_wear, percent_sitting, percent_not_sitting
  - Sleep removal is not done here; downstream manuscript/statistical analysis applies the sleep/non-wear cutoff.
  - Output: `data/summaries/<SEQN>_sleep_posture_<summary_epoch>_epoch.csv`
  - Example:
    - python scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS --summary-epoch 1h --out data/summaries/62161_sleep_posture_1h_epoch.csv
    - python scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS --summary-epoch 30s --out data/summaries/62161_sleep_posture_30s_epoch.csv

- `scripts/export_10s_dataset.py`
  - Purpose: Preserve the CHAP 10-second predictions and aligned SWaN state as compact participant-day Parquet files before intermediate prediction CSVs are deleted.
  - Output layout: `data/epoch_10s/model=<MODEL>/Dataset=<CYCLE>/ID=<SEQN>/Day=<YYYY-MM-DD>/part-0.parquet`
  - File columns: `epoch_index`, `sitting`, `sleep_state`, `valid_sleep`, `waking`, and `sleep_nonwear`. SWaN state codes are `0=WEAR`, `1=SLEEP`, `2=NON-WEAR`, and `-1=missing`.
  - Example:
    - python scripts/export_10s_dataset.py --participant-id 83724 --dataset 2013-14 --model CHAP_ALL_ADULTS --output-root data/epoch_10s

- `scripts/summarize_10s_dataset.py`
  - Purpose: Incrementally create participant-day and participant-level summaries from the durable 10-second Parquet dataset.
  - Unchanged participant-day inputs are skipped. Participant summaries are refreshed only when new or updated days are found.
  - Participant-day output includes the complete SWaN-state-by-sitting cross-tab as explicit count and hour columns.
  - It also calculates gap-aware waking sedentary bouts, immediate breaks in sitting, sustained breaks, and time accumulated in bouts lasting at least 10, 20, 30, and 60 minutes.
  - Bouts use the 10-second CHAP grid. SWaN states are natively 30 seconds and aligned to three 10-second rows; sleep/non-wear ends bouts but does not count as a break.
  - Output roots:
    - `data/epoch_10s_summaries/participant_day/model=<MODEL>/Dataset=<CYCLE>/ID=<SEQN>/Day=<DATE>/part-0.parquet`
    - `data/epoch_10s_summaries/participant/model=<MODEL>/Dataset=<CYCLE>/ID=<SEQN>/part-0.parquet`
  - Example:
    - python scripts/summarize_10s_dataset.py --input-root data/epoch_10s --output-root data/epoch_10s_summaries

- `scripts/run_participant_pipeline.py`
  - Purpose: Orchestrate the full participant pipeline via subprocess with optional conda env selection per step.
  - Steps: optional download -> sleep (SWaN) -> posture (DeepPostures) -> optional durable 10-second Parquet export -> optional epoch-level summary
  - Example (dry run):
    - python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --dry-run
  - Example (execute, use envs, skip first/last incomplete days for both steps):
    - python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --download --skip-incomplete-days-sleep --skip-incomplete-days-posture --sleep-conda-env sklearn023 --posture-conda-env deepposture --posture-model CHAP_ALL_ADULTS
  - Example (also write a 30-second summary for one participant):
    - python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --download --skip-incomplete-days-sleep --skip-incomplete-days-posture --sleep-conda-env sklearn023 --posture-conda-env deepposture --posture-model CHAP_ALL_ADULTS --summarize --summary-epoch 30s

- `scripts/batch_pipeline.py`
  - Purpose: Batch-process a list of participants and independently produce durable 10-second Parquet files, summary-epoch output, or both.
  - Input batch file: CSV/TXT with rows like `2011-12,62161` (ignore blank lines and `#` comments).
  - Default master CSV (compressed): `data/summaries/batch_sleep_posture_hourly.csv.gz` for `--summary-epoch 1h`; non-hourly runs use names such as `data/summaries/batch_sleep_posture_30s_epoch.csv.gz`. Use `--no-compress-master` to write plain CSV.
  - Disk-saving default: per-participant summaries are written to a temporary file and deleted after appending to the master CSV. Pass `--keep-participant-summaries` if you want to keep each `data/summaries/<SEQN>_sleep_posture_<summary_epoch>_epoch.csv` for debugging.
  - Example:
    - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --download
    - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --summary-epoch 30s --download
    - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP --sleep-conda-env sklearn023 --posture-conda-env deepposture --posture-site wrist --export-10s --skip-summary --export-10s-output-root data/epoch_10s --download

## Quickstart

1) Create two conda environments (sleep and posture) as described above.
2) Prepare a batch file under `batches/` with lines like `2011-12,62161`.
3) Run the batch pipeline (downloads raw data as needed, runs sleep and posture, summarizes, and appends to a master file):

   - python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --download

4) Find outputs under:

- Sleep: `data/sleep_predictions/<SEQN>/predictions/*.csv`
- Posture: `data/predictions/<SEQN>/<MODEL>/*.csv`
- Epoch-level summary (per participant): `data/summaries/<SEQN>_sleep_posture_<summary_epoch>_epoch.csv` (only if `--keep-participant-summaries` is used in batch mode; otherwise summaries are appended directly to the master and the temp files are removed)
- Batch master summary: `data/summaries/batch_sleep_posture_hourly.csv.gz` for 1-hour summaries or `data/summaries/batch_sleep_posture_<summary_epoch>_epoch.csv.gz` for non-hourly summaries. Gzip is used by default; pass `--no-compress-master` for plain CSV.
- Durable 10-second data: `data/epoch_10s/model=<MODEL>/Dataset=<CYCLE>/ID=<SEQN>/Day=<YYYY-MM-DD>/part-0.parquet` when `--export-10s` is used. Pass `--posture-wrist-include-probability` to include `prob_sitting` as float32.

Specify `--summary-epoch` as a duration string such as `10s`, `30s`, `1min`, `5m`, `20m`, `30m`, `45m`, `1h`, or `2h`. The duration must be a multiple of the 10-second CHAP base epoch and must divide evenly into 24 hours. If the duration divides evenly into one hour, `Hour` is populated and `epoch` is the within-hour epoch index. If it does not divide evenly into one hour, `Hour` is left blank and `epoch` is the within-day epoch index.

You can also add coarser epoch labels in the same output with `--epoch-columns`. For example, run once at 30-second resolution and include 20-minute and 30-minute grouping columns:

```bash
python scripts/batch_pipeline.py \
  --batch-file batches/batch_1.txt \
  --model CHAP_ALL_ADULTS \
  --sleep-conda-env sklearn023 \
  --posture-conda-env deepposture-gpu \
  --posture-site wrist \
  --summary-epoch 30s \
  --epoch-columns 20m 30m \
  --download
```

This creates columns such as `epoch_20m` and `epoch_30m`. These are grouping labels, not separate metric columns. Use `Hour` for hourly grouping. Use the finest `--summary-epoch` you need, then aggregate rows downstream using the coarser epoch columns.

## Cluster batch (Unity GPU) 💻

We include SLURM scripts to run each `batches/batch_<N>.txt` as a job array on the Unity cluster GPU partition:

- Wrist model with sleep included: `cluster/run_batch_unity_wrist_sleep_array.sh`
- Wrist model with sleep, durable 10-second Parquet only: `cluster/run_batch_unity_wrist_sleep_10s_array.sh`
- Incremental participant-day and participant summary refresh: `cluster/run_epoch_10s_summary_refresh.sh`
- Legacy/general GPU launcher: `cluster/run_batch_unity_gpu_array.sh`
- Requests GPU partition, 1 task, 4 cores, 40GB RAM, 10 days, long QoS
- Wrist/sleep array: 25 tasks with concurrency 25 (`1-25%25`)
- Default storage roots are `/work/pi_jstauden_umass_edu/SBpaper_data` and `/work/pi_jstauden_umass_edu/SBpaper_outputs`
- Uses conda environments from this repo’s YAMLs

Submit from repo root:

```bash
sbatch cluster/run_batch_unity_wrist_sleep_array.sh
```

To produce only the durable 10-second Parquet dataset and skip summary-epoch CSV creation:

```bash
sbatch cluster/run_batch_unity_wrist_sleep_10s_array.sh
```

The 10-second array writes persistent outputs under `$ARRAY_OUTPUT_ROOT/epoch_10s` by default. Raw data, processed data, and prediction CSVs remain temporary and are deleted after each successful participant export.

While the 10-second production array is running, submit the summary refresh
whenever you want to incorporate newly completed participant-days:

```bash
sbatch cluster/run_epoch_10s_summary_refresh.sh
```

The refresh job is idempotent. Existing unchanged days are skipped, updated
participants are refreshed, and a lock prevents two refresh jobs from writing
the same summaries simultaneously. Set `WRITE_CSV_SNAPSHOTS=1` only when a
combined CSV snapshot is needed:

```bash
sbatch --export=ALL,WRITE_CSV_SNAPSHOTS=1 cluster/run_epoch_10s_summary_refresh.sh
```

After changing the summary metric definitions, rebuild existing day and
participant summaries once:

```bash
sbatch --export=ALL,OVERWRITE_DAYS=1,REFRESH_ALL_PARTICIPANTS=1 \
  cluster/run_epoch_10s_summary_refresh.sh
```

The wrist/sleep launcher includes SWaN sleep prediction and uses `--posture-site wrist`. Customize env names (`SLEEP_ENV`, `POSTURE_ENV`), model, wrist device, summary epoch, storage roots, or array range in the script. For a 30-second summary run, submit:

```bash
sbatch --export=ALL,SUMMARY_EPOCH=30s cluster/run_batch_unity_wrist_sleep_array.sh
```

To add coarser epoch grouping columns in the SLURM output:

```bash
sbatch --export=ALL,SUMMARY_EPOCH=30s,EPOCH_COLUMNS="20m 30m" cluster/run_batch_unity_wrist_sleep_array.sh
```

The wrist/sleep launcher uses two layers of protection against hung runs. First, `PARTICIPANT_TIMEOUT` limits the whole participant pipeline, defaulting to 7200 seconds. Second, SWaN sleep chunks run normally first with `SWAN_TIMEOUT`, defaulting to 300 seconds per chunk. If a normal SWaN chunk times out, the script retries that chunk in an isolated worker process. Set `SWAN_WORKER_FALLBACK=0` to disable this fallback, or pass `--sleep-swan-use-worker` manually if you want every SWaN chunk to run in a worker. Defaults in the launcher are `SLEEP_DAY_CHUNKS=3`, `SLEEP_CHUNK_OVERLAP=30`, `SWAN_RETRIES=3`, `SWAN_RETRY_TIMEOUT=300`, `SWAN_MAX_SUBDIVISION_DEPTH=4`, and `SWAN_MIN_CHUNK_MINUTES=3`.

### GPU posture environment on cluster

Create a CUDA-enabled posture environment using the provided YAML and validate GPU access:

```bash
# One-time: create envs on the cluster login node
conda env create -f environments/swan.yml          # sleep
conda env create -f environments/posture-gpu.yml   # posture (GPU)

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

The SLURM script requests a GPU with `#SBATCH --gpus=1` and sets `POSTURE_ENV=deepposture-gpu`. If your cluster uses different partition names or CUDA versions, adjust `-p` and the `pytorch-cuda` version in `environments/posture-gpu.yml` accordingly.

## Outputs and definitions

- Sleep CSV (per day): 30‑second windows with `START_TIME`, `STOP_TIME`, and either `STATE ∈ {WEAR, SLEEP, NON‑WEAR}` or numeric `PREDICTED` values that can be mapped to those states.
- Posture CSV (per day): 10‑second rows with `timestamp`, `prediction ∈ {sitting, not-sitting}`.
- Epoch-level summary columns (per participant):
  - `ID`: participant identifier
  - `Dataset`: NHANES cycle label, when provided by batch mode, for example `2011-2012`
  - `Day`: calendar day
  - `DayType`: `Weekday` or `Weekend`
  - `Hour`: hour of day containing the summary epoch, coded 1-24 when the requested epoch divides evenly into one hour; left blank for epochs such as 45 minutes that do not align within each hour
  - `epoch`: within-hour epoch index when the requested epoch divides evenly into one hour; otherwise within-day epoch index
  - `epoch_start`, `epoch_end`: start and end timestamps for the summary epoch
  - `epoch_duration_seconds`, `epoch_duration_hours`: duration of the summary epoch; downstream analyses use this to convert percentages to hours
  - `percent_sleep_nonwear`: percent of the summary epoch classified by SWaN as SLEEP or NON-WEAR
  - `percent_wear`: percent of the summary epoch classified by SWaN as WEAR
  - `percent_sitting`: percent of the summary epoch classified by CHAP as sitting, regardless of SWaN sleep/wear state
  - `percent_not_sitting`: percent of the summary epoch classified by CHAP as not sitting, regardless of SWaN sleep/wear state
  - `n_base_epochs`, `n_sleep_nonwear_epochs`, `n_wear_epochs`, `n_sitting_epochs`, `n_not_sitting_epochs`: counts of 10-second base epochs used to build each row
  - `summary_epoch`: requested summary resolution, such as `1h` or `30s`
  - `model`: posture model used, such as `CHAP_ALL_ADULTS`

This upstream repository intentionally does not remove sleep/non-wear from sitting. The downstream manuscript/statistical-analysis repository applies the calibrated sleep/non-wear cutoff and computes waking sedentary time.

## Notes on reproducibility

- Use `conda run -n <env>` to pin binary dependencies across steps.
- The YAML files (`environments/environment-swan.yml`, `environments/environment-posture.yml`) are included and preferred for setup. If you customize locally, you can export your updates:
- conda env export -n sklearn023 > environments/environment-swan.yml
- conda env export -n deepposture > environments/environment-posture.yml
- Batch runs will, by default, clean up participant directories after summarization to save disk; pass `--no-cleanup` to keep all intermediates.
