# step_by_step_pipeline.py
"""
This script demonstrates a step-by-step pipeline:
1. Preprocesses the example CSV file using the pre_process_data.py functionality.
2. Saves the output in a new directory.
3. Runs make_predictions.py on the preprocessed data.
"""

import os  # For path manipulations
import subprocess  # For running external scripts
import argparse # For command-line argument parsing
import pathlib # For path manipulations
import sys # For sys.executable
import shutil # For file operations
import glob # For file pattern matching
import tempfile # For temporary directories
from typing import Optional, Callable, Any # For type hints

# Optional helper to convert raw NHANES .sensor.csv files into daily ActiGraph CSVs
try:
    from prepare_DeepPosture_format import prepare_deeppostures_format as _prepare_deeppostures_format
    prepare_deeppostures_format: Optional[Callable[..., Any]] = _prepare_deeppostures_format
except Exception:
    prepare_deeppostures_format = None

# Resolve paths relative to the repository root regardless of cwd
THIS_FILE = pathlib.Path(__file__).resolve() # this script produces absolute path
REPO_ROOT = THIS_FILE.parents[1]

# The DeepPosture model code has been moved inside the nested directory 'posture_library/MSSE-2021'.
# For backward compatibility (in case someone still has the old layout), we detect which path exists.
_candidate_new = REPO_ROOT / "scripts" / "posture_library" / "MSSE-2021"
_candidate_old = REPO_ROOT / "MSSE-2021"  # legacy (pre-move) ## This is not needed anymore



def find_model_root(override: Optional[str] = None):
    """Return a pathlib.Path to the model root. If override provided, validate and return it."""
    if override:
        p = pathlib.Path(override)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Provided model root does not exist: {override}")

    if _candidate_new.exists():
        return _candidate_new
    if _candidate_old.exists():
        return _candidate_old
    raise FileNotFoundError(
        "Could not locate the MSSE-2021 model directory. Expected at 'posture_library/MSSE-2021' (new) or 'MSSE-2021' (legacy)."
    )

# These will be set after argument parsing to allow CLI overrides
PREPROCESS_SCRIPT = None
PREDICT_SCRIPT = None
GT3X_FREQUENCY = 80
DOWN_SAMPLE_FREQUENCY = 10

def run_preprocessing(main_csv_dir, preprocessed_dir, preprocess_script, gt3x_frequency, down_sample_frequency, skip_incomplete_days=False, verbose=False):
    print(f"[get_posture_predictions] Starting preprocessing for: {main_csv_dir}")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Filter CSV files if skip_incomplete_days is True
    if skip_incomplete_days:
        temp_dir = tempfile.mkdtemp()
        csv_files = sorted(glob.glob(os.path.join(main_csv_dir, "*.csv*")))

        if len(csv_files) > 2:  # Only filter if we have more than 2 files
            # Skip first and last CSV files
            filtered_files = csv_files[1:-1]
            print(f"[getpred_pipeline] Filtering out incomplete days: skipping {len(csv_files) - len(filtered_files)} files")
            print(f"[getpred_pipeline] Skipped files: {os.path.basename(csv_files[0])}, {os.path.basename(csv_files[-1])}")

            # Copy filtered files to temp directory
            for file_path in filtered_files:
                shutil.copy2(file_path, temp_dir)

            csv_source_dir = temp_dir
        else:
            print(f"[getpred_pipeline] Not enough CSV files to filter (found {len(csv_files)})")
            csv_source_dir = main_csv_dir
    else:
        csv_source_dir = main_csv_dir

    cmd = [
        sys.executable, str(preprocess_script),
        "--gt3x-dir", csv_source_dir,
        "--pre-processed-dir", preprocessed_dir,
        "--gt3x-frequency", str(gt3x_frequency),
        "--down-sample-frequency", str(down_sample_frequency),
    ]
    if verbose:
        print(f"[getpred_pipeline] Running preprocessing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Clean up temporary directory if created
    if skip_incomplete_days and 'temp_dir' in locals():
        shutil.rmtree(temp_dir)

    print(f"[getpred_pipeline] Preprocessing complete for: {main_csv_dir}")
## Need to add the padding option to improve predictions for the last incomplete window
def run_predictions(preprocessed_dir, predictions_dir, predict_script, model, padding, verbose=False):
    print(f"[getpred_pipeline] Starting predictions for: {preprocessed_dir} with model: {model}")
    os.makedirs(predictions_dir, exist_ok=True)
    cmd = [
        sys.executable, str(predict_script),
        "--model", model,
        "--pre-processed-dir", preprocessed_dir,
        "--predictions-dir", predictions_dir,
        "--padding", str(padding)
    ]
    if verbose:
        print(f"[getpred_pipeline] Running predictions command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[getpred_pipeline] Predictions complete for: {preprocessed_dir}")

if __name__ == "__main__":
    print("[getpred_pipeline] Script started.")
    parser = argparse.ArgumentParser(description="DeepPostures participant pipeline")
    parser.add_argument('--participant-id', type=str, required=True, help='Participant ID to process')
    parser.add_argument('--model', type=str, default='CHAP_A', help='Model subfolder to use for predictions (e.g., CHAP_A, CHAP_ALL_ADULTS)')
    parser.add_argument('--padding', type=str, default='drop', help='Padding for predictions (default: drop)')
    parser.add_argument('--skip-incomplete-days', action='store_true', help='Skip first and last day CSV files (incomplete data)')
    parser.add_argument('--model-root', type=str, default=None, help='Path to MSSE-2021 model root (overrides automatic detection)')
    parser.add_argument('--gt3x-frequency', type=int, default=GT3X_FREQUENCY, help=f'GT3X sample frequency (default: {GT3X_FREQUENCY})')
    parser.add_argument('--down-sample-frequency', type=int, default=DOWN_SAMPLE_FREQUENCY, help=f'Down-sample frequency (default: {DOWN_SAMPLE_FREQUENCY})')
    parser.add_argument('--pre-processed-dir', type=str, default=None, help='Override preprocessed output directory')
    parser.add_argument('--predictions-dir', type=str, default=None, help='Override predictions output directory')
    parser.add_argument('--preprocess-only', action='store_true', help='Only run preprocessing and exit')
    parser.add_argument('--predict-only', action='store_true', help='Only run predictions (requires preprocessed data to exist)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging for subprocess commands')
    args = parser.parse_args()

    participant_id = args.participant_id
    model = args.model
    padding = args.padding
    skip_incomplete_days = args.skip_incomplete_days
    print(f"[getpred_pipeline] Processing participant: {participant_id} with model: {model}")
    # Prefer per-day processed CSVs if present; otherwise fall back to raw GT3X sensor CSVs
    processed_dir_path = (REPO_ROOT / "data" / "processed" / participant_id).resolve()
    raw_dir_11 = (REPO_ROOT / "data" / "raw" / "2011-12" / participant_id).resolve()
    raw_dir_13 = (REPO_ROOT / "data" / "raw" / "2013-14" / participant_id).resolve()

    csv_source_dir = None
    source_kind = None  # 'processed' (daily) or 'raw' (hourly)
    # Check processed per-day first
    if processed_dir_path.exists() and any(processed_dir_path.glob('*.csv')):
        csv_source_dir = str(processed_dir_path)
        source_kind = 'processed'
        print(f"[getpred_pipeline] Using processed day-level CSVs: {csv_source_dir}")
    else:
        # Fallback to raw: prefer 2011-12, then 2013-14
        for candidate in [raw_dir_11, raw_dir_13]:
            if candidate.exists() and any(candidate.glob('*.sensor.csv')):
                csv_source_dir = str(candidate)
                source_kind = 'raw'
                print(f"[getpred_pipeline] Using raw GT3X hourly files: {csv_source_dir}")
                break
        # As a looser fallback (in case filenames don't end with .sensor.csv), accept any CSVs
        if csv_source_dir is None:
            for candidate in [raw_dir_11, raw_dir_13]:
                if candidate.exists() and any(candidate.glob('*.csv')):
                    csv_source_dir = str(candidate)
                    source_kind = 'raw'
                    print(f"[getpred_pipeline] Using raw CSV files: {csv_source_dir}")
                    break

        # If we only have raw input, try auto-converting to daily ActiGraph CSVs once
        if source_kind == 'raw' and not (processed_dir_path.exists() and any(processed_dir_path.glob('*.csv'))):
            # Determine cycle from the selected raw directory
            if csv_source_dir is not None and str(raw_dir_11) in csv_source_dir:
                cycle = '2011-12'
            elif csv_source_dir is not None and str(raw_dir_13) in csv_source_dir:
                cycle = '2013-14'
            else:
                cycle = '2011-12'
            if prepare_deeppostures_format is not None:
                print(f"[getpred_pipeline] Auto-preparing ActiGraph daily CSVs from raw for participant {participant_id} (cycle {cycle})...")
                ok = prepare_deeppostures_format(
                    dataset_name=cycle,
                    participant_id=participant_id,
                    dest_dir=str((REPO_ROOT / 'data' / 'raw').resolve()),
                    processed_dir=str((REPO_ROOT / 'data' / 'processed').resolve()),
                    delete_raw=False,
                    delete_newformat=False,
                )
                if ok and any(processed_dir_path.glob('*.csv')):
                    csv_source_dir = str(processed_dir_path)
                    source_kind = 'processed'
                    print(f"[getpred_pipeline] Converted and switching to processed day-level CSVs: {csv_source_dir}")
                else:
                    print("[getpred_pipeline] Warning: auto-preparation did not produce day-level CSVs; continuing with raw input.")
            else:
                print("[getpred_pipeline] Note: prepare_DeepPosture_format not available; cannot auto-convert raw to daily CSVs.")

    if csv_source_dir is None:
        print(f"[getpred_pipeline] ERROR: Could not locate input files for participant {participant_id}. Searched: {processed_dir_path}, {raw_dir_11}, {raw_dir_13}")
        sys.exit(1)
    preprocessed_dir = args.pre_processed_dir if args.pre_processed_dir else str((REPO_ROOT / "data" / "preprocessed" / participant_id).resolve())
    predictions_dir = args.predictions_dir if args.predictions_dir else str((REPO_ROOT / "data" / "predictions" / participant_id).resolve())

    # Resolve model root and script locations
    model_root = find_model_root(args.model_root)
    PREPROCESS_SCRIPT = (model_root / "pre_process_data.py").resolve()
    PREDICT_SCRIPT = (model_root / "make_predictions.py").resolve()

    try:
        if args.predict_only and not args.preprocess_only:
            # Only run predictions; assume preprocessed data exists
            run_predictions(preprocessed_dir, predictions_dir, PREDICT_SCRIPT, model, padding, verbose=args.verbose)
        elif args.preprocess_only and not args.predict_only:
            # When skipping incomplete days, only applicable to processed day-level CSVs
            if skip_incomplete_days and source_kind != 'processed':
                print("[getpred_pipeline] Note: --skip-incomplete-days is only applied for day-level processed CSVs; using full raw input.")
            run_preprocessing(csv_source_dir, preprocessed_dir, PREPROCESS_SCRIPT, args.gt3x_frequency, args.down_sample_frequency, (skip_incomplete_days and source_kind == 'processed'), verbose=args.verbose)
        else:
            # Full pipeline
            if skip_incomplete_days and source_kind != 'processed':
                print("[getpred_pipeline] Note: --skip-incomplete-days is only applied for day-level processed CSVs; using full raw input.")
            run_preprocessing(csv_source_dir, preprocessed_dir, PREPROCESS_SCRIPT, args.gt3x_frequency, args.down_sample_frequency, (skip_incomplete_days and source_kind == 'processed'), verbose=args.verbose)
            run_predictions(preprocessed_dir, predictions_dir, PREDICT_SCRIPT, model, padding, verbose=args.verbose)

        print(f"[] Pipeline complete. Predictions saved in: {predictions_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[getpred_pipeline] External script failed with return code {e.returncode}")
        raise



#python scripts/posture_library/MSSE-2021/pre_process_data.py  --gt3x-dir data/processed/62161 --pre-processed-dir data/preprocessed/ --gt3x-frequency 80 --down-sample-frequency 10
# python scripts/get_posture_predictions.py --participant-id 62161 --skip-incomplete-days --model CHAP_ALL_ADULTS