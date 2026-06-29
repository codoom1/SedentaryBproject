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
WRIST_PREDICT_SCRIPT = (REPO_ROOT / "scripts" / "CHAP2" / "main_finetune.py").resolve()
GT3X_FREQUENCY = 80
DOWN_SAMPLE_FREQUENCY = 10

# Try to import a helper for day completeness checks; fall back to naive logic if unavailable
try:
    sys.path.insert(0, str((REPO_ROOT / "scripts").resolve()))
    from helper_scripts.compute_vm_aug_predictions import check_day_completeness  # type: ignore
except Exception:
    check_day_completeness = None  # type: ignore

def run_preprocessing(main_csv_dir, preprocessed_dir, preprocess_script, gt3x_frequency, down_sample_frequency, skip_incomplete_days=False, verbose=False):
    print(f"[get_posture_predictions] Starting preprocessing for: {main_csv_dir}")
    os.makedirs(preprocessed_dir, exist_ok=True)
    temp_dir = None

    # Filter CSV files if skip_incomplete_days is True
    if skip_incomplete_days:
        temp_dir = tempfile.mkdtemp()
        csv_files = sorted(glob.glob(os.path.join(main_csv_dir, "*.csv*")))

        if not csv_files:
            print("[getpred_pipeline] ERROR: No day-level CSV files found to preprocess.")
            sys.exit(1)

        # New rule: keep only strict 24h days (00:00:00 to next 00:00:00) regardless of file count
        kept_files = []
        skipped_files = []

        if check_day_completeness is not None:
            for fp in csv_files:
                ok, start_time, end_time, reason = check_day_completeness(fp)
                # Accept the helper's completeness decision: starts at 00:00 and ends at 23:59:xx or next 00:00, duration 23–25h
                if ok:
                    kept_files.append(fp)
                else:
                    skipped_files.append((fp, start_time, end_time, reason))
        else:
            print("[getpred_pipeline] Warning: strict completeness checker unavailable; falling back to naive first/last skip logic.")
            kept_files = csv_files[1:-1] if len(csv_files) > 2 else []
            for idx, fp in enumerate(csv_files):
                if fp not in kept_files:
                    skipped_files.append((fp, None, None, 'fallback_naive_skip'))

        if skipped_files:
            print(f"[getpred_pipeline] Skipping {len(skipped_files)} incomplete day file(s):")
            for fp, st, et, rsn in skipped_files:
                base = os.path.basename(fp)
                print(f"  - {base} (start={st}, end={et}) reason={rsn}")

        if not kept_files:
            print("[getpred_pipeline] ERROR: All day-level files are incomplete under strict 24h rule. Nothing to preprocess.")
            # Clean up temp_dir before exiting
            shutil.rmtree(temp_dir)
            sys.exit(1)

        print(f"[getpred_pipeline] Keeping {len(kept_files)} file(s) after filtering incomplete days.")
        for file_path in kept_files:
            shutil.copy2(file_path, temp_dir)
        csv_source_dir = temp_dir
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
    try:
        subprocess.run(cmd, check=True)
    finally:
        # Clean up temporary directory if created, even when preprocessing fails.
        if temp_dir is not None and os.path.exists(temp_dir):
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


def _looks_like_chap1_day_dir(path: pathlib.Path) -> bool:
    """Return True for wrapper layout: <root>/<day>/<day>.h5."""
    if not path.exists() or not path.is_dir():
        return False
    day_dirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return any((day_dir / f"{day_dir.name}.h5").exists() for day_dir in day_dirs)


def _looks_like_chap1_subject_root(path: pathlib.Path) -> bool:
    """Return True for original CHAP1 layout: <root>/<subject>/<day>.h5."""
    if not path.exists() or not path.is_dir():
        return False
    subject_dirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for subject_dir in subject_dirs:
        if any(child.is_file() and child.suffix == ".h5" for child in subject_dir.iterdir()):
            return True
    return False


def _make_day_dir_view(source_dir: pathlib.Path, participant_id: str) -> tempfile.TemporaryDirectory:
    """Create <tmp>/<participant>/<day>/<day>.h5 from flat day-level .h5 files."""
    tmp = tempfile.TemporaryDirectory(prefix="chap1_predict_")
    tmp_participant_dir = pathlib.Path(tmp.name) / participant_id
    tmp_participant_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix == ".h5")
    if not h5_files:
        tmp.cleanup()
        raise FileNotFoundError(f"No flat .h5 files found under: {source_dir}")

    for h5_file in h5_files:
        day_name = h5_file.stem
        day_dir = tmp_participant_dir / day_name
        day_dir.mkdir(parents=True, exist_ok=True)
        link_path = day_dir / h5_file.name
        os.symlink(h5_file.resolve(), link_path)

    return tmp


def resolve_chap1_prediction_dir(preprocessed_dir, participant_id, verbose=False):
    """Normalize CHAP1 predict-only inputs without moving user data.

    CHAP1's make_predictions.py can consume either:
    - wrapper day layout: <participant>/<day>/<day>.h5
    - original root layout: <root>/<subject>/<day>.h5

    This helper also accepts parent roots and flat day-level .h5 directories by
    resolving or creating a temporary symlink view.
    """
    path = pathlib.Path(preprocessed_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed directory not found: {path}")

    participant_path = path / participant_id
    if participant_path.exists() and participant_path.is_dir():
        path = participant_path

    if _looks_like_chap1_day_dir(path) or _looks_like_chap1_subject_root(path):
        return str(path), None

    flat_h5_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".h5")
    if flat_h5_files:
        tmp = _make_day_dir_view(path, participant_id)
        resolved = pathlib.Path(tmp.name) / participant_id
        if verbose:
            print(f"[getpred_pipeline] Created temporary CHAP1 day-dir view: {resolved}")
        return str(resolved), tmp

    raise FileNotFoundError(
        "Could not find CHAP1 preprocessed .h5 files. Expected one of: "
        f"{path}/<day>/<day>.h5, {path}/<subject>/<day>.h5, or flat {path}/*.h5"
    )


def run_wrist_predictions(
    preprocessed_dir,
    predictions_dir,
    predict_script,
    participant_id,
    wrist_model,
    wrist_checkpoint,
    wrist_device,
    wrist_batch_size,
    wrist_num_workers,
    padding,
    wrist_include_probability=False,
    wrist_pin_mem=True,
    verbose=False,
    show_progress=False,
):
    print(f"[getpred_pipeline] Starting wrist predictions for: {preprocessed_dir}")
    os.makedirs(predictions_dir, exist_ok=True)
    cmd = [
        sys.executable, str(predict_script),
        "--predict_only",
        "--participant_id", participant_id,
        "--data_path", preprocessed_dir,
        "--model", wrist_model,
        "--make_prediction",
        "--device", wrist_device,
        "--batch_size", str(wrist_batch_size),
        "--num_workers", str(wrist_num_workers),
        "--prediction_dir", predictions_dir,
        "--padding", padding,
    ]
    if wrist_checkpoint:
        cmd += ["--eval", wrist_checkpoint]
    if wrist_include_probability:
        cmd.append("--include_probability")
    if wrist_pin_mem:
        cmd.append("--pin_mem")
    else:
        cmd.append("--no_pin_mem")
    if show_progress:
        cmd.append("--show_prediction_progress")
    if verbose:
        print(f"[getpred_pipeline] Running wrist predictions command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[getpred_pipeline] Wrist predictions complete for: {preprocessed_dir}")

if __name__ == "__main__":
    print("[getpred_pipeline] Script started.")
    parser = argparse.ArgumentParser(description="DeepPostures participant pipeline")
    parser.add_argument('--participant-id', type=str, required=True, help='Participant ID to process')
    parser.add_argument('--model', type=str, default='CHAP_A', help='Model subfolder to use for predictions (e.g., CHAP_A, CHAP_ALL_ADULTS)')
    parser.add_argument(
        '--padding',
        choices=('drop', 'zero', 'wrap'),
        default='wrap',
        help='Final partial-sequence handling for hip or wrist predictions (default: wrap)',
    )
    parser.add_argument('--skip-incomplete-days', action='store_true', help='Skip day-level CSVs that are not full days: start at 00:00:00 and end at 23:59:xx or next 00:00:00 (23–25 hours)')
    parser.add_argument('--model-root', type=str, default=None, help='Path to MSSE-2021 model root (overrides automatic detection)')
    parser.add_argument('--gt3x-frequency', type=int, default=GT3X_FREQUENCY, help=f'GT3X sample frequency (default: {GT3X_FREQUENCY})')
    parser.add_argument('--down-sample-frequency', type=int, default=DOWN_SAMPLE_FREQUENCY, help=f'Down-sample frequency (default: {DOWN_SAMPLE_FREQUENCY})')
    parser.add_argument('--pre-processed-dir', type=str, default=None, help='Override preprocessed output directory')
    parser.add_argument('--predictions-dir', type=str, default=None, help='Override predictions output directory')
    parser.add_argument('--posture-site', choices=['hip', 'wrist'], default='hip', help='Choose hip to keep current make_predictions.py flow or wrist to use CHAP2 main_finetune.py')
    parser.add_argument('--show-prediction-progress', action='store_true', dest='show_prediction_progress', help='Print wrist prediction batch/file progress when using --posture-site wrist')
    parser.add_argument('--no-show-prediction-progress', action='store_false', dest='show_prediction_progress', help='Disable wrist prediction batch/file progress output')
    parser.add_argument('--wrist-model', type=str, default='CHAP', help='Wrist model name passed to CHAP2 main_finetune.py (default: CHAP)')
    parser.add_argument('--wrist-checkpoint', type=str, default=None, help='Checkpoint path for wrist CHAP2 prediction (forwarded as --eval)')
    parser.add_argument('--wrist-device', type=str, default='cpu', help='Device for wrist CHAP2 prediction (e.g., cpu, cuda, mps)')
    parser.add_argument('--wrist-batch-size', type=int, default=40, help='Batch size for wrist CHAP2 prediction')
    parser.add_argument('--wrist-num-workers', type=int, default=0, help='DataLoader workers for wrist CHAP2 prediction')
    parser.add_argument('--wrist-include-probability', action='store_true', help='Include prob_sitting in wrist output CSVs')
    parser.add_argument('--wrist-pin-mem', action='store_true', help='Enable pin_memory for wrist CHAP2 DataLoader')
    parser.add_argument('--wrist-no-pin-mem', action='store_false', dest='wrist_pin_mem', help='Disable pin_memory for wrist CHAP2 DataLoader')
    parser.set_defaults(wrist_pin_mem=True, show_prediction_progress=True)
    parser.add_argument('--preprocess-only', action='store_true', help='Only run preprocessing and exit')
    parser.add_argument('--predict-only', action='store_true', help='Only run predictions (requires preprocessed data to exist)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging for subprocess commands')
    args = parser.parse_args()

    participant_id = args.participant_id
    model = args.model
    padding = args.padding
    skip_incomplete_days = args.skip_incomplete_days
    posture_site = args.posture_site
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
    if posture_site == 'wrist':
        predictions_dir = args.predictions_dir if args.predictions_dir else str((REPO_ROOT / "data" / "predictions").resolve())
    else:
        predictions_dir = args.predictions_dir if args.predictions_dir else str((REPO_ROOT / "data" / "predictions" / participant_id).resolve())

    # Resolve model root and script locations
    model_root = find_model_root(args.model_root)
    PREPROCESS_SCRIPT = (model_root / "pre_process_data.py").resolve()
    PREDICT_SCRIPT = (model_root / "make_predictions.py").resolve()

    try:
        if args.predict_only and not args.preprocess_only:
            # Only run predictions; assume preprocessed data exists
            if posture_site == 'wrist':
                run_wrist_predictions(
                    preprocessed_dir,
                    predictions_dir,
                    WRIST_PREDICT_SCRIPT,
                    participant_id,
                    wrist_model=args.wrist_model,
                    wrist_checkpoint=args.wrist_checkpoint,
                    wrist_device=args.wrist_device,
                    wrist_batch_size=args.wrist_batch_size,
                    wrist_num_workers=args.wrist_num_workers,
                    padding=padding,
                    wrist_include_probability=args.wrist_include_probability,
                    wrist_pin_mem=args.wrist_pin_mem,
                    verbose=args.verbose,
                    show_progress=args.show_prediction_progress,
                )
            else:
                prediction_input_dir, prediction_tmp = resolve_chap1_prediction_dir(
                    preprocessed_dir,
                    participant_id,
                    verbose=args.verbose,
                )
                try:
                    run_predictions(prediction_input_dir, predictions_dir, PREDICT_SCRIPT, model, padding, verbose=args.verbose)
                finally:
                    if prediction_tmp is not None:
                        prediction_tmp.cleanup()
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
            if posture_site == 'wrist':
                run_wrist_predictions(
                    preprocessed_dir,
                    predictions_dir,
                    WRIST_PREDICT_SCRIPT,
                    participant_id,
                    wrist_model=args.wrist_model,
                    wrist_checkpoint=args.wrist_checkpoint,
                    wrist_device=args.wrist_device,
                    wrist_batch_size=args.wrist_batch_size,
                    wrist_num_workers=args.wrist_num_workers,
                    padding=padding,
                    wrist_include_probability=args.wrist_include_probability,
                    wrist_pin_mem=args.wrist_pin_mem,
                    verbose=args.verbose,
                    show_progress=args.show_prediction_progress,
                )
            else:
                prediction_input_dir, prediction_tmp = resolve_chap1_prediction_dir(
                    preprocessed_dir,
                    participant_id,
                    verbose=args.verbose,
                )
                try:
                    run_predictions(prediction_input_dir, predictions_dir, PREDICT_SCRIPT, model, padding, verbose=args.verbose)
                finally:
                    if prediction_tmp is not None:
                        prediction_tmp.cleanup()

        print(f"[] Pipeline complete. Predictions saved in: {predictions_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[getpred_pipeline] External script failed with return code {e.returncode}")
        raise



#python scripts/posture_library/MSSE-2021/pre_process_data.py  --gt3x-dir data/processed/62161 --pre-processed-dir data/preprocessed/ --gt3x-frequency 80 --down-sample-frequency 10
# python scripts/get_posture_predictions.py --participant-id 62161 --skip-incomplete-days --model CHAP_ALL_ADULTS
