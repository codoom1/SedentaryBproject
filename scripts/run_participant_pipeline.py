#!/usr/bin/env python3
"""
Lightweight pipeline runner (Python) for one participant.

Steps:
 1) Optionally download participant archive and extract using functions in download_participantPAMdata.py
 2) Run sleep classification by-day using scripts/sleep_scripts/sleep_classify.py
 3) Run posture prediction using scripts/get_posture_predictions.py
 4) Optionally export durable 10-second Parquet files
 5) Optionally summarize sleep/posture predictions to the requested epoch

This script intentionally does not modify existing scripts. It calls them via subprocess.
It provides a --dry-run mode which only prints the commands and validates the called scripts with --help (safe).

Usage examples:
  python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --dry-run
  python scripts/run_participant_pipeline.py --participant-id 62161 --cycle 2011-12 --download --skip-incomplete-days-sleep --skip-incomplete-days-posture --sleep-conda-env sklearn023 --posture-conda-env deepposture

"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Paths to scripts
REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_participantPAMdata.py"
SLEEP_SCRIPT = REPO_ROOT / "scripts" / "sleep_scripts" / "sleep_classify.py"
POSTURE_SCRIPT = REPO_ROOT / "scripts" / "get_posture_predictions.py"
SUMMARIZER_SCRIPT = REPO_ROOT / "scripts" / "summarize_participant.py"
EXPORT_10S_SCRIPT = REPO_ROOT / "scripts" / "export_10s_dataset.py"
MODEL_ROOT_NEW = REPO_ROOT / "scripts" / "posture_library" / "MSSE-2021"
MODEL_ROOT_SPACE = REPO_ROOT / "scripts" / "posture library" / "MSSE-2021"
MODEL_ROOT_LEGACY = REPO_ROOT / "MSSE-2021"

try:
    sys.path.insert(0, str((REPO_ROOT / "scripts").resolve()))
    from helper_scripts.compute_vm_aug_predictions import check_day_completeness  # type: ignore
except Exception:
    check_day_completeness = None  # type: ignore


def run_cmd(cmd, dry_run=False):
    """Run a command (list) using the same Python executable; in dry_run, show and run with --help to validate."""
    logger.info("CMD: %s", " ".join(map(str, cmd)))
    if dry_run:
        # For safety, run the target script with --help to ensure it's callable
        help_cmd = list(cmd)
        # Replace arguments with --help if executable is the python script
        # If the last element is a script path, append --help
        if any(str(x).endswith('.py') for x in help_cmd):
            help_cmd.append("--help")
        try:
            subprocess.run(help_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Help command failed: %s", e)
            raise
        return

    # Real execution
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %s", e.returncode)
        raise


def download_and_extract(participant_id: str, cycle: str, dest_root: Path, dry_run: bool = False):
    """Use download_participant_archive_only to fetch and extract the archive into dest_root.
    This imports the helper function directly and calls it to avoid double-download. If dry_run is True,
    this function will only print what it would do.
    """
    logger.info("Will download participant %s (cycle %s) into %s", participant_id, cycle, dest_root)

    if dry_run:
        logger.info("Dry-run: skipping actual download")
        return None

    # Use subprocess to call the helper script (keeps behavior consistent)
    dest_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(DOWNLOAD_SCRIPT), participant_id, cycle, str(dest_root), "--extract", "--remove-archive"]
    # The helper prints the result; run it and parse nothing here
    run_cmd(cmd, dry_run=False)

    # After extraction, the helper will create dest_root / participant_id
    extracted_dir = dest_root / participant_id
    if not extracted_dir.exists():
        logger.warning("Expected extracted directory not found: %s", extracted_dir)
        return None

    return extracted_dir


def processed_days_all_incomplete(participant_id: str) -> bool:
    """Return True only when processed day CSVs exist and all fail the completeness check."""
    processed_dir = REPO_ROOT / "data" / "processed" / participant_id
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files or check_day_completeness is None:
        return False

    saw_complete = False
    for csv_file in csv_files:
        try:
            ok, _start_time, _end_time, _reason = check_day_completeness(str(csv_file))
        except Exception:
            return False
        if ok:
            saw_complete = True
            break

    return not saw_complete


def default_summary_out(participant_id: str, skip_sleep: bool, summary_epoch: str) -> Path:
    summary_type = "posture_only" if skip_sleep else "sleep_posture"
    return REPO_ROOT / "data" / "summaries" / f"{participant_id}_{summary_type}_{summary_epoch}_epoch.csv"


def main():
    parser = argparse.ArgumentParser(description="Run participant pipeline: download -> sleep (by-day) -> posture -> optional summary")
    parser.add_argument("--participant-id", required=True, help="Participant ID (SEQN)")
    parser.add_argument("--cycle", default="2011-12", help="NHANES cycle (2011-12 or 2013-14)")
    parser.add_argument("--download", action="store_true", help="Download and extract participant archive before running (default: false)")
    parser.add_argument("--raw-dest", default=str(REPO_ROOT / "data" / "raw"), help="Root directory to store extracted raw data (default: data/raw)")
    parser.add_argument("--sleep-output", default=str(REPO_ROOT / "data" / "sleep_predictions"), help="Directory for sleep outputs")
    parser.add_argument("--skip-sleep", action="store_true", help="Skip sleep predictions entirely and run posture only")
    parser.add_argument("--posture-model", default="CHAP_ALL_ADULTS", help="Model to pass to posture script")
    parser.add_argument(
        "--skip-incomplete-days-sleep",
        action="store_true",
        help="Sleep: skip any calendar day without a full day of data (start 00:00; end 23:59:xx or next 00:00; 23–25h)"
    )
    parser.add_argument(
        "--skip-incomplete-days-posture",
        action="store_true",
        help="Posture: skip day-level CSVs that are not full days (start 00:00; end 23:59:xx or next 00:00; 23–25h)"
    )
    parser.add_argument("--sleep-tmp-dir", type=str, default=None, help="Temp directory to use for the sleep step (forwarded as --tmp-dir)")
    parser.add_argument("--sleep-conda-env", type=str, default=None, help="Conda env name to run the sleep step in (e.g., sklearn023). If omitted uses current Python")
    parser.add_argument("--posture-conda-env", type=str, default=None, help="Conda env name to run the posture step in (e.g., deepposture). If omitted uses current Python")
    parser.add_argument("--posture-site", choices=["hip", "wrist"], default="hip", help="Choose hip to keep the current make_predictions.py flow or wrist to use CHAP2 main_finetune.py")
    parser.add_argument("--posture-show-prediction-progress", action="store_true", dest="posture_show_prediction_progress", help="Forward --show-prediction-progress to get_posture_predictions (wrist mode)")
    parser.add_argument("--posture-no-show-prediction-progress", action="store_false", dest="posture_show_prediction_progress", help="Disable wrist prediction progress output")
    parser.add_argument("--posture-wrist-model", default="CHAP", help="Forward --wrist-model to get_posture_predictions")
    parser.add_argument("--posture-wrist-checkpoint", default=None, help="Forward --wrist-checkpoint to get_posture_predictions")
    parser.add_argument("--posture-wrist-device", default="cpu", help="Forward --wrist-device to get_posture_predictions")
    parser.add_argument("--posture-wrist-batch-size", type=int, default=40, help="Forward --wrist-batch-size to get_posture_predictions")
    parser.add_argument("--posture-wrist-num-workers", type=int, default=0, help="Forward --wrist-num-workers to get_posture_predictions")
    parser.add_argument("--posture-padding", choices=("drop", "zero", "wrap"), default="wrap", help="Handle the final partial posture sequence")
    parser.add_argument("--posture-wrist-include-probability", action="store_true", help="Include prob_sitting in wrist CSVs and 10-second Parquet export")
    parser.add_argument("--posture-wrist-pin-mem", action="store_true", help="Enable pin_memory for wrist prediction DataLoader")
    parser.add_argument("--posture-wrist-no-pin-mem", action="store_false", dest="posture_wrist_pin_mem", help="Disable pin_memory for wrist prediction DataLoader")
    parser.set_defaults(posture_wrist_pin_mem=True, posture_show_prediction_progress=True)
    parser.add_argument("--sleep-only-dates", nargs='*', default=None, help="Optional list of dates (YYYY-MM-DD) to process for sleep --by-day; useful for debugging a problematic day")
    parser.add_argument("--sleep-day-chunks", type=int, default=None, help="Override number of chunks per day for sleep (forwarded to --day-chunks)")
    parser.add_argument("--sleep-chunk-overlap", type=int, default=None, help="Override chunk overlap seconds for sleep (forwarded to --chunk-overlap-seconds)")
    parser.add_argument("--sleep-debug", action='store_true', help="Enable debug logging for the sleep step (forwards --debug)")
    parser.add_argument("--sleep-swan-timeout", type=int, default=None, help="Timeout in seconds for each SWaN chunk (forwards --swan-timeout-seconds)")
    parser.add_argument("--sleep-swan-use-worker", action='store_true', help="Run SWaN first pass via isolated worker with timeout (forwards --swan-use-worker)")
    parser.add_argument("--sleep-swan-worker-fallback", action='store_true', help="Run SWaN normally first; if it times out, retry that chunk in an isolated worker")
    parser.add_argument("--sleep-swan-retries", type=int, default=None, help="Number of retry attempts per chunk (forwards --swan-retries)")
    parser.add_argument("--sleep-swan-retry-timeout", type=int, default=None, help="Timeout seconds for retry attempts (forwards --swan-retry-timeout-seconds)")
    parser.add_argument("--sleep-max-subdivision-depth", type=int, default=None, help="Maximum depth for recursive chunk subdivision (forwards --max-subdivision-depth)")
    parser.add_argument("--sleep-min-chunk-minutes", type=int, default=None, help="Minimum chunk duration in minutes before subdivision stops (forwards --min-chunk-minutes)")
    parser.add_argument("--summarize", action="store_true", help="After posture prediction, summarize outputs to epoch-level CSV rows")
    parser.add_argument("--summary-epoch", default="1h", help="Summary epoch for --summarize, e.g. 10s, 30s, 1min, 5m, 30m, or 1h")
    parser.add_argument("--epoch-columns", nargs="*", default=None, help="Optional coarser epoch label columns to add, e.g. --epoch-columns 20m 30m. Use Hour for hourly grouping.")
    parser.add_argument("--summary-out", default=None, help="Optional output CSV for --summarize. Defaults to data/summaries/<ID>_<type>_<summary_epoch>_epoch.csv")
    parser.add_argument("--export-10s", action="store_true", help="Export durable participant-day 10-second Parquet files after prediction")
    parser.add_argument("--export-10s-output-root", default=str(REPO_ROOT / "data" / "epoch_10s"), help="Root directory for durable 10-second Parquet files")
    parser.add_argument("--export-10s-include-sleep-probabilities", action="store_true", help="Include SWaN probability columns in the 10-second Parquet files")
    parser.add_argument("--export-10s-overwrite", action="store_true", help="Overwrite existing participant-day 10-second Parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: validate commands but do not execute heavy tasks")

    args = parser.parse_args()

    pid = args.participant_id
    cycle = args.cycle
    raw_dest_root = Path(args.raw_dest)

    # Step 1: download (optional)
    extracted_dir = None
    if args.download:
        extracted_dir = download_and_extract(pid, cycle, raw_dest_root / cycle, dry_run=args.dry_run)
    else:
        # If not downloading, assume data is already present under data/raw/<cycle>/<participant_id>
        candidate = raw_dest_root / cycle / pid
        if candidate.exists():
            extracted_dir = candidate
        else:
            logger.warning("No extracted raw data found at %s. You can pass --download to fetch it.", candidate)

    # Build sleep command (optionally run inside a specified conda env)
    base_sleep = [str(SLEEP_SCRIPT), "--participant-id", pid, "--data-dir", str(raw_dest_root / cycle), "--output-dir", args.sleep_output, "--by-day"]
    if args.sleep_tmp_dir:
        base_sleep += ["--tmp-dir", args.sleep_tmp_dir]
    if args.skip_incomplete_days_sleep:
        base_sleep.append("--skip-incomplete-days")
    if args.sleep_only_dates:
        base_sleep += ["--only-dates"] + args.sleep_only_dates
    if args.sleep_day_chunks is not None:
        base_sleep += ["--day-chunks", str(args.sleep_day_chunks)]
    if args.sleep_chunk_overlap is not None:
        base_sleep += ["--chunk-overlap-seconds", str(args.sleep_chunk_overlap)]
    if args.sleep_debug:
        base_sleep.append("--debug")
    if args.sleep_swan_timeout is not None:
        base_sleep += ["--swan-timeout-seconds", str(args.sleep_swan_timeout)]
    if args.sleep_swan_use_worker:
        base_sleep.append("--swan-use-worker")
    if args.sleep_swan_worker_fallback:
        base_sleep.append("--swan-worker-fallback")
    if args.sleep_swan_retries is not None:
        base_sleep += ["--swan-retries", str(args.sleep_swan_retries)]
    if args.sleep_swan_retry_timeout is not None:
        base_sleep += ["--swan-retry-timeout-seconds", str(args.sleep_swan_retry_timeout)]
    if args.sleep_max_subdivision_depth is not None:
        base_sleep += ["--max-subdivision-depth", str(args.sleep_max_subdivision_depth)]
    if args.sleep_min_chunk_minutes is not None:
        base_sleep += ["--min-chunk-minutes", str(args.sleep_min_chunk_minutes)]
    if args.sleep_conda_env:
        sleep_cmd = ["conda", "run", "-n", args.sleep_conda_env, "--no-capture-output", "python"] + base_sleep
    else:
        sleep_cmd = [sys.executable] + base_sleep

    # Build posture command (optionally run inside a specified conda env)
    # Prefer the in-repo model root under scripts/posture_library/MSSE-2021; fallback to legacy path
    model_root_arg = None
    if MODEL_ROOT_NEW.exists():
        model_root_arg = str(MODEL_ROOT_NEW)
    elif MODEL_ROOT_SPACE.exists():
        model_root_arg = str(MODEL_ROOT_SPACE)
    elif MODEL_ROOT_LEGACY.exists():
        model_root_arg = str(MODEL_ROOT_LEGACY)

    base_posture = [str(POSTURE_SCRIPT), "--participant-id", pid, "--model", args.posture_model]
    base_posture += ["--posture-site", args.posture_site]
    if args.posture_show_prediction_progress:
        base_posture.append("--show-prediction-progress")
    base_posture += ["--wrist-model", args.posture_wrist_model]
    if args.posture_wrist_checkpoint:
        base_posture += ["--wrist-checkpoint", args.posture_wrist_checkpoint]
    base_posture += ["--wrist-device", args.posture_wrist_device]
    base_posture += ["--wrist-batch-size", str(args.posture_wrist_batch_size)]
    base_posture += ["--wrist-num-workers", str(args.posture_wrist_num_workers)]
    base_posture += ["--padding", args.posture_padding]
    if args.posture_wrist_include_probability:
        base_posture.append("--wrist-include-probability")
    if args.posture_wrist_pin_mem:
        base_posture.append("--wrist-pin-mem")
    else:
        base_posture.append("--wrist-no-pin-mem")
    if model_root_arg:
        base_posture += ["--model-root", model_root_arg]
    if args.skip_incomplete_days_posture:
        base_posture.append("--skip-incomplete-days")
    if args.posture_conda_env:
        posture_cmd = ["conda", "run", "-n", args.posture_conda_env, "--no-capture-output", "python"] + base_posture
    else:
        posture_cmd = [sys.executable] + base_posture

    summary_out = Path(args.summary_out) if args.summary_out else default_summary_out(pid, args.skip_sleep, args.summary_epoch)
    summary_cmd = [
        sys.executable,
        str(SUMMARIZER_SCRIPT),
        "--participant-id", pid,
        "--model", args.posture_model,
        "--out", str(summary_out),
        "--summary-epoch", args.summary_epoch,
        "--dataset", cycle,
    ]
    if args.epoch_columns:
        summary_cmd += ["--epoch-columns"] + args.epoch_columns
    if args.skip_sleep:
        summary_cmd.append("--skip-sleep")

    export_10s_cmd = [
        sys.executable,
        str(EXPORT_10S_SCRIPT),
        "--participant-id", pid,
        "--dataset", cycle,
        "--model", args.posture_model,
        "--output-root", args.export_10s_output_root,
        "--data-root", str(REPO_ROOT / "data"),
    ]
    if args.skip_sleep:
        export_10s_cmd.append("--skip-sleep")
    if args.export_10s_include_sleep_probabilities:
        export_10s_cmd.append("--include-sleep-probabilities")
    if args.export_10s_overwrite:
        export_10s_cmd.append("--overwrite")

    # Dry run: print commands and validate scripts
    if args.dry_run:
        logger.info("Dry-run: validating commands with --help")
        if not args.skip_sleep:
            run_cmd([sys.executable, str(SLEEP_SCRIPT)], dry_run=True)
        run_cmd([sys.executable, str(POSTURE_SCRIPT)], dry_run=True)
        if args.export_10s:
            run_cmd([sys.executable, str(EXPORT_10S_SCRIPT)], dry_run=True)
        if args.summarize:
            run_cmd([sys.executable, str(SUMMARIZER_SCRIPT)], dry_run=True)
        logger.info("Dry-run complete. Commands to be run:")
        if args.skip_sleep:
            logger.info("sleep_cmd: [skipped]")
        else:
            logger.info("sleep_cmd: %s", " ".join(sleep_cmd))
        logger.info("posture_cmd: %s", " ".join(posture_cmd))
        if args.export_10s:
            logger.info("export_10s_cmd: %s", " ".join(export_10s_cmd))
        if args.summarize:
            logger.info("summary_cmd: %s", " ".join(summary_cmd))
        return

    # Execute sleep
    if args.skip_sleep:
        logger.info("Skipping sleep classification for participant %s (--skip-sleep)", pid)
    else:
        logger.info("Running sleep classification for participant %s", pid)
        try:
            run_cmd(sleep_cmd, dry_run=False)
        except subprocess.CalledProcessError as e:
            if e.returncode == 2 and args.skip_incomplete_days_sleep:
                # Sleep script signals no complete days when using by-day + skip flag
                logger.warning(
                    "Participant %s: all days are incomplete per completeness rule (start 00:00; end ~23:59 or next 00:00; 23–25h). Skipping posture and ending this participant run.",
                    pid
                )
                # End gracefully so batch processing can move on to the next participant
                return
            # Other failures: propagate
            raise

    # Execute posture
    logger.info("Running posture prediction for participant %s", pid)
    try:
        run_cmd(posture_cmd, dry_run=False)
    except subprocess.CalledProcessError:
        if args.skip_incomplete_days_posture and processed_days_all_incomplete(pid):
            logger.warning(
                "Participant %s: all processed posture day files are incomplete under the completeness rule. "
                "Skipping posture and ending this participant run.",
                pid,
            )
            return
        raise

    if args.export_10s:
        logger.info("Exporting durable 10-second Parquet files for participant %s", pid)
        run_cmd(export_10s_cmd, dry_run=False)

    if args.summarize:
        logger.info("Summarizing participant %s to %s", pid, summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(summary_cmd, dry_run=False)

    logger.info(
        "Pipeline finished. Sleep outputs: %s; Posture outputs under data/predictions/<participant_id>",
        args.sleep_output,
    )


if __name__ == '__main__':
    main()


# python scripts/run_participant_pipeline.py \
#   --participant-id 62161 \
#   --cycle 2011-12 \
#   --posture-model CHAP_ALL_ADULTS \
#   --skip-incomplete-days-sleep \
#   --skip-incomplete-days-posture \
#   --sleep-conda-env sklearn023 \
#   --posture-conda-env deepposture\
#   --download\
#   --tmp-dir data/tmp/62161


## Wrist model compatible example usage
# python scripts/run_participant_pipeline.py \
#   --participant-id 62193 \
#   --cycle 2011-12 \
#   --skip-sleep \
#   --posture-site wrist \
#   --posture-wrist-model CHAP \
#   --posture-show-prediction-progress \
#   --posture-conda-env deepposture
#   --skip-incomplete-days-posture


 #python scripts/run_participant_pipeline.py  --participant-id 62193  --cycle 2011-12 --skip-sleep --posture-site wrist --posture-show-prediction-progress  --posture-conda-env deepposture --skip-incomplete-days-posture --download
