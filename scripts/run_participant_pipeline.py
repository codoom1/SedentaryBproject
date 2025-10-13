#!/usr/bin/env python3
"""
Lightweight pipeline runner (Python) for one participant.

Steps:
 1) Optionally download participant archive and extract using functions in download_participantPAMdata.py
 2) Run sleep classification by-day using scripts/sleep_scripts/sleep_classify.py
 3) Run posture prediction using scripts/get_posture_predictions.py

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
MODEL_ROOT_NEW = REPO_ROOT / "scripts" / "posture_library" / "MSSE-2021"
MODEL_ROOT_SPACE = REPO_ROOT / "scripts" / "posture library" / "MSSE-2021"
MODEL_ROOT_LEGACY = REPO_ROOT / "MSSE-2021"


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


def main():
    parser = argparse.ArgumentParser(description="Run participant pipeline: download -> sleep (by-day) -> posture")
    parser.add_argument("--participant-id", required=True, help="Participant ID (SEQN)")
    parser.add_argument("--cycle", default="2011-12", help="NHANES cycle (2011-12 or 2013-14)")
    parser.add_argument("--download", action="store_true", help="Download and extract participant archive before running (default: false)")
    parser.add_argument("--raw-dest", default=str(REPO_ROOT / "data" / "raw"), help="Root directory to store extracted raw data (default: data/raw)")
    parser.add_argument("--sleep-output", default=str(REPO_ROOT / "data" / "sleep_predictions"), help="Directory for sleep outputs")
    parser.add_argument("--posture-model", default="CHAP_ALL_ADULTS", help="Model to pass to posture script")
    parser.add_argument("--skip-incomplete-days-sleep", action="store_true", help="Skip incomplete days when running sleep (if supported)")
    parser.add_argument("--skip-incomplete-days-posture", action="store_true", help="Skip incomplete days when running posture predictions")
    parser.add_argument("--sleep-conda-env", type=str, default=None, help="Conda env name to run the sleep step in (e.g., sklearn023). If omitted uses current Python")
    parser.add_argument("--posture-conda-env", type=str, default=None, help="Conda env name to run the posture step in (e.g., deepposture). If omitted uses current Python")
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
    if args.skip_incomplete_days_sleep:
        base_sleep.append("--skip-incomplete-days")
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
    if model_root_arg:
        base_posture += ["--model-root", model_root_arg]
    if args.skip_incomplete_days_posture:
        base_posture.append("--skip-incomplete-days")
    if args.posture_conda_env:
        posture_cmd = ["conda", "run", "-n", args.posture_conda_env, "--no-capture-output", "python"] + base_posture
    else:
        posture_cmd = [sys.executable] + base_posture

    # Dry run: print commands and validate scripts
    if args.dry_run:
        logger.info("Dry-run: validating commands with --help")
        run_cmd([sys.executable, str(SLEEP_SCRIPT)], dry_run=True)
        run_cmd([sys.executable, str(POSTURE_SCRIPT)], dry_run=True)
        logger.info("Dry-run complete. Commands to be run:")
        logger.info("sleep_cmd: %s", " ".join(sleep_cmd))
        logger.info("posture_cmd: %s", " ".join(posture_cmd))
        return

    # Execute sleep
    logger.info("Running sleep classification for participant %s", pid)
    run_cmd(sleep_cmd, dry_run=False)

    # Execute posture
    logger.info("Running posture prediction for participant %s", pid)
    run_cmd(posture_cmd, dry_run=False)

    logger.info("Pipeline finished. Sleep outputs: %s; Posture outputs under data/predictions/<participant_id>", args.sleep_output)


if __name__ == '__main__':
    main()


# python scripts/run_participant_pipeline.py \
#   --participant-id 62161 \
#   --cycle 2011-12 \
#   --posture-model CHAP_ALL_ADULTS \
#   --skip-incomplete-days-sleep \
#   --skip-incomplete-days-posture \
#   --sleep-conda-env sklearn023 \
#   --posture-conda-env deepposture