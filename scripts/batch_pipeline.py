#!/usr/bin/env python3
"""
Batch runner to process lists of participants using two scripts:
- run_participant_pipeline.py: download (optional) -> sleep -> posture
- summarize_participant.py: per-participant hourly summary

Batch file format:
- CSV or TXT. Each non-empty line must be: "cycle,participant_id"

For each participant the script:
- runs `run_participant_pipeline.py` (optionally with --download and conda envs)
- runs `summarize_participant.py` to create a per-participant summary CSV
- appends the per-participant summary to a master CSV
- optionally deletes participant-specific data directories to free disk (disable with --no-cleanup)
## python scripts/batch_pipeline.py  --batch-file batches/batch_1.txt  --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --download 

Usage:
    python scripts/batch_pipeline.py \
        --batch-file batches/batch_1.txt \
        --model CHAP_ALL_ADULTS \
        --sleep-conda-env sklearn023 \
        --posture-conda-env deepposture \
        --download

"""
import argparse
import subprocess
from pathlib import Path
import shutil
import csv
import sys
import logging
from typing import Optional, List, Tuple
import tempfile
import gzip

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def run_cmd(cmd, dry_run=False):
    logger.info('CMD: %s', ' '.join(cmd))
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        # try to include any stderr text if available
        try:
            stderr = e.stderr.decode('utf-8') if e.stderr else None
        except Exception:
            stderr = None
        logger.error('Command failed: %s (returncode=%s)\n%s', ' '.join(cmd), e.returncode, stderr or '')
        return e.returncode


def append_summary(master_csv: Path, part_csv: Path, compress: bool = False):
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    if not part_csv.exists():
        logger.warning('Per-participant summary not found: %s', part_csv)
        return

    if not master_csv.exists():
        # first file: copy with header (compressed or not)
        if compress:
            # Write entire participant file into a new gzip master
            with part_csv.open('r', newline='') as src, gzip.open(master_csv, 'wt', newline='') as dst:  # type: ignore
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(part_csv, master_csv)
        logger.info('Created master summary: %s', master_csv)
    else:
        if compress:
            # Append rows skipping header into gzip file
            with part_csv.open('r', newline='') as src, gzip.open(master_csv, 'at', newline='') as dst:  # type: ignore
                reader = csv.reader(src)
                writer = csv.writer(dst)
                count = 0
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    writer.writerow(row)
                    count += 1
            logger.info('Appended %s rows (gz) from %s to %s', count, part_csv, master_csv)
        else:
            if pd is not None:
                # append without header using pandas
                df = pd.read_csv(part_csv)  # type: ignore
                df.to_csv(master_csv, mode='a', header=False, index=False)  # type: ignore
                logger.info('Appended %s rows from %s to %s', len(df), part_csv, master_csv)
            else:
                # Fallback: append CSV rows manually, skipping header
                with part_csv.open('r', newline='') as src, master_csv.open('a', newline='') as dst:
                    reader = csv.reader(src)
                    writer = csv.writer(dst)
                    count = 0
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        writer.writerow(row)
                        count += 1
                logger.info('Appended %s rows from %s to %s (fallback)', count, part_csv, master_csv)


def cleanup_participant(participant_id: str, cycle: str, model: str):
    repo = Path.cwd()
    paths = [
        repo / 'data' / 'raw' / cycle / participant_id,
        repo / 'data' / 'processed' / participant_id,
        repo / 'data' / 'preprocessed' / participant_id,
        repo / 'data' / 'sleep_predictions' / participant_id,
        repo / 'data' / 'predictions' / participant_id,
    ]
    for p in paths:
        if p.exists():
            try:
                shutil.rmtree(p)
                logger.info('Deleted: %s', p)
            except Exception as e:
                logger.warning('Failed to delete %s: %s', p, e)


def process_batch(batch_file: Path, model: str, master_out: Path, sleep_conda_env: Optional[str] = None,
                  posture_conda_env: Optional[str] = None, download: bool = False, dry_run: bool = False,
                  keep_on_error: bool = False,
                  do_cleanup: bool = True,
                  keep_participant_summaries: bool = False,
                  compress_master: bool = True):

    repo = Path.cwd()
    run_pipeline_py = repo / 'scripts' / 'run_participant_pipeline.py'
    summarizer_py = repo / 'scripts' / 'summarize_participant.py'

    if not batch_file.exists():
        logger.error('Batch file not found: %s', batch_file)
        return

    rows: List[Tuple[str, str]] = []
    with open(batch_file, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            # allow comments
            if isinstance(r[0], str) and r[0].strip().startswith('#'):
                continue
            if len(r) < 2:
                logger.error('Invalid row in batch file (expected "cycle,participant_id"): %s', r)
                continue
            rows.append((r[0].strip(), r[1].strip()))

    for idx, (cycle, pid) in enumerate(rows, 1):
        logger.info('[%d/%d] Processing participant %s (cycle=%s)', idx, len(rows), pid, cycle)

        # If raw data is missing and we're not passing --download, warn the user; otherwise
        # let run_participant_pipeline.py handle the download when called with --download.
        raw_dir = repo / 'data' / 'raw' / cycle / pid
        if not raw_dir.exists() and not download:
            logger.warning('No extracted raw data found at %s. Pass --download to fetch it before running.', raw_dir)

        # Build pipeline command
        pipeline_cmd = [sys.executable, str(run_pipeline_py), '--participant-id', pid, '--cycle', cycle, '--posture-model', model]
        # pass env names to the participant pipeline so it runs steps in the right conda envs
        if sleep_conda_env:
            pipeline_cmd += ['--sleep-conda-env', sleep_conda_env]
        if posture_conda_env:
            pipeline_cmd += ['--posture-conda-env', posture_conda_env]
        if download:
            pipeline_cmd += ['--download']
        # Always skip incomplete days for both sleep and posture
        pipeline_cmd += ['--skip-incomplete-days-sleep', '--skip-incomplete-days-posture']

        rc = run_cmd(pipeline_cmd, dry_run=dry_run)
        if rc != 0:
            logger.error('Pipeline failed for %s (rc=%s)', pid, rc)
            if not keep_on_error:
                logger.info('Skipping summarization and cleanup for %s', pid)
                continue

        # Build summarizer command with either a temp output or a persistent file
        if keep_participant_summaries:
            part_out = repo / 'data' / 'summaries' / f"{pid}_sleep_posture_hourly.csv"
            summarizer_cmd = [sys.executable, str(summarizer_py), '--participant-id', pid, '--model', model, '--out', str(part_out)]

            if dry_run:
                run_cmd(summarizer_cmd, dry_run=True)
                logger.info('Dry-run: skipping append and cleanup for %s', pid)
                continue

            rc2 = run_cmd(summarizer_cmd, dry_run=dry_run)
            if rc2 != 0:
                logger.error('Summarizer failed for %s (rc=%s)', pid, rc2)
                if not keep_on_error:
                    continue
            append_summary(master_out, part_out, compress=compress_master)
        else:
            # Use a temporary directory to avoid persisting per-participant summaries
            with tempfile.TemporaryDirectory() as tmpdir:
                part_out = Path(tmpdir) / f"{pid}.csv"
                summarizer_cmd = [sys.executable, str(summarizer_py), '--participant-id', pid, '--model', model, '--out', str(part_out)]

                if dry_run:
                    run_cmd(summarizer_cmd, dry_run=True)
                    logger.info('Dry-run: skipping append and cleanup for %s', pid)
                    continue

                rc2 = run_cmd(summarizer_cmd, dry_run=dry_run)
                if rc2 != 0:
                    logger.error('Summarizer failed for %s (rc=%s)', pid, rc2)
                    if not keep_on_error:
                        continue

                append_summary(master_out, part_out, compress=compress_master)

        # Cleanup participant directories to save disk (unless disabled)
        if do_cleanup:
            cleanup_participant(pid, cycle or 'unknown_cycle', model)

    logger.info('Batch processing complete. Master summary at: %s', master_out)


def main():
    parser = argparse.ArgumentParser(description='Run batch pipeline for many participants')
    parser.add_argument('--batch-file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--master-out', default='data/summaries/batch_sleep_posture_hourly.csv.gz')
    parser.add_argument('--sleep-conda-env', default=None)
    parser.add_argument('--posture-conda-env', default=None)
    parser.add_argument('--download', action='store_true', help='Run with --download to fetch raw archives')
    parser.add_argument('--dry-run', action='store_true', help='Print commands but do not execute')
    parser.add_argument('--keep-on-error', action='store_true', help='Do not stop on participant error; skip summarization/cleanup')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not delete participant data after summarization')
    parser.add_argument('--keep-participant-summaries', action='store_true', help='Keep per-participant hourly summary CSVs (default: not kept)')
    # Compression flags (default: compress master). --no-compress-master overrides.
    parser.add_argument('--compress-master', dest='compress_master', action='store_true', help='Compress the master CSV (gzip). Default behavior.')
    parser.add_argument('--no-compress-master', dest='compress_master', action='store_false', help='Do not compress the master CSV.')
    parser.set_defaults(compress_master=True)

    args = parser.parse_args()

    # Normalize master_out extension when compressing by default
    master_out_path = Path(args.master_out)
    if args.compress_master and master_out_path.suffix != '.gz':
        # If user supplied a non-gz path but compression is enabled, append .gz for clarity
        master_out_path = Path(str(master_out_path) + '.gz')
        logger.info('Using compressed master output: %s', master_out_path)

    process_batch(Path(args.batch_file), args.model, master_out_path,
                  sleep_conda_env=args.sleep_conda_env,
                  posture_conda_env=args.posture_conda_env,
                  download=args.download,
                  dry_run=args.dry_run,
                  keep_on_error=args.keep_on_error,
                  do_cleanup=(not args.no_cleanup),
                  keep_participant_summaries=args.keep_participant_summaries,
                  compress_master=args.compress_master)


if __name__ == '__main__':
    main()

# python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture