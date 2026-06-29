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

    python scripts/batch_pipeline.py \
        --batch-file batches/batch_1.txt \
        --model CHAP_ALL_ADULTS \
        --posture-conda-env deepposture \
        --skip-sleep

"""
import argparse ## for command-line argument parsing
import subprocess ## for running shell commands
from pathlib import Path ## for filesystem path manipulations
import shutil ## for file operations
import csv ## for CSV file handling
import sys ## for system-specific parameters and functions
import logging ## for logging messages
from typing import Optional, List, Tuple, Sequence
import tempfile
import gzip

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def run_cmd(cmd, dry_run=False, timeout=None):
    logger.info('CMD: %s', ' '.join(cmd))
    if dry_run:
        return 0
    try:
        res = subprocess.run(cmd, check=True, timeout=timeout)
        return res.returncode
    except subprocess.TimeoutExpired as e:
        logger.error('Command timed out after %ss: %s', timeout, ' '.join(cmd))
        return 124  # convention: 124 for timeout
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


def build_summarizer_cmd(
    summarizer_py: Path,
    participant_id: str,
    model: str,
    out_path: Path,
    skip_sleep: bool = False,
    summary_epoch: str = '1h',
    dataset: Optional[str] = None,
    epoch_columns: Optional[Sequence[str]] = None,
):
    cmd = [
        sys.executable,
        str(summarizer_py),
        '--participant-id', participant_id,
        '--model', model,
        '--out', str(out_path),
        '--summary-epoch', summary_epoch,
    ]
    if epoch_columns:
        cmd += ['--epoch-columns'] + list(epoch_columns)
    if dataset:
        cmd += ['--dataset', dataset]
    if skip_sleep:
        cmd += ['--skip-sleep']
    return cmd


def build_posture_cmd(
    posture_script_py: Path,
    participant_id: str,
    model: str,
    posture_site: str,
    posture_wrist_model: str,
    posture_wrist_checkpoint: Optional[str],
    posture_wrist_device: str,
    posture_wrist_batch_size: int,
    posture_wrist_num_workers: int,
    posture_padding: str,
    posture_wrist_include_probability: bool,
    posture_wrist_pin_mem: bool,
    posture_show_prediction_progress: bool,
):
    cmd = [
        sys.executable, str(posture_script_py),
        '--participant-id', participant_id,
        '--model', model,
        '--posture-site', posture_site,
        '--skip-incomplete-days',
        '--wrist-model', posture_wrist_model,
        '--wrist-device', posture_wrist_device,
        '--wrist-batch-size', str(posture_wrist_batch_size),
        '--wrist-num-workers', str(posture_wrist_num_workers),
        '--padding', posture_padding,
    ]
    if posture_wrist_checkpoint:
        cmd += ['--wrist-checkpoint', posture_wrist_checkpoint]
    if posture_wrist_include_probability:
        cmd += ['--wrist-include-probability']
    if posture_wrist_pin_mem:
        cmd += ['--wrist-pin-mem']
    else:
        cmd += ['--wrist-no-pin-mem']
    if posture_show_prediction_progress:
        cmd += ['--show-prediction-progress']
    return cmd


def process_batch(batch_file: Path, model: str, master_out: Path, sleep_conda_env: Optional[str] = None,
                  posture_conda_env: Optional[str] = None, download: bool = False, dry_run: bool = False,
                  keep_on_error: bool = False,
                  do_cleanup: bool = True,
                  keep_participant_summaries: bool = False,
                  compress_master: bool = True,
                  do_summary: bool = True,
                  export_10s: bool = False,
                  export_10s_output_root: str = 'data/epoch_10s',
                  export_10s_include_sleep_probabilities: bool = False,
                  export_10s_overwrite: bool = False,
                  skip_sleep: bool = False,
                  posture_site: str = 'hip',
                  posture_show_prediction_progress: bool = True,
                  posture_wrist_model: str = 'CHAP',
                  posture_wrist_checkpoint: Optional[str] = None,
                  posture_wrist_device: str = 'cpu',
                  posture_wrist_batch_size: int = 40,
                  posture_wrist_num_workers: int = 0,
                  posture_padding: str = 'wrap',
                  posture_wrist_include_probability: bool = False,
                  posture_wrist_pin_mem: bool = True,
                  summary_epoch: str = '1h',
                  epoch_columns: Optional[Sequence[str]] = None,
                  sleep_tmp_dir: Optional[str] = None,
                  sleep_day_chunks: Optional[int] = None,
                  sleep_chunk_overlap: Optional[int] = None,
                  sleep_swan_timeout: Optional[int] = None,
                  sleep_swan_use_worker: bool = False,
                  sleep_swan_worker_fallback: bool = False,
                  sleep_swan_retries: Optional[int] = None,
                  sleep_swan_retry_timeout: Optional[int] = None,
                  sleep_max_subdivision_depth: Optional[int] = None,
                  sleep_min_chunk_minutes: Optional[int] = None,
                  failed_out: Optional[str] = None,
                  participant_timeout: int = 1000):
    '''Process a batch of participants as specified in the batch file.
    args:
        batch_file: Path to CSV/TXT file with lines of "cycle,participant_id"
        model: Posture model name to use
        master_out: Path to master summary CSV (appended per participant)
        sleep_conda_env: Name of conda env for sleep step
        posture_conda_env: Name of conda env for posture step
        download: Whether to pass --download to participant pipeline
        dry_run: If True, print commands but do not execute
        keep_on_error: If True, do not stop on participant error; skip summarization/cleanup
        do_cleanup: If True, delete participant data after summarization
        keep_participant_summaries: If True, keep per-participant summary CSVs
        compress_master: If True, compress the master CSV with gzip
        do_summary: If True, create and append requested summary-epoch output
        export_10s: If True, export durable participant-day 10-second Parquet files
        export_10s_output_root: Root directory for durable 10-second Parquet files
        export_10s_include_sleep_probabilities: Include SWaN probabilities in Parquet files
        export_10s_overwrite: Overwrite existing participant-day Parquet files
        skip_sleep: If True, skip sleep predictions and run posture only
        posture_site: hip or wrist posture pipeline
        posture_show_prediction_progress: Show wrist prediction progress
        posture_wrist_model: Wrist model name to forward to participant pipeline
        posture_wrist_checkpoint: Optional wrist checkpoint path
        posture_wrist_device: Wrist device to use
        posture_wrist_batch_size: Wrist batch size
        posture_wrist_num_workers: Wrist DataLoader workers
        posture_wrist_include_probability: Include wrist probabilities in outputs
        posture_wrist_pin_mem: Enable pin_memory for wrist DataLoader
        summary_epoch: Summary epoch passed to summarize_participant.py, e.g. 10s, 30s, 1min, 5m, 30m, or 1h
        epoch_columns: Optional coarser epoch label columns to add to summary output
        sleep_tmp_dir: Optional temp directory for sleep step
        sleep_day_chunks: Optional number of chunks per day for sleep
        sleep_chunk_overlap: Optional overlap seconds between sleep chunks
        sleep_swan_timeout: Optional timeout in seconds for each SWaN worker chunk
        sleep_swan_use_worker: Run SWaN chunks in isolated worker processes
        sleep_swan_worker_fallback: Run SWaN normally first; use worker only after a timeout
        sleep_swan_retries: Optional retry attempts per SWaN chunk
        sleep_swan_retry_timeout: Optional timeout in seconds for SWaN retry attempts
        sleep_max_subdivision_depth: Optional max depth for sleep chunk subdivision
        sleep_min_chunk_minutes: Optional min chunk duration in minutes for sleep subdivision
        failed_out: Optional failed-participant CSV/TXT path
        participant_timeout: Timeout in seconds for each participant pipeline run
    '''

    repo = Path.cwd()
    run_pipeline_py = repo / 'scripts' / 'run_participant_pipeline.py'
    summarizer_py = repo / 'scripts' / 'summarize_participant.py'

    if not batch_file.exists():
        logger.error('Batch file not found: %s', batch_file)
        return

    # Ensure provided sleep tmp base exists if specified
    if sleep_tmp_dir:
        try:
            Path(sleep_tmp_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning('Could not create sleep tmp dir %s: %s', sleep_tmp_dir, e)

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
        pipeline_cmd += ['--posture-site', posture_site]
        # pass env names to the participant pipeline so it runs steps in the right conda envs
        if sleep_conda_env:
            pipeline_cmd += ['--sleep-conda-env', sleep_conda_env]
        if posture_conda_env:
            pipeline_cmd += ['--posture-conda-env', posture_conda_env]
        if posture_show_prediction_progress:
            pipeline_cmd += ['--posture-show-prediction-progress']
        pipeline_cmd += ['--posture-wrist-model', posture_wrist_model]
        if posture_wrist_checkpoint:
            pipeline_cmd += ['--posture-wrist-checkpoint', posture_wrist_checkpoint]
        pipeline_cmd += ['--posture-wrist-device', posture_wrist_device]
        pipeline_cmd += ['--posture-wrist-batch-size', str(posture_wrist_batch_size)]
        pipeline_cmd += ['--posture-wrist-num-workers', str(posture_wrist_num_workers)]
        pipeline_cmd += ['--posture-padding', posture_padding]
        if posture_wrist_include_probability:
            pipeline_cmd += ['--posture-wrist-include-probability']
        if posture_wrist_pin_mem:
            pipeline_cmd += ['--posture-wrist-pin-mem']
        else:
            pipeline_cmd += ['--posture-wrist-no-pin-mem']
        if download:
            pipeline_cmd += ['--download']
        if skip_sleep:
            pipeline_cmd += ['--skip-sleep']
        if export_10s:
            pipeline_cmd += ['--export-10s', '--export-10s-output-root', export_10s_output_root]
        if export_10s_include_sleep_probabilities:
            pipeline_cmd += ['--export-10s-include-sleep-probabilities']
        if export_10s_overwrite:
            pipeline_cmd += ['--export-10s-overwrite']
        if sleep_tmp_dir:
            pipeline_cmd += ['--sleep-tmp-dir', sleep_tmp_dir]
        # Advanced robust sleep options
        if sleep_day_chunks is not None:
            pipeline_cmd += ['--sleep-day-chunks', str(sleep_day_chunks)]
        if sleep_chunk_overlap is not None:
            pipeline_cmd += ['--sleep-chunk-overlap', str(sleep_chunk_overlap)]
        if sleep_swan_timeout is not None:
            pipeline_cmd += ['--sleep-swan-timeout', str(sleep_swan_timeout)]
        if sleep_swan_use_worker:
            pipeline_cmd += ['--sleep-swan-use-worker']
        if sleep_swan_worker_fallback:
            pipeline_cmd += ['--sleep-swan-worker-fallback']
        if sleep_swan_retries is not None:
            pipeline_cmd += ['--sleep-swan-retries', str(sleep_swan_retries)]
        if sleep_swan_retry_timeout is not None:
            pipeline_cmd += ['--sleep-swan-retry-timeout', str(sleep_swan_retry_timeout)]
        if sleep_max_subdivision_depth is not None:
            pipeline_cmd += ['--sleep-max-subdivision-depth', str(sleep_max_subdivision_depth)]
        if sleep_min_chunk_minutes is not None:
            pipeline_cmd += ['--sleep-min-chunk-minutes', str(sleep_min_chunk_minutes)]
        # Always skip incomplete days for both sleep and posture
        if not skip_sleep:
            pipeline_cmd += ['--skip-incomplete-days-sleep']
        pipeline_cmd += ['--skip-incomplete-days-posture']

        # Apply the participant timeout only when the sleep step is enabled.
        pipeline_timeout = None if skip_sleep else participant_timeout

        # Timeout and failed logging logic
        if failed_out:
            batch_failed_path = Path(failed_out)
            batch_failed_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            batch_failed_dir = Path('batch_failed')
            batch_failed_dir.mkdir(exist_ok=True)
            batch_failed_path = batch_failed_dir / (batch_file.stem + '_failed.txt')
        rc = run_cmd(pipeline_cmd, dry_run=dry_run, timeout=pipeline_timeout)
        if rc != 0:
            logger.error('Pipeline failed or timed out for %s (rc=%s)', pid, rc)
            # Log to batch-specific failed file in batch_failed/
            with open(batch_failed_path, 'a') as failf:
                failf.write(f"{cycle},{pid}\n")
            logger.info('Logged failed participant %s,%s to %s', cycle, pid, batch_failed_path)

            if not do_summary:
                if do_cleanup:
                    cleanup_participant(pid, cycle or 'unknown_cycle', model)
                continue

            # Attempt to salvage: if sleep completed some days, run posture for those days, then summarize
            sleep_preds_dir_failed = repo / 'data' / 'sleep_predictions' / pid / 'predictions'
            if skip_sleep:
                posture_dir_failed = repo / 'data' / 'predictions' / pid / model
                if posture_dir_failed.exists() and any(posture_dir_failed.glob('*.csv')):
                    logger.info('Sleep was skipped for participant %s, but posture outputs exist. Attempting posture-only summarization.', pid)
                    try:
                        if keep_participant_summaries:
                            part_out_failed = repo / 'data' / 'summaries' / f"{pid}_posture_only_{summary_epoch}_epoch.csv"
                            summarizer_cmd_failed = build_summarizer_cmd(
                                summarizer_py, pid, model, part_out_failed,
                                skip_sleep=True, summary_epoch=summary_epoch, dataset=cycle,
                                epoch_columns=epoch_columns
                            )
                            if not dry_run:
                                rc_salvage = run_cmd(summarizer_cmd_failed, dry_run=dry_run)
                                if rc_salvage == 0:
                                    append_summary(master_out, part_out_failed, compress=compress_master)
                                    logger.info('Salvaged posture-only data for %s and appended to master', pid)
                                else:
                                    logger.warning('Failed to salvage posture-only data for %s', pid)
                        else:
                            with tempfile.TemporaryDirectory() as tmpdir_failed:
                                part_out_failed = Path(tmpdir_failed) / f"{pid}.csv"
                                summarizer_cmd_failed = build_summarizer_cmd(
                                    summarizer_py, pid, model, part_out_failed,
                                    skip_sleep=True, summary_epoch=summary_epoch, dataset=cycle,
                                    epoch_columns=epoch_columns
                                )
                                if not dry_run:
                                    rc_salvage = run_cmd(summarizer_cmd_failed, dry_run=dry_run)
                                    if rc_salvage == 0:
                                        append_summary(master_out, part_out_failed, compress=compress_master)
                                        logger.info('Salvaged posture-only data for %s and appended to master', pid)
                                    else:
                                        logger.warning('Failed to salvage posture-only data for %s', pid)
                    except Exception as e:
                        logger.warning('Exception while trying to salvage posture-only data for %s: %s', pid, e)
                else:
                    logger.info('Sleep was skipped for participant %s and no posture outputs were found to summarize.', pid)
            elif sleep_preds_dir_failed.exists() and any(sleep_preds_dir_failed.glob('*.csv')):
                sleep_day_count = len(list(sleep_preds_dir_failed.glob('*.csv')))
                logger.info('Found %d sleep prediction day(s) for failed participant %s. Running posture to salvage data.', sleep_day_count, pid)

                try:
                    # Build and run posture command to process whatever days are available
                    posture_script_py = repo / 'scripts' / 'get_posture_predictions.py'
                    posture_cmd_salvage = build_posture_cmd(
                        posture_script_py,
                        pid,
                        model,
                        posture_site,
                        posture_wrist_model,
                        posture_wrist_checkpoint,
                        posture_wrist_device,
                        posture_wrist_batch_size,
                        posture_wrist_num_workers,
                        posture_padding,
                        posture_wrist_include_probability,
                        posture_wrist_pin_mem,
                        posture_show_prediction_progress,
                    )

                    # Run posture in the appropriate conda env if specified
                    if posture_conda_env:
                        posture_cmd_salvage = ['conda', 'run', '-n', posture_conda_env, '--no-capture-output', 'python'] + posture_cmd_salvage[1:]

                    if not dry_run:
                        logger.info('Running posture prediction for salvage: %s', ' '.join(posture_cmd_salvage))
                        rc_posture_salvage = run_cmd(posture_cmd_salvage, dry_run=dry_run)
                        if rc_posture_salvage != 0:
                            logger.warning('Posture salvage run failed for %s (rc=%s), but continuing with summarization attempt', pid, rc_posture_salvage)

                    # Now run summarizer - it will only summarize days where BOTH sleep and posture exist
                    if keep_participant_summaries:
                        part_out_failed = repo / 'data' / 'summaries' / f"{pid}_sleep_posture_{summary_epoch}_epoch.csv"
                        summarizer_cmd_failed = build_summarizer_cmd(
                            summarizer_py, pid, model, part_out_failed,
                            summary_epoch=summary_epoch, dataset=cycle,
                            epoch_columns=epoch_columns
                        )
                        if not dry_run:
                            rc_salvage = run_cmd(summarizer_cmd_failed, dry_run=dry_run)
                            if rc_salvage == 0:
                                append_summary(master_out, part_out_failed, compress=compress_master)
                                logger.info('Salvaged partial data for %s and appended to master', pid)
                            else:
                                logger.warning('Failed to salvage partial data for %s', pid)
                    else:
                        with tempfile.TemporaryDirectory() as tmpdir_failed:
                            part_out_failed = Path(tmpdir_failed) / f"{pid}.csv"
                            summarizer_cmd_failed = build_summarizer_cmd(
                                summarizer_py, pid, model, part_out_failed,
                                summary_epoch=summary_epoch, dataset=cycle,
                                epoch_columns=epoch_columns
                            )
                            if not dry_run:
                                rc_salvage = run_cmd(summarizer_cmd_failed, dry_run=dry_run)
                                if rc_salvage == 0:
                                    append_summary(master_out, part_out_failed, compress=compress_master)
                                    logger.info('Salvaged partial data for %s and appended to master', pid)
                                else:
                                    logger.warning('Failed to salvage partial data for %s', pid)
                except Exception as e:
                    logger.warning('Exception while trying to salvage partial data for %s: %s', pid, e)
            else:
                logger.info('No sleep predictions found to salvage for failed participant %s', pid)

            # Cleanup if enabled
            if do_cleanup:
                cleanup_participant(pid, cycle or 'unknown_cycle', model)
            continue

        if not skip_sleep:
            # If sleep had no complete days, participant runner will skip posture and produce no per-day sleep outputs.
            sleep_preds_dir = repo / 'data' / 'sleep_predictions' / pid / 'predictions'
            if not (sleep_preds_dir.exists() and any(sleep_preds_dir.glob('*.csv'))):
                logger.warning('Participant %s: no complete sleep-day outputs found; skipping requested outputs.', pid)
                if do_cleanup and not dry_run:
                    cleanup_participant(pid, cycle or 'unknown_cycle', model)
                continue

        posture_dir = repo / 'data' / 'predictions' / pid / model
        if not (posture_dir.exists() and any(posture_dir.glob('*.csv'))):
            logger.warning('Participant %s: no posture outputs found; skipping requested outputs.', pid)
            if do_cleanup and not dry_run:
                cleanup_participant(pid, cycle or 'unknown_cycle', model)
            continue

        if not do_summary:
            logger.info('Participant %s: 10-second export complete; summary output disabled.', pid)
            if do_cleanup and not dry_run:
                cleanup_participant(pid, cycle or 'unknown_cycle', model)
            continue

        # Build summarizer command with either a temp output or a persistent file
        if keep_participant_summaries:
            suffix = f"posture_only_{summary_epoch}_epoch" if skip_sleep else f"sleep_posture_{summary_epoch}_epoch"
            part_out = repo / 'data' / 'summaries' / f"{pid}_{suffix}.csv"
            summarizer_cmd = build_summarizer_cmd(
                summarizer_py, pid, model, part_out,
                skip_sleep=skip_sleep, summary_epoch=summary_epoch, dataset=cycle,
                epoch_columns=epoch_columns
            )

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
                summarizer_cmd = build_summarizer_cmd(
                    summarizer_py, pid, model, part_out,
                    skip_sleep=skip_sleep, summary_epoch=summary_epoch, dataset=cycle,
                    epoch_columns=epoch_columns
                )

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

    if do_summary:
        logger.info('Batch processing complete. Master summary at: %s', master_out)
    else:
        logger.info('Batch processing complete. Summary output was disabled.')


def main():
    parser = argparse.ArgumentParser(description='Run batch pipeline for many participants')
    parser.add_argument('--batch-file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--master-out', default='data/summaries/batch_sleep_posture_hourly.csv.gz')
    parser.add_argument('--sleep-conda-env', default=None)
    parser.add_argument('--posture-conda-env', default=None)
    parser.add_argument('--posture-site', choices=['hip', 'wrist'], default='hip', help='Choose hip or wrist posture pipeline for batch runs')
    parser.add_argument('--posture-show-prediction-progress', action='store_true', dest='posture_show_prediction_progress', help='Show wrist prediction progress during batch runs')
    parser.add_argument('--posture-no-show-prediction-progress', action='store_false', dest='posture_show_prediction_progress', help='Disable wrist prediction progress during batch runs')
    parser.add_argument('--posture-wrist-model', default='CHAP', help='Wrist model name to forward to participant pipeline')
    parser.add_argument('--posture-wrist-checkpoint', default=None, help='Optional wrist checkpoint path to forward to participant pipeline')
    parser.add_argument('--posture-wrist-device', default='cpu', help='Wrist device to use during batch runs')
    parser.add_argument('--posture-wrist-batch-size', type=int, default=40, help='Wrist batch size to forward to participant pipeline')
    parser.add_argument('--posture-wrist-num-workers', type=int, default=0, help='Wrist DataLoader workers to forward to participant pipeline')
    parser.add_argument('--posture-padding', choices=('drop', 'zero', 'wrap'), default='wrap', help='Handle the final partial posture sequence')
    parser.add_argument('--posture-wrist-include-probability', action='store_true', help='Include prob_sitting in wrist CSVs and 10-second Parquet export')
    parser.add_argument('--posture-wrist-pin-mem', dest='posture_wrist_pin_mem', action='store_true', help='Enable pin_memory for wrist prediction DataLoader')
    parser.add_argument('--posture-wrist-no-pin-mem', dest='posture_wrist_pin_mem', action='store_false', help='Disable pin_memory for wrist prediction DataLoader')
    parser.set_defaults(posture_wrist_pin_mem=True, posture_show_prediction_progress=True)
    parser.add_argument('--download', action='store_true', help='Run with --download to fetch raw archives')
    parser.add_argument('--skip-sleep', action='store_true', help='Skip sleep predictions and run posture only for each participant')
    parser.add_argument('--summary-epoch', default='1h', help='Summary epoch passed to summarize_participant.py, e.g. 10s, 30s, 1min, 5m, 30m, or 1h')
    parser.add_argument('--epoch-columns', nargs='*', default=None, help='Optional coarser epoch label columns to add, e.g. --epoch-columns 20m 30m. Use Hour for hourly grouping.')
    parser.add_argument('--skip-summary', action='store_true', help='Do not create or append summary-epoch output; requires --export-10s')
    parser.add_argument('--export-10s', action='store_true', help='Export durable participant-day 10-second Parquet files before cleanup')
    parser.add_argument('--export-10s-output-root', default='data/epoch_10s', help='Root directory for durable 10-second Parquet files')
    parser.add_argument('--export-10s-include-sleep-probabilities', action='store_true', help='Include SWaN probability columns in 10-second Parquet files')
    parser.add_argument('--export-10s-overwrite', action='store_true', help='Overwrite existing participant-day 10-second Parquet files')
    parser.add_argument('--sleep-tmp-dir', default=None, help='Temp directory for sleep step (forwarded to participant runner)')
    parser.add_argument('--sleep-day-chunks', type=int, default=None, help='Number of chunks per day for sleep (forwarded to participant runner)')
    parser.add_argument('--sleep-chunk-overlap', type=int, default=None, help='Overlap seconds between sleep chunks (forwarded to participant runner)')
    parser.add_argument('--sleep-swan-timeout', type=int, default=None, help='Timeout in seconds for each SWaN worker chunk (forwarded to participant runner)')
    parser.add_argument('--sleep-swan-use-worker', action='store_true', help='Run SWaN chunks in isolated worker processes with timeout enforcement')
    parser.add_argument('--sleep-swan-worker-fallback', action='store_true', help='Run SWaN normally first; if it times out, retry that chunk in an isolated worker')
    parser.add_argument('--sleep-swan-retries', type=int, default=None, help='Retry attempts per SWaN chunk (forwarded to participant runner)')
    parser.add_argument('--sleep-swan-retry-timeout', type=int, default=None, help='Timeout in seconds for SWaN retry attempts (forwarded to participant runner)')
    parser.add_argument('--sleep-max-subdivision-depth', type=int, default=None, help='Maximum depth for recursive chunk subdivision (forwarded to participant runner)')
    parser.add_argument('--sleep-min-chunk-minutes', type=int, default=None, help='Minimum chunk duration in minutes before subdivision stops (forwarded to participant runner)')
    parser.add_argument('--failed-out', default=None, help='Optional failed participant output path. Rows are cycle,participant_id.')
    parser.add_argument('--participant-timeout', type=int, default=1000, help='Timeout in seconds for each participant pipeline run when sleep is enabled')
    parser.add_argument('--dry-run', action='store_true', help='Print commands but do not execute')
    parser.add_argument('--keep-on-error', action='store_true', help='Do not stop on participant error; skip summarization/cleanup')
    parser.add_argument('--no-cleanup', action='store_true', help='Do not delete participant data after summarization')
    parser.add_argument('--keep-participant-summaries', action='store_true', help='Keep per-participant hourly summary CSVs (default: not kept)')
    # Compression flags (default: compress master). --no-compress-master overrides.
    parser.add_argument('--compress-master', dest='compress_master', action='store_true', help='Compress the master CSV (gzip). Default behavior.')
    parser.add_argument('--no-compress-master', dest='compress_master', action='store_false', help='Do not compress the master CSV.')
    parser.set_defaults(compress_master=True)

    args = parser.parse_args()
    if args.skip_summary and not args.export_10s:
        parser.error('--skip-summary requires --export-10s so the pipeline produces a durable output')

    default_master_out = 'data/summaries/batch_sleep_posture_hourly.csv.gz'
    if args.summary_epoch == '1h':
        epoch_sleep_master_out = default_master_out
        epoch_posture_master_out = 'data/summaries/batch_posture_only_hourly.csv.gz'
    else:
        epoch_sleep_master_out = f"data/summaries/batch_sleep_posture_{args.summary_epoch}_epoch.csv.gz"
        epoch_posture_master_out = f"data/summaries/batch_posture_only_{args.summary_epoch}_epoch.csv.gz"

    # Normalize master_out extension when compressing by default
    selected_master_out = args.master_out
    if args.skip_sleep and args.master_out == default_master_out:
        selected_master_out = epoch_posture_master_out
        logger.info('Using posture-only master output: %s', selected_master_out)
    elif (not args.skip_sleep) and args.master_out == default_master_out:
        selected_master_out = epoch_sleep_master_out
        logger.info('Using sleep/posture master output: %s', selected_master_out)

    master_out_path = Path(selected_master_out)
    if args.compress_master and master_out_path.suffix != '.gz':
        # If user supplied a non-gz path but compression is enabled, append .gz for clarity
        master_out_path = Path(str(master_out_path) + '.gz')
        logger.info('Using compressed master output: %s', master_out_path)
    elif not args.compress_master and master_out_path.suffix == '.gz':
        master_out_path = master_out_path.with_suffix('')
        logger.info('Using uncompressed master output: %s', master_out_path)

    process_batch(Path(args.batch_file), args.model, master_out_path,
                  sleep_conda_env=args.sleep_conda_env,
                  posture_conda_env=args.posture_conda_env,
                  download=args.download,
                  dry_run=args.dry_run,
                  keep_on_error=args.keep_on_error,
                  do_cleanup=(not args.no_cleanup),
                  keep_participant_summaries=args.keep_participant_summaries,
                  compress_master=args.compress_master,
                  do_summary=(not args.skip_summary),
                  export_10s=args.export_10s,
                  export_10s_output_root=args.export_10s_output_root,
                  export_10s_include_sleep_probabilities=args.export_10s_include_sleep_probabilities,
                  export_10s_overwrite=args.export_10s_overwrite,
                  skip_sleep=args.skip_sleep,
                  posture_site=args.posture_site,
                  posture_show_prediction_progress=args.posture_show_prediction_progress,
                  posture_wrist_model=args.posture_wrist_model,
                  posture_wrist_checkpoint=args.posture_wrist_checkpoint,
                  posture_wrist_device=args.posture_wrist_device,
                  posture_wrist_batch_size=args.posture_wrist_batch_size,
                  posture_wrist_num_workers=args.posture_wrist_num_workers,
                  posture_padding=args.posture_padding,
                  posture_wrist_include_probability=args.posture_wrist_include_probability,
                  posture_wrist_pin_mem=args.posture_wrist_pin_mem,
                  summary_epoch=args.summary_epoch,
                  epoch_columns=args.epoch_columns,
                  sleep_tmp_dir=args.sleep_tmp_dir,
                  sleep_day_chunks=args.sleep_day_chunks,
                  sleep_chunk_overlap=args.sleep_chunk_overlap,
                  sleep_swan_timeout=args.sleep_swan_timeout,
                  sleep_swan_use_worker=args.sleep_swan_use_worker,
                  sleep_swan_worker_fallback=args.sleep_swan_worker_fallback,
                  sleep_swan_retries=args.sleep_swan_retries,
                  sleep_swan_retry_timeout=args.sleep_swan_retry_timeout,
                  sleep_max_subdivision_depth=args.sleep_max_subdivision_depth,
                  sleep_min_chunk_minutes=args.sleep_min_chunk_minutes,
                  failed_out=args.failed_out,
                  participant_timeout=args.participant_timeout)


if __name__ == '__main__':
    main()

# python scripts/batch_pipeline.py --batch-file batches/batch_1.txt --model CHAP_ALL_ADULTS --sleep-conda-env sklearn023 --posture-conda-env deepposture --sleep-tmp-dir data/tmp --download -master-out data/summaries/my_custom_master.csv.gz

## Wrist compatibility example:
# python scripts/batch_pipeline.py   --batch-file batches/batch_1.txt   --model CHAP   --posture-conda-env deepposture  --skip-sleep --download --posture-site wrist
