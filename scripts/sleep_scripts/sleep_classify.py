#!/usr/bin/env python3
"""
Sleep Classification for SedentaryBehaviour Project using SWaN_accel Package

This script works with raw accelerometer data files from NHANES and applies 
SWaN sleep classification to detect sleep/wake periods.

The SWaN (Sleep/Wake Analysis) algorithm uses accelerometer data to predict
sleep and wake periods. This script implements both first pass (initial
sleep detection) and optional second pass (refinement with self-reported logs).

Usage:
    # Basic usage (processes all data at once)
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12
    
    # Process by calendar day with per-day outputs (recommended)
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --by-day
    
    # Skip incomplete days (requires full 23-25 hour days starting at 00:00:00)
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --by-day --skip-incomplete-days
    
    # Process specific dates only
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --by-day --only-dates 2000-01-11 2000-01-12
    
    # Custom output directory and chunk settings
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --output-dir data/sleep_predictions --by-day --day-chunks 3 --chunk-overlap-seconds 30
    
    # Enable debug logging
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --by-day --debug
    
    # Use isolated worker process with timeout (robust for large datasets)
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --by-day --swan-use-worker --swan-timeout-seconds 300

Author: Generated for SedentaryBehaviour project
Date: October 2025
"""

# Set BLAS/OMP thread caps BEFORE importing numpy/pandas/sklearn to ensure they take effect
import os as _osenv
_osenv.environ.setdefault("OMP_NUM_THREADS", "1")
_osenv.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_osenv.environ.setdefault("MKL_NUM_THREADS", "1")
_osenv.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import json
import glob
import tempfile
import shutil
import signal
from contextlib import contextmanager

# Try to import SWaN_accel
try:
    import SWaN_accel as swan
    from SWaN_accel import swan_first_pass, swan_second_pass, classify
    SWAN_AVAILABLE = True
except ImportError:
    print("Warning: SWaN_accel package not found. Please install it using:")
    print("  pip install SWaN_accel")
    SWAN_AVAILABLE = False

# Configure a real logger. Default INFO; toggled by --debug.
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class SwanTimeoutError(TimeoutError):
    """Raised when an in-process SWaN call exceeds the requested timeout."""


@contextmanager
def swan_alarm_timeout(timeout_seconds):
    """Best-effort timeout for in-process SWaN calls.

    This avoids worker overhead for normal cases. A subprocess worker is still
    needed for hard-kill protection when a library call cannot be interrupted.
    """
    if not timeout_seconds or timeout_seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum, frame):
        raise SwanTimeoutError(f"SWaN call timed out after {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_raw_sensor_data(participant_dir):
    """
    Load all raw sensor CSV files for a participant.
    
    Parameters
    ----------
    participant_dir : str or Path
        Directory containing the participant's sensor CSV files
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns: timestamp, X, Y, Z
    """
    participant_dir = Path(participant_dir)
    
    # Find all sensor CSV files
    sensor_files = sorted(glob.glob(str(participant_dir / "*.sensor.csv")))
    
    if not sensor_files:
        raise FileNotFoundError(f"No sensor CSV files found in {participant_dir}")
    
    logger.info(f"Found {len(sensor_files)} sensor files for participant")
    
    # Load and concatenate all files
    dfs = []
    for sensor_file in sensor_files:
        try:
            df = pd.read_csv(sensor_file)
            
            # Rename columns if needed
            if 'HEADER_TIMESTAMP' in df.columns:
                df = df.rename(columns={'HEADER_TIMESTAMP': 'timestamp'})
            
            # Parse timestamp with explicit format to avoid warnings
            # NHANES format: "2000-01-08 17:30:00.000"
            # Parse timestamp with explicit formats; fall back to pandas parser
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
                except ValueError:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Keep only necessary columns
            df = df[['timestamp', 'X', 'Y', 'Z']]
            
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} rows from {Path(sensor_file).name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {sensor_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data could be loaded from sensor files")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded total of {len(combined_df)} data points")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


def _parse_ts(s: str):
    """Parse timestamp string to pandas.Timestamp with common formats."""
    s = s.strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        try:
            return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                return pd.to_datetime(s)
            except Exception:
                return None


def list_sensor_files(participant_dir: Path):
    return sorted(glob.glob(str(Path(participant_dir) / "*.sensor.csv")))


def build_file_time_index(participant_dir: Path):
    """Quickly scan each sensor CSV to get (min_ts, max_ts) without reading whole file.
    Returns list of dicts: {path, min_ts, max_ts}."""
    import csv as _csv
    files = list_sensor_files(participant_dir)
    index = []
    for i, fp in enumerate(files, 1):
        p = Path(fp)
        min_ts = None
        max_ts = None
        # First non-header row for min_ts
        try:
            with p.open('r', newline='') as f:
                reader = _csv.reader(f)
                header = next(reader, None)
                ts_col = None
                if header:
                    # Common names
                    if 'timestamp' in header:
                        ts_col = header.index('timestamp')
                    elif 'HEADER_TIMESTAMP' in header:
                        ts_col = header.index('HEADER_TIMESTAMP')
                    elif 'HEADER_TIME_STAMP' in header:
                        ts_col = header.index('HEADER_TIME_STAMP')
                    else:
                        ts_col = 0
                for row in reader:
                    if not row:
                        continue
                    ts_idx = ts_col if ts_col is not None else 0
                    if ts_idx >= len(row):
                        continue
                    ts = _parse_ts(row[ts_idx])
                    if ts is not None:
                        min_ts = ts
                        break
        except Exception as e:
            logger.warning("Failed to read first row for %s: %s", p, e)
        # Last row for max_ts: read last chunk
        try:
            with p.open('rb') as bf:
                try:
                    bf.seek(-65536, os.SEEK_END)
                except OSError:
                    bf.seek(0)
                tail_bytes = bf.read()
                try:
                    tail_text = tail_bytes.decode(errors='ignore')
                except Exception:
                    # If already str (shouldn't happen), coerce
                    tail_text = str(tail_bytes)
                tail = tail_text.splitlines()
                # Walk from end to find the last non-empty, non-header line
                for line in reversed(tail):
                    if not line.strip():
                        continue
                    if line.lower().startswith('timestamp') or 'HEADER' in line:
                        continue
                    parts = line.split(',')
                    if not parts:
                        continue
                    ts = _parse_ts(parts[0])
                    if ts is not None:
                        max_ts = ts
                        break
        except Exception as e:
            logger.warning("Failed to read last row for %s: %s", p, e)
        index.append({
            'path': p,
            'min_ts': min_ts,
            'max_ts': max_ts,
            'min_ns': int(min_ts.value) if min_ts is not None else -1,
            'max_ns': int(max_ts.value) if max_ts is not None else -1,
        })
        if i % 50 == 0:
            logger.info("Indexed %d/%d files...", i, len(files))
    # Filter files with no timestamps
    valid = [it for it in index if it['min_ts'] is not None and it['max_ts'] is not None]
    if not valid:
        raise ValueError(f"No valid timestamp ranges found in {participant_dir}")
    # Sort by min_ts
    def _key_min_ns(d):
        try:
            return int(d.get('min_ns', -1))
        except Exception:
            return -1
    valid.sort(key=_key_min_ns)
    logger.info("File time index built: %d files", len(valid))
    return valid


def prepare_data_for_swan(df, sampling_rate=80):
    """
    Prepare accelerometer data for SWaN analysis.
    
    SWaN expects data in a specific format with regular sampling intervals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: timestamp, X, Y, Z
    sampling_rate : int
        Sampling rate in Hz (default: 80 for NHANES GT3X+)
        
    Returns
    -------
    pd.DataFrame
        Prepared dataframe ready for SWaN analysis
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate sampling interval
    expected_interval = pd.Timedelta(seconds=1/sampling_rate)
    
    # Check actual sampling rate
    actual_intervals = df['timestamp'].diff()
    median_interval = actual_intervals.median()
    
    logger.info(f"Expected interval: {expected_interval}")
    logger.info(f"Median actual interval: {median_interval}")
    
    # SWaN expects acceleration in g units (which NHANES data already provides)
    # No conversion needed
    
    return df


def run_swan_first_pass(df, sampling_rate=80, output_dir=None, participant_id=None):
    """
    Run SWaN first pass sleep detection.
    
    The first pass identifies sleep/wake periods based solely on accelerometer
    data without requiring self-reported sleep logs.
    
    Based on the actual SWAN implementation, this function:
    1. Groups data into 30-second windows
    2. Computes features for each window
    3. Uses trained model to predict: WEAR (0), SLEEP (1), or NON-WEAR (2)
    
    Parameters
    ----------
    df : pd.DataFrame
        Accelerometer data with columns: timestamp, X, Y, Z
    sampling_rate : int
        Sampling rate in Hz
    output_dir : str or Path, optional
        Directory to save results
    participant_id : str, optional
        Participant ID for labeling outputs
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added sleep/wake predictions
    """
    if not SWAN_AVAILABLE:
        raise ImportError("SWaN_accel package is not installed")
    
    logger.info("Running SWaN first pass sleep detection...")
    
    # Prepare dataframe with SWAN expected column names
    swan_df = df.copy()
    swan_df.rename(columns={
        'timestamp': 'HEADER_TIME_STAMP',
        'X': 'X_ACCELERATION_METERS_PER_SECOND_SQUARED',
        'Y': 'Y_ACCELERATION_METERS_PER_SECOND_SQUARED',
        'Z': 'Z_ACCELERATION_METERS_PER_SECOND_SQUARED'
    }, inplace=True)
    
    # Ensure timestamp is datetime
    swan_df['HEADER_TIME_STAMP'] = pd.to_datetime(swan_df['HEADER_TIME_STAMP'], errors='coerce')

    # Coerce accelerations to numeric and drop non-finite values
    num_cols = [
        'X_ACCELERATION_METERS_PER_SECOND_SQUARED',
        'Y_ACCELERATION_METERS_PER_SECOND_SQUARED',
        'Z_ACCELERATION_METERS_PER_SECOND_SQUARED'
    ]
    before_rows = len(swan_df)
    for c in num_cols:
        if c in swan_df.columns:
            swan_df[c] = pd.to_numeric(swan_df[c], errors='coerce')
    # Drop rows with invalid timestamps or NaNs in any accel column
    swan_df = swan_df.dropna(subset=['HEADER_TIME_STAMP'] + [c for c in num_cols if c in swan_df.columns])
    # Remove non-finite values (inf/-inf)
    for c in num_cols:
        if c in swan_df.columns:
            swan_df = swan_df[np.isfinite(swan_df[c])]
    # Deduplicate and sort by timestamp to avoid pathological grouping behavior
    dup_count = swan_df['HEADER_TIME_STAMP'].duplicated().sum()
    if dup_count:
        logger.warning("Detected %d duplicated timestamps in chunk; keeping first occurences.", int(dup_count))
        swan_df = swan_df.drop_duplicates(subset=['HEADER_TIME_STAMP'], keep='first')
    swan_df = swan_df.sort_values('HEADER_TIME_STAMP').reset_index(drop=True)
    after_rows = len(swan_df)
    dropped = before_rows - after_rows
    if dropped > 0:
        logger.warning("Sanitized input for SWaN: dropped %d problematic rows (non-numeric/NaN/non-finite/dups)", int(dropped))
    if after_rows == 0:
        raise ValueError("No valid rows remain after sanitizing input for SWaN first pass")
    
    # Create temporary output file
    if output_dir and participant_id:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_output = output_dir / f"{participant_id}_swan_first_pass_raw.csv"
    else:
        temp_output = "temp_swan_output.csv"
    
    logger.info(f"Processing {len(swan_df)} data points...")
    logger.info(f"Date range: {swan_df['HEADER_TIME_STAMP'].min()} to {swan_df['HEADER_TIME_STAMP'].max()}")
    
    # swan_first_pass imported at module top when SWAN_AVAILABLE
    
    # Run SWAN first pass
    # The main() function expects: df, file_path, sampling_rate
    swan_first_pass.main(
        df=swan_df,
        file_path=str(temp_output),
        sampling_rate=sampling_rate
    )
    
    # Load the results
    logger.info(f"Loading SWAN predictions from {temp_output}")
    swan_results = pd.read_csv(temp_output)
    
    # SWAN predictions:
    # 0 = WEAR (awake)
    # 1 = SLEEP
    # 2 = NON-WEAR
    
    # Map to original timestamps (merge back with input data)
    swan_results['HEADER_TIME_STAMP'] = pd.to_datetime(swan_results['HEADER_TIME_STAMP'])
    
    # Create window-level predictions
    result_summary = swan_results[['HEADER_TIME_STAMP', 'STOP_TIME', 'PREDICTED', 
                                   'PROB_WEAR', 'PROB_SLEEP', 'PROB_NWEAR']].copy()
    result_summary = result_summary.rename({ 'HEADER_TIME_STAMP': 'START_TIME' }, axis=1)
    result_summary['START_TIME'] = pd.to_datetime(result_summary['START_TIME'])
    result_summary['STOP_TIME'] = pd.to_datetime(result_summary['STOP_TIME'])
    
    # Map predictions to readable labels
    prediction_map = {0: 'WEAR', 1: 'SLEEP', 2: 'NON-WEAR'}
    result_summary['STATE'] = result_summary['PREDICTED'].map(prediction_map)
    
    # Calculate summary statistics
    total_windows = len(result_summary)
    sleep_windows = (result_summary['PREDICTED'] == 1).sum()
    wear_windows = (result_summary['PREDICTED'] == 0).sum()
    nonwear_windows = (result_summary['PREDICTED'] == 2).sum()
    
    total_hours = total_windows * 30 / 3600  # 30-second windows
    sleep_hours = sleep_windows * 30 / 3600
    wear_hours = wear_windows * 30 / 3600
    nonwear_hours = nonwear_windows * 30 / 3600
    
    logger.info(f"Processed {total_windows} windows (30-second each)")
    logger.info(f"Sleep: {sleep_windows} windows ({sleep_hours:.2f} hours, {sleep_windows/total_windows*100:.1f}%)")
    logger.info(f"Wear: {wear_windows} windows ({wear_hours:.2f} hours, {wear_windows/total_windows*100:.1f}%)")
    logger.info(f"Non-wear: {nonwear_windows} windows ({nonwear_hours:.2f} hours, {nonwear_windows/total_windows*100:.1f}%)")
    
    # Save results if output directory provided
    if output_dir and participant_id:
        # Save window-level predictions
        output_file = output_dir / f"{participant_id}_swan_predictions.csv"
        result_summary.to_csv(output_file, index=False, float_format="%.3f")
        logger.info(f"Saved predictions to {output_file}")
        
        # Create summary
        summary = {
            'participant_id': participant_id,
            'total_hours': float(total_hours),
            'total_windows': int(total_windows),
            'sleep': {
                'windows': int(sleep_windows),
                'hours': float(sleep_hours),
                'percentage': float(sleep_windows/total_windows*100)
            },
            'wear': {
                'windows': int(wear_windows),
                'hours': float(wear_hours),
                'percentage': float(wear_windows/total_windows*100)
            },
            'non_wear': {
                'windows': int(nonwear_windows),
                'hours': float(nonwear_hours),
                'percentage': float(nonwear_windows/total_windows*100)
            },
            'date_range': {
                'start': str(result_summary['START_TIME'].min()),
                'end': str(result_summary['STOP_TIME'].max())
            }
        }
        
        summary_file = output_dir / f"{participant_id}_swan_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        print("\n" + "="*70)
        print(f"SWAN First Pass Results for Participant {participant_id}")
        print("="*70)
        print(f"Total duration: {total_hours:.2f} hours ({total_windows} windows)")
        print(f"Sleep:    {sleep_hours:.2f} hours ({summary['sleep']['percentage']:.1f}%)")
        print(f"Wear:     {wear_hours:.2f} hours ({summary['wear']['percentage']:.1f}%)")
        print(f"Non-wear: {nonwear_hours:.2f} hours ({summary['non_wear']['percentage']:.1f}%)")
        print("="*70 + "\n")
    
    return result_summary


def main():
    """Main function to run sleep classification."""
    parser = argparse.ArgumentParser(
        description='Run SWaN sleep classification on NHANES raw accelerometer data'
    )
    
    parser.add_argument(
        '--participant-id',
        required=True,
        help='Participant ID (SEQN)'
    )
    
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing raw sensor CSV files (e.g., data/raw/2011-12)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/sleep_predictions',
        help='Output directory for sleep predictions (default: data/sleep_predictions)'
    )
    
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=80,
        help='Sampling rate in Hz (default: 80 for NHANES GT3X+)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--by-day',
        action='store_true',
        help='Process participant data one calendar day at a time and save per-day outputs'
    )
    parser.add_argument(
        '--only-dates',
        nargs='*',
        default=None,
        help='Optional list of YYYY-MM-DD dates to process (only with --by-day). Example: --only-dates 2000-01-11 2000-01-12'
    )
    parser.add_argument(
        '--skip-incomplete-days',
        action='store_true',
        help='Skip any calendar day without a full day of data: must start at 00:00:00 and end at 23:59:xx or next 00:00:00 (23–25 hours)'
    )
    parser.add_argument(
        '--day-chunks',
        type=int,
        default=3,
        help='When processing by day, split each day into this many chunks (default: 3)'
    )
    parser.add_argument(
        '--chunk-overlap-seconds',
        type=int,
        default=30,
        help='Overlap seconds between adjacent chunks to avoid boundary effects (default: 30)'
    )
    parser.add_argument(
        '--tmp-dir',
        default=None,
        help='Directory for temporary working files (defaults to $TMPDIR or system temp)'
    )
    parser.add_argument(
        '--swan-timeout-seconds',
        type=int,
        default=180,
        help='Timeout in seconds for each SWaN chunk; if exceeded, the chunk is marked failed and processing continues (default: 180)'
    )
    parser.add_argument(
        '--swan-use-worker',
        action='store_true',
        help='Run SWaN first-pass in an isolated subprocess worker with timeout enforcement'
    )
    parser.add_argument(
        '--swan-worker-fallback',
        action='store_true',
        help='Run SWaN in-process first; if it times out, retry that chunk in an isolated worker'
    )
    parser.add_argument(
        '--swan-retries',
        type=int,
        default=5,
        help='Number of retry attempts for a chunk if SWaN fails or times out (default: 1)'
    )
    parser.add_argument(
        '--swan-retry-timeout-seconds',
        type=int,
        default=None,
        help='Timeout in seconds for retry attempts (defaults to --swan-timeout-seconds if not provided)'
    )
    parser.add_argument(
        '--max-subdivision-depth',
        type=int,
        default=3,
        help='Maximum depth for recursive chunk subdivision when a chunk fails (default: 3, meaning up to 8x subdivision)'
    )
    parser.add_argument(
        '--min-chunk-minutes',
        type=int,
        default=5,
        help='Minimum chunk size in minutes before giving up on subdivision (default: 5 minutes)'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Check if SWaN is available
    if not SWAN_AVAILABLE:
        logger.error("SWaN_accel package not found. Please install it:")
        logger.error("  pip install SWaN_accel")
        sys.exit(1)
    
    # Construct participant directory path
    participant_dir = Path(args.data_dir) / args.participant_id

    if not participant_dir.exists():
        logger.error(f"Participant directory not found: {participant_dir}")
        sys.exit(1)

    try:
        # Build a time index so we can load only data needed per day
        logger.info("Indexing sensor files for participant %s", args.participant_id)
        file_index = build_file_time_index(participant_dir)

        # Also load once if not by-day (kept for backward compatibility)
        df_full = None

        def read_files_for_window(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
            """Read only files overlapping [window_start, window_end) and filter to that window."""
            overlaps = [it for it in file_index if (it['max_ts'] >= window_start) and (it['min_ts'] < window_end)]
            if not overlaps:
                return pd.DataFrame(columns=['timestamp', 'X', 'Y', 'Z'])
            dfs = []
            for it in overlaps:
                p = it['path']
                try:
                    dfp = pd.read_csv(p, usecols=None, low_memory=False)
                    if 'HEADER_TIMESTAMP' in dfp.columns:
                        dfp = dfp.rename(columns={'HEADER_TIMESTAMP': 'timestamp'})
                    # Parse timestamps
                    try:
                        dfp['timestamp'] = pd.to_datetime(dfp['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
                    except Exception:
                        dfp['timestamp'] = pd.to_datetime(dfp['timestamp'], errors='coerce')
                    # Keep necessary columns only if present
                    cols = [c for c in ['timestamp', 'X', 'Y', 'Z'] if c in dfp.columns]
                    dfp = dfp[cols]
                    # Coerce numeric and drop obvious bad rows early
                    for c in ['X', 'Y', 'Z']:
                        if c in dfp.columns:
                            dfp[c] = pd.to_numeric(dfp[c], errors='coerce')
                    subset_cols = ['timestamp'] + [c for c in ['X','Y','Z'] if c in dfp.columns]
                    # Use boolean mask instead of dropna(subset=...) to avoid type-stub issues
                    if subset_cols:
                        mask = dfp[subset_cols].notna().all(axis=1)
                        dfp = dfp[mask]
                    for c in ['X','Y','Z']:
                        if c in dfp.columns:
                            dfp = dfp[np.isfinite(dfp[c])]
                    # Filter to slice window with small overlaps
                    dfp = dfp[(dfp['timestamp'] >= window_start) & (dfp['timestamp'] < window_end)]
                    if not dfp.empty:
                        dfs.append(dfp)
                except Exception as e:
                    logger.warning("Failed reading %s: %s", p, e)
            if not dfs:
                return pd.DataFrame(columns=['timestamp', 'X', 'Y', 'Z'])
            dd = pd.concat(dfs, ignore_index=True)
            # Deduplicate timestamps and sort
            if 'timestamp' in dd.columns:
                if dd['timestamp'].duplicated().any():
                    dups = int(dd['timestamp'].duplicated().sum())
                    logger.warning("[sleep_by_day] Window read: detected %d duplicate timestamps; dropping duplicates.", dups)
                    dd = dd.drop_duplicates(subset=['timestamp'], keep='first')
                dd = dd.sort_values('timestamp').reset_index(drop=True)
            return dd

        def process_participant_by_day(participant_id, output_dir, overlap_seconds=0, min_seconds=300, sampling_rate=80, skip_incomplete_days=False, day_chunks: int = 3, chunk_overlap_seconds: int = 30):
            """
            Split a participant DataFrame into calendar days, run SWaN per day,
            trim overlapping windows produced by the overlap, and save day-level
            CSV and JSON summary files under output_dir/<participant_id>/
            Returns a list of per-day result dicts.
            """
            # Determine overall span from file index
            start_day = pd.Timestamp(min(it['min_ts'] for it in file_index)).normalize()
            end_day = pd.Timestamp(max(it['max_ts'] for it in file_index)).normalize()
            cur = start_day
            out = []
            part_out_dir = Path(output_dir) / str(participant_id)
            preds_dir = part_out_dir / 'predictions'
            sums_dir = part_out_dir / 'summaries'
            preds_dir.mkdir(parents=True, exist_ok=True)
            sums_dir.mkdir(parents=True, exist_ok=True)

            all_days = []
            # Build optional allowlist of dates
            only_set = None
            if args.only_dates:
                try:
                    only_set = set(pd.to_datetime(d).date() for d in args.only_dates)
                except Exception:
                    logger.warning("--only-dates contains unparsable entries; ignoring filter: %s", args.only_dates)
                    only_set = None

            while cur <= end_day:
                day_start = pd.Timestamp(cur)
                day_end = day_start + pd.Timedelta(days=1)
                # No day-level overlap; select strictly the day window
                all_days.append(day_start.date())
                if only_set is not None and day_start.date() not in only_set:
                    logger.info("[sleep_by_day] %s: skipping due to --only-dates filter", day_start.date())
                    cur += pd.Timedelta(days=1)
                    continue
                logger.info("[sleep_by_day] %s: selecting files for strict day window %s to %s", day_start.date(), day_start, day_end)
                df_day = read_files_for_window(day_start, day_end)
                if df_day.empty:
                    out.append({'date': day_start.date().isoformat(), 'status': 'no_data'})
                    cur += pd.Timedelta(days=1)
                    continue

                # Completeness check aligned with helper: starts at 00:00:00, 23–25 hours total,
                # and ends around 23:59:xx of same day or 00:00:xx of next day.
                df_strict = df_day[(df_day['timestamp'] >= day_start) & (df_day['timestamp'] < day_end)]
                has_any = not df_strict.empty
                if has_any:
                    strict_first = df_strict['timestamp'].min()
                    strict_last = df_strict['timestamp'].max()
                    tol = pd.Timedelta(seconds=1)
                    starts_midnight = strict_first <= (day_start + tol)
                    span_seconds = (strict_last - strict_first).total_seconds()
                    duration_hours = span_seconds / 3600.0
                    has_full_day_data = 23.0 <= duration_hours <= 25.0
                    # For end time, accept near end-of-day (23:59:xx) or next-day midnight
                    end_time = (strict_last + pd.Timedelta(seconds=0)).time()
                    ends_properly = ((end_time.hour == 0 and end_time.minute == 0) or (end_time.hour == 23 and end_time.minute == 59))
                    is_full_day = starts_midnight and has_full_day_data and ends_properly
                else:
                    strict_first = None
                    strict_last = None
                    span_seconds = 0.0
                    duration_hours = 0.0
                    is_full_day = False

                duration = (df_day['timestamp'].max() - df_day['timestamp'].min()).total_seconds()
                date_str = day_start.strftime('%Y-%m-%d')
                day_out_dir = part_out_dir

                # New rule: if skipping incomplete days, skip ANY day that is not a full 24 hours
                if skip_incomplete_days and not is_full_day:
                    # Log with details so users can see which day was skipped and why
                    logger.info(
                        "[sleep_by_day] Skipping incomplete day %s for participant %s: first=%s last=%s span=%.2f h (requires 23–25h, start at 00:00 and end ~23:59 or next 00:00)",
                        date_str, participant_id, strict_first, strict_last, duration_hours
                    )
                    out.append({
                        'date': date_str,
                        'status': 'skipped_incomplete',
                        'reason': 'not_full_day_23_25h',
                        'strict_first': str(strict_first) if strict_first is not None else None,
                        'strict_last': str(strict_last) if strict_last is not None else None,
                        'span_seconds': span_seconds
                    })
                    cur += pd.Timedelta(days=1)
                    continue

                # Retain legacy insufficient data guard for extremely tiny slices (e.g., corrupt input)
                if duration < min_seconds:
                    out.append({'date': date_str, 'status': 'insufficient_data', 'duration_s': duration})
                    cur += pd.Timedelta(days=1)
                    continue

                # Run SWaN first pass in chunks within the day window, with small internal overlap
                import time as _time
                proc_part_id = f"{participant_id}_{date_str}"
                # Choose temp base: CLI --tmp-dir > $TMPDIR > system default
                tmp_base = args.tmp_dir or os.environ.get('TMPDIR')
                # Ensure tmp base exists if provided; fall back to system temp on failure
                if tmp_base:
                    try:
                        Path(tmp_base).mkdir(parents=True, exist_ok=True)
                    except Exception as _e:
                        logger.warning("[sleep_by_day] Could not create tmp base %s: %s. Falling back to system temp.", tmp_base, _e)
                        tmp_base = None
                # Compute chunk boundaries
                chunks = max(1, int(day_chunks))
                ovlp = pd.Timedelta(seconds=max(0, int(chunk_overlap_seconds)))
                total_td = (day_end - day_start)
                step_td = total_td / chunks
                chunk_results = []
                failed_chunks = 0
                try:
                    with tempfile.TemporaryDirectory(prefix=f"swan_{proc_part_id}_", dir=tmp_base) as tmpd:
                        logger.info("[sleep_by_day] Using temp dir: %s", tmpd)
                        
                        # Define recursive subdivision function for robust processing
                        def process_chunk_recursive(chunk_df, chunk_start, chunk_end, chunk_label, depth=0):
                            """
                            Process a chunk with SWaN. If it fails after retries, subdivide and retry smaller pieces.
                            Returns: list of successful result DataFrames
                            """
                            max_depth = int(args.max_subdivision_depth)
                            min_chunk_td = pd.Timedelta(minutes=int(args.min_chunk_minutes))
                            chunk_duration = chunk_end - chunk_start
                            
                            if chunk_df.empty:
                                return []
                            
                            # Check if chunk is too small to subdivide further
                            if chunk_duration < min_chunk_td:
                                logger.warning(
                                    "[subdivision] %s chunk %s: < %d min, cannot subdivide further. GIVING UP on this segment.",
                                    date_str, chunk_label, args.min_chunk_minutes
                                )
                                return []
                            
                            # Try processing this chunk with retries
                            attempts = max(1, int(args.swan_retries))
                            worker_after_timeout = False
                            for attempt in range(1, attempts + 1):
                                try:
                                    seg_part_id = f"{proc_part_id}_{chunk_label}_d{depth}"
                                    run_worker = args.swan_use_worker or worker_after_timeout
                                    if run_worker:
                                        import subprocess as _sp
                                        worker_in = Path(tmpd) / f"{seg_part_id}_a{attempt}_worker_in.csv"
                                        worker_out = Path(tmpd) / f"{seg_part_id}_a{attempt}_swan_first_pass_raw.csv"
                                        _wdf = chunk_df.rename(columns={
                                            'timestamp': 'HEADER_TIME_STAMP',
                                            'X': 'X_ACCELERATION_METERS_PER_SECOND_SQUARED',
                                            'Y': 'Y_ACCELERATION_METERS_PER_SECOND_SQUARED',
                                            'Z': 'Z_ACCELERATION_METERS_PER_SECOND_SQUARED'
                                        }).copy()
                                        _wdf['HEADER_TIME_STAMP'] = pd.to_datetime(_wdf['HEADER_TIME_STAMP'], errors='coerce')
                                        _wdf = _wdf.dropna(subset=['HEADER_TIME_STAMP'])
                                        _wdf.to_csv(worker_in, index=False)
                                        worker_script = Path(__file__).with_name('swan_worker.py')
                                        timeout_s = int(args.swan_timeout_seconds)
                                        if attempt > 1 and args.swan_retry_timeout_seconds:
                                            timeout_s = int(args.swan_retry_timeout_seconds)
                                        cmd = [sys.executable, str(worker_script), '--input', str(worker_in), '--output', str(worker_out), '--sampling-rate', str(sampling_rate)]
                                        _sp.run(cmd, check=True, timeout=max(30, timeout_s))
                                        seg_res = pd.read_csv(worker_out)
                                        if 'HEADER_TIME_STAMP' in seg_res.columns:
                                            seg_res = seg_res.rename({'HEADER_TIME_STAMP': 'START_TIME'}, axis=1)
                                    else:
                                        timeout_s = int(args.swan_timeout_seconds)
                                        if attempt > 1 and args.swan_retry_timeout_seconds:
                                            timeout_s = int(args.swan_retry_timeout_seconds)
                                        try:
                                            with swan_alarm_timeout(timeout_s):
                                                seg_res = run_swan_first_pass(
                                                    chunk_df,
                                                    sampling_rate=sampling_rate,
                                                    output_dir=tmpd,
                                                    participant_id=seg_part_id
                                                )
                                        except SwanTimeoutError:
                                            if not args.swan_worker_fallback:
                                                raise
                                            worker_after_timeout = True
                                            logger.warning(
                                                "[subdivision] %s chunk %s depth=%d: normal SWaN timed out on attempt %d; retrying this chunk in worker",
                                                date_str,
                                                chunk_label,
                                                depth,
                                                attempt,
                                            )
                                            import subprocess as _sp
                                            worker_in = Path(tmpd) / f"{seg_part_id}_a{attempt}_fallback_worker_in.csv"
                                            worker_out = Path(tmpd) / f"{seg_part_id}_a{attempt}_fallback_swan_first_pass_raw.csv"
                                            _wdf = chunk_df.rename(columns={
                                                'timestamp': 'HEADER_TIME_STAMP',
                                                'X': 'X_ACCELERATION_METERS_PER_SECOND_SQUARED',
                                                'Y': 'Y_ACCELERATION_METERS_PER_SECOND_SQUARED',
                                                'Z': 'Z_ACCELERATION_METERS_PER_SECOND_SQUARED'
                                            }).copy()
                                            _wdf['HEADER_TIME_STAMP'] = pd.to_datetime(_wdf['HEADER_TIME_STAMP'], errors='coerce')
                                            _wdf = _wdf.dropna(subset=['HEADER_TIME_STAMP'])
                                            _wdf.to_csv(worker_in, index=False)
                                            worker_script = Path(__file__).with_name('swan_worker.py')
                                            cmd = [sys.executable, str(worker_script), '--input', str(worker_in), '--output', str(worker_out), '--sampling-rate', str(sampling_rate)]
                                            _sp.run(cmd, check=True, timeout=max(30, timeout_s))
                                            seg_res = pd.read_csv(worker_out)
                                            if 'HEADER_TIME_STAMP' in seg_res.columns:
                                                seg_res = seg_res.rename({'HEADER_TIME_STAMP': 'START_TIME'}, axis=1)
                                    # Success! Return results
                                    logger.info("[subdivision] %s chunk %s depth=%d: SUCCESS on attempt %d/%d (%d windows)",
                                                date_str, chunk_label, depth, attempt, attempts, len(seg_res))
                                    return [seg_res]
                                except Exception as _e:
                                    try:
                                        import subprocess as _sp
                                        is_timeout = isinstance(_e, _sp.TimeoutExpired)
                                    except Exception:
                                        is_timeout = False
                                    if is_timeout:
                                        logger.error("[subdivision] %s chunk %s depth=%d: timeout (attempt %d/%d)",
                                                     date_str, chunk_label, depth, attempt, attempts)
                                    else:
                                        logger.error("[subdivision] %s chunk %s depth=%d: failed (attempt %d/%d): %s",
                                                     date_str, chunk_label, depth, attempt, attempts, str(_e)[:100])
                            
                            # All retries failed. Try subdivision if depth allows
                            if depth >= max_depth:
                                logger.error(
                                    "[subdivision] %s chunk %s: FAILED after %d attempts at max depth %d. Cannot subdivide further.",
                                    date_str, chunk_label, attempts, max_depth
                                )
                                return []
                            
                            # Subdivide into 2 halves and recurse
                            mid_point = chunk_start + (chunk_end - chunk_start) / 2
                            logger.warning(
                                "[subdivision] %s chunk %s: subdividing at depth %d into 2 halves (%.1f min each)",
                                date_str, chunk_label, depth, (chunk_duration.total_seconds() / 60) / 2
                            )
                            
                            left_df = chunk_df[(chunk_df['timestamp'] >= chunk_start) & (chunk_df['timestamp'] < mid_point)]
                            right_df = chunk_df[(chunk_df['timestamp'] >= mid_point) & (chunk_df['timestamp'] < chunk_end)]
                            
                            results = []
                            if not left_df.empty:
                                results.extend(process_chunk_recursive(left_df, chunk_start, mid_point, f"{chunk_label}L", depth + 1))
                            if not right_df.empty:
                                results.extend(process_chunk_recursive(right_df, mid_point, chunk_end, f"{chunk_label}R", depth + 1))
                            
                            return results
                        
                        # Main chunk processing loop with recursive subdivision
                        for k in range(chunks):
                            c_start = day_start + k * step_td
                            c_end = day_start + (k + 1) * step_td
                            # Add internal overlap except on outer boundaries
                            if k > 0:
                                c_start = c_start - ovlp
                            if k < (chunks - 1):
                                c_end = c_end + ovlp
                            c_start = max(c_start, day_start)
                            c_end = min(c_end, day_end)
                            seg = df_day[(df_day['timestamp'] >= c_start) & (df_day['timestamp'] < c_end)]
                            if seg.empty:
                                logger.info("[sleep_by_day] %s chunk %d/%d: no data after filtering (%s to %s)", date_str, k+1, chunks, c_start, c_end)
                                continue
                            t0 = _time.perf_counter()
                            logger.info("[sleep_by_day] %s chunk %d/%d: processing %d rows (%s to %s)", date_str, k+1, chunks, len(seg), c_start, c_end)
                            chunk_label = f"c{k+1}of{chunks}"
                            
                            # Use recursive subdivision for robustness
                            results = process_chunk_recursive(seg, c_start, c_end, chunk_label, depth=0)
                            
                            dt = _time.perf_counter() - t0
                            if results:
                                total_windows = sum(len(r) for r in results)
                                logger.info("[sleep_by_day] %s chunk %d/%d: done in %.1fs (%d windows from %d piece(s))",
                                            date_str, k+1, chunks, dt, total_windows, len(results))
                                chunk_results.extend(results)
                            else:
                                failed_chunks += 1
                                logger.error("[sleep_by_day] %s chunk %d/%d: FAILED completely after all subdivision attempts", date_str, k+1, chunks)
                except Exception as e:
                    logger.error(f"SWaN failed for {participant_id} on {date_str}: {e}")
                    out.append({'date': date_str, 'status': 'swan_failed', 'error': str(e)})
                    cur += pd.Timedelta(days=1)
                    continue
                
                # Check results - with subdivision, we should have processed everything
                if not chunk_results:
                    logger.error("[sleep_by_day] %s: NO DATA processed (all chunks failed even after subdivision)", date_str)
                    out.append({'date': date_str, 'status': 'all_chunks_failed', 'n_failed_chunks': int(failed_chunks)})
                    cur += pd.Timedelta(days=1)
                    continue
                
                if failed_chunks > 0:
                    logger.warning("[sleep_by_day] %s: %d/%d chunks FAILED despite subdivision, but continuing with %d successful pieces",
                                   date_str, failed_chunks, chunks, len(chunk_results))
                
                swan_result = pd.concat(chunk_results, ignore_index=True)

                # Normalize time columns and drop duplicate-named columns
                if 'HEADER_TIME_STAMP' in swan_result.columns:
                    if 'START_TIME' in swan_result.columns:
                        # Both exist; drop HEADER_TIME_STAMP to avoid duplicate label issues
                        swan_result = swan_result.drop(columns=['HEADER_TIME_STAMP'])
                    else:
                        swan_result = swan_result.rename({'HEADER_TIME_STAMP': 'START_TIME'}, axis=1)
                # Drop any duplicate column names (keep first occurrence)
                if swan_result.columns.duplicated().any():
                    dup_names = list(swan_result.columns[swan_result.columns.duplicated()])
                    logger.warning("[sleep_by_day] Dropping duplicate columns after concat: %s", dup_names)
                    swan_result = swan_result.loc[:, ~swan_result.columns.duplicated()]

                # Ensure START_TIME exists
                if 'START_TIME' not in swan_result.columns:
                    logger.error("[sleep_by_day] Missing START_TIME in SWaN results; available: %s", list(swan_result.columns))
                    out.append({'date': date_str, 'status': 'swan_failed', 'error': 'missing_START_TIME'})
                    cur += pd.Timedelta(days=1)
                    continue
                # Coerce START_TIME to datetime robustly, ensuring a single Series
                st = swan_result['START_TIME']
                if isinstance(st, pd.DataFrame):
                    st = st.iloc[:, 0]
                swan_result['START_TIME'] = pd.to_datetime(st, errors='coerce')
                if 'STOP_TIME' in swan_result.columns:
                    stp = swan_result['STOP_TIME']
                    if isinstance(stp, pd.DataFrame):
                        stp = stp.iloc[:, 0]
                    swan_result['STOP_TIME'] = pd.to_datetime(stp, errors='coerce')
                # Trim to strict calendar day
                swan_trim = swan_result[(swan_result['START_TIME'] >= day_start) & (swan_result['START_TIME'] < day_end)].copy()
                # Remove duplicate rows (default keeps first)
                swan_trim = swan_trim.drop_duplicates()
                if 'STATE' not in swan_trim.columns and 'PREDICTED' in swan_trim.columns:
                    prediction_map = {0: 'WEAR', 1: 'SLEEP', 2: 'NON-WEAR'}
                    swan_trim['STATE'] = pd.to_numeric(
                        swan_trim['PREDICTED'], errors='coerce'
                    ).map(prediction_map)

                pred_file = preds_dir / f"{date_str}_sleep_predictions.csv"
                summary_file = sums_dir / f"{date_str}_sleep_summary.json"
                # Ensure directories exist before saving (defensive check)
                preds_dir.mkdir(parents=True, exist_ok=True)
                sums_dir.mkdir(parents=True, exist_ok=True)
                swan_trim.to_csv(pred_file, index=False)

                # Build summary
                try:
                    sleep_windows = int((swan_trim['PREDICTED'] == 1).sum()) if 'PREDICTED' in swan_trim else None
                    total_windows = len(swan_trim)
                except Exception:
                    sleep_windows = None
                    total_windows = len(swan_trim)

                summary = {
                    'participant_id': participant_id,
                    'date': date_str,
                    'n_windows': int(total_windows),
                    'sleep_windows': sleep_windows,
                    'duration_s': duration
                }
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)

                out.append({'date': date_str, 'status': 'processed', 'pred_file': str(pred_file), 'summary_file': str(summary_file)})
                cur += pd.Timedelta(days=1)

            return out

        # If user requested by-day processing, run that and exit
        if args.by_day:
            logger.info("Processing participant by calendar day (--by-day)")
            results = process_participant_by_day(
                args.participant_id,
                args.output_dir,
                overlap_seconds=0,
                min_seconds=300,
                sampling_rate=args.sampling_rate,
                skip_incomplete_days=args.skip_incomplete_days,
                day_chunks=args.day_chunks,
                chunk_overlap_seconds=args.chunk_overlap_seconds
            )
            processed_cnt = sum(1 for r in results if r.get('status') == 'processed')
            skipped_cnt = sum(1 for r in results if r.get('status') == 'skipped_incomplete')
            logger.info(
                "By-day processing complete: %d processed, %d skipped as incomplete (total days=%d)",
                processed_cnt, skipped_cnt, len(results)
            )
            # If user requested to skip incomplete days and none remain, error out so the pipeline stops
            if args.skip_incomplete_days and processed_cnt == 0:
                logger.error(
                    "No complete days found for participant %s after applying completeness rule (start 00:00 and end ~23:59 or next 00:00, 23–25h). Skipping further steps.",
                    args.participant_id
                )
                print(json.dumps(results, indent=2))
                sys.exit(2)
            print(json.dumps(results, indent=2))
            sys.exit(0)
        
        # Run SWaN first pass (non by-day)
        if df_full is None:
            logger.info("Loading full data once (non by-day path)")
            df_full = load_raw_sensor_data(participant_dir)
        results = run_swan_first_pass(
            df_full,
            sampling_rate=args.sampling_rate,
            output_dir=args.output_dir,
            participant_id=args.participant_id
        )
        
        logger.info("Sleep classification completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during sleep classification: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
