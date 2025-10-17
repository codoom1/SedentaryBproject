#!/usr/bin/env python3
"""
Sleep Classification for SedentaryBehaviour Project using SWaN_accel Package

This script works with raw accelerometer data files from NHANES and applies 
SWaN sleep classification to detect sleep/wake periods.

The SWaN (Sleep/Wake Analysis) algorithm uses accelerometer data to predict
sleep and wake periods. This script implements both first pass (initial
sleep detection) and optional second pass (refinement with self-reported logs).

Usage:
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12
    python scripts/sleep_scripts/sleep_classify.py --participant-id 62163 --data-dir data/raw/2011-12 --output-dir data/sleep_predictions

Author: Generated for SedentaryBehaviour project
Date: October 2025
"""

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

# Try to import SWaN_accel
try:
    import SWaN_accel as swan
    from SWaN_accel import swan_first_pass, swan_second_pass, classify
    SWAN_AVAILABLE = True
except ImportError:
    print("Warning: SWaN_accel package not found. Please install it using:")
    print("  pip install SWaN_accel")
    SWAN_AVAILABLE = False

# Disable logging: define a no-op logger so existing calls don't output
class _NoOpLogger:
    def debug(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def setLevel(self, *args, **kwargs):
        pass

logger = _NoOpLogger()


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
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                # Fallback if format doesn't match (e.g., missing milliseconds)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            
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
    swan_df['HEADER_TIME_STAMP'] = pd.to_datetime(swan_df['HEADER_TIME_STAMP'])
    
    # Create temporary output file
    if output_dir and participant_id:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_output = output_dir / f"{participant_id}_swan_first_pass_raw.csv"
    else:
        temp_output = "temp_swan_output.csv"
    
    logger.info(f"Processing {len(swan_df)} data points...")
    logger.info(f"Date range: {swan_df['HEADER_TIME_STAMP'].min()} to {swan_df['HEADER_TIME_STAMP'].max()}")
    
    # Import SWAN first pass function
    from SWaN_accel import swan_first_pass
    
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
    result_summary.rename(columns={'HEADER_TIME_STAMP': 'START_TIME'}, inplace=True)
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
    parser.add_argument('--skip-incomplete-days', action='store_true', help='Skip first and last sensor files (incomplete days)')
    
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
        # Load raw data (do NOT assume each sensor file corresponds to a calendar day)
        logger.info(f"Loading data for participant {args.participant_id}")
        df = load_raw_sensor_data(participant_dir)

        def process_participant_by_day(df, participant_id, output_dir, overlap_seconds=30, min_seconds=300, sampling_rate=80, skip_incomplete_days=False):
            """
            Split a participant DataFrame into calendar days, run SWaN per day,
            trim overlapping windows produced by the overlap, and save day-level
            CSV and JSON summary files under output_dir/<participant_id>/
            Returns a list of per-day result dicts.
            """
            df = df.sort_values('timestamp').reset_index(drop=True)
            start_day = df['timestamp'].min().normalize()
            end_day = df['timestamp'].max().normalize()
            cur = start_day
            out = []
            part_out_dir = Path(output_dir) / str(participant_id)
            preds_dir = part_out_dir / 'predictions'
            sums_dir = part_out_dir / 'summaries'
            preds_dir.mkdir(parents=True, exist_ok=True)
            sums_dir.mkdir(parents=True, exist_ok=True)

            all_days = []
            while cur <= end_day:
                day_start = pd.Timestamp(cur)
                day_end = day_start + pd.Timedelta(days=1)
                slice_start = day_start - pd.Timedelta(seconds=overlap_seconds)
                slice_end = day_end + pd.Timedelta(seconds=overlap_seconds)
                all_days.append(day_start.date())
                df_day = df[(df['timestamp'] >= slice_start) & (df['timestamp'] < slice_end)].copy()
                if df_day.empty:
                    out.append({'date': day_start.date().isoformat(), 'status': 'no_data'})
                    cur += pd.Timedelta(days=1)
                    continue

                duration = (df_day['timestamp'].max() - df_day['timestamp'].min()).total_seconds()
                date_str = day_start.strftime('%Y-%m-%d')
                day_out_dir = part_out_dir

                if duration < min_seconds:
                    # If skip_incomplete_days is enabled and this is the first or last calendar day,
                    # mark as skipped_incomplete instead of insufficient_data
                    is_first_day = (day_start.date() == all_days[0])
                    is_last_day = (day_start.date() == all_days[-1])
                    if skip_incomplete_days and (is_first_day or is_last_day):
                        out.append({'date': date_str, 'status': 'skipped_incomplete', 'duration_s': duration})
                        cur += pd.Timedelta(days=1)
                        continue
                    out.append({'date': date_str, 'status': 'insufficient_data', 'duration_s': duration})
                    cur += pd.Timedelta(days=1)
                    continue

                # Run SWaN first pass on the day's slice in a temporary directory to avoid
                # polluting the day_out_dir with SWaN's intermediate outputs. We'll then
                # save only the trimmed predictions to preds_dir and the summary to sums_dir.
                proc_part_id = f"{participant_id}_{date_str}"
                try:
                    with tempfile.TemporaryDirectory(prefix=f"swan_{proc_part_id}_") as tmpd:
                        swan_result = run_swan_first_pass(
                            df_day,
                            sampling_rate=sampling_rate,
                            output_dir=tmpd,
                            participant_id=proc_part_id
                        )
                        # run_swan_first_pass will write its own files in tmpd; we'll read the
                        # returned DataFrame and move only what we want below.
                except Exception as e:
                    logger.error(f"SWaN failed for {participant_id} on {date_str}: {e}")
                    out.append({'date': date_str, 'status': 'swan_failed', 'error': str(e)})
                    cur += pd.Timedelta(days=1)
                    continue

                # swan_result is a window-level DataFrame with START_TIME column
                if 'START_TIME' not in swan_result.columns:
                    # If different naming, try HEADER_TIME_STAMP
                    if 'HEADER_TIME_STAMP' in swan_result.columns:
                        swan_result = swan_result.rename(columns={'HEADER_TIME_STAMP': 'START_TIME'})

                swan_result['START_TIME'] = pd.to_datetime(swan_result['START_TIME'])
                # Trim to strict calendar day
                swan_trim = swan_result[(swan_result['START_TIME'] >= day_start) & (swan_result['START_TIME'] < day_end)].copy()
                swan_trim = swan_trim.drop_duplicates(subset=['START_TIME', 'STOP_TIME', 'PREDICTED'])

                pred_file = preds_dir / f"{date_str}_sleep_predictions.csv"
                summary_file = sums_dir / f"{date_str}_sleep_summary.json"
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
            results = process_participant_by_day(df, args.participant_id, args.output_dir, overlap_seconds=30, min_seconds=300, sampling_rate=args.sampling_rate, skip_incomplete_days=args.skip_incomplete_days)
            logger.info(f"By-day processing complete: {results}")
            print(json.dumps(results, indent=2))
            sys.exit(0)
        
        # Run SWaN first pass
        results = run_swan_first_pass(
            df,
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