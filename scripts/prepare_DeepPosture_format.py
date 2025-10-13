#!/usr/bin/env python3
"""
Prepare data in DeepPostures format for each participant.
Expects raw .sensor.csv files already downloaded in data/raw/<cycle>/<participant_id>/ directory.
Converts them to ActiGraph format in data/processed/<participant_id>/ directory.

Usage:
    python scripts/prepare_deeppostures_format.py <dataset_name> <participant_id>
    python scripts/prepare_deeppostures_format.py "2011-12" 62163
    python scripts/prepare_deeppostures_format.py "2013-14" 62161 --delete-raw
"""

import os
import sys
import shutil
from pathlib import Path
import re
import pandas as pd
import argparse
from datetime import datetime


def get_sensor_files(part_dir):
    """Return all .sensor.csv or .sensor.csv.gz files in a directory."""
    part_path = Path(part_dir)
    if not part_path.exists():
        return []
    files = list(part_path.glob("*.sensor.csv")) + list(part_path.glob("*.sensor.csv.gz"))
    return sorted(files)


def extract_date_from_filename(fname):
    """Extract YYYY-MM-DD from filename."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", fname)
    return m.group(0) if m else None


def extract_time_from_filename(fname):
    """Extract HH:MM:SS from filename pattern YYYY-MM-DD-HH-MM-SS-sss."""
    m = re.search(r"\d{4}-\d{2}-\d{2}-(\d{2})-(\d{2})-(\d{2})-\d{3}", fname)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return None


def prepare_deeppostures_format(
    dataset_name,
    participant_id,
    dest_dir="data/raw",
    processed_dir="data/processed",
    delete_raw=False,
    delete_newformat=False
):
    """
    Convert raw .sensor.csv files to ActiGraph format for DeepPostures.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., "2011-12" or "2013-14")
    participant_id : int or str
        Participant SEQN identifier
    dest_dir : str
        Base directory containing raw data (default: "data/raw")
    processed_dir : str
        Base directory for processed output (default: "data/processed")
    delete_raw : bool
        Whether to delete raw files after processing (default: False)
    delete_newformat : bool
        Whether to delete processed files after creation (default: False)
    """
    print(f"\nProcessing participant: {participant_id} in dataset {dataset_name} ...")
    
    # Check if raw data exists
    part_dir = Path(dest_dir) / dataset_name / str(participant_id)
    
    if not part_dir.exists():
        print(f"[ERROR] Participant directory not found: {part_dir}")
        print(f"        Please ensure raw data exists at this location.")
        return False
    
    sensor_files = get_sensor_files(part_dir)
    
    if not sensor_files:
        print(f"[ERROR] No sensor files found for participant {participant_id} in {part_dir}")
        return False
    
    print(f"[FOUND] {len(sensor_files)} sensor file(s) for participant {participant_id}")
    
    # Create output directory for this participant
    out_dir = Path(processed_dir) / str(participant_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract date from filename (YYYY-MM-DD)
    files_by_date = {}
    for sensor_file in sensor_files:
        date = extract_date_from_filename(sensor_file.name)
        if date:
            files_by_date.setdefault(date, []).append(sensor_file)
    
    if not files_by_date:
        print(f"[ERROR] Could not extract dates from filenames")
        return False
    
    print(f"[INFO] Found {len(files_by_date)} unique date(s)")
    
    # Process each date
    for date, files in sorted(files_by_date.items()):
        day_data_list = []
        
        for sensor_file in files:
            try:
                if sensor_file.suffix == ".gz":
                    df = pd.read_csv(sensor_file, compression="gzip")
                else:
                    df = pd.read_csv(sensor_file)
                
                # Check for required columns
                keep_cols = [c for c in ["X", "Y", "Z"] if c in df.columns]
                if len(keep_cols) < 3:
                    print(f"[WARNING] Missing X/Y/Z columns in {sensor_file.name}")
                    continue
                
                day_data_list.append(df[keep_cols])
                
            except Exception as e:
                print(f"[WARNING] Failed to read {sensor_file.name}: {e}")
                continue
        
        if not day_data_list:
            print(f"    [SKIP] No valid data for calendar day: {date}")
            continue
        
        # Combine all data for this day
        day_data = pd.concat(day_data_list, ignore_index=True)
        
        # Extract earliest start time from filenames for this day
        start_times = []
        for f in files:
            t = extract_time_from_filename(f.name)
            if t:
                start_times.append(t)
        
        if start_times:
            start_time = sorted(start_times)[0]
        else:
            start_time = "00:00:00"
        
        # Prepare ActiGraph/ActiLife header
        date_obj = pd.to_datetime(date)
        now = datetime.now()
        
        header_lines = [
            "------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.3 Firmware v3.2.1 date format M/d/yyyy at 80 Hz  Filter Normal -----------",
            "Serial Number: UNKNOWN",
            f"Start Time {start_time}",
            f"Start Date {date_obj.strftime('%m/%d/%Y')}",
            "Epoch Period (hh:mm:ss) 00:00:00",
            f"Download Time {now.strftime('%H:%M:%S')}",
            f"Download Date {now.strftime('%m/%d/%Y')}",
            "Current Memory Address: 0",
            "Current Battery Voltage: 4.07     Mode = 12",
            "--------------------------------------------------"
        ]
        
        # Write to file in ActiGraph format
        out_file = out_dir / f"{date}.csv"
        with open(out_file, "w") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("Accelerometer X,Accelerometer Y,Accelerometer Z\n")
            day_data.to_csv(f, index=False, header=False)
        
        print(f"    [SAVED] ActiGraph format for {date}: {out_file}")
    
    # Optionally delete raw files
    if delete_raw:
        for sensor_file in sensor_files:
            sensor_file.unlink()
            print(f"    [DELETED] Raw file: {sensor_file.name}")
        # Remove directory if empty
        if part_dir.exists() and not any(part_dir.iterdir()):
            part_dir.rmdir()
            print(f"    [DELETED] Empty directory: {part_dir}")
    
    # Optionally delete processed files
    if delete_newformat:
        for out_file in out_dir.glob("*.csv"):
            out_file.unlink()
            print(f"    [DELETED] Processed file: {out_file.name}")
        # Remove directory if empty
        if out_dir.exists() and not any(out_dir.iterdir()):
            out_dir.rmdir()
            print(f"    [DELETED] Empty directory: {out_dir}")
    
    print(f"    [COMPLETE] Processing finished for participant {participant_id}")
    return True


def prepare_multiple_participants(
    dataset_name,
    participant_ids,
    dest_dir="data/raw",
    processed_dir="data/processed",
    delete_raw=False,
    delete_newformat=False
):
    """
    Process multiple participants.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., "2011-12" or "2013-14")
    participant_ids : list
        List of participant SEQN identifiers
    dest_dir : str
        Base directory containing raw data (default: "data/raw")
    processed_dir : str
        Base directory for processed output (default: "data/processed")
    delete_raw : bool
        Whether to delete raw files after processing (default: False)
    delete_newformat : bool
        Whether to delete processed files after creation (default: False)
    
    Returns
    -------
    dict
        Dictionary with success/failure counts
    """
    results = {"success": 0, "failed": 0}
    
    for pid in participant_ids:
        success = prepare_deeppostures_format(
            dataset_name,
            pid,
            dest_dir=dest_dir,
            processed_dir=processed_dir,
            delete_raw=delete_raw,
            delete_newformat=delete_newformat
        )
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total participants: {len(participant_ids)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(
        description='Convert raw NHANES accelerometer data to DeepPostures ActiGraph format'
    )
    
    parser.add_argument(
        'dataset_name',
        help='Dataset name (e.g., "2011-12" or "2013-14")'
    )
    
    parser.add_argument(
        'participant_id',
        help='Participant SEQN identifier'
    )
    
    parser.add_argument(
        '--dest-dir',
        default='data/raw',
        help='Base directory containing raw data (default: data/raw)'
    )
    
    parser.add_argument(
        '--processed-dir',
        default='data/processed',
        help='Base directory for processed output (default: data/processed)'
    )
    
    parser.add_argument(
        '--delete-raw',
        action='store_true',
        help='Delete raw files after processing'
    )
    
    parser.add_argument(
        '--delete-newformat',
        action='store_true',
        help='Delete processed files after creation'
    )
    
    args = parser.parse_args()
    
    success = prepare_deeppostures_format(
        args.dataset_name,
        args.participant_id,
        dest_dir=args.dest_dir,
        processed_dir=args.processed_dir,
        delete_raw=args.delete_raw,
        delete_newformat=args.delete_newformat
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
