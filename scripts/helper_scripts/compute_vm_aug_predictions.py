"""
Compute 10-second vector magnitude (VM) features from raw accelerometer CSV files.

Overview
--------
This script parses daily accelerometer CSV files that contain a small header with
"Start Time" and "Start Date" lines followed by a column header line:

		Accelerometer X,Accelerometer Y,Accelerometer Z

Assumptions and conventions:
- Sampling rate is 80 Hz (80 samples per second).
- VM is computed per-sample as sqrt(x^2 + y^2 + z^2).
- Features are aggregated in non-overlapping 10-second windows (800 samples per window).
- Each input CSV is assumed to represent (roughly) a single day of data.

Outputs
-------
- For each input day, a per-day CSV is written with columns:
	[timestamp (window start), avg_vm_10s, is_midnight_block]
- Optionally, a combined CSV that concatenates all per-day VM windows is produced.

Augmentation (optional)
-----------------------
If a predictions CSV for the same day is provided (via --predictions-csv), the
script will merge VM features with predictions on the 'timestamp' column and
write a single augmented output file containing both predictions and VM fields.

Sleep predictions (30s -> 10s)
------------------------------
When a sleep predictions CSV (30-second cadence) is provided via --sleep-predictions-csv,
the script expands each 30s row into three 10s rows (at t, t+10s, t+20s) duplicating
the sleep prediction values so they align with the 10s VM and posture prediction windows.

CLI usage (examples)
--------------------
VM only (single day; complete-day enforced):
	python "scripts/helper scripts/compute_vm_aug_predictions.py" \
		--input-csv data/processed/62596/2011-12-01.csv \
		--output-dir data/features/62596

Augment posture predictions + VM (single output file):
	python "scripts/helper scripts/compute_vm_aug_predictions.py" \
		--input-csv data/processed/62596/2011-12-01.csv \
		--predictions-csv data/predictions/62596/CHAP_ALL_ADULTS/2011-12-01.csv \
		--output-path data/augmented_predictions/62596/CHAP_ALL_ADULTS/2011-12-01.csv \
		--participant-id 62596

Augment sleep (30s->10s) + VM (single output file):
	python "scripts/helper scripts/compute_vm_aug_predictions.py" \
		--input-csv data/processed/62596/2011-12-01.csv \
		--sleep-predictions-csv data/sleep_predictions/62596/predictions/2011-12-01_sleep_predictions.csv \
		--output-path data/augmented_predictions/62596/sleep_only/2011-12-01.csv \
		--participant-id 62596

Augment posture + sleep (expanded to 10s) + VM together:
	python "scripts/helper scripts/compute_vm_aug_predictions.py" \
		--input-csv data/processed/62596/2011-12-01.csv \
		--predictions-csv data/predictions/62596/CHAP_ALL_ADULTS/2011-12-01.csv \
		--sleep-predictions-csv data/sleep_predictions/62596/predictions/2011-12-01_sleep_predictions.csv \
		--output-path data/augmented_predictions/62596/CHAP_ALL_ADULTS/2011-12-01_augmented.csv \
		--participant-id 62596

Auto-run pipeline if input CSV is missing (downloads if needed):
	python scripts/helper_scripts/compute_vm_aug_predictions.py --input-csv data/processed/62596/2000-01-12.csv --output-path data/augmented_predictions/62596/CHAP_ALL_ADULTS/2011-12-01.csv --participant-id 62596 --date 2000-01-12

Participant-first workflow (no dates needed; auto-run default):
	# Process ALL available days for a participant; cycle inferred from constants
	python scripts/helper_scripts/compute_vm_aug_predictions.py --participant-id 62596 --posture-model CHAP_ALL_ADULTS

	# Process a specific day only
	python scripts/helper_scripts/compute_vm_aug_predictions.py --participant-id 62596 --date 2011-12-01

Notes
-----
- The script contains utilities to validate whether a file appears to be a
	"complete" day (starts at 00:00:00 and has ~24 hours of data).
- The code aims not to change input values precision but will report diagnostics
	about apparent decimal precision in the data.

Wrapper
-------
This module also exposes a convenience function `run_pipeline_and_augment_single_day` that
uses `scripts/run_participant_pipeline.py` to generate predictions, then computes VM for a
target day and merges posture and sleep predictions (expanding sleep from 30s to 10s) into
one augmented CSV output.

Python usage example (wrapper):
	from pathlib import Path
	from scripts.helper_scripts.compute_vm import run_pipeline_and_augment_single_day

	out_path = run_pipeline_and_augment_single_day(
			participant_id="62596",
			date_str="2011-12-01",
			cycle="2011-12",
			posture_model="CHAP_ALL_ADULTS",
			# sleep_conda_env="sklearn023", posture_conda_env="deepposture",  # optional
	)
	print("Augmented file:", out_path)
"""

import os
import sys
import argparse
import numpy as np
import csv
from datetime import datetime, timedelta
import pandas as pd
import subprocess
from pathlib import Path
import shutil
from typing import Optional, List


def parse_header_and_start_time(csv_path):
	"""
	Read the entire CSV, parse header metadata, and locate the first data row.

	Args:
		csv_path (str): Path to the daily accelerometer CSV file.

	Returns:
		tuple[datetime, int, list[str]]: A tuple containing
			- start_datetime: The recording start datetime parsed from header.
			- data_start_line_idx: The 0-based index of the first data row (after headers).
			- lines: All lines from the file (kept to avoid a second file read).

	Raises:
		ValueError: If required header fields or data start line cannot be found.
	"""
	with open(csv_path, 'r') as f:
		lines = f.readlines()
	start_time = None
	start_date = None
	data_start_idx = None
	for i, line in enumerate(lines):
		if line.startswith('Start Time'):
			start_time = line.split('Start Time')[1].strip()
		if line.startswith('Start Date'):
			start_date = line.split('Start Date')[1].strip()
		if line.strip() == 'Accelerometer X,Accelerometer Y,Accelerometer Z':
			data_start_idx = i + 1
			break
	if not (start_time and start_date and data_start_idx):
		raise ValueError(f"Header parsing failed for {csv_path}")
	# Parse datetime
	dt_str = f"{start_date} {start_time}"
	try:
		start_dt = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
	except Exception:
		# Try alternative format
		start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
	return start_dt, data_start_idx, lines


def process_csv(csv_path):
	"""
	Compute 10-second VM features from a single daily accelerometer CSV.

	Steps:
	1) Parse header to obtain recording start time and locate the first data row.
	2) Load the accelerometer rows (x, y, z) as floats.
	3) Partition the sequence into non-overlapping 10-second windows (800 samples).
	4) For each window, compute per-sample VM and then average VM over the window.
	5) Emit tuples of (window_start_timestamp, avg_vm_10s, is_midnight_block).

	Assumptions:
	- Sampling frequency is 80 Hz.
	- Input lines after the header contain exactly 3 comma-separated values.

	Args:
		csv_path (str): Path to the daily accelerometer CSV file.

	Returns:
		list[tuple[str, float, int]]: List of VM features for each 10s window.
			Each tuple is ("YYYY-MM-DD HH:MM:SS", avg_vm_10s, is_midnight_block).
	"""
	start_dt, data_start_idx, lines = parse_header_and_start_time(csv_path)
	# Read data rows
	data_rows = []
	for line in lines[data_start_idx:]:
		parts = line.strip().split(',')
		if len(parts) != 3:
			continue
		try:
			x, y, z = map(float, parts)
			data_rows.append((x, y, z))
		except Exception:
			continue
	# Group into 10s windows (80Hz)
	window_size = 80 * 10
	n_windows = len(data_rows) // window_size
	vm_features = []

	print(f"Processing {len(data_rows)} data points into {n_windows} windows of {window_size} samples each")

	# Check data precision by examining first few samples
	if len(data_rows) > 0:
		sample_precision_analysis = []
		for i, (x, y, z) in enumerate(data_rows[:10]):
			# Count decimal places for each axis
			x_str, y_str, z_str = str(x), str(y), str(z)
			x_decimals = len(x_str.split('.')[-1]) if '.' in x_str else 0
			y_decimals = len(y_str.split('.')[-1]) if '.' in y_str else 0
			z_decimals = len(z_str.split('.')[-1]) if '.' in z_str else 0
			sample_precision_analysis.append((x_decimals, y_decimals, z_decimals))

		max_decimals = max(max(row) for row in sample_precision_analysis)
		print(f"Input data precision: up to {max_decimals} decimal places")
		print(f"Sample data: {data_rows[0]} -> VM = {np.sqrt(sum(x**2 for x in data_rows[0])):.6f}")

	for i in range(n_windows):
		window = data_rows[i * window_size:(i + 1) * window_size]

		# Simple VM computation - no need for high precision if input is limited
		vms = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in window]
		avg_vm = np.mean(vms)  # Standard precision is fine

		# Check for identical values (likely due to data precision limits)
		unique_vms = len(set(f"{vm:.3f}" for vm in vms))  # Round to 3dp like input data
		if unique_vms == 1:
			print(f"INFO: Window {i} has identical VM values (rounded to 3dp): {vms[0]:.3f}")
		elif unique_vms < 50:  # Less than 50 unique values in 800 samples suggests limited precision
			print(f"INFO: Window {i} has only {unique_vms} unique VM values (3dp precision) out of {len(vms)} samples")

		window_start = start_dt + timedelta(seconds=i * 10)
		is_midnight_block = 1 if window_start.time() == datetime.strptime("00:00:00", "%H:%M:%S").time() else 0

		# Store with appropriate precision (3 decimal places to match input)
		vm_features.append((
			window_start.strftime("%Y-%m-%d %H:%M:%S"),
			round(avg_vm, 6),  # Keep 6 decimals for the average, but understand precision limits
			is_midnight_block
		))

		# Debug output for first few windows
		if i < 3:
			print(f"Window {i}: avg_vm={avg_vm:.6f}, unique_values={unique_vms}, samples={len(vms)}")

	return vm_features


def process_csv_to_daily_vm(csv_path, out_dir):
	"""
	Process a single day CSV into a VM features CSV file.

	Args:
		csv_path (str): Path to input daily accelerometer CSV.
		out_dir (str): Directory where the per-day VM CSV will be written.

	Returns:
		str | None: Path to the generated per-day VM CSV on success, else None.

	Side effects:
		Writes a CSV file named "vm_<input_basename>.csv" into out_dir.

	Notes:
		The output schema is: timestamp, avg_vm_10s, is_midnight_block.
	"""
	try:
		# Extract date from filename for output
		fname = os.path.basename(csv_path)
		if fname.endswith('.csv'):
			date_part = fname[:-4]  # Remove .csv extension
		else:
			date_part = fname

		# Compute VM features for this day
		vm_features = process_csv(csv_path)

		if not vm_features:
			print(f"[WARN] No VM features computed for {fname}")
			return None

		# Write to individual day file
		out_path = os.path.join(out_dir, f'vm_{date_part}.csv')
		with open(out_path, 'w', newline='') as fout:
			writer = csv.writer(fout)
			writer.writerow(['timestamp', 'avg_vm_10s', 'is_midnight_block'])
			for row in vm_features:
				writer.writerow(row)

		print(f"[SUCCESS] Wrote {len(vm_features)} VM windows to {out_path}")
		return out_path

	except Exception as e:
		print(f"[ERROR] Failed to process {csv_path}: {e}")
		return None


def parse_header_only(csv_path):
	"""
	Parse only the header portion of a CSV to get start time and data start index.

	This avoids reading the entire file into memory. The function streams lines
	until it finds the line immediately preceding data.

	Args:
		csv_path (str): Path to the daily accelerometer CSV file.

	Returns:
		tuple[datetime, int]: start_datetime and data_start_line_idx.

	Raises:
		ValueError: If header fields or data start marker are not found.
	"""
	with open(csv_path, 'r') as f:
		start_time = None
		start_date = None
		data_start_idx = None

		for i, line in enumerate(f):
			if line.startswith('Start Time'):
				start_time = line.split('Start Time')[1].strip()
			elif line.startswith('Start Date'):
				start_date = line.split('Start Date')[1].strip()
			elif line.strip() == 'Accelerometer X,Accelerometer Y,Accelerometer Z':
				data_start_idx = i + 1
				break

		if not (start_time and start_date and data_start_idx):
			raise ValueError(f"Header parsing failed for {csv_path}")

		# Parse datetime
		dt_str = f"{start_date} {start_time}"
		try:
			start_dt = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
		except Exception:
			# Try alternative format
			start_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

	return start_dt, data_start_idx


def check_day_completeness(csv_path):
	"""
	Heuristically determine whether the CSV represents a complete day of data.

	Criteria:
	- Start time must be exactly 00:00:00.
	- Duration must be approximately a full day (23–25 hours allowed).
	- End time must be either 00:00 of next day or 23:59:xx of same day.

	Args:
		csv_path (str): Path to the daily accelerometer CSV file.

	Returns:
		tuple[bool, datetime.time | None, datetime.time | None, str]:
			(is_complete, start_time, end_time, reason/description)
	"""
	try:
		# Parse header efficiently
		start_dt, data_start_idx = parse_header_only(csv_path)

		# Count total lines efficiently using wc -l
		import subprocess
		result = subprocess.run(['wc', '-l', csv_path], capture_output=True, text=True)
		total_lines = int(result.stdout.split()[0])

		# Estimate data rows (total lines - header lines)
		data_row_count = total_lines - data_start_idx

		# Estimate end time (assuming 80 Hz sampling)
		duration_seconds = data_row_count / 80.0
		end_dt = start_dt + timedelta(seconds=duration_seconds)

		start_time = start_dt.time()
		end_time = end_dt.time()

		# Check if day is complete
		starts_at_midnight = (start_time.hour == 0 and start_time.minute == 0 and start_time.second == 0)

		# For end time, a complete 24-hour day will end at 00:00:00 of the next day
		# OR in the 23:59:xx minute (for slightly less than 24 hours)
		# We'll accept both as "complete"
		duration_hours = duration_seconds / 3600.0
		ends_properly = (
			(end_time.hour == 0 and end_time.minute == 0) or  # Exactly 24 hours
			(end_time.hour == 23 and end_time.minute == 59)   # Almost 24 hours
		)

		# Also check that we have approximately a full day of data (23-25 hours)
		has_full_day_data = 23.0 <= duration_hours <= 25.0

		if not starts_at_midnight:
			reason = f"starts at {start_time} (not 00:00:00)"
			return False, start_time, end_time, reason

		if not has_full_day_data:
			reason = f"only {duration_hours:.1f} hours of data (need 23-25 hours)"
			return False, start_time, end_time, reason

		if not ends_properly:
			reason = f"ends at {end_time} (should be 00:00:xx or 23:59:xx)"
			return False, start_time, end_time, reason

		return True, start_time, end_time, f"complete day ({duration_hours:.1f} hours)"

	except Exception as e:
		return False, None, None, f"error checking: {e}"



# --------------------
# Wrapper functionality
# --------------------
def _find_prediction_file(pred_dir: Path, date_str: str) -> Optional[Path]:
	"""Return a posture prediction CSV path for the given day if found.

	Tries pred_dir/date_str.csv first, then scans for files starting with date_str.
	"""
	exact = pred_dir / f"{date_str}.csv"
	if exact.exists():
		return exact
	if pred_dir.exists():
		for p in sorted(pred_dir.glob(f"{date_str}*.csv")):
			return p
	return None


def _expand_sleep_30s_to_10s(sleep_df: pd.DataFrame, target_day: str) -> Optional[pd.DataFrame]:
	"""Filter sleep predictions to target_day and expand from 30s to 10s cadence.

	Returns a DataFrame with 'timestamp' as string (YYYY-MM-DD HH:MM:SS),
	or None if no rows match the target_day.
	"""
	sdf = sleep_df.copy()
	# Normalize timestamp column from possible alternatives
	if 'timestamp' not in sdf.columns:
		for alt in ['START_TIME', 'HEADER_TIME_STAMP', 'start_time', 'StartTime']:
			if alt in sdf.columns:
				sdf = sdf.rename(columns={alt: 'timestamp'})
				break
	if 'timestamp' not in sdf.columns:
		print("[WARN] Sleep predictions CSV missing 'timestamp' (and no known alternatives); skipping sleep merge")
		return None
	sdf['timestamp'] = pd.to_datetime(sdf['timestamp'], errors='coerce')
	sdf = sdf.dropna(subset=['timestamp'])
	day_mask = sdf['timestamp'].dt.strftime('%Y-%m-%d') == target_day
	if not day_mask.any():
		return None
	sdf = sdf.loc[day_mask].copy()
	expanded = []
	for _, row in sdf.iterrows():
		base_ts = row['timestamp']
		for offset in (0, 10, 20):
			new_row = row.copy()
			new_row['timestamp'] = base_ts + pd.to_timedelta(offset, unit='s')
			expanded.append(new_row)
	out = pd.DataFrame(expanded)
	out['timestamp'] = out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
	return out


def run_pipeline_and_augment_single_day(
	participant_id: str,
	date_str: str,
	cycle: Optional[str] = None,
	posture_model: str = "CHAP_ALL_ADULTS",
	repo_root: Optional[Path] = None,
	processed_root: Optional[Path] = None,
	predictions_root: Optional[Path] = None,
	sleep_root: Optional[Path] = None,
	output_root: Optional[Path] = None,
	sleep_conda_env: str = "sklearn023",
	posture_conda_env: str = "deepposture",
	skip_incomplete_days_sleep: bool = True,
	skip_incomplete_days_posture: bool = True,
	download: bool = True,
) -> Path:
	"""
	Orchestrate pipeline -> VM compute -> augmentation for a single participant/day.

	- Runs scripts/run_participant_pipeline.py to generate posture and sleep predictions.
	- Computes VM for the daily processed accelerometer CSV (complete-day enforced).
	- Merges posture predictions and expanded sleep predictions (30s->10s) with VM.

	Args:
		participant_id: SEQN / participant ID.
		date_str: Target day in YYYY-MM-DD.
		cycle: NHANES cycle (e.g., 2011-12).
		posture_model: Posture model name to pass to posture step.
		repo_root: Repo root path; inferred from this file if None.
		processed_root: Root for processed daily CSVs (default: repo_root/data/processed).
		predictions_root: Root for posture predictions (default: repo_root/data/predictions).
		sleep_root: Root for sleep predictions (default: repo_root/data/sleep_predictions).
		output_root: Root for augmented outputs (default: repo_root/data/augmented_predictions).
		sleep_conda_env: Optional conda env for sleep step.
		posture_conda_env: Optional conda env for posture step.
		skip_incomplete_days_sleep: Pass through to pipeline sleep step.
		skip_incomplete_days_posture: Pass through to pipeline posture step.

	Returns:
		Path to the augmented CSV written under output_root/participant_id/posture_model/date_str.csv
	"""
	# Resolve roots
	# This file is at scripts/helper_scripts/, repo root is two levels up
	repo_root = repo_root or Path(__file__).resolve().parents[2]
	processed_root = processed_root or (repo_root / "data" / "processed")
	predictions_root = predictions_root or (repo_root / "data" / "predictions")
	sleep_root = sleep_root or (repo_root / "data" / "sleep_predictions")
	output_root = output_root or (repo_root / "data" / "augmented_predictions")

	# Infer cycle if not provided from constants/participants.csv
	if cycle is None:
		constants_path = repo_root / "constants" / "participants.csv"
		cycle = infer_cycle_for_participant(constants_path, participant_id)
		if cycle is None:
			raise ValueError(f"Unable to infer cycle for participant {participant_id} from {constants_path}")

	# 1) Run participant pipeline to ensure predictions exist
	pipeline_script = repo_root / "scripts" / "run_participant_pipeline.py"
	base_sleep_flags = ["--participant-id", participant_id, "--cycle", cycle]
	# Build command, include optional flags
	cmd = [sys.executable, str(pipeline_script)] + base_sleep_flags + ["--posture-model", posture_model]
	if skip_incomplete_days_sleep:
		cmd.append("--skip-incomplete-days-sleep")
	if skip_incomplete_days_posture:
		cmd.append("--skip-incomplete-days-posture")
	if sleep_conda_env:
		cmd += ["--sleep-conda-env", sleep_conda_env]
	if posture_conda_env:
		cmd += ["--posture-conda-env", posture_conda_env]

	# Request download by default
	if download:
		cmd.append("--download")

	# Execute pipeline
	subprocess.run(cmd, check=True)

	# 2) Locate input files
	pid = str(participant_id)
	input_csv = processed_root / pid / f"{date_str}.csv"
	if not input_csv.exists():
		raise FileNotFoundError(f"Processed daily CSV not found: {input_csv}")

	pred_dir = predictions_root / pid / posture_model
	pred_csv = _find_prediction_file(pred_dir, date_str)
	if not pred_csv:
		raise FileNotFoundError(f"Posture predictions CSV for {date_str} not found in {pred_dir}")

	sleep_csv_path = sleep_root / pid / "predictions" / f"{date_str}_sleep_predictions.csv"

	# 3) Compute VM rows for the day
	vm_rows = process_csv(str(input_csv))
	vm_df = pd.DataFrame(vm_rows, columns=['timestamp', 'avg_vm_10s', 'is_midnight_block'])

	# 4) Load posture predictions and normalize day/timestamp
	pred_df = pd.read_csv(pred_csv)
	if 'timestamp' not in pred_df.columns:
		raise ValueError("Predictions CSV missing 'timestamp'")
	# participant validation best-effort
	if 'participant_id' in pred_df.columns:
		pred_pids = pred_df['participant_id'].astype(str).unique()
		if not (len(pred_pids) == 1 and pred_pids[0] == pid):
			raise ValueError(f"Participant mismatch in predictions: {pred_pids} vs {pid}")
	pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], errors='coerce')
	pred_df = pred_df.dropna(subset=['timestamp'])
	pred_df['timestamp'] = pred_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
	pred_df = pred_df[pred_df['timestamp'].str.startswith(date_str)].copy()
	if pred_df.empty:
		raise ValueError(f"No predictions rows for {date_str} in {pred_csv}")

	merged = pd.merge(pred_df, vm_df, on='timestamp', how='left')

	# 5) Optional sleep expansion and merge
	if sleep_csv_path.exists():
		sleep_df = pd.read_csv(sleep_csv_path)
		if 'participant_id' in sleep_df.columns:
			sleep_pids = sleep_df['participant_id'].astype(str).unique()
			if not (len(sleep_pids) == 1 and sleep_pids[0] == pid):
				raise ValueError(f"Participant mismatch in sleep predictions: {sleep_pids} vs {pid}")
		sleep10 = _expand_sleep_30s_to_10s(sleep_df, date_str)
		if sleep10 is not None:
			merged = pd.merge(merged, sleep10, on='timestamp', how='left', suffixes=('', '_sleep'))

	# 6) Write output
	out_path = output_root / pid / posture_model / f"{date_str}.csv"
	out_path.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(out_path, index=False)
	return out_path


def infer_cycle_for_participant(constants_csv: Path, participant_id: str) -> Optional[str]:
	"""Infer cycle for a participant by scanning constants/participants.csv.

	The constants file is expected to contain rows of form: cycle,participant_id
	with an optional header. Returns the first matching cycle as a string.
	"""
	if not constants_csv.exists():
		return None
	pid = str(participant_id)
	with constants_csv.open('r', newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			if not row or row[0].strip().startswith('#'):
				continue
			if len(row) < 2:
				continue
			c0 = (row[0] or '').strip().lstrip('\ufeff').lower()
			c1 = (row[1] or '').strip().lower()
			# Skip header
			if c0 == 'cycle' and c1 == 'participant_id':
				continue
			cycle = row[0].strip()
			pid_row = row[1].strip()
			if pid_row == pid:
				return cycle
	return None


def process_participant_all_days(
	participant_id: str,
	cycle: Optional[str] = None,
	posture_model: str = "CHAP_ALL_ADULTS",
	repo_root: Optional[Path] = None,
	processed_root: Optional[Path] = None,
	predictions_root: Optional[Path] = None,
	sleep_root: Optional[Path] = None,
	output_root: Optional[Path] = None,
	sleep_conda_env: str = "sklearn023",
	posture_conda_env: str = "deepposture",
	skip_incomplete_days_sleep: bool = True,
	skip_incomplete_days_posture: bool = True,
	download: bool = True,
) -> List[Path]:
	"""Run pipeline for the participant (ensuring predictions), then process all days.

	Returns list of written augmented CSV paths.
	"""
	repo_root = repo_root or Path(__file__).resolve().parents[2]
	processed_root = processed_root or (repo_root / "data" / "processed")
	predictions_root = predictions_root or (repo_root / "data" / "predictions")
	sleep_root = sleep_root or (repo_root / "data" / "sleep_predictions")
	output_root = output_root or (repo_root / "data" / "augmented_predictions")

	# Infer cycle if not provided
	if cycle is None:
		constants_path = repo_root / "constants" / "participants.csv"
		cycle = infer_cycle_for_participant(constants_path, participant_id)
		if cycle is None:
			raise ValueError(f"Unable to infer cycle for participant {participant_id} from {constants_path}")

	# Run participant pipeline once
	pipeline_script = repo_root / "scripts" / "run_participant_pipeline.py"
	cmd = [sys.executable, str(pipeline_script), "--participant-id", participant_id, "--cycle", cycle, "--posture-model", posture_model]
	if skip_incomplete_days_sleep:
		cmd.append("--skip-incomplete-days-sleep")
	if skip_incomplete_days_posture:
		cmd.append("--skip-incomplete-days-posture")
	if sleep_conda_env:
		cmd += ["--sleep-conda-env", sleep_conda_env]
	if posture_conda_env:
		cmd += ["--posture-conda-env", posture_conda_env]
	if download:
		cmd.append("--download")
	subprocess.run(cmd, check=True)

	# Process each daily CSV under processed_root/pid/
	pid = str(participant_id)
	day_dir = processed_root / pid
	if not day_dir.exists():
		raise FileNotFoundError(f"Processed directory not found for participant {pid}: {day_dir}")

	written: List[Path] = []
	for daily_csv in sorted(day_dir.glob("*.csv")):
		date_str = daily_csv.stem
		# reuse single-day augmentation without re-running pipeline per day
		pred_dir = predictions_root / pid / posture_model
		pred_csv = _find_prediction_file(pred_dir, date_str)
		if pred_csv is None:
			# Skip days without predictions (should be rare if pipeline ran)
			continue
		# Compute VM
		vm_rows = process_csv(str(daily_csv))
		vm_df = pd.DataFrame(vm_rows, columns=['timestamp', 'avg_vm_10s', 'is_midnight_block'])
		# Load posture
		pred_df = pd.read_csv(pred_csv)
		pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], errors='coerce')
		pred_df = pred_df.dropna(subset=['timestamp'])
		pred_df['timestamp'] = pred_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
		pred_df = pred_df[pred_df['timestamp'].str.startswith(date_str)].copy()
		if pred_df.empty:
			continue
		merged = pd.merge(pred_df, vm_df, on='timestamp', how='left')
		# Sleep (optional)
		sleep_csv_path = sleep_root / pid / "predictions" / f"{date_str}_sleep_predictions.csv"
		if sleep_csv_path.exists():
			sleep_df = pd.read_csv(sleep_csv_path)
			sleep10 = _expand_sleep_30s_to_10s(sleep_df, date_str)
			if sleep10 is not None:
				merged = pd.merge(merged, sleep10, on='timestamp', how='left', suffixes=('', '_sleep'))
		# Write
		out_path = output_root / pid / posture_model / f"{date_str}.csv"
		out_path.parent.mkdir(parents=True, exist_ok=True)
		merged.to_csv(out_path, index=False)
		written.append(out_path)

	return written




def main():
	"""
	CLI entry point to compute VM features for exactly one daily CSV file.

	Command-line arguments:
		--input-csv (str, required): Path to a single daily accelerometer CSV.
		--output-dir (str): Directory where the VM CSV will be written. Default: data/features

	Behavior:
		- Validates the input CSV is a complete day per check_day_completeness().
		- Writes one per-day VM file into output-dir (no combined output).
	"""
	parser = argparse.ArgumentParser(description="Compute 10s VM features, optionally augment with posture and sleep. With only --participant-id, auto-run pipeline and process all days by default.")
	parser.add_argument('--input-csv', help='Path to a single daily accelerometer CSV file (if not provided, use --participant-id workflows)')
	parser.add_argument('--output-dir', default='data/features', help='Directory to write the per-day VM CSV file (when not augmenting)')
	parser.add_argument('--predictions-csv', help='Optional: path to a predictions CSV for the same day to augment with VM')
	parser.add_argument('--output-path', help='Optional: exact output path for augmented predictions+VM CSV (required when using --predictions-csv)')
	parser.add_argument('--sleep-predictions-csv', help='Optional: path to a sleep predictions CSV (30s cadence) to be expanded to 10s and merged')
	parser.add_argument('--participant-id', help='Participant ID; required for participant-first workflows')
	# Auto-run wrapper/pipeline options
	parser.add_argument('--auto-run-pipeline', dest='auto_run_pipeline', action='store_true', default=True, help='Run the participant pipeline automatically (default: enabled)')
	parser.add_argument('--no-auto-run-pipeline', dest='auto_run_pipeline', action='store_false', help='Disable auto-running the participant pipeline')
	parser.add_argument('--download', dest='download', action='store_true', default=True, help='Download raw data when auto-running pipeline (default: enabled)')
	parser.add_argument('--no-download', dest='download', action='store_false', help='Do not download raw data when auto-running pipeline')
	parser.add_argument('--cycle', help='NHANES cycle; if omitted, inferred from constants/participants.csv')
	parser.add_argument('--posture-model', default='CHAP_ALL_ADULTS', help='Posture model to use when auto-running pipeline')
	parser.add_argument('--date', help='Target day (YYYY-MM-DD). Required for auto-run if cannot infer from paths')
	args = parser.parse_args()

	input_csv = args.input_csv

	# Helper: infer date (YYYY-MM-DD) from provided paths or --date
	def _infer_day() -> Optional[str]:
		candidates = [args.input_csv, args.predictions_csv, args.sleep_predictions_csv]
		for p in candidates:
			if not p:
				continue
			base = os.path.basename(p)
			# Handle names like 2011-12-01_sleep_predictions.csv
			if base and len(base) >= 10:
				maybe = base[:10]
				try:
					datetime.strptime(maybe, '%Y-%m-%d')
					return maybe
				except Exception:
					pass
		return args.date

	inferred_day = _infer_day()

	if input_csv and not os.path.isfile(input_csv):
		if args.auto_run_pipeline and args.participant_id and inferred_day:
			# Use the higher-level wrapper to create augmented output end-to-end
			print("[INFO] Input CSV missing. Auto-running pipeline wrapper...")
			out_path = run_pipeline_and_augment_single_day(
				participant_id=str(args.participant_id),
				date_str=inferred_day,
				cycle=args.cycle,
				posture_model=args.posture_model,
				download=bool(args.download),
			)
			print(f"[INFO] Wrapper wrote: {out_path}")
			# If a specific output path is requested, copy to that path
			if args.output_path:
				os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
				shutil.copyfile(out_path, args.output_path)
				print(f"[INFO] Copied to requested output-path: {args.output_path}")
			return
		else:
			print(f"[ERROR] Input CSV does not exist: {input_csv}")
			if not args.participant_id:
				print("Hint: provide --participant-id and --auto-run-pipeline to fetch/generate data")
			if not inferred_day:
				print("Hint: provide --date YYYY-MM-DD or a path containing the date to auto-run")
			sys.exit(1)

	# Participant-first workflows (no input-csv) -> process all days or a specific day
	if not input_csv and args.participant_id:
		if not args.auto_run_pipeline:
			print("[ERROR] Participant-first workflow requires auto-run enabled (default)")
			sys.exit(1)
		if args.date:
			# Single day
			out_path = run_pipeline_and_augment_single_day(
				participant_id=str(args.participant_id),
				date_str=args.date,
				cycle=args.cycle,
				posture_model=args.posture_model,
				download=bool(args.download),
			)
			print(f"[INFO] Wrote: {out_path}")
			return
		else:
			# All days for participant
			written = process_participant_all_days(
				participant_id=str(args.participant_id),
				cycle=args.cycle,
				posture_model=args.posture_model,
				download=bool(args.download),
			)
			print(f"[INFO] Wrote {len(written)} augmented files for participant {args.participant_id}")
			for p in written[:5]:
				print(f" - {p}")
			if len(written) > 5:
				print(" ...")
			return

	# Enforce complete day requirement
	is_complete, start_time, end_time, reason = check_day_completeness(input_csv)
	if not is_complete:
		print(f"[ABORT] Incomplete day: {reason}")
		sys.exit(2)
	else:
		print(f"[OK] Complete day detected: starts at {start_time}, ends at {end_time}")

	# Determine target day string (YYYY-MM-DD) from the input CSV header or filename
	try:
		start_dt_hdr, _ = parse_header_only(input_csv)
		target_day = start_dt_hdr.strftime('%Y-%m-%d')
	except Exception:
		base = os.path.basename(input_csv)
		target_day = base.split('.')[0][:10]
	print(f"Target day: {target_day}")

	# If augmenting predictions (and optional sleep), require output-path and produce a single merged file
	if args.predictions_csv or args.sleep_predictions_csv:
		predictions_csv = args.predictions_csv
		output_path = args.output_path
		if not output_path:
			print("[ERROR] --output-path is required when providing predictions and/or sleep predictions")
			sys.exit(4)
		if predictions_csv and not os.path.isfile(predictions_csv):
			print(f"[ERROR] Predictions CSV does not exist: {predictions_csv}")
			sys.exit(5)
		if args.sleep_predictions_csv and not os.path.isfile(args.sleep_predictions_csv):
			print(f"[ERROR] Sleep predictions CSV does not exist: {args.sleep_predictions_csv}")
			sys.exit(5)

		# Compute VM features in-memory
		vm_rows = process_csv(input_csv)
		vm_df = pd.DataFrame(vm_rows, columns=['timestamp', 'avg_vm_10s', 'is_midnight_block'])

		# Load predictions (if provided) and merge on timestamp
		merged = None
		if predictions_csv:
			try:
				pred_df = pd.read_csv(predictions_csv)
			except Exception as e:
				print(f"[ERROR] Failed to read predictions CSV: {e}")
				sys.exit(6)
			if 'timestamp' not in pred_df.columns:
				print("[ERROR] Predictions CSV is missing 'timestamp' column")
				sys.exit(7)

			# Optional participant validation
			if args.participant_id:
				pid = str(args.participant_id)
				if 'participant_id' in pred_df.columns:
					pred_pids = pred_df['participant_id'].astype(str).unique()
					if len(pred_pids) > 1 or pred_pids[0] != pid:
						print(f"[ERROR] Participant ID mismatch in predictions CSV. Expected {pid}, found {list(pred_pids)}")
						sys.exit(7)
				elif pid not in os.path.abspath(predictions_csv):
					print(f"[WARN] Participant ID {pid} not found in predictions path; continuing")

			# Normalize timestamps and filter to target day
			pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'], errors='coerce')
			pred_df = pred_df.dropna(subset=['timestamp'])
			pred_df['timestamp'] = pred_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
			day_mask = pred_df['timestamp'].str.startswith(target_day)
			if not day_mask.any():
				print(f"[ERROR] Predictions file does not contain rows for {target_day}")
				sys.exit(7)
			pred_df = pred_df.loc[day_mask].copy()

			merged = pd.merge(pred_df, vm_df, on='timestamp', how='left')
		else:
			# Start with VM only if posture predictions not provided
			merged = vm_df.copy()

		# Sleep predictions (30s) -> expand to 10s and merge
		if args.sleep_predictions_csv:
			try:
				sleep_df = pd.read_csv(args.sleep_predictions_csv)
			except Exception as e:
				print(f"[ERROR] Failed to read sleep predictions CSV: {e}")
				sys.exit(8)
			if 'timestamp' not in sleep_df.columns:
				print("[ERROR] Sleep predictions CSV is missing 'timestamp' column")
				sys.exit(9)

			# Optional participant validation (path or column)
			if args.participant_id:
				pid = str(args.participant_id)
				if 'participant_id' in sleep_df.columns:
					sleep_pids = sleep_df['participant_id'].astype(str).unique()
					if len(sleep_pids) > 1 or sleep_pids[0] != pid:
						print(f"[ERROR] Participant ID mismatch in sleep predictions CSV. Expected {pid}, found {list(sleep_pids)}")
						sys.exit(9)
				elif pid not in os.path.abspath(args.sleep_predictions_csv):
					print(f"[WARN] Participant ID {pid} not found in sleep predictions path; continuing")

			# Ensure 'timestamp' is datetime, filter to target day, then expand 30s->10s
			sleep_df['timestamp'] = pd.to_datetime(sleep_df['timestamp'], errors='coerce')
			sleep_df = sleep_df.dropna(subset=['timestamp'])
			sleep_day_mask = sleep_df['timestamp'].dt.strftime('%Y-%m-%d') == target_day
			if not sleep_day_mask.any():
				print(f"[WARN] Sleep predictions file does not contain rows for {target_day}; skipping sleep merge")
				sleep10_df = None
			else:
				sleep_df = sleep_df.loc[sleep_day_mask].copy()
				# Create expanded rows at t, t+10s, t+20s duplicating predictions
				expanded_list = []
				for _, row in sleep_df.iterrows():
					base_ts = row['timestamp']
					for offset in (0, 10, 20):
						new_row = row.copy()
						new_row['timestamp'] = base_ts + pd.to_timedelta(offset, unit='s')
						expanded_list.append(new_row)
				sleep10_df = pd.DataFrame(expanded_list)

			if sleep10_df is not None:
				# Format timestamps to match VM/predictions string format
				sleep10_df['timestamp'] = sleep10_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
				merged = pd.merge(merged, sleep10_df, on='timestamp', how='left', suffixes=('', '_sleep'))
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		merged.to_csv(output_path, index=False)

		# Summary
		vm_matches = merged['avg_vm_10s'].notna().sum() if 'avg_vm_10s' in merged.columns else 0
		sleep_cols = [c for c in merged.columns if c.endswith('_sleep') or 'sleep' in c.lower()]
		print(f"\n=== SUMMARY (Augmented) ===")
		print(f"Wrote: {output_path}")
		print(f"VM matches: {vm_matches}/{len(merged)}")
		if sleep_cols:
			# Use NumPy to compute row-wise any without type checker axis complaints
			sleep_mask_np = merged[sleep_cols].notna().to_numpy()
			sleep_matches = int(np.any(sleep_mask_np, axis=1).sum())
			print(f"Sleep matches (at least one sleep col non-null): {sleep_matches}/{len(merged)}")
		return

	# Otherwise, write a per-day VM CSV into output-dir
	out_dir = args.output_dir
	os.makedirs(out_dir, exist_ok=True)
	print(f"\n--- Processing single file ---\n{input_csv}")
	result_path = process_csv_to_daily_vm(input_csv, out_dir)

	if not result_path:
		print(f"[ERROR] Failed to produce VM file for: {input_csv}")
		sys.exit(3)

	# Count windows for a brief summary
	try:
		with open(result_path, 'r') as f:
			total_windows = sum(1 for _ in f) - 1  # minus header
	except Exception:
		total_windows = 'unknown'

	print(f"\n=== SUMMARY ===")
	print(f"Wrote: {os.path.basename(result_path)}")
	print(f"Total VM windows computed: {total_windows}")


if __name__ == '__main__':
	main()

