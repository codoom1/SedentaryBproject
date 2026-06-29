#!/usr/bin/env python3
"""
Summarize participant sleep and posture predictions into epoch-level CSV rows.

Output columns include:
ID, Dataset, Day, DayType, Hour, epoch, epoch_start, epoch_end,
epoch_duration_seconds, epoch_duration_hours, percent_sleep_nonwear,
percent_wear, percent_sitting, percent_not_sitting, summary_epoch, model.

Rules:
- Sleep predictions: 30-second windows with START_TIME/STOP_TIME and STATE in {WEAR,SLEEP,NON-WEAR}.
- Posture predictions: 10-second rows with timestamp and prediction in {sitting, not-sitting}.

Alignment strategy:
- Expand sleep predictions to 10-second resolution (replicate each 30s row into three 10s slots) so every 10s posture row can be matched to a sleep state. This gives sleep precedence easily.
- For each requested summary epoch, compute:
  - percent_sleep_nonwear = 100 * (number of 10s slots labeled SLEEP or NON-WEAR) / total slots in epoch
  - percent_wear = 100 * (number of 10s slots labeled WEAR) / total slots in epoch
  - percent_sitting = 100 * (number of CHAP posture slots labeled 'sitting') / total slots in epoch
  - percent_not_sitting = 100 * (number of CHAP posture slots labeled 'not-sitting') / total slots in epoch

Notes:
- Sleep removal is not performed here. The manuscript/statistical-analysis repo applies the sleep/non-wear cutoff.
- The script expects files in:
  data/sleep_predictions/<ID>/predictions/<YYYY-MM-DD>_sleep_predictions.csv
  data/predictions/<ID>/<MODEL>/<YYYY-MM-DD>.csv

Usage:
  python3 scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS
  python3 scripts/summarize_participant.py --participant-id 62193 --model CHAP --skip-sleep
  python3 scripts/summarize_participant.py --participant-id 62193 --model CHAP_ALL_ADULTS --summary-epoch 30s

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, Sequence
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_summary_epoch(summary_epoch: str) -> int:
    key = summary_epoch.strip().lower()
    aliases = {'hourly': '1h', 'hour': '1h'}
    key = aliases.get(key, key)

    match = re.fullmatch(r'(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours)', key)
    if not match:
        raise ValueError(
            "Unsupported --summary-epoch '{0}'. Use durations like 10s, 30s, "
            "1min, 5m, 30m, or 1h.".format(summary_epoch)
        )

    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        raise ValueError("--summary-epoch must be greater than zero")

    if unit in {'s', 'sec', 'secs', 'second', 'seconds'}:
        seconds = value
    elif unit in {'m', 'min', 'mins', 'minute', 'minutes'}:
        seconds = value * 60
    elif unit in {'h', 'hr', 'hrs', 'hour', 'hours'}:
        seconds = value * 3600
    else:
        raise ValueError(f"Unsupported --summary-epoch unit: {unit}")

    if seconds % 10 != 0:
        raise ValueError("--summary-epoch must be a multiple of the 10-second CHAP base epoch")
    if 86400 % seconds != 0:
        raise ValueError("--summary-epoch must divide evenly into 24 hours")

    return seconds


def format_epoch_label(summary_epoch: str) -> str:
    seconds = parse_summary_epoch(summary_epoch)
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def parse_extra_epoch_columns(summary_epoch_seconds: int, epoch_columns: Optional[Sequence[str]]):
    parsed = []
    seen = set()
    for epoch_column in epoch_columns or []:
        seconds = parse_summary_epoch(epoch_column)
        label = format_epoch_label(epoch_column)
        column = f"epoch_{label}"
        if seconds < summary_epoch_seconds:
            raise ValueError(
                f"--epoch-columns value '{epoch_column}' is finer than --summary-epoch. "
                "Run the summarizer at the finest epoch you need, then add coarser epoch columns."
            )
        if seconds % summary_epoch_seconds != 0:
            raise ValueError(
                f"--epoch-columns value '{epoch_column}' must be an exact multiple of --summary-epoch"
            )
        if column in seen:
            continue
        parsed.append((column, seconds))
        seen.add(column)
    return parsed


def normalize_dataset(dataset: Optional[str]) -> str:
    if dataset is None:
        return ''
    dataset = str(dataset)
    return {'2011-12': '2011-2012', '2013-14': '2013-2014'}.get(dataset, dataset)


def day_type(day_value: str) -> str:
    weekday = pd.Timestamp(day_value).dayofweek
    return 'Weekend' if weekday >= 5 else 'Weekday'


def read_sleep_day(sleep_file: Path):
    df = pd.read_csv(sleep_file)
    # Parse times
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    df['STOP_TIME'] = pd.to_datetime(df['STOP_TIME'])
    if 'STATE' in df.columns:
        state = df['STATE'].astype(str).str.strip().str.upper()
        state = state.str.replace('_', '-', regex=False).str.replace(' ', '-', regex=False)
        df['STATE'] = state.replace({
            'NONWEAR': 'NON-WEAR',
            'NON-WEAR': 'NON-WEAR',
            'NWEAR': 'NON-WEAR',
            'SLEEPING': 'SLEEP',
            'WAKE': 'WEAR',
            'WAKING': 'WEAR',
        })
    elif 'PREDICTED' in df.columns:
        prediction_map = {0: 'WEAR', 1: 'SLEEP', 2: 'NON-WEAR'}
        predicted = pd.to_numeric(df['PREDICTED'], errors='coerce')
        df['STATE'] = predicted.map(prediction_map)
    else:
        raise ValueError(f"No STATE or PREDICTED column in {sleep_file}")

    before = len(df)
    df = df.dropna(subset=['START_TIME', 'STOP_TIME', 'STATE'])
    valid_states = {'WEAR', 'SLEEP', 'NON-WEAR'}
    invalid_state_mask = ~df['STATE'].isin(valid_states)
    if invalid_state_mask.any():
        invalid_values = sorted(df.loc[invalid_state_mask, 'STATE'].dropna().unique().tolist())
        logger.warning(
            'Sleep file %s has %d rows with unsupported STATE values %s; dropping them',
            sleep_file,
            int(invalid_state_mask.sum()),
            invalid_values,
        )
        df = df.loc[~invalid_state_mask].copy()
    df = df.sort_values('START_TIME')
    if df['START_TIME'].duplicated().any():
        duplicate_count = int(df['START_TIME'].duplicated().sum())
        logger.warning(
            'Sleep file %s has %d duplicate START_TIME rows; keeping first after sorting',
            sleep_file,
            duplicate_count,
        )
        df = df.drop_duplicates(subset=['START_TIME'], keep='first')
    dropped = before - len(df)
    if dropped:
        logger.warning('Sleep file %s dropped %d invalid sleep rows', sleep_file, int(dropped))
    return df


def read_posture_day(posture_file: Path):
    df = pd.read_csv(posture_file)
    # parse timestamp column name variations
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    else:
        raise ValueError(f"No timestamp column in {posture_file}")
    if 'prediction' not in df.columns:
        raise ValueError(f"No prediction column in {posture_file}")

    prediction = df['prediction'].astype(str).str.strip().str.lower()
    df['prediction'] = prediction.replace({
        'sit': 'sitting',
        'seated': 'sitting',
        'not sitting': 'not-sitting',
        'not_sitting': 'not-sitting',
        'notsitting': 'not-sitting',
        'non-sitting': 'not-sitting',
        'non_sitting': 'not-sitting',
        'standing': 'not-sitting',
    })
    valid_predictions = {'sitting', 'not-sitting'}
    invalid_prediction_mask = ~df['prediction'].isin(valid_predictions)
    if invalid_prediction_mask.any():
        invalid_values = sorted(
            df.loc[invalid_prediction_mask, 'prediction'].dropna().unique().tolist()
        )
        logger.warning(
            'Posture file %s has %d rows with unsupported prediction values %s; treating them as not-sitting',
            posture_file,
            int(invalid_prediction_mask.sum()),
            invalid_values,
        )
        df.loc[invalid_prediction_mask, 'prediction'] = 'not-sitting'
    return df


def expand_sleep_to_10s(sleep_df):
    # For each 30s window create three 10s slots starting at START_TIME + 0, +10s, +20s
    rows = []
    for _, r in sleep_df.iterrows():
        start = r['START_TIME']
        state = r.get('STATE', None)
        # create three 10s timestamps representing the start of each 10s slot
        for off in [0, 10, 20]:
            ts = start + pd.Timedelta(seconds=off)
            rows.append({'timestamp': ts, 'STATE': state})
    sdf = pd.DataFrame(rows)
    return sdf


def summarize_participant(
    participant_id: str,
    model: str,
    out_csv: Path,
    skip_sleep: bool = False,
    summary_epoch: str = '1h',
    dataset: Optional[str] = None,
    epoch_columns: Optional[Sequence[str]] = None,
):
    summary_epoch_seconds = parse_summary_epoch(summary_epoch)
    extra_epoch_columns = parse_extra_epoch_columns(summary_epoch_seconds, epoch_columns)
    epoch_duration_hours = summary_epoch_seconds / 3600.0
    epoch_divides_hour = 3600 % summary_epoch_seconds == 0
    dataset_value = normalize_dataset(dataset)

    repo = Path.cwd()
    sleep_dir = repo / 'data' / 'sleep_predictions' / participant_id / 'predictions'
    posture_dir = repo / 'data' / 'predictions' / participant_id / model

    if not skip_sleep and not sleep_dir.exists():
        logger.error('Sleep predictions directory not found: %s', sleep_dir)
        return False
    if not posture_dir.exists():
        logger.error('Posture predictions directory not found: %s', posture_dir)
        return False

    # Gather all days present in the required inputs for the selected mode.
    sleep_days = []
    if sleep_dir.exists():
        sleep_days = sorted([p.name for p in sleep_dir.glob('*_sleep_predictions.csv')])
        sleep_days = [name.split('_sleep_predictions.csv')[0] for name in sleep_days]
    posture_days = sorted([p.stem for p in posture_dir.glob('*.csv')])

    if skip_sleep:
        logger.info('Summarizing participant %s in posture-only mode (--skip-sleep)', participant_id)
        days = posture_days
    else:
        days = sorted(set(sleep_days) | set(posture_days))
    logger.info('Found days: %s', days)
    if not days:
        logger.error('No day files found to summarize for participant %s', participant_id)
        return False

    out_rows = []

    for day in days:
        logger.info('Processing day %s', day)
        sleep_file = sleep_dir / f"{day}_sleep_predictions.csv"
        posture_file = posture_dir / f"{day}.csv"

        if not posture_file.exists():
            logger.warning('Posture file missing for %s; skipping day', day)
            continue

        posture_df = read_posture_day(posture_file)

        # posture timestamps are 10s resolution already; ensure rounded to 10s
        posture_df['timestamp'] = posture_df['timestamp'].dt.round('10s')
        posture_df = posture_df.dropna(subset=['timestamp', 'prediction'])
        posture_df = posture_df.sort_values('timestamp')
        if posture_df['timestamp'].duplicated().any():
            duplicate_count = int(posture_df['timestamp'].duplicated().sum())
            logger.warning(
                'Posture file %s has %d duplicate timestamp rows; keeping first after sorting',
                posture_file,
                duplicate_count,
            )
            posture_df = posture_df.drop_duplicates(subset=['timestamp'], keep='first')
        posture_df.set_index('timestamp', inplace=True)

        if skip_sleep:
            merged = posture_df.copy()
            merged['STATE'] = 'WEAR'
        else:
            if not sleep_file.exists():
                logger.warning('Sleep file missing for %s; skipping day', day)
                continue
            sleep_df = read_sleep_day(sleep_file)
            # Expand sleep to 10s resolution
            sleep_10 = expand_sleep_to_10s(sleep_df)
            # set index to timestamp for fast join
            sleep_10.set_index('timestamp', inplace=True)

            # Join posture with sleep states; sleep takes precedence
            merged = posture_df.join(sleep_10, how='left')

            # If any posture rows miss STATE (unlikely), set to WEAR for posture-based measures
            merged['STATE'] = merged['STATE'].fillna('WEAR')

        # Create hour and day columns from index
        merged = merged.reset_index()
        merged['day'] = merged['timestamp'].dt.date.astype(str)
        midnight = merged['timestamp'].dt.normalize()
        seconds_since_midnight = (
            (merged['timestamp'] - midnight).dt.total_seconds().astype(int)
        )
        epoch_start_seconds = (seconds_since_midnight // summary_epoch_seconds) * summary_epoch_seconds
        merged['epoch_start'] = midnight + pd.to_timedelta(epoch_start_seconds, unit='s')
        merged['epoch_end'] = merged['epoch_start'] + pd.to_timedelta(summary_epoch_seconds, unit='s')
        merged['hour'] = merged['epoch_start'].dt.hour + 1 if epoch_divides_hour else pd.NA
        if epoch_divides_hour:
            merged['epoch'] = ((epoch_start_seconds % 3600) // summary_epoch_seconds).astype(int) + 1
        else:
            merged['epoch'] = (epoch_start_seconds // summary_epoch_seconds).astype(int) + 1

        # For each requested summary epoch compute metrics.
        grouped = merged.groupby(['day', 'epoch', 'epoch_start', 'epoch_end'])
        for (d, epoch, epoch_start, epoch_end), g in grouped:
            total_slots = len(g)
            if total_slots == 0:
                continue
            h = int(pd.Timestamp(epoch_start).hour + 1) if epoch_divides_hour else None
            sleep_or_nwear = g['STATE'].isin(['SLEEP', 'NON-WEAR']).sum()
            wear_mask = g['STATE'] == 'WEAR'
            wear_slots = wear_mask.sum()
            sitting_slots = (g['prediction'] == 'sitting').sum()
            not_sitting_slots = (g['prediction'] != 'sitting').sum()

            percent_sleep_nonwear = 100.0 * sleep_or_nwear / total_slots
            percent_wear = 100.0 * wear_slots / total_slots
            percent_sitting = 100.0 * sitting_slots / total_slots
            percent_not_sitting = 100.0 * not_sitting_slots / total_slots

            row = {
                'ID': participant_id,
                'Dataset': dataset_value,
                'Day': d,
                'DayType': day_type(d),
                'Hour': h,
                'epoch': int(epoch),
                'epoch_start': pd.Timestamp(epoch_start).isoformat(),
                'epoch_end': pd.Timestamp(epoch_end).isoformat(),
                'epoch_duration_seconds': int(summary_epoch_seconds),
                'epoch_duration_hours': epoch_duration_hours,
                'percent_sleep_nonwear': percent_sleep_nonwear,
                'percent_wear': percent_wear,
                'percent_sitting': percent_sitting,
                'percent_not_sitting': percent_not_sitting,
                'n_base_epochs': int(total_slots),
                'n_sleep_nonwear_epochs': int(sleep_or_nwear),
                'n_wear_epochs': int(wear_slots),
                'n_sitting_epochs': int(sitting_slots),
                'n_not_sitting_epochs': int(not_sitting_slots),
                'summary_epoch': summary_epoch,
                'model': model
            }

            row_start_seconds = int(
                (pd.Timestamp(epoch_start) - pd.Timestamp(epoch_start).normalize()).total_seconds()
            )
            for column, seconds in extra_epoch_columns:
                row[column] = (row_start_seconds // seconds) + 1

            out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        logger.error('No summary rows were produced for participant %s', participant_id)
        return False
    # Round percentage columns to 2 decimal places
    pct_cols = [c for c in ['percent_sleep_nonwear', 'percent_wear', 'percent_sitting', 'percent_not_sitting'] if c in out_df.columns]
    if not out_df.empty and pct_cols:
        out_df[pct_cols] = out_df[pct_cols].round(2)
    if 'epoch_duration_hours' in out_df.columns:
        out_df['epoch_duration_hours'] = out_df['epoch_duration_hours'].round(8)
    out_df = out_df.sort_values(['ID', 'Day', 'epoch_start', 'epoch']).reset_index(drop=True)

    out_df.to_csv(out_csv, index=False)
    logger.info('Wrote summary to %s', out_csv)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--participant-id', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', default='data/summaries/sleep_posture_hourly.csv')
    parser.add_argument('--skip-sleep', action='store_true', help='Summarize posture-only predictions, treating all posture rows as WEAR')
    parser.add_argument('--summary-epoch', default='1h', help='Summary epoch duration, e.g. 10s, 30s, 1min, 5m, 30m, or 1h (default: 1h)')
    parser.add_argument('--epoch-columns', nargs='*', default=None, help='Optional coarser epoch label columns to add, e.g. --epoch-columns 20m 30m. Use Hour for hourly grouping.')
    parser.add_argument('--dataset', default=None, help='Optional NHANES cycle label, e.g. 2011-12 or 2013-14')
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ok = summarize_participant(
        args.participant_id,
        args.model,
        Path(args.out),
        skip_sleep=args.skip_sleep,
        summary_epoch=args.summary_epoch,
        dataset=args.dataset,
        epoch_columns=args.epoch_columns,
    )
    raise SystemExit(0 if ok else 1)

## Example command
# python scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS --out data/summaries/62161_sleep_posture_hourly.csv
