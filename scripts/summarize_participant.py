#!/usr/bin/env python3
"""
Summarize participant sleep and posture predictions into hourly CSV rows.

Output columns: ID, Day, Hour, percent_sleep_nonwear, percent_sitting, percent_not_sitting

Rules:
- Sleep predictions: 30-second windows with START_TIME/STOP_TIME and STATE in {WEAR,SLEEP,NON-WEAR}.
- Posture predictions: 10-second rows with timestamp and prediction in {sitting, not-sitting}.

Alignment strategy:
- Expand sleep predictions to 10-second resolution (replicate each 30s row into three 10s slots) so every 10s posture row can be matched to a sleep state. This gives sleep precedence easily.
- For each day and hour (1..24), compute:
  - percent_sleep_nonwear = 100 * (number of 10s slots labeled SLEEP or NON-WEAR) / total slots in hour
  - percent_sitting = 100 * (number of posture slots labeled 'sitting' AND corresponding sleep state == WEAR) / total wear slots in hour
  - percent_not_sitting = 100 * (number of posture slots labeled 'not-sitting' AND corresponding sleep state == WEAR) / total wear slots in hour

Notes:
- If there are zero wear slots in an hour, percent_sitting and percent_not_sitting will be NaN (or 0 depending on preference). We'll set them to 0.0 for robustness.
- The script expects files in:
  data/sleep_predictions/<ID>/predictions/<YYYY-MM-DD>_sleep_predictions.csv
  data/predictions/<ID>/<MODEL>/<YYYY-MM-DD>.csv

Usage:
  python3 scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS

"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def read_sleep_day(sleep_file: Path):
    df = pd.read_csv(sleep_file)
    # Parse times
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    df['STOP_TIME'] = pd.to_datetime(df['STOP_TIME'])
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


def summarize_participant(participant_id: str, model: str, out_csv: Path):
    repo = Path.cwd()
    sleep_dir = repo / 'data' / 'sleep_predictions' / participant_id / 'predictions'
    posture_dir = repo / 'data' / 'predictions' / participant_id / model

    if not sleep_dir.exists():
        logger.error('Sleep predictions directory not found: %s', sleep_dir)
        return
    if not posture_dir.exists():
        logger.error('Posture predictions directory not found: %s', posture_dir)
        return

    # Gather all days present in either directory
    sleep_days = sorted([p.name for p in sleep_dir.glob('*_sleep_predictions.csv')])
    # normalize names to YYYY-MM-DD
    sleep_days = [name.split('_sleep_predictions.csv')[0] for name in sleep_days]
    posture_days = sorted([p.stem for p in posture_dir.glob('*.csv')])

    days = sorted(set(sleep_days) | set(posture_days))
    logger.info('Found days: %s', days)

    out_rows = []

    for day in days:
        logger.info('Processing day %s', day)
        sleep_file = sleep_dir / f"{day}_sleep_predictions.csv"
        posture_file = posture_dir / f"{day}.csv"

        if not sleep_file.exists():
            logger.warning('Sleep file missing for %s; skipping day', day)
            continue
        if not posture_file.exists():
            logger.warning('Posture file missing for %s; skipping day', day)
            continue

        sleep_df = read_sleep_day(sleep_file)
        posture_df = read_posture_day(posture_file)

        # Expand sleep to 10s resolution
        sleep_10 = expand_sleep_to_10s(sleep_df)
        # set index to timestamp for fast join
        sleep_10.set_index('timestamp', inplace=True)

        # posture timestamps are 10s resolution already; ensure rounded to 10s
        posture_df['timestamp'] = posture_df['timestamp'].dt.round('10s')
        posture_df.set_index('timestamp', inplace=True)

        # Join posture with sleep states; sleep takes precedence
        merged = posture_df.join(sleep_10, how='left')

        # If any posture rows miss STATE (unlikely), set to WEAR for posture-based measures
        merged['STATE'] = merged['STATE'].fillna('WEAR')

        # Create hour and day columns from index
        merged = merged.reset_index()
        merged['day'] = merged['timestamp'].dt.date.astype(str)
        merged['hour'] = merged['timestamp'].dt.hour + 1  # make 1..24

        # For each hour compute metrics
        grouped = merged.groupby(['day', 'hour'])
        for (d, h), g in grouped:
            total_slots = len(g)
            if total_slots == 0:
                continue
            sleep_or_nwear = g['STATE'].isin(['SLEEP', 'NON-WEAR']).sum()
            percent_sleep_nonwear = 100.0 * sleep_or_nwear / total_slots

            # wear-only slots
            wear_mask = g['STATE'] == 'WEAR'
            wear_slots = wear_mask.sum()

            # percent_wear is percent of the hour that is WEAR
            percent_wear = 100.0 * wear_slots / total_slots

            # compute sitting/not-sitting as percent of the whole hour so they sum to percent_wear
            if wear_slots == 0:
                percent_sitting = 0.0
                percent_not_sitting = 0.0
            else:
                sitting_slots = ((g['prediction'] == 'sitting') & wear_mask).sum()
                not_sitting_slots = ((g['prediction'] != 'sitting') & wear_mask).sum()
                percent_sitting = 100.0 * sitting_slots / total_slots
                percent_not_sitting = 100.0 * not_sitting_slots / total_slots

            out_rows.append({
                'ID': participant_id,
                'Day': d,
                'Hour': int(h),
                'percent_sleep_nonwear': percent_sleep_nonwear,
                'percent_wear': percent_wear,
                'percent_sitting': percent_sitting,
                'percent_not_sitting': percent_not_sitting
            })

    out_df = pd.DataFrame(out_rows)
    # Round percentage columns to 2 decimal places
    pct_cols = [c for c in ['percent_sleep_nonwear', 'percent_wear', 'percent_sitting', 'percent_not_sitting'] if c in out_df.columns]
    if not out_df.empty and pct_cols:
        out_df[pct_cols] = out_df[pct_cols].round(2)

    out_df.to_csv(out_csv, index=False)
    logger.info('Wrote summary to %s', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--participant-id', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', default='data/summaries/sleep_posture_hourly.csv')
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summarize_participant(args.participant_id, args.model, Path(args.out))

## Example command
# python scripts/summarize_participant.py --participant-id 62161 --model CHAP_ALL_ADULTS --out data/summaries/62161_sleep_posture_hourly.csv