#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def list_hour_files(root: Path, pid: str, day: str):
    d = root / pid
    return sorted([p for p in d.glob(f"GT3XPLUS-AccelerationCalibrated-*.{day}-*-*-*-*-P0000.sensor.csv")])


def main():
    ap = argparse.ArgumentParser(description="Inspect raw window for anomalies")
    ap.add_argument("--root", required=True, help="Root of raw dir (e.g., data/raw/2013-14)")
    ap.add_argument("--participant", required=True)
    ap.add_argument("--start", required=True, help="Start ISO timestamp, e.g., 2000-01-11T00:00:00")
    ap.add_argument("--end", required=True, help="End ISO timestamp, e.g., 2000-01-11T03:00:30")
    args = ap.parse_args()

    root = Path(args.root)
    pid = str(args.participant)
    t0 = pd.to_datetime(args.start)
    t1 = pd.to_datetime(args.end)

    # Collect candidate files by hour range
    day = t0.strftime("%Y-%m-%d")
    pdir = root / pid
    files = sorted([p for p in pdir.glob(f"*{day}-*.sensor.csv")])
    # Filter roughly by hour to reduce load
    hours = {t0.strftime('%H'), (t0 + pd.Timedelta(hours=1)).strftime('%H'), (t0 + pd.Timedelta(hours=2)).strftime('%H'), (t0 + pd.Timedelta(hours=3)).strftime('%H')}
    files = [p for p in files if any(f"-{h}-" in p.name for h in hours)]
    print(f"Inspecting {len(files)} files: {[p.name for p in files]}")

    total = 0
    nan_ts = 0
    dup_ts = 0
    non_mono = 0
    nan_xyz = 0
    inf_xyz = 0
    extreme = 0
    minv = {c: None for c in ['X', 'Y', 'Z']}
    maxv = {c: None for c in ['X', 'Y', 'Z']}

    for p in files:
        df = pd.read_csv(p)
        if 'HEADER_TIMESTAMP' in df.columns:
            df = df.rename(columns={'HEADER_TIMESTAMP': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[(df['timestamp'] >= t0) & (df['timestamp'] < t1)]
        if df.empty:
            continue
        total += len(df)
        nan_ts += df['timestamp'].isna().sum()
        # Duplicates and monotonic
        dup_ts += int(df['timestamp'].duplicated().sum())
        non_mono += int((~df['timestamp'].is_monotonic_increasing))
        for c in ['X', 'Y', 'Z']:
            if c in df.columns:
                nan_xyz += df[c].isna().sum()
                inf_xyz += int(np.isfinite(df[c]).sum() != len(df))
                vmin = float(df[c].min())
                vmax = float(df[c].max())
                minv[c] = vmin if minv[c] is None else min(minv[c], vmin)
                maxv[c] = vmax if maxv[c] is None else max(maxv[c], vmax)
                extreme += int(((df[c].abs() > 100).any()))

    print("Summary:")
    print({
        'rows': total,
        'nan_timestamps': nan_ts,
        'duplicate_timestamps': dup_ts,
        'non_monotonic_flag': non_mono,
        'nan_xyz_total': nan_xyz,
        'inf_xyz_files_flag_count': inf_xyz,
        'extreme_abs_gt_100_files_flag_count': extreme,
        'min_vals': minv,
        'max_vals': maxv,
    })


if __name__ == '__main__':
    main()
