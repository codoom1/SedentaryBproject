#!/usr/bin/env python3
"""
Isolated SWaN first-pass worker.

Reads a pre-sanitized CSV with columns expected by SWaN:
 - HEADER_TIME_STAMP (datetime)
 - X_ACCELERATION_METERS_PER_SECOND_SQUARED
 - Y_ACCELERATION_METERS_PER_SECOND_SQUARED
 - Z_ACCELERATION_METERS_PER_SECOND_SQUARED

Runs SWaN_accel.swan_first_pass.main and writes window-level raw output CSV to --output.

This isolation allows the caller to enforce a subprocess timeout.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np

# Cap BLAS/OMP/NumExpr threads inside worker to reduce stall/oversubscription risk
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from SWaN_accel import swan_first_pass
except Exception as e:
    print(f"ERROR: Failed to import SWaN_accel.swan_first_pass: {e}", file=sys.stderr)
    sys.exit(1)


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure timestamp column present and is datetime
    ts_col = None
    for c in ("HEADER_TIME_STAMP", "HEADER_TIMESTAMP", "timestamp", "START_TIME"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No timestamp column found in worker input")
    if ts_col != "HEADER_TIME_STAMP":
        df = df.rename(columns={ts_col: "HEADER_TIME_STAMP"})
    df["HEADER_TIME_STAMP"] = pd.to_datetime(df["HEADER_TIME_STAMP"], errors="coerce")

    # Coerce accelerations numeric and drop bad rows
    for c in [
        "X_ACCELERATION_METERS_PER_SECOND_SQUARED",
        "Y_ACCELERATION_METERS_PER_SECOND_SQUARED",
        "Z_ACCELERATION_METERS_PER_SECOND_SQUARED",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need_cols = ["HEADER_TIME_STAMP"] + [
        c for c in [
            "X_ACCELERATION_METERS_PER_SECOND_SQUARED",
            "Y_ACCELERATION_METERS_PER_SECOND_SQUARED",
            "Z_ACCELERATION_METERS_PER_SECOND_SQUARED",
        ]
        if c in df.columns
    ]
    df = df.dropna(subset=need_cols)
    for c in need_cols:
        if c != "HEADER_TIME_STAMP":
            df = df[np.isfinite(df[c])]
    # Drop duplicate timestamps, keep first, and sort
    if df["HEADER_TIME_STAMP"].duplicated().any():
        df = df.drop_duplicates(subset=["HEADER_TIME_STAMP"], keep="first")
    df = df.sort_values("HEADER_TIME_STAMP").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after sanitization in worker")
    return df


def main():
    p = argparse.ArgumentParser(description="SWaN first-pass worker")
    p.add_argument("--input", required=True, help="Path to input CSV for SWaN")
    p.add_argument("--output", required=True, help="Path to output CSV (raw window-level)")
    p.add_argument("--sampling-rate", type=int, default=80, help="Sampling rate in Hz")
    args = p.parse_args()

    try:
        # Focused read for performance and stability
        df = pd.read_csv(
            args.input,
            low_memory=False,
        )
    except Exception as e:
        print(f"ERROR: Worker failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        df = _sanitize(df)
    except Exception as e:
        print(f"ERROR: Worker input sanitization failed: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        swan_first_pass.main(df=df, file_path=args.output, sampling_rate=args.sampling_rate)
    except Exception as e:
        print(f"ERROR: SWaN first-pass execution failed: {e}", file=sys.stderr)
        sys.exit(4)

    # Validate output exists and is non-empty
    try:
        # Read a few rows to confirm write success
        out_df = pd.read_csv(args.output, nrows=1)
    except Exception as e:
        print(f"ERROR: Worker produced no readable output CSV: {e}", file=sys.stderr)
        sys.exit(5)


if __name__ == "__main__":
    main()
