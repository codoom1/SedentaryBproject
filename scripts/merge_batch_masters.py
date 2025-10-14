#!/usr/bin/env python3
"""
Merge per-task master CSV(.gz) files produced by the array batch runner into
one consolidated master CSV(.gz), preserving a single header.

Usage:
  python scripts/merge_batch_masters.py \
    --inputs data/summaries/master_batch_*.csv.gz \
    --output data/summaries/batch_sleep_posture_hourly.csv.gz

Notes:
  - Inputs can be .csv or .csv.gz in any mix.
  - Header is taken from the first non-empty file; subsequent files will have
    their header skipped.
  - The output extension determines compression (.gz -> gzip).
"""
from __future__ import annotations

import argparse
import glob
import gzip
from pathlib import Path
from typing import TextIO


def open_any(path: Path, mode: str) -> TextIO:
    # Always use text modes ('rt', 'wt', 'at') for both gz and plain files
    if path.suffix == '.gz':
        return gzip.open(path, mode)  # type: ignore[return-value]
    f = open(path, mode)
    return f  # type: ignore[return-value]


def main():
    ap = argparse.ArgumentParser(description='Merge per-task master CSVs into one')
    ap.add_argument('--inputs', required=True, help='Glob for input files, e.g., data/summaries/master_batch_*.csv.gz')
    ap.add_argument('--output', required=True, help='Output path, e.g., data/summaries/batch_sleep_posture_hourly.csv.gz')
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.inputs))
    if not inputs:
        raise SystemExit(f'No inputs matched: {args.inputs}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    lines_written = 0

    # Choose text write/append depending on file existence
    mode = 'wt'  # always text write, regardless of extension

    with open_any(out_path, mode) as out_f:
        for i, p in enumerate(inputs, 1):
            in_path = Path(p)
            if not in_path.exists() or in_path.stat().st_size == 0:
                continue
            with open_any(in_path, 'rt') as in_f:
                for j, line in enumerate(in_f):
                    if j == 0:
                        if wrote_header:
                            continue  # skip subsequent headers
                        else:
                            wrote_header = True
                    out_f.write(line)
                    lines_written += 1

    print(f'Merged {len(inputs)} files into {out_path} (lines: {lines_written})')


if __name__ == '__main__':
    main()
