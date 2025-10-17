#!/usr/bin/env python3
"""
Merge all .csv.gz files in data/project_data/SBnovel_outputs/ into a single
gzipped CSV at data/project_data/SBnovel_outputs_merged.csv.gz.

This streams line-by-line to keep memory usage low and writes the header once.
"""

from __future__ import annotations

import gzip
import os
import re
import sys
from glob import glob
from typing import List


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(__file__))  # repo root
SRC_DIR = os.path.join(WORKSPACE_ROOT, "data", "project_data", "SBnovel_outputs")
DEST_PATH = os.path.join(WORKSPACE_ROOT, "data", "project_data", "SBnovel_outputs_merged.csv.gz")


def natural_key(path: str):
    """Sort key that extracts numbers to get batch_1, batch_2, ..., batch_10 order."""
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def find_source_files(src_dir: str) -> List[str]:
    files = glob(os.path.join(src_dir, "*.csv.gz"))
    files.sort(key=natural_key)
    return files


def merge_csv_gz(files: List[str], dest_path: str) -> None:
    if not files:
        print(f"No .csv.gz files found to merge in: {SRC_DIR}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    total_rows = 0
    first_header: str | None = None

    with gzip.open(dest_path, mode="wt", encoding="utf-8", newline="") as out_f:
        for i, fpath in enumerate(files):
            with gzip.open(fpath, mode="rt", encoding="utf-8", newline="") as in_f:
                try:
                    header = next(in_f)
                except StopIteration:
                    print(f"Warning: {fpath} is empty; skipping.")
                    continue

                if i == 0:
                    first_header = header
                    out_f.write(header)
                else:
                    if first_header is not None and header != first_header:
                        # Header mismatch; proceed but log.
                        print(f"Warning: Header mismatch in {fpath}. Using first header.")
                    # Skip header line for subsequent files

                # Write remaining lines
                for line in in_f:
                    out_f.write(line)
                    total_rows += 1

            print(f"Merged {os.path.basename(fpath)}")

    print(f"Done. Wrote: {dest_path}")
    print(f"Data rows (excluding header): {total_rows}")


def main(argv: List[str]) -> int:
    src = SRC_DIR
    dest = DEST_PATH

    # Optional CLI overrides
    if len(argv) > 1:
        src = argv[1]
    if len(argv) > 2:
        dest = argv[2]

    files = find_source_files(src)
    print(f"Found {len(files)} files to merge in {src}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    merge_csv_gz(files, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
