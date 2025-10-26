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
from pathlib import Path
from typing import List
import shutil
import subprocess


# Resolve repository root as the grandparent of this file's directory (scripts/helper_scripts/ -> scripts/ -> repo root)
WORKSPACE_ROOT = str(Path(__file__).resolve().parents[2])
SRC_DIR = os.path.join(WORKSPACE_ROOT, "data", "project_data")
DEST_PATH = os.path.join(WORKSPACE_ROOT, "data", "project_data", "finalnew_data.csv.gz")


def natural_key(path: str):
    """Sort key that extracts numbers to get batch_1, batch_2, ..., batch_10 order."""
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def find_source_files(src_dir: str) -> List[str]:
    files = glob(os.path.join(src_dir, "*.csv.gz"))
    files.sort(key=natural_key)
    return files


def merge_csv_gz(files: List[str], dest_path: str, src_dir_for_msg: str) -> None:
    if not files:
        print(f"No .csv.gz files found to merge in: {src_dir_for_msg}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Fast path: if pigz is available, use multi-threaded compression in a single pass
    pigz_path = shutil.which("pigz")
    if pigz_path is not None:
        with open(dest_path, "wb") as out_gz:
            threads = max(1, (os.cpu_count() or 2))
            proc = subprocess.Popen([pigz_path, "-1", f"-p{threads}", "-c"], stdin=subprocess.PIPE, stdout=out_gz)
            assert proc.stdin is not None
            try:
                first_header: bytes | None = None
                for i, fpath in enumerate(files):
                    with gzip.open(fpath, mode="rb") as in_f:
                        header = in_f.readline()
                        if not header:
                            print(f"Warning: {fpath} is empty; skipping.")
                            continue
                        if i == 0:
                            first_header = header
                            proc.stdin.write(header)
                        else:
                            if first_header is not None and header != first_header:
                                print(f"Warning: Header mismatch in {fpath}. Using first header.")
                            # header skipped for subsequent files
                        # stream remainder
                        while True:
                            chunk = in_f.read(1024 * 1024 * 4)
                            if not chunk:
                                break
                            proc.stdin.write(chunk)
                    print(f"Merged {os.path.basename(fpath)}")
            finally:
                proc.stdin.close()
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"pigz failed with exit code {ret}")
        print(f"Done. Wrote: {dest_path}")
        return

    # Fallback: Python gzip compression (single threaded)
    first_header_fb: bytes | None = None
    with gzip.open(dest_path, mode="wb", compresslevel=1) as out_f:  # faster
        for i, fpath in enumerate(files):
            with gzip.open(fpath, mode="rb") as in_f:
                header = in_f.readline()
                if not header:
                    print(f"Warning: {fpath} is empty; skipping.")
                    continue
                if i == 0:
                    first_header_fb = header
                    out_f.write(header)
                else:
                    if first_header_fb is not None and header != first_header_fb:
                        print(f"Warning: Header mismatch in {fpath}. Using first header.")
                # Stream-copy remaining bytes in chunks
                while True:
                    chunk = in_f.read(1024 * 1024 * 4)
                    if not chunk:
                        break
                    out_f.write(chunk)
            print(f"Merged {os.path.basename(fpath)}")
    print(f"Done. Wrote: {dest_path}")


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

    merge_csv_gz(files, dest, src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
