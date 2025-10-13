#!/usr/bin/env python3
"""
Create fixed-size batch files from constants/participants.csv.

Each output file contains lines of the form:
    cycle,participant_id

Defaults:
- Input: constants/participants.csv
- Output directory: batches/
- Batch size: 100
- Prefix: batch_
- Start index: 1
- Dedupe identical (cycle,participant_id) pairs by default

Examples:
  python scripts/make_batches_from_constants.py --batch-size 100
  python scripts/make_batches_from_constants.py --batch-size 250 --shuffle --seed 42
  python scripts/make_batches_from_constants.py --participants constants/participants.csv --out-dir batches --prefix nhanes_ --start-index 10
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple, Set


def read_participants(csv_path: Path, dedupe: bool = True) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            # skip commented lines
            if isinstance(row[0], str) and row[0].strip().startswith('#'):
                continue
            # skip header if present
            if i == 0 and len(row) >= 2 and row[0].strip().lower() == 'cycle' and row[1].strip().lower() == 'participant_id':
                continue
            if len(row) < 2:
                continue
            cycle = row[0].strip()
            pid = row[1].strip()
            if not cycle or not pid:
                continue
            key = (cycle, pid)
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)
            pairs.append(key)
    return pairs


def write_batches(pairs: List[Tuple[str, str]], out_dir: Path, batch_size: int, prefix: str, start_index: int = 1) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    # chunk
    idx = start_index
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        out_path = out_dir / f"{prefix}{idx}.txt"
        with out_path.open('w', newline='') as f:
            for (cycle, pid) in chunk:
                f.write(f"{cycle},{pid}\n")
        files.append(out_path)
        idx += 1
    return files

def write_batches_by_sizes(pairs: List[Tuple[str, str]], out_dir: Path, sizes: List[int], prefix: str, start_index: int = 1) -> List[Path]:
    """Write batches with explicit sizes per file. Skips zero-sized batches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    offset = 0
    idx = start_index
    for sz in sizes:
        if sz <= 0:
            continue
        chunk = pairs[offset:offset + sz]
        if not chunk:
            break
        out_path = out_dir / f"{prefix}{idx}.txt"
        with out_path.open('w', newline='') as f:
            for (cycle, pid) in chunk:
                f.write(f"{cycle},{pid}\n")
        files.append(out_path)
        idx += 1
        offset += sz
    return files


def main():
    ap = argparse.ArgumentParser(description='Create fixed-size batch files from constants/participants.csv')
    ap.add_argument('--participants', default='constants/participants.csv', help='Path to participants CSV')
    ap.add_argument('--out-dir', default='batches', help='Output directory for batch files')
    ap.add_argument('--batch-size', type=int, default=100, help='Number of rows per batch file')
    ap.add_argument('--num-batches', type=int, default=None, help='Desired number of batch files; overrides --batch-size if provided')
    ap.add_argument('--prefix', default='batch_', help='Output filename prefix (prefix<N>.txt)')
    ap.add_argument('--start-index', type=int, default=1, help='Starting index for batch numbering')
    ap.add_argument('--shuffle', action='store_true', help='Shuffle participants before batching')
    ap.add_argument('--seed', type=int, default=None, help='Random seed when using --shuffle')
    ap.add_argument('--no-dedupe', action='store_true', help='Do not deduplicate identical (cycle,participant_id) pairs')
    ap.add_argument('--dry-run', action='store_true', help='Print summary without writing files')
    args = ap.parse_args()

    csv_path = Path(args.participants)
    out_dir = Path(args.out_dir)

    pairs = read_participants(csv_path, dedupe=(not args.no_dedupe))
    total = len(pairs)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    # Determine batching strategy
    if args.num_batches and args.num_batches > 0:
        nb = args.num_batches
        q, r = divmod(total, nb)
        # Distribute remainder across the first r batches (sizes differ by at most 1)
        sizes = [(q + 1) if i < r else q for i in range(nb)]
        effective_files = sum(1 for s in sizes if s > 0)
        print(f"Found {total} participants -> {effective_files} batch file(s) (requested {nb}); sizes: first {r} have {q+1}, remaining have {q}")
        if args.dry_run:
            return
        files = write_batches_by_sizes(pairs, out_dir, sizes, args.prefix, args.start_index)
        for p in files:
            print(f"Wrote {p}")
        return
    else:
        num_files = (total + args.batch_size - 1) // args.batch_size if total else 0
        print(f"Found {total} participants -> {num_files} batch file(s) of size {args.batch_size} (last may be smaller)")

        if args.dry_run:
            return

        files = write_batches(pairs, out_dir, args.batch_size, args.prefix, args.start_index)
        for p in files:
            print(f"Wrote {p}")


if __name__ == '__main__':
    main()
