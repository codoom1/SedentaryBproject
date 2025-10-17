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

New options:
- --dedupe-by: choose 'pair' (default) to dedupe by (cycle,participant_id) or 'participant' to dedupe by participant_id across cycles.

Examples:
    python scripts/make_batches_from_constants.py --batch-size 100
    python scripts/make_batches_from_constants.py --batch-size 250 --shuffle --seed 42
    python scripts/make_batches_from_constants.py --participants constants/participants.csv --out-dir batches --prefix nhanes_ --start-index 10
    python scripts/make_batches_from_constants.py --dedupe-by participant  # unique participants across cycles
    # Exclude participants already present in final data
    python scripts/make_batches_from_constants.py --exclude-from-final data/project_data/final_data.csv.gz --num-batches 25 --shuffle --seed 42 --clean
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
from pathlib import Path
from typing import List, Tuple, Set, Literal, Sequence


def read_participants(
    csv_path: Path,
    dedupe: bool = True,
    dedupe_by: Literal['pair', 'participant'] = 'pair',
) -> List[Tuple[str, str]]:
    """Read (cycle, participant_id) rows from CSV.

    - dedupe=True with dedupe_by='pair' removes duplicate (cycle,participant_id) pairs.
    - dedupe=True with dedupe_by='participant' keeps only the first occurrence of a participant_id across cycles.
    """
    pairs: List[Tuple[str, str]] = []
    # Track seen keys based on dedupe strategy
    seen: Set[object] = set()
    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for _, row in enumerate(reader):
            if not row:
                continue
            # skip commented lines
            first_cell = row[0] if isinstance(row[0], str) else ''
            if first_cell.strip().startswith('#'):
                continue
            # skip header if present (regardless of row index); handle BOM
            c0 = (row[0] if isinstance(row[0], str) else '').strip().lstrip('\ufeff').lower()
            c1 = (row[1] if len(row) > 1 and isinstance(row[1], str) else '').strip().lower()
            if len(row) >= 2 and c0 == 'cycle' and c1 == 'participant_id':
                continue
            if len(row) < 2:
                continue
            cycle = (row[0] if isinstance(row[0], str) else '').strip()
            pid = (row[1] if isinstance(row[1], str) else '').strip()
            if not cycle or not pid:
                continue
            key = (cycle, pid)
            if dedupe:
                if dedupe_by == 'pair':
                    if key in seen:  # type: ignore[arg-type]
                        continue
                    seen.add(key)  # type: ignore[arg-type]
                else:  # dedupe by participant_id
                    if pid in seen:  # type: ignore[operator]
                        continue
                    seen.add(pid)  # type: ignore[arg-type]
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


ID_CANDIDATES = [
    'ID', 'id', 'participant_id', 'participant', 'pid', 'subject',
    'ParticipantID', 'participantId', 'person_id', 'user_id'
]


def _find_id_column(headers: Sequence[str]) -> str | None:
    lower_map = {h.lower(): h for h in headers}
    for cand in ID_CANDIDATES:
        if cand in headers:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_final_participant_ids(final_path: Path) -> Set[str]:
    """Load participant IDs from a gzipped CSV final data file."""
    ids: Set[str] = set()
    if not final_path.exists():
        print(f"Warning: --exclude-from-final path not found: {final_path}")
        return ids
    # Read as gz CSV with DictReader
    with gzip.open(final_path, mode='rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"Warning: No header found in {final_path}; cannot exclude participants")
            return ids
        id_col = _find_id_column(reader.fieldnames)
        if not id_col:
            print(f"Warning: Could not identify participant ID column in {final_path}; headers: {reader.fieldnames}")
            return ids
        for row in reader:
            pid = row.get(id_col)
            if pid is not None:
                ids.add(str(pid))
    return ids


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
    ap.add_argument('--dedupe-by', choices=['pair', 'participant'], default='pair',
                    help="Dedupe strategy when not using --no-dedupe. 'pair' = (cycle,participant_id) (default); 'participant' = participant_id across cycles")
    ap.add_argument('--clean', action='store_true', help='Remove existing batch files with the same prefix before writing')
    ap.add_argument('--exclude-from-final', default=None, help='Path to gzipped CSV final data file; exclude participants already present in this file')
    args = ap.parse_args()

    csv_path = Path(args.participants)
    out_dir = Path(args.out_dir)

    pairs = read_participants(csv_path, dedupe=(not args.no_dedupe), dedupe_by=args.dedupe_by)
    total = len(pairs)

    # Optionally exclude participants already present in final data
    if args.exclude_from_final:
        final_path = Path(args.exclude_from_final)
        final_ids = load_final_participant_ids(final_path)
        before = len(pairs)
        pairs = [(c, pid) for (c, pid) in pairs if pid not in final_ids]
        after = len(pairs)
        excluded = before - after
        print(f"Excluding participants found in {final_path}: {excluded} excluded; {after} remaining of {before}.")
        # Recompute total after exclusion so sizes and counts are correct
        total = after
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    # Optionally clean existing batch files for this prefix to avoid stale overlaps across runs
    if not args.dry_run and args.clean:
        out_dir.mkdir(parents=True, exist_ok=True)
        to_delete = list(out_dir.glob(f"{args.prefix}*.txt"))
        for p in to_delete:
            try:
                p.unlink()
            except Exception as e:
                print(f"Warning: failed to remove {p}: {e}")

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

# python scripts/make_batches_from_constants.py --num-batches 25 --shuffle --seed 42 --out-dir batches --exclude-from-final data/project_data/final_data.csv.gz --clean