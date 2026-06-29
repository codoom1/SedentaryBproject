#!/usr/bin/env python3
"""Check participant coverage for durable 10-second epoch outputs.

The 10-second export is partitioned as:
    model=<MODEL>/Dataset=<CYCLE>/ID=<SEQN>/Day=<YYYY-MM-DD>/part-0.parquet

This script compares those partitions with constants/participants.csv and can
write a missing-participant CSV plus SLURM batch files for reruns.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            cycle = row[0].strip()
            pid = row[1].strip()
            if cycle.lower() == "cycle" and pid.lower() == "participant_id":
                continue
            if not cycle or not pid:
                continue
            pair = (cycle, pid)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def parse_partition(path: Path) -> tuple[str | None, str | None, str | None]:
    model = dataset = pid = None
    for part in path.parts:
        if part.startswith("model="):
            model = part.split("=", 1)[1]
        elif part.startswith("Dataset="):
            dataset = part.split("=", 1)[1]
        elif part.startswith("ID="):
            pid = part.split("=", 1)[1]
    return model, dataset, pid


def find_present(
    output_root: Path,
    model_filter: str | None,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], int]]:
    present: set[tuple[str, str]] = set()
    day_counts: dict[tuple[str, str], set[str]] = defaultdict(set)

    for parquet in output_root.glob("model=*/Dataset=*/ID=*/Day=*/part-*.parquet"):
        model, dataset, pid = parse_partition(parquet)
        if not dataset or not pid:
            continue
        if model_filter and model != model_filter:
            continue
        key = (dataset, pid)
        present.add(key)
        day = parquet.parent.name
        if day.startswith("Day="):
            day_counts[key].add(day.split("=", 1)[1])

    return present, {key: len(days) for key, days in day_counts.items()}


def write_pairs(path: Path, pairs: list[tuple[str, str]], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["cycle", "participant_id"])
        writer.writerows(pairs)


def write_batches(
    out_dir: Path,
    pairs: list[tuple[str, str]],
    batch_size: int,
    prefix: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, start in enumerate(range(0, len(pairs), batch_size), start=1):
        batch = pairs[start : start + batch_size]
        path = out_dir / f"{prefix}{index}.txt"
        write_pairs(path, batch, header=False)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare constants/participants.csv with epoch_10s parquet output."
    )
    parser.add_argument(
        "--participants",
        default="constants/participants.csv",
        type=Path,
        help="Manifest CSV with cycle,participant_id columns.",
    )
    parser.add_argument(
        "--output-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s",
        type=Path,
        help="Root of durable 10-second Parquet output.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model partition to count, for example CHAP.",
    )
    parser.add_argument(
        "--missing-out",
        default=None,
        type=Path,
        help="Optional CSV path for missing participants.",
    )
    parser.add_argument(
        "--batch-out-dir",
        default=None,
        type=Path,
        help="Optional directory for rerun batch files from missing participants.",
    )
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--batch-prefix", default="missing_10s_batch_")
    args = parser.parse_args()

    manifest = read_manifest(args.participants)
    expected = set(manifest)
    present, day_counts = find_present(args.output_root, args.model)
    present_expected = expected & present
    missing = [pair for pair in manifest if pair not in present]
    extra = present - expected

    print(f"participants_expected={len(manifest)}")
    print(f"participants_with_epoch_10s={len(present_expected)}")
    print(f"participants_missing_epoch_10s={len(missing)}")
    print(f"extra_output_participants_not_in_manifest={len(extra)}")
    if day_counts:
        counts = [day_counts[pair] for pair in present_expected if pair in day_counts]
        print(f"participant_day_min={min(counts)}")
        print(f"participant_day_max={max(counts)}")
        print(f"participant_day_total={sum(counts)}")

    if args.missing_out:
        write_pairs(args.missing_out, missing, header=True)
        print(f"missing_csv={args.missing_out}")

    if args.batch_out_dir:
        if args.batch_size <= 0:
            raise SystemExit("--batch-size must be positive")
        files = write_batches(args.batch_out_dir, missing, args.batch_size, args.batch_prefix)
        print(f"missing_batch_files={len(files)}")
        if files:
            print(f"missing_batch_first={files[0]}")
            print(f"missing_batch_last={files[-1]}")


if __name__ == "__main__":
    main()
