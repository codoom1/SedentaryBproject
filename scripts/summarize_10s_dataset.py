#!/usr/bin/env python3
"""Incrementally summarize durable 10-second Parquet files."""

import argparse
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EPOCH_SECONDS = 10
STATE_NAMES = {-1: "missing", 0: "wear", 1: "sleep", 2: "nonwear"}
LONG_BOUT_MINUTES = (10, 20, 30, 60)
SUSTAINED_BREAK_SECONDS = (30, 60, 120)
PARTITION_PATTERN = re.compile(
    r"model=(?P<model>[^/]+)/Dataset=(?P<Dataset>[^/]+)/ID=(?P<ID>[^/]+)/Day=(?P<Day>[^/]+)/part-0\.parquet$"
)


def parse_partitions(path: Path) -> Optional[Dict[str, str]]:
    match = PARTITION_PATTERN.search(path.as_posix())
    return match.groupdict() if match else None


def atomic_write_parquet(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".parquet.tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        df.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def day_output_path(output_root: Path, parts: Dict[str, str]) -> Path:
    return (
        output_root
        / "participant_day"
        / f"model={parts['model']}"
        / f"Dataset={parts['Dataset']}"
        / f"ID={parts['ID']}"
        / f"Day={parts['Day']}"
        / "part-0.parquet"
    )


def participant_output_path(output_root: Path, parts: Dict[str, str]) -> Path:
    return (
        output_root
        / "participant"
        / f"model={parts['model']}"
        / f"Dataset={parts['Dataset']}"
        / f"ID={parts['ID']}"
        / "part-0.parquet"
    )


def add_bout_and_break_metrics(df: pd.DataFrame, row: Dict[str, object]) -> None:
    """Add gap-aware waking sedentary bout and break metrics to a day summary."""
    ordered = df.sort_values("epoch_index").drop_duplicates("epoch_index", keep="first").copy()
    ordered["waking"] = ordered["valid_sleep"] & ordered["sleep_state"].eq(0)
    ordered["consecutive"] = ordered["epoch_index"].diff().eq(1)
    ordered["prev_waking"] = ordered["waking"].shift(fill_value=False)
    ordered["prev_sitting"] = ordered["sitting"].shift(fill_value=False)

    waking_sitting = ordered["waking"] & ordered["sitting"]
    ordered["new_bout"] = waking_sitting & ~(
        ordered["consecutive"] & ordered["prev_waking"] & ordered["prev_sitting"]
    )
    ordered["bout_id"] = ordered["new_bout"].cumsum()
    bouts = (
        ordered.loc[waking_sitting]
        .groupby("bout_id", sort=False)
        .size()
        .astype("int64")
    )
    bout_minutes = bouts * EPOCH_SECONDS / 60

    row["number_sedentary_bouts"] = int(len(bouts))
    row["mean_bout_duration_minutes"] = float(bout_minutes.mean()) if len(bouts) else 0.0
    row["median_bout_duration_minutes"] = float(bout_minutes.median()) if len(bouts) else 0.0
    row["maximum_bout_duration_minutes"] = float(bout_minutes.max()) if len(bouts) else 0.0

    for threshold in LONG_BOUT_MINUTES:
        hours = float(bout_minutes.loc[bout_minutes >= threshold].sum() / 60)
        row[f"sedentary_hours_in_bouts_ge_{threshold}min"] = hours
        row[f"percent_sedentary_time_in_bouts_ge_{threshold}min"] = (
            100 * hours / row["waking_sedentary_hours"]
            if row["waking_sedentary_hours"] > 0
            else np.nan
        )

    ordered["break_in_sitting"] = (
        ordered["consecutive"]
        & ordered["waking"]
        & ~ordered["sitting"]
        & ordered["prev_waking"]
        & ordered["prev_sitting"]
    )
    row["number_breaks"] = int(ordered["break_in_sitting"].sum())
    row["breaks_per_sedentary_hour"] = (
        row["number_breaks"] / row["waking_sedentary_hours"]
        if row["waking_sedentary_hours"] > 0
        else np.nan
    )
    row["breaks_per_waking_hour"] = (
        row["number_breaks"] / row["waking_hours"] if row["waking_hours"] > 0 else np.nan
    )

    run_boundary = (
        ~ordered["consecutive"]
        | ordered["waking"].ne(ordered["waking"].shift(fill_value=False))
        | ordered["sitting"].ne(ordered["sitting"].shift(fill_value=False))
    )
    ordered["run_id"] = run_boundary.cumsum()
    non_sitting_runs = ordered.loc[ordered["waking"] & ~ordered["sitting"]].groupby(
        "run_id", sort=False
    )
    break_run_lengths = []
    for _, run in non_sitting_runs:
        if bool(run["break_in_sitting"].iloc[0]):
            break_run_lengths.append(len(run))
    for seconds in SUSTAINED_BREAK_SECONDS:
        minimum_epochs = seconds // EPOCH_SECONDS
        row[f"sustained_breaks_ge_{seconds}sec"] = int(
            sum(length >= minimum_epochs for length in break_run_lengths)
        )


def summarize_day(source: Path, parts: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(source, columns=["epoch_index", "sitting", "sleep_state", "valid_sleep"])
    if df.empty:
        raise ValueError(f"No rows in {source}")

    df["sitting"] = df["sitting"].astype(bool)
    df["sleep_state"] = pd.to_numeric(df["sleep_state"], errors="coerce").fillna(-1).astype("int8")
    df["valid_sleep"] = df["valid_sleep"].astype(bool)

    row = {
        "DayType": "Weekend" if pd.Timestamp(parts["Day"]).dayofweek >= 5 else "Weekday",
        "n_epochs": int(len(df)),
        "recorded_hours": len(df) * EPOCH_SECONDS / 3600,
        "first_epoch_index": int(df["epoch_index"].min()),
        "last_epoch_index": int(df["epoch_index"].max()),
        "n_epoch_gaps": int((df["epoch_index"].sort_values().diff().dropna() != 1).sum()),
        "valid_sleep_epochs": int(df["valid_sleep"].sum()),
        "missing_sleep_epochs": int((~df["valid_sleep"]).sum()),
        "sitting_epochs": int(df["sitting"].sum()),
        "not_sitting_epochs": int((~df["sitting"]).sum()),
    }

    for state_code, state_name in STATE_NAMES.items():
        state_mask = df["sleep_state"].eq(state_code)
        row[f"{state_name}_not_sitting_epochs"] = int((state_mask & ~df["sitting"]).sum())
        row[f"{state_name}_sitting_epochs"] = int((state_mask & df["sitting"]).sum())

    epoch_count_columns = [
        name for name in row if name.endswith("_epochs") and name != "n_epochs"
    ]
    for column in epoch_count_columns:
        row[column.replace("_epochs", "_hours")] = row[column] * EPOCH_SECONDS / 3600

    row["waking_sedentary_hours"] = row["wear_sitting_hours"]
    row["all_sitting_hours"] = row["sitting_hours"]
    row["waking_hours"] = row["wear_sitting_hours"] + row["wear_not_sitting_hours"]
    row["sleep_hours"] = row["sleep_sitting_hours"] + row["sleep_not_sitting_hours"]
    row["nonwear_hours"] = row["nonwear_sitting_hours"] + row["nonwear_not_sitting_hours"]
    row["waking_sedentary_percent"] = (
        100 * row["waking_sedentary_hours"] / row["waking_hours"]
        if row["waking_hours"] > 0
        else np.nan
    )
    row["analysis_epoch_seconds"] = EPOCH_SECONDS
    row["swan_native_epoch_seconds"] = 30
    add_bout_and_break_metrics(df, row)
    return pd.DataFrame([row])


def summarize_participant(day_files: Iterable[Path], parts: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for path in sorted(day_files):
        frame = pd.read_parquet(path)
        day_parts = parse_partitions(path)
        if day_parts is None:
            continue
        frame["Day"] = day_parts["Day"]
        frames.append(frame)
    days = pd.concat(frames, ignore_index=True)
    if days.empty:
        raise ValueError(f"No participant-day summaries for participant {parts['ID']}")

    identifiers = {"Day", "DayType"}
    numeric_columns = [
        column
        for column in days.columns
        if column not in identifiers and pd.api.types.is_numeric_dtype(days[column])
    ]
    row = {
        "valid_days": int(len(days)),
        "weekday_days": int((days["DayType"] == "Weekday").sum()),
        "weekend_days": int((days["DayType"] == "Weekend").sum()),
        "first_day": str(days["Day"].min()),
        "last_day": str(days["Day"].max()),
    }
    for column in numeric_columns:
        row[f"mean_{column}"] = days[column].mean()
    return pd.DataFrame([row])


def write_csv_snapshot(dataset_root: Path, destination: Path) -> None:
    import pyarrow.dataset as ds

    if not dataset_root.exists():
        return
    table = ds.dataset(dataset_root, format="parquet", partitioning="hive").to_table()
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_pandas().to_csv(destination, index=False, compression="gzip")
    logger.info("Wrote CSV snapshot: %s", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/epoch_10s")
    parser.add_argument("--output-root", default="data/epoch_10s_summaries")
    parser.add_argument("--overwrite-days", action="store_true")
    parser.add_argument("--refresh-all-participants", action="store_true")
    parser.add_argument("--write-csv-snapshots", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    source_files = sorted(input_root.glob("model=*/Dataset=*/ID=*/Day=*/part-0.parquet"))
    if not source_files:
        raise SystemExit(f"No 10-second Parquet files found under {input_root}")

    affected_participants: Dict[tuple, Dict[str, str]] = {}
    processed_days = 0
    skipped_days = 0
    failed_days = 0

    for source in source_files:
        parts = parse_partitions(source)
        if parts is None:
            logger.warning("Could not parse partitions; skipping %s", source)
            failed_days += 1
            continue
        destination = day_output_path(output_root, parts)
        if (
            destination.exists()
            and not args.overwrite_days
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            skipped_days += 1
            continue
        try:
            atomic_write_parquet(summarize_day(source, parts), destination)
            logger.info("Wrote participant-day summary: %s", destination)
            affected_participants[(parts["model"], parts["Dataset"], parts["ID"])] = parts
            processed_days += 1
        except Exception as exc:
            logger.warning("Could not summarize %s; it may still be writing: %s", source, exc)
            failed_days += 1

    participant_roots = sorted(
        (output_root / "participant_day").glob("model=*/Dataset=*/ID=*")
    )
    refreshed_participants = 0
    for participant_root in participant_roots:
        day_files = sorted(participant_root.glob("Day=*/part-0.parquet"))
        if not day_files:
            continue
        parts = parse_partitions(day_files[0])
        if parts is None:
            continue
        key = (parts["model"], parts["Dataset"], parts["ID"])
        destination = participant_output_path(output_root, parts)
        newest_day_mtime = max(path.stat().st_mtime for path in day_files)
        needs_refresh = (
            args.refresh_all_participants
            or key in affected_participants
            or not destination.exists()
            or destination.stat().st_mtime < newest_day_mtime
        )
        if not needs_refresh:
            continue
        atomic_write_parquet(summarize_participant(day_files, parts), destination)
        logger.info("Wrote participant summary: %s", destination)
        refreshed_participants += 1

    if args.write_csv_snapshots:
        write_csv_snapshot(
            output_root / "participant_day",
            output_root / "participant_day_snapshot.csv.gz",
        )
        write_csv_snapshot(
            output_root / "participant",
            output_root / "participant_snapshot.csv.gz",
        )

    logger.info(
        "Complete: processed_days=%d skipped_unchanged_days=%d failed_days=%d "
        "refreshed_participants=%d",
        processed_days,
        skipped_days,
        failed_days,
        refreshed_participants,
    )


if __name__ == "__main__":
    main()
