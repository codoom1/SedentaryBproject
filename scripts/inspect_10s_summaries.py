#!/usr/bin/env python3
"""Inspect raw 10-second data and downstream summaries for one participant."""

import argparse
import random
import re
from pathlib import Path

import pandas as pd


ID_PATTERN = re.compile(r"ID=([^/]+)")
PARTITION_PATTERN = re.compile(
    r"model=(?P<model>[^/]+)/Dataset=(?P<Dataset>[^/]+)/ID=(?P<ID>[^/]+)"
    r"(?:/Day=(?P<Day>[^/]+))?"
)
SHORT_DAY_COLUMNS = [
    "recorded_hours",
    "waking_hours",
    "waking_sedentary_hours",
    "number_sedentary_bouts",
    "number_breaks",
    "mean_bout_duration_minutes",
    "maximum_bout_duration_minutes",
    "breaks_per_sedentary_hour",
    "percent_sedentary_time_in_bouts_ge_30min",
    "sustained_breaks_ge_30sec",
]
SHORT_PARTICIPANT_COLUMNS = [
    "valid_days",
    "mean_recorded_hours",
    "mean_waking_hours",
    "mean_waking_sedentary_hours",
    "mean_number_sedentary_bouts",
    "mean_number_breaks",
    "mean_maximum_bout_duration_minutes",
    "mean_breaks_per_sedentary_hour",
    "mean_percent_sedentary_time_in_bouts_ge_30min",
]


def participant_id_from_path(path: Path) -> str:
    match = ID_PATTERN.search(path.as_posix())
    if not match:
        raise ValueError(f"Could not find ID partition in {path}")
    return match.group(1)


def partitions_from_path(path: Path) -> dict:
    match = PARTITION_PATTERN.search(path.as_posix())
    if not match:
        raise ValueError(f"Could not parse partition fields from {path}")
    return {k: v for k, v in match.groupdict().items() if v is not None}


def add_partitions(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.copy()
    for column, value in reversed(list(partitions_from_path(path).items())):
        if column not in out.columns:
            out.insert(0, column, value)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s_summaries",
    )
    parser.add_argument(
        "--raw-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s",
    )
    parser.add_argument("--participant-id", default=None)
    parser.add_argument("--day", default=None, help="Specific date to inspect, e.g. 2000-01-07")
    parser.add_argument(
        "--view",
        choices=["raw", "day", "participant", "all"],
        required=True,
        help="Choose which dataset level to inspect",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw-rows", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for the selected view. For --view all, suffixes are added.",
    )
    parser.add_argument("--full", action="store_true", help="Print every summary column")
    args = parser.parse_args()

    root = Path(args.summary_root)
    day_files = sorted(
        (root / "participant_day").glob("model=*/Dataset=*/ID=*/Day=*/part-0.parquet")
    )
    if not day_files:
        raise SystemExit(f"No participant-day summary files found under {root}")

    participant_ids = sorted({participant_id_from_path(path) for path in day_files})
    if args.participant_id is not None:
        participant_id = str(args.participant_id)
        if participant_id not in participant_ids:
            raise SystemExit(f"No summaries found for participant {participant_id}")
    else:
        participant_id = random.Random(args.seed).choice(participant_ids)

    selected_days = [
        path for path in day_files if participant_id_from_path(path) == participant_id
    ]
    first_day = selected_days[0]
    participant_file = (
        root
        / "participant"
        / first_day.parts[-5]
        / first_day.parts[-4]
        / first_day.parts[-3]
        / "part-0.parquet"
    )

    day_frames = []
    for path in selected_days:
        frame = add_partitions(pd.read_parquet(path), path)
        day_frames.append(frame)
    days = pd.concat(day_frames, ignore_index=True).sort_values("Day")

    rng = random.Random(args.seed)
    available_days = sorted(days["Day"].astype(str).tolist())
    if args.day is not None:
        selected_day = str(args.day)
        if selected_day not in available_days:
            raise SystemExit(
                f"No summary day {selected_day} for participant {participant_id}. "
                f"Available: {', '.join(available_days)}"
            )
    else:
        selected_day = rng.choice(available_days)

    print(f"Participant: {participant_id}")
    if args.view in {"raw", "all"}:
        print(f"Selected day: {selected_day}")
        print(f"Raw root: {args.raw_root}")
    if args.view in {"day", "participant", "all"}:
        print(f"Participant-days: {len(days)}")
        print(f"Summary root: {root}")
    print()

    if args.view in {"raw", "all"}:
        raw_root = Path(args.raw_root)
        raw_candidates = sorted(
            raw_root.glob(
                f"model=*/Dataset=*/ID={participant_id}/Day={selected_day}/part-0.parquet"
            )
        )
        print(f"Raw 10-second data for Day={selected_day}")
        if not raw_candidates:
            print("Raw participant-day file not found.")
            raw = None
            raw_file = None
        else:
            raw_file = raw_candidates[0]
            raw = add_partitions(pd.read_parquet(raw_file), raw_file)
            print(f"File: {raw_file}")
            print(f"Rows: {len(raw)}")
            print()
            print(raw.head(args.raw_rows).to_string(index=False))
            print()
            print("Raw SWaN-by-sitting counts")
            print(pd.crosstab(raw["sleep_state"], raw["sitting"]).to_string())
            if args.output_csv and args.view == "raw":
                write_csv(raw, Path(args.output_csv))
        print()

    if args.view in {"day", "all"}:
        print("Participant-day summaries")
        day_view = days.copy()
        if args.full:
            print(day_view.T.to_string())
        else:
            columns = ["Day"] + [column for column in SHORT_DAY_COLUMNS if column in days.columns]
            print(day_view[columns].to_string(index=False))
        if args.output_csv and args.view == "day":
            write_csv(day_view, Path(args.output_csv))
        print()

    if args.view in {"participant", "all"}:
        print("Participant-level summary")
        if not participant_file.exists():
            print(f"Not found yet: {participant_file}")
            return

        participant = add_partitions(pd.read_parquet(participant_file), participant_file)
        if args.full:
            print(participant.T.to_string())
        else:
            columns = [
                column for column in SHORT_PARTICIPANT_COLUMNS if column in participant.columns
            ]
            print(participant[columns].T.to_string(header=False))
        if args.output_csv and args.view == "participant":
            write_csv(participant, Path(args.output_csv))

    if args.output_csv and args.view == "all":
        base = Path(args.output_csv)
        stem = base.with_suffix("")
        suffix = base.suffix or ".csv"
        if "raw" in locals() and raw is not None:
            write_csv(raw, Path(f"{stem}_raw{suffix}"))
        write_csv(days, Path(f"{stem}_participant_day{suffix}"))
        if participant_file.exists():
            write_csv(participant, Path(f"{stem}_participant{suffix}"))


if __name__ == "__main__":
    main()
