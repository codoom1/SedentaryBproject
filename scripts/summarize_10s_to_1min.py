#!/usr/bin/env python3
"""Create incremental 1-minute CHAP/SWaN data and join NHANES PAXMIN states.

Two subcommands are provided:

cache-paxmin
    Stream a large PAXMIN XPT once and write compact participant Parquet files.

summarize
    Aggregate available 10-second participant-day Parquet files to one minute,
    then join the cached NHANES minute record when available.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARTITION_PATTERN = re.compile(
    r"model=(?P<model>[^/]+)/Dataset=(?P<Dataset>[^/]+)/ID=(?P<ID>[^/]+)"
    r"/Day=(?P<Day>[^/]+)/part-0\.parquet$"
)
NHANES_STATE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
NHANES_STATE_NAMES = {0: "wear", 1: "sleep", 2: "nonwear", 3: "unknown"}
EXPECTED_XPT_BYTES = {"2011-2012": 8_125_196_000, "2013-2014": 9_351_691_760}
PAXMIN_KEEP_COLUMNS = [
    "SEQN",
    "PAXDAYM",
    "PAXDAYWM",
    "PAXSSNMP",
    "PAXTSM",
    "PAXAISMM",
    "PAXMTSM",
    "PAXMXM",
    "PAXMYM",
    "PAXMZM",
    "PAXPREDM",
    "PAXTRANM",
    "PAXLXMM",
    "PAXLXSDM",
    "PAXQFM",
    "PAXFLGSM",
]


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


def parse_parts(path: Path) -> dict[str, str]:
    match = PARTITION_PATTERN.search(path.as_posix())
    if not match:
        raise ValueError(f"Could not parse partition path: {path}")
    return match.groupdict()


def cache_path(cache_root: Path, dataset: str, participant_id: str) -> Path:
    return cache_root / f"Dataset={dataset}" / f"ID={participant_id}" / "part-0.parquet"


def output_path(output_root: Path, parts: dict[str, str]) -> Path:
    return (
        output_root
        / f"model={parts['model']}"
        / f"Dataset={parts['Dataset']}"
        / f"ID={parts['ID']}"
        / f"Day={parts['Day']}"
        / "part-0.parquet"
    )


def normalize_seqn(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64").astype(str)


def write_cached_participant(
    frame: pd.DataFrame,
    dataset: str,
    cache_root: Path,
    overwrite: bool,
) -> bool:
    if frame.empty:
        return False
    participant_id = str(int(float(frame["SEQN"].iloc[0])))
    destination = cache_path(cache_root, dataset, participant_id)
    if destination.exists() and not overwrite:
        return False

    available = [column for column in PAXMIN_KEEP_COLUMNS if column in frame.columns]
    out = frame[available].copy()
    out["SEQN"] = normalize_seqn(out["SEQN"])
    for column in available:
        if column not in {"SEQN", "PAXFLGSM"}:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(["PAXDAYM", "PAXSSNMP"]).reset_index(drop=True)
    out["nhanes_minute_index"] = out.groupby("PAXDAYM", sort=False).cumcount().astype("int16")
    atomic_write_parquet(out, destination)
    logger.info("Cached PAXMIN participant %s: %s rows", participant_id, len(out))
    return True


def cache_paxmin(args: argparse.Namespace) -> None:
    xpt = Path(args.xpt)
    if not xpt.exists():
        raise SystemExit(f"PAXMIN file not found: {xpt}")
    expected_bytes = EXPECTED_XPT_BYTES[args.dataset]
    actual_bytes = xpt.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"PAXMIN download is incomplete: {xpt} has {actual_bytes} bytes; "
            f"expected {expected_bytes}. Let download_nhanes_paxmin.sh finish first."
        )

    wanted: set[str] | None = None
    if args.epoch_10s_root:
        wanted = {
            path.name.split("=", 1)[1]
            for path in Path(args.epoch_10s_root).glob(
                f"model=*/Dataset={args.dataset}/ID=*"
            )
        }
        logger.info("Restricting cache to %d participant IDs found in 10-second data", len(wanted))

    reader = pd.read_sas(
        xpt,
        format="xport",
        iterator=True,
        chunksize=args.chunk_rows,
        encoding="latin1",
    )
    carry = pd.DataFrame()
    cached = 0
    rows_read = 0
    previous_seqn: int | None = None

    for chunk_number, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        seqn = pd.to_numeric(chunk["SEQN"], errors="coerce").round().astype("Int64")
        if not seqn.is_monotonic_increasing:
            raise RuntimeError("PAXMIN is not sorted by SEQN; streaming cache would be unsafe")
        if previous_seqn is not None and int(seqn.dropna().iloc[0]) < previous_seqn:
            raise RuntimeError("PAXMIN SEQN order moved backwards")

        last_seqn = seqn.dropna().iloc[-1]
        complete = chunk.loc[seqn.ne(last_seqn)].copy()
        carry = chunk.loc[seqn.eq(last_seqn)].copy()
        previous_seqn = int(last_seqn)

        for participant_value, participant in complete.groupby("SEQN", sort=False):
            participant_id = str(int(float(participant_value)))
            if wanted is None or participant_id in wanted:
                cached += int(
                    write_cached_participant(
                        participant, args.dataset, Path(args.cache_root), args.overwrite
                    )
                )

        logger.info(
            "PAXMIN chunk=%d rows_read=%d cached_participants=%d",
            chunk_number,
            rows_read,
            cached,
        )

    if not carry.empty:
        participant_id = str(int(float(carry["SEQN"].iloc[0])))
        if wanted is None or participant_id in wanted:
            cached += int(
                write_cached_participant(
                    carry, args.dataset, Path(args.cache_root), args.overwrite
                )
            )
    logger.info("PAXMIN cache complete: rows_read=%d new_participants=%d", rows_read, cached)


def aggregate_10s(source: Path) -> pd.DataFrame:
    df = pd.read_parquet(source)
    if df.empty:
        raise ValueError(f"No rows in {source}")
    df = df.sort_values("epoch_index").drop_duplicates("epoch_index", keep="first")
    df["minute_index"] = (pd.to_numeric(df["epoch_index"]) // 6).astype("int16")
    df["sitting"] = df["sitting"].astype(bool)
    df["valid_sleep"] = df["valid_sleep"].astype(bool)
    df["sleep_state"] = pd.to_numeric(df["sleep_state"], errors="coerce").fillna(-1).astype("int8")
    df["half_minute"] = (pd.to_numeric(df["epoch_index"]) % 6 >= 3).astype("int8")
    df["swan_wear"] = df["sleep_state"].eq(0).astype("int8")
    df["swan_sleep"] = df["sleep_state"].eq(1).astype("int8")
    df["swan_nonwear"] = df["sleep_state"].eq(2).astype("int8")
    df["swan_missing"] = (~df["valid_sleep"]).astype("int8")

    grouped = df.groupby("minute_index", sort=True, observed=True)
    out = grouped.agg(
        n_10s_epochs=("epoch_index", "size"),
        chap_sitting_10s_epochs=("sitting", "sum"),
        swan_wear_10s_epochs=("swan_wear", "sum"),
        swan_sleep_10s_epochs=("swan_sleep", "sum"),
        swan_nonwear_10s_epochs=("swan_nonwear", "sum"),
        swan_missing_10s_epochs=("swan_missing", "sum"),
    )

    # Select the modal valid state for each native 30-second half. Ties use
    # the lower state code, matching the previous implementation.
    valid_states = df.loc[df["sleep_state"].ge(0)]
    state_counts = (
        valid_states.groupby(
            ["minute_index", "half_minute", "sleep_state"],
            sort=True,
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["minute_index", "half_minute", "count", "sleep_state"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["minute_index", "half_minute"])
    )
    half_states = state_counts.pivot(
        index="minute_index", columns="half_minute", values="sleep_state"
    ).reindex(out.index)
    first_state = (
        half_states[0] if 0 in half_states.columns else pd.Series(-1, index=out.index)
    ).fillna(-1).astype("int8")
    second_state = (
        half_states[1] if 1 in half_states.columns else pd.Series(-1, index=out.index)
    ).fillna(-1).astype("int8")

    out.insert(0, "hour", (out.index // 60).astype("int8"))
    out.insert(1, "minute_of_hour", (out.index % 60).astype("int8"))
    out["complete_minute"] = out["n_10s_epochs"].eq(6)
    out["chap_sitting_seconds"] = out["chap_sitting_10s_epochs"] * 10
    out["chap_sitting_fraction"] = (
        out["chap_sitting_10s_epochs"] / out["n_10s_epochs"]
    )
    out["chap_sitting_majority"] = out["chap_sitting_10s_epochs"].ge(3)
    out["chap_sitting_any"] = out["chap_sitting_10s_epochs"].gt(0)
    out["chap_sitting_full_minute"] = out["chap_sitting_10s_epochs"].eq(6)
    out["swan_state_first_30s"] = first_state
    out["swan_state_second_30s"] = second_state
    out["swan_transition"] = (
        first_state.ge(0) & second_state.ge(0) & first_state.ne(second_state)
    )
    out["swan_state_1min"] = np.where(
        first_state.eq(second_state), first_state, -2
    ).astype("int8")
    out["swan_waking_1min"] = out["swan_state_1min"].eq(0)
    out["waking_sedentary_1min"] = (
        out["swan_waking_1min"] & out["chap_sitting_majority"]
    )

    probability_columns = [
        column
        for column in ["prob_sitting", "prob_wear", "prob_sleep", "prob_nonwear"]
        if column in df.columns
    ]
    if probability_columns:
        probability_means = grouped[probability_columns].mean().rename(
            columns={column: f"{column}_mean" for column in probability_columns}
        )
        out = out.join(probability_means)

    out = out.reset_index()
    ordered_columns = [
        "minute_index",
        "hour",
        "minute_of_hour",
        "n_10s_epochs",
        "complete_minute",
        "chap_sitting_10s_epochs",
        "chap_sitting_seconds",
        "chap_sitting_fraction",
        "chap_sitting_majority",
        "chap_sitting_any",
        "chap_sitting_full_minute",
        "swan_state_first_30s",
        "swan_state_second_30s",
        "swan_transition",
        "swan_state_1min",
        "swan_wear_10s_epochs",
        "swan_sleep_10s_epochs",
        "swan_nonwear_10s_epochs",
        "swan_missing_10s_epochs",
        "swan_waking_1min",
        "waking_sedentary_1min",
    ]
    ordered_columns += [
        f"{column}_mean" for column in probability_columns
    ]
    out = out[ordered_columns]
    for column in [
        "chap_sitting_fraction",
        "prob_sitting_mean",
        "prob_wear_mean",
        "prob_sleep_mean",
        "prob_nonwear_mean",
    ]:
        if column in out:
            out[column] = out[column].astype("float32")
    return out


def choose_pax_anchor(
    pax: pd.DataFrame, available_days: Iterable[str]
) -> tuple[int | None, str]:
    dates = sorted(pd.Timestamp(day) for day in available_days)
    if not dates:
        return None, "no_10s_days"

    day_info = (
        pax.groupby("PAXDAYM", sort=True)
        .agg(n_minutes=("PAXDAYM", "size"), weekday=("PAXDAYWM", "first"))
        .reset_index()
    )
    full = day_info.loc[day_info["n_minutes"].eq(1440)].copy()
    if full.empty:
        return None, "no_full_nhanes_days"

    first_date = dates[0]
    expected_weekday = ((first_date.dayofweek + 1) % 7) + 1  # pandas Mon=0; NHANES Sun=1
    candidates = full.loc[full["weekday"].eq(expected_weekday), "PAXDAYM"].astype(int).tolist()
    if not candidates:
        candidates = full["PAXDAYM"].astype(int).tolist()

    pax_days = set(pd.to_numeric(day_info["PAXDAYM"], errors="coerce").dropna().astype(int))
    full_days = set(pd.to_numeric(full["PAXDAYM"], errors="coerce").dropna().astype(int))
    scored = []
    for anchor in candidates:
        score = 0
        for date in dates:
            wear_day = anchor + int((date - first_date).days)
            if wear_day in pax_days:
                score += 1
            if wear_day in full_days:
                score += 2
        scored.append((score, -anchor, anchor))
    anchor = max(scored)[2]
    status = "weekday_and_sequence" if anchor in candidates else "sequence_only"
    return anchor, status


def add_nhanes_columns(
    minute: pd.DataFrame,
    pax: pd.DataFrame,
    wear_day: int,
    alignment_status: str,
) -> pd.DataFrame:
    pax_day = pax.loc[pd.to_numeric(pax["PAXDAYM"], errors="coerce").eq(wear_day)].copy()
    if pax_day.empty:
        minute["nhanes_match"] = False
        minute["nhanes_wear_day"] = wear_day
        minute["nhanes_alignment_status"] = "wear_day_missing"
        return minute

    rename = {
        "PAXDAYM": "nhanes_wear_day",
        "PAXDAYWM": "nhanes_weekday",
        "PAXSSNMP": "nhanes_start_sample",
        "PAXTSM": "nhanes_seconds_with_data",
        "PAXAISMM": "nhanes_idle_sleep_samples",
        "PAXMTSM": "nhanes_mims_triaxial",
        "PAXMXM": "nhanes_mims_x",
        "PAXMYM": "nhanes_mims_y",
        "PAXMZM": "nhanes_mims_z",
        "PAXPREDM": "nhanes_state_raw",
        "PAXTRANM": "nhanes_transition",
        "PAXLXMM": "nhanes_lux_mean",
        "PAXLXSDM": "nhanes_lux_sd",
        "PAXQFM": "nhanes_quality_flag_count",
        "PAXFLGSM": "nhanes_quality_flags",
    }
    keep = ["nhanes_minute_index"] + [column for column in rename if column in pax_day]
    pax_day = pax_day[keep].rename(columns=rename)
    pax_day["nhanes_state"] = (
        pd.to_numeric(pax_day["nhanes_state_raw"], errors="coerce")
        .map(NHANES_STATE_MAP)
        .fillna(-1)
        .astype("int8")
    )
    pax_day["nhanes_state_name"] = pax_day["nhanes_state"].map(NHANES_STATE_NAMES).fillna("missing")
    pax_day["nhanes_match"] = True
    pax_day["nhanes_alignment_status"] = alignment_status
    out = minute.merge(
        pax_day,
        left_on="minute_index",
        right_on="nhanes_minute_index",
        how="left",
        validate="one_to_one",
    ).drop(columns=["nhanes_minute_index"], errors="ignore")
    out["swan_nhanes_comparable"] = (
        out["swan_state_1min"].isin([0, 1, 2])
        & out["nhanes_state"].isin([0, 1, 2])
        & ~out["nhanes_transition"].fillna(False).astype(bool)
    )
    out["swan_nhanes_state_agree"] = (
        out["swan_nhanes_comparable"]
        & out["swan_state_1min"].eq(out["nhanes_state"])
    )
    return out


def summarize(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    cache_root = Path(args.paxmin_cache_root)
    sources = sorted(
        input_root.glob(
            f"model={args.model}/Dataset={args.dataset}/ID=*/Day=*/part-0.parquet"
        )
    )
    if args.participant_id:
        wanted = {str(value) for value in args.participant_id}
        sources = [path for path in sources if parse_parts(path)["ID"] in wanted]
    if args.limit_days:
        sources = sources[: args.limit_days]
    if not sources:
        raise SystemExit("No matching 10-second participant-day files found")

    days_by_participant: dict[str, list[Path]] = {}
    for source in sources:
        days_by_participant.setdefault(parse_parts(source)["ID"], []).append(source)

    written = skipped = failed = 0
    for participant_id, participant_sources in days_by_participant.items():
        participant_sources = sorted(participant_sources)
        pax_file = cache_path(cache_root, args.dataset, participant_id)
        pax = pd.read_parquet(pax_file) if pax_file.exists() else None
        if pax is None and args.require_nhanes:
            logger.warning("No cached PAXMIN for ID=%s; skipping", participant_id)
            continue

        available_days = [parse_parts(path)["Day"] for path in participant_sources]
        anchor, alignment_status = (
            choose_pax_anchor(pax, available_days) if pax is not None else (None, "paxmin_not_cached")
        )
        first_date = pd.Timestamp(min(available_days))

        for source in participant_sources:
            parts = parse_parts(source)
            destination = output_path(output_root, parts)
            newest_input = source.stat().st_mtime
            if pax_file.exists():
                newest_input = max(newest_input, pax_file.stat().st_mtime)
            if (
                destination.exists()
                and not args.overwrite
                and destination.stat().st_mtime >= newest_input
            ):
                skipped += 1
                continue
            try:
                minute = aggregate_10s(source)
                minute.insert(0, "day", parts["Day"])
                if pax is not None and anchor is not None:
                    wear_day = anchor + int((pd.Timestamp(parts["Day"]) - first_date).days)
                    minute = add_nhanes_columns(minute, pax, wear_day, alignment_status)
                else:
                    minute["nhanes_match"] = False
                    minute["nhanes_alignment_status"] = alignment_status
                atomic_write_parquet(minute, destination)
                logger.info(
                    "Wrote 1-minute data ID=%s Day=%s rows=%d NHANES_matches=%d",
                    participant_id,
                    parts["Day"],
                    len(minute),
                    int(minute["nhanes_match"].fillna(False).sum()),
                )
                written += 1
            except Exception as exc:
                logger.warning("Could not summarize %s: %s", source, exc)
                failed += 1

    logger.info("Complete: written=%d skipped=%d failed=%d", written, skipped, failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache-paxmin", help="Partition a PAXMIN XPT by participant")
    cache.add_argument("--xpt", required=True)
    cache.add_argument("--dataset", required=True, choices=("2011-2012", "2013-2014"))
    cache.add_argument(
        "--cache-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/paxmin_cache",
    )
    cache.add_argument("--epoch-10s-root", default=None)
    cache.add_argument("--chunk-rows", type=int, default=250_000)
    cache.add_argument("--overwrite", action="store_true")
    cache.set_defaults(func=cache_paxmin)

    summary = subparsers.add_parser("summarize", help="Build incremental 1-minute data")
    summary.add_argument(
        "--input-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_10s",
    )
    summary.add_argument(
        "--output-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/epoch_1min",
    )
    summary.add_argument(
        "--paxmin-cache-root",
        default="/work/pi_jstauden_umass_edu/SBpaper_outputs_10s/paxmin_cache",
    )
    summary.add_argument("--model", default="CHAP")
    summary.add_argument("--dataset", required=True, choices=("2011-2012", "2013-2014"))
    summary.add_argument("--participant-id", nargs="*", default=None)
    summary.add_argument("--limit-days", type=int, default=None)
    summary.add_argument("--require-nhanes", action="store_true")
    summary.add_argument("--overwrite", action="store_true")
    summary.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
