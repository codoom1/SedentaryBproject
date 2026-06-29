#!/usr/bin/env python3
"""Export CHAP and SWaN predictions as compact participant-day 10-second Parquet files."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SLEEP_STATE_CODES = {"WEAR": 0, "SLEEP": 1, "NON-WEAR": 2}


def normalize_dataset(dataset: str) -> str:
    return {"2011-12": "2011-2012", "2013-14": "2013-2014"}.get(dataset, dataset)


def normalize_sleep_state(series: pd.Series) -> pd.Series:
    state = series.astype(str).str.strip().str.upper()
    state = state.str.replace("_", "-", regex=False).str.replace(" ", "-", regex=False)
    return state.replace(
        {
            "NONWEAR": "NON-WEAR",
            "NWEAR": "NON-WEAR",
            "SLEEPING": "SLEEP",
            "WAKE": "WEAR",
            "WAKING": "WEAR",
        }
    )


def read_posture_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in df.columns else "time" if "time" in df.columns else None
    if timestamp_col is None or "prediction" not in df.columns:
        raise ValueError(f"Posture file must contain timestamp/time and prediction columns: {path}")

    timestamp = pd.to_datetime(df[timestamp_col], errors="coerce").dt.round("10s")
    prediction = df["prediction"].astype(str).str.strip().str.lower().replace(
        {
            "sit": "sitting",
            "seated": "sitting",
            "not sitting": "not-sitting",
            "not_sitting": "not-sitting",
            "notsitting": "not-sitting",
            "non-sitting": "not-sitting",
            "non_sitting": "not-sitting",
            "standing": "not-sitting",
        }
    )
    out = pd.DataFrame({"timestamp": timestamp, "sitting": prediction.eq("sitting")})
    if "prob_sitting" in df.columns:
        out["prob_sitting"] = pd.to_numeric(df["prob_sitting"], errors="coerce")
    elif "probability" in df.columns:
        # Backward compatibility for older wrist CSVs where probability meant
        # confidence in the predicted class rather than P(sitting).
        confidence = pd.to_numeric(df["probability"], errors="coerce")
        out["prob_sitting"] = np.where(out["sitting"], confidence, 1.0 - confidence)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    return out.drop_duplicates(subset=["timestamp"], keep="first")


def read_sleep_day(path: Path, include_probabilities: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "START_TIME" not in df.columns:
        raise ValueError(f"Sleep file must contain START_TIME: {path}")

    if "STATE" in df.columns:
        state = normalize_sleep_state(df["STATE"])
    elif "PREDICTED" in df.columns:
        state = pd.to_numeric(df["PREDICTED"], errors="coerce").map(
            {0: "WEAR", 1: "SLEEP", 2: "NON-WEAR"}
        )
    else:
        raise ValueError(f"Sleep file must contain STATE or PREDICTED: {path}")

    base = pd.DataFrame(
        {
            "sleep_start": pd.to_datetime(df["START_TIME"], errors="coerce").dt.round("10s"),
            "sleep_state": state.map(SLEEP_STATE_CODES),
        }
    )
    probability_columns = {
        "PROB_WEAR": "prob_wear",
        "PROB_SLEEP": "prob_sleep",
        "PROB_NWEAR": "prob_nonwear",
    }
    if include_probabilities:
        for source, target in probability_columns.items():
            base[target] = pd.to_numeric(df[source], errors="coerce") if source in df.columns else np.nan

    base = base.dropna(subset=["sleep_start", "sleep_state"]).sort_values("sleep_start")
    base = base.drop_duplicates(subset=["sleep_start"], keep="first")

    expanded = base.loc[base.index.repeat(3)].copy()
    expanded["offset_seconds"] = expanded.groupby(level=0).cumcount() * 10
    expanded["timestamp"] = expanded["sleep_start"] + pd.to_timedelta(
        expanded["offset_seconds"], unit="s"
    )
    keep = ["timestamp", "sleep_state"]
    if include_probabilities:
        keep += list(probability_columns.values())
    return expanded[keep].reset_index(drop=True)


def output_path(output_root: Path, model: str, dataset: str, participant_id: str, day: str) -> Path:
    return (
        output_root
        / f"model={model}"
        / f"Dataset={normalize_dataset(dataset)}"
        / f"ID={participant_id}"
        / f"Day={day}"
        / "part-0.parquet"
    )


def export_participant(
    participant_id: str,
    dataset: str,
    model: str,
    output_root: Path,
    data_root: Path,
    skip_sleep: bool = False,
    include_sleep_probabilities: bool = False,
    overwrite: bool = False,
) -> int:
    posture_dir = data_root / "predictions" / participant_id / model
    sleep_dir = data_root / "sleep_predictions" / participant_id / "predictions"
    if not posture_dir.exists():
        raise FileNotFoundError(f"Posture prediction directory not found: {posture_dir}")
    if not skip_sleep and not sleep_dir.exists():
        raise FileNotFoundError(f"Sleep prediction directory not found: {sleep_dir}")

    posture_files = sorted(posture_dir.glob("*.csv"))
    if not posture_files:
        raise FileNotFoundError(f"No posture day files found: {posture_dir}")

    written = 0
    available = 0
    for posture_file in posture_files:
        day = posture_file.stem
        destination = output_path(output_root, model, dataset, participant_id, day)
        if destination.exists() and not overwrite:
            logger.info("Already exists; skipping: %s", destination)
            available += 1
            continue

        posture = read_posture_day(posture_file)
        if posture.empty:
            logger.warning("No valid posture rows; skipping %s", posture_file)
            continue

        if skip_sleep:
            epochs = posture.copy()
            epochs["sleep_state"] = pd.Series(0, index=epochs.index, dtype="int8")
        else:
            sleep_file = sleep_dir / f"{day}_sleep_predictions.csv"
            if not sleep_file.exists():
                logger.warning("No matching sleep file; skipping day %s", day)
                continue
            sleep = read_sleep_day(sleep_file, include_sleep_probabilities)
            epochs = posture.merge(sleep, on="timestamp", how="left", validate="one_to_one")

        day_start = epochs["timestamp"].dt.normalize()
        epoch_index = ((epochs["timestamp"] - day_start).dt.total_seconds() // 10)
        epochs["epoch_index"] = epoch_index.astype("int16")
        epochs["sleep_state"] = epochs["sleep_state"].fillna(-1).astype("int8")
        epochs["valid_sleep"] = epochs["sleep_state"].ge(0)
        epochs["waking"] = epochs["sleep_state"].eq(SLEEP_STATE_CODES["WEAR"])
        epochs["sleep_nonwear"] = epochs["sleep_state"].isin(
            [SLEEP_STATE_CODES["SLEEP"], SLEEP_STATE_CODES["NON-WEAR"]]
        )
        epochs["sitting"] = epochs["sitting"].astype("bool")

        columns = [
            "epoch_index",
            "sitting",
            "sleep_state",
            "valid_sleep",
            "waking",
            "sleep_nonwear",
        ]
        if "prob_sitting" in epochs.columns:
            epochs["prob_sitting"] = pd.to_numeric(
                epochs["prob_sitting"], errors="coerce"
            ).astype("float32")
            columns.append("prob_sitting")
        if include_sleep_probabilities and not skip_sleep:
            for column in ["prob_wear", "prob_sleep", "prob_nonwear"]:
                epochs[column] = pd.to_numeric(epochs[column], errors="coerce").astype("float32")
                columns.append(column)

        epochs = epochs[columns].sort_values("epoch_index").reset_index(drop=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        epochs.to_parquet(destination, engine="pyarrow", compression="zstd", index=False)
        logger.info(
            "Wrote %s rows to %s (%s missing SWaN matches)",
            len(epochs),
            destination,
            int((~epochs["valid_sleep"]).sum()),
        )
        written += 1
        available += 1

    if written == 0:
        logger.warning("No new 10-second Parquet files written for participant %s", participant_id)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--dataset", required=True, help="NHANES cycle, e.g. 2011-12 or 2013-14")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", default="data/epoch_10s")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--skip-sleep", action="store_true", help="Export posture-only epochs as waking")
    parser.add_argument("--include-sleep-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for 10-second Parquet export. Install it in the environment "
            "running this script, for example: conda install -n deepposture -c conda-forge pyarrow"
        ) from exc

    written = export_participant(
        participant_id=args.participant_id,
        dataset=args.dataset,
        model=args.model,
        output_root=Path(args.output_root),
        data_root=Path(args.data_root),
        skip_sleep=args.skip_sleep,
        include_sleep_probabilities=args.include_sleep_probabilities,
        overwrite=args.overwrite,
    )
    raise SystemExit(0 if written > 0 else 1)


if __name__ == "__main__":
    main()
