#!/usr/bin/env python3
"""Compare CHAP1 Python and CHAP1R predictions for one preprocessed H5 file.

Example:
  python scripts/demo_compare_chap1_python_r.py \
    --h5 data/preprocessed/62193/2000-01-07/2000-01-07.h5 \
    --model CHAP_ALL_ADULTS
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAP1_ROOT = REPO_ROOT / "scripts" / "posture_library" / "MSSE-2021"

MODEL_SPECS = {
    "CHAP_A": {"amp_factor": 2, "bi_lstm_window_size": 9},
    "CHAP_B": {"amp_factor": 4, "bi_lstm_window_size": 9},
    "CHAP_C": {"amp_factor": 2, "bi_lstm_window_size": 7},
    "CHAP_ALL_ADULTS": {"amp_factor": 2, "bi_lstm_window_size": 7},
    "CHAP_CHILDREN": {"amp_factor": 4, "bi_lstm_window_size": 3},
    "CHAP_AUSDIAB": {"amp_factor": 4, "bi_lstm_window_size": 9},
}

LABEL_MAP = {-1: "no-label", 0: "sitting", 1: "not-sitting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CHAP1 Python and CHAP1R predictions on one preprocessed day H5."
    )
    parser.add_argument("--h5", required=True, help="Path to one CHAP1 preprocessed .h5 day file.")
    parser.add_argument("--model", default="CHAP_ALL_ADULTS", choices=sorted(MODEL_SPECS))
    parser.add_argument("--down-sample-frequency", type=int, default=10)
    parser.add_argument("--padding", choices=("drop", "zero", "wrap"), default="wrap")
    parser.add_argument(
        "--n-windows",
        type=int,
        default=84,
        help="Limit comparison to the first N valid 10-second windows. Use --all for the whole file.",
    )
    parser.add_argument("--all", action="store_true", help="Compare all valid windows in the H5 file.")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data" / "tmp" / "chap1_python_r_demo"),
        help="Directory for CSV outputs and comparison_summary.json.",
    )
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable to use.")
    return parser.parse_args()


def bi_lstm_win_size(model_name: str, down_sample_frequency: int) -> int:
    spec = MODEL_SPECS[model_name]
    return (60 // int(down_sample_frequency)) * int(spec["bi_lstm_window_size"])


def read_h5_day(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        time = h5["time"][:]
        data = h5["data"][:]
        if data.ndim == 3 and data.shape[0] != len(time) and data.shape[2] == len(time):
            data = np.transpose(data, (2, 1, 0))
        label = h5["label"][:] if "label" in h5 else np.full(len(time), -1, dtype=np.int64)
        sleeping = h5["sleeping"][:] if "sleeping" in h5 else np.zeros(len(time), dtype=np.int64)
        non_wear = h5["non_wear"][:] if "non_wear" in h5 else np.zeros(len(time), dtype=np.int64)
    return {
        "time": np.asarray(time),
        "data": np.asarray(data),
        "label": np.asarray(label, dtype=np.int64),
        "sleeping": np.asarray(sleeping, dtype=np.int64),
        "non_wear": np.asarray(non_wear, dtype=np.int64),
    }


def day_segments(day: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    segments = []
    data_batch = []
    time_batch = []
    label_batch = []

    def flush() -> None:
        nonlocal data_batch, time_batch, label_batch
        if time_batch:
            segments.append(
                (
                    np.asarray(data_batch, dtype=np.float32),
                    np.asarray(time_batch),
                    np.asarray(label_batch, dtype=np.int64),
                )
            )
        data_batch = []
        time_batch = []
        label_batch = []

    for x, t, s, nw, label in zip(
        day["data"], day["time"], day["sleeping"], day["non_wear"], day["label"]
    ):
        if int(s) == 1 or int(nw) == 1:
            flush()
            continue
        data_batch.append(x)
        time_batch.append(t)
        label_batch.append(label)

    flush()
    return segments


def apply_limit(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_windows: int | None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if n_windows is None:
        return segments
    remaining = int(n_windows)
    limited = []
    for x, timestamps, labels in segments:
        if remaining <= 0:
            break
        take = min(remaining, len(timestamps))
        limited.append((x[:take], timestamps[:take], labels[:take]))
        remaining -= take
    return limited


def pad_segment(
    x: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    win_size: int,
    padding: str,
    down_sample_frequency: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, bool]:
    border = x.shape[0] % win_size
    wrapped = False
    zeroed = False
    deficit = 0
    if border == 0:
        return x, timestamps, labels, deficit, wrapped, zeroed

    if padding == "drop":
        return x[:-border], timestamps[:-border], labels[:-border], deficit, wrapped, zeroed

    deficit = win_size - border
    labels_padded = np.full(deficit, -1, dtype=np.int64)

    if padding == "zero":
        x_padded = np.zeros((deficit,) + x.shape[1:], dtype=x.dtype)
        increment = int(down_sample_frequency)
        timestamps_padded = np.full(deficit, timestamps[-1]) + np.array(
            [increment * (i + 1) for i in range(deficit)]
        )
        return (
            np.vstack((x, x_padded)),
            np.hstack((timestamps, timestamps_padded)),
            np.hstack((labels, labels_padded)),
            deficit,
            wrapped,
            True,
        )

    if x.shape[0] < win_size:
        raise ValueError("Cannot use wrap padding when segment length is shorter than bi_lstm_win_size.")
    x_last_p1 = x[:-border]
    x_last_p2 = x[-win_size:]
    return (
        np.vstack((x_last_p1, x_last_p2)),
        timestamps,
        np.hstack((labels, labels_padded)),
        deficit,
        True,
        zeroed,
    )


def load_python_model(model_name: str, win_size: int):
    sys.path.insert(0, str(CHAP1_ROOT))
    from model import CNNBiLSTMModel  # pylint: disable=import-error,import-outside-toplevel
    from utils import load_model_weights  # pylint: disable=import-error,import-outside-toplevel

    spec = MODEL_SPECS[model_name]
    model = CNNBiLSTMModel(
        amp_factor=int(spec["amp_factor"]),
        bi_lstm_win_size=win_size,
        num_classes=2,
    )
    checkpoint = CHAP1_ROOT / "pre-trained-models-pt" / f"{model_name}.pth"
    load_model_weights(model, str(checkpoint), weights_only=True)
    model.eval()
    return model


def predict_python(
    h5_path: Path,
    model_name: str,
    down_sample_frequency: int,
    padding: str,
    n_windows: int | None,
    output_csv: Path,
) -> pd.DataFrame:
    win_size = bi_lstm_win_size(model_name, down_sample_frequency)
    model = load_python_model(model_name, win_size)
    segments = apply_limit(day_segments(read_h5_day(h5_path)), n_windows)

    rows = []
    for segment_id, (x, timestamps, labels) in enumerate(segments):
        x, timestamps, labels, deficit, wrapped, zeroed = pad_segment(
            x, timestamps, labels, win_size, padding, down_sample_frequency
        )
        if x.shape[0] == 0:
            continue

        n_seq = x.shape[0] // win_size
        if n_seq == 0:
            continue
        chunks = np.split(x.squeeze(), n_seq)

        probs = []
        preds = []
        with torch.no_grad():
            for chunk_start in range(0, len(chunks), 16):
                batch = np.asarray(chunks[chunk_start : chunk_start + 16], dtype=np.float32)
                inputs = torch.from_numpy(batch).view(-1, 100, 3, 1).permute(0, 3, 1, 2)
                prob = torch.sigmoid(model(inputs))
                probs.extend(prob.detach().cpu().numpy().flatten().tolist())
                preds.extend(torch.round(prob).detach().cpu().numpy().flatten().astype(int).tolist())

        if padding == "wrap" and wrapped:
            probs = probs[:-win_size] + probs[-(x.shape[0] % win_size or win_size) :]
            preds = preds[:-win_size] + preds[-(x.shape[0] % win_size or win_size) :]
        elif padding == "zero" and zeroed and deficit > 0:
            probs = probs[:-deficit]
            preds = preds[:-deficit]
            timestamps = timestamps[:-deficit]
            labels = labels[:-deficit]

        for t, label, prob, pred in zip(timestamps, labels, probs, preds):
            rows.append(
                {
                    "segment": segment_id,
                    "timestamp": datetime.fromtimestamp(float(t)).strftime("%Y-%m-%d %H:%M:%S"),
                    "label": LABEL_MAP[int(label)],
                    "probability": float(prob),
                    "prediction": LABEL_MAP[int(pred)],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def write_r_helper(script_path: Path) -> None:
    script_path.write_text(
        r'''
args <- commandArgs(trailingOnly = TRUE)
params <- list()
for (i in seq(1, length(args), by = 2)) {
  params[[sub("^--", "", args[[i]])]] <- args[[i + 1]]
}

source("R/chap2r_utils.R")

label_text <- function(values) {
  ifelse(values == -1L, "no-label", ifelse(values == 0L, "sitting", "not-sitting"))
}

model_name <- params[["model"]]
down_sample_frequency <- as.integer(params[["down_sample_frequency"]])
bi_lstm_win_size <- as.integer(
  60L %/% down_sample_frequency * get_chap_model_spec(model_name)$specs$bi_lstm_window_size
)

model <- build_chap_model(model_name, down_sample_frequency = down_sample_frequency)
model <- load_chap2r_weights(model, get_chap_model_checkpoint(model_name))

day <- read_day_h5(params[["h5"]])
segments <- input_iterator_segments_from_day(day, train = FALSE)
n_windows <- as.integer(params[["n_windows"]])
if (n_windows < 0L) n_windows <- NA_integer_

rows <- list()
remaining <- n_windows
for (seg_id in seq_along(segments)) {
  seg <- segments[[seg_id]]
  if (!is.na(remaining)) {
    if (remaining <= 0L) break
    take <- min(remaining, length(seg$timestamps))
    seg$x <- seg$x[seq_len(take), , , drop = FALSE]
    seg$timestamps <- seg$timestamps[seq_len(take)]
    seg$labels <- seg$labels[seq_len(take)]
    remaining <- remaining - take
  }

  y <- segment_predict(
    model = model,
    x = seg$x,
    bi_lstm_win_size = bi_lstm_win_size,
    padding = params[["padding"]],
    return_probabilities = TRUE,
    batch_n_seq = 16L
  )
  if (length(y$prediction) == 0L) next

  n_out <- length(y$prediction)
  ts_fmt <- format(
    as.POSIXct(seg$timestamps[seq_len(n_out)], origin = "1970-01-01", tz = Sys.timezone()),
    "%Y-%m-%d %H:%M:%S"
  )
  rows[[length(rows) + 1L]] <- data.frame(
    segment = seg_id - 1L,
    timestamp = ts_fmt,
    label = label_text(as.integer(seg$labels[seq_len(n_out)])),
    probability = as.numeric(y$probability),
    prediction = label_text(as.integer(y$prediction)),
    stringsAsFactors = FALSE
  )
}

out <- if (length(rows) == 0L) {
  data.frame(segment = integer(), timestamp = character(), label = character(),
             probability = numeric(), prediction = character())
} else {
  do.call(rbind, rows)
}
utils::write.csv(out, params[["output"]], row.names = FALSE, quote = TRUE)
'''.lstrip(),
        encoding="utf-8",
    )


def predict_r(
    h5_path: Path,
    model_name: str,
    down_sample_frequency: int,
    padding: str,
    n_windows: int | None,
    output_csv: Path,
    rscript: str,
    out_dir: Path,
) -> pd.DataFrame:
    helper = out_dir / "run_chap1r_demo.R"
    write_r_helper(helper)
    cmd = [
        rscript,
        str(helper),
        "--h5",
        str(h5_path),
        "--model",
        model_name,
        "--down_sample_frequency",
        str(down_sample_frequency),
        "--padding",
        padding,
        "--n_windows",
        str(n_windows if n_windows is not None else -1),
        "--output",
        str(output_csv),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return pd.read_csv(output_csv)


def compare_outputs(python_df: pd.DataFrame, r_df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    row_count_match = len(python_df) == len(r_df)
    n = min(len(python_df), len(r_df))
    p = python_df.iloc[:n].reset_index(drop=True)
    r = r_df.iloc[:n].reset_index(drop=True)

    probability_diff = (p["probability"] - r["probability"]).abs() if n else pd.Series(dtype=float)
    summary = {
        "python_rows": int(len(python_df)),
        "r_rows": int(len(r_df)),
        "row_count_match": bool(row_count_match),
        "timestamps_match": bool(p["timestamp"].equals(r["timestamp"])) if n else row_count_match,
        "labels_match": bool(p["label"].equals(r["label"])) if n else row_count_match,
        "predictions_match": bool(p["prediction"].equals(r["prediction"])) if n else row_count_match,
        "max_abs_probability_diff": float(probability_diff.max()) if n else 0.0,
        "mean_abs_probability_diff": float(probability_diff.mean()) if n else 0.0,
    }
    summary["pass"] = bool(
        summary["row_count_match"]
        and summary["timestamps_match"]
        and summary["labels_match"]
        and summary["predictions_match"]
        and summary["max_abs_probability_diff"] < 1e-6
    )
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows = None if args.all else args.n_windows
    python_csv = out_dir / "python_predictions.csv"
    r_csv = out_dir / "r_predictions.csv"

    print(f"[demo] H5: {h5_path}")
    print(f"[demo] model={args.model} padding={args.padding} n_windows={n_windows or 'all'}")
    python_df = predict_python(
        h5_path,
        args.model,
        args.down_sample_frequency,
        args.padding,
        n_windows,
        python_csv,
    )
    r_df = predict_r(
        h5_path,
        args.model,
        args.down_sample_frequency,
        args.padding,
        n_windows,
        r_csv,
        args.rscript,
        out_dir,
    )
    summary = compare_outputs(python_df, r_df, out_dir)

    print("\nCHAP1 Python vs CHAP1R comparison")
    print(f"rows: {summary['python_rows']} vs {summary['r_rows']}")
    print(f"timestamps match: {summary['timestamps_match']}")
    print(f"labels match: {summary['labels_match']}")
    print(f"predictions match: {summary['predictions_match']}")
    print(f"max probability diff: {summary['max_abs_probability_diff']:.3e}")
    print(f"mean probability diff: {summary['mean_abs_probability_diff']:.3e}")
    print(f"outputs: {out_dir}")
    print("PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#python scripts/demo_compare_chap1_python_r.py --h5 data/preprocessed/62193/2000-01-07/2000-01-07.h5 --model CHAP_ALL_ADULTS
