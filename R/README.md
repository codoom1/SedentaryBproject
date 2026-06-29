# CHAP1R Pipeline

This directory contains the R implementation of the CHAP1 / DeepPostures
CNN-BiLSTM posture model used by:

```text
scripts/posture_library/MSSE-2021
```

The filenames currently use `chap2r` for historical reasons, but this R path is
for the CHAP1/MSSE-2021 model family, not the later CHAP2 submitted models under
`scripts/CHAP2/SUBMIT_RESULT`.

## Active Files

The active prediction path is:

```text
R/chap2r_predict_from_h5.R
  -> R/chap2r_utils.R
      -> R/chap2r_model.R
```

`chap2r_predict_from_h5.R`
: Command-line entry point. Reads preprocessed participant H5 files, loads a
CHAP1R model, runs inference, and writes posture prediction CSVs.

`chap2r_utils.R`
: Model configuration, checkpoint lookup, H5 reading, wear-segment creation,
prediction padding, probability generation, and CSV writing.

`chap2r_model.R`
: R torch implementation of the CHAP1 CNN-BiLSTM architecture. It matches
`scripts/posture_library/MSSE-2021/model.py`.

`convert_pt_to_rtorch.R`
: Utility script for converting Python `.pth` checkpoints to R `.rds`
checkpoints. It is not needed for normal prediction if the `.rds` files already
exist.

`CHAP2R.R`
: Older standalone scratch/demo file. It duplicates model code and runs checks
at top level. It is not part of the active pipeline.

## Required Inputs

### Preprocessed H5 Directory

The R prediction script expects a participant-level directory under a
preprocessed root:

```text
data/preprocessed/
  62193/
    2000-01-07/
      2000-01-07.h5
    2000-01-08/
      2000-01-08.h5
```

This nested layout is produced by this project's wrapper scripts. The R reader
searches recursively, so it can find `.h5` files inside day folders.

Each day H5 file should contain these datasets:

```text
time       numeric vector, one Unix timestamp per 10-second window
data       numeric array, expected as [window, 100, 3]
sleeping   integer vector, 1 means sleep window
non_wear   integer vector, 1 means non-wear window
label      integer vector, -1 no label, 0 sitting, 1 not-sitting
```

The `data` array stores one 10-second accelerometer window per row:

```text
window count x 100 samples x 3 axes
```

At the default 10 Hz downsample frequency, `100` samples equals 10 seconds.

### Model Name

Supported CHAP1 model names:

```text
CHAP_A
CHAP_B
CHAP_C
CHAP_ALL_ADULTS
CHAP_CHILDREN
CHAP_AUSDIAB
```

The default and recommended model is:

```text
CHAP_ALL_ADULTS
```

Model settings:

| Model | `amp_factor` | `bi_lstm_window_size` in minutes | Default 10 Hz sequence length |
|---|---:|---:|---:|
| `CHAP_A` | 2 | 9 | 54 windows |
| `CHAP_B` | 4 | 9 | 54 windows |
| `CHAP_C` | 2 | 7 | 42 windows |
| `CHAP_ALL_ADULTS` | 2 | 7 | 42 windows |
| `CHAP_CHILDREN` | 4 | 3 | 18 windows |
| `CHAP_AUSDIAB` | 4 | 9 | 54 windows |

The sequence length is computed as:

```r
60 %/% down_sample_frequency * bi_lstm_window_size
```

With `down_sample_frequency = 10`, `CHAP_ALL_ADULTS` uses:

```text
60 / 10 * 7 = 42 10-second windows
```

### Checkpoint Files

By default, R loads `.rds` checkpoints from:

```text
scripts/posture_library/MSSE-2021/pre-trained-models-rtorch/
```

Example:

```text
scripts/posture_library/MSSE-2021/pre-trained-models-rtorch/CHAP_ALL_ADULTS.rds
```

These `.rds` files are converted versions of the Python `.pth` checkpoints in:

```text
scripts/posture_library/MSSE-2021/pre-trained-models-pt/
```

To regenerate them, use:

```bash
Rscript R/convert_pt_to_rtorch.R
```

That conversion script requires the `deepposture` conda environment through
`reticulate`, because it imports Python `torch`.

## Basic Prediction Command

Run CHAP1R predictions for one participant:

```bash
Rscript R/chap2r_predict_from_h5.R \
  --preprocessed-dir data/preprocessed \
  --subject-id 62193 \
  --model-name CHAP_ALL_ADULTS \
  --padding wrap \
  --down-sample-frequency 10 \
  --output-file data/predictions/62193/CHAP_ALL_ADULTS \
  --segment \
  --output-label \
  --output-probability \
  --overwrite
```

This writes one CSV per calendar day into:

```text
data/predictions/62193/CHAP_ALL_ADULTS/
```

Example output:

```text
data/predictions/62193/CHAP_ALL_ADULTS/2000-01-07.csv
data/predictions/62193/CHAP_ALL_ADULTS/2000-01-08.csv
```

## Command Arguments

`--preprocessed-dir`
: Root directory containing participant folders. For the project layout, use
`data/preprocessed`.

`--subject-id`
: Participant folder name under `--preprocessed-dir`, for example `62193`.

`--model-name`
: CHAP1 model name. Defaults to `CHAP_ALL_ADULTS`.

`--checkpoint`
: Optional checkpoint override. If omitted, the script uses the `.rds` checkpoint
for `--model-name` from `pre-trained-models-rtorch`.

`--output-file`
: Output directory or CSV path. Directory output is recommended. If omitted, the
script writes to `data/predictions/<subject-id>/<model-name>`.

`--down-sample-frequency`
: Frequency of the preprocessed windows. Default is `10`.

`--padding`
: How to handle a wear segment whose length is not divisible by the model's
BiLSTM sequence length.

Supported values:

```text
drop
zero
wrap
```

`drop`
: Drop the final incomplete sequence.

`zero`
: Pad the final incomplete sequence with zero-valued windows, then remove padded
predictions from the output.

`wrap`
: Reuse the final full sequence to cover the segment tail, then keep only the
tail predictions. This requires the segment to be at least one full sequence
long. This is the default.

`--segment`
: Include a zero-based `segment` column. Segments are contiguous awake/wear
regions separated by sleep or non-wear.

`--output-label`
: Include the H5 ground-truth label column.

`--output-probability`
: Include the sigmoid probability from the binary CHAP model. This defaults to
enabled in the current argument parser.

`--overwrite`
: Replace any existing day CSV on first write for that day.

`--append`
: Append to existing output files instead of overwriting.

## Output CSV Structure

With `--segment --output-label --output-probability`, output columns are:

```text
segment,timestamp,label,probability,prediction
```

`segment`
: Zero-based wear segment index.

`timestamp`
: Window start time formatted as `YYYY-MM-DD HH:MM:SS`.

`label`
: Human-readable H5 label. One of:

```text
no-label
sitting
not-sitting
```

`probability`
: `sigmoid(logit)` from the binary model. This is the probability used for the
threshold rule.

`prediction`
: `sitting` when probability is below `0.5`; `not-sitting` when probability is
at least `0.5`.

## Segmentation Rules

The R pipeline follows the CHAP1 Python prediction path:

1. Read all day H5 files for the subject in sorted order.
2. Build continuous wear segments.
3. Break a segment whenever:
   ```text
   sleeping == 1
   non_wear == 1
   ```
4. Keep labels as metadata during prediction.
5. Apply padding within each segment.
6. Run model inference in BiLSTM-sized chunks.
7. Split output rows back into day CSV files by timestamp.

This means a segment can continue across midnight if the original consecutive
day files have no sleep/non-wear break at the boundary. The output is still
written by calendar day.

## Python-vs-R Demo

Use the demo script to prove that CHAP1R matches CHAP1 Python on one H5 file:

```bash
/opt/homebrew/anaconda3/envs/deepposture/bin/python \
  scripts/demo_compare_chap1_python_r.py \
  --h5 data/preprocessed/62193/2000-01-07/2000-01-07.h5 \
  --model CHAP_ALL_ADULTS
```

Default behavior:

```text
model: CHAP_ALL_ADULTS
padding: wrap
n_windows: 84
```

`84` windows equals two full `CHAP_ALL_ADULTS` BiLSTM sequences
(`2 * 42`), so the demo runs quickly while still testing the sequence model.

The demo writes:

```text
data/tmp/chap1_python_r_demo/python_predictions.csv
data/tmp/chap1_python_r_demo/r_predictions.csv
data/tmp/chap1_python_r_demo/comparison_summary.json
```

Expected result:

```text
CHAP1 Python vs CHAP1R comparison
rows: 84 vs 84
timestamps match: True
labels match: True
predictions match: True
max probability diff: 5.551e-16
mean probability diff: 2.419e-16
PASS
```

To compare the whole H5 file:

```bash
/opt/homebrew/anaconda3/envs/deepposture/bin/python \
  scripts/demo_compare_chap1_python_r.py \
  --h5 data/preprocessed/62193/2000-01-07/2000-01-07.h5 \
  --model CHAP_ALL_ADULTS \
  --all
```

## Demo Script Arguments

`--h5`
: Required. One preprocessed day H5 file.

`--model`
: CHAP1 model name. Default is `CHAP_ALL_ADULTS`.

`--down-sample-frequency`
: Default is `10`.

`--padding`
: One of `drop`, `zero`, or `wrap`. Default is `wrap`.

`--n-windows`
: Number of valid windows to compare. Default is `84`.

`--all`
: Compare all valid awake/wear windows in the H5 file.

`--out-dir`
: Output folder for the Python CSV, R CSV, generated R helper, and JSON summary.

`--rscript`
: Rscript executable. Default is `Rscript`.

## Dependencies

R packages:

```r
torch
rhdf5
abind
```

Python packages for the demo:

```text
h5py
numpy
pandas
torch
```

The demo should be run in an environment that has Python torch and the CHAP1
dependencies installed. In this project that is usually:

```bash
conda run -n deepposture python scripts/demo_compare_chap1_python_r.py --help
```

## Notes And Caveats

- This R pipeline is currently validating CHAP1/MSSE-2021 behavior.
- CHAP2 submitted checkpoints under `scripts/CHAP2/SUBMIT_RESULT` are separate
and are not the target of this R pipeline yet.
- The `.rds` checkpoint conversion must preserve LSTM direction names correctly:
  R torch uses `l1` where Python torch uses `l0`; the active loader maps these
  names before loading weights.
- The R pipeline recursively reads nested participant/day H5 files. Python
  CHAP1's original `make_predictions.py` expects a different layout, so use
  `scripts/get_posture_predictions.py` or the demo script when comparing paths.
