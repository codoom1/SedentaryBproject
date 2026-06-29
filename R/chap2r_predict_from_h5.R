# CHAP2R prediction script for preprocessed .h5 day files.
#
# Usage example:
# Rscript R/chap2r_predict_from_h5.R \
#   --preprocessed-dir data/preprocessed \
#   --subject-id 62596 \
#   --model-name CHAP_ALL_ADULTS \
#   --padding wrap \
#   --down-sample-frequency 10
#
# Optional flags:
#   --segment
#   --output-label
#
# Notes:
# - --checkpoint is optional; if omitted, the default for --model-name is used.
# - --output-file can be a directory (recommended). The script writes one CSV per day.

library(torch)

# Source utility helpers (this file sources chap2r_model.R internally).
source('R/chap2r_utils.R')





run_predictions_h5 <- function(
  preprocessed_dir,
  subject_id,
  checkpoint_path = NULL,
  output_file = NULL,
  model_name = "CHAP_ALL_ADULTS",
  down_sample_frequency = 10L,
  padding = "wrap",
  include_segment = FALSE,
  include_label = FALSE,
  include_probability = FALSE,
  overwrite = TRUE,
  label_map = c("sitting", "not-sitting", "no-label")
) {
  if (is.null(checkpoint_path) || checkpoint_path == "") {
    checkpoint_path <- get_chap_model_checkpoint(model_name)
    message("Using default checkpoint path: ", checkpoint_path)
  }

  spec_rec <- get_chap_model_spec(model_name)
  spec <- spec_rec$specs
  bi_lstm_window_size <- spec$bi_lstm_window_size
  bi_lstm_win_size <- as.integer(60L %/% as.integer(down_sample_frequency) * bi_lstm_window_size)

  model <- build_chap_model(
    model_name = model_name,
    down_sample_frequency = down_sample_frequency,
    num_classes = 2L
  )

  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  model <- load_chap2r_weights(model, checkpoint_path, device = device)

  if (is.null(output_file) || output_file == "") {
    output_file <- file.path("data", "predictions", subject_id, model_name)
    message("Using default output directory: ", output_file)
  }

  output_dir <- if (grepl("\\.csv$", output_file, ignore.case = TRUE)) dirname(output_file) else output_file

  segments <- input_iterator_segments(preprocessed_dir, subject_id, train = FALSE)
  if (length(segments) == 0L) {
    stop("No valid wear segments were found in .h5 files.")
  }

  wrote_any <- FALSE
  written_days <- character(0)

  for (seg_id in seq_along(segments)) {
    seg <- segments[[seg_id]]

    y_out <- segment_predict(
      model = model,
      x = seg$x,
      bi_lstm_win_size = bi_lstm_win_size,
      padding = padding,
      device = device,
      return_probabilities = include_probability
    )

    if (include_probability) {
      y_pred <- y_out$prediction
      y_prob <- y_out$probability
    } else {
      y_pred <- y_out
      y_prob <- NULL
    }

    if (length(y_pred) == 0L) next

    n_pred <- length(y_pred)
    timestamps <- seg$timestamps[seq_len(n_pred)]
    labels <- if (include_label) seg$labels[seq_len(n_pred)] else NULL

    ts_posix <- as.POSIXct(timestamps, origin = "1970-01-01", tz = Sys.timezone())
    day_keys <- format(as.Date(ts_posix), "%Y-%m-%d")

    for (day_name in unique(day_keys)) {
      idx <- which(day_keys == day_name)
      day_labels <- if (include_label) labels[idx] else NULL
      first_write_for_day <- !day_name %in% written_days

      save_chap_day_predictions(
        timestamps = timestamps[idx],
        predictions = y_pred[idx],
        probabilities = if (include_probability) y_prob[idx] else NULL,
        output_dir = output_dir,
        day_name = day_name,
        labels = day_labels,
        include_label = include_label,
        segment_id = seg_id - 1L,
        include_segment = include_segment,
        include_probability = include_probability,
        overwrite = overwrite && first_write_for_day,
        label_map = label_map
      )
      written_days <- union(written_days, day_name)
      wrote_any <- TRUE
    }
  }

  if (!wrote_any) {
    stop("No valid wear segments were found in .h5 files.")
  }

  invisible(NULL)
}

parse_args <- function(argv) {
  out <- list(
    preprocessed_dir = NULL,
    subject_id = NULL,
    checkpoint = NULL,
    output_file = NULL,
    model_name = "CHAP_ALL_ADULTS",
    down_sample_frequency = 10L,
    padding = "wrap",
    segment = FALSE,
    output_label = FALSE,
    output_probability = TRUE,
    overwrite = TRUE
  )

  i <- 1L
  while (i <= length(argv)) {
    key <- argv[i]
    if (key %in% c("--segment")) {
      out$segment <- TRUE
      i <- i + 1L
      next
    }
    if (key %in% c("--output-label")) {
      out$output_label <- TRUE
      i <- i + 1L
      next
    }
    if (key %in% c("--output-probability")) {
      out$output_probability <- TRUE
      i <- i + 1L
      next
    }
    if (key %in% c("--append")) {
      out$overwrite <- FALSE
      i <- i + 1L
      next
    }
    if (key %in% c("--overwrite")) {
      out$overwrite <- TRUE
      i <- i + 1L
      next
    }

    if (i == length(argv)) stop("Missing value for argument: ", key)
    val <- argv[i + 1L]

    if (key == "--preprocessed-dir") out$preprocessed_dir <- val
    else if (key == "--subject-id") out$subject_id <- val
    else if (key == "--checkpoint") out$checkpoint <- val
    else if (key == "--output-file") out$output_file <- val
    else if (key == "--model-name") out$model_name <- val
    else if (key == "--down-sample-frequency") out$down_sample_frequency <- as.integer(val)
    else if (key == "--padding") out$padding <- val
    else stop("Unknown argument: ", key)

    i <- i + 2L
  }

  required <- c("preprocessed_dir", "subject_id", "checkpoint")
  if (is.null(out$checkpoint)) {
    required <- c("preprocessed_dir", "subject_id")
  }
  missing <- required[vapply(required, function(k) is.null(out[[k]]), logical(1))]
  if (length(missing) > 0L) {
    stop("Missing required arguments: ", paste(missing, collapse = ", "))
  }

  out
}

if (sys.nframe() == 0L) {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  run_predictions_h5(
    preprocessed_dir = args$preprocessed_dir,
    subject_id = args$subject_id,
    checkpoint_path = args$checkpoint,
    output_file = args$output_file,
    model_name = args$model_name,
    down_sample_frequency = args$down_sample_frequency,
    padding = args$padding,
    include_segment = args$segment,
    include_label = args$output_label,
    include_probability = args$output_probability,
    overwrite = args$overwrite
  )
}
