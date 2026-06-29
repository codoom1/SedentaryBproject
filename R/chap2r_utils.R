# Helper functions for the CHAP model in R

library(torch)
library(rhdf5)

source('R/chap2r_model.R')

## Function to retrieve model specifications based on model name.
get_chap_model_spec <- function(model_name) {
  specs <- list(
    CHAP_A = list(amp_factor = 2L, bi_lstm_window_size = 9L),
    CHAP_B = list(amp_factor = 4L, bi_lstm_window_size = 9L),
    CHAP_C = list(amp_factor = 2L, bi_lstm_window_size = 7L),
    CHAP_ALL_ADULTS = list(amp_factor = 2L, bi_lstm_window_size = 7L),
    CHAP_CHILDREN = list(amp_factor = 4L, bi_lstm_window_size = 3L),
    CHAP_AUSDIAB = list(amp_factor = 4L, bi_lstm_window_size = 9L)
  )
  if (!model_name %in% names(specs)) {
    stop("Unknown model_name: ", model_name)
  }

  return(list(specs = specs[[model_name]], model_name = model_name))
}

## Function to get pretrained checkpoint path for a CHAP model name.
get_chap_model_checkpoint <- function(model_name) {
  spec_rec <- get_chap_model_spec(model_name)
  file.path(
    "scripts",
    "posture_library",
    "MSSE-2021",
    "pre-trained-models-rtorch",
    paste0(spec_rec$model_name, ".rds")
  )
}

## Function to build a CHAP model from a model name and runtime settings.
build_chap_model <- function(model_name, down_sample_frequency = 10L, num_classes = 2L) {
  spec_rec <- get_chap_model_spec(model_name)
  spec <- spec_rec$specs
  resolved_model_name <- spec_rec$model_name

  if (!exists("CNNBiLSTMModel", mode = "function")) {
    stop("CNNBiLSTMModel is not available. Source R/chap2r_model.R before calling build_chap_model().")
  }

  if (!identical(resolved_model_name, model_name)) {
    stop("Resolved model name does not match requested model name.")
  }

  bi_lstm_win_size <- as.integer(60L %/% as.integer(down_sample_frequency) * spec$bi_lstm_window_size)
  CNNBiLSTMModel(
    amp_factor = spec$amp_factor,
    bi_lstm_win_size = bi_lstm_win_size,
    num_classes = num_classes
  )
}



## Example tests (optional):
# print(get_chap_model_spec("CHAP_ALL_ADULTS"))
# model <- build_chap_model("CHAP_ALL_ADULTS")
# print(model)



read_h5_dataset <- function(h5_file, dataset_name) {
  if (requireNamespace("rhdf5", quietly = TRUE)) {
    return(rhdf5::h5read(h5_file, dataset_name))
  }
  if (requireNamespace("hdf5r", quietly = TRUE)) {
    h5 <- hdf5r::H5File$new(h5_file, mode = "r")
    on.exit(h5$close_all(), add = TRUE)
    return(h5[[dataset_name]]$read())
  }
  stop("Install either 'rhdf5' or 'hdf5r' to read .h5 files.")
}

read_day_h5 <- function(h5_file) {
  time <- as.numeric(read_h5_dataset(h5_file, "time"))
  data <- read_h5_dataset(h5_file, "data")
  data_dim <- dim(data)

  if (length(data_dim) == 3L && data_dim[1] != length(time) && data_dim[3] == length(time)) {
    data <- aperm(data, c(3, 2, 1))
  }

  list(
    time = time,
    data = data,
    sleeping = as.integer(read_h5_dataset(h5_file, "sleeping")),
    non_wear = as.integer(read_h5_dataset(h5_file, "non_wear")),
    label = as.integer(read_h5_dataset(h5_file, "label"))
  )
}

list_subject_h5_files <- function(preprocessed_dir, subject_id) {
  # Discover all day-level .h5 files for a subject in deterministic order.
  subject_dir <- file.path(preprocessed_dir, subject_id)
  print(paste("Looking for .h5 files under:", subject_dir))
  if (!dir.exists(subject_dir)) {
    stop("Subject directory not found: ", subject_dir)
  }

  files <- list.files(subject_dir, pattern = "\\.h5$", full.names = TRUE, recursive = TRUE)
  if (length(files) == 0L) {
    print(paste("No .h5 files found under:", subject_dir))
    stop("No .h5 files found under: ", subject_dir)
  }

  files <- sort(files)
  print(paste("Found .h5 files:", length(files)))
  files
}

input_iterator_segments_from_day <- function(day, train = FALSE) {
  # Build contiguous wear segments within a single day payload.
  # This keeps the same segmentation rules used by input_iterator_segments,
  # but avoids holding all subject files in memory at once.
  segments <- list()
  data_batch <- list()
  timestamps_batch <- numeric(0)
  label_batch <- integer(0)

  flush_segment <- function() {
    if (length(timestamps_batch) > 0L) {
      x_arr <- simplify2array(data_batch)
      print(paste("Shape of x_arr:", paste(dim(x_arr), collapse = "x")))

      if (length(dim(x_arr)) == 2L) {
        x_arr <- array(x_arr, dim = c(1L, dim(x_arr)[1], dim(x_arr)[2]))
      } else if (length(dim(x_arr)) == 3L) {
        x_arr <- aperm(x_arr, c(3, 1, 2))
      }

      segments[[length(segments) + 1L]] <<- list(
        x = x_arr,
        timestamps = timestamps_batch,
        labels = label_batch
      )
    }

    data_batch <<- list()
    timestamps_batch <<- numeric(0)
    label_batch <<- integer(0)
  }

  n <- length(day$time)
  for (i in seq_len(n)) {
    s <- day$sleeping[i]
    nw <- day$non_wear[i]
    l <- day$label[i]

    # Break segments at sleep/non-wear; in train mode also break at missing labels.
    if (s == 1L || nw == 1L || (train && l == -1L)) {
      flush_segment()
      next
    }

    data_batch[[length(data_batch) + 1L]] <- day$data[i, , ]
    timestamps_batch <- c(timestamps_batch, day$time[i])
    label_batch <- c(label_batch, l)
  }

  flush_segment()
  segments
}

## Example test for day-by-day helper flow (optional):
# preprocessed_dir <- "data/preprocessed"
# subject_id <- "62596"
# files <- list_subject_h5_files(preprocessed_dir, subject_id)
# day <- read_day_h5(files[[1]])
# day_segments <- input_iterator_segments_from_day(day, train = FALSE)
# print(length(day_segments))
# if (length(day_segments) > 0L) {
#   print(dim(day_segments[[1]]$x))
#   print(head(day_segments[[1]]$timestamps))
# }


input_iterator_segments <- function(preprocessed_dir, subject_id, train = FALSE) {
  subject_dir <- file.path(preprocessed_dir, subject_id)
  print(paste("Looking for .h5 files under:", subject_dir))
  if (!dir.exists(subject_dir)) {
    stop("Subject directory not found: ", subject_dir)
  }

  files <- list.files(subject_dir, pattern = "\\.h5$", full.names = TRUE, recursive = TRUE)
  if (length(files) == 0L) {
    print(paste("No .h5 files found under: ", subject_dir))
    stop("No .h5 files found under: ", subject_dir)
  }
  print(paste("Found .h5 files:", length(files)))
  files <- sort(files)

  segments <- list()
  data_batch <- list()
  timestamps_batch <- numeric(0)
  label_batch <- integer(0)

  flush_segment <- function() {
    if (length(timestamps_batch) > 0L) {
      x_arr <- simplify2array(data_batch)
      ## Show one example of the shape of x_arr for debugging
      print(paste("Shape of x_arr:", paste(dim(x_arr), collapse = "x")))

      if (length(dim(x_arr)) == 2L) {
        x_arr <- array(x_arr, dim = c(1L, dim(x_arr)[1], dim(x_arr)[2]))
      } else if (length(dim(x_arr)) == 3L) {
        x_arr <- aperm(x_arr, c(3, 1, 2))
      }

      segments[[length(segments) + 1L]] <<- list(
        x = x_arr,
        timestamps = timestamps_batch,
        labels = label_batch
      )
    }
    data_batch <<- list()
    timestamps_batch <<- numeric(0)
    label_batch <<- integer(0)
  }

  for (f in files) {
    day <- read_day_h5(f)
    ## Show one example of the shape of day$data for debugging
    print(paste("Processing file:", f))
    print(paste("Shape of day$data:", paste(dim(day$data), collapse = "x")))

    
    n <- length(day$time)
    for (i in seq_len(n)) {
      s <- day$sleeping[i]
     #print(paste("Index:", i, "Sleeping:", s, "Non-wear:", day$non_wear[i], "Label:", day$label[i]))
      nw <- day$non_wear[i]
      l <- day$label[i]

      if (s == 1L || nw == 1L || (train && l == -1L)) {
        flush_segment()
        next
      }

      data_batch[[length(data_batch) + 1L]] <- day$data[i, , ]
      timestamps_batch <- c(timestamps_batch, day$time[i])
      label_batch <- c(label_batch, l)
    }

  }

  flush_segment()
  segments
}


## Example test (optional):
# path <- "data/preprocessed"
# subject_id <- "62193"
# segments <- input_iterator_segments(path, subject_id)
# print(length(segments))
# print(head(segments[[1]]$x))


## Function to load model weights from a checkpoint file (.rds or .pth) into a given model.

load_chap2r_weights <- function(model, checkpoint_path, device = NULL) {
  if (!file.exists(checkpoint_path)) {
    stop("Checkpoint not found: ", checkpoint_path)
  }

  ext <- tolower(tools::file_ext(checkpoint_path))

  arr_state <- NULL
  if (ext == "rds") {
    arr_state <- readRDS(checkpoint_path)
  } else if (ext %in% c("pth", "pt")) {
    arr_state <- torch_load(checkpoint_path)
    if (is.list(arr_state) && !is.null(arr_state$state_dict)) {
      arr_state <- arr_state$state_dict
    }
    if (is.list(arr_state) && !is.null(arr_state$model_state_dict)) {
      arr_state <- arr_state$model_state_dict
    }
  } else if (ext == "h5") {
    stop(".h5 checkpoint weights are not directly supported here. Convert .h5 -> .pth/.rds first.")
  } else {
    stop("Unsupported checkpoint extension: .", ext, " (use .rds, .pth, or .pt)")
  }

  model_keys <- names(model$state_dict())

  alt_keys <- function(k) {
    c(
      k,
      sub("_l1_reverse$", "_l0_reverse", k),
      sub("_l1$", "_l0", k),
      sub("_l0_reverse$", "_l1_reverse", k),
      sub("_l0$", "_l1", k)
    )
  }

  to_tensor <- function(x) {
    if (inherits(x, "torch_tensor")) {
      t <- x$to(dtype = torch_float())
    } else {
      t <- torch_tensor(x, dtype = torch_float())
    }
    if (!is.null(device)) t <- t$to(device = device)
    t
  }

  aligned <- list()
  missing <- character(0)

  for (k in model_keys) {
    cands <- unique(alt_keys(k))
    hit <- cands[cands %in% names(arr_state)]
    if (length(hit) == 0L) {
      missing <- c(missing, k)
      next
    }
    aligned[[k]] <- to_tensor(arr_state[[hit[1]]])
  }

  if (length(missing) > 0L) {
    stop("Missing state keys after remap: ", paste(missing, collapse = ", "))
  }

  model$load_state_dict(aligned, strict = TRUE)
  model$eval()
  model
}


## Example test (optional):
# model_name <- "CHAP_ALL_ADULTS"
# model <- build_chap_model(model_name)
# checkpoint_path <- get_chap_model_checkpoint(model_name)
# model <- load_chap2r_weights(model, checkpoint_path)
# print("Model weights loaded successfully.")




as_nhwc3 <- function(x) {
  d <- dim(x)
  if (length(d) == 4L) {
    if (d[4] == 1L) {
      return(x[, , , 1, drop = TRUE])
    }
    stop("Unexpected 4D shape for x. Expected last dim == 1.")
  }
  if (length(d) == 3L) return(x)
  stop("Unexpected x shape. Expected 3D or 4D tensor-like array.")
}





segment_predict <- function(model, x, bi_lstm_win_size, padding = "wrap", downsample_window = 0.1, device = NULL, return_probabilities = FALSE, batch_n_seq = 64L) {
  if (!padding %in% c("drop", "zero", "wrap")) {
    stop("padding must be one of: drop, zero, wrap")
  }

  x <- as_nhwc3(x)
  print(paste("segment_predict: input shape", paste(dim(x), collapse = "x")))
  print(paste("segment_predict: bi_lstm_win_size", bi_lstm_win_size, "padding", padding))
  n <- dim(x)[1]
  border <- n %% bi_lstm_win_size

  wrapped <- FALSE
  zeroed <- FALSE
  deficit <- 0L

  if (border != 0L) {
    if (padding == "drop") {
      keep <- seq_len(n - border)
      x <- x[keep, , , drop = FALSE]
    } else {
      deficit <- bi_lstm_win_size - border
      if (padding == "zero") {
        x_pad <- array(0, dim = c(deficit, dim(x)[2], dim(x)[3]))
        x <- abind::abind(x, x_pad, along = 1)
        zeroed <- TRUE
      }
      if (padding == "wrap") {
        if (n < bi_lstm_win_size) {
          stop("Cannot use wrap padding when segment length is shorter than bi_lstm_win_size")
        }
        x_last_p1 <- x[seq_len(n - border), , , drop = FALSE]
        x_last_p2 <- x[(n - bi_lstm_win_size + 1L):n, , , drop = FALSE]
        x <- abind::abind(x_last_p1, x_last_p2, along = 1)
        wrapped <- TRUE
      }
    }
  }

  if (dim(x)[1] == 0L) return(integer(0))

  n_windows <- dim(x)[1]
  n_seq <- n_windows %/% bi_lstm_win_size
  print(paste("segment_predict: n_windows", n_windows, "n_seq", n_seq, "border", border))
  if (n_seq == 0L) return(integer(0))

  preds <- integer(0)
  probs_all <- numeric(0)
  batch_n_seq <- max(1L, as.integer(batch_n_seq))
  n_batches <- ceiling(n_seq / batch_n_seq)
  progress_every <- max(1L, n_batches %/% 10L)

  flatten_seq_order <- function(m) {
    if (is.null(dim(m))) return(as.numeric(m))
    as.numeric(t(m))
  }

  for (b in seq_len(n_batches)) {
    seq_start <- (b - 1L) * batch_n_seq + 1L
    seq_end <- min(n_seq, b * batch_n_seq)
    if (b == 1L || b == n_batches || (b %% progress_every) == 0L) {
      print(paste("segment_predict: processing batch", b, "of", n_batches, "(seq", seq_start, "to", seq_end, ")"))
    }

    idx_start <- (seq_start - 1L) * bi_lstm_win_size + 1L
    idx_end <- seq_end * bi_lstm_win_size
    chunk <- x[idx_start:idx_end, , , drop = FALSE] # [batch_seq*win, 100, 3]

    t_in <- torch_tensor(chunk, dtype = torch_float())
    if (!is.null(device)) t_in <- t_in$to(device = device)
    t_in <- t_in$unsqueeze(4)$permute(c(1, 4, 2, 3)) # [batch_seq*win, 1, 100, 3]

    out <- model(t_in)
    prob <- torch_sigmoid(out)
    probs_vec <- flatten_seq_order(as.array(prob$cpu()))
    pred <- (prob >= 0.5)$to(dtype = torch_int())
    preds <- c(preds, as.integer(flatten_seq_order(as.array(pred$cpu()))))
    probs_all <- c(probs_all, probs_vec)
  }

  if (padding == "wrap" && wrapped) {
    preds <- c(preds[seq_len(length(preds) - bi_lstm_win_size)], tail(preds, border))
    probs_all <- c(probs_all[seq_len(length(probs_all) - bi_lstm_win_size)], tail(probs_all, border))
  } else if (padding == "zero" && zeroed && deficit > 0L) {
    preds <- head(preds, length(preds) - deficit)
    probs_all <- head(probs_all, length(probs_all) - deficit)
  }

  print(paste("segment_predict: completed with", length(preds), "predictions"))

  if (return_probabilities) {
    return(list(prediction = preds, probability = probs_all))
  }

  preds
}



## Function to save one day's predictions to a CSV file.
save_chap_day_predictions <- function(
  timestamps,
  predictions,
  probabilities = NULL,
  output_dir,
  day_name = NULL,
  labels = NULL,
  include_label = FALSE,
  segment_id = NULL,
  include_segment = FALSE,
  include_probability = FALSE,
  overwrite = TRUE,
  label_map = c("sitting", "not-sitting", "no-label")
) {
  n_out <- min(length(timestamps), length(predictions))
  if (n_out == 0L) {
    stop("No timestamps/predictions to save.")
  }

  timestamps <- timestamps[seq_len(n_out)]
  predictions <- predictions[seq_len(n_out)]
  ts_fmt <- format(
    as.POSIXct(timestamps, origin = "1970-01-01", tz = Sys.timezone()),
    "%Y-%m-%d %H:%M:%S"
  )

  label_text <- function(values) {
    ifelse(
      values == -1L,
      label_map[3],
      ifelse(values == 0L, label_map[1], label_map[2])
    )
  }

  out_df <- data.frame(timestamp = ts_fmt, stringsAsFactors = FALSE)

  if (include_segment) {
    out_df <- data.frame(
      segment = if (is.null(segment_id)) NA_integer_ else as.integer(segment_id),
      out_df,
      stringsAsFactors = FALSE
    )
  }

  if (include_label) {
    if (is.null(labels)) {
      stop("labels must be provided when include_label = TRUE.")
    }
    out_df$label <- label_text(as.integer(labels[seq_len(n_out)]))
  }

  if (include_probability) {
    if (is.null(probabilities)) {
      stop("probabilities must be provided when include_probability = TRUE.")
    }
    out_df$probability <- as.numeric(probabilities[seq_len(n_out)])
  }

  out_df$prediction <- label_text(as.integer(predictions))

  if (is.null(day_name)) {
    day_name <- format(as.Date(ts_fmt[1]), "%Y-%m-%d")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_file <- file.path(output_dir, paste0(day_name, ".csv"))
  if (file.exists(output_file) && overwrite) {
    file.remove(output_file)
  }

  if (file.exists(output_file)) {
    old_df <- utils::read.csv(output_file, stringsAsFactors = FALSE)

    # Align column schemas so appends work even if optional columns changed
    # between runs (e.g., segment/label toggled on or off).
    all_cols <- union(names(old_df), names(out_df))
    missing_in_old <- setdiff(all_cols, names(old_df))
    missing_in_new <- setdiff(all_cols, names(out_df))

    if (length(missing_in_old) > 0L) {
      for (col in missing_in_old) old_df[[col]] <- NA
    }
    if (length(missing_in_new) > 0L) {
      for (col in missing_in_new) out_df[[col]] <- NA
    }

    old_df <- old_df[, all_cols, drop = FALSE]
    out_df <- out_df[, all_cols, drop = FALSE]
    out_df <- rbind(old_df, out_df)
  }
  utils::write.csv(out_df, output_file, row.names = FALSE, quote = TRUE)
  message("Wrote predictions: ", output_file)

  invisible(out_df)
}



## Example test (optional):
# down_sample_frequency <- 10L
# model_name <- "CHAP_ALL_ADULTS"
# spec_rec <- get_chap_model_spec(model_name)
# spec <- spec_rec$specs
# bi_lstm_win_size <- as.integer(60L %/% down_sample_frequency * spec$bi_lstm_window_size)
# model <- build_chap_model(model_name)
# checkpoint_path <- get_chap_model_checkpoint(model_name)
# model <- load_chap2r_weights(model, checkpoint_path)
#
# path <- "data/preprocessed"
# subject_id <- "62193"
# segments <- input_iterator_segments(path, subject_id)
#
# if (length(segments) == 0L) stop("No segments available.")
# x_in <- segments[[1]]$x
# y_pred <- segment_predict(
#   model = model,
#   x = x_in,
#   bi_lstm_win_size = bi_lstm_win_size,
#   padding = "wrap",
#   device = NULL
# )
#
# n_show <- min(length(y_pred), length(segments[[1]]$timestamps))
# ts_fmt <- format(
#   as.POSIXct(segments[[1]]$timestamps[seq_len(n_show)], origin = "1970-01-01", tz = Sys.timezone()),
#   "%Y-%m-%d %H:%M:%S"
# )
# pred_txt <- ifelse(y_pred[seq_len(n_show)] == 0L, "sitting", "not-sitting")
# pred_df <- data.frame(timestamp = ts_fmt, prediction = pred_txt, stringsAsFactors = FALSE)
# print(head(pred_df, 20))
