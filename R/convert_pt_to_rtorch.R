library(reticulate)

use_condaenv("deepposture", required = TRUE)
py_torch <- import("torch", convert = FALSE)
builtins <- import_builtins(convert = FALSE)

src_dir <- "scripts/posture_library/MSSE-2021/pre-trained-models-pt"
out_dir <- "scripts/posture_library/MSSE-2021/pre-trained-models-rtorch"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

py_keys_chr <- function(obj) {
  # Force Python keys-view -> Python list -> R character vector
  k <- py_to_r(builtins$list(obj$keys()))
  as.character(unlist(k, use.names = FALSE))
}

extract_state_dict <- function(obj) {
  keys <- tryCatch(py_keys_chr(obj), error = function(e) character(0))
  if (length(keys) == 0) return(obj)

  if ("state_dict" %in% keys) return(obj[["state_dict"]])
  if ("model_state_dict" %in% keys) return(obj[["model_state_dict"]])
  obj
}

files <- sort(list.files(src_dir, pattern = "\\.pth$", full.names = TRUE))
stopifnot(length(files) > 0)

for (f in files) {
  cat("Converting:", basename(f), "\n")

  ckpt <- tryCatch(
    py_torch$load(f, map_location = "cpu", weights_only = TRUE),
    error = function(e) py_torch$load(f, map_location = "cpu")
  )

  sd <- extract_state_dict(ckpt)
  keys <- py_keys_chr(sd)

  arr_state <- list()
  for (k in keys) {
    arr_state[[k]] <- py_to_r(sd[[k]]$detach()$cpu()$numpy())
  }

  out_file <- file.path(out_dir, sub("\\.pth$", ".rds", basename(f)))
  saveRDS(arr_state, out_file)
  cat("  ->", out_file, "\n")
}

cat("Done.\n")
