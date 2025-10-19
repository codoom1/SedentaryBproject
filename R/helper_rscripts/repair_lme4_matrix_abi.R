#!/usr/bin/env Rscript

# Rebuild lme4 against the currently installed Matrix to resolve ABI mismatch.
# Safe to run multiple times. Works best inside the project (renv will auto-activate).

suppressPackageStartupMessages({
  if (file.exists("renv/activate.R")) {
    source("renv/activate.R")
  }
})

msg <- function(...) cat(sprintf(...), "\n")

safe_pkg_version <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(pkg))
}

msg("Project: %s", basename(getwd()))
msg("R: %s", getRversion())
msg("Library paths:")
print(.libPaths())

matrix_ver_before <- safe_pkg_version("Matrix")
lme4_ver_before   <- safe_pkg_version("lme4")
msg("Matrix (before): %s", matrix_ver_before)
msg("lme4   (before): %s", lme4_ver_before)

rebuild_lme4 <- function() {
  if (requireNamespace("renv", quietly = TRUE)) {
    msg("Rebuilding lme4 via renv::rebuild('lme4') …")
    renv::rebuild("lme4")
  } else {
    msg("renv not available; installing lme4 from source via install.packages …")
    install.packages("lme4", type = "source")
  }
}

try(rebuild_lme4(), silent = FALSE)

# Verify load and versions after rebuild
abi_ok <- TRUE
warns  <- NULL

withCallingHandlers({
  suppressPackageStartupMessages({
    library(Matrix)
    library(lme4)
  })
}, warning = function(w) {
  warns <<- c(warns, conditionMessage(w))
  invokeRestart("muffleWarning")
})

if (length(warns)) {
  abi_warn <- grep("ABI version mismatch|check_dep_version", warns, value = TRUE)
  if (length(abi_warn)) {
    abi_ok <- FALSE
    msg("Warning detected after rebuild: %s", abi_warn[[1]])
  }
}

matrix_ver_after <- safe_pkg_version("Matrix")
lme4_ver_after   <- safe_pkg_version("lme4")
msg("Matrix  (after): %s", matrix_ver_after)
msg("lme4    (after): %s", lme4_ver_after)

if (isTRUE(abi_ok)) {
  msg("Success: lme4 loads cleanly against Matrix. ABI mismatch resolved.")
  quit(status = 0)
} else {
  msg("Still seeing ABI mismatch. Options:\n  1) Clean rebuild of both Matrix and lme4: renv::rebuild(c('Matrix','lme4'))\n  2) Restore lockfile state: renv::restore()\n  3) Downgrade Matrix to a pre-ABI bump (e.g., 1.6-5) and reinstall lme4")
  quit(status = 1)
}
