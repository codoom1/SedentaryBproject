# Setup renv for this project and install required packages

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

# 1) Ensure renv is available
ensure_package("renv")

# 2) Initialize renv (idempotent)
if (!dir.exists("renv")) {
  renv::init(bare = TRUE)  # minimal init; we'll explicitly install packages below
} else {
  message("[INFO] renv already initialized; activating...")
  renv::activate()
}

# 3) Install project packages
pkgs <- c(
  "dplyr", "readr", "ggplot2", "tidyr", "tibble",
  "lme4", "emmeans", "pbkrtest", "scales"
)
renv::install(pkgs)

# 4) Snapshot lockfile
renv::snapshot(prompt = FALSE)

message("[DONE] renv initialized and packages installed. Lockfile updated.")
