



## Load the big data using a RELATIVE path (configurable)
## Priority order:
##  1. Environment variable MAIN_DATA_FILE if set
##  2. data/maindata/combined/merged_big_dataset_with_quality.csv.gz
##  3. data/maindata/summary_merged.csv[.gz] as fallback
data_load <- function() {
  
  candidates <- "data/project_data/final_data.csv.gz"
  candidates <- candidates[!is.na(candidates)]
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop("No main data file found. Set MAIN_DATA_FILE env var or place file under data/maindata/.")
  }
  message("[INFO] Loading main data file: ", existing[1])
  df <- .safe_read_csv(existing[1])
  # Add DayType if not present
  if (!("DayType" %in% names(df)) && ("Day" %in% names(df))) {
    df <- add_daytype_from_day(df, day_col = "Day", out_col = "DayType")
  }
  # Merge covariates from nonsleep dataset if available
  ns_path <- "constants/nosleep_data.csv.gz"  # repository path
  if (file.exists(ns_path)) {
    df <- merge_nonsleep_covariates(df, nonsleep_path = ns_path, id_col = "ID")
  } else {
    message("[WARN] Nonsleep dataset not found at ", ns_path, "; proceeding without covariates.")
  }
  df <- dplyr::arrange(df, Day, ID, Hour)
  df
}


# -------------------------------------------------------------------
# Safe CSV reader: tries readr, then data.table::fread, then base utils
# Supports gz files via gzfile() in base reader
# -------------------------------------------------------------------
.safe_read_csv <- function(path) {
  # Try readr first, if available (guard against corrupt namespace)
  can_readr <- tryCatch(requireNamespace("readr", quietly = TRUE), error = function(e) FALSE)
  if (isTRUE(can_readr)) {
    out <- tryCatch(suppressMessages(readr::read_csv(path, progress = FALSE)), error = function(e) NULL)
    if (!is.null(out)) return(out)
    message("[WARN] readr failed to read ", path, "; falling back to alternative reader.")
  }
  # Try data.table::fread if available (guard namespace)
  can_dt <- tryCatch(requireNamespace("data.table", quietly = TRUE), error = function(e) FALSE)
  if (isTRUE(can_dt)) {
    out <- tryCatch(data.table::fread(path, data.table = FALSE, showProgress = FALSE), error = function(e) NULL)
    if (!is.null(out)) return(out)
  }
  # Fallback to base utils (supports gz via gzfile)
  con <- try(gzfile(path, "rt"), silent = TRUE)
  if (!inherits(con, "try-error")) {
    on.exit(try(close(con), silent = TRUE), add = TRUE)
    return(utils::read.csv(con))
  }
  # If gzfile failed (maybe not gz), try plain read.csv
  return(utils::read.csv(path))
}

# Silence R CMD check notes for NSE columns used in dplyr/ggplot pipelines
utils::globalVariables(c(
  "Height","Weight","ID",".height_m","Hour","Hour_num","emmean",
  "lower.CL","upper.CL","asymp.LCL","asymp.UCL","emm_var","emm_level",
  ".","DayType"
))


# -------------------------------------------------------------------
# 1) Derive DayType (Weekday/Weekend) from a Day column
#    - Expects Day to be coercible to Date (YYYY-mm-dd)
#    - Adds/overwrites DayType with values: "Weekday" or "Weekend"
# -------------------------------------------------------------------
add_daytype_from_day <- function(df, day_col = "Day", out_col = "DayType") {
  if (!(day_col %in% names(df))) {
    stop("Column '", day_col, "' not found in data frame.")
  }
  # Coerce to Date; tolerate character/factor inputs
  day_vec <- tryCatch(as.Date(df[[day_col]]), error = function(e) NA)
  if (all(is.na(day_vec))) {
    stop("Could not coerce '", day_col, "' to Date for DayType derivation.")
  }
  wd <- weekdays(day_vec)
  is_weekend <- grepl("Saturday|Sunday", wd, ignore.case = TRUE)
  df[[out_col]] <- ifelse(is_weekend, "Weekend", "Weekday")
  # Return with DayType as factor for consistency
  df[[out_col]] <- factor(df[[out_col]], levels = c("Weekday", "Weekend"), ordered = FALSE)
  df
}

# -------------------------------------------------------------------
# 2) Merge participant covariates from constants/nosleep_dataset.csv.gz
#    - Attaches: Age, Height, Weight, BMI (computed), Gender, Marital_status,
#      RatioItop, Interview_weight, Exam_weight, PSU, Stratum, data_quality_level
#    - Left-join by ID (keeps all rows of df)
#    - Height assumed in cm, Weight in kg (per sample); BMI = kg / m^2
# -------------------------------------------------------------------
merge_nonsleep_covariates <- function(df,
                                      nonsleep_path = "constants/nosleep_data.csv.gz",
                                      id_col = "ID") {
  if (!file.exists(nonsleep_path)) {
    stop("Nonsleep dataset not found at ", nonsleep_path)
  }
  if (!(id_col %in% names(df))) {
    stop("Column '", id_col, "' not found in primary data frame.")
  }
  ns <- .safe_read_csv(nonsleep_path)
  if (!(id_col %in% names(ns))) {
    stop("Column '", id_col, "' not found in nonsleep dataset.")
  }
  # Standardize common column names if necessary (case-insensitive helpers)
  get_col <- function(tbl, candidates) {
    # return first matching name from candidates in tbl, else NA
    for (nm in candidates) {
      if (nm %in% names(tbl)) return(nm)
    }
    # try case-insensitive match
    for (nm in names(tbl)) {
      if (tolower(nm) %in% tolower(candidates)) return(nm)
    }
    NA_character_
  }
  age_col   <- get_col(ns, c("Age"))
  ht_col    <- get_col(ns, c("Height"))
  wt_col    <- get_col(ns, c("Weight"))
  g_col     <- get_col(ns, c("Gender","Sex"))
  mar_col   <- get_col(ns, c("Marital_status","Marital Status","marital_status"))
  ratio_col <- get_col(ns, c("RatioItop","PIR","RatioIToP","ratio_itop"))
  iw_col    <- get_col(ns, c("Interview_weight","Interview Weight","WTINT2YR","interview_weight"))
  ew_col    <- get_col(ns, c("Exam_weight","Exam Weight","WTMEC2YR","exam_weight"))
  psu_col   <- get_col(ns, c("PSU","psu"))
  str_col   <- get_col(ns, c("Stratum","stratum","SDMVSTRA"))
  dql_col   <- get_col(ns, c("data_quality_level","data_quality","Data_quality_level"))

  # Create a reduced table for join
  keep <- c(id_col, age_col, ht_col, wt_col, g_col, mar_col, ratio_col, iw_col, ew_col, psu_col, str_col, dql_col)
  keep <- unique(keep[!is.na(keep)])
  ns_keep <- ns[, keep, drop = FALSE]

  # Rename for standardized output names
  rename_map <- c()
  if (!is.na(age_col)) rename_map[age_col] <- "Age"
  if (!is.na(ht_col))  rename_map[ht_col]  <- "Height"
  if (!is.na(wt_col))  rename_map[wt_col]  <- "Weight"
  if (!is.na(g_col))   rename_map[g_col]   <- "Gender"
  if (!is.na(mar_col)) rename_map[mar_col] <- "Marital_status"
  if (!is.na(ratio_col)) rename_map[ratio_col] <- "RatioItop"
  if (!is.na(iw_col))  rename_map[iw_col]  <- "Interview_weight"
  if (!is.na(ew_col))  rename_map[ew_col]  <- "Exam_weight"
  if (!is.na(psu_col)) rename_map[psu_col] <- "PSU"
  if (!is.na(str_col)) rename_map[str_col] <- "Stratum"
  if (!is.na(dql_col)) rename_map[dql_col] <- "data_quality_level"
  ns_keep <- dplyr::rename(ns_keep, dplyr::all_of(rename_map))

  # Collapse to one row per participant ID to prevent row multiplication on join
  if (nrow(ns_keep) > 0) {
    ns_keep <- ns_keep |> dplyr::group_by(.data[[id_col]]) |> dplyr::slice_head(n = 1) |> dplyr::ungroup()
  }

  # Perform left join by ID
  out <- dplyr::left_join(df, ns_keep, by = setNames(id_col, id_col))

  # Ensure data_quality_level exists even if absent
  if (!"data_quality_level" %in% names(out)) {
    out$data_quality_level <- NA_character_
  }

  # Compute BMI when Height & Weight present
  if (all(c("Height","Weight") %in% names(out))) {
    out <- out %>% dplyr::mutate(
      BMI = dplyr::if_else(is.na(Height) | is.na(Weight) | Height <= 0,
                           as.numeric(NA),
                           Weight / ((Height/100) ^ 2))
    )
  } else {
    if (!("BMI" %in% names(out))) out$BMI <- rep(NA_real_, nrow(out))
  }

  out
}


preprocess_sed_data <- function(
  df,
  remove_partial_days   = FALSE,  # default disabled (final_data has no PartialDay)
  require_one_full_day  = FALSE,  # default disabled (depends on PartialDay)
  adults_only           = FALSE,
  factor_id_hour        = FALSE,
  split_by_daytype      = FALSE,
  enforce_hour_unordered = TRUE,
  set_hour_treatment     = TRUE,
  drop_flagged          = FALSE,
  flagged_ids           = NULL,
  flag_col              = NULL
) {
  # packages
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Package 'dplyr' is required.")
  }
  library(dplyr)
  
  # basic column checks (fail early with helpful messages where critical)
  needed_min <- c("ID", "Hour")
  missing_min <- setdiff(needed_min, names(df))
  if (length(missing_min)) {
    stop("Missing required columns: ", paste(missing_min, collapse = ", "))
  }
  out <- df
  # Derive DayType if absent but Day present
  if (!("DayType" %in% names(out))) {
    if ("Day" %in% names(out)) {
      out <- add_daytype_from_day(out, day_col = "Day", out_col = "DayType")
    } else {
      stop("'DayType' not found and cannot derive because 'Day' column is missing.")
    }
  }
  # Ensure DayType is a factor
  if (!is.factor(out$DayType)) out$DayType <- factor(out$DayType, levels = c("Weekday","Weekend"))
  
  # (Optional) drop flagged rows or IDs
  if (drop_flagged) {
    if (!is.null(flag_col)) {
      if (!flag_col %in% names(out)) stop("flag_col '", flag_col, "' not found in df.")
      out <- out %>% dplyr::filter(!.data[[flag_col]])
    }
    if (!is.null(flagged_ids)) {
      out <- out %>% dplyr::filter(!(ID %in% flagged_ids))
    }
  }
  
  # Note: Partial-day logic removed (final_data has no PartialDay)
  # Adults only (Age >= 18) – apply only if Age exists
  if (adults_only && ("Age" %in% names(out))) {
    out <- out %>% dplyr::filter(.data$Age >= 18)
  } else if (adults_only) {
    warning("adults_only=TRUE but 'Age' column not found; skipping adult filter.")
  }
  
  # Convert types for modeling
  # Always make DayType a factor (helps consistent splitting)
  if (!is.factor(out$DayType)) out$DayType <- factor(out$DayType)
  
  # If requested, coerce ID and Hour to factors
  if (factor_id_hour) {
    out$ID <- factor(out$ID)              # ensure factor
    out$ID <- droplevels(out$ID)          # drop unused
    # Hour as factor with levels 1..24 (if numeric), but NOT ordered
    if (is.numeric(out$Hour)) {
      hr_lvls <- sort(unique(out$Hour))
      out$Hour <- factor(out$Hour, levels = hr_lvls, ordered = FALSE)
    } else {
      out$Hour <- factor(out$Hour, ordered = FALSE)
    }
  }
  
  # Even if factor_id_hour == FALSE, we still want Hour to be unordered (to avoid contr.poly)
  if (enforce_hour_unordered) {
    # If Hour is ordered, drop ordering; if not a factor, coerce carefully
    if (is.ordered(out$Hour)) {
      out$Hour <- factor(out$Hour, levels = levels(out$Hour), ordered = FALSE)
    } else if (!is.factor(out$Hour)) {
      hr_lvls <- sort(unique(out$Hour))
      out$Hour <- factor(out$Hour, levels = hr_lvls, ordered = FALSE)
    }
  }
  
  # Apply treatment contrasts to Hour (keeps design sparse for lmer)
  if (set_hour_treatment && is.factor(out$Hour)) {
    contrasts(out$Hour) <- contr.treatment(nlevels(out$Hour))
  }
  
  # ---------------------------
  # Age groups, BMI and BMI category (robust to missing columns)
  # ---------------------------
  weight_col <- names(out)[grepl("^weight$|weight", names(out), ignore.case = TRUE)]
  height_col <- names(out)[grepl("^height$|height", names(out), ignore.case = TRUE)]
  weight_col <- if (length(weight_col)) weight_col[1] else NULL
  height_col <- if (length(height_col)) height_col[1] else NULL

  if (!is.null(weight_col) && !is.null(height_col)) {
    out <- out %>%
      dplyr::mutate(
        .height_m = ifelse(is.na(.data[[height_col]]), NA_real_, .data[[height_col]] / 100),
        BMI = dplyr::if_else(
          is.na(.data[[weight_col]]) | is.na(.height_m) | (.height_m <= 0),
          as.numeric(NA),
          .data[[weight_col]] / (.height_m ^ 2)
        )
      ) %>%
      dplyr::select(-.height_m)
  } else {
    if (!("BMI" %in% names(out))) out$BMI <- NA_real_
  }

  # Age groups and BMI categories
  if ("Age" %in% names(out)) {
    out <- out %>%
      dplyr::mutate(
        age_group = dplyr::case_when(
          .data$Age < 18 ~ "<18",
          .data$Age >= 18 & .data$Age <= 34 ~ "18-34",
          .data$Age >= 35 & .data$Age <= 44 ~ "35-44",
          .data$Age >= 45 & .data$Age <= 64 ~ "45-64",
          .data$Age >= 65 ~ "65+",
          TRUE ~ NA_character_
        ),
        bmi_cat = dplyr::case_when(
          .data$Age < 18 ~ NA_character_,
          !is.na(BMI) & BMI < 25 ~ "Under/Normal (<25)",
          !is.na(BMI) & BMI >= 25 & BMI < 30 ~ "Overweight (25-<30)",
          !is.na(BMI) & BMI >= 30 ~ "Obesity (>=30)",
          TRUE ~ NA_character_
        )
      )
    # factorize
    base_lvls <- c("18-34", "35-44", "45-64", "65+")
    if (any(out$Age < 18, na.rm = TRUE)) base_lvls <- c("<18", base_lvls)
    out$age_group <- factor(out$age_group, levels = base_lvls, ordered = FALSE)
    out$bmi_cat   <- factor(out$bmi_cat, levels = c("Under/Normal (<25)", "Overweight (25-<30)", "Obesity (>=30)"), ordered = FALSE)
  } else {
    # create placeholder factors to keep downstream models happy
    out$age_group <- factor(rep(NA_character_, nrow(out)), levels = c("<18","18-34","35-44","45-64","65+"))
    out$bmi_cat   <- factor(rep(NA_character_, nrow(out)), levels = c("Under/Normal (<25)", "Overweight (25-<30)", "Obesity (>=30)"))
  }
  
  # Optionally split into weekday/weekend datasets
  if (split_by_daytype) {
    weekday_df <- out %>% dplyr::filter(.data$DayType == "Weekday")
    weekend_df <- out %>% dplyr::filter(.data$DayType == "Weekend")
    
    # provenance
    info <- list(
      remove_partial_days   = remove_partial_days,
      require_one_full_day  = require_one_full_day,
      adults_only           = adults_only,
      factor_id_hour        = factor_id_hour,
      enforce_hour_unordered= enforce_hour_unordered,
      set_hour_treatment    = set_hour_treatment,
      split_by_daytype      = TRUE
    )
    attr(weekday_df, "preprocess_info") <- info
    attr(weekend_df, "preprocess_info") <- info
    
    return(list(
      all     = out,
      weekday = weekday_df,
      weekend = weekend_df
    ))
  }
  
  # attach provenance
  attr(out, "preprocess_info") <- list(
    remove_partial_days    = remove_partial_days,
    require_one_full_day   = require_one_full_day,
    adults_only            = adults_only,
    factor_id_hour         = factor_id_hour,
    enforce_hour_unordered = enforce_hour_unordered,
    set_hour_treatment     = set_hour_treatment,
    split_by_daytype       = FALSE
  )
  out
}


## -------------------------------------------------------------------------
## Helper: prepare subset data for diagonal-G models (sampling & weights)
## -------------------------------------------------------------------------
prepare_diagG_datasets <- function(weekday_df, weekend_df, n_select = 5000,
                                   id_col = "ID", weight_col = "Exam_weight",
                                   scale_weights = TRUE, verbose = TRUE) {
  stopifnot(id_col %in% names(weekday_df), id_col %in% names(weekend_df))
  stopifnot(weight_col %in% names(weekday_df), weight_col %in% names(weekend_df))
  
  # Collect unique IDs
  ids_weekday_all <- unique(weekday_df[[id_col]])
  ids_weekend_all <- unique(weekend_df[[id_col]])
  
  # Allow n_select > available count gracefully
  n_wd <- min(n_select, length(ids_weekday_all))
  n_we <- min(n_select, length(ids_weekend_all))
  ids_weekday <- ids_weekday_all[seq_len(n_wd)]
  ids_weekend <- ids_weekend_all[seq_len(n_we)]
  
  sub_weekday <- dplyr::filter(weekday_df, .data[[id_col]] %in% ids_weekday)
  sub_weekend <- dplyr::filter(weekend_df, .data[[id_col]] %in% ids_weekend)
  
  if (!(weight_col %in% names(sub_weekday)) || !(weight_col %in% names(sub_weekend))) {
    stop("Weight column '", weight_col, "' not found in both datasets.")
  }
  
  if (scale_weights) {
    sub_weekday$scaled_weight <- sub_weekday[[weight_col]] / mean(sub_weekday[[weight_col]], na.rm = TRUE)
    sub_weekend$scaled_weight <- sub_weekend[[weight_col]] / mean(sub_weekend[[weight_col]], na.rm = TRUE)
  }
  
  weights_day <- sub_weekday[[weight_col]]
  weights_end <- sub_weekend[[weight_col]]
  
  if (verbose) {
    message(sprintf("[prepare_diagG_datasets] Weekday IDs: %d (requested %d); Weekend IDs: %d (requested %d)",
                    n_wd, n_select, n_we, n_select))
  }
  
  list(
    data_model_weekday = sub_weekday,
    data_model_weekend = sub_weekend,
    weights_day = weights_day,
    weights_end = weights_end,
    ids_weekday = ids_weekday,
    ids_weekend = ids_weekend
  )
}



# Helper to fit a diagonal-G model for a data frame with Hour as 1..24
fit_diag_by_hour <- function(df, weights, extra_fixed = NULL) {
  # Ensure Hour is a factor with full 1..24 support (even if some hours absent)
  df <- df %>% mutate(Hour = factor(.data$Hour, levels = 1:24))
  # Build 24 numeric dummy columns for RANDOM part
  X <- model.matrix(~ 0 + Hour, df)              # Hour1 ... Hour24
  colnames(X) <- make.names(colnames(X))
  md <- cbind(df, as.data.frame(X))
  re_cols <- paste(colnames(X), collapse = " + ")
  fixed_terms <- c("Hour", extra_fixed)
  fixed_terms <- unique(fixed_terms[!is.na(fixed_terms) & nzchar(fixed_terms)])
  fixed_rhs <- paste(fixed_terms, collapse = " + ")
  form <- as.formula(paste0("percent_sitting ~ ", fixed_rhs, " + (0 + ", re_cols, " || ID)"))
  ctl  <- lmerControl(optimizer = "bobyqa")
  lmer(form, data = md, REML = TRUE, control = ctl, weights = weights)
}



# ---------------------------------------------------------------------
# compute_emm_hour_quality: produces emm_res list with $raw and $with_ci
# Hour-only rows are tagged as emm_var="Overall", emm_level="Overall"
# ---------------------------------------------------------------------
compute_emm_hour_quality <- function(model_weekday, model_weekend,
                                     emm_spec = NULL,
                                     extra_fixed = NULL,
                                     type = "response",
                                     hour_levels = 1:24,
                                     weekday_label = "Weekday",
                                     weekend_label = "Weekend",
                                     weighting = c("equal","proportional")) {
  weighting <- match.arg(weighting)
  specs <- character(0)
  
  if (!is.null(emm_spec)) {
    specs <- emm_spec
  } else if (!is.null(extra_fixed)) {
    specs <- c("Hour", paste0("Hour+", extra_fixed))
  } else {
    specs <- "Hour"
  }
  
  run_for_model <- function(mod, day_label) {
    out_list <- list()
    for (sp in specs) {
      fml <- as.formula(paste0("~ ", sp))
      emm_df <- as.data.frame(emmeans::emmeans(mod, fml, type = type,
                                               weights = if (weighting == "proportional") "proportional" else NULL))
      vars_in_spec <- strsplit(sp, "\\+")[[1]] |> trimws()
      other_vars <- setdiff(vars_in_spec, "Hour")
      
      # Tag Hour-only specs as Overall (no NA)
      if (length(other_vars) == 0 || all(other_vars == "")) {
        emm_df$emm_var   <- "Overall"
        emm_df$emm_level <- "Overall"
      } else if (length(other_vars) == 1) {
        v <- other_vars
        emm_df$emm_var <- v
        if (v %in% names(emm_df)) {
          emm_df$emm_level <- as.character(emm_df[[v]])
        } else {
          emm_df$emm_level <- NA_character_
        }
      } else {
        emm_df$emm_var <- paste(other_vars, collapse = "+")
        present_vars <- intersect(other_vars, names(emm_df))
        if (length(present_vars)) {
          emm_df$emm_level <- apply(emm_df[, present_vars, drop = FALSE], 1, function(r) paste(r, collapse = "_"))
        } else {
          emm_df$emm_level <- NA_character_
        }
      }
      
      emm_df$DayType <- day_label
      emm_df$emm_spec <- sp
      out_list[[sp]] <- emm_df
    }
    bind_rows(out_list)
  }
  
  emm_wd <- run_for_model(model_weekday, weekday_label)
  emm_we <- run_for_model(model_weekend, weekend_label)
  
  combined <- bind_rows(emm_wd, emm_we) %>%
    mutate(
      Hour = if ("Hour" %in% names(.)) factor(Hour, levels = as.character(hour_levels), ordered = TRUE) else NA,
      Hour_num = if (!is.na(Hour[1])) as.numeric(as.character(Hour)) else NA_real_
    )
  
  # Standardize CI names
  if (all(c("lower.CL","upper.CL") %in% names(combined))) {
    combined_ci <- mutate(combined, LCL = .data$lower.CL, UCL = .data$upper.CL)
  } else if (all(c("asymp.LCL","asymp.UCL") %in% names(combined))) {
    combined_ci <- mutate(combined, LCL = .data$asymp.LCL, UCL = .data$asymp.UCL)
  } else {
    combined_ci <- mutate(combined, LCL = NA_real_, UCL = NA_real_)
  }
  
  list(raw = combined, with_ci = combined_ci)
}

# ---------------------------------------------------------------------
# plot_emm_by_hour: plots Hour (1-24) vs emmean with ribbons and grouping
# - Expects emm_res from compute_emm_hour_quality()
# - Treats emm_var/emm_level == "Overall" specially (thicker black ribbon/line)
# ---------------------------------------------------------------------
plot_emm_by_hour <- function(emm_res,
                             which_table = c("with_ci","raw"),
                             var_to_plot = NULL,         # pick one emm_var or NULL auto
                             overall_label = "Overall",  # label used internally by compute_emm...
                             include_overall = TRUE,     # NEW: toggle to draw overall line/ribbon
                             overall_legend_label = "all adults", # NEW: legend text for overall
                             overall_ci_alpha = 0.22,
                             point_size = 1.6,
                             line_size = 0.8,
                             overall_line_size = 1.1,
                             palette = NULL,
                             linetypes = NULL,
                             facet_scales = "free_y") {
  which_table <- match.arg(which_table)
  tbl <- if (which_table == "with_ci") emm_res$with_ci else emm_res$raw
  
  # Defensive column setup
  if (!"emm_var" %in% names(tbl))   tbl$emm_var   <- NA_character_
  if (!"emm_level" %in% names(tbl)) tbl$emm_level <- NA_character_
  if (!"Hour_num" %in% names(tbl) && "Hour" %in% names(tbl)) tbl$Hour_num <- as.numeric(as.character(tbl$Hour))
  
  required_cols <- c("Hour_num", "emmean", "DayType", "emm_var", "emm_level")
  missing_cols <- setdiff(required_cols, names(tbl))
  if (length(missing_cols)) stop("emm table missing columns: ", paste(missing_cols, collapse = ", "))
  
  tbl <- tbl %>%
    mutate(
      Hour_num = as.numeric(Hour_num),
      emm_var = as.character(emm_var),
      emm_level = as.character(emm_level),
      # treat any empty/NA emm_var or emm_level as Overall (match compute function behavior)
      emm_var = ifelse(is.na(emm_var) | emm_var == "" , overall_label, emm_var),
      emm_level = ifelse(is.na(emm_level) | emm_level == "" , overall_label, emm_level),
      # internal flag for rows that are truly overall aggregate
      is_overall = (tolower(emm_var) == tolower(overall_label)) & (tolower(emm_level) == tolower(overall_label))
    )
  
  detected_vars <- unique(na.omit(tbl$emm_var))
  detected_vars <- setdiff(detected_vars, overall_label) # non-overall variables
  
  # Select what to plot
  if (!is.null(var_to_plot)) {
    if (!var_to_plot %in% detected_vars) stop("var_to_plot not found in results.")
    plot_tbl <- tbl %>% filter(emm_var == var_to_plot | is_overall)
    facet_by_var <- FALSE
  } else {
    if (length(detected_vars) <= 1) {
      plot_tbl <- tbl
      facet_by_var <- FALSE
    } else {
      facet_by_var <- TRUE
      plot_tbl <- tbl %>% filter(emm_var %in% detected_vars | is_overall)
    }
  }
  
  # Ensure LCL/UCL exist
  if (!all(c("LCL","UCL") %in% names(plot_tbl))) {
    plot_tbl <- plot_tbl %>% mutate(LCL = emmean, UCL = emmean)
    warning("No LCL/UCL in table; ribbons collapse to lines.")
  }
  
  # Split main levels and overall
  plot_df_main <- plot_tbl %>% filter(!is_overall & emm_var != overall_label)
  plot_df_overall <- plot_tbl %>% filter(is_overall)
  
  # If including overall in the legend, create a plotting-level column that maps overall to the legend label
  # and uses original emm_level values for the non-overall rows.
  plot_df_main <- plot_df_main %>% mutate(emm_level_plot = emm_level)
  if (include_overall) {
    plot_df_overall <- plot_df_overall %>% mutate(emm_level_plot = overall_legend_label)
  } else {
    # if not including overall, ensure it's empty so we don't draw it
    plot_df_overall <- plot_df_overall %>% mutate(emm_level_plot = NA_character_)
  }
  
  # Colors/linetypes defaults for present emm_level_plot values (exclude NA)
  non_overall_plot_levels <- sort(unique(na.omit(plot_df_main$emm_level_plot)))
  present_plot_levels <- c(non_overall_plot_levels, if (include_overall) overall_legend_label else character(0))
  
  if (is.null(palette)) {
    nlev <- max(3, length(present_plot_levels))
    cols <- scales::hue_pal()(nlev)
    # assign first colors to non-overall levels, last color reserved for overall (but we override below to black)
    pal_vals <- setNames(cols[seq_len(length(non_overall_plot_levels))], non_overall_plot_levels)
    if (include_overall) pal_vals[[overall_legend_label]] <- "black"
    palette <- pal_vals
  } else {
    # ensure overall gets a color (black) if not provided and included
    if (include_overall && !(overall_legend_label %in% names(palette))) {
      palette[[overall_legend_label]] <- "black"
    }
  }
  
  if (is.null(linetypes)) {
    linetypes <- setNames(rep("solid", length(non_overall_plot_levels)), non_overall_plot_levels)
    if (include_overall) linetypes[[overall_legend_label]] <- "solid"
  } else {
    if (include_overall && !(overall_legend_label %in% names(linetypes))) {
      linetypes[[overall_legend_label]] <- "solid"
    }
  }
  
  # Build plot: ribbons/lines for main levels and optional overall (overall uses emm_level_plot to appear in legend)
  p <- ggplot() +
    # main ribbons (by emm_level_plot)
    geom_ribbon(data = plot_df_main,
                aes(x = Hour_num, ymin = LCL, ymax = UCL, fill = emm_level_plot,
                    group = interaction(emm_level_plot, DayType)),
                alpha = 0.25, colour = NA)
  
  # overall ribbon: only draw if include_overall == TRUE and there are rows
  if (include_overall && nrow(plot_df_overall) > 0) {
    # draw overall ribbon in black; still map fill for legend consistency but set show.legend = TRUE
    p <- p + geom_ribbon(data = plot_df_overall,
                         aes(x = Hour_num, ymin = LCL, ymax = UCL, fill = emm_level_plot,
                             group = interaction(emm_level_plot, DayType)),
                         alpha = overall_ci_alpha, colour = NA)
  }
  
  # main lines & points
  p <- p +
    geom_line(data = plot_df_main,
              aes(x = Hour_num, y = emmean, color = emm_level_plot, linetype = emm_level_plot,
                  group = interaction(emm_level_plot, DayType)),
              linewidth = line_size) +
    geom_point(data = plot_df_main,
               aes(x = Hour_num, y = emmean, color = emm_level_plot,
                   group = interaction(emm_level_plot, DayType)),
               size = point_size)
  
  # overall lines & points (if included). To include in legend, map color/linetype to emm_level_plot
  if (include_overall && nrow(plot_df_overall) > 0) {
    p <- p +
      geom_line(data = plot_df_overall,
                aes(x = Hour_num, y = emmean, color = emm_level_plot, linetype = emm_level_plot,
                    group = interaction(emm_level_plot, DayType)),
                linewidth = overall_line_size) +
      geom_point(data = plot_df_overall,
                 aes(x = Hour_num, y = emmean, color = emm_level_plot,
                     group = interaction(emm_level_plot, DayType)),
                 size = point_size + 0.6)
  }
  
  p <- p +
    scale_x_continuous(breaks = seq(1, 24, by = 2), limits = c(1, 24)) +
    labs(x = "Hour", y = "Estimated % sedentary",
         color = NULL, linetype = NULL, fill = NULL) +
    theme_classic(base_size = 12) +
    theme(legend.position = "right")
  
  # Now build scales only for levels present
  present_levels <- unique(c(plot_df_main$emm_level_plot, if (include_overall) plot_df_overall$emm_level_plot else character(0)))
  present_levels <- na.omit(present_levels)
  
  # Colors: ensure palette covers present_levels; fallback to hue for any missing
  vals <- palette[intersect(names(palette), present_levels)]
  missing_levels <- setdiff(present_levels, names(vals))
  if (length(missing_levels)) {
    fallback <- setNames(scales::hue_pal()(length(missing_levels)), missing_levels)
    vals <- c(vals, fallback)
  }
  p <- p + scale_color_manual(values = vals) + scale_fill_manual(values = vals)
  
  # Linetypes: ensure covers present_levels
  lt_vals <- linetypes[intersect(names(linetypes), present_levels)]
  missing_lt <- setdiff(present_levels, names(lt_vals))
  if (length(missing_lt)) {
    lt_vals <- c(lt_vals, setNames(rep("solid", length(missing_lt)), missing_lt))
  }
  p <- p + scale_linetype_manual(values = lt_vals)
  
  # Facet: multiple variables -> facet_grid(emm_var ~ DayType); otherwise facet_wrap by DayType
  if (facet_by_var) {
    p <- p + facet_grid(emm_var ~ DayType, scales = facet_scales)
  } else {
    p <- p + facet_wrap(~DayType)
  }
  
  # Cosmetic adjustments
  p <- p + theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )
  
  return(p)
}

# ---------------------------------------------------------------------
# Example usage:
# ---------------------------------------------------------------------
# 1) compute emmeans (Hour + age_group and Hour + bmi_cat automatically)
# emm_res <- compute_emm_hour_quality(m_weekday_diag, m_weekend_diag,
#                                     extra_fixed = c("age_group", "bmi_cat"),
#                                     type = "response",
#                                     weighting = "proportional")
#
# 2) Plot only age_group (with Overall included)
# p_age <- plot_emm_by_hour(emm_res, which_table = "with_ci", var_to_plot = "age_group")
# print(p_age)
#
# 3) Plot everything (facets for age_group and bmi_cat, with Overall shown in each panel)
# p_all <- plot_emm_by_hour(emm_res, which_table = "with_ci")
# print(p_all)
