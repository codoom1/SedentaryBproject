
rm(list = ls()) ## Clear environment for clean start

## --- Project Root Resolution -------------------------------------------------
## Prefer running from project root; if not, attempt to set relative to this file.
## Avoid hard-coded user-specific absolute paths for portability.
this_file <- tryCatch(normalizePath(sys.frames()[[1]]$ofile), error = function(e) NA)
if (!is.na(this_file)) {
  proj_root <- normalizePath(file.path(dirname(this_file), ".."))
  if (file.exists(file.path(proj_root, "README.md"))) {
    setwd(proj_root)
  }
}
## If still not at project root, user can manually set via: setwd("/path/to/DeepPostures")

## Load required libraries
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(lme4)
library(emmeans)
library(pbkrtest)
#library(tibble)

## Load helper functions for preprocessing and modeling
source("R/model_utils.R")

# show the first few rows after arranging to confirm ordering
## Load data
df <- data_load()
colnames(df)
unique(df$data_quality_level)
head(df)
length(unique(df$ID))
df$data_quality <- ifelse(df$data_quality_level=="header_only","no_flagged_data","flagged_data")

## Data preprocessing for modeling
# 1. Remove data for partial days
# 2. Include only participant with at least 1 full day data.
# 3. Option to include only 18+(Adults) years data
# 4. Option to make ID and Hour categorical
# 5. Option to divide the dataframe into two datasets(weekday dataset and Weekend dataset)
# (Optional for now). option to drop participants with some flagged data
# Apply preprocess function to get separate datasets for Weekday vs Weekend (plus 'all')
splits <- preprocess_sed_data(df,
                              require_one_full_day = TRUE,
                              adults_only = TRUE,
                              factor_id_hour = TRUE,          # ensure ID + Hour are factors
                              enforce_hour_unordered = TRUE,  # critical to avoid contr.poly
                              set_hour_treatment = TRUE,
                              split_by_daytype = TRUE)
## get all processed data
all_data     <- splits$all
colnames(all_data)





## get weekday data
weekday_data <- splits$weekday
head(weekday_data)
dim(weekday_data)
str(weekday_data)
## get weekend data
weekend_data <- splits$weekend
head(weekend_data)
dim(weekend_data)




# Prepare data (previous inline code wrapped into function above)
prep_subset <- prepare_diagG_datasets(weekday_data, weekend_data, n_select = 1000,
                                      id_col = "ID", weight_col = "Exam_weight")
data_model_weekday <- prep_subset$data_model_weekday
data_model_weekend <- prep_subset$data_model_weekend
weights_day <- prep_subset$weights_day
weights_end <- prep_subset$weights_end
ids_select_weekday <- prep_subset$ids_weekday
ids_select_weekend <- prep_subset$ids_weekend




# Silence R CMD check NOTES for non-standard evaluation variable lookups
utils::globalVariables(c("Hour", "Hour_num", "emmean", "asymp.LCL", "asymp.UCL",
                         "data_quality_level", "DayType", "lower.CL", "upper.CL", "ID",
                         "LCL","UCL", "age_group", "bmi_cat"))


m_weekday_diag <- fit_diag_by_hour(data_model_weekday,weights=weights_day,
                                   extra_fixed = c("age_group","bmi_cat"))
m_weekend_diag <- fit_diag_by_hour(data_model_weekend, weights=weights_end,
                                   extra_fixed = c("age_group","bmi_cat"))
summary(m_weekend_diag)
# (Optional) sanity checks: should be 24 one-column blocks (no Corr table)
# unlist(getME(m_weekday_diag,"cnms")); length(getME(m_weekday_diag,"theta"))
# unlist(getME(m_weekend_diag,"cnms")); length(getME(m_weekend_diag,"theta"))


# # EMMs on the response scale (wrapped into helper)
# overall_weighting_toggle <- "proportional"  # change to "proportional" if desired
# emm_res   <- compute_emm_hour_quality(m_weekday_diag, m_weekend_diag,
#                                       emm_spec = "Hour+data_quality_level",
#                                       type = "response",
#                                       weighting = overall_weighting_toggle)
# emm_all   <- emm_res$raw
# emm_all2  <- emm_res$with_ci


# ---------------------------------------------------------------------
# Example usage:
# ---------------------------------------------------------------------
# 1) compute emmeans (Hour + age_group and Hour + bmi_cat automatically)
emm_res <- compute_emm_hour_quality(m_weekday_diag, m_weekend_diag,
                                    extra_fixed = c("age_group", "bmi_cat"),
                                    type = "response",
                                    weighting = "proportional")

View(emm_res$with_ci)

# 2) Plot only age_group (with Overall included)
p_age <- plot_emm_by_hour(emm_res,
                          which_table = "with_ci", 
                          var_to_plot = "age_group",
                          overall_line_size = 0.4,
                          point_size = 1,
                          line_size = 0.5)
print(p_age)

p_bmi_cat <- plot_emm_by_hour(emm_res,
                              which_table = "with_ci",
                              var_to_plot = "bmi_cat",
                              overall_line_size = 0.4,
                              point_size = 1,
                              line_size = 0.5,
                              include_overall = FALSE)
print(p_bmi_cat )


# 3) Plot everything (facets for age_group and bmi_cat, with Overall shown in each panel)
p_all <- plot_emm_by_hour(emm_res, 
                          which_table = "with_ci",
                          overall_line_size = 0.8,
                          point_size = 1.5,
                          line_size = .6,
                          include_overall = FALSE)
print(p_all)

# Relabel legend entries for data quality levels
quality_label_map <- c(
  header_only      = "no anomalies",
  low_data_q1      = "low anomalies",
  medium_data_q2   = "medium anomalies",
  high_data_q3     = "high anomalies",
  very_high_data_q4= "extreme anomalies"
)

# quality_label_map <- c(
#   no_flagged_data      = "no anomalies",
#   flagged_data      = "has anomalies"
# )

emm_all2 <- emm_all2 %>%
  mutate(
    quality_label = dplyr::recode(.data$data_quality_level, !!!quality_label_map),
    quality_label = factor(quality_label,
                           levels = c("no anomalies","low anomalies","medium anomalies","high anomalies","extreme anomalies"))
  )

# emm_all2 <- emm_all2 %>%
#   mutate(
#     quality_label = dplyr::recode(.data$data_quality, !!!quality_label_map),
#     quality_label = factor(quality_label,
#                            levels = c("no anomalies","has anomalies"))
#   )
# ---------------------------------------------------------------------------
# Overall (quality-collapsed) EMMs: marginal over data_quality_level
# This averages over levels (equal weighting by default). For size-weighted
# marginal means, use emmeans(..., weights = 'proportional').
# ---------------------------------------------------------------------------
overall_res  <- compute_emm_hour_quality(m_weekday_diag, m_weekend_diag,
                                         emm_spec = "Hour", type = "response",
                                         weighting = overall_weighting_toggle)
overall_all2 <- overall_res$with_ci %>% mutate(source = "overall_marginal",
                                               quality_label = "Overall")

# Combine anomaly-level and overall for unified legend
plot_df_main <- emm_all2
plot_df_overall <- overall_all2

# Prepare factor levels including Overall at end
legend_levels <- c(levels(plot_df_main$quality_label), "Overall")
plot_df_overall$quality_label <- factor(plot_df_overall$quality_label, levels = legend_levels)
plot_df_main$quality_label    <- factor(plot_df_main$quality_label, levels = legend_levels)

# Define colors and linetypes
quality_colors <- c(
  "no anomalies" = "#1b9e77",
  "low anomalies" = "#d95f02",
  "medium anomalies" = "#7570b3",
  "high anomalies" = "#e7298a",
  "extreme anomalies" = "#66a61e",
  "Overall" = "#000000"
)
quality_linetypes <- c(
  "no anomalies" = "solid",
  "low anomalies" = "solid",
  "medium anomalies" = "solid",
  "high anomalies" = "solid",
  "extreme anomalies" = "solid",
  "Overall" = "dashed"
)

# # Define colors and linetypes
# quality_colors <- c(
#   "no anomalies" = "#1b9e77",
#   "has anomalies" = "#d95f02",
#   "Overall" = "#000000"
# )
# quality_linetypes <- c(
#   "no anomalies" = "solid",
#   "has anomalies" = "solid",
#   "Overall" = "dashed"
# )

overall_ci_alpha <- 0.40

p1 <- ggplot() +
  # anomaly-level ribbons
  geom_ribbon(data = plot_df_main,
              aes(x = Hour_num, ymin = LCL, ymax = UCL, fill = quality_label,
                  group = interaction(quality_label, DayType)),
              alpha = 0.25, colour = NA) +
  # overall ribbon (thicker alpha, no outline)
  geom_ribbon(data = plot_df_overall,
              aes(x = Hour_num, ymin = LCL, ymax = UCL,
                  group = interaction(quality_label, DayType)),
              alpha = overall_ci_alpha, fill = "black", colour = NA) +
  # anomaly-level lines
  geom_line(data = plot_df_main,
            aes(Hour_num, emmean, color = quality_label, linetype = quality_label,
                group = interaction(quality_label, DayType)), linewidth = 0.7) +
  geom_point(data = plot_df_main,
             aes(Hour_num, emmean, color = quality_label,
                 group = interaction(quality_label, DayType)), size = 1.6) +
  # overall line & points
  geom_line(data = plot_df_overall,
            aes(Hour_num, emmean, color = quality_label, linetype = quality_label,
                group = interaction(quality_label, DayType)), linewidth = 1.1) +
  geom_point(data = plot_df_overall,
             aes(Hour_num, emmean, color = quality_label,
                 group = interaction(quality_label, DayType)), size = 1.9) +
  scale_color_manual(values = quality_colors) +
  scale_linetype_manual(values = quality_linetypes) +
  scale_fill_manual(values = quality_colors[setdiff(names(quality_colors),"Overall")], guide = "none") +
  scale_x_continuous(breaks = seq(1, 24, by = 2), limits = c(1, 24)) +
  labs(x = "Hour", y = "Estimated % sedentary",
       title = NULL,
       color = "Anomaly level", linetype = "Anomaly level") +
  guides(fill = "none") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right") +
  facet_wrap(~DayType)

# Add custom theme adjustments for better readability
p1<- p1 + theme(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(hjust = 0.5)
)




# Save the plot
anomalies_comp <- p1
ggsave("results/data_qual_plots/anomalies_comp.png", plot = p1, 
       width = 10, height = 6, dpi = 300)
ggsave("results/data_qual_plots/anomalies_comp.pdf", plot = p1, 
       width = 10, height = 6, dpi = 300)

# Display the plot
print(p1)

#### =========Inverstigating anomaly levels in the data=====####

cat("\n =======Anomaly levels====\n")

# Create grouping: "no anomalies" vs "has anomalies"
all2_data <- all_data %>%
  mutate(
    quality_label = recode(data_quality_level, !!!quality_label_map),
    quality_label = factor(quality_label,
                           levels = c("no anomalies","low anomalies","medium anomalies","high anomalies","extreme anomalies")),
    anomaly_group = ifelse(quality_label == "no anomalies", "no anomalies", "has anomalies")
  )

# Counts for "has anomalies" stacks
has_anomalies_counts <- all2_data %>%
  distinct(ID, quality_label,anomaly_group) %>%
  filter(quality_label != "no anomalies") %>%  # only stack anomalies
  count(anomaly_group, quality_label) %>%
  group_by(anomaly_group) %>%
  mutate(percent = (n / sum(n)) * 100) %>%
  ungroup()

has_anomalies_counts <- has_anomalies_counts %>%
  arrange(quality_label) %>%
  group_by(anomaly_group) %>%
  mutate(
    ymax = cumsum(n),
    ymin = ymax - n,
    y_text = ymax - n/2  # center for each label
  ) %>%
  ungroup()

# Overall totals for both bars
bar_totals <- all2_data %>%
  distinct(ID, quality_label) %>%
  mutate(anomaly_group = ifelse(quality_label == "no anomalies", "no anomalies", "has anomalies")) %>%
  count(anomaly_group) %>%
  mutate(total_percent = (n / nrow(all2_data %>% distinct(ID))) * 100)

has_anomalies_counts <- has_anomalies_counts %>%
  mutate(quality_label = factor(quality_label,
                                levels = c("extreme anomalies", "high anomalies", "medium anomalies","low anomalies" )))



# Plot
p <- ggplot()  +
  # Stacked "has anomalies" bar
  geom_bar(data = has_anomalies_counts,
           aes(x = anomaly_group, y = n, fill = quality_label),
           stat = "identity") +
  # Percentages inside each stack
  geom_text(
    data = has_anomalies_counts,
    aes(x = anomaly_group, y = y_text, label = paste0(sprintf("%.1f", percent), "%")),
    size = 3
  ) +
  # Overall % on top of stacked bar
  geom_text(data = bar_totals %>% filter(anomaly_group == "has anomalies"),
            aes(x = anomaly_group, y = n, label = paste0(sprintf("%.1f", total_percent), "%")),
            vjust = -0.5, inherit.aes = FALSE, size = 5) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "Data Quality", y = "Number of Participants",ylim=c(0,9000),
       title = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right",
        axis.text.x = element_text(angle = 0, hjust = 0.5))+
  # "No anomalies" bar
  geom_bar(data = bar_totals %>% filter(anomaly_group == "no anomalies"),
           aes(x = anomaly_group, y = n),
           stat = "identity", fill = "blue") +
  # Add percentage centered inside the "no anomalies" bar
  geom_text(data = bar_totals %>% filter(anomaly_group == "no anomalies"),
            aes(x = anomaly_group, y = n, label = paste0(sprintf("%.1f", total_percent), "%")),
            inherit.aes = FALSE,vjust = -.5, size = 5, color = "black")+
  scale_y_continuous(breaks = seq(0, 7000, by = 1000), limits = c(0, 7000))

p<- p + theme(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(hjust = 0.5)
)
print(p)



# Save the plot
anomaly_part <- p
ggsave("results/data_qual_plots/anomaly_part.png", plot = p, 
       width = 10, height = 6, dpi = 300)
ggsave("results/data_qual_plots/anomaly_part.pdf", plot = p, 
       width = 10, height = 6, dpi = 300)




## (Optional)
## Define a function to automate the entire process
## from preprocessing to modeling and plotting.
fit_mixed_hour_models <- function(
    df,
    # preprocessing controls
    remove_partial_days = TRUE,
    require_one_full_day = TRUE,
    adults_only = TRUE,
    factor_id_hour = TRUE,
    # modeling controls
    model_types = c("intercept", "diagG", "corr"),
    weight_col = "Exam_weight",
    scale_weights = TRUE,
    diagG_use_weights = TRUE,
    corr_use_weights = TRUE,
    intercept_use_weights = TRUE,
    n_diagG_ids = NULL,            # optional: restrict IDs (NULL = all)
  # new options
  extra_fixed = NULL,            # character vector of additional fixed-effect variable names
  subset_ids = NULL,             # vector of IDs to keep (applied after preprocessing)
  subset_filter = NULL,          # function(df)->logical OR single character expression evaluated within df
    subset_n_ids = NULL,           # numeric: choose this many IDs (intersection of weekday/weekend) if subset_ids not supplied
    subset_sample_seed = NULL,     # seed for reproducible sampling when subset_n_ids used
    subset_sample_method = c("random","first"), # selection method when using subset_n_ids
  verbose = TRUE,
  # overall overlay / weighting options (for future extension)
  overall_weighting = c("equal","proportional"),
  include_overall_overlay = FALSE,
  overall_line_linetype = "dashed",
  overall_line_size = 1.1
    ,plot_by_quality = FALSE              # NEW: produce quality-level (anomaly) plot like standalone diagG plot
    ,quality_factor = "data_quality_level" # factor name for anomalies
    ,quality_label_map = c(                # default relabels
      header_only = "no anomalies",
      low_data_q1 = "low anomalies",
      medium_data_q2 = "medium anomalies",
      high_data_q3 = "high anomalies",
      very_high_data_q4 = "extreme anomalies"
    )
) {
  model_types <- unique(match.arg(model_types, choices = c("intercept", "diagG", "corr"), several.ok = TRUE))
  subset_sample_method <- match.arg(subset_sample_method)
  overall_weighting <- match.arg(overall_weighting)

  # Auto-enable quality plot if user supplied the quality factor in extra_fixed but didn't toggle plot_by_quality
  if (!plot_by_quality && !is.null(extra_fixed) && quality_factor %in% extra_fixed) {
    plot_by_quality <- TRUE
    if (verbose) message("[fit_mixed_hour_models] Auto-enabled plot_by_quality because '", quality_factor, "' is in extra_fixed.")
  }

  # 1) Preprocess
  splits <- preprocess_sed_data(
    df,
    remove_partial_days = remove_partial_days,
    require_one_full_day = require_one_full_day,
    adults_only = adults_only,
    factor_id_hour = factor_id_hour,
    enforce_hour_unordered = TRUE,
    set_hour_treatment = TRUE,
    split_by_daytype = TRUE
  )
  weekday_df <- splits$weekday
  weekend_df <- splits$weekend
  all_data <- splits$all
  
  quality_label_map <- c(
    header_only      = "no anomalies",
    low_data_q1      = "low anomalies",
    medium_data_q2   = "medium anomalies",
    high_data_q3     = "high anomalies",
    very_high_data_q4= "extreme anomalies"
  )
  
  all2_data <- all_data %>% mutate(
    quality_label = recode(data_quality_level, !!!quality_label_map),
    quality_label = factor(quality_label,levels = c("no anomalies","low anomalies","medium anomalies","high anomalies","extreme anomalies"))
    )
  # First, calculate the total number of unique IDs
  total_ids <- all2_data %>% 
    distinct(ID) %>% 
    nrow()
  
  # Then create the quality counts with percentages
  quality_counts <- all2_data %>%
    distinct(ID, quality_label) %>%  # Get unique id-quality combinations
    count(quality_label) %>%         # Count by quality level
    mutate(percent = (n / total_ids) * 100) %>%  # Calculate percentage
    arrange(desc(n))                      # Sort by count
  
  # View the result
  print(quality_counts)
  
  # Create the bar chart
 p2 <-  ggplot(quality_counts, aes(x = quality_label, y = n, fill = quality_label)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = paste0(n, "(", sprintf("%.2f", percent),")")), vjust = -0.5, size = 4) +  # Add count labels on bars
    labs(x = "Data Quality Level", 
         y = "Number of Unique IDs",
         title = "Number of Unique Participants by Data Quality Level") +
    scale_fill_brewer(palette = "Set2") +  # Use a nice color palette
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",  # Remove legend since x-axis shows categories
          axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x labels if needed
  
  print(p2)
  
  # Derive subset_ids from numeric request if provided ---------------------
  if (is.null(subset_ids) && !is.null(subset_n_ids)) {
    ids_pool <- intersect(unique(weekday_df$ID), unique(weekend_df$ID))
    if (length(ids_pool) == 0) stop("No overlapping IDs between weekday and weekend data for numeric subsetting.")
    n_target <- min(subset_n_ids, length(ids_pool))
    if (!is.null(subset_sample_seed)) set.seed(subset_sample_seed)
    if (subset_sample_method == "random") {
      subset_ids <- sample(ids_pool, n_target)
    } else {
      subset_ids <- ids_pool[seq_len(n_target)]
    }
    if (verbose) {
      if (n_target < subset_n_ids) {
        message(sprintf("[fit_mixed_hour_models] Requested %d IDs but only %d available; using all.", subset_n_ids, n_target))
      }
      message(sprintf("[fit_mixed_hour_models] Selected %d IDs via %s method for subsetting.", n_target, subset_sample_method))
    }
  }

  # Optional subsetting -----------------------------------------------------
  apply_subset <- function(dat) {
    if (!is.null(subset_ids)) {
      dat <- dplyr::filter(dat, ID %in% subset_ids)
    }
    if (!is.null(subset_filter)) {
      if (is.function(subset_filter)) {
        keep <- subset_filter(dat)
        if (!is.logical(keep) || length(keep) != nrow(dat)) {
          stop("subset_filter function must return logical vector of length nrow(data).")
        }
      } else if (is.character(subset_filter) && length(subset_filter) == 1) {
        keep <- with(dat, eval(parse(text = subset_filter)))
        if (!is.logical(keep) || length(keep) != nrow(dat)) {
          stop("subset_filter expression must evaluate to logical vector of length nrow(data).")
        }
      } else {
        stop("subset_filter must be a function or a single character expression.")
      }
      dat <- dat[keep, , drop = FALSE]
    }
    dat
  }

  weekday_df <- apply_subset(weekday_df)
  weekend_df <- apply_subset(weekend_df)

  if (nrow(weekday_df) == 0) stop("No weekday rows after filtering.")
  if (nrow(weekend_df) == 0) stop("No weekend rows after filtering.")

  # Ensure Hour is unordered factor with treatment contrasts
  weekday_df$Hour <- factor(weekday_df$Hour, ordered = FALSE)
  weekend_df$Hour <- factor(weekend_df$Hour, ordered = FALSE)
  contrasts(weekday_df$Hour) <- contr.treatment(nlevels(weekday_df$Hour))
  contrasts(weekend_df$Hour) <- contr.treatment(nlevels(weekend_df$Hour))

  # Validate extra_fixed vars
  if (!is.null(extra_fixed)) {
    if (!all(extra_fixed %in% names(weekday_df)) || !all(extra_fixed %in% names(weekend_df))) {
      missing_wd <- setdiff(extra_fixed, names(weekday_df))
      missing_we <- setdiff(extra_fixed, names(weekend_df))
      stop("Some extra_fixed variables are missing. Weekday missing: ", paste(missing_wd, collapse=","),
           "; Weekend missing: ", paste(missing_we, collapse=","))
    }
  }

  # Weight handling ---------------------------------------------------------
  if (!is.null(weight_col)) {
    if (!(weight_col %in% names(weekday_df) && weight_col %in% names(weekend_df))) {
      stop("weight_col '", weight_col, "' not found in both weekday & weekend data frames.")
    }
    if (scale_weights) {
      weekday_df$scaled_weight <- weekday_df[[weight_col]] / mean(weekday_df[[weight_col]], na.rm = TRUE)
      weekend_df$scaled_weight <- weekend_df[[weight_col]] / mean(weekend_df[[weight_col]], na.rm = TRUE)
    }
    w_wd <- if (scale_weights) weekday_df$scaled_weight else weekday_df[[weight_col]]
    w_we <- if (scale_weights) weekend_df$scaled_weight else weekend_df[[weight_col]]
  } else {
    w_wd <- w_we <- NULL
  }

  ctl_fast <- lmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
  ctl_full <- lmerControl(optimizer = "bobyqa",calc.derivs = FALSE)

  models <- list()
  emmeans_list <- list()

  # Helper: compute emmeans + tidy for a given pair of weekday/weekend models
  collect_emm <- function(m_wd, m_we, label) {
    emm_wd <- as.data.frame(emmeans(m_wd, ~ Hour))  # Expect lower.CL/upper.CL OR asymp.*
    emm_wd$DayType <- "Weekday"
    emm_we <- as.data.frame(emmeans(m_we, ~ Hour))
    emm_we$DayType <- "Weekend"
    emm_all <- bind_rows(emm_wd, emm_we) %>%
      mutate(Hour = factor(Hour, levels = as.character(1:24), ordered = TRUE),
             Hour_num = as.numeric(as.character(Hour)),
             model_type = label)
    # Standardize CI columns to LCL/UCL so downstream code is stable
    if (all(c("lower.CL","upper.CL") %in% names(emm_all))) {
      emm_all$LCL <- emm_all$lower.CL; emm_all$UCL <- emm_all$upper.CL
    } else if (all(c("asymp.LCL","asymp.UCL") %in% names(emm_all))) {
      emm_all$LCL <- emm_all$asymp.LCL; emm_all$UCL <- emm_all$asymp.UCL
    } else {
      emm_all$LCL <- NA_real_; emm_all$UCL <- NA_real_
    }
    emm_all
  }

  # (1) Intercept-only model ------------------------------------------------
  if ("intercept" %in% model_types) {
    if (verbose) message("[fit_mixed_hour_models] Fitting intercept models ...")
  fixed_terms_intercept <- c("Hour", extra_fixed)
  fixed_terms_intercept <- unique(fixed_terms_intercept[!is.na(fixed_terms_intercept) & nzchar(fixed_terms_intercept)])
  rhs_intercept <- paste(fixed_terms_intercept, collapse = " + ")
  form_intercept <- as.formula(paste0("PercentSedentary ~ ", rhs_intercept, " + (1 | ID)"))
  m_weekday_intercept <- lmer(form_intercept, data = weekday_df,
                                REML = TRUE, control = ctl_fast,
                                weights = if (intercept_use_weights) w_wd else NULL)
  m_weekend_intercept <- lmer(form_intercept, data = weekend_df,
                                REML = TRUE, control = ctl_fast,
                                weights = if (intercept_use_weights) w_we else NULL)
    models$intercept <- list(weekday = m_weekday_intercept, weekend = m_weekend_intercept)
    emmeans_list$intercept <- collect_emm(m_weekday_intercept, m_weekend_intercept, "intercept")
  }

  # (2) Diagonal-G model (uncorrelated per-hour random effects) -------------
  if ("diagG" %in% model_types) {
    if (verbose) message("[fit_mixed_hour_models] Fitting diagonal-G models ...")
    # Optionally restrict IDs for speed
    if (!is.null(n_diagG_ids)) {
      keep_ids_wd <- unique(weekday_df$ID)[seq_len(min(n_diagG_ids, length(unique(weekday_df$ID))))]
      keep_ids_we <- unique(weekend_df$ID)[seq_len(min(n_diagG_ids, length(unique(weekend_df$ID))))]
      wd_sub <- dplyr::filter(weekday_df, ID %in% keep_ids_wd)
      we_sub <- dplyr::filter(weekend_df, ID %in% keep_ids_we)
      w_wd_sub <- if (intercept_use_weights) w_wd[weekday_df$ID %in% keep_ids_wd] else NULL
      w_we_sub <- if (intercept_use_weights) w_we[weekend_df$ID %in% keep_ids_we] else NULL
    } else {
      wd_sub <- weekday_df; we_sub <- weekend_df; w_wd_sub <- w_wd; w_we_sub <- w_we
    }
    # Support old versions of fit_diag_by_hour without extra_fixed argument
    fd_formals <- try(formals(fit_diag_by_hour), silent = TRUE)
    has_extra <- !inherits(fd_formals, "try-error") && "extra_fixed" %in% names(fd_formals)
    if (has_extra) {
      m_weekday_diag <- fit_diag_by_hour(wd_sub, weights = if (diagG_use_weights) w_wd_sub else NULL,
                                         extra_fixed = extra_fixed)
      m_weekend_diag <- fit_diag_by_hour(we_sub, weights = if (diagG_use_weights) w_we_sub else NULL,
                                         extra_fixed = extra_fixed)
    } else {
      if (!is.null(extra_fixed)) {
        warning("extra_fixed specified but current fit_diag_by_hour() does not accept it; update Modeling.R definition to use this feature.")
      }
      m_weekday_diag <- fit_diag_by_hour(wd_sub, weights = if (diagG_use_weights) w_wd_sub else NULL)
      m_weekend_diag <- fit_diag_by_hour(we_sub, weights = if (diagG_use_weights) w_we_sub else NULL)
    }
    models$diagG <- list(weekday = m_weekday_diag, weekend = m_weekend_diag)
    emmeans_list$diagG <- collect_emm(m_weekday_diag, m_weekend_diag, "diagG")
  }

  # (3) Correlated random hour effects (full covariance) --------------------
  if ("corr" %in% model_types) {
    if (verbose) message("[fit_mixed_hour_models] Fitting correlated (full G) models ...")
    # For correlated structure: (0 + Hour | ID) gives a 24x24 covariance (can be heavy)
  fixed_terms_corr <- c("Hour", extra_fixed)
  fixed_terms_corr <- unique(fixed_terms_corr[!is.na(fixed_terms_corr) & nzchar(fixed_terms_corr)])
  rhs_corr <- paste(fixed_terms_corr, collapse = " + ")
  form_corr <- as.formula(paste0("PercentSedentary ~ ", rhs_corr, " + (0 + Hour | ID)"))
  m_weekday_corr <- lmer(form_corr, data = weekday_df,
                           REML = TRUE, control = ctl_full,
                           weights = if (corr_use_weights) w_wd else NULL)
  m_weekend_corr <- lmer(form_corr, data = weekend_df,
                           REML = TRUE, control = ctl_full,
                           weights = if (corr_use_weights) w_we else NULL)
    models$corr <- list(weekday = m_weekday_corr, weekend = m_weekend_corr)
    emmeans_list$corr <- collect_emm(m_weekday_corr, m_weekend_corr, "corr")
  }

  # Choose a default plot model preference: diagG > corr > intercept
  plot_choice <- if ("diagG" %in% names(emmeans_list)) {
    emmeans_list$diagG
  } else if ("corr" %in% names(emmeans_list)) {
    emmeans_list$corr
  } else {
    emmeans_list$intercept
  }

  # Default simple DayType plot (if not plotting by quality levels)
  p <- NULL
  if (!plot_by_quality) {
    p <- ggplot(plot_choice, aes(Hour_num, emmean, color = DayType, fill = DayType)) +
      geom_ribbon(aes(ymin = LCL, ymax = UCL), alpha = 0.30, colour = NA) +
      geom_line(linewidth = 0.2) +
      geom_point(size = 1) +
      scale_x_continuous(breaks = seq(1, 24, by = 2), limits = c(1, 24)) +
      labs(x = "Hour", y = "Estimated % Sedentary",
           title = sprintf("Sedentary Behavior by Hour (%s model)", unique(plot_choice$model_type))) +
      theme_classic(base_size = 12) +
      theme(legend.title = element_blank(), legend.position = "bottom")
    if (include_overall_overlay) {
      overall_df <- plot_choice
      p <- p +
        geom_line(data = overall_df,
                  aes(Hour_num, emmean), inherit.aes = FALSE,
                  linetype = overall_line_linetype, linewidth = overall_line_size, color = "black")
      p<- p + theme(
        strip.background = element_blank(),
        strip.text = element_text(face = "bold"),
        plot.title = element_text(hjust = 0.5)
      )
      print(p)
    }
  } else {
    # Build quality-level emmeans for selected model type (Hour + quality_factor)
    chosen_model_type <- unique(plot_choice$model_type)[1]
    chosen_models <- models[[chosen_model_type]]
    if (is.null(chosen_models)) stop("Internal error: chosen model type not found in models list.")
    q_res <- compute_emm_hour_quality(chosen_models$weekday, chosen_models$weekend,
                                      emm_spec = paste0("Hour+", quality_factor),
                                      type = "response",
                                      weighting = overall_weighting)
    q_all  <- q_res$raw
    q_all2 <- q_res$with_ci
    # Recode quality labels
    if (!all(names(quality_label_map) %in% unique(q_all2[[quality_factor]]))) {
      # proceed silently; only recode those present
    }
    q_all2 <- q_all2 %>%
      mutate(quality_label = {
        orig <- as.character(.data[[quality_factor]])
        mapped <- unname(quality_label_map[orig])
        # For any values not in the map, keep original
        mapped[is.na(mapped)] <- orig[is.na(mapped)]
        mapped
      })
    # Order factor levels: map order + any extras appended
    base_lvls <- quality_label_map
    ordered_levels <- unique(c(unname(base_lvls), setdiff(unique(q_all2$quality_label), unname(base_lvls))))
    q_all2$quality_label <- factor(q_all2$quality_label, levels = ordered_levels)
    # Overall overlay (marginal over quality)
    overall_q <- compute_emm_hour_quality(chosen_models$weekday, chosen_models$weekend,
                                          emm_spec = "Hour", type = "response",
                                          weighting = overall_weighting)$with_ci %>%
      mutate(quality_label = "Overall")
    # Combine factor levels for legend
    q_all2$quality_label <- factor(q_all2$quality_label, levels = c(levels(q_all2$quality_label), "Overall"))
    overall_q$quality_label <- factor(overall_q$quality_label, levels = levels(q_all2$quality_label))
    # Color & linetype scales
    palette_base <- c(
      "no anomalies" = "#1b9e77",
      "low anomalies" = "#d95f02",
      "medium anomalies" = "#7570b3",
      "high anomalies" = "#e7298a",
      "extreme anomalies" = "#66a61e"
    )
    # Add any missing levels with grey
    missing_cols <- setdiff(levels(q_all2$quality_label), names(palette_base))
    if (length(missing_cols) > 0) {
      add_cols <- rep("grey29", length(missing_cols)); names(add_cols) <- missing_cols
      palette_base <- c(palette_base, add_cols)
    }
    palette_full <- c(palette_base, Overall = "#000000")
    ltypes <- rep("solid", length(palette_full)); names(ltypes) <- names(palette_full); ltypes["Overall"] <- overall_line_linetype
    # Build plot
    p <- ggplot() +
      geom_ribbon(data = q_all2,
                  aes(x = Hour_num, ymin = LCL, ymax = UCL, fill = quality_label,
                      group = interaction(quality_label, DayType)), alpha = 0.25, colour = NA) +
      geom_ribbon(data = overall_q,
                  aes(x = Hour_num, ymin = LCL, ymax = UCL,
                      group = interaction(quality_label, DayType)), alpha = 0.40, fill = "black", colour = NA) +
      geom_line(data = q_all2,
                aes(Hour_num, emmean, color = quality_label, linetype = quality_label,
                    group = interaction(quality_label, DayType)), linewidth = 0.2) +
      geom_point(data = q_all2,
                 aes(Hour_num, emmean, color = quality_label,
                     group = interaction(quality_label, DayType)), size = 1) +
      geom_line(data = overall_q,
                aes(Hour_num, emmean, color = quality_label, linetype = quality_label,
                    group = interaction(quality_label, DayType)), linewidth = overall_line_size) +
      geom_point(data = overall_q,
                 aes(Hour_num, emmean, color = quality_label,
                     group = interaction(quality_label, DayType)), size = 1) +
      scale_color_manual(values = palette_full) +
      scale_linetype_manual(values = ltypes) +
      scale_fill_manual(values = palette_full[setdiff(names(palette_full), "Overall")], guide = "none") +
      scale_x_continuous(breaks = seq(1,24,2), limits = c(1,24)) +
      labs(x = "Hour", y = "Estimated % sedentary",
           title = sprintf("Sedentary behavior by hour (%s model)", chosen_model_type),
           color = "Anomaly level", linetype = "Anomaly level") +
      theme_classic(base_size = 12) +
      theme(legend.position = "right") +
      facet_wrap(~DayType)
    p <-p+ theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5)
    )
    print(p)
    # Replace plot_choice with quality plot data for export if desired
    plot_choice <- q_all2
  }

  # Save consolidated plot (named by chosen model)
  # Derive a stable tag for filenames
  if (exists("chosen_model_type")) {
    base_type <- chosen_model_type
  } else if ("model_type" %in% names(plot_choice) && length(unique(plot_choice$model_type)) >= 1) {
    base_type <- unique(plot_choice$model_type)[1]
  } else {
    base_type <- "model"
  }
  out_tag <- gsub("[^A-Za-z0-9_]+", "_", base_type)
  png_file <- sprintf("plots/sedentary_by_hour_qualitylevel_%s_model.png", out_tag)
  pdf_file <- sprintf("plots/sedentary_by_hour_qualitylevel_%s_model.pdf", out_tag)
  ggsave(png_file, plot = p, width = 10, height = 6, dpi = 300)
  ggsave(pdf_file, plot = p, width = 10, height = 6)

  # Bind all emmeans results into a single data frame for export
  all_emm <- bind_rows(emmeans_list)

  subset_info <- list(
    subset_ids = if (!is.null(subset_ids)) length(unique(subset_ids)) else NULL,
    subset_filter = if (!is.null(subset_filter)) TRUE else FALSE,
    subset_n_ids_requested = if (!is.null(subset_n_ids)) subset_n_ids else NULL,
    subset_n_ids_actual = if (!is.null(subset_ids)) length(unique(subset_ids)) else NULL,
    subset_sample_method = if (!is.null(subset_n_ids)) subset_sample_method else NULL
  )

  list(
    models = models,
    emmeans = emmeans_list,
    emmeans_combined = all_emm,
  plot = p,
  plot_by_quality = plot_by_quality,
  quality_factor = quality_factor,
  quality_label_map = quality_label_map,
    model_types = model_types,
  weight_info = list(weight_col = weight_col, scaled = scale_weights,
             overall_weighting = overall_weighting),
    extra_fixed = extra_fixed,
    subset_info = subset_info
  )
}

##### ------------------------- Use the function ---------------------- ####

# Create plots directory if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots", recursive = TRUE)
}

# Create results directory if it doesn't exist
if (!dir.exists("results/modeling")) {
  dir.create("results/modeling", recursive = TRUE)
}

# Example (default: fits intercept + diagG + corr, no overlays):
# result <- fit_mixed_hour_models(df)

# Example: diagG-only fit with overall overlay, proportional weighting,
# extra fixed effects, and random subsetting of 10000 IDs for speed.
# (Adjust variable names in extra_fixed to those present in your data.)

cat("\n == Running the diagonal G model with data_quality_level fixed effect==\n")

# diagG_result <- fit_mixed_hour_models(
#   df,
#   model_types = "diagG",            # only diagonal-G random effects
#   extra_fixed = c("data_quality_level"),  # example additional fixed effects
#   subset_n_ids = 10000,               # sample 1200 overlapping IDs
#   subset_sample_seed = 123,          # reproducible sampling
#   subset_sample_method = "random",   # or "first"
#   scale_weights = TRUE,              # create & use scaled weights
#   diagG_use_weights = TRUE,
#   include_overall_overlay = TRUE,    # add dashed overall curve
#   overall_weighting = "proportional",# average Hours weighting by cell sizes
#   overall_line_linetype = "dashed",
#   overall_line_size = .15,
#   verbose = TRUE
# )
### Save model details(optional)
# saveRDS(diagG_result, file = "results/models_object/diagG_models.rds")

# Access pieces from save model :
diagG_result <- readRDS(file="results/models_object/diagG_models.rds")

summary(diagG_result$models$diagG$weekday )
diagG_result$emmeans$diagG         
diagG_result$plot                  
diagG_result$subset_info           
diagG_result$weight_info          
VarCorr(diagG_result$models$diagG$weekday)

# Display plot
print(diagG_result$plot)


#### Full correlation structure model
cat("\n == Running the Full G model with data_quality_level fixed effect==\n")

# FullG_result <- fit_mixed_hour_models(
#   df,
#   model_types = "corr",            # only Full-G random effects
#   extra_fixed = c("data_quality_level"),  # example additional fixed effects
#   subset_n_ids = 10000,               # sample 10000 overlapping IDs
#   subset_sample_seed = 123,          # reproducible sampling
#   subset_sample_method = "random",   # or "first"
#   scale_weights = TRUE,              # create & use scaled weights
#   diagG_use_weights = TRUE,
#   include_overall_overlay = TRUE,    # add dashed overall curve
#   overall_weighting = "proportional",# average Hours weighting by cell sizes
#   overall_line_linetype = "dashed",
#   overall_line_size = .15,
#   verbose = TRUE
# )

### Save model details(optional)
# saveRDS(FullG_result, file = "results/models_object/FullG_models.rds")

# Access pieces from save model :
FullG_result <- readRDS(file="results/models_object/FullG_models.rds")

summary(FullG_result$models$corr$weekday )
FullG_result$emmeans$corr         
FullG_result$plot                  
FullG_result$subset_info           
FullG_result$weight_info          
VarCorr(FullG_result$models$corr$weekday)
# Display and save the plot
print(FullG_result$plot)

## Let arrange the plots
library(patchwork)
p1 <- FullG_result$plot
p2 <- diagG_result$plot
combined <- p1 / p2
full_diag_G<- combined

ggsave(filename ="results/data_qual_plots/full_diag_G.png" )
ggsave(filename ="results/data_qual_plots/full_diag_G.pdf" )
## combine Full G and Diagonal G model emmeans
full_diag_emmeans <- rbind(FullG_result$emmeans$corr, diagG_result$emmeans$diagG)

### ======Compare Full G and diagonal G models
cat("\n == Full G vs Diag G models with a plot ==\n")
plt <- function(df){
  min_hr <- min(df$Hour_num)
  max_hr <- max(df$Hour_num)
  df$model_type[df$model_type == "corr"] <- "Full_G"
  df$model_type[df$model_type == "diagG"] <- "diag_G"
pp <- ggplot() +
  # 95 % CI intervals for estimates
  geom_ribbon(data = df,
              aes(x = Hour_num, ymin = LCL, ymax = UCL, fill = model_type,
                  group = interaction(model_type, DayType)),
              alpha = 0.25, colour = NA) +
  # G type-level lines
  geom_line(data = df,
            aes(Hour_num, emmean, color = model_type,
                group = interaction(model_type, DayType)), linewidth = 0.2) +
  geom_point(data = df,
             aes(Hour_num, emmean, color = model_type,
                 group = interaction(model_type, DayType)), size = 1)  +
  scale_x_continuous(breaks = seq(1, 24, by = 2), limits = c(min_hr, max_hr)) +
  labs(x = "Hour", y = "Estimated % sedentary",
       title = NULL,
       color = "Corr structure", linetype = "corr structure") +
  guides(fill = "none") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right") +
  facet_wrap(~DayType)
# Add custom theme adjustments for better readability
pp <- pp + theme(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  plot.title = element_text(hjust = 0.5)
)
return(pp)
}
compare_full_diagG <- plt(df = full_diag_emmeans)
## save image for comparing G types
ggsave(filename = "results/data_qual_plots/compare_full_diagG.png")
ggsave(filename = "results/data_qual_plots/compare_full_diagG.pdf")


### Let zoom in to see the hours of higher activity levels
intense_hours <- seq(10,24, 1)
full_diag_emmeans_intense <- full_diag_emmeans %>% filter(Hour_num %in% intense_hours)
plt(df = full_diag_emmeans_intense)

cat("\n=== MODELING COMPLETED ===\n")




