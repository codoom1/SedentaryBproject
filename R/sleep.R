# improved_activity_plots.R
# Dependencies: tidyverse, scales, RColorBrewer
# install.packages(c("tidyverse","scales","RColorBrewer"))

library(tidyverse)
library(scales)
library(RColorBrewer)

# --------- User parameters (tweak as needed) ----------
data_path <- "data/project_data/final_data.csv.gz"

# IDs: set manually OR auto pick first two
 id1 <- "64229"; id2 <- "64179"
if (!exists("id1") || !exists("id2")) {
  ids_tmp <- readr::read_csv(data_path, col_types = cols(ID = col_character()), n_max = 5000)
  uniq_ids <- ids_tmp %>% distinct(ID) %>% pull(ID)
  if (length(uniq_ids) < 2) stop("Need at least two unique IDs in the dataset")
  id1 <- uniq_ids[1]; id2 <- uniq_ids[2]
}
selected_ids <- c(id1, id2)
print(paste("Selected IDs:", paste(selected_ids, collapse = ", ")))
# wake detection
wear_threshold <- 50  # percent_wear > threshold => candidate wake hour
min_run <- 2          # require >= min_run consecutive hours to count as a block

# output files and sizes
out_pdf <- "plots/activity_profiles_improved.pdf"    # vector for print
out_png_preview <- "plots/activity_profiles_preview_improved.png"  # quick check

# Customizable figure dimensions (in inches)
fig_width <- 16   # Change this value as needed
fig_height <- 8   # Change this value as needed

poster_width_in <- 48; poster_height_in <- 24
preview_width_in <- 16; preview_height_in <- 8
dpi_preview <- 200; dpi_print <- 300

# --------- Read and prepare data ----------
df <- readr::read_csv(
  data_path,
  col_types = cols(
    ID = col_character(),
    Day = col_date(),
    Hour = col_double(),
    percent_sleep_nonwear = col_double(),
    percent_wear = col_double(),
    percent_sitting = col_double(),
    percent_not_sitting = col_double()
  )
)

head(unique(df$ID))

## Load the nosleep data and merge to df on ID, Day, Hour
nosleep_data <- readr::read_csv('constants/nosleep_data.csv.gz',
                                col_types = cols(
                                  ID = col_character(),
                                  Date = col_date(),
                                  Hour = col_double(),
                                  PercentSedentary = col_double()
                                )) %>%
  rename(Day = Date, percent_sedentary = PercentSedentary) %>%
  select(ID, Day, Hour, percent_sedentary)

head(nosleep_data)
# Merge nosleep data with df
df <- df %>%
  left_join(nosleep_data, by = c("ID", "Day", "Hour"))

head(df$percent_sedentary)
head(unique(df$ID))
avg_hourly <- df %>%
  filter(ID %in% selected_ids) %>%
  group_by(ID, Hour) %>%
  summarise(across(starts_with("percent_"), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  arrange(ID, Hour)

head(avg_hourly)

# helper: find contiguous TRUE runs (returns start/end hours for runs >= min_run)
find_runs <- function(log_vec, hrs, min_run = 1) {
  if (length(log_vec) == 0) return(tibble(start = numeric(0), end = numeric(0), len = integer(0)))
  r <- rle(as.integer(log_vec))
  ends <- cumsum(r$lengths); starts <- ends - r$lengths + 1
  runs <- tibble(is_true = r$values == 1, start_idx = starts, end_idx = ends, len = r$lengths)
  runs <- runs %>% filter(is_true & len >= min_run)
  if (nrow(runs) == 0) return(tibble(start = numeric(0), end = numeric(0), len = integer(0)))
  runs %>% mutate(start = hrs[start_idx], end = hrs[end_idx]) %>% select(start, end, len)
}

# detect main wake block per ID (longest run)
wake_blocks <- avg_hourly %>%
  group_by(ID) %>%
  summarise(
    runs = list(find_runs(percent_wear > wear_threshold, Hour, min_run)), 
    .groups = "drop"
  ) %>%
  mutate(
    main = map(runs, ~ {
      if (nrow(.x) == 0) {
        tibble(start = NA_real_, end = NA_real_, len = NA_integer_)
      } else {
        .x %>% arrange(desc(len)) %>% slice(1)
      }
    })
  ) %>%
  unnest(main) %>%
  select(ID, start, end, len)
wake_blocks


# rect_df for shading
rect_df <- wake_blocks %>% filter(!is.na(start) & !is.na(end)) %>%
  mutate(xmin = start - 0.5, xmax = end + 0.5, ymin = -Inf, ymax = Inf)

# long format for plotting multiple metrics
plot_long <- avg_hourly %>%
  pivot_longer(cols = starts_with("percent_"), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric,
                         percent_sleep_nonwear = "Sleep",
                         percent_wear = "Wake",
                         percent_sitting = "Sitting",
                         percent_sedentary = "Sedentary",
                         percent_not_sitting = "Not sitting"))

# color palettes
id_colors <- brewer.pal(3, "Dark2")[1:2]   # 2 colors for IDs
metric_colors <- c("Worn (Wake)" = "#1b9e77", "Not sitting" = "#d95f02",
                   "Sitting" = "#7570b3", "Sleep / Non-wear" = "#66a61e")

# ---------- THEME helper ----------
make_theme <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size * 1.8, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = base_size * 1.05, hjust = 0.5),
      axis.title = element_text(size = base_size * 1.05),
      axis.text = element_text(size = base_size * 0.9),
      strip.text = element_text(size = base_size * 1.15, face = "bold"),
      legend.position = "bottom",
      legend.key.width = unit(1.2, "cm")
    )
}

plot_long_new <- plot_long %>% filter(metric %in% c('Sleep','Sedentary'))
# Ensure ID is character in both datasets
plot_long_new <- plot_long_new %>% mutate(ID = as.character(ID))
rect_df2 <- rect_df %>% 
  mutate(ID = as.character(ID),
         xmax = pmin(xmax, 24))  # Cap xmax at 24 to avoid clipping

# Ensure both IDs are present
all_ids <- sort(unique(plot_long_new$ID))

# Create anonymous labels for IDs
id_labels <- setNames(c("Participant A", "Participant B"), all_ids)

# Apply anonymous labels to both datasets
plot_long_new <- plot_long_new %>% 
  mutate(ID = factor(ID, levels = all_ids, labels = c("Participant A", "Participant B")))
rect_df2 <- rect_df2 %>% 
  mutate(ID = factor(ID, levels = all_ids, labels = c("Participant A", "Participant B")))

# Split rect_df2 by ID for separate geom layers (workaround for faceting issue)
rect_id1 <- rect_df2 %>% filter(ID == "Participant A")
rect_id2 <- rect_df2 %>% filter(ID == "Participant B")

p_facet <- ggplot(plot_long_new, aes(x = Hour, y = value, color = metric, group = metric)) +
  facet_wrap(~ID, nrow = 1) +
  geom_rect(
    data = rect_id1,
    aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 100),
    inherit.aes = FALSE,
    fill = "#FFEB3B", color = NA, alpha = 0.35  # Bright yellow
  ) +
  geom_rect(
    data = rect_id2,
    aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 100),
    inherit.aes = FALSE,
    fill = "#4FC3F7", color = NA, alpha = 0.35  # Bright cyan
  ) +
  geom_point(size = 4) +  # Increased from 1 to 3
  geom_line(linewidth = 2) +  # Increased line thickness
  geom_abline(intercept = 50, slope = 0, linewidth = 0.8, linetype = "dashed") +
  scale_x_continuous(breaks = c(seq(1, 24, 3), 24), limits = c(1, 24)) +
  scale_y_continuous(labels = scales::percent_format(scale = 1), breaks = seq(0, 100, 10), limits = c(0, 100)) +
  labs(x = "Hour of day", y = "Percent (%)", color = NULL) +
  guides(color = guide_legend(nrow = 1, byrow = FALSE)) +
  theme_classic(base_size = 25) +  # Increased base font size
  theme(
    # Thicker axes lines
    axis.line = element_line(linewidth = 1.2, color = "black"),
    axis.ticks = element_line(linewidth = 1, color = "black"),
    axis.ticks.length = unit(0.3, "cm"),
    
    # Larger text elements
    axis.title = element_text(size = 25, face = "bold"),
    axis.text = element_text(size = 16, color = "black"),
    strip.text = element_text(size = 18, face = "bold"),
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 20, face = "bold"),
    
    # Thicker panel border
    panel.border = element_rect(color = "black", linewidth = 1.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # Legend adjustments
    legend.position = "bottom",
    legend.key.size = unit(1.5, "cm")
  )

# Display the plot
print(p_facet)

# Save to PDF with customizable dimensions
cat("\nSaving plot to PDF:", out_pdf, "\n")
cat("Dimensions:", fig_width, "x", fig_height, "inches\n")
ggsave(out_pdf, plot = p_facet, width = fig_width, height = fig_height, units = "in", device = "pdf")

# Optional: also save a PNG preview
cat("Saving PNG preview:", out_png_preview, "\n")
ggsave(out_png_preview, plot = p_facet, width = fig_width, height = fig_height, units = "in", dpi = 300)
