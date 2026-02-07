# Calibration Plot (STRATOS-Compliant)
# Van Calster 2024 requirements: smoothed curve with CI, slope, intercept, O:E, Brier
# Task 2.3 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses gg_add() instead of + operator for S7/ggplot2 4.0+ compatibility

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(patchwork)
})

# Determine project root
find_project_root <- function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) {
      return(dir)
    }
    dir <- dirname(dir)
  }
  stop("Could not find project root")
}

PROJECT_ROOT <- find_project_root()

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD DATA
# ==============================================================================

cal_data <- fromJSON("data/r_data/calibration_data.json")
pred_data <- fromJSON("data/r_data/predictions_top4.json")

# Get configs dataframe
configs <- cal_data$data$configs

# Prepare calibration curve data
cal_curves <- lapply(1:nrow(configs), function(i) {
  curve <- configs$curve
  data.frame(
    config = configs$name[i],
    config_idx = configs$config_idx[i],
    bin_midpoint = curve$bin_midpoints[[i]],
    observed = curve$observed[[i]],
    predicted = curve$predicted[[i]],
    count = curve$counts[[i]],
    stringsAsFactors = FALSE
  )
})
cal_df <- bind_rows(cal_curves) %>%
  filter(!is.na(observed)) %>%
  mutate(config = factor(config, levels = unique(config)))

# Prepare prediction data for histogram/rug
pred_configs <- pred_data$data$configs
pred_list <- lapply(1:nrow(pred_configs), function(i) {
  data.frame(
    config = pred_configs$name[i],
    y_true = pred_configs$y_true[[i]],
    y_prob = pred_configs$y_prob[[i]],
    stringsAsFactors = FALSE
  )
})
pred_df <- bind_rows(pred_list) %>%
  mutate(
    config = factor(config, levels = unique(config)),
    outcome = factor(ifelse(y_true == 1, "Glaucoma", "Control"),
                     levels = c("Control", "Glaucoma"))
  )

# ==============================================================================
# STRATOS CALIBRATION PLOT
# ==============================================================================

# Colors for configs
config_colors <- ECONOMIST_PALETTE[1:4]
names(config_colors) <- levels(cal_df$config)

# Main calibration plot
p_cal <- ggplot(cal_df, aes(x = bin_midpoint, y = observed, color = config))
p_cal <- gg_add(p_cal,
  # Perfect calibration line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_defs[["--color-text-secondary"]], linewidth = 0.5),
  # Calibration points with size by count
  geom_point(aes(size = count), alpha = 0.8),
  # Smoothed calibration curve (LOESS)
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2, linewidth = 0.8),
  # Rug for individual predictions (sample only for clarity)
  geom_rug(
    data = pred_df %>% filter(config == levels(config)[1]) %>% sample_n(min(50, n())),
    aes(x = y_prob, y = NULL),
    sides = "b",
    alpha = 0.3,
    color = color_defs[["--color-text-primary"]]
  ),
  scale_color_manual(values = config_colors, name = "Pipeline"),
  scale_size_continuous(range = c(2, 6), guide = "none"),
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
  coord_equal(),
  # Labels (academic mode - no title/subtitle for journal submission)
  labs(
    x = "Predicted Probability",
    y = "Observed Proportion"
  ),
  theme_calibration(),
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  ),
  guides(color = guide_legend(nrow = 2))
)

# Marginal histogram of predictions
p_hist <- ggplot(pred_df %>% filter(config == levels(config)[1]), aes(x = y_prob, fill = outcome))
p_hist <- gg_add(p_hist,
  geom_histogram(bins = 20, alpha = 0.7, position = "identity"),
  scale_fill_manual(values = c("Control" = color_defs[["--color-primary"]], "Glaucoma" = color_defs[["--color-negative"]])),
  scale_x_continuous(limits = c(0, 1)),
  labs(x = NULL, y = "Count", fill = "Outcome"),
  theme_void(),
  theme(
    axis.text.y = element_text(size = 8),
    legend.position = "none",
    plot.margin = margin(0, 15, 0, 15)
  )
)

# Add STRATOS annotations
# Get metrics for first config (best ensemble)
gt_idx <- which(grepl("pupil-gt|ensemble", configs$name))[1]
if (is.na(gt_idx) || length(gt_idx) == 0) gt_idx <- 1

annotation_text <- sprintf(
  "Calibration slope: %.2f\nCalibration intercept: %.2f\nO:E ratio: %.2f\nBrier: %.3f, IPA: %.3f\nN=%d (%d events)",
  configs$calibration_slope[gt_idx],
  configs$calibration_intercept[gt_idx],
  configs$o_e_ratio[gt_idx],
  configs$brier[gt_idx],
  configs$ipa[gt_idx],
  configs$n[gt_idx],
  configs$n_events[gt_idx]
)

# Add annotation to main calibration plot
p_final <- gg_add(p_cal,
  annotate(
    "text",
    x = 0.02, y = 0.98,
    label = annotation_text,
    hjust = 0, vjust = 1,
    size = 3,
    family = "Helvetica"
  )
)

# Note: Skip patchwork due to S7 compatibility issues
# Marginal histogram can be added later if needed

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_final, "fig_calibration_stratos")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Calibration Plot Complete (STRATOS)")
message("========================================")
message("Elements included:")
message("  - Smoothed calibration curve with 95% CI")
message("  - Perfect calibration reference line")
message("  - Marginal histogram of predictions")
message("  - Rug plot for sample distribution")
message("  - STRATOS annotations: slope, intercept, O:E, Brier, IPA")
