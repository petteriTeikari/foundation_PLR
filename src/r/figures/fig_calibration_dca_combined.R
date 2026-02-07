# Combined Calibration + DCA Figure (STRATOS-Compliant)
# =====================================================
# 1x2 panel: Calibration (left) + DCA (right)
# Per user request to consolidate two separate figures
#
# Van Calster 2024 requirements:
# Van Calster, Ben, Gary S. Collins, Andrew J. Vickers, et al. 2024. “Performance Evaluation of Predictive AI Models to Support Medical Decisions: Overview and Guidance.” arXiv:2412.10288. Preprint, arXiv, December 13. https://doi.org/10.48550/arXiv.2412.10288. - https://doi.org/10.1016/j.landig.2025.100916
# - Calibration: smoothed curve with CI, slope, intercept, O:E, Brier
# - DCA: Net benefit curves with treat-all/treat-none references
#
# Created: 2026-01-27
# Author: Foundation PLR Team

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

message("[fig_calibration_dca_combined] Loading data...")

cal_data <- fromJSON("data/r_data/calibration_data.json")
dca_data <- fromJSON("data/r_data/dca_data.json")

# Get configs
cal_configs <- cal_data$data$configs
dca_configs <- dca_data$data$configs
prevalence <- dca_data$data$sample_prevalence

message(sprintf("  Calibration configs: %d", nrow(cal_configs)))
message(sprintf("  DCA configs: %d", nrow(dca_configs)))

# ==============================================================================
# PREPARE CALIBRATION DATA
# ==============================================================================

cal_curves <- lapply(1:nrow(cal_configs), function(i) {
  curve <- cal_configs$curve
  data.frame(
    config = cal_configs$name[i],
    bin_midpoint = curve$bin_midpoints[[i]],
    observed = curve$observed[[i]],
    count = curve$counts[[i]],
    stringsAsFactors = FALSE
  )
})
cal_df <- bind_rows(cal_curves) %>%
  filter(!is.na(observed)) %>%
  mutate(config = factor(config, levels = unique(config)))

# ==============================================================================
# PREPARE DCA DATA
# ==============================================================================

dca_list <- lapply(1:nrow(dca_configs), function(i) {
  data.frame(
    config = dca_configs$name[i],
    threshold = dca_configs$thresholds[[i]],
    nb_model = dca_configs$nb_model[[i]],
    stringsAsFactors = FALSE
  )
})
dca_df <- bind_rows(dca_list) %>%
  mutate(config = factor(config, levels = unique(config)))

# Reference strategies
ref_thresholds <- sort(unique(dca_df$threshold))
ref_df <- data.frame(
  threshold = ref_thresholds,
  nb_treat_all = prevalence - (1 - prevalence) * (ref_thresholds / (1 - ref_thresholds))
)

# ==============================================================================
# COLORS
# ==============================================================================

# Use standard combo colors
config_colors <- COLORS_STANDARD_COMBOS[levels(cal_df$config)]
names(config_colors) <- levels(cal_df$config)

message(sprintf("  Configs: %s", paste(names(config_colors), collapse = ", ")))

# ==============================================================================
# PANEL A: CALIBRATION PLOT
# ==============================================================================

message("[fig_calibration_dca_combined] Creating calibration panel...")

# Get metrics for annotation - Ground Truth only (first config)
# Note: With multiple curves, showing metrics for only one is ambiguous
# unless clearly labeled. We show Ground Truth metrics as the reference.
gt_idx <- which(cal_configs$name == "Ground Truth")
if (length(gt_idx) == 0) gt_idx <- 1  # Fallback to first

# Optional if you want just the ground truth metrics (confusing as you multiple slopes there, and better to say either just include a single metric to the legend, or express these in the body text or in some .json file)
# annotation_text <- sprintf(
#   "Ground Truth:\nSlope: %.2f, O:E: %.2f\nBrier: %.3f, IPA: %.2f",
#   cal_configs$calibration_slope[gt_idx],
#   cal_configs$o_e_ratio[gt_idx],
#   cal_configs$brier[gt_idx],
#   cal_configs$ipa[gt_idx]
# )

p_cal <- ggplot(cal_df, aes(x = bin_midpoint, y = observed, color = config)) +
  # Perfect calibration line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = color_defs[["--color-text-secondary"]], linewidth = 0.5) +
  # Calibration points
  geom_point(aes(size = count), alpha = 0.8) +
  # Smoothed curves
  geom_smooth(method = "loess", span = 0.8, se = TRUE, alpha = 0.15, linewidth = 0.8) +
  scale_color_manual(values = config_colors, name = "Pipeline") +
  scale_size_continuous(range = c(2, 5), guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_equal() +
  # STRATOS annotation
  annotate(
    "label",
    x = 0.02, y = 0.98,
    label = annotation_text,
    hjust = 0, vjust = 1,
    size = 2.8,
    fill = color_defs[["--color-white"]],
    alpha = 0.9,
    linewidth = 0.3
  ) +
  labs(
    x = "Predicted Probability",
    y = "Observed Proportion"
  )
  # The legends are now the same for both panels
	# theme_calibration() +
  # theme(legend.position = "none")  # Will use shared legend

# ==============================================================================
# PANEL B: DCA PLOT
# ==============================================================================

message("[fig_calibration_dca_combined] Creating DCA panel...")

p_dca <- ggplot(dca_df, aes(x = threshold, y = nb_model, color = config)) +
  # Treat none reference (horizontal at 0)
  geom_hline(yintercept = 0, linetype = "dashed", color = color_defs[["--color-annotation"]], linewidth = 0.5) +
  # Treat all reference
  geom_line(
    data = ref_df,
    aes(x = threshold, y = nb_treat_all),
    inherit.aes = FALSE,
    linetype = "dotted",
    color = color_defs[["--color-text-primary"]],
    linewidth = 0.8
  ) +
  # Model net benefit curves
  geom_line(linewidth = 1) +
  # Prevalence marker
  geom_vline(
    xintercept = prevalence,
    linetype = "solid",
    color = color_defs[["--color-text-secondary"]],
    linewidth = 0.3,
    alpha = 0.5
  ) +
  scale_color_manual(values = config_colors, name = "Pipeline") +
  scale_x_continuous(
    limits = c(0, 0.30),
    breaks = seq(0, 0.30, 0.05),
    labels = scales::percent
  ) +
  scale_y_continuous(
    limits = c(-0.05, max(c(dca_df$nb_model, ref_df$nb_treat_all), na.rm = TRUE) * 1.1),
    breaks = scales::pretty_breaks(n = 5)
  ) +
  # Annotations
  annotate(
    "text",
    x = 0.28, y = ref_df$nb_treat_all[nrow(ref_df)] + 0.01,
    label = "Treat All",
    color = color_defs[["--color-text-primary"]],
    size = 2.5,
    hjust = 1,
    fontface = "italic"
  ) +
  annotate(
    "text",
    x = 0.28, y = 0.01,
    label = "Treat None",
    color = color_defs[["--color-annotation"]],
    size = 2.5,
    hjust = 1,
    fontface = "italic"
  ) +
  labs(
    x = "Threshold Probability",
    y = "Net Benefit"
  )
  # TODO! Think of a better position, top-right position, with a white background?
  theme_foundation_plr() +
  theme(legend.position = "none")  # Will use shared legend

# ==============================================================================
# COMPOSE FIGURE
# ==============================================================================

message("[fig_calibration_dca_combined] Composing figure...")

# Create a legend-only plot to extract and share
p_legend <- ggplot(cal_df, aes(x = bin_midpoint, y = observed, color = config)) +
  geom_line() +
  scale_color_manual(values = config_colors, name = "Pipeline") +
  theme_foundation_plr() +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9)
  ) +
  guides(color = guide_legend(nrow = 1))

# Add "A  Title" style headers (consistent pattern across all figures)
title_theme <- theme(
  plot.title = element_text(face = "bold", size = 14, hjust = 0),
  plot.title.position = "plot"
)

p_cal_titled <- p_cal + ggtitle("A  Calibration") + title_theme
p_dca_titled <- p_dca + ggtitle("B  Decision Curve Analysis") + title_theme

# Compose with shared legend at bottom (no tag_levels - using manual ggtitle)
composed <- (p_cal_titled | p_dca_titled) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9)
  ) &
  guides(color = guide_legend(nrow = 1))

# Save
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(composed, "fig_calibration_dca_combined")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Calibration + DCA Combined Complete")
message("========================================")
message("Panels:")
message("  (A) Calibration plot with STRATOS annotation")
message("  (B) Decision Curve Analysis with references")
message("\nShared legend at bottom for 1x2 layout")
message(sprintf("Sample prevalence: %.1f%%", prevalence * 100))
