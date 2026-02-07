#!/usr/bin/env Rscript
# ==============================================================================
# ROC + Risk-Coverage Combined Figure (1x2 panels)
# ==============================================================================
#
# Panel A (Left): ROC curves (TPR vs FPR) - higher AUROC = better discrimination
# Panel B (Right): Risk-Coverage curves - lower AURC = better uncertainty calibration
#
# Shows 9 model configurations (8 handpicked + Top-10 Mean aggregate)
#
# References:
#   - Geifman & El-Yaniv (2017) "Selective Classification for DNNs"
#   - OATML BDL Benchmarks (diabetic retinopathy diagnosis)
#   - torch-uncertainty AURC implementation
#   - Sanofi risk.assessr R package
#
# Data Source: data/r_data/roc_rc_data.json
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(jsonlite)
})

# Find project root
PROJECT_ROOT <- (function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  stop("Could not find project root")
})()

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# ==============================================================================
# Load Data
# ==============================================================================
message("[fig_roc_rc_combined] Loading data...")

data <- validate_data_source("roc_rc_data.json")
message("  Loaded ", data$data$n_configs, " configs from file")

# MAIN FIGURE: Filter to 5 standard preprocessing categories
# Extended combos and Top-10 Mean belong in supplementary
STANDARD_COMBO_IDS <- c("ground_truth", "best_ensemble", "best_single_fm", "deep_learning", "traditional")
data$data$configs <- Filter(function(cfg) cfg$id %in% STANDARD_COMBO_IDS, data$data$configs)
data$data$n_configs <- length(data$data$configs)
message("  Filtered to ", data$data$n_configs, " standard combos for main figure")

# Map config IDs to category names (consistent with other figures)
CATEGORY_NAMES <- c(
  "ground_truth" = "Ground Truth",
  "best_ensemble" = "Ensemble FM",
  "best_single_fm" = "Single-model FM",
  "deep_learning" = "Deep Learning",
  "traditional" = "Traditional"
)

# Add category_name to each config
for (i in seq_along(data$data$configs)) {
  cfg_id <- data$data$configs[[i]]$id
  data$data$configs[[i]]$category_name <- CATEGORY_NAMES[[cfg_id]]
}

# ==============================================================================
# Prepare Data
# ==============================================================================

# Build ROC dataframe with CI if available
build_roc_df <- function(configs) {
  df_list <- lapply(configs, function(cfg) {
    fpr <- unlist(cfg$roc$fpr)
    tpr <- unlist(cfg$roc$tpr)
    n <- length(fpr)

    # Check for CI data
    has_ci <- !is.null(cfg$roc$has_ci) && cfg$roc$has_ci
    tpr_lo <- if (has_ci) unlist(cfg$roc$tpr_ci_lo) else rep(NA, n)
    tpr_hi <- if (has_ci) unlist(cfg$roc$tpr_ci_hi) else rep(NA, n)

    data.frame(
      config = rep(cfg$category_name, n),
      config_id = rep(cfg$id, n),
      fpr = fpr,
      tpr = tpr,
      tpr_lo = tpr_lo,
      tpr_hi = tpr_hi,
      auroc = rep(cfg$roc$auroc, n),
      has_ci = rep(has_ci, n),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(df_list)
}

# Build RC dataframe
build_rc_df <- function(configs) {
  df_list <- lapply(configs, function(cfg) {
    coverage <- unlist(cfg$rc$coverage)
    risk <- unlist(cfg$rc$risk)
    n <- length(coverage)
    data.frame(
      config = rep(cfg$category_name, n),
      config_id = rep(cfg$id, n),
      coverage = coverage,
      risk = risk,
      aurc = rep(cfg$rc$aurc, n),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(df_list)
}

roc_df <- build_roc_df(data$data$configs)
rc_df <- build_rc_df(data$data$configs)

message("  ROC points: ", nrow(roc_df))
message("  RC points: ", nrow(rc_df))

# Get colors from YAML
color_defs <- load_color_definitions()

# Use standard category colors (consistent with other figures)
colors <- c(
  "Ground Truth" = resolve_color("--color-category-ground-truth", color_defs),
  "Ensemble FM" = resolve_color("--color-category-ensemble", color_defs),
  "Single-model FM" = resolve_color("--color-category-foundation-model", color_defs),
  "Deep Learning" = resolve_color("--color-category-deep-learning", color_defs),
  "Traditional" = resolve_color("--color-category-traditional", color_defs)
)

# Define category order (best to worst, consistent with other figures)
category_order <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
roc_df$config <- factor(roc_df$config, levels = category_order)
rc_df$config <- factor(rc_df$config, levels = category_order)

# Create legend labels with AUROC values (for ROC panel)
legend_labels_auroc <- sapply(data$data$configs, function(cfg) {
  sprintf("%s (%.3f)", cfg$category_name, cfg$roc$auroc)
})
names(legend_labels_auroc) <- sapply(data$data$configs, function(cfg) cfg$category_name)

# Create legend labels with AURC values (for RC panel)
legend_labels_aurc <- sapply(data$data$configs, function(cfg) {
  sprintf("%s (%.3f)", cfg$category_name, cfg$rc$aurc)
})
names(legend_labels_aurc) <- sapply(data$data$configs, function(cfg) cfg$category_name)

message("  Colors: ", paste(names(colors), collapse = ", "))

# ==============================================================================
# Create ROC Panel (with AUROC legend)
# ==============================================================================
message("[fig_roc_rc_combined] Creating ROC panel...")

# Filter to configs with CI for ribbon layer
roc_ci_df <- roc_df %>% filter(has_ci)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = config, fill = config)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = color_defs[["--color-text-secondary"]], linewidth = 0.5) +
  # CI ribbons (alpha = 0.2 for subtle bands)
  geom_ribbon(
    data = roc_ci_df,
    aes(ymin = tpr_lo, ymax = tpr_hi),
    alpha = 0.2,
    color = NA
  ) +
  # Main ROC lines
  geom_line(linewidth = 0.8, alpha = 0.9) +
  scale_color_manual(values = colors, name = NULL, labels = legend_labels_auroc) +
  scale_fill_manual(values = colors, guide = "none") +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  coord_equal() +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  ggtitle("A  ROC Curves") +
  theme_foundation_plr() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.title.position = "plot",
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 7),
    legend.key.width = unit(0.6, "cm"),
    plot.margin = margin(5, 10, 5, 5)
  ) +
  guides(color = guide_legend(nrow = 2))

# ==============================================================================
# Create RC Panel (with AURC legend)
# ==============================================================================
message("[fig_roc_rc_combined] Creating RC panel...")

p_rc <- ggplot(rc_df, aes(x = coverage, y = risk, color = config)) +
  geom_line(linewidth = 0.8, alpha = 0.9) +
  scale_color_manual(values = colors, name = NULL, labels = legend_labels_aurc) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(
    x = "Coverage",
    y = "Risk (Error Rate)"
  ) +
  ggtitle("B  Risk-Coverage") +
  theme_foundation_plr() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.title.position = "plot",
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 7),
    legend.key.width = unit(0.6, "cm"),
    plot.margin = margin(5, 5, 5, 10)
  ) +
  guides(color = guide_legend(nrow = 2))

# ==============================================================================
# Compose Figure with Separate Legends Per Panel
# ==============================================================================
message("[fig_roc_rc_combined] Composing figure...")

# Each panel keeps its own legend (no guides = "collect")
# Panel titles use ggtitle() with "A  Title" format (no separate tag_levels)
composed <- (p_roc | p_rc) +
  plot_layout(widths = c(1, 1))

# ==============================================================================
# Save Figure (using figure system)
# ==============================================================================
message("[fig_roc_rc_combined] Saving figure...")

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(
  composed,
  "fig_roc_rc_combined"
)

message("[fig_roc_rc_combined] DONE")
