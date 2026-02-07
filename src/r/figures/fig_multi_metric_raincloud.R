# Figure 2: Multi-Metric Raincloud 2×2 Panel
# ==========================================
# Shows preprocessing effects across STRATOS metrics:
# - (A) AUROC by Pipeline Type (Discrimination)
# - (B) Scaled Brier (IPA) by Type (Overall/Calibration)
# - (C) Net Benefit @ pt=0.10 by Type (Clinical Utility)
# - (D) O:E Ratio by Type (Calibration-in-the-large)
#
# Style: Uses figure_style.yaml for UPPERCASE panel labels (A, B, C, D)
# AUROC MCID: 0.05 (not 0.02)
# Net benefit threshold: pt=0.10 recommended
#
# Created: 2026-01-27
# Updated: 2026-01-28 (harmonized panel labels via figure_style.yaml)
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(ggdist)  # For raincloud half-eye distributions
})

# ==============================================================================
# SETUP
# ==============================================================================

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
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/load_style.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load colors from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# ==============================================================================
# PIPELINE TYPE CLASSIFICATION (via YAML category_loader)
# ==============================================================================

# Use categorize_outlier_methods() from category_loader.R
# This replaces hardcoded case_when() patterns with YAML-driven categorization.
# Colors from COLORS_PIPELINE_TYPE in color_palettes.R (which loads from YAML)

# ==============================================================================
# SINGLE METRIC RAINCLOUD
# ==============================================================================

#' Create a single raincloud panel for one metric
#'
#' @param data Data frame with pipeline_type and metric column
#' @param metric Name of the metric column
#' @param metric_label Display label for the metric
#' @param panel_title Panel title (e.g., "Discrimination") - appears after tag letter
#' @param higher_is_better If TRUE, higher values are better (default TRUE)
#' @param show_legend Whether to show legend (default FALSE)
#' @return ggplot object
#' @export
create_raincloud_metric <- function(data, metric, metric_label, panel_title = NULL,
                                    higher_is_better = TRUE, show_legend = FALSE) {

  type_colors <- COLORS_PIPELINE_TYPE

  # Filter out any NA or Unknown categories (data quality issue)
  data <- data[!is.na(data$pipeline_type) & data$pipeline_type != "Unknown", ]

  # Order factor levels appropriately (includes all 5 valid categories)
  data$pipeline_type <- factor(
    data$pipeline_type,
    levels = c("Ground Truth", "Ensemble", "Foundation Model", "Deep Learning", "Traditional")
  )

  # Drop unused factor levels to prevent gaps
  data$pipeline_type <- droplevels(data$pipeline_type)

  # Build the plot with raincloud components
  p <- ggplot(data, aes(x = .data[[metric]], y = pipeline_type, fill = pipeline_type))
  p <- gg_add(p,
    # Half-violin (distribution shape)
    # CRITICAL: height=0.5 prevents vertical overlap between categories
    # Ground Truth (n=7) spikes high without height limit
    ggdist::stat_halfeye(
      adjust = 2.0,           # Wide bandwidth = smooth density
      height = 0.5,           # MAXIMUM height per row (prevents overlap)
      .width = 0,
      justification = -0.02,
      point_colour = NA,
      slab_linewidth = 0,
      alpha = 0.7,
      trim = TRUE
    ),
    # Boxplot (quartiles)
    geom_boxplot(
      width = 0.1,
      outlier.shape = NA,
      alpha = 0.7
    ),
    # Jittered points (individual data)
    geom_jitter(
      aes(color = pipeline_type),
      width = 0,
      height = 0.05,
      alpha = 0.5,
      size = 1.2
    ),
    # Colors
    scale_fill_manual(values = type_colors, name = "Pipeline Type"),
    scale_color_manual(values = type_colors, name = "Pipeline Type"),
    # Labels - NO subtitle, title will be combined with tag by patchwork
    labs(
      x = metric_label,
      y = NULL,
      title = panel_title
    ),
    theme_foundation_plr(),
    theme(
      axis.text.y = element_text(size = 9),
      # Title styling to match compose_figures pattern
      plot.title = element_text(size = 11, face = "plain", hjust = 0),
      plot.title.position = "plot"
    )
  )

  # Handle legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  }

  return(p)
}

# ==============================================================================
# COMBINED 2×2 RAINCLOUD PANEL
# ==============================================================================

#' Create Multi-Metric Raincloud 2×2 panel
#'
#' Shows AUROC, Scaled Brier, Net Benefit, and O:E Ratio
#' across pipeline types.
#'
#' @param data Data frame with pipeline_type and metric columns
#' @param style Optional pre-loaded style config (from load_figure_style())
#' @return patchwork object
#' @export
create_multi_metric_raincloud <- function(data, style = NULL) {
  # Load style from YAML (single source of truth)
  if (is.null(style)) style <- load_figure_style()

  # Panel labels and titles: "A  Discrimination" style (letter + title on same line)
  # This matches fig_calibration_dca_combined.png styling
  panel_font <- style$panel_labels$font_family
  panel_size <- style$panel_labels$font_size

  # Create individual panels with combined letter+title headers
  # Using ggtitle to create "A  Title" format (NO separate patchwork tags)

  # (A) AUROC - Discrimination
  p_auroc <- create_raincloud_metric(
    data, "auroc", get_metric_name("auroc", style),
    panel_title = NULL,
    higher_is_better = TRUE
  ) + ggtitle(paste0("A  ", get_category_name("discrimination", style))) +
    theme(plot.title = element_text(family = panel_font, face = "bold", size = panel_size, hjust = 0))

  # (B) Scaled Brier (IPA) - Overall Performance
  p_ipa <- create_raincloud_metric(
    data, "scaled_brier", get_metric_name("scaled_brier", style),
    panel_title = NULL,
    higher_is_better = TRUE
  ) + ggtitle(paste0("B  ", get_category_name("overall_performance", style))) +
    theme(plot.title = element_text(family = panel_font, face = "bold", size = panel_size, hjust = 0))

  # (C) Net Benefit @ pt=0.10 - Clinical Utility
  p_nb <- create_raincloud_metric(
    data, "net_benefit_10pct", get_metric_name("net_benefit", style),
    panel_title = NULL,
    higher_is_better = TRUE
  ) + ggtitle(paste0("C  ", get_category_name("clinical_utility", style))) +
    theme(plot.title = element_text(family = panel_font, face = "bold", size = panel_size, hjust = 0))

  # (D) O:E Ratio - Calibration
  p_oe <- create_raincloud_metric(
    data, "o_e_ratio", get_metric_name("o_e_ratio", style),
    panel_title = NULL,
    higher_is_better = FALSE
  ) + ggtitle(paste0("D  ", get_category_name("calibration", style))) +
    theme(plot.title = element_text(family = panel_font, face = "bold", size = panel_size, hjust = 0))

  # Combine using patchwork (2×2 layout)
  # NO tag_levels since we manually added "A Title" style headers
  combined <- (p_auroc + p_ipa) / (p_nb + p_oe)

  return(combined)
}

# ==============================================================================
# MAIN EXECUTION (when sourced as script)
# ==============================================================================

if (sys.nframe() == 0) {
  message("Loading data from data/r_data/essential_metrics.csv...")

  # Load data
  metrics_df <- read.csv(file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv"))

  # Filter to CatBoost classifier only (fixed classifier per research question)
  cat_df <- metrics_df %>%
    filter(classifier == "CatBoost") %>%
    mutate(pipeline_type = categorize_outlier_methods(outlier_method))

  message(sprintf("Found %d CatBoost configurations", nrow(cat_df)))
  message("\nPipeline type distribution:")
  print(table(cat_df$pipeline_type))

  # Check for required columns
  required_cols <- c("auroc", "scaled_brier", "net_benefit_10pct", "o_e_ratio")
  missing_cols <- setdiff(required_cols, colnames(cat_df))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  # Summary statistics by pipeline type
  message("\n=== Summary Statistics by Pipeline Type ===")
  summary_stats <- cat_df %>%
    group_by(pipeline_type) %>%
    summarise(
      n = n(),
      auroc_mean = mean(auroc, na.rm = TRUE),
      auroc_sd = sd(auroc, na.rm = TRUE),
      ipa_mean = mean(scaled_brier, na.rm = TRUE),
      nb_mean = mean(net_benefit_10pct, na.rm = TRUE),
      oe_mean = mean(o_e_ratio, na.rm = TRUE),
      .groups = "drop"
    )
  print(summary_stats)

  # Create figure
  message("\nCreating Multi-Metric Raincloud 2×2 panel...")
  p <- create_multi_metric_raincloud(cat_df)

  # Save
  # Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
  save_publication_figure(p, "fig_multi_metric_raincloud")

  message("\n========================================")
  message("Multi-Metric Raincloud Figure Complete")
  message("========================================")
  message("Panels included:")
  message("  (a) AUROC by Pipeline Type (Discrimination)")
  message("  (b) Scaled Brier (IPA) by Type (Overall)")
  message("  (c) Net Benefit @ pt=10% by Type (Clinical Utility)")
  message("  (d) O:E Ratio by Type (Calibration-in-the-large)")
}
