# Figure 1: STRATOS Core 2×2 Panel
# =================================
# Combined figure with ROC, Calibration, DCA, and Probability Distributions
#
# Van Calster 2024 requirements:
# - Smoothed calibration curve with CI
# - Annotation box with slope, intercept, O:E, Brier, IPA
# - DCA with treat-all and treat-none reference lines
# - Probability distributions by outcome
#
# Expert review requirements (2026-01-27):
# - Panel labels: UPPERCASE (A,B,C,D) with bold 14pt font (via figure_style.yaml)
# - Net benefit threshold: pt=0.10 recommended
#
# Style: Uses figure_style.yaml for panel labels (UPPERCASE A, B, C, D)
#
# Created: 2026-01-27
# Updated: 2026-01-28 (harmonized panel labels via figure_style.yaml)
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(patchwork)
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

# Source dependencies
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/load_style.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# PIPELINE COLOR HELPER
# ==============================================================================
# Colors from centralized COLORS_STANDARD_COMBOS in color_palettes.R

#' Ensure config names are standard IDs
#'
#' Data exports now use standard IDs directly (ground_truth, best_ensemble, etc.)
#' This function validates and passes through standard IDs.
#'
#' @param names Character vector of config names to map
#' @return Character vector of standard IDs
map_config_to_standard_id <- function(names) {
  # Standard IDs that are valid
  valid_ids <- c("ground_truth", "best_ensemble", "best_single_fm", "traditional")

  sapply(names, function(n) {
    if (n %in% valid_ids) {
      n  # Already a standard ID
    } else {
      warning(sprintf("Unknown config name: %s", n))
      n  # Return as-is
    }
  }, USE.NAMES = FALSE)
}

# ==============================================================================
# PANEL 1: ROC CURVES
# ==============================================================================

#' Create ROC panel with CI bands
#'
#' @param roc_data Data frame with config, fpr, tpr columns
#' @param show_legend Whether to show legend (default TRUE)
#' @return ggplot object
#' @export
create_roc_panel <- function(roc_data, show_legend = TRUE) {
  # Get pipeline colors
  pipeline_colors <- COLORS_STANDARD_COMBOS
  # Ensure config order matches colors
  configs <- unique(roc_data$config)
  color_mapping <- pipeline_colors[configs]
  names(color_mapping) <- configs

  # Build the plot
  p <- ggplot(roc_data, aes(x = fpr, y = tpr, color = config))
  p <- gg_add(p,
    # Diagonal reference line (random classifier)
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = COLOR_REFERENCE_LINE, linewidth = 0.5),
    # ROC curves
    geom_line(linewidth = 0.8),
    # Colors
    scale_color_manual(values = color_mapping, name = "Pipeline"),
    # Axes
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0.01)),
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0.01, 0.01)),
    coord_equal(),
    # Labels
    labs(
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ),
    theme_foundation_plr()
  )

  # Handle legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p, theme(legend.position = "bottom", legend.direction = "horizontal"))
  }

  return(p)
}

# ==============================================================================
# PANEL 2: CALIBRATION PLOT
# ==============================================================================

#' Create calibration panel with STRATOS annotation box
#'
#' @param cal_data Data frame with config, bin_midpoint, observed, count columns
#' @param cal_metrics Data frame with config, calibration_slope, calibration_intercept,
#'                    o_e_ratio, brier, scaled_brier columns
#' @param highlight_config Which config to show in annotation (default first)
#' @param show_legend Whether to show legend (default TRUE)
#' @return ggplot object
#' @export
create_calibration_panel <- function(cal_data, cal_metrics, highlight_config = NULL, show_legend = TRUE) {
  # Get pipeline colors
  pipeline_colors <- COLORS_STANDARD_COMBOS
  configs <- unique(cal_data$config)
  color_mapping <- pipeline_colors[configs]
  names(color_mapping) <- configs

  # Determine which config to highlight in annotation
  if (is.null(highlight_config)) {
    highlight_config <- configs[1]
  }

  # Get metrics for highlighted config
  highlight_metrics <- cal_metrics[cal_metrics$config == highlight_config, ]

  # Build annotation text with STRATOS-required metrics
  if (nrow(highlight_metrics) > 0) {
    annotation_text <- sprintf(
      "Calibration slope: %.2f\nCalibration intercept: %.2f\nO:E ratio: %.2f\nBrier: %.3f\nScaled Brier (IPA): %.2f",
      highlight_metrics$calibration_slope[1],
      highlight_metrics$calibration_intercept[1],
      highlight_metrics$o_e_ratio[1],
      highlight_metrics$brier[1],
      highlight_metrics$scaled_brier[1]
    )
  } else {
    annotation_text <- ""
  }

  # Build the plot
  p <- ggplot(cal_data, aes(x = bin_midpoint, y = observed, color = config))
  p <- gg_add(p,
    # Perfect calibration line
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = COLOR_ANNOTATION_SECONDARY, linewidth = 0.5),
    # Calibration points with size by count
    geom_point(aes(size = count), alpha = 0.7),
    # Smoothed calibration curve (LOESS)
    geom_smooth(method = "loess", span = 0.75, se = TRUE, alpha = 0.15, linewidth = 0.8),
    # Colors
    scale_color_manual(values = color_mapping, name = "Pipeline"),
    scale_size_continuous(range = c(2, 6), guide = "none"),
    # Axes
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
    coord_equal(),
    # STRATOS annotation box (required)
    annotate(
      "label",
      x = 0.02, y = 0.98,
      label = annotation_text,
      hjust = 0, vjust = 1,
      size = 2.8,
      family = "Helvetica",
      fill = color_defs[["--color-white"]],
      alpha = 0.85,
      linewidth = 0.3
    ),
    # Labels
    labs(
      x = "Predicted Probability",
      y = "Observed Proportion"
    ),
    theme_calibration()
  )

  # Handle legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p, theme(legend.position = "bottom", legend.direction = "horizontal"))
  }

  return(p)
}

# ==============================================================================
# PANEL 3: DECISION CURVE ANALYSIS
# ==============================================================================

#' Create DCA panel with treat-all/treat-none references
#'
#' @param dca_data Data frame with config, threshold, nb_model columns
#' @param prevalence Sample prevalence for treat-all calculation
#' @param show_legend Whether to show legend (default TRUE)
#' @return ggplot object
#' @export
create_dca_panel <- function(dca_data, prevalence, show_legend = TRUE) {
  # Get pipeline colors
  pipeline_colors <- COLORS_STANDARD_COMBOS
  configs <- unique(dca_data$config)
  color_mapping <- pipeline_colors[configs]
  names(color_mapping) <- configs

  # Create reference strategy data
  thresholds <- sort(unique(dca_data$threshold))
  ref_df <- data.frame(
    threshold = thresholds,
    nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds)),
    nb_treat_none = 0
  )

  # Build the plot
  p <- ggplot(dca_data, aes(x = threshold, y = nb_model, color = config))
  p <- gg_add(p,
    # Treat none reference (horizontal at NB=0) - REQUIRED by STRATOS
    geom_hline(yintercept = 0, linetype = "dashed", color = COLOR_REFERENCE_LINE, linewidth = 0.5),
    # Treat all reference - REQUIRED by STRATOS
    geom_line(
      data = ref_df,
      aes(x = threshold, y = nb_treat_all),
      inherit.aes = FALSE,
      linetype = "dotted",
      color = COLOR_ANNOTATION_TEXT,
      linewidth = 0.8
    ),
    # Model net benefit curves
    geom_line(linewidth = 1),
    # Prevalence marker
    geom_vline(
      xintercept = prevalence,
      linetype = "solid",
      color = COLOR_ANNOTATION_SECONDARY,
      linewidth = 0.3,
      alpha = 0.5
    ),
    # Recommended threshold marker (pt=0.10 per expert review)
    geom_vline(
      xintercept = 0.10,
      linetype = "dotted",
      color = COLOR_EMPHASIS,
      linewidth = 0.3
    ),
    # Colors
    scale_color_manual(values = color_mapping, name = "Pipeline"),
    # Axes (X-axis 0.05-0.30 per STRATOS recommendation)
    scale_x_continuous(
      limits = c(0, 0.30),
      breaks = seq(0, 0.30, 0.05),
      labels = scales::percent
    ),
    scale_y_continuous(
      limits = c(-0.05, max(c(dca_data$nb_model, ref_df$nb_treat_all), na.rm = TRUE) * 1.1),
      breaks = scales::pretty_breaks(n = 5)
    ),
    # Annotations for reference lines
    annotate(
      "text",
      x = 0.28, y = ref_df$nb_treat_all[which.min(abs(ref_df$threshold - 0.28))] + 0.01,
      label = "Treat All",
      color = COLOR_ANNOTATION_TEXT,
      size = 2.5,
      hjust = 1,
      fontface = "italic"
    ),
    annotate(
      "text",
      x = 0.28, y = 0.01,
      label = "Treat None",
      color = COLOR_REFERENCE_LINE,
      size = 2.5,
      hjust = 1,
      fontface = "italic"
    ),
    # Labels
    labs(
      x = "Threshold Probability",
      y = "Net Benefit"
    ),
    theme_foundation_plr()
  )

  # Handle legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p, theme(legend.position = "bottom", legend.direction = "horizontal"))
  }

  return(p)
}

# ==============================================================================
# PANEL 4: PROBABILITY DISTRIBUTIONS
# ==============================================================================

#' Create probability distribution panel by outcome
#'
#' @param pred_data Data frame with config, y_true, y_prob columns
#' @param show_legend Whether to show legend (default TRUE)
#' @return ggplot object
#' @export
create_prob_dist_panel <- function(pred_data, show_legend = TRUE) {
  # Add outcome labels
  pred_df <- pred_data %>%
    mutate(outcome = factor(
      ifelse(y_true == 1, "Glaucoma", "Control"),
      levels = c("Control", "Glaucoma")
    ))

  # Get unique configs for faceting
  configs <- unique(pred_df$config)

  # Build the plot - density ridges or stacked histograms
  p <- ggplot(pred_df, aes(x = y_prob, fill = outcome))
  p <- gg_add(p,
    geom_density(alpha = 0.6, position = "identity", linewidth = 0.3),
    facet_wrap(~config, ncol = 2, scales = "free_y"),
    # Outcome colors
    scale_fill_manual(
      values = COLORS_OUTCOME,      name = "Outcome"
    ),
    # Axes
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)),
    # Labels
    labs(
      x = "Predicted Probability",
      y = "Density"
    ),
    theme_foundation_plr(),
    theme(
      strip.text = element_text(size = 9, face = "bold")
    )
  )

  # Handle legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p, theme(legend.position = "bottom", legend.direction = "horizontal"))
  }

  return(p)
}

# ==============================================================================
# COMBINED 2×2 PANEL
# ==============================================================================

#' Create STRATOS Core 2×2 panel figure
#'
#' Combines ROC, Calibration, DCA, and Probability Distribution panels
#' per Van Calster 2024 STRATOS guidelines.
#'
#' @param data_list List containing roc, calibration, calibration_metrics, dca, prevalence, prob_dist
#' @param show_panel_legends Whether to show legends in individual panels (default FALSE)
#' @param style Optional pre-loaded style config (from load_figure_style())
#' @return patchwork object
#' @export
create_stratos_core_panel <- function(data_list, show_panel_legends = FALSE, style = NULL) {
  # Load style from YAML (single source of truth)
  if (is.null(style)) style <- load_figure_style()

  # Panel title theme (consistent across all figures)
  title_theme <- theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.title.position = "plot"
  )

  # Create individual panels with "A  Title" style headers
  p_roc <- create_roc_panel(data_list$roc, show_legend = show_panel_legends) +
    ggtitle("A  ROC Curves") + title_theme

  p_cal <- create_calibration_panel(
    data_list$calibration,
    data_list$calibration_metrics,
    show_legend = show_panel_legends
  ) + ggtitle("B  Calibration") + title_theme

  p_dca <- create_dca_panel(data_list$dca, data_list$prevalence, show_legend = show_panel_legends) +
    ggtitle("C  Decision Curve Analysis") + title_theme

  p_prob <- create_prob_dist_panel(data_list$prob_dist, show_legend = show_panel_legends) +
    ggtitle("D  Probability Distributions") + title_theme

  # Combine using patchwork (2×2 layout)
  # Panel titles use ggtitle() - no separate tag_levels needed
  combined <- (p_roc + p_cal) / (p_dca + p_prob)

  return(combined)
}

# ==============================================================================
# MAIN EXECUTION (when sourced as script)
# ==============================================================================

if (sys.nframe() == 0) {
  message("Loading data from data/r_data/...")

  # Load data
  roc_rc_data <- fromJSON(file.path(PROJECT_ROOT, "data/r_data/roc_rc_data.json"))
  cal_data <- fromJSON(file.path(PROJECT_ROOT, "data/r_data/calibration_data.json"))
  dca_data <- fromJSON(file.path(PROJECT_ROOT, "data/r_data/dca_data.json"))
  pred_data <- fromJSON(file.path(PROJECT_ROOT, "data/r_data/predictions_top4.json"))
  metrics_df <- read.csv(file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv"))

  # Prepare ROC data - configs is a data.frame, not a list
  configs_df <- roc_rc_data$data$configs
  roc_df <- do.call(rbind, lapply(1:nrow(configs_df), function(i) {
    data.frame(
      config = configs_df$id[i],
      fpr = configs_df$roc$fpr[[i]],
      tpr = configs_df$roc$tpr[[i]],
      stringsAsFactors = FALSE
    )
  }))
  # Filter to standard 4 configs
  roc_df <- roc_df %>% filter(config %in% c("ground_truth", "best_ensemble", "best_single_fm", "traditional"))

  # Prepare calibration data - map truncated names to standard IDs
  cal_configs <- cal_data$data$configs
  cal_df <- do.call(rbind, lapply(1:nrow(cal_configs), function(i) {
    data.frame(
      config = cal_configs$name[i],
      bin_midpoint = cal_configs$curve$bin_midpoints[[i]],
      observed = cal_configs$curve$observed[[i]],
      count = cal_configs$curve$counts[[i]],
      stringsAsFactors = FALSE
    )
  })) %>%
    filter(!is.na(observed)) %>%
    mutate(config = map_config_to_standard_id(config))
  # Note: No filter applied - use all available standard configs

  # Calibration metrics from CSV (already computed in extraction pipeline)
  cal_metrics_df <- metrics_df %>%
    filter(outlier_method == "pupil-gt" | grepl("ensemble", outlier_method)) %>%
    filter(classifier == "CatBoost") %>%
    select(outlier_method, imputation_method, calibration_slope, calibration_intercept, o_e_ratio, brier, scaled_brier) %>%
    head(4) %>%
    mutate(config = c("ground_truth", "best_ensemble", "best_single_fm", "traditional")[1:n()])

  # Prepare DCA data - map truncated names to standard IDs
  dca_configs <- dca_data$data$configs
  dca_df <- do.call(rbind, lapply(1:nrow(dca_configs), function(i) {
    data.frame(
      config = dca_configs$name[i],
      threshold = dca_configs$thresholds[[i]],
      nb_model = dca_configs$nb_model[[i]],
      stringsAsFactors = FALSE
    )
  })) %>%
    mutate(config = map_config_to_standard_id(config))
  # Note: No filter - use all available configs
  prevalence <- dca_data$data$sample_prevalence

  # Prepare prediction data - map truncated names to standard IDs
  pred_configs <- pred_data$data$configs
  prob_df <- do.call(rbind, lapply(1:nrow(pred_configs), function(i) {
    data.frame(
      config = pred_configs$name[i],
      y_true = pred_configs$y_true[[i]],
      y_prob = pred_configs$y_prob[[i]],
      stringsAsFactors = FALSE
    )
  })) %>%
    mutate(config = map_config_to_standard_id(config))
  # Note: No filter - use all available configs

  # Create combined data list
  data_list <- list(
    roc = roc_df,
    calibration = cal_df,
    calibration_metrics = cal_metrics_df,
    dca = dca_df,
    prevalence = prevalence,
    prob_dist = prob_df
  )

  # Create figure
  message("Creating STRATOS Core 2×2 panel...")
  p <- create_stratos_core_panel(data_list)

  # Save
  # Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
  save_publication_figure(p, "fig_stratos_core")

  message("\n========================================")
  message("STRATOS Core Figure Complete")
  message("========================================")
  message("Panels included:")
  message("  (a) ROC Curves")
  message("  (b) Calibration Plot with STRATOS annotation box")
  message("  (c) DCA with treat-all/treat-none references")
  message("  (d) Probability Distributions by outcome")
}
