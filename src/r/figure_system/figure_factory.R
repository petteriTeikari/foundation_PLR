# Figure Factory - Flexible Decomposable Figure Components
# =========================================================
# Creates individual ggplot2 objects that can be used standalone or composed.
#
# Key design principles:
# 1. Functions return ggplot objects (not save directly)
# 2. infographic=FALSE by default (for journal submission)
# 3. show_legend parameter for composition flexibility
# 4. All styling through theme_foundation_plr
# 5. Colors loaded from configs/VISUALIZATION/colors.yaml (Single Source of Truth)
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

# Null coalescing operator (needed before sourcing common.R)
`%||%` <- function(x, y) if (is.null(x)) y else x

# Load shared utilities using robust path detection
.get_script_dir <- function() {
  # Try multiple methods to find script directory
  if (!is.null(sys.frame(1)$ofile)) {
    return(dirname(sys.frame(1)$ofile))
  }
  # Fallback: find project root and navigate
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) {
      return(file.path(dir, "src/r/figure_system"))
    }
    dir <- dirname(dir)
  }
  stop("Could not find figure_system directory")
}

# Source common utilities if available
tryCatch({
  script_dir <- .get_script_dir()
  common_path <- file.path(script_dir, "common.R")
  config_loader_path <- file.path(script_dir, "config_loader.R")
  if (file.exists(common_path)) {
    source(common_path, local = FALSE)
  }
  if (file.exists(config_loader_path)) {
    source(config_loader_path, local = FALSE)
  }
}, error = function(e) {
  # Fallback definitions if common.R can't be loaded
  message("Warning: Could not load common.R: ", e$message)
})

# Load color definitions at startup for use throughout this file
.ff_color_defs <- NULL
tryCatch({
  .ff_color_defs <- load_color_definitions()
}, error = function(e) {
  message("Warning: Could not load color definitions: ", e$message)
})

# Define local versions if not loaded from common.R
if (!exists("find_project_root", mode = "function")) {
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
}

# Note: load_category_colors and load_reference_colors are defined in common.R
# They load colors from configs/VISUALIZATION/combos.yaml (Single Source of Truth)
# If common.R failed to load, we need to source it explicitly
if (!exists("load_category_colors", mode = "function")) {
  project_root <- find_project_root()
  source(file.path(project_root, "src/r/figure_system/common.R"), local = FALSE)
}

if (!exists("ensure_theme_loaded", mode = "function")) {
  ensure_theme_loaded <- function() {
    if (!exists("theme_forest", mode = "function", envir = globalenv())) {
      project_root <- find_project_root()
      source(file.path(project_root, "src/r/theme_foundation_plr.R"), local = FALSE)
      source(file.path(project_root, "src/r/color_palettes.R"), local = FALSE)
    }
  }
}

# Aliases for backward compatibility
.find_project_root <- find_project_root
.ensure_theme_loaded <- ensure_theme_loaded

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

#' Validate input data for forest plots
#' @param data data.frame to validate
#' @param required_cols character vector of required column names
#' @param fn_name name of calling function for error messages
.validate_forest_data <- function(data, required_cols, fn_name) {
  if (is.null(data)) {
    stop(paste0(fn_name, ": data cannot be NULL"))
  }
  if (!is.data.frame(data)) {
    stop(paste0(fn_name, ": data must be a data.frame"))
  }
  if (nrow(data) == 0) {
    stop(paste0(fn_name, ": data must have at least one row"))
  }
  missing <- setdiff(required_cols, names(data))
  if (length(missing) > 0) {
    stop(paste0(fn_name, ": data missing required columns: ", paste(missing, collapse = ", ")))
  }
}

# ==============================================================================
# FOREST PLOT: Outlier Detection Methods
# ==============================================================================

#' Create forest plot for outlier detection methods
#'
#' @param data data.frame with columns: outlier_method, outlier_display_name,
#'             auroc_mean, auroc_ci_lo, auroc_ci_hi, category, n_configs
#' @param infographic logical; if TRUE, include title/subtitle/caption
#'                    (default FALSE for journal submission)
#' @param show_legend logical; if TRUE, show legend (default TRUE)
#' @param x_limits numeric vector of length 2 for x-axis limits (default c(0.5, 1.0))
#'
#' @return ggplot object
#' @export
create_forest_outlier <- function(data,
                                   infographic = FALSE,
                                   show_legend = TRUE,
                                   x_limits = c(0.5, 1.0)) {
  # Validate input
  required_cols <- c("outlier_display_name", "auroc_mean", "auroc_ci_lo", "auroc_ci_hi", "category")
  .validate_forest_data(data, required_cols, "create_forest_outlier")

  # Ensure theme is loaded
  .ensure_theme_loaded()

  # Prepare data
  plot_data <- data %>%
    mutate(
      display_name = factor(
        outlier_display_name,
        levels = rev(outlier_display_name[order(auroc_mean)])
      )
    )

  # Category colors from YAML config (Single Source of Truth)
  category_colors <- load_category_colors("outlier")
  ref_colors <- load_reference_colors()

  # Get ground truth AUROC for reference line
  gt_auroc <- plot_data$auroc_mean[plot_data$category == "Ground Truth"]
  gt_auroc <- if (length(gt_auroc) > 0) gt_auroc[1] else NA

  # Build plot using gg_add for S7 compatibility
  p <- ggplot(plot_data, aes(x = auroc_mean, y = display_name, color = category))

  p <- gg_add(p,
    # Reference line at random chance
    geom_vline(
      xintercept = 0.5,
      linetype = "dotted",
      color = ref_colors$random_chance,
      linewidth = 0.5
    )
  )

  # Add ground truth reference if available
  if (!is.na(gt_auroc)) {
    p <- gg_add(p,
      geom_vline(
        xintercept = gt_auroc,
        linetype = "dashed",
        color = ref_colors$ground_truth,
        linewidth = 0.5
      )
    )
  }

  p <- gg_add(p,
    # Point + CI whiskers
    geom_pointrange(
      aes(xmin = auroc_ci_lo, xmax = auroc_ci_hi),
      size = 0.6,
      linewidth = 0.8
    ),
    # Color scale
    scale_color_manual(values = category_colors, name = "Method Type"),
    # X-axis
    scale_x_continuous(
      limits = x_limits,
      breaks = seq(x_limits[1], x_limits[2], 0.1)
    ),
    # Base theme
    theme_forest()
  )

  # Apply legend setting
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p,
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal"
      ),
      guides(color = guide_legend(nrow = 1))
    )
  }

  # Labels depend on infographic mode
  if (infographic) {
    n_methods <- nrow(plot_data)
    n_configs <- sum(plot_data$n_configs, na.rm = TRUE)
    p <- gg_add(p,
      labs(
        title = "Outlier detection method performance",
        subtitle = "AUROC with 95% CI (CatBoost classifier, aggregated across imputation methods)",
        x = "AUROC",
        y = NULL,
        caption = paste0(
          "Dotted red: Random chance (AUROC=0.5). Dashed gold: Ground Truth.\n",
          "N = ", n_configs, " configurations across ", n_methods, " outlier methods."
        )
      ),
      theme(plot.caption = element_text(hjust = 0, lineheight = 1.2))
    )
  } else {
    # Journal mode: minimal labels
    p <- gg_add(p,
      labs(
        title = NULL,
        subtitle = NULL,
        x = "AUROC",
        y = NULL,
        caption = NULL
      )
    )
  }

  return(p)
}

# ==============================================================================
# FOREST PLOT: Imputation Methods
# ==============================================================================

#' Create forest plot for imputation methods
#'
#' @param data data.frame with columns: imputation_method, imputation_display_name,
#'             auroc_mean, auroc_ci_lo, auroc_ci_hi, category, n_configs
#' @param infographic logical; if TRUE, include title/subtitle/caption
#'                    (default FALSE for journal submission)
#' @param show_legend logical; if TRUE, show legend (default TRUE)
#' @param x_limits numeric vector of length 2 for x-axis limits (default c(0.5, 1.0))
#'
#' @return ggplot object
#' @export
create_forest_imputation <- function(data,
                                      infographic = FALSE,
                                      show_legend = TRUE,
                                      x_limits = c(0.5, 1.0)) {
  # Validate input
  required_cols <- c("imputation_display_name", "auroc_mean", "auroc_ci_lo", "auroc_ci_hi", "category")
  .validate_forest_data(data, required_cols, "create_forest_imputation")

  # Ensure theme is loaded
  .ensure_theme_loaded()

  # Prepare data
  plot_data <- data %>%
    mutate(
      display_name = factor(
        imputation_display_name,
        levels = rev(imputation_display_name[order(auroc_mean)])
      )
    )

  # Category colors from YAML config (Single Source of Truth)
  category_colors <- load_category_colors("imputation")
  ref_colors <- load_reference_colors()

  # Get ground truth AUROC for reference line
  gt_auroc <- plot_data$auroc_mean[plot_data$category == "Ground Truth"]
  gt_auroc <- if (length(gt_auroc) > 0) gt_auroc[1] else NA

  # Build plot
  p <- ggplot(plot_data, aes(x = auroc_mean, y = display_name, color = category))

  p <- gg_add(p,
    geom_vline(
      xintercept = 0.5,
      linetype = "dotted",
      color = ref_colors$random_chance,
      linewidth = 0.5
    )
  )

  if (!is.na(gt_auroc)) {
    p <- gg_add(p,
      geom_vline(
        xintercept = gt_auroc,
        linetype = "dashed",
        color = ref_colors$ground_truth,
        linewidth = 0.5
      )
    )
  }

  p <- gg_add(p,
    geom_pointrange(
      aes(xmin = auroc_ci_lo, xmax = auroc_ci_hi),
      size = 0.6,
      linewidth = 0.8
    ),
    scale_color_manual(values = category_colors, name = "Method Type"),
    scale_x_continuous(
      limits = x_limits,
      breaks = seq(x_limits[1], x_limits[2], 0.1)
    ),
    theme_forest()
  )

  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p,
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal"
      ),
      guides(color = guide_legend(nrow = 1))
    )
  }

  if (infographic) {
    n_methods <- nrow(plot_data)
    n_configs <- sum(plot_data$n_configs, na.rm = TRUE)
    p <- gg_add(p,
      labs(
        title = "Imputation method performance",
        subtitle = "AUROC with 95% CI (CatBoost classifier, aggregated across outlier methods)",
        x = "AUROC",
        y = NULL,
        caption = paste0(
          "Dotted red: Random chance (AUROC=0.5). Dashed gold: Ground Truth.\n",
          "N = ", n_configs, " configurations across ", n_methods, " imputation methods."
        )
      ),
      theme(plot.caption = element_text(hjust = 0, lineheight = 1.2))
    )
  } else {
    p <- gg_add(p,
      labs(
        title = NULL,
        subtitle = NULL,
        x = "AUROC",
        y = NULL,
        caption = NULL
      )
    )
  }

  return(p)
}

# ==============================================================================
# CALIBRATION PLOT (STRATOS-Compliant)
# ==============================================================================

#' Create STRATOS-compliant calibration plot
#'
#' @param cal_df data.frame with calibration curve data:
#'               config, bin_midpoint, observed, predicted, count
#' @param pred_df data.frame with prediction data:
#'               config, y_true, y_prob, outcome
#' @param annotations list with STRATOS annotations for one config:
#'                    calibration_slope, calibration_intercept, o_e_ratio, brier, ipa, n, n_events
#' @param infographic logical; if TRUE, include title/subtitle/caption
#' @param show_legend logical; if TRUE, show legend
#'
#' @return ggplot object
#' @export
create_calibration <- function(cal_df,
                                pred_df = NULL,
                                annotations = NULL,
                                infographic = FALSE,
                                show_legend = TRUE) {
  # Ensure theme is loaded
  .ensure_theme_loaded()

  # Config colors from YAML (Single Source of Truth)
  color_defs <- .ff_color_defs %||% load_color_definitions()
  config_colors <- c(
    color_defs[["--color-primary"]],
    color_defs[["--color-secondary"]],
    color_defs[["--color-category-ensemble"]],
    color_defs[["--color-accent"]]
  )
  names(config_colors) <- levels(cal_df$config)[1:min(4, length(levels(cal_df$config)))]

  # Build plot
  p <- ggplot(cal_df, aes(x = bin_midpoint, y = observed, color = config))

  p <- gg_add(p,
    # Perfect calibration line
    geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                color = color_defs[["--color-text-secondary"]], linewidth = 0.5),
    # Calibration points with size by count
    geom_point(aes(size = count), alpha = 0.8),
    # Smoothed calibration curve (LOESS)
    geom_smooth(method = "loess", se = TRUE, alpha = 0.2, linewidth = 0.8)
  )

  # Add rug for predictions if available
  if (!is.null(pred_df) && nrow(pred_df) > 0) {
    first_config <- levels(cal_df$config)[1]
    rug_data <- pred_df %>%
      filter(config == first_config) %>%
      sample_n(min(50, n()))
    if (nrow(rug_data) > 0) {
      p <- gg_add(p,
        geom_rug(
          data = rug_data,
          aes(x = y_prob, y = NULL),
          sides = "b",
          alpha = 0.3,
          color = color_defs[["--color-text-primary"]],
          inherit.aes = FALSE
        )
      )
    }
  }

  p <- gg_add(p,
    scale_color_manual(values = config_colors, name = "Pipeline"),
    scale_size_continuous(range = c(2, 6), guide = "none"),
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)),
    coord_equal(),
    theme_calibration()
  )

  # Add STRATOS annotations if provided
  if (!is.null(annotations)) {
    annotation_text <- sprintf(
      "Slope: %.2f\nIntercept: %.2f\nO:E: %.2f\nBrier: %.3f, IPA: %.3f\nN=%d (%d events)",
      annotations$calibration_slope,
      annotations$calibration_intercept,
      annotations$o_e_ratio,
      annotations$brier,
      annotations$ipa,
      annotations$n,
      annotations$n_events
    )
    p <- gg_add(p,
      annotate(
        "text",
        x = 0.02, y = 0.98,
        label = annotation_text,
        hjust = 0, vjust = 1,
        size = 2.5,
        family = "Helvetica"
      )
    )
  }

  # Legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p,
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal"
      ),
      guides(color = guide_legend(nrow = 2))
    )
  }

  # Labels
  if (infographic) {
    p <- gg_add(p,
      labs(
        title = "Calibration Plot",
        subtitle = "STRATOS-compliant calibration assessment",
        x = "Predicted Probability",
        y = "Observed Proportion",
        caption = "Dashed line: perfect calibration. Points sized by bin count."
      )
    )
  } else {
    p <- gg_add(p,
      labs(
        x = "Predicted Probability",
        y = "Observed Proportion"
      )
    )
  }

  return(p)
}

# ==============================================================================
# DECISION CURVE ANALYSIS (STRATOS-Compliant)
# ==============================================================================

#' Create STRATOS-compliant DCA plot
#'
#' @param dca_df data.frame with DCA data:
#'               config, threshold, nb_model, nb_treat_all, nb_treat_none
#' @param prevalence numeric; disease prevalence for annotations
#' @param infographic logical; if TRUE, include title/subtitle/caption
#' @param show_legend logical; if TRUE, show legend
#' @param x_max numeric; maximum threshold to display (default 0.30)
#'
#' @return ggplot object
#' @export
create_dca <- function(dca_df,
                        prevalence = NULL,
                        infographic = FALSE,
                        show_legend = TRUE,
                        x_max = 0.30) {
  # Ensure theme is loaded
  .ensure_theme_loaded()

  # Config colors from YAML (Single Source of Truth)
  color_defs <- .ff_color_defs %||% load_color_definitions()
  config_colors <- c(
    color_defs[["--color-primary"]],
    color_defs[["--color-secondary"]],
    color_defs[["--color-category-ensemble"]],
    color_defs[["--color-accent"]]
  )
  names(config_colors) <- levels(dca_df$config)[1:min(4, length(levels(dca_df$config)))]

  # Create reference data
  ref_thresholds <- unique(dca_df$threshold)
  ref_thresholds <- ref_thresholds[ref_thresholds <= x_max]

  if (!is.null(prevalence)) {
    ref_df <- data.frame(
      threshold = ref_thresholds,
      nb_treat_all = prevalence - (1 - prevalence) * (ref_thresholds / (1 - ref_thresholds))
    )
  } else {
    # Use first config's treat_all values
    first_config <- levels(dca_df$config)[1]
    ref_df <- dca_df %>%
      filter(config == first_config, threshold <= x_max) %>%
      select(threshold, nb_treat_all) %>%
      distinct()
  }

  # Build plot
  p <- ggplot(dca_df %>% filter(threshold <= x_max), aes(x = threshold, y = nb_model, color = config))

  p <- gg_add(p,
    # Treat none reference (horizontal at 0)
    geom_hline(yintercept = 0, linetype = "dashed",
               color = color_defs[["--color-annotation"]], linewidth = 0.5),
    # Treat all reference
    geom_line(
      data = ref_df,
      aes(x = threshold, y = nb_treat_all),
      inherit.aes = FALSE,
      linetype = "dotted",
      color = color_defs[["--color-text-primary"]],
      linewidth = 0.8
    ),
    # Model net benefit curves
    geom_line(linewidth = 1),
    # Key threshold markers
    geom_vline(
      xintercept = c(0.05, 0.10, 0.15, 0.20),
      linetype = "dotted",
      color = color_defs[["--color-border"]],
      linewidth = 0.3
    )
  )

  # Prevalence marker if provided
  if (!is.null(prevalence)) {
    p <- gg_add(p,
      geom_vline(
        xintercept = prevalence,
        linetype = "solid",
        color = color_defs[["--color-text-secondary"]],
        linewidth = 0.3
      ),
      annotate(
        "text",
        x = prevalence + 0.005, y = 0.25,
        label = paste0("Prev.\n", round(prevalence * 100, 1), "%"),
        color = color_defs[["--color-text-secondary"]],
        size = 2.2,
        hjust = 0,
        vjust = 1
      )
    )
  }

  p <- gg_add(p,
    # Annotations for reference lines
    annotate(
      "text",
      x = x_max - 0.02, y = ref_df$nb_treat_all[nrow(ref_df)] + 0.01,
      label = "Treat All",
      color = color_defs[["--color-text-primary"]],
      size = 2.5,
      hjust = 1,
      fontface = "italic"
    ),
    annotate(
      "text",
      x = x_max - 0.02, y = 0.01,
      label = "Treat None",
      color = color_defs[["--color-annotation"]],
      size = 2.5,
      hjust = 1,
      fontface = "italic"
    ),
    scale_color_manual(values = config_colors, name = "Pipeline"),
    scale_x_continuous(
      limits = c(0, x_max),
      breaks = seq(0, x_max, 0.05),
      labels = scales::percent
    ),
    scale_y_continuous(
      limits = c(-0.05, 0.30),
      breaks = seq(0, 0.30, 0.05)
    ),
    theme_foundation_plr()
  )

  # Legend
  if (!show_legend) {
    p <- gg_add(p, theme(legend.position = "none"))
  } else {
    p <- gg_add(p,
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal"
      ),
      guides(color = guide_legend(nrow = 2))
    )
  }

  # Labels
  if (infographic) {
    p <- gg_add(p,
      labs(
        title = "Decision Curve Analysis",
        subtitle = "Net benefit across threshold probabilities",
        x = "Threshold Probability",
        y = "Net Benefit",
        caption = "Dotted: Treat All. Dashed: Treat None (NB=0)."
      )
    )
  } else {
    p <- gg_add(p,
      labs(
        x = "Threshold Probability",
        y = "Net Benefit"
      )
    )
  }

  return(p)
}
