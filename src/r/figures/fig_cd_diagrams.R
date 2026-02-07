# Figure 3: Critical Difference (CD) Diagrams
# ============================================
# Shows statistical comparisons of preprocessing methods using
# Friedman test + Nemenyi post-hoc comparisons.
#
# Uses scmamp::plotCD() for standard Demšar (2006) style diagrams:
# - Horizontal rank axis at top
# - Method names with vertical+horizontal bars
# - CD bracket showing critical difference
# - Clique bars connecting methods not significantly different
#
# Based on Demšar (2006): "Statistical Comparisons of Classifiers over Multiple Data Sets"
#
# Created: 2026-01-27
# Updated: 2026-01-27 - Switched to scmamp::plotCD wrapper
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
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

# Source the CD diagram wrapper (uses scmamp::plotCD)
source(file.path(PROJECT_ROOT, "src/r/figure_system/cd_diagram.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

#' Prepare CD matrix from metrics data
#'
#' Creates a matrix suitable for create_cd_diagram():
#' - Rows = folds/datasets (replicates)
#' - Columns = methods (what we compare)
#'
#' @param df Data frame with method and metric columns
#' @param method_col Name of column containing method names
#' @param metric_col Name of column containing metric values
#' @param fold_col Name of column containing fold/replicate identifier
#' @return Matrix ready for create_cd_diagram()
prepare_cd_matrix_from_df <- function(df, method_col, metric_col, fold_col) {
  wide_df <- df %>%
    select(all_of(c(method_col, metric_col, fold_col))) %>%
    pivot_wider(
      id_cols = all_of(fold_col),
      names_from = all_of(method_col),
      values_from = all_of(metric_col)
    )

  # Convert to matrix: rows = folds, columns = methods
  mat <- as.matrix(wide_df[, -1])
  rownames(mat) <- wide_df[[fold_col]]

  return(mat)
}

#' Create pseudo-fold replicates from cross-tabulated data
#'
#' When we have outlier x imputation combinations but no explicit folds,
#' treat one dimension as "folds" to create the replication structure
#' needed for CD diagrams.
#'
#' Handles sparse data by:
#' 1. First filtering to rows (folds) that have data for most methods
#' 2. Then filtering to columns (methods) that are complete in those rows
#'
#' @param df Data frame with outlier_method, imputation_method, and metric columns
#' @param compare_col Which column to compare (methods as columns)
#' @param fold_col Which column to use as folds (replicates)
#' @param metric_col Name of metric column
#' @param min_folds Minimum number of folds required (default 3)
#' @return Matrix ready for create_cd_diagram()
create_cd_matrix_cross <- function(df, compare_col, fold_col, metric_col = "auroc",
                                    min_folds = 3) {
  # Create crosstab to see coverage
  crosstab <- table(df[[fold_col]], df[[compare_col]])

  # Find folds (rows) that have most methods
  methods_per_fold <- rowSums(crosstab > 0)

  # Keep folds with at least 80% of max coverage
  max_coverage <- max(methods_per_fold)
  good_folds <- names(methods_per_fold)[methods_per_fold >= 0.8 * max_coverage]

  if (length(good_folds) < min_folds) {
    # Fall back to all folds with at least 2 methods
    good_folds <- names(methods_per_fold)[methods_per_fold >= 2]
  }

  # Filter data to good folds
  df_filtered <- df %>% filter(.data[[fold_col]] %in% good_folds)

  # Now pivot
  wide_df <- df_filtered %>%
    select(all_of(c(compare_col, fold_col, metric_col))) %>%
    pivot_wider(
      id_cols = all_of(fold_col),
      names_from = all_of(compare_col),
      values_from = all_of(metric_col)
    )

  mat <- as.matrix(wide_df[, -1])
  rownames(mat) <- wide_df[[fold_col]]

  # Remove columns (methods) with any NA in the kept folds
  complete_cols <- colSums(is.na(mat)) == 0
  mat <- mat[, complete_cols, drop = FALSE]

  # Ensure we have at least min_folds rows
  if (nrow(mat) < min_folds) {
    warning(sprintf("Only %d folds available, CD diagram may be unreliable", nrow(mat)))
  }

  return(mat)
}

# ==============================================================================
# COMBINED MULTI-PANEL FIGURE
# ==============================================================================

#' Create multi-panel CD diagram figure
#'
#' Arranges CD diagrams in a vertical layout using base R graphics.
#' Adapts layout based on available data (2 or 3 panels).
#'
#' @param outlier_matrix Matrix for outlier method comparison
#' @param imputation_matrix Matrix for imputation method comparison
#' @param combined_matrix Matrix for combined pipeline comparison (or NULL)
#' @param output_path Path to save the combined figure
#' @param width Figure width in inches
#' @param height Figure height in inches
# NOTE: Do NOT define CD_BACKGROUND here - it would shadow the function from cd_diagram.R
# Use color_defs directly for local references

create_cd_multi_panel <- function(outlier_matrix, imputation_matrix, category_matrix,
                                   output_path, width = 14, height = 14,
                                   show_category_legend = FALSE) {

  # Get background color from color definitions
  bg_color <- color_defs[["--color-background"]]

  # Determine number of panels
  n_panels <- 2  # Always have outlier and imputation
  has_categories <- !is.null(category_matrix) && nrow(category_matrix) >= 2
  if (has_categories) n_panels <- 3

  # Calculate panel height - more space for each panel
  panel_height <- 5

  # Use Economist off-white background
  # figure-system-exception: CD diagrams use base R graphics (not ggplot2), so png() is required
  png(output_path, width = width, height = n_panels * panel_height, units = "in", res = 300,
      bg = bg_color)

  # Layout with margin space - wide margins for method names
  # mar = c(bottom, left, top, right)
  par(mfrow = c(n_panels, 1), mar = c(4, 14, 3, 14), oma = c(0, 0, 2, 0),
      bg = bg_color, family = "sans", col.main = color_defs[["--color-text-primary"]])

  # Panel A: Outlier Detection Methods
  # IMPORTANT: reset_par = FALSE to preserve mfrow layout
  create_cd_diagram(outlier_matrix, abbreviate = TRUE, cex = 1.0,
                    left_margin = 14, right_margin = 14, reset_par = FALSE,
                    show_category_legend = FALSE,
                    method_type = "outlier_detection")
  mtext("(a) Outlier Detection Methods", side = 3, line = 1, font = 2, cex = 1.0, col = color_defs[["--color-text-primary"]])

  # Panel B: Imputation Methods
  create_cd_diagram(imputation_matrix, abbreviate = TRUE, cex = 1.0,
                    left_margin = 14, right_margin = 14, reset_par = FALSE,
                    show_category_legend = FALSE,
                    method_type = "imputation")
  mtext("(b) Imputation Methods", side = 3, line = 1, font = 2, cex = 1.0, col = color_defs[["--color-text-primary"]])

  # Panel C: Preprocessing Categories (5 categories consistent with other figures)
  if (has_categories) {
    create_cd_diagram(category_matrix, abbreviate = FALSE, cex = 1.0,
                      left_margin = 14, right_margin = 14, reset_par = FALSE,
                      show_category_legend = FALSE,
                      method_type = "outlier_detection")
    mtext("(c) Preprocessing Categories", side = 3, line = 1, font = 2, cex = 1.0, col = color_defs[["--color-text-primary"]])
  }

  dev.off()
  message("Saved combined CD diagrams: ", output_path)
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if (sys.nframe() == 0) {
  # Parse command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  show_category_legend <- "--color" %in% args || "--legend" %in% args

  if (show_category_legend) {
    message("Category legend ENABLED (--color or --legend flag detected)")
  }

  message("Loading data from data/r_data/essential_metrics.csv...")

  metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")
  if (!file.exists(metrics_path)) {
    stop("Data file not found: ", metrics_path,
         "\nRun: python scripts/extract_all_configs_to_duckdb.py first")
  }

  metrics_df <- read.csv(metrics_path)

  # Filter to CatBoost classifier (fixed classifier per research design)
  cat_df <- metrics_df %>%
    filter(classifier == "CatBoost")

  message(sprintf("Found %d CatBoost configurations", nrow(cat_df)))

  # ==========================================================================
  # Panel A: Outlier Detection Methods
  # ==========================================================================
  # Compare outlier methods, using imputation methods as "folds"

  message("\n=== Preparing Outlier Methods Matrix ===")
  outlier_matrix <- create_cd_matrix_cross(
    cat_df,
    compare_col = "outlier_method",
    fold_col = "imputation_method",
    metric_col = "auroc"
  )
  message(sprintf("Outlier matrix: %d methods x %d imputation folds",
                  ncol(outlier_matrix), nrow(outlier_matrix)))

  # ==========================================================================
  # Panel B: Imputation Methods
  # ==========================================================================
  # Compare imputation methods, using outlier methods as "folds"

  message("\n=== Preparing Imputation Methods Matrix ===")
  imputation_matrix <- create_cd_matrix_cross(
    cat_df,
    compare_col = "imputation_method",
    fold_col = "outlier_method",
    metric_col = "auroc"
  )
  message(sprintf("Imputation matrix: %d methods x %d outlier folds",
                  ncol(imputation_matrix), nrow(imputation_matrix)))

  # ==========================================================================
  # Panel C: Preprocessing Categories (5 categories, consistent with other figures)
  # ==========================================================================
  # Compare 5 preprocessing categories:
  # Ground Truth, Ensemble FM, Single-model FM, Deep Learning, Traditional

  message("\n=== Preparing Preprocessing Categories Matrix ===")

  # Assign categories based on outlier method (using category_loader.R)
  category_df <- cat_df %>%
    mutate(
      raw_category = sapply(outlier_method, get_outlier_category),
      display_category = sapply(raw_category, to_display_category)
    )

  # Aggregate AUROC by category and imputation method
  # imputation_method serves as "folds" (replicates) for the CD test
  category_agg <- category_df %>%
    group_by(display_category, imputation_method) %>%
    summarise(auroc = mean(auroc, na.rm = TRUE), .groups = "drop")

  # Create CD matrix: rows = imputation methods (folds), columns = categories
  category_matrix <- create_cd_matrix_cross(
    category_agg,
    compare_col = "display_category",
    fold_col = "imputation_method",
    metric_col = "auroc",
    min_folds = 3
  )

  # Reorder columns to match standard category order
  category_order <- get_category_order()
  available_cats <- intersect(category_order, colnames(category_matrix))
  category_matrix <- category_matrix[, available_cats, drop = FALSE]

  message(sprintf("Category matrix: %d categories x %d imputation folds",
                  ncol(category_matrix), nrow(category_matrix)))

  # ==========================================================================
  # Generate Combined 3-Panel Figure (ONLY OUTPUT)
  # ==========================================================================
  # Per figure_layouts.yaml: only fig_cd_diagrams.png goes to supplementary
  # NO individual panel outputs

  output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  message("\n=== Creating Combined 3-Panel CD Figure ===")
  output_path <- file.path(output_dir, "fig_cd_diagrams.png")

  create_cd_multi_panel(
    outlier_matrix,
    imputation_matrix,
    category_matrix,
    output_path = output_path,
    width = 14,
    height = 16,
    show_category_legend = FALSE  # Category legend broken, keep disabled
  )

  # ==========================================================================
  # Summary
  # ==========================================================================

  message("\n========================================")
  message("CD Diagrams Complete (Demšar Style)")
  message("========================================")
  message(sprintf("Outlier methods compared: %d", ncol(outlier_matrix)))
  message(sprintf("Imputation methods compared: %d", ncol(imputation_matrix)))
  message(sprintf("Preprocessing categories compared: %d", ncol(category_matrix)))
  message("\nFigure generated:")
  message(sprintf("  - %s", output_path))
}
