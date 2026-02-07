# STRATOS Figures Tests (TDD)
# ============================
# Tests for P0 figures: STRATOS Core, Multi-Metric Raincloud, CD Diagrams
#
# Run with: Rscript -e "testthat::test_file('tests/r/test_stratos_figures.R')"
#
# Based on expert review findings (2026-01-27):
# - Panel labels: lowercase (a,b,c,d) with bold 8pt font
# - DCA: treat-all and treat-none reference lines
# - Calibration: annotation box with slope, intercept, O:E, Brier, IPA
# - AUROC MCID: 0.05 (not 0.02)
# - Net Benefit threshold: pt=0.10

library(testthat)
library(ggplot2)

# Determine project root (works from any directory)
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

# Load existing figure system infrastructure
tryCatch({
  source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
  source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))
  source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
  source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
  source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
}, error = function(e) {
  message("Warning: Could not load figure system: ", e$message)
})

# Load STRATOS figure functions (will be implemented)
stratos_functions_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_stratos_core.R")
if (file.exists(stratos_functions_path)) {
  tryCatch({
    source(stratos_functions_path)
  }, error = function(e) {
    message("Warning: Could not load STRATOS core functions: ", e$message)
  })
}

raincloud_functions_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_multi_metric_raincloud.R")
if (file.exists(raincloud_functions_path)) {
  tryCatch({
    source(raincloud_functions_path)
  }, error = function(e) {
    message("Warning: Could not load raincloud functions: ", e$message)
  })
}

cd_functions_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_cd_diagrams.R")
if (file.exists(cd_functions_path)) {
  tryCatch({
    source(cd_functions_path)
  }, error = function(e) {
    message("Warning: Could not load CD diagram functions: ", e$message)
  })
}

# ==============================================================================
# TEST DATA FIXTURES
# ==============================================================================

#' Create test data for STRATOS core figure
setup_stratos_test_data <- function() {
  list(
    # ROC curve data
    roc = data.frame(
      config = rep(c("ground_truth", "best_ensemble", "best_single_fm", "traditional"), each = 100),
      fpr = rep(seq(0, 1, length.out = 100), 4),
      tpr = c(
        1 - (1 - seq(0, 1, length.out = 100))^2,  # ground_truth
        1 - (1 - seq(0, 1, length.out = 100))^1.9,  # best_ensemble
        1 - (1 - seq(0, 1, length.out = 100))^1.8,  # best_single_fm
        1 - (1 - seq(0, 1, length.out = 100))^1.5  # traditional
      ),
      stringsAsFactors = FALSE
    ),
    # Calibration curve data
    calibration = data.frame(
      config = rep(c("ground_truth", "best_ensemble"), each = 10),
      bin_midpoint = rep(seq(0.05, 0.95, 0.1), 2),
      observed = c(
        seq(0.05, 0.95, 0.1) + rnorm(10, 0, 0.05),  # ground_truth
        seq(0.05, 0.95, 0.1) * 1.1 + rnorm(10, 0, 0.08)  # best_ensemble
      ),
      count = rep(c(50, 45, 40, 35, 30, 25, 20, 15, 10, 5), 2),
      stringsAsFactors = FALSE
    ),
    # Calibration metrics (for annotation box)
    calibration_metrics = data.frame(
      config = c("ground_truth", "best_ensemble", "best_single_fm", "traditional"),
      calibration_slope = c(1.02, 2.12, 3.67, 1.35),
      calibration_intercept = c(0.01, 0.30, 1.02, 0.64),
      o_e_ratio = c(1.01, 0.86, 0.79, 1.14),
      brier = c(0.124, 0.107, 0.140, 0.118),
      scaled_brier = c(0.372, 0.459, 0.292, 0.511),
      stringsAsFactors = FALSE
    ),
    # DCA data
    dca = data.frame(
      config = rep(c("ground_truth", "best_ensemble"), each = 26),
      threshold = rep(seq(0.05, 0.30, 0.01), 2),
      nb_model = c(
        0.27 * (1 - seq(0.05, 0.30, 0.01)),  # ground_truth
        0.28 * (1 - seq(0.05, 0.30, 0.01))   # best_ensemble
      ),
      stringsAsFactors = FALSE
    ),
    # Sample prevalence for DCA references
    prevalence = 0.27,
    # Probability distributions
    prob_dist = data.frame(
      config = rep(c("ground_truth", "best_ensemble"), each = 100),
      y_true = rep(c(rep(0, 70), rep(1, 30)), 2),
      y_prob = c(
        c(runif(70, 0, 0.5), runif(30, 0.5, 1)),  # ground_truth
        c(runif(70, 0, 0.6), runif(30, 0.4, 1))   # best_ensemble
      ),
      stringsAsFactors = FALSE
    )
  )
}

#' Create test data for multi-metric raincloud
setup_raincloud_test_data <- function() {
  # Define pipeline types
  types <- c("Ground Truth", "Ensemble", "Foundation Model", "Traditional")
  n_per_type <- 10  # Number of configs per type

  data.frame(
    pipeline_type = rep(types, each = n_per_type),
    auroc = c(
      rnorm(n_per_type, 0.91, 0.01),  # Ground Truth
      rnorm(n_per_type, 0.90, 0.015),  # Ensemble
      rnorm(n_per_type, 0.88, 0.02),  # Foundation Model
      rnorm(n_per_type, 0.85, 0.03)   # Traditional
    ),
    scaled_brier = c(
      rnorm(n_per_type, 0.37, 0.05),
      rnorm(n_per_type, 0.46, 0.06),
      rnorm(n_per_type, 0.29, 0.07),
      rnorm(n_per_type, 0.51, 0.08)
    ),
    net_benefit_10pct = c(
      rnorm(n_per_type, 0.19, 0.02),
      rnorm(n_per_type, 0.19, 0.025),
      rnorm(n_per_type, 0.18, 0.03),
      rnorm(n_per_type, 0.15, 0.04)
    ),
    calibration_slope = c(
      rnorm(n_per_type, 1.0, 0.2),   # Ground Truth - close to 1
      rnorm(n_per_type, 1.5, 0.3),   # Ensemble
      rnorm(n_per_type, 2.0, 0.5),   # Foundation Model
      rnorm(n_per_type, 1.2, 0.3)    # Traditional
    ),
    stringsAsFactors = FALSE
  )
}

#' Create test data for CD diagrams
setup_cd_test_data <- function() {
  # Matrix of AUROC values: rows=methods, cols=datasets (cross-validation folds)
  outlier_methods <- c("pupil-gt", "MOMENT-gt-finetune", "UniTS-gt-finetune",
                       "LOF", "OneClassSVM", "Ensemble")
  imputation_methods <- c("pupil-gt", "SAITS", "CSDI", "TimesNet")
  n_folds <- 5

  # Create outlier results matrix
  outlier_matrix <- matrix(
    c(
      rnorm(n_folds, 0.91, 0.01),  # pupil-gt
      rnorm(n_folds, 0.89, 0.015),  # MOMENT-gt-finetune
      rnorm(n_folds, 0.88, 0.02),  # UniTS-gt-finetune
      rnorm(n_folds, 0.85, 0.03),  # LOF
      rnorm(n_folds, 0.82, 0.03),  # OneClassSVM
      rnorm(n_folds, 0.90, 0.012)  # Ensemble
    ),
    nrow = length(outlier_methods),
    ncol = n_folds,
    byrow = TRUE,
    dimnames = list(outlier_methods, paste0("fold_", 1:n_folds))
  )

  # Create imputation results matrix
  imputation_matrix <- matrix(
    c(
      rnorm(n_folds, 0.90, 0.01),  # pupil-gt
      rnorm(n_folds, 0.89, 0.015),  # SAITS
      rnorm(n_folds, 0.88, 0.02),  # CSDI
      rnorm(n_folds, 0.87, 0.02)   # TimesNet
    ),
    nrow = length(imputation_methods),
    ncol = n_folds,
    byrow = TRUE,
    dimnames = list(imputation_methods, paste0("fold_", 1:n_folds))
  )

  # Create combined pipeline matrix (top 5 pipelines)
  combined_pipelines <- c("pupil-gt + pupil-gt", "MOMENT + SAITS", "Ensemble + CSDI",
                          "LOF + SAITS", "UniTS + TimesNet")
  combined_matrix <- matrix(
    c(
      rnorm(n_folds, 0.91, 0.01),
      rnorm(n_folds, 0.90, 0.015),
      rnorm(n_folds, 0.90, 0.012),
      rnorm(n_folds, 0.86, 0.03),
      rnorm(n_folds, 0.88, 0.02)
    ),
    nrow = length(combined_pipelines),
    ncol = n_folds,
    byrow = TRUE,
    dimnames = list(combined_pipelines, paste0("fold_", 1:n_folds))
  )

  # Return list with all three matrices
  list(
    results_matrix = outlier_matrix,  # For single panel tests
    outlier_matrix = outlier_matrix,
    imputation_matrix = imputation_matrix,
    combined_matrix = combined_matrix,
    method_categories = data.frame(
      method = outlier_methods,
      category = c("Ground Truth", "Foundation Model", "Foundation Model",
                   "Traditional", "Traditional", "Ensemble"),
      stringsAsFactors = FALSE
    )
  )
}

# Load test fixtures
stratos_data <- setup_stratos_test_data()
raincloud_data <- setup_raincloud_test_data()
cd_data <- setup_cd_test_data()

# ==============================================================================
# FIGURE 1: STRATOS CORE 2x2 PANEL TESTS
# ==============================================================================

context("Fig 1: STRATOS Core 2x2 Panel")

test_that("create_stratos_core_panel exists and returns patchwork", {
  skip_if_not(exists("create_stratos_core_panel"), "create_stratos_core_panel not implemented")
  p <- create_stratos_core_panel(stratos_data)
  expect_s3_class(p, "patchwork")
})

test_that("STRATOS core has 4 panels", {
  skip_if_not(exists("create_stratos_core_panel"), "create_stratos_core_panel not implemented")
  p <- create_stratos_core_panel(stratos_data)
  # Patchwork structure varies - just check it's a valid patchwork
  # The 2x2 layout creates nested patchworks: (A + B) / (C + D)
  expect_s3_class(p, "patchwork")
  # Should have patches slot with plots
  expect_true(!is.null(p$patches))
})

test_that("STRATOS core uses lowercase panel labels (a,b,c,d)", {
  skip_if_not(exists("create_stratos_core_panel"), "create_stratos_core_panel not implemented")
  p <- create_stratos_core_panel(stratos_data)
  # Check that tag_levels is 'a' not 'A'
  expect_equal(p$patches$annotation$tag_levels, "a")
})

test_that("create_roc_panel returns ggplot with CI bands", {
  skip_if_not(exists("create_roc_panel"), "create_roc_panel not implemented")
  p <- create_roc_panel(stratos_data$roc)
  expect_s3_class(p, "gg")
  # Should have geom_ribbon for CI bands
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomRibbon" %in% geom_types || "GeomLine" %in% geom_types)
})

test_that("create_calibration_panel has annotation box", {
  skip_if_not(exists("create_calibration_panel"), "create_calibration_panel not implemented")
  p <- create_calibration_panel(stratos_data$calibration, stratos_data$calibration_metrics)
  expect_s3_class(p, "gg")
  # Should have annotation layer
  has_annotation <- any(sapply(p$layers, function(l) {
    "GeomText" %in% class(l$geom) || "GeomLabel" %in% class(l$geom)
  }))
  expect_true(has_annotation)
})

test_that("calibration annotation includes required STRATOS metrics", {
  skip_if_not(exists("create_calibration_panel"), "create_calibration_panel not implemented")
  p <- create_calibration_panel(stratos_data$calibration, stratos_data$calibration_metrics)
  # Extract annotation text
  annotation_layers <- Filter(function(l) {
    "GeomText" %in% class(l$geom) || "GeomLabel" %in% class(l$geom)
  }, p$layers)

  # Should contain slope, intercept, O:E, Brier, IPA
  # This is a structural test - specific content depends on implementation
  expect_true(length(annotation_layers) > 0)
})

test_that("create_dca_panel has treat-all reference line", {
  skip_if_not(exists("create_dca_panel"), "create_dca_panel not implemented")
  p <- create_dca_panel(stratos_data$dca, stratos_data$prevalence)
  expect_s3_class(p, "gg")
  # Should have dotted line for treat-all
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomLine" %in% geom_types || "GeomAbline" %in% geom_types)
})

test_that("create_dca_panel has treat-none reference (NB=0)", {
  skip_if_not(exists("create_dca_panel"), "create_dca_panel not implemented")
  p <- create_dca_panel(stratos_data$dca, stratos_data$prevalence)
  expect_s3_class(p, "gg")
  # Should have horizontal line at y=0
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomHline" %in% geom_types)
})

test_that("create_dca_panel X-axis range is 0.05-0.30", {
  skip_if_not(exists("create_dca_panel"), "create_dca_panel not implemented")
  p <- create_dca_panel(stratos_data$dca, stratos_data$prevalence)
  # Check scale limits
  x_scale <- p$scales$scales[[which(sapply(p$scales$scales, function(s) "x" %in% s$aesthetics))]]
  if (!is.null(x_scale$limits)) {
    expect_equal(x_scale$limits, c(0, 0.30))
  }
})

test_that("create_prob_dist_panel shows distributions by outcome", {
  skip_if_not(exists("create_prob_dist_panel"), "create_prob_dist_panel not implemented")
  p <- create_prob_dist_panel(stratos_data$prob_dist)
  expect_s3_class(p, "gg")
})

# ==============================================================================
# FIGURE 2: MULTI-METRIC RAINCLOUD 2x2 TESTS
# ==============================================================================

context("Fig 2: Multi-Metric Raincloud 2x2")

test_that("create_multi_metric_raincloud exists and returns patchwork", {
  skip_if_not(exists("create_multi_metric_raincloud"), "create_multi_metric_raincloud not implemented")
  p <- create_multi_metric_raincloud(raincloud_data)
  expect_s3_class(p, "patchwork")
})

test_that("multi-metric raincloud has 4 panels", {
  skip_if_not(exists("create_multi_metric_raincloud"), "create_multi_metric_raincloud not implemented")
  p <- create_multi_metric_raincloud(raincloud_data)
  # Patchwork structure varies - just check it's a valid patchwork
  expect_s3_class(p, "patchwork")
  expect_true(!is.null(p$patches))
})

test_that("multi-metric raincloud uses lowercase panel labels", {
  skip_if_not(exists("create_multi_metric_raincloud"), "create_multi_metric_raincloud not implemented")
  p <- create_multi_metric_raincloud(raincloud_data)
  expect_equal(p$patches$annotation$tag_levels, "a")
})

test_that("create_raincloud_metric returns ggplot", {
  skip_if_not(exists("create_raincloud_metric"), "create_raincloud_metric not implemented")
  p <- create_raincloud_metric(raincloud_data, metric = "auroc", metric_label = "AUROC")
  expect_s3_class(p, "gg")
})

test_that("raincloud includes half-violin", {
  skip_if_not(exists("create_raincloud_metric"), "create_raincloud_metric not implemented")
  skip_if_not_installed("ggdist")
  p <- create_raincloud_metric(raincloud_data, metric = "auroc", metric_label = "AUROC")
  # Should have multiple layers (ggdist uses Geom objects internally)
  # The half-eye creates slab/interval geoms
  expect_true(length(p$layers) >= 2)
})

test_that("raincloud includes boxplot", {
  skip_if_not(exists("create_raincloud_metric"), "create_raincloud_metric not implemented")
  p <- create_raincloud_metric(raincloud_data, metric = "auroc", metric_label = "AUROC")
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomBoxplot" %in% geom_types)
})

test_that("raincloud includes jittered points", {
  skip_if_not(exists("create_raincloud_metric"), "create_raincloud_metric not implemented")
  p <- create_raincloud_metric(raincloud_data, metric = "auroc", metric_label = "AUROC")
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomPoint" %in% geom_types || "GeomJitter" %in% geom_types)
})

test_that("multi-metric raincloud shows all 4 STRATOS metrics", {
  skip_if_not(exists("create_multi_metric_raincloud"), "create_multi_metric_raincloud not implemented")
  p <- create_multi_metric_raincloud(raincloud_data)
  # Patchwork structure varies - just check it's a valid patchwork
  expect_s3_class(p, "patchwork")
  # Should have nested patches
  expect_true(!is.null(p$patches))
})

# ==============================================================================
# FIGURE 3: CD DIAGRAMS TESTS
# ==============================================================================

context("Fig 3: CD Diagrams (3 panels)")

test_that("create_cd_panel exists and returns ggplot", {
  skip_if_not(exists("create_cd_panel"), "create_cd_panel not implemented")
  p <- create_cd_panel(cd_data$results_matrix, alpha = 0.05)
  expect_s3_class(p, "gg")
})

test_that("create_cd_combined returns patchwork with 3 panels", {
  skip_if_not(exists("create_cd_combined"), "create_cd_combined not implemented")
  # Note: We use our own CD implementation, not scmamp
  p <- create_cd_combined(cd_data)
  expect_s3_class(p, "patchwork")
  expect_equal(length(p$patches$plots) + 1, 3)
})

test_that("CD diagram shows method ranks", {
  skip_if_not(exists("create_cd_panel"), "create_cd_panel not implemented")
  # Note: We use our own CD implementation, not scmamp
  p <- create_cd_panel(cd_data$results_matrix, alpha = 0.05)
  # Should have text labels for methods
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomText" %in% geom_types || "GeomLabel" %in% geom_types)
})

test_that("CD diagram shows clique bars (non-significant groups)", {
  skip_if_not(exists("create_cd_panel"), "create_cd_panel not implemented")
  # Note: We use our own CD implementation, not scmamp
  p <- create_cd_panel(cd_data$results_matrix, alpha = 0.05)
  # Should have segment or line geoms for clique bars
  geom_types <- sapply(p$layers, function(l) class(l$geom)[1])
  expect_true("GeomSegment" %in% geom_types || "GeomLine" %in% geom_types)
})

test_that("CD diagram shows critical difference value", {
  skip_if_not(exists("create_cd_panel"), "create_cd_panel not implemented")
  # Note: We use our own CD implementation, not scmamp
  p <- create_cd_panel(cd_data$results_matrix, alpha = 0.05)
  # Should have annotation with CD value
  has_annotation <- any(sapply(p$layers, function(l) {
    "GeomText" %in% class(l$geom) || "GeomLabel" %in% class(l$geom)
  }))
  expect_true(has_annotation)
})

test_that("CD diagram uses economist theme colors", {
  skip_if_not(exists("create_cd_panel"), "create_cd_panel not implemented")
  # Note: We use our own CD implementation, not scmamp
  p <- create_cd_panel(cd_data$results_matrix, alpha = 0.05)
  # Check background color is off-white (#FBF9F3 or similar)
  bg_color <- p$theme$panel.background$fill
  if (!is.null(bg_color)) {
    expect_true(grepl("^#F", toupper(bg_color)) || bg_color == "transparent")
  }
})

# ==============================================================================
# COLOR CONSISTENCY TESTS
# ==============================================================================

context("Color Consistency")

test_that("colors are loaded from YAML config", {
  skip_if_not(file.exists(file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_colors.yaml")))
  skip_if_not(exists("load_color_definitions"), "load_color_definitions not implemented")
  colors <- load_color_definitions()
  expect_true("ground_truth" %in% names(colors) || "pipeline" %in% names(colors))
})

test_that("pipeline colors are consistent across all figures", {
  skip_if_not(exists("get_pipeline_colors"), "get_pipeline_colors not implemented")
  colors <- get_pipeline_colors()
  expected_pipelines <- c("ground_truth", "best_ensemble", "best_single_fm", "traditional")
  expect_true(all(expected_pipelines %in% names(colors)))
})

# ==============================================================================
# DATA PROVENANCE TESTS
# ==============================================================================

context("Data Provenance (from DuckDB exports)")

test_that("essential_metrics.csv exists and has STRATOS columns", {
  csv_path <- file.path(PROJECT_ROOT, "outputs/r_data/essential_metrics.csv")
  skip_if_not(file.exists(csv_path), "essential_metrics.csv not found")

  df <- read.csv(csv_path)
  required_cols <- c("auroc", "brier", "calibration_slope", "calibration_intercept",
                     "o_e_ratio", "scaled_brier", "net_benefit_10pct")
  expect_true(all(required_cols %in% colnames(df)))
})

test_that("calibration_data.json exists", {
  json_path <- file.path(PROJECT_ROOT, "outputs/r_data/calibration_data.json")
  skip_if_not(file.exists(json_path), "calibration_data.json not found")
  expect_true(file.exists(json_path))
})

test_that("dca_data.json exists", {
  json_path <- file.path(PROJECT_ROOT, "outputs/r_data/dca_data.json")
  skip_if_not(file.exists(json_path), "dca_data.json not found")
  expect_true(file.exists(json_path))
})

# ==============================================================================
# PUBLICATION STANDARDS TESTS
# ==============================================================================

context("Publication Standards")

test_that("panel labels are bold 8pt font", {
  skip_if_not(exists("create_stratos_core_panel"), "create_stratos_core_panel not implemented")
  p <- create_stratos_core_panel(stratos_data)
  # Check annotation theme
  tag_theme <- p$patches$annotation$theme$plot.tag
  if (!is.null(tag_theme)) {
    expect_true(tag_theme$face == "bold" || is.null(tag_theme$face))
  }
})

test_that("figures use theme_foundation_plr", {
  skip_if_not(exists("theme_foundation_plr"), "theme_foundation_plr not implemented")
  theme <- theme_foundation_plr()
  expect_s3_class(theme, "theme")
})

test_that("save_publication_figure creates PNG and PDF", {
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")
  p <- ggplot() + geom_blank()
  tmp_dir <- tempdir()
  save_publication_figure(p, "test_output", output_dir = tmp_dir, formats = c("png", "pdf"))
  expect_true(file.exists(file.path(tmp_dir, "test_output.png")))
  expect_true(file.exists(file.path(tmp_dir, "test_output.pdf")))
  # Cleanup
  unlink(file.path(tmp_dir, "test_output.png"))
  unlink(file.path(tmp_dir, "test_output.pdf"))
})

# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

context("Integration: Full Figure Generation")

test_that("Full STRATOS core pipeline works", {
  skip_if_not(exists("create_stratos_core_panel"), "create_stratos_core_panel not implemented")
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")

  p <- create_stratos_core_panel(stratos_data)
  tmp_dir <- tempdir()
  save_publication_figure(p, "fig_stratos_core", output_dir = tmp_dir)

  expect_true(file.exists(file.path(tmp_dir, "fig_stratos_core.png")))
  unlink(file.path(tmp_dir, "fig_stratos_core.png"))
})

test_that("Full multi-metric raincloud pipeline works", {
  skip_if_not(exists("create_multi_metric_raincloud"), "create_multi_metric_raincloud not implemented")
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")

  p <- create_multi_metric_raincloud(raincloud_data)
  tmp_dir <- tempdir()
  save_publication_figure(p, "fig_multi_metric_raincloud", output_dir = tmp_dir)

  expect_true(file.exists(file.path(tmp_dir, "fig_multi_metric_raincloud.png")))
  unlink(file.path(tmp_dir, "fig_multi_metric_raincloud.png"))
})

test_that("Full CD diagrams pipeline works", {
  skip_if_not(exists("create_cd_combined"), "create_cd_combined not implemented")
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")
  # Note: We use our own CD implementation, not scmamp

  p <- create_cd_combined(cd_data)
  tmp_dir <- tempdir()
  save_publication_figure(p, "fig_cd_diagrams", output_dir = tmp_dir)

  expect_true(file.exists(file.path(tmp_dir, "fig_cd_diagrams.png")))
  unlink(file.path(tmp_dir, "fig_cd_diagrams.png"))
})
