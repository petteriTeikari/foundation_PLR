# Figure System Tests
# ====================
# TDD tests for the flexible decomposable figure system.
#
# Run with: Rscript -e "testthat::test_file('tests/r/test_figure_system.R')"

library(testthat)
library(ggplot2)

# Determine project root (works from any directory)
find_project_root <- function() {
  # Look for common markers
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

# Load test fixtures
source(file.path(PROJECT_ROOT, "tests/r/fixtures/test_data.R"))

# Load figure system (will fail until implemented)
tryCatch({
  source(file.path(PROJECT_ROOT, "src/r/figure_system/figure_factory.R"))
  source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
  source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
}, error = function(e) {
  message("Figure system not yet implemented: ", e$message)
})

# ==============================================================================
# TEST DATA SETUP
# ==============================================================================

outlier_data <- setup_outlier_test_data()
imputation_data <- setup_imputation_test_data()

# ==============================================================================
# UNIT TESTS: create_forest_outlier
# ==============================================================================

test_that("create_forest_outlier returns ggplot object", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  p <- create_forest_outlier(outlier_data)
  expect_s3_class(p, "gg")
})

test_that("create_forest_outlier: infographic=FALSE removes annotations", {

  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  p <- create_forest_outlier(outlier_data, infographic = FALSE)
  expect_null(p$labels$title)
  expect_null(p$labels$subtitle)
  expect_null(p$labels$caption)
})

test_that("create_forest_outlier: infographic=TRUE includes annotations", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  p <- create_forest_outlier(outlier_data, infographic = TRUE)
  expect_false(is.null(p$labels$title))
})

test_that("create_forest_outlier: show_legend=FALSE removes legend", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  p <- create_forest_outlier(outlier_data, show_legend = FALSE)
  # Check theme has legend.position = "none"
  expect_equal(p$theme$legend.position, "none")
})

test_that("create_forest_outlier: errors on NULL data", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  expect_error(create_forest_outlier(NULL), "cannot be NULL")
})

test_that("create_forest_outlier: errors on empty data frame", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  expect_error(create_forest_outlier(data.frame()), "at least one row")
})

test_that("create_forest_outlier: errors on missing required columns", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  bad_data <- data.frame(x = 1, y = 2)
  expect_error(create_forest_outlier(bad_data), "missing required columns")
})

test_that("create_forest_outlier: handles NA in auroc_mean gracefully", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  data_with_na <- outlier_data
  data_with_na$auroc_mean[1] <- NA
  # Should still create plot (ggplot handles NAs)
  p <- create_forest_outlier(data_with_na)
  expect_s3_class(p, "gg")
})

test_that("create_forest_outlier: handles missing Ground Truth category", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  data_no_gt <- outlier_data[outlier_data$category != "Ground Truth", ]
  # Should work without ground truth reference line
  p <- create_forest_outlier(data_no_gt)
  expect_s3_class(p, "gg")
})

# ==============================================================================
# UNIT TESTS: create_forest_imputation
# ==============================================================================

test_that("create_forest_imputation returns ggplot object", {
  skip_if_not(exists("create_forest_imputation"), "create_forest_imputation not implemented")
  p <- create_forest_imputation(imputation_data)
  expect_s3_class(p, "gg")
})

test_that("create_forest_imputation: infographic=FALSE removes annotations", {
  skip_if_not(exists("create_forest_imputation"), "create_forest_imputation not implemented")
  p <- create_forest_imputation(imputation_data, infographic = FALSE)
  expect_null(p$labels$title)
  expect_null(p$labels$subtitle)
  expect_null(p$labels$caption)
})

# ==============================================================================
# UNIT TESTS: compose_figures
# ==============================================================================

test_that("compose_figures creates patchwork object", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p1 <- ggplot() + geom_blank()
  p2 <- ggplot() + geom_blank()
  composed <- compose_figures(list(p1, p2), layout = "2x1")
  expect_s3_class(composed, "patchwork")
})
test_that("compose_figures: errors on empty plot list", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  expect_error(compose_figures(list()), "at least one plot")
})

test_that("compose_figures: errors on non-list input", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p <- ggplot() + geom_blank()
  expect_error(compose_figures(p), "must be a list")
})

test_that("compose_figures: errors on invalid layout string", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p <- ggplot() + geom_blank()
  expect_error(compose_figures(list(p), layout = "invalid"), "Invalid layout")
})

test_that("compose_figures: errors when plot count mismatches layout", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p <- ggplot() + geom_blank()
  expect_error(compose_figures(list(p), layout = "2x2"), "requires 4 plots")
})

test_that("compose_figures: errors when too many plots for layout", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p <- ggplot() + geom_blank()
  expect_error(compose_figures(list(p, p, p), layout = "2x1"), "requires 2 plots")
})

test_that("compose_figures: errors when plots list contains non-ggplot", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p <- ggplot() + geom_blank()
  expect_error(compose_figures(list(p, "not a plot")), "not ggplot objects")
})

test_that("compose_figures: 2x1 layout works", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p1 <- ggplot() + geom_blank()
  p2 <- ggplot() + geom_blank()
  composed <- compose_figures(list(p1, p2), layout = "2x1")
  expect_s3_class(composed, "patchwork")
})

test_that("compose_figures: 1x2 layout works", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p1 <- ggplot() + geom_blank()
  p2 <- ggplot() + geom_blank()
  composed <- compose_figures(list(p1, p2), layout = "1x2")
  expect_s3_class(composed, "patchwork")
})

test_that("compose_figures: tag_levels adds panel labels", {
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  p1 <- ggplot() + geom_blank()
  p2 <- ggplot() + geom_blank()
  composed <- compose_figures(list(p1, p2), layout = "2x1", tag_levels = "A")
  # Patchwork stores annotation info
  expect_s3_class(composed, "patchwork")
})

# ==============================================================================
# UNIT TESTS: save_publication_figure
# ==============================================================================

test_that("save_publication_figure creates files", {
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")
  p <- ggplot() + geom_blank()
  tmp_dir <- tempdir()
  save_publication_figure(p, "test_fig", output_dir = tmp_dir, formats = c("png"))
  expect_true(file.exists(file.path(tmp_dir, "test_fig.png")))
  # Cleanup
  unlink(file.path(tmp_dir, "test_fig.png"))
})

test_that("save_publication_figure: errors on invalid width", {
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")
  p <- ggplot() + geom_blank()
  expect_error(save_publication_figure(p, "test", width = -10), "positive")
})

test_that("save_publication_figure: errors on zero width", {
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")
  p <- ggplot() + geom_blank()
  expect_error(save_publication_figure(p, "test", width = 0), "positive")
})

# ==============================================================================
# UNIT TESTS: compose_from_config
# ==============================================================================

test_that("compose_from_config loads from YAML", {
  skip_if_not(exists("compose_from_config"), "compose_from_config not implemented")
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  skip_if_not(file.exists(config_path), "Config not created")

  # Provide test data for each panel
  data_list <- list(
    outlier_summary = outlier_data,
    imputation_summary = imputation_data
  )

  composed <- compose_from_config(
    "fig_forest_combined",
    infographic = FALSE,
    config_path = config_path,
    data_list = data_list
  )
  expect_s3_class(composed, "patchwork")
})

test_that("compose_from_config: errors on unknown figure", {
  skip_if_not(exists("compose_from_config"), "compose_from_config not implemented")
  expect_error(compose_from_config("nonexistent_figure"), "not found")
})

# ==============================================================================
# CONFIG VALIDATION TESTS
# ==============================================================================

test_that("figure_layouts.yaml exists", {
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  expect_true(
    file.exists(config_path),
    info = "figure_layouts.yaml must exist"
  )
})

test_that("figure_layouts.yaml is valid YAML", {
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  skip_if_not(file.exists(config_path), "Config not created")
  config <- yaml::read_yaml(config_path)
  expect_true("version" %in% names(config))
  expect_true("figures" %in% names(config))
})

test_that("figure_layouts.yaml contains fig_forest_combined", {
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  skip_if_not(file.exists(config_path), "Config not created")
  config <- yaml::read_yaml(config_path)
  expect_true("fig_forest_combined" %in% names(config$figures))
})

test_that("figure_layouts.yaml figures have required fields", {
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  skip_if_not(file.exists(config_path), "Config not created")
  config <- yaml::read_yaml(config_path)
  for (fig_name in names(config$figures)) {
    fig <- config$figures[[fig_name]]
    expect_true("layout" %in% names(fig), info = paste(fig_name, "missing layout"))
    expect_true("panels" %in% names(fig), info = paste(fig_name, "missing panels"))
  }
})

# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

test_that("Full pipeline: create + compose + save works", {
  skip_if_not(exists("create_forest_outlier"), "create_forest_outlier not implemented")
  skip_if_not(exists("create_forest_imputation"), "create_forest_imputation not implemented")
  skip_if_not(exists("compose_figures"), "compose_figures not implemented")
  skip_if_not(exists("save_publication_figure"), "save_publication_figure not implemented")

  # Create panels
  p1 <- create_forest_outlier(outlier_data, infographic = FALSE, show_legend = FALSE)
  p2 <- create_forest_imputation(imputation_data, infographic = FALSE, show_legend = FALSE)

  # Compose
  composed <- compose_figures(list(p1, p2), layout = "2x1", tag_levels = "A")

  # Save
  tmp_dir <- tempdir()
  save_publication_figure(composed, "test_combined", output_dir = tmp_dir, formats = c("png"))

  # Verify
  expect_true(file.exists(file.path(tmp_dir, "test_combined.png")))

  # Cleanup
  unlink(file.path(tmp_dir, "test_combined.png"))
})
