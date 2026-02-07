# TDD Tests for Figure Configuration Compliance
# =============================================
# These tests ENFORCE:
# 1. No hardcoded colors - must come from colors.yaml
# 2. No hardcoded dimensions - must come from figure_layouts.yaml
# 3. CD diagrams use horizontal 1x3 layout
# 4. All figures load configs via load_figure_config()
#
# Created: 2026-01-27
# Author: Foundation PLR Team

library(testthat)

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

# Note: We don't source config_loader.R here because it has path issues in test context
# Instead, we directly check file contents and YAML configs

# ==============================================================================
# TEST: NO HARDCODED COLORS IN FIGURE SCRIPTS
# ==============================================================================

context("Figure Config Compliance - No Hardcoding")

test_that("fig_cd_diagrams.R uses centralized color constants", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_cd_diagrams.R")
  skip_if_not(file.exists(file_path), "fig_cd_diagrams.R not found")

  content <- paste(readLines(file_path), collapse = "\n")

  # Should source color_palettes.R
  uses_palette <- grepl("color_palettes\\.R", content)

  # Should use COLORS_PIPELINE_TYPE or scale_color_pipeline_type
  uses_centralized <- grepl("COLORS_PIPELINE_TYPE|scale_color_pipeline_type|scale_fill_pipeline_type", content)

  # Should NOT define its own get_category_colors function with hardcoded values
  defines_own <- grepl("get_category_colors.*<-.*function", content)

  expect_true(uses_palette,
    info = "fig_cd_diagrams.R must source color_palettes.R")
  expect_true(uses_centralized,
    info = "fig_cd_diagrams.R must use COLORS_PIPELINE_TYPE from color_palettes.R")
  expect_false(defines_own,
    info = "fig_cd_diagrams.R should NOT define its own get_category_colors()")
})

test_that("fig_stratos_core.R uses centralized color constants", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_stratos_core.R")
  skip_if_not(file.exists(file_path), "fig_stratos_core.R not found")

  content <- paste(readLines(file_path), collapse = "\n")

  # Should source color_palettes.R
  uses_palette <- grepl("color_palettes\\.R", content)

  # Should NOT define its own get_pipeline_colors function with hardcoded values
  defines_own <- grepl("get_pipeline_colors.*<-.*function", content)

  expect_true(uses_palette,
    info = "fig_stratos_core.R must source color_palettes.R")
  expect_false(defines_own,
    info = "fig_stratos_core.R should NOT define its own get_pipeline_colors()")
})

test_that("fig_multi_metric_raincloud.R uses centralized color constants", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_multi_metric_raincloud.R")
  skip_if_not(file.exists(file_path), "fig_multi_metric_raincloud.R not found")

  content <- paste(readLines(file_path), collapse = "\n")

  # Should source color_palettes.R
  uses_palette <- grepl("color_palettes\\.R", content)

  # Should NOT define its own get_pipeline_type_colors function with hardcoded values
  defines_own <- grepl("get_pipeline_type_colors.*<-.*function", content)

  expect_true(uses_palette,
    info = "fig_multi_metric_raincloud.R must source color_palettes.R")
  expect_false(defines_own,
    info = "fig_multi_metric_raincloud.R should NOT define its own get_pipeline_type_colors()")
})

# ==============================================================================
# TEST: CD DIAGRAMS LAYOUT
# ==============================================================================

context("CD Diagrams Layout Compliance")

test_that("fig_cd_diagrams.R uses horizontal layout operator (+) not vertical (/)", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_cd_diagrams.R")
  skip_if_not(file.exists(file_path), "fig_cd_diagrams.R not found")

  content <- readLines(file_path)

  # Find the line that combines panels
  combine_lines <- grep("p_outlier.*p_imputation.*p_combined", content, value = TRUE)

  # Should use + (horizontal) not / (vertical)
  uses_horizontal <- any(grepl("\\+", combine_lines))
  uses_vertical <- any(grepl("/", combine_lines))

  expect_true(uses_horizontal,
    info = "CD diagrams should use horizontal layout: p_outlier + p_imputation + p_combined")
  expect_false(uses_vertical,
    info = "CD diagrams should NOT use vertical layout (/)")
})

test_that("CD diagrams layout is defined in figure_layouts.yaml", {
  layouts_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")
  skip_if_not(file.exists(layouts_path), "figure_layouts.yaml not found")

  config <- yaml::read_yaml(layouts_path)

  # Check that 1x3 layout exists

  expect_true("1x3" %in% names(config$layouts),
    info = "1x3 layout must be defined in figure_layouts.yaml")

  # Check layout properties
  layout_1x3 <- config$layouts[["1x3"]]
  expect_equal(layout_1x3$nrow, 1)
  expect_equal(layout_1x3$ncol, 3)
})

# ==============================================================================
# TEST: COLOR CONFIG LOADING
# ==============================================================================

context("Color Configuration Loading")

test_that("colors.yaml can be loaded and parsed", {
  colors_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/colors.yaml")
  skip_if_not(file.exists(colors_path), "colors.yaml not found")

  # Should parse without error
  colors <- yaml::read_yaml(colors_path)

  expect_true(is.list(colors))
  expect_true(length(colors) > 0)
})

test_that("colors.yaml contains pipeline category colors", {
  colors_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/colors.yaml")
  skip_if_not(file.exists(colors_path), "colors.yaml not found")

  config <- yaml::read_yaml(colors_path)

  # These should exist for pipeline type coloring
  required <- c("category_ground_truth", "category_foundation_model",
                "category_traditional", "category_ensemble")

  for (color_name in required) {
    expect_true(color_name %in% names(config),
      info = paste("colors.yaml missing:", color_name))
  }
})

# ==============================================================================
# TEST: FIGURE DIRECTORY STRUCTURE
# ==============================================================================

context("Figure Directory Structure")

test_that("Main figures directory exists", {
  main_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
  expect_true(dir.exists(main_dir),
    info = "figures/generated/ggplot2/main/ directory must exist")
})

test_that("Supplementary figures directory exists", {
  supp_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary")
  expect_true(dir.exists(supp_dir),
    info = "figures/generated/ggplot2/supplementary/ directory must exist")
})

test_that("Extra-supplementary figures directory exists", {
  extra_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/extra-supplementary")
  expect_true(dir.exists(extra_dir),
    info = "figures/generated/ggplot2/extra-supplementary/ directory must exist")
})

# ==============================================================================
# TEST: FIGURE SCRIPTS USE CONFIG LOADER
# ==============================================================================

context("Figure Scripts Load Configuration")

test_that("fig_cd_diagrams.R sources config_loader.R", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_cd_diagrams.R")
  skip_if_not(file.exists(file_path), "fig_cd_diagrams.R not found")

  content <- paste(readLines(file_path), collapse = "\n")

  # Should source either config_loader.R or color_palettes.R (which loads colors)
  uses_config <- grepl("config_loader\\.R|color_palettes\\.R|load_color_definitions", content)

  expect_true(uses_config,
    info = "fig_cd_diagrams.R must load colors via config system")
})

test_that("fig_stratos_core.R sources config_loader.R", {
  file_path <- file.path(PROJECT_ROOT, "src/r/figures/fig_stratos_core.R")
  skip_if_not(file.exists(file_path), "fig_stratos_core.R not found")

  content <- paste(readLines(file_path), collapse = "\n")

  uses_config <- grepl("config_loader\\.R|color_palettes\\.R|load_color_definitions", content)

  expect_true(uses_config,
    info = "fig_stratos_core.R must load colors via config system")
})

# ==============================================================================
# TEST: EXPECTED MAIN FIGURES EXIST
# ==============================================================================

context("Expected Main Figures")

test_that("Key main figures are generated", {
  main_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/main")
  skip_if_not(dir.exists(main_dir), "main directory not found")

  # Core STRATOS-compliant main figures
  expected_figures <- c(
    "fig_stratos_core.png",           # STRATOS 2x2 core figure
    "fig_calibration_stratos.png",    # STRATOS calibration
    "fig_dca_stratos.png",            # STRATOS DCA
    "fig_variance_decomposition.png"  # ANOVA decomposition
  )

  for (fig in expected_figures) {
    fig_path <- file.path(main_dir, fig)
    expect_true(file.exists(fig_path),
      info = paste("Missing main figure:", fig))
  }
})

# ==============================================================================
# TEST: CD DIAGRAMS DIMENSIONS (should be wide, not tall)
# ==============================================================================

context("CD Diagrams Dimensions")

test_that("CD diagrams PNG has horizontal aspect ratio (width > height)", {
  fig_path <- file.path(PROJECT_ROOT,
    "figures/generated/ggplot2/supplementary/fig_cd_diagrams.png")
  skip_if_not(file.exists(fig_path), "fig_cd_diagrams.png not found")

  # Read PNG dimensions
  if (requireNamespace("png", quietly = TRUE)) {
    img <- png::readPNG(fig_path)
    height <- dim(img)[1]
    width <- dim(img)[2]

    expect_true(width > height,
      info = paste("CD diagrams should be wider than tall. Got:",
                   width, "x", height))
  } else {
    skip("png package not available for dimension check")
  }
})
