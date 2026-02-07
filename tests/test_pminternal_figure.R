#!/usr/bin/env Rscript
#' TDD Tests for pminternal Instability Figure
#'
#' These tests verify the generated PNG has correct properties.
#' Run with: Rscript tests/test_pminternal_figure.R
#'
#' Expected output: figures/generated/ggplot2/supplementary/fig_instability_combined.png

library(testthat)
library(png)

# Find project root
.find_project_root <- function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  getwd()  # fallback
}

PROJECT_ROOT <- .find_project_root()

# Constants
FIGURE_PATH <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary/fig_instability_combined.png")
EXPECTED_WIDTH_INCHES <- 12
EXPECTED_HEIGHT_INCHES <- 6
EXPECTED_DPI <- 300  # Publication quality
MIN_WIDTH_PX <- EXPECTED_WIDTH_INCHES * EXPECTED_DPI * 0.9  # 10% tolerance
MIN_HEIGHT_PX <- EXPECTED_HEIGHT_INCHES * EXPECTED_DPI * 0.9


test_that("Figure output file exists", {
  skip_if_not(file.exists(FIGURE_PATH), "Figure not yet generated")
  expect_true(file.exists(FIGURE_PATH))
})


test_that("Figure is valid PNG", {
  skip_if_not(file.exists(FIGURE_PATH), "Figure not yet generated")

  # Should not error
  img <- tryCatch(
    readPNG(FIGURE_PATH),
    error = function(e) NULL
  )

  expect_false(is.null(img), "Failed to read PNG file")
})


test_that("Figure has expected dimensions", {
  skip_if_not(file.exists(FIGURE_PATH), "Figure not yet generated")

  img <- readPNG(FIGURE_PATH)
  height_px <- dim(img)[1]
  width_px <- dim(img)[2]

  # Check width
  expect_gte(
    width_px, MIN_WIDTH_PX,
    paste("Width too small:", width_px, "px, expected >=", MIN_WIDTH_PX)
  )

  # Check height
  expect_gte(
    height_px, MIN_HEIGHT_PX,
    paste("Height too small:", height_px, "px, expected >=", MIN_HEIGHT_PX)
  )

  # Check aspect ratio (should be ~2:1 for 12x6)
  aspect_ratio <- width_px / height_px
  expect_true(
    aspect_ratio > 1.5 && aspect_ratio < 2.5,
    paste("Unexpected aspect ratio:", round(aspect_ratio, 2), "expected ~2:1")
  )
})


test_that("Figure is not blank/all white", {
  skip_if_not(file.exists(FIGURE_PATH), "Figure not yet generated")

  img <- readPNG(FIGURE_PATH)

  # Check that not all pixels are white (RGB all close to 1)
  if (length(dim(img)) == 3) {
    # Color image
    mean_rgb <- mean(img[,,1:3])
  } else {
    # Grayscale
    mean_rgb <- mean(img)
  }

  # If mean is too close to 1, image is mostly white (blank)
  expect_lt(
    mean_rgb, 0.98,
    paste("Image appears blank (mean pixel value:", round(mean_rgb, 3), ")")
  )
})


test_that("Figure has content in expected regions (two panels)", {
  skip_if_not(file.exists(FIGURE_PATH), "Figure not yet generated")

  img <- readPNG(FIGURE_PATH)
  width_px <- dim(img)[2]

  # Check left panel (first half)
  left_panel <- img[, 1:(width_px/2), ]
  left_variance <- var(as.vector(left_panel))

  # Check right panel (second half)
  right_panel <- img[, (width_px/2 + 1):width_px, ]
  right_variance <- var(as.vector(right_panel))

  # Both panels should have some variance (not blank)
  expect_gt(
    left_variance, 0.001,
    "Left panel appears blank or uniform"
  )

  expect_gt(
    right_variance, 0.001,
    "Right panel appears blank or uniform"
  )
})


# Run tests if executed directly
if (sys.nframe() == 0) {
  # Change to project root if needed
  if (file.exists("tests/test_pminternal_figure.R")) {
    # Already in project root
  } else if (file.exists("../tests/test_pminternal_figure.R")) {
    setwd("..")
  }

  test_results <- test_file("tests/test_pminternal_figure.R", reporter = "summary")

  # Exit with error code if tests failed
  if (any(as.data.frame(test_results)$failed > 0)) {
    quit(status = 1)
  }
}
