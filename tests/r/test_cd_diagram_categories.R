# CD Diagram Category Aggregation Tests
# ======================================
# TDD tests for Panel C: 5 preprocessing categories instead of individual pipelines.
#
# The 5 categories (consistent with other figures):
# 1. Ground Truth
# 2. Ensemble FM
# 3. Single-model FM
# 4. Deep Learning
# 5. Traditional
#
# Run with: Rscript -e "testthat::test_file('tests/r/test_cd_diagram_categories.R')"

library(testthat)
library(dplyr)

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

# Source common first (provides find_project_root for category_loader)
source(file.path(PROJECT_ROOT, "src/r/figure_system/common.R"))
# Source config_loader (provides load_color_definitions)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
# Source category loader
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))

# ==============================================================================
# TEST: Category Assignment from Method Names
# ==============================================================================

test_that("outlier methods map to correct raw categories", {
  # Ground Truth

  expect_equal(get_outlier_category("pupil-gt"), "Ground Truth")


  # Foundation Model (will display as "Single-model FM")
  expect_equal(get_outlier_category("MOMENT-gt-finetune"), "Foundation Model")
  expect_equal(get_outlier_category("MOMENT-gt-zeroshot"), "Foundation Model")
  expect_equal(get_outlier_category("UniTS-gt-finetune"), "Foundation Model")

  # Deep Learning
  expect_equal(get_outlier_category("TimesNet-gt"), "Deep Learning")

  # Traditional
  expect_equal(get_outlier_category("LOF"), "Traditional")
  expect_equal(get_outlier_category("OneClassSVM"), "Traditional")
  expect_equal(get_outlier_category("PROPHET"), "Traditional")
  expect_equal(get_outlier_category("SubPCA"), "Traditional")

  # Ensemble (will display as "Ensemble FM")
  expect_equal(get_outlier_category("ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"), "Ensemble")
  expect_equal(get_outlier_category("ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"), "Ensemble")
})

# ==============================================================================
# TEST: Display Name Mapping
# ==============================================================================

test_that("categories map to display names for figures", {
  # Direct mappings
  expect_equal(to_display_category("Ground Truth"), "Ground Truth")
  expect_equal(to_display_category("Traditional"), "Traditional")
  expect_equal(to_display_category("Deep Learning"), "Deep Learning")

  # Renamed categories
  expect_equal(to_display_category("Foundation Model"), "Single-model FM")
  expect_equal(to_display_category("Ensemble"), "Ensemble FM")
})

# ==============================================================================
# TEST: Category Order
# ==============================================================================

test_that("category order is consistent with other figures", {
  expected_order <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
  expect_equal(get_category_order(), expected_order)
})

# ==============================================================================
# TEST: Aggregation Produces Exactly 5 Categories
# ==============================================================================

test_that("aggregation by category produces exactly 5 groups", {
  # Load real metrics data
  metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")
  skip_if_not(file.exists(metrics_path), "Metrics file not available")

  df <- read.csv(metrics_path)

  # Filter to CatBoost
  df <- df %>% filter(classifier == "CatBoost")

  # Add category column
  df <- df %>%
    mutate(category = sapply(outlier_method, get_outlier_category)) %>%
    mutate(display_category = sapply(category, to_display_category))

  # Should have exactly 5 unique display categories
  expect_equal(length(unique(df$display_category)), 5)

  # All expected categories should be present
  expected <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
  expect_true(all(expected %in% unique(df$display_category)))
})

# ==============================================================================
# TEST: Category CD Matrix Structure
# ==============================================================================

test_that("category CD matrix has 5 columns (one per category)", {
  # Load real metrics data
  metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")
  skip_if_not(file.exists(metrics_path), "Metrics file not available")

  df <- read.csv(metrics_path)
  df <- df %>% filter(classifier == "CatBoost")

  # Create category matrix
  # Rows = imputation methods (folds), Columns = outlier categories
  category_df <- df %>%
    mutate(category = sapply(outlier_method, get_outlier_category)) %>%
    mutate(display_category = sapply(category, to_display_category)) %>%
    group_by(display_category, imputation_method) %>%
    summarise(auroc = mean(auroc, na.rm = TRUE), .groups = "drop")

  # Pivot to matrix
  wide_df <- category_df %>%
    tidyr::pivot_wider(
      id_cols = imputation_method,
      names_from = display_category,
      values_from = auroc
    )

  mat <- as.matrix(wide_df[, -1])

  # Should have exactly 5 columns
  expect_equal(ncol(mat), 5)

  # Column names should be the 5 categories
  expected_cats <- c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
  expect_true(all(colnames(mat) %in% expected_cats))
})

# ==============================================================================
# TEST: Category Aggregation Produces Reasonable Values
# ==============================================================================

test_that("category aggregation produces reasonable AUROC values", {
  metrics_path <- file.path(PROJECT_ROOT, "data/r_data/essential_metrics.csv")
  skip_if_not(file.exists(metrics_path), "Metrics file not available")

  df <- read.csv(metrics_path)
  df <- df %>% filter(classifier == "CatBoost")

  category_df <- df %>%
    mutate(category = sapply(outlier_method, get_outlier_category)) %>%
    mutate(display_category = sapply(category, to_display_category)) %>%
    group_by(display_category) %>%
    summarise(mean_auroc = mean(auroc, na.rm = TRUE), .groups = "drop")

  # Ground Truth should have highest AUROC (~0.91)
  gt_auroc <- category_df %>% filter(display_category == "Ground Truth") %>% pull(mean_auroc)
  expect_gt(gt_auroc, 0.85)

  # All categories should have AUROC > 0.5 (better than random)
  expect_true(all(category_df$mean_auroc > 0.5))

  # Ground Truth should be among the top performers (top 2)
  sorted_cats <- category_df %>% arrange(desc(mean_auroc)) %>% pull(display_category)
  expect_true("Ground Truth" %in% sorted_cats[1:2],
              info = paste("Ground Truth should be in top 2. Actual order:", paste(sorted_cats, collapse = ", ")))
})
