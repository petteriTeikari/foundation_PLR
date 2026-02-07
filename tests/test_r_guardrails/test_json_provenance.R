# Test: JSON Data Provenance Validation
# ======================================
# Guardrail tests ensuring all JSON exports have proper metadata.
#
# Run: Rscript -e "testthat::test_file('tests/test_r_guardrails/test_json_provenance.R')"
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(testthat)
  library(jsonlite)
})

# Find project root
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
R_DATA_DIR <- file.path(PROJECT_ROOT, "outputs/r_data")

# ==============================================================================
# TEST: All JSON exports have metadata
# ==============================================================================

test_that("All JSON exports have metadata section", {
  json_files <- list.files(R_DATA_DIR, pattern = "\\.json$", full.names = TRUE)

  if (length(json_files) == 0) {
    skip("No JSON files found in outputs/r_data/")
  }

  missing_metadata <- c()

  for (json_file in json_files) {
    tryCatch({
      data <- fromJSON(json_file, simplifyVector = FALSE)
      if (is.null(data$metadata)) {
        missing_metadata <- c(missing_metadata, basename(json_file))
      }
    }, error = function(e) {
      warning(sprintf("Could not parse %s: %s", basename(json_file), e$message))
    })
  }

  if (length(missing_metadata) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: %d JSON files missing metadata section:\n\n  %s\n\nFIX: Add metadata with generator, data_source, and db_hash.",
      length(missing_metadata),
      paste(missing_metadata, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: All JSON exports have data source hash
# ==============================================================================

test_that("All JSON exports have data_source with db_hash", {
  json_files <- list.files(R_DATA_DIR, pattern = "\\.json$", full.names = TRUE)

  if (length(json_files) == 0) {
    skip("No JSON files found")
  }

  missing_hash <- c()

  for (json_file in json_files) {
    tryCatch({
      data <- fromJSON(json_file, simplifyVector = FALSE)

      # Check for db_hash (may be in different locations)
      has_hash <- FALSE
      if (!is.null(data$metadata$data_source$db_hash)) has_hash <- TRUE
      if (!is.null(data$metadata$db_hash)) has_hash <- TRUE
      if (!is.null(data$metadata$data_source$hash)) has_hash <- TRUE

      if (!has_hash) {
        missing_hash <- c(missing_hash, basename(json_file))
      }
    }, error = function(e) {
      # Skip unparseable files
    })
  }

  if (length(missing_hash) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: %d JSON files missing db_hash for provenance tracking:\n\n  %s\n\nFIX: Run python scripts/fix_json_provenance.py or regenerate exports.",
      length(missing_hash),
      paste(missing_hash, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: All JSON exports have generator field
# ==============================================================================

test_that("All JSON exports have generator field", {
  json_files <- list.files(R_DATA_DIR, pattern = "\\.json$", full.names = TRUE)

  if (length(json_files) == 0) {
    skip("No JSON files found")
  }

  missing_generator <- c()

  for (json_file in json_files) {
    tryCatch({
      data <- fromJSON(json_file, simplifyVector = FALSE)
      if (is.null(data$metadata$generator)) {
        missing_generator <- c(missing_generator, basename(json_file))
      }
    }, error = function(e) {
      # Skip unparseable files
    })
  }

  if (length(missing_generator) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: %d JSON files missing generator field:\n\n  %s\n\nFIX: Add metadata.generator with the export script path.",
      length(missing_generator),
      paste(missing_generator, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: No synthetic data indicators
# ==============================================================================

test_that("Predictions are from REAL data, not synthetic", {
  json_files <- list.files(R_DATA_DIR, pattern = "prediction.*\\.json$", full.names = TRUE)

  if (length(json_files) == 0) {
    skip("No prediction JSON files found")
  }

  synthetic_indicators <- c("synthetic", "random", "fake", "mock", "test_data")
  suspicious_files <- c()

  for (json_file in json_files) {
    tryCatch({
      content <- readLines(json_file, warn = FALSE)
      content_str <- paste(content, collapse = "\n")

      for (indicator in synthetic_indicators) {
        if (grepl(indicator, content_str, ignore.case = TRUE)) {
          suspicious_files <- c(suspicious_files, basename(json_file))
          break
        }
      }
    }, error = function(e) {
      # Skip unreadable files
    })
  }

  if (length(suspicious_files) > 0) {
    msg <- sprintf(
      "CRITICAL: %d files may contain synthetic data:\n\n  %s\n\nThis is a CRITICAL FAILURE for scientific integrity!",
      length(suspicious_files),
      paste(suspicious_files, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: ROC/RC data has correct schema
# ==============================================================================

test_that("roc_rc_data.json has correct schema", {
  json_path <- file.path(R_DATA_DIR, "roc_rc_data.json")

  if (!file.exists(json_path)) {
    skip("roc_rc_data.json not found - run export_roc_rc_data.py first")
  }

  data <- fromJSON(json_path, simplifyVector = FALSE)

  # Check structure
  expect_true("data" %in% names(data), info = "Missing 'data' section")
  expect_true("configs" %in% names(data$data), info = "Missing 'configs' in data")
  expect_gte(length(data$data$configs), 9, info = "Expected at least 9 configs")

  # Check each config
  for (config in data$data$configs) {
    expect_true(!is.null(config$id), info = sprintf("Config missing 'id'"))
    expect_true(!is.null(config$roc), info = sprintf("Config '%s' missing 'roc'", config$id))
    expect_true(!is.null(config$rc), info = sprintf("Config '%s' missing 'rc'", config$id))

    # ROC curve validity
    if (!is.null(config$roc$fpr)) {
      expect_equal(config$roc$fpr[[1]], 0, tolerance = 0.01,
                   info = "ROC FPR should start at 0")
      expect_equal(tail(config$roc$fpr, 1)[[1]], 1, tolerance = 0.01,
                   info = "ROC FPR should end at 1")
    }
  }
})

# ==============================================================================
# TEST: Selective classification data has correct schema
# ==============================================================================

test_that("selective_classification_data.json has correct schema", {
  json_path <- file.path(R_DATA_DIR, "selective_classification_data.json")

  if (!file.exists(json_path)) {
    skip("selective_classification_data.json not found - run export script first")
  }

  data <- fromJSON(json_path, simplifyVector = FALSE)

  # Check structure
  expect_true("data" %in% names(data))
  expect_true("retention_levels" %in% names(data$data))
  expect_true("configs" %in% names(data$data))

  # Check retention levels
  levels <- unlist(data$data$retention_levels)
  expect_true(all(levels >= 0 & levels <= 1),
              info = "Retention levels must be between 0 and 1")

  # Check configs have required fields
  for (config in data$data$configs) {
    expect_true(!is.null(config$auroc_at_retention))
    expect_true(!is.null(config$net_benefit_at_retention))
    expect_true(!is.null(config$scaled_brier_at_retention))
  }
})

message("\n[test_json_provenance.R] All provenance tests completed.")
