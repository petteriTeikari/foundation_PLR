# Foundation PLR Data Loading Utilities
# Load JSON/CSV data exported from Python pipeline
# Created: 2026-01-25

library(jsonlite)
library(dplyr)
library(readr)

# ==============================================================================
# PROJECT ROOT FINDER
# ==============================================================================

.find_project_root <- function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) {
      return(dir)
    }
    dir <- dirname(dir)
  }
  # Fallback to working directory
  getwd()
}

PROJECT_ROOT <- .find_project_root()

# ==============================================================================
# DATA PATHS (Canonical location: data/)
# ==============================================================================

DATA_DIR <- file.path(PROJECT_ROOT, "data", "r_data")
SHAP_DIR <- "figures/generated/shap"

# ==============================================================================
# LOADING FUNCTIONS
# ==============================================================================

#' Load essential metrics CSV
#' @return data.frame with all config metrics
#' @export
load_essential_metrics <- function(path = file.path(DATA_DIR, "essential_metrics.csv")) {
  if (!file.exists(path)) {
    stop("Metrics file not found: ", path, "\nRun: python scripts/export_data_for_r.py")
  }
  read_csv(path, show_col_types = FALSE)
}

#' Load top-10 config JSON
#' @return list with config details
#' @export
load_top10_configs <- function(path = file.path(DATA_DIR, "top10_configs.json")) {
  if (!file.exists(path)) {
    stop("Config file not found: ", path, "\nRun: python scripts/export_data_for_r.py")
  }
  fromJSON(path)
}

#' Load SHAP feature importance
#' @return data.frame with feature importance per config
#' @export
load_shap_importance <- function(path = file.path(DATA_DIR, "shap_feature_importance.json")) {
  if (!file.exists(path)) {
    # Try the original SHAP figures data
    alt_path <- file.path(SHAP_DIR, "shap_figures_data.json")
    if (file.exists(alt_path)) {
      message("Using shap_figures_data.json from figures directory")
      path <- alt_path
    } else {
      stop("SHAP file not found: ", path, "\nRun: python scripts/export_shap_for_r.py")
    }
  }
  fromJSON(path)
}

#' Load per-sample SHAP values for beeswarm plots
#' @return data.frame with sample-level SHAP values
#' @export
load_shap_samples <- function(path = file.path(DATA_DIR, "shap_per_sample.json")) {
  if (!file.exists(path)) {
    stop("SHAP samples not found: ", path, "\nRun: python scripts/export_shap_for_r.py")
  }
  fromJSON(path)
}

#' Load VIF analysis
#' @return data.frame with VIF per feature
#' @export
load_vif <- function(path = file.path(DATA_DIR, "vif_analysis.json")) {
  if (!file.exists(path)) {
    stop("VIF file not found: ", path, "\nRun: python scripts/compute_vif.py")
  }
  fromJSON(path)
}

#' Load DCA curves data
#' @export
load_dca <- function(path = file.path(DATA_DIR, "dca_curves.json")) {
  if (!file.exists(path)) {
    stop("DCA file not found: ", path)
  }
  fromJSON(path)
}

#' Load calibration data
#' @export
load_calibration <- function(path = file.path(DATA_DIR, "calibration_data.json")) {
  if (!file.exists(path)) {
    stop("Calibration file not found: ", path)
  }
  fromJSON(path)
}

# ==============================================================================
# DATA VALIDATION
# ==============================================================================

#' Validate JSON metadata
#' @param data List loaded from JSON
#' @return TRUE if valid, error otherwise
validate_json_metadata <- function(data) {
  if (!"metadata" %in% names(data)) {
    warning("JSON missing metadata field")
    return(FALSE)
  }

  required <- c("created", "schema_version", "generator")
  missing <- required[!required %in% names(data$metadata)]
  if (length(missing) > 0) {
    warning("Metadata missing fields: ", paste(missing, collapse = ", "))
    return(FALSE)
  }

  TRUE
}

#' Load JSON with metadata validation
#' @export
load_json_with_validation <- function(path) {
  data <- fromJSON(path)
  if (validate_json_metadata(data)) {
    message("Data source: ", data$metadata$generator)
    message("Created: ", data$metadata$created)
  }
  data
}

# ==============================================================================
# DATA TRANSFORMATIONS
# ==============================================================================

#' Prepare SHAP data for beeswarm plot
#' @export
prepare_shap_beeswarm <- function(shap_data) {
  # Reshape to long format
  configs <- shap_data$configs

  all_data <- lapply(seq_along(configs), function(i) {
    cfg <- configs[[i]]
    importance <- cfg$feature_importance

    data.frame(
      config = cfg$name,
      feature = importance$feature,
      mean_abs_shap = importance$mean_abs_shap,
      std = importance$std,
      ci_lo = importance$ci_lo,
      ci_hi = importance$ci_hi,
      stringsAsFactors = FALSE
    )
  })

  bind_rows(all_data)
}

#' Add wavelength column based on feature name
#' @export
add_wavelength <- function(df) {
  df %>%
    mutate(
      wavelength = case_when(
        grepl("^Blue_|^amp_bin_0|^amp_bin_1|^amp_bin_2|^amp_bin_3|^amp_bin_4", feature) ~ "Blue (469nm)",
        grepl("^Red_|^amp_bin_5|^amp_bin_6|^amp_bin_7|^amp_bin_8|^amp_bin_9", feature) ~ "Red (640nm)",
        TRUE ~ "Combined"
      )
    )
}

#' Compute VIF concern level
#' @export
add_vif_concern <- function(df) {
  df %>%
    mutate(
      concern = case_when(
        VIF >= 10 ~ "High",
        VIF >= 5 ~ "Moderate",
        TRUE ~ "OK"
      ),
      concern = factor(concern, levels = c("OK", "Moderate", "High"))
    )
}

message("Data loading utilities ready.")
message("Use load_shap_importance(), load_vif(), etc.")
