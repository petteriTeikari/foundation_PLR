# Test: YAML Configuration Validation
# ====================================
# Guardrail tests for YAML config completeness and consistency.
#
# Run: Rscript -e "testthat::test_file('tests/test_r_guardrails/test_yaml_configs.R')"
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(testthat)
  library(yaml)
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
VIZ_CONFIG_DIR <- file.path(PROJECT_ROOT, "configs/VISUALIZATION")

# ==============================================================================
# TEST: All combo_source references in figure_layouts.yaml exist
# ==============================================================================

test_that("All combo_source references exist in plot_hyperparam_combos.yaml", {
  layouts <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "figure_layouts.yaml"))
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  invalid_refs <- c()

  for (fig_id in names(layouts$figures)) {
    fig <- layouts$figures[[fig_id]]
    if (!is.null(fig$combo_source)) {
      combo_source <- fig$combo_source

      # Check if it's a preset group
      is_preset <- combo_source %in% names(combos$preset_groups)

      # Check if it's a direct section
      is_section <- combo_source %in% names(combos)

      if (!is_preset && !is_section) {
        invalid_refs <- c(invalid_refs, sprintf("%s: %s", fig_id, combo_source))
      }
    }
  }

  if (length(invalid_refs) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: Figures reference non-existent combo_source:\n\n  %s",
      paste(invalid_refs, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: All combos have required fields
# ==============================================================================

test_that("All combos have required fields (id, name, outlier_method, imputation_method)", {
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  required_fields <- c("id", "name", "outlier_method", "imputation_method")
  missing <- list()

  combo_sections <- c("standard_combos", "extended_combos")
  for (section in combo_sections) {
    for (combo in combos[[section]]) {
      for (field in required_fields) {
        if (is.null(combo[[field]])) {
          missing[[length(missing) + 1]] <- list(
            section = section,
            combo_id = combo$id %||% "MISSING_ID",
            field = field
          )
        }
      }
    }
  }

  if (length(missing) > 0) {
    msg <- "GUARDRAIL VIOLATION: Combos missing required fields:\n\n"
    for (m in missing) {
      msg <- paste0(msg, sprintf("  %s.%s: missing '%s'\n", m$section, m$combo_id, m$field))
    }
    fail(msg)
  }
})

# ==============================================================================
# TEST: All color_refs reference color_definitions
# ==============================================================================

test_that("All color_ref values exist in color_definitions", {
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  color_defs <- names(combos$color_definitions)
  invalid_refs <- c()

  # Check shap_figure_combos
  if (!is.null(combos$shap_figure_combos$configs)) {
    for (combo in combos$shap_figure_combos$configs) {
      if (!is.null(combo$color_ref)) {
        if (!combo$color_ref %in% color_defs) {
          invalid_refs <- c(invalid_refs, sprintf("shap_figure_combos.%s: %s",
                                                   combo$id, combo$color_ref))
        }
      }
    }
  }

  if (length(invalid_refs) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: color_ref values not in color_definitions:\n\n  %s",
      paste(invalid_refs, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: No duplicate combo IDs
# ==============================================================================

test_that("No duplicate combo IDs across sections", {
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  all_ids <- c()
  duplicates <- c()

  combo_sections <- c("standard_combos", "extended_combos")
  for (section in combo_sections) {
    for (combo in combos[[section]]) {
      if (!is.null(combo$id)) {
        if (combo$id %in% all_ids) {
          duplicates <- c(duplicates, combo$id)
        }
        all_ids <- c(all_ids, combo$id)
      }
    }
  }

  if (length(duplicates) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: Duplicate combo IDs:\n\n  %s",
      paste(duplicates, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: Preset groups reference valid combo IDs
# ==============================================================================

test_that("All preset group combos exist", {
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  # Collect all valid IDs
  valid_ids <- c()
  for (combo in combos$standard_combos) {
    valid_ids <- c(valid_ids, combo$id)
  }
  for (combo in combos$extended_combos) {
    valid_ids <- c(valid_ids, combo$id)
  }

  invalid_refs <- c()

  for (preset_name in names(combos$preset_groups)) {
    preset <- combos$preset_groups[[preset_name]]
    for (combo_id in preset$combos) {
      if (!combo_id %in% valid_ids) {
        invalid_refs <- c(invalid_refs, sprintf("%s: %s", preset_name, combo_id))
      }
    }
  }

  if (length(invalid_refs) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: Preset groups reference invalid combo IDs:\n\n  %s",
      paste(invalid_refs, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: No misleading combo names
# ==============================================================================

test_that("No misleading 'baseline' or 'simple' names for FM pipelines", {
  combos <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "plot_hyperparam_combos.yaml"))

  fm_indicators <- c("MOMENT", "SAITS", "CSDI", "TimesNet", "UniTS")
  misleading <- c()

  combo_sections <- c("standard_combos", "extended_combos")
  for (section in combo_sections) {
    for (combo in combos[[section]]) {
      combo_id <- combo$id %||% ""
      imputation <- combo$imputation_method %||% ""

      # Check if uses FM
      uses_fm <- any(sapply(fm_indicators, function(fm) grepl(fm, imputation)))

      # Check if name suggests "simple" or "baseline"
      if (uses_fm && (grepl("simple", combo_id, ignore.case = TRUE) ||
                       grepl("baseline", combo_id, ignore.case = TRUE))) {
        misleading <- c(misleading, sprintf("%s (uses %s)", combo_id, imputation))
      }
    }
  }

  if (length(misleading) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: Misleading combo names (FM used but called 'simple/baseline'):\n\n  %s",
      paste(misleading, collapse = "\n  ")
    )
    fail(msg)
  }
})

# ==============================================================================
# TEST: Figure layouts have required dimensions
# ==============================================================================

test_that("All figures have required dimension fields", {
  layouts <- yaml::read_yaml(file.path(VIZ_CONFIG_DIR, "figure_layouts.yaml"))

  missing_dims <- c()

  for (fig_id in names(layouts$figures)) {
    fig <- layouts$figures[[fig_id]]

    if (is.null(fig$dimensions) ||
        is.null(fig$dimensions$width) ||
        is.null(fig$dimensions$height)) {
      missing_dims <- c(missing_dims, fig_id)
    }
  }

  if (length(missing_dims) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: Figures missing dimension specifications:\n\n  %s",
      paste(missing_dims, collapse = "\n  ")
    )
    fail(msg)
  }
})

message("\n[test_yaml_configs.R] All YAML config tests completed.")

# Null coalescing operator for R < 4.1
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}
