# Test: No Hardcoded Values in R Figure Scripts
# ==============================================
# Guardrail tests ensuring R scripts load from YAML/JSON, not hardcoded values.
#
# Run: Rscript -e "testthat::test_file('tests/test_r_guardrails/test_no_hardcoded_values.R')"
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(testthat)
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

# ==============================================================================
# TEST: No hardcoded model names in figure scripts
# ==============================================================================

test_that("No hardcoded model names in figure scripts", {
  figure_scripts <- list.files(
    file.path(PROJECT_ROOT, "src/r/figures"),
    pattern = "\\.R$",
    full.names = TRUE
  )

  if (length(figure_scripts) == 0) {
    skip("No R figure scripts found")
  }

  # Patterns that indicate hardcoded model names
  banned_patterns <- c(
    '"pupil-gt"',           # Model name in quotes
    '"MOMENT-gt-finetune"', # Model name
    '"ensemble-LOF',        # Model name
    '"OneClassSVM"',        # Model name
    '"SAITS"',              # Imputation method
    '"CSDI"'                # Imputation method
  )

  violations <- list()

  for (script in figure_scripts) {
    content <- paste(readLines(script, warn = FALSE), collapse = "\n")

    for (pattern in banned_patterns) {
      # Check for pattern outside comments
      lines <- strsplit(content, "\n")[[1]]
      for (i in seq_along(lines)) {
        line <- lines[i]
        # Skip comments
        if (grepl("^\\s*#", line)) next

        if (grepl(pattern, line, fixed = TRUE)) {
          # Allow in certain contexts
          # - String replacement calls
          # - Variable definitions from YAML
          # - Display name mapping
          allowed_contexts <- c("yaml::", "fromJSON", "_display_name", "load_", "%||%")
          if (any(sapply(allowed_contexts, function(ctx) grepl(ctx, line)))) next

          violations[[length(violations) + 1]] <- list(
            file = basename(script),
            line = i,
            pattern = pattern,
            content = substr(trimws(line), 1, 60)
          )
        }
      }
    }
  }

  if (length(violations) > 0) {
    msg <- "GUARDRAIL VIOLATION: Hardcoded model names in R scripts!\n\n"
    for (v in violations) {
      msg <- paste0(msg, sprintf("  %s:%d\n    Pattern: %s\n    Line: %s\n\n",
                                  v$file, v$line, v$pattern, v$content))
    }
    msg <- paste0(msg, "FIX: Load model names from YAML config or display_names.yaml")
    fail(msg)
  }
})

# ==============================================================================
# TEST: No hardcoded color vectors
# ==============================================================================

test_that("No hardcoded color vectors in figure scripts", {
  figure_scripts <- list.files(
    file.path(PROJECT_ROOT, "src/r/figures"),
    pattern = "\\.R$",
    full.names = TRUE
  )

  if (length(figure_scripts) == 0) {
    skip("No R figure scripts found")
  }

  # Pattern for hardcoded hex color vectors
  color_vector_pattern <- 'c\\("[^"]*#[0-9A-Fa-f]{6}'

  violations <- list()

  for (script in figure_scripts) {
    content <- paste(readLines(script, warn = FALSE), collapse = "\n")
    lines <- strsplit(content, "\n")[[1]]

    for (i in seq_along(lines)) {
      line <- lines[i]
      if (grepl("^\\s*#", line)) next  # Skip comments

      if (grepl(color_vector_pattern, line, perl = TRUE)) {
        violations[[length(violations) + 1]] <- list(
          file = basename(script),
          line = i,
          content = substr(trimws(line), 1, 80)
        )
      }
    }
  }

  if (length(violations) > 0) {
    msg <- "GUARDRAIL VIOLATION: Hardcoded color vectors in R scripts!\n\n"
    for (v in violations) {
      msg <- paste0(msg, sprintf("  %s:%d\n    %s\n\n", v$file, v$line, v$content))
    }
    msg <- paste0(msg, "FIX: Load colors from plot_hyperparam_combos.yaml via config_loader.R")
    fail(msg)
  }
})

# ==============================================================================
# TEST: All figure scripts load config from YAML
# ==============================================================================

test_that("All figure scripts use config loading functions", {
  figure_scripts <- list.files(
    file.path(PROJECT_ROOT, "src/r/figures"),
    pattern = "^fig.*\\.R$",  # Only files starting with "fig"
    full.names = TRUE
  )

  if (length(figure_scripts) == 0) {
    skip("No R figure scripts found")
  }

  # Patterns that indicate proper config loading
  config_patterns <- c(
    "load_figure_config",
    "load_figure_combos",
    "yaml::read_yaml",
    "validate_data_source",
    "load_figure_all"
  )

  missing_config <- c()

  for (script in figure_scripts) {
    content <- paste(readLines(script, warn = FALSE), collapse = "\n")

    has_config_loading <- any(sapply(config_patterns, function(p) grepl(p, content)))

    if (!has_config_loading) {
      missing_config <- c(missing_config, basename(script))
    }
  }

  if (length(missing_config) > 0) {
    msg <- sprintf(
      "GUARDRAIL VIOLATION: %d figure scripts don't load config from YAML:\n\n  %s\n\nFIX: Add config loading using config_loader.R functions.",
      length(missing_config),
      paste(missing_config, collapse = "\n  ")
    )
    # This is a warning, not a failure, since some scripts might have valid reasons
    warning(msg)
  }
})

# ==============================================================================
# TEST: No on-the-fly model selection (e.g., configs[1:4])
# ==============================================================================

test_that("No on-the-fly model selection", {
  figure_scripts <- list.files(
    file.path(PROJECT_ROOT, "src/r/figures"),
    pattern = "\\.R$",
    full.names = TRUE
  )

  if (length(figure_scripts) == 0) {
    skip("No R figure scripts found")
  }

  # Patterns indicating on-the-fly selection
  banned_patterns <- c(
    "configs\\[1:4\\]",        # Hardcoded slice
    "configs\\[1:min",         # Hardcoded slice with min
    "head\\(configs",          # Taking first N
    "configs %>% head",        # Pipe version
    'grepl.*"MOMENT".*filter'  # Pattern-based filtering
  )

  violations <- list()

  for (script in figure_scripts) {
    content <- paste(readLines(script, warn = FALSE), collapse = "\n")

    for (pattern in banned_patterns) {
      if (grepl(pattern, content, perl = TRUE)) {
        violations[[length(violations) + 1]] <- list(
          file = basename(script),
          pattern = pattern
        )
      }
    }
  }

  if (length(violations) > 0) {
    msg <- "GUARDRAIL VIOLATION: On-the-fly model selection detected!\n\n"
    for (v in violations) {
      msg <- paste0(msg, sprintf("  %s: uses pattern '%s'\n", v$file, v$pattern))
    }
    msg <- paste0(msg, "\nFIX: Define which configs to use in YAML, not in R code.")
    fail(msg)
  }
})

# ==============================================================================
# TEST: case_when not used for display names
# ==============================================================================

test_that("case_when not used for hardcoded display names", {
  figure_scripts <- list.files(
    file.path(PROJECT_ROOT, "src/r/figures"),
    pattern = "\\.R$",
    full.names = TRUE
  )

  if (length(figure_scripts) == 0) {
    skip("No R figure scripts found")
  }

  # Pattern for case_when with hardcoded strings (display name conversion)
  # This catches: case_when(method == "pupil-gt" ~ "Ground Truth", ...)
  case_when_pattern <- 'case_when.*~.*"[A-Z]'

  violations <- list()

  for (script in figure_scripts) {
    content <- paste(readLines(script, warn = FALSE), collapse = "\n")
    lines <- strsplit(content, "\n")[[1]]

    for (i in seq_along(lines)) {
      line <- lines[i]
      if (grepl("^\\s*#", line)) next

      if (grepl(case_when_pattern, line, perl = TRUE)) {
        # Check if it's for display name conversion
        if (grepl("display|label|name", line, ignore.case = TRUE)) {
          violations[[length(violations) + 1]] <- list(
            file = basename(script),
            line = i,
            content = substr(trimws(line), 1, 80)
          )
        }
      }
    }
  }

  if (length(violations) > 0) {
    msg <- "GUARDRAIL VIOLATION: Using case_when for display name conversion!\n\n"
    for (v in violations) {
      msg <- paste0(msg, sprintf("  %s:%d\n    %s\n\n", v$file, v$line, v$content))
    }
    msg <- paste0(msg, "FIX: Use apply_display_names() from common.R instead.")
    fail(msg)
  }
})

message("\n[test_no_hardcoded_values.R] All guardrail tests completed.")
