# Category Loader for Figure System
# ==================================
# SINGLE SOURCE OF TRUTH - Loads method categories from YAML.
# REPLACES ALL case_when() categorization blocks in R scripts!
#
# DO NOT use case_when(grepl(...)) for method categorization.
# Use these functions instead:
#   - categorize_outlier_methods()
#   - categorize_imputation_methods()
#   - get_category_colors()
#
# Created: 2026-01-28
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(yaml)
})

# Source common utilities
if (!exists("find_project_root", mode = "function")) {
  source(file.path(dirname(sys.frame(1)$ofile), "common.R"))
}

# Source config_loader for load_color_definitions (used for fallback colors)
if (!exists("load_color_definitions", mode = "function")) {
  source(file.path(dirname(sys.frame(1)$ofile), "config_loader.R"))
}

# Emergency fallback color (only used when YAML completely fails to load)
# This is the ONLY place this hex value should exist in the codebase
.EMERGENCY_FALLBACK_COLOR <- "#888888"  # nolint: hardcoded_color

# ==============================================================================
# YAML LOADING
# ==============================================================================

#' Load category mapping from YAML
#'
#' Loads configs/mlflow_registry/category_mapping.yaml.
#' This is the SINGLE SOURCE OF TRUTH for method categories.
#'
#' @return List with outlier_method_categories and imputation_method_categories
#' @export
load_category_mapping <- function() {
  project_root <- find_project_root()
  path <- file.path(project_root, "configs/mlflow_registry/category_mapping.yaml")

  if (!file.exists(path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: category_mapping.yaml not found at: %s\n\n",
      "This file is required for method categorization.",
      path
    ))
  }

  yaml::read_yaml(path)
}

#' Load method abbreviations from YAML
#'
#' Loads configs/mlflow_registry/method_abbreviations.yaml.
#' Used for CD diagrams where space is limited.
#'
#' @return List with outlier_method_abbreviations, imputation_method_abbreviations, etc.
#' @export
load_method_abbreviations <- function() {
  project_root <- find_project_root()
  path <- file.path(project_root, "configs/mlflow_registry/method_abbreviations.yaml")

  if (!file.exists(path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: method_abbreviations.yaml not found at: %s",
      path
    ))
  }

  yaml::read_yaml(path)
}

#' Load display names from YAML
#'
#' Loads configs/mlflow_registry/display_names.yaml.
#' Used for human-readable method names in legends.
#'
#' @return List with outlier_methods, imputation_methods, classifiers
#' @export
load_display_names <- function() {
  project_root <- find_project_root()
  path <- file.path(project_root, "configs/mlflow_registry/display_names.yaml")

  if (!file.exists(path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: display_names.yaml not found at: %s",
      path
    ))
  }

  yaml::read_yaml(path)
}

# ==============================================================================
# CATEGORY ASSIGNMENT (REPLACES case_when())
# ==============================================================================

#' Get category for a single outlier method
#'
#' Uses YAML config - NOT hardcoded patterns!
#' This function replaces all case_when(grepl(...)) blocks.
#'
#' @param method Character: outlier method name from database
#' @param config Optional: pre-loaded config (for efficiency in loops)
#' @return Character: category name (e.g., "Ground Truth", "Foundation Model")
#' @export
get_outlier_category <- function(method, config = NULL) {
  if (is.null(config)) {
    config <- load_category_mapping()
  }

  cats <- config$outlier_method_categories

  # Check exact matches first
  if (!is.null(cats$exact) && method %in% names(cats$exact)) {
    return(cats$exact[[method]])
  }

  # Check pattern matches in order
  for (rule in cats$patterns) {
    if (grepl(rule$pattern, method)) {
      return(rule$category)
    }
  }

  return("Unknown")
}

#' Get category for a single imputation method
#'
#' @param method Character: imputation method name
#' @param config Optional: pre-loaded config
#' @return Character: category name
#' @export
get_imputation_category <- function(method, config = NULL) {
  if (is.null(config)) {
    config <- load_category_mapping()
  }

  cats <- config$imputation_method_categories

  # Check exact matches first
  if (!is.null(cats$exact) && method %in% names(cats$exact)) {
    return(cats$exact[[method]])
  }

  # Check pattern matches in order
  for (rule in cats$patterns) {
    if (grepl(rule$pattern, method)) {
      return(rule$category)
    }
  }

  return("Unknown")
}

#' Categorize a vector of outlier methods
#'
#' Vectorized version for use in dplyr mutate() pipelines.
#' USE THIS instead of case_when(grepl(...)).
#'
#' @param methods Character vector: outlier method names
#' @return Character vector: categories
#' @examples
#' data %>% mutate(category = categorize_outlier_methods(outlier_method))
#' @export
categorize_outlier_methods <- function(methods) {
  config <- load_category_mapping()
  sapply(methods, get_outlier_category, config = config, USE.NAMES = FALSE)
}

#' Categorize a vector of imputation methods
#'
#' @param methods Character vector: imputation method names
#' @return Character vector: categories
#' @export
categorize_imputation_methods <- function(methods) {
  config <- load_category_mapping()
  sapply(methods, get_imputation_category, config = config, USE.NAMES = FALSE)
}

# ==============================================================================
# CATEGORY DISPLAY NAMES (for figure consistency)
# ==============================================================================

#' Convert raw category to display category
#'
#' Maps raw category names to figure-consistent display names:
#'   - "Foundation Model" -> "Single-model FM"
#'   - "Ensemble" -> "Ensemble FM"
#'   - Others unchanged
#'
#' @param category Character: raw category name
#' @return Character: display category name
#' @export
to_display_category <- function(category) {
  # Mapping from raw category to display category
  display_map <- c(
    "Ground Truth" = "Ground Truth",
    "Foundation Model" = "Single-model FM",
    "Ensemble" = "Ensemble FM",
    "Deep Learning" = "Deep Learning",
    "Traditional" = "Traditional",
    "Unknown" = "Unknown"
  )

  if (category %in% names(display_map)) {
    return(display_map[[category]])
  }

  warning(sprintf("Unknown category '%s', returning as-is", category))
  return(category)
}

#' Vectorized version of to_display_category
#'
#' @param categories Character vector of raw category names
#' @return Character vector of display category names
#' @export
to_display_categories <- function(categories) {
  sapply(categories, to_display_category, USE.NAMES = FALSE)
}

#' Get standard category order for figures
#'
#' Returns the canonical order of 5 categories used across all figures.
#' This ensures consistency in legends and facets.
#'
#' @return Character vector of 5 display category names in standard order
#' @export
get_category_order <- function() {
  c("Ground Truth", "Ensemble FM", "Single-model FM", "Deep Learning", "Traditional")
}

# ==============================================================================
# DISPLAY NAMES (REPLACES hardcoded strings)
# ==============================================================================

#' Get display name for an outlier method
#'
#' @param method Character: raw method name
#' @param config Optional: pre-loaded display_names config
#' @return Character: human-readable display name
#' @export
get_outlier_display_name <- function(method, config = NULL) {
  if (is.null(config)) {
    config <- load_display_names()
  }

  if (method %in% names(config$outlier_methods)) {
    entry <- config$outlier_methods[[method]]
    # Handle both old format (string) and new format (object with display_name)
    if (is.list(entry)) {
      return(entry$display_name)
    } else {
      return(entry)
    }
  }

  # Fallback: return as-is
  warning(sprintf("No display name found for outlier method: %s", method))
  return(method)
}

#' Get display name for an imputation method
#'
#' @param method Character: raw method name
#' @param config Optional: pre-loaded display_names config
#' @return Character: human-readable display name
#' @export
get_imputation_display_name <- function(method, config = NULL) {
  if (is.null(config)) {
    config <- load_display_names()
  }

  if (method %in% names(config$imputation_methods)) {
    entry <- config$imputation_methods[[method]]
    if (is.list(entry)) {
      return(entry$display_name)
    } else {
      return(entry)
    }
  }

  warning(sprintf("No display name found for imputation method: %s", method))
  return(method)
}

# ==============================================================================
# ABBREVIATIONS (REPLACES 73 hardcoded lines in cd_diagram.R)
# ==============================================================================

#' Get abbreviation for a method (outlier or imputation)
#'
#' @param method Character: method name
#' @param method_type Character: "outlier" or "imputation"
#' @param config Optional: pre-loaded abbreviations config
#' @return Character: short abbreviation for CD diagrams
#' @export
get_method_abbreviation <- function(method, method_type = "outlier", config = NULL) {
  if (is.null(config)) {
    config <- load_method_abbreviations()
  }

  abbrevs <- if (method_type == "outlier") {
    config$outlier_method_abbreviations
  } else {
    config$imputation_method_abbreviations
  }

  if (method %in% names(abbrevs)) {
    return(abbrevs[[method]])
  }

  # Fallback: truncate to max_length
  max_len <- config$abbreviation_rules$fallback_truncate %||% 10
  return(substr(method, 1, max_len))
}

#' Get abbreviation for a pipeline (outlier + imputation)
#'
#' @param outlier_method Character: outlier method name
#' @param imputation_method Character: imputation method name
#' @param config Optional: pre-loaded abbreviations config
#' @return Character: short pipeline abbreviation
#' @export
get_pipeline_abbreviation <- function(outlier_method, imputation_method, config = NULL) {
  if (is.null(config)) {
    config <- load_method_abbreviations()
  }

  # Check if exact pipeline is defined
  sep <- config$abbreviation_rules$separator %||% "+"
  pipeline_key <- paste(outlier_method, imputation_method, sep = " + ")

  if (!is.null(config$pipeline_abbreviations) &&
      pipeline_key %in% names(config$pipeline_abbreviations)) {
    return(config$pipeline_abbreviations[[pipeline_key]])
  }

  # Build from components
  out_abbrev <- get_method_abbreviation(outlier_method, "outlier", config)
  imp_abbrev <- get_method_abbreviation(imputation_method, "imputation", config)

  return(paste(out_abbrev, imp_abbrev, sep = sep))
}

# ==============================================================================
# COLORS (for category-based coloring)
# ==============================================================================

#' Get colors for method categories
#'
#' Returns a named vector of colors for categories.
#' USE THIS instead of hardcoded color vectors.
#'
#' @param config Optional: pre-loaded category_mapping config
#' @return Named character vector: category -> hex color
#' @export
get_category_colors <- function(config = NULL) {
  if (is.null(config)) {
    config <- load_category_mapping()
  }

  # Get color definitions
  color_defs <- config$category_colors

  # Map category display names to colors via outlier_category_display
  colors <- sapply(names(config$outlier_category_display), function(cat_name) {
    display_info <- config$outlier_category_display[[cat_name]]
    color_ref <- display_info$color_ref

    if (color_ref %in% names(color_defs)) {
      return(color_defs[[color_ref]])
    }

    warning(sprintf("Color not found for category '%s', using muted color", cat_name))
    return(color_defs[["--color-text-muted"]] %||% .EMERGENCY_FALLBACK_COLOR)
  })

  names(colors) <- names(config$outlier_category_display)
  return(colors)
}

#' Get a single category color
#'
#' @param category Character: category name (e.g., "Ground Truth")
#' @param config Optional: pre-loaded config
#' @return Character: hex color code
#' @export
get_category_color <- function(category, config = NULL) {
  colors <- get_category_colors(config)

  if (category %in% names(colors)) {
    return(colors[[category]])
  }

  # Get color_defs for fallback
  color_defs <- tryCatch(
    load_color_definitions(),
    error = function(e) list(`--color-text-muted` = .EMERGENCY_FALLBACK_COLOR)
  )
  warning(sprintf("Unknown category '%s', using muted color", category))
  return(color_defs[["--color-text-muted"]] %||% .EMERGENCY_FALLBACK_COLOR)
}

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

#' Create category-colored palette for ggplot scale_color_manual
#'
#' Returns a named vector suitable for scale_color_manual(values = ...).
#'
#' @param data Data frame with a 'category' column
#' @return Named character vector for scale_color_manual
#' @export
create_category_palette <- function(data) {
  categories <- unique(data$category)
  colors <- get_category_colors()

  # Filter to only categories present in data
  palette <- colors[categories]

  # Fill missing with gray
  missing <- setdiff(categories, names(colors))
  if (length(missing) > 0) {
    # Get muted color from config for fallback
    color_defs <- tryCatch(
      load_color_definitions(),
      error = function(e) list(`--color-text-muted` = .EMERGENCY_FALLBACK_COLOR)
    )
    muted_color <- color_defs[["--color-text-muted"]] %||% .EMERGENCY_FALLBACK_COLOR
    warning(sprintf("Unknown categories: %s", paste(missing, collapse = ", ")))
    for (cat in missing) {
      palette[[cat]] <- muted_color
    }
  }

  return(palette)
}

message("[category_loader] Category loader module loaded successfully")
