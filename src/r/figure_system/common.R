# Common Utilities for Figure System
# ===================================
# Shared functions used across figure_factory.R, compose_figures.R, save_figure.R
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(yaml)
})

# ==============================================================================
# PROJECT ROOT FINDER
# ==============================================================================

#' Find the project root directory
#'
#' Searches upward from current directory for common project markers.
#'
#' @return Character string with absolute path to project root
#' @export
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

# ==============================================================================
# COLOR LOADING FROM YAML
# ==============================================================================

# Cache for loaded colors
.color_cache <- new.env(parent = emptyenv())

#' Load category colors from YAML config
#'
#' Loads colors from configs/VISUALIZATION/combos.yaml color_definitions
#' and maps them to category names used in forest plots.
#'
#' @param category_type Character: "outlier" or "imputation" to get appropriate palette
#' @return Named character vector of colors
#' @export
load_category_colors <- function(category_type = "outlier") {
  # Check cache
  cache_key <- paste0("category_colors_", category_type)
  if (exists(cache_key, envir = .color_cache)) {
    return(get(cache_key, envir = .color_cache))
  }

  # Load from YAML (Single Source of Truth: combos.yaml)
  project_root <- find_project_root()
  combos_path <- file.path(project_root, "configs/VISUALIZATION/combos.yaml")

  if (!file.exists(combos_path)) {
    stop("combos.yaml not found! This is REQUIRED for color definitions.")
  }

  combos_config <- yaml::read_yaml(combos_path)
  color_defs <- combos_config$color_definitions

  if (is.null(color_defs)) {
    stop("color_definitions not found in combos.yaml!")
  }

  # Map category names to colors from YAML
  category_colors <- c(
    "Ground Truth" = color_defs[["--color-category-ground-truth"]],
    "Foundation Model" = color_defs[["--color-category-foundation-model"]],
    "Traditional" = color_defs[["--color-category-traditional"]],
    "Deep Learning" = color_defs[["--color-category-deep-learning"]],
    "Ensemble" = color_defs[["--color-category-ensemble"]]
  )

  # Cache the result
  assign(cache_key, category_colors, envir = .color_cache)

  return(category_colors)
}

#' Load reference line colors
#'
#' @return Named list with `random_chance` and `ground_truth` colors
#' @export
load_reference_colors <- function() {
  project_root <- find_project_root()
  combos_path <- file.path(project_root, "configs/VISUALIZATION/combos.yaml")

  if (!file.exists(combos_path)) {
    stop("combos.yaml not found! This is REQUIRED for color definitions.")
  }

  combos_config <- yaml::read_yaml(combos_path)
  color_defs <- combos_config$color_definitions

  list(
    random_chance = color_defs[["--color-negative"]],
    ground_truth = color_defs[["--color-ground-truth"]]
  )
}

# ==============================================================================
# NULL COALESCING OPERATOR
# ==============================================================================

# Define if not already available (R < 4.1)
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

# ==============================================================================
# DISPLAY NAME LOADING FROM YAML (Single Source of Truth)
# ==============================================================================

# Cache for display names
.display_names_cache <- new.env(parent = emptyenv())

#' Load display names from YAML config
#'
#' Loads the single source of truth for all display names.
#' NEVER use hardcoded case_when for display names - always use this!
#'
#' @return Named list with sections: outlier_methods, imputation_methods, classifiers, categories
#' @export
load_display_names <- function() {
  cache_key <- "display_names"
  if (exists(cache_key, envir = .display_names_cache)) {
    return(get(cache_key, envir = .display_names_cache))
  }

  project_root <- find_project_root()
  yaml_path <- file.path(project_root, "configs/mlflow_registry/display_names.yaml")

  if (!file.exists(yaml_path)) {
    stop("CRITICAL: display_names.yaml not found at: ", yaml_path)
  }

  display_names <- yaml::read_yaml(yaml_path)
  assign(cache_key, display_names, envir = .display_names_cache)
  return(display_names)
}

#' Get display name for an outlier method
#'
#' @param method Character: raw method name from data
#' @return Character: publication-friendly display name
#' @export
get_outlier_display_name <- function(method) {
  names_list <- load_display_names()$outlier_methods
  if (method %in% names(names_list)) {
    return(names_list[[method]])
  }
  warning("No display name for outlier method: ", method, " - using raw name")
  return(method)
}

#' Get display name for an imputation method
#'
#' @param method Character: raw method name from data
#' @return Character: publication-friendly display name
#' @export
get_imputation_display_name <- function(method) {
  names_list <- load_display_names()$imputation_methods
  if (method %in% names(names_list)) {
    return(names_list[[method]])
  }
  warning("No display name for imputation method: ", method, " - using raw name")
  return(method)
}

#' Apply display names to a data frame (vectorized)
#'
#' Adds *_display_name columns for outlier_method and imputation_method
#'
#' @param df Data frame with outlier_method and/or imputation_method columns
#' @return Data frame with added display name columns
#' @export
apply_display_names <- function(df) {
  names_config <- load_display_names()

  if ("outlier_method" %in% names(df)) {
    df$outlier_display_name <- vapply(df$outlier_method, function(m) {
      entry <- names_config$outlier_methods[[m]]
      if (is.list(entry) && !is.null(entry$display_name)) {
        entry$display_name
      } else if (is.character(entry)) {
        entry
      } else {
        m
      }
    }, character(1))
  }

  if ("imputation_method" %in% names(df)) {
    df$imputation_display_name <- vapply(df$imputation_method, function(m) {
      entry <- names_config$imputation_methods[[m]]
      if (is.list(entry) && !is.null(entry$display_name)) {
        entry$display_name
      } else if (is.character(entry)) {
        entry
      } else {
        m
      }
    }, character(1))
  }

  return(df)
}

# ==============================================================================
# THEME LOADING
# ==============================================================================

#' Ensure theme functions are loaded
#'
#' Lazily loads theme_foundation_plr.R and color_palettes.R if not already available.
#'
#' @export
ensure_theme_loaded <- function() {
  if (!exists("theme_forest", mode = "function", envir = globalenv())) {
    project_root <- find_project_root()
    source(file.path(project_root, "src/r/theme_foundation_plr.R"), local = FALSE)
    source(file.path(project_root, "src/r/color_palettes.R"), local = FALSE)
  }
}
