# Config Loader for Figure System
# ================================
# Centralized config loading with validation.
# ENFORCES: All figure params come from YAML, nothing hardcoded.
#
# Created: 2026-01-27
# Author: Foundation PLR Team
#
# CRITICAL: This module is the GUARDRAIL for YAML-driven figure generation.
# All R figure scripts MUST use these functions to load configuration.

suppressPackageStartupMessages({
  library(yaml)
  library(jsonlite)
})

# Source common utilities (find_project_root, %||%)
if (!exists("find_project_root", mode = "function")) {
  source(file.path(dirname(sys.frame(1)$ofile), "common.R"))
}

# ==============================================================================
# FIGURE CONFIG LOADING
# ==============================================================================

#' Load figure configuration from YAML
#'
#' Loads configuration for a specific figure from figure_layouts.yaml.
#' FAILS LOUDLY if the figure is not defined - this is a GUARDRAIL.
#'
#' @param figure_id Character: e.g., "fig_selective_classification"
#' @return List with validated config including layout, dimensions, panels, etc.
#' @export
load_figure_config <- function(figure_id) {
  project_root <- find_project_root()
  layouts_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")

  if (!file.exists(layouts_path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: figure_layouts.yaml not found at: %s\n",
      layouts_path
    ))
  }

  config <- yaml::read_yaml(layouts_path)

  if (!figure_id %in% names(config$figures)) {
    available <- paste(names(config$figures), collapse = ", ")
    stop(sprintf(
      "GUARDRAIL VIOLATION: Figure '%s' not found in figure_layouts.yaml.\n\nAvailable figures: %s\n\nYou MUST add the figure to YAML before creating the R script.",
      figure_id,
      available
    ))
  }

  fig_config <- config$figures[[figure_id]]
  fig_config$figure_id <- figure_id

  # Inject global settings
  fig_config$output_settings <- config$output_settings
  fig_config$fonts <- config$fonts
  fig_config$tag_styles <- config$tag_styles

  # Validate required fields
  required_fields <- c("layout", "dimensions", "filename")
  missing <- setdiff(required_fields, names(fig_config))
  if (length(missing) > 0) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: Figure '%s' missing required fields: %s\n\nAdd these to figure_layouts.yaml.",
      figure_id, paste(missing, collapse = ", ")
    ))
  }

  message(sprintf("[config_loader] Loaded figure config: %s", figure_id))
  return(fig_config)
}

# ==============================================================================
# COMBO LOADING
# ==============================================================================

#' Load combos from plot_hyperparam_combos.yaml
#'
#' Loads combo definitions from the SINGLE SOURCE OF TRUTH.
#' NEVER hardcode combos in R scripts - always load from YAML.
#'
#' @param combo_source Character: Section name in YAML, e.g., "standard_combos",
#'                     "extended_combos", "shap_figure_combos", or preset name like "main_4"
#' @return List of combo configs with colors, names, pipeline configs
#' @export
load_figure_combos <- function(combo_source) {
  project_root <- find_project_root()
  combos_path <- file.path(project_root, "configs/VISUALIZATION/plot_hyperparam_combos.yaml")

  if (!file.exists(combos_path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: plot_hyperparam_combos.yaml not found at: %s",
      combos_path
    ))
  }

  config <- yaml::read_yaml(combos_path)

  # Check if it's a preset group reference
  if (combo_source %in% names(config$preset_groups)) {
    preset <- config$preset_groups[[combo_source]]
    combo_ids <- preset$combos
    message(sprintf("[config_loader] Using preset group '%s': %s",
                    combo_source, paste(combo_ids, collapse = ", ")))
    return(get_combos_by_ids(combo_ids, config))
  }

  # Check direct section
  if (!combo_source %in% names(config)) {
    available <- paste(names(config), collapse = ", ")
    stop(sprintf(
      "GUARDRAIL VIOLATION: Combo source '%s' not found in plot_hyperparam_combos.yaml.\n\nAvailable sources: %s",
      combo_source,
      available
    ))
  }

  section <- config[[combo_source]]

  # Handle section with nested 'configs' key (e.g., shap_figure_combos)
  if ("configs" %in% names(section)) {
    combos <- section$configs
  } else if (is.list(section) && !is.null(section[[1]]$id)) {
    # Direct list of combos (e.g., standard_combos)
    combos <- section
  } else {
    stop(sprintf(
      "GUARDRAIL VIOLATION: Section '%s' has unexpected structure.",
      combo_source
    ))
  }

  # Validate each combo has required fields
  for (i in seq_along(combos)) {
    combo <- combos[[i]]
    if (is.null(combo$id)) {
      stop(sprintf("GUARDRAIL VIOLATION: Combo %d in '%s' missing 'id' field", i, combo_source))
    }
  }

  message(sprintf("[config_loader] Loaded %d combos from '%s'",
                  length(combos), combo_source))
  return(combos)
}

#' Get combos by IDs from config
#'
#' @param combo_ids Character vector of combo IDs to retrieve
#' @param config Full YAML config (from read_yaml)
#' @return List of combo configs
get_combos_by_ids <- function(combo_ids, config) {
  all_combos <- list()

  # Collect from standard_combos
  for (combo in config$standard_combos) {
    all_combos[[combo$id]] <- combo
  }

  # Collect from extended_combos
  for (combo in config$extended_combos) {
    all_combos[[combo$id]] <- combo
  }

  # Collect from shap_figure_combos if exists
  if ("shap_figure_combos" %in% names(config)) {
    for (combo in config$shap_figure_combos$configs) {
      all_combos[[combo$id]] <- combo
    }
  }

  # Return requested combos in order
  result <- list()
  for (id in combo_ids) {
    if (id %in% names(all_combos)) {
      result[[length(result) + 1]] <- all_combos[[id]]
    } else {
      stop(sprintf("GUARDRAIL VIOLATION: Combo ID '%s' not found in YAML", id))
    }
  }

  return(result)
}

# ==============================================================================
# DATA SOURCE VALIDATION
# ==============================================================================

#' Validate data file exists and has correct schema
#'
#' Loads JSON data with provenance validation. Fails if file doesn't exist
#' or is missing required keys.
#'
#' @param data_source Filename in data/r_data/ or full path
#' @param required_keys Top-level keys that must exist (default: metadata, data)
#' @return Parsed JSON data as list
#' @export
validate_data_source <- function(data_source, required_keys = c("metadata", "data")) {
  project_root <- find_project_root()

  # Handle both filename and full path
  if (grepl("^/|^\\.\\./", data_source)) {
    data_path <- data_source
  } else {
    data_path <- file.path(project_root, "data/r_data", data_source)
  }

  if (!file.exists(data_path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: Data file '%s' not found.\n\nRun the export script first: python scripts/export_XXX_for_r.py\n\nSearched at: %s",
      data_source, data_path
    ))
  }

  data <- jsonlite::fromJSON(data_path, simplifyVector = FALSE)

  missing_keys <- setdiff(required_keys, names(data))
  if (length(missing_keys) > 0) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: Data file '%s' missing required keys: %s\n\nRegenerate with proper schema.",
      data_source, paste(missing_keys, collapse = ", ")
    ))
  }

  # Log provenance for audit trail
  if (!is.null(data$metadata$data_source$db_hash)) {
    message(sprintf("[config_loader] Data provenance: %s (hash: %s)",
                    basename(data_path),
                    substr(data$metadata$data_source$db_hash, 1, 8)))
  } else if (!is.null(data$metadata$generator)) {
    message(sprintf("[config_loader] Data generator: %s", data$metadata$generator))
  }

  return(data)
}

# ==============================================================================
# COLOR RESOLUTION
# ==============================================================================

#' Load color definitions from YAML
#'
#' @return Named list of color definitions (CSS variable style names -> hex)
#' @export
load_color_definitions <- function() {
  project_root <- find_project_root()
  combos_path <- file.path(project_root, "configs/VISUALIZATION/plot_hyperparam_combos.yaml")

  config <- yaml::read_yaml(combos_path)
  return(config$color_definitions)
}

#' Resolve a color_ref to actual hex color
#'
#' GUARDRAIL: Only accepts color_ref (--color-xxx) or hex colors.
#' Warns if raw hex is used outside of color_definitions.
#'
#' @param color_ref Character: e.g., "--color-ground-truth" or "#666666"
#' @param color_definitions Named list of color definitions (optional, loaded if NULL)
#' @return Character: hex color code
#' @export
resolve_color <- function(color_ref, color_definitions = NULL) {
  if (is.null(color_definitions)) {
    color_definitions <- load_color_definitions()
  }

  # If it's a CSS variable reference
  if (startsWith(color_ref, "--")) {
    if (color_ref %in% names(color_definitions)) {
      return(color_definitions[[color_ref]])
    }
    stop(sprintf(
      "GUARDRAIL VIOLATION: Color '%s' not found in color_definitions.\n\nAdd it to plot_hyperparam_combos.yaml color_definitions section.",
      color_ref
    ))
  }

  # If it's already a hex color, allow but warn
  if (grepl("^#[0-9A-Fa-f]{6}$", color_ref)) {
    warning(sprintf(
      "Using raw hex color '%s' - prefer color_ref for consistency",
      color_ref
    ))
    return(color_ref)
  }

  stop(sprintf(
    "GUARDRAIL VIOLATION: Invalid color format '%s'. Must be --color-xxx ref or #RRGGBB hex.",
    color_ref
  ))
}

#' Get colors for a list of combos
#'
#' Resolves color_ref or color_var for each combo.
#'
#' @param combos List of combo configs
#' @return Named character vector: combo name -> hex color
#' @export
get_combo_colors <- function(combos) {
  color_defs <- load_color_definitions()

  colors <- sapply(combos, function(combo) {
    # Try color_ref first (new style), then color_var (old style)
    ref <- combo$color_ref %||% combo$color_var %||% combo$color

    if (is.null(ref)) {
      warning(sprintf("Combo '%s' has no color defined, using muted color", combo$id))
      return(color_defs[["--color-text-muted"]])
    }

    resolve_color(ref, color_defs)
  })

  # Name by display name if available, otherwise id
  names(colors) <- sapply(combos, function(c) c$name %||% c$id)

  return(colors)
}

# ==============================================================================
# FIGURE CATEGORY LOOKUP
# ==============================================================================

#' Get output directory for a figure based on its category
#'
#' @param figure_id Character: figure ID to look up
#' @return Character: output directory path
#' @export
get_figure_output_dir <- function(figure_id) {
  project_root <- find_project_root()
  layouts_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")

  config <- yaml::read_yaml(layouts_path)

  # Search through categories
  for (cat_name in names(config$figure_categories)) {
    cat_info <- config$figure_categories[[cat_name]]
    if (figure_id %in% cat_info$figures) {
      return(cat_info$output_dir)
    }
  }

  # Default to main output directory
  warning(sprintf(
    "Figure '%s' not found in any category - using default output directory",
    figure_id
  ))
  return(config$output_settings$directory)
}

# ==============================================================================
# METRIC VALIDATION
# ==============================================================================

#' Validate that a metric is defined in the registry
#'
#' @param metric_name Character: metric name to validate
#' @return TRUE if valid, stops with error if not
#' @export
validate_metric <- function(metric_name) {
  project_root <- find_project_root()
  metrics_path <- file.path(project_root, "configs/mlflow_registry/metrics/classification.yaml")

  if (!file.exists(metrics_path)) {
    warning("Metrics registry not found, skipping validation")
    return(TRUE)
  }

  metrics <- yaml::read_yaml(metrics_path)

  if (!metric_name %in% names(metrics)) {
    available <- paste(names(metrics), collapse = ", ")
    stop(sprintf(
      "GUARDRAIL VIOLATION: Metric '%s' not in registry.\n\nAvailable: %s",
      metric_name, available
    ))
  }

  return(TRUE)
}

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

#' Load everything needed for a figure in one call
#'
#' @param figure_id Character: figure ID
#' @return List with config, combos (if applicable), color_definitions
#' @export
load_figure_all <- function(figure_id) {
  fig_config <- load_figure_config(figure_id)

  result <- list(
    config = fig_config,
    color_definitions = load_color_definitions()
  )

  # Load combos if combo_source is specified
  if (!is.null(fig_config$combo_source)) {
    result$combos <- load_figure_combos(fig_config$combo_source)
    result$combo_colors <- get_combo_colors(result$combos)
  }

  # Load data if data_source is specified at figure level
  if (!is.null(fig_config$data_source)) {
    result$data <- validate_data_source(fig_config$data_source)
  }

  return(result)
}

message("[config_loader] Config loader module loaded successfully")
