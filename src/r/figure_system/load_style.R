# Figure Style Loader - SINGLE SOURCE OF TRUTH
# =============================================
# ALL figure scripts MUST source this file and use these functions.
# ZERO hardcoding of aesthetics in individual scripts.
#
# Usage:
#   source("src/r/figure_system/load_style.R")
#   style <- load_figure_style()
#   p + get_panel_label_theme(style)
#
# Created: 2026-01-28

suppressPackageStartupMessages({
  library(yaml)
  library(ggplot2)
})

# Cache for style config
.style_cache <- new.env(parent = emptyenv())

#' Find project root
#' @return Path to project root
find_project_root_style <- function() {
  if (exists("PROJECT_ROOT", envir = globalenv())) {
    return(get("PROJECT_ROOT", envir = globalenv()))
  }

  current <- getwd()
  while (current != dirname(current)) {
    if (file.exists(file.path(current, "pyproject.toml"))) {
      return(current)
    }
    current <- dirname(current)
  }
  stop("Could not find project root (no pyproject.toml found)")
}

#' Load figure style from YAML - SINGLE SOURCE OF TRUTH
#'
#' All figure scripts MUST call this before creating plots.
#' Style is cached after first load.
#'
#' @param force_reload Reload from file even if cached
#' @return List with all style parameters
#' @export
load_figure_style <- function(force_reload = FALSE) {
  if (!force_reload && exists("style", envir = .style_cache)) {
    return(get("style", envir = .style_cache))
  }

  project_root <- find_project_root_style()
  style_path <- file.path(project_root, "configs/VISUALIZATION/figure_style.yaml")

  if (!file.exists(style_path)) {
    stop(sprintf(
      "GUARDRAIL VIOLATION: figure_style.yaml not found at: %s\n\nCreate it per docs/planning/figure-production-grade.md",
      style_path
    ))
  }

  config <- yaml::read_yaml(style_path)

  # Validate required keys
  required <- c("panel_labels", "legend", "pipeline_display_names", "metric_display_names")
  missing <- setdiff(required, names(config))
  if (length(missing) > 0) {
    stop("Missing required style config: ", paste(missing, collapse = ", "))
  }

  # Cache and return

  assign("style", config, envir = .style_cache)
  message("[load_style] Figure style loaded from YAML")
  return(config)
}

#' Get pipeline display name from YAML
#'
#' @param pipeline_id Pipeline identifier (e.g., "ground_truth", "best_ensemble")
#' @param style Optional pre-loaded style config
#' @return Display name string
#' @export
get_pipeline_name <- function(pipeline_id, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  name <- style$pipeline_display_names[[pipeline_id]]
  if (is.null(name)) {
    available <- paste(names(style$pipeline_display_names), collapse = ", ")
    stop(sprintf(
      "Unknown pipeline: '%s'\n\nAvailable: %s",
      pipeline_id, available
    ))
  }
  return(name)
}

#' Get metric display name from YAML
#'
#' @param metric_id Metric identifier (e.g., "auroc", "scaled_brier")
#' @param style Optional pre-loaded style config
#' @return Display name string
#' @export
get_metric_name <- function(metric_id, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  name <- style$metric_display_names[[metric_id]]
  if (is.null(name)) {
    available <- paste(names(style$metric_display_names), collapse = ", ")
    stop(sprintf(
      "Unknown metric: '%s'\n\nAvailable: %s",
      metric_id, available
    ))
  }
  return(name)
}

#' Get category display name from YAML
#'
#' @param category_id Category identifier (e.g., "discrimination", "overall_performance")
#' @param style Optional pre-loaded style config
#' @return Display name string
#' @export
get_category_name <- function(category_id, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  name <- style$category_display_names[[category_id]]
  if (is.null(name)) {
    available <- paste(names(style$category_display_names), collapse = ", ")
    stop(sprintf(
      "Unknown category: '%s'\n\nAvailable: %s",
      category_id, available
    ))
  }
  return(name)
}

#' Get panel interpretation text from YAML
#'
#' @param metric_id Metric identifier (e.g., "auroc", "aurc")
#' @param style Optional pre-loaded style config
#' @return Interpretation string (e.g., "Higher = Better")
#' @export
get_panel_interpretation <- function(metric_id, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  text <- style$panel_interpretation[[metric_id]]
  if (is.null(text)) return("")
  return(text)
}

#' Get standardized panel label theme for patchwork
#'
#' Returns a theme() object with panel label styling from YAML.
#' Use in plot_annotation() calls.
#'
#' @param style Optional pre-loaded style config
#' @return ggplot2 theme object
#' @export
get_panel_label_theme <- function(style = NULL) {
  if (is.null(style)) style <- load_figure_style()

  panel <- style$panel_labels

  theme(
    plot.tag = element_text(
      family = panel$font_family,
      face = if (panel$font_weight == "bold") "bold" else "plain",
      size = panel$font_size,
      color = panel$color
    )
  )
}

#' Get tag_levels for patchwork
#'
#' Returns the tag_levels parameter value from YAML.
#' Use in plot_annotation() calls.
#'
#' @param style Optional pre-loaded style config
#' @return String for tag_levels (e.g., "A")
#' @export
get_tag_levels <- function(style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  return(style$panel_labels$tag_levels)
}

#' Build legend label with metric value
#'
#' Creates standardized legend labels like "Ground Truth (AUROC: 0.911)"
#'
#' @param pipeline_id Pipeline identifier
#' @param metric_id Metric identifier
#' @param value Metric value
#' @param style Optional pre-loaded style config
#' @return Formatted string
#' @export
build_legend_label <- function(pipeline_id, metric_id, value, style = NULL) {
  if (is.null(style)) style <- load_figure_style()
  pipeline_name <- get_pipeline_name(pipeline_id, style)
  metric_name <- get_metric_name(metric_id, style)
  sprintf("%s (%s: %.3f)", pipeline_name, metric_name, value)
}

message("[load_style] Style loader module loaded successfully")
