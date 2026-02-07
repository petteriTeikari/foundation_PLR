# Compose Figures - Multi-panel Figure Composition
# =================================================
# Uses patchwork to combine multiple ggplot2 objects into publication-ready
# multi-panel figures.
#
# Key features:
# 1. Layout-driven composition (2x1, 1x2, 2x2, etc.)
# 2. Automatic panel labeling (A, B, C or a, b, c or 1, 2, 3)
# 3. Config-driven composition from YAML
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(patchwork)
  library(yaml)
})

# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

# Null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Project root finder (local definition for robustness)
.find_project_root <- function() {
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

# Try to load shared common.R if available
tryCatch({
  common_path <- file.path(.find_project_root(), "src/r/figure_system/common.R")
  if (file.exists(common_path) && !exists("find_project_root", mode = "function")) {
    source(common_path, local = FALSE)
  }
}, error = function(e) NULL)

# ==============================================================================
# LAYOUT DEFINITIONS
# ==============================================================================

# Internal layout specifications matching figure_layouts.yaml
.LAYOUTS <- list(
  "1x1" = list(nrow = 1, ncol = 1, n_panels = 1),
  "2x1" = list(nrow = 2, ncol = 1, n_panels = 2),
  "1x2" = list(nrow = 1, ncol = 2, n_panels = 2),
  "2x2" = list(nrow = 2, ncol = 2, n_panels = 4),
  "3x1" = list(nrow = 3, ncol = 1, n_panels = 3),
  "1x3" = list(nrow = 1, ncol = 3, n_panels = 3)
)

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

#' Validate plots input
#' @param plots list of ggplot objects
#' @param fn_name name of calling function for error messages
.validate_plots <- function(plots, fn_name) {
  if (!is.list(plots)) {
    stop(paste0(fn_name, ": plots must be a list"))
  }
  if (length(plots) == 0) {
    stop(paste0(fn_name, ": plots must contain at least one plot"))
  }

  # Check each element is a ggplot
  non_gg <- sapply(plots, function(p) {
    !inherits(p, "gg") && !inherits(p, "ggplot")
  })
  if (any(non_gg)) {
    bad_indices <- which(non_gg)
    stop(paste0(
      fn_name, ": elements at positions ",
      paste(bad_indices, collapse = ", "),
      " are not ggplot objects"
    ))
  }
}

#' Validate layout string
#' @param layout string like "2x1"
#' @param fn_name name of calling function for error messages
.validate_layout <- function(layout, fn_name) {
  if (!layout %in% names(.LAYOUTS)) {
    valid <- paste(names(.LAYOUTS), collapse = ", ")
    stop(paste0(fn_name, ": Invalid layout '", layout, "'. Valid layouts: ", valid))
  }
}

# ==============================================================================
# PANEL TITLE HELPER
# ==============================================================================

#' Add a panel label and title to a ggplot
#'
#' Adds "A. Title" style header to a plot for multi-panel figures.
#'
#' @param plot ggplot object
#' @param label Panel label (e.g., "A", "B")
#' @param title Panel title (e.g., "Outlier Detection Method")
#' @param label_family Font family for label (default: "Neue Haas Grotesk Display Pro")
#' @param title_family Font family for title (default: same as label_family)
#' @param label_size Font size for label in points (default: 14)
#' @param title_size Font size for title in points (default: 12)
#'
#' @return ggplot object with panel header added
#' @export
add_panel_title <- function(plot,
                             label,
                             title = NULL,
                             label_family = "Neue Haas Grotesk Display Pro",
                             title_family = NULL,
                             label_size = 14,
                             title_size = 12) {

  if (is.null(title_family)) title_family <- label_family

  # Build the title text: "A   Title" (label bold, title regular weight)
  # Using spaces for separation since we can't mix fonts in one element easily
  if (!is.null(title) && nchar(title) > 0) {
    full_title <- paste0(label, "    ", title)
  } else {
    full_title <- label
  }

  # Add as ggtitle with custom styling
  plot + labs(title = full_title) +
    theme(
      plot.title = element_text(
        family = label_family,
        face = "bold",
        size = label_size,
        hjust = 0,
        margin = margin(b = 8)
      ),
      plot.title.position = "plot"
    )
}

# ==============================================================================
# MAIN COMPOSITION FUNCTION
# ==============================================================================

#' Compose multiple ggplot objects into a single figure
#'
#' @param plots list of ggplot objects
#' @param layout character string specifying layout (e.g., "2x1", "1x2", "2x2")
#' @param tag_levels character for panel labels: "A", "a", "1", "i", or NULL for none
#' @param widths relative widths for columns (only for layouts with >1 column)
#' @param heights relative heights for rows (only for layouts with >1 row)
#' @param panel_titles optional character vector of titles for each panel
#' @param tag_font Font family for panel tags (default: "Neue Haas Grotesk Display Pro")
#' @param tag_size Font size for panel tags in points (default: 14)
#'
#' @return patchwork object
#' @export
compose_figures <- function(plots,
                             layout = "2x1",
                             tag_levels = "A",
                             widths = NULL,
                             heights = NULL,
                             panel_titles = NULL,
                             tag_font = "Neue Haas Grotesk Display Pro",
                             tag_size = 14) {
  # Validate inputs
  .validate_plots(plots, "compose_figures")
  .validate_layout(layout, "compose_figures")

  # Get layout spec
  layout_spec <- .LAYOUTS[[layout]]

  # Check plot count matches layout
  expected <- layout_spec$n_panels
  actual <- length(plots)
  if (actual != expected) {
    stop(paste0(
      "compose_figures: Layout '", layout, "' requires ", expected,
      " plots, but ", actual, " provided"
    ))
  }

  # Generate panel labels based on tag_levels
  labels <- NULL
  if (!is.null(tag_levels)) {
    labels <- switch(tag_levels,
      "A" = LETTERS[1:length(plots)],
      "a" = letters[1:length(plots)],
      "1" = as.character(1:length(plots)),
      "i" = tolower(as.roman(1:length(plots))),
      LETTERS[1:length(plots)]  # default
    )
  }

  # If panel_titles provided, add them to each plot using add_panel_title
  if (!is.null(panel_titles) && !is.null(labels)) {
    if (length(panel_titles) != length(plots)) {
      stop(paste0(
        "compose_figures: panel_titles length (", length(panel_titles),
        ") must match number of plots (", length(plots), ")"
      ))
    }
    for (i in seq_along(plots)) {
      plots[[i]] <- add_panel_title(
        plots[[i]],
        label = labels[i],
        title = panel_titles[i],
        label_family = tag_font,
        label_size = tag_size
      )
    }
    # Don't use patchwork's tag_levels since we added them manually
    tag_levels <- NULL
  }

  # Special case: single plot
  if (layout == "1x1") {
    composed <- plots[[1]]
    if (!is.null(tag_levels)) {
      composed <- composed + plot_annotation(
        tag_levels = tag_levels,
        theme = theme(plot.tag = element_text(
          family = tag_font,
          face = "bold",
          size = tag_size
        ))
      )
    }
    class(composed) <- c("patchwork", class(composed))
    return(composed)
  }

  # Compose using patchwork
  # Start with first plot
  composed <- plots[[1]]

  # Add remaining plots based on layout
  if (layout %in% c("2x1", "3x1")) {
    # Vertical stacking
    for (i in 2:length(plots)) {
      composed <- composed / plots[[i]]
    }
  } else if (layout %in% c("1x2", "1x3")) {
    # Horizontal arrangement
    for (i in 2:length(plots)) {
      composed <- composed | plots[[i]]
    }
  } else if (layout == "2x2") {
    # 2x2 grid: (p1 | p2) / (p3 | p4)
    top_row <- plots[[1]] | plots[[2]]
    bottom_row <- plots[[3]] | plots[[4]]
    composed <- top_row / bottom_row
  }

  # Apply tag levels with custom font (only if we didn't use panel_titles)
  if (!is.null(tag_levels)) {
    composed <- composed + plot_annotation(
      tag_levels = tag_levels,
      theme = theme(plot.tag = element_text(
        family = tag_font,
        face = "bold",
        size = tag_size
      ))
    )
  }

  # Apply custom widths/heights if provided
  if (!is.null(widths) || !is.null(heights)) {
    composed <- composed + plot_layout(
      nrow = layout_spec$nrow,
      ncol = layout_spec$ncol,
      widths = widths,
      heights = heights
    )
  }

  return(composed)
}

# ==============================================================================
# CONFIG-DRIVEN COMPOSITION
# ==============================================================================

#' Compose figures from YAML configuration
#'
#' This function reads figure specifications from configs/VISUALIZATION/figure_layouts.yaml
#' and creates the specified composite figure.
#'
#' @param figure_name name of the figure in the YAML (e.g., "fig_forest_combined")
#' @param infographic logical; override infographic mode for all panels
#' @param config_path optional path to config file (default: auto-detect)
#' @param data_list optional named list of data frames for each panel's data_source
#'
#' @return patchwork object
#' @export
compose_from_config <- function(figure_name,
                                 infographic = FALSE,
                                 config_path = NULL,
                                 data_list = NULL) {
  # Load config
  if (is.null(config_path)) {
    project_root <- .find_project_root()
    config_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")
  }

  if (!file.exists(config_path)) {
    stop(paste0("compose_from_config: Config file not found: ", config_path))
  }

  config <- yaml::read_yaml(config_path)

  # Find the figure
  if (!figure_name %in% names(config$figures)) {
    available <- paste(names(config$figures), collapse = ", ")
    stop(paste0(
      "compose_from_config: Figure '", figure_name,
      "' not found in config. Available: ", available
    ))
  }

  fig_spec <- config$figures[[figure_name]]

  # Load figure factory if needed
  project_root <- .find_project_root()
  source(file.path(project_root, "src/r/figure_system/figure_factory.R"))

  # Create each panel
  plots <- list()
  for (panel in fig_spec$panels) {
    fn_name <- panel$figure_function
    data_source <- panel$data_source

    # Get the function
    if (!exists(fn_name, mode = "function")) {
      stop(paste0("compose_from_config: Function '", fn_name, "' not found"))
    }
    fn <- get(fn_name)

    # Get data
    if (!is.null(data_list) && data_source %in% names(data_list)) {
      data <- data_list[[data_source]]
    } else {
      stop(paste0(
        "compose_from_config: Data source '", data_source,
        "' not provided in data_list"
      ))
    }

    # Determine infographic mode (parameter overrides config)
    panel_infographic <- if (!is.null(panel$infographic)) panel$infographic else infographic
    panel_show_legend <- if (!is.null(panel$show_legend)) panel$show_legend else TRUE

    # Create the panel
    p <- fn(data, infographic = panel_infographic, show_legend = panel_show_legend)
    plots <- c(plots, list(p))
  }

  # Get layout and tag_levels from config
  layout <- fig_spec$layout
  tag_levels <- fig_spec$tag_levels

  # Compose
  composed <- compose_figures(plots, layout = layout, tag_levels = tag_levels)

  return(composed)
}
