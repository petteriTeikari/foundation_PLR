# Save Figure - Publication-Ready Figure Export
# ==============================================
# Saves ggplot2/patchwork objects in multiple formats with consistent settings.
#
# Key features:
# 1. Multi-format output (PDF, PNG, TIFF, EPS)
# 2. Publication-quality settings (DPI, fonts, dimensions)
# 3. Consistent file naming
#
# Created: 2026-01-27
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(ggplot2)
  library(yaml)
})

# Null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Project root finder
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

# ==============================================================================
# LOAD CONFIG SETTINGS
# ==============================================================================

#' Load figure dimensions from figure_registry.yaml
#' @param figure_name name of figure (e.g., "fig_forest_combined")
#' @return list with width, height or NULL if not found
.load_figure_dimensions <- function(figure_name) {
  project_root <- .find_project_root()
  registry_path <- file.path(project_root, "configs/VISUALIZATION/figure_registry.yaml")

  if (!file.exists(registry_path)) {
    return(NULL)
  }

  registry <- yaml::read_yaml(registry_path)

  # Search in r_figures (R-generated figures with explicit dimensions)
  if (!is.null(registry$r_figures[[figure_name]])) {
    styling <- registry$r_figures[[figure_name]]$styling
    if (!is.null(styling)) {
      # R figures have explicit width/height, not semantic
      if (!is.null(styling$width) && !is.null(styling$height)) {
        return(list(width = as.numeric(styling$width), height = as.numeric(styling$height)))
      }
      return(.resolve_semantic_dimensions(styling))
    }
  }

  # Search in main_figures
  if (!is.null(registry$main_figures[[figure_name]])) {
    styling <- registry$main_figures[[figure_name]]$styling
    if (!is.null(styling)) {
      return(.resolve_semantic_dimensions(styling))
    }
  }

  # Search in supplementary_figures
  if (!is.null(registry$supplementary_figures[[figure_name]])) {
    styling <- registry$supplementary_figures[[figure_name]]$styling
    if (!is.null(styling)) {
      return(.resolve_semantic_dimensions(styling))
    }
  }

  return(NULL)
}

#' Resolve semantic dimension names to actual values
#' @param styling list with width (e.g., "single", "double") and aspect ratio
#' @return list with numeric width and height
.resolve_semantic_dimensions <- function(styling) {
  # Semantic width definitions (journal standard column widths in inches)
  width_map <- list(
    single = 3.5,    # Single column
    double = 7.0,    # Double column
    full = 7.5       # Full page width
  )

  width_str <- styling$width %||% "double"
  aspect <- styling$aspect %||% 0.75

  # Resolve semantic width
  if (is.character(width_str)) {
    width <- width_map[[width_str]]
    if (is.null(width)) {
      # Try to parse as numeric
      width <- as.numeric(width_str)
      if (is.na(width)) width <- 7.0
    }
  } else {
    width <- as.numeric(width_str)
  }

  # Calculate height from aspect ratio
  height <- width * aspect

  return(list(width = width, height = height))
}

#' Load global output settings from YAML config
#' @return list with formats, dpi, directory, figure_categories
.load_output_settings <- function() {
  project_root <- .find_project_root()
  config_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")

  if (file.exists(config_path)) {
    config <- yaml::read_yaml(config_path)
    return(list(
      formats = config$output_settings$formats %||% c("png"),
      dpi = config$output_settings$dpi %||% 300,
      directory = config$output_settings$directory %||% "figures/generated/ggplot2/",
      figure_categories = config$figure_categories %||% list()
    ))
  }

  # Fallback defaults
  list(
    formats = c("png"),
    dpi = 300,
    directory = "figures/generated/ggplot2/",
    figure_categories = list()
  )
}

#' Get output directory for a specific figure based on category
#' @param filename base filename (without extension)
#' @param config_settings loaded config settings
#' @return output directory path
.get_figure_output_dir <- function(filename, config_settings) {
  project_root <- .find_project_root()
  categories <- config_settings$figure_categories

  # Check each category for the figure
  for (cat_name in names(categories)) {
    cat_info <- categories[[cat_name]]
    if (filename %in% cat_info$figures) {
      # Found the category - use its output_dir
      return(file.path(project_root, cat_info$output_dir))
    }
  }

  # Not found in any category - use default directory
  return(file.path(project_root, config_settings$directory))
}

# Cache for output settings
.output_settings_cache <- NULL

#' Get cached output settings
.get_output_settings <- function() {
  if (is.null(.output_settings_cache)) {
    .output_settings_cache <<- .load_output_settings()
  }
  .output_settings_cache
}

# ==============================================================================
# DEFAULT SETTINGS (legacy, now loaded from YAML)
# ==============================================================================

.PUBLICATION_DEFAULTS <- list(
  dpi = list(
    screen = 150,
    print = 300,
    publication = 600
  ),
  dimensions = list(
    single_column = 3.5,
    double_column = 7.0,
    full_page_height = 9.0
  )
)

# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

#' Validate numeric dimension
#' @param value numeric value to validate
#' @param name parameter name for error messages
#' @param fn_name function name for error messages
.validate_positive <- function(value, name, fn_name) {
  if (!is.numeric(value) || length(value) != 1 || value <= 0) {
    stop(paste0(fn_name, ": ", name, " must be a positive number"))
  }
}

#' Validate output directory
#' @param output_dir directory path
#' @param fn_name function name for error messages
.validate_output_dir <- function(output_dir, fn_name) {
  if (!dir.exists(output_dir)) {
    # Try to create it
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    if (!dir.exists(output_dir)) {
      stop(paste0(fn_name, ": Cannot create output directory: ", output_dir))
    }
  }
}

# ==============================================================================
# MAIN SAVE FUNCTION
# ==============================================================================

#' Save a publication-quality figure in multiple formats
#'
#' Formats are loaded from configs/VISUALIZATION/figure_layouts.yaml by default.
#' The global setting `output_settings.formats` controls which formats are generated.
#'
#' IMPORTANT: Dimensions should NOT be hardcoded in script calls!
#' This function will automatically look up dimensions from figure_registry.yaml
#' based on the filename. Only pass width/height for ad-hoc or debugging purposes.
#'
#' @param plot ggplot or patchwork object to save
#' @param filename base filename without extension (e.g., "fig_forest_combined")
#' @param output_dir directory to save files (default: from YAML config)
#' @param width figure width in inches (default: NULL = load from figure_registry.yaml)
#' @param height figure height in inches (default: NULL = load from figure_registry.yaml)
#' @param dpi resolution for raster formats (default: from YAML config, typically 300)
#' @param formats character vector of formats (default: NULL = load from YAML config)
#' @param device_pdf PDF device to use: "cairo_pdf" or "pdf" (default: "cairo_pdf")
#' @param bg background color (default: "white")
#'
#' @return invisible list of saved file paths
#' @export
save_publication_figure <- function(plot,
                                     filename,
                                     output_dir = NULL,
                                     width = NULL,
                                     height = NULL,
                                     dpi = NULL,
                                     formats = NULL,
                                     device_pdf = "cairo_pdf",
                                     bg = "white") {
  # Load defaults from YAML config
  config_settings <- .get_output_settings()

  # Try to load dimensions from figure_registry.yaml
  if (is.null(width) || is.null(height)) {
    registry_dims <- .load_figure_dimensions(filename)
    if (!is.null(registry_dims)) {
      if (is.null(width)) width <- registry_dims$width
      if (is.null(height)) height <- registry_dims$height
      message(sprintf("Loaded dimensions from registry: %s x %s inches", width, height))
    } else {
      # Fallback defaults if not in registry
      if (is.null(width)) width <- 7.0  # double column default
      if (is.null(height)) height <- 5.25  # 0.75 aspect ratio
      message(sprintf("Figure '%s' not in registry, using defaults: %s x %s inches", filename, width, height))
    }
  }

  # Apply config defaults for NULL parameters
  if (is.null(output_dir)) {
    # Route to correct subdirectory based on figure_categories
    output_dir <- .get_figure_output_dir(filename, config_settings)
  }
  if (is.null(dpi)) {
    dpi <- config_settings$dpi
  }
  if (is.null(formats)) {
    formats <- config_settings$formats
  }

  # Validate inputs
  .validate_positive(width, "width", "save_publication_figure")
  .validate_positive(height, "height", "save_publication_figure")
  .validate_positive(dpi, "dpi", "save_publication_figure")
  .validate_output_dir(output_dir, "save_publication_figure")

  # Validate plot
  if (!inherits(plot, c("gg", "ggplot", "patchwork"))) {
    stop("save_publication_figure: plot must be a ggplot or patchwork object")
  }

  # Validate formats
  valid_formats <- c("pdf", "png", "tiff", "eps", "svg", "jpg", "jpeg")
  invalid <- setdiff(tolower(formats), valid_formats)
  if (length(invalid) > 0) {
    warning(paste0(
      "save_publication_figure: Unknown format(s) ignored: ",
      paste(invalid, collapse = ", ")
    ))
    formats <- intersect(tolower(formats), valid_formats)
  }

  if (length(formats) == 0) {
    stop("save_publication_figure: No valid formats specified")
  }

  # Save in each format
  saved_files <- list()

  for (fmt in formats) {
    filepath <- file.path(output_dir, paste0(filename, ".", fmt))

    tryCatch({
      if (fmt == "pdf") {
        # PDF with optional cairo
        if (device_pdf == "cairo_pdf" && capabilities("cairo")) {
          ggsave(
            filepath, plot,
            width = width, height = height,
            device = cairo_pdf,
            bg = bg
          )
        } else {
          ggsave(
            filepath, plot,
            width = width, height = height,
            device = "pdf",
            bg = bg
          )
        }
      } else if (fmt == "png") {
        ggsave(
          filepath, plot,
          width = width, height = height,
          dpi = dpi,
          device = "png",
          bg = bg
        )
      } else if (fmt == "tiff") {
        ggsave(
          filepath, plot,
          width = width, height = height,
          dpi = dpi,
          device = "tiff",
          compression = "lzw",
          bg = bg
        )
      } else if (fmt == "eps") {
        ggsave(
          filepath, plot,
          width = width, height = height,
          device = "eps",
          bg = bg
        )
      } else if (fmt == "svg") {
        ggsave(
          filepath, plot,
          width = width, height = height,
          device = "svg",
          bg = bg
        )
      } else if (fmt %in% c("jpg", "jpeg")) {
        ggsave(
          filepath, plot,
          width = width, height = height,
          dpi = dpi,
          device = "jpeg",
          bg = bg
        )
      }

      message("Saved: ", filepath)
      saved_files[[fmt]] <- filepath

    }, error = function(e) {
      warning(paste0("Failed to save ", fmt, ": ", e$message))
    })
  }

  invisible(saved_files)
}

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

#' Save figure from YAML config specifications
#'
#' Uses the output settings from figure_layouts.yaml for a specific figure.
#'
#' @param plot ggplot or patchwork object
#' @param figure_name name of figure in config (e.g., "fig_forest_combined")
#' @param output_dir optional override for output directory
#' @param dpi optional override for DPI
#'
#' @return invisible list of saved file paths
#' @export
save_from_config <- function(plot,
                              figure_name,
                              output_dir = NULL,
                              dpi = 300) {
  # Load config
  project_root <- .find_project_root()
  config_path <- file.path(project_root, "configs/VISUALIZATION/figure_layouts.yaml")

  if (!file.exists(config_path)) {
    stop("save_from_config: Config file not found")
  }

  config <- yaml::read_yaml(config_path)

  # Find figure spec
  if (!figure_name %in% names(config$figures)) {
    stop(paste0("save_from_config: Figure '", figure_name, "' not found in config"))
  }

  fig_spec <- config$figures[[figure_name]]

  # Get dimensions
  dims <- fig_spec$dimensions
  width <- dims$width
  height <- dims$height

  # Get formats from outputs
  formats <- c()
  if (!is.null(fig_spec$outputs$pdf)) formats <- c(formats, "pdf")
  if (!is.null(fig_spec$outputs$png)) formats <- c(formats, "png")
  if (!is.null(fig_spec$outputs$tiff)) formats <- c(formats, "tiff")
  if (!is.null(fig_spec$outputs$eps)) formats <- c(formats, "eps")

  if (length(formats) == 0) {
    formats <- c("pdf", "png")  # default
  }

  # Get filename from first output
  filename <- sub("\\.[^.]+$", "", basename(fig_spec$outputs$pdf %||% fig_spec$outputs$png))

  # Use output_dir from config or parameter
  if (is.null(output_dir)) {
    output_dir <- config$metadata$output_directory
    if (is.null(output_dir)) {
      output_dir <- "figures/generated/ggplot2"
    }
  }

  # Save
  save_publication_figure(
    plot = plot,
    filename = filename,
    output_dir = output_dir,
    width = width,
    height = height,
    dpi = dpi,
    formats = formats
  )
}
