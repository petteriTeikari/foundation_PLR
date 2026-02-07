# Figure: Individual Subject PLR Traces
# ======================================
# Shows raw and processed PLR signals with light stimulus visualization.
#
# Configuration-driven: All settings loaded from figure_layouts.yaml
# Light protocol: Blue (469nm) first, then Red (640nm)
#
# Usage:
#   Rscript r/figures/fig_subject_traces.R --class control
#   Rscript r/figures/fig_subject_traces.R --class glaucoma
#
# Created: 2026-01-31
# Author: Foundation PLR Team

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(yaml)
  library(jsonlite)
})

# ==============================================================================
# PROJECT ROOT FINDER
# ==============================================================================

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

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load color definitions from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# ==============================================================================
# COMMAND LINE ARGUMENT PARSING
# ==============================================================================

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # Default: generate both
  class_filter <- "both"

  for (i in seq_along(args)) {
    if (args[i] == "--class" && i < length(args)) {
      class_filter <- args[i + 1]
    }
  }

  if (!class_filter %in% c("control", "glaucoma", "both")) {
    stop("Invalid --class argument. Must be: control, glaucoma, or both")
  }

  return(list(class_filter = class_filter))
}

# ==============================================================================
# CONFIGURATION LOADER
# ==============================================================================

#' Load figure configuration from YAML
#'
#' @param figure_id The figure identifier (e.g., "fig_subject_traces_control")
#' @return List with figure configuration
load_figure_config <- function(figure_id) {
  config_path <- file.path(PROJECT_ROOT, "configs/VISUALIZATION/figure_layouts.yaml")

  if (!file.exists(config_path)) {
    stop("Configuration file not found: ", config_path)
  }

  config <- yaml::read_yaml(config_path)

  if (!figure_id %in% names(config$figures)) {
    stop("Figure not found in config: ", figure_id)
  }

  config$figures[[figure_id]]
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

#' Load subject traces from JSON
#'
#' Returns NULL with warning if data not available (graceful degradation).
#'
#' @param data_source Filename in data/r_data/
#' @return Data frame with all subject traces, or NULL if unavailable
load_subject_traces <- function(data_source = "subject_traces.json") {
  json_path <- file.path(PROJECT_ROOT, "data", "r_data", data_source)

  if (!file.exists(json_path)) {
    warning(
      "\n",
      "========================================\n",
      "SUBJECT TRACES DATA NOT AVAILABLE\n",
      "========================================\n",
      "File not found: ", json_path, "\n\n",
      "This data requires access to the SERI PLR database.\n",
      "The subject trace figures cannot be generated without it.\n\n",
      "If you have database access, run:\n",
      "  uv run python scripts/export_subject_traces_for_r.py\n\n",
      "This is expected in public/CI environments.\n",
      "========================================"
    )
    return(NULL)
  }

  data <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  # Helper function to safely unlist while preserving NULLs as NA
  safe_unlist <- function(x) {
    sapply(x, function(v) if (is.null(v)) NA_real_ else as.numeric(v))
  }

  # Convert subjects list to tidy data frame
  subjects_df <- bind_rows(lapply(data$subjects, function(s) {
    n <- length(s$time)
    data.frame(
      subject_id = rep(s$subject_id, n),
      class_label = rep(s$class_label, n),
      outlier_pct = rep(s$outlier_pct, n),
      note = rep(s$note, n),
      outlier_level = rep(s$outlier_level, n),
      time = safe_unlist(s$time),
      pupil_orig = safe_unlist(s$pupil_orig),  # Original signal (gray)
      pupil_gt = safe_unlist(s$pupil_gt),       # Ground truth (black)
      outlier_mask = as.integer(safe_unlist(s$outlier_mask)),
      blue_stimulus = safe_unlist(s$blue_stimulus),
      red_stimulus = safe_unlist(s$red_stimulus),
      stringsAsFactors = FALSE
    )
  }))

  # Add light protocol metadata
  attr(subjects_df, "light_protocol") <- data$metadata$light_protocol
  attr(subjects_df, "metadata") <- data$metadata

  message("Loaded ", length(unique(subjects_df$subject_id)), " subjects from ", json_path)

  subjects_df
}

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

#' Create single subject trace panel
#'
#' @param df Data for single subject
#' @param cfg Figure configuration
#' @param panel_label Panel label (A, B, C, etc.)
#' @param global_y_range Global y-axis range for consistent comparison
#' @return ggplot object
create_trace_panel <- function(df, cfg, panel_label = NULL, global_y_range = NULL) {

  # Get subject info
  subject_id <- unique(df$subject_id)
  class_label <- unique(df$class_label)
  outlier_pct <- unique(df$outlier_pct)
  note <- unique(df$note)

  # Build title
  title_text <- sprintf("%s (%s) - %.1f%% outliers",
                        subject_id,
                        tools::toTitleCase(class_label),
                        outlier_pct)

  # Light protocol timing from config
  light <- cfg$light_protocol
  blue_onset <- light$blue_onset
  blue_offset <- light$blue_offset
  red_onset <- light$red_onset
  red_offset <- light$red_offset

  # Get colors from color_defs
  # Color scheme:
  #   - pupil_orig (original with artifacts) = MID-GRAY
  #   - pupil_gt (denoised ground truth) = BLACK
  #   - outliers = ORANGE/RED for visibility
  blue_color <- color_defs[["--color-primary"]]     # Blue stimulus
  red_color <- color_defs[["--color-negative"]]     # Red stimulus
  orig_color <- color_defs[["--color-text-muted"]]  # Mid-gray for original signal
  gt_color <- color_defs[["--color-text-primary"]]  # Black for ground truth
  outlier_color <- color_defs[["--color-traditional"]]  # Orange for outliers

  # Display settings
  disp <- cfg$display
  alpha_orig <- disp$alpha_raw %||% 0.5
  alpha_gt <- disp$alpha_processed %||% 1.0
  stimulus_alpha <- light$alpha %||% 0.15

  # Y-axis range: use global range for consistent comparison
  y_range <- global_y_range

  # Start plot
  p <- ggplot(df, aes(x = time))

  # Add light stimulus regions as filled rectangles
  if (light$show_stimulus && !is.null(y_range)) {
    y_min <- y_range[1]
    y_max <- y_range[2]

    # Blue stimulus region
    p <- p + annotate(
      "rect",
      xmin = blue_onset, xmax = blue_offset,
      ymin = y_min, ymax = y_max,
      fill = blue_color,
      alpha = stimulus_alpha
    )

    # Red stimulus region
    p <- p + annotate(
      "rect",
      xmin = red_onset, xmax = red_offset,
      ymin = y_min, ymax = y_max,
      fill = red_color,
      alpha = stimulus_alpha
    )

    # Add dashed vertical lines at onset/offset
    p <- p +
      geom_vline(xintercept = blue_onset, linetype = "dashed",
                 color = blue_color, linewidth = 0.5, alpha = 0.7) +
      geom_vline(xintercept = blue_offset, linetype = "dashed",
                 color = blue_color, linewidth = 0.5, alpha = 0.7) +
      geom_vline(xintercept = red_onset, linetype = "dashed",
                 color = red_color, linewidth = 0.5, alpha = 0.7) +
      geom_vline(xintercept = red_offset, linetype = "dashed",
                 color = red_color, linewidth = 0.5, alpha = 0.7)
  }

  # Add ORIGINAL signal (mid-gray, with artifacts)
  if (disp$show_raw) {
    p <- p + geom_line(
      aes(y = pupil_orig),
      color = orig_color,
      alpha = alpha_orig,
      linewidth = 0.4,
      na.rm = TRUE
    )
  }

  # Add GROUND TRUTH signal (black, denoised)
  if (disp$show_processed) {
    p <- p + geom_line(
      aes(y = pupil_gt),
      color = gt_color,
      alpha = alpha_gt,
      linewidth = 0.7,
      na.rm = TRUE
    )
  }

  # Highlight outlier points on the ORIGINAL signal
  # Clamp outlier y-values to visible range (outliers beyond range shown at boundary)
  if (disp$show_outliers) {
    outlier_df <- df[df$outlier_mask == 1, ]
    if (nrow(outlier_df) > 0 && !is.null(y_range)) {
      # Clamp y-values to visible range
      outlier_df$pupil_clamped <- pmax(pmin(outlier_df$pupil_orig, y_range[2]), y_range[1])

      p <- p + geom_point(
        data = outlier_df,
        aes(y = pupil_clamped),
        color = outlier_color,
        size = 1.0,
        alpha = 0.9,
        na.rm = TRUE
      )
    }
  }

  # Apply theme and labels
  p <- p +
    labs(
      title = if (!is.null(panel_label)) paste0(panel_label, "  ", title_text) else title_text,
      x = cfg$x_axis$label %||% "Time (s)",
      y = cfg$y_axis$label %||% "Pupil Size (normalized)"
    ) +
    theme_foundation_plr() +
    theme(
      plot.title = element_text(size = 10, face = "bold", hjust = 0),
      axis.text = element_text(size = 8),
      axis.title = element_text(size = 9),
      panel.grid.minor = element_blank()
    )

  # Fix y-axis for consistent comparison
  if (!is.null(y_range)) {
    p <- p + coord_cartesian(ylim = y_range)
  }

  p
}

#' Create multi-panel subject traces figure
#'
#' @param data Data frame with all subjects
#' @param cfg Figure configuration
#' @param class_filter Which class to show ("control" or "glaucoma")
#' @param global_y_range Global y-axis range [y_min, y_max] for consistent comparison
#' @return Combined ggplot object
create_subject_traces_figure <- function(data, cfg, class_filter, global_y_range) {

  # Filter by class
  data <- data[data$class_label == class_filter, ]

  # Get unique subjects in order (high outlier first, then avg, then low)
  subject_order <- data %>%
    select(subject_id, outlier_level, outlier_pct) %>%
    distinct() %>%
    mutate(level_order = case_when(
      outlier_level == "high_outlier" ~ 1,
      outlier_level == "average_outlier" ~ 2,
      outlier_level == "low_outlier" ~ 3,
      TRUE ~ 4
    )) %>%
    arrange(level_order, desc(outlier_pct)) %>%
    pull(subject_id)

  # Create panel for each subject with GLOBAL y-axis range
  panels <- lapply(seq_along(subject_order), function(i) {
    sid <- subject_order[i]
    df <- data[data$subject_id == sid, ]
    panel_label <- LETTERS[i]
    create_trace_panel(df, cfg, panel_label = panel_label, global_y_range = global_y_range)
  })

  # Combine with patchwork
  n_panels <- length(panels)

  if (n_panels == 6) {
    # 3x2 layout
    combined <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]]) / (panels[[5]] | panels[[6]])
  } else if (n_panels == 4) {
    combined <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]])
  } else {
    combined <- wrap_plots(panels, ncol = 2)
  }

  combined
}

# ==============================================================================
# LEGEND HELPER
# ==============================================================================

#' Create a standalone legend for the figure
#'
#' @param cfg Figure configuration
#' @return ggplot legend as grob
create_legend <- function(cfg) {
  # Get colors
  gt_color <- color_defs[["--color-positive"]]
  raw_color <- color_defs[["--color-text-muted"]]
  outlier_color <- color_defs[["--color-traditional"]]
  blue_color <- color_defs[["--color-primary"]]
  red_color <- color_defs[["--color-negative"]]

  # Create dummy data for legend
  legend_data <- data.frame(
    x = 1:3,
    y = 1:3,
    type = c("Ground Truth", "Raw Signal", "Outliers")
  )

  legend_plot <- ggplot(legend_data, aes(x = x, y = y)) +
    geom_line(aes(color = "Ground Truth"), linewidth = 0.8) +
    geom_line(aes(color = "Raw Signal"), linewidth = 0.5) +
    geom_point(aes(color = "Outliers"), size = 2) +
    scale_color_manual(
      values = c(
        "Ground Truth" = gt_color,
        "Raw Signal" = raw_color,
        "Outliers" = outlier_color
      ),
      name = ""
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal"
    )

  legend_plot
}

# ==============================================================================
# NULL COALESCING OPERATOR
# ==============================================================================

`%||%` <- function(x, y) if (is.null(x)) y else x

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if (sys.nframe() == 0) {
  message("\n========================================")
  message("Subject Traces Figure Generation")
  message("========================================")

  # Parse arguments
  args <- parse_args()
  class_filter <- args$class_filter

  message("Class filter: ", class_filter)

  # Load data (returns NULL if unavailable)
  message("\nLoading subject traces...")
  data <- load_subject_traces("subject_traces.json")

  # Graceful exit if data not available
  if (is.null(data)) {
    message("\n[SKIPPED] Subject traces figures not generated (data unavailable)")
    message("This is expected in public/CI environments without database access.")
    message("\n========================================")
    quit(save = "no", status = 0)  # Exit cleanly, not with error
  }

  # Get light protocol from data
  protocol <- attr(data, "light_protocol")
  message(sprintf("Light protocol: Blue %.1fs-%.1fs, Red %.1fs-%.1fs",
                  protocol$blue_start, protocol$blue_end,
                  protocol$red_start, protocol$red_end))

  # Get global y-axis range from data (for consistent comparison across all panels)
  metadata <- attr(data, "metadata")
  y_range_info <- metadata$y_axis_range
  global_y_range <- c(y_range_info$y_min, y_range_info$y_max)
  message(sprintf("Global Y-axis range: [%.1f, %.1f]", global_y_range[1], global_y_range[2]))

  # Generate figures
  classes_to_generate <- if (class_filter == "both") {
    c("control", "glaucoma")
  } else {
    class_filter
  }

  for (class_name in classes_to_generate) {
    message("\n--- Generating ", class_name, " figure ---")

    # Load config for this figure
    fig_id <- paste0("fig_subject_traces_", class_name)
    cfg <- load_figure_config(fig_id)

    message("  Config loaded: ", cfg$display_name)

    # Create figure with global y-axis range
    fig <- create_subject_traces_figure(data, cfg, class_name, global_y_range)

    # Output path
    output_dir <- file.path(PROJECT_ROOT, "figures/generated/ggplot2/supplementary")
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }

    # Save using figure system
    save_publication_figure(
      fig,
      cfg$filename,
      output_dir = output_dir,
      width = cfg$dimensions$width,
      height = cfg$dimensions$height
    )

    message("  Saved: ", file.path(output_dir, paste0(cfg$filename, ".png")))
  }

  message("\n========================================")
  message("Subject Traces Figure Complete")
  message("========================================\n")
}
