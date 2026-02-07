# Foundation PLR Publication Theme for ggplot2
# Economist-style theme with off-white background
# Created: 2026-01-25
#
# Inspired by: https://altaf-ali.github.io/ggplot_tutorial/challenge.html
# Features:
#   - Off-white (#FBF9F3) background
#   - Bold sans-serif headings (Helvetica Bold as Neue Haas Grotesk substitute)
#   - Minimal grid lines
#   - Colorblind-safe palette

suppressPackageStartupMessages(library(ggplot2))

# ==============================================================================
# GGPLOT2 4.0+ / S7 COMPATIBILITY
# ==============================================================================
# ggplot2 4.0+ uses S7 internally which can conflict with the + operator.
# This helper function provides a workaround using ggplot_add().

#' Add layers to ggplot using ggplot_add (S7-compatible)
#'
#' @param p A ggplot object
#' @param ... Layers to add
#' @return A ggplot object with added layers
#' @export
#' @note Fixed object_name capture using substitute() on full list
gg_add <- function(p, ...) {
  dots <- as.list(substitute(list(...)))[-1]  # Capture expressions properly

layers <- list(...)
  for (i in seq_along(layers)) {
    obj_name <- if (i <= length(dots)) deparse(dots[[i]]) else paste0("layer_", i)
    p <- ggplot_add(layers[[i]], p, object_name = obj_name)
  }
  p
}

# ==============================================================================
# ECONOMIST-STYLE COLOR DEFINITIONS
# ==============================================================================

# Economist off-white background (user preference)
ECONOMIST_BG <- "#FBF9F3"
ECONOMIST_GRID <- "#D4D4D4"
ECONOMIST_TEXT <- "#333333"
ECONOMIST_CAPTION <- "#666666"
ECONOMIST_TITLE <- "#000000"

# Economist-inspired colorblind-safe palette
# Red for emphasis, blues for comparison, grays for context
ECONOMIST_COLORS <- c(
  "#E3120B",  # Economist red (primary)
  "#006BA2",  # Dark blue
  "#3EBCD2",  # Light blue/cyan
  "#379A8B",  # Teal
  "#EBB434",  # Gold/yellow
  "#932834",  # Dark red
  "#999999"   # Gray
)

#' Economist-style theme for Foundation PLR figures
#'
#' @param base_size Base font size (default 11)
#' @param base_family Font family (default "Helvetica" - close to Neue Haas Grotesk)
#' @return A ggplot2 theme object
#' @export
theme_foundation_plr <- function(base_size = 11,
                                  base_family = "Helvetica") {

  # Size hierarchy
  title_size <- base_size + 3
  subtitle_size <- base_size + 1
  axis_title_size <- base_size
  axis_text_size <- base_size - 1
  caption_size <- base_size - 2

  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Plot background - Economist off-white
      plot.background = element_rect(fill = ECONOMIST_BG, colour = NA),
      panel.background = element_rect(fill = ECONOMIST_BG, colour = NA),

      # Grid - minimal horizontal lines only (Economist style)
      panel.grid.major.y = element_line(colour = ECONOMIST_GRID, linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),

      # No panel border (Economist uses axis lines instead)
      panel.border = element_blank(),

      # Axes
      axis.line.x = element_line(colour = ECONOMIST_TEXT, linewidth = 0.5),
      axis.line.y = element_blank(),
      axis.title = element_text(
        size = axis_title_size,
        colour = ECONOMIST_TEXT,
        face = "plain"
      ),
      axis.title.x = element_text(margin = margin(t = 10)),
      axis.title.y = element_text(margin = margin(r = 10), angle = 90),
      axis.text = element_text(size = axis_text_size, colour = ECONOMIST_TEXT),
      axis.ticks.x = element_line(colour = ECONOMIST_TEXT, linewidth = 0.3),
      axis.ticks.y = element_blank(),
      axis.ticks.length = unit(0.15, "cm"),

      # Legend
      legend.background = element_rect(fill = ECONOMIST_BG, colour = NA),
      legend.key = element_rect(fill = ECONOMIST_BG, colour = NA),
      legend.text = element_text(size = axis_text_size, colour = ECONOMIST_TEXT),
      legend.title = element_text(
        size = axis_text_size,
        colour = ECONOMIST_TEXT,
        face = "bold"
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.margin = margin(b = 5),
      legend.key.size = unit(0.8, "lines"),

      # Title - Bold black heading (Economist style)
      plot.title = element_text(
        size = title_size,
        face = "bold",
        colour = ECONOMIST_TITLE,
        hjust = 0,
        margin = margin(b = 5)
      ),
      plot.subtitle = element_text(
        size = subtitle_size,
        colour = ECONOMIST_TEXT,
        hjust = 0,
        margin = margin(b = 15)
      ),
      plot.caption = element_text(
        size = caption_size,
        colour = ECONOMIST_CAPTION,  # Lighter than body text for source citations
        hjust = 0,  # Left-aligned like Economist source citations
        margin = margin(t = 10)
      ),
      plot.title.position = "plot",
      plot.caption.position = "plot",

      # Facets
      strip.background = element_rect(fill = ECONOMIST_BG, colour = NA),
      strip.text = element_text(
        size = axis_title_size,
        face = "bold",
        colour = ECONOMIST_TEXT
      ),

      # Margins
      plot.margin = margin(15, 15, 10, 15),

      # Complete theme
      complete = TRUE
    )
}

#' Theme variant for calibration plots (with marginal histogram)
#' @export
theme_calibration <- function(...) {
  theme_foundation_plr(...) %+replace%
    theme(
      legend.position = "right",
      legend.direction = "vertical",
      panel.grid.major.x = element_line(colour = ECONOMIST_GRID, linewidth = 0.3)
    )
}

#' Theme variant for forest plots
#' @export
theme_forest <- function(...) {
  theme_foundation_plr(...) %+replace%
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(colour = ECONOMIST_GRID, linewidth = 0.3),
      axis.ticks.y = element_blank()
    )
}

#' Theme variant for heatmaps
#' @export
theme_heatmap <- function(...) {
  theme_foundation_plr(...) %+replace%
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "right"
    )
}

#' Theme variant for scatter plots (like the Economist example)
#' @export
theme_scatter <- function(...) {
  theme_foundation_plr(...) %+replace%
    theme(
      panel.grid.major.x = element_line(colour = ECONOMIST_GRID, linewidth = 0.3)
    )
}

# ==============================================================================
# ECONOMIST COLOR SCALES
# ==============================================================================

#' Economist-style discrete color scale
#' @export
scale_color_economist <- function(...) {
  scale_color_manual(values = ECONOMIST_COLORS, ...)
}

#' Economist-style discrete fill scale
#' @export
scale_fill_economist <- function(...) {
  scale_fill_manual(values = ECONOMIST_COLORS, ...)
}

# Note: Don't auto-set theme_set() as it can cause issues with some ggplot2 versions
# Users should call: theme_set(theme_foundation_plr()) manually if desired

message("Foundation PLR Economist-style theme loaded.")
message("Background: off-white (#FBF9F3)")
message("Use theme_foundation_plr() or scale_color_economist()")
