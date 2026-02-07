#!/usr/bin/env Rscript
# ==============================================================================
# Selective Classification Figure (1x3 panels)
# ==============================================================================
#
# Shows how model performance changes when rejecting uncertain predictions.
# X-axis: Retained data (1.0 → 0.1) - as we retain less, only confident preds remain
# Y-axis: AUROC, Net Benefit, Scaled Brier - should IMPROVE as retention decreases
#
# Clinical interpretation: "refer 50% most uncertain patients to expert"
#
# References:
#   - Geifman & El-Yaniv (2017) "Selective Classification for DNNs"
#   - OATML BDL Benchmarks (diabetic retinopathy diagnosis)
#   - torch-uncertainty AURC implementation
#
# Data Source: data/r_data/selective_classification_data.json
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(jsonlite)
})

# Find project root
PROJECT_ROOT <- (function() {
  markers <- c("pyproject.toml", "CLAUDE.md", ".git")
  dir <- getwd()
  while (dir != dirname(dir)) {
    if (any(file.exists(file.path(dir, markers)))) return(dir)
    dir <- dirname(dir)
  }
  stop("Could not find project root")
})()

# Source figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# ==============================================================================
# Load Data
# ==============================================================================
message("[fig_selective_classification] Loading data...")

data <- validate_data_source("selective_classification_data.json")
message("  Loaded ", length(data$data$configs), " configs")
message("  Retention levels: ", paste(range(unlist(data$data$retention_levels)), collapse = " to "))

# ==============================================================================
# Prepare Data
# ==============================================================================

build_panel_df <- function(data, metric) {
  metric_col <- paste0(metric, "_at_retention")

  df_list <- lapply(seq_along(data$data$configs), function(i) {
    cfg <- data$data$configs[[i]]
    retention_levels <- unlist(data$data$retention_levels)
    values_list <- cfg[[metric_col]]
    values <- sapply(values_list, function(v) {
      if (is.null(v)) NA_real_ else as.numeric(v)
    })
    n <- length(retention_levels)
    data.frame(
      config = rep(cfg$name, n),
      config_id = rep(cfg$id, n),
      retained = retention_levels,
      value = values,
      stringsAsFactors = FALSE
    )
  })

  df <- bind_rows(df_list) %>%
    filter(!is.na(value))

  return(df)
}

# Get colors from YAML
color_defs <- load_color_definitions()

# Resolve colors for each config (using YAML-defined fallback)
colors <- sapply(data$data$configs, function(cfg) {
  ref <- cfg$color_ref
  if (is.null(ref)) {
    # Use semantic fallback color from YAML instead of hardcoding
    ref <- "--color-text-secondary"
  }
  resolve_color(ref, color_defs)
})
names(colors) <- sapply(data$data$configs, function(cfg) cfg$name)
message("  Colors: ", paste(names(colors), collapse = ", "))

# ==============================================================================
# Create Panel Function
# ==============================================================================
create_panel <- function(data, metric, y_label, y_limits = NULL, colors, show_legend = FALSE) {
  df <- build_panel_df(data, metric)

  config_order <- sapply(data$data$configs, function(cfg) cfg$name)
  df$config <- factor(df$config, levels = config_order)

  p <- ggplot(df, aes(x = retained, y = value, color = config)) +
    geom_line(linewidth = 1) +
    geom_point(size = 1.5, alpha = 0.7) +
    scale_color_manual(values = colors, name = NULL) +
    scale_x_reverse(
      limits = c(1.0, 0.1),
      breaks = seq(1.0, 0.1, -0.2),
      labels = scales::percent_format(accuracy = 1)
    ) +
    labs(x = "Retained Data", y = y_label) +
    theme_foundation_plr() +
    theme(
      legend.position = if (show_legend) "right" else "none",
      plot.margin = margin(5, 10, 5, 5)
    )

  if (!is.null(y_limits)) {
    p <- p + scale_y_continuous(limits = y_limits)
  }

  return(p)
}

# ==============================================================================
# Create Panels
# ==============================================================================
message("[fig_selective_classification] Creating panels...")

# Create panels - one with legend that will be collected at bottom
# Each panel gets its own "A  Title" style header
p1 <- create_panel(data, "auroc", "AUROC", y_limits = c(0.5, 1.0), colors, show_legend = TRUE) +
  ggtitle("A  Discrimination (AUROC)") +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0), plot.title.position = "plot")

p2 <- create_panel(data, "net_benefit", "Net Benefit", y_limits = NULL, colors, show_legend = FALSE) +
  ggtitle("B  Clinical Utility") +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0), plot.title.position = "plot")

p3 <- create_panel(data, "scaled_brier", "Scaled Brier (IPA)", y_limits = NULL, colors, show_legend = FALSE) +
  ggtitle("C  Overall Performance") +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0), plot.title.position = "plot")

# ==============================================================================
# Compose Figure with Shared Bottom Legend
# ==============================================================================
message("[fig_selective_classification] Composing figure...")

# Compose panels with shared legend at bottom
# Panel titles use ggtitle() with "A  Title" format (no separate tag_levels)
composed <- (p1 | p2 | p3) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.key.width = unit(1.0, "cm"),
    legend.text = element_text(size = 9)
  ) &
  guides(color = guide_legend(nrow = 1))

# ==============================================================================
# Save Figure (using figure system)
# ==============================================================================
message("[fig_selective_classification] Saving figure...")

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(
  composed,
  "fig_selective_classification"
)

message("[fig_selective_classification] DONE")
