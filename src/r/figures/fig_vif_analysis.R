# VIF (Variance Inflation Factor) Analysis Plot
# Economist-style ggplot2 visualization
# Task 4.1 from ggplot2-viz-creation-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses gg_add() instead of + operator for S7/ggplot2 4.0+ compatibility

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
})

# Determine project root
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
source(file.path(PROJECT_ROOT, "src/r/figure_system/compose_figures.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# LOAD DATA
# ==============================================================================

vif_data <- fromJSON("data/r_data/vif_analysis.json")

# Extract aggregate VIF
vif_df <- as_tibble(vif_data$data$aggregate) %>%
  mutate(
    feature = factor(feature, levels = rev(feature)),
    concern = factor(concern, levels = c("OK", "Moderate", "High")),
    wavelength = case_when(
      grepl("^Blue_", feature) ~ "Blue (469nm)",
      grepl("^Red_", feature) ~ "Red (640nm)",
      TRUE ~ "Combined"
    )
  )

# ==============================================================================
# FIGURE 1: VIF Bar Chart colored by Wavelength
# ==============================================================================

# Color bars by wavelength (not concern level - avoids confusion)
wavelength_colors <- c(
  "Blue (469nm)" = color_defs[["--color-primary"]],
  "Red (640nm)" = color_defs[["--color-negative"]]
)

p_vif <- ggplot(vif_df, aes(x = VIF_mean, y = feature, fill = wavelength))
p_vif <- gg_add(p_vif,
  geom_col(width = 0.7),
  # Threshold lines in neutral gray
  geom_vline(
    xintercept = 10,
    linetype = "dashed",
    color = color_defs[["--color-text-secondary"]],
    linewidth = 0.5
  ),
  geom_vline(
    xintercept = 20,
    linetype = "dashed",
    color = color_defs[["--color-text-primary"]],
    linewidth = 0.5
  ),
  # Annotations at bottom - "High" slightly higher than "Moderate"
  annotate(
    "text",
    x = 11, y = 0.2,
    label = "Moderate",
    color = color_defs[["--color-text-secondary"]],
    size = 2.3,
    hjust = 0,
    fontface = "italic"
  ),
  annotate(
    "text",
    x = 21, y = 0.55,
    label = "High",
    color = color_defs[["--color-text-primary"]],
    size = 2.3,
    hjust = 0,
    fontface = "italic"
  ),
  # Extend y-axis down to show labels
  coord_cartesian(ylim = c(-0.2, 8.5), clip = "off"),
  scale_fill_manual(values = wavelength_colors, name = "Stimulus"),
  scale_x_continuous(
    breaks = c(0, 25, 50, 75, 100, 125),
    limits = c(0, 130)
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Mean VIF",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = "top",
    legend.justification = "left",
    legend.margin = margin(b = 10)
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_vif, "fig_vif_analysis")

# ==============================================================================
# FIGURE 2: VIF by Wavelength (Faceted)
# ==============================================================================

p_vif_facet <- ggplot(vif_df, aes(x = VIF_mean, y = feature, fill = wavelength))
p_vif_facet <- gg_add(p_vif_facet,
  geom_col(width = 0.7),
  geom_errorbarh(
    aes(xmin = VIF_min, xmax = VIF_max),
    height = 0.3,
    color = color_defs[["--color-text-primary"]],
    linewidth = 0.3
  ),
  geom_vline(
    xintercept = 10,
    linetype = "dashed",
    color = color_defs[["--color-annotation"]],
    linewidth = 0.5
  ),
  scale_fill_manual(
    values = c(
      "Blue (469nm)" = color_defs[["--color-primary"]],
      "Red (640nm)" = color_defs[["--color-negative"]]
    ),
    name = "Stimulus"
  ),
  facet_wrap(~ wavelength, scales = "free_y", ncol = 1),
  # Labels (academic mode)
  labs(
    x = "VIF",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 11)
  )
)

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_vif_facet, "fig_vif_by_wavelength")

# ==============================================================================
# FIGURE 3: COMBINED VIF (1x2 layout)
# ==============================================================================

# Add "A  Title" style headers (consistent pattern across all figures)
title_theme <- theme(
  plot.title = element_text(face = "bold", size = 14, hjust = 0),
  plot.title.position = "plot"
)

p_vif_titled <- p_vif + ggtitle("A  Overall VIF") + title_theme
p_vif_facet_titled <- p_vif_facet + ggtitle("B  VIF by Wavelength") + title_theme

# Compose with patchwork (no tag_levels - using manual ggtitle)
fig_vif_combined <- (p_vif_titled | p_vif_facet_titled) +
  patchwork::plot_layout(widths = c(1, 1))

# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(fig_vif_combined, "fig_vif_combined")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("VIF Analysis Figures Complete")
message("========================================")
message("Output files:")
message("  - fig_vif_analysis.pdf (Main VIF bar chart)")
message("  - fig_vif_by_wavelength.pdf (Faceted by wavelength)")
message("\nKey findings:")
message(paste("  - High VIF features:", sum(vif_df$concern == "High")))
message(paste("  - Moderate VIF features:", sum(vif_df$concern == "Moderate")))
message(paste("  - OK features:", sum(vif_df$concern == "OK")))
