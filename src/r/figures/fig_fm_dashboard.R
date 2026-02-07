# Foundation Model Dashboard
# Shows FM performance across different tasks (outlier detection, imputation, featurization)
# Task 2.8 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
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
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))
source(file.path(PROJECT_ROOT, "src/r/color_palettes.R"))

# Load color definitions from YAML
color_defs <- load_color_definitions()

# ==============================================================================
# DATA (from manuscript findings)
# ==============================================================================

# Foundation model performance across tasks
# Values based on paper findings
fm_dashboard <- data.frame(
  task = c("Outlier Detection", "Outlier Detection",
           "Imputation", "Imputation",
           "Feature Extraction", "Feature Extraction"),
  method = c("Foundation Model", "Traditional/Ground Truth",
             "Foundation Model", "Traditional/Ground Truth",
             "Foundation Model", "Traditional/Ground Truth"),
  performance = c(
    # Outlier Detection: FM vs Ground Truth
    0.88, 0.91,  # FM competitive but GT still best
    # Imputation: FM vs Traditional
    0.89, 0.88,  # FM slightly better
    # Feature Extraction: FM embeddings vs Handcrafted
    0.74, 0.83   # FM much worse
  ),
  stringsAsFactors = FALSE
)

# Add verdict column
fm_dashboard <- fm_dashboard %>%
  group_by(task) %>%
  mutate(
    fm_perf = performance[method == "Foundation Model"],
    trad_perf = performance[method == "Traditional/Ground Truth"],
    verdict = case_when(
      fm_perf > trad_perf + 0.02 ~ "FM Better",
      fm_perf < trad_perf - 0.02 ~ "Traditional Better",
      TRUE ~ "Comparable"
    )
  ) %>%
  ungroup() %>%
  mutate(
    task = factor(task, levels = c("Outlier Detection", "Imputation", "Feature Extraction"))
  )

# ==============================================================================
# DASHBOARD PLOT
# ==============================================================================

# Bar plot with grouped bars
p_dashboard <- ggplot(fm_dashboard, aes(x = task, y = performance, fill = method))
p_dashboard <- gg_add(p_dashboard,
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.9),
  # Value labels
  geom_text(
    aes(label = sprintf("%.2f", performance)),
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 4,
    fontface = "bold"
  ),
  # Color scale (from YAML config)
  scale_fill_manual(
    values = c("Foundation Model" = color_defs[["--color-secondary"]], "Traditional/Ground Truth" = color_defs[["--color-primary"]]),
    name = "Method Type"
  ),
  # Y-axis
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = expansion(mult = c(0, 0.15))
  ),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = NULL,
    y = "AUROC (downstream classification)"
  ),
  theme_foundation_plr(),
  theme(
    legend.position = "top",
    axis.text.x = element_text(size = 11, face = "bold")
  )
)

# Add verdict annotations
verdict_data <- fm_dashboard %>%
  select(task, verdict) %>%
  distinct() %>%
  mutate(
    verdict_color = case_when(
      verdict == "FM Better" ~ color_defs[["--color-positive"]],
      verdict == "Traditional Better" ~ color_defs[["--color-negative"]],
      TRUE ~ color_defs[["--color-text-secondary"]]
    ),
    y_pos = 0.95
  )

p_dashboard <- gg_add(p_dashboard,
  geom_text(
    data = verdict_data,
    aes(x = task, y = y_pos, label = verdict, color = verdict),
    inherit.aes = FALSE,
    size = 3.5,
    fontface = "italic"
  ),
  scale_color_manual(
    values = c(
      "FM Better" = color_defs[["--color-positive"]],
      "Traditional Better" = color_defs[["--color-negative"]],
      "Comparable" = color_defs[["--color-text-secondary"]]
    ),
    guide = "none"
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_dashboard, "fig_fm_dashboard")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Foundation Model Dashboard Complete")
message("========================================")
message("Key findings by task:")
for (t in unique(fm_dashboard$task)) {
  subset <- fm_dashboard %>% filter(task == t)
  message(sprintf("  %s: FM=%.2f, Trad=%.2f (%s)",
                  t, subset$fm_perf[1], subset$trad_perf[1], subset$verdict[1]))
}
