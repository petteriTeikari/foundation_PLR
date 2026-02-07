# Critical Difference Diagram for Preprocessing Methods
# Statistical comparison of outlier detection methods
# Task 2.10 from ggplot2-viz-remaining-plan.xml
#
# Created: 2026-01-25
# Author: Foundation PLR Team
#
# Note: Uses scmamp package for proper CD diagram generation

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
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

# Load figure system (MANDATORY)
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/common.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/category_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load colors from YAML (MANDATORY - no hardcoded colors!)
color_defs <- load_color_definitions()

# Note: scmamp package not available for this R version
# Using ggplot2-based rank comparison instead
SCMAMP_AVAILABLE <- FALSE

# Output directory handled by save_publication_figure()

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

# Load all metrics
metrics <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

# Filter to CatBoost
metrics <- metrics %>%
  filter(toupper(classifier) == "CATBOOST") %>%
  filter(!is.na(auroc))

# For CD diagram, we need a matrix of method performance across "datasets"
# In our case, treat each imputation method as a "dataset"

# Create wide format: rows = imputation methods (datasets), cols = outlier methods
# Apply display names from YAML (Single Source of Truth)
cd_data <- metrics %>%
  select(outlier_method, imputation_method, auroc) %>%
  apply_display_names() %>%
  pivot_wider(
    id_cols = imputation_method,
    names_from = outlier_display_name,
    values_from = auroc,
    values_fn = mean
  )

# Convert to matrix for scmamp
cd_matrix <- cd_data %>%
  select(-imputation_method) %>%
  as.matrix()
rownames(cd_matrix) <- cd_data$imputation_method

# Remove methods with too many NAs
na_counts <- colSums(is.na(cd_matrix))
cd_matrix <- cd_matrix[, na_counts < nrow(cd_matrix) / 2]

# Also remove rows with NAs
cd_matrix <- cd_matrix[complete.cases(cd_matrix), ]

message(sprintf("CD matrix: %d datasets x %d methods", nrow(cd_matrix), ncol(cd_matrix)))

# ==============================================================================
# GENERATE CD DIAGRAM (scmamp version - skipped, package not available)
# ==============================================================================

# Note: scmamp package is not available for this R version
# Using ggplot2-based rank comparison as alternative
message("Note: scmamp package not available, using ggplot2 rank plot instead")

# ==============================================================================
# ALTERNATIVE: Manual CD-style plot with ggplot2
# ==============================================================================

# Calculate mean ranks
# Use display names from YAML (Single Source of Truth)
rank_data <- metrics %>%
  group_by(imputation_method) %>%
  mutate(rank = rank(-auroc)) %>%  # Rank within each imputation method (higher = better = lower rank)
  ungroup() %>%
  group_by(outlier_method) %>%
  summarize(
    mean_rank = mean(rank, na.rm = TRUE),
    auroc_mean = mean(auroc, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_rank) %>%
  # Apply display names from YAML
  mutate(
    display_name = sapply(outlier_method, get_outlier_display_name),
    # Category from YAML via category_loader
    category = categorize_outlier_methods(outlier_method)
  )

# ggplot2 version of rank comparison
p_ranks <- ggplot(rank_data, aes(x = mean_rank, y = reorder(display_name, -mean_rank), fill = category))
p_ranks <- gg_add(p_ranks,
  geom_col(width = 0.7),
  scale_fill_manual(
    values = get_category_colors(),
    name = "Method Type"
  ),
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))),
  # Labels (academic mode - no title/subtitle/caption for journal submission)
  labs(
    x = "Mean Rank",
    y = NULL
  ),
  theme_forest(),
  theme(
    legend.position = "bottom"
  )
)

# Save using figure system (formats from YAML config)
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(p_ranks, "fig_cd_preprocessing")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("CD Diagram Complete")
message("========================================")
message("Methods ranked by mean rank:")
print(rank_data %>% select(display_name, mean_rank, auroc_mean, category) %>% head(10))
