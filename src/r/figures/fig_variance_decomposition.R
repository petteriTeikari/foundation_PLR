# Variance Decomposition Plot (2-Panel Lollipop)
# ===============================================
# Shows η² (eta-squared) for factors in factorial design
#
# Panel A: Preprocessing factors only (fixed CatBoost)
# Panel B: All factors including classifier (shows classifier dominates)
#
# Narrative: "Classifier choice explains most variance, but that's known -
# everyone expects CatBoost > LR. Our focus is preprocessing variance."
#
# Created: 2026-01-25
# Updated: 2026-01-27 - Redesigned as 2-panel lollipop per user feedback
# Author: Foundation PLR Team

# ==============================================================================
# SETUP
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
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
# LOAD DATA
# ==============================================================================

# Load metrics (all classifiers for Panel B)
metrics_all <- read_csv("data/r_data/essential_metrics.csv", show_col_types = FALSE)

message(sprintf("Loaded %d total configurations", nrow(metrics_all)))
message(sprintf("Classifiers: %s", paste(unique(metrics_all$classifier), collapse = ", ")))

# ==============================================================================
# HELPER: Compute ANOVA eta-squared
# ==============================================================================

compute_eta_squared <- function(df, formula_str) {
  # Fit ANOVA model
  model <- aov(as.formula(formula_str), data = df)
  anova_table <- summary(model)[[1]]

  # Calculate eta-squared (proportion of variance explained)
  ss_total <- sum(anova_table$`Sum Sq`)

  # Build results - handle missing F/p values gracefully
  factor_names <- rownames(anova_table)
  n_factors <- length(factor_names)

  # F and p values may have NAs or be shorter (missing for Residuals)
  f_vals <- rep(NA_real_, n_factors)
  p_vals <- rep(NA_real_, n_factors)
  if (!is.null(anova_table$`F value`) && length(anova_table$`F value`) > 0) {
    f_vals[1:length(anova_table$`F value`)] <- anova_table$`F value`
  }
  if (!is.null(anova_table$`Pr(>F)`) && length(anova_table$`Pr(>F)`) > 0) {
    p_vals[1:length(anova_table$`Pr(>F)`)] <- anova_table$`Pr(>F)`
  }

  data.frame(
    factor_raw = factor_names,
    sum_sq = anova_table$`Sum Sq`,
    df = anova_table$Df,
    f_value = f_vals,
    p_value = p_vals,
    eta_squared = anova_table$`Sum Sq` / ss_total,
    stringsAsFactors = FALSE
  )
}

# ==============================================================================
# PANEL A: Preprocessing Only (CatBoost fixed)
# ==============================================================================

message("\n=== Panel A: Preprocessing Factors (CatBoost Only) ===")

# Filter to CatBoost
metrics_catboost <- metrics_all %>%
  filter(toupper(classifier) == "CATBOOST") %>%
  filter(!is.na(auroc)) %>%
  mutate(
    outlier_method = as.factor(outlier_method),
    imputation_method = as.factor(imputation_method)
  )

message(sprintf("CatBoost configurations: %d", nrow(metrics_catboost)))

# Compute ANOVA (additive model - no interaction due to sparse cells)
# Note: With 57 rows for 11×7 design, many cells have <1 observation
# so we cannot estimate interaction effects reliably
eta_preprocessing <- compute_eta_squared(
  metrics_catboost,
  "auroc ~ outlier_method + imputation_method"
)

# Add friendly labels
eta_preprocessing <- eta_preprocessing %>%
  mutate(
    factor = case_when(
      grepl("outlier_method", factor_raw) ~ "Outlier\nMethod",
      grepl("imputation_method", factor_raw) ~ "Imputation\nMethod",
      grepl("Residuals", factor_raw) ~ "Residual",
      TRUE ~ factor_raw
    ),
    pct = eta_squared * 100,
    label = sprintf("%.1f%%", pct)
  ) %>%
  filter(factor != "Residual")

message("Eta-squared (preprocessing):")
print(eta_preprocessing[, c("factor", "pct")])

# ==============================================================================
# PANEL B: All Factors Including Classifier
# ==============================================================================

message("\n=== Panel B: All Factors Including Classifier ===")

# Prepare data with all classifiers
metrics_full <- metrics_all %>%
  filter(!is.na(auroc)) %>%
  mutate(
    outlier_method = as.factor(outlier_method),
    imputation_method = as.factor(imputation_method),
    classifier = as.factor(classifier)
  )

message(sprintf("All configurations: %d", nrow(metrics_full)))

# Compute ANOVA with classifier
eta_full <- compute_eta_squared(
  metrics_full,
  "auroc ~ classifier + outlier_method + imputation_method"
)

# Add friendly labels
eta_full <- eta_full %>%
  mutate(
    factor = case_when(
      grepl("classifier", factor_raw) ~ "Classifier",
      grepl("outlier_method", factor_raw) ~ "Outlier\nMethod",
      grepl("imputation_method", factor_raw) ~ "Imputation\nMethod",
      grepl("Residuals", factor_raw) ~ "Residual",
      TRUE ~ factor_raw
    ),
    pct = eta_squared * 100,
    label = sprintf("%.1f%%", pct),
    highlight = factor == "Classifier"  # Highlight classifier dominance
  ) %>%
  filter(factor != "Residual")

message("Eta-squared (full model):")
print(eta_full[, c("factor", "pct")])

# ==============================================================================
# CREATE LOLLIPOP PANELS
# ==============================================================================

# Colors for factors (from YAML config)
factor_colors <- c(
  "Outlier\nMethod" = color_defs[["--color-primary"]],       # Economist blue
  "Imputation\nMethod" = color_defs[["--color-secondary"]],  # Light blue
  "Classifier" = color_defs[["--color-negative"]]            # Red (highlight dominance)
)

# Lollipop chart theme modifications (extends theme_foundation_plr)
# Note: Using inline theme object per CRITICAL-FAILURE-004 - no custom theme functions
lollipop_theme_mods <- theme(
  axis.text.y = element_text(size = 10, face = "bold"),
  panel.grid.major.x = element_line(color = color_defs[["--color-grid"]], linewidth = 0.5),
  panel.grid.minor.x = element_blank()
)

# Panel A: Preprocessing only
p_preprocessing <- ggplot(eta_preprocessing, aes(x = pct, y = reorder(factor, pct))) +
  # Lollipop stick
  geom_segment(
    aes(x = 0, xend = pct, y = factor, yend = factor, color = factor),
    linewidth = 1.5
  ) +
  # Lollipop head
  geom_point(aes(color = factor), size = 5) +
  # Value labels
  geom_text(
    aes(label = label),
    hjust = -0.3,
    size = 3.5,
    fontface = "bold"
  ) +
  scale_color_manual(values = factor_colors, guide = "none") +
  scale_x_continuous(
    limits = c(0, max(eta_preprocessing$pct) * 1.25),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = expression(eta^2~"(% variance explained)"),
    y = NULL
  ) +
  ggtitle("A  Preprocessing Only") +
  theme_foundation_plr() + lollipop_theme_mods +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.title.position = "plot"
  )

# Panel B: Full model with classifier
p_full <- ggplot(eta_full, aes(x = pct, y = reorder(factor, pct))) +
  # Lollipop stick
  geom_segment(
    aes(x = 0, xend = pct, y = factor, yend = factor, color = factor),
    linewidth = 1.5
  ) +
  # Lollipop head
  geom_point(aes(color = factor), size = 5) +
  # Value labels
  geom_text(
    aes(label = label),
    hjust = -0.3,
    size = 3.5,
    fontface = "bold"
  ) +
  scale_color_manual(values = factor_colors, guide = "none") +
  scale_x_continuous(
    limits = c(0, max(eta_full$pct) * 1.15),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = expression(eta^2~"(% variance explained)"),
    y = NULL
  ) +
  ggtitle("B  Full Model") +
  theme_foundation_plr() + lollipop_theme_mods +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.title.position = "plot"
  )

# ==============================================================================
# COMPOSE FIGURE
# ==============================================================================

message("\n=== Composing 2-Panel Figure ===")

# Panel titles use ggtitle() with "A  Title" format (no separate tag_levels)
composed <- (p_preprocessing | p_full)

# Save using figure system
# Dimensions loaded from configs/VISUALIZATION/figure_registry.yaml
save_publication_figure(composed, "fig_variance_decomposition")

# ==============================================================================
# SUMMARY
# ==============================================================================

message("\n========================================")
message("Variance Decomposition Complete")
message("========================================")
message("Panel A: Preprocessing only (CatBoost fixed)")
outlier_pct <- eta_preprocessing$pct[eta_preprocessing$factor == "Outlier\nMethod"]
impute_pct <- eta_preprocessing$pct[eta_preprocessing$factor == "Imputation\nMethod"]
if (length(outlier_pct) > 0) message(sprintf("  - Outlier Method: %.1f%%", outlier_pct))
if (length(impute_pct) > 0) message(sprintf("  - Imputation Method: %.1f%%", impute_pct))

message("\nPanel B: Full model (all factors)")
classifier_pct <- eta_full$pct[eta_full$factor == "Classifier"]
preprocess_pct <- sum(eta_full$pct[eta_full$factor != "Classifier"])
message(sprintf("  - Classifier: %.1f%% (dominates)", classifier_pct))
message(sprintf("  - Preprocessing combined: %.1f%%", preprocess_pct))
message("\nNarrative: Classifier choice explains most variance,")
message("but that's expected (CatBoost > LogisticRegression).")
message("Our focus is preprocessing variance within best classifier.")
