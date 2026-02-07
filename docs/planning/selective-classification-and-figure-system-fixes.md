# Selective Classification Figures + ggplot2 System Fixes

**Status**: REVIEWED BY 3 AGENTS - READY FOR EXECUTION
**Created**: 2026-01-27
**GitHub Issue**: [#6](https://github.com/petteriTeikari/foundation_PLR/issues/6)

---

## What This Plan Covers

### New Figures to Implement

1. **`fig_selective_classification`** (Main, 1×3 panels)
   - Shows how model performance changes when rejecting uncertain predictions
   - X-axis: Retained data (1.0 → 0.1) — fewer samples = only confident predictions
   - Y-axis: AUROC, Net Benefit, Scaled Brier (should INCREASE as retention decreases)
   - Clinical use: "refer 50% most uncertain patients to expert"

2. **`fig_roc_rc_combined`** (Supplementary, 1×2 panels)
   - Left: ROC curves (TPR vs FPR) for 9 model configurations
   - Right: Risk-Coverage curves (Risk vs Coverage) — lower AURC = better uncertainty

3. **`fig_prob_dist_combined`** (Main) — combine existing standalone figures
4. **`fig_variance_combined`** (Main) — rewrite as 2-column lollipop chart

### System Fixes

- **Directory routing**: Main figures go to `main/`, not ROOT
- **Data deduplication**: Ground truth has 3× duplicate predictions (189 vs 63)
- **YAML enforcement**: All figures use config_loader.R, not hardcoded values

### Key Concept: Selective Classification

```
At 100% retention (keep all samples):  Baseline AUROC ~0.91
At 50% retention (keep confident 50%): AUROC should be HIGHER (~0.95?)
At 10% retention (keep confident 10%): AUROC should be HIGHEST

Why? By rejecting uncertain predictions, we keep only those the model
is confident about. IF confidence is well-calibrated, these are the
correct predictions → higher performance on retained set.

Clinical application: Model handles confident cases, experts handle uncertain.
```

---

## Executive Summary

Three reviewer agents (Data Pipeline, R/ggplot2, Architecture) identified critical issues. This plan consolidates their findings into an actionable implementation order.

---

## Critical Issues Summary

| Issue | Severity | Source |
|-------|----------|--------|
| Ground truth 3x duplicate predictions | CRITICAL | Data Pipeline |
| `main/` directory EMPTY | CRITICAL | Architecture |
| `top10_catboost` view missing | CRITICAL | Data Pipeline |
| Missing R scripts for 4 figures | HIGH | R/ggplot2 |
| Figure routing not working | HIGH | Architecture |
| `compose_from_config()` unused | MEDIUM | Architecture |

---

## Phase 1: Data Pipeline Fixes (Python)

### 1.1 Fix Ground Truth Deduplication

**Problem**: `pupil-gt + pupil-gt` has 189 predictions (3 MLflow runs × 63 subjects), others have 63.

**Fix in both export scripts**:

```python
def get_predictions_for_combo(conn, outlier_method, imputation_method, classifier):
    """Get predictions with deduplication by subject_id."""
    query = """
        WITH dedup AS (
            SELECT subject_id, y_true, y_prob,
                   ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY mlflow_run_id) as rn
            FROM predictions
            WHERE outlier_method = ?
              AND imputation_method = ?
              AND classifier = ?
        )
        SELECT y_true, y_prob
        FROM dedup
        WHERE rn = 1
        ORDER BY subject_id
    """
    return conn.execute(query, [outlier_method, imputation_method, classifier]).fetchall()
```

### 1.2 Fix Top-10 Fallback Logic

**Problem**: Fallback sorts by `AVG(y_prob)` instead of actual AUROC.

**Fix**: Compute AUROC properly in fallback:

```python
from sklearn.metrics import roc_auc_score

def get_top10_configs_fallback(conn):
    combos = conn.execute("""
        SELECT DISTINCT outlier_method, imputation_method
        FROM predictions WHERE classifier = 'CATBOOST'
    """).fetchall()

    auroc_scores = []
    for outlier, imputation in combos:
        y_true, y_prob = get_predictions_deduplicated(conn, outlier, imputation, 'CATBOOST')
        if len(np.unique(y_true)) == 2:
            auroc = roc_auc_score(y_true, y_prob)
            auroc_scores.append((outlier, imputation, auroc))

    auroc_scores.sort(key=lambda x: x[2], reverse=True)
    return auroc_scores[:10]
```

### 1.3 Validate AURC/RC Computation

**Current implementation is CORRECT** per reviewer:
- Confidence = `|y_prob - 0.5|` (distance from decision boundary)
- Samples sorted by confidence descending
- Risk = error rate on accepted samples
- Coverage = fraction of samples accepted

**Note on user request**: User wants **rejection ratio** on x-axis (not coverage).
- Rejection ratio = 1 - coverage
- So x-axis goes 0→1 where 0 = accept all, 1 = reject all

### 1.4 Re-export JSON Files

After fixes, re-run:
```bash
python scripts/export_selective_classification_for_r.py
python scripts/export_roc_rc_data.py
```

---

## Phase 2: R Script Creation

### 2.1 `fig_selective_classification.R` (NEW)

**Layout**: 1×3 (three panels side by side)
**X-axis**: Rejection ratio (1 - coverage)
**Panels**: AUROC, Net Benefit, Scaled Brier

```r
# src/r/figures/fig_selective_classification.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(jsonlite)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("pyproject.toml"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load data
data <- validate_data_source("selective_classification_data.json")
colors <- get_main4_colors()  # From config

# Create panel function
create_panel <- function(data, metric, y_label, colors, show_legend = FALSE) {
  df <- lapply(seq_along(data$data$configs), function(i) {
    cfg <- data$data$configs[[i]]
    metric_col <- paste0(metric, "_at_retention")
    data.frame(
      config = cfg$name,
      retained = data$data$retention_levels,  # Fraction of data retained
      value = cfg[[metric_col]]
    )
  }) %>% bind_rows() %>% filter(!is.na(value))

  p <- ggplot(df, aes(x = retained, y = value, color = config)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = colors, name = NULL) +
    # X-axis: retained data, 1.0 on RIGHT (baseline), decreasing LEFT (higher perf)
    scale_x_reverse(limits = c(1.0, 0.1), breaks = seq(1.0, 0.1, -0.1)) +
    labs(x = "Retained Data", y = y_label) +
    theme_foundation_plr()

  if (!show_legend) p <- p + theme(legend.position = "none")
  return(p)
}

# Create panels
p1 <- create_panel(data, "auroc", "AUROC", colors)
p2 <- create_panel(data, "net_benefit", "Net Benefit", colors)
p3 <- create_panel(data, "scaled_brier", "Scaled Brier (IPA)", colors, show_legend = TRUE)

# Compose
composed <- (p1 | p2 | p3) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))

# Save to MAIN directory
save_publication_figure(composed, "fig_selective_classification",
                        category = "main", width = 14, height = 5)
```

### 2.2 `fig_roc_rc_combined.R` (NEW)

**Layout**: 1×2 (ROC left, RC right)
**Models**: 8 handpicked + Top-10 Mean aggregate

```r
# src/r/figures/fig_roc_rc_combined.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(jsonlite)
})

PROJECT_ROOT <- rprojroot::find_root(rprojroot::has_file("pyproject.toml"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/config_loader.R"))
source(file.path(PROJECT_ROOT, "src/r/figure_system/save_figure.R"))
source(file.path(PROJECT_ROOT, "src/r/theme_foundation_plr.R"))

# Load data
data <- validate_data_source("roc_rc_curves_data.json")

# Build ROC dataframe
roc_df <- lapply(data$data$configs, function(cfg) {
  data.frame(
    config = cfg$name,
    fpr = cfg$roc$fpr,
    tpr = cfg$roc$tpr,
    auroc = cfg$roc$auroc
  )
}) %>% bind_rows()

# Build RC dataframe
rc_df <- lapply(data$data$configs, function(cfg) {
  data.frame(
    config = cfg$name,
    coverage = cfg$rc$coverage,
    risk = cfg$rc$risk,
    aurc = cfg$rc$aurc
  )
}) %>% bind_rows()

# Get colors from config
colors <- get_extended_colors()  # 9 colors for 8 handpicked + top10_mean

# ROC panel
p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = config)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = colors, name = NULL) +
  coord_equal() +
  labs(x = "False Positive Rate", y = "True Positive Rate",
       subtitle = "Higher AUROC = Better") +
  theme_foundation_plr() +
  theme(legend.position = "none")

# RC panel (Risk-Coverage)
p_rc <- ggplot(rc_df, aes(x = coverage, y = risk, color = config)) +
  geom_line(linewidth = 0.8) +
  scale_color_manual(values = colors, name = NULL) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "Coverage", y = "Risk (Error Rate)",
       subtitle = "Lower AURC = Better") +
  theme_foundation_plr() +
  theme(legend.position = "right")

# Compose
composed <- (p_roc | p_rc) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(face = "bold", size = 14))

# Save to SUPPLEMENTARY directory
save_publication_figure(composed, "fig_roc_rc_combined",
                        category = "supplementary", width = 14, height = 7)
```

### 2.3 `fig_prob_dist_combined.R` (NEW)

Combine existing standalone prob_dist figures into 1×2 layout.

### 2.4 `fig_variance_combined.R` (Rewrite)

Convert from single-column bar to 2-column horizontal lollipop:
- Panel A: Preprocessing effects (CatBoost fixed)
- Panel B: Full pipeline including classifier

---

## Phase 3: Fix Directory Routing

### 3.1 Update `save_publication_figure()`

Add explicit `category` parameter:

```r
save_publication_figure <- function(plot, name, category = NULL,
                                    width = NULL, height = NULL) {
  config <- yaml::read_yaml("configs/VISUALIZATION/figure_layouts.yaml")

  # Determine output directory
  if (!is.null(category)) {
    output_dir <- config$figure_categories[[category]]$output_dir
  } else {
    # Try to auto-detect from figure name
    output_dir <- .get_figure_output_dir(name, config)
  }

  # Create directory if needed
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Save
  filepath <- file.path(output_dir, paste0(name, ".png"))
  ggsave(filepath, plot, width = width, height = height, dpi = 300)
  message(sprintf("Saved: %s", filepath))
}
```

### 3.2 Create `main/` Directory

```bash
mkdir -p figures/generated/ggplot2/main
```

### 3.3 Clean Up ROOT Directory

```bash
# After combined versions are created, delete standalones
rm figures/generated/ggplot2/fig_prob_dist_by_outcome.png
rm figures/generated/ggplot2/fig_prob_dist_faceted.png
rm figures/generated/ggplot2/fig04_variance_decomposition.png
rm figures/generated/ggplot2/fig_shap_importance_gt.png
rm figures/generated/ggplot2/fig_shap_importance_ensemble.png
rm figures/generated/ggplot2/fig_vif_analysis.png
rm figures/generated/ggplot2/fig_vif_by_wavelength.png
```

---

## Phase 4: Regenerate All Figures

```bash
# 1. Fix and re-export data
python scripts/export_selective_classification_for_r.py
python scripts/export_roc_rc_data.py

# 2. Generate all R figures
Rscript src/r/figures/generate_all_r_figures.R

# 3. Verify output
ls -la figures/generated/ggplot2/main/       # Should have 6 figures
ls -la figures/generated/ggplot2/supplementary/  # Should have 10 figures
ls -la figures/generated/ggplot2/extra-supplementary/  # Should have 4 figures
ls -la figures/generated/ggplot2/*.png       # Should be EMPTY
```

---

## Phase 5: Validation

### Expected Final State

```
figures/generated/ggplot2/
├── main/                              # 6 figures
│   ├── fig_forest_combined.png
│   ├── fig_calibration_dca_combined.png
│   ├── fig_prob_dist_combined.png
│   ├── fig_variance_combined.png
│   ├── fig_shap_importance_multi.png
│   └── fig_selective_classification.png  ← NEW
│
├── supplementary/                     # 10 figures
│   ├── cd_preprocessing.png
│   ├── fig05_shap_beeswarm.png
│   ├── fig06_specification_curve.png
│   ├── fig07_heatmap_preprocessing.png
│   ├── fig_R7_featurization_comparison.png
│   ├── fig_raincloud_auroc.png
│   ├── fig_shap_heatmap.png
│   ├── fig_shap_importance_combined.png
│   ├── fig_vif_combined.png
│   └── fig_roc_rc_combined.png       ← NEW
│
└── extra-supplementary/               # 4 figures
    ├── fig_M3_factorial_matrix.png
    ├── fig_R8_fm_dashboard.png
    ├── fig_shap_gt_vs_ensemble.png
    └── fig_shap_gt_vs_ensemble_bars.png
```

---

## Implementation Order

| Step | Task | Files | Complexity |
|------|------|-------|------------|
| 1 | Fix deduplication in export scripts | `scripts/export_*.py` | Low |
| 2 | Fix top10 fallback logic | `scripts/export_roc_rc_data.py` | Medium |
| 3 | Re-export JSON data | Run scripts | Low |
| 4 | Create `fig_selective_classification.R` | New file | Medium |
| 5 | Create `fig_roc_rc_combined.R` | New file | Medium |
| 6 | Create `fig_prob_dist_combined.R` | New file | Low |
| 7 | Rewrite `fig_variance_combined.R` | Rewrite | Medium |
| 8 | Fix `save_publication_figure()` routing | `save_figure.R` | Low |
| 9 | Create main/ directory | Bash | Trivial |
| 10 | Regenerate all figures | Run R script | Low |
| 11 | Clean up ROOT directory | Bash | Trivial |
| 12 | Validate outputs | Manual check | Low |

---

## Reference: Selective Classification / Retention Curves

### Key Concept (User Clarification)

```
Retention Curve (AUC vs Retained Data):
- X-axis: Retained Data fraction (1.0 = keep all, 0.5 = keep top 50% confident)
- Y-axis: AUROC (or Accuracy) on retained samples
- At 100% retention: baseline AUROC (~0.91 in our case)
- As we reject uncertain samples (move LEFT): AUROC should INCREASE
- This works IF confidence is well-calibrated

Key insight: "referring 50% most uncertain patients to an expert"
- Clinical use case: model handles confident predictions, experts handle uncertain ones
- Good uncertainty = AUROC increases significantly when rejecting uncertain samples
- Poor uncertainty = AUROC stays flat (model doesn't know what it doesn't know)
```

### Visual Reference (from OATML bdl-benchmarks)

```
        89 ─┬─────────────────────────
           │     ╲
   AUC  87 │      ╲  MC Dropout
           │       ╲
        85 │        ╲___
           │             ╲
        83 │              ╲ Deterministic
           │               ╲
        81 ─┴─────────────────────────
           0.5   0.6   0.7   0.8   0.9   1.0
                    retained data

Interpretation:
- At 1.0 (retain all): All models ~84% AUC
- At 0.5 (retain confident 50%): MC Dropout reaches 88%, Deterministic only 85%
- MC Dropout has BETTER uncertainty calibration (steeper curve)
```

### Our Figure: `fig_selective_classification`

**Corrected specification**:
- X-axis: **Retained Data** (1.0 → 0.1, right to left like the reference)
- Y-axis panels: AUROC, Net Benefit, Scaled Brier
- At retention=1.0: baseline metrics (AUROC ~0.91)
- As retention decreases: metrics should IMPROVE if confidence is calibrated

**Alternative X-axis** (user's original request):
- Could use Rejection Ratio (0 → 0.9) going left to right
- Rejection Ratio = 1 - Retention
- Same interpretation, just flipped axis

---

## References (User-Provided)

### 1. OATML BDL Benchmarks - Diabetic Retinopathy Diagnosis
**URL**: https://github.com/OATML/bdl-benchmarks/tree/alpha/baselines/diabetic_retinopathy_diagnosis

Key paper: Leibig et al. - Uses uncertainty for pre-screening, referring uncertain patients to experts.

Methods benchmarked:
- MC Dropout (Gal & Ghahramani, 2015)
- Mean-Field Variational Inference (Peterson & Anderson, 1987; Wen et al., 2018)
- Deep Ensembles (Lakshminarayanan et al., 2016)
- Ensemble MC Dropout (Smith & Gal, 2018)

### 2. Benchmarking Uncertainty Estimation Performance
**URL**: https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance

Provides AURC implementation and selective classification metrics.

### 3. Failure Detection Paper (arXiv 2305.15508)
**URL**: https://arxiv.org/html/2305.15508v4

Comprehensive overview of selective classification, Risk-Coverage curves, and AURC.

### 4. torch-uncertainty AURC Implementation
**URL**: https://torch-uncertainty.github.io/generated/torch_uncertainty.metrics.classification.AURC.html

Reference implementation in PyTorch. AURC = "main metric for Selective Classification (SC) performance assessment."

### 5. Sanofi risk.assessr (R Package)
**URL**: https://github.com/Sanofi-Public/risk.assessr

R-native implementation for clinical/pharma context. Validated for regulatory use.

### 6. Geifman & El-Yaniv 2017 (NeurIPS)
**Paper**: "Selective Classification for Deep Neural Networks"

Original paper defining selective classification framework and Risk-Coverage curves.

---

## Implementation Note: X-Axis Direction

The exported JSON has `retention_levels: [0.1, 0.15, ..., 1.0]` going UP.

For the plot matching the OATML reference style:
```r
# Option A: Retained data on x-axis (1.0 → 0.1, decreasing left)
scale_x_reverse(limits = c(1.0, 0.1))

# Option B: Keep as-is but interpret correctly
# x=1.0 (right) = retain all = baseline
# x=0.1 (left) = retain only 10% most confident = should be higher AUROC
```

The key is that **performance should increase as we move LEFT** (fewer retained samples = only confident predictions kept).

---

## Approval Checklist

- [ ] Data pipeline fixes reviewed
- [ ] R script templates reviewed
- [ ] Directory structure approved
- [ ] Implementation order approved

**Ready for execution upon user approval.**
