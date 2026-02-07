# Figure Categorization and Grouping Plan

**Status**: YAML CONFIG DONE, FIGURES ORGANIZED
**Created**: 2026-01-27
**Goal**: Categorize figures into main/supplementary/extra-supplementary with YAML-driven configuration

---

## Overview

Figures need to be categorized by publication destination:

| Category | Destination | Count |
|----------|-------------|-------|
| **Main** | Main article body | 7 |
| **Supplementary** | Journal supplementary materials | 10 |
| **Extra-Supplementary** | Internal research archive only | 4 |

**Total**: 21 figures

---

## Output Directory Structure

```
figures/generated/ggplot2/
├── main/                           # Main article figures
│   ├── fig_forest_combined.png
│   ├── fig_calibration_dca_combined.png
│   ├── fig04_variance_decomposition.png
│   ├── fig05_shap_beeswarm.png
│   ├── fig_prob_dist_by_outcome.png
│   ├── fig_prob_dist_faceted.png
│   └── fig_shap_importance_multi.png
│
├── supplementary/                  # Journal supplementary materials
│   ├── cd_preprocessing.png
│   ├── fig06_specification_curve.png
│   ├── fig07_heatmap_preprocessing.png
│   ├── fig_R7_featurization_comparison.png
│   ├── fig_raincloud_auroc.png
│   ├── fig_shap_heatmap.png
│   ├── fig_shap_importance_ensemble.png
│   ├── fig_shap_importance_gt.png
│   ├── fig_vif_analysis.png
│   └── fig_vif_by_wavelength.png
│
└── extra-supplementary/            # Internal archive only
    ├── fig_M3_factorial_matrix.png
    ├── fig_R8_fm_dashboard.png
    ├── fig_shap_gt_vs_ensemble.png
    └── fig_shap_gt_vs_ensemble_bars.png
```

**Extra-supplementary archive location:**
```
/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/latent-methods-results/
```

---

## YAML Configuration Design

### Option A: Extend `figure_layouts.yaml` (Recommended)

Add categorization to existing config:

```yaml
# configs/VISUALIZATION/figure_layouts.yaml

# ... existing output_settings ...

figure_categories:
  main:
    description: "Main article figures"
    output_dir: "figures/generated/ggplot2/main"
    figures:
      - fig_forest_combined
      - fig_calibration_dca_combined
      - fig04_variance_decomposition
      - fig05_shap_beeswarm
      - fig_prob_dist_by_outcome
      - fig_prob_dist_faceted
      - fig_shap_importance_multi

  supplementary:
    description: "Journal supplementary materials"
    output_dir: "figures/generated/ggplot2/supplementary"
    figures:
      - cd_preprocessing
      - fig06_specification_curve
      - fig07_heatmap_preprocessing
      - fig_R7_featurization_comparison
      - fig_raincloud_auroc
      - fig_shap_heatmap
      - fig_shap_importance_ensemble
      - fig_shap_importance_gt
      - fig_vif_analysis
      - fig_vif_by_wavelength

  extra_supplementary:
    description: "Internal research archive only"
    output_dir: "figures/generated/ggplot2/extra-supplementary"
    archive_to: "/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/latent-methods-results/"
    figures:
      - fig_M3_factorial_matrix
      - fig_R8_fm_dashboard
      - fig_shap_gt_vs_ensemble
      - fig_shap_gt_vs_ensemble_bars
```

### Option B: Separate `figure_registry.yaml`

Create dedicated registry file:

```yaml
# configs/VISUALIZATION/figure_registry.yaml

version: "1.0.0"

figures:
  fig_forest_combined:
    category: main
    description: "Combined forest plot (Outlier + Imputation)"
    panels: ["A: Outlier Detection", "B: Imputation"]
    width: 10
    height: 12

  fig_calibration_dca_combined:
    category: main
    description: "Combined calibration and DCA (STRATOS)"
    panels: ["A: Calibration", "B: DCA"]
    width: 9
    height: 14

  cd_preprocessing:
    category: supplementary
    description: "Critical difference diagram for preprocessing"
    width: 10
    height: 8

  fig_M3_factorial_matrix:
    category: extra_supplementary
    description: "Factorial design matrix"
    width: 10
    height: 9

  # ... etc for all figures
```

---

## Categorization Summary

### Main Figures (7)

| Figure | Description | Panels |
|--------|-------------|--------|
| `fig_forest_combined` | Preprocessing method comparison | A: Outlier, B: Imputation |
| `fig_calibration_dca_combined` | STRATOS clinical assessment | A: Calibration, B: DCA |
| `fig04_variance_decomposition` | ANOVA η² decomposition | Single |
| `fig05_shap_beeswarm` | SHAP feature importance | Single |
| `fig_prob_dist_by_outcome` | Probability distributions | Single |
| `fig_prob_dist_faceted` | Faceted probability distributions | Multi-panel |
| `fig_shap_importance_multi` | Multi-config SHAP comparison | Multi-panel |

### Supplementary Figures (10)

| Figure | Description | Rationale for Supplementary |
|--------|-------------|----------------------------|
| `cd_preprocessing` | CD diagram ranking | Supporting evidence |
| `fig06_specification_curve` | Specification curve | Methodological detail |
| `fig07_heatmap_preprocessing` | Full factorial heatmap | Detailed results |
| `fig_R7_featurization_comparison` | Handcrafted vs embeddings | Supporting comparison |
| `fig_raincloud_auroc` | AUROC distribution | Visual supplement |
| `fig_shap_heatmap` | SHAP value heatmap | Detailed SHAP |
| `fig_shap_importance_ensemble` | Ensemble SHAP | Pipeline-specific |
| `fig_shap_importance_gt` | Ground truth SHAP | Pipeline-specific |
| `fig_vif_analysis` | VIF multicollinearity | Diagnostics |
| `fig_vif_by_wavelength` | VIF by wavelength | Detailed diagnostics |

### Extra-Supplementary Figures (4)

| Figure | Description | Rationale for Internal Only |
|--------|-------------|----------------------------|
| `fig_M3_factorial_matrix` | Design matrix visualization | Experimental design documentation |
| `fig_R8_fm_dashboard` | FM performance dashboard | Internal summary |
| `fig_shap_gt_vs_ensemble` | GT vs Ensemble SHAP scatter | Exploratory analysis |
| `fig_shap_gt_vs_ensemble_bars` | GT vs Ensemble SHAP bars | Exploratory analysis |

---

## Implementation Plan

### Phase 1: YAML Configuration

1. Add `figure_categories` section to `configs/VISUALIZATION/figure_layouts.yaml`
2. Define output directories and figure lists for each category

### Phase 2: Update `save_publication_figure()`

Modify `src/r/figure_system/save_figure.R` to:
1. Accept optional `category` parameter
2. Look up category from YAML if not provided
3. Route output to correct subdirectory

```r
save_publication_figure <- function(plot, name, width = NULL, height = NULL, category = NULL) {
  # Load config
  config <- load_figure_config()

  # Determine category from YAML lookup if not provided
  if (is.null(category)) {
    category <- get_figure_category(name, config)
  }

  # Get output directory for category
  output_dir <- config$figure_categories[[category]]$output_dir

  # ... rest of save logic
}
```

### Phase 3: Update `generate_all_r_figures.R`

1. Create output subdirectories if they don't exist
2. Pass category to save function (or let it auto-detect from YAML)
3. Optionally copy extra-supplementary to archive location

### Phase 4: Archive Script

Create `scripts/archive_extra_supplementary.sh`:
```bash
#!/bin/bash
# Copy extra-supplementary figures to manuscript archive
ARCHIVE_DIR="/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/latent-methods-results/"
cp figures/generated/ggplot2/extra-supplementary/*.png "$ARCHIVE_DIR"
```

---

## Migration Steps

1. Create subdirectories:
   ```bash
   mkdir -p figures/generated/ggplot2/{main,supplementary,extra-supplementary}
   ```

2. Move existing figures to correct subdirectories (one-time migration)

3. Update YAML config with categorization

4. Update save_figure.R to use category-based routing

5. Regenerate all figures with new routing

---

## Validation

After implementation, verify:
- [ ] YAML config has all 21 figures categorized
- [ ] Each figure appears in exactly ONE category
- [ ] Output directories exist and are writable
- [ ] `generate_all_r_figures.R` routes correctly
- [ ] Extra-supplementary archive script works

---

---

## COMPLETED (2026-01-27)

### Actions Taken

1. **Added `figure_categories` section to `configs/VISUALIZATION/figure_layouts.yaml`**
   - Defines main, supplementary, extra_supplementary categories
   - Specifies output directories for each
   - Lists figures in each category

2. **Created directory structure:**
   ```
   figures/generated/ggplot2/
   ├── main/           (7 figures)
   ├── supplementary/  (10 figures)
   └── extra-supplementary/  (4 figures)
   ```

3. **Organized existing figures into correct subdirectories**

### Current Counts

| Category | Count | Status |
|----------|-------|--------|
| Main | 7 | ✅ Organized |
| Supplementary | 10 | ✅ Organized |
| Extra-Supplementary | 4 | ✅ Organized |
| **Total** | **21** | ✅ Complete |

---

## NEXT: Figure Composition

### Specified Combinations

#### Main Figures

| Combined Figure | Layout | Panel A (Left) | Panel B (Right) |
|-----------------|--------|----------------|-----------------|
| `fig_prob_dist_combined` | 1x2 | `fig_prob_dist_by_outcome` | `fig_prob_dist_faceted` |
| `fig_calibration_dca_combined` | 1x2 | Calibration | DCA | *(already exists, verify layout)* |

#### Supplementary Figures

| Combined Figure | Layout | Panel A (Left) | Panel B (Right) |
|-----------------|--------|----------------|-----------------|
| `fig_shap_importance_combined` | 1x2 | `fig_shap_importance_gt` | `fig_shap_importance_ensemble` |
| `fig_vif_combined` | 1x2 | `fig_vif_analysis` | `fig_vif_by_wavelength` |

### Updated Figure Counts After Combination

**Main figures:** 7 → 6 (prob_dist combined into 1)
- `fig_forest_combined` (A: Outlier, B: Imputation) - 2x1
- `fig_calibration_dca_combined` (A: Calibration, B: DCA) - 1x2
- `fig_prob_dist_combined` (A: By Outcome, B: Faceted) - 1x2 **NEW**
- `fig04_variance_decomposition`
- `fig05_shap_beeswarm`
- `fig_shap_importance_multi`

**Supplementary figures:** 10 → 8 (SHAP combined, VIF combined)
- `cd_preprocessing`
- `fig06_specification_curve`
- `fig07_heatmap_preprocessing`
- `fig_R7_featurization_comparison`
- `fig_raincloud_auroc`
- `fig_shap_heatmap`
- `fig_shap_importance_combined` (A: GT, B: Ensemble) - 1x2 **NEW**
- `fig_vif_combined` (A: Analysis, B: By Wavelength) - 1x2 **NEW**

**Extra-supplementary:** 4 (unchanged)

### Implementation Tasks

1. [ ] Update `generate_all_r_figures.R` to create `fig_prob_dist_combined`
2. [ ] Verify `fig_calibration_dca_combined` uses 1x2 layout (not 2x1)
3. [ ] Create `fig_shap_importance_combined` in supplementary
4. [ ] Create `fig_vif_combined` in supplementary
5. [ ] Remove standalone versions that are now combined
6. [ ] Update YAML figure_categories with new combined figure names

---

## NEW FIGURES TO IMPLEMENT (Added 2026-01-27)

### 1. `fig_selective_classification` - Main Figure (NEW)

**Layout**: 1 row × 3 columns (1x3)

**Description**: Shows how selective classification (abstaining on uncertain predictions) affects three key metrics. Models that "know what they don't know" will show better performance at lower retention (higher rejection).

| Panel | X-axis | Y-axis | Description |
|-------|--------|--------|-------------|
| A (Left) | Rejection Ratio (0→1) | AUROC | Discrimination at each retention level |
| B (Center) | Rejection Ratio (0→1) | DCA (Decision Curve Analysis) | Clinical utility at each retention level |
| C (Right) | Rejection Ratio (0→1) | Net Benefit | Net benefit at clinical threshold |

**Data Source**: `outputs/r_data/selective_classification_data.json`
- Already exported with 19 retention levels (0.1 to 1.0)
- Contains 4 main combos: ground_truth, best_ensemble, best_single_fm, traditional
- Metrics: auroc_at_retention, net_benefit_at_retention, scaled_brier_at_retention

**Note**: The exported JSON uses `retention_levels` and `rejection_ratios` arrays. Rejection ratio = 1 - retention.

**Implementation Notes**:
- Use rejection ratio on x-axis (0 = accept all, 1 = reject all)
- Show all 4 main combos as colored lines
- Include confidence bands if available
- Add reference lines (e.g., "treat all" baseline for Net Benefit)

### 2. `fig_roc_rc_combined` - Supplementary Figure (NEW)

**Layout**: 1 row × 2 columns (1x2)

**Description**: Combined ROC and Risk-Coverage (RC) curves for comprehensive model comparison. Shows both discrimination (ROC) and uncertainty calibration (RC).

| Panel | Plot Type | Content |
|-------|-----------|---------|
| A (Left) | ROC Curve | TPR vs FPR for 8 handpicked models + Top-10 aggregate |
| B (Right) | RC Curve | Risk vs Coverage for same models |

**Data Source**: `outputs/r_data/roc_rc_curves_data.json`
- Contains 10 combos including Top-10 Mean aggregate
- ROC: fpr, tpr arrays per combo
- RC: coverage, risk arrays per combo
- AURC metric for ranking

**Models to Include** (from `configs/VISUALIZATION/plot_hyperparam_combos.yaml`):
1. ground_truth
2. best_ensemble
3. best_single_fm
4. traditional
5. moment_full
6. lof_moment
7. timesnet_full
8. units_pipeline
9. simple_baseline (OneClassSVM + MOMENT-zeroshot)
10. Top-10 Mean (aggregate of top performers)

**Implementation Notes**:
- Use consistent colors across both panels (same legend)
- Add diagonal reference line on ROC (random classifier)
- Show AUROC in ROC panel legend
- Show AURC in RC panel legend
- RC curve: X = coverage (fraction retained), Y = risk (error rate)

---

### Updated Figure Counts After New Additions

**Main figures:** 6 → 7 (+1 selective classification)
- `fig_forest_combined` (A: Outlier, B: Imputation) - 2x1
- `fig_calibration_dca_combined` (A: Calibration, B: DCA) - 1x2
- `fig_prob_dist_combined` (A: By Outcome, B: Faceted) - 1x2
- `fig04_variance_decomposition`
- `fig05_shap_beeswarm`
- `fig_shap_importance_multi`
- **`fig_selective_classification` (A: AUROC, B: DCA, C: Net Benefit) - 1x3 NEW**

**Supplementary figures:** 8 → 9 (+1 ROC/RC combined)
- `cd_preprocessing`
- `fig06_specification_curve`
- `fig07_heatmap_preprocessing`
- `fig_R7_featurization_comparison`
- `fig_raincloud_auroc`
- `fig_shap_heatmap`
- `fig_shap_importance_combined` (A: GT, B: Ensemble) - 1x2
- `fig_vif_combined` (A: Analysis, B: By Wavelength) - 1x2
- **`fig_roc_rc_combined` (A: ROC, B: RC) - 1x2 NEW**

**Extra-supplementary:** 4 (unchanged)

**Total:** 21 → 20 (after all combinations and additions)

---

### Data Export Status for New Figures

| Figure | JSON Export | Status |
|--------|-------------|--------|
| `fig_selective_classification` | `outputs/r_data/selective_classification_data.json` | ✅ Already exported |
| `fig_roc_rc_combined` | `outputs/r_data/roc_rc_curves_data.json` | ✅ Already exported |

Both JSON files contain library-agnostic data suitable for ggplot2, matplotlib, or D3.js.

---

## Questions Resolved

| Question | Decision |
|----------|----------|
| Where to store categorization? | `configs/VISUALIZATION/figure_layouts.yaml` |
| Flat vs nested directories? | Nested: `main/`, `supplementary/`, `extra-supplementary/` |
| Auto-detect category? | Yes, lookup from YAML by figure name |
| Archive extra-supplementary? | Yes, copy to manuscript repo |
