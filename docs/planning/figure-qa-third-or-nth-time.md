# Figure QA - Third (or Nth) Time

**Created**: 2026-01-27
**Status**: PLANNING - NOT YET EXECUTED

---

## User Requirements (Verbatim)

> "The following grouping needs to happen: 1) 2-column layout with fig_prob_dist_by_outcome.png on the left, and fig_prob_dist_faceted.png on the right; 2) convert fig_calibration_dca_combined.png also two 2-col and 1-row layout. From the supplementary figures: 3) two-col layout for these two /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/supplementary/fig_shap_importance_ensemble.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/supplementary/fig_shap_importance_gt.png; 4) 2-col layout for these two: /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/supplementary/fig_vif_analysis.png
> /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/figures/generated/ggplot2/supplementary/fig_vif_by_wavelength.png. And let's contunue with some other QA issues when you are done adding these to the plan"

---

## Required Figure Groupings

### Main Figures

| Combined Figure | Layout | Panel A (Left) | Panel B (Right) |
|-----------------|--------|----------------|-----------------|
| `fig_prob_dist_combined` | 1x2 | `fig_prob_dist_by_outcome.png` | `fig_prob_dist_faceted.png` |
| `fig_calibration_dca_combined` | 1x2 | Calibration | DCA |

### Supplementary Figures

| Combined Figure | Layout | Panel A (Left) | Panel B (Right) |
|-----------------|--------|----------------|-----------------|
| `fig_shap_importance_combined` | 1x2 | `fig_shap_importance_gt.png` | `fig_shap_importance_ensemble.png` |
| `fig_vif_combined` | 1x2 | `fig_vif_analysis.png` | `fig_vif_by_wavelength.png` |

---

## New Figures (From Previous Session)

### Main: `fig_selective_classification`

**Layout**: 1 row × 3 columns (1x3)

| Panel | X-axis | Y-axis |
|-------|--------|--------|
| A (Left) | Rejection Ratio (0→1) | AUROC |
| B (Center) | Rejection Ratio (0→1) | DCA |
| C (Right) | Rejection Ratio (0→1) | Net Benefit |

**Data Source**: `outputs/r_data/selective_classification_data.json` ✅ Already exported

### Supplementary: `fig_roc_rc_combined`

**Layout**: 1 row × 2 columns (1x2)

| Panel | Content |
|-------|---------|
| A (Left) | ROC Curve (8 models + Top-10) |
| B (Right) | RC Curve (same models) |

**Data Source**: `outputs/r_data/roc_rc_curves_data.json` ✅ Already exported

---

## Implementation Checklist

### Figure Composition Tasks

- [ ] **1.** Create `fig_prob_dist_combined` (1x2): prob_dist_by_outcome (L) + prob_dist_faceted (R)
- [ ] **2.** Convert `fig_calibration_dca_combined` to 1x2 layout (Calibration L, DCA R)
- [ ] **3.** Create `fig_shap_importance_combined` (1x2): shap_importance_gt (L) + shap_importance_ensemble (R)
- [ ] **4.** Create `fig_vif_combined` (1x2): vif_analysis (L) + vif_by_wavelength (R)
- [ ] **5.** Create `fig_selective_classification` (1x3): AUROC, DCA, Net Benefit vs rejection ratio
- [ ] **6.** Create `fig_roc_rc_combined` (1x2): ROC curves (L) + RC curves (R)

### Cleanup Tasks

- [ ] Remove standalone versions after combining (delete from source folder)
- [ ] Update YAML figure_categories with new combined figure names
- [ ] Verify all combined figures save to correct subdirectory (main vs supplementary)

### QA Verification

- [ ] All main figures in `figures/generated/ggplot2/main/`
- [ ] All supplementary figures in `figures/generated/ggplot2/supplementary/`
- [ ] PNG-only output (no SVG/PDF)
- [ ] Run `pytest tests/test_figure_qa/`

---

## Other QA Issues (To Be Identified)

*User mentioned "continue with some other QA issues" - will be added after grouping tasks are planned.*

---
