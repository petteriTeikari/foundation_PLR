# SHAP Importance Multi-Config Figure Fix

## Status: IN PROGRESS (partial implementation done)

## Problem

`fig_shap_importance_multi.png` has two issues:
1. Does NOT include ground truth (pupil-gt + pupil-gt) - it's at index 5 in data but script takes first 4
2. Missing "Top-10 Mean" aggregate as 6th "model" discussed in manuscript

## Current State (What's Been Done)

### 1. Added `shap_figure_combos` to YAML (DONE)
File: `configs/VISUALIZATION/plot_hyperparam_combos.yaml`

Added new section with 6 configs:
- ground_truth (pupil-gt + pupil-gt) - REQUIRED reference
- best_ensemble (ensemble-LOF-MOMENT- + CSDI)
- ensemble_thresholded (ensembleThresholded- + CSDI)
- moment_saits (MOMENT-gt-finetune + SAITS)
- ensemble_timesnet (ensemble-LOF-MOMENT- + TimesNet)
- top10_mean (__AGGREGATE_TOP10__) - averaged SHAP across all 10 configs

### 2. Updated R Script (DONE)
File: `src/r/figures/fig_shap_importance.R`

Changed Figure 2 section to:
- Load configs from YAML instead of hardcoded first 4
- Pattern-match config names from JSON data
- Compute Top-10 Mean aggregate
- Use YAML-defined colors

### 3. Added Auto-Routing to save_figure.R (DONE)
File: `src/r/figure_system/save_figure.R`

Added:
- `.get_figure_output_dir()` function - routes figures to main/supplementary/extra based on figure_categories
- Updated `save_publication_figure()` to use routing when output_dir is NULL

### 4. Updated figure_categories (DONE)
File: `configs/VISUALIZATION/figure_layouts.yaml`

- Moved `fig05_shap_beeswarm` from main → supplementary (per user)
- Fixed categorization

## What Still Needs Testing

1. Run `Rscript src/r/figures/fig_shap_importance.R` to verify:
   - Ground truth config is found and included
   - Top-10 Mean aggregate is computed correctly
   - Figure saves to correct directory (main/)
   - Legend shows all 6 configs with correct colors

2. Verify all figures now route to correct subdirectories automatically

## Files Modified

| File | Change |
|------|--------|
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Added shap_figure_combos section |
| `src/r/figures/fig_shap_importance.R` | YAML-driven config selection, Top-10 Mean |
| `src/r/figure_system/save_figure.R` | Auto-routing based on figure_categories |
| `configs/VISUALIZATION/figure_layouts.yaml` | Fixed fig05_shap_beeswarm category |

## Remaining Items (NOT STARTED)

### Variance Decomposition Redesign
See: `.claude/planning/variance-decomposition-redesign.md`

- Change from bars to horizontal lollipop chart
- Two-panel 1x2 layout:
  - Panel A: Preprocessing effects (CatBoost fixed)
  - Panel B: Full pipeline including classifier
