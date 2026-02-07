# Figure Batch QA Update Plan
## Ensuring All Figures Use Academic Mode (No Infographics)

**Created:** 2026-01-27
**Status:** IN PROGRESS - Consolidating redundant figures
**Goal:** Audit and update ALL figure generation scripts to produce publication-ready academic figures without infographic annotations (titles, subtitles, captions embedded in figures).

---

## 🚨 CONSOLIDATION PRINCIPLE

**ONE figure per concept. No duplicates. No demos. No variants.**

### What We Need
| Figure | Panels | Source | Status |
|--------|--------|--------|--------|
| `fig_forest_combined.png` | A: Outlier, B: Imputation | `generate_all_r_figures.R` | ✅ DONE |
| `fig_calibration_dca_combined.png` | A: Calibration, B: DCA | `generate_all_r_figures.R` | ✅ DONE |
| ... other single-concept figures ... | N/A | Individual scripts | ✅ DONE |

### What We DON'T Need
- ❌ `fig02_forest_outlier.png` (standalone) - REDUNDANT (deleted)
- ❌ `fig03_forest_imputation.png` (standalone) - REDUNDANT (deleted)
- ❌ `fig_forest_combined_from_config.png` - REDUNDANT (deleted)
- ❌ `fig_forest_outlier_infographic.png` - REDUNDANT (deleted)
- ❌ `fig_calibration_stratos.png` (standalone) - REDUNDANT (now in combined)
- ❌ `fig_dca_stratos.png` (standalone) - REDUNDANT (now in combined)
- ❌ Any "demo" or "test" figures

---

## Executive Summary

The codebase has **60+ figure generation scripts** across two systems:
- **R system** (`src/r/figures/`): 18 figure scripts + 4 system files
- **Python system** (`src/viz/`): 38 visualization modules

**Problem:** Legacy scripts may have hardcoded titles/captions that should be removed for journal submission. The new `figure_factory.R` system supports `infographic=FALSE` by default, but old scripts haven't been migrated.

**Solution:** Audit all scripts, migrate to composable system where appropriate, ensure all figures are in academic mode.

---

## Current State Analysis

### R Figures Inventory

| Script | Has Hardcoded Title/Caption | Uses New System | Status |
|--------|---------------------------|-----------------|--------|
| `fig02_forest_outlier.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig03_forest_imputation.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig04_variance_decomposition.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig05_shap_beeswarm.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig06_specification_curve.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig07_heatmap_preprocessing.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_calibration_stratos.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_dca_stratos.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_M3_factorial_matrix.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_R7_featurization_comparison.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_R8_fm_dashboard.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_prob_dist_by_outcome.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_raincloud_auroc.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_shap_importance.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_shap_gt_vs_ensemble.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_shap_heatmap.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `fig_vif_analysis.R` | ❌ REMOVED | ✅ YES | ✅ DONE |
| `cd_preprocessing.R` | ❌ REMOVED | ✅ YES | ✅ DONE |

### Python Figures Inventory

| Script | Has Hardcoded Title/Caption | Priority |
|--------|---------------------------|----------|
| `featurization_comparison.py` | ? CHECK | MEDIUM |
| `foundation_model_dashboard.py` | ? CHECK | MEDIUM |
| `factorial_matrix.py` | ? CHECK | LOW |
| `utility_matrix.py` | ? CHECK | LOW |
| `cd_diagram_preprocessing.py` | ? CHECK | LOW |
| `retained_metric.py` | ? CHECK | HIGH (BROKEN) |
| `calibration_plot.py` | ? CHECK | HIGH |
| `dca_plot.py` | ? CHECK | HIGH |
| `forest_plot.py` | ? CHECK | HIGH |
| `generate_instability_figures.py` | ? CHECK | MEDIUM |

---

## Action Plan

### Phase 1: Audit (Automated)

**Goal:** Scan all figure scripts for `title`, `subtitle`, `caption`, `labs(` patterns

```bash
# R scripts audit
grep -rn "labs\s*(" src/r/figures/ | grep -E "title|subtitle|caption"
grep -rn "ggtitle\s*(" src/r/figures/

# Python scripts audit
grep -rn "\.set_title\|ax\.title\|fig\.suptitle" src/viz/
grep -rn "plt\.title\|plt\.suptitle" src/viz/
```

**Output:** List of files with hardcoded annotations

### Phase 2: Migration Strategy

For each script, decide:

| Strategy | When to Use | Effort |
|----------|-------------|--------|
| **Full Migration** | Forest plots, frequently reused | HIGH |
| **Quick Fix** | One-off scripts, rarely regenerated | LOW |
| **Skip** | Already academic mode | NONE |

**Full Migration** = Rewrite using `figure_factory.R` / `compose_figures.R`
**Quick Fix** = Remove/comment out hardcoded labels, keep script structure

### Phase 3: Implementation

#### 3.1 High Priority Scripts (Forest Plots)

- [ ] `fig02_forest_outlier.R` - Migrate to `create_forest_outlier(infographic=FALSE)`
- [ ] `fig03_forest_imputation.R` - Migrate to `create_forest_imputation(infographic=FALSE)`
- [ ] Combined figure already done via `demo_usage.R`

#### 3.2 STRATOS Main Figures

- [ ] `fig_calibration_stratos.R` - Check/remove titles
- [ ] `fig_dca_stratos.R` - Check/remove titles
- [ ] `fig_prob_dist_by_outcome.R` - Check/remove titles

#### 3.3 Supplementary Figures

- [ ] `fig04_variance_decomposition.R`
- [ ] `fig05_shap_beeswarm.R`
- [ ] `fig06_specification_curve.R`
- [ ] `fig07_heatmap_preprocessing.R`
- [ ] `fig_raincloud_auroc.R`
- [ ] SHAP figures
- [ ] VIF figures

#### 3.4 Python Figures

- [ ] Audit all `src/viz/*.py` for hardcoded titles
- [ ] Apply `setup_style()` consistently
- [ ] Remove `plt.title()` / `ax.set_title()` calls where present

### Phase 4: Batch Generation

**Command to generate ALL figures:**

```bash
# R figures (run each script)
for script in src/r/figures/fig*.R; do
    echo "Running: $script"
    Rscript "$script" 2>&1 | tee -a figures_generation.log
done

# Python figures
python src/viz/generate_all_figures.py --all 2>&1 | tee -a figures_generation.log
```

### Phase 5: QA Validation

```bash
# Run figure QA tests
pytest tests/test_figure_qa/ -v

# Visual inspection checklist
# - No titles embedded in figures
# - No captions embedded in figures
# - Consistent styling
# - Colorblind-safe
# - Proper DPI (300+)
```

---

## Success Criteria

1. **ZERO hardcoded titles/captions** in any generated figure
2. **All main figures** generated successfully
3. **All figures** pass QA tests
4. **Consistent styling** across R and Python figures
5. **PNG-only output** (no PDF per user requirement)

---

## Files to Create/Modify

### New Files
- [ ] `src/r/figures/generate_all_r_figures.R` - Batch runner for R figures

### Modified Files
- [ ] `src/r/figures/fig02_forest_outlier.R` - Remove hardcoded labels
- [ ] `src/r/figures/fig03_forest_imputation.R` - Remove hardcoded labels
- [ ] (others based on audit)

### Config Updates
- [ ] Ensure `configs/VISUALIZATION/figure_layouts.yaml` has all figures registered

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking working figures | HIGH | Test each change individually |
| Missing figures | MEDIUM | Cross-reference with registry |
| Style inconsistency | LOW | Use shared theme functions |
| Data not available | HIGH | Run extraction first |

---

## Timeline

| Phase | Estimated Effort |
|-------|-----------------|
| Audit | 1 task |
| Migration decisions | 1 task |
| Implementation | 3-5 tasks |
| Batch generation | 1 task |
| QA validation | 1 task |

---

## Reviewer Agents Required

1. **Audit Agent** - Scan all scripts for hardcoded annotations
2. **R Code Reviewer** - Validate R script updates follow best practices
3. **Figure QA Agent** - Run QA tests and visual inspection
4. **Integration Reviewer** - Ensure all figures work together

---

## Questions for User

1. Should deprecated figures (e.g., `fig_retained_auroc`) be deleted or kept for reference?
2. For Python figures, should we add `infographic` parameter support or just ensure academic mode?
3. Which figures are mandatory for the current submission vs. nice-to-have?

---

## Appendix: Figure Registry Cross-Reference

See: `configs/VISUALIZATION/figure_registry.yaml` for complete list of registered figures and their requirements.
