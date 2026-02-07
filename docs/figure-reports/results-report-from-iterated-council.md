# Results Section Report from Iterated LLM Council

**Date**: 2026-01-30 (Updated)
**Source**: Iterated LLM Council execution (4 iterations) + Data Verification
**Manuscript**: PLR Foundation Model for TVST

---

## Summary

The Iterated LLM Council reviewed and refined the Results section across 4 iterations. This report documents the key metrics, figures, and pending items identified during the review process.

**IMPORTANT UPDATE (2026-01-30)**: All metrics have been verified against the source database and JSON files. Several calibration values were corrected.

---

## Key Verified Numbers in Results Section

### Overall Pipeline Performance (VERIFIED 2026-01-30)

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| Total configurations | 316 | DuckDB essential_metrics | ✅ Verified |
| Best AUROC | **0.9130** | ensemble + CSDI + CatBoost | ✅ Verified |
| Ground Truth AUROC | **0.9110** | pupil-gt + pupil-gt + CatBoost | ✅ Verified |
| Traditional AUROC | **0.8599** | LOF + SAITS + TabPFN | ✅ Verified |
| Best Single FM AUROC | **0.9099** | MOMENT-gt-finetune + SAITS + CatBoost | ✅ Verified |

### Calibration Assessment (CORRECTED 2026-01-30)

**⚠️ CRITICAL CORRECTION**: Previous reports cited incorrect calibration slopes. All values have been re-verified from `calibration_data.json`.

| Configuration | Cal. Slope | O:E Ratio | Brier | IPA | Status |
|---------------|------------|-----------|-------|-----|--------|
| Ground Truth | **0.52** | 0.82 | 0.135 | 0.315 | ✅ Verified |
| Best Ensemble | **0.30** | 0.86 | 0.102 | 0.481 | ✅ Verified |
| Best Single FM | **0.65** | 0.73 | 0.155 | 0.214 | ✅ Verified |
| Traditional | **0.07** | 1.34 | 0.122 | 0.381 | ✅ Verified |

**Interpretation**: All methods show overfitting (calibration slope < 1), meaning predictions are more extreme than warranted. Traditional (LOF + SAITS) shows the most severe overfitting (slope = 0.07).

### Decision Curve Analysis (VERIFIED 2026-01-30)

| Configuration | NB @5% | NB @10% | NB @15% | NB @20% |
|---------------|--------|---------|---------|---------|
| Ground Truth | 0.231 | **0.189** | 0.141 | 0.127 |
| Best Ensemble | 0.231 | **0.189** | 0.170 | 0.175 |
| Best Single FM | 0.231 | **0.189** | 0.141 | 0.087 |
| Traditional | 0.198 | **0.182** | 0.163 | 0.159 |

**Prevalence**: 26.9% (56 events / 208 subjects)

### Model Instability (NEW - 2026-01-30)

**3-Panel Instability Analysis** (Riley 2023 style):

| Configuration | 95% CI Width | Interpretation |
|---------------|--------------|----------------|
| Ground Truth | **0.089** | Most stable |
| Best Ensemble | **0.109** | Comparable stability |
| Traditional | **0.272** | **3× higher instability** |

---

## Figures - UPDATED SOURCES

### Main Figures (for results.tex)

**Source**: `figures/generated/ggplot2/main/`
**Report**: `figures/generated/ggplot2/main/main-figure-report.md`

| Figure | Filename | Description | Status |
|--------|----------|-------------|--------|
| 1 | `fig_calibration_dca_combined.png` | Calibration + DCA combined | ✅ Ready |
| 2 | `fig_forest_combined.png` | Forest plot combined | ✅ Ready |
| 3 | `fig_multi_metric_raincloud.png` | Multi-metric raincloud | ✅ Ready |
| 4 | `fig_prob_dist_combined.png` | Probability distribution | ✅ Ready |
| 5 | `fig_shap_importance_combined.png` | SHAP importance | ✅ Ready |
| 6 | `fig_variance_decomposition.png` | Variance decomposition | ✅ Ready |

### Supplementary Figures (for supplementary.tex)

**Source**: `figures/generated/ggplot2/supplementary/`
**Report**: `figures/generated/ggplot2/supplementary/supplementary-figure-report.md`

| Figure | Filename | Description | Status |
|--------|----------|-------------|--------|
| S1 | `fig_roc_rc_combined.png` | ROC + Risk-Coverage | ✅ Ready |
| S2 | `fig_selective_classification.png` | Selective classification | ✅ Ready |
| S3 | `fig_cd_diagrams.png` | Critical difference diagrams | ✅ Ready |
| S4 | `fig_cd_preprocessing.png` | Preprocessing method ranks | ✅ Ready |
| S5 | `fig_raincloud_auroc.png` | AUROC distribution raincloud | ✅ Ready |
| S6 | `fig_prob_dist_faceted.png` | Probability faceted (all 4) | ✅ Ready |
| S7 | `fig_shap_importance_multi.png` | Multi-config SHAP | ✅ Ready |
| S8 | `fig_specification_curve.png` | Specification curve | ✅ Ready |
| S9 | `fig_instability_combined.png` | **3-panel instability (NEW)** | ✅ Ready |

---

## COMPLETED Verification Tasks

- [x] **Cross-check AUROC values** - Verified against DuckDB (2026-01-30)
- [x] **Verify calibration slopes** - CORRECTED from 1.625/3.665/0.812 to 0.52/0.30/0.65/0.07
- [x] **Verify DCA net benefit values** - Verified against dca_data.json
- [x] **Add instability figure** - New 3-panel figure created (2026-01-30)
- [x] **Update figure reports** - main-figure-report.md and supplementary-figure-report.md updated

---

## REMAINING Action Items

### For Manuscript Integration

1. [ ] **Copy main figures** from `ggplot2/main/` to manuscript `figures/generated/`
2. [ ] **Update \includegraphics paths** in results.tex
3. [ ] **Update calibration slope values** in results.tex (CRITICAL - values were wrong!)
4. [ ] **Create supplementary.tex** with figures from `ggplot2/supplementary/`
5. [ ] **Add instability analysis paragraph** referencing Figure S9

### Figure Mapping (Old → New)

| Current Reference | Replace With | Notes |
|-------------------|--------------|-------|
| Calibration slopes 1.625, 3.665, 0.812 | **0.52, 0.30, 0.65, 0.07** | CRITICAL FIX |
| Missing | `fig_instability_combined.png` | New 3-panel figure |

---

## Statistical Concerns (Council Review)

1. **Winner's curse**: Top-10 mean AUROC should be reported alongside best
2. **Multiple comparisons**: 316 configurations, no formal correction applied
3. **EPV**: 56 events, EPV ranges from 0.055 (raw embeddings) to 7.0 (handcrafted)
4. **Calibration interpretation**: All methods overfitted - this is a new finding from corrected data

---

## Related Files

- **Results section**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/results.tex`
- **Main figure report**: `figures/generated/ggplot2/main/main-figure-report.md`
- **Supplementary figure report**: `figures/generated/ggplot2/supplementary/supplementary-figure-report.md`
- **Database**: `data/public/foundation_plr_results_stratos.db`
- **Calibration data**: `data/r_data/calibration_data.json`
- **DCA data**: `data/r_data/dca_data.json`
- **Instability data**: `data/r_data/pminternal_bootstrap_predictions.json`

### Council Review Files

- `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/iterated-llm-council/04-improvements-from-comments/`

---

## Council Review History

| Iteration | R3 Biostatistics | Key Finding |
|-----------|------------------|-------------|
| 1 | MINOR (6/10) | EPV concerns, calibration inconsistency |
| 2 | MINOR (8/10) | Winner's curse added, age confounding acknowledged |
| 3 | MINOR (8/10) | PCA verification outstanding |
| 4 | ACCEPT (8.5/10) | EPV caveat added, but verification still needed |

---

*Report updated: 2026-01-30*
*Data verification completed: 2026-01-30*
*Calibration values CORRECTED: 2026-01-30*
