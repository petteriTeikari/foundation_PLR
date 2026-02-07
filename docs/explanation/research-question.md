# Research Question

## The Core Question

> **How do preprocessing choices (outlier detection → imputation) affect downstream prediction quality when using handcrafted physiological features for glaucoma screening?**

## What This Means

### We FIX the Classifier

CatBoost is established as the best classifier. We don't compare classifiers—that's not the research question.

### We VARY the Preprocessing

We systematically test combinations of:

- **11 outlier detection methods** (foundation models vs traditional)
- **7 imputation methods** (deep learning vs classical)

### We MEASURE Downstream Effects

Not just AUROC, but ALL STRATOS-compliant metrics:

- Discrimination (AUROC)
- Calibration (slope, intercept, O:E ratio)
- Overall (Brier, Scaled Brier)
- Clinical utility (Net Benefit, DCA)

## What This Is NOT About

| ❌ NOT This | ✅ This Instead |
|------------|-----------------|
| Comparing classifiers | Fix classifier, vary preprocessing |
| Maximizing AUROC | Measure all STRATOS metrics |
| Generic ML benchmarking | Preprocessing sensitivity analysis |
| Foundation model features | Handcrafted physiological features |

## Why Foundation Models for Preprocessing?

Foundation models (MOMENT, UniTS, TimesNet) are generic time-series models trained on large datasets. We ask:

1. **Can they detect artifacts** as well as human experts?
2. **Can they reconstruct** missing signal segments accurately?
3. **Do they reduce error propagation** compared to traditional methods?

## Key Insight: Embeddings Underperform

Foundation model embeddings (used as features) underperform handcrafted features by **9 percentage points** (0.740 vs 0.830 AUROC).

**Why?** Generic embeddings don't capture domain-specific PLR physiology. Handcrafted features encode expert knowledge about glaucoma biomarkers.

## Data Provenance

| Dataset | N | Source |
|---------|---|--------|
| Najjar et al. 2023 | 322 | Original study |
| Our classification subset | 208 | 152 control + 56 glaucoma |
| Our preprocessing subset | 507 | All with ground truth masks |

!!! warning "Do Not Compare Directly"
    Our AUROC (0.913) cannot be compared to Najjar's (0.94) due to different subsets and different goals.
