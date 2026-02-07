# RESEARCH QUESTION - READ EVERY SESSION

This is the **high-level vision** that applies to ALL work in this repo.

## The Question

> **How do preprocessing choices (outlier detection → imputation) affect downstream classification performance when using handcrafted physiological features for glaucoma screening?**

## What This Means

1. **FIX the classifier** (CatBoost) - it's already established as best
2. **VARY the preprocessing** (outlier method × imputation method)
3. **MEASURE downstream effect** on classification AUROC
4. **COMPARE** foundation models vs traditional methods for preprocessing

## What This Is NOT About

- Comparing classifiers (CatBoost vs XGBoost) - **IRRELEVANT**
- Maximizing AUROC - **NOT THE GOAL**
- Generic ML benchmarking - **MISS THE POINT**

## Data Provenance

**Source**: Najjar et al. 2023, Br J Ophthalmol (DOI: 10.1136/bjophthalmol-2021-319938)

| Dataset | N | AUROC | Notes |
|---------|---|-------|-------|
| Najjar original | 322 | 0.94 | Full Singapore dataset |
| Our subset (classify) | 208 | 0.913 | 152 control + 56 glaucoma |
| Our subset (preprocess) | 507 | N/A | All with ground truth masks |

**DO NOT compare our AUROC directly to Najjar's!** Different subset, different goal.

## Key Findings (What We've Shown)

| Finding | Value | Meaning |
|---------|-------|---------|
| Best AUROC | 0.913 | With ground truth + CatBoost |
| Preprocessing effect | η²=0.15 | Matters, but less than classifier |
| Handcrafted vs Embeddings | 0.830 vs 0.740 | **Embeddings underperform by 9pp!** |
| FM for preprocessing | Competitive | FMs useful for outlier/imputation |

## Sister Repos

- **Manuscript**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/`
- **Literature**: `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/appendix-literature-review/`
