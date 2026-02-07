# /manuscript - Navigate to Manuscript Context

Access the LaTeX manuscript and literature review for Foundation PLR.

## Usage

- `/manuscript` - Show manuscript locations and key files
- `/manuscript methods` - Show methods section files
- `/manuscript results` - Show results section files
- `/manuscript lit` - Show literature review sections

## Manuscript Location

```
/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/
├── latent-methods-results/    # LaTeX methods + results
│   ├── methods/               # 11 method section files
│   ├── results/               # 8 result section files + artifacts
│   └── figures/               # Figure TeX includes
├── appendix-literature-review/ # 23 literature review sections
├── planning/                   # Planning documents
├── data/                       # DuckDB databases
└── figures/generated/          # Output figures
```

## Key Files

### Methods (latent-methods-results/methods/)
- `methods-00-overview.tex` - Overview
- `methods-01-study-population.tex` - Subject counts (507/208)
- `methods-03-factorial-design.tex` - Experimental design
- `methods-04-outlier-detection.tex` - 15 outlier methods
- `methods-05-imputation.tex` - 7 imputation methods
- `methods-07-classification.tex` - CatBoost + baselines

### Results (latent-methods-results/results/)
- `results-00-overview.tex` - Summary
- `results-02-outlier-detection.tex` - Outlier performance
- `results-03-imputation.tex` - Imputation MAE
- `results-04-classification.tex` - AUROC tables
- `results-05-factorial-analysis.tex` - Variance decomposition

### Literature Review (appendix-literature-review/)
- `section-01-plr-physiology.tex` - PLR basics
- `section-06-foundation-models-timeseries.tex` - FM background
- `section-13-calibration.tex` - Calibration theory
- `section-17-anomaly-detection.tex` - Outlier methods

## Data Provenance

**Original study**: Najjar et al. 2023, Br J Ophthalmol
- 322 subjects, AUROC 0.94
- We use a SUBSET (507 preprocess, 208 classify)
- DO NOT compare AUROC values directly!

## Research Question Reminder

> How do preprocessing choices (outlier detection → imputation) affect downstream classification?

**NOT about comparing classifiers!**
