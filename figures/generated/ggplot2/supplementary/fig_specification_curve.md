# Specification Curve Analysis

## What This Figure Shows

The specification curve (also called "multiverse analysis") displays the results of ALL analytical choices tested in our study, sorted by their outcome metric (AUROC).

**X-axis**: Configuration rank (1 = best, 57 = worst)
**Y-axis**: AUROC with 95% bootstrap confidence intervals
**Colors**: Pipeline type based on ground truth usage
- **Yellow (Both GT)**: Ground truth for both outlier detection AND imputation
- **Orange (Outlier GT)**: Ground truth only for outlier detection
- **Blue (Automated)**: Fully automated preprocessing (no ground truth)

**Dashed yellow line**: Best achievable performance (AUROC = 0.913)

## Scientific Background

The specification curve methodology was introduced by:

> **Simonsohn U, Simmons JP, Nelson LD (2020)**. "Specification curve analysis." *Nature Human Behaviour*, 4(11):1208-1214. DOI: 10.1038/s41562-020-0912-z

Related foundational work:

> **Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W (2016)**. "Increasing Transparency Through a Multiverse Analysis." *Perspectives on Psychological Science*, 11(5):702-712. DOI: 10.1177/1745691616658637

Application to clinical prediction models:

> **Riley RD et al. (2023)**. "Clinical prediction models and the multiverse of madness." *BMC Medicine*. DOI: 10.1186/s12916-023-02849-7

## Why This Matters

### The "Garden of Forking Paths" Problem

When building a predictive model, researchers make many analytical choices:
- Which outlier detection method to use?
- Which imputation method to use?
- Which classifier to use?
- Which hyperparameters?

Each choice creates a "fork" leading to different results. Traditional papers report only the "best" path, hiding the variability.

### What Specification Curves Reveal

1. **Robustness**: If results are consistent across many specifications, findings are robust
2. **Sensitivity**: Which choices matter most for the outcome?
3. **Transparency**: Shows the full range of results, not just cherry-picked best

## Interpretation of Our Results

### Key Finding: Optimal AUROC = 0.913

With optimal preprocessing choices, we achieve AUROC = 0.913, which is:
- Only 0.017 below the Najjar 2023 benchmark (0.93)
- Clinically useful for glaucoma screening

### Robustness: Median AUROC = 0.878

Even with suboptimal choices:
- Median AUROC across all 57 configurations = 0.878
- IQR = 0.028 (relatively narrow spread)
- Most configurations yield clinically acceptable performance

### Ground Truth Effect

The color coding reveals that:
- **Both GT (yellow)** configurations cluster at the top
- **Automated (blue)** configurations span the full range
- Some automated pipelines match or exceed ground truth performance

## Technical Notes

- **N = 57 configurations**: All use CatBoost classifier (fixed per research design)
- **Configurations vary by**: Outlier detection method (11) × Imputation method (7), minus invalid combinations
- **CIs**: 95% bootstrap confidence intervals (1000 iterations)
- **Ranking**: By mean AUROC, descending

## Related Figures

- **fig_variance_decomposition.png**: Shows η² (variance explained) by preprocessing factor
- **fig_cd_diagrams.png**: Statistical comparison of methods (Nemenyi test)
- **fig_heatmap_preprocessing.png**: AUROC by outlier × imputation combination
