# Classification

Stage 4 of the pipeline: Training and evaluating classifiers.

## Default Classifier: CatBoost

!!! note "Fixed Classifier"
    CatBoost is the default and recommended classifier. The research question is about **preprocessing effects**, not classifier comparison.

### Why CatBoost?

| Property | Value |
|----------|-------|
| Mean AUROC | 0.878 |
| Best AUROC | 0.913 |
| Handles categorical | Yes |
| GPU support | Yes |
| Overfitting protection | Built-in |

## Bootstrap Evaluation

All results use bootstrap validation:

```yaml
CLS_EVALUATION:
  BOOTSTRAP:
    n_iterations: 1000
    alpha_CI: 0.95
```

This provides:

- Robust confidence intervals
- Per-iteration metrics for statistical tests
- Subject-wise stability analysis

## STRATOS Metrics

Classification is evaluated with all STRATOS-compliant metrics:

### Discrimination

- **AUROC**: Area Under ROC Curve (95% CI)

### Calibration

- **Calibration slope**: Should be ~1.0
- **Calibration intercept**: Should be ~0.0
- **O:E ratio**: Observed/Expected ratio

### Overall Performance

- **Brier score**: Proper scoring rule
- **Scaled Brier (IPA)**: Interpretable proportion

### Clinical Utility

- **Net Benefit**: At clinical threshold
- **DCA curves**: Decision Curve Analysis

## Running Classification

```bash
# Default (CatBoost with best preprocessing)
python -m src.classification.flow_classification

# With specific preprocessing
python -m src.classification.flow_classification \
    outlier_method=MOMENT-gt-finetune \
    imputation_method=SAITS
```

## API Reference

::: src.classification.flow_classification
    options:
      show_root_heading: true
