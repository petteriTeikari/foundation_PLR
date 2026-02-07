# STRATOS Metrics

## What is STRATOS?

STRATOS (STRengthening Analytical Thinking for Observational Studies) is an initiative providing guidance for predictive AI model evaluation in medicine.

**Reference:** Van Calster B, Collins GS, Vickers AJ, et al. "Performance evaluation of predictive AI models to support medical decisions: Overview and guidance." STRATOS Initiative Topic Group 6.

## The STRATOS Core Set

These measures MUST be reported for clinical prediction models:

### 1. Discrimination

**Question:** Does the model rank patients correctly?

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **AUROC** | Area Under ROC Curve | 0.5 = random, 1.0 = perfect |

!!! note
    AUROC is semi-proper—it measures ranking, not probability accuracy.

### 2. Calibration

**Question:** Do predicted probabilities match observed frequencies?

| Metric | Description | Target |
|--------|-------------|--------|
| **Calibration plot** | Visual check | Points on diagonal |
| **Calibration slope** | Weak calibration | Close to 1.0 |
| **Calibration intercept** | Mean calibration | Close to 0.0 |
| **O:E ratio** | Observed/Expected | Close to 1.0 |

### 3. Overall Performance

**Question:** Combined discrimination + calibration?

| Metric | Description | Notes |
|--------|-------------|-------|
| **Brier score** | Mean squared error | Lower is better |
| **Scaled Brier (IPA)** | Interpretable Brier | 0-1 scale, higher better |

### 4. Clinical Utility

**Question:** Is the model useful for decisions?

| Metric | Description | Notes |
|--------|-------------|-------|
| **Net Benefit** | Clinical value | At threshold(s) |
| **DCA curves** | Decision Curve Analysis | Across thresholds |

## What NOT to Use

STRATOS explicitly recommends against:

| ❌ Metric | Problem |
|----------|---------|
| F1 score | Improper + ignores TN |
| AUPRC | Ignores TN, unclear interpretation |
| pAUROC | No decision-analytic basis |
| Accuracy | Improper for clinical thresholds ≠ 0.5 |

## Why This Matters

> "Selecting appropriate performance measures is essential for predictive AI models that are developed to be used in medical practice, because **poorly performing models may harm patients and lead to increased costs**."
> — Van Calster et al. 2024

## Implementation in Foundation PLR

All experiments automatically compute STRATOS metrics:

```python
# Automatic computation via bootstrap_evaluation
metrics = {
    "auroc": ...,
    "brier": ...,
    "calibration_slope": ...,
    "calibration_intercept": ...,
    "o_e_ratio": ...,
    "net_benefit_5pct": ...,
    "net_benefit_10pct": ...,
    "net_benefit_20pct": ...,
}
```

See the [API Reference](../api-reference/stats.md) for implementation details.
