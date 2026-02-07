# Statistics (`src/stats/`)

> **You just trained a model. It has AUROC = 0.91. Is it good?**
>
> The honest answer: *You don't have enough information to know.*

AUROC tells you the model can rank patients. It doesn't tell you whether the predicted probabilities are meaningful, whether the model is useful for clinical decisions, or whether individual predictions are reliable.

This module implements the **STRATOS-compliant** statistical framework that answers the questions AUROC can't:

| Question | Metric | Module |
|----------|--------|--------|
| Can it rank patients? | AUROC | `classifier_metrics.py` |
| Are probabilities accurate? | Calibration | `calibration_extended.py` |
| Is it clinically useful? | Net Benefit | `clinical_utility.py` |
| Can I trust individual predictions? | Instability | `pminternal_wrapper.py` |

**CRITICAL**: Report ALL these metrics. AUROC alone is misleading.

[![STRATOS metrics computation flow diagram showing how raw predictions are transformed into the 5 mandatory metric domains — discrimination, calibration, overall performance, clinical utility, and probability distributions — via bootstrap resampling with 1000 iterations, following Van Calster, Ben, Gary S. Collins, Andrew J. Vickers, et al. 2024 arXiv:2412.10288 and Van Calster, Ben, Gary S. Collins, Andrew J. Vickers, et al. 2025 Lancet Digital Health STRATOS Initiative Topic Group 6 guidelines for performance evaluation of predictive AI models to support medical decisions](../../docs/repo-figures/assets/fig-repo-59-stratos-computation-flow.jpg)](../../docs/repo-figures/assets/fig-repo-59-stratos-computation-flow.jpg)

*Figure: STRATOS Metrics — from predictions to publication. Each of the 406 (outlier x imputation x classifier) configurations feeds through the same bootstrap pipeline to produce all 5 metric domains required by [Van Calster et al. 2024](https://arxiv.org/abs/2412.10288) (arXiv) / [2025](https://doi.org/10.1016/S2589-7500(24)00249-0) (Lancet Digital Health). See [computation flow plan](../../docs/repo-figures/figure-plans/fig-repo-59-stratos-computation-flow.md) for details.*

---

## Visual Guide: Understanding the Metrics

This section explains **what each metric means** and **how to interpret it**.

> 💡 **Two ways to use this guide:**
> - **Quick reference**: Use the tables to interpret your numbers
> - **Deep dive**: Read the explanations and follow cross-references

For the academic justification of *why* these metrics matter, see [STRATOS Guidelines](../../docs/tutorials/stratos-metrics.md).

<details open>
<summary><b>1. Discrimination: Can the model rank patients correctly?</b></summary>

### AUROC (Area Under ROC Curve)

**What it measures**: The probability that a randomly chosen positive case is ranked higher than a randomly chosen negative case.

**Intuition**: If you pick one glaucoma patient and one healthy person at random, how often does the model give the glaucoma patient a higher probability?

| AUROC | Interpretation | Clinical Guidance |
|-------|----------------|-------------------|
| 0.5 | Random guessing | Model is useless |
| 0.5-0.7 | Poor | Rarely acceptable clinically |
| 0.7-0.8 | Acceptable | Minimum for clinical use |
| 0.8-0.9 | Excellent | Strong clinical utility |
| > 0.9 | Outstanding | Check for data leakage! |

*Classification per Hosmer & Lemeshow (2000)*

**Code:**
```python
from src.stats.classifier_metrics import compute_auroc
auroc = compute_auroc(y_true, y_prob)  # Returns: 0.913
```

**AUROC is necessary but NOT sufficient** - a model can rank perfectly but give meaningless probabilities. Always check calibration!

</details>

<details open>
<summary><b>2. Calibration: Do probabilities match reality?</b></summary>

![Calibration for clinical prediction models: a well-calibrated model predicts 30% when approximately 30% of patients truly have the disease. Shows calibration slope (ideal: 1.0), intercept (ideal: 0.0), O:E ratio (ideal: 1.0), and a smoothed calibration curve with confidence interval.](../../docs/repo-figures/assets/fig-repo-39-calibration-explained.jpg)

**What it measures**: Do predicted probabilities match actual outcome frequencies?

**Intuition**: If your model says "70% chance of glaucoma" for 100 patients, do approximately 70 of them actually have glaucoma?

### Calibration Slope (Weak Calibration)

| Slope | Interpretation |
|-------|----------------|
| < 1.0 | **Overfitting** - predictions too extreme |
| = 1.0 | Perfect calibration |
| > 1.0 | **Underfitting** - predictions too conservative |

**Code:**
```python
from src.stats.calibration_extended import compute_calibration_slope
slope = compute_calibration_slope(y_true, y_prob)  # Target: ~1.0
```

### Calibration Intercept (Calibration-in-the-Large)

| Intercept | Interpretation |
|-----------|----------------|
| < 0 | **Overconfident** - predictions too high on average |
| = 0 | Perfect |
| > 0 | **Underconfident** - predictions too low on average |

### O:E Ratio (Observed / Expected)

| O:E Ratio | Interpretation |
|-----------|----------------|
| < 1.0 | Model overpredicts events |
| = 1.0 | Perfect |
| > 1.0 | Model underpredicts events |

**Code:**
```python
from src.stats.calibration_extended import (
    compute_calibration_slope,
    compute_calibration_intercept,
    compute_oe_ratio,
)
slope = compute_calibration_slope(y_true, y_prob)
intercept = compute_calibration_intercept(y_true, y_prob)
oe_ratio = compute_oe_ratio(y_true, y_prob)
```

*For academic details, see [STRATOS Guidelines - Calibration](../../docs/tutorials/stratos-metrics.md#2-calibration)*

</details>

<details open>
<summary><b>3. Overall Performance: Combined discrimination + calibration</b></summary>

### Brier Score

**What it measures**: Mean squared error between predictions and outcomes.

```
Brier = (1/N) × Σ(predicted - actual)²
```

| Brier | Interpretation |
|-------|----------------|
| 0.0 | Perfect predictions |
| < 0.25 | Good (better than predicting prevalence) |
| = prevalence × (1-prevalence) | No better than guessing prevalence |

### Scaled Brier (IPA - Index of Prediction Accuracy)

**What it measures**: Improvement over null model (always predicting prevalence).

```
IPA = 1 - (Brier / Brier_null)
```

| IPA | Interpretation |
|-----|----------------|
| < 0 | **Worse than guessing** |
| = 0 | Same as always predicting prevalence |
| > 0 | Better than null model |
| = 1 | Perfect |

**Code:**
```python
from src.stats.scaled_brier import compute_brier_score, compute_scaled_brier
brier = compute_brier_score(y_true, y_prob)
ipa = compute_scaled_brier(y_true, y_prob)
```

</details>

<details open>
<summary><b>4. Clinical Utility: Is the model useful for decisions?</b></summary>

![Decision Curve Analysis (Vickers & Elkin 2006): net benefit plotted against threshold probability. A useful model provides more net benefit than both the 'treat all' and 'treat none' strategies across clinically relevant thresholds.](../../docs/repo-figures/assets/fig-repo-40-net-benefit-dca.jpg)

**What it measures**: Does using this model improve patient outcomes compared to default strategies?

### Net Benefit

The key insight: **False positives and false negatives have different costs.**

At threshold probability p:
- You treat if P(disease) > p
- The threshold reflects your harm-benefit trade-off

| Net Benefit | Interpretation |
|-------------|----------------|
| > "Treat All" line | Model is useful at this threshold |
| > 0 | Better than treating no one |
| < 0 | Harmful - don't use the model here |

**Code:**
```python
from src.stats.clinical_utility import compute_net_benefit, decision_curve_analysis

# Single threshold
nb = compute_net_benefit(y_true, y_prob, threshold=0.15)

# DCA curve
dca = decision_curve_analysis(y_true, y_prob, thresholds=np.arange(0.05, 0.40, 0.01))
```

*For academic details, see [STRATOS Guidelines - Clinical Utility](../../docs/tutorials/stratos-metrics.md#5-clinical-utility)*

</details>

<details open>
<summary><b>5. Model Stability: Are predictions reliable?</b></summary>

![pminternal instability plot (Riley 2023, BMC Medicine): individual predictions vary across 200 bootstrap samples. Each subject is a row; horizontal spread indicates prediction instability. Wide spread means the model is unreliable for that patient.](../../docs/repo-figures/assets/fig-repo-27d-how-to-read-instability-plot.jpg)

**What it measures**: How consistent are individual predictions across bootstrap resamples?

**Intuition**: A patient with prediction 0.65 might have bootstrap predictions ranging from 0.4 to 0.8. That patient's prediction is **unstable** and shouldn't be trusted.

### MAPE (Mean Absolute Prediction Error)

| MAPE | Interpretation |
|------|----------------|
| Low | Stable, reliable predictions |
| High | Predictions vary significantly - flag for review |

**Code:**
```python
from src.stats.pminternal_wrapper import run_pminternal_analysis

# Requires R + pminternal package
results = run_pminternal_analysis(predictions_df, n_bootstrap=200)
```

*Implements Riley et al. 2023 - see [Reading Instability Plots](../../docs/tutorials/reading-plots.md#instability-plots)*

</details>

<details open>
<summary><b>6. Uncertainty Analysis: When should the model abstain?</b></summary>

![Risk-coverage plot for selective classification (Geifman & El-Yaniv 2017): x-axis is coverage (fraction of predictions made), y-axis is risk (error rate). Good uncertainty estimates produce a steep initial drop. Measured by AURC (lower is better).](../../docs/repo-figures/assets/fig-repo-27f-how-to-read-risk-coverage.jpg)

**What it measures**: Can the model identify predictions it's uncertain about?

### AURC (Area Under Risk-Coverage Curve)

| AURC | Interpretation |
|------|----------------|
| Low | Good uncertainty estimates - model knows what it doesn't know |
| High | Poor uncertainty - model is confidently wrong |

### Selective Classification

At coverage c: "Make predictions for the c% most confident cases"

**Code:**
```python
from src.stats.uncertainty_propagation import compute_aurc, selective_classification

aurc = compute_aurc(y_true, y_prob, confidence_scores)
result = selective_classification(y_true, y_prob, coverage=0.8)
```

*See [Reading Risk-Coverage Plots](../../docs/tutorials/reading-plots.md#risk-coverage-aurc)*

</details>

---

## Overview

## Module Structure

```
stats/
├── __init__.py
│
├── classifier_metrics.py        # AUROC, sensitivity, specificity
├── calibration_metrics.py       # Basic calibration
├── calibration_extended.py      # Slope, intercept, O:E ratio
├── classifier_calibration.py    # Calibration integration
├── scaled_brier.py              # Scaled Brier (IPA)
├── clinical_utility.py          # Net Benefit, DCA
│
├── pminternal_wrapper.py        # R pminternal wrapper (Riley 2023)
├── pminternal_analysis.py       # Instability analysis
│
├── bootstrap.py                 # Bootstrap utilities
├── effect_sizes.py              # Effect size calculations
├── fdr_correction.py            # Multiple testing correction
├── variance_decomposition.py    # ANOVA decomposition
│
├── uncertainty_quantification.py  # Per-patient uncertainty
├── uncertainty_propagation.py     # AURC, selective classification
├── decision_uncertainty.py        # Decision uncertainty
│
├── imputation_metrics.py        # MAE, RMSE for imputation
│
├── _types.py                    # Type definitions
├── _validation.py               # Input validation
└── _exceptions.py               # Custom exceptions
```

## STRATOS Core Set (MANDATORY)

| Category | Metric | Function |
|----------|--------|----------|
| **Discrimination** | AUROC | `classifier_metrics.compute_auroc()` |
| **Calibration** | Slope | `calibration_extended.compute_calibration_slope()` |
| **Calibration** | Intercept | `calibration_extended.compute_calibration_intercept()` |
| **Calibration** | O:E ratio | `calibration_extended.compute_oe_ratio()` |
| **Overall** | Brier | `scaled_brier.compute_brier_score()` |
| **Overall** | Scaled Brier | `scaled_brier.compute_scaled_brier()` |
| **Clinical** | Net Benefit | `clinical_utility.compute_net_benefit()` |
| **Clinical** | DCA | `clinical_utility.decision_curve_analysis()` |

## Key Functions

### Discrimination

```python
from src.stats.classifier_metrics import compute_auroc

auroc = compute_auroc(y_true, y_prob)
# Returns: 0.913
```

### Calibration

```python
from src.stats.calibration_extended import (
    compute_calibration_slope,
    compute_calibration_intercept,
    compute_oe_ratio,
)

slope = compute_calibration_slope(y_true, y_prob)
intercept = compute_calibration_intercept(y_true, y_prob)
oe_ratio = compute_oe_ratio(y_true, y_prob)
```

### Clinical Utility

```python
from src.stats.clinical_utility import (
    compute_net_benefit,
    decision_curve_analysis,
)

# Single threshold
nb = compute_net_benefit(y_true, y_prob, threshold=0.15)

# DCA curve across thresholds
dca = decision_curve_analysis(
    y_true, y_prob,
    thresholds=np.arange(0.05, 0.40, 0.01)
)
```

### Scaled Brier (IPA)

```python
from src.stats.scaled_brier import compute_scaled_brier

# IPA = 1 - (Brier / Brier_null)
ipa = compute_scaled_brier(y_true, y_prob)
```

### Model Stability (pminternal)

```python
from src.stats.pminternal_wrapper import run_pminternal_analysis

# Requires R + pminternal package
results = run_pminternal_analysis(
    predictions_df,
    n_bootstrap=200
)
```

## Bootstrap Confidence Intervals

```python
from src.stats.bootstrap import bootstrap_metric

# Bootstrap any metric
result = bootstrap_metric(
    y_true, y_prob,
    metric_fn=compute_auroc,
    n_iterations=1000,
    alpha=0.95
)
# Returns: {'mean': 0.913, 'ci_lo': 0.851, 'ci_hi': 0.955}
```

## Uncertainty Analysis

```python
from src.stats.uncertainty_propagation import (
    compute_aurc,
    selective_classification,
)

# Area Under Risk-Coverage curve
aurc = compute_aurc(y_true, y_prob, confidence_scores)

# Selective classification at coverage level
result = selective_classification(y_true, y_prob, coverage=0.8)
```

## pminternal (Riley 2023)

For prediction instability analysis, we wrap the R `pminternal` package:

```python
from src.stats.pminternal_wrapper import run_pminternal_analysis

# Requires R 4.4+ with pminternal installed
instability_results = run_pminternal_analysis(
    predictions=bootstrap_predictions,  # (n_samples, n_bootstrap)
    y_true=y_true
)
```

**What pminternal provides:**
- Prediction instability plots
- MAPE (Mean Absolute Prediction Error)
- Calibration instability across bootstrap
- Classification instability

## What NOT to Use (Per STRATOS)

| Metric | Problem |
|--------|---------|
| F1 score | Improper, ignores TN |
| AUPRC | Ignores TN |
| pAUROC | No decision-analytic basis |
| Accuracy | Improper for clinical thresholds |

## References

1. **Van Calster et al. 2024** - STRATOS metrics guidelines
2. **Riley et al. 2023** - pminternal for model stability
3. **Vickers & Elkin 2006** - Decision curve analysis

## See Also

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Pipeline overview
- [src/viz/README.md](../viz/README.md) - Visualization of metrics
- [STRATOS_CALIBRATION_IMPLEMENTATION.md](../../docs/STRATOS_CALIBRATION_IMPLEMENTATION.md) - Implementation notes
