# STRATOS-Compliant Calibration Metrics Implementation

## Overview

This document describes the implementation of calibration slope and intercept metrics
following STRATOS/TRIPOD-AI guidelines for clinical prediction models.

**Date**: 2026-01-22
**Author**: Claude Code implementation based on Van Calster et al. recommendations

## References

Key publications informing this implementation:

1. Van Calster, B., McLernon, D.J., van Smeden, M., et al. (2019).
   **"Calibration: the Achilles heel of predictive analytics."**
   BMC Medicine, 17, 230.
   [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996/)

2. Van Calster, B., Nieboer, D., Vergouwe, Y., et al. (2016).
   **"A calibration hierarchy for risk models was defined: from utopia to empirical data."**
   Journal of Clinical Epidemiology.

3. pmcalibration R package documentation.
   [GitHub](https://stephenrho.github.io/pmcalibration/)
   [CRAN](https://cran.r-project.org/web/packages/pmcalibration/index.html)

4. CalibrationCurves R package.
   [CRAN](https://cran.r-project.org/web/packages/CalibrationCurves/index.html)

## Calibration Metrics

### 1. Calibration Slope (ζ)

**Definition**: The calibration slope evaluates the spread of estimated risks and has a target value of 1.

- Slope < 1: Risk predictions are too extreme (overfitting)
- Slope > 1: Risk predictions are not extreme enough (underfitting)
- Slope = 1: Perfect calibration

**Model**:
```
logit(E[Y|π̂]) = α + ζ × logit(π̂)
```

Where:
- Y is the binary outcome
- π̂ is the predicted probability
- ζ is the calibration slope
- α is the calibration intercept

### 2. Calibration Intercept (α) / Calibration-in-the-Large

**Definition**: The calibration intercept assesses overall under/overestimation of risks.

- Intercept < 0: Average overestimation of risk
- Intercept > 0: Average underestimation of risk
- Intercept = 0: Perfect calibration-in-the-large

**Two approaches exist**:

#### Approach A: Single GLM (rms::val.prob style)
```R
model <- glm(y ~ logit_p, family = binomial())
intercept <- coef(model)["(Intercept)"]
slope <- coef(model)["logit_p"]
```

#### Approach B: Separate GLMs (pmcalibration/Van Calster 2016)
```R
# Intercept: slope fixed to 1 (offset)
model_int <- glm(y ~ offset(logit_p), family = binomial())
intercept <- coef(model_int)["(Intercept)"]

# Slope: separate model
model_slope <- glm(y ~ logit_p, family = binomial())
slope <- coef(model_slope)["logit_p"]
```

### 3. O:E Ratio (Observed/Expected)

**Definition**: Ratio of observed events to expected (predicted) events.

```
O:E = Σy_true / Σπ̂
```

- O:E = 1: Perfect calibration-in-the-large
- O:E > 1: Underestimation (model predicts fewer events than observed)
- O:E < 1: Overestimation (model predicts more events than observed)

## Implementation Choice

We implement **Approach A** (single GLM) for the following reasons:

1. **Consistency with sklearn**: Our Python fallback uses sklearn LogisticRegression which fits both parameters in a single model
2. **Simplicity**: Single model provides both slope and intercept with proper covariance
3. **Established practice**: Used by rms::val.prob which is widely cited
4. **Comparable results**: When calibration is reasonable, both approaches give similar results

## Python Implementation

Located in `src/stats/pminternal_wrapper.py`:

```python
# Pure Python fallback
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

eps = 1e-10
p_clipped = np.clip(y_prob, eps, 1 - eps)
lp = logit(p_clipped)

lr = LogisticRegression(penalty=None, solver='lbfgs')
lr.fit(lp.reshape(-1, 1), y_true)

slope = lr.coef_[0, 0]
intercept = lr.intercept_[0]
```

## R Implementation

When R and pROC are available, we use:

```R
# Logit transform
eps <- 1e-10
p_clipped <- pmax(pmin(y_prob, 1-eps), eps)
lp <- log(p_clipped / (1 - p_clipped))

# Fit logistic calibration model
model <- glm(y_true ~ lp, family=binomial())
slope <- coef(model)["lp"]
intercept <- coef(model)["(Intercept)"]
```

## Confidence Intervals

### Via Standard Errors (Normal Approximation)
```
CI_slope = slope ± z × SE_slope
CI_intercept = intercept ± z × SE_intercept
```

### Via Bootstrap (for instability analysis)
The `instability_analysis()` function computes bootstrap distribution of calibration slope,
providing empirical confidence intervals and the instability index (CV of slope).

## Model Instability Analysis

Following Riley (2023), we compute:

- **Instability Index**: Coefficient of variation (CV) of calibration slope across bootstrap samples
- **Stability Rating**:
  - CV < 0.10: "stable"
  - CV 0.10-0.20: "moderate"
  - CV > 0.20: "unstable"

## Interpretation Guidelines (STRATOS)

| Metric | Ideal | Warning | Poor |
|--------|-------|---------|------|
| Calibration Slope | 0.9-1.1 | 0.7-0.9 or 1.1-1.3 | <0.7 or >1.3 |
| Calibration Intercept | -0.1 to 0.1 | -0.2 to 0.2 | <-0.2 or >0.2 |
| O:E Ratio | 0.9-1.1 | 0.8-1.2 | <0.8 or >1.2 |

## Limitations

1. **Slope/intercept are summary statistics**: They reduce calibration to 2 numbers,
   which may miss local miscalibration. Always examine calibration curves as well.

2. **Assumes logistic model**: The calibration model assumes logistic relationship,
   which may not hold for highly miscalibrated predictions.

3. **Sensitive to extreme predictions**: Predictions near 0 or 1 can strongly influence
   the logit transformation and fitted slope.

## Validation

Our implementation is validated against:
- `rms::val.prob` R function
- `pmcalibration` R package
- `CalibrationCurves` R package

Test suite: `tests/unit/test_pminternal_wrapper.py`
