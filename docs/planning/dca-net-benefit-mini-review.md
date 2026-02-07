# Decision Curve Analysis and Net Benefit: A Mini-Review for Clinical Prediction Models

**Purpose:** Synthesized literature backing for results.tex and discussion.tex sections
**Date:** 2026-02-01
**Sources:** Vickers 2006, 2019; Van Calster 2018; Harrell (blog); Millard 2025; Alkan 2025; MSKCC resources

---

## Executive Summary

Decision Curve Analysis (DCA) is a method for evaluating the **clinical utility** of prediction models by incorporating the relative costs of false positives and false negatives through the concept of **net benefit**. Unlike discrimination metrics (AUROC) or calibration measures, DCA directly answers: "Would using this model lead to better clinical decisions than alternative strategies?"

---

## 1. Why Net Benefit Matters: Beyond AUROC

### The AUROC Limitation

AUROC measures how well a model **ranks** patients but says nothing about whether using the model would **improve clinical decisions**. Two models with identical AUROC can have vastly different clinical utility depending on:

1. **Calibration** - Are predicted probabilities accurate?
2. **Threshold choice** - At what probability do we intervene?
3. **Clinical context** - What are the costs of errors?

> "Performance measures like sensitivity, specificity, and area under the receiver operating characteristic curve do not account for clinical consequences. It is therefore common for published models to perform well on these measures but have little clinical value." — Vickers et al. (2019)

### The Net Benefit Solution

Net benefit directly incorporates clinical consequences:

```
NB(t) = TP/N - FP/N × t/(1-t)
```

Where:
- `t` = threshold probability (probability at which we intervene)
- `t/(1-t)` = **exchange rate** (how many FPs we tolerate per TP)

**Clinical interpretation:** "At threshold t, net benefit equals the proportion of true positives, minus the proportion of false positives weighted by the odds of the threshold."

---

## 2. The Exchange Rate: Clinical Decision Theory

### What the Threshold Means

The threshold probability reflects the **implicit exchange rate** a clinician accepts:

| Threshold | Exchange Rate | Interpretation |
|-----------|---------------|----------------|
| 5% | 0.053 | Accept 19 unnecessary interventions per true case |
| 10% | 0.111 | Accept 9 unnecessary interventions per true case |
| 15% | 0.176 | Accept 5.7 unnecessary interventions per true case |
| 20% | 0.25 | Accept 4 unnecessary interventions per true case |
| 33% | 0.5 | Accept 2 unnecessary interventions per true case |
| 50% | 1.0 | Equal weight to missing a case vs. unnecessary intervention |

**Key insight from Vickers (2019):**
> "The threshold probability can be thought of as the probability above which a patient would choose to be treated and below which a patient would choose not to be treated."

### Deriving from Decision Theory

The threshold t is derived from:

```
t = Cost(FP) / [Cost(FP) + Cost(FN)]
```

If missing a cancer (FN) is 19× worse than an unnecessary biopsy (FP), then:
```
t = 1 / (1 + 19) = 5%
```

This connects DCA to **formal decision theory** and makes the clinical stakes explicit.

---

## 3. Comparison Strategies: Treat All vs. Treat None

### The Two Default Strategies

Every DCA plot includes two reference strategies:

1. **Treat All:** Give intervention to everyone regardless of prediction
   - Net benefit = (TP + FN)/N - (FP + TN)/N × t/(1-t)
   - Simplifies to: Prevalence - (1-Prevalence) × t/(1-t)
   - Good at **low thresholds** where FP penalty is small

2. **Treat None:** Give intervention to no one
   - Net benefit = 0 (by definition)
   - Good at **high thresholds** where FP penalty is large

### The Model Must Beat Both

A useful model must have net benefit:
- **Greater than Treat All** at thresholds where Treat All is positive
- **Greater than zero** (Treat None) across the clinically relevant range

> "At any threshold, a model is only useful if it has a higher net benefit than both default strategies." — MSKCC Decision Curve Analysis

---

## 4. Common Errors in DCA Presentation (Harrell 2020)

Frank Harrell identified seven common mistakes in DCA reporting:

### Error 1: Unspecified Clinical Decisions

**Problem:** Presenting DCA without stating what clinical decision the model informs.

**Solution:** Explicitly state: "This model is intended to guide the decision to [refer for glaucoma exam / perform biopsy / initiate treatment]."

### Error 2: Excessive Threshold Range

**Problem:** Showing thresholds from 0% to 100% when clinically implausible.

**Solution:** Restrict to the **clinically relevant range**. For glaucoma screening:
- Lower bound: ~3.5% (global prevalence)
- Upper bound: ~20-35% (resource-constrained settings)

### Error 3: Excessive White Space

**Problem:** Y-axis starting at large negative values, compressing the useful range.

**Solution:** Start Y-axis at -0.01 or -0.02, just below where net benefit becomes negative.

### Error 4: Unsmoothed Statistical Noise

**Problem:** Raw DCA curves show sampling noise, creating misleading local peaks.

**Solution:** Apply LOESS or kernel smoothing to represent the "scientific truth" rather than random variation.

### Error 5: Recommending Specific Thresholds

**Problem:** Authors recommend "optimal" thresholds based on where curves peak.

**Solution:** Present the curve across the range; let clinicians choose based on their context.

### Error 6: Claiming Success Despite Negative Findings

**Problem:** Concluding the model is useful when it barely beats Treat All/None.

**Solution:** Honest interpretation—small net benefit differences may not justify implementation costs.

### Error 7: Ignoring Overfitting

**Problem:** Computing DCA on training data without cross-validation adjustment.

**Solution:** Apply the same bootstrap/cross-validation used for other metrics to DCA.

---

## 5. Relationship to Proper Scoring Rules (Millard 2025)

### Brier Score and DCA Connection

Millard (2025) establishes the mathematical relationship between:
- **Decision curves** (net benefit vs. threshold)
- **Cost curves** (expected loss vs. threshold)
- **Brier curves** (Brier score contribution vs. threshold)

Key insight:
> "The expected cost curve is related to the calibration-refinement decomposition of Brier score."

### Practical Implications

1. **Well-calibrated models** have better net benefit across thresholds
2. **Brier score** (a proper scoring rule) provides an overall summary
3. **DCA** provides the **threshold-specific** view needed for decision-making

For our preprocessing comparison:
- Models with similar AUROC may have different net benefit profiles
- Calibration differences (from preprocessing) affect clinical utility
- Report **both** Brier score (overall) and DCA (threshold-specific)

---

## 6. Application to Glaucoma Screening

### Clinical Context

| Parameter | Value | Source |
|-----------|-------|--------|
| Global glaucoma prevalence | 3.54% | Tham et al. 2014 |
| Target sensitivity | 86.2% | Najjar et al. 2023 |
| Target specificity | 82.1% | Najjar et al. 2023 |

### Clinically Relevant Thresholds

| Scenario | Threshold | Rationale |
|----------|-----------|-----------|
| **Population screening** | 5-10% | High sensitivity priority; referral is low-cost |
| **Targeted screening** | 10-20% | Balanced approach for at-risk populations |
| **Resource-limited** | 20-35% | Minimize referrals; higher threshold acceptable |

### What Our Study Shows

By varying **preprocessing methods** while holding the classifier constant (CatBoost):
- Ground truth preprocessing sets the upper bound of achievable net benefit
- Foundation model preprocessing (MOMENT, UniTS) approaches ground truth performance
- Traditional methods (LOF) show lower net benefit at moderate thresholds
- The gap between methods is **largest at moderate thresholds (10-20%)**

---

## 7. Reporting Checklist for DCA (Van Calster 2018)

### Mandatory Elements

- [ ] State the clinical decision being evaluated
- [ ] Specify the clinically relevant threshold range
- [ ] Include Treat All and Treat None reference lines
- [ ] Restrict Y-axis to meaningful range
- [ ] Apply appropriate smoothing
- [ ] Report net benefit values at key thresholds
- [ ] Discuss confidence/uncertainty in estimates

### Optional but Recommended

- [ ] Show interventions avoided per 1000 patients
- [ ] Present net benefit differences vs. reference strategy
- [ ] Include sensitivity analysis for threshold uncertainty
- [ ] Report computational reproducibility (JSON data)

---

## 8. Key Quotes for Manuscript

### For Methods Section

> "Decision curve analysis is a method for evaluating prediction models that incorporates clinical consequences by weighting the benefits of true positives against the harms of false positives at clinically relevant threshold probabilities."

### For Results Section

> "Net benefit was computed across threshold probabilities from 5% to 35%, representing the range of clinical decision points for glaucoma referral. The preprocessing method affected clinical utility primarily at moderate thresholds (10-20%), with foundation model-based preprocessing achieving net benefit within X% of ground truth."

### For Discussion Section

> "Unlike discrimination metrics (AUROC), which measure ranking ability, decision curve analysis directly addresses whether the model would improve clinical decisions. Our findings suggest that preprocessing choices have clinically meaningful effects on net benefit, particularly at moderate threshold probabilities where screening programs typically operate."

---

## 9. Implementation Notes

### R Package: dcurves

```r
library(dcurves)

# Basic DCA
dca(outcome ~ prediction, data = df,
    thresholds = seq(0.05, 0.35, by = 0.01))

# With confidence intervals
dca(outcome ~ prediction, data = df,
    thresholds = seq(0.05, 0.35, by = 0.01),
    bootstraps = 1000)
```

### Python Implementation

```python
def net_benefit(y_true, y_prob, threshold):
    """Compute net benefit at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)

    exchange_rate = threshold / (1 - threshold)
    return tp/n - fp/n * exchange_rate
```

---

## 10. References

1. **Vickers AJ, Elkin EB.** Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.

2. **Vickers AJ, van Calster B, Steyerberg EW.** A simple, step-by-step guide to interpreting decision curve analysis. Diagn Progn Res. 2019;3:18.

3. **Van Calster B, Wynants L, Verbeek JFM, et al.** Reporting and Interpreting Decision Curve Analysis: A Guide for Investigators. Eur Urol. 2018;74(6):796-804.

4. **Harrell FE.** Extended Decision Curve Analysis. fharrell.com/post/edca/ (2020).

5. **Millard SP.** Comparison of decision curves, cost curves, and Brier curves for binary classification. Preprints.org. 2025.

6. **Alkan M, et al.** Artificial Intelligence-Driven Clinical Decision Support Systems. arXiv:2501.09628. 2025.

7. **Steyerberg EW, Vergouwe Y.** Towards better clinical prediction models: seven steps for development and an ABCD for validation. Eur Heart J. 2014;35(29):1925-1931.

8. **Van Calster B, McLernon DJ, van Smeden M, et al.** Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230.

---

## Summary for Authors

**Key points to emphasize in the manuscript:**

1. **DCA complements, not replaces, AUROC and calibration metrics** - we report all three per STRATOS guidelines

2. **The exchange rate interpretation makes clinical stakes explicit** - a 15% threshold means accepting ~6 unnecessary referrals per true case

3. **Preprocessing effects are threshold-dependent** - methods may be equivalent at some thresholds but different at others

4. **We follow Harrell's guidelines** to avoid common DCA presentation errors

5. **Net benefit directly answers the clinical question** - "Would this model improve screening decisions?"

6. **Our figures (fig-repo-41 and fig-repo-42) are designed for expert-level understanding** while remaining accessible to clinical audiences
