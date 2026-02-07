# Figure Plan: DCA Threshold Sensitivity Analysis (fig-repo-42)

**Status:** Planning
**Priority:** High
**Type:** Comparative Analysis Figure
**Output:** Supplementary Material (potentially main)

---

## Overview

An expert-level figure showing **how preprocessing choices affect clinical utility across different threshold preferences**. This goes beyond basic DCA by:

1. Showing **uncertainty bands** around net benefit curves
2. Identifying **threshold ranges where models differ significantly**
3. Highlighting **clinical decision points** specific to glaucoma screening

## The Clinical Question

> "Does my preprocessing choice matter clinically, or do all methods perform similarly at the thresholds relevant to glaucoma screening?"

## Visual Concept: "The Decision Landscape"

### Layout (2-row × 2-column grid)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              PREPROCESSING EFFECTS ON CLINICAL UTILITY                   │
├────────────────────────────────┬────────────────────────────────────────┤
│                                │                                        │
│   PANEL A:                     │   PANEL B:                             │
│   Full DCA with CI bands       │   Net Benefit Difference               │
│   (4 preprocessing methods)    │   (vs Ground Truth)                    │
│                                │                                        │
│   • Ground Truth (reference)   │   Shows Δ(NB) with CI                  │
│   • Best Ensemble              │   Horizontal line at 0                 │
│   • Best Single FM             │   Shaded regions where                 │
│   • Traditional (LOF)          │   difference is significant            │
│                                │                                        │
├────────────────────────────────┼────────────────────────────────────────┤
│                                │                                        │
│   PANEL C:                     │   PANEL D:                             │
│   Clinical Threshold Zones     │   Summary: Interventions               │
│                                │   Avoided per 1000 Patients            │
│   Glaucoma-specific:           │                                        │
│   • Conservative (5%)          │   Bar chart showing net                │
│   • Moderate (10%)             │   interventions at each                │
│   • Aggressive (20%)           │   threshold for each method            │
│                                │                                        │
└────────────────────────────────┴────────────────────────────────────────┘
```

### Panel A: DCA with Bootstrap Confidence Intervals

**What it shows:**
- X-axis: Threshold probability (3.5% to 35%)
- Y-axis: Net Benefit (clinical utility)
- Lines: 4 preprocessing pipelines
- Shaded bands: 95% bootstrap CI for each curve
- Reference lines: "Treat All" and "Treat None" (dashed)

**Critical design elements:**
1. **Restricted range**: Only show clinically plausible thresholds (not 0-100%)
2. **CI overlap analysis**: Where bands overlap = methods are equivalent
3. **Smoothed curves**: Apply loess smoothing per Harrell's recommendation

**Glaucoma-specific threshold range:**
- Lower bound: ~3.5% (global glaucoma prevalence per Tham 2014)
- Upper bound: ~20-35% (depends on clinical context)

### Panel B: Net Benefit Difference Plot

**What it shows:**
- X-axis: Threshold probability
- Y-axis: Δ(Net Benefit) = NB(method) - NB(Ground Truth)
- Horizontal reference line at 0
- Color-coded by whether difference is statistically significant

**Interpretation:**
- Points above 0: Method is BETTER than ground truth (at that threshold)
- Points below 0: Method is WORSE than ground truth
- Shaded region: 95% CI crosses 0 = no significant difference

This directly answers: "Does preprocessing matter at threshold X?"

### Panel C: Clinical Threshold Annotation

**What it shows:**
- Same DCA plot as Panel A but with **annotated vertical zones**
- Three clinical scenarios:

| Scenario | Threshold | Rationale |
|----------|-----------|-----------|
| **Conservative** | 5% | Early detection priority; accept many false positives |
| **Moderate** | 10% | Balanced approach; standard screening |
| **Aggressive** | 20% | Resource-constrained; minimize unnecessary referrals |

**Annotations:**
- Vertical shaded bands for each threshold zone
- Text labels explaining clinical meaning
- Arrows pointing to where each method performs best

### Panel D: Interventions Avoided Bar Chart

**What it shows:**
For a cohort of 1000 patients, at each threshold:
- How many **unnecessary referrals avoided** (vs treat-all)
- How many **true cases correctly identified** (vs treat-none)

This is the **most clinically intuitive** metric:
> "Using Ensemble+CSDI instead of LOF+SAITS at 15% threshold avoids 23 unnecessary referrals per 1000 patients while identifying the same number of glaucoma cases."

**Bar chart design:**
- Grouped bars by preprocessing method
- Y-axis: Count per 1000 patients
- Colors: Semantic (green = benefit, orange = neutral, red = harm)

## Key Clinical Messages

1. **At low thresholds (5-10%)**: All models perform similarly - preprocessing choice less critical
2. **At moderate thresholds (10-20%)**: Foundation model preprocessing shows advantage
3. **At high thresholds (>20%)**: Differences between methods become clinically significant
4. **Ground truth** is not always best - well-tuned ensemble may match or exceed

## Data Requirements

### From DuckDB extraction:

```sql
SELECT
    outlier_method,
    imputation_method,
    classifier,
    net_benefit_5pct,
    net_benefit_10pct,
    net_benefit_15pct,
    net_benefit_20pct,
    -- Need bootstrap samples for CIs
    y_true,
    y_prob
FROM essential_metrics
WHERE classifier = 'CatBoost'
```

### Bootstrap CI calculation:

For each bootstrap iteration:
1. Resample (y_true, y_prob) pairs
2. Compute net benefit at each threshold
3. Store percentile-based CIs

## Harrell's Seven Errors Addressed

| Error | Our Solution |
|-------|--------------|
| 1. Unspecified clinical decision | "Refer for comprehensive glaucoma exam" |
| 2. Excessive threshold range | Restricted to 3.5%-35% |
| 3. Excessive white space | Y-axis starts at -0.02 |
| 4. Unsmoothed curves | LOESS smoothing applied |
| 5. Threshold recommendation | No single threshold recommended |
| 6. Ignoring negative findings | Report when methods are equivalent |
| 7. No overfitting correction | Bootstrap CIs from cross-validated predictions |

## Technical Implementation

### Python Code Structure

```python
from src.viz.plot_config import setup_style, COLORS, save_figure
import numpy as np

def compute_net_benefit(y_true, y_prob, threshold):
    """Compute net benefit at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)

    if threshold >= 1:
        return 0.0

    exchange_rate = threshold / (1 - threshold)
    return tp/n - fp/n * exchange_rate

def bootstrap_net_benefit(y_true, y_prob, threshold, n_boot=1000):
    """Bootstrap CI for net benefit."""
    nbs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        nb = compute_net_benefit(y_true[idx], y_prob[idx], threshold)
        nbs.append(nb)
    return np.percentile(nbs, [2.5, 97.5])
```

### Smoothing (Harrell-recommended)

```python
from scipy.ndimage import gaussian_filter1d

# Apply smoothing to net benefit curves
smoothed_nb = gaussian_filter1d(net_benefit_values, sigma=2)
```

## Quality Criteria

- [ ] No hardcoded colors (use COLORS dict)
- [ ] Bootstrap CIs computed correctly
- [ ] LOESS/Gaussian smoothing applied
- [ ] Clinically meaningful threshold range (3.5%-35%)
- [ ] Clear panel labels and annotations
- [ ] Accessible color palette (colorblind-safe)
- [ ] JSON data saved for reproducibility
- [ ] Dimensions: 14 × 10 inches (grid layout)

## References

- Vickers AJ, Elkin EB (2006). Decision curve analysis. Med Decis Making
- Harrell FE. Extended Decision Curve Analysis. fharrell.com/post/edca/
- Van Calster B et al. (2018). Reporting and Interpreting Decision Curve Analysis
- Tham YC et al. (2014). Global prevalence of glaucoma. Ophthalmology (3.54% prevalence)

---

## Status Tracking

- [ ] Data extraction from DuckDB
- [ ] Bootstrap CI computation
- [ ] Panel A implementation (DCA with CI)
- [ ] Panel B implementation (difference plot)
- [ ] Panel C implementation (threshold zones)
- [ ] Panel D implementation (interventions bar chart)
- [ ] Integration and layout
- [ ] Harrell's 7 errors checklist review
- [ ] JSON export
- [ ] Figure QA tests
