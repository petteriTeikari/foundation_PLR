# Figure Plan: DCA Expert Mechanics (fig-repo-41)

**Status:** Planning
**Priority:** High
**Type:** Educational / Methodology Figure
**Output:** Supplementary Material (potentially main)

---

## Overview

An expert-level figure explaining the **mechanics of net benefit calculation** in Decision Curve Analysis. Unlike basic DCA plots that simply show curves, this figure deconstructs the mathematics and provides intuitive understanding of what net benefit means clinically.

## The Core Insight

Net benefit answers: **"For every 100 patients, how many TRUE cases would we correctly identify, penalized by the FALSE alarms we'd tolerate?"**

The formula:
```
NB(t) = TP/N - FP/N × t/(1-t)
```

The term `t/(1-t)` is the **exchange rate**: how many false positives we'd trade for one true positive.

## Visual Concept: "The Exchange Rate Panel"

### Layout (3-panel horizontal)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NET BENEFIT: THE CLINICAL EXCHANGE                    │
├──────────────────────┬──────────────────────┬──────────────────────────┤
│                      │                      │                          │
│   PANEL A:           │   PANEL B:           │   PANEL C:               │
│   Exchange Rate      │   Decomposition      │   Clinical Meaning       │
│   Curve              │   at Threshold       │   (Patient Flow)         │
│                      │                      │                          │
│   y = t/(1-t)        │   Stacked bars       │   100 patients →         │
│   for t ∈ [0,1]      │   showing TP - FP    │   decisions →            │
│                      │   components         │   outcomes               │
│                      │                      │                          │
└──────────────────────┴──────────────────────┴──────────────────────────┘
```

### Panel A: The Exchange Rate Curve

**What it shows:**
- X-axis: Threshold probability (0% to 50%)
- Y-axis: Exchange rate t/(1-t)
- Key annotations:
  - At 10% threshold: exchange rate = 0.11 (9 FPs = 1 TP)
  - At 20% threshold: exchange rate = 0.25 (4 FPs = 1 TP)
  - At 33% threshold: exchange rate = 0.5 (2 FPs = 1 TP)
  - At 50% threshold: exchange rate = 1.0 (1 FP = 1 TP)

**Clinical interpretation callout:**
> "At a 10% threshold, clinicians accept that it takes 9 unnecessary biopsies to find 1 cancer.
> At 20%, only 4 unnecessary biopsies per cancer found."

### Panel B: Net Benefit Decomposition

**What it shows:**
A stacked/waterfall chart at a SINGLE threshold (e.g., 15%) showing:
- **Green bar (positive):** True Positive Rate contribution (TP/N)
- **Red bar (negative):** False Positive penalty (FP/N × t/(1-t))
- **Net result:** Net Benefit = Green - Red

Show for 3 scenarios:
1. **Perfect model** (all TP, no FP)
2. **Our model** (realistic TP/FP tradeoff)
3. **Treat-all** (everyone gets treatment)

This makes visible WHY treat-all fails at high thresholds (FP penalty dominates).

### Panel C: Patient Flow Diagram

**What it shows:**
A Sankey-style flow for 100 patients:

```
100 patients
    │
    ├─→ Model predicts HIGH risk (60)
    │       ├─→ True Positive (40) → Correct treatment
    │       └─→ False Positive (20) → Unnecessary treatment
    │
    └─→ Model predicts LOW risk (40)
            ├─→ True Negative (30) → Correct no-treatment
            └─→ False Negative (10) → Missed cases
```

With annotations showing:
- **Benefit:** 40 correctly treated
- **Cost:** 20 unnecessarily treated × exchange rate
- **Net benefit calculation**

## Key Messages to Convey

1. **Net benefit is NOT accuracy** - it weights errors by clinical importance
2. **Threshold choice implies an exchange rate** - every threshold has clinical meaning
3. **The "treat all" line fails at high thresholds** because FP penalty overwhelms
4. **Models must beat "treat none"** across clinically relevant range

## Data Requirements

For our preprocessing comparison (glaucoma detection):

| Model | TP | FP | TN | FN | N |
|-------|----|----|----|----|---|
| Ground truth + CatBoost | Extract from predictions | | | | 208 |
| Ensemble + CSDI + CatBoost | Extract from predictions | | | | 208 |
| LOF + SAITS + TabPFN | Extract from predictions | | | | 208 |

Compute at thresholds: 5%, 10%, 15%, 20%, 25%, 30%

## Technical Implementation

### Python/Matplotlib

```python
# Panel A: Exchange rate curve
thresholds = np.linspace(0.01, 0.5, 100)
exchange_rates = thresholds / (1 - thresholds)

# Panel B: Decomposition bars (waterfall chart)
# Use matplotlib's bar charts with positive/negative stacking

# Panel C: Sankey diagram
# Consider using plotly or matplotlib-sankey
```

### Colors (from COLORS dict)

- True Positives: `COLORS["benefit"]` or green
- False Positives: `COLORS["cost"]` or red
- Net Benefit: `COLORS["primary"]`

## Relationship to Millard 2025 (Brier Curves)

Millard shows that:
- **Brier score at threshold t** = mean(Brier) evaluated at threshold
- **Cost curve** from Brier decomposition has geometric relationship to DCA
- Our figure should acknowledge this but focus on **clinical interpretability**

Key insight from Millard:
> "The expected cost curve is related to the calibration-refinement decomposition of Brier score"

We can add a small inset or footnote referencing that net benefit connects to proper scoring rules.

## Quality Criteria

- [ ] No hardcoded colors (use COLORS dict)
- [ ] Exchange rate annotations at key thresholds
- [ ] Clear visual hierarchy (most important info emphasized)
- [ ] Accessible to clinicians unfamiliar with DCA
- [ ] JSON data saved for reproducibility
- [ ] Dimensions: 14 × 6 inches (wide format for 3 panels)

## References

- Vickers AJ, Elkin EB (2006). Decision curve analysis. Med Decis Making
- Vickers AJ et al. (2019). A simple, step-by-step guide to interpreting decision curve analysis
- Millard SP (2025). Comparison of decision curves, cost curves, and Brier curves
- Van Calster B et al. (2018). Reporting and Interpreting Decision Curve Analysis

---

## Status Tracking

- [ ] Data extraction from DuckDB
- [ ] Panel A implementation
- [ ] Panel B implementation
- [ ] Panel C implementation
- [ ] Integration and layout
- [ ] Color and accessibility review
- [ ] JSON export
- [ ] Figure QA tests
