# fig-trans-12: M-GAM: Missing Values as Features

**Status**: 📋 PLANNED
**Tier**: 3 - Alternative Approaches
**Target Persona**: E-commerce analysts, EHR data scientists, statisticians

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-12 |
| Type | Architecture + example diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Introduce M-GAM (Missing-aware Generalized Additive Model) as the appropriate approach for interpretable modeling when missingness itself is informative—contrasting with the imputation approach.

---

## 3. Key Message

> "When missing data isn't corruption but information (store closed, patient didn't visit, sensor intentionally off), M-GAM treats missingness as a feature rather than a bug. This maintains interpretability while often outperforming imputation."

---

## 4. Literature Source

McTavish et al. (2024, NeurIPS): "Interpretable Generalized Additive Models for Datasets with Missing Values"

Key insight (Proposition 3.1): "Even with perfect imputation, models using missingness as a value can outperform models using imputed data."

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  M-GAM: Missing Values as Features                                         │
│  When Missingness IS Information                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE PROBLEM WITH IMPUTATION                                               │
│  ───────────────────────────                                               │
│                                                                            │
│  Original Data:    X1=5, X2=3, X3=?                                        │
│                                                                            │
│  After Imputation: X1=5, X2=3, X3=f(X1,X2)=7                               │
│                                                                            │
│  The model sees:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │     X1      │      X2      │         X3                         │      │
│  │   ────────  │   ────────   │   ────────────────────             │      │
│  │    ╱╲       │     ╱╲       │     NOW A 3D FUNCTION!             │      │
│  │   ╱  ╲      │    ╱  ╲      │     X3 = f(X1, X2)                 │      │
│  │  ╱    ╲     │   ╱    ╲     │     ← NOT INTERPRETABLE            │      │
│  │ ╱      ╲    │  ╱      ╲    │                                    │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                            │
│  ⚠️ Imputation creates multivariate dependencies that break GAM structure! │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  M-GAM APPROACH                                                            │
│  ──────────────                                                            │
│                                                                            │
│  Original Data:    X1=5, X2=3, X3=MISSING                                  │
│                                                                            │
│  M-GAM sees:       X1=5, X2=3, M3=1 (missingness indicator)                │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  No Missing Data        │   When X3 is Missing                  │      │
│  ├─────────────────────────┼───────────────────────────────────────┤      │
│  │     X1      X2      X3  │    X1*     X2*     [X3 removed]       │      │
│  │   ────────────────────  │  ────────────────                     │      │
│  │    ╱╲     ╱╲     ╱╲     │   ╱╲      ╱╲                          │      │
│  │   ╱  ╲   ╱  ╲   ╱  ╲    │  ╱  ╲    ╱  ╲   ← ADJUSTED curves    │      │
│  │  ╱    ╲ ╱    ╲ ╱    ╲   │ ╱    ╲  ╱    ╲     for missingness   │      │
│  └─────────────────────────┴───────────────────────────────────────┘      │
│                                                                            │
│  ✓ Still univariate shape functions → INTERPRETABLE                       │
│  ✓ Missingness indicator captured → INFORMATIVE                           │
│  ✓ ℓ0 regularization prevents overfitting → SPARSE                        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHEN TO USE M-GAM (Not TSFMs)                                             │
│  ─────────────────────────────                                             │
│                                                                            │
│  Scenario                        │ Imputation │ M-GAM                      │
│  ────────────────────────────────┼────────────┼──────────────────────────  │
│  Sensor failure (PLR blink)      │ ✓ Yes      │ ✗ No                       │
│  Store closed on Sunday          │ ✗ No       │ ✓ Yes                      │
│  Patient skipped appointment     │ ✗ No       │ ✓ Yes                      │
│  Lab test too expensive to run   │ ✗ No       │ ✓ Yes                      │
│  Data transmission error         │ ✓ Yes      │ ✗ No                       │
│                                                                            │
│  Rule: If missingness = measurement error → Impute                        │
│        If missingness = real-world cause  → M-GAM                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  KEY THEORETICAL RESULT (McTavish 2024)                                    │
│  ──────────────────────────────────────                                    │
│                                                                            │
│  Proposition 3.1: Perfect imputation can REDUCE model performance.         │
│                                                                            │
│  "When missingness is informative, the Bayes-optimal model using           │
│   missingness as a value outperforms the Bayes-optimal model using         │
│   perfectly imputed data."                                                 │
│                                                                            │
│  Translation: Even if you could impute perfectly, you'd still do worse    │
│               than treating missingness as a feature!                      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"M-GAM: Missing Values as Features"

### Caption
"When missingness is informative (store closures, skipped appointments, expensive tests), imputation destroys valuable signal. M-GAM (McTavish et al. 2024) treats missingness indicators as explicit features while maintaining the interpretable univariate structure of GAMs. Key insight: even perfect imputation can reduce model performance when missingness correlates with outcomes. Use M-GAM for business/healthcare data where gaps have meaning; use imputation (TSFMs) for sensor data where gaps are measurement errors."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a diagram explaining M-GAM for missing value handling.

TOP - Problem with Imputation:
Show how imputing X3=f(X1,X2) creates a multivariate dependency
Simple GAM curves becoming a 3D surface = not interpretable

MIDDLE - M-GAM Approach:
Side-by-side: "No Missing" vs "X3 Missing"
Show adjusted univariate curves when missingness is detected
Highlight: still interpretable, missingness captured

BOTTOM LEFT - When to use table:
Imputation vs M-GAM for different scenarios
(sensor failure vs store closed vs patient skipped)

BOTTOM RIGHT - Key theoretical result:
Proposition 3.1 callout box
"Perfect imputation can reduce performance"

Style: Academic, GAM curve visualizations, interpretability emphasis
```

---

## 8. Alt Text

"Diagram explaining M-GAM for handling missing values. Top section shows problem with imputation: when X3 is imputed as function of X1 and X2, it creates multivariate dependencies that break GAM interpretability. Middle section shows M-GAM approach: uses adjusted univariate shape curves when missingness is detected, maintaining interpretability while capturing missingness information. Bottom left shows decision table for when to use imputation (sensor errors) versus M-GAM (store closures, skipped appointments). Bottom right highlights McTavish 2024 Proposition 3.1: perfect imputation can reduce model performance when missingness is informative."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
