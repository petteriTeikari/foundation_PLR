# fig-repo-28: STRATOS Metrics: Beyond AUROC

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-28 |
| **Title** | STRATOS Metrics: Beyond AUROC |
| **Complexity Level** | L2 (Statistical concept) |
| **Target Persona** | Biostatistician, Research Scientist |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P0 |
| **Aspect Ratio** | 16:9 |

## Purpose

Explain the 5 STRATOS performance domains—why AUROC alone is insufficient for clinical model evaluation.

## Key Message

"AUROC measures discrimination but ignores calibration and clinical utility. STRATOS guidelines require ALL 5 domains for proper model evaluation."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STRATOS METRICS: BEYOND AUROC                                │
│                    Van Calster et al. 2024                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM WITH AUROC ALONE                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  A model with AUROC = 0.90 could still:                                         │
│  ❌ Predict 50% probability when true risk is 5% (miscalibrated)                │
│  ❌ Be useless at clinical decision thresholds (no utility)                     │
│  ❌ Have unstable predictions across patients                                   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE 5 STRATOS PERFORMANCE DOMAINS                                              │
│  ══════════════════════════════════                                             │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │   1. DISCRIMINATION                    2. CALIBRATION                     │ │
│  │   ══════════════════                   ═══════════════                    │ │
│  │                                                                           │ │
│  │   "Can the model RANK patients?"       "Do predictions match reality?"    │ │
│  │                                                                           │ │
│  │   📊 AUROC (0.0 - 1.0)                 📈 Calibration slope (ideal: 1.0)  │ │
│  │      Higher = better ranking            Slope < 1 = overfitting           │ │
│  │                                         Slope > 1 = underfitting          │ │
│  │                                                                           │ │
│  │                                        📉 Calibration intercept (ideal: 0)│ │
│  │                                         Measures systematic bias          │ │
│  │                                                                           │ │
│  │                                        ⚖️ O:E ratio (ideal: 1.0)          │ │
│  │                                         Observed / Expected events        │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │   3. OVERALL PERFORMANCE               4. CLASSIFICATION                  │ │
│  │   ══════════════════════               ═════════════════                  │ │
│  │                                                                           │ │
│  │   "Discrimination + Calibration"       "At a specific threshold"          │ │
│  │                                                                           │ │
│  │   📊 Brier score (0.0 - 1.0)           📊 Sensitivity, Specificity        │ │
│  │      Lower = better                       At chosen threshold             │ │
│  │                                                                           │ │
│  │   📊 Scaled Brier / IPA                ⚠️ F1 NOT RECOMMENDED              │ │
│  │      Compares to null model               (ignores true negatives)        │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │   5. CLINICAL UTILITY                                                     │ │
│  │   ═══════════════════                                                     │ │
│  │                                                                           │ │
│  │   "Is the model useful for DECISIONS?"                                    │ │
│  │                                                                           │ │
│  │   📊 Net Benefit (threshold-specific)                                     │ │
│  │      Accounts for: benefits of true positives vs harms of false positives │ │
│  │                                                                           │ │
│  │   📈 Decision Curve Analysis (DCA)                                        │ │
│  │      Net benefit across threshold range (e.g., 5% - 40%)                  │ │
│  │                                                                           │ │
│  │   This is what actually matters for clinical deployment!                  │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  METRICS WE REPORT                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  │ Domain         │ Metric                │ Reported │                         │
│  │ ─────────────  │ ─────────────────────  │ ──────── │                         │
│  │ Discrimination │ AUROC with 95% CI      │   ✅     │                         │
│  │ Calibration    │ Slope, Intercept, O:E  │   ✅     │                         │
│  │ Overall        │ Brier, Scaled Brier    │   ✅     │                         │
│  │ Classification │ Sens, Spec at 15%      │   ✅     │                         │
│  │ Clinical       │ Net Benefit, DCA       │   ✅     │                         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  STRATOS SAYS DO NOT USE                                                        │
│  ═══════════════════════                                                        │
│                                                                                 │
│  ❌ F1 Score (ignores true negatives)                                           │
│  ❌ AUPRC alone (ignores true negatives)                                        │
│  ❌ Accuracy at 0.5 threshold (wrong for prevalence ≠ 50%)                      │
│  ❌ Youden index optimization (assumes equal costs)                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Problem statement**: Why AUROC alone fails
2. **5 domains grid**: Discrimination, Calibration, Overall, Classification, Clinical
3. **Metrics table**: What we report for each domain
4. **Do not use list**: STRATOS-banned metrics

## Text Content

### Title Text
"STRATOS Metrics: 5 Domains for Proper Model Evaluation"

### Caption
Following STRATOS guidelines (Van Calster 2024), we report metrics across all 5 performance domains: discrimination (AUROC), calibration (slope, intercept, O:E ratio), overall (Brier, scaled Brier), classification (sensitivity, specificity at clinical threshold), and clinical utility (Net Benefit, DCA). AUROC alone is insufficient—a well-discriminating model can still be poorly calibrated or clinically useless.

## Prompts for Nano Banana Pro

### Style Prompt
Five-domain grid showing STRATOS performance categories. Each domain as a card with icon, title, and key metrics. "Do not use" section with warning symbols. Clean, medical research aesthetic. Metrics table at bottom.

### Content Prompt
Create a STRATOS metrics overview:

**TOP - Problem**:
- "AUROC alone is insufficient"
- 3 bullet points of what AUROC misses

**MIDDLE - 5 Domain Grid**:
- 5 cards arranged in 2-3 layout
- Each with: Domain name, "What it measures", Key metrics
- Icons for each domain

**BOTTOM LEFT - Metrics Table**:
- Domain | Metric | Reported (checkmarks)

**BOTTOM RIGHT - Banned**:
- Red X marks: F1, AUPRC alone, Accuracy at 0.5, Youden

## Alt Text

STRATOS performance metrics diagram showing 5 evaluation domains. Discrimination: AUROC with CI. Calibration: slope, intercept, O:E ratio. Overall: Brier, scaled Brier. Classification: sensitivity, specificity at threshold. Clinical utility: Net Benefit, DCA. Table shows all metrics reported. Banned metrics: F1, AUPRC alone, accuracy at 0.5, Youden index.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
