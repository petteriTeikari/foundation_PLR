# fig-repo-30: How to Read Instability Plots

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-30 |
| **Title** | How to Read Prediction Instability Plots |
| **Complexity Level** | L2-L3 (Statistical visualization + methodology) |
| **Target Persona** | All |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Teach readers how to interpret prediction instability plots—a key visualization for understanding how stable individual predictions are across bootstrap resamples.

## Key Message

"Instability plots show how much each patient's prediction varies when the model is retrained on bootstrap samples. Wide vertical spread = unstable prediction. Stable models have points clustered tightly around the diagonal."

## Literature Foundation

| Source | Key Contribution |
|--------|------------------|
| Riley et al. 2023 | BMC Medicine - "Clinical prediction models and the multiverse of madness" |
| Rhodes et al. 2025 | pminternal R package for model stability analysis |
| Riley et al. 2025 | BMJ - Uncertainty in risk predictions |

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     HOW TO READ PREDICTION INSTABILITY PLOTS                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS AN INSTABILITY PLOT?                                                   │
│  ════════════════════════════                                                   │
│                                                                                 │
│  Shows how individual predictions vary when the model is retrained on           │
│  different bootstrap samples of the training data.                              │
│                                                                                 │
│  KEY INSIGHT: A prediction can have TWO kinds of uncertainty:                   │
│  1. EPISTEMIC: Model uncertainty (instability across retraining)                │
│  2. ALEATORIC: Irreducible noise in the data                                    │
│                                                                                 │
│  Instability plots visualize EPISTEMIC uncertainty.                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ANATOMY OF AN INSTABILITY PLOT                                                 │
│  ═══════════════════════════════                                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Bootstrap                                                              │   │
│  │  Prediction   1.0 ┤                                          ∕         │   │
│  │  (y-axis)         │                                       ∕            │   │
│  │               0.8 ┤                          ╱╲        ∕   ← Wide CI   │   │
│  │                   │                       ╱    ╲    ∕      = UNSTABLE  │   │
│  │               0.6 ┤                    ╱        ╲∕                      │   │
│  │                   │          ┃      ╱                                   │   │
│  │               0.4 ┤          ┃   ╱    ← Narrow CI = STABLE             │   │
│  │                   │        ∕ ┃∕                                         │   │
│  │               0.2 ┤     ∕   ∕                                           │   │
│  │                   │  ∕   ∕     45° DIAGONAL:                            │   │
│  │               0.0 ┼∕───∕──────perfect agreement with                    │   │
│  │                   0.0  0.2  0.4  0.6  0.8  1.0   original model        │   │
│  │                                                                         │   │
│  │                   Original Model Prediction (x-axis)                    │   │
│  │                                                                         │   │
│  │   EACH VERTICAL LINE = One patient                                      │   │
│  │   • Line spans 95% CI from bootstrap models                             │   │
│  │   • Center point = mean prediction across bootstraps                    │   │
│  │   • Dot on x-axis = original model prediction                           │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  READING THE PLOT                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  1. OVERALL SPREAD                                                              │
│     ──────────────                                                              │
│     • Tight around diagonal = stable model                                      │
│     • Wide spread = unstable predictions                                        │
│     • Asymmetric spread = calibration issues                                    │
│                                                                                 │
│  2. INDIVIDUAL PATIENT LINES                                                    │
│     ─────────────────────────                                                   │
│     • Short line = stable prediction for this patient                           │
│     • Long line = unstable prediction—flag for second opinion                   │
│     • Line crossing 0.5 = classification might flip!                            │
│                                                                                 │
│  3. POSITION ON X-AXIS                                                          │
│     ──────────────────                                                          │
│     • Extreme predictions (near 0 or 1) often more stable                       │
│     • Mid-range predictions (0.3-0.7) often less stable                         │
│     • This is where "triage" decisions are hardest                              │
│                                                                                 │
│  4. MAPE (Mean Absolute Prediction Error)                                       │
│     ─────────────────────────────────────                                       │
│     • Per-patient: mean_b |p̂ᵢ⁽ᵇ⁾ - p̂ᵢ| across bootstrap iterations           │
│     • Global MAPE: average of per-patient MAPE across all patients              │
│     • Lower MAPE = more stable predictions                                      │
│     • Typical range: 0.02-0.10 for well-behaved clinical models                 │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE BOOTSTRAP PROCESS (Riley 2023 Table 1)                                     │
│  ══════════════════════════════════════════                                     │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  STEP 1:  Train original model on full dataset                          │   │
│  │           → Get prediction p̂ᵢ for each patient i                       │   │
│  │                                                                         │   │
│  │  STEP 2:  For b = 1 to B (e.g., B = 1000):                              │   │
│  │           a) Draw bootstrap sample (with replacement)                    │   │
│  │           b) Retrain model on bootstrap sample                           │   │
│  │           c) Get predictions p̂ᵢ⁽ᵇ⁾ for ORIGINAL patients               │   │
│  │                                                                         │   │
│  │  STEP 3:  For each patient, compute:                                    │   │
│  │           • Mean prediction across bootstraps                            │   │
│  │           • 95% CI: [2.5th percentile, 97.5th percentile]               │   │
│  │           • MAPE: mean |p̂ᵢ⁽ᵇ⁾ - p̂ᵢ| across all b                       │   │
│  │                                                                         │   │
│  │  STEP 4:  Plot each patient as vertical line from CI_lo to CI_hi        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXAMPLE: PLR Preprocessing Comparison                                          │
│  ═════════════════════════════════════                                          │
│  (ILLUSTRATIVE VALUES - actual results in manuscript figures)                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Ground Truth Pipeline     FM Ensemble Pipeline     Traditional Pipeline│   │
│  │  MAPE ≈ low                MAPE ≈ moderate          MAPE ≈ higher       │   │
│  │                                                                         │   │
│  │  ┃ ┃┃┃┃ ┃┃                 ╱╲  ╱╲                   ╱╲    ╱╲  ╱╲        │   │
│  │  ┃┃┃┃┃┃┃┃∕                ╱  ╲╱  ╲∕                ╱  ╲  ╱  ╲╱  ╲       │   │
│  │  ┃┃┃┃∕ ∕                 ╱  ∕ ∕                    ╱    ╲╱    ∕   ╲      │   │
│  │  ∕                       ∕                        ∕                      │   │
│  │  Tight = Stable          Moderate spread          Wide = Unstable       │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Interpretation:                                                                │
│  • Ground truth preprocessing → most stable predictions                        │
│  • FM ensemble → moderately stable (acceptable for clinical use)               │
│  • Traditional methods → least stable (predictions may not be reliable)        │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CLASSIFICATION INSTABILITY INDEX (CII)                                         │
│  ═══════════════════════════════════════                                        │
│                                                                                 │
│  For binary classification, CII measures how often predicted CLASS changes:    │
│                                                                                 │
│  CII = proportion of bootstraps where predicted class ≠ original class         │
│                                                                                 │
│  Example:                                                                       │
│  • Patient with p̂ = 0.48, CII = 0.42                                          │
│    → 42% of bootstrap models classify this patient differently!                 │
│    → FLAG: Do not trust this classification                                     │
│                                                                                 │
│  • Patient with p̂ = 0.95, CII = 0.02                                          │
│    → Only 2% of models disagree—very stable classification                     │
│                                                                                 │
│  NOTE: CII is computed at threshold t (default 0.5). For clinical              │
│  applications, use the clinically relevant decision threshold.                  │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON MISTAKES                                                                │
│  ═══════════════                                                                │
│                                                                                 │
│  ❌ "Instability means the model is wrong"                                      │
│     → Instability means the model is UNCERTAIN—this is information!            │
│                                                                                 │
│  ❌ "All predictions should be stable"                                          │
│     → Some patients are inherently ambiguous; expect instability near 0.5       │
│                                                                                 │
│  ❌ "Wide CI means we need more data"                                           │
│     → May be true, but could also mean feature space is noisy                   │
│                                                                                 │
│  ❌ "Only looking at aggregate MAPE"                                            │
│     → Individual patients may have very different stability                     │
│                                                                                 │
│  ❌ "Comparing instability across models with different thresholds"             │
│     → Must use same classification threshold for fair comparison                │
│                                                                                 │
│  ❌ "Using out-of-bag predictions for instability analysis"                     │
│     → Must predict on ALL patients with each bootstrap model                    │
│                                                                                 │
│  ❌ "Not reporting bootstrap iteration count B"                                 │
│     → Always state B (e.g., B=1000) for reproducibility                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation in This Repository

### R Code Location
`src/r/figures/fig_instability_combined.R`

### Key Implementation Details
```r
# Load bootstrap predictions from JSON
predictions <- fromJSON("data/bootstrap_predictions.json")

# For each patient, compute CI across bootstrap models
patient_summary <- predictions %>%
  group_by(patient_id) %>%
  summarise(
    original_pred = first(original_prediction),
    mean_pred = mean(bootstrap_prediction),
    ci_lo = quantile(bootstrap_prediction, 0.025),
    ci_hi = quantile(bootstrap_prediction, 0.975),
    mape = mean(abs(bootstrap_prediction - original_prediction)),
    cii = mean(as.integer(bootstrap_prediction > 0.5) !=
               as.integer(original_prediction > 0.5))
  )

# Plot: vertical lines for each patient
ggplot(patient_summary, aes(x = original_pred)) +
  geom_segment(aes(y = ci_lo, yend = ci_hi, x = original_pred, xend = original_pred),
               alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +  # Diagonal
  labs(x = "Original Model Prediction",
       y = "Bootstrap Model Predictions (95% CI)")
```

### pminternal R Package Functions
```r
library(pminternal)

# Model stability analysis (Rhodes 2025)
validate(model, data,
         cal_plot = TRUE,           # Calibration stability
         pred_plot = TRUE,          # Prediction instability plot
         mape_plot = TRUE)          # MAPE distribution

# Specific instability functions
prediction_stability(model, data, B = 1000)
mape_stability(model, data, B = 1000)
calibration_stability(model, data, B = 1000)
classification_stability(model, data, B = 1000, threshold = 0.5)
```

## Content Elements

1. **Instability plot anatomy**: Labeled diagram showing vertical lines, diagonal, and axes
2. **Reading guide**: 4 numbered steps (overall spread, individual lines, position, MAPE)
3. **Bootstrap process**: Step-by-step explanation from Riley 2023
4. **Example interpretation**: Three-panel comparison of preprocessing pipelines
5. **CII explanation**: Classification Instability Index for binary outcomes
6. **Common mistakes**: What NOT to conclude

## Text Content

### Title Text
"How to Read Prediction Instability Plots"

### Caption
Prediction instability plots (Riley et al. 2023) show how individual risk predictions vary when the model is retrained on bootstrap samples. Each vertical line represents one patient: the x-position is the original prediction, and the line spans the 95% CI from bootstrap models. Tight clustering around the diagonal indicates stable predictions. Wide spread indicates epistemic uncertainty—these patients might warrant a "second opinion." MAPE (Mean Absolute Prediction Error) summarizes overall stability.

## Prompts for Nano Banana Pro

### Style Prompt
Educational diagram explaining prediction instability plot interpretation. Scatter plot with vertical CI lines for each patient, diagonal reference line. Three-panel comparison example. BMC Medicine style per Riley 2023.

### Content Prompt
Create an instability plot reading guide:

**TOP - Anatomy**:
- Scatter plot with vertical lines (one per patient)
- 45° diagonal reference line
- Axes: original prediction (x) vs bootstrap predictions (y)
- Callouts for stable vs unstable patients

**MIDDLE - Reading Steps**:
- 4 numbered steps with icons
- 1: Overall spread around diagonal
- 2: Individual patient lines (short = stable)
- 3: Position on x-axis matters
- 4: MAPE summary statistic

**BOTTOM - Example**:
- Three-panel comparison: Ground Truth, FM Ensemble, Traditional
- MAPE values annotated

**SIDEBAR - CII**:
- Classification Instability Index explanation
- Example with p̂ = 0.48

## Alt Text

Educational diagram explaining prediction instability plot interpretation. Shows anatomy with vertical lines for each patient representing 95% CI of bootstrap predictions, plotted against original model predictions. Diagonal line shows perfect agreement. Reading guide: (1) check overall spread around diagonal, (2) examine individual lines (short = stable, long = unstable), (3) note that mid-range predictions are often less stable, (4) MAPE summarizes average instability. Example compares three pipelines: ground truth (MAPE 0.032, tight), FM ensemble (MAPE 0.041, moderate), traditional (MAPE 0.089, wide spread). CII section explains classification instability for binary outcomes.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md

## References

- Riley RD, Collins GS, Ensor J, et al. (2023) Clinical prediction models and the multiverse of madness. BMC Medicine 21:112.
- Rhodes C, et al. (2025) pminternal: Internal validation of clinical prediction models. R package.
- Riley RD, Snell KI, Collins GS, et al. (2025) Uncertainty in risk predictions. BMJ 388:e077740.
