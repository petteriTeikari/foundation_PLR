# fig-repo-39: Calibration Metrics Explained

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-39 |
| **Title** | Calibration Metrics Explained |
| **Complexity Level** | L2 (Statistical concept) |
| **Target Persona** | Biostatistician, Research Scientist |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain calibration slope, intercept, and O:E ratio—critical STRATOS metrics beyond AUROC.

## Key Message

"A well-calibrated model predicts 30% risk when 30% of such patients actually have disease. Slope=1, intercept=0, O:E=1 indicate perfect calibration."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CALIBRATION METRICS EXPLAINED                                │
│                    "Do predictions match reality?"                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHAT IS CALIBRATION?                                                           │
│  ═════════════════════                                                          │
│                                                                                 │
│  DISCRIMINATION: "Can the model RANK patients?"                                 │
│  CALIBRATION:    "Do predicted probabilities MATCH observed frequencies?"       │
│                                                                                 │
│  A model can have excellent AUROC (0.90) but terrible calibration:              │
│  → Predicts 50% when true risk is 5%                                            │
│  → Unusable for clinical decisions!                                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE CALIBRATION PLOT                                                           │
│  ════════════════════                                                           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │  Observed    1.0 ┤                                          ╱           │   │
│  │  proportion      │                                       ╱              │   │
│  │  (actual         │                                    ╱                 │   │
│  │  disease     0.8 ┤                                 ╱                    │   │
│  │  rate in         │                              ╱                       │   │
│  │  each bin)       │                           ╱                          │   │
│  │              0.6 ┤                        ╱                             │   │
│  │                  │                     ╱          ← IDEAL LINE          │   │
│  │                  │                  ╱             (slope=1, intercept=0)│   │
│  │              0.4 ┤               ╱                                      │   │
│  │                  │            ╱                                         │   │
│  │                  │         ╱                                            │   │
│  │              0.2 ┤      ╱   ●──●──●──●         ← ACTUAL MODEL           │   │
│  │                  │   ╱          (may deviate from ideal)                │   │
│  │                  │╱                                                     │   │
│  │              0.0 ┼──────────────────────────────────────────────────    │   │
│  │                  0.0    0.2    0.4    0.6    0.8    1.0                 │   │
│  │                              Predicted probability                      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE THREE METRICS                                                              │
│  ═════════════════                                                              │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                        │    │
│  │  1. CALIBRATION SLOPE                                                  │    │
│  │  ════════════════════                                                  │    │
│  │                                                                        │    │
│  │  Ideal: 1.0                                                            │    │
│  │                                                                        │    │
│  │  Slope < 1:  Model is OVERCONFIDENT (extreme predictions too extreme)  │    │
│  │              Predicts 90% when should be 70%                           │    │
│  │              Predicts 10% when should be 30%                           │    │
│  │                                                                        │    │
│  │  Slope > 1:  Model is UNDERCONFIDENT (predictions too moderate)        │    │
│  │              Predicts 60% when should be 80%                           │    │
│  │                                                                        │    │
│  │  Common issue: slope < 1 in overfitted models                          │    │
│  │                                                                        │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                        │    │
│  │  2. CALIBRATION INTERCEPT                                              │    │
│  │  ════════════════════════                                              │    │
│  │                                                                        │    │
│  │  Ideal: 0.0                                                            │    │
│  │                                                                        │    │
│  │  Intercept > 0: Model UNDERESTIMATES risk systematically               │    │
│  │                 Average prediction lower than average outcome          │    │
│  │                                                                        │    │
│  │  Intercept < 0: Model OVERESTIMATES risk systematically                │    │
│  │                 Average prediction higher than average outcome         │    │
│  │                                                                        │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                        │    │
│  │  3. O:E RATIO (Observed / Expected)                                    │    │
│  │  ══════════════════════════════════                                    │    │
│  │                                                                        │    │
│  │  Ideal: 1.0                                                            │    │
│  │                                                                        │    │
│  │  O:E = (# actual events) / (sum of predicted probabilities)            │    │
│  │                                                                        │    │
│  │  O:E > 1: More events than predicted (underestimation)                 │    │
│  │  O:E < 1: Fewer events than predicted (overestimation)                 │    │
│  │                                                                        │    │
│  │  Example: 56 glaucoma cases, model predicts total of 40 → O:E = 1.4    │    │
│  │                                                                        │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INTERPRETATION GUIDE                                                           │
│  ═══════════════════                                                            │
│                                                                                 │
│  │ Slope │ Intercept │ O:E │ Interpretation                                   │ │
│  │ ───── │ ───────── │ ─── │ ──────────────────────────────────────────────── │ │
│  │ 1.0   │ 0.0       │ 1.0 │ PERFECT calibration (rare in practice)           │ │
│  │ 0.8   │ 0.0       │ 1.0 │ Overconfident but mean-calibrated                 │ │
│  │ 1.0   │ 0.1       │ 1.2 │ Underestimates overall risk                       │ │
│  │ 0.7   │ -0.1      │ 0.8 │ Overconfident AND overestimates                   │ │
│                                                                                 │
│  Our results: slope ≈ 0.98, intercept ≈ -0.02, O:E ≈ 1.02                       │
│  → Near-perfect calibration!                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **What is calibration**: Discrimination vs calibration distinction
2. **Calibration plot**: Visual with ideal line and model curve
3. **Three metrics cards**: Slope, intercept, O:E with interpretations
4. **Interpretation table**: Combinations of values and meaning

## Text Content

### Title Text
"Calibration Metrics: Do Predictions Match Reality?"

### Caption
Calibration measures whether predicted probabilities match observed outcomes. Three metrics: (1) Calibration slope (ideal=1; <1 means overconfident, >1 means underconfident); (2) Calibration intercept (ideal=0; >0 means systematic underestimation); (3) O:E ratio (ideal=1; observed events / sum of predictions). Our pipeline achieves slope≈0.98, intercept≈-0.02, O:E≈1.02—near-perfect calibration.

## Prompts for Nano Banana Pro

### Style Prompt
Calibration plot with ideal line and actual curve. Three metric explanation cards stacked vertically. Interpretation table. Medical research aesthetic with clean statistical presentation.

### Content Prompt
Create a calibration metrics explanation:

**TOP - Concept**:
- "Discrimination: ranking" vs "Calibration: probability accuracy"

**MIDDLE - Calibration Plot**:
- X: predicted probability, Y: observed proportion
- Diagonal ideal line, actual model curve

**BOTTOM - Three Metrics**:
- Three cards: Slope (ideal 1.0), Intercept (ideal 0.0), O:E (ideal 1.0)
- Each with meaning of high/low values

**FOOTER - Table**:
- Combinations and interpretations

## Alt Text

Calibration metrics explanation diagram. Calibration plot shows ideal diagonal line (slope=1, intercept=0) vs actual model curve. Three metrics explained: calibration slope (ideal 1.0; <1 overconfident, >1 underconfident), calibration intercept (ideal 0.0; >0 underestimates, <0 overestimates), O:E ratio (ideal 1.0; observed events divided by predicted total). Interpretation table shows combinations of values and clinical meaning.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
