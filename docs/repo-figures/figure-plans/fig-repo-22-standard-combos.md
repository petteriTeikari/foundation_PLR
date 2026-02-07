# fig-repo-22: The 4 Standard Combos

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-22 |
| **Title** | The 4 Standard Combos: Comparing Preprocessing Pipelines |
| **Complexity Level** | L2 (Concept explanation) |
| **Target Persona** | All |
| **Location** | docs/user-guide/, Root README |
| **Priority** | P0 |
| **Aspect Ratio** | 16:10 |

## Purpose

Introduce the 4 standard preprocessing combinations used throughout all figures—these are the "cast of characters" for every visualization.

## Key Message

"Four preprocessing pipelines represent the key comparison: ground truth (ceiling), best FM ensemble, best single FM, and traditional methods. Every figure compares these same 4."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    THE 4 STANDARD COMBOS                                        │
│                    ═══════════════════════                                      │
│                    Consistent comparison across all figures                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   COMBO 1: GROUND TRUTH (Ceiling Performance)           AUROC: 0.911   │   │
│  │   ════════════════════════════════════════════                         │   │
│  │                                                                         │   │
│  │   Outlier:     pupil-gt         (human-annotated masks)                │   │
│  │   Imputation:  pupil-gt         (human-supervised reconstruction)      │   │
│  │   Classifier:  CatBoost         (fixed across all)                     │   │
│  │                                                                         │   │
│  │   🎯 "What's possible with perfect preprocessing"                      │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   COMBO 2: BEST ENSEMBLE (Highest Automated)            AUROC: 0.913   │   │
│  │   ══════════════════════════════════════════                           │   │
│  │                                                                         │   │
│  │   Outlier:     ensemble-LOF-MOMENT-...  (7 methods combined)           │   │
│  │   Imputation:  CSDI                     (deep learning)                │   │
│  │   Classifier:  CatBoost                                                │   │
│  │                                                                         │   │
│  │   🏆 "Actually BEATS ground truth by combining methods"                │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   COMBO 3: BEST SINGLE FM                               AUROC: 0.910   │   │
│  │   ═══════════════════════════                                          │   │
│  │                                                                         │   │
│  │   Outlier:     MOMENT-gt-finetune    (single foundation model)         │   │
│  │   Imputation:  SAITS                 (deep learning imputation)        │   │
│  │   Classifier:  CatBoost                                                │   │
│  │                                                                         │   │
│  │   🔬 "Best we can do with ONE foundation model"                        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │   COMBO 4: TRADITIONAL BASELINE                         AUROC: 0.860   │   │
│  │   ═════════════════════════════                                        │   │
│  │                                                                         │   │
│  │   Outlier:     LOF                   (Local Outlier Factor)            │   │
│  │   Imputation:  SAITS                 (deep learning imputation)        │   │
│  │   Classifier:  TabPFN                (varied for this baseline)        │   │
│  │                                                                         │   │
│  │   📊 "Traditional ML without foundation models"                        │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY THESE 4?                                                                   │
│  ═══════════                                                                    │
│                                                                                 │
│  ✓ Ground truth = theoretical ceiling (what's achievable)                       │
│  ✓ Ensemble = practical ceiling (automated, beats GT!)                          │
│  ✓ Single FM = simplest FM approach                                             │
│  ✓ Traditional = pre-FM baseline                                                │
│                                                                                 │
│  More than 4 curves = visual clutter. These 4 tell the complete story.          │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHERE YOU'LL SEE THESE                                                         │
│  ══════════════════════                                                         │
│                                                                                 │
│  📈 ROC curves           Always these 4 colors                                  │
│  📊 Calibration plots    Same 4 pipelines                                       │
│  📉 DCA curves           Same 4 pipelines                                       │
│  📋 All comparison tables Ground truth always included                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Four combo cards**: Each showing outlier + imputation + classifier
2. **AUROC badges**: Performance for each combo
3. **One-line summaries**: What each combo represents
4. **Why 4?**: Justification for limiting to 4 curves
5. **Usage examples**: Where these appear in figures

## Text Content

### Title Text
"The 4 Standard Combos: Consistent Comparison Across All Figures"

### Caption
Every visualization in this repository compares the same 4 preprocessing pipelines: (1) Ground truth (human-annotated, AUROC 0.911) shows the performance ceiling; (2) Best ensemble (7 methods combined, AUROC 0.913) actually exceeds ground truth; (3) Best single FM (MOMENT + SAITS, AUROC 0.910) shows what one foundation model achieves; (4) Traditional baseline (LOF + SAITS, AUROC 0.860) represents pre-FM methods. CatBoost classifier is fixed for fair preprocessing comparison.

## Prompts for Nano Banana Pro

### Style Prompt
Four stacked cards showing preprocessing combinations. Each card in distinct color matching figure palette. AUROC badges on right side. Icons for each step (outlier, imputation, classifier). Clean medical research aesthetic. Matte colors, subtle shadows on cards.

### Content Prompt
Create a "4 Standard Combos" comparison card:

**CARD 1 (Blue/Green)**: "GROUND TRUTH"
- Three rows: pupil-gt → pupil-gt → CatBoost
- Badge: "AUROC: 0.911"
- Tag: "Ceiling Performance"

**CARD 2 (Orange)**: "BEST ENSEMBLE"
- Three rows: ensemble → CSDI → CatBoost
- Badge: "AUROC: 0.913"
- Tag: "Beats Ground Truth!"

**CARD 3 (Purple)**: "BEST SINGLE FM"
- Three rows: MOMENT-gt-finetune → SAITS → CatBoost
- Badge: "AUROC: 0.910"
- Tag: "One FM Approach"

**CARD 4 (Gray)**: "TRADITIONAL"
- Three rows: LOF → SAITS → TabPFN
- Badge: "AUROC: 0.860"
- Tag: "Baseline"

**FOOTER**:
- "These 4 appear in every figure for consistent comparison"

## Alt Text

Four preprocessing combination cards showing the standard comparisons used across all figures. Card 1: Ground truth (pupil-gt + pupil-gt + CatBoost, AUROC 0.911). Card 2: Best ensemble (7-method ensemble + CSDI + CatBoost, AUROC 0.913). Card 3: Best single FM (MOMENT-gt-finetune + SAITS + CatBoost, AUROC 0.910). Card 4: Traditional baseline (LOF + SAITS + TabPFN, AUROC 0.860).

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/user-guide/
