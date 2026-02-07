# fig-repo-02: The Preprocessing Pipeline

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-02 |
| **Title** | The Preprocessing Pipeline |
| **Complexity Level** | L2 (Process overview) |
| **Target Persona** | All |
| **Location** | Root README.md, ARCHITECTURE.md |
| **Priority** | P1 (Critical) |

## Purpose

Show the complete data flow from raw pupil signal to classification prediction, highlighting where preprocessing choices matter.

## Key Message

"Preprocessing happens in two stages (outlier detection → imputation), and choices at each stage affect final prediction quality."

## Visual Concept

**Horizontal pipeline with error propagation emphasis:**

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐
│ Raw PLR  │ → │   OUTLIER    │ → │ IMPUTATION │ → │   FEATURE   │ → │ CLASSIFICATION│
│  Signal  │    │  DETECTION   │    │            │    │ EXTRACTION  │    │   (CatBoost) │
│          │    │  (11 methods)│    │ (8 methods)│    │ (handcraft) │    │              │
└──────────┘    └──────────────┘    └────────────┘    └─────────────┘    └──────────────┘
                      │                    │                                    │
                      ▼                    ▼                                    ▼
               ┌────────────┐       ┌────────────┐                      ┌────────────┐
               │ Errors here│  →   │propagate   │    →                 │affect AUROC│
               │            │       │here        │                      │  & metrics │
               └────────────┘       └────────────┘                      └────────────┘
```

## Content Elements

### Required Elements
1. Five pipeline stages (raw → outlier → imputation → features → classification)
2. Method counts at each stage (11 outlier, 8 imputation, 1 classifier)
3. Error propagation arrows showing downstream effects
4. Final output (AUROC, calibration metrics)

### Optional Elements
1. Example method names at each stage
2. Ground truth baseline indicator
3. STRATOS metrics list

## Text Content

### Title Text
"The Preprocessing Pipeline: Where Choices Matter"

### Labels/Annotations
- Stage 1: "Raw PLR Signal (N=507 subjects, ~1M timepoints)"
- Stage 2: "Outlier Detection (11 methods: LOF, MOMENT, UniTS, TimesNet...)"
- Stage 3: "Imputation (8 methods: SAITS, CSDI, MOMENT, linear...)"
- Stage 4: "Feature Extraction (amplitude bins + latency)"
- Stage 5: "Classification (N=208 labeled: 152 control, 56 glaucoma)"
- Error arrow: "Errors propagate downstream"
- Output: "STRATOS Metrics: AUROC, Brier, Calibration, Net Benefit"
- Baseline note: "Ground truth (pupil-gt) = human-annotated baseline for comparison"

### Caption (for embedding)
The preprocessing pipeline shows how raw PLR signals flow through outlier detection and imputation before feature extraction and classification. Errors at early stages propagate through the entire pipeline.

## Technical Notes

- **Data source**: Pipeline architecture from ARCHITECTURE.md, methods.tex
- **Subject counts**: N=507 for preprocessing (outlier + imputation), N=208 for classification (152 control, 56 glaucoma)
- **Ground truth**: `pupil-gt` is human-annotated baseline for comparison, NOT a detection/imputation method
- **Real methods**: LOF, OneClassSVM, MOMENT, UniTS, TimesNet (outlier); SAITS, CSDI, MOMENT, linear (imputation)
- **Dependencies**: fig-repo-01 provides context
- **Updates needed**: If method counts change (currently 11 outlier, 8 imputation)

## Prompts for Nano Banana Pro

### Style Prompt
Technical process diagram. Clean horizontal flow with consistent box sizes. Use subtle gradients to indicate data quality (noisy on left, clean on right). Arrow connections between stages. Error propagation shown with red-tinted downward arrows. Professional blue/gray color scheme.

### Content Prompt
Create a horizontal 5-stage pipeline diagram for pupil signal preprocessing:
1. Raw Signal (wavy line icon, "N=507 subjects")
2. Outlier Detection box (magnifying glass icon, "11 methods: LOF, MOMENT, UniTS...")
3. Imputation box (puzzle piece icon, "8 methods: SAITS, CSDI, MOMENT...")
4. Feature Extraction box (bar chart icon, "handcrafted features")
5. Classification box (decision tree icon, "N=208 labeled subjects")

Below stages 2-3, show downward arrows labeled "errors propagate" leading to stage 5's outcome. At the end, show output metrics.

Include a small annotation box: "Ground truth (pupil-gt) = human-annotated baseline, not a detection method"

### Refinement Notes
- Emphasize the error propagation concept visually
- Make "11 methods" and "8 methods" prominent
- The pipeline should read left-to-right naturally

## Alt Text

Five-stage horizontal pipeline showing raw PLR signal flowing through outlier detection, imputation, feature extraction, to classification, with error propagation arrows showing downstream effects.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
