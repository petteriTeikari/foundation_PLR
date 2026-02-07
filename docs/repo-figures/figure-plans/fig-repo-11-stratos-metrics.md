# fig-repo-11: STRATOS Metrics Explained

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-11 |
| **Title** | STRATOS: Beyond AUROC |
| **Complexity Level** | L2 (For biostatisticians) |
| **Target Persona** | Biostatistician, Research Scientist |
| **Location** | docs/, README.md |
| **Priority** | P2 (High) |

## Purpose

Explain why we report multiple metrics (not just AUROC) and what each metric tells us about model quality.

## Key Message

"AUROC alone isn't enough. STRATOS guidelines require discrimination, calibration, AND clinical utility metrics for proper model evaluation."

## Visual Concept

**Metric categories with what they measure:**

```
┌─────────────────────────────────────────────────────────────────┐
│              STRATOS: COMPREHENSIVE MODEL EVALUATION            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DISCRIMINATION                                         │   │
│  │  "Does the model rank patients correctly?"              │   │
│  │  ┌─────────┐                                            │   │
│  │  │ AUROC   │  0.91 ████████████████████░░░░ 1.0        │   │
│  │  └─────────┘                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CALIBRATION                                            │   │
│  │  "Do predicted probabilities match reality?"            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Slope: 0.98 │  │ Intercept:  │  │ Brier: 0.13 │     │   │
│  │  │ (ideal: 1)  │  │ -0.01       │  │ (lower=better)│    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CLINICAL UTILITY                                       │   │
│  │  "Is the model useful for decisions?"                   │   │
│  │  ┌───────────────────────────────────────────────────┐ │   │
│  │  │ Net Benefit at 15% threshold: 0.199               │ │   │
│  │  │ (How much better than treating everyone?)         │ │   │
│  │  └───────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ⚠️  AUROC alone can be misleading!                            │
│      A model with high AUROC but poor calibration              │
│      gives wrong probability estimates.                        │
└─────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Three metric categories (discrimination, calibration, utility)
2. Key question each category answers
3. Specific metrics with example values
4. Warning about AUROC-only evaluation

### Optional Elements
1. Visual calibration curve example
2. DCA curve example
3. Link to Van Calster 2024 paper

## Text Content

### Title Text
"STRATOS: Complete Model Evaluation"

### Labels/Annotations
- Discrimination: "Does it rank correctly?" → AUROC
- Calibration: "Are probabilities accurate?" → Slope, Intercept, Brier
- Clinical Utility: "Is it useful?" → Net Benefit
- Warning: "AUROC alone can be misleading"

### Caption (for embedding)
STRATOS guidelines require evaluating models on discrimination (AUROC), calibration (slope, Brier), and clinical utility (Net Benefit) - not just AUROC alone.

## Prompts for Nano Banana Pro

### Style Prompt
Educational infographic with medical/statistical aesthetic. Three stacked panels with distinct colors. Include metric values as examples. Warning callout at bottom. Clean, professional.

### Content Prompt
Create a three-panel infographic for STRATOS metrics:
1. DISCRIMINATION panel: AUROC with progress bar visual
2. CALIBRATION panel: Three boxes for slope, intercept, Brier score
3. CLINICAL UTILITY panel: Net Benefit with explanation

At bottom: Warning box about AUROC-only evaluation being incomplete.

### Refinement Notes
- Make it accessible to biostatisticians who may not know all terms
- Include the question each metric answers
- Emphasize the warning about AUROC-only evaluation

## Alt Text

Three-panel infographic showing STRATOS metrics: Discrimination (AUROC), Calibration (slope, intercept, Brier), and Clinical Utility (Net Benefit), with warning that AUROC alone is insufficient.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
