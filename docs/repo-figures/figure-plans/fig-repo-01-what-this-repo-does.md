# fig-repo-01: What Does This Repository Do?

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-01 |
| **Title** | What Does This Repository Do? |
| **Complexity Level** | L1 (ELI5) |
| **Target Persona** | All (especially PI, non-technical) |
| **Location** | Root README.md (hero image) |
| **Priority** | P1 (Critical) |

## Purpose

This is the HERO IMAGE for the repository. A non-technical ophthalmologist should understand within 5 seconds what this repository is about and why it matters for glaucoma screening.

## Key Message

"This repository tests whether AI foundation models can clean pupil measurement signals better than traditional methods, improving glaucoma screening accuracy."

## Visual Concept

**Split-panel comparison with narrative flow:**

```
LEFT SIDE                    RIGHT SIDE
┌─────────────────┐         ┌─────────────────┐
│  [Noisy pupil   │   →     │  [Clean pupil   │
│   signal with   │         │   signal after  │
│   blinks/noise] │         │   preprocessing]│
└─────────────────┘         └─────────────────┘
        ↓                           ↓
┌─────────────────┐         ┌─────────────────┐
│  "Traditional   │   vs    │  "AI Foundation │
│   Methods"      │         │   Models"       │
│  (LOF, SVM)     │         │  (MOMENT, UniTS)│
└─────────────────┘         └─────────────────┘
        ↓                           ↓
┌─────────────────────────────────────────────┐
│     Which gives BETTER glaucoma screening?  │
│            (AUROC: 0.86 → 0.91)             │
└─────────────────────────────────────────────┘
```

## Content Elements

### Required Elements
1. Raw PLR signal with visible artifacts (blinks, noise)
2. Cleaned PLR signal (smooth curve)
3. "Traditional" vs "AI Foundation Models" comparison
4. Outcome metric (AUROC improvement)
5. Glaucoma screening context (eye icon)

### Optional Elements
1. Time axis labels (0-7 seconds)
2. Foundation model logos (MOMENT, UniTS, TimesNet)
3. "507 subjects" data point

## Text Content

### Title Text
"Foundation Models for Pupil Signal Preprocessing"

### Labels/Annotations
- Raw signal: "Noisy pupil measurement (blinks, artifacts)"
- Clean signal: "Preprocessed signal ready for analysis"
- Traditional: "Traditional Methods: LOF, SVM, interpolation"
- AI: "Foundation Models: MOMENT, UniTS, TimesNet"
- Outcome: "Result: Screening accuracy 0.86 → 0.91 AUROC"

### Caption (for embedding)
This repository evaluates whether time series foundation models can improve pupillary light reflex preprocessing for glaucoma screening compared to traditional outlier detection methods.

## Technical Notes

- **Data source**: SERI PLR Glaucoma dataset (Najjar et al. 2023)
- **Dependencies**: None (standalone hero image)
- **Updates needed**: If AUROC values change

## Prompts for Nano Banana Pro

### Style Prompt
Clean, professional scientific infographic. Medical/healthcare aesthetic with soft blues and greens. Minimalist design with clear visual hierarchy. No decorative elements. Accessible color palette (colorblind-friendly). White background.

### Content Prompt
Create a horizontal infographic showing the transformation of a noisy pupil measurement signal (left, with visible spikes representing blinks) to a clean signal (right, smooth curve). In the middle, show two paths: "Traditional Methods" (simple icons) and "AI Foundation Models" (neural network icon). At the bottom, show the outcome: improved glaucoma screening accuracy. Include a small eye icon to represent ophthalmology context.

### Refinement Notes
- Ensure the signal curves look like real pupillometry data (constriction then recovery)
- The "noise" should look like blink artifacts (sharp vertical drops)
- Keep text minimal - this should work even at small sizes

## Alt Text

Infographic showing raw pupil signal with blink artifacts being cleaned by traditional methods or AI foundation models, resulting in improved glaucoma screening accuracy.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
