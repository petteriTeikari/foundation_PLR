# fig-repo-19: Subject Stratification: 507 vs 208

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-19 |
| **Title** | Subject Stratification: 507 vs 208 |
| **Complexity Level** | L1 (Concept explanation) |
| **Target Persona** | All |
| **Location** | docs/user-guide/, Root README |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Clarify why subject counts differ between preprocessing (507) and classification (208)—this is study design, not data loss.

## Key Message

"507 subjects for preprocessing evaluation, 208 for classification. The difference is LABELS, not lost data."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SUBJECT STRATIFICATION: 507 vs 208                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TOTAL DATASET: 507 SUBJECTS                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                         │   │
│  │    ALL 507 SUBJECTS HAVE:                                               │   │
│  │    ✓ Raw PLR recordings (1981 timepoints each)                          │   │
│  │    ✓ Ground truth outlier masks (human-annotated blinks)                │   │
│  │    ✓ Ground truth denoised signals (human-supervised)                   │   │
│  │                                                                         │   │
│  │    Used for: PREPROCESSING EVALUATION                                   │   │
│  │    (outlier detection, imputation quality)                              │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌────────────────────────────────┬────────────────────────────────────────┐   │
│  │         LABELED (208)          │          UNLABELED (299)               │   │
│  ├────────────────────────────────┼────────────────────────────────────────┤   │
│  │                                │                                        │   │
│  │  👁️ 152 Healthy Controls       │  ❓ No disease labels                  │   │
│  │  👁️  56 Glaucoma Patients      │     (preprocessing only)               │   │
│  │                                │                                        │   │
│  │  Used for: CLASSIFICATION      │  Used for: PREPROCESSING               │   │
│  │  (train classifier, evaluate)  │  (outlier/imputation benchmarks)       │   │
│  │                                │                                        │   │
│  │  152 + 56 = 208 subjects       │  299 subjects                          │   │
│  │                                │                                        │   │
│  └────────────────────────────────┴────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  COMMON CONFUSION                                                               │
│  ════════════════                                                               │
│                                                                                 │
│  ❌ WRONG: "We lost 299 subjects"                                               │
│  ✅ RIGHT: "299 subjects lack disease labels, so can't train classifiers"       │
│                                                                                 │
│  All 507 subjects contribute to preprocessing benchmarks.                       │
│  Only 208 labeled subjects can be used for classification.                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  REPORTING CORRECTLY                                                            │
│  ═══════════════════                                                            │
│                                                                                 │
│  "Preprocessing methods evaluated on N=507 subjects"                            │
│  "Classification models trained/evaluated on N=208 labeled subjects"            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Nested rectangle diagram**: 507 total → 208 labeled + 299 unlabeled
2. **What all 507 have**: Ground truth masks, denoised signals
3. **Labeled breakdown**: 152 controls + 56 glaucoma
4. **Common confusion callout**: "Lost" vs "unlabeled"
5. **Correct reporting examples**: How to cite subject counts

## Text Content

### Title Text
"Subject Stratification: 507 vs 208"

### Caption
All 507 subjects have ground truth for preprocessing evaluation (outlier masks, denoised signals). Only 208 subjects (152 healthy + 56 glaucoma) have disease labels for classification. The 299 unlabeled subjects aren't "lost"—they contribute to preprocessing benchmarks but can't be used to train classifiers.

## Prompts for Nano Banana Pro

### Style Prompt
Nested rectangle diagram showing dataset stratification. Large outer box (507 subjects) containing two inner boxes (208 labeled, 299 unlabeled). Eye icons for subjects. Clear breakdown of 152+56=208. "Confusion callout" box with X and checkmark. Clean, informative, medical research context. Matte colors.

### Content Prompt
Create a subject stratification diagram:

**TOP - Full Dataset**:
- Large rectangle: "507 SUBJECTS"
- List what all have: ground truth masks, denoised signals

**MIDDLE - Stratification**:
- Two side-by-side rectangles inside the large one
- LEFT (blue): "208 LABELED" → 152 Healthy + 56 Glaucoma
- RIGHT (gray): "299 UNLABELED" → preprocessing only

**BOTTOM - Confusion Callout**:
- X mark: "We lost 299 subjects" (wrong)
- Checkmark: "299 lack labels but contribute to preprocessing" (right)

**FOOTER - Reporting**:
- Two example sentences for correct citation

## Alt Text

Subject stratification diagram showing 507 total subjects. All have ground truth for preprocessing. Split into 208 labeled (152 healthy controls + 56 glaucoma patients) used for classification, and 299 unlabeled used only for preprocessing benchmarks. Clarifies that unlabeled subjects aren't "lost" but simply lack disease labels.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/user-guide/
