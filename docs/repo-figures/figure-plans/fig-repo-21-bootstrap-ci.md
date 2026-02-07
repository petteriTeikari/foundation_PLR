# fig-repo-21: Bootstrap Confidence Intervals: 1000 Iterations

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-21 |
| **Title** | Bootstrap Confidence Intervals: 1000 Iterations |
| **Complexity Level** | L2 (Statistical concept) |
| **Target Persona** | Biostatistician, Research Scientist |
| **Location** | docs/concepts-for-researchers.md |
| **Priority** | P1 |
| **Aspect Ratio** | 16:10 |

## Purpose

Explain bootstrap methodology for confidence intervals in an accessible way—why 1000 iterations, how CIs are computed.

## Key Message

"One AUROC number is a guess. 1000 bootstrap samples give us confidence: the true AUROC is likely between 0.851 and 0.955."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BOOTSTRAP CONFIDENCE INTERVALS: 1000 ITERATIONS               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE PROBLEM WITH ONE NUMBER                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  Test set: 208 subjects                                                         │
│  AUROC = 0.913                                                                  │
│                                                                                 │
│  But how confident are we? What if we had different test subjects?              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  THE BOOTSTRAP SOLUTION                                                         │
│  ══════════════════════                                                         │
│                                                                                 │
│  REPEAT 1000 TIMES:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                          │  │
│  │  1. Sample 208 subjects WITH REPLACEMENT (some appear twice, some not)   │  │
│  │       👤👤👤👤👤...👤  (208 subjects, but with repeats)                  │  │
│  │                                                                          │  │
│  │  2. Compute AUROC on this sample                                         │  │
│  │       Iteration 1: 0.907                                                 │  │
│  │       Iteration 2: 0.921                                                 │  │
│  │       Iteration 3: 0.895                                                 │  │
│  │       ...                                                                │  │
│  │       Iteration 1000: 0.918                                              │  │
│  │                                                                          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  3. After 1000 iterations, we have a DISTRIBUTION:                              │
│                                                                                 │
│          0.80    0.85    0.90    0.95    1.00                                   │
│            │       │       │       │       │                                    │
│            │       ┌───────┴───────┐       │                                    │
│            │      ╱                 ╲      │                                    │
│            │    ╱                     ╲    │                                    │
│            │  ╱                         ╲  │                                    │
│            │╱█████████████████████████████╲│                                    │
│                  ▲                   ▲                                          │
│                0.851               0.955                                        │
│               (2.5%)              (97.5%)                                        │
│                                                                                 │
│            └──────── 95% CI ────────┘                                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INTERPRETATION                                                                 │
│  ══════════════                                                                 │
│                                                                                 │
│  Mean AUROC: 0.913                                                              │
│  95% CI: [0.851, 0.955]                                                         │
│                                                                                 │
│  "We're 95% confident the true AUROC lies between 0.851 and 0.955"              │
│                                                                                 │
│  Wide CI = uncertain → Need more data                                           │
│  Narrow CI = confident → Results are stable                                     │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WHY 1000 ITERATIONS?                                                           │
│  ════════════════════                                                           │
│                                                                                 │
│  Too few (50):   CI edges are noisy, unreliable                                 │
│  Just right (1000): Smooth CI, good precision                                   │
│  More (10000):   Diminishing returns, 10× slower                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

1. **Problem statement**: One number isn't enough
2. **Bootstrap process**: Resample with replacement → compute → repeat
3. **Distribution histogram**: Bell curve with CI bounds marked
4. **Interpretation**: What the CI means in plain language
5. **Why 1000**: Trade-off between precision and computation

## Text Content

### Title Text
"Bootstrap Confidence Intervals: 1000 Iterations"

### Caption
A single AUROC value (0.913) doesn't tell us how confident we should be. Bootstrap resampling creates 1000 virtual test sets by sampling with replacement, computing AUROC on each. The resulting distribution gives us a 95% confidence interval [0.851, 0.955]—we're 95% confident the true AUROC falls in this range. 1000 iterations balance precision with computation time.

## Prompts for Nano Banana Pro

### Style Prompt
Statistical explanation with histogram visualization. Step-by-step process diagram. Bell curve distribution with vertical CI bounds. Numbers and percentages clearly labeled. Friendly but accurate statistical presentation. Economist-style data visualization. Matte colors, medical research context.

### Content Prompt
Create a bootstrap explanation diagram:

**TOP - Problem**:
- "One AUROC = 0.913, but how confident are we?"

**MIDDLE - Process Box**:
- Three numbered steps: Sample → Compute → Repeat 1000x
- Show example iteration numbers (0.907, 0.921, etc.)

**CENTER - Histogram**:
- Bell curve distribution of 1000 AUROC values
- Vertical lines at 0.851 (2.5%) and 0.955 (97.5%)
- Shaded 95% CI region

**BOTTOM - Interpretation**:
- Mean: 0.913, CI: [0.851, 0.955]
- Plain language: "95% confident true AUROC is in this range"
- Why 1000: "Too few = noisy, too many = slow, 1000 = just right"

## Alt Text

Bootstrap confidence interval explanation. Problem: single AUROC (0.913) lacks confidence measure. Process: resample 208 subjects with replacement 1000 times, compute AUROC each time. Result: histogram distribution with mean 0.913 and 95% CI bounds at 0.851 (2.5th percentile) and 0.955 (97.5th percentile). Interpretation: 95% confident true AUROC falls within CI range.

## Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in docs/concepts-for-researchers.md
