# Documentation Hierarchy Improvement Plan

## The Problem

Currently, metric explanations are disconnected:
- `src/stats/README.md` lists functions but doesn't explain **what the metrics mean**
- `docs/tutorials/stratos-metrics.md` explains the academic framework but is disconnected from code
- No clear hierarchy between "how to interpret" and "why these metrics matter academically"

## The Solution: Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 3: Academic Framework (docs/tutorials/)                       │
│  "WHY these metrics - academic recommendations"                      │
│                                                                      │
│  ├── stratos-metrics.md       ← Van Calster 2024 framework          │
│  └── tripod-ai.md             ← TRIPOD+AI reporting checklist       │
└─────────────────────────────────────────────────────────────────────┘
                              ↑ cross-references
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 2: Metric Interpretation (src/stats/README.md)                │
│  "WHAT each metric means and HOW to interpret it"                    │
│                                                                      │
│  Sections:                                                           │
│  ├── Discrimination Metrics (AUROC, sensitivity, specificity)       │
│  ├── Calibration Metrics (slope, intercept, O:E ratio)              │
│  ├── Overall Performance (Brier, Scaled Brier)                       │
│  ├── Clinical Utility (Net Benefit, DCA)                             │
│  ├── Model Stability (pminternal instability)                        │
│  └── Uncertainty Analysis (AURC, selective classification)           │
│                                                                      │
│  Each section has:                                                   │
│  • Visual figure explaining the metric                               │
│  • "What good values look like"                                      │
│  • "What bad values mean"                                            │
│  • Code examples                                                     │
│  • Cross-reference to academic framework                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↑ documents
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: Code Implementation (src/stats/*.py)                       │
│  "HOW to compute each metric"                                        │
│                                                                      │
│  ├── classifier_metrics.py      → AUROC, sensitivity, specificity   │
│  ├── calibration_extended.py    → slope, intercept, O:E ratio       │
│  ├── scaled_brier.py            → Brier, IPA                        │
│  ├── clinical_utility.py        → Net Benefit, DCA                  │
│  ├── pminternal_wrapper.py      → instability via R                 │
│  └── uncertainty_propagation.py → AURC, selective classification    │
└─────────────────────────────────────────────────────────────────────┘
```

## Action Items

### 1. Restructure `src/stats/README.md`

Add detailed "Visual Guide" sections for each metric category:

**Discrimination Section:**
- Figure: `fig-repo-27-cd-diagram-reading.jpg` (ranking interpretation)
- Explain: AUROC = probability of correct ranking
- Good: > 0.8, Excellent: > 0.9
- Bad: < 0.7 = barely better than random
- Cross-ref: [STRATOS Guidelines](../tutorials/stratos-metrics.md#1-discrimination)

**Calibration Section:**
- Figure: `fig-repo-39-calibration-explained.jpg`
- Explain: Do probabilities match outcomes?
- Slope: should be ~1.0 (spread of predictions)
- Intercept: should be ~0.0 (average bias)
- O:E ratio: should be ~1.0 (observed/expected)
- Cross-ref: [STRATOS Guidelines](../tutorials/stratos-metrics.md#2-calibration)

**Clinical Utility Section:**
- Figure: `fig-repo-40-net-benefit-dca.jpg`
- Explain: Is the model useful for decisions?
- Net Benefit formula and interpretation
- DCA curves: compare to "treat all" and "treat none"
- Cross-ref: [STRATOS Guidelines](../tutorials/stratos-metrics.md#5-clinical-utility)

**Uncertainty Section:**
- Figure: `fig-repo-27f-how-to-read-risk-coverage.jpg`
- Explain: When should model abstain?
- AURC: lower is better
- Selective classification trade-off
- Cross-ref: [Reading Plots Guide](../tutorials/reading-plots.md#risk-coverage-aurc)

**Instability Section:**
- Figure: `fig-repo-27d-how-to-read-instability-plot.jpg`
- Explain: Are predictions stable?
- pminternal interpretation
- Cross-ref: [Reading Plots Guide](../tutorials/reading-plots.md#instability-plots)

### 2. Create `docs/tutorials/tripod-ai.md`

Add TRIPOD+AI reporting checklist as academic reference:
- What to report in methods section
- What to report in results section
- Common mistakes to avoid
- Cross-references to where we implement each item

### 3. Update `docs/tutorials/stratos-metrics.md`

Add cross-references back to code:
- Each metric section links to `src/stats/README.md#section`
- "Implementation details: see [src/stats/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/README.md)"

### 4. Update Cross-References

**From code README → tutorials:**
```markdown
*For academic justification, see [STRATOS Guidelines](../tutorials/stratos-metrics.md)*
```

**From tutorials → code README:**
```markdown
*For implementation details, see [src/stats/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/README.md)*
```

## Figures Needed

| Section | Figure | Status |
|---------|--------|--------|
| Discrimination | CD diagram | ✅ `fig-repo-27-cd-diagram-reading.jpg` |
| Calibration | Calibration curve | ✅ `fig-repo-39-calibration-explained.jpg` |
| Clinical Utility | DCA | ✅ `fig-repo-40-net-benefit-dca.jpg` |
| Uncertainty | Risk-coverage | ✅ `fig-repo-27f-how-to-read-risk-coverage.jpg` |
| Instability | pminternal | ✅ `fig-repo-27d-how-to-read-instability-plot.jpg` |
| Overall | Brier score | ❌ Need to create or use existing |

## Expected Outcome

A researcher exploring `src/stats/`:
1. Opens `README.md`
2. Sees visual guide explaining each metric category
3. Understands WHAT each metric means
4. Finds code examples
5. Clicks cross-reference to understand WHY (academic framework)

A researcher reading the paper:
1. Opens `docs/tutorials/stratos-metrics.md`
2. Understands the STRATOS framework
3. Clicks cross-reference to see HOW we implemented it
4. Can verify our implementation matches guidelines
