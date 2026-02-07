# fig-trans-01: When NOT to Impute

**Status**: 📋 PLANNED
**Tier**: 1 - Honest Limitations
**Target Persona**: Humanitarian/Logistics, EHR professionals

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-01 |
| Type | Comparison diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Show professionals when imputation is **harmful** - specifically in domains where:
- Missing data encodes real events (not measurement errors)
- "Artifacts" are actually signals
- Temporal gaps are informative

---

## 3. Key Message

> "Not all missing data is corruption. In logistics and healthcare, gaps often encode reality - a warehouse closure, a conflict zone, a patient who didn't visit. Imputing over these events destroys valuable information."

---

## 4. Literature Sources

| Source | Key Finding |
|--------|-------------|
| Van Ness et al. 2023 | "Missingness indicators outperform imputation when missingness is informative" |
| McTavish et al. 2024 | "Perfect imputation can reduce best possible model performance" (Proposition 3.1) |
| Le Morvan et al. 2021 | "Impute-then-predict models sacrifice expressiveness with perfect imputation" |
| Sperrin et al. 2020 | "In predictive modeling, inclusion of missingness indicators is recommended" |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  WHEN TO IMPUTE vs WHEN NOT TO                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ✓ IMPUTE (PLR)                    ✗ DON'T IMPUTE (Logistics)             │
│  ─────────────────                 ───────────────────────────             │
│                                                                            │
│  [Pupil trace with blink]          [Supply chain with gap]                 │
│                                                                            │
│     ∧     ∧                          ────────────────────────              │
│    / \   / \                                   │                           │
│   /   \_/   \──────────                        │ ← Gap = warehouse          │
│  /  ↑blink↑  \                                 │   closed due to            │
│ /             \                                │   missile strike           │
│                                                                            │
│  The blink is NOT data.            The gap IS data.                        │
│  The pupil was there,              No shipments occurred.                  │
│  we just couldn't see it.          Imputing = inventing fake deliveries.   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DECISION CRITERIA                                                         │
│                                                                            │
│  ┌─────────────────────────────┐                                           │
│  │ Is the gap a measurement    │                                           │
│  │ error or sensor failure?    │                                           │
│  └──────────┬──────────────────┘                                           │
│             │                                                              │
│      YES    │    NO                                                        │
│      ↓      │    ↓                                                         │
│  ┌──────────┴──────────┐                                                   │
│  │ IMPUTE              │     ┌────────────────────────────┐                │
│  │ Reconstruct the     │     │ DON'T IMPUTE               │                │
│  │ true underlying     │     │ Use missingness as feature │                │
│  │ signal              │     │ (M-GAM, GMAN approach)     │                │
│  └─────────────────────┘     └────────────────────────────┘                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Content Elements

### Left Panel: Imputation Makes Sense (PLR)
- Pupil trace with blink artifact
- Ground truth underlying signal (dashed)
- Label: "The pupil was there - we just couldn't measure it"
- Arrow: "Impute to recover true physiological signal"

### Right Panel: Imputation is Harmful (Logistics)
- Supply chain time series with gap
- External event annotation (conflict, holiday, closure)
- Label: "The gap IS the data - no deliveries occurred"
- Warning: "Imputing = inventing fake events"

### Bottom: Decision Framework
- Simple flowchart
- Key question: "Is the gap a measurement error?"
- Two paths: Impute vs Use Missingness as Feature

---

## 7. Text Content

### Title
"When NOT to Impute: Missingness as Information"

### Caption
"In biomedical signal processing, missing data typically represents measurement artifacts (blinks, sensor failures) that obscure a real underlying signal. Imputation is appropriate. However, in logistics, EHRs, and event-driven domains, gaps often encode real events—a closed warehouse, a conflict zone, a patient who didn't visit. Imputing over these gaps destroys valuable information. The M-GAM and GMAN approaches treat missingness as a feature rather than a bug."

### Callout Box
"Proposition (McTavish 2024): Perfect imputation can reduce the best possible model performance. When missingness is informative, the Bayes-optimal model using missingness as a value outperforms the Bayes-optimal model using perfectly imputed data."

---

## 8. Nano Banana Pro Prompts

### Primary Prompt
```
Create a professional scientific comparison diagram titled "When NOT to Impute: Missingness as Information"

LEFT SIDE (green theme):
- Show a simplified pupil diameter trace over time
- Include a gap labeled "blink" with dashed line showing "true signal underneath"
- Label: "Biomedical: Gap = measurement error"
- Subtitle: "The pupil was there - we just couldn't see it"
- Icon: checkmark with "IMPUTE"

RIGHT SIDE (red theme):
- Show a supply chain time series with a real gap
- Annotate the gap with "Warehouse closed - conflict zone"
- Label: "Logistics: Gap = real event"
- Subtitle: "No deliveries occurred - this IS the data"
- Icon: X mark with "DON'T IMPUTE"

CENTER BOTTOM:
- Simple decision tree: "Is the gap a measurement error?" → YES: Impute → NO: Use missingness as feature

Style:
- Clean, academic, no decorative elements
- Economist magazine aesthetic
- Professional color palette (not garish)
- Sans-serif fonts
- High information density
```

### Refinement Prompt
```
Refine the comparison diagram to emphasize the key insight:

Add subtle visual metaphor:
- Left: pupil trace as a continuous ribbon, blink as temporary occlusion
- Right: supply chain as discrete blocks, gap as genuine absence

Ensure colorblind safety:
- Use blue/orange instead of green/red if needed
- Add pattern fills as secondary encoding

Add citation callout box with McTavish 2024 proposition about perfect imputation reducing model performance.
```

---

## 9. Alt Text

"A comparison diagram with two panels. Left panel shows a biomedical signal (pupil trace) with a gap caused by a blink, labeled 'IMPUTE - the signal existed, we just couldn't measure it'. Right panel shows a logistics time series with a gap caused by a warehouse closure in a conflict zone, labeled 'DON'T IMPUTE - the gap IS the information'. A decision flowchart at the bottom asks 'Is the gap a measurement error?' with paths to 'Impute' for yes and 'Use missingness as feature' for no."

---

## 10. Related Figures

- fig-trans-03: Domain Fit Matrix (expands on which domains suit imputation)
- fig-trans-11: GMAN for Event-Conditioned Time Series (alternative approach)
- fig-trans-12: M-GAM: Missing Values as Features (technical details)

---

## 11. Validation Checklist

- [ ] Does it respect logistics/humanitarian professionals? (Not patronizing)
- [ ] Is the biomedical case accurate? (Blinks are artifacts)
- [ ] Is the McTavish citation accurate? (Proposition 3.1)
- [ ] Colorblind safe?
- [ ] Alt text complete?

---

*Last updated: 2026-02-01*
