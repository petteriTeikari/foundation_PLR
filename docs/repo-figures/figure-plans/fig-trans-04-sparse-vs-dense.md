# fig-trans-04: Sparse vs Dense: Different Beasts

**Status**: 📋 PLANNED
**Tier**: 1 - Honest Limitations
**Target Persona**: All lay professionals

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-04 |
| Type | Waveform comparison |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Visually demonstrate the fundamental difference between dense time series (where TSFMs work) and sparse time series (where they don't), helping professionals identify which category their data falls into.

---

## 3. Key Message

> "30 samples per second vs 1 sample per day are completely different problems. The math that works for one doesn't work for the other."

---

## 4. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  SPARSE vs DENSE: Different Beasts                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DENSE TIME SERIES (TSFMs work here)                                       │
│  ───────────────────────────────────                                       │
│                                                                            │
│  PLR Signal (30 Hz = 30 samples/second)                                    │
│  ╭─╮   ╭─╮                                                                 │
│  │ │   │ │   ╭───────────────────────────────                              │
│  │ ╰───╯ ╰───╯                                                             │
│  │                                                                         │
│  └────────────────────────────────────────────────────────────             │
│   0        1        2        3        4        5 seconds                   │
│   ↑                                                                        │
│   150 data points in 5 seconds                                             │
│                                                                            │
│  Properties:                                                               │
│  • Continuous underlying process                                           │
│  • Gaps = measurement errors (fixable)                                     │
│  • Temporal patterns are learnable                                         │
│  • Neighboring points are correlated                                       │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SPARSE TIME SERIES (TSFMs fail here)                                      │
│  ────────────────────────────────────                                      │
│                                                                            │
│  Supply Chain Data (1 sample/day)                                          │
│                                                                            │
│      ●           ●                   ●       ●   ●                         │
│                      ●                                                     │
│  ────────────────────────────────────────────────────────────              │
│   Mon  Tue  Wed  Thu  Fri  Sat  Sun  Mon  Tue  Wed  Thu                   │
│   ↑                                                                        │
│   11 data points in 11 days                                                │
│                                                                            │
│  Properties:                                                               │
│  • Discrete events, not continuous process                                 │
│  • Gaps = real events (holidays, closures)                                 │
│  • External factors dominate patterns                                      │
│  • Points may be independent                                               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE MATH IS DIFFERENT                                                     │
│  ─────────────────────                                                     │
│                                                                            │
│  Dense:  x(t+dt) ≈ x(t) + noise     →  Interpolation makes sense          │
│  Sparse: x(t+1d) ≠ f(x(t))          →  Interpolation is meaningless       │
│                                                                            │
│  Dense:  "Fill the gap with nearby values"                                 │
│  Sparse: "The gap IS the information"                                      │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  IDENTIFY YOUR DATA                                                        │
│                                                                            │
│  Question                              Dense (TSFM)    Sparse (Other)      │
│  ────────────────────────────────────  ────────────    ────────────        │
│  Samples per minute?                   >1              <1                  │
│  Is underlying process continuous?     Yes             No/Maybe            │
│  Are gaps measurement errors?          Yes             No                  │
│  Do neighbors predict each other?      Yes             Weakly              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Content Elements

### Top Panel: Dense Time Series
- PLR waveform at 30 Hz
- Annotation: "150 data points in 5 seconds"
- Properties list emphasizing continuity

### Middle Panel: Sparse Time Series
- Supply chain discrete points
- Annotation: "11 data points in 11 days"
- Properties list emphasizing discreteness

### Math Section
- Simple formulas showing why interpolation works/fails
- Plain language explanation

### Decision Checklist
- 4 questions to identify data type
- Clear binary answers

---

## 6. Text Content

### Title
"Sparse vs Dense: Different Beasts"

### Caption
"Dense time series (>1 sample/second) have continuous underlying processes where gaps represent measurement errors—interpolation makes sense. Sparse time series (<1 sample/day) are discrete events where gaps often encode real information—interpolation destroys data. TSFMs are designed for the former. If your data looks like the bottom panel, use domain-specific approaches like GMAN or M-GAM instead."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a visual comparison of dense vs sparse time series data.

TOP PANEL - Dense (PLR signal):
- Smooth continuous waveform at 30 Hz
- Show many data points close together
- Annotation: "150 points in 5 seconds"
- Green/blue color scheme (this works with TSFMs)

BOTTOM PANEL - Sparse (Supply chain):
- Scattered discrete points, one per day
- Large gaps between points
- Annotation: "11 points in 11 days"
- Orange/red color scheme (TSFMs fail here)

MIDDLE - Math comparison:
- Dense: "x(t+dt) ≈ x(t)" - interpolation works
- Sparse: "gap IS information" - don't interpolate

BOTTOM - Decision checklist:
Table with questions to identify data type

Style: Academic, clear contrast between panels
```

---

## 8. Alt Text

"Visual comparison of dense vs sparse time series. Top panel shows PLR signal at 30 Hz with 150 data points in 5 seconds as a smooth continuous curve. Bottom panel shows supply chain data with only 11 scattered points over 11 days. Middle section explains math: dense data allows interpolation because neighboring points are correlated, sparse data gaps encode real information. Bottom checklist helps identify which category data belongs to."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
