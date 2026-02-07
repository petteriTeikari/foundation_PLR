# fig-trans-20: Choose Your Approach (Decision Tree)

**Status**: 📋 PLANNED
**Tier**: 1 - Core Translational
**Target Persona**: All lay professionals, decision makers

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-20 |
| Type | Decision tree flowchart |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 16" × 12" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Provide a comprehensive decision tree that guides readers from their data characteristics to the appropriate approach—whether that's TSFMs, traditional methods, domain-specific models, or entirely different paradigms like GMAN or M-GAM.

---

## 3. Key Message

> "There's no universal 'best' approach. Your data characteristics—sampling rate, gap meaning, domain expertise—determine which path to take. This decision tree helps you find YOUR path."

---

## 4. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  CHOOSE YOUR APPROACH: A Decision Tree                                          │
│  From Data Characteristics to Method Selection                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                    ┌─────────────────────────┐                                  │
│                    │    START: Your Data     │                                  │
│                    └───────────┬─────────────┘                                  │
│                                │                                                │
│                    ┌───────────▼─────────────┐                                  │
│                    │ Sampling rate > 1 Hz?   │                                  │
│                    └───────────┬─────────────┘                                  │
│                         │              │                                        │
│                       YES            NO                                         │
│                         │              │                                        │
│              ┌──────────▼────┐    ┌───▼───────────────────────┐                │
│              │ DENSE SIGNAL  │    │ SPARSE/IRREGULAR SIGNAL   │                │
│              │ (PLR, ECG,    │    │ (EHR, business, logistics) │                │
│              │  vibration)   │    └───────────┬───────────────┘                │
│              └──────┬────────┘                │                                │
│                     │                         │                                │
│         ┌───────────▼─────────────┐   ┌──────▼──────────────────┐              │
│         │ Gaps = measurement      │   │ Gaps = information      │              │
│         │ errors?                 │   │ (meaningful missingness)│              │
│         └───────────┬─────────────┘   └───────────┬─────────────┘              │
│              │              │                     │                            │
│            YES            NO                    YES                            │
│              │              │                     │                            │
│    ┌─────────▼──────┐ ┌────▼────────┐    ┌──────▼──────────────┐              │
│    │ TSFMs/Impute   │ │ Source      │    │ M-GAM               │              │
│    │ (MOMENT, SAITS)│ │ Separation  │    │ (missingness as     │              │
│    └────────┬───────┘ │ needed      │    │  feature)           │              │
│             │         └─────────────┘    └─────────────────────┘              │
│    ┌────────▼────────────┐                                                     │
│    │ Have domain         │                                                     │
│    │ expertise?          │                                                     │
│    └─────────┬───────────┘                                                     │
│         │          │                                                           │
│       YES        NO                                                            │
│         │          │                                                           │
│   ┌─────▼──────┐ ┌─▼─────────────────┐                                        │
│   │ Handcraft  │ │ FM Embeddings     │                                        │
│   │ features   │ │ (accept ~9pp loss │                                        │
│   │ (higher    │ │  for convenience) │                                        │
│   │  accuracy) │ └───────────────────┘                                        │
│   └────────────┘                                                               │
│                                                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SPECIAL CASES                                                                  │
│  ─────────────                                                                  │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │  Scenario                           │ Approach                          │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Event-conditioned sparse data      │ GMAN (Graph Mixing)              │    │
│  │  (logistics, disaster response)     │                                  │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Multiple overlapping sources       │ Source separation first,         │    │
│  │  (wearable lung + heart + ambient)  │ then task-specific model        │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Real-time constraints              │ Simple baselines (linear,        │    │
│  │  (< 100ms latency required)         │ moving average)                  │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Small data (< 1000 samples)        │ Simple baselines, not DL         │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Interpretability required          │ GAMs, linear models,             │    │
│  │  (clinical, regulatory)             │ handcrafted features             │    │
│  │  ────────────────────────────────────┼─────────────────────────────────│    │
│  │  Zero-shot cross-domain             │ Foundation models                │    │
│  │  (prototype quickly)                │ (MOMENT, TimesFM)                │    │
│  │                                                                         │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  THE HONEST SUMMARY                                                             │
│  ─────────────────                                                              │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                         │    │
│  │  Domain            │ Typical Best Choice  │ FM Role                    │    │
│  │  ──────────────────┼─────────────────────┼──────────────────────────  │    │
│  │  Biosignals (PLR,  │ Handcrafted +       │ Preprocessing, not         │    │
│  │  ECG, EEG)         │ traditional ML      │ embeddings                 │    │
│  │  ──────────────────┼─────────────────────┼──────────────────────────  │    │
│  │  Industrial (vib,  │ Domain + simple DL  │ Anomaly detection          │    │
│  │  audio, grid)      │                     │                            │    │
│  │  ──────────────────┼─────────────────────┼──────────────────────────  │    │
│  │  Business/sparse   │ GMAN, M-GAM,        │ Limited (wrong             │    │
│  │                    │ Prophet             │ assumptions)               │    │
│  │  ──────────────────┼─────────────────────┼──────────────────────────  │    │
│  │  Exploratory/POC   │ FMs directly        │ Primary (speed > accuracy) │    │
│  │                                                                         │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  Key insight: TSFMs are TOOLS, not solutions. They help with preprocessing     │
│  in dense signals, but they don't replace domain knowledge or fix wrong        │
│  assumptions about your data.                                                   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Text Content

### Title
"Choose Your Approach: A Decision Tree"

### Caption
"No universal 'best' method exists—your data characteristics determine the right approach. Dense signals (>1 Hz) with measurement-error gaps → TSFMs for preprocessing, handcrafted features for classification. Sparse signals with meaningful missingness → M-GAM. Event-conditioned logistics → GMAN. Real-time or small data → simple baselines. The honest truth: foundation models are tools for preprocessing, not replacements for domain expertise."

---

## 6. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a comprehensive decision tree for method selection.

MAIN TREE:
Start → "Sampling > 1 Hz?"
  YES → Dense Signal branch
    "Gaps = errors?" YES → TSFMs → "Have expertise?" → Handcraft/Embeddings
    "Gaps = errors?" NO → Source separation
  NO → Sparse Signal branch
    "Gaps = information?" → M-GAM

SPECIAL CASES TABLE:
Event-conditioned → GMAN
Multiple sources → Source separation
Real-time → Simple baselines
Small data → Simple baselines
Interpretability → GAMs, linear
Zero-shot → Foundation models

HONEST SUMMARY TABLE:
Domain vs Typical Best Choice vs FM Role
(Biosignals, Industrial, Business, Exploratory)

Style: Flowchart with clear decision nodes, tables for special cases
```

---

## 7. Alt Text

"Comprehensive decision tree for method selection. Main tree starts with sampling rate question: greater than 1 Hz leads to dense signal branch (TSFMs, handcrafted features), less than 1 Hz leads to sparse signal branch (M-GAM, GMAN). Special cases table covers event-conditioned data (GMAN), multiple sources (separation first), real-time (simple baselines), small data (simple baselines), interpretability (GAMs), zero-shot (FMs). Summary table shows by domain: biosignals favor handcrafted features, industrial favors domain plus DL, business favors GMAN/M-GAM, exploratory favors FMs for speed."

---

## 8. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
