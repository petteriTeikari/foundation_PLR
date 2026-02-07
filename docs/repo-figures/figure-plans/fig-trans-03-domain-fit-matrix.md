# fig-trans-03: Domain Fit Matrix

**Status**: 📋 PLANNED
**Tier**: 1 - Honest Limitations
**Target Persona**: All lay professionals, data scientists

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-03 |
| Type | Heat map / matrix |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 12" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Provide an honest, at-a-glance assessment of which domains are suitable for TSFM-based preprocessing and which require alternative approaches.

---

## 3. Key Message

> "TSFMs are not universal. They work for dense, regularly-sampled signals with learnable temporal dynamics. For sparse, irregular, or event-driven data, use purpose-built alternatives."

---

## 4. Literature Sources

| Source | Key Finding |
|--------|-------------|
| Jin et al. 2024 | "LLMs fail on datasets without clear periodicity" |
| Schoenegger & Park 2023 | "GPT-4 ≈ 50% on real forecasting tournaments" |
| Hewamalage et al. 2022 | "ML researchers adopt flawed evaluation practices" |
| Various TSFM papers | Foundation models trained on dense time series |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  DOMAIN FIT MATRIX: Where TSFMs Work (and Don't)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                        TSFM FIT SCORE                                      │
│                  ─────────────────────────                                 │
│                  Poor │ Fair │ Good │ Excellent                            │
│                    1  │  2   │  3   │    4                                 │
│                                                                            │
│  ┌────────────────┬─────┬─────┬─────┬─────┬───────────────────────────┐    │
│  │ Domain         │Anom │Imput│Forec│Class│ Notes                     │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ DENSE SIGNALS  │     │     │     │     │                           │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ PLR (30 Hz)    │ ██  │ ██  │ █   │ █   │ Imputation good,          │    │
│  │                │     │     │     │     │ classification needs       │    │
│  │                │     │     │     │     │ domain features           │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ ECG (500 Hz)   │ ███ │ ███ │ ██  │ ███ │ Mature FMs exist          │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ Seismic        │ ███ │ ██  │ █   │ ██  │ DASFormer emerging        │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ Vibration      │ ██  │ ██  │ █   │ ██  │ Statistical still strong  │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ Speech (16kHz) │ ███ │ ██  │ N/A │ ██  │ Domain-specific wins      │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ SPARSE SIGNALS │     │     │     │     │                           │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ EHR (irregular)│ █   │ X   │ █   │ █   │ Missingness informative,  │    │
│  │                │     │     │     │     │ use GMAN/M-GAM            │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ Business KPIs  │ █   │ X   │ █   │ N/A │ ARIMA/Prophet still win   │    │
│  ├────────────────┼─────┼─────┼─────┼─────┼───────────────────────────┤    │
│  │ Logistics      │ █   │ X   │ █   │ █   │ Event-conditioned needed  │    │
│  └────────────────┴─────┴─────┴─────┴─────┴───────────────────────────┘    │
│                                                                            │
│  Legend:                                                                   │
│  ███ = Excellent (4)  ██ = Good (3)  █ = Fair (2)  X = Don't use (1)      │
│  N/A = Task not applicable                                                 │
│                                                                            │
│  Tasks: Anom = Anomaly Detection, Imput = Imputation,                      │
│         Forec = Forecasting, Class = Classification                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Content Elements

### Matrix Structure
- Rows: Domains (grouped by dense vs sparse)
- Columns: Tasks (Anomaly Detection, Imputation, Forecasting, Classification)
- Cells: Fit score (1-4) + color coding

### Dense Signals (Green zone)
- PLR (this repo): Good for anomaly/imputation, poor for classification without domain features
- ECG: Excellent across all tasks (mature foundation models)
- EEG: Good to excellent
- Seismic: Good for anomaly detection (DASFormer emerging)
- Vibration: Good, but statistical features still competitive
- Speech/Audio: Good, but domain-specific models often win

### Sparse Signals (Red zone)
- EHR: Poor fit - structural missingness, use Neural ODE/GMAN
- Business KPIs: Poor fit - ARIMA/Prophet still win
- Logistics: Poor fit - event-conditioned approaches needed

### Key Factors Column
Brief explanation for each domain's fit score

---

## 7. Text Content

### Title
"Domain Fit Matrix: Where TSFMs Work (and Don't)"

### Caption
"Time Series Foundation Models (TSFMs) are not universal solutions. They excel at preprocessing tasks (anomaly detection, imputation) on dense, regularly-sampled signals where temporal dynamics are learnable. They underperform on sparse or irregular signals (EHRs, business metrics) where missingness is informative and external events drive patterns. For classification tasks, domain-specific features often outperform generic TSFM embeddings. This matrix provides an honest assessment based on published benchmarks and our experiments."

### Footnotes
- "Scores based on published literature and our experiments"
- "X indicates imputation is harmful, not just ineffective"
- "N/A indicates task not applicable to domain"

---

## 8. Nano Banana Pro Prompts

### Primary Prompt
```
Create a professional heat map matrix titled "Domain Fit Matrix: Where TSFMs Work (and Don't)"

Structure:
- Rows: Domains grouped into "DENSE SIGNALS" (PLR, ECG, EEG, Seismic, Vibration, Speech) and "SPARSE SIGNALS" (EHR, Business KPIs, Logistics)
- Columns: Tasks (Anomaly Detection, Imputation, Forecasting, Classification, Key Factors)
- Cells: Color-coded scores from 1-4 plus brief notes

Color scheme:
- Excellent (4): Dark blue
- Good (3): Medium blue
- Fair (2): Light blue
- Don't use (1/X): Orange/red
- N/A: Gray

Key visual elements:
- Clear dividing line between dense and sparse signal groups
- Legend explaining scores and X/N/A symbols
- "Key Factors" column with brief explanations

Style:
- Clean, academic heat map
- Economist magazine clarity
- Sans-serif fonts
- High information density without clutter
```

### Refinement Prompt
```
Refine the heat map to add:

1. Sampling rate annotation for each domain (e.g., "PLR (30 Hz)", "ECG (500 Hz)", "EHR (irregular)")

2. Small icons or badges for domains with mature foundation models:
   - ECG: "FM: ECG-FM, MIRA"
   - EEG: "FM: LaBraM"
   - Seismic: "FM: DASFormer"

3. Alternative approach annotations for sparse signals:
   - EHR: "Use: Neural ODE, GMAN"
   - Business: "Use: ARIMA, Prophet"
   - Logistics: "Use: GMAN"

4. Ensure colorblind safety with pattern fills as secondary encoding
```

---

## 9. Alt Text

"A heat map matrix showing Time Series Foundation Model (TSFM) fit scores across different domains and tasks. Dense signal domains (PLR, ECG, EEG, Seismic, Vibration, Speech) show mostly good to excellent scores for anomaly detection and imputation. Sparse signal domains (EHR, Business KPIs, Logistics) show poor scores with X marks indicating imputation should not be used. A legend explains the 1-4 scoring system and notes that X indicates imputation is harmful, not just ineffective."

---

## 10. Related Figures

- fig-trans-01: When NOT to Impute (details on why sparse signals get X)
- fig-trans-04: Sparse vs Dense: Different Beasts (visual comparison)
- fig-trans-10: The Dense Signal Club (focus on transferable domains)

---

## 11. Validation Checklist

- [ ] Scores backed by citations?
- [ ] PLR scores match our experiments? (Imputation good, classification needs features)
- [ ] ECG scores reflect mature FMs? (ECG-FM, MIRA exist)
- [ ] Sparse signal warnings clear?
- [ ] Colorblind safe?
- [ ] Alt text complete?

---

*Last updated: 2026-02-01*
