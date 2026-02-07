# fig-trans-10: The Dense Signal Club

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: All technical professionals

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-10 |
| Type | Visual membership diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 12" × 12" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Create a memorable visual showing which domains belong to the "dense signal club" where TSFM preprocessing concepts transfer, and which domains are excluded.

---

## 3. Key Message

> "If your signal has >1 sample/second, continuous underlying dynamics, and physically interpretable artifacts—welcome to the club. The preprocessing concepts from this PLR repository will work for you."

---

## 4. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  THE DENSE SIGNAL CLUB                                                     │
│  Where TSFM Preprocessing Concepts Transfer                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                         ┌─────────────────────────┐                        │
│                         │    CLUB MEMBERSHIP      │                        │
│                         │    REQUIREMENTS         │                        │
│                         ├─────────────────────────┤                        │
│                         │ ✓ >1 sample/second      │                        │
│                         │ ✓ Continuous process    │                        │
│                         │ ✓ Gaps = errors         │                        │
│                         │ ✓ Neighbors correlated  │                        │
│                         └─────────────────────────┘                        │
│                                    │                                       │
│          ┌─────────────────────────┼─────────────────────────┐             │
│          │                         │                         │             │
│          ▼                         ▼                         ▼             │
│  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐      │
│  │  BIOSIGNALS   │        │  ENGINEERING  │        │    AUDIO      │      │
│  ├───────────────┤        ├───────────────┤        ├───────────────┤      │
│  │ 👁️ PLR (30Hz) │        │ ⚡ Grid (60Hz) │        │ 🎤 Speech     │      │
│  │ ❤️ ECG (500Hz)│        │ 🔧 Vibration   │        │    (16kHz)    │      │
│  │ 🧠 EEG (256Hz)│        │    (5kHz)     │        │ 🎵 Music      │      │
│  │ 💓 PPG (125Hz)│        │ 🌍 Seismic    │        │    (48kHz)    │      │
│  │               │        │    (100Hz)    │        │               │      │
│  └───────────────┘        └───────────────┘        └───────────────┘      │
│                                                                            │
│                    ════════════════════════════════                        │
│                           THE VELVET ROPE                                  │
│                    ════════════════════════════════                        │
│                                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                     NOT IN THE CLUB                                │    │
│  │                     (Different math needed)                        │    │
│  ├───────────────────────────────────────────────────────────────────┤    │
│  │                                                                    │    │
│  │  📋 EHR Data           📊 Business KPIs      🚚 Logistics          │    │
│  │  (irregular,           (daily/weekly,        (event-driven,        │    │
│  │   gaps = info)          no artifacts)         external shocks)     │    │
│  │                                                                    │    │
│  │  → Use: Neural ODE     → Use: ARIMA          → Use: GMAN           │    │
│  │         GMAN                  Prophet               M-GAM          │    │
│  │                                                                    │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHAT CLUB MEMBERS SHARE                                                   │
│  ───────────────────────                                                   │
│                                                                            │
│  • Anomaly detection algorithms (LOF, autoencoders) work across domains   │
│  • Imputation architectures (SAITS, CSDI, diffusion) transfer             │
│  • Evaluation methodology (bootstrap, cross-validation) applies           │
│  • Ground truth can be hand-annotated with high inter-rater reliability   │
│                                                                            │
│  WHAT DIFFERS                                                              │
│  ────────────                                                              │
│                                                                            │
│  • Feature definitions (domain-specific biomarkers, spectral features)    │
│  • Threshold calibration (what counts as "anomalous")                     │
│  • Classification targets (disease vs fault vs noise type)                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Content Elements

### Club Requirements Card
- 4 criteria for membership
- Clear, memorable checklist

### Member Categories
- Biosignals: PLR, ECG, EEG, PPG
- Engineering: Grid, Vibration, Seismic
- Audio: Speech, Music

### "Not in Club" Section
- EHR, Business, Logistics
- Alternative approaches for each

### What Transfers vs What Doesn't
- Algorithms that work across domains
- Domain-specific elements

---

## 6. Text Content

### Title
"The Dense Signal Club"

### Caption
"TSFM preprocessing concepts transfer across domains that share four properties: high sampling rate (>1 Hz), continuous underlying process, gaps representing measurement errors (not information), and correlated neighboring samples. PLR, ECG, vibration, seismic, and audio all belong to this 'club'. EHRs, business KPIs, and logistics data don't—they need domain-specific approaches like GMAN, M-GAM, or traditional forecasting methods."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a visual "club membership" diagram for dense time series domains.

TOP - Membership card:
Box showing 4 requirements: >1 sample/second, continuous process, gaps=errors, neighbors correlated

MIDDLE - Three member categories in connected boxes:
- Biosignals (PLR, ECG, EEG, PPG with icons)
- Engineering (Grid, Vibration, Seismic with icons)
- Audio (Speech, Music with icons)

VELVET ROPE DIVIDER

BOTTOM - "Not in Club" section:
- EHR, Business KPIs, Logistics
- Each with recommended alternative approach

FOOTER:
What transfers (algorithms, architectures) vs what differs (features, thresholds)

Style: Exclusive club metaphor but academic, not kitschy
```

---

## 8. Alt Text

"Visual membership diagram for 'The Dense Signal Club'. Top shows membership requirements: greater than 1 sample per second, continuous underlying process, gaps represent errors, neighbors are correlated. Three member categories shown: Biosignals (PLR, ECG, EEG, PPG), Engineering (Grid, Vibration, Seismic), and Audio (Speech, Music). Below a divider, 'Not in Club' section shows EHR, Business KPIs, and Logistics with recommended alternative approaches for each. Footer lists what transfers between members (algorithms, architectures) versus what is domain-specific (features, thresholds)."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
