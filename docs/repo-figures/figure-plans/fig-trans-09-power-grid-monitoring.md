# fig-trans-09: Power Grid Monitoring: 60 Hz is Regular Too

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: Electrical engineers, utility operators, smart grid developers

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-09 |
| Type | Domain parallel diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Show that power grid monitoring shares structural similarities with PLR preprocessing—both are dense, regular signals with known periodicities and physically interpretable anomalies.

---

## 3. Key Message

> "A 60 Hz power grid signal is as regular as a 30 Hz pupil signal. Voltage sags, transients, and harmonic distortion are 'artifacts' that anomaly detection algorithms can find—the same algorithms that find eye blinks."

---

## 4. Structural Parallels

| Aspect | PLR | Power Grid |
|--------|-----|------------|
| **Sampling rate** | 30 Hz | 30-100 Hz (PMU), up to kHz for protection |
| **Fundamental freq** | Light stimulus cycle | 50/60 Hz mains frequency |
| **Artifacts** | Blinks, saccades | Harmonics, transients, sags |
| **Events of interest** | Pupil response | Faults, outages, attacks |
| **Anomaly detection** | LOF, MOMENT | Graph DNNs, statistical |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  POWER GRID MONITORING: 60 Hz is Regular Too                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PLR SIGNAL (30 Hz)                  GRID VOLTAGE (60 Hz fundamental)      │
│  ─────────────────                   ────────────────────────────────      │
│                                                                            │
│  ∧     ∧                             ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲          │
│ / \   / \──────────                        │                               │
│/   \_/   \                                 │ ← voltage sag                 │
│  ↑blink↑                                 ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲              │
│                                                                            │
│  Regular periodicity: Light stimulus   Regular periodicity: 60 Hz mains    │
│  Artifact: Eyelid occlusion            Artifact: Harmonic distortion       │
│  Event: Pupil constriction             Event: Fault, switching transient   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ANOMALY TYPES                                                             │
│  ─────────────                                                             │
│                                                                            │
│  PLR Anomalies         │ Grid Anomalies                                    │
│  ──────────────────────┼─────────────────────────────────────────────────  │
│  Blinks (signal loss)  │ Outages (signal loss)                             │
│  Tracking errors       │ Sensor failures                                   │
│  Saccades (jumps)      │ Switching transients (jumps)                      │
│  Drift                 │ Frequency deviation                               │
│  Saturation            │ Voltage sags/swells                               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SHARED PREPROCESSING PIPELINE                                             │
│  ─────────────────────────────                                             │
│                                                                            │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │ Raw Signal│───►│ Quality      │───►│ Anomaly      │───►│ Downstream │  │
│  │           │    │ Filtering    │    │ Detection    │    │ Analysis   │  │
│  └───────────┘    └──────────────┘    └──────────────┘    └────────────┘  │
│                                                                            │
│  PLR:  pupil_raw   →  remove blinks  →  detect patterns →  classify        │
│  Grid: voltage_raw →  remove noise   →  detect faults   →  respond         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHY TSFMS COULD HELP GRID MONITORING                                      │
│  ────────────────────────────────────                                      │
│                                                                            │
│  ✓ Dense, regularly-sampled data (same as PLR)                             │
│  ✓ Known periodicity (60 Hz vs light stimulus cycle)                       │
│  ✓ Physically interpretable anomalies                                      │
│  ✓ Large amounts of unlabeled data for pretraining                         │
│                                                                            │
│  Current SOTA: Graph Deviation Networks, CNN+RNN hybrids                   │
│  Opportunity: Zero-shot cross-grid transfer with TSFMs                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"Power Grid Monitoring: 60 Hz is Regular Too"

### Caption
"Power grid voltage signals share structural similarities with PLR data: both are dense, regularly-sampled with known periodicities, and exhibit physically interpretable anomalies. Blinks in PLR parallel voltage sags in grids; tracking errors parallel sensor failures. The same anomaly detection algorithms (LOF, autoencoders) that find eye blinks can find grid faults. Current state-of-art uses Graph Deviation Networks and CNN+RNN hybrids; TSFMs offer potential for zero-shot cross-grid transfer learning."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a parallel comparison of PLR and power grid monitoring.

TOP - Signal comparison:
- Left: PLR waveform with blink artifact (30 Hz)
- Right: Voltage waveform with sag (60 Hz sinusoid)
- Annotations showing parallel anomaly types

MIDDLE - Anomaly type comparison table:
PLR anomalies vs Grid anomalies (blinks↔outages, drift↔frequency deviation)

BOTTOM - Shared pipeline:
Raw Signal → Quality Filtering → Anomaly Detection → Downstream Analysis
Show parallel paths for PLR and Grid

FOOTER:
"Why TSFMs could help" - bullet points on structural similarities

Style: Utility/engineering context, professional, no decorative elements
```

---

## 8. Alt Text

"Parallel comparison of PLR and power grid signal processing. Top shows PLR waveform with blink at 30 Hz alongside grid voltage waveform with sag at 60 Hz. Middle table maps PLR anomalies to grid anomalies: blinks to outages, tracking errors to sensor failures, saccades to transients. Bottom shows shared pipeline for both domains. Footer lists structural similarities that make TSFMs applicable: dense sampling, known periodicity, interpretable anomalies."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
