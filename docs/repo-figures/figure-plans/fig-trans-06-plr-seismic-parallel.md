# fig-trans-06: PLR ↔ Seismic: Anomaly Detection Transfers

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: Geophysicists, seismologists, environmental scientists

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-06 |
| Type | Side-by-side comparison |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Show that PLR preprocessing concepts translate directly to seismic signal processing—both are dense, regularly-sampled signals with physically interpretable artifacts.

---

## 3. Key Message

> "Earthquake detection is fundamentally anomaly detection. The same algorithms that find eye blinks in pupil data can find seismic events in ground motion data."

---

## 4. Structural Parallels

| Aspect | PLR | Seismic |
|--------|-----|---------|
| **Sampling rate** | 30 Hz | 100-500 Hz |
| **Signal type** | Autonomic response | Ground motion |
| **Artifacts to remove** | Blinks, tracking loss | Instrument noise, power line (50/60 Hz) |
| **Events to detect** | Light stimulus response | Earthquakes, tremors |
| **Ground truth** | Human annotation | Seismologist labels |
| **TSFM status** | This repo (early) | DASFormer (emerging, 2025) |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PLR ↔ SEISMIC: Anomaly Detection Transfers                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PUPILLOMETRY (30 Hz)                SEISMOLOGY (100 Hz)                   │
│  ───────────────────                 ──────────────────                    │
│                                                                            │
│  [Pupil trace with blink]            [Seismic trace with earthquake]       │
│       ∧     ∧                              ╱╲                              │
│      / \   / \──────────                 ╱    ╲                            │
│     /   \_/   \                    ─────╱      ╲─────────────              │
│    /  ↑blink↑  \                       ↑earthquake↑                        │
│                                                                            │
│  Artifact: Eyelid closes             Artifact: 50 Hz power line noise      │
│  Event: Pupil constricts             Event: Ground shakes                  │
│  Goal: Detect blinks, keep response  Goal: Detect quakes, remove noise     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SHARED CHALLENGE: SIGNAL vs ARTIFACT vs EVENT                             │
│  ─────────────────────────────────────────────                             │
│                                                                            │
│  │ Raw Signal │──►│ Artifact Detection │──►│ Event Detection │            │
│                                                                            │
│  PLR:    pupil_raw  →  find blinks      →  find light response             │
│  Seismic: raw_motion →  find inst. noise →  find earthquakes               │
│                                                                            │
│  Same pipeline! Different domain knowledge for thresholds.                 │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  METHODS THAT TRANSFER                                                     │
│  ─────────────────────                                                     │
│                                                                            │
│  Method          │ PLR Usage              │ Seismic Usage                  │
│  ────────────────┼────────────────────────┼──────────────────────────────  │
│  LOF             │ Outlier detection      │ Noise spike detection          │
│  Autoencoders    │ Anomaly scoring        │ DeepDenoiser (2021)            │
│  Diffusion       │ CSDI imputation        │ Cold Diffusion (2024)          │
│  Foundation      │ MOMENT (this repo)     │ DASFormer (2025)               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  KEY REFERENCES                                                            │
│                                                                            │
│  PLR: This repository                                                      │
│  Seismic: DASFormer (2025) - Self-supervised TSFM for earthquake detection │
│           Cold Diffusion for Seismic Denoising (2024)                      │
│           DeepDenoiser - CNN for seismic signal enhancement                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"PLR ↔ Seismic: Anomaly Detection Transfers"

### Caption
"Pupillometry and seismology share the same preprocessing challenge: separate signal from artifacts, then detect events of interest. The algorithms transfer directly—LOF for outlier detection, autoencoders for denoising, diffusion models for reconstruction. DASFormer (2025) applies TSFM pre-training to earthquake detection using the same self-supervised approach that works for biosignals. The domain knowledge differs (blinks vs power line noise), but the mathematical framework is identical."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a parallel comparison of PLR pupillometry and seismic signal processing.

LEFT PANEL - PLR (30 Hz):
- Pupil diameter trace with a blink artifact
- Annotation showing artifact vs signal
- Blue color scheme

RIGHT PANEL - Seismic (100 Hz):
- Ground motion trace with an earthquake event
- Annotation showing noise vs event
- Earth-tone color scheme

MIDDLE - Shared pipeline:
Flow diagram: Raw Signal → Artifact Detection → Event Detection
Show how same pipeline applies to both domains

BOTTOM - Methods table:
LOF, Autoencoders, Diffusion, Foundation Models
Show parallel usage in both domains

Style: Academic, emphasize structural similarity
```

---

## 8. Alt Text

"Side-by-side comparison of PLR and seismic signal processing. Left shows pupil trace with blink artifact at 30 Hz. Right shows seismic trace with earthquake event at 100 Hz. Middle section shows shared pipeline: Raw Signal → Artifact Detection → Event Detection, applicable to both domains. Bottom table lists methods that transfer: LOF, autoencoders, diffusion models, and foundation models (MOMENT for PLR, DASFormer for seismic)."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
