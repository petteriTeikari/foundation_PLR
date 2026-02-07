# fig-trans-08: Source Separation: Lung/Heart/Ambient

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: Wearable health engineers, respiratory scientists, medical device developers

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-08 |
| Type | Component diagram |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 10" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Show that wearable acoustic monitoring for lung sounds requires the same preprocessing concepts as PLR—separating signal from artifact—but with the added complexity of multiple overlapping sources (lung, heart, ambient).

---

## 3. Key Message

> "A stethoscope-like wearable captures lung sounds, heart sounds, and environmental noise as a mixture. Separating these is analogous to PLR artifact removal, but with multiple signals to preserve instead of one."

---

## 4. Literature Sources

Based on `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/background-research/older-archives/wearableMic_signal-compressed.md`:

- Grooby et al. (2023): Signal separation for lung sound analysis
- McLane et al. (2023): Flexible sensor design for body sounds
- Rennoll et al. (2023): Impedance-matched microphone design
- Yang and Zhao (2023): Acoustic wake-up for sparse event monitoring

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  SOURCE SEPARATION: Lung / Heart / Ambient                                 │
│  Wearable Acoustic Health Monitoring                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE MIXTURE PROBLEM                                                       │
│  ───────────────────                                                       │
│                                                                            │
│  ┌─────────────┐                                                           │
│  │  Wearable   │    Records MIXTURE:                                       │
│  │  Microphone │    ════════════════                                       │
│  │     🎤      │    Lung + Heart + Ambient + Friction                      │
│  └─────────────┘                                                           │
│                                                                            │
│  Mixed Signal:  ╱╲_/╲╱╲__╱╲_╱╲/╲_╱╲╱╲_/╲__╱╲_╱╲/╲_                        │
│                                                                            │
│  Contains:      🫁 Lung sounds (wheeze, crackles, breath)                  │
│                 ❤️ Heart sounds (S1, S2, murmurs)                          │
│                 🔊 Ambient noise (traffic, speech, HVAC)                   │
│                 ⚡ Sensor artifacts (friction, movement)                   │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PLR vs LUNG SOUND: Parallel Challenges                                    │
│  ───────────────────────────────────────                                   │
│                                                                            │
│  Challenge        │ PLR                   │ Lung Sounds                    │
│  ─────────────────┼───────────────────────┼──────────────────────────────  │
│  Signal of        │ Pupil diameter        │ Lung sounds                    │
│  interest         │                       │                                │
│  Artifacts        │ Blinks, tracking      │ Heart, ambient, friction       │
│  Sampling         │ 30 Hz                 │ 8-16 kHz                       │
│  Separation       │ Not needed (1 signal) │ Critical (4+ sources)          │
│  Ground truth     │ Human labels          │ Expert annotation              │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SEPARATION PIPELINE                                                       │
│  ───────────────────                                                       │
│                                                                            │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Mixed   │───►│ Artifact    │───►│ Source      │───►│ Downstream  │     │
│  │ Signal  │    │ Removal     │    │ Separation  │    │ Analysis    │     │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                                            │
│  PLR:            Blink detection    (not needed)      Classification       │
│  Lung:           Friction removal   Lung/Heart split  Disease detection    │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  MULTI-SENSOR ADVANTAGE                                                    │
│  ──────────────────────                                                    │
│                                                                            │
│  Single Mic (blind):     ┌───────┐                                         │
│  All sources mixed       │  🎤   │  →  Separation is hard                  │
│                          └───────┘                                         │
│                                                                            │
│  Multi-Mic (informed):   ┌───────┐  Body mic (lung + heart)                │
│  Reference helps         │  🎤   │                                         │
│  separation              │  🎤   │  Reference mic (ambient only)           │
│                          └───────┘  →  Separation is easier                │
│                                                                            │
│  Analogy: PLR has known light stimulus timing (like a reference)           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"Source Separation: Lung/Heart/Ambient"

### Caption
"Wearable acoustic monitoring captures a mixture of lung sounds, heart sounds, ambient noise, and sensor artifacts. Separating these into clinically useful signals parallels PLR preprocessing—both require distinguishing signal from corruption. The key difference: lung monitoring needs source separation (multiple signals to preserve), not just artifact removal (one signal to clean). Multi-sensor setups with reference microphones transform 'blind' separation into 'informed' separation, analogous to how PLR uses known stimulus timing."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a diagram explaining wearable lung sound source separation.

TOP - The Mixture Problem:
- Wearable microphone icon recording a mixed signal
- Show four components: lung sounds, heart sounds, ambient noise, friction artifacts
- Each with distinct icon (lungs, heart, speaker, lightning)

MIDDLE - Comparison table:
PLR vs Lung Sounds on: signal of interest, artifacts, sampling rate, separation need

BOTTOM - Pipeline:
Mixed Signal → Artifact Removal → Source Separation → Downstream Analysis
Show PLR and Lung parallel paths

SIDEBAR - Multi-sensor advantage:
Single mic (blind) vs Multi-mic (informed) comparison

Style: Medical device context, clean diagram, no sci-fi effects
```

---

## 8. Alt Text

"Diagram explaining wearable lung sound source separation. Top section shows a wearable microphone capturing a mixture of lung sounds, heart sounds, ambient noise, and friction artifacts. Middle table compares PLR and lung sound processing challenges. Bottom shows pipeline: Mixed Signal → Artifact Removal → Source Separation → Analysis. Sidebar illustrates multi-sensor advantage: single microphone requires blind separation while multiple microphones enable informed separation using reference signals."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
