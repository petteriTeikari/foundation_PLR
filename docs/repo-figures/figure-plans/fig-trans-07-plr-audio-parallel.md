# fig-trans-07: PLR ↔ Audio: Denoising Concepts

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: Audio engineers, speech scientists, music technologists

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-07 |
| Type | Side-by-side comparison |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Show that PLR denoising concepts parallel speech enhancement and music source separation—all are dense signals requiring artifact removal while preserving the underlying structure.

---

## 3. Key Message

> "Whether you're removing eye blinks from pupil data or background noise from speech, the core challenge is the same: separate signal from corruption without destroying the information you want."

---

## 4. Structural Parallels

| Aspect | PLR | Speech | Music |
|--------|-----|--------|-------|
| **Sampling rate** | 30 Hz | 16-48 kHz | 44.1-48 kHz |
| **Artifact type** | Blinks, tracking | Background noise | Recording noise |
| **Signal structure** | Smooth response | Harmonic patterns | Multi-source mixture |
| **Denoising goal** | Remove blinks | Remove noise | Separate sources |
| **Method family** | Autoencoders, diffusion | DeepFilterNet, GTCRN | U-Net, Mamba |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PLR ↔ AUDIO: Denoising Concepts                                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  THE SAME CHALLENGE AT DIFFERENT SCALES                                    │
│                                                                            │
│  PLR (30 Hz)           SPEECH (16 kHz)        MUSIC (48 kHz)              │
│  ──────────            ────────────           ───────────                  │
│                                                                            │
│  ∧     ∧               ╱╲╱╲╱╲╱╲╱╲            [Complex waveform]           │
│ / \   / \──────        ╱╲╱╲╱╲╱╲╱╲                                         │
│/   \_/   \            ╱╲╱╲╱╲╱╲╱╲╱╲                                         │
│  ↑blink↑              ↑background↑            ↑mixed sources↑              │
│                                                                            │
│  Signal: Pupil size    Signal: Speech         Signal: Vocals + instruments │
│  Noise: Blinks         Noise: Environment     Noise: Other sources        │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SHARED MATHEMATICAL FRAMEWORK                                             │
│  ─────────────────────────────                                             │
│                                                                            │
│  observed(t) = signal(t) + artifact(t)                                     │
│                                                                            │
│  Goal: Estimate signal(t) given only observed(t)                           │
│                                                                            │
│  Methods:                                                                  │
│  • Masking: Learn which parts are artifact                                 │
│  • Reconstruction: Learn to generate clean signal                          │
│  • Separation: Learn to split mixture into sources                         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ARCHITECTURE FAMILIES                                                     │
│                                                                            │
│  Architecture    │ PLR              │ Audio                                │
│  ────────────────┼──────────────────┼────────────────────────────────────  │
│  Autoencoders    │ Anomaly scoring  │ U-Net for source separation         │
│  Masking         │ Outlier masks    │ Spectrogram masking                 │
│  Diffusion       │ CSDI, SAITS      │ Cold Diffusion for speech           │
│  State-space     │ TimesNet         │ Mamba for music separation          │
│  Transformers    │ MOMENT           │ Transformer denoising               │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  KEY DIFFERENCE: Domain Representation                                     │
│                                                                            │
│  PLR: Works directly in time domain (30 Hz is manageable)                  │
│  Audio: Often uses time-frequency (spectrogram) domain                     │
│         Because 48 kHz × minutes = billions of samples                     │
│                                                                            │
│  The concepts transfer; the preprocessing adapts to scale.                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Text Content

### Title
"PLR ↔ Audio: Denoising Concepts"

### Caption
"Denoising a 30 Hz pupil signal and enhancing 48 kHz speech share the same mathematical framework: estimate the clean signal from a corrupted observation. The architecture families (autoencoders, masking, diffusion, transformers) apply to both domains. The key difference is representation—PLR works in time domain directly, while audio often uses spectrograms due to scale. Domain-specific models still outperform generic ones, but the conceptual transfer is direct."

---

## 7. Prompts for Nano Banana Pro

### Primary Prompt
```
Create a three-way comparison of PLR, speech, and music denoising.

TOP ROW - Three signal examples:
- PLR: Smooth curve with blink artifact (30 Hz)
- Speech: Waveform with background noise (16 kHz)
- Music: Complex waveform with mixed sources (48 kHz)

MIDDLE - Mathematical framework:
"observed = signal + artifact"
"Goal: Estimate signal given observed"

BOTTOM - Architecture comparison table:
Show autoencoders, masking, diffusion, state-space, transformers
Applied to both PLR and audio domains

NOTE BOX:
"Key difference: PLR works in time domain, audio uses spectrograms"

Style: Academic, show conceptual unity across different scales
```

---

## 8. Alt Text

"Three-way comparison of denoising in PLR, speech, and music. Top row shows signal examples at different sampling rates (30 Hz, 16 kHz, 48 kHz) each with their characteristic artifacts. Middle section shows shared mathematical framework: observed = signal + artifact. Bottom table compares architecture families (autoencoders, masking, diffusion, transformers) applied to both domains. Note explains that PLR works in time domain while audio uses spectrograms due to scale differences."

---

## 9. Status

- [x] Draft created
- [ ] Generated
- [ ] Placed in documentation
