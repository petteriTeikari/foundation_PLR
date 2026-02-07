# fig-trans-05: PLR ↔ Vibration: Same Problem, Different Domain

**Status**: 📋 PLANNED
**Tier**: 2 - Translational Parallels
**Target Persona**: Industrial engineers, predictive maintenance professionals

---

## 1. Metadata

| Field | Value |
|-------|-------|
| Figure ID | fig-trans-05 |
| Type | Side-by-side comparison |
| Style | 75% manuscript + 25% Economist |
| Dimensions | 14" × 8" |
| Format | PDF (vector) + PNG (300 DPI) |

---

## 2. Purpose

Demonstrate that preprocessing challenges in PLR pupillometry and industrial vibration monitoring are structurally identical, enabling concept transfer.

---

## 3. Key Message

> "The preprocessing pipeline is domain-agnostic. Whether you're removing eye blinks from pupil signals or sensor dropouts from vibration data, the mathematical problem is the same: detect anomalies, reconstruct the true underlying signal."

---

## 4. Structural Parallels

| Aspect | PLR (Pupillometry) | Vibration Monitoring |
|--------|-------------------|---------------------|
| **Sampling rate** | 30 Hz | 5-25 kHz (downsampled to 25-50 Hz) |
| **Signal type** | Autonomic response | Mechanical motion |
| **Artifacts** | Blinks, tracking loss, saccades | Sensor dropouts, electrical noise, transmission errors |
| **Ground truth** | Hand-annotated by experts | Hand-labeled by engineers |
| **Periodicity** | Light stimulus cycle (known) | Machine rotation frequency (known) |
| **Anomaly detection goal** | Find blinks to remove | Find dropouts to remove (vs faults to alert) |
| **Imputation goal** | Reconstruct pupil trajectory | Reconstruct vibration signature |
| **Classification goal** | Glaucoma vs healthy | Normal vs bearing fault |

---

## 5. Visual Concept

```
┌────────────────────────────────────────────────────────────────────────────┐
│  PLR ↔ VIBRATION: Same Problem, Different Domain                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PUPILLOMETRY (30 Hz)                VIBRATION MONITORING (25 Hz)          │
│  ───────────────────                 ─────────────────────────             │
│                                                                            │
│  Signal:                             Signal:                               │
│    ∧     ∧                             ╱╲╱╲╱╲╱╲╱╲╱╲                        │
│   / \   / \                                   │                            │
│  /   \_/   \──────────                  ╱╲╱╲╱│╲╱╲╱╲                        │
│ /  ↑blink↑  \                               │ ↑                            │
│      │                                      │ dropout                      │
│      │                                      │                              │
│  Artifact: Eyelid occlusion          Artifact: Transmission error          │
│  Physics: No light reaches pupil     Physics: No data transmitted          │
│  Solution: Impute from context       Solution: Impute from context         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SHARED PREPROCESSING PIPELINE                                             │
│  ─────────────────────────────                                             │
│                                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Raw Signal  │───►│ Anomaly     │───►│ Imputation  │───►│ Feature     │  │
│  │             │    │ Detection   │    │ /Reconstruct│    │ Extraction  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                            │
│  PLR: blinks,        PLR: LOF, MOMENT   PLR: SAITS, CSDI  PLR: amplitude   │
│       tracking       Vib: statistical,  Vib: similar      bins, latency    │
│  Vib: dropouts,          same methods        methods      Vib: RMS,        │
│       noise              work                             kurtosis         │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  WHAT TRANSFERS vs WHAT DOESN'T                                            │
│                                                                            │
│  ✓ TRANSFERS:                        ✗ DOMAIN-SPECIFIC:                    │
│  • Anomaly detection algorithms      • Feature definitions                 │
│  • Imputation architectures          • Threshold calibration               │
│  • Evaluation methodology            • Classification models               │
│  • Pipeline orchestration            • Domain expertise for labeling       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Content Elements

### Top Section: Signal Comparison
- Left: PLR trace with blink artifact (annotated)
- Right: Vibration trace with dropout artifact (annotated)
- Arrows connecting analogous elements

### Middle Section: Shared Pipeline
- 4-stage pipeline diagram (Raw → Detect → Impute → Features)
- Domain-specific method names under each stage
- Emphasis on shared architecture

### Bottom Section: Transfer Summary
- Two columns: What transfers vs What's domain-specific
- Clear separation of generic vs specialized components

---

## 7. Text Content

### Title
"PLR ↔ Vibration: Same Problem, Different Domain"

### Caption
"Pupillometry and industrial vibration monitoring share identical preprocessing challenges. Both are dense, regularly-sampled signals with physically interpretable artifacts (blinks vs sensor dropouts) that need detection and removal before downstream analysis. The anomaly detection algorithms (LOF, MOMENT, statistical methods) and imputation architectures (SAITS, CSDI) transfer directly. Only the domain-specific features and classification targets differ. This repo's preprocessing pipeline can be forked for vibration monitoring with minimal modification."

### Callout Box
"Transfer Potential: The same preprocessing code that removes eye blinks from pupil signals can remove sensor dropouts from vibration data. The mathematical problem is identical."

---

## 8. Nano Banana Pro Prompts

### Primary Prompt
```
Create a professional scientific comparison figure titled "PLR ↔ Vibration: Same Problem, Different Domain"

TOP ROW - Signal Comparison (two panels):
Left panel: Pupil diameter trace over time (simplified)
- Show smooth curve with a gap labeled "blink artifact"
- Annotate: "30 Hz sampling, autonomic response to light"
- Color: blue theme

Right panel: Vibration acceleration trace over time
- Show oscillating signal with a gap labeled "sensor dropout"
- Annotate: "25 Hz sampling, mechanical rotation"
- Color: orange theme

MIDDLE ROW - Shared Pipeline:
- 4-box flow diagram: Raw Signal → Anomaly Detection → Imputation → Feature Extraction
- Under each box, list methods that work in BOTH domains
- Arrows connecting the boxes

BOTTOM ROW - Transfer Summary:
Two columns:
- Left (green checkmarks): "What Transfers" - algorithms, architectures, methodology
- Right (red X marks): "Domain-Specific" - features, thresholds, classification targets

Style:
- Clean, academic
- Economist magazine clarity
- Parallel structure emphasizing similarities
- Professional color palette
```

### Refinement Prompt
```
Refine to add:

1. Small "sampling rate" badges on each signal panel (30 Hz, 25 Hz)

2. Mathematical notation showing the same formula applies:
   - "Anomaly score: d(x_t, neighborhood)"
   - "Imputation: x̂_t = f(x_{t-k:t-1}, x_{t+1:t+k})"

3. Domain expert silhouettes:
   - Ophthalmologist next to PLR panel
   - Maintenance engineer next to vibration panel
   - Caption: "Different experts, same math"

4. Ensure visual balance between panels
```

---

## 9. Alt Text

"A comparison figure with three sections. Top section shows two time series side by side: a pupil diameter trace with a blink artifact (left) and a vibration acceleration trace with a sensor dropout (right). Both artifacts are visually similar - gaps in otherwise smooth signals. Middle section shows a shared preprocessing pipeline with four stages (Raw Signal, Anomaly Detection, Imputation, Feature Extraction) that applies to both domains. Bottom section lists what transfers between domains (algorithms, architectures, methodology) versus what is domain-specific (features, thresholds, classification targets)."

---

## 10. Related Figures

- fig-trans-06: PLR ↔ Seismic parallel
- fig-trans-07: PLR ↔ Audio parallel
- fig-trans-10: The Dense Signal Club (all transferable domains)
- fig-trans-15: PLR Code: What's Domain-Specific?

---

## 11. Validation Checklist

- [ ] Vibration monitoring accurately represented?
- [ ] Artifact types correctly analogous?
- [ ] Sampling rates realistic?
- [ ] Transfer claims defensible?
- [ ] Industrial engineer would recognize their domain?
- [ ] Colorblind safe?
- [ ] Alt text complete?

---

*Last updated: 2026-02-01*
