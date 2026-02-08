# fig-repo-98: Pupil Signal Preprocessing Challenges

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-98 |
| **Title** | The Universal Preprocessing Bottleneck Across Pupillometry Hardware |
| **Version** | v3-final |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | Research Scientist, Biostatistician |
| **Location** | Root README.md (Pupil Preprocessing references section) |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Illustrate that raw pupil signals from ANY acquisition device -- handheld pupillometers, research eye trackers, smartphone cameras -- share the same artifact types, making preprocessing the universal bottleneck. This figure accompanies the "Pupil Preprocessing" references section in the README.

## Key Message

"Preprocessing is device-agnostic: blinks and segmentation glitches affect all pupillometry hardware equally. High-quality image segmentation (SAMv3) is a prerequisite — TSFMs handle what remains after segmentation, not raw device garbage."

## Visual Concept (v3 — final)

Three-column layout: hardware diversity → shared artifacts + segmentation prerequisite → TSFM pipeline.

**Key changes from v1/v2:**
- Device: SERI prototype pupillometer from Singapore (red/blue chromatic stimulus lights visible in eyepiece)
- Center column: Only 2 shared instrumentation artifacts (blinks + segmentation glitches), NOT baseline drift
- New inset: "Segmentation Prerequisite" panel emphasizing SAMv3 resolves partial occlusion uncertainty
- Right column: NO baseline correction block, NO cEEMD — pipeline ends at Imputation → clean PLR waveform
- PLR output: Classic "cliff and slope" trace (flat baseline → vertical drop → flat valley → slow exponential recovery)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  THE UNIVERSAL PREPROCESSING BOTTLENECK                                         │
├──────────────────────┬───────────────────────────────┬─────────────────────────┤
│                      │                               │                         │
│  ACQUISITION         │  ARTIFACTS & SEGMENTATION     │  TSFM PIPELINE          │
│  DEVICES             │                               │                         │
│                      │  ┌───────────────────────┐    │  ┌──────────────┐       │
│  ┌────────────┐      │  │ SHARED ARTIFACTS      │    │  │ Outlier      │       │
│  │ Handheld   │──────│─→│  • Blink (dropout)    │    │  │ Detection    │       │
│  │ Pupillometer│     │  │  • Segmentation       │    │  │ MOMENT, LOF  │       │
│  │(SERI, SG)  │      │  │    Glitches (noise)   │────│─→└──────┬───────┘       │
│  └────────────┘      │  └───────────────────────┘    │         │               │
│                      │                               │  ┌──────▼───────┐       │
│  ┌────────────┐      │  ┌───────────────────────┐    │  │ Imputation   │       │
│  │ Eye Tracker│──────│─→│ SEGMENTATION          │    │  │ SAITS, CSDI  │       │
│  │(EyeLink,   │     │  │ PREREQUISITE          │    │  └──────┬───────┘       │
│  │ Tobii)     │      │  │                       │    │         │               │
│  └────────────┘      │  │ [Eye with partial     │    │         ▼               │
│                      │  │  occlusion]           │    │  ┌──────────────┐       │
│  ┌────────────┐      │  │  Poor Seg ✗ │SAMv3 ✓ │    │  │ Reconstructed│       │
│  │ Smartphone │──────│─→│ "Uncertainty resolved │    │  │ PLR Waveform │       │
│  │ Camera     │      │  │  by segmentation,     │    │  │ (cliff+slope)│       │
│  └────────────┘      │  │  NOT by TSFM"         │    │  └──────────────┘       │
│                      │  └───────────────────────┘    │                         │
│                      │                               │                         │
│                      │  PHYSIOLOGICAL                │                         │
│                      │  Hippus (0.3 Hz) —            │                         │
│                      │  experimental control,        │                         │
│                      │  not algorithms               │                         │
│                      │                               │                         │
├──────────────────────┴───────────────────────────────┴─────────────────────────┤
│  KEY INSIGHT: Metrological artifacts (blinks, segmentation glitches) are       │
│  targets for TSFM preprocessing. Partial occlusion must be resolved by         │
│  high-quality segmentation (SAMv3), not downstream algorithms.                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Left-to-right convergence: diverse hardware → shared artifacts + segmentation → TSFM pipeline"
spatial_anchors:
  devices_column:
    x: 0.15
    y: 0.45
    content: "Three acquisition device types stacked vertically (30% width)"
  artifacts_column:
    x: 0.45
    y: 0.45
    content: "Shared artifacts + segmentation prerequisite inset (40% width)"
  pipeline_column:
    x: 0.82
    y: 0.45
    content: "Two-stage TSFM pipeline → clean PLR waveform (30% width)"
  insight_box:
    x: 0.5
    y: 0.92
    content: "Metrological vs physiological + SAMv3 distinction callout"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Handheld pupillometer | `highlight_accent` | SERI prototype (Singapore), black device with red/blue chromatic stimulus lights visible in eyepiece |
| Eye tracker | `secondary_pathway` | EyeLink, Tobii, Pupil Labs research systems |
| Smartphone camera | `secondary_pathway` | PupilSense-type mobile acquisition |
| Blink artifact | `abnormal_warning` | Complete pupil occlusion dropout, signal drops to zero |
| Segmentation glitch | `abnormal_warning` | Noisy burst/spike from pupil tracking loss |
| Segmentation prerequisite | `highlight_accent` | SAMv3 resolves partial occlusion — TSFM cannot handle segmentation garbage |
| Hippus | `secondary_pathway` | 0.3 Hz spontaneous oscillation (physiological, requires experimental control) |
| Outlier detection | `outlier_detection` | MOMENT, LOF |
| Imputation | `imputation` | SAITS, CSDI |
| Reconstructed PLR | `features` | Classic cliff-and-slope waveform: flat baseline → vertical drop → flat valley → slow exponential recovery |

### Removed from v1/v2
| Name | Reason for Removal |
|------|-------------------|
| Baseline drift | Not a shared instrumentation artifact — more related to adaptation/fatigue |
| Partial occlusion (as artifact) | Moved to segmentation prerequisite — resolved by SAMv3, not TSFM |
| Baseline correction block | Not part of our TSFM pipeline — removed from right column |
| cEEMD | Not used in our pipeline |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| All 3 devices | Shared artifacts | Converging arrows (gold, gray, gray) | "Same signal, same problems" |
| Shared artifacts | TSFM pipeline | Arrow | "Algorithmic correction" |
| Segmentation inset | TSFM pipeline | Annotation | "Prerequisite: resolved BEFORE TSFM" |
| Hippus | Separate note | Dashed arrow | "Experimental control, not preprocessing" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "SHARED SIGNAL ARTIFACTS" | Blinks (dropout to zero) + Segmentation glitches (noisy bursts) | Center top |
| "SEGMENTATION PREREQUISITE" | Poor segmentation ✗ vs SAMv3 ✓ — partial occlusion resolved by segmentation, not TSFM | Center middle (new inset) |
| "PHYSIOLOGICAL" | Hippus (0.3 Hz) — requires experimental control | Center bottom |
| "KEY INSIGHT" | Metrological artifacts → TSFM. Partial occlusion → SAMv3 segmentation. | Bottom strip |

## Text Content

### Labels (Max 30 chars each)
- Label 1: Handheld Pupillometer (SERI)
- Label 2: Eye Tracker
- Label 3: Smartphone Camera
- Label 4: Blink Artifact
- Label 5: Segmentation Glitches
- Label 6: Segmentation Prerequisite
- Label 7: Outlier Detection
- Label 8: Imputation
- Label 9: Reconstructed PLR

### Caption (for embedding)
Pupil preprocessing challenges are shared across all acquisition hardware. Blinks and segmentation glitches affect all devices equally, while partial occlusion must be resolved by high-quality image segmentation (SAMv3) before TSFM-based outlier detection and imputation can reconstruct the PLR waveform.

## Prompts for Nano Banana Pro

### Style Prompt
High-fidelity medical infographic, clean Scientific American aesthetic. Matte finish, no glowing sci-fi effects. Background color #FBF9F3 (off-white). Typography: Helvetica, clean dark charcoal. Ray-traced ambient occlusion, soft volumetric lighting from top-left. Hairline vector callout lines.

### Content Prompt
Create a three-column infographic showing the universal preprocessing bottleneck in pupillometry:

**LEFT COLUMN (30% width) — Hardware:**
1. Top Panel (Gold Outline): A specific black handheld pupillometer device with a distinct eyepiece showing internal RED and BLUE stimulus lights (reference: SERI prototype, Singapore). Label: "Handheld Pupillometer (SERI, Singapore)".
2. Middle Panel (Gray): A standard desktop eye tracker setup (monitor + chinrest). Label: "Eye Tracker (EyeLink, Tobii)".
3. Bottom Panel (Gray): A smartphone displaying an eye scanning UI. Label: "Smartphone Camera".
Three curving arrows (gold, gray, gray) merge toward the center column.

**CENTER COLUMN (40% width) — Artifacts & Segmentation:**
1. Top Section "Shared Signal Artifacts": Two distinct waveform glitches:
   (A) "Blink Artifact": Clean signal drops vertically to zero, holds, returns.
   (B) "Segmentation Glitches": Noisy burst/spike pattern from tracking loss.
2. Middle Inset "Segmentation Prerequisite" (CRITICAL): Schematic eye with eyelid partially covering pupil. Split into two icons: "Poor Segmentation" (Red ✗, noisy mask including eyelid) vs "High-Quality SAMv3" (Green ✓, clean circular mask excluding eyelid). Annotation: "Uncertainty must be resolved by segmentation (SAMv3), not TSFM".
3. Bottom Section "Physiological": Subtle sine-wave trace labeled "Hippus (0.3 Hz) — Requires Experimental Control".

**RIGHT COLUMN (30% width) — TSFM Pipeline:**
1. Input arrow from center column.
2. Block 1 (Gold): Rounded rectangle "Outlier Detection". Subtext: "MOMENT, LOF".
3. Block 2 (Blue): Rounded rectangle "Imputation". Subtext: "SAITS, CSDI".
4. NO baseline correction block — arrow flows directly from Imputation into final plot.
5. Output: Clean PLR waveform — classic "cliff and slope" curve: flat baseline → vertical drop → flat valley → slow exponential recovery. Label: "Reconstructed PLR".

### Negative Prompt
baseline correction box, cEEMD, glowing edges, neon, cyberpunk, brain anatomy, messy text, garbled labels, white background, stock photo, 3d render gloss, reflection, low resolution, jpeg artifacts

### Refinement Notes
- Emphasize the handheld pupillometer (our device) with gold accent; use actual SERI prototype appearance
- Eye trackers and smartphones in secondary gray
- The segmentation prerequisite inset is the KEY new element — must be visually prominent
- PLR waveform must be realistic cliff-and-slope, not a sine wave
- Keep signal waveforms recognizable but schematic
- SAMv3 reference: https://github.com/facebookresearch/sam3

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "target_model": "Nano Banana Pro",
    "figure_id": "fig-repo-98-v3-final",
    "style_reference": "STYLE-GUIDE.md (Medical Illustration / Economist)",
    "aspect_ratio": "16:9"
  },
  "generation_parameters": {
    "aesthetic_score": 9,
    "rendering_engine": "Octane Render (Matte)",
    "lighting": "Soft volumetric, top-left diffuse",
    "background_hex": "#FBF9F3"
  },
  "layout_architecture": {
    "columns": "3 (Left: Hardware, Center: Artifacts & Segmentation, Right: TSFM Pipeline)",
    "flow": "Left-to-Right convergence"
  },
  "master_prompt": {
    "style_header": "High-fidelity medical infographic, clean Scientific American aesthetic. Matte finish, no glowing sci-fi effects. Background color #FBF9F3 (off-white). Typography: Helvetica, clean dark charcoal.",

    "column_1_hardware": {
      "position": "Left Column (30% width)",
      "elements": [
        "1. Top Panel (Gold Outline): A specific black handheld pupillometer device. The device has a distinct eyepiece showing internal RED and BLUE stimulus lights (reference: SERI prototype). Label: 'Handheld Pupillometer (SERI, Singapore)'.",
        "2. Middle Panel (Gray): A standard desktop eye tracker setup (monitor + chinrest). Label: 'Eye Tracker (EyeLink, Tobii)'.",
        "3. Bottom Panel (Gray): A smartphone displaying an eye scanning UI. Label: 'Smartphone Camera'.",
        "Connections: Three curving arrows (gold, gray, gray) merging towards the center column."
      ]
    },

    "column_2_artifacts_logic": {
      "position": "Center Column (40% width)",
      "container_style": "Rounded panel with internal dividers",
      "elements": [
        "1. Top Section 'Shared Signal Artifacts': Display two distinct waveform glitches. (A) 'Blink Artifact': A clean signal that drops vertically to zero, holds, and returns. (B) 'Segmentation Glitches': A noisy burst/spike pattern caused by tracking loss.",
        "2. Middle Inset 'Segmentation Prerequisite' (CRITICAL): A visual callout box. Show a schematic eye with an eyelid partially covering the pupil. Split into two small icons: 'Poor Segmentation' (Red X, noisy mask including eyelid) vs 'High-Quality SAMv3' (Green Check, clean circular mask excluding eyelid). Annotation: 'Uncertainty must be resolved by segmentation (SAMv3), not TSFM'.",
        "3. Bottom Section 'Physiological': A subtle sine-wave trace labeled 'Hippus (0.3Hz) - Requires Experimental Control'."
      ]
    },

    "column_3_pipeline": {
      "position": "Right Column (30% width)",
      "flow": "Vertical Downward Flow",
      "elements": [
        "1. Input Arrow: Large arrow arriving from the Center Column.",
        "2. Block 1 (Gold): Rounded rectangle labeled 'Outlier Detection'. Subtext: 'MOMENT, LOF'.",
        "3. Block 2 (Blue): Rounded rectangle labeled 'Imputation'. Subtext: 'SAITS, CSDI'.",
        "4. Output Graph (NO Baseline Box): The arrow flows directly from Imputation into a final clean plot.",
        "5. The PLR Waveform: A clean 'Cliff and Slope' curve. Flat baseline -> vertical drop -> flat valley -> slow exponential recovery curve. Label: 'Reconstructed PLR'."
      ]
    },

    "negative_prompt": "baseline correction box, cEEMD, glowing edges, neon, cyberpunk, brain anatomy, messy text, garbled labels, white background, stock photo, 3d render gloss, reflection, low resolution, jpeg artifacts"
  }
}
```

## Alt Text

Three-column infographic showing pupillometry preprocessing challenges: three device types (handheld pupillometer from SERI Singapore, eye tracker, smartphone) converge on shared artifacts (blinks, segmentation glitches) that feed into a two-stage TSFM pipeline (outlier detection, imputation) producing a reconstructed PLR waveform. A central inset emphasizes that partial occlusion must be resolved by high-quality image segmentation (SAMv3) before TSFM processing, not by the TSFM stack itself. Physiological artifacts (hippus) require experimental control.

## Status

- [x] Draft created
- [x] v3 spec finalized (Gemini feedback incorporated)
- [ ] PNG regenerated with v3 spec
- [ ] Review passed
- [ ] JPG asset converted (rounded corners)
- [ ] Placed in README/docs
