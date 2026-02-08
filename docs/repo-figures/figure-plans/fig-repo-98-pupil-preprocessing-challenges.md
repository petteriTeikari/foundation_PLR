# fig-repo-98: Pupil Signal Preprocessing Challenges

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-98 |
| **Title** | The Universal Preprocessing Bottleneck Across Pupillometry Hardware |
| **Complexity Level** | L2 (Technical concept) |
| **Target Persona** | Research Scientist, Biostatistician |
| **Location** | Root README.md (Pupil Preprocessing references section) |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Illustrate that raw pupil signals from ANY acquisition device -- handheld pupillometers, research eye trackers, smartphone cameras -- share the same artifact types, making preprocessing the universal bottleneck. This figure accompanies the "Pupil Preprocessing" references section in the README.

## Key Message

"Preprocessing is device-agnostic: blinks, tracking failures, and baseline drift affect all pupillometry hardware equally. The methods evaluated in this repository apply regardless of acquisition platform."

## Visual Concept

Three-column layout showing hardware diversity converging on shared preprocessing challenges:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  THE UNIVERSAL PREPROCESSING BOTTLENECK                                         │
├──────────────────────┬──────────────────────┬──────────────────────────────────┤
│                      │                      │                                  │
│  ACQUISITION         │  SHARED ARTIFACTS    │  PREPROCESSING                   │
│  DEVICES             │                      │  PIPELINE                        │
│                      │                      │                                  │
│  ┌────────────┐      │   ╔═══════════╗      │  ┌─────────────┐                │
│  │ Handheld   │──────│──→║  Blinks   ║──────│─→│  Outlier    │                │
│  │ Pupillometer│     │   ╠═══════════╣      │  │  Detection  │                │
│  │ (SERI, NL) │      │   ║  Tracking ║      │  └──────┬──────┘                │
│  └────────────┘      │   ║  failures ║      │         │                        │
│                      │   ╠═══════════╣      │  ┌──────▼──────┐                │
│  ┌────────────┐      │   ║  Baseline ║      │  │  Imputation │                │
│  │ Eye Tracker│──────│──→║  drift    ║──────│─→│  (TSFM/DL) │                │
│  │(EyeLink,   │     │   ╠═══════════╣      │  └──────┬──────┘                │
│  │ Tobii)     │      │   ║  Partial  ║      │         │                        │
│  └────────────┘      │   ║  occlusion║      │  ┌──────▼──────┐                │
│                      │   ╠═══════════╣      │  │  Baseline   │                │
│  ┌────────────┐      │   ║  Hippus   ║      │  │  Correction │                │
│  │ Smartphone │──────│──→║  (0.3 Hz) ║──────│─→│             │                │
│  │ Camera     │      │   ╚═══════════╝      │  └─────────────┘                │
│  └────────────┘      │                      │                                  │
│                      │                      │                                  │
├──────────────────────┴──────────────────────┴──────────────────────────────────┤
│                                                                                 │
│  KEY INSIGHT: Metrological artifacts (blinks, tracking failures) are targets    │
│  for TSFM-based preprocessing. Physiological artifacts (hippus, accommodation)  │
│  require experimental control, not algorithmic correction.                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Left-to-right convergence: diverse hardware → shared artifacts → common pipeline"
spatial_anchors:
  devices_column:
    x: 0.15
    y: 0.45
    content: "Three acquisition device types stacked vertically"
  artifacts_column:
    x: 0.45
    y: 0.45
    content: "Shared artifact types in central column"
  pipeline_column:
    x: 0.78
    y: 0.45
    content: "Three-stage preprocessing pipeline"
  insight_box:
    x: 0.5
    y: 0.88
    content: "Metrological vs physiological distinction callout"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Handheld pupillometer | `highlight_accent` | SERI device, infrared 30 Hz, chromatic stimuli |
| Eye tracker | `secondary_pathway` | EyeLink, Tobii, Pupil Labs research systems |
| Smartphone camera | `secondary_pathway` | PupilSense-type mobile acquisition |
| Blink artifact | `abnormal_warning` | Complete pupil occlusion, 100-400 ms |
| Tracking failure | `abnormal_warning` | Lost pupil boundary, noisy diameter |
| Baseline drift | `abnormal_warning` | Slow pupil size change from fatigue/adaptation |
| Partial occlusion | `abnormal_warning` | Eyelid interference with ellipse fitting |
| Hippus | `secondary_pathway` | 0.2-0.5 Hz spontaneous oscillation (physiological) |
| Outlier detection | `outlier_detection` | MOMENT, LOF, UniTS, ensembles |
| Imputation | `imputation` | SAITS, CSDI, MOMENT zero-shot |
| Baseline correction | `features` | cEEMD, filtering |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| All 3 devices | Shared artifacts | Converging arrows | "Same signal, same problems" |
| Shared artifacts | Preprocessing | Arrow | "Algorithmic correction" |
| Hippus | Separate note | Dashed arrow | "Experimental control, not preprocessing" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "METROLOGICAL" | Blinks, tracking failures = targets for TSFM preprocessing | Center-left |
| "PHYSIOLOGICAL" | Hippus, accommodation = experimental design, not algorithms | Center-right |
| "KEY INSIGHT" | Preprocessing methods are device-agnostic | Bottom strip |

## Text Content

### Labels (Max 30 chars each)
- Label 1: Handheld Pupillometer
- Label 2: Eye Tracker
- Label 3: Smartphone Camera
- Label 4: Blink Artifacts
- Label 5: Tracking Failures
- Label 6: Baseline Drift
- Label 7: Outlier Detection
- Label 8: Signal Reconstruction
- Label 9: Metrological vs Physiological

### Caption (for embedding)
Pupil preprocessing challenges are shared across all acquisition hardware -- handheld pupillometers, eye trackers, and smartphone cameras all produce the same artifact types requiring algorithmic correction.

## Prompts for Nano Banana Pro

### Style Prompt
Medical illustration quality, ray-traced ambient occlusion, soft volumetric lighting, Economist off-white background (#FBF9F3), clean editorial layout, elegant scientific illustration, matte finishes, hairline vector callout lines, professional data visualization. Eye anatomy cross-section showing pupil-iris boundary for the handheld device panel.

### Content Prompt
Create a three-column infographic showing the universal preprocessing bottleneck in pupillometry:

LEFT COLUMN - Three device types stacked vertically: (1) A handheld chromatic pupillometer device with infrared camera highlighted in gold, (2) A desktop eye tracker system in muted gray, (3) A smartphone with pupil detection overlay in muted gray. The handheld device is visually emphasized as the primary device.

CENTER COLUMN - Shared artifact types displayed as signal waveform snippets: blink (sudden dropout), tracking failure (noisy burst), baseline drift (slow wandering), partial occlusion (intermittent noise). Each shown as a small annotated waveform. Below, a dashed separator with "physiological" artifacts (hippus oscillation) shown separately with a note about experimental control.

RIGHT COLUMN - Three-stage preprocessing pipeline flowing downward: outlier detection (with small MOMENT/LOF labels), imputation/reconstruction, baseline correction. Clean signal emerging at bottom.

Converging arrows from all three devices point to the shared artifact column. The key insight strip at bottom reads: device-agnostic preprocessing.

### Refinement Notes
- Emphasize the handheld pupillometer (our device) with gold accent
- Eye trackers and smartphones in secondary gray
- The metrological vs physiological distinction should be visually clear
- Keep signal waveforms recognizable but schematic

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-98",
    "title": "The Universal Preprocessing Bottleneck Across Pupillometry Hardware"
  },
  "content_architecture": {
    "primary_message": "Preprocessing challenges are device-agnostic: all pupillometry hardware produces the same artifact types requiring algorithmic correction",
    "layout_flow": "Three-column left-to-right: diverse devices converge on shared artifacts, feed into common preprocessing pipeline",
    "spatial_anchors": {
      "devices": {"x": 0.15, "y": 0.45},
      "artifacts": {"x": 0.45, "y": 0.45},
      "pipeline": {"x": 0.78, "y": 0.45},
      "insight": {"x": 0.5, "y": 0.88}
    },
    "key_structures": [
      {"name": "Handheld Pupillometer", "role": "highlight_accent", "is_highlighted": true, "labels": ["SERI device", "30 Hz IR", "Chromatic"]},
      {"name": "Eye Tracker", "role": "secondary_pathway", "is_highlighted": false, "labels": ["EyeLink", "Tobii"]},
      {"name": "Smartphone", "role": "secondary_pathway", "is_highlighted": false, "labels": ["PupilSense"]},
      {"name": "Blink Artifact", "role": "abnormal_warning", "is_highlighted": true, "labels": ["100-400 ms dropout"]},
      {"name": "Tracking Failure", "role": "abnormal_warning", "is_highlighted": false, "labels": ["Noisy burst"]},
      {"name": "Baseline Drift", "role": "abnormal_warning", "is_highlighted": false, "labels": ["Slow wandering"]},
      {"name": "Outlier Detection", "role": "outlier_detection", "is_highlighted": true, "labels": ["MOMENT, LOF"]},
      {"name": "Imputation", "role": "imputation", "is_highlighted": true, "labels": ["SAITS, CSDI"]},
      {"name": "Baseline Correction", "role": "features", "is_highlighted": false, "labels": ["cEEMD"]}
    ],
    "callout_boxes": [
      {"heading": "METROLOGICAL", "body_text": "Blinks, tracking failures: targets for TSFM preprocessing"},
      {"heading": "PHYSIOLOGICAL", "body_text": "Hippus, accommodation: require experimental control"},
      {"heading": "KEY INSIGHT", "body_text": "Methods are device-agnostic"}
    ]
  }
}
```

## Alt Text

Three-column infographic showing pupillometry preprocessing challenges: three device types (handheld pupillometer, eye tracker, smartphone) converge on shared artifacts (blinks, tracking failures, baseline drift) that feed into a common preprocessing pipeline (outlier detection, imputation, baseline correction). Metrological artifacts are algorithmic targets; physiological artifacts require experimental control.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
