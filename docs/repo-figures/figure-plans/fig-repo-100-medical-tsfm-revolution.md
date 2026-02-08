# fig-repo-100: The Medical Biosignal TSFM Revolution

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-100 |
| **Title** | The Medical Biosignal Foundation Model Convergence |
| **Complexity Level** | L2-L3 (Technical landscape) |
| **Target Persona** | Research Scientist, ML Engineer |
| **Location** | Root README.md (Towards medical biosignal TSFMs section) |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:9 |

## Purpose

Visualize the convergence from modality-specific biosignal foundation models (EEG FMs, ECG FMs, PPG FMs) toward a hypothetical universal medical TSFM. Shows where MIRA sits (medical but forecasting-only) and the gap: a full-stack medical TSFM that can preprocess ANY biosignal.

## Key Message

"Medical biosignal FMs are proliferating by modality (EEG, ECG, PPG), but no single model yet handles all preprocessing tasks across modalities -- the post-MIRA frontier."

## Visual Concept

Evolutionary convergence diagram: three modality streams (EEG, ECG, PPG) flowing from left toward a convergence point on the right, with MIRA as an intermediate step and the gap clearly marked:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  THE MEDICAL BIOSIGNAL FM CONVERGENCE                                           │
│                                                                                 │
│  2024                2025                2025-2026           FRONTIER            │
│  ─────              ─────               ─────────           ────────            │
│                                                                                 │
│  ┌───────┐     ┌──────────────┐                                                │
│  │ EEG   │     │ EEGFormer    │                                                │
│  │       │────→│ Neuro-GPT    │──────────┐                                     │
│  │       │     │ (TUH Corpus) │          │                                     │
│  └───────┘     └──────────────┘          │        ┌─────────────────────┐      │
│                                          │        │                     │      │
│  ┌───────┐     ┌──────────────┐          ├───────→│  UNIVERSAL MEDICAL  │      │
│  │ ECG   │     │ ECG-FM       │          │        │  TSFM               │      │
│  │       │────→│ BenchECG     │──────────┤        │                     │      │
│  │       │     │ OpenECG      │          │        │  Full-stack:        │      │
│  └───────┘     │ (1.2M rec)   │          │        │  • Anomaly detection│      │
│                └──────────────┘          │        │  • Imputation       │      │
│                                          │        │  • Classification   │      │
│  ┌───────┐     ┌──────────────┐          │        │  • Forecasting      │      │
│  │ PPG   │     │ PaPaGei      │──────────┘        │                     │      │
│  │       │────→│ Pulse-PPG    │                    │  Pretrained on:     │      │
│  │       │     │ PPG-Distill  │                    │  EEG + ECG + PPG +  │      │
│  └───────┘     │ (57K+ hours) │          ┌───┐    │  PLR + ERG + ...    │      │
│                └──────────────┘          │   │    └─────────────────────┘      │
│                                          │ ? │              ▲                   │
│  ┌────────────────────────────┐          │   │              │                   │
│  │  MIRA (2025)               │──────────┘   │              │                   │
│  │  454B medical timepoints   │  Forecasting │              │                   │
│  │  MIMIC-III/IV, PTB-XL     │  ONLY        │    GAP: No full-stack           │
│  │  ✅ Forecasting            │──────────────┘    medical TSFM exists          │
│  │  ❌ Anomaly detection      │                                                │
│  │  ❌ Imputation             │                                                │
│  │  ❌ Classification         │                                                │
│  └────────────────────────────┘                                                │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐            │
│  │  INSIGHT: Generalist TSFMs (MOMENT) currently outperform       │            │
│  │  domain-specific models by 27% win score in full-tuning        │            │
│  │  (Kataria 2025), but the gap narrows with domain pretraining.  │            │
│  └─────────────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Left-to-right timeline: modality-specific models converge toward universal medical TSFM"
spatial_anchors:
  modality_origins:
    x: 0.1
    y: 0.4
    content: "Three biosignal modality icons (EEG, ECG, PPG waveforms)"
  modality_models:
    x: 0.35
    y: 0.4
    content: "Modality-specific FM clusters"
  mira_node:
    x: 0.55
    y: 0.72
    content: "MIRA as partial step (medical but forecasting-only)"
  convergence_point:
    x: 0.8
    y: 0.4
    content: "Hypothetical universal medical TSFM"
  insight_strip:
    x: 0.5
    y: 0.92
    content: "Generalist vs specialist insight from Kataria 2025"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| EEG waveform | `primary_pathway` | Characteristic multi-channel neural oscillation |
| ECG waveform | `primary_pathway` | PQRST complex |
| PPG waveform | `primary_pathway` | Photoplethysmography pulse |
| EEGFormer + Neuro-GPT | `foundation_model` | EEG-specific FMs (TUH Corpus, ~1500 hours) |
| ECG-FM + BenchECG + OpenECG | `foundation_model` | ECG-specific FMs (1.2M recordings) |
| PaPaGei + Pulse-PPG + PPG-Distill | `foundation_model` | PPG-specific FMs (57K+ hours) |
| MIRA | `abnormal_warning` | Medical TSFM, 454B timepoints, forecasting-ONLY |
| Universal Medical TSFM | `highlight_accent` | Hypothetical full-stack: anomaly + imputation + classification + forecasting |
| Convergence arrows | `primary_pathway` | Three modality streams merging |
| Gap indicator | `abnormal_warning` | No full-stack medical TSFM yet exists |
| Kataria finding | `callout_box` | Specialist +27% win score over generalist in full-tuning |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| EEG models | Convergence | Flowing arrow | "EEG representations" |
| ECG models | Convergence | Flowing arrow | "ECG representations" |
| PPG models | Convergence | Flowing arrow | "PPG representations" |
| MIRA | Convergence | Dashed arrow + X | "Forecasting only" |
| Convergence | Universal TSFM | Bold arrow + ? | "The frontier" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "MIRA (2025)" | 454B medical timepoints, forecasting-only | Bottom-left |
| "THE GAP" | No full-stack medical TSFM for anomaly + imputation + classification | Right side |
| "SPECIALIST vs GENERALIST" | Kataria 2025: domain-specific +27% over generalist in full-tuning | Bottom strip |

## Text Content

### Labels (Max 30 chars each)
- Label 1: EEG Foundation Models
- Label 2: ECG Foundation Models
- Label 3: PPG Foundation Models
- Label 4: MIRA (Forecasting Only)
- Label 5: Universal Medical TSFM
- Label 6: Full-Stack Gap
- Label 7: Anomaly Detection
- Label 8: Imputation
- Label 9: Classification
- Label 10: Forecasting

### Caption (for embedding)
The medical biosignal FM landscape: modality-specific models (EEG, ECG, PPG) are proliferating, but no universal medical TSFM yet supports all preprocessing tasks. MIRA provides medical pretraining but only forecasting. The frontier is a full-stack medical TSFM.

## Prompts for Nano Banana Pro

### Style Prompt
Medical illustration quality, ray-traced ambient occlusion, soft volumetric lighting, Economist off-white background (#FBF9F3), elegant scientific illustration, clean editorial layout, muted color accents, flowing organic arrows showing convergence, professional data visualization, Scientific American infographic style.

### Content Prompt
Create a convergence diagram showing the medical biosignal foundation model revolution:

LEFT SIDE - Three biosignal modality streams, each represented by a characteristic waveform: EEG (multi-channel brain oscillations in deep blue), ECG (PQRST complex in teal), PPG (pulse waveform in gold). Each has small model name cards below.

CENTER - The three streams flow rightward through organic arrows, showing individual modality-specific foundation models: EEG branch has EEGFormer and Neuro-GPT, ECG branch has ECG-FM and OpenECG, PPG branch has PaPaGei and Pulse-PPG.

BOTTOM-CENTER - MIRA positioned as a separate node below the main flow, connected by a dashed arrow with a warning indicator. Card shows "454B medical timepoints" but lists only forecasting as supported, with anomaly detection, imputation, classification crossed out.

RIGHT SIDE - Convergence point marked with a gold-accented star showing "Universal Medical TSFM" with four task capabilities listed (anomaly, imputation, classification, forecasting). A question mark indicates this is hypothetical.

BOTTOM STRIP - Insight callout from Kataria 2025: specialists beat generalists by 27% in full-tuning.

### Refinement Notes
- The convergence metaphor should feel evolutionary, not forced
- MIRA should be clearly positioned as a partial step (medical but incomplete)
- The gap should be visually obvious (maybe a dotted outline for the universal TSFM)
- Characteristic waveforms for each modality should be recognizable at a glance
- Keep the three modality colors consistent (deep blue EEG, teal ECG, gold PPG)

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-100",
    "title": "The Medical Biosignal Foundation Model Convergence"
  },
  "content_architecture": {
    "primary_message": "Medical biosignal FMs are proliferating by modality, but no universal full-stack medical TSFM exists yet -- the post-MIRA frontier",
    "layout_flow": "Left-to-right convergence: three modality streams merge toward hypothetical universal medical TSFM",
    "spatial_anchors": {
      "modality_origins": {"x": 0.1, "y": 0.4},
      "modality_models": {"x": 0.35, "y": 0.4},
      "mira": {"x": 0.55, "y": 0.72},
      "convergence": {"x": 0.8, "y": 0.4},
      "insight": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {"name": "EEG Stream", "role": "primary_pathway", "is_highlighted": true, "labels": ["EEGFormer", "Neuro-GPT"]},
      {"name": "ECG Stream", "role": "primary_pathway", "is_highlighted": true, "labels": ["ECG-FM", "OpenECG"]},
      {"name": "PPG Stream", "role": "primary_pathway", "is_highlighted": true, "labels": ["PaPaGei", "Pulse-PPG"]},
      {"name": "MIRA", "role": "abnormal_warning", "is_highlighted": true, "labels": ["454B timepoints", "Forecasting ONLY"]},
      {"name": "Universal Medical TSFM", "role": "highlight_accent", "is_highlighted": true, "labels": ["Full-stack", "Hypothetical"]}
    ],
    "callout_boxes": [
      {"heading": "MIRA", "body_text": "Medical pretraining (MIMIC-III/IV) but forecasting-only"},
      {"heading": "THE GAP", "body_text": "No full-stack medical TSFM for anomaly + imputation + classification"},
      {"heading": "INSIGHT", "body_text": "Specialist models +27% over generalists in full-tuning (Kataria 2025)"}
    ]
  }
}
```

## Alt Text

Convergence diagram showing three biosignal modality streams (EEG, ECG, PPG) with their respective foundation models flowing toward a hypothetical universal medical TSFM. MIRA positioned as partial step (medical but forecasting-only). The gap: no full-stack medical TSFM yet supports anomaly detection, imputation, and classification across modalities.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
