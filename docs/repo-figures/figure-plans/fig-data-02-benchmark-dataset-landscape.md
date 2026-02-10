# fig-data-02: Common TS Benchmark Dataset Landscape

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-data-02 |
| **Title** | Common TS Benchmark Dataset Landscape |
| **Complexity Level** | L2 (Research Scientist) |
| **Target Persona** | Research Scientist |
| **Location** | `data/README.md`, root `README.md` |
| **Priority** | P2 (High) |
| **Aspect Ratio** | 16:10 |

## Purpose

Map standard benchmark datasets used in TSFM literature (ETTh1/h2, Weather, ECL, PSM, MSL, SMAP, SMD, SWaT), showing which models were pretrained/evaluated on which datasets, and where PLR sits in this landscape. Developers need to understand the domain gap when adapting TSFMs to medical biosignals.

## Key Message

"TSFMs are benchmarked on industrial/weather datasets. PLR is a medical biosignal with different characteristics -- understanding this gap informs adaptation choices."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               COMMON TS BENCHMARK DATASET LANDSCAPE                              │
│               Where PLR fits among standard TSFM benchmarks                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DATASET PROPERTIES                                                              │
│  ══════════════════                                                              │
│                                                                                  │
│  │ Dataset  │ Domain      │ Length    │ Hz    │ Ch │ Task          │ Used By    │
│  │ ──────── │ ─────────── │ ──────── │ ───── │ ── │ ───────────── │ ────────── │
│  │ ETTh1/h2 │ Energy      │ 17,420   │ 1/hr  │ 7  │ Forecast      │ M,U,T      │
│  │ Weather  │ Meteorology │ 52,696   │ 10min │ 21 │ Forecast      │ M,U,T      │
│  │ ECL      │ Electricity │ 26,304   │ 1/hr  │ 321│ Forecast      │ M,U,T      │
│  │ PSM      │ Industrial  │ 87,841   │ 1min  │ 25 │ Anomaly Det.  │ U,T        │
│  │ MSL      │ Aerospace   │ 58,317   │ —     │ 55 │ Anomaly Det.  │ U,T        │
│  │ SMAP     │ Aerospace   │ 135,183  │ —     │ 25 │ Anomaly Det.  │ U,T        │
│  │ SMD      │ Server      │ 708,420  │ 1min  │ 38 │ Anomaly Det.  │ U,T        │
│  │ SWaT     │ Industrial  │ 496,800  │ 1sec  │ 51 │ Anomaly Det.  │ T          │
│  │ ──────── │ ─────────── │ ──────── │ ───── │ ── │ ───────────── │ ────────── │
│  │ PLR      │ MEDICAL     │ 1,981    │ 30Hz  │ 1  │ Anom+Imp+Cls  │ This work  │
│  │          │ (biosignal) │          │       │    │               │            │
│                                                                                  │
│  Legend: M=MOMENT, U=UniTS, T=TimesNet, S=SAITS                                │
│                                                                                  │
│  KEY DIFFERENCES                                                                │
│  ═══════════════                                                                │
│                                                                                  │
│  PLR vs standard benchmarks:                                                    │
│  • Length: ~2K vs 17K-700K (PLR is SHORT)                                       │
│  • Channels: 1 vs 7-321 (PLR is UNIVARIATE)                                    │
│  • Sampling: 30Hz physiological vs hourly/minute industrial                     │
│  • Domain: Medical biosignal vs energy/weather/server                           │
│  • Tasks: Combined anomaly+imputation+classification pipeline                   │
│                                                                                  │
│  These differences explain WHY adaptation code is needed                        │
│  (padding, windowing, format conversion)                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Standard datasets | `secondary_pathway` | ETTh, Weather, ECL, PSM, MSL, SMAP, SMD, SWaT |
| PLR dataset | `highlight_accent` | Highlighted as outlier medical domain |
| Model columns | `foundation_model` | Which model uses which dataset |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "DOMAIN GAP" | "PLR is a short, univariate medical biosignal. Standard benchmarks are long, multivariate industrial/weather signals." | Bottom right |

## Text Content

### Labels
- PLR: 1,981 x 1ch x 30Hz
- Benchmarks: 17K-700K x 7-321ch
- Domain gap: medical vs industrial

### Caption
Standard TSFM benchmarks (ETTh1/h2, Weather, ECL, PSM, MSL, SMAP, SMD, SWaT) are long, multivariate industrial or weather datasets. PLR is a short (1,981 timepoints), univariate (1 channel), high-frequency (30Hz) medical biosignal. This domain gap drives the need for adaptation code (padding, windowing, format conversion) in `src/data_io/`.

## Prompts for Nano Banana Pro

### Style Prompt
Matrix/landscape table. Rows: datasets with properties. Columns: domain, length, Hz, channels, task, models. PLR row highlighted as outlier. Key differences callout below. Clean, structured, technical style.

### Content Prompt
Create "TS Benchmark Dataset Landscape" infographic:

**TOP**: Property table with 9 datasets + PLR highlighted

**BOTTOM**: Key differences callout:
- Length comparison (2K vs 700K)
- Channel count (1 vs 321)
- Domain (medical vs industrial)
- Why adaptation is needed

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/MODELS/MOMENT.yaml` | MOMENT: pretrained on diverse time-series corpora |
| `configs/MODELS/TimesNet.yaml` | TimesNet: Time-Series-Library benchmark suite |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/ts_format.py` | `write_as_psm()`: Adapts PLR to PSM benchmark format |
| `src/data_io/torch_data.py` | Handles PLR-specific padding/windowing for short sequences |

## Extension Guide

When onboarding a new TSFM:
1. Check which benchmark datasets the model was pretrained/evaluated on
2. Compare PLR properties (length, channels, Hz) with those benchmarks
3. Identify the format gap and implement adapters in `src/data_io/ts_format.py`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "data-02",
    "title": "Common TS Benchmark Dataset Landscape"
  },
  "content_architecture": {
    "primary_message": "PLR is a short, univariate medical biosignal among long, multivariate industrial benchmarks. This domain gap drives adaptation needs.",
    "layout_flow": "Table at top with PLR highlighted, key differences callout below",
    "spatial_anchors": {
      "dataset_table": {"x": 0.5, "y": 0.35},
      "plr_highlight": {"x": 0.5, "y": 0.6},
      "key_differences": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "Standard Benchmarks",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["ETTh", "Weather", "ECL", "PSM", "MSL", "SMAP"]
      },
      {
        "name": "PLR Dataset",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["1981 x 1ch x 30Hz", "Medical biosignal"]
      }
    ],
    "callout_boxes": [
      {"heading": "DOMAIN GAP", "body_text": "Short (2K vs 700K), univariate (1 vs 321 channels), medical vs industrial. Adaptation code bridges this gap."}
    ]
  }
}
```

## Alt Text

Dataset landscape table comparing 8 standard TSFM benchmarks (ETTh, Weather, ECL, PSM, MSL, SMAP, SMD, SWaT) with PLR. PLR highlighted as outlier: short (1981 vs 17K-700K), univariate (1 vs 7-321 channels), medical biosignal vs industrial/weather.

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in data/README.md
