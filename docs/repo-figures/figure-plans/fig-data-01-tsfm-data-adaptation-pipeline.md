# fig-data-01: TSFM Data Adaptation Pipeline

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-data-01 |
| **Title** | TSFM Data Adaptation Pipeline |
| **Complexity Level** | L3 (Research Scientist / ML Engineer) |
| **Target Persona** | ML Engineer |
| **Location** | `data/README.md`, `src/data_io/README.md` |
| **Priority** | P1 (Critical) |
| **Aspect Ratio** | 16:10 |

## Purpose

Show how a single raw PLR signal (507 subjects x 1981 timepoints) transforms into four model-specific formats. Developers onboarding a new TSFM need to understand which code paths handle format conversion and what each model expects.

## Key Message

"One PLR signal, four formats -- each TSFM has unique sequence length, NaN handling, and windowing requirements. All conversions live in `src/data_io/`."

## Visual Concept

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               TSFM DATA ADAPTATION PIPELINE                                      │
│               One PLR signal → Four model-specific formats                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  RAW PLR SIGNAL (507 subjects x 1981 timepoints)                                │
│  ═══════════════════════════════════════════════                                 │
│                                                                                  │
│            ┌──── Common Preprocessing ────┐                                     │
│            │  Trim/pad, outlier masking    │                                     │
│            │  src/data_io/data_utils.py    │                                     │
│            └──────────┬───────────────────┘                                     │
│                       │                                                          │
│         ┌─────────────┼─────────────┬──────────────┐                            │
│         ▼             ▼             ▼              ▼                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐                   │
│  │  MOMENT     │ │  UniTS      │ │  TimesNet   │ │  SAITS      │                   │
│  ├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤                   │
│  │ NaN: Keep   │ │ NaN: Median │ │ NaN: Median │ │ NaN: Keep   │                   │
│  │  + mask     │ │  fill       │ │  fill       │ │  + mask     │                   │
│  │ Seq: 512    │ │ Seq: Native │ │ Seq: Native │ │ Seq: Native │                   │
│  │ Pad: 2048   │ │ Format: PSM │ │ Format:     │ │ Format:     │                   │
│  │ Windows: 4  │ │  CSV        │ │  PyPOTS     │ │  PyPOTS     │                   │
│  │  x 512      │ │             │ │  dict       │ │  dict       │                   │
│  │ Output:     │ │ Output:     │ │ Output:     │ │ Output:     │                   │
│  │ Tensor      │ │ CSV files   │ │ dict        │ │ dict        │                   │
│  │ Dataset     │ │             │ │             │ │             │                   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘                   │
│                                                                                  │
│  Code Paths:                                                                    │
│  ─────────────                                                                  │
│  MOMENT:   data_utils.py → transform_data_for_momentfm()                       │
│            torch_data.py → PLRDataset (pad, window, mask)                       │
│  UniTS:    ts_format.py  → write_as_psm()                                      │
│  TimesNet: ts_format.py  → write_as_pypots_dict() (via PyPOTS)                 │
│  SAITS:    ts_format.py  → write_as_pypots_dict() (via PyPOTS)                 │
│                                                                                  │
│  Config:   configs/MODELS/{MOMENT,TimesNet,SAITS,CSDI}.yaml                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Raw PLR Signal | `raw_signal` | Input: 507 x 1981 array |
| Common Preprocessing | `outlier_detection` | Shared trim/pad/mask step |
| MOMENT branch | `foundation_model` | NaN-tolerant, pad to 2048, 4x512 windows |
| UniTS branch | `foundation_model` | Median fill, PSM CSV format |
| TimesNet branch | `traditional_method` | Median fill, PyPOTS dict |
| SAITS branch | `traditional_method` | NaN-tolerant, PyPOTS dict |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| Raw Signal | Common Preprocessing | Arrow | "trim/pad" |
| Common Preprocessing | MOMENT | Arrow | "pad 2048, split 4x512" |
| Common Preprocessing | UniTS | Arrow | "median fill, write CSV" |
| Common Preprocessing | TimesNet | Arrow | "median fill, PyPOTS" |
| Common Preprocessing | SAITS | Arrow | "keep NaN, PyPOTS" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "NaN HANDLING" | "MOMENT and SAITS use input_mask for NaN. UniTS and TimesNet require median fill." | Right side |

## Text Content

### Labels
- Raw PLR: 507 x 1981
- MOMENT: pad→2048, 4 windows
- UniTS: PSM CSV format
- TimesNet: PyPOTS dict
- SAITS: PyPOTS dict

### Caption
Raw PLR signals (507 subjects, 1981 timepoints each) pass through common preprocessing then diverge into four model-specific formats. MOMENT uses NaN-tolerant input with masking, padding to 2048 and splitting into 4x512 windows. UniTS expects PSM-format CSV with median-filled NaN values. TimesNet and SAITS use PyPOTS dict format, with TimesNet requiring median fill and SAITS accepting NaN with masks.

## Prompts for Nano Banana Pro

### Style Prompt
Pipeline/funnel diagram. Single input at top fanning into four branches below. Each branch is a distinct color-coded card showing model name, NaN handling, sequence length, and output format. Code path annotations at bottom. Clean, technical, architectural style.

### Content Prompt
Create "TSFM Data Adaptation Pipeline" diagram:

**TOP**: Raw PLR signal block (507 x 1981 timepoints)

**MIDDLE**: Common preprocessing box with arrow down

**BOTTOM**: Four parallel cards for MOMENT, UniTS, TimesNet, SAITS showing:
- NaN handling strategy
- Sequence length / windowing
- Output format
- Code path reference

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/MODELS/MOMENT.yaml` | MOMENT: seq_len=512, pad_ts=True, split_subjects_to_windows=True |
| `configs/MODELS/TimesNet.yaml` | TimesNet: via PyPOTS, Time-Series-Library format |
| `configs/MODELS/SAITS.yaml` | SAITS: via PyPOTS, NaN-tolerant |
| `configs/MODELS/CSDI.yaml` | CSDI: via PyPOTS, diffusion-based imputation |

## Code Paths

| Module | Role |
|--------|------|
| `src/data_io/data_utils.py` | `transform_data_for_momentfm()`: MOMENT-specific padding and windowing |
| `src/data_io/torch_data.py` | `PLRDataset`: PyTorch dataset with padding, windowing, masking |
| `src/data_io/ts_format.py` | `write_as_psm()`, `write_as_pypots_dict()`: format converters |
| `src/data_io/data_wrangler.py` | `convert_df_to_dict()`: DataFrame to data dict conversion |

## Extension Guide

To add a new TSFM format:
1. Identify expected input format from the model's documentation
2. Add a `write_as_{format}()` function to `src/data_io/ts_format.py`
3. Add model config YAML to `configs/MODELS/`
4. Wire into pipeline via `src/anomaly_detection/` or `src/imputation/`
5. Add tests verifying format correctness

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "data-01",
    "title": "TSFM Data Adaptation Pipeline"
  },
  "content_architecture": {
    "primary_message": "One PLR signal transforms into four model-specific formats, each with unique NaN handling, sequence length, and output format requirements.",
    "layout_flow": "Top-down funnel: raw signal → common preprocessing → four parallel branches",
    "spatial_anchors": {
      "raw_signal": {"x": 0.5, "y": 0.1},
      "preprocessing": {"x": 0.5, "y": 0.3},
      "moment": {"x": 0.15, "y": 0.6},
      "units": {"x": 0.38, "y": 0.6},
      "timesnet": {"x": 0.62, "y": 0.6},
      "saits": {"x": 0.85, "y": 0.6},
      "code_paths": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Raw PLR Signal",
        "role": "raw_signal",
        "is_highlighted": true,
        "labels": ["507 subjects", "1981 timepoints"]
      },
      {
        "name": "MOMENT",
        "role": "foundation_model",
        "is_highlighted": true,
        "labels": ["NaN+mask", "pad 2048", "4x512 windows"]
      },
      {
        "name": "UniTS",
        "role": "foundation_model",
        "is_highlighted": false,
        "labels": ["median fill", "PSM CSV"]
      },
      {
        "name": "TimesNet",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["median fill", "PyPOTS dict"]
      },
      {
        "name": "SAITS",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["NaN+mask", "PyPOTS dict"]
      }
    ],
    "callout_boxes": [
      {"heading": "NaN HANDLING", "body_text": "MOMENT and SAITS tolerate NaN with input_mask. UniTS and TimesNet require median fill."}
    ]
  }
}
```

## Alt Text

Pipeline diagram showing raw PLR signal (507 subjects, 1981 timepoints) flowing through common preprocessing into four model-specific branches: MOMENT (NaN+mask, pad 2048, 4x512 windows, TensorDataset), UniTS (median fill, PSM CSV), TimesNet (median fill, PyPOTS dict), SAITS (NaN+mask, PyPOTS dict).

## Status

- [x] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in data/README.md
