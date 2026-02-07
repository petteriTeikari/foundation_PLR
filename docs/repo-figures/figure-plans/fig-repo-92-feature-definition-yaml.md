# fig-repo-92: From YAML to Signal: How PLR Features Are Defined

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-92 |
| **Title** | From YAML to Signal: How PLR Features Are Defined |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist |
| **Location** | `configs/PLR_FEATURIZATION/README.md`, `docs/explanation/features.md` |
| **Priority** | P3 (Medium) |

## Purpose

Show how each handcrafted PLR (Pupillary Light Reflex) feature is defined declaratively in YAML and maps to a specific time window and statistic on the raw PLR signal. This is a key architectural decision: features are NOT hardcoded in Python -- they are defined in configuration and the featurization code reads the YAML to extract features. This figure also contrasts the handcrafted approach with the embedding approach.

## Key Message

Each handcrafted feature is defined in `featuresBaseline.yaml` with four parameters: `time_from` (reference point), `time_start`/`time_end` (window bounds), `measure` (what to measure), and `stat` (summary statistic). This maps directly to a segment of the PLR signal waveform.

## Content Specification

### Panel 1: PLR Signal Waveform with Feature Windows

```
Pupil Diameter (normalized)
│
│ ─────────────────╮                                    ╭─────────────────
│                   ╲                                  ╱
│                    ╲                ╭──────────────╮╱
│    BASELINE         ╲              │  RECOVERY    │
│    [-5s, 0s]         ╲            ╱│  WINDOW      │
│    stat: median       ╲          ╱ │              │
│                        ╲        ╱  │              │
│                         ╲      ╱   │              │
│                          ╲    ╱    │              │
│                           ╲  ╱     │              │
│                            ╲╱      │              │
│                        MINIMUM     │              │
│                                    │              │
│───┬──────────┬──────────┬──────────┬──────────────┬──────────── time (s)
│  -5          0         1.5         5             12         15
│              │                     │              │
│         STIMULUS               STIMULUS       POST-
│         ONSET                  OFFSET         RESPONSE
│              │                                    │
│    ┌─────────┼─────────────────────┐              │
│    │  CONSTRICTION WINDOW          │              │
│    │  [onset, 0 to 15s]           │              │
│    │  MAX_CONSTRICTION: stat=min  │              │
│    │  PHASIC: stat=min (0-5s)    │              │
│    └───────────────────────────────┘              │
│                                                    │
│    ┌───────────────────────────────────────────────┐
│    │  POST-ILLUMINATION PUPIL RESPONSE             │
│    │  PIPR: [offset, 0 to 15s], stat=min          │
│    │  PIPR_AUC: [offset, 0 to 12s], stat=AUC     │
│    └───────────────────────────────────────────────┘
```

### Panel 2: YAML Feature Definitions (Actual Config)

```
File: configs/PLR_FEATURIZATION/featuresBaseline.yaml

FEATURES_METADATA:
  name: 'baseline'
  version: 1.0
  feature_method: 'handcrafted_features'

FEATURES:

  MAX_CONSTRICTION:             ← Feature name
    time_from: 'onset'          ← Reference: stimulus onset
    time_start: 0               ← Window start (seconds)
    time_end: 15                ← Window end (seconds)
    measure: 'amplitude'        ← What to measure
    stat: 'min'                 ← Summary: minimum value in window

  PHASIC:
    time_from: 'onset'
    time_start: 0
    time_end: 5
    measure: 'amplitude'
    stat: 'min'                 ← Minimum of first 5s after onset

  SUSTAINED:
    time_from: 'offset'
    time_start: -5
    time_end: 0
    measure: 'amplitude'
    stat: 'min'                 ← Minimum in last 5s before offset

  PIPR:
    time_from: 'offset'
    time_start: 0
    time_end: 15
    measure: 'amplitude'
    stat: 'min'                 ← Post-illumination minimum

  BASELINE:
    time_from: 'onset'
    time_start: -5
    time_end: 0
    measure: 'amplitude'
    stat: 'median'              ← Median of 5s before stimulus

  PIPR_AUC:
    time_from: 'offset'
    time_start: 0
    time_end: 12
    measure: 'amplitude'
    stat: 'AUC'                 ← Area under curve post-offset
```

### Panel 3: YAML Parameter Reference

```
PARAMETER DEFINITIONS
═════════════════════

  time_from:      Reference point on the PLR signal
                  ├── 'onset'   → Stimulus onset (light turns on)
                  └── 'offset'  → Stimulus offset (light turns off)

  time_start:     Start of feature window (seconds relative to time_from)
                  Can be negative (before reference point)

  time_end:       End of feature window (seconds relative to time_from)

  measure:        What physical quantity to extract
                  ├── 'amplitude' → Pupil diameter values
                  └── 'timing'    → Time to reach a condition

  stat:           Summary statistic applied to the window
                  ├── 'min'       → Minimum value (deepest constriction)
                  ├── 'max'       → Maximum value (peak recovery)
                  ├── 'mean'      → Average value
                  ├── 'median'    → Median value (robust to outliers)
                  ├── 'AUC'       → Area under curve
                  └── 'slope'     → Linear slope (rate of change)
```

### Panel 4: Handcrafted vs Embedding Contrast

```
APPROACH 1: HANDCRAFTED (configs/PLR_FEATURIZATION/)
══════════════════════════════════════════════════════
  PLR Signal → YAML-defined windows → 6 features per subject
                                       │
  ├── MAX_CONSTRICTION  (scalar)       │
  ├── PHASIC            (scalar)       │── 6-dimensional
  ├── SUSTAINED         (scalar)       │   feature vector
  ├── PIPR              (scalar)       │
  ├── BASELINE          (scalar)       │
  └── PIPR_AUC          (scalar)       │

  Config: configs/PLR_FEATURIZATION/featuresBaseline.yaml
  Code:   src/featurization/featurize_PLR.py
          src/featurization/featurizer_PLR_subject.py

APPROACH 2: EMBEDDINGS (configs/PLR_EMBEDDING/)
══════════════════════════════════════════════════════
  PLR Signal → MOMENT foundation model → 768-d embedding per subject
                                          │
  ├── dim_0   (learned)                   │
  ├── dim_1   (learned)                   │── 768-dimensional
  ├── ...                                 │   embedding vector
  └── dim_767 (learned)                   │

  Config: configs/PLR_EMBEDDING/
  Code:   src/featurization/embedding/
  Tasks:  "embedding", "reconstruction", "forecasting"

  Key difference:
  ├── Handcrafted: interpretable, domain-driven, 6 features
  └── Embeddings: learned, model-driven, 768 features
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/PLR_FEATURIZATION/featuresBaseline.yaml` | Main feature definitions (6 features) |
| `configs/PLR_FEATURIZATION/featuresSimple.yaml` | Simplified feature set variant |
| `configs/PLR_EMBEDDING/` | MOMENT embedding configuration |
| `configs/PLR_FEATURIZATION/README.md` | Feature documentation |

## Code Paths

| Module | Role |
|--------|------|
| `src/featurization/featurize_PLR.py` | Main featurization entry point |
| `src/featurization/featurizer_PLR_subject.py` | Per-subject feature extraction |
| `src/featurization/feature_utils.py` | Window extraction utilities |
| `src/featurization/feature_log.py` | Feature logging to MLflow |
| `src/featurization/flow_featurization.py` | Prefect flow for featurization |
| `src/featurization/subflow_handcrafted_featurization.py` | Handcrafted subflow |
| `src/featurization/embedding/` | MOMENT embedding extraction |
| `src/featurization/visualize_features.py` | Feature visualization utilities |

## Extension Guide

To add a new handcrafted feature:
1. Edit `configs/PLR_FEATURIZATION/featuresBaseline.yaml`
2. Add a new entry with `time_from`, `time_start`, `time_end`, `measure`, `stat`
3. If new `stat` type needed, implement in `src/featurization/feature_utils.py`
4. Re-run featurization: the code reads YAML dynamically
5. Update `FEATURES_METADATA.version` to track config changes

To add a new feature set:
1. Create `configs/PLR_FEATURIZATION/featuresNew.yaml`
2. Update `FEATURES_METADATA.name` to distinguish from baseline
3. Hydra override: `+PLR_FEATURIZATION=featuresNew`

To use a different embedding model:
1. Add model config in `configs/PLR_EMBEDDING/`
2. Implement extraction in `src/featurization/embedding/`
3. Register the model's task modes (embedding, reconstruction, forecasting)

Note: This is a repo documentation figure - shows HOW features are defined in config, NOT which features perform best.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-92",
    "title": "From YAML to Signal: How PLR Features Are Defined"
  },
  "content_architecture": {
    "primary_message": "Each handcrafted feature is defined in YAML with time_from, time_start, time_end, measure, and stat parameters that map to specific PLR signal segments.",
    "layout_flow": "Top: PLR waveform with overlaid feature windows. Middle: YAML definitions with arrows to waveform. Bottom: handcrafted vs embedding contrast.",
    "spatial_anchors": {
      "waveform": {"x": 0.05, "y": 0.02, "width": 0.9, "height": 0.3},
      "yaml_defs": {"x": 0.05, "y": 0.35, "width": 0.55, "height": 0.3},
      "param_ref": {"x": 0.65, "y": 0.35, "width": 0.3, "height": 0.3},
      "contrast": {"x": 0.05, "y": 0.7, "width": 0.9, "height": 0.25}
    },
    "key_structures": [
      {
        "name": "PLR Waveform",
        "role": "raw_signal",
        "is_highlighted": true,
        "labels": ["Pupil diameter over time"]
      },
      {
        "name": "Feature Windows",
        "role": "features",
        "is_highlighted": true,
        "labels": ["YAML-defined segments"]
      },
      {
        "name": "Handcrafted",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["6 features", "interpretable"]
      },
      {
        "name": "Embeddings",
        "role": "foundation_model",
        "is_highlighted": false,
        "labels": ["768 features", "learned"]
      }
    ],
    "callout_boxes": [
      {"heading": "CONFIG-DRIVEN", "body_text": "Features are NOT hardcoded in Python. YAML defines the windows; code reads YAML dynamically."},
      {"heading": "TWO APPROACHES", "body_text": "Handcrafted: 6 interpretable features from YAML. Embeddings: 768 learned dimensions from MOMENT."}
    ]
  }
}
```

## Alt Text

PLR signal waveform with color-coded time windows showing how each handcrafted feature maps from YAML configuration to a specific signal segment. Below, contrast between 6 handcrafted features and 768-dimensional MOMENT embeddings.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
