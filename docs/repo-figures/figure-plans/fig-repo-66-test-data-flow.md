# fig-repo-66: Where Test Data Comes From

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-66 |
| **Title** | Where Test Data Comes From |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `tests/README.md`, `docs/explanation/test-architecture.md` |
| **Priority** | P2 (High) |

## Purpose

Show the 4 data channels that feed the test suite, their isolation boundaries, and which test markers use which channel. Developers need to understand that synthetic data never touches production paths, and that tests degrade gracefully when real data is absent.

## Key Message

Test data flows through 4 isolated channels: synthetic DB (for unit/guardrail tests), demo subjects (for visualization demos), real DB (for integration/e2e, skipped if absent), and in-memory fixtures (for pure function tests). Synthetic NEVER touches production.

## Content Specification

### Panel 1: Four Data Channels

```
┌─────────────────────────────────────────────────────────────────────────┐
│         WHERE TEST DATA COMES FROM: 4 CHANNELS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CHANNEL 1: Synthetic DB               CHANNEL 2: Demo Subjects        │
│  ┌──────────────────────────┐          ┌──────────────────────────┐    │
│  │ data/synthetic/           │          │ configs/demo_subjects.yaml│    │
│  │   SYNTH_PLR_DEMO.db      │          │                          │    │
│  │                           │          │ 8 subjects:              │    │
│  │ Contains:                 │          │   Control: H001-H004     │    │
│  │   Synthetic PLR signals   │          │   Glaucoma: G001-G004    │    │
│  │   Known ground truths     │          │                          │    │
│  │   Deterministic outputs   │          │ Stratified by:           │    │
│  │                           │          │   Outlier percentage      │    │
│  │ Used by:                  │          │   Signal quality          │    │
│  │   @mark.unit              │          │                          │    │
│  │   @mark.guardrail         │          │ Used by:                 │    │
│  │                           │          │   Visualization tests     │    │
│  │ Created by:               │          │   Demo data fixtures      │    │
│  │   src/synthetic/ modules  │          │                          │    │
│  └──────────────────────────┘          └──────────────────────────┘    │
│                                                                         │
│  CHANNEL 3: Real DB (optional)         CHANNEL 4: In-Memory Fixtures   │
│  ┌──────────────────────────┐          ┌──────────────────────────┐    │
│  │ data/public/              │          │ tests/conftest.py         │    │
│  │   foundation_plr_         │          │                          │    │
│  │   results.db              │          │ sample_plr_array()       │    │
│  │   cd_diagram_data.duckdb  │          │   np.ndarray (8x1981)   │    │
│  │                           │          │   Synthetic PLR response │    │
│  │ Contains:                 │          │                          │    │
│  │   316 real configs        │          │ sample_outlier_mask()    │    │
│  │   STRATOS metrics         │          │   5% random outliers     │    │
│  │   Bootstrap CIs           │          │                          │    │
│  │                           │          │ minimal_cfg()            │    │
│  │ Used by:                  │          │   OmegaConf mock config  │    │
│  │   @mark.integration       │          │                          │    │
│  │   @mark.e2e               │          │ mock_mlflow()            │    │
│  │                           │          │   Patched MLflow calls   │    │
│  │ If absent:                │          │                          │    │
│  │   Tests SKIP gracefully   │          │ Used by:                 │    │
│  │   (Group A skips)         │          │   All test types          │    │
│  └──────────────────────────┘          │   No file I/O needed     │    │
│                                         └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Isolation Boundary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ISOLATION RULE: Synthetic NEVER Touches Production                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    PRODUCTION ZONE                                      │
│                    ┌──────────────────────────────────┐                 │
│                    │ data/public/                      │                 │
│                    │   foundation_plr_results.db      │                 │
│                    │   cd_diagram_data.duckdb         │                 │
│                    │                                   │                 │
│                    │ figures/generated/                │                 │
│                    │   (from real data only)           │                 │
│                    │                                   │                 │
│                    │ data/r_data/                      │                 │
│                    │   (R script outputs)              │                 │
│          ┌────────┤                                   │                 │
│          │ BANNED │ /home/petteri/mlruns/              │                 │
│          │CROSSING│   (410 real MLflow runs)           │                 │
│          │        └──────────────────────────────────┘                 │
│          │                                                              │
│          │        ┌──────────────────────────────────┐                 │
│          │        │ data/synthetic/                    │                 │
│          │        │   SYNTH_PLR_DEMO.db               │                 │
│          │        │                                   │                 │
│          │        │ In-memory fixtures                 │                 │
│          │        │   (np.random arrays, mocks)        │                 │
│          └────────┤                                   │                 │
│                    │ TESTING ZONE                      │                 │
│                    └──────────────────────────────────┘                 │
│                                                                         │
│  ENFORCEMENT:                                                           │
│  ├── Pre-commit: extraction-isolation-check                             │
│  ├── Pre-commit: figure-isolation-check                                 │
│  ├── Config: configs/data_isolation.yaml                                │
│  └── Test: test_data_provenance.py (P0 figure QA)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Channel-to-Marker Mapping

```
DATA CHANNEL             TEST MARKERS THAT USE IT
───────────────────────  ──────────────────────────────────────
Synthetic DB             unit, guardrail (deterministic checks)
Demo Subjects YAML       unit (visualization, feature extraction)
Real DB (optional)       integration, e2e, data (skip if absent)
In-Memory Fixtures       unit, guardrail (pure functions, no I/O)

GRACEFUL DEGRADATION:
  Real DB absent   → 43 tests skip (Group A)
  R_DATA_DIR absent → 34 tests skip (Group B)
  Demo DB absent    → 11 tests skip (Group E)
  In-memory fixtures → ALWAYS available (no skips)
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: 4 channels → isolation boundary → marker mapping"
spatial_anchors:
  four_channels:
    x: 0.5
    y: 0.25
    content: "4 data channels with contents, users, and creation method"
  isolation_boundary:
    x: 0.5
    y: 0.6
    content: "Production vs testing zone with enforcement mechanisms"
  marker_mapping:
    x: 0.5
    y: 0.9
    content: "Which markers use which data channel"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/demo_subjects.yaml` | 8 demo subjects (H001-H004 control, G001-G004 glaucoma) |
| `configs/data_isolation.yaml` | Isolation rules for synthetic vs production data |
| `data/private/subject_lookup.yaml` | Maps Hxxx/Gxxx to PLRxxxx (gitignored) |
| `tests/conftest.py` | Path constants: RESULTS_DB, DEMO_DB, R_DATA_DIR |

## Code Paths

| Module | Role |
|--------|------|
| `tests/conftest.py` | Path constants (RESULTS_DB, DEMO_DB), in-memory fixtures |
| `tests/test_figure_qa/conftest.py` | R_DATA_DIR and FIGURES_DIR path fixtures |
| `scripts/check_extraction_isolation.py` | Pre-commit: synthetic not in production |
| `scripts/check_figure_isolation.py` | Pre-commit: synthetic not in figures/ |
| `src/utils/data_mode.py` | Runtime data mode (synthetic vs production) |

## Extension Guide

To add a new data channel:
1. Define the path constant in `tests/conftest.py`
2. Create a fixture with `pytest.skip()` if the path does not exist
3. Document which test markers should use this channel
4. Add isolation enforcement if the channel contains sensitive data
5. Update `configs/data_isolation.yaml` with the new boundary

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-66",
    "title": "Where Test Data Comes From"
  },
  "content_architecture": {
    "primary_message": "Test data flows through 4 isolated channels: synthetic DB, demo subjects, real DB (optional), and in-memory fixtures. Synthetic NEVER touches production.",
    "layout_flow": "Top-down: 4 channels, isolation boundary, marker mapping",
    "spatial_anchors": {
      "four_channels": {"x": 0.5, "y": 0.25},
      "isolation_boundary": {"x": 0.5, "y": 0.6},
      "marker_mapping": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Synthetic DB",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["data/synthetic/SYNTH_PLR_DEMO.db", "unit + guardrail"]
      },
      {
        "name": "Demo Subjects",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["configs/demo_subjects.yaml", "H001-H004, G001-G004"]
      },
      {
        "name": "Real DB (optional)",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["data/public/foundation_plr_results.db", "integration + e2e"]
      },
      {
        "name": "In-Memory Fixtures",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["sample_plr_array()", "minimal_cfg()", "mock_mlflow()"]
      },
      {
        "name": "Isolation Boundary",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["BANNED CROSSING", "extraction-isolation-check", "figure-isolation-check"]
      }
    ],
    "callout_boxes": [
      {"heading": "ISOLATION RULE", "body_text": "Synthetic data NEVER touches production paths. Enforced by pre-commit hooks and figure QA tests."}
    ]
  }
}
```

## Alt Text

Diagram of 4 test data channels (synthetic DB, demo subjects, real DB, in-memory fixtures) separated by an isolation boundary with enforcement mechanisms.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
