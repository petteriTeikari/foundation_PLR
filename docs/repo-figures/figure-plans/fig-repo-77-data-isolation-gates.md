# fig-repo-77: Synthetic vs Production: The Isolation Boundary

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-77 |
| **Title** | Synthetic vs Production: The Isolation Boundary |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `ARCHITECTURE.md`, `data/README.md` |
| **Priority** | P2 (High) |

## Purpose

Show the strict boundary between synthetic data (used for testing) and production data (from MLflow). Developers need to understand that cross-contamination triggers test failures, and that four independent enforcement gates ensure synthetic data never appears in production artifacts or generated figures.

## Key Message

Synthetic data (for testing) and production data (from MLflow) are isolated by directory structure, config, pre-commit hooks, and integration tests. Cross-contamination triggers immediate failures.

## Content Specification

### Panel 1: Two Isolation Zones

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA ISOLATION ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─── PRODUCTION ZONE ──────────────┐   ┌─── SYNTHETIC ZONE ─────────┐ │
│  │                                    │   │                            │ │
│  │  data/public/                     │   │  data/synthetic/            │ │
│  │  ├── foundation_plr_results.db    │   │  └── SYNTH_PLR_DEMO.db     │ │
│  │  └── cd_diagram_data.duckdb      │   │     (8 demo subjects)      │ │
│  │                                    │   │                            │ │
│  │  /home/petteri/mlruns/            │   │  src/synthetic/             │ │
│  │  └── 410 real MLflow runs         │   │  └── Data generators       │ │
│  │     (542 pickles, 1000 bootstrap) │   │     (isolated module)      │ │
│  │                                    │   │                            │ │
│  │  figures/generated/               │   │  tests/                     │ │
│  │  ├── *.png (from real data)       │   │  ├── Unit tests use        │ │
│  │  └── *.json (figure sidecars)     │   │  │   synthetic fixtures    │ │
│  │                                    │   │  └── Skip if real DB       │ │
│  │  data/r_data/                     │   │      absent (graceful)     │ │
│  │  └── R-exported CSV files         │   │                            │ │
│  │                                    │   │                            │ │
│  └────────────────────────────────────┘   └────────────────────────────┘ │
│                                                                          │
│             ╳ ─ ─ ─ ─ ─ ─ ISOLATION BOUNDARY ─ ─ ─ ─ ─ ─ ╳            │
│                  Cross-contamination = IMMEDIATE FAILURE                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: The 4 Enforcement Gates

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    4 ENFORCEMENT GATES                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GATE 1: EXTRACTION ISOLATION (pre-commit)                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Hook: extraction-isolation-check                                  │  │
│  │  Script: scripts/check_extraction_isolation.py                    │  │
│  │  Triggers on: scripts/extract*.py, src/utils/data_mode.py,       │  │
│  │               configs/data_isolation.yaml                         │  │
│  │  Checks: Extraction scripts do not reference synthetic paths      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  GATE 2: FIGURE ISOLATION (pre-commit)                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Hook: figure-isolation-check                                     │  │
│  │  Script: scripts/check_figure_isolation.py                        │  │
│  │  Triggers on: src/viz/, figures/, configs/data_isolation.yaml     │  │
│  │  Checks: Generated figures reference only production data         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  GATE 3: CONFIG-DRIVEN ISOLATION                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  File: configs/data_isolation.yaml                                │  │
│  │  Defines: production paths, synthetic paths, allowed boundaries  │  │
│  │  Read by: extraction scripts, isolation check scripts            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  GATE 4: INTEGRATION TESTS (pytest)                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  tests/integration/test_extraction_isolation.py                   │  │
│  │  tests/integration/test_figure_routing.py                        │  │
│  │  tests/integration/test_mlflow_isolation.py                      │  │
│  │  tests/e2e/test_synthetic_isolation_e2e.py                       │  │
│  │  Checks: End-to-end verification that no synthetic data leaks    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: What Cross-Contamination Looks Like

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CROSS-CONTAMINATION SCENARIOS                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SCENARIO A: Synthetic DB in production path                            │
│  data/public/SYNTH_PLR_DEMO.db  ← WRONG: synthetic in production      │
│  Caught by: extraction-isolation-check (Gate 1)                         │
│                                                                          │
│  SCENARIO B: Figure generated from synthetic data                       │
│  figures/generated/fig_R7_calibration.png sourcing SYNTH_PLR_DEMO.db   │
│  Caught by: figure-isolation-check (Gate 2), test_data_provenance.py   │
│                                                                          │
│  SCENARIO C: Extraction script using synthetic path                     │
│  scripts/extract_all_configs_to_duckdb.py → data/synthetic/            │
│  Caught by: extraction-isolation-check (Gate 1)                         │
│                                                                          │
│  SCENARIO D: Test using production data without skipif guard            │
│  Direct import of foundation_plr_results.db without path check         │
│  Caught by: test_data_location_policy.py (Gate 4)                      │
│                                                                          │
│  HISTORICAL: CRITICAL-FAILURE-001 (Synthetic Data in Figures)           │
│  → Claude generated calibration plots with SYNTHETIC data               │
│  → Prompted creation of the 4-gate isolation architecture               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/data_isolation.yaml` | Defines production vs synthetic path boundaries |
| `.pre-commit-config.yaml` | Hook definitions: extraction-isolation-check, figure-isolation-check |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/check_extraction_isolation.py` | Gate 1: Pre-commit check for extraction scripts |
| `scripts/check_figure_isolation.py` | Gate 2: Pre-commit check for figure outputs |
| `tests/integration/test_extraction_isolation.py` | Gate 4: Integration test for extraction isolation |
| `tests/integration/test_figure_routing.py` | Gate 4: Integration test for figure routing |
| `tests/integration/test_mlflow_isolation.py` | Gate 4: Integration test for MLflow data isolation |
| `tests/e2e/test_synthetic_isolation_e2e.py` | Gate 4: End-to-end synthetic isolation test |
| `tests/test_guardrails/test_data_location_policy.py` | Gate 4: Data location policy enforcement |
| `src/utils/data_mode.py` | Utility for determining production vs synthetic mode |

## Extension Guide

To add a new data path that must be isolation-aware:
1. Add the path to `configs/data_isolation.yaml` under the appropriate zone (production/synthetic)
2. Update `scripts/check_extraction_isolation.py` if the path is used in extraction
3. Update `scripts/check_figure_isolation.py` if the path is used in figure generation
4. Add a test in `tests/integration/` verifying the isolation boundary
5. Ensure `conftest.py` fixtures use `pytest.mark.skipif(not path.exists())` for optional paths

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-77",
    "title": "Synthetic vs Production: The Isolation Boundary"
  },
  "content_architecture": {
    "primary_message": "Synthetic data and production data are isolated by directory, config, pre-commit hooks, and integration tests. Cross-contamination triggers immediate failures.",
    "layout_flow": "Two zones (production/synthetic) at top, 4 gates below, contamination scenarios at bottom",
    "spatial_anchors": {
      "production_zone": {"x": 0.3, "y": 0.2},
      "synthetic_zone": {"x": 0.7, "y": 0.2},
      "isolation_boundary": {"x": 0.5, "y": 0.35},
      "gates": {"x": 0.5, "y": 0.65},
      "scenarios": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Production Zone",
        "role": "healthy_normal",
        "is_highlighted": true,
        "labels": ["data/public/", "figures/generated/"]
      },
      {
        "name": "Synthetic Zone",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["data/synthetic/", "tests/"]
      },
      {
        "name": "Gate 1: Extraction Isolation",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Pre-commit hook"]
      },
      {
        "name": "Gate 2: Figure Isolation",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["Pre-commit hook"]
      },
      {
        "name": "Gate 3: Config",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["data_isolation.yaml"]
      },
      {
        "name": "Gate 4: Integration Tests",
        "role": "abnormal_warning",
        "is_highlighted": false,
        "labels": ["pytest integration"]
      }
    ],
    "callout_boxes": [
      {"heading": "CRITICAL-FAILURE-001", "body_text": "This architecture was created after synthetic data appeared in publication figures."}
    ]
  }
}
```

## Alt Text

Two-zone diagram showing production and synthetic data directories separated by an isolation boundary with four enforcement gates.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
