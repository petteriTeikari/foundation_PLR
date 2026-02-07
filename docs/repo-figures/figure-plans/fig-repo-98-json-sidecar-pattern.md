# fig-repo-98: JSON Sidecars: Every Figure's Reproducibility Passport

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-98 |
| **Title** | JSON Sidecars: Every Figure's Reproducibility Passport |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `figures/README.md`, `docs/explanation/figure-reproducibility.md` |
| **Priority** | P2 (High) |

## Purpose

Document the JSON sidecar pattern that ensures every generated figure is reproducible. Each PNG has a companion JSON file containing all numeric data, source database hash, combo identifiers, and generation metadata. This enables figure reproduction without re-running the pipeline, reviewer verification of data provenance, and detection of stale figures.

## Key Message

Every figure has a companion JSON sidecar that serves as its "reproducibility passport": it contains all data needed to recreate the figure, traces provenance to the source database, and is tested automatically by the figure QA suite.

## Content Specification

### Panel 1: The Sidecar Pattern

```
figures/generated/
|
+-- fig_R7_calibration.png              <-- The visible figure
|
+-- fig_R7_calibration.json             <-- The sidecar (PUBLIC, committed)
|   |
|   +-- Contains ALL numeric data needed to recreate the figure
|   +-- Traces provenance to source database
|   +-- Committed to git (aggregate data, no patient info)
|
+-- data/
    +-- fig_R7_calibration_TEST.json    <-- Subject-level (PRIVATE, gitignored)
        |
        +-- Contains per-patient predictions
        +-- NEVER committed (patient privacy)
        +-- Generated locally by make analyze
```

### Panel 2: JSON Sidecar Structure

```
{
  "figure_id": "R7",
  "generated_at": "2026-02-06T12:00:00",

  // PROVENANCE: Where did this data come from?
  "source_db": "data/public/foundation_plr_results.db",
  "source_hash": "sha256:abc123def456...",      <-- DB file hash at generation time

  // WHAT: Which preprocessing combos are shown?
  "combos_used": [
    "ground_truth",
    "best_ensemble",
    "best_single_fm",
    "traditional"
  ],

  // DATA: All numeric values in the figure
  "data": {
    "ground_truth": {
      "auroc": ...,
      "calibration_slope": ...,
      "calibration_intercept": ...,
      "curve_points": [[x1, y1], [x2, y2], ...]
    },
    "best_ensemble": {
      ...
    }
  },

  // SUMMARY: Aggregate statistics
  "summary_statistics": {
    "n_subjects": 208,
    "n_bootstrap": 1000,
    "ci_level": 0.95
  }
}
```

### Panel 3: How Sidecars Are Created

```
# In any figure script (src/viz/*.py):

from src.viz.plot_config import setup_style, save_figure, COLORS

setup_style()

# ... create figure using DuckDB data ...

# save_figure() automatically creates the JSON sidecar
save_figure(
    fig,                           # matplotlib Figure object
    "fig_R7_calibration",          # base name (no extension)
    data={                         # dict --> written as .json alongside .png
        "figure_id": "R7",
        "source_db": str(db_path),
        "combos_used": combo_ids,
        "data": per_combo_data,
        "summary_statistics": summary
    }
)

# Result:
#   figures/generated/fig_R7_calibration.png   (created)
#   figures/generated/fig_R7_calibration.json  (created from data dict)
```

### Panel 4: save_figure() Function Signature

```
save_figure(
    fig: plt.Figure,
    name: str,
    data: Optional[Dict[str, Any]] = None,    # --> JSON sidecar
    formats: List[str] = None,                 # Default: from figure_layouts.yaml
    output_dir: Optional[Path] = None,         # Default: figures/generated/
    synthetic: Optional[bool] = None,          # Auto-detect from data_mode
) -> Path

Key behaviors:
  - If data= is provided, writes {name}.json alongside {name}.png
  - If synthetic=True (or auto-detected), routes to figures/synthetic/
  - Formats loaded from configs/VISUALIZATION/figure_layouts.yaml
  - Returns Path to the primary output file (PNG)
```

### Panel 5: Privacy Levels

```
+----------------------------------------------------------------------+
|  PRIVACY MODEL                                                         |
|                                                                        |
|  PUBLIC (committed to git):                                            |
|  +-------------------------------+                                     |
|  | Aggregate JSON sidecars       |  figures/generated/*.json           |
|  | Summary statistics            |  n_subjects=208, n_bootstrap=1000  |
|  | Per-combo metrics             |  auroc, calibration_slope, etc.    |
|  | Curve points (aggregated)     |  calibration curve bins            |
|  +-------------------------------+                                     |
|                                                                        |
|  PRIVATE (gitignored):                                                 |
|  +-------------------------------+                                     |
|  | Subject-level predictions     |  data/*_TEST.json                  |
|  | Per-patient y_true, y_prob    |  Individual risk scores            |
|  | Subject lookup table          |  data/private/subject_lookup.yaml  |
|  | Original PLR subject codes    |  PLRxxxx identifiers              |
|  +-------------------------------+                                     |
|                                                                        |
|  Controlled by:                                                        |
|  - figure_registry.yaml: json_privacy field (public / private)        |
|  - .gitignore: data/private/*, *_TEST.json patterns                   |
|  - configs/demo_subjects.yaml: H001-H004 (anonymized codes)          |
+----------------------------------------------------------------------+
```

### Panel 6: QA Testing

```
tests/test_figure_qa/test_data_provenance.py
|
+-- test_every_png_has_json()
|     For every .png in figures/generated/:
|       assert corresponding .json exists
|       "Figure without sidecar = unreproducible"
|
+-- test_json_has_required_fields()
|     For every .json sidecar:
|       assert "figure_id" in data
|       assert "source_db" in data
|       assert "data" in data
|       "Sidecar without provenance = useless"
|
+-- test_source_hash_matches_db()
|     If source_db path exists:
|       compute sha256 of current DB
|       compare to source_hash in JSON
|       WARN if mismatch (figure may be stale)
|
+-- test_no_synthetic_in_production()
      assert "_synthetic_warning" not in data
      "Synthetic data must not appear in production figures"
```

### Panel 7: Reproducibility Chain

```
FORWARD (generation):
  DuckDB --> src/viz/figure_script.py --> save_figure(data=...) --> PNG + JSON

BACKWARD (verification):
  JSON sidecar --> source_hash --> matches DuckDB? --> data matches figure?

REPRODUCTION (from JSON alone):
  JSON sidecar --> extract data dict --> re-plot without DuckDB or MLflow
  (Useful for: reviewers, collaborators, future re-analysis)

STALENESS DETECTION:
  JSON source_hash != current DB sha256 --> figure is STALE
  Re-run: python src/viz/generate_all_figures.py --figure R7
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: file pattern, JSON structure, creation flow, privacy, QA, chain"
spatial_anchors:
  file_pattern:
    x: 0.5
    y: 0.1
    content: "PNG + JSON + private JSON file layout"
  json_structure:
    x: 0.5
    y: 0.28
    content: "JSON sidecar fields and structure"
  creation:
    x: 0.5
    y: 0.45
    content: "save_figure() creates sidecar automatically"
  privacy:
    x: 0.5
    y: 0.6
    content: "Public (committed) vs Private (gitignored)"
  qa:
    x: 0.5
    y: 0.78
    content: "test_data_provenance.py verifies every sidecar"
  chain:
    x: 0.5
    y: 0.92
    content: "Forward generation, backward verification, reproduction"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `configs/VISUALIZATION/figure_registry.yaml` | `json_privacy` field per figure (public/private) |
| `configs/VISUALIZATION/figure_layouts.yaml` | `output_settings.formats` (png, svg) |
| `.gitignore` | Patterns for private JSON (`*_TEST.json`, `data/private/*`) |
| `configs/demo_subjects.yaml` | Anonymized subject codes (H001-H004, G001-G004) |

## Code Paths

| Module | Role |
|--------|------|
| `src/viz/plot_config.py` | `save_figure()` function that creates PNG + JSON sidecar |
| `src/utils/data_mode.py` | `is_synthetic_mode()`, `get_figures_dir_for_mode()` (routing logic) |
| `tests/test_figure_qa/test_data_provenance.py` | QA tests: every PNG has JSON, required fields present |
| `tests/test_figure_qa/conftest.py` | Fixtures: `figure_dir`, `json_files`, `png_files` |
| `src/viz/generate_all_figures.py` | Entry point: `--figure R7` generates specific figure with sidecar |

## Extension Guide

To add a JSON sidecar to a new figure:
1. In your figure script, collect all plotted data into a `dict`
2. Include provenance: `figure_id`, `source_db`, `source_hash`
3. Include data: per-combo metrics, curve points, summary statistics
4. Pass the dict to `save_figure(fig, "name", data=your_dict)`
5. Set `json_privacy` in `figure_registry.yaml` (public for aggregate, private for subject-level)
6. Run `pytest tests/test_figure_qa/test_data_provenance.py -v` to verify

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-98",
    "title": "JSON Sidecars: Every Figure's Reproducibility Passport"
  },
  "content_architecture": {
    "primary_message": "Every figure has a companion JSON sidecar containing all data, provenance, and metadata needed for reproduction and verification.",
    "layout_flow": "Top-down: file pattern, JSON structure, creation, privacy, QA, reproducibility chain",
    "spatial_anchors": {
      "file_pattern": {"x": 0.5, "y": 0.1},
      "json_structure": {"x": 0.5, "y": 0.28},
      "creation": {"x": 0.5, "y": 0.45},
      "privacy": {"x": 0.5, "y": 0.6},
      "qa": {"x": 0.5, "y": 0.78},
      "chain": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {
        "name": "Sidecar File Pattern",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["fig_*.png", "fig_*.json (PUBLIC)", "data/*_TEST.json (PRIVATE)"]
      },
      {
        "name": "JSON Structure",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["figure_id", "source_hash", "combos_used", "data", "summary_statistics"]
      },
      {
        "name": "save_figure() Function",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["save_figure(fig, name, data=dict)", "Auto-creates JSON sidecar"]
      },
      {
        "name": "Privacy Model",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["PUBLIC: aggregate metrics", "PRIVATE: subject-level (gitignored)"]
      },
      {
        "name": "QA Tests",
        "role": "healthy_normal",
        "is_highlighted": false,
        "labels": ["test_every_png_has_json", "test_required_fields", "test_no_synthetic"]
      }
    ],
    "callout_boxes": [
      {"heading": "KEY PRINCIPLE", "body_text": "Every PNG must have a JSON sidecar. A figure without provenance is unreproducible."},
      {"heading": "PRIVACY", "body_text": "Aggregate JSON = PUBLIC (committed). Subject-level JSON = PRIVATE (gitignored for patient privacy)."}
    ]
  }
}
```

## Alt Text

Diagram of the JSON sidecar pattern: each figure PNG has a companion JSON with provenance, data, and metadata. Public sidecars committed; private subject-level data gitignored.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
