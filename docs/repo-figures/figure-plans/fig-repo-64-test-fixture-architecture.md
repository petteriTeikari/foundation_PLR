# fig-repo-64: conftest.py Hierarchy: Fixtures from Root to Leaf

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-64 |
| **Title** | conftest.py Hierarchy: Fixtures from Root to Leaf |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `tests/README.md`, `docs/explanation/test-architecture.md` |
| **Priority** | P2 (High) |

## Purpose

Document the tree hierarchy of conftest.py files, showing how fixtures are scoped (session, module, function) and inherited. Developers need to know where a fixture is defined, what scope it uses, and which test directories can access it.

## Key Message

Test fixtures follow a tree hierarchy from root conftest.py (session-scoped DB connections and path constants) down to per-directory conftest.py files (module-scoped mocks and auto-applied markers). Fixture scope determines lifetime and sharing.

## Content Specification

### Panel 1: conftest.py Tree Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│           conftest.py HIERARCHY: ROOT TO LEAF                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  tests/conftest.py  (ROOT — all tests inherit these)                   │
│  ├── PATH CONSTANTS (module-level, not fixtures):                      │
│  │     PROJECT_ROOT = Path(__file__).parent.parent                     │
│  │     RESULTS_DB   = PROJECT_ROOT / "data/public/foundation_plr_...db"│
│  │     CD_DIAGRAM_DB = PROJECT_ROOT / "data/public/cd_diagram_...db"   │
│  │     R_DATA_DIR   = PROJECT_ROOT / "data/r_data"                     │
│  │     FIGURES_DIR  = PROJECT_ROOT / "figures/generated"               │
│  │     DEMO_DB      = PROJECT_ROOT / "data/synthetic/SYNTH_PLR_..."    │
│  │                                                                     │
│  ├── FIXTURES (function-scoped unless noted):                          │
│  │     project_root()         → Path    (function)                     │
│  │     demo_data_path()       → Path    (function, uses DEMO_DB)       │
│  │     demo_data_available()  → bool    (function)                     │
│  │     minimal_cfg()          → OmegaConf  (function, mock config)     │
│  │     outlier_detection_cfg()→ OmegaConf  (function)                  │
│  │     imputation_cfg()       → OmegaConf  (function)                  │
│  │     sample_plr_array()     → np.ndarray (function, 8x1981)          │
│  │     sample_plr_array_3d()  → np.ndarray (function, 8x1981x1)       │
│  │     sample_outlier_mask()  → np.ndarray (function)                  │
│  │     mock_mlflow()          → dict       (function, patched MLflow)  │
│  │     force_cpu()            → None       (function, patches torch)   │
│  │     temp_artifacts_dir()   → Path       (function, tempdir)         │
│  │     temp_duckdb_path()     → Path       (function)                  │
│  │     skip_if_no_demo_data() → None       (function, conditional)     │
│  │                                                                     │
│  ├── tests/unit/conftest.py                                            │
│  │     pytest_collection_modifyitems() → auto-applies @mark.unit       │
│  │                                                                     │
│  ├── tests/integration/conftest.py                                     │
│  │     pytest_collection_modifyitems() → auto-applies @mark.integration│
│  │                                                                     │
│  ├── tests/e2e/conftest.py                                             │
│  │     pytest_collection_modifyitems() → auto-applies @mark.e2e        │
│  │                                                                     │
│  ├── tests/test_guardrails/conftest.py                                 │
│  │     PROJECT_ROOT (re-defined)                                       │
│  │     pytest_collection_modifyitems() → auto-applies @mark.guardrail  │
│  │                                                                     │
│  ├── tests/test_no_hardcoding/conftest.py                              │
│  │     pytest_collection_modifyitems() → auto-applies @mark.guardrail  │
│  │                                                                     │
│  ├── tests/test_figure_qa/conftest.py                                  │
│  │     PROJECT_ROOT, R_DATA_DIR, FIGURES_DIR, GOLDEN_DIR (re-defined)  │
│  │     calibration_json_path()  → Path (function, skip if missing)     │
│  │     calibration_data()       → dict (function, loads JSON)          │
│  │     dca_json_path()          → Path (function, skip if missing)     │
│  │     dca_data()               → dict (function, loads JSON)          │
│  │     predictions_json_path()  → Path (function, skip if missing)     │
│  │     predictions_data()       → dict (function, loads JSON)          │
│  │     all_json_files()         → list (function, globs r_data/*.json) │
│  │     all_figure_files()       → list (function, globs *.pdf + *.png) │
│  │     ggplot2_figures()        → list (function, globs ggplot2/*)     │
│  │     golden_dir()             → Path (function, creates if needed)   │
│  │     pytest_collection_modifyitems() → auto-applies @mark.data       │
│  │                                                                     │
│  └── tests/test_figure_generation/conftest.py                          │
│        pytest_collection_modifyitems() → auto-applies @mark.unit       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Fixture Scope Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  FIXTURE SCOPE: LIFETIME AND SHARING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SESSION SCOPE (once per test run)                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Currently: None explicitly session-scoped             │       │
│  │ Path constants (RESULTS_DB, etc.) serve this role     │       │
│  │ as module-level variables (effectively session-scoped)│       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  MODULE SCOPE (once per test file)                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Not currently used (available for future optimization)│       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  FUNCTION SCOPE (once per test function — DEFAULT)               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ ALL current fixtures: sample_plr_array, minimal_cfg,  │       │
│  │ mock_mlflow, temp_artifacts_dir, calibration_data,    │       │
│  │ all_json_files, etc.                                  │       │
│  │                                                        │       │
│  │ Advantage: complete isolation between tests            │       │
│  │ Tradeoff: re-creates data for every test function      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  INHERITANCE RULE:                                               │
│  Root conftest.py fixtures are available to ALL test files.      │
│  Per-directory conftest.py fixtures are available ONLY to tests  │
│  in that directory and its subdirectories.                        │
└─────────────────────────────────────────────────────────────────┘
```

### Panel 3: Key Path Constants

```
CONSTANT         PATH                                         USED BY
─────────────── ──────────────────────────────────────────── ─────────────
PROJECT_ROOT     Path(__file__).parent.parent                 All tests
RESULTS_DB       data/public/foundation_plr_results.db        Group A skips
CD_DIAGRAM_DB    data/public/cd_diagram_data.duckdb           Group A skips
R_DATA_DIR       data/r_data                                  Group B skips
FIGURES_DIR      figures/generated                             Group D skips
DEMO_DB          data/synthetic/SYNTH_PLR_DEMO.db             Group E skips
GOLDEN_DIR       tests/golden_images                           Visual regression
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: tree hierarchy → scope diagram → path constants"
spatial_anchors:
  tree_hierarchy:
    x: 0.5
    y: 0.3
    content: "Root-to-leaf conftest.py tree with fixtures at each level"
  scope_diagram:
    x: 0.5
    y: 0.65
    content: "Session/module/function scope explanation"
  path_constants:
    x: 0.5
    y: 0.9
    content: "Canonical path constants referenced by all tests"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `tests/conftest.py` | Root fixtures: path constants, data arrays, mocks, configs |
| `tests/test_figure_qa/conftest.py` | Figure QA fixtures: JSON loaders, figure globbers |
| `tests/*/conftest.py` | Per-directory marker auto-application |
| `configs/demo_subjects.yaml` | 8 demo subjects used by demo_data fixtures |

## Code Paths

| Module | Role |
|--------|------|
| `tests/conftest.py` | Root conftest (path constants, 15+ fixtures, skip logic) |
| `tests/test_figure_qa/conftest.py` | Figure QA fixtures (JSON data, figure files, golden dir) |
| `tests/test_guardrails/conftest.py` | Guardrail marker + PROJECT_ROOT override |
| `tests/unit/conftest.py` | Unit marker auto-application |
| `tests/integration/conftest.py` | Integration marker auto-application |
| `tests/e2e/conftest.py` | E2E marker auto-application |
| `tests/test_figure_generation/conftest.py` | Unit marker for figure generation tests |
| `tests/test_no_hardcoding/conftest.py` | Guardrail marker for hardcoding checks |

## Extension Guide

To add a new fixture:
1. Decide the scope: function (default, best isolation), module (shared within file), session (shared across run)
2. Decide the level: root `tests/conftest.py` (all tests) vs per-directory conftest.py (specific tests)
3. If the fixture depends on a file, add a `pytest.skip()` guard for graceful degradation
4. If it is a path constant, add it to the canonical section at the top of `tests/conftest.py`
5. Follow the naming convention: `sample_*` for synthetic data, `*_path` for file paths, `*_cfg` for configs

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-64",
    "title": "conftest.py Hierarchy: Fixtures from Root to Leaf"
  },
  "content_architecture": {
    "primary_message": "Test fixtures follow a tree hierarchy from root conftest.py (session-scoped DB paths) down to per-directory conftest.py files (auto-applied markers and specialized fixtures).",
    "layout_flow": "Top-down: tree hierarchy, scope diagram, path constant table",
    "spatial_anchors": {
      "tree_hierarchy": {"x": 0.5, "y": 0.3},
      "scope_diagram": {"x": 0.5, "y": 0.65},
      "path_constants": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Root conftest.py",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["PROJECT_ROOT", "RESULTS_DB", "R_DATA_DIR", "FIGURES_DIR", "DEMO_DB"]
      },
      {
        "name": "Per-directory conftest.py",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["unit/", "integration/", "e2e/", "test_figure_qa/", "test_guardrails/"]
      },
      {
        "name": "Fixture Scope",
        "role": "callout_box",
        "is_highlighted": true,
        "labels": ["session (path constants)", "function (all fixtures)", "inheritance rule"]
      }
    ],
    "callout_boxes": [
      {"heading": "INHERITANCE RULE", "body_text": "Root fixtures are available to ALL tests. Per-directory fixtures are scoped to that directory and its children."}
    ]
  }
}
```

## Alt Text

Tree diagram of conftest.py hierarchy from root to leaf directories, showing fixture definitions, scopes, and path constants at each level.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
