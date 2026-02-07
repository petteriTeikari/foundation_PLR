# fig-repo-78: How Tests Find Their Data: Path Resolution Chain

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-78 |
| **Title** | How Tests Find Their Data: Path Resolution Chain |
| **Complexity Level** | L4 |
| **Target Persona** | ML Engineer |
| **Location** | `tests/README.md`, `CONTRIBUTING.md` |
| **Priority** | P2 (High) |

## Purpose

Show the complete path resolution chain from `Path(__file__)` to a running test fixture. Developers need to understand how tests find databases, how missing files trigger graceful skips instead of crashes, and how fixture scoping controls resource sharing.

## Key Message

Test data paths resolve through a chain: PROJECT_ROOT to canonical path constants to conftest.py fixtures to skipif decorators. Missing files cause graceful skips, not crashes.

## Content Specification

### Panel 1: The Resolution Chain

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PATH RESOLUTION CHAIN                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Step 1: Compute PROJECT_ROOT                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  tests/conftest.py (line 20):                                      │  │
│  │  PROJECT_ROOT = Path(__file__).parent.parent                      │  │
│  │  → /home/.../foundation_PLR/                                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  Step 2: Define Canonical Path Constants                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  tests/conftest.py (lines 21-25):                                  │  │
│  │                                                                    │  │
│  │  RESULTS_DB   = PROJECT_ROOT / "data/public/                      │  │
│  │                   foundation_plr_results.db"                       │  │
│  │  CD_DIAGRAM_DB = PROJECT_ROOT / "data/public/                     │  │
│  │                   cd_diagram_data.duckdb"                          │  │
│  │  R_DATA_DIR   = PROJECT_ROOT / "data/r_data"                     │  │
│  │  FIGURES_DIR  = PROJECT_ROOT / "figures/generated"                │  │
│  │  DEMO_DB      = PROJECT_ROOT / "data/synthetic/                   │  │
│  │                   SYNTH_PLR_DEMO.db"                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  Step 3: Create Fixtures with Scope                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  @pytest.fixture                                                   │  │
│  │  def project_root() -> Path:                                      │  │
│  │      return PROJECT_ROOT                                          │  │
│  │                                                                    │  │
│  │  @pytest.fixture                                                   │  │
│  │  def demo_data_path(project_root) -> Path:                        │  │
│  │      return DEMO_DB                                               │  │
│  │                                                                    │  │
│  │  @pytest.fixture                                                   │  │
│  │  def demo_data_available(demo_data_path) -> bool:                 │  │
│  │      return demo_data_path.exists()                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                                                      │
│                    ▼                                                      │
│  Step 4: Skip If Not Available                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  @pytest.mark.skipif(                                              │  │
│  │      not RESULTS_DB.exists(),                                     │  │
│  │      reason="Production DB not available"                         │  │
│  │  )                                                                 │  │
│  │  def test_essential_metrics():                                    │  │
│  │      conn = duckdb.connect(str(RESULTS_DB))                       │  │
│  │      ...                                                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Fixture Hierarchy Across conftest.py Files

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFTEST.PY FIXTURE HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  tests/conftest.py (ROOT)                                               │
│  ├── Scope: module/function (default)                                   │
│  ├── project_root, demo_data_path, demo_data_available                 │
│  ├── minimal_cfg (OmegaConf mock configuration)                        │
│  └── Used by ALL test files                                             │
│  │                                                                       │
│  ├── tests/test_figure_qa/conftest.py                                   │
│  │   ├── Scope: session (shared across all QA tests)                    │
│  │   ├── figure_dir → FIGURES_DIR                                       │
│  │   ├── json_files → scans figures/generated/*.json                    │
│  │   └── png_files → scans figures/generated/*.png                      │
│  │                                                                       │
│  ├── tests/test_figure_generation/conftest.py                           │
│  │   ├── Scope: function (fresh per test)                               │
│  │   ├── db_connection → DuckDB connection                              │
│  │   └── Mock data for figure unit tests                                │
│  │                                                                       │
│  ├── tests/test_no_hardcoding/conftest.py                               │
│  │   └── Shared paths for hardcoding detection tests                    │
│  │                                                                       │
│  └── tests/test_guardrails/conftest.py                                  │
│      └── Guardrail-specific fixtures                                    │
│                                                                          │
│  SCOPE HIERARCHY:                                                       │
│  session > module > class > function                                    │
│  └── session = one DB connection for all tests in the session           │
│  └── function = fresh fixture per test function (isolation)             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 3: Graceful Skip Behavior

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHAT HAPPENS WHEN A FILE DOESN'T EXIST                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RESULTS_DB exists?                                                     │
│  ├── YES → 316-row metrics table available → integration tests run     │
│  └── NO  → pytest.skip("Production DB not available")                  │
│            → Test counted as SKIPPED (not FAILED)                       │
│            → This is normal for fresh clones                            │
│                                                                          │
│  R_DATA_DIR exists?                                                     │
│  ├── YES → R-exported CSVs available → figure QA tests run            │
│  └── NO  → 34 tests skip (Group B skip group)                         │
│            → Fix: make r-figures                                        │
│                                                                          │
│  FIGURES_DIR has PNGs?                                                  │
│  ├── YES → Figure QA tests run → check provenance, rendering          │
│  └── NO  → 24 tests skip (Group D skip group)                         │
│            → Fix: make analyze                                          │
│                                                                          │
│  DEMO_DB exists?                                                        │
│  ├── YES → Synthetic demo data available → unit tests use it          │
│  └── NO  → 11 tests skip (Group E skip group)                         │
│            → Fix: make analyze (creates demo data)                     │
│                                                                          │
│  RESOLUTION PATH:                                                       │
│  make extract → make analyze → make r-figures → 0 skips               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `tests/conftest.py` | Root: PROJECT_ROOT, canonical path constants, base fixtures |
| `tests/test_figure_qa/conftest.py` | Figure QA: session-scoped figure directory scanning |
| `tests/test_figure_generation/conftest.py` | Figure gen: function-scoped DB connections |
| `tests/test_no_hardcoding/conftest.py` | Hardcoding: shared paths for detection tests |
| `tests/test_guardrails/conftest.py` | Guardrails: guardrail-specific fixtures |

## Code Paths

| Module | Role |
|--------|------|
| `tests/conftest.py` | Root conftest: PROJECT_ROOT, RESULTS_DB, CD_DIAGRAM_DB, R_DATA_DIR, FIGURES_DIR, DEMO_DB |
| `tests/test_figure_qa/conftest.py` | Figure QA fixtures: figure_dir, json_files, png_files |
| `tests/test_figure_generation/conftest.py` | Figure generation fixtures: db_connection, mock data |

## Extension Guide

To add a new canonical data path for tests:
1. Define the constant in `tests/conftest.py` (e.g., `NEW_PATH = PROJECT_ROOT / "data/new/"`)
2. Create a fixture with appropriate scope (session for shared resources, function for isolated)
3. Add `@pytest.mark.skipif(not NEW_PATH.exists(), reason="...")` to tests that need it
4. Document which skip group the new path belongs to (A-H)
5. Add the resolution step to the make pipeline

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-78",
    "title": "How Tests Find Their Data: Path Resolution Chain"
  },
  "content_architecture": {
    "primary_message": "Test data paths resolve through a chain: PROJECT_ROOT to canonical constants to fixtures to skipif decorators. Missing files cause graceful skips.",
    "layout_flow": "Top-down: PROJECT_ROOT at top, through constants, fixtures, skipif, to test execution at bottom",
    "spatial_anchors": {
      "project_root": {"x": 0.5, "y": 0.1},
      "constants": {"x": 0.5, "y": 0.3},
      "fixtures": {"x": 0.5, "y": 0.5},
      "skipif": {"x": 0.5, "y": 0.7},
      "skip_behavior": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "PROJECT_ROOT",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Path(__file__).parent.parent"]
      },
      {
        "name": "Canonical Constants",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["RESULTS_DB", "CD_DIAGRAM_DB", "DEMO_DB"]
      },
      {
        "name": "conftest.py Fixtures",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["session/function scope"]
      },
      {
        "name": "skipif Decorators",
        "role": "healthy_normal",
        "is_highlighted": true,
        "labels": ["Graceful skip"]
      }
    ],
    "callout_boxes": [
      {"heading": "GRACEFUL DEGRADATION", "body_text": "Missing files trigger pytest.skip(), not crashes. Fresh clones see 181 skips, resolved by make extract + analyze."}
    ]
  }
}
```

## Alt Text

Flowchart showing test path resolution from PROJECT_ROOT through canonical constants and conftest fixtures to skipif decorators for graceful degradation.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
