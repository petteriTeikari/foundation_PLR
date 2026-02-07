# fig-repo-61: Test Skip Groups A-H: What's Skipped and Why

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-61 |
| **Title** | Test Skip Groups A-H: What's Skipped and Why |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `tests/README.md`, `docs/explanation/test-architecture.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show developers why 181 tests skip on a fresh clone and how to systematically resolve each group. Every skip has a known root cause and a documented fix command. This eliminates the "why are so many tests skipping?" confusion during onboarding.

## Key Message

181 test skips are organized into 7 groups (A-H), each with a distinct root cause and a single fix command. Running `make extract`, `make analyze`, and `make r-figures-all` in sequence resolves all skips to zero.

## Content Specification

### Panel 1: Skip Group Diagnostic Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│           TEST SKIP GROUPS: A DIAGNOSTIC MAP (~181 total)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GROUP A: DB Paths (~43 skips)         GROUP B: R_DATA_DIR (~34 skips) │
│  ┌───────────────────────────┐         ┌───────────────────────────┐   │
│  │ Root: Production DuckDB    │         │ Root: R output data       │   │
│  │ not at expected path       │         │ not yet generated         │   │
│  │                            │         │                           │   │
│  │ Files affected:            │         │ Files affected:           │   │
│  │  test_artifact_consistency │         │  test_figure_qa/*         │   │
│  │  test_extraction_registry  │         │  test_data_provenance     │   │
│  │  test_full_pipeline        │         │  test_no_nan_ci           │   │
│  │  test_decomposition/*      │         │  test_publication_stds    │   │
│  │  test_extraction_*         │         │  test_rendering_artifacts │   │
│  │                            │         │                           │   │
│  │ Skip condition:            │         │ Skip condition:           │   │
│  │  RESULTS_DB.exists() = F   │         │  R_DATA_DIR.exists() = F  │   │
│  │                            │         │                           │   │
│  │ FIX: make extract          │         │ FIX: make r-figures-all   │   │
│  └───────────────────────────┘         └───────────────────────────┘   │
│                                                                         │
│  GROUP D: Figure Filenames (~24)       GROUP E: Demo Data (~11)        │
│  ┌───────────────────────────┐         ┌───────────────────────────┐   │
│  │ Root: Generated figure     │         │ Root: Demo subjects not   │   │
│  │ PNGs not yet created       │         │ yet created from pipeline │   │
│  │                            │         │                           │   │
│  │ Skip condition:            │         │ Skip condition:           │   │
│  │  FIGURES_DIR/*.png empty   │         │  DEMO_DB.exists() = F     │   │
│  │                            │         │                           │   │
│  │ FIX: make analyze          │         │ FIX: make analyze         │   │
│  └───────────────────────────┘         └───────────────────────────┘   │
│                                                                         │
│  GROUP F: Manuscript (~10)    GROUP G: TDD Stubs (~6)  GROUP H: (~9)  │
│  ┌───────────────────┐        ┌───────────────────┐    ┌────────────┐  │
│  │ Root: LaTeX        │        │ Root: Placeholder  │    │ Root:      │  │
│  │ artifacts need     │        │ tests for future   │    │ Vendored   │  │
│  │ full pipeline run  │        │ features           │    │ exception  │  │
│  │                    │        │                    │    │ allowances │  │
│  │ FIX: make analyze  │        │ FIX: Implement     │    │            │  │
│  └───────────────────┘        │ then remove skip   │    │ No fix     │  │
│                                └───────────────────┘    │ needed     │  │
│                                                          └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Resolution Flowchart

```
FRESH CLONE                     AFTER EXTRACTION              AFTER ANALYSIS
┌──────────────┐                ┌──────────────┐              ┌──────────────┐
│ 2042 passed  │  make extract  │ 2042 passed  │ make analyze │ 2042 passed  │
│ 181 skipped  │ ────────────▶  │ ~138 skipped │ ──────────▶  │ ~80 skipped  │
│ 0 failed     │                │ 0 failed     │              │ 0 failed     │
└──────────────┘                └──────────────┘              └──────────────┘
                                                                     │
                                                              make r-figures-all
                                                                     │
                                                                     ▼
                                                              ┌──────────────┐
                                                              │ 2042 passed  │
                                                              │ ~15 skipped  │
                                                              │ 0 failed     │
                                                              └──────────────┘
                                                              (remaining: TDD
                                                               stubs + vendored)
```

### Panel 3: Skip Condition Implementation

```
# tests/conftest.py — How skips are triggered

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DB   = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
CD_DIAGRAM_DB = PROJECT_ROOT / "data" / "public" / "cd_diagram_data.duckdb"
R_DATA_DIR   = PROJECT_ROOT / "data" / "r_data"
FIGURES_DIR  = PROJECT_ROOT / "figures" / "generated"
DEMO_DB      = PROJECT_ROOT / "data" / "synthetic" / "SYNTH_PLR_DEMO.db"

# Pattern: graceful skip if file not present
@pytest.fixture
def results_db_path():
    if not RESULTS_DB.exists():
        pytest.skip("Production DB not available")  ← Group A
    return RESULTS_DB
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: map → flowchart → implementation"
spatial_anchors:
  skip_map:
    x: 0.5
    y: 0.25
    content: "7 skip groups with root causes and fix commands"
  resolution_flow:
    x: 0.5
    y: 0.65
    content: "Linear resolution path: extract → analyze → r-figures"
  implementation:
    x: 0.5
    y: 0.9
    content: "conftest.py skip condition code"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `tests/conftest.py` | Canonical data path constants (RESULTS_DB, R_DATA_DIR, etc.) |
| `configs/demo_subjects.yaml` | 8 demo subjects (H001-H004, G001-G004) |
| `Makefile` | `extract`, `analyze`, `r-figures-all` targets |

## Code Paths

| Module | Role |
|--------|------|
| `tests/conftest.py` | Root conftest with skip logic and path constants |
| `tests/test_figure_qa/conftest.py` | Figure QA specific fixtures (R_DATA_DIR dependent) |
| `tests/test_figure_generation/conftest.py` | Auto-applies `unit` marker |
| `tests/integration/conftest.py` | Auto-applies `integration` marker |
| `tests/test_guardrails/conftest.py` | Auto-applies `guardrail` marker |

## Extension Guide

To add a new skip group:
1. Define a new path constant in `tests/conftest.py`
2. Create a fixture with `pytest.skip()` if the path does not exist
3. Document the root cause and fix command in this figure plan
4. Add the group to the resolution flowchart

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-61",
    "title": "Test Skip Groups A-H: What's Skipped and Why"
  },
  "content_architecture": {
    "primary_message": "181 test skips are organized into 7 groups (A-H), each with a distinct root cause and fix command.",
    "layout_flow": "Top-down: skip group map, resolution flowchart, implementation detail",
    "spatial_anchors": {
      "skip_map": {"x": 0.5, "y": 0.25},
      "resolution_flow": {"x": 0.5, "y": 0.65},
      "implementation": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Skip Group Map",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Group A: DB Paths", "Group B: R_DATA_DIR", "Group D: Figures", "Group E: Demo", "Group F: Manuscript", "Group G: TDD", "Group H: Vendored"]
      },
      {
        "name": "Resolution Flowchart",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["make extract", "make analyze", "make r-figures-all"]
      }
    ],
    "callout_boxes": [
      {"heading": "KEY INSIGHT", "body_text": "Three make targets resolve all resolvable skips: extract, analyze, r-figures-all."}
    ]
  }
}
```

## Alt Text

Diagnostic map of 7 test skip groups (A-H) showing root cause and fix command for each, with a resolution flowchart from 181 skips to near-zero.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
