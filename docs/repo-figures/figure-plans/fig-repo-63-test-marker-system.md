# fig-repo-63: pytest Markers: How 2000+ Tests Are Partitioned

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-63 |
| **Title** | pytest Markers: How 2000+ Tests Are Partitioned |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer / Research Scientist |
| **Location** | `tests/README.md`, `docs/explanation/test-architecture.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show how 7 pytest markers partition the test suite into logical groups for selective execution. Developers need to know which marker to use for fast iteration versus full validation, and how markers overlap.

## Key Message

7 pytest markers partition 2000+ tests into fast/slow, data-dependent/independent, and R-required groups, enabling selective execution from 90-second unit sweeps to full integration runs.

## Content Specification

### Panel 1: Marker Definitions

```
┌─────────────────────────────────────────────────────────────────────────┐
│            pytest MARKERS: 7 GROUPS, 2000+ TESTS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MARKER          COUNT   DESCRIPTION                AUTO-APPLIED BY     │
│  ─────────────── ──────  ──────────────────────────  ──────────────────  │
│  unit            ~239    Pure functions, no I/O      unit/conftest.py    │
│  integration     ~96     Demo data, cross-module     integration/        │
│  e2e             ~18     Full pipeline, slow         e2e/conftest.py     │
│  slow            ~31     > 30 seconds runtime        Manual @mark.slow   │
│  guardrail       ~150+   Code quality, file scan     test_guardrails/    │
│  │                                                   test_no_hardcoding/ │
│  data            varies  Needs data/r_data or        test_figure_qa/     │
│  │                       data/public files           conftest.py         │
│  r_required      varies  Needs Rscript binary        Manual @mark        │
│                                                                         │
│  Defined in: pyproject.toml [tool.pytest.ini_options].markers           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Marker Overlap Venn Diagram

```
                      ┌───────────────────────────────────────────┐
                      │              ALL TESTS (~2042)             │
                      │                                           │
                      │   ┌─────────────────────────────┐        │
                      │   │        unit (~239)           │        │
                      │   │                              │        │
                      │   │   ┌────────────────────┐     │        │
                      │   │   │ guardrail overlap  │     │        │
                      │   │   │ (test_figure_gen/) │     │        │
                      │   │   └────────────────────┘     │        │
                      │   └─────────────────────────────┘        │
                      │                                           │
                      │   ┌─────────────────────────────┐        │
                      │   │     integration (~96)        │        │
                      │   │   ┌──────────────────┐       │        │
                      │   │   │ slow overlap     │       │        │
                      │   │   │ (~20 tests)      │       │        │
                      │   │   └──────────────────┘       │        │
                      │   └─────────────────────────────┘        │
                      │                                           │
                      │   ┌────────────────┐                     │
                      │   │ e2e (~18)      │                     │
                      │   │ ┌────────────┐ │                     │
                      │   │ │ slow+e2e   │ │                     │
                      │   │ │ overlap    │ │                     │
                      │   │ └────────────┘ │                     │
                      │   └────────────────┘                     │
                      │                                           │
                      │   ┌──────────────────────────────┐       │
                      │   │    guardrail (~150+)          │       │
                      │   │  test_guardrails/             │       │
                      │   │  test_no_hardcoding/          │       │
                      │   └──────────────────────────────┘       │
                      │                                           │
                      │   ┌──────────┐  ┌───────────────┐        │
                      │   │  data    │  │  r_required   │        │
                      │   │ (varies) │  │  (varies)     │        │
                      │   └──────────┘  └───────────────┘        │
                      └───────────────────────────────────────────┘

  KEY OVERLAPS:
  - slow overlaps with integration and e2e (tests > 30s in both)
  - guardrail overlaps with unit (file-scanning tests are fast)
  - data is orthogonal (depends on file existence, not test type)
  - r_required is orthogonal (depends on R installation)
```

### Panel 3: Auto-Application Mechanism

```
# Each test directory has a conftest.py that auto-applies markers:

tests/
├── conftest.py                       ← Root: path constants, skip logic
├── unit/conftest.py                  ← Auto-applies @mark.unit
├── integration/conftest.py           ← Auto-applies @mark.integration
├── e2e/conftest.py                   ← Auto-applies @mark.e2e
├── test_guardrails/conftest.py       ← Auto-applies @mark.guardrail
├── test_no_hardcoding/conftest.py    ← Auto-applies @mark.guardrail
├── test_figure_qa/conftest.py        ← Auto-applies @mark.data
└── test_figure_generation/conftest.py ← Auto-applies @mark.unit

# Implementation pattern (every per-directory conftest.py):
def pytest_collection_modifyitems(items):
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
```

### Panel 4: Command Cheatsheet

```
┌─────────────────────────────────────────────────────────────────────┐
│  COMMAND CHEATSHEET                                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FAST DEVELOPMENT LOOP                                               │
│  pytest -m unit                        # ~239 tests, ~30s            │
│  pytest -m guardrail                   # Code quality only           │
│  pytest -m "unit or guardrail"         # CI Tier 1 equivalent        │
│                                                                      │
│  SELECTIVE EXECUTION                                                 │
│  pytest -m "not slow"                  # Skip long-running tests     │
│  pytest -m "integration and not r_required"  # Integration, no R     │
│  pytest -m "not data"                  # Skip data-dependent tests   │
│                                                                      │
│  FULL VALIDATION                                                     │
│  pytest -m "integration or e2e"        # CI Tier 3 equivalent        │
│  pytest                                # Everything (2042 tests)     │
│                                                                      │
│  SPECIFIC SUBSYSTEMS                                                 │
│  pytest tests/test_figure_qa/ -v       # Figure QA (zero tolerance)  │
│  pytest tests/test_no_hardcoding/ -v   # Anti-hardcoding checks      │
│  pytest tests/test_guardrails/ -v      # All guardrails              │
└─────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: definitions → overlaps → auto-application → cheatsheet"
spatial_anchors:
  marker_table:
    x: 0.5
    y: 0.15
    content: "7 marker definitions with counts and auto-apply source"
  venn_diagram:
    x: 0.5
    y: 0.4
    content: "Overlap relationships between markers"
  auto_apply:
    x: 0.5
    y: 0.65
    content: "conftest.py hierarchy and auto-application mechanism"
  cheatsheet:
    x: 0.5
    y: 0.88
    content: "Command cheatsheet for common scenarios"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `pyproject.toml` | `[tool.pytest.ini_options].markers` -- 7 marker definitions |
| `pyproject.toml` | `addopts = ["--strict-markers"]` -- unknown markers cause errors |
| `tests/*/conftest.py` | Per-directory `pytest_collection_modifyitems()` auto-apply |

## Code Paths

| Module | Role |
|--------|------|
| `tests/unit/conftest.py` | Auto-applies `unit` marker |
| `tests/integration/conftest.py` | Auto-applies `integration` marker |
| `tests/e2e/conftest.py` | Auto-applies `e2e` marker |
| `tests/test_guardrails/conftest.py` | Auto-applies `guardrail` marker |
| `tests/test_no_hardcoding/conftest.py` | Auto-applies `guardrail` marker |
| `tests/test_figure_qa/conftest.py` | Auto-applies `data` marker |
| `tests/test_figure_generation/conftest.py` | Auto-applies `unit` marker |

## Extension Guide

To add a new pytest marker:
1. Define the marker in `pyproject.toml` under `[tool.pytest.ini_options].markers`
2. Create a `conftest.py` in the test directory with `pytest_collection_modifyitems()` to auto-apply
3. If the marker should gate a CI job, update `.github/workflows/ci.yml` with the `-m` filter
4. Document the marker in this figure plan

Note: `--strict-markers` in `addopts` ensures any undeclared marker causes an immediate error.

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-63",
    "title": "pytest Markers: How 2000+ Tests Are Partitioned"
  },
  "content_architecture": {
    "primary_message": "7 pytest markers partition 2000+ tests into fast/slow, data-dependent/independent, and R-required groups for selective execution.",
    "layout_flow": "Top-down: definitions, overlap Venn, auto-application, command cheatsheet",
    "spatial_anchors": {
      "marker_table": {"x": 0.5, "y": 0.15},
      "venn_diagram": {"x": 0.5, "y": 0.4},
      "auto_apply": {"x": 0.5, "y": 0.65},
      "cheatsheet": {"x": 0.5, "y": 0.88}
    },
    "key_structures": [
      {
        "name": "Marker Definitions Table",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["unit", "integration", "e2e", "slow", "guardrail", "data", "r_required"]
      },
      {
        "name": "Overlap Venn Diagram",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["slow+integration", "slow+e2e", "guardrail+unit"]
      },
      {
        "name": "Command Cheatsheet",
        "role": "callout_box",
        "is_highlighted": false,
        "labels": ["pytest -m unit", "pytest -m 'not slow'"]
      }
    ],
    "callout_boxes": [
      {"heading": "STRICT MARKERS", "body_text": "Undeclared markers cause immediate errors via --strict-markers in pyproject.toml."}
    ]
  }
}
```

## Alt Text

Diagram showing 7 pytest markers partitioning 2000+ tests with a Venn diagram of overlaps and a command cheatsheet for selective execution.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
