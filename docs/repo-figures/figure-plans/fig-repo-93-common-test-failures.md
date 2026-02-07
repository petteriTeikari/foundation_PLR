# fig-repo-93: My Tests Failed: A Diagnostic Flowchart

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-93 |
| **Title** | My Tests Failed: A Diagnostic Flowchart |
| **Complexity Level** | L2 |
| **Target Persona** | All (new developer onboarding through experienced contributor) |
| **Location** | `tests/README.md`, `docs/explanation/debugging.md`, `CONTRIBUTING.md` |
| **Priority** | P1 (Critical) |

## Purpose

Provide a single decision tree that any developer can follow when pytest reports failures. Most test failures in this repository fall into 5 predictable categories with known fixes. This figure eliminates the "my tests are failing, what do I do?" question during onboarding and daily development.

## Key Message

Most test failures are caused by missing data (run `make extract && make analyze`), not by broken code. Five categories cover nearly all failures, each with a single fix command.

## Content Specification

### Panel 1: Diagnostic Decision Tree

```
"Tests Failed"
  |
  +-- Error contains "FileNotFoundError" or "pytest.skip"?
  |   |
  |   YES --> Missing data files (most common failure!)
  |           |
  |           +-- Which file?
  |               +-- foundation_plr_results.db  --> make extract
  |               +-- cd_diagram_data.duckdb     --> make extract
  |               +-- data/r_data/*              --> make r-figures-all
  |               +-- figures/generated/*.png     --> make analyze
  |               +-- SYNTH_PLR_DEMO.db          --> make synthetic-db
  |               |
  |               +-- TIP: Run all three in sequence:
  |                   make extract && make analyze && make r-figures-all
  |
  +-- Error contains "ModuleNotFoundError"?
  |   |
  |   YES --> Missing Python package
  |           FIX: uv sync
  |           (NEVER: pip install, conda install)
  |
  +-- Error contains "Pre-commit hook failed"?
  |   |
  |   YES --> Which hook?
  |           +-- ruff           --> Auto-fixed. Run: git add <files> && git commit
  |           +-- registry-      --> Update ALL 5 anti-cheat layers:
  |           |   integrity         registry_canary.yaml, classification.yaml,
  |           |                     registry.py, test_registry.py, pre-commit args
  |           +-- computation-   --> Remove banned import from src/viz/*.py
  |           |   decoupling        (sklearn, scipy.stats, src/stats/*)
  |           +-- r-hardcoding   --> Replace hex color with load_color_definitions()
  |           |   -check            Replace ggsave() with save_publication_figure()
  |           +-- renv-sync-     --> Known pre-existing failure.
  |               check             Bypass: SKIP=renv-sync-check git commit
  |
  +-- Error contains "R not found" or "Rscript" error?
  |   |
  |   YES --> R runtime not installed
  |           Option A: Install R >= 4.4 from CRAN (system-level)
  |           Option B: make r-docker-test (Docker, no local R needed)
  |           Option C: pytest -m "not r_required" (skip R tests)
  |
  +-- None of the above?
      |
      --> Genuine test failure
          1. Read the assertion message carefully
          2. Check if test is in tests/test_guardrails/ (code quality rule)
          3. Fix the source code, not the test
          4. Re-run: pytest tests/path/to/failing_test.py -v
```

### Panel 2: Most Common Failure (Callout)

```
+-----------------------------------------------------------------------+
|  MOST COMMON: "181 tests skipped" on fresh clone                       |
|                                                                         |
|  This is NORMAL. Tests skip gracefully when data files are absent.     |
|  See fig-repo-61 (Test Skip Groups A-H) for the full breakdown.       |
|                                                                         |
|  Quick resolution:                                                      |
|    make extract        --> resolves ~43 skips (Group A: DB paths)       |
|    make analyze        --> resolves ~45 skips (Groups D+E+F: figures)   |
|    make r-figures-all  --> resolves ~34 skips (Group B: R data)         |
|    Remaining ~15 skips --> TDD stubs + vendored exceptions (expected)   |
+-----------------------------------------------------------------------+
```

### Panel 3: Quick Command Reference

```
+--------------------------+--------------------------------------------+
| Symptom                  | Fix Command                                |
+--------------------------+--------------------------------------------+
| FileNotFoundError on DB  | make extract                               |
| ModuleNotFoundError      | uv sync                                    |
| ruff hook fails          | git add <auto-fixed files>                 |
| registry hook fails      | Update 5 anti-cheat layers                 |
| computation hook fails   | Remove banned import from src/viz/         |
| R not found              | Install R >= 4.4 OR make r-docker-test     |
| Lots of pytest.skip      | make extract && make analyze               |
+--------------------------+--------------------------------------------+
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: decision tree as central element, callout box below, command table at bottom"
spatial_anchors:
  decision_tree:
    x: 0.5
    y: 0.35
    content: "Main diagnostic flowchart with 5 branches"
  callout_box:
    x: 0.5
    y: 0.72
    content: "Most common failure: 181 skips on fresh clone"
  command_table:
    x: 0.5
    y: 0.92
    content: "Quick symptom-to-fix reference table"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `tests/conftest.py` | Path constants (RESULTS_DB, R_DATA_DIR, FIGURES_DIR) and skip logic |
| `.pre-commit-config.yaml` | 7 pre-commit hooks with trigger conditions |
| `Makefile` | `extract`, `analyze`, `r-figures-all`, `r-docker-test` targets |
| `pyproject.toml` | pytest markers and configuration |

## Code Paths

| Module | Role |
|--------|------|
| `tests/conftest.py` | Root conftest with graceful skip fixtures |
| `tests/test_figure_qa/conftest.py` | Figure QA fixtures (R_DATA_DIR dependent) |
| `scripts/verify_registry_integrity.py` | Registry anti-cheat verification |
| `scripts/check_computation_decoupling.py` | Import ban enforcement in `src/viz/` |
| `scripts/check_r_hardcoding.py` | R hardcoding prevention |

## Extension Guide

To add a new diagnostic branch:
1. Identify the error pattern (the string users see in pytest output)
2. Determine the root cause and single fix command
3. Add a new branch to the decision tree in Panel 1
4. Add the symptom/fix pair to the command reference table in Panel 3
5. If the fix involves a `make` target, verify the target exists in `Makefile`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-93",
    "title": "My Tests Failed: A Diagnostic Flowchart"
  },
  "content_architecture": {
    "primary_message": "Most test failures are caused by missing data, not broken code. Five categories cover nearly all failures, each with a single fix command.",
    "layout_flow": "Top-down: decision tree, callout box, command reference table",
    "spatial_anchors": {
      "decision_tree": {"x": 0.5, "y": 0.35},
      "callout_box": {"x": 0.5, "y": 0.72},
      "command_table": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {
        "name": "Diagnostic Decision Tree",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["FileNotFoundError", "ModuleNotFoundError", "Pre-commit hook", "R not found", "Genuine failure"]
      },
      {
        "name": "Most Common Failure Callout",
        "role": "callout_box",
        "is_highlighted": true,
        "labels": ["181 skips on fresh clone", "make extract", "make analyze"]
      },
      {
        "name": "Command Reference Table",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Symptom", "Fix Command"]
      }
    ],
    "callout_boxes": [
      {"heading": "MOST COMMON", "body_text": "181 test skips on fresh clone is normal. Run make extract && make analyze to resolve."},
      {"heading": "KEY RULE", "body_text": "NEVER use pip install or conda install. Always uv sync."}
    ]
  }
}
```

## Alt Text

Decision tree for diagnosing test failures: five branches for missing data, missing packages, pre-commit hooks, R runtime, and genuine bugs, each with a fix command.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
