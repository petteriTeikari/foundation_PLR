# fig-repo-65: Figure QA: 7 Test Files, Zero Tolerance

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-65 |
| **Title** | Figure QA: 7 Test Files, Zero Tolerance |
| **Complexity Level** | L3 |
| **Target Persona** | Research Scientist / ML Engineer |
| **Location** | `tests/test_figure_qa/README.md`, `docs/explanation/figure-qa.md` |
| **Priority** | P1 (Critical) |

## Purpose

Document the figure QA test system that enforces scientific integrity on every generated figure. This system exists because of CRITICAL-FAILURE-001 (synthetic data in figures). Every test file targets a specific failure mode, organized by priority from P0 (data fraud) to P3 (accessibility).

## Key Message

Figure QA has 7 specialized test files organized by priority (P0=synthetic fraud, P1=statistical validity, P2=rendering quality, P3=accessibility). ALL must pass before any figure is committed. Zero tolerance means zero exceptions.

## Content Specification

### Panel 1: Priority-Ordered Test Files

```
┌─────────────────────────────────────────────────────────────────────────┐
│         FIGURE QA: 7 TEST FILES, ZERO TOLERANCE                         │
│         Command: pytest tests/test_figure_qa/ -v                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  P0  DATA INTEGRITY (Critical — scientific fraud prevention)            │
│  ╔═════════════════════════════════════════════════════════════════╗    │
│  ║  test_data_provenance.py                                        ║    │
│  ║  ├── Verifies figure data comes from real MLflow runs           ║    │
│  ║  ├── Checks source_hash matches production DB                   ║    │
│  ║  ├── Rejects synthetic/generated data in figure JSONs           ║    │
│  ║  └── Example failure: "Figure uses SYNTH_PLR data, not real"    ║    │
│  ║                                                                  ║    │
│  ║  Origin: CRITICAL-FAILURE-001 (synthetic data in calibration    ║    │
│  ║  plots shipped as real model predictions)                        ║    │
│  ╚═════════════════════════════════════════════════════════════════╝    │
│                                                                         │
│  P1  STATISTICAL VALIDITY (High — scientific correctness)               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  test_statistical_validity.py                                    │   │
│  │  ├── Validates metric values are in plausible ranges             │   │
│  │  ├── Checks confidence intervals do not overlap impossibly       │   │
│  │  ├── Verifies sample sizes match expected N                      │   │
│  │  └── Example failure: "AUROC CI width is 0.0 (degenerate)"      │   │
│  │                                                                  │   │
│  │  test_no_nan_ci.py                                               │   │
│  │  ├── Scans all JSON sidecar files for NaN values                 │   │
│  │  ├── Checks CI bounds are finite numbers                         │   │
│  │  ├── Validates bootstrap produced actual distributions           │   │
│  │  └── Example failure: "calibration_slope_ci_lower is NaN"        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  P2  PUBLICATION STANDARDS (Medium — journal requirements)              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  test_publication_standards.py                                   │   │
│  │  ├── Checks DPI >= 100 for all generated figures                 │   │
│  │  ├── Validates dimensions match config specifications            │   │
│  │  ├── Verifies font sizes are readable at publication scale       │   │
│  │  └── Example failure: "Figure DPI is 72, minimum is 100"         │   │
│  │                                                                  │   │
│  │  test_rendering_artifacts.py                                     │   │
│  │  ├── Checks for clipped content, missing axis labels             │   │
│  │  ├── Validates legends are present and readable                  │   │
│  │  ├── Verifies no overlapping text elements                       │   │
│  │  └── Example failure: "Y-axis label is missing"                  │   │
│  │                                                                  │   │
│  │  test_visual_rendering.py                                        │   │
│  │  ├── Visual regression against golden images                     │   │
│  │  ├── Detects unexpected layout changes                           │   │
│  │  ├── Uses perceptual hashing for comparison                      │   │
│  │  └── Example failure: "Figure hash differs from golden by >10%"  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  P3  ACCESSIBILITY (Lower priority — inclusive design)                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  test_accessibility.py                                           │   │
│  │  ├── Checks color palette is colorblind-safe                     │   │
│  │  ├── Validates sufficient contrast ratios                        │   │
│  │  ├── Verifies patterns/markers supplement color encoding         │   │
│  │  └── Example failure: "Red-green only palette detected"          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: QA in the Figure Lifecycle

```
                    FIGURE GENERATION LIFECYCLE
                    ─────────────────────────

1. Code                2. Generate              3. QA GATE
   src/viz/*.py  ──▶     PNG + JSON sidecar  ──▶   pytest tests/test_figure_qa/
                                                      │
                                                      ├── ALL PASS → 4. Commit
                                                      │
                                                      └── ANY FAIL → BLOCKED
                                                          ├── P0 fail = data integrity crisis
                                                          ├── P1 fail = statistical bug
                                                          ├── P2 fail = rendering fix needed
                                                          └── P3 fail = accessibility fix

   ENFORCEMENT:
   ├── Pre-commit: figure-isolation-check (prevents synthetic data)
   ├── CI: test-fast includes figure QA tests
   └── Rule: CRITICAL-FAILURE-001 in .claude/docs/meta-learnings/
```

### Panel 3: Test Fixture Dependencies

```
tests/test_figure_qa/conftest.py
├── PROJECT_ROOT, R_DATA_DIR, FIGURES_DIR, GOLDEN_DIR
│
├── calibration_json_path()  → skips if data/r_data/calibration_data.json missing
├── dca_json_path()          → skips if data/r_data/dca_data.json missing
├── predictions_json_path()  → skips if data/r_data/predictions_top4.json missing
├── all_json_files()         → skips if data/r_data/ directory missing
├── all_figure_files()       → skips if figures/generated/ directory missing
├── ggplot2_figures()        → skips if figures/generated/ggplot2/ missing
└── golden_dir()             → creates tests/golden_images/ if needed
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: priority-ordered test files → lifecycle integration → fixture dependencies"
spatial_anchors:
  priority_table:
    x: 0.5
    y: 0.35
    content: "7 test files organized P0-P3 with example failures"
  lifecycle:
    x: 0.5
    y: 0.72
    content: "QA gate in the figure generation lifecycle"
  fixtures:
    x: 0.5
    y: 0.92
    content: "conftest.py fixture dependencies and skip conditions"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `tests/test_figure_qa/conftest.py` | Fixtures for JSON data, figure files, golden images |
| `.pre-commit-config.yaml` | `figure-isolation-check` hook (prevents synthetic in figures/) |
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-001-synthetic-data-in-figures.md` | Incident report |

## Code Paths

| Module | Role |
|--------|------|
| `tests/test_figure_qa/test_data_provenance.py` | P0: Synthetic data detection |
| `tests/test_figure_qa/test_statistical_validity.py` | P1: Metric range and CI validation |
| `tests/test_figure_qa/test_no_nan_ci.py` | P1: NaN detection in bootstrap outputs |
| `tests/test_figure_qa/test_publication_standards.py` | P2: DPI, dimensions, font checks |
| `tests/test_figure_qa/test_rendering_artifacts.py` | P2: Missing labels, clipping |
| `tests/test_figure_qa/test_visual_rendering.py` | P2: Visual regression vs golden images |
| `tests/test_figure_qa/test_accessibility.py` | P3: Colorblind safety, contrast |
| `scripts/check_figure_isolation.py` | Pre-commit: synthetic data gate |

## Extension Guide

To add a new figure QA test:
1. Determine the priority level (P0-P3) based on the failure mode
2. Add the test to the appropriate existing file, or create a new file if the failure mode is distinct
3. If the test needs new fixture data, add fixtures to `tests/test_figure_qa/conftest.py`
4. Ensure the test skips gracefully (with `pytest.skip()`) when data is not available
5. All figure QA tests auto-receive `@mark.data` via the conftest `pytest_collection_modifyitems()`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-65",
    "title": "Figure QA: 7 Test Files, Zero Tolerance"
  },
  "content_architecture": {
    "primary_message": "Figure QA has 7 specialized test files organized by priority (P0-P3). ALL must pass before any figure commit.",
    "layout_flow": "Top-down: priority-ordered test files, lifecycle gate, fixture dependencies",
    "spatial_anchors": {
      "priority_table": {"x": 0.5, "y": 0.35},
      "lifecycle": {"x": 0.5, "y": 0.72},
      "fixtures": {"x": 0.5, "y": 0.92}
    },
    "key_structures": [
      {
        "name": "P0 Data Integrity",
        "role": "abnormal_warning",
        "is_highlighted": true,
        "labels": ["test_data_provenance.py", "CRITICAL-FAILURE-001"]
      },
      {
        "name": "P1 Statistical Validity",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["test_statistical_validity.py", "test_no_nan_ci.py"]
      },
      {
        "name": "P2 Publication Standards",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["test_publication_standards.py", "test_rendering_artifacts.py", "test_visual_rendering.py"]
      },
      {
        "name": "P3 Accessibility",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["test_accessibility.py"]
      }
    ],
    "callout_boxes": [
      {"heading": "ZERO TOLERANCE", "body_text": "ALL 7 test files must pass. There is no such thing as a low-priority scientific integrity issue."},
      {"heading": "ORIGIN STORY", "body_text": "CRITICAL-FAILURE-001: Claude generated calibration plots with SYNTHETIC data instead of real predictions."}
    ]
  }
}
```

## Alt Text

Priority-ordered diagram of 7 figure QA test files from P0 data integrity to P3 accessibility, showing example failures and the zero-tolerance enforcement gate.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
