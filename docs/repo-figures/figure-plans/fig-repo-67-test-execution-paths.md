# fig-repo-67: Running Tests: Pick Your Path

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-67 |
| **Title** | Running Tests: Pick Your Path |
| **Complexity Level** | L2 |
| **Target Persona** | All (new developer onboarding) |
| **Location** | `tests/README.md`, `CONTRIBUTING.md`, `docs/explanation/test-architecture.md` |
| **Priority** | P1 (Critical) |

## Purpose

Provide a decision tree for running tests, comparing speed, coverage, and requirements of each path. New developers should be able to pick the right test execution path within 10 seconds of reading this figure.

## Key Message

Three primary test execution paths serve different needs: local (fastest feedback, ~90s), Docker (CI parity, ~10min), and GitHub Actions (automatic PR gate). Additional paths exist for figure QA, R tests, and specific file targeting.

## Content Specification

### Panel 1: Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────┐
│         RUNNING TESTS: PICK YOUR PATH                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "I want to run tests"                                                  │
│  │                                                                      │
│  ├─── Quick check while coding?                                         │
│  │    └── make test-local                                               │
│  │        Speed: ~90 seconds                                            │
│  │        Coverage: Tier 1 (unit + guardrail)                           │
│  │        Requires: Python 3.11 + uv                                    │
│  │        When: Every save, before git commit                           │
│  │                                                                      │
│  ├─── Full CI parity on my machine?                                     │
│  │    └── make test-all                                                 │
│  │        Speed: ~10 minutes                                            │
│  │        Coverage: All tiers (Docker)                                   │
│  │        Requires: Docker                                              │
│  │        When: Before pushing, after major changes                     │
│  │                                                                      │
│  ├─── Just figure QA?                                                   │
│  │    └── pytest tests/test_figure_qa/ -v                               │
│  │        Speed: ~30 seconds                                            │
│  │        Coverage: 7 QA test files (P0-P3)                             │
│  │        Requires: Python 3.11 + data/r_data/                          │
│  │        When: After generating any figure                             │
│  │                                                                      │
│  ├─── R figure tests?                                                   │
│  │    └── make r-docker-test                                            │
│  │        Speed: ~5 minutes                                             │
│  │        Coverage: R script syntax + figure generation                 │
│  │        Requires: Docker (R environment inside)                       │
│  │        When: After modifying src/r/ files                            │
│  │                                                                      │
│  ├─── Specific test file?                                               │
│  │    └── pytest tests/path/to/test_file.py -v                         │
│  │        Speed: Instant (single file)                                  │
│  │        Coverage: Just that file                                      │
│  │        Requires: Python 3.11 + uv                                    │
│  │        When: Debugging a specific failure                            │
│  │                                                                      │
│  └─── PR submitted?                                                     │
│       └── GitHub Actions runs automatically                             │
│           Speed: ~35 minutes (wall time)                                │
│           Coverage: All tiers + quality gates + R lint                  │
│           Requires: Nothing (cloud)                                     │
│           When: Every push to PR branch                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Comparison Matrix

```
┌───────────────────┬──────────┬───────────────┬─────────────┬───────────┐
│  PATH             │ SPEED    │ COVERAGE      │ REQUIRES    │ WHEN      │
├───────────────────┼──────────┼───────────────┼─────────────┼───────────┤
│ make test-local   │ ~90s     │ Tier 1        │ Python+uv   │ Every save│
│ make test-all     │ ~10 min  │ All tiers     │ Docker      │ Pre-push  │
│ pytest test_qa/   │ ~30s     │ Figure QA     │ Python+data │ Post-gen  │
│ make r-docker-test│ ~5 min   │ R tests       │ Docker      │ R changes │
│ pytest file.py    │ <10s     │ Single file   │ Python+uv   │ Debugging │
│ GitHub Actions    │ ~35 min  │ Everything    │ Nothing     │ Auto (PR) │
└───────────────────┴──────────┴───────────────┴─────────────┴───────────┘
```

### Panel 3: What Each Path Actually Runs

```
make test-local
  └── PREFECT_DISABLED=1 uv run pytest tests/ -m "unit or guardrail" -n auto

make test-all
  └── docker compose run test pytest tests/ -n auto -v

pytest tests/test_figure_qa/ -v
  └── Runs all 7 QA test files (P0-P3), auto-applies @mark.data

make r-docker-test
  └── docker compose run r-figures Rscript -e "parse('file.R')"
      + R figure generation tests

pytest tests/path/to/file.py -v
  └── Direct pytest execution, inherits root conftest.py fixtures

GitHub Actions (.github/workflows/ci.yml)
  └── lint ─┬─ test-fast ── test-integration
            ├─ quality-gates
            └─ r-lint
```

### Panel 4: Common Workflows

```
┌─────────────────────────────────────────────────────────────────────┐
│  COMMON WORKFLOWS                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WORKFLOW 1: Normal development iteration                            │
│  Edit code → make test-local → fix failures → git commit             │
│                                                                      │
│  WORKFLOW 2: Before creating a PR                                    │
│  make test-local → make test-all → git push → GitHub Actions auto    │
│                                                                      │
│  WORKFLOW 3: After generating a figure                               │
│  python src/viz/generate_all_figures.py --figure R7                  │
│  → pytest tests/test_figure_qa/ -v → git commit                     │
│                                                                      │
│  WORKFLOW 4: After modifying R code                                  │
│  Edit src/r/figures/*.R → make r-docker-test → git commit            │
│                                                                      │
│  WORKFLOW 5: Debugging a specific failure                            │
│  pytest tests/unit/test_feature.py::test_specific -v --tb=long      │
└─────────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-down: decision tree → comparison matrix → implementation details → workflows"
spatial_anchors:
  decision_tree:
    x: 0.5
    y: 0.25
    content: "6 execution paths as a decision tree"
  comparison_matrix:
    x: 0.5
    y: 0.52
    content: "Speed, coverage, requirements comparison"
  implementation:
    x: 0.5
    y: 0.72
    content: "What each path actually executes"
  workflows:
    x: 0.5
    y: 0.9
    content: "5 common development workflows"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `Makefile` | `test-local`, `test-all`, `test-figures`, `r-docker-test` targets |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline (5 jobs) |
| `pyproject.toml` | pytest addopts, markers, testpaths |
| `docker-compose.yml` | Docker service definitions for test execution |

## Code Paths

| Module | Role |
|--------|------|
| `Makefile` | Local test targets (test-local, test-all, test-figures) |
| `.github/workflows/ci.yml` | CI pipeline: lint, test-fast, quality-gates, test-integration, r-lint |
| `docker-compose.yml` | Docker services: test, r-figures, dev |
| `tests/conftest.py` | Root conftest shared by all execution paths |
| `pyproject.toml` | pytest configuration (addopts, markers, testpaths) |

## Extension Guide

To add a new test execution path:
1. Define the make target in `Makefile` with the appropriate pytest `-m` filter
2. If the path requires Docker, add a service to `docker-compose.yml`
3. Document the speed, coverage, and requirements in this figure plan
4. Add a common workflow example showing when to use the new path
5. If the path should run in CI, add a job to `.github/workflows/ci.yml`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-67",
    "title": "Running Tests: Pick Your Path"
  },
  "content_architecture": {
    "primary_message": "6 test execution paths serve different needs: local (~90s), Docker (~10min), figure QA (~30s), R tests (~5min), single file (<10s), and GitHub Actions (~35min auto).",
    "layout_flow": "Top-down: decision tree, comparison matrix, implementation, workflows",
    "spatial_anchors": {
      "decision_tree": {"x": 0.5, "y": 0.25},
      "comparison_matrix": {"x": 0.5, "y": 0.52},
      "implementation": {"x": 0.5, "y": 0.72},
      "workflows": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Decision Tree",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["make test-local", "make test-all", "pytest test_qa/", "make r-docker-test", "pytest file.py", "GitHub Actions"]
      },
      {
        "name": "Comparison Matrix",
        "role": "callout_box",
        "is_highlighted": true,
        "labels": ["Speed", "Coverage", "Requires", "When"]
      },
      {
        "name": "Common Workflows",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Normal dev", "Pre-PR", "Post-figure", "R code", "Debugging"]
      }
    ],
    "callout_boxes": [
      {"heading": "QUICK START", "body_text": "For most development: make test-local (~90s). Before PR: make test-all. After figures: pytest tests/test_figure_qa/ -v."}
    ]
  }
}
```

## Alt Text

Decision tree for 6 test execution paths comparing speed, coverage, and requirements, with common workflow examples for development, PRs, figures, and debugging.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
