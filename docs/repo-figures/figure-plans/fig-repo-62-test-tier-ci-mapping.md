# fig-repo-62: From Test Tiers to CI Jobs

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-62 |
| **Title** | From Test Tiers to CI Jobs |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `.github/workflows/README.md`, `docs/explanation/ci-architecture.md` |
| **Priority** | P1 (Critical) |

## Purpose

Map the local test tier system to the GitHub Actions CI pipeline. Developers need to understand which local tests correspond to which CI jobs, what runs in parallel, and the dependency chain that gates merging.

## Key Message

4 local test tiers map to 5 CI jobs with explicit dependency chains and parallelism. Lint, test-fast, quality-gates, and r-lint run in parallel; test-integration depends on test-fast passing. Total wall time is approximately 35 minutes.

## Content Specification

### Panel 1: Tier-to-Job Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LOCAL TEST TIERS  ──────▶  GITHUB ACTIONS JOBS             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LOCAL TIER                    CI JOB                                   │
│  ┌─────────────────────┐      ┌──────────────────────────────────┐     │
│  │ Tier 0: Lint         │─────▶│ lint  (timeout: 5 min)           │     │
│  │ ruff check + format  │      │ uvx ruff check src/ tests/       │     │
│  │ No project deps      │      │ uvx ruff format --check          │     │
│  └─────────────────────┘      └──────────────────────────────────┘     │
│                                         │                               │
│  ┌─────────────────────┐      ┌────────▼─────────────────────────┐     │
│  │ Tier 1: Unit +       │─────▶│ test-fast  (timeout: 10 min)    │     │
│  │ Guardrail            │      │ -m "unit or guardrail"           │     │
│  │ Fast, no I/O         │      │ pytest-xdist -n auto             │     │
│  └─────────────────────┘      └──────────────────────────────────┘     │
│                                         │                               │
│  ┌─────────────────────┐      ┌────────▼─────────────────────────┐     │
│  │ Quality Gates        │─────▶│ quality-gates  (timeout: 10 min)│     │
│  │ Registry + Decoupling│      │ verify_registry_integrity.py     │     │
│  │ + Parallel systems   │      │ check_computation_decoupling.py  │     │
│  └─────────────────────┘      │ check_parallel_systems.py        │     │
│                                └──────────────────────────────────┘     │
│                                         │                               │
│  ┌─────────────────────┐      ┌────────▼─────────────────────────┐     │
│  │ Tier 3: Integration  │─────▶│ test-integration  (timeout: 15m)│     │
│  │ + E2E                │      │ -m "integration or e2e"          │     │
│  │ Demo data, slow      │      │ needs: test-fast                 │     │
│  └─────────────────────┘      └──────────────────────────────────┘     │
│                                                                         │
│                                ┌──────────────────────────────────┐     │
│                          ─────▶│ r-lint  (timeout: 10 min)        │     │
│  (No local tier)               │ Rscript -e "parse('file.R')"    │     │
│                                │ Parallel with all above          │     │
│                                └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Dependency DAG

```
                    push / PR trigger
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
         ┌────────┐  ┌─────────┐  ┌────────┐
         │  lint  │  │test-fast│  │ r-lint │
         │  ~30s  │  │  ~3min  │  │ ~2min  │
         └────────┘  └────┬────┘  └────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              ┌──────────┐  ┌─────────────────┐
              │ quality- │  │test-integration  │
              │ gates    │  │ needs: test-fast │
              │ ~2min    │  │ ~10min           │
              └──────────┘  └─────────────────┘

   PARALLEL ──────────────  SEQUENTIAL ─────▶
   (no dependency)          (needs: keyword)

   Total wall time: ~35 minutes (dominated by test-integration)
```

### Panel 3: CI Configuration Details

```
# .github/workflows/ci.yml

on:
  pull_request:
    branches: [main]
    paths-ignore: ['*.md', 'docs/**', '.claude/**']
  push:
    branches: [main]
  workflow_dispatch:                    ← Manual trigger

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true             ← Cancels older runs on same branch

env:
  PYTHON_VERSION: "3.11"
  MPLBACKEND: Agg                      ← Headless matplotlib
```

## Spatial Anchors

```yaml
layout_flow: "Left-to-right: local tiers → CI jobs, then top-down DAG"
spatial_anchors:
  tier_mapping:
    x: 0.5
    y: 0.3
    content: "Side-by-side mapping of local tiers to CI jobs"
  dependency_dag:
    x: 0.5
    y: 0.7
    content: "DAG showing parallel and sequential job execution"
  ci_config:
    x: 0.5
    y: 0.9
    content: "Key workflow configuration snippets"
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.github/workflows/ci.yml` | Main CI workflow (5 jobs, triggers, concurrency) |
| `pyproject.toml` | pytest markers, addopts, testpaths |
| `Makefile` | Local test targets (`test-local`, `test-all`, etc.) |

## Code Paths

| Module | Role |
|--------|------|
| `.github/workflows/ci.yml` | CI workflow definition (5 jobs) |
| `scripts/verify_registry_integrity.py` | Quality gate: registry anti-cheat |
| `scripts/check_computation_decoupling.py` | Quality gate: import ban enforcement |
| `scripts/check_parallel_systems.py` | Quality gate: parallel system verification |
| `tests/conftest.py` | Root test configuration and fixtures |

## Extension Guide

To add a new CI job:
1. Add the job definition to `.github/workflows/ci.yml`
2. Specify `needs:` dependencies if the job depends on another
3. Set an appropriate `timeout-minutes`
4. If the job requires a new test marker, add it to `pyproject.toml` `[tool.pytest.ini_options].markers`
5. Create a conftest.py `pytest_collection_modifyitems()` to auto-apply the marker

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-62",
    "title": "From Test Tiers to CI Jobs"
  },
  "content_architecture": {
    "primary_message": "4 local test tiers map to 5 CI jobs with explicit dependency chains and parallelism, totaling ~35 minutes wall time.",
    "layout_flow": "Left-to-right tier mapping, then top-down dependency DAG",
    "spatial_anchors": {
      "tier_mapping": {"x": 0.5, "y": 0.3},
      "dependency_dag": {"x": 0.5, "y": 0.7},
      "ci_config": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Tier-to-Job Mapping",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Tier 0: lint", "Tier 1: test-fast", "Quality Gates", "Tier 3: integration", "R Lint"]
      },
      {
        "name": "Dependency DAG",
        "role": "secondary_pathway",
        "is_highlighted": true,
        "labels": ["parallel", "needs: test-fast", "~35 min wall time"]
      }
    ],
    "callout_boxes": [
      {"heading": "KEY INSIGHT", "body_text": "lint, test-fast, quality-gates, and r-lint run in parallel. Only test-integration waits for test-fast."}
    ]
  }
}
```

## Alt Text

Diagram mapping 4 local test tiers to 5 GitHub Actions CI jobs, showing parallel execution and sequential dependency chains with approximate durations.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
