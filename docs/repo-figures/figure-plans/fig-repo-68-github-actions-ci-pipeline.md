# fig-repo-68: GitHub Actions CI Pipeline

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-68 |
| **Title** | CI Pipeline: From Push to Green Check |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `.github/workflows/README.md`, `docs/explanation/ci-cd.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show how the unified CI workflow (`ci.yml`) orchestrates 5 jobs with explicit parallelism and dependency chains. Developers need to understand which jobs gate which, what triggers the pipeline, and where to look when a job fails.

## Key Message

The CI pipeline runs 5 jobs: 3 in parallel (lint, test-fast, r-lint), quality-gates parallel with test-fast, then integration tests after test-fast passes. Total wall time is approximately 35 minutes.

## Content Specification

### Panel 1: Trigger Conditions

```
TRIGGER CONDITIONS
┌─────────────────────────────────────────────────────────────────────┐
│  push:                                                               │
│    branches: [main]                                                  │
│                                                                      │
│  pull_request:                                                       │
│    branches: [main]                                                  │
│    paths-ignore: ['*.md', 'docs/**', '.claude/**']                   │
│                                                                      │
│  workflow_dispatch:   (manual trigger)                                │
│                                                                      │
│  concurrency:                                                        │
│    group: ci-${{ github.ref }}                                       │
│    cancel-in-progress: true                                          │
│    (Cancels stale runs when new commits push to same branch)         │
└─────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Job DAG (Primary Content)

```
                        git push / PR opened
                              │
                              ▼
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  lint         │ │  test-fast   │ │  r-lint      │
     │  (Tier 0)     │ │  (Tier 1)    │ │              │
     │  ~30s         │ │  ~2-3 min    │ │  ~2 min      │
     │               │ │              │ │              │
     │  ruff check   │ │  pytest      │ │  Rscript -e  │
     │  ruff format  │ │  -m "unit or │ │  parse()     │
     │  --check      │ │   guardrail" │ │  for all .R  │
     │               │ │  -n auto     │ │  files       │
     │  No deps      │ │  (xdist)     │ │              │
     │  installed    │ │              │ │  R 4.5.2     │
     └──────────────┘ └──────┬───────┘ └──────────────┘
                              │
              ┌───────────────┤
              │               │
              ▼               ▼
     ┌──────────────┐ ┌──────────────────┐
     │ quality-gates │ │ test-integration │
     │  (~2 min)     │ │  (Tier 3)        │
     │               │ │  ~5-10 min       │
     │  verify_      │ │                  │
     │  registry_    │ │  pytest           │
     │  integrity.py │ │  -m "integration │
     │               │ │   or e2e"        │
     │  check_       │ │  -n auto         │
     │  computation_ │ │  (xdist)         │
     │  decoupling.py│ │                  │
     │               │ │  needs:          │
     │  check_       │ │    test-fast     │
     │  parallel_    │ │                  │
     │  systems.py   │ │                  │
     └──────────────┘ └──────────────────┘
       (PARALLEL)       (SEQUENTIAL)
       no dependency     depends on
       chain             test-fast

    ═══════════════════════════════════════
    TOTAL WALL TIME: ~15-35 min
    (depends on cache hits and test count)
    ═══════════════════════════════════════
```

### Panel 3: Job Details

| Job | Timeout | Runner | Key Steps | Dependencies |
|-----|---------|--------|-----------|--------------|
| `lint` | 5 min | ubuntu-latest | `uvx ruff check`, `uvx ruff format --check` | None (no project deps) |
| `test-fast` | 10 min | ubuntu-latest | `uv sync --dev --frozen`, pytest with xdist | None |
| `r-lint` | 10 min | ubuntu-latest | `r-lib/actions/setup-r@v2`, parse all `.R` files | None |
| `quality-gates` | 10 min | ubuntu-latest | 3 validation scripts | None (runs in parallel) |
| `test-integration` | 15 min | ubuntu-latest | pytest integration + e2e with xdist | `needs: test-fast` |

### Panel 4: Failure Modes

```
WHEN A JOB FAILS                        WHERE TO LOOK
┌────────────────────┐                  ┌────────────────────────────────┐
│ lint fails         │ ─────────────── │ Run: uvx ruff check src/       │
│                    │                  │ Auto-fix: uvx ruff check --fix │
├────────────────────┤                  ├────────────────────────────────┤
│ test-fast fails    │ ─────────────── │ Check test-results-tier1       │
│                    │                  │ artifact (JUnit XML)           │
├────────────────────┤                  ├────────────────────────────────┤
│ quality-gates fails│ ─────────────── │ Registry: scripts/verify_      │
│                    │                  │ registry_integrity.py          │
│                    │                  │ Decoupling: check imports in   │
│                    │                  │ src/viz/                       │
├────────────────────┤                  ├────────────────────────────────┤
│ test-integration   │ ─────────────── │ Needs production data or       │
│ fails              │                  │ demo subjects to pass fully    │
├────────────────────┤                  ├────────────────────────────────┤
│ r-lint fails       │ ─────────────── │ Syntax error in src/r/figures/ │
│                    │                  │ Fix R script, re-push          │
└────────────────────┘                  └────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom DAG with horizontal parallel branches"
spatial_anchors:
  trigger:
    x: 0.5
    y: 0.05
    content: "Trigger conditions (push, PR, manual)"
  parallel_row:
    x: 0.5
    y: 0.3
    content: "3 parallel jobs: lint, test-fast, r-lint"
  quality_gates:
    x: 0.25
    y: 0.6
    content: "Quality gates (parallel, no dependency)"
  integration:
    x: 0.75
    y: 0.6
    content: "Integration tests (depends on test-fast)"
  failure_guide:
    x: 0.5
    y: 0.85
    content: "Failure modes and diagnostic actions"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Trigger block | `callout_box` | Push/PR/dispatch conditions |
| lint job | `traditional_method` | Lightweight style check |
| test-fast job | `primary_pathway` | Core unit + guardrail tests |
| r-lint job | `secondary_pathway` | R syntax validation |
| quality-gates job | `primary_pathway` | Registry + decoupling enforcement |
| test-integration job | `primary_pathway` | Full integration + e2e tests |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| Trigger | lint | Arrow | "parallel" |
| Trigger | test-fast | Arrow | "parallel" |
| Trigger | r-lint | Arrow | "parallel" |
| Trigger | quality-gates | Arrow | "parallel" |
| test-fast | test-integration | Dependency Arrow | "needs: test-fast" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "PARALLEL" | lint, test-fast, r-lint, quality-gates start simultaneously | Middle-left |
| "SEQUENTIAL" | test-integration waits for test-fast | Middle-right |
| "CANCEL" | Stale runs cancelled on new push (concurrency group) | Top-right |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.github/workflows/ci.yml` | Unified CI workflow definition (5 jobs) |
| `pyproject.toml` | ruff configuration for lint job |
| `uv.lock` | Python dependency lockfile (cache key for test jobs) |

## Code Paths

| Module | Role |
|--------|------|
| `scripts/verify_registry_integrity.py` | Quality gate: checks 11/8/5 method counts |
| `scripts/check_computation_decoupling.py` | Quality gate: no sklearn in src/viz/ |
| `scripts/check_parallel_systems.py` | Quality gate: no duplicate config systems |
| `src/r/figures/*.R` | Files checked by r-lint job |

## Extension Guide

To add a new CI job:
1. Add job definition in `.github/workflows/ci.yml` under `jobs:`
2. Set `needs:` to specify dependency chain (or omit for parallel execution)
3. Set `timeout-minutes:` to prevent runaway jobs
4. Use `astral-sh/setup-uv@v4` with `enable-cache: true` for Python jobs
5. Use `r-lib/actions/setup-r@v2` for R jobs
6. Add to the DAG diagram in this figure plan

To add a new quality gate:
1. Create script in `scripts/check_<name>.py`
2. Add step in `quality-gates` job
3. Add corresponding pre-commit hook in `.pre-commit-config.yaml`

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-68",
    "title": "CI Pipeline: From Push to Green Check"
  },
  "content_architecture": {
    "primary_message": "5 CI jobs run with 3 parallel and 1 sequential dependency, totaling ~35 min wall time.",
    "layout_flow": "Top-to-bottom DAG with horizontal parallel branches",
    "spatial_anchors": {
      "trigger": {"x": 0.5, "y": 0.05},
      "parallel_row": {"x": 0.5, "y": 0.3},
      "quality_gates": {"x": 0.25, "y": 0.6},
      "integration": {"x": 0.75, "y": 0.6},
      "failure_guide": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "Trigger Conditions",
        "role": "callout_box",
        "is_highlighted": false,
        "labels": ["push: main", "pull_request: main", "workflow_dispatch"]
      },
      {
        "name": "lint",
        "role": "traditional_method",
        "is_highlighted": false,
        "labels": ["Tier 0", "ruff check + format", "~30s", "no deps"]
      },
      {
        "name": "test-fast",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Tier 1", "unit + guardrail", "xdist parallel", "~2-3 min"]
      },
      {
        "name": "r-lint",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["R 4.5.2", "parse() syntax check", "~2 min"]
      },
      {
        "name": "quality-gates",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["registry integrity", "computation decoupling", "parallel systems"]
      },
      {
        "name": "test-integration",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Tier 3", "integration + e2e", "needs: test-fast", "~5-10 min"]
      }
    ],
    "callout_boxes": [
      {"heading": "PARALLEL", "body_text": "lint, test-fast, r-lint, and quality-gates all start simultaneously on push/PR."},
      {"heading": "SEQUENTIAL", "body_text": "test-integration waits for test-fast to pass before starting."},
      {"heading": "CONCURRENCY", "body_text": "cancel-in-progress: true cancels stale CI runs when a new push arrives on the same branch."}
    ]
  }
}
```

## Alt Text

Directed acyclic graph showing 5 CI jobs: lint, test-fast, and r-lint run in parallel, quality-gates runs in parallel, test-integration depends on test-fast.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
