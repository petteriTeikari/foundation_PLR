# fig-repo-88: 40+ Make Targets: Organized by Purpose

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-88 |
| **Title** | 40+ Make Targets: Organized by Purpose |
| **Complexity Level** | L2 |
| **Target Persona** | All |
| **Location** | `README.md`, `Makefile` (header comment) |
| **Priority** | P2 (High) |

## Purpose

Provide a visual map of all Makefile targets grouped by purpose. New developers face 40+ targets and need to know which ones to use first, which are for specialized tasks, and how targets relate to each other. The figure highlights "start here" entry points.

## Key Message

Make targets are organized into 8 categories. Start with `make test-local` for development, `make reproduce` for the full pipeline, `make figures` for publication figures.

## Content Specification

### Panel 1: Grouped Target Map

```
                              make help
                                 │
    ┌────────────┬───────────┬───┴───┬──────────┬──────────┬─────────────┬──────────┐
    │            │           │       │          │          │             │          │
PIPELINE     FIGURES    TESTING   TESTING   DOCKER     DOCKER    REGISTRY   EXPERIMENT
(reproduce)  (generate) (Docker)  (local)   (Python)   (R)       (verify)   (manage)
    │            │           │       │          │          │             │          │
    │            │           │       │          │          │             │          │
┌───┴────┐  ┌───┴────┐  ┌───┴──┐ ┌─┴──────┐ ┌┴────────┐ ┌┴──────────┐ ┌┴────────┐ ┌┴─────────┐
│reproduce│  │figures │  │test  │ │test-   │ │docker-  │ │r-docker-  │ │verify-  │ │list-     │
│ ★       │  │        │  │      │ │local ★ │ │build    │ │build      │ │registry-│ │experiment│
│extract  │  │figure  │  │test- │ │test-   │ │docker-  │ │r-docker-  │ │integrity│ │run-      │
│analyze  │  │ ID=R7  │  │fast  │ │local-  │ │run      │ │run        │ │         │ │experiment│
│reproduce│  │figures-│  │test- │ │all     │ │docker-  │ │r-docker-  │ │check-   │ │new-      │
│-from-   │  │list    │  │data  │ │test-   │ │test     │ │test       │ │registry │ │experiment│
│checkpt  │  │        │  │test- │ │figures │ │docker-  │ │r-docker-  │ │test-    │ │validate- │
│verify-  │  │        │  │all   │ │test-viz│ │shell    │ │shell      │ │registry │ │experiment│
│extract  │  │        │  │      │ │test-   │ │docker-  │ │           │ │         │ │          │
│         │  │        │  │      │ │registry│ │compose- │ │           │ │         │ │          │
│         │  │        │  │      │ │test-   │ │up       │ │           │ │         │ │          │
│         │  │        │  │      │ │integr  │ │docker-  │ │           │ │         │ │          │
│         │  │        │  │      │ │type-chk│ │compose- │ │           │ │         │ │          │
│         │  │        │  │      │ │        │ │down     │ │           │ │         │ │          │
└────────┘  └────────┘  └──────┘ └────────┘ └─────────┘ └───────────┘ └─────────┘ └──────────┘
                                     │
                              R FIGURES
                              ┌──────────────┐
                              │r-figures-all │
                              │r-figures-    │
                              │ sprint1/2/3  │
                              │r-figures-    │
                              │ stratos      │
                              │r-validate    │
                              │r-clean       │
                              └──────────────┘

  OTHER:
  ┌─────────────────┐
  │ compliance       │
  │ validate         │
  │ clean            │
  │ setup            │
  │ install-hooks    │
  └─────────────────┘

  ★ = START HERE (recommended first commands for new developers)
```

### Panel 2: "Start Here" Quick Reference

```
NEW DEVELOPER:
  1. make test-local           ★ Verify setup works (Tier 1, ~90s)
  2. make test-figures         ★ Check figure QA (if making figures)
  3. make check-registry       ★ Verify method counts (if editing registry)

REPRODUCING RESULTS:
  1. make reproduce            Full pipeline: MLflow → DuckDB → Figures
  2. make reproduce-from-checkpoint  Skip extraction, use existing DuckDB
  3. make verify-extraction    Confirm extraction succeeded

MAKING FIGURES:
  1. make figures              All Python matplotlib figures
  2. make figure ID=R7         Single figure by ID
  3. make figures-list         See available figure IDs
  4. make r-figures-all        All R ggplot2 figures
  5. make r-figures-sprint1    Just Sprint 1 R figures

CI PARITY:
  1. make test                 Docker-based Tier 1 (same as CI)
  2. make test-all             All tiers in Docker
  3. make r-docker-test        R environment test in Docker
```

### Panel 3: Target Dependencies

```
reproduce ─────────┬── extract (Block 1: MLflow → DuckDB)
                   └── analyze (Block 2: DuckDB → Figures/Stats)

check-registry ────┬── verify-registry-integrity (script check)
                   └── test-registry (pytest assertions)

r-figures-all ─────┬── r-figures-sprint1 (7 figures)
                   ├── r-figures-sprint2 (3 figures)
                   └── r-figures-sprint3 (6 figures)

docker-run ────────── docker-build (builds image first)
r-docker-run ──────── r-docker-build (builds R image first)
```

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `Makefile` | All target definitions (465 lines) |
| `scripts/test-docker.sh` | Docker test runner (used by `make test*`) |
| `scripts/reproduce_all_results.py` | Pipeline runner (used by `make reproduce*`) |
| `configs/VISUALIZATION/figure_registry.yaml` | Figure IDs for `make figure ID=X` |

## Code Paths

| Module | Role |
|--------|------|
| `Makefile` | Target definitions, dependency chains |
| `scripts/test-docker.sh` | Docker test execution (`--tier 1`, `--data`, `--all`) |
| `scripts/reproduce_all_results.py` | `--extract-only`, `--analyze-only`, `--from-checkpoint` |
| `src/viz/generate_all_figures.py` | `--figure ID`, `--list` |
| `scripts/verify_registry_integrity.py` | Cross-layer registry verification |
| `scripts/validate_experiments.py` | Experiment config validation |
| `scripts/validate_figures.py` | R figure validation |

## Extension Guide

To add a new Make target:
1. Add `.PHONY` declaration at the top of the relevant section
2. Define the target with its dependencies
3. Add help text in the `help` target
4. If the target uses Docker, follow the existing pattern (build first, then run)
5. Update this figure plan if the target belongs to a new category

Target naming conventions:
- `test-*`: Testing targets (local and Docker)
- `docker-*`: Docker management targets
- `r-*`: R-related targets
- `*-all`: Run all variants of a category

Note: This is a repo documentation figure - shows HOW make targets are organized, NOT research results.

## JSON Export Block

```json
{
  "meta": {
    "figure_id": "repo-88",
    "title": "40+ Make Targets: Organized by Purpose"
  },
  "content_architecture": {
    "primary_message": "Make targets are organized into 8 categories. Start with make test-local for development, make reproduce for full pipeline.",
    "layout_flow": "Top-down tree from 'make help' branching into 8 category groups, with 'Start Here' callout",
    "spatial_anchors": {
      "root": {"x": 0.5, "y": 0.02},
      "pipeline": {"x": 0.05, "y": 0.15},
      "figures": {"x": 0.18, "y": 0.15},
      "testing_docker": {"x": 0.31, "y": 0.15},
      "testing_local": {"x": 0.44, "y": 0.15},
      "docker_py": {"x": 0.57, "y": 0.15},
      "docker_r": {"x": 0.7, "y": 0.15},
      "registry": {"x": 0.83, "y": 0.15},
      "start_here": {"x": 0.3, "y": 0.75, "width": 0.4, "height": 0.2}
    },
    "key_structures": [
      {
        "name": "make test-local",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["START HERE", "~90s"]
      },
      {
        "name": "make reproduce",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Full pipeline"]
      },
      {
        "name": "make figures",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["All Python figures"]
      }
    ],
    "callout_boxes": [
      {"heading": "START HERE", "body_text": "New developers: make test-local (verify setup), make test-figures (figure QA), make check-registry (method counts)."},
      {"heading": "CI PARITY", "body_text": "make test runs the same tests as GitHub Actions CI. Use for pre-push verification."}
    ]
  }
}
```

## Alt Text

Grouped tree diagram showing 40-plus Makefile targets organized into 8 categories: pipeline, figures, testing Docker, testing local, Docker Python, Docker R, registry, and experiments. Star markers indicate starting points for new developers.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
