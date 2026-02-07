# fig-repo-71: Docker Build Pipeline (CI)

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-71 |
| **Title** | Docker Workflow: Build, Test, Push |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `docs/explanation/docker.md`, `.github/workflows/README.md` |
| **Priority** | P2 (High) |

## Purpose

Show the DAG of the `docker.yml` GitHub Actions workflow: how 3 Docker images are built in parallel, tested, and conditionally pushed to GHCR. Developers need to understand the build pipeline, what triggers it (only Docker-related file changes), and why push only happens on main + manual dispatch.

## Key Message

docker.yml builds 3 images in parallel (R 45min, test 30min, full 60min), runs Python tests in Docker after the full build, then conditionally pushes R and full images to GHCR on main branch with manual dispatch only.

## Content Specification

### Panel 1: Trigger Conditions

```
TRIGGER CONDITIONS (docker.yml)
┌─────────────────────────────────────────────────────────────────────┐
│  pull_request:                                                       │
│    branches: [main]                                                  │
│    paths:                           ← ONLY when Docker files change  │
│      - 'Dockerfile'                                                  │
│      - 'Dockerfile.r'                                                │
│      - 'Dockerfile.test'                                             │
│      - 'docker-compose.yml'                                          │
│      - 'renv.lock'                                                   │
│      - 'pyproject.toml'                                              │
│      - 'uv.lock'                                                     │
│      - '.github/workflows/docker.yml'                                │
│                                                                      │
│  workflow_dispatch:   (manual trigger from GitHub UI)                 │
│                                                                      │
│  NOTE: No push trigger. Docker builds only on PR or manual dispatch. │
└─────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Job DAG (Primary Content)

```
                    PR to main (Docker paths changed)
                    OR workflow_dispatch (manual)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  build-r     │ │  build-test  │ │  build-full  │
     │              │ │              │ │              │
     │  timeout:    │ │  timeout:    │ │  timeout:    │
     │  45 min      │ │  30 min      │ │  60 min      │
     │              │ │              │ │              │
     │  Dockerfile  │ │  Dockerfile  │ │  Dockerfile  │
     │  .r          │ │  .test       │ │  (full)      │
     │              │ │              │ │              │
     │  STEPS:      │ │  STEPS:      │ │  STEPS:      │
     │  1. Checkout │ │  1. Checkout │ │  1. Checkout │
     │  2. Buildx   │ │  2. Buildx   │ │  2. Buildx   │
     │  3. Build    │ │  3. Build    │ │  3. Build    │
     │     (no push)│ │     (no push)│ │     (no push)│
     │  4. Test R:  │ │  4. Run      │ │  4. Test Py  │
     │     library  │ │     Tier 1   │ │  5. Test R   │
     │     (ggplot2,│ │     tests    │ │  6. Test Node│
     │     pminter- │ │     in Docker│ │              │
     │     nal)     │ │              │ │              │
     │  5. Test R   │ │              │ │              │
     │     syntax   │ │              │ │              │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            │                │                ▼
            │                │       ┌──────────────┐
            │                │       │ test-python  │
            │                │       │              │
            │                │       │ timeout:     │
            │                │       │ 30 min       │
            │                │       │              │
            │                │       │ needs:       │
            │                │       │   build-full │
            │                │       │              │
            │                │       │ STEPS:       │
            │                │       │ 1. Rebuild   │
            │                │       │    full img  │
            │                │       │    (cached)  │
            │                │       │ 2. pytest    │
            │                │       │    in Docker │
            │                │       │    (no integ)│
            │                │       └──────┬───────┘
            │                │               │
            ▼                ▼               ▼
     ┌────────────────────────────────────────────┐
     │              push-images                     │
     │                                              │
     │  CONDITIONAL:                                │
     │    if: github.ref == 'refs/heads/main'       │
     │        AND event == 'workflow_dispatch'       │
     │                                              │
     │  needs: [build-r, build-test,                │
     │          build-full, test-python]             │
     │                                              │
     │  STEPS:                                      │
     │  1. Login to GHCR                            │
     │  2. Build + Push R image                     │
     │     → ghcr.io/$REPO-r:latest                 │
     │     → ghcr.io/$REPO-r:<sha>                  │
     │  3. Build + Push full image                  │
     │     → ghcr.io/$REPO:latest                   │
     │     → ghcr.io/$REPO:<sha>                    │
     │                                              │
     │  Tags: latest + git SHA                      │
     │  Cache: GitHub Actions cache (GHA)           │
     └────────────────────────────────────────────┘
```

### Panel 3: Cache Strategy

```
DOCKER BUILD CACHE (GitHub Actions Cache)
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  All build steps use:                                             │
│    cache-from: type=gha        ← Read from GHA cache             │
│    cache-to: type=gha,mode=max ← Write all layers to cache       │
│                                                                   │
│  First build:  ~45-60 min (no cache)                              │
│  Cached build: ~5-10 min (only changed layers rebuilt)            │
│                                                                   │
│  Cache invalidation triggers:                                     │
│    renv.lock changed   → R package layer rebuilt                  │
│    uv.lock changed     → Python package layer rebuilt             │
│    Dockerfile changed  → Full rebuild from changed instruction    │
│    Source code changed → Only final COPY layers rebuilt           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Panel 4: What Gets Pushed to GHCR

```
GITHUB CONTAINER REGISTRY (ghcr.io)
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ONLY pushed on: main branch + manual workflow_dispatch           │
│  NOT pushed on: PRs, feature branches, automatic pushes          │
│                                                                   │
│  R Image:                                                         │
│    ghcr.io/<owner>/foundation_plr-r:latest                       │
│    ghcr.io/<owner>/foundation_plr-r:<git-sha>                    │
│                                                                   │
│  Full Image:                                                      │
│    ghcr.io/<owner>/foundation_plr:latest                         │
│    ghcr.io/<owner>/foundation_plr:<git-sha>                      │
│                                                                   │
│  Test image: NOT pushed (built and discarded after tests pass)   │
│  Shiny image: NOT pushed (local use only)                        │
│                                                                   │
│  Permissions: contents:read, packages:write                       │
│  Auth: GITHUB_TOKEN (automatic)                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Spatial Anchors

```yaml
layout_flow: "Top-to-bottom DAG: 3 parallel builds → test-python → conditional push"
spatial_anchors:
  trigger:
    x: 0.5
    y: 0.05
    content: "Trigger: PR (Docker paths) or manual dispatch"
  parallel_builds:
    x: 0.5
    y: 0.25
    content: "3 parallel builds: build-r, build-test, build-full"
  test_python:
    x: 0.7
    y: 0.5
    content: "test-python: depends on build-full"
  push_images:
    x: 0.5
    y: 0.75
    content: "push-images: conditional on main + manual dispatch"
  cache_strategy:
    x: 0.5
    y: 0.9
    content: "GHA cache strategy"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| build-r | `secondary_pathway` | Builds Dockerfile.r, tests R packages |
| build-test | `secondary_pathway` | Builds Dockerfile.test, runs Tier 1 tests |
| build-full | `primary_pathway` | Builds full Dockerfile, tests all 3 runtimes |
| test-python | `primary_pathway` | Runs pytest in Docker after full build |
| push-images | `highlight_accent` | Conditional push to GHCR (main + dispatch only) |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| Trigger | build-r | Arrow | "parallel" |
| Trigger | build-test | Arrow | "parallel" |
| Trigger | build-full | Arrow | "parallel" |
| build-full | test-python | Dependency | "needs: build-full" |
| build-r | push-images | Dependency | "needs" |
| build-test | push-images | Dependency | "needs" |
| build-full | push-images | Dependency | "needs" |
| test-python | push-images | Dependency | "needs" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "PARALLEL" | All 3 builds start simultaneously | Top |
| "CONDITIONAL" | push-images only runs on main + workflow_dispatch | Bottom |
| "GHA CACHE" | type=gha,mode=max caches all Docker layers | Bottom-right |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `.github/workflows/docker.yml` | Docker CI workflow (5 jobs) |
| `Dockerfile` | Full dev image built by build-full |
| `Dockerfile.r` | R figure image built by build-r |
| `Dockerfile.test` | Lean test image built by build-test |

## Code Paths

| Module | Role |
|--------|------|
| `renv.lock` | Triggers rebuild when R deps change |
| `pyproject.toml` + `uv.lock` | Triggers rebuild when Python deps change |
| `docker-compose.yml` | Triggers workflow on change (path filter) |
| `tests/` | Executed by build-test and test-python jobs |
| `src/r/figures/*.R` | Syntax-checked by build-r job |

## Extension Guide

To add a new Docker image to the CI pipeline:
1. Create `Dockerfile.<name>` in repo root
2. Add `build-<name>` job in `.github/workflows/docker.yml`
3. Use `docker/setup-buildx-action@v3` for BuildKit
4. Use `docker/build-push-action@v5` with `push: false`, `load: true`
5. Add validation steps (test commands)
6. Add to `needs:` array of `push-images` job
7. If pushing to GHCR, add metadata extraction and push step
8. Add Dockerfile path to `paths:` trigger filter
9. Update this figure plan with the new job

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-71",
    "title": "Docker Workflow: Build, Test, Push"
  },
  "content_architecture": {
    "primary_message": "docker.yml builds 3 images in parallel, tests Python in Docker, then conditionally pushes to GHCR on main + manual dispatch.",
    "layout_flow": "Top-to-bottom DAG: 3 parallel builds, test-python depends on build-full, push-images depends on all",
    "spatial_anchors": {
      "trigger": {"x": 0.5, "y": 0.05},
      "parallel_builds": {"x": 0.5, "y": 0.25},
      "test_python": {"x": 0.7, "y": 0.5},
      "push_images": {"x": 0.5, "y": 0.75},
      "cache_strategy": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "build-r",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Dockerfile.r", "45 min timeout", "test: library(ggplot2)", "test: R syntax"]
      },
      {
        "name": "build-test",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["Dockerfile.test", "30 min timeout", "run Tier 1 tests"]
      },
      {
        "name": "build-full",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["Dockerfile", "60 min timeout", "test Python + R + Node.js"]
      },
      {
        "name": "test-python",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["needs: build-full", "30 min timeout", "pytest in Docker"]
      },
      {
        "name": "push-images",
        "role": "highlight_accent",
        "is_highlighted": true,
        "labels": ["CONDITIONAL", "main + workflow_dispatch", "R image + full image", "GHCR"]
      }
    ],
    "callout_boxes": [
      {"heading": "PARALLEL BUILDS", "body_text": "build-r, build-test, and build-full all start simultaneously. No dependencies between them."},
      {"heading": "CONDITIONAL PUSH", "body_text": "push-images only runs on main branch AND manual workflow_dispatch. PR builds never push."},
      {"heading": "GHA CACHE", "body_text": "All builds use GitHub Actions cache (type=gha,mode=max). Subsequent builds reuse cached layers."}
    ]
  }
}
```

## Alt Text

DAG of Docker CI workflow: 3 parallel build jobs feed into test-python, then conditional push-images to GHCR on main branch only.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
