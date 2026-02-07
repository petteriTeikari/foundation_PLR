# fig-repo-69: Docker Multi-Image Architecture

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-69 |
| **Title** | Four Docker Images, Four Purposes |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `docs/explanation/docker.md`, root `README.md` |
| **Priority** | P1 (Critical) |

## Purpose

Show the four specialized Docker images in the repository, their base images, what they include, their approximate sizes, and when to use each one. Developers need to pick the right image for their task without reading all four Dockerfiles.

## Key Message

4 specialized Docker images serve different needs: full dev (Python+R+Node, ~2GB), R-only figures (renv-pinned, ~1GB), lean test (Python-only multi-stage, ~400MB), and Shiny (legacy GT tools, port 3838).

## Content Specification

### Panel 1: Four Images Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FOUR DOCKER IMAGES, FOUR PURPOSES                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Dockerfile (FULL DEV)                Dockerfile.r (R FIGURES)            │
│  ┌─────────────────────────┐          ┌─────────────────────────┐        │
│  │ Base: rocker/tidyverse   │          │ Base: rocker/tidyverse   │        │
│  │       :4.5.2             │          │       :4.5.2             │        │
│  │                          │          │                          │        │
│  │ + Python 3.11 (uv 0.5)  │          │ + renv 1.1.6             │        │
│  │ + R 4.5.2 (renv 1.1.6)  │          │ + renv.lock pinned       │        │
│  │ + Node.js 20 LTS        │          │ + ggplot2, pminternal    │        │
│  │ + All Python deps        │          │ + dcurves, pROC          │        │
│  │ + All R deps             │          │                          │        │
│  │ + npm packages           │          │ No Python, No Node.js    │        │
│  │                          │          │                          │        │
│  │ SIZE: ~2 GB              │          │ SIZE: ~1 GB              │        │
│  │ USE:  Full dev env       │          │ USE:  R figure gen       │        │
│  │ CMD:  bash (interactive) │          │ CMD:  Rscript setup.R    │        │
│  └─────────────────────────┘          └─────────────────────────┘        │
│                                                                           │
│  Dockerfile.test (LEAN CI)            Dockerfile.shiny (INTERACTIVE)     │
│  ┌─────────────────────────┐          ┌─────────────────────────┐        │
│  │ Base: python:3.11-slim   │          │ Base: rocker/shiny       │        │
│  │       -bookworm          │          │       :4.5.2             │        │
│  │                          │          │                          │        │
│  │ Multi-stage build:       │          │ + renv 1.1.6             │        │
│  │   builder → runtime      │          │ + Shiny Server           │        │
│  │                          │          │ + Cairo, xvfb            │        │
│  │ + Python 3.11 (uv 0.5)  │          │ + hht, missForest        │        │
│  │ + pytest + xdist         │          │ + changepoint, imputeTS  │        │
│  │                          │          │                          │        │
│  │ No R, No Node.js         │          │ PORT: 3838               │        │
│  │                          │          │ USE:  GT creation tools  │        │
│  │ SIZE: ~400 MB            │          │ CMD:  shiny-server       │        │
│  │ USE:  Fast CI tests      │          │                          │        │
│  │ CMD:  pytest (Tier 1)    │          │ Apps:                    │        │
│  └─────────────────────────┘          │  /inspect_outliers       │        │
│                                        │  /inspect_EMD            │        │
│                                        └─────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Base Image Relationships

```
                    Docker Hub
                       │
          ┌────────────┼────────────────┐
          │            │                │
          ▼            ▼                ▼
  python:3.11-slim  rocker/tidyverse  rocker/shiny
  -bookworm         :4.5.2           :4.5.2
     │                 │                │
     │            ┌────┴────┐           │
     │            │         │           │
     ▼            ▼         ▼           ▼
  Dockerfile   Dockerfile  Dockerfile  Dockerfile
  .test        (FULL)      .r          .shiny
  (multi-      (multi-     (single-    (single-
   stage)       stage)      stage)      stage)
```

### Panel 3: Multi-Stage Build Pattern (Dockerfile.test)

```
STAGE 1: builder                     STAGE 2: runtime
┌──────────────────────────┐        ┌──────────────────────────┐
│ python:3.11-slim-bookworm│        │ python:3.11-slim-bookworm│
│                          │        │                          │
│ apt: build-essential,    │        │ apt: git, libgomp1       │
│      git, libgomp1       │        │ (NO build-essential)     │
│                          │        │                          │
│ COPY uv from ghcr.io    │        │ COPY uv from ghcr.io     │
│ COPY pyproject.toml      │        │ COPY --from=builder      │
│ COPY uv.lock             │        │   /opt/venv → /opt/venv  │
│                          │        │                          │
│ uv venv /opt/venv        │        │ COPY src/ tests/ configs/│
│ uv sync --frozen --dev   │        │      scripts/            │
│                          │        │                          │
│ /opt/venv = compiled     │  ───▶  │ CMD: pytest -m "unit     │
│ Python packages          │  COPY  │   or guardrail" -n auto  │
└──────────────────────────┘        └──────────────────────────┘
  ~800 MB (with compilers)            ~400 MB (runtime only)

  Build deps discarded ──────▶ Smaller final image
```

### Panel 4: Multi-Stage Build Pattern (Dockerfile, full dev)

```
STAGE 1: python-builder              STAGE 2: final (rocker/tidyverse:4.5.2)
┌──────────────────────────┐        ┌──────────────────────────┐
│ python:3.11-slim-bookworm│        │ rocker/tidyverse:4.5.2   │
│                          │        │                          │
│ apt: build-essential     │        │ + Python 3.11            │
│ COPY uv from ghcr.io    │        │   (deadsnakes PPA)       │
│ uv sync --frozen --no-dev│        │ + Node.js 20 LTS        │
│                          │        │   (nodesource)           │
│ /opt/venv = compiled     │        │                          │
│ Python packages          │  ───▶  │ COPY --from=python-      │
│                          │  COPY  │   builder /opt/venv      │
└──────────────────────────┘        │                          │
                                    │ renv::restore()          │
                                    │ npm ci --production      │
                                    │                          │
                                    │ COPY src/ configs/ etc.  │
                                    │ HEALTHCHECK: python +    │
                                    │   R + node               │
                                    └──────────────────────────┘
```

### Panel 5: Layer Caching Strategy

```
LAYER CACHING: What changes rarely is built first

Layer 1 (base):    Base image (rocker/python)          ← Rarely changes
Layer 2 (deps):    renv.lock / uv.lock / package.json  ← Changes on dep update
Layer 3 (restore): renv::restore() / uv sync / npm ci  ← Cached if lock unchanged
Layer 4 (code):    COPY src/ tests/ configs/            ← Changes every commit
Layer 5 (dirs):    mkdir outputs/ figures/              ← Never changes

Rebuild from Layer 4 on code change (~30s)
Rebuild from Layer 3 on dep change (~10-30 min)
Rebuild from Layer 1 on base image bump (~45-60 min)
```

## Spatial Anchors

```yaml
layout_flow: "2x2 grid with base image tree above and caching below"
spatial_anchors:
  full_dev:
    x: 0.25
    y: 0.2
    content: "Dockerfile (FULL): rocker + Python + R + Node.js"
  r_figures:
    x: 0.75
    y: 0.2
    content: "Dockerfile.r: rocker + renv pinned R packages"
  lean_test:
    x: 0.25
    y: 0.5
    content: "Dockerfile.test: python:3.11-slim multi-stage"
  shiny:
    x: 0.75
    y: 0.5
    content: "Dockerfile.shiny: rocker/shiny for GT tools"
  multi_stage:
    x: 0.5
    y: 0.75
    content: "Multi-stage build pattern (builder → runtime)"
  caching:
    x: 0.5
    y: 0.9
    content: "Layer caching strategy"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| Dockerfile (FULL) | `primary_pathway` | Full dev environment, all 3 runtimes |
| Dockerfile.r | `secondary_pathway` | R-only figure generation image |
| Dockerfile.test | `primary_pathway` | Lean multi-stage CI test image |
| Dockerfile.shiny | `secondary_pathway` | Legacy interactive Shiny tools |
| python-builder stage | `outlier_detection` | Temporary build stage, discarded |
| final stage | `classification` | Slim runtime with compiled deps copied in |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| python:3.11-slim | Dockerfile.test | Base image | "multi-stage" |
| python:3.11-slim | Dockerfile (builder) | Base image | "builder stage" |
| rocker/tidyverse:4.5.2 | Dockerfile (final) | Base image | "final stage" |
| rocker/tidyverse:4.5.2 | Dockerfile.r | Base image | "single stage" |
| rocker/shiny:4.5.2 | Dockerfile.shiny | Base image | "single stage" |
| builder stage | final stage | COPY | "/opt/venv" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "MULTI-STAGE" | Builder compiles deps, runtime copies binaries only | Below test image |
| "uv 0.5.14" | All Python images pin uv version via ghcr.io/astral-sh/uv:0.5.14 | Top-right |
| "renv 1.1.6" | All R images pin renv version for lockfile compatibility | Top-right |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `Dockerfile` | Full development environment (Python + R + Node.js) |
| `Dockerfile.r` | R-only figure generation environment |
| `Dockerfile.test` | Lean Python-only test environment |
| `Dockerfile.shiny` | Legacy Shiny server for GT creation tools |
| `renv.lock` | Pinned R package versions (used by all R images) |
| `pyproject.toml` + `uv.lock` | Pinned Python dependencies (used by Python images) |
| `apps/visualization/package.json` | Node.js dependencies (used by full image) |

## Code Paths

| Module | Role |
|--------|------|
| `src/r/figures/*.R` | R figure scripts executed in Dockerfile.r |
| `src/r/setup.R` | R environment verification script |
| `src/tools/ground-truth-creation/` | Shiny apps mounted in Dockerfile.shiny |
| `apps/visualization/` | React+D3.js app built in full image |
| `renv/activate.R` | renv bootstrap script copied into all R images |
| `renv/settings.json` | renv configuration copied into all R images |

## Extension Guide

To add a new Docker image:
1. Create `Dockerfile.<purpose>` in the repository root
2. Choose base image: `python:3.11-slim-bookworm` (Python-only) or `rocker/tidyverse:4.5.2` (R-included)
3. Use multi-stage build if compilation is needed (separate builder from runtime)
4. Pin tool versions: `COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /usr/local/bin/uv`
5. Copy lockfiles before source code (layer caching)
6. Add service to `docker-compose.yml`
7. Add build job to `.github/workflows/docker.yml`
8. Update this figure plan with the new image

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-69",
    "title": "Four Docker Images, Four Purposes"
  },
  "content_architecture": {
    "primary_message": "4 specialized Docker images serve different needs: full dev (~2GB), R figures (~1GB), lean test (~400MB), and Shiny (interactive GT tools).",
    "layout_flow": "2x2 grid of image cards with base image tree and caching diagram",
    "spatial_anchors": {
      "full_dev": {"x": 0.25, "y": 0.2},
      "r_figures": {"x": 0.75, "y": 0.2},
      "lean_test": {"x": 0.25, "y": 0.5},
      "shiny": {"x": 0.75, "y": 0.5},
      "multi_stage": {"x": 0.5, "y": 0.75},
      "caching": {"x": 0.5, "y": 0.9}
    },
    "key_structures": [
      {
        "name": "Dockerfile (FULL)",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["rocker/tidyverse:4.5.2", "Python 3.11 + R 4.5.2 + Node.js 20", "~2GB", "full dev env"]
      },
      {
        "name": "Dockerfile.r (R FIGURES)",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["rocker/tidyverse:4.5.2", "renv 1.1.6 + renv.lock", "~1GB", "R figure gen"]
      },
      {
        "name": "Dockerfile.test (LEAN)",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["python:3.11-slim-bookworm", "multi-stage build", "~400MB", "fast CI tests"]
      },
      {
        "name": "Dockerfile.shiny (INTERACTIVE)",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["rocker/shiny:4.5.2", "port 3838", "GT creation tools"]
      }
    ],
    "callout_boxes": [
      {"heading": "MULTI-STAGE BUILD", "body_text": "Dockerfile.test uses builder stage for compilation, copies only binaries to slim runtime. Cuts image size by ~50%."},
      {"heading": "LAYER CACHING", "body_text": "Lockfiles (renv.lock, uv.lock) copied before source code. Dep install cached unless lockfile changes."},
      {"heading": "PINNED VERSIONS", "body_text": "uv 0.5.14, renv 1.1.6, R 4.5.2, Python 3.11 — all pinned for reproducibility."}
    ]
  }
}
```

## Alt Text

Four Docker image cards in a 2x2 grid showing full dev, R figures, lean test, and Shiny images with their base images, sizes, and purposes.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
