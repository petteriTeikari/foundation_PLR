# fig-repo-70: Docker Compose Service Map

## Metadata

| Field | Value |
|-------|-------|
| **ID** | fig-repo-70 |
| **Title** | Docker Compose: 7 Services, 2 Profiles |
| **Complexity Level** | L3 |
| **Target Persona** | ML Engineer |
| **Location** | `docs/explanation/docker.md`, root `README.md` |
| **Priority** | P2 (High) |

## Purpose

Show the 7 services defined in `docker-compose.yml`, their images, volume mounts, port mappings, and profile grouping. Developers need to understand which service to use, what data each can access, and how read-only vs read-write mounts enforce data safety.

## Key Message

docker-compose.yml defines 7 services across 3 profiles: 5 default services (dev, r-figures, test, test-fast, test-data), and 2 profile-gated services (viz on port 3000, shiny on port 3838). Volume mounts enforce read-only access for data safety.

## Content Specification

### Panel 1: Service Topology

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  docker-compose.yml: 7 SERVICES, 2 PROFILES              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  DEFAULT PROFILE (no --profile needed)                                    │
│  ════════════════════════════════════                                     │
│                                                                           │
│  ┌─── dev ──────────────────┐  ┌─── r-figures ─────────────┐            │
│  │ IMAGE: foundation-plr    │  │ IMAGE: foundation-plr-r   │            │
│  │        :latest           │  │        :latest             │            │
│  │ CMD:   bash (interactive)│  │ CMD:   Rscript (run all    │            │
│  │                          │  │        src/r/figures/*.R)  │            │
│  │ VOLUMES (rw):            │  │                            │            │
│  │  src/ scripts/ tests/    │  │ VOLUMES:                   │            │
│  │  configs/ apps/          │  │  src/r/        (ro)        │            │
│  │  figures/generated/      │  │  outputs/r_data (ro)       │            │
│  │  outputs/                │  │  configs/       (ro)       │            │
│  │ VOLUMES (ro):            │  │  figures/generated/        │            │
│  │  data/public/            │  │    ggplot2/    (rw)        │            │
│  │                          │  │                            │            │
│  │ CACHES:                  │  │                            │            │
│  │  uv-cache (named vol)   │  │                            │            │
│  │  renv-cache (named vol) │  │                            │            │
│  └──────────────────────────┘  └────────────────────────────┘            │
│                                                                           │
│  ┌─── test ─────────────────┐  ┌─── test-fast ────────────┐            │
│  │ IMAGE: foundation-plr-   │  │ IMAGE: foundation-plr-   │            │
│  │        test:latest       │  │        test:latest        │            │
│  │ CMD:   pytest -m "unit   │  │ CMD:   pytest -m "unit    │            │
│  │        or guardrail"     │  │        or guardrail"      │            │
│  │        -n auto           │  │        -n auto            │            │
│  │                          │  │                           │            │
│  │ VOLUMES (ro):            │  │ VOLUMES (ro):             │            │
│  │  src/ tests/ configs/    │  │  src/ tests/ configs/     │            │
│  │  scripts/                │  │  scripts/                 │            │
│  └──────────────────────────┘  └───────────────────────────┘            │
│                                                                           │
│  ┌─── test-data ────────────┐                                            │
│  │ IMAGE: foundation-plr-   │                                            │
│  │        test:latest       │                                            │
│  │ CMD:   pytest -m "unit   │                                            │
│  │   or guardrail or data"  │                                            │
│  │        -n auto           │                                            │
│  │                          │                                            │
│  │ VOLUMES (ro):            │                                            │
│  │  src/ tests/ configs/    │                                            │
│  │  scripts/                │                                            │
│  │  data/r_data/            │  ← additional data mount                   │
│  │  data/public/            │  ← additional data mount                   │
│  └──────────────────────────┘                                            │
│                                                                           │
│  PROFILE: viz                          PROFILE: shiny                    │
│  ═══════════                           ══════════════                     │
│  ┌─── viz ──────────────────┐  ┌─── shiny ─────────────────┐            │
│  │ IMAGE: foundation-plr    │  │ IMAGE: foundation-plr-    │            │
│  │        :latest           │  │        shiny:latest       │            │
│  │ CMD:   npm start         │  │ CMD:   shiny-server       │            │
│  │ PORT:  3000:3000         │  │ PORT:  3838:3838          │            │
│  │                          │  │                           │            │
│  │ VOLUMES (rw):            │  │ VOLUMES (rw):             │            │
│  │  apps/visualization/     │  │  src/tools/ground-truth-  │            │
│  │                          │  │    creation/ → /srv/      │            │
│  │ node_modules excluded    │  │    shiny-server/          │            │
│  │  via anonymous volume    │  │                           │            │
│  │                          │  │ ENV:                      │            │
│  │ START:                   │  │  DOCKER_CONTAINER=TRUE    │            │
│  │  --profile viz up        │  │                           │            │
│  └──────────────────────────┘  │ START:                    │            │
│                                │  --profile shiny up       │            │
│                                └───────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Panel 2: Volume Mount Safety Model

```
HOST FILESYSTEM                    CONTAINER
──────────────────────────────    ──────────────────────────
src/           ─── rw ──────▶    /project/src/
tests/         ─── rw ──────▶    /project/tests/
configs/       ─── rw ──────▶    /project/configs/
scripts/       ─── rw ──────▶    /project/scripts/
apps/          ─── rw ──────▶    /project/apps/
figures/gen/   ─── rw ──────▶    /project/figures/generated/
outputs/       ─── rw ──────▶    /project/outputs/

data/public/   ─── ro ──────▶    /project/data/public/
data/r_data/   ─── ro ──────▶    /project/data/r_data/
outputs/r_data ─── ro ──────▶    /project/outputs/r_data/
src/r/         ─── ro ──────▶    /project/src/r/

RULE: Data directories are READ-ONLY (ro) in test
      and r-figures services. Only dev service has
      write access to source code directories.
```

### Panel 3: Named Volume Caches

```
NAMED VOLUMES (persist between container restarts)
┌──────────────────────────────────────────────────┐
│  foundation-plr-uv-cache                          │
│    Mount: /root/.cache/uv                         │
│    Used by: dev service                           │
│    Purpose: Cache uv downloads between runs       │
│                                                   │
│  foundation-plr-renv-cache                        │
│    Mount: /project/renv/cache                     │
│    Used by: dev service                           │
│    Purpose: Cache R package builds between runs   │
└──────────────────────────────────────────────────┘
```

### Panel 4: Command Cheatsheet

```
COMMANDS                                          SERVICE
─────────────────────────────────────────────    ─────────
docker-compose up -d dev                         dev (background)
docker-compose exec dev bash                     dev (shell)
docker-compose run --rm r-figures                r-figures (one-shot)
docker-compose run --rm test                     test (one-shot)
docker-compose run --rm test-fast                test-fast (one-shot)
docker-compose run --rm test-data                test-data (one-shot)
docker-compose --profile viz up                  viz (port 3000)
docker-compose --profile shiny up                shiny (port 3838)
docker-compose down                              stop all
```

## Spatial Anchors

```yaml
layout_flow: "Top grid of 5 default services, bottom row of 2 profile services"
spatial_anchors:
  dev:
    x: 0.2
    y: 0.15
    content: "dev: full interactive environment"
  r_figures:
    x: 0.7
    y: 0.15
    content: "r-figures: R figure generation"
  test_services:
    x: 0.5
    y: 0.4
    content: "test / test-fast / test-data: Python test runners"
  viz:
    x: 0.25
    y: 0.65
    content: "viz: React+D3.js app (profile: viz)"
  shiny:
    x: 0.75
    y: 0.65
    content: "shiny: Shiny GT tools (profile: shiny)"
  volume_model:
    x: 0.5
    y: 0.85
    content: "Volume mount safety: rw vs ro"
```

## Content Elements

### Key Structures
| Name | Role (Semantic Tag) | Description |
|------|---------------------|-------------|
| dev service | `primary_pathway` | Full interactive dev environment |
| r-figures service | `secondary_pathway` | One-shot R figure generation |
| test service | `primary_pathway` | Unit + guardrail test runner |
| test-fast service | `primary_pathway` | Fast subset test runner |
| test-data service | `primary_pathway` | Test runner with data mounts |
| viz service | `highlight_accent` | React+D3.js visualization (profile-gated) |
| shiny service | `highlight_accent` | Shiny GT creation tools (profile-gated) |

### Relationships/Connections
| From | To | Type | Label |
|------|-----|------|-------|
| dev | foundation-plr:latest | Uses image | "full image" |
| r-figures | foundation-plr-r:latest | Uses image | "R image" |
| test, test-fast, test-data | foundation-plr-test:latest | Uses image | "lean image" |
| viz | foundation-plr:latest | Uses image | "full image" |
| shiny | foundation-plr-shiny:latest | Uses image | "shiny image" |
| host src/ | container /project/src/ | Volume | "rw" |
| host data/public/ | container /project/data/public/ | Volume | "ro" |

### Callout Boxes
| Title | Content | Location |
|-------|---------|----------|
| "PROFILES" | viz and shiny require --profile flag to start | Bottom |
| "READ-ONLY" | Data directories mounted as ro in test services | Right side |
| "NAMED VOLUMES" | uv-cache and renv-cache persist between container restarts | Bottom-right |

## Config Locations

| Config File | Purpose |
|-------------|---------|
| `docker-compose.yml` | All 7 service definitions, volume mounts, profiles |
| `Dockerfile` | Image for dev and viz services |
| `Dockerfile.r` | Image for r-figures service |
| `Dockerfile.test` | Image for test, test-fast, test-data services |
| `Dockerfile.shiny` | Image for shiny service |

## Code Paths

| Module | Role |
|--------|------|
| `src/r/figures/*.R` | R scripts executed by r-figures service |
| `apps/visualization/` | React+D3.js app served by viz service |
| `src/tools/ground-truth-creation/` | Shiny apps served by shiny service |
| `tests/` | Test files executed by test services |

## Extension Guide

To add a new Docker Compose service:
1. Add service definition in `docker-compose.yml` under `services:`
2. Choose image: existing (`foundation-plr-test:latest`) or new Dockerfile
3. Set volume mounts: use `ro` for data, `rw` for output directories
4. If the service should not start by default, add `profiles: [profile-name]`
5. If the service exposes a port, add `ports: ["HOST:CONTAINER"]`
6. Set environment variables: `PREFECT_DISABLED=1`, `PYTHONPATH=/project/src`
7. Update this figure plan with the new service

Note: Performance comparisons are in the manuscript, not this repository.

## JSON Export Block (for Gemini)

```json
{
  "meta": {
    "figure_id": "repo-70",
    "title": "Docker Compose: 7 Services, 2 Profiles"
  },
  "content_architecture": {
    "primary_message": "7 Docker Compose services: 5 default (dev, r-figures, test, test-fast, test-data) and 2 profile-gated (viz on 3000, shiny on 3838). Volume mounts enforce read-only data access.",
    "layout_flow": "Top grid of default services, bottom row of profile-gated services, volume mount diagram below",
    "spatial_anchors": {
      "dev": {"x": 0.2, "y": 0.15},
      "r_figures": {"x": 0.7, "y": 0.15},
      "test_services": {"x": 0.5, "y": 0.4},
      "viz": {"x": 0.25, "y": 0.65},
      "shiny": {"x": 0.75, "y": 0.65},
      "volume_model": {"x": 0.5, "y": 0.85}
    },
    "key_structures": [
      {
        "name": "dev",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["foundation-plr:latest", "bash", "rw volumes", "full dev"]
      },
      {
        "name": "r-figures",
        "role": "secondary_pathway",
        "is_highlighted": false,
        "labels": ["foundation-plr-r:latest", "Rscript", "ro data, rw figures"]
      },
      {
        "name": "test / test-fast / test-data",
        "role": "primary_pathway",
        "is_highlighted": true,
        "labels": ["foundation-plr-test:latest", "pytest", "ro volumes"]
      },
      {
        "name": "viz",
        "role": "highlight_accent",
        "is_highlighted": false,
        "labels": ["profile: viz", "port 3000", "React + D3.js"]
      },
      {
        "name": "shiny",
        "role": "highlight_accent",
        "is_highlighted": false,
        "labels": ["profile: shiny", "port 3838", "GT creation tools"]
      }
    ],
    "callout_boxes": [
      {"heading": "PROFILES", "body_text": "viz and shiny require --profile flag: docker-compose --profile viz up"},
      {"heading": "READ-ONLY MOUNTS", "body_text": "Data dirs (data/public/, outputs/r_data/) are mounted read-only in test and r-figures services."},
      {"heading": "NAMED VOLUMES", "body_text": "uv-cache and renv-cache are named volumes that persist between container restarts for faster rebuilds."}
    ]
  }
}
```

## Alt Text

Service topology diagram of 7 Docker Compose services: dev, r-figures, test, test-fast, test-data as default, viz and shiny as profile-gated, with volume mount annotations.

## Status

- [ ] Draft created
- [ ] Review passed
- [ ] Generated
- [ ] Placed in README/docs
