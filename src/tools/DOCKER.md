# Docker Usage for Legacy R Shiny Tools

This document describes how to run the legacy ground truth creation tools using Docker.

## Overview

The ground truth creation tools (`inspect_outliers` and `inspect_EMD`) are R Shiny applications used during the original dataset annotation process. They are preserved in Docker for:

1. **Reproducibility** - Run the exact annotation environment used in the study
2. **Documentation** - Understand the annotation workflow
3. **Future annotation** - If additional ground truth data needs to be created

## Important Note

**These apps require the original SERI PLR dataset to function.** Without data mounted at `/data/PLR`, the apps will return HTTP 500 errors during initialization. This is expected behavior - the Docker infrastructure is preserved to document the exact annotation environment, not for standalone use.

To verify the Docker infrastructure works, you can access the sample apps:
- http://localhost:3838/01_hello (should work without data)

## Quick Start

```bash
# Build and start the Shiny server
docker compose --profile shiny up shiny

# Access the apps in your browser:
# - Outlier inspection: http://localhost:3838/inspect_outliers
# - EMD inspection: http://localhost:3838/inspect_EMD
```

## Building the Image

```bash
# Build the Shiny Docker image
docker build -t foundation-plr-shiny -f Dockerfile.shiny .

# Or use docker compose
docker compose --profile shiny build shiny
```

## Running the Container

### Using docker compose (Recommended)

```bash
# Start in foreground (see logs)
docker compose --profile shiny up shiny

# Start in background
docker compose --profile shiny up -d shiny

# Stop
docker compose --profile shiny down
```

### Using docker directly

```bash
docker run -p 3838:3838 \
  -v $(pwd)/src/tools/ground-truth-creation:/srv/shiny-server/ground-truth-creation \
  -e DOCKER_CONTAINER=TRUE \
  foundation-plr-shiny
```

## Data Configuration

The Shiny apps read data paths from a configuration file. When running in Docker, you need to:

1. **Mount your data directory** in docker compose.yml:

```yaml
shiny:
  volumes:
    # ... existing volumes ...
    - /path/to/your/PLR/data:/data/PLR:rw
```

2. **Create/update config/paths.csv** in the ground-truth-creation directory with Docker-compatible paths (no header row):

```csv
data_in,linux,/data/PLR/input
data_out,linux,/data/PLR/output
data_in,windows,C:/Data/PLR/input
data_out,windows,C:/Data/PLR/output
```

## Applications

### inspect_outliers

Interactive tool for reviewing and correcting automatic outlier detection results.

**Purpose**: Allow human annotators to:
- View raw PLR signals with detected outliers
- Exclude additional outlier points (too conservative)
- Include back points incorrectly marked as outliers (too aggressive)
- Save corrected annotations

**Access**: http://localhost:3838/inspect_outliers

### inspect_EMD

Interactive tool for Empirical Mode Decomposition (EMD) component selection.

**Purpose**: Allow human annotators to:
- View decomposed IMF (Intrinsic Mode Function) components
- Assign each IMF to signal categories (noise, hiFreq, loFreq, base)
- Generate denoised signal from selected components
- Save mapping and reconstructed signals

**Access**: http://localhost:3838/inspect_EMD

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_CONTAINER` | Set to `TRUE` to enable Docker-specific path handling | Not set |

## Troubleshooting

### Apps not loading

1. Check container logs: `docker compose --profile shiny logs shiny`
2. Verify Shiny server is running: `docker compose --profile shiny exec shiny ps aux`
3. Check symlinks exist: `docker compose --profile shiny exec shiny ls -la /srv/shiny-server/`

### Data not found

1. Verify volume mounts in docker compose.yml
2. Check paths.csv configuration
3. Ensure data files exist at mounted paths

### R package errors

The Dockerfile installs these packages not in renv.lock:
- hht (Hilbert-Huang Transform)
- missForest (missing data imputation)
- changepoint (changepoint detection)
- imputeTS (time series imputation)
- doParallel (parallel processing)
- moments (statistical moments)
- Cairo (graphics rendering)

If additional packages are needed, add them to the Dockerfile and rebuild.

## Dependencies

### System libraries (in Docker image)

- `libcairo2-dev` - Cairo graphics library
- `libgtk2.0-dev` - GTK+ development files
- `xvfb` - Virtual framebuffer for headless rendering
- `xauth` - X authentication
- `xfonts-base` - Base X fonts
- `libxt-dev` - X Toolkit intrinsics
- `libgit2-dev` - Git library (for devtools/remotes)

### R packages

See `renv.lock` for the full list plus the additional packages mentioned above.

## Architecture

```
/srv/shiny-server/
├── ground-truth-creation/           # Mounted from src/tools/ground-truth-creation
│   ├── shiny-apps/
│   │   ├── inspect_outliers/
│   │   │   ├── ui.R
│   │   │   └── server.R
│   │   └── inspect_EMD/
│   │       ├── ui.R
│   │       └── server.R
│   ├── config/                      # Configuration files
│   ├── PLR_IO/                      # I/O utilities
│   └── PLR_reconstruction/          # Signal reconstruction utilities
├── inspect_outliers -> ground-truth-creation/shiny-apps/inspect_outliers
└── inspect_EMD -> ground-truth-creation/shiny-apps/inspect_EMD
```

The symlinks at the root level allow accessing apps via simple URLs like `/inspect_outliers` instead of `/ground-truth-creation/shiny-apps/inspect_outliers`.
