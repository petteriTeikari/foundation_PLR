# Comprehensive Plan: Legacy R Tools Documentation & Docker Containerization

**Created:** 2026-01-31
**Status:** Planning Complete
**Related Issue:** https://github.com/petteriTeikari/foundation_PLR/issues/8

## Executive Summary

This document consolidates the plans for:
1. **Docker containerization** of the legacy R Shiny tools
2. **Documentation** for manuscript readers
3. **Integration** with the existing R environment (renv)

---

## Part 1: Docker Containerization

### Current State

The project already has:
- `Dockerfile.r` - rocker/tidyverse:4.5.2 with renv
- `docker-compose.yml` - services for dev, r-figures, test, viz
- `renv.lock` - reproducible R packages

### Missing R Packages

The legacy Shiny tools require packages NOT in current renv.lock:

| Package | Version | Purpose | Used By |
|---------|---------|---------|---------|
| `hht` | latest | CEEMD decomposition | `lowLevel_decomposition_wrappers.R` |
| `missForest` | latest | RF imputation | `lowLevel_imputation_wrappers.R` |
| `changepoint` | latest | PELT detection | `changepoint_detection.R` |
| `imputeTS` | latest | Time series imputation | `lowLevel_imputation_wrappers.R` |
| `doParallel` | latest | Parallel missForest | `lowLevel_imputation_wrappers.R` |
| `moments` | latest | Statistical moments | `inspect_EMD/server.R` |
| `Cairo` | latest | High-quality graphics | All Shiny apps |

### Recommended Approach: Dockerfile.shiny

Create a new `Dockerfile.shiny` based on `rocker/shiny:4.5.2`:

```dockerfile
# Dockerfile.shiny - R Shiny environment for legacy PLR tools
FROM rocker/shiny:4.5.2

LABEL maintainer="Foundation PLR Team"
LABEL description="R Shiny environment for PLR ground truth creation tools"

# System dependencies for Cairo
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev libgtk2.0-dev xvfb xauth xfonts-base libxt-dev libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# renv setup
ENV RENV_VERSION=1.1.6
RUN R -e "install.packages('remotes', repos = 'https://cloud.r-project.org'); \
          remotes::install_version('renv', version = '1.1.6', repos = 'https://cloud.r-project.org')"

WORKDIR /project

# Copy renv files
COPY renv.lock .Rprofile renv/activate.R renv/settings.json ./renv/

# Restore base packages
RUN R -e "options(warn = 1); renv::restore(prompt = FALSE)"

# Install Shiny-specific packages
RUN R -e "install.packages(c('hht', 'missForest', 'changepoint', 'imputeTS', \
                              'doParallel', 'moments', 'Cairo'), \
                           repos = 'https://cloud.r-project.org')"

# Copy Shiny apps
COPY src/tools/ground-truth-creation/ /srv/shiny-server/ground-truth-creation/

# Create symlinks
RUN ln -s /srv/shiny-server/ground-truth-creation/shiny-apps/inspect_outliers \
          /srv/shiny-server/inspect_outliers && \
    ln -s /srv/shiny-server/ground-truth-creation/shiny-apps/inspect_EMD \
          /srv/shiny-server/inspect_EMD

EXPOSE 3838
CMD ["/usr/bin/shiny-server"]
```

### Docker-Compose Addition

```yaml
  # R Shiny Apps for Ground Truth Creation (Legacy Tools)
  shiny:
    build:
      context: .
      dockerfile: Dockerfile.shiny
    image: foundation-plr-shiny:latest
    container_name: foundation-plr-shiny
    ports:
      - "3838:3838"
    volumes:
      - ./src/tools/ground-truth-creation/shiny-apps:/srv/shiny-server/apps:ro
      - ./data:/project/data:rw
    profiles:
      - shiny  # Start with: docker-compose --profile shiny up
```

### Known Issue: rstudioapi Dependency

The Shiny apps use `rstudioapi::getActiveDocumentContext()$path` to find their location. This fails in Docker. Required fix in server.R files:

```r
# Replace rstudioapi path detection with:
if (Sys.getenv("DOCKER_CONTAINER") == "TRUE") {
  path_base <- "/project/src/tools/ground-truth-creation"
} else if (rstudioapi::isAvailable()) {
  # Existing RStudio detection
} else {
  path_base <- getwd()
}
```

---

## Part 2: Documentation Plan

### Documentation Files to Create

| File | Priority | Purpose |
|------|----------|---------|
| `src/tools/README.md` | **HIGH** | Main entry point |
| `src/tools/ground-truth-creation/README.md` | **HIGH** | Pipeline overview |
| `src/tools/ground-truth-creation/imputation/README.md` | **HIGH** | MissForest params |
| `src/tools/ground-truth-creation/denoising/README.md` | **HIGH** | CEEMD params |
| `src/tools/ground-truth-creation/shiny-apps/README.md` | **MEDIUM** | How to run apps |
| `src/tools/ground-truth-creation/supporting/README.md` | **MEDIUM** | Utilities |
| `src/tools/docs/WORKFLOW.md` | **MEDIUM** | Clean markdown workflow |
| `src/tools/DOCKER.md` | **LOW** | Container instructions |

### Key Content for Each

#### `src/tools/README.md`
- Purpose of the tools
- Connection to the paper (ground truth creation)
- Subject counts (507 vs 208)
- Links to subdirectories
- Video demo reference

#### `src/tools/ground-truth-creation/README.md`
- 8-step workflow diagram
- Input/output formats
- Prerequisites (R packages, system deps)
- References (MissForest, CEEMD papers)

#### `imputation/README.md` (CRITICAL)
- MissForest parameters used:
  - `maxiter`: 10 (default)
  - `ntree`: 100 (default)
  - `parallelize`: 'variables'
- OOB error values
- Human verification process

#### `denoising/README.md` (CRITICAL)
- CEEMD parameters used:
  - `trials`: 100
  - `nimf`: 10
  - `noise.amp`: sd(LOESS residuals)
- IMF classification scheme
- Performance notes (hht vs Rlibeemd)

#### `shiny-apps/README.md`
- How to run each app
- Interface description
- Screenshots reference
- Troubleshooting

---

## Part 3: Implementation Checklist

### Phase 1: File Copy & Tests ✅ COMPLETED
- [x] Create directory structure
- [x] Copy Shiny apps (inspect_outliers, inspect_EMD)
- [x] Copy imputation code (MissForest)
- [x] Copy denoising code (cEEMD)
- [x] Copy supporting utilities
- [x] Copy video demo
- [x] Create R syntax smoke tests

### Phase 2: Docker (NEW)
- [ ] Create `Dockerfile.shiny`
- [ ] Add shiny service to `docker-compose.yml`
- [ ] Update renv.lock with missing packages
- [ ] Fix rstudioapi path detection in server.R files
- [ ] Test Docker build and run
- [ ] Document Docker commands in README

### Phase 3: Documentation (NEW)
- [ ] Create `src/tools/README.md`
- [ ] Create `src/tools/ground-truth-creation/README.md`
- [ ] Create `imputation/README.md`
- [ ] Create `denoising/README.md`
- [ ] Create `shiny-apps/README.md`
- [ ] Create `supporting/README.md`
- [ ] Extract clean workflow from wiki HTML → WORKFLOW.md
- [ ] Add parameters-used.md with exact GT settings

### Phase 4: Integration
- [ ] Update main project README with tools section
- [ ] Update GitHub issue #8 with completion status
- [ ] Update methods.tex supplementary materials
- [ ] Add git LFS for video file (11.3 MB)

---

## Part 4: User Commands Reference

### Building & Running Shiny

```bash
# Build Shiny image
docker build -t foundation-plr-shiny -f Dockerfile.shiny .

# Run with Shiny Server (all apps)
docker-compose --profile shiny up -d
# Access: http://localhost:3838/inspect_outliers/
# Access: http://localhost:3838/inspect_EMD/

# Run single app (development)
docker run --rm -p 3838:3838 \
  -v $(pwd)/data:/project/data \
  foundation-plr-shiny \
  R -e "shiny::runApp('/srv/shiny-server/inspect_outliers', host='0.0.0.0', port=3838)"

# Stop
docker-compose --profile shiny down
```

### Running Tests

```bash
# R syntax tests
Rscript tests/test_legacy_r_tools/test_r_syntax.R "$(pwd)"

# All legacy tool tests (when numpy available)
uv run pytest tests/test_legacy_r_tools/ -v
```

---

## Part 5: References

### Academic Papers

1. **MissForest:** Waljee AK et al. (2013) "Comparison of imputation methods for missing laboratory data in medicine." BMJ Open. http://dx.doi.org/10.1136/bmjopen-2013-002847

2. **CEEMD/libeemd:** Luukko PJJ, Helske J, Räsänen E (2017) "Introducing libeemd: A program package for performing the ensemble empirical mode decomposition." arXiv:1707.00487

3. **PLR Data:** Najjar RP et al. (2023) "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma." Br J Ophthalmol.

### Original Repositories

- R-PLR: https://github.com/petteriTeikari/R-PLR
- deepPLR: https://github.com/petteriTeikari/deepPLR

---

## Appendix: Copied Files Inventory

### From R-PLR (R-PLR-master/)

| File | Target | Lines |
|------|--------|-------|
| `Apps_Shiny/inspect_outliers/ui.R` | `shiny-apps/inspect_outliers/` | ~100 |
| `Apps_Shiny/inspect_outliers/server.R` | `shiny-apps/inspect_outliers/` | ~800 |
| `Apps_Shiny/inspect_EMD/ui.R` | `shiny-apps/inspect_EMD/` | ~50 |
| `Apps_Shiny/inspect_EMD/server.R` | `shiny-apps/inspect_EMD/` | ~600 |
| `PLR_reconstruction/subfunctions/lowLevel_imputation_wrappers.R` | `imputation/` | ~300 |
| `PLR_reconstruction/batch_AnalyzeAndReImpute.R` | `imputation/` | ~200 |
| `PLR_reconstruction/subfunctions/lowLevel_decomposition_wrappers.R` | `denoising/` | ~400 |
| `PLR_reconstruction/subfunctions/lowLevel_denoising_wrappers.R` | `denoising/` | ~200 |
| `PLR_artifacts/subfunctions/changepoint_detection.R` | `supporting/` | ~100 |
| `PLR_analysis/subfunctions/compute_PLR_features.R` | `supporting/` | ~300 |

### From deepPLR (deepPLR-master/)

| File | Target | Lines |
|------|--------|-------|
| `PLR_dataAugmentation/R_CEEMD_augm/PLR_augmentation.R` | `supporting/` | ~200 |

### Media

| File | Size | Notes |
|------|------|-------|
| `inspect-outliers-demo-2018.mp4` | 11.3 MB | Consider git-lfs |
