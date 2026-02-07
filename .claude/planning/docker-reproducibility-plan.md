# Docker Reproducibility Plan for Foundation PLR

## Purpose

This document provides context for implementing a Docker-based reproducible research environment for the Foundation PLR project. It captures the theoretical background, technical requirements, and implementation strategy to enable cold-start implementation sessions.

## Theoretical Background

### The Reproducibility Crisis (Baker 2016, Pimentel 2019)

From our literature review (`section-22-mlops-reproducability.tex`):

> "A large-scale study of Jupyter notebooks associated with published papers found that only 879 out of 22,578 (3.9%) Python notebooks could be fully reproduced" (Pimentel et al. 2019)

> "Of notebooks with declared dependencies, fewer than two-thirds could even have their dependencies installed successfully, and of those that ran, only a fraction produced identical results to the original."

**Key failure modes (from Figure fig-22-04):**
1. **Dependency conflicts (~33%)** - Package versions change, dependencies break
2. **Environment mismatches (~25%)** - OS differences, system libraries, GPU drivers
3. **Hidden configurations (~20%)** - Undocumented parameters, environment variables

### MLOps Maturity Levels (Kreuzberger 2023)

| Level | Description | Current State |
|-------|-------------|---------------|
| 0 | Ad hoc (Jupyter notebooks, no version control) | Most academic research |
| 1 | Pipeline automation (scripts with config) | **Foundation PLR is here** |
| 2 | CI/CD for ML (automated testing, experiment tracking) | **Target with Docker** |
| 3 | Continuous training (auto-retraining) | Future work |
| 4 | Full automation (drift detection, monitoring) | Clinical deployment |

### Core Reproducibility Stack

From the literature review (Figure fig-22-01-code-versioning):

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Code Versioning (Git)                                  │
│   - Cryptographic commit hashes prevent code drift              │
│   - Already implemented in Foundation PLR                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Dependency Lock Files                                  │
│   - Python: uv.lock (PEP 751 compliant, cross-platform)         │
│   - R: PINNED_VERSIONS in setup script                          │
│   - Node.js: package-lock.json                                  │
│   → ALREADY IMPLEMENTED in setup-dev-environment.sh             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Containerization (Docker)                              │
│   - Packages entire OS environment                              │
│   - CUDA runtime, system libraries, environment variables       │
│   - Reproducible images from Dockerfiles                        │
│   → THIS IS WHAT WE NEED TO IMPLEMENT                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Scientific References

From `vessops.bib` (to cite in documentation):

1. **Baker 2016** - "1,500 scientists lift the lid on reproducibility" (Nature) - Survey showing >70% of scientists failed to reproduce others' experiments
2. **Pimentel 2019** - Large-scale study of Jupyter notebook reproducibility (3.9% success rate)
3. **Mitchell 2019** - "Model Cards for Model Reporting" (FAT*) - Documentation standard for ML models
4. **Boettiger 2015** - "An introduction to Docker for reproducible research" - Foundational paper on containerization for science
5. **Nüst 2019** - "containerit: Generating Dockerfiles for reproducible research with R"
6. **Cook 2017** - "Docker for Data Science" (book)
7. **Sochat 2019** - Evaluation of Docker containers for scientific workloads

### "Explicit, Versioned Specification" Principle

From the setup script header comment:
> "The solution requires explicit, versioned specification of the entire computational environment."

This principle drives our approach:
- **Python**: `uv.lock` provides exact versions of all 275 packages
- **R**: `PINNED_VERSIONS` dictionary in setup script
- **Node.js**: `package-lock.json` with exact versions
- **System**: Docker image with exact base image tag

---

## Current Environment Components

### From `setup-dev-environment.sh` Output

#### Python Environment
- **Manager**: uv (Astral, 10-100x faster than pip)
- **Python**: 3.10.12 (system)
- **Packages**: 275 resolved in uv.lock
- **Virtual env**: `.venv/` in project root

#### R Environment
- **Version**: R 4.5.2 (from CRAN, not Ubuntu's old version)
- **Key packages** (pinned versions):
  - Hmisc 5.2-1
  - survival 3.7-0
  - MASS 7.3-61
  - mgcv 1.9-1
  - pROC 1.18.5
  - dcurves 0.5.0
  - pmcalibration 0.1.0
  - pminternal 0.1.0 (critical for STRATOS compliance)

#### Node.js Environment
- **Manager**: nvm
- **Node.js**: v20.20.0 LTS
- **npm**: 10.8.2
- **Packages**: 246 in `apps/visualization/node_modules/`

#### Development Tools
- ruff 0.14.13 (Python linter)
- pre-commit 4.5.1
- just 1.46.0 (command runner)
- Docker 29.1.3

---

## Docker Implementation Strategy

### Multi-Stage Build Approach

```dockerfile
# Stage 1: Base system with R and system dependencies
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04 AS base

# Stage 2: Python environment
FROM base AS python-env
# Install uv, create venv, sync dependencies

# Stage 3: R environment
FROM python-env AS r-env
# Install R 4.5.2 from CRAN with pinned packages

# Stage 4: Node.js environment
FROM r-env AS node-env
# Install nvm, Node.js 20 LTS, npm packages

# Stage 5: Final research environment
FROM node-env AS research
# Copy project code, set up working directory
```

### Critical Components to Include

1. **Base Image**: `nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04`
   - Ensures GPU compatibility for deep learning models
   - Matches CUDA version used in experiments

2. **System Dependencies** (from setup script):
   ```
   build-essential curl wget git ca-certificates gnupg
   libssl-dev libffi-dev python3-dev
   libcurl4-openssl-dev libxml2-dev (for R packages)
   libsodium-dev (for Hmisc sodium dependency)
   ```

3. **R Installation** (CRAN method):
   - Add CRAN GPG key
   - Add CRAN repository for Ubuntu
   - Pin CRAN packages over Ubuntu's
   - Install R 4.5.2

4. **Version Pinning Strategy**:
   ```
   Python: COPY uv.lock . && uv sync --frozen
   R: Use remotes::install_version() with exact versions
   Node.js: COPY package-lock.json . && npm ci
   ```

### Proposed Dockerfile Structure

```dockerfile
# =============================================================================
# Foundation PLR - Reproducible Research Environment
# =============================================================================
# Based on: section-22-mlops-reproducability.tex
# "The solution requires explicit, versioned specification of the entire
#  computational environment." (Van Calster et al. 2024, STRATOS)
# =============================================================================

ARG CUDA_VERSION=12.4.0
ARG CUDNN_VERSION=9
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10
ARG NODE_VERSION=20
ARG R_VERSION=4.5.2

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ... (full implementation in actual Dockerfile)
```

---

## Files to Create

### 1. `Dockerfile` (project root)
Main development environment image.

### 2. `docker-compose.yml`
Services:
- `foundation-plr`: Main research container
- `mlflow`: Experiment tracking server (optional)
- Volumes for data persistence

### 3. `.dockerignore`
Exclude:
- `.git/`
- `mlruns/` (too large, use volume mount)
- `*.db` (databases, mount as volumes)
- `.venv/` (recreated in container)
- `node_modules/` (recreated in container)

### 4. `docker/` Directory
- `Dockerfile.dev` - Development environment
- `Dockerfile.ci` - CI/CD lightweight image
- `scripts/entrypoint.sh` - Container entry point

### 5. `justfile` Additions
```just
# Docker commands
docker-build:
    docker build -t foundation-plr:latest .

docker-run:
    docker run -it --gpus all -v $(pwd):/workspace foundation-plr:latest

docker-shell:
    docker run -it --gpus all -v $(pwd):/workspace foundation-plr:latest /bin/bash
```

---

## Acceptance Criteria

### Must Have
- [ ] `docker build` completes successfully
- [ ] All Python dependencies installed (verify with `uv sync`)
- [ ] All R packages installed (verify pminternal loads)
- [ ] All Node.js dependencies installed (verify npm ci)
- [ ] GPU accessible in container (verify `nvidia-smi`)
- [ ] MLflow experiments can be run inside container
- [ ] Can reproduce AUROC results from paper

### Should Have
- [ ] Multi-stage build for smaller final image
- [ ] docker-compose for easy startup
- [ ] Pre-built image on GitHub Container Registry (GHCR)
- [ ] CI workflow to build and test image

### Nice to Have
- [ ] Singularity/Apptainer recipe for HPC clusters
- [ ] DevContainer configuration for VS Code
- [ ] GPU-less variant for CI testing

---

## Implementation Checklist

### Phase 1: Basic Dockerfile
1. [ ] Create Dockerfile with system dependencies
2. [ ] Add Python environment (uv)
3. [ ] Add R environment (CRAN)
4. [ ] Add Node.js environment (nvm)
5. [ ] Test build locally

### Phase 2: Version Pinning
1. [ ] Copy and use uv.lock
2. [ ] Implement R package version pinning
3. [ ] Use npm ci for exact Node.js versions
4. [ ] Document exact base image digest

### Phase 3: Integration
1. [ ] Create docker-compose.yml
2. [ ] Add volume mounts for data
3. [ ] Configure GPU passthrough
4. [ ] Add healthchecks

### Phase 4: CI/CD
1. [ ] GitHub Actions workflow for build
2. [ ] Push to GHCR
3. [ ] Add reproducibility test job
4. [ ] Document in README

---

## References

### Project Documentation
- `CLAUDE.md` - Project context and rules
- `scripts/setup-dev-environment.sh` - Current setup script
- `uv.lock` - Python dependency lock file

### Scientific Literature
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/appendix-literature-review/section-22-mlops-reproducability.tex`
- `/home/petteri/Dropbox/manuscriptDrafts/vesselMLOps/vessops.bib`

### External Documentation
- [uv documentation](https://docs.astral.sh/uv/)
- [Docker best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
