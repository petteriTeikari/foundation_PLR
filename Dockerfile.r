# Dockerfile.r - R environment for figure generation
#
# This Dockerfile creates a reproducible R environment for generating
# publication-quality figures using renv for package management.
#
# Build: docker build -t foundation-plr-r -f Dockerfile.r .
# Run:   docker run --rm -v $(pwd)/figures/generated/ggplot2:/project/figures/generated/ggplot2 foundation-plr-r
# Test:  docker run --rm foundation-plr-r Rscript -e "library(ggplot2); library(pminternal); cat('SUCCESS\n')"
#
# For scientific reproducibility, this image:
# 1. Uses rocker/tidyverse with pinned R version (4.5.2)
# 2. Restores exact package versions from renv.lock
# 3. Includes all STRATOS-compliant statistical packages
#
# =============================================================================

# Use rocker/tidyverse which has many packages pre-installed
# This includes ggplot2, dplyr, tidyr, etc. and all system dependencies
# Pinned 2026-02-13 for reproducibility (TRIPOD-Code area B)
FROM rocker/tidyverse:4.5.2@sha256:17dca9381149911b201184ab46e6c8628b68ddc1386b9562bc26ca8b6b4c6f81

LABEL maintainer="Foundation PLR Team"
LABEL description="R environment for ggplot2 figure generation with STRATOS compliance"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/petteri-lab/foundation_plr"

# =============================================================================
# Additional system dependencies
# =============================================================================
# rocker/tidyverse already has most dependencies, add any missing ones
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgit2-dev \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# renv setup
# =============================================================================
# Install renv for reproducible package management
ENV RENV_VERSION=1.1.6
RUN R -e "install.packages('remotes', repos = 'https://cloud.r-project.org'); remotes::install_version('renv', version = '1.1.6', repos = 'https://cloud.r-project.org')"

# Set working directory
WORKDIR /project

# =============================================================================
# Package installation (layer caching optimization)
# =============================================================================
# Copy renv files FIRST for Docker layer caching
# These change less frequently than source code, so Docker can cache this layer
COPY renv.lock renv.lock
COPY .Rprofile .Rprofile
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json

# Restore R packages from lockfile
# Note: We don't use clean=TRUE to avoid issues with recommended packages
# This step is cached unless renv.lock changes
RUN R -e "options(warn = 1); renv::restore(prompt = FALSE)"

# =============================================================================
# Application code
# =============================================================================
# Copy R source code
COPY src/r/ src/r/

# Copy configs needed by R scripts
COPY configs/ configs/

# =============================================================================
# Output directories
# =============================================================================
# Create directories for figure output
RUN mkdir -p figures/generated/ggplot2 \
    && mkdir -p outputs/r_data \
    && mkdir -p outputs/tables

# =============================================================================
# Environment configuration
# =============================================================================
# Set locale for proper font/character rendering
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Ensure R uses project library
ENV R_LIBS_USER=/project/renv/library

# =============================================================================
# Default command
# =============================================================================
# Default: run setup.R to verify environment
# Override with specific script as needed
CMD ["Rscript", "-e", "source('src/r/setup.R'); cat('R environment ready\\n')"]
