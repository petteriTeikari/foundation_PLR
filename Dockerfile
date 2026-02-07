# Dockerfile - Full Development Environment
#
# Multi-stage build combining:
# - Python 3.11 with uv package manager
# - R 4.4 with renv for reproducible packages
# - Node.js 20 LTS for visualization app
#
# Build: docker build -t foundation-plr .
# Run:   docker run --rm -it foundation-plr
# Test:  docker run --rm foundation-plr python -c "import duckdb; print('OK')"
#
# For scientific reproducibility, this image:
# 1. Uses pinned base images with specific versions
# 2. Uses uv for Python, renv for R, npm for Node.js
# 3. Includes all STRATOS-compliant statistical packages
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Python environment builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS python-builder

# Install build dependencies for Python packages requiring compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Pin uv version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /project

# Copy Python dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv /opt/venv && \
    uv sync --frozen --no-dev

# Copy Python itself for the final image
RUN cp -r /usr/local/lib/python3.11 /opt/python-lib && \
    cp /usr/local/bin/python3.11 /opt/python-bin

# -----------------------------------------------------------------------------
# Stage 2: Final combined image (based on rocker for R 4.4)
# -----------------------------------------------------------------------------
FROM rocker/tidyverse:4.5.2 AS final

LABEL maintainer="Foundation PLR Team"
LABEL description="Full development environment for Foundation PLR (Python + R + Node.js)"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/petteri-lab/foundation_plr"

# =============================================================================
# System dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11 (from deadsnakes PPA for Ubuntu)
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    # Additional tools
    libgit2-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python setup
# =============================================================================
# Copy uv for package management
# Pin uv version for reproducibility
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# Copy the pre-built Python virtual environment
COPY --from=python-builder /opt/venv /opt/venv

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Fix venv symlinks to point to this image's Python
RUN rm /opt/venv/bin/python /opt/venv/bin/python3 /opt/venv/bin/python3.11 && \
    ln -s /usr/bin/python3.11 /opt/venv/bin/python && \
    ln -s python /opt/venv/bin/python3 && \
    ln -s python /opt/venv/bin/python3.11

# Set Python to use the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# =============================================================================
# R setup - renv already installed in rocker image
# =============================================================================
ENV RENV_VERSION=1.1.6
RUN R -e "install.packages('remotes', repos = 'https://cloud.r-project.org'); remotes::install_version('renv', version = '1.1.6', repos = 'https://cloud.r-project.org')"

# =============================================================================
# Working directory
# =============================================================================
WORKDIR /project

# =============================================================================
# R packages
# =============================================================================
COPY renv.lock renv.lock
COPY .Rprofile .Rprofile
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json

# Restore R packages from lockfile
RUN R -e "options(warn = 1); renv::restore(prompt = FALSE)"

# =============================================================================
# Python project files
# =============================================================================
COPY pyproject.toml uv.lock ./

# =============================================================================
# Node.js packages (visualization app)
# =============================================================================
COPY apps/visualization/package*.json apps/visualization/
# Install Node.js packages for visualization app
RUN cd apps/visualization && npm ci --omit=dev

# =============================================================================
# Application code
# =============================================================================
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY Makefile ./
COPY apps/ apps/

# =============================================================================
# Output directories
# =============================================================================
RUN mkdir -p figures/generated/ggplot2 \
    && mkdir -p figures/generated/matplotlib \
    && mkdir -p outputs/r_data \
    && mkdir -p outputs/tables \
    && mkdir -p outputs/latex

# =============================================================================
# Environment configuration
# =============================================================================
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV R_LIBS_USER=/project/renv/library
ENV PYTHONPATH=/project/src
ENV PREFECT_DISABLED=1

# =============================================================================
# Health check
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD bash -c "python -c 'import duckdb' && R -e 'library(ggplot2)' && node --version"

# =============================================================================
# Default command
# =============================================================================
CMD ["bash", "-c", "echo '=== Foundation PLR Development Environment ===' && python --version && R --version | head -1 && node --version && echo '=== Ready ===' && bash"]
