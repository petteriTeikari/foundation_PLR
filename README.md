# Foundation PLR

[![CI](https://github.com/petteriTeikari/foundation_PLR/actions/workflows/ci.yml/badge.svg)](https://github.com/petteriTeikari/foundation_PLR/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![R 4.4+](https://img.shields.io/badge/R-4.4+-276DC3.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://github.com/astral-sh/uv)

**Test Suite:**
[![Unit Tests](https://img.shields.io/badge/unit-passing-brightgreen)](#test-architecture)
[![Guardrails](https://img.shields.io/badge/guardrails-passing-brightgreen)](#test-architecture)
[![Integration](https://img.shields.io/badge/integration-passing-brightgreen)](#test-architecture)
[![E2E](https://img.shields.io/badge/e2e-passing-brightgreen)](#test-architecture)
[![Data](https://img.shields.io/badge/data-local%20only-blue)](#test-architecture)
[![R Required](https://img.shields.io/badge/R%20tests-local%20only-blue)](#test-architecture)

> **Research Question**: How do preprocessing choices (outlier detection → imputation) affect downstream classification performance for glaucoma screening using handcrafted physiological features?

## 30-Second Summary

This repository evaluates whether **foundation models** (MOMENT, UniTS, TimesNet) can replace traditional methods (LOF, SVM, linear interpolation) for **PLR biosignal preprocessing** in glaucoma screening.

**Key Findings:**
- Best AUROC: **0.913** (ground truth preprocessing + CatBoost)
- Handcrafted features outperform FM embeddings by **9 percentage points**
- Foundation models are **competitive** for outlier detection and imputation
- Preprocessing choice matters (η² = 0.15) but less than classifier choice (which is obvious if you compare Catboost to Logistic Regression)

**Data**: 507 subjects from Najjar et al. 2023 (Br J Ophthalmol), 208 with classification labels.

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| **Understand the pipeline** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Set up development environment** | [Quick Start](#quick-start-development-environment) |
| **Run the experiment** | [Running the Pipeline](#running-the-pipeline) |
| **Understand the code** | [src/README.md](src/README.md) |
| **See configuration options** | [configs/README.md](configs/README.md) |
| **Generate figures** | [figures/README.md](figures/README.md) |
| **Contribute** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Work with Claude Code** | [.claude/README.md](.claude/README.md) |

### By Role

| Role | Start Here |
|------|------------|
| **New contributor** | [CONTRIBUTING.md](CONTRIBUTING.md) → [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Researcher** | [ARCHITECTURE.md](ARCHITECTURE.md) → [src/stats/README.md](src/stats/README.md) |
| **Data scientist** | [src/README.md](src/README.md) → [configs/README.md](configs/README.md) |
| **Visualization** | [figures/README.md](figures/README.md) → [apps/visualization/README.md](apps/visualization/README.md) |

## Quick Start with Docker (Recommended)

The fastest way to get a reproducible environment:

```bash
# Build the full development image (Python + R + Node.js)
make docker-build

# Run tests in Docker
make docker-test

# Enter interactive shell
make docker-shell

# Generate R figures in Docker
make r-docker-run
```

### Docker Compose (Development)

```bash
# Start development environment
docker-compose up -d dev

# Enter container
docker-compose exec dev bash

# Generate R figures
docker-compose run r-figures

# Run Python tests
docker-compose run test
```

### Pre-built Images

Pre-built images are available from GitHub Container Registry:

```bash
# Full development environment (Python + R + Node.js)
docker pull ghcr.io/petteri-lab/foundation_plr:latest

# R-only for figures
docker pull ghcr.io/petteri-lab/foundation_plr-r:latest
```

See [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) for details.

---

## Quick Start: Local Development Environment

If you prefer local installation over Docker:

```bash
./scripts/infra/setup-dev-environment.sh
```

The script auto-detects your OS (Ubuntu/Debian, macOS, Windows via Git Bash) and installs everything needed.

**Note! Only been tested for Ubuntu 22.04 at the moment**

### Manual Setup (if needed)

1. **Install [uv](https://astral.sh/blog/uv)** (package manager):
   ```bash
   pip install uv
   ```

2. **Create virtual environment**:
   ```bash
   uv venv --python 3.11
   uv sync
   ```

3. **Activate environment**:
   ```bash
   source .venv/bin/activate
   ```

## Running the Pipeline

### 1. Start the Prefect server

```bash
prefect server start
```

### 2. Run the experiment

```bash
python src/pipeline_PLR.py
```

This uses the default Hydra config from [`configs/defaults.yaml`](configs/defaults.yaml).

### 3. View results

**MLflow UI** (experiment tracking):
```bash
export MLFLOW_TRACKING_URI='file:////home/petteri/Dropbox/mlruns'
mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

**Prefect UI** (workflow monitoring):
```bash
# Open http://127.0.0.1:4200/dashboard
```

## Project Structure

```
foundation_PLR/
├── src/                    # Source code (pipeline stages)
│   ├── anomaly_detection/  # Stage 1: 11 outlier methods
│   ├── imputation/         # Stage 2: 8 imputation methods
│   ├── featurization/      # Stage 3: Feature extraction
│   ├── classification/     # Stage 4: CatBoost + evaluation
│   ├── stats/              # STRATOS-compliant metrics
│   ├── viz/                # Figure generation
│   ├── data_io/            # Data import/export
│   └── log_helpers/        # MLflow integration
├── configs/                # Hydra configuration
├── tests/                  # Test suite
├── apps/visualization/     # React + D3.js interactive figures
├── figures/                # Generated figures + JSON data
├── notebooks/              # Jupyter tutorials
├── scripts/                # Utility scripts
├── docs/                   # Additional documentation
├── assets/                 # Images and static files
├── outputs/                # Generated outputs
└── .claude/                # AI assistant context
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed pipeline documentation.

## The Pipeline

```
Raw PLR Signal → Outlier Detection → Imputation → Featurization → Classification → STRATOS Metrics
                  (gt+8 methods+ensembles)       (gt+5 methods+ensembles)  (handcrafted+Moment)    (CatBoost + 4 alternative classifiers)
```

**NOT about comparing classifiers** — we fix CatBoost and vary preprocessing.

### Available Methods

| Stage | Methods |
|-------|---------|
| **Outlier Detection** | pupil-gt, MOMENT-gt-finetune, MOMENT-gt-zeroshot, UniTS-gt-finetune, TimesNet-gt, LOF, OneClassSVM, SubPCA, PROPHET, ensembles |
| **Imputation** | pupil-gt, SAITS, CSDI, TimesNet, MOMENT-finetune, MOMENT-zeroshot, ensemble |
| **Classification** | CatBoost (primary), XGBoost, TabPFN, TabM, LogisticRegression |

### Output Metrics ([STRATOS](http://arxiv.org/abs/2412.10288)-compliant)

| Category | Metrics |
|----------|---------|
| **Discrimination** | AUROC with 95% CI |
| **Calibration** | Slope, intercept, O:E ratio, Brier |
| **Clinical Utility** | Net Benefit, Decision Curve Analysis |

## Configuration

All configuration uses [Hydra](https://hydra.cc/):

```bash
# Default run
python src/pipeline_PLR.py

# Custom config
python src/pipeline_PLR.py --config-name=hyperparam_sweep

# Override parameters
python src/pipeline_PLR.py PREFECT.PROCESS_FLOWS.CLASSIFICATION=true
```

See [configs/README.md](configs/README.md) for configuration details.

## Synthetic Data Isolation

This project implements a **4-gate isolation architecture** to prevent synthetic data from contaminating production artifacts. This is critical for research integrity.

### Why Isolation Matters

Running the pipeline with synthetic data without proper isolation could:
- Mix synthetic runs with real experiments in MLflow
- Include synthetic runs in production DuckDB databases
- Overwrite real figures with synthetic visualizations

### The 4-Gate Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GATE 0: Data Mode Detection (src/utils/data_mode.py)       │
│  • Env var: FOUNDATION_PLR_SYNTHETIC=1                      │
│  • Config: EXPERIMENT.is_synthetic=true                     │
│  • Filename: contains "SYNTH_" or "synthetic"               │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
   SYNTHETIC MODE                          PRODUCTION MODE
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ GATE 1: MLflow      │                 │ GATE 1: MLflow      │
│ • Prefix: __SYNTH_  │                 │ • No prefix         │
│ • Tag: is_synth=true│                 │ • Tag: is_synth=false│
│ • Exp: synth_PLR_*  │                 │ • Exp: PLR_*        │
└─────────────────────┘                 └─────────────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ GATE 2: Extraction  │                 │ GATE 2: Extraction  │
│ • REJECT if synth   │                 │ • REJECT if synth   │
│ • Separate script   │                 │ • Standard script   │
└─────────────────────┘                 └─────────────────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ GATE 3: Outputs     │                 │ GATE 3: Outputs     │
│ outputs/synthetic/  │                 │ outputs/            │
│ figures/synthetic/  │                 │ figures/generated/  │
└─────────────────────┘                 └─────────────────────┘
```

### Running Synthetic Experiments

```bash
# Method 1: Environment variable
FOUNDATION_PLR_SYNTHETIC=1 python src/pipeline_PLR.py

# Method 2: Use synthetic config
python src/pipeline_PLR.py --config-name=synthetic_run

# Method 3: Override in config
python src/pipeline_PLR.py +EXPERIMENT.is_synthetic=true
```

### Extracting Synthetic Results

```bash
# Extract ONLY synthetic runs to separate database
python scripts/extraction/extract_synthetic_to_duckdb.py
# Output: outputs/synthetic/synthetic_foundation_plr_results.db

# Production extraction REJECTS synthetic runs automatically
python scripts/extraction/extract_all_configs_to_duckdb.py
```

### Key Module: `src/utils/data_mode.py`

Central detection and validation logic (604 lines, fully documented):

| Function | Purpose |
|----------|---------|
| `is_synthetic_mode()` | Check if env var sets synthetic mode |
| `is_synthetic_from_config(cfg)` | Check if Hydra config indicates synthetic |
| `get_data_mode(cfg=None)` | Returns "synthetic" or "production" |
| `validate_run_for_production_extraction()` | Rejects synthetic runs |
| `get_results_db_path_for_mode(synthetic)` | Returns correct output path |
| `get_figures_dir_for_mode(synthetic)` | Routes figures correctly |

### Configuration: `configs/data_isolation.yaml`

Defines all isolation parameters including:
- Detection triggers (env vars, config keys, filename patterns)
- MLflow prefixes and tags
- Extraction rejection criteria
- Output directory paths
- JSON metadata fields for synthetic data

### Pre-commit Hooks

Two hooks ensure isolation integrity:

```yaml
# .pre-commit-config.yaml
- id: extraction-isolation-check   # Validates extraction scripts
- id: figure-isolation-check       # Validates figure routing
```

### Test Coverage

| Test Suite | Coverage |
|------------|----------|
| `tests/unit/test_data_mode.py` | 31 unit tests for detection logic |
| `tests/integration/test_mlflow_isolation.py` | 16 tests for MLflow prefixing |
| `tests/integration/test_extraction_isolation.py` | 15 tests for DuckDB extraction |
| `tests/integration/test_figure_routing.py` | 13 tests for figure output routing |
| `tests/e2e/test_synthetic_isolation_e2e.py` | 18 end-to-end isolation tests |

Run all isolation tests:
```bash
pytest tests/unit/test_data_mode.py tests/integration/test_*isolation*.py tests/e2e/test_synthetic*.py -v
```

## Test Architecture

The test suite is organized into **tiers** matching the CI pipeline. Each tier has a dedicated pytest marker:

| Tier | Marker | What It Tests | CI | Local |
|------|--------|---------------|:---:|:-----:|
| **0** | — | Lint (ruff check + format) | Y | Y |
| **1** | `unit` | Pure function tests, no external deps | Y | Y |
| **1** | `guardrail` | Code quality scans (hardcoding, decoupling) | Y | Y |
| **2** | — | Quality gates (registry, decoupling scripts) | Y | Y |
| **3** | `integration` | Integration with demo/synthetic data | Y | Y |
| **3** | `e2e` | Full pipeline end-to-end | Y | Y |
| **Local** | `data` | Tests requiring `data/public/` or `data/r_data/` | — | Y |
| **Local** | `r_required` | Tests requiring R/Rscript | — | Y |

```bash
# Run by tier
make test-fast          # Tier 1: unit + guardrail (parallel, ~2 min)
make test-integration   # Tier 3: integration + e2e

# Run by marker
pytest -m unit          # Only unit tests
pytest -m guardrail     # Only code quality guardrails
pytest -m data          # Data-dependent tests (local only)
pytest -m r_required    # R environment tests (local only)

# Full local suite
PREFECT_DISABLED=1 MPLBACKEND=Agg pytest tests/ \
  --ignore=tests/test_docker_r.py --ignore=tests/test_docker_full.py
```

## Documentation Index

### Architecture & API

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Pipeline overview, error propagation, module entry points |
| [docs/API_ARCHITECTURE.md](docs/API_ARCHITECTURE.md) | **Full UML/Mermaid diagrams**, class hierarchies, sequence diagrams |
| [.claude/KNOWLEDGE_GRAPH.md](.claude/KNOWLEDGE_GRAPH.md) | **Knowledge graph** for Graph RAG retrieval |

### Development Guides

| Document | Description |
|----------|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, code style, PR process |
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions (research focus, rules) |
| [src/README.md](src/README.md) | Source code organization by pipeline stage |
| [configs/README.md](configs/README.md) | Hydra configuration system |
| [tests/README.md](tests/README.md) | Testing guide, Docker policy |

### Output & Visualization

| Document | Description |
|----------|-------------|
| [figures/README.md](figures/README.md) | Figure generation, JSON data, naming |
| [Figure Catalogue](docs/repo-figures/figure-alt-text-catalog.md) | Complete index of all ~130 repository figures with descriptions |
| [apps/visualization/README.md](apps/visualization/README.md) | React + D3.js interactive figures |
| [.claude/README.md](.claude/README.md) | Claude Code context system |

### Module Documentation

Each source module has its own README with function index:

| Module | Description | Key Functions |
|--------|-------------|---------------|
| [src/anomaly_detection/](src/anomaly_detection/README.md) | 11 outlier detection methods | `detect_outliers()`, `run_lof()` |
| [src/imputation/](src/imputation/README.md) | 8 imputation methods | `impute_signal()`, `linear_interpolate()` |
| [src/featurization/](src/featurization/README.md) | Feature extraction | `extract_amplitude_bins()` |
| [src/classification/](src/classification/README.md) | CatBoost + evaluation | `bootstrap_evaluate()` |
| [src/stats/](src/stats/README.md) | STRATOS metrics | `calibration_slope()`, `net_benefit()` |
| [src/viz/](src/viz/README.md) | Figure generation | `setup_style()`, `save_figure()` |
| [src/data_io/](src/data_io/README.md) | Database operations | `load_plr_database()` |
| [src/log_helpers/](src/log_helpers/README.md) | MLflow integration | `init_mlflow()`, `log_metrics()` |
| [src/utils/](src/utils/) | Utilities including data mode | `is_synthetic_mode()`, `get_data_mode()` |

## Reproducibility

This project uses a **dual lockfile approach** for reproducibility:

| Language | Lockfile | Tool | Restore Command |
|----------|----------|------|-----------------|
| **Python** | `uv.lock` | [uv](https://github.com/astral-sh/uv) | `uv sync` |
| **R** | `renv.lock` | [renv](https://rstudio.github.io/renv/) | `renv::restore()` |

### R Environment

The R codebase (17 figure scripts in `src/r/`) uses:

- **renv** for package locking (113 packages)
- **Docker** for full isolation (`Dockerfile.r`)
- **Pre-commit hook** for sync verification

```bash
# Generate R figures locally
make r-figures-all

# Generate R figures in Docker (recommended for reproducibility)
make r-docker-run

# Check renv.lock is in sync
Rscript scripts/infra/check_renv_sync.R
```

### Future: R4R for Large-Scale R Projects

If the R codebase expands significantly (100+ scripts), consider adopting [R4R](https://doi.org/10.1145/3736731.3746156) (Donat-Bouillud et al. 2025, ACM REP '25) for automated artifact generation:

> R4R achieves **97.5% reproducibility** on R notebooks through dynamic dependency tracing and automatic Docker/renv.lock generation.

**When to adopt R4R:**
- R codebase grows beyond ~50 scripts
- Multiple contributors adding R packages independently
- Need for runtime dependency tracing (catching dynamically loaded packages)

**Current approach is sufficient** for our 17-script R codebase with controlled package additions.

See: [R Code Reproducibility Analysis](manuscripts/foundationPLR/planning/r-code-reproducibility-analysis.md) for details.

## Data Provenance

**Source**: Najjar RP, et al. "Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma." Br J Ophthalmol 2023;107:663–670.

| Dataset | N | AUROC | Notes |
|---------|---|-------|-------|
| Najjar original | 322 | 0.94 | Full Singapore dataset |
| Our subset (classification) | 208 | 0.913 | 152 control + 56 glaucoma |
| Our subset (preprocessing) | 507 | N/A | All with ground truth masks |

## License

[MIT](LICENSE)

## Citation

If you use this code, please cite:

```
TO-COME-?
```

And the original data source:

```bibtex
@article{najjar2023pupillometry,
  title={Handheld chromatic pupillometry can accurately and rapidly reveal functional loss in glaucoma},
  author={Najjar, Raymond P and others},
  journal={British Journal of Ophthalmology},
  volume={107},
  pages={663--670},
  year={2023}
}
```
