# TRIPOD-Code Compliance Mapping

> **Status**: Speculative compliance mapping against the TRIPOD-Code *protocol* paper.
> The final TRIPOD-Code checklist has not yet been published. This document maps repository
> practices against the anticipated areas (A--H) described in the protocol.
>
> **Reference**: Pollard TJ, Sounack M, et al. (2026). "TRIPOD-Code: Reporting guideline
> for code sharing in studies developing or validating clinical prediction models --
> A protocol." *Diagnostic and Prognostic Research*, 10, 3.
> DOI: [10.1186/s41512-025-00217-4](https://doi.org/10.1186/s41512-025-00217-4)
>
> This document will be updated when the final TRIPOD-Code checklist is published.

---

## Area A: Code Availability

| Item | Evidence |
|------|----------|
| Public repository | [github.com/petteriTeikari/foundation_PLR](https://github.com/petteriTeikari/foundation_PLR) |
| License | MIT ([`LICENSE`](../LICENSE)) |
| Zenodo DOI | **TO BE CREATED** at manuscript submission |
| Tagged release | Repository will receive a versioned release at submission time |
| Persistent identifier | Zenodo concept DOI will provide a version-independent persistent identifier |

The repository has been public since development began. All source code, configuration,
and non-private data are committed to the `main` branch.

---

## Area B: Software Dependencies

All three language ecosystems used in the project have deterministic dependency locking.

| Ecosystem | Lock file | Manager | Key constraints |
|-----------|-----------|---------|-----------------|
| Python 3.11+ | [`pyproject.toml`](../pyproject.toml) + [`uv.lock`](../uv.lock) | [uv](https://docs.astral.sh/uv/) | `pip`/`conda` banned by project rules ([`.claude/rules/20-package-management.md`](../.claude/rules/20-package-management.md)) |
| R >= 4.4 | [`renv.lock`](../renv.lock) (~288K, full version pinning) | [renv](https://rstudio.github.io/renv/) | System R from CRAN; `conda install r-*` banned |
| Node.js 20 LTS | [`apps/visualization/package.json`](../apps/visualization/package.json) + [`.nvmrc`](../.nvmrc) | npm | Version pinned to Node 20 via `.nvmrc` |

### Docker images (sha256 digest-pinned)

All four Dockerfiles pin base images by sha256 digest to guarantee bitwise-identical
base layers regardless of tag mutation.

| Dockerfile | Base image | Purpose |
|------------|-----------|---------|
| [`Dockerfile`](../Dockerfile) | `python:3.11-slim-bookworm` + `rocker/tidyverse:4.5.2` | Full pipeline (Python + R) |
| [`Dockerfile.test`](../Dockerfile.test) | `python:3.11-slim-bookworm` | CI test runner |
| [`Dockerfile.r`](../Dockerfile.r) | `rocker/tidyverse:4.5.2` | R-only analyses |
| [`Dockerfile.shiny`](../Dockerfile.shiny) | `rocker/shiny:4.5.2` | Interactive Shiny dashboard |

Digest-pinning date: 2026-02-13. All images include uv installed from a digest-pinned
`ghcr.io/astral-sh/uv` image.

### Dependency locking strategy

- **Python**: `uv.lock` provides cross-platform, hash-verified resolution. `uv sync`
  recreates the exact environment. Adding dependencies via `uv add` updates both
  `pyproject.toml` and `uv.lock`.
- **R**: `renv.lock` records package versions, sources, and checksums.
  `renv::restore()` recreates the library.
- **Node.js**: `package-lock.json` provides deterministic installs via `npm ci`.

---

## Area C: License

| Item | File | Notes |
|------|------|-------|
| Software license | [`LICENSE`](../LICENSE) | MIT License (Copyright 2024--2026 Petteri Teikari) |
| Machine-readable citation | [`CITATION.cff`](../CITATION.cff) | CFF format for automated citation extraction |
| Zenodo metadata | [`.zenodo.json`](../.zenodo.json) | Metadata for Zenodo archival deposit |

The MIT license permits unrestricted reuse, modification, and redistribution, including
commercial use, with attribution.

---

## Area D: Code Structure and Modularity

### Architecture documentation

The repository includes a detailed architecture document ([`ARCHITECTURE.md`](../ARCHITECTURE.md),
~500 lines) with Mermaid diagrams describing the two-block pipeline design.

### Two-block architecture

```
Block 1: EXTRACTION              Block 2: ANALYSIS
MLflow runs --> DuckDB            DuckDB --> Figures / Tables / LaTeX
(all computation)                 (read-only visualization)
```

This separation is enforced by pre-commit hooks and CI tests that ban metric computation
imports (e.g., `sklearn.metrics`) from visualization code in `src/viz/`.

### Configuration

- [Hydra](https://hydra.cc/) configuration system with configs in [`configs/`](../configs/)
- Anti-hardcoding rules enforced by CI: no hex colors, literal paths, method names,
  or dimension constants in source code
- Directory structure enforced by project conventions and CI checks

### Module organization

| Directory | Responsibility |
|-----------|---------------|
| `src/outlier_detection/` | 11 outlier detection methods |
| `src/imputation/` | 8 imputation methods |
| `src/featurization/` | Handcrafted physiological feature extraction |
| `src/classification/` | Classifier training and evaluation |
| `src/stats/` | Statistical metrics and wrappers |
| `src/viz/` | Read-only visualization (DuckDB -> figures) |
| `src/orchestration/` | Prefect workflow orchestration |
| `src/data_io/` | Registry, DuckDB export, data I/O |
| `configs/` | Hydra configs, registry, visualization params |

---

## Area E: Testing

### Test suite

The repository contains 145 test files across 6 categories, comprising 2000+ individual
test cases.

| Category | Purpose | Example |
|----------|---------|---------|
| Unit tests | Individual function correctness | `tests/test_stats/`, `tests/test_data_io/` |
| Integration tests | Pipeline stage interactions | `tests/test_integration/` |
| Figure QA | Data provenance, rendering, accessibility | `tests/test_figure_qa/` |
| Guardrail tests | Computation decoupling, registry integrity | `tests/test_guardrails/` |
| Notebook tests | Quarto notebook execution | CI via `notebook-tests.yml` |
| Security analysis | Dependency vulnerability scanning | CI via `ciso-assistant-security.yml` |

See [`tests/README.md`](../tests/README.md) for the full test taxonomy.

### Continuous integration

Six GitHub Actions workflows provide a 4-tier CI structure:

| Workflow | File | Scope |
|----------|------|-------|
| Main CI | `ci.yml` | Lint, type-check, unit/integration tests |
| Config integrity | `config-integrity.yml` | Registry validation, config consistency |
| Docker build | `docker.yml` | Container build verification |
| Documentation | `deploy-docs.yml` | MkDocs build and deployment |
| Notebooks | `notebook-tests.yml` | Quarto notebook execution tests |
| Security | `ciso-assistant-security.yml` | Dependency and code security analysis |

### Guardrail tests

Critical invariants are enforced as automated tests:

- **Computation decoupling**: `src/viz/` modules must not import metric computation
  functions; they read pre-computed values from DuckDB only.
- **Registry integrity**: Outlier method count must equal 11, imputation method count
  must equal 8, classifier count must equal 5.
- **Figure data provenance**: Figures must not use synthetic data; data lineage must be
  traceable to MLflow runs.

---

## Area F: Reproducibility

### Data availability

| Dataset | Records | Access | Path |
|---------|---------|--------|------|
| Demo subjects (de-identified) | 8 | Public | [`configs/demo_subjects.yaml`](../configs/demo_subjects.yaml) |
| Synthetic data | 32 subjects | Public | [`data/synthetic/SYNTH_PLR_DEMO.db`](../data/synthetic/SYNTH_PLR_DEMO.db) |
| Full research dataset | 507 subjects | Private (patient data) | Local DuckDB, not in repository |

### 4-gate data isolation

The repository uses a 4-gate architecture to separate public and private data:

1. **Gate 1**: Raw patient data (never committed, local only)
2. **Gate 2**: MLflow experiment tracking (local `/home/petteri/mlruns/`)
3. **Gate 3**: Extracted metrics in DuckDB (de-identified, shareable)
4. **Gate 4**: Public figures and tables (committed to repository)

### Reproduction commands

| Command | Scope |
|---------|-------|
| `make reproduce` | Full pipeline from raw data (requires private data) |
| `make reproduce-from-checkpoint` | Analysis only, from DuckDB checkpoint |
| `make extract` | Extraction block only (MLflow -> DuckDB) |
| `make analyze` | Analysis block only (DuckDB -> figures) |

### Docker reproduction

For full environment reproduction without local dependency management:

```bash
docker build -t foundation-plr .
docker run foundation-plr make reproduce-from-checkpoint
```

### Data lineage

The complete data lineage is documented in [`ARCHITECTURE.md`](../ARCHITECTURE.md):

```
MLflow runs --> DuckDB (extraction) --> CSV/JSON (intermediate) --> Figures (output)
```

Each figure includes a companion JSON data file for numeric reproducibility, generated
by the `save_figure()` utility.

---

## Area G: Code Documentation by Pipeline Stage

| Stage | Code location | Entry point | Notes |
|-------|--------------|-------------|-------|
| Outlier Detection | `src/outlier_detection/` | 11 methods via [`configs/mlflow_registry/`](../configs/mlflow_registry/) | Foundation models + traditional methods |
| Imputation | `src/imputation/` | 8 methods via registry | CSDI, SAITS, linear interpolation, etc. |
| Feature Extraction | `src/featurization/` | Handcrafted PLR features | Fixed; not varied in experiments |
| Classification | `src/classification/` | CatBoost (fixed) | Classifier is held constant per research design |
| Evaluation | `src/stats/`, `src/viz/` | STRATOS-compliant across 5 metric domains | Discrimination, calibration, clinical utility, overall, distributions |
| Orchestration | `src/orchestration/` | Prefect flows (`extraction_flow.py`, `analysis_flow.py`) | Two-block pipeline coordination |
| Data I/O | `src/data_io/` | `registry.py`, `streaming_duckdb_export.py` | Single source of truth for valid methods |
| Configuration | `configs/` | Hydra YAML configs | Registry, visualization, demo subjects |

### Method registry

The [`configs/mlflow_registry/`](../configs/mlflow_registry/) directory serves as the
single source of truth for valid experimental parameters. The Python interface
([`src/data_io/registry.py`](../src/data_io/registry.py)) provides validation functions
that prevent use of orphan or invalid method names from MLflow.

### Evaluation metrics

Metrics follow the STRATOS Initiative reporting guidelines (Van Calster et al. 2024).
The metric registry ([`src/viz/metric_registry.py`](../src/viz/metric_registry.py))
defines six metric sets:

1. **STRATOS Core**: AUROC, calibration slope, O:E ratio, Brier score, Net Benefit
2. **Discrimination**: AUROC with bootstrap 95% CI
3. **Calibration**: Slope, intercept, O:E ratio, smoothed calibration curves
4. **Clinical Utility**: Net Benefit at 5/10/15/20% thresholds, DCA curves
5. **Outlier Detection**: Method-specific detection performance
6. **Imputation**: Reconstruction accuracy metrics

---

## Area H: Long-Term Archival

| Mechanism | Status | Details |
|-----------|--------|---------|
| Zenodo deposit | **Planned** for submission | Will provide concept DOI (version-independent) |
| GitHub releases | Will be created at submission | Tagged release matching manuscript version |
| `CITATION.cff` | Present | Machine-readable citation metadata |
| `.zenodo.json` | Present | Zenodo deposit metadata |
| Lock files | Present | `uv.lock`, `renv.lock`, `package-lock.json` for exact reproduction |
| Docker digests | Present | sha256-pinned base images in all 4 Dockerfiles |

The combination of Zenodo archival, digest-pinned Docker images, and deterministic lock
files across three ecosystems aims to ensure long-term reproducibility independent of
upstream package registry availability.

---

## Beyond TRIPOD-Code

The following practices go beyond the anticipated TRIPOD-Code requirements.

### 3-layer provenance tracking

1. **MLflow experiment tracking**: All training runs logged with parameters, metrics,
   and artifacts.
2. **DuckDB extraction**: Metrics materialized from MLflow into queryable tables with
   full parameter metadata.
3. **Figure JSON data**: Every figure includes a companion JSON file containing all
   plotted numeric values, enabling verification without re-running the pipeline.

### Frozen registry for publication

The method registry ([`configs/mlflow_registry/`](../configs/mlflow_registry/)) is frozen
at publication time. Registry integrity is verified by CI tests that assert exact method
counts (11 outlier, 8 imputation, 5 classifier). Any change to the registry breaks CI.

### Pre-commit hooks

10 pre-commit hooks enforce code quality standards on every commit, including:

- R hardcoding checks (no hex colors, no `ggsave()`)
- Computation decoupling enforcement (no metric imports in `src/viz/`)
- Ruff linting and formatting
- YAML/JSON validation

### AI-assisted development

This repository was developed with AI assistance (Claude Code). The AI contribution is
limited to code generation, test writing, documentation, and refactoring. AI was **not**
used for:

- Statistical analysis design or interpretation
- Clinical decision-making or threshold selection
- Data collection, labeling, or clinical annotation
- Selection of evaluation metrics (guided by STRATOS guidelines)

---

## Limitations

- The full research dataset (507 subjects) cannot be shared publicly due to patient
  privacy constraints. The synthetic dataset (32 subjects) and demo subject configuration
  (8 de-identified subjects) are provided for pipeline verification.
- `make reproduce` requires access to local MLflow data. External reproducibility is
  supported via `make reproduce-from-checkpoint` using the DuckDB checkpoint.
- R package versions are pinned via `renv.lock` but binary compatibility depends on
  the system R installation version.

---

*This document will be updated when the final TRIPOD-Code checklist is published.
Current mapping is against the protocol paper (Pollard, Sounack et al. 2026,
DOI: [10.1186/s41512-025-00217-4](https://doi.org/10.1186/s41512-025-00217-4)).*
