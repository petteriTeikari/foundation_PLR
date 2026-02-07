# Knowledge Graph: Foundation PLR Documentation

> **Navigation Index**: This page maps concepts to their explanations, implementations, and visualizations - Obsidian-style.

---

## How to Use This Graph

| You want to... | Start here |
|----------------|------------|
| **Understand a metric** | [Concept Index](#concept-index) → Tutorial → Code |
| **Find code for X** | [Code Index](#code-index) |
| **Interpret a figure** | [Figure Index](#figure-index) |
| **Learn the pipeline** | [Learning Paths](#learning-paths) |

---

## Concept Index

| Concept | Tutorial (L1) | Code (L3) | Key Figures |
|---------|---------------|-----------|-------------|
| **STRATOS Framework** | [stratos-metrics.md](tutorials/stratos-metrics.md) | [`src/stats/`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/README.md) | fig-repo-28 |
| **Discrimination (AUROC)** | [stratos-metrics.md#discrimination](tutorials/stratos-metrics.md#1-discrimination) | [`classifier_metrics.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/classifier_metrics.py) | ROC curves |
| **Calibration** | [stratos-metrics.md#calibration](tutorials/stratos-metrics.md#2-calibration) | [`calibration_extended.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/calibration_extended.py) | fig-repo-39 |
| **Net Benefit / DCA** | [stratos-metrics.md#clinical-utility](tutorials/stratos-metrics.md#5-clinical-utility) | [`clinical_utility.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/clinical_utility.py) | fig-repo-40 |
| **Brier Score** | [stratos-metrics.md#overall-performance](tutorials/stratos-metrics.md#3-overall-performance) | [`scaled_brier.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/scaled_brier.py) | — |
| **Prediction Instability** | [reading-plots.md#instability](tutorials/reading-plots.md#instability-plots) | [`pminternal_wrapper.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/pminternal_wrapper.py) | fig-repo-27d |
| **AURC / Uncertainty** | [reading-plots.md#risk-coverage](tutorials/reading-plots.md#risk-coverage-aurc) | [`uncertainty_propagation.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/uncertainty_propagation.py) | fig-repo-27f |
| **CD Diagrams** | [reading-plots.md#cd-diagrams](tutorials/reading-plots.md#critical-difference-diagrams) | [`cd_diagram_preprocessing.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/viz/cd_diagram_preprocessing.py) | fig-repo-27 |
| **SHAP Values** | [reading-plots.md#shap](tutorials/reading-plots.md#shap-values) | [`generate_shap_figures.py`](https://github.com/petteriTeikari/foundation_PLR/blob/main/scripts/generate_shap_figures.py) | fig-repo-27e |
| **Reproducibility** | [reproducibility.md](tutorials/reproducibility.md) | [`Dockerfile`](https://github.com/petteriTeikari/foundation_PLR/blob/main/Dockerfile), [`uv.lock`](https://github.com/petteriTeikari/foundation_PLR/blob/main/uv.lock) | fig-repro-* |
| **UV / Polars / DuckDB** | [dependencies.md](tutorials/dependencies.md) | [`pyproject.toml`](https://github.com/petteriTeikari/foundation_PLR/blob/main/pyproject.toml) | fig-repo-14-16 |

---

## Code Index

### Statistics (`src/stats/`)

| Module | Implements | Tutorial Link |
|--------|------------|---------------|
| `classifier_metrics.py` | AUROC, sensitivity, specificity | [Discrimination](tutorials/stratos-metrics.md#1-discrimination) |
| `calibration_extended.py` | Slope, intercept, O:E ratio | [Calibration](tutorials/stratos-metrics.md#2-calibration) |
| `scaled_brier.py` | Brier, Scaled Brier (IPA) | [Overall Performance](tutorials/stratos-metrics.md#3-overall-performance) |
| `clinical_utility.py` | Net Benefit, DCA | [Clinical Utility](tutorials/stratos-metrics.md#5-clinical-utility) |
| `pminternal_wrapper.py` | R pminternal interop | [Instability Plots](tutorials/reading-plots.md#instability-plots) |
| `uncertainty_propagation.py` | AURC, selective classification | [Risk-Coverage](tutorials/reading-plots.md#risk-coverage-aurc) |

### Visualization (`src/viz/`)

| Module | Creates | Interpretation |
|--------|---------|----------------|
| `calibration_plot.py` | Smoothed calibration curves | [Calibration Plots](tutorials/reading-plots.md#calibration-plots) |
| `dca_plot.py` | Decision curve analysis | [DCA](tutorials/reading-plots.md#decision-curve-analysis-dca) |
| `cd_diagram_preprocessing.py` | CD diagrams | [CD Diagrams](tutorials/reading-plots.md#critical-difference-diagrams) |
| `fig_instability_plots.py` | Prediction instability | [Instability Plots](tutorials/reading-plots.md#instability-plots) |
| `prob_distribution.py` | Probability distributions | [Discrimination](tutorials/stratos-metrics.md#1-discrimination) |

### R Figures (`src/r/figures/`)

| Script | Purpose | Interpretation |
|--------|---------|----------------|
| `fig_calibration_stratos.R` | STRATOS calibration | [Calibration](tutorials/stratos-metrics.md#2-calibration) |
| `fig_cd_diagrams.R` | CD diagram generation | [CD Diagrams](tutorials/reading-plots.md#critical-difference-diagrams) |
| `fig_instability_pminternal.R` | pminternal plots | [Instability](tutorials/reading-plots.md#instability-plots) |

---

## Figure Index

### STRATOS Metrics Figures

| Figure ID | Concept | Location |
|-----------|---------|----------|
| fig-repo-28 | STRATOS overview | [docs/repo-figures/generated/](repo-figures/generated/) |
| fig-repo-39 | Calibration explained | [Calibration tutorial](tutorials/stratos-metrics.md#2-calibration) |
| fig-repo-40 | Net Benefit / DCA | [Clinical utility tutorial](tutorials/stratos-metrics.md#5-clinical-utility) |

### Plot Interpretation Figures

| Figure ID | Shows How to Read | Tutorial Section |
|-----------|-------------------|------------------|
| fig-repo-27 | CD diagrams | [CD Diagrams](tutorials/reading-plots.md#critical-difference-diagrams) |
| fig-repo-27b | Raincloud plots | [Raincloud](tutorials/reading-plots.md#raincloud-plots) |
| fig-repo-27c | Specification curves | [Spec Curve](tutorials/reading-plots.md#specification-curve-analysis) |
| fig-repo-27d | Instability plots | [Instability](tutorials/reading-plots.md#instability-plots) |
| fig-repo-27e | SHAP values | [SHAP](tutorials/reading-plots.md#shap-values) |
| fig-repo-27f | Risk-coverage | [AURC](tutorials/reading-plots.md#risk-coverage-aurc) |

### Reproducibility Figures

| Figure ID | Concept | Tutorial |
|-----------|---------|----------|
| fig-repro-01 | Crisis in numbers | [Reproducibility](tutorials/reproducibility.md) |
| fig-repro-14 | Lockfiles | [Dependencies](tutorials/dependencies.md) |
| fig-repro-20 | DuckDB single source | [Reproducibility](tutorials/reproducibility.md) |

---

## Learning Paths

### For Newcomers (30 min)

```
1. README.md
   └── "What is this project?"

2. docs/tutorials/stratos-metrics.md
   └── "Why AUROC isn't enough"

3. docs/tutorials/reading-plots.md
   └── "How to read these visualizations"
```

### For Developers (1 hour)

```
1. ARCHITECTURE.md
   └── Pipeline overview

2. src/stats/README.md
   └── Metric implementations

3. src/viz/README.md
   └── Figure generation

4. configs/VISUALIZATION/figure_registry.yaml
   └── Available figures
```

### For Reproducibility (45 min)

```
1. docs/tutorials/reproducibility.md
   └── Why this matters

2. docs/tutorials/dependencies.md
   └── UV, Polars, DuckDB

3. Makefile
   └── Available commands

4. make reproduce
   └── Run the pipeline
```

### For R Users (30 min)

```
1. src/r/README.md
   └── R figure system

2. src/r/figure_system/README.md
   └── API documentation

3. src/r/figures/_TEMPLATE.R
   └── Create new figures
```

---

## Cross-Reference Matrix

### Documentation ↔ Code

| Document | Primary Code | Config |
|----------|--------------|--------|
| `stratos-metrics.md` | `src/stats/*.py` | `configs/defaults.yaml` |
| `reading-plots.md` | `src/viz/*.py`, `src/r/figures/*.R` | `configs/VISUALIZATION/` |
| `reproducibility.md` | `Dockerfile`, `Makefile` | `pyproject.toml`, `renv.lock` |
| `dependencies.md` | `pyproject.toml` | — |

### Code → Documentation (Reverse Links)

| Code File | Explained In |
|-----------|--------------|
| `calibration_extended.py` | [src/stats/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/README.md#calibration), [stratos-metrics.md](tutorials/stratos-metrics.md#2-calibration) |
| `clinical_utility.py` | [src/stats/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/stats/README.md#clinical-utility), [stratos-metrics.md](tutorials/stratos-metrics.md#5-clinical-utility) |
| `pminternal_wrapper.py` | [src/r/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/r/README.md), [reading-plots.md](tutorials/reading-plots.md#instability-plots) |
| `calibration_plot.py` | [src/viz/README.md](https://github.com/petteriTeikari/foundation_PLR/blob/main/src/viz/README.md), [reading-plots.md](tutorials/reading-plots.md#calibration-plots) |

---

## Navigation Shortcuts

| Starting Point | To Find | Path |
|----------------|---------|------|
| Root README | Any concept | README → Tutorials → Specific topic |
| Any figure | Interpretation | Figure → reading-plots.md |
| Any metric | Implementation | Metric → src/stats/README.md |
| Any viz code | STRATOS context | Code → stratos-metrics.md |

---

## Key References

| Topic | Primary Paper | DOI |
|-------|---------------|-----|
| STRATOS Framework | Van Calster 2024 | [10.1007/s10654-024-01168-2](https://doi.org/10.1007/s10654-024-01168-2) |
| Calibration | Van Calster 2019 | [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7) |
| Decision Curves | Vickers & Elkin 2006 | [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361) |
| Prediction Instability | Riley 2023 | [10.1186/s12916-023-02961-2](https://doi.org/10.1186/s12916-023-02961-2) |
| TRIPOD+AI | Collins 2024 | [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) |
| AUROC Interpretation | Hosmer & Lemeshow 2000 | Applied Logistic Regression (textbook) |

---

## Maintenance Notes

This knowledge graph should be updated when:
- New tutorials are added
- New code modules are created
- New figures are generated
- Cross-references change

Last updated: 2026-02-01
