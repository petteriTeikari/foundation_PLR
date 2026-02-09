# Notebooks

Quarto tutorial notebooks for researchers wanting to understand and extend the Foundation PLR pipeline.

## Why Quarto (Not Jupyter)

This project uses **Quarto (`.qmd`) exclusively**. Jupyter (`.ipynb`) and marimo files are rejected by pre-commit hooks.

<!-- TODO: Add fig-nb-01 (Notebook Landscape) comparing Jupyter/Quarto/Marimo -->

<details>
<summary>Why not Jupyter?</summary>

Jupyter notebooks store code in JSON, producing noisy git diffs. More critically, **96% of published Jupyter notebooks fail to reproduce** (Pimentel et al. 2019, MSR) and 76.88% contain hidden execution-order dependencies (Pimentel et al. 2021, Empirical Software Engineering).

Quarto solves this with plain-text `.qmd` files, mandatory top-to-bottom execution via `quarto render`, and the `freeze` mechanism for caching computation results.

<!-- TODO: Add fig-nb-02 (Hidden State Problem) showing execution order issues -->
</details>

| Rule | Enforcement |
|------|-------------|
| `.qmd` format only | Pre-commit hook + `.gitignore` + CI |
| No heavy logic in cells | Import from `src/` modules (thin notebook pattern) |
| Data via DuckDB read-only | AST-based import detection bans `sklearn.metrics` |
| No hardcoded colors/paths | Pre-commit regex + pattern checks |
| No `.ipynb` anywhere in repo | Repo-wide scan (not just `notebooks/`) |
| No marimo notebooks | `import marimo` detection in `.py` files |
| Must render in CI | `quarto render` in GitHub Actions |

## Available Notebooks

| Notebook | Audience | What You Learn |
|----------|----------|----------------|
| `01-pipeline-walkthrough.qmd` | New researchers, PIs | Research question, 4-stage pipeline, DuckDB exploration, STRATOS metrics overview |
| `02-reproduce-and-extend.qmd` | Researchers reproducing results | Two-block architecture, running analysis, custom queries, DCA curves, contribution workflow |

### 01 -- Pipeline Walkthrough

Walks through the research question, data provenance (Najjar et al. 2023), the four-stage preprocessing pipeline (outlier detection, imputation, featurization, classification), and demonstrates querying the results database. Includes 4 Mermaid diagrams and interactive DuckDB queries.

### 02 -- Reproduce and Extend

Shows the two-block architecture (extraction vs analysis), how to reproduce published numbers with assertions, query calibration/DCA data, and build custom analyses. Demonstrates the thin notebook pattern: all computation lives in `src/`, notebooks only orchestrate and visualize.

## Architecture

<!-- TODO: Add fig-nb-03 (Our Quarto Architecture) showing enforcement layers -->

```
_quarto.yml                    Project config (freeze, theme, error:false)
    |
    +-- 01-pipeline-walkthrough.qmd
    |       |-- DuckDB read-only queries
    |       |-- 4 Mermaid diagrams (pipeline, data, methods, metrics)
    |       +-- matplotlib histogram (via setup_style() + COLORS)
    |
    +-- 02-reproduce-and-extend.qmd
    |       |-- DuckDB read-only queries
    |       |-- src.viz.plot_config (imported for styling)
    |       |-- Assertions against published numbers
    |       +-- DCA curve visualization
    |
    +-- extensions/
            |-- _template.qmd      Contribution template
            +-- README.md           DuckDB table reference
```

### The Thin Notebook Pattern

<!-- TODO: Add fig-nb-07 (Notebook to Production) showing pattern comparison -->

Notebooks are **orchestration layers**, not computation engines:

- **Logic** lives in `src/stats/`, `src/viz/` (tested by 2042 pytest tests)
- **Data** comes from DuckDB (`data/public/foundation_plr_results.db`)
- **Style** uses `setup_style()` + `COLORS` dict from `plot_config.py`
- **Results** are never computed in cells -- only read from the database

### Quarto Freeze

<!-- TODO: Add fig-nb-04 (Quarto Freeze) showing CI time machine -->

The `freeze: auto` setting in `_quarto.yml` captures computation results in `_freeze/`. CI renders from frozen results without needing Python, DuckDB, or any compute dependencies -- just Quarto + Pandoc.

## Quick Start

```bash
# From project root -- activate environment
source .venv/bin/activate

# Preview notebooks interactively (opens browser)
cd notebooks
quarto preview

# Or render to HTML (output in notebooks/_output/)
quarto render
```

## Prerequisites

```bash
# Quarto CLI (>= 1.6)
# See: https://quarto.org/docs/get-started/

# Python environment (from project root)
uv sync --dev

# Set Quarto to use project Python
export QUARTO_PYTHON=.venv/bin/python
```

## Enforcement & Security

<!-- TODO: Add fig-nb-05 (Testing Landscape) showing 4-layer pyramid -->
<!-- TODO: Add fig-nb-06 (MLSecOps) showing attack surface map -->

Three layers enforce notebook quality:

| Layer | Tool | What It Catches |
|-------|------|-----------------|
| **Pre-commit** | `check_notebook_format.py` | `.ipynb` files, banned imports (AST), hex colors, `.savefig()`, sensitive data patterns, marimo |
| **CI smoke test** | `quarto render` | ImportError, FileNotFound, stale DB queries, execution errors |
| **CI policy check** | Same script as pre-commit | Repo-wide enforcement on PRs and pushes to `main` |

The pre-commit hook uses **AST parsing** (not regex) to detect banned imports like `sklearn.metrics` and `sklearn.calibration`. This catches all import forms including `from sklearn import metrics` and aliased imports.

Sensitive data patterns (patient IDs matching `PLR\d{4}`, absolute home paths) are scanned in **code cells only** to avoid false positives from documentation.

## Contributing New Notebooks

See `extensions/README.md` for the contribution guide and DuckDB table reference.

```bash
cp extensions/_template.qmd extensions/my_analysis.qmd
# Edit, then render:
quarto render extensions/my_analysis.qmd
```

## CI Workflow

The `.github/workflows/notebook-tests.yml` workflow:
1. Triggers on PRs touching `notebooks/`, `src/`, or validation scripts
2. Triggers on pushes to `main` touching `notebooks/` or `src/`
3. Runs `check_notebook_format.py` (repo-wide policy check)
4. Runs `quarto render --to html` (smoke test)
5. Uploads rendered output as artifact (7-day retention)
