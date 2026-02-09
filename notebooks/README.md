# Notebooks (`notebooks/`)

Quarto tutorial notebooks for researchers wanting to understand and extend the Foundation PLR pipeline.

## Quarto-Only Policy

This project uses **Quarto (`.qmd`) exclusively** for notebooks. Jupyter (`.ipynb`) and marimo files are not accepted.

| Rule | Enforcement |
|------|-------------|
| `.qmd` format only | Pre-commit hook + `.gitignore` + CI |
| No heavy logic in cells | Import from `src/` modules |
| Data via DuckDB read-only | Pre-commit bans `sklearn.metrics` |
| No hardcoded colors/paths | Pre-commit pattern check |
| Must render in CI | `quarto render` in GitHub Actions |

**Why Quarto?** Plain-text Markdown with clean git diffs, built-in Mermaid diagrams, `freeze` caching for CI, and native support for both Python and R.

## Available Notebooks

| Notebook | Audience | What You Learn |
|----------|----------|----------------|
| `01-pipeline-walkthrough.qmd` | New researchers, PIs | Research question, pipeline architecture, DuckDB exploration, STRATOS metrics |
| `02-reproduce-and-extend.qmd` | Researchers reproducing results, data scientists | Running the analysis flow, custom analyses, contribution workflow |

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Preview notebooks interactively (opens browser)
cd notebooks
quarto preview

# Or render to HTML
quarto render
# Output in notebooks/_output/
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

## Contributing New Notebooks

See `extensions/README.md` for the contribution guide and template.

```bash
cp extensions/_template.qmd extensions/my_analysis.qmd
# Edit, then render:
quarto render extensions/my_analysis.qmd
```

## Rendering in CI

The `.github/workflows/notebook-tests.yml` workflow renders all notebooks on PRs that touch `notebooks/` or `src/`. This catches stale imports, broken DB queries, and execution errors.
