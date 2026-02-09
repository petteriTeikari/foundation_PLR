# Plan: Production-Grade Quarto Tutorial Notebooks

**Branch:** `feat/quarto-notebook-tutorials`
**Created:** 2026-02-09
**Status:** PLANNING

---

## 1. Problem Statement

### 1.1 Current State: Three Stale Jupyter Notebooks

All three notebooks in `notebooks/` were created early in the project and are **severely stale**:

| Notebook | Lines | Issues Found |
|----------|-------|-------------|
| `comprehensive_guide.ipynb` | ~44 cells | Wrong DB paths (`../outputs/`), non-existent `foundation_plr_distributions.db`, hardcoded hex colors (`#3498db`, `#e74c3c`), wrong subject count ("63 people"), heavy inline function defs, compares classifiers (WRONG research question) |
| `data_access_tutorial.ipynb` | ~54 cells | Wrong DB paths (`../outputs/`), non-existent DB files, references classifiers instead of preprocessing pipelines, uses `plt.savefig()` |
| `reproducibility_tutorial.ipynb` | ~37 cells | Imports from `src.stats` that don't exist (`cohens_d`, `hedges_g`, `brier_decomposition`, `monte_carlo_classifier_uncertainty`, `clinical_decision_stability`, `sensitivity_analysis_delta`), wrong DB paths, 60+ line `evaluate_classifier()` defined inline, compares classifiers |

**None of these notebooks can execute.** They would fail on the first cell that tries to connect to a database or import a non-existent module.

### 1.2 Philosophy: Notebooks Are NOT Production

> "There is no good way to put notebooks in production." -- Maria Vechtomova, MLOps Tech Lead

This project runs "normally" via:
- `make reproduce` / `make analyze` / `make extract` (Makefile)
- Prefect flows (`extraction_flow.py`, `analysis_flow.py`)
- R scripts in `src/r/figures/` for ggplot2 figures

Notebooks serve **one purpose**: pedagogical documentation for researchers wanting to understand the repo. They are an **alternative lens** alongside:
- Static documentation (README, ARCHITECTURE.md, MkDocs)
- LLM-assisted exploration (Claude Code, Cursor)
- Mermaid/UML diagrams

### 1.3 Goals

1. **Convert .ipynb to .qmd (Quarto)** -- plain-text, git-friendly, supports Mermaid natively
2. **Reduce from 3 notebooks to 2** -- focused, non-overlapping
3. **Keep code execution trivial** -- `make analyze` or simple DuckDB reads, no heavy computation
4. **Add CI smoke tests** -- `quarto render` fails on broken notebooks
5. **Prevent staleness** -- CI catches drift; freeze caches expensive outputs
6. **Show data scientists how to contribute** -- Quarto-only policy, with test harness

---

## 2. Research Summary: Why Quarto

Based on comprehensive research (Ten Simple Rules for Jupyter [@ruleTenSimpleRules2019], Quaranta 2022, marimo comparison, Quarto CI best practices):

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Format** | `.qmd` (Quarto) | Plain-text Markdown, clean git diffs, no output metadata bloat |
| **vs Jupyter** | Replace `.ipynb` | `.ipynb` = JSON blobs, unreadable diffs, merge conflicts |
| **vs marimo** | Skip | Python-only (we use R+Python), newer/less mature CI |
| **Diagrams** | Mermaid (built-in) | Native `{mermaid}` blocks in `.qmd`, no extensions needed |
| **Caching** | `freeze: auto` | Re-execute only when source changes; commit `_freeze/` |
| **CI testing** | `quarto render --to html` | Smoke test: fails if any cell errors |
| **Linting** | Ruff native (`.ipynb` removal makes this moot) | Extract logic to importable `.py` modules |
| **Logging** | Custom loguru sink + callout blocks | Collapsible HTML `<details>` for Prefect/loguru output |
| **Output stripping** | Not needed | `.qmd` files don't embed outputs |

---

## 3. Architecture: Two Tutorial Notebooks

### 3.1 Notebook 1: `01-pipeline-walkthrough.qmd`

**Audience:** Researcher unfamiliar with the project, wants to understand end-to-end flow.

**Execution model:** Runs `make reproduce-from-checkpoint` on synthetic data (fast, no MLflow needed). Alternatively, reads pre-computed DuckDB directly.

**Content outline:**

```
1. Research Question (Mermaid: pipeline diagram)
   - Fix classifier, vary preprocessing
   - NOT about comparing classifiers

2. The Data: Chromatic Pupillometry (Mermaid: stimulus protocol diagram)
   - What is PLR? (ELI5 for PIs)
   - Blue (469nm) vs Red (640nm) stimulus
   - 30 Hz, 60s recording, 8 handcrafted features

3. Pipeline Architecture (Mermaid: 4-stage flow with method counts)
   - [1] Outlier Detection (11 methods)
   - [2] Imputation (8 methods)
   - [3] Featurization (FIXED: handcrafted)
   - [4] Classification (FIXED: CatBoost)

4. Running the Pipeline
   - Option A: `make reproduce` (full, needs MLflow)
   - Option B: `make analyze` (from DuckDB checkpoint)
   - Option C: Read results directly (this notebook)

5. Exploring Results in DuckDB
   - Connect to `data/public/foundation_plr_results.db`
   - Tables: essential_metrics, calibration_curves, dca_curves, etc.
   - SQL examples: top configs, AUROC distribution, preprocessing effect

6. Key Findings (read from DB, display as tables)
   - Best AUROC: 0.913 (ground truth + CatBoost)
   - Preprocessing effect: eta-squared = 0.15
   - Handcrafted vs Embeddings: 9pp gap

7. STRATOS Metrics Explained (Mermaid: 5-domain diagram)
   - Why AUROC alone is insufficient
   - Calibration, DCA, Net Benefit at clinical thresholds

8. Generating Publication Figures
   - `make r-figures-all` (R/ggplot2)
   - `make figures` (Python/matplotlib)
   - Show example output images
```

**Code cells:** ~15-20 cells, all READ-ONLY from DuckDB. No sklearn, no model training, no heavy computation.

### 3.2 Notebook 2: `02-reproduce-and-extend.qmd`

**Audience:** Researcher wanting to reproduce results OR data scientist wanting to add new analyses.

**Execution model:** Demonstrates running the Prefect analysis flow, then shows how to query the resulting DuckDB for custom analyses.

**Content outline:**

```
1. Reproduction Overview (Mermaid: two-block architecture)
   - Block 1: MLflow -> DuckDB (extraction, needs raw data)
   - Block 2: DuckDB -> Figures/Stats/LaTeX (analysis, portable)

2. Running the Analysis Flow
   - `make analyze` calls `scripts/reproduce_all_results.py --analyze-only`
   - Which calls Prefect `analysis_flow()`
   - Steps: check_public_data -> generate_figures -> compute_statistics -> latex_artifacts
   - Loguru output display (collapsible callout block)

3. What the Flow Produces
   - `outputs/figures/` -- manuscript figures
   - `artifacts/latex/numbers.tex` -- inline LaTeX macros
   - `analysis_report.json` -- run summary

4. Custom Analyses from DuckDB
   - Connect read-only to `foundation_plr_results.db`
   - Example: New calibration metric (ICI)
   - Example: Custom threshold analysis
   - Example: Preprocessing pipeline ranking by multiple metrics

5. For Data Scientists: Adding New Analyses
   (Mermaid: contribution workflow diagram)

   a. The Contract: Quarto-Only Policy
      - All notebook contributions MUST be .qmd files
      - No .ipynb, no marimo -- enforced by .gitignore + pre-commit
      - Logic goes in src/ modules; notebooks are orchestration + narrative

   b. Step-by-Step: Adding a New Statistical Test
      - Write function in src/stats/your_analysis.py
      - Write unit test in tests/unit/test_your_analysis.py
      - Create tutorial notebook in notebooks/extensions/your_analysis.qmd
      - The notebook imports from src/stats/ and reads from DuckDB
      - CI runs `quarto render` to validate

   c. Step-by-Step: Adding a New Visualization
      - Write plot function in src/viz/your_plot.py
      - Register in figure_registry.yaml
      - Create notebook showing the plot with narrative
      - The analysis_flow can optionally call your plot

   d. Integration with Existing Pipeline
      - Your .qmd can be called by Prefect: `quarto render notebook.qmd`
      - Or called by Make: add a target in Makefile
      - Papermill-style parameterization via YAML header `params:`

6. Verifying Reproducibility
   - Expected outputs checklist
   - SHA256 checksums: `make verify-data`
   - CI badge status

7. Troubleshooting
   - Common errors and fixes
   - Missing private data (expected for external researchers)
   - R package installation issues
```

**Code cells:** ~10-15 cells. The Prefect flow call is 1 cell. Rest is DuckDB reads and custom analysis examples.

---

## 4. Implementation Plan

### Phase 1: Infrastructure (T1-T3)

**T1: Install Quarto and configure project**
```bash
# Install Quarto CLI (Linux)
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.40/quarto-1.6.40-linux-amd64.deb
sudo dpkg -i quarto-1.6.40-linux-amd64.deb

# Add Jupyter kernel dependency
uv add --dev jupyter jupyterlab
```

Create `notebooks/_quarto.yml`:
```yaml
project:
  type: default
  output-dir: _output

execute:
  freeze: auto
  echo: true
  warning: false
  error: true     # Halt on errors (don't silently continue)

format:
  html:
    theme: cosmo
    code-fold: true
    code-summary: "Show code"
    toc: true
    toc-depth: 3
    number-sections: true
    fig-width: 10
    fig-height: 6
    fig-dpi: 150
    mermaid:
      theme: neutral
```

**T2: Add .gitignore rules and pre-commit hook**

`.gitignore` additions:
```gitignore
# Notebooks: Quarto-only policy
notebooks/*.ipynb
notebooks/_output/
# Keep _freeze/ for reproducible CI
!notebooks/_freeze/
```

Pre-commit hook `scripts/validation/check_notebook_format.py`:
- Scan for any `.ipynb` files in `notebooks/` (FAIL if found)
- Validate all `.qmd` files have required YAML header fields
- Check no hardcoded paths, hex colors, or method names

**T3: Add Quarto render CI workflow**

`.github/workflows/notebook-tests.yml`:
```yaml
name: Notebook Tests
on:
  pull_request:
    paths:
      - 'notebooks/**'
      - 'src/**'  # Source changes could break notebooks
jobs:
  render:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: quarto-dev/quarto-actions/setup@v2
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: |
          cd notebooks
          quarto render --to html
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rendered-notebooks
          path: notebooks/_output/
```

### Phase 2: Convert Notebooks (T4-T5)

**T4: Create `notebooks/01-pipeline-walkthrough.qmd`**

Write from scratch (do NOT mechanically convert the stale `.ipynb` files). Use:
- Mermaid diagrams for pipeline, stimulus protocol, STRATOS domains
- DuckDB read-only queries against `data/public/foundation_plr_results.db`
- `#| label:` on every code chunk
- No function definitions > 5 lines (import from `src/` instead)
- No sklearn, no model training
- Display pre-generated figure PNGs where appropriate

**T5: Create `notebooks/02-reproduce-and-extend.qmd`**

Write from scratch. Include:
- Prefect flow explanation with Mermaid sequence diagram
- `make analyze` demonstration (or subprocess call)
- Loguru output in collapsible callout block
- Data scientist contribution guide with Mermaid workflow
- Custom analysis examples (ICI metric, threshold sweep)

### Phase 3: Cleanup and Testing (T6-T8)

**T6: Delete old .ipynb files**
- Remove `comprehensive_guide.ipynb`, `data_access_tutorial.ipynb`, `reproducibility_tutorial.ipynb`
- Update `notebooks/README.md` to reflect Quarto policy

**T7: Render locally and commit `_freeze/`**
```bash
cd notebooks
quarto render
git add _freeze/
```

**T8: Run full CI validation**
- Push branch, verify GitHub Actions workflow passes
- Verify rendered HTML output in artifacts

### Phase 4: Documentation (T9)

**T9: Update root README and notebook README**
- Update notebooks section in root README
- Rewrite `notebooks/README.md` with:
  - Quarto-only policy statement
  - How to render locally (`quarto render` or `quarto preview`)
  - How data scientists can contribute new notebooks
  - Link to contribution template

---

## 5. Staleness Prevention Strategy

### 5.1 CI as the Safety Net

The `notebook-tests.yml` workflow triggers on changes to:
- `notebooks/**` -- direct notebook changes
- `src/**` -- source code changes that could break imports

If a refactoring breaks a notebook import, CI catches it immediately.

### 5.2 Freeze for Expensive Outputs

Notebooks that call `make analyze` or heavy DB queries use `freeze: auto`:
- First render: execute and cache results in `_freeze/`
- Subsequent renders: skip execution unless `.qmd` source changed
- CI uses frozen results (fast) unless the notebook itself changed

### 5.3 Minimal Code, Maximum Narrative

The anti-staleness design principle: **notebooks contain almost no logic**.

| What notebooks DO | What notebooks DON'T do |
|---|---|
| `import duckdb; conn.execute("SELECT ...")` | Define 60-line `evaluate_classifier()` functions |
| `import subprocess; subprocess.run(["make", "analyze"])` | Import from `src.stats` modules that may change |
| Display Mermaid diagrams | Compute bootstrap CIs |
| Show pre-generated figure PNGs | Call `plt.savefig()` |
| Explain the research question | Train classifiers |

By keeping code cells to simple DB reads and subprocess calls, the notebooks are insulated from internal refactorings. The only breakage vectors are:
1. DuckDB table schema changes (rare, would break everything)
2. Makefile target renames (rare, well-documented)
3. Figure output path changes (caught by CI)

### 5.4 Quarterly Render Check

Add a scheduled CI job (monthly or quarterly) that renders all notebooks fresh (ignoring freeze) to catch silent staleness from upstream data changes:

```yaml
on:
  schedule:
    - cron: '0 6 1 */3 *'  # First day of every quarter
```

---

## 6. Data Scientist Contribution Workflow

### 6.1 The Contract

```
notebooks/
  _quarto.yml           # Project config (committed)
  _freeze/              # Cached outputs (committed)
  01-pipeline-walkthrough.qmd   # Tutorial 1
  02-reproduce-and-extend.qmd   # Tutorial 2
  extensions/           # Data scientist contributions
    _template.qmd       # Contribution template
    README.md           # Contribution guide
```

**Rules:**
1. All contributions MUST be `.qmd` files
2. Heavy logic MUST live in `src/` modules with unit tests
3. Data access MUST go through `data/public/foundation_plr_results.db` (read-only)
4. No `.ipynb`, no marimo, no R Markdown -- Quarto only
5. `quarto render` must pass in CI

### 6.2 Template: `notebooks/extensions/_template.qmd`

```yaml
---
title: "Your Analysis Title"
author: "Your Name"
date: today
format:
  html:
    code-fold: true
    toc: true
jupyter: python3
execute:
  echo: true
  warning: false
  error: true
params:
  db_path: "../data/public/foundation_plr_results.db"
---
```

### 6.3 How Quarto Notebooks Integrate with Prefect

A data scientist's `.qmd` can be called from the pipeline:

```python
# In a Prefect task or Makefile target
import subprocess
subprocess.run(["quarto", "render", "notebooks/extensions/my_analysis.qmd",
                "--to", "html"], check=True)
```

Or via parameterized execution:
```bash
quarto render my_analysis.qmd -P db_path:path/to/custom.db
```

This keeps the Prefect flow as the orchestrator while allowing data scientists to author rich narrative documents that produce reproducible outputs.

---

## 7. Loguru Display in Quarto

For the notebook that runs `make analyze`, we need to display Prefect/loguru output nicely.

**Approach:** Custom loguru sink that collects messages, then display in a Quarto callout block.

```python
#| label: setup-logging
#| echo: false
from loguru import logger
from IPython.display import display, HTML
import sys

log_messages = []

def notebook_sink(message):
    record = message.record
    level = record["level"].name
    color = {"DEBUG": "#6c757d", "INFO": "#0d6efd",
             "WARNING": "#ffc107", "ERROR": "#dc3545"}.get(level, "#000")
    log_messages.append(
        f'<span style="color:{color};font-family:monospace;">'
        f'[{record["time"]:%H:%M:%S}] [{level}] {record["message"]}</span>'
    )

logger.remove()
logger.add(notebook_sink, level="INFO")
logger.add(sys.stderr, level="WARNING")  # Still show warnings in cell output
```

Then after running the flow:

```python
#| label: show-logs
#| output: asis
#| echo: false
print("::: {.callout-note collapse='true' title='Execution Log (" +
      str(len(log_messages)) + " messages)'}")
print("<br>".join(log_messages))
print(":::")
```

**Alternative (simpler):** Just capture subprocess stdout/stderr and display in a callout:
```python
#| label: run-analysis
import subprocess
result = subprocess.run(
    ["make", "analyze"],
    capture_output=True, text=True, cwd=".."
)
```

```python
#| label: show-output
#| output: asis
#| echo: false
print("::: {.callout-note collapse='true' title='make analyze output'}")
print("```")
print(result.stdout[-3000:])  # Last 3000 chars
print("```")
print(":::")
```

---

## 8. File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `notebooks/_quarto.yml` | Quarto project config |
| `notebooks/01-pipeline-walkthrough.qmd` | Tutorial 1: understand the project |
| `notebooks/02-reproduce-and-extend.qmd` | Tutorial 2: reproduce + contribute |
| `notebooks/extensions/_template.qmd` | Contribution template |
| `notebooks/extensions/README.md` | Contribution guide |
| `.github/workflows/notebook-tests.yml` | CI smoke test |
| `scripts/validation/check_notebook_format.py` | Pre-commit: enforce .qmd-only |

### Modified Files
| File | Change |
|------|--------|
| `notebooks/README.md` | Rewrite for Quarto policy |
| `.gitignore` | Add `.ipynb` exclusion, `_output/` exclusion, `_freeze/` keep |
| `.pre-commit-config.yaml` | Add notebook format check hook |

### Deleted Files
| File | Reason |
|------|--------|
| `notebooks/comprehensive_guide.ipynb` | Stale, replaced by `01-pipeline-walkthrough.qmd` |
| `notebooks/data_access_tutorial.ipynb` | Stale, merged into `01-pipeline-walkthrough.qmd` |
| `notebooks/reproducibility_tutorial.ipynb` | Stale, replaced by `02-reproduce-and-extend.qmd` |

---

## 9. Verification Checklist

- [ ] `quarto render` succeeds locally for both `.qmd` files
- [ ] No `.ipynb` files remain in `notebooks/`
- [ ] `_freeze/` directory committed with cached outputs
- [ ] CI workflow (`notebook-tests.yml`) passes
- [ ] Pre-commit hook rejects `.ipynb` file additions
- [ ] All DuckDB queries point to `data/public/foundation_plr_results.db`
- [ ] No hardcoded hex colors, paths, or method names
- [ ] No sklearn/scipy imports in notebook cells (read from DB instead)
- [ ] Mermaid diagrams render correctly in HTML output
- [ ] Data scientist contribution template is functional
- [ ] Root README updated with notebook section

---

## 10. Dependencies

### System
- Quarto CLI >= 1.6 (not yet installed)

### Python (via uv)
- `jupyter` + `jupyterlab` (for Quarto's Jupyter engine)
- Already have: `duckdb`, `pandas`, `numpy`, `matplotlib`

### No new R dependencies needed
(R figures are generated by `make r-figures-all`, not by notebooks)

---

## 11. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Quarto not available in CI | LOW | `quarto-dev/quarto-actions/setup@v2` handles installation |
| DB schema changes break notebooks | LOW | Notebooks use stable tables (`essential_metrics`), CI catches |
| Data scientists submit `.ipynb` anyway | MEDIUM | Pre-commit hook + `.gitignore` + clear README policy |
| Notebooks become stale again | LOW | CI renders on every PR touching `notebooks/` or `src/` |
| Quarto `convert` bugs drop cells | LOW | We write `.qmd` from scratch, not converting mechanically |
| `freeze` masks broken notebooks | LOW | Quarterly scheduled CI renders fresh (ignoring freeze) |
