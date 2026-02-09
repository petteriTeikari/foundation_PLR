# Notebook Figure Coverage Plan

## Context

The `notebooks/` directory was migrated from stale Jupyter to production-grade Quarto.
The `notebooks/README.md` needs 6-8 educational figures explaining the notebook
ecosystem, our design choices, testing, security, and reproducibility patterns.

These figures use the **Nano Banana Pro** content/style decoupling system.
Individual figure plans go to `docs/repo-figures/figure-plans/fig-nb-*.md`.

## Existing Coverage (Gap Analysis)

| Topic | Existing Figure | Gap |
|-------|----------------|-----|
| Jupyter failure modes (ELI5) | fig-repro-02a | Covered |
| Jupyter failure modes (Expert) | fig-repro-02b | Covered |
| Five horsemen of irreproducibility | fig-repro-03 | Mentions notebooks |
| Lockfiles as time machine | fig-repro-14 | Covered (dependency angle) |
| **Quarto vs alternatives** | NONE | **New figure needed** |
| **Notebook-to-production patterns** | NONE | **New figure needed** |
| **Notebook testing landscape** | NONE | **New figure needed** |
| **MLSecOps for notebooks** | NONE | **New figure needed** |
| **Our Quarto architecture** | NONE | **New figure needed** |
| **Quarto freeze mechanism** | NONE | **New figure needed** |
| **Library-from-notebook (nbdev/fusen)** | NONE | **New figure needed** |
| **Rougier's 10 rules applied** | NONE | **New figure needed** |

## Proposed Figures (8 total)

### fig-nb-01: The Notebook Landscape (Decision Matrix)

**ID**: fig-nb-01
**Complexity**: L2 (Data Scientist)
**Key Message**: "Choose your notebook tool based on your primary goal: publication (Quarto), exploration (Jupyter), reproducibility (Marimo)."

Three-column comparison: Jupyter | Quarto | Marimo with decision tree at bottom.

| Dimension | Jupyter | Quarto | Marimo |
|-----------|---------|--------|--------|
| File format | JSON (.ipynb) | Plain text (.qmd) | Pure Python (.py) |
| Git diffs | Poor (JSON noise) | Excellent | Excellent |
| Reproducibility | 4% success rate | Good (render = fresh exec) | Excellent (DAG) |
| Multi-language | Via kernels | Python + R + Julia in one doc | Python only |
| Publishing | nbconvert (limited) | HTML/PDF/Word/ePub/websites | Web apps |
| Interactivity | ipywidgets | Static output | Native reactive UI |
| Execution model | Any-order cells | Render-time (fresh) | Reactive DAG |

Decision flowchart:
- Need R + Python? -> Quarto
- Need reactive dashboard? -> Marimo
- Need massive ecosystem? -> Jupyter (but add reproducibility guards)
- Need publication? -> Quarto
- Need version control? -> Quarto or Marimo

**Sources**: Pimentel et al. 2019/2021 (4% reproducibility), marimo.io/features, quarto.org

---

### fig-nb-02: Why 96% of Notebooks Fail (The Hidden State Problem)

**ID**: fig-nb-02
**Complexity**: L1 (ELI5)
**Key Message**: "Hidden state is the silent killer. The notebook LOOKS correct but isn't -- because cells were run out of order."

**NOTE**: Complements existing fig-repro-02a/02b but focuses specifically on the
EXECUTION ORDER problem with a concrete visual example, not the dependency/import angle.

Visual: Two-panel "Spot the Difference" showing:
- LEFT: Notebook during development (cells run 1, 5, 3, 7, 2 -- variable `df` defined in deleted cell still in memory)
- RIGHT: Same notebook re-run top-to-bottom -- cell 7 fails with NameError because cell 3 was deleted

Stats callout: "76.88% of notebooks have hidden state skips" (Pimentel 2021)

**Key citation**: Pimentel et al. 2021, Empirical Software Engineering (PMC8106381)

---

### fig-nb-03: Our Quarto Architecture (How the Tutorials Work)

**ID**: fig-nb-03
**Complexity**: L2 (Data Scientist)
**Key Message**: "Notebooks are thin orchestration layers. All logic lives in src/, all data in DuckDB, all style in configs."

Shows Foundation PLR's specific architecture:

```
_quarto.yml (project config: freeze, theme, error:false)
    |
    +-- 01-pipeline-walkthrough.qmd
    |       |-- DuckDB (read-only) <-- data/public/foundation_plr_results.db
    |       |-- Mermaid diagrams (4 embedded)
    |       +-- matplotlib (via setup_style() + COLORS dict)
    |
    +-- 02-reproduce-and-extend.qmd
    |       |-- DuckDB (read-only)
    |       |-- src.viz.plot_config (imported)
    |       |-- Contribution examples (eval:false)
    |       +-- Assertions against published numbers
    |
    +-- extensions/
            |-- _template.qmd
            +-- README.md (DuckDB table reference)
```

Three enforcement layers:
1. Pre-commit: check_notebook_format.py (AST-based import detection)
2. CI: quarto render as smoke test (error:false = halt on errors)
3. Freeze: _freeze/ committed = CI doesn't need compute env

---

### fig-nb-04: Quarto Freeze -- Your CI Time Machine

**ID**: fig-nb-04
**Complexity**: L1 (ELI5)
**Key Message**: "Freeze captures computation results locally. CI only does lightweight rendering -- no Python, R, or GPU needed."

Metaphor: "freeze = photograph of your computation"

Timeline:
1. Developer runs `quarto render` locally (needs Python, DuckDB, GPU, etc.)
2. Results stored in `_freeze/` (JSON + figures)
3. `_freeze/` committed to git
4. CI runs `quarto render` on the frozen results -- only Pandoc needed
5. HTML output produced without executing any code

Contrast with "without freeze":
- CI needs: Python 3.11, uv sync, DuckDB, matplotlib, all 200+ packages
- CI time: 5+ minutes
- With freeze: Just Quarto + Pandoc
- CI time: 30 seconds

**Complement**: fig-repro-14 (lockfiles). Freeze is the COMPUTATION time machine;
lockfiles are the DEPENDENCY time machine. Together they make CI trivial.

**Source**: quarto.org/docs/projects/code-execution.html, Mine Cetinkaya-Rundel's Quarto tips

---

### fig-nb-05: Notebook Testing Landscape

**ID**: fig-nb-05
**Complexity**: L3 (ML Engineer)
**Key Message**: "Different tools test different things. Use quarto render for smoke tests, nbval for regression, testbook for unit tests."

Pyramid diagram (bottom = most common, top = most specific):

```
         /    AST-based   \    ← Our check_notebook_format.py
        /   Static Analysis \   (banned imports, hex colors, savefig)
       /─────────────────────\
      /    Quarto Render      \  ← Smoke test: does it execute top-to-bottom?
     /      (nbmake equiv)     \   Catches: ImportError, FileNotFound, stale queries
    /───────────────────────────\
   /      Output Regression      \ ← nbval: did outputs change?
  /          (nbval)              \  Catches: silent numerical drift
 /─────────────────────────────────\
/      Unit Testing (testbook)      \  ← Test individual functions IN notebooks
/       + pytest for src/ modules    \  Catches: logic errors
```

Tool matrix:
| Tool | What it tests | Our usage |
|------|--------------|-----------|
| check_notebook_format.py | Policy (no .ipynb, no sklearn) | Pre-commit + CI |
| quarto render | Execution (smoke test) | CI on every PR |
| pytest (src/) | Logic (functions) | CI + pre-commit |
| nbval | Output regression | Not used (freeze handles this) |
| testbook | Notebook functions | Not used (logic in src/) |

**Sources**: nbmake (Treebeard), nbval (arXiv:2001.04808), Kitware best practices

---

### fig-nb-06: MLSecOps for Notebooks (Attack Surface Map)

**ID**: fig-nb-06
**Complexity**: L3 (ML Engineer / Security)
**Key Message**: "Notebooks are executable documents. Every render is code execution. Defense requires multiple layers."

Attack surface diagram showing 6 vectors:

1. **Malicious .ipynb outputs** (CVE-2021-32797/98: XSS to RCE via Google Caja bypass)
2. **Pickle deserialization** (CVE-2025-1716: picklescan bypass, 89% evasion rate)
3. **Hidden state exploitation** (credentials in JSON metadata survive output clearing)
4. **Quarto render = unsandboxed execution** (rendering .qmd = running untrusted code)
5. **Supply chain** (malicious extensions, compromised packages in notebook env)
6. **Exposed instances** (Qubitstrike 2023, Panamorfi DDoS 2024, ransomware 2024)

Our mitigations:
| Vector | Our Defense |
|--------|-----------|
| .ipynb smuggling | Repo-wide scan (not just notebooks/), .gitignore, AST check |
| Unsafe imports | AST-based detection (sklearn.metrics, sklearn.calibration) |
| Sensitive data | PLR code + abs path scanner in code cells |
| Hardcoded secrets | Pre-commit + CI policy checks |
| Unsandboxed render | error:false (halt on errors), eval:false for examples |
| Marimo smuggling | import marimo detection in .py files |

CVE timeline: 2021-2025, showing doubling of Jupyter CVEs between 2023-2024.

**Sources**: Google Security Research (GHSA-c469-p3jp-2vhx), JFrog picklescan report 2025,
OWASP ML Security Top 10, Aqua Security (Jupyter ransomware), Darktrace (cryptominer campaigns)

---

### fig-nb-07: From Notebook to Production (The Thin Notebook Pattern)

**ID**: fig-nb-07
**Complexity**: L2 (Data Scientist)
**Key Message**: "Notebooks orchestrate; src/ computes. This is the only pattern that scales."

Four patterns compared (good -> bad gradient):

```
PATTERN 1: Thin Notebook (RECOMMENDED - our approach)
notebook → imports from src/ → src/ has pytest tests → production-ready

PATTERN 2: nbdev Literate Programming
notebook IS the source → #|export extracts → package + docs + tests
Good for: library development

PATTERN 3: Notebook-as-Orchestrator (Databricks/Netflix)
master notebook → worker notebooks → DAG scheduling
Good for: Spark/cloud pipelines

PATTERN 4: Fat Notebook (ANTI-PATTERN)
notebook has ALL logic → no tests → not importable → "it works on my machine"
```

Foundation PLR's implementation:
- Notebooks: 10-21 cells each, all < 15 lines
- Logic: in `src/stats/`, `src/viz/`
- Data: DuckDB read-only queries
- Style: `setup_style()` + `COLORS` dict from plot_config.py
- Testing: `pytest tests/` (2042 tests), `quarto render` (smoke)

Rougier's Rule #1 applied: "Know Your Audience" -- notebooks for researchers,
src/ for developers, configs/ for reproducibility.

**Sources**: Ploomber (nbs-production), Databricks best practices, Netflix tech blog,
Rougier et al. 2014 (Ten Simple Rules for Better Figures, PLOS Comp Biol)

---

### fig-nb-08: Building Libraries from Notebooks (nbdev and fusen)

**ID**: fig-nb-08
**Complexity**: L2 (Data Scientist)
**Key Message**: "If you want to build libraries: Use nbdev (Python) or fusen (R). They turn notebooks into tested, documented packages."

Two-column comparison:

**nbdev (Python, by fast.ai)**:
```
notebook.ipynb
  |-- #| export → lib/module.py
  |-- assertions → test suite
  |-- markdown → documentation site
  |-- nbdev_export → Python package on PyPI
```
- Source of truth: notebook
- Output: Python package + Quarto docs site
- Test: every cell is a test by default
- Git: custom merge driver solves JSON conflicts
- Success: fastai v2, fastcore, ghapi

**fusen (R, by ThinkR)**:
```
dev_history.Rmd
  |-- function chunks → R/module.R
  |-- example chunks → vignettes/
  |-- test chunks → tests/testthat/
  |-- fusen::inflate() → Full R package
```
- Source of truth: RMarkdown/Quarto
- Output: CRAN-ready R package
- Test: testthat chunks extracted
- Insight: docs-first → better-documented packages

**When NOT to use**: If your project is an application (not a library),
use the thin notebook pattern instead (fig-nb-07).

**Sources**: nbdev.fast.ai, fast.ai blog (nbdev2+Quarto announcement 2022),
fusen CRAN page, GitHub Blog (nbdev literate programming)

## Figure Naming Convention

All notebook-related figures use prefix `fig-nb-`:

| ID | Title | Complexity | Priority |
|----|-------|-----------|----------|
| fig-nb-01 | The Notebook Landscape (Decision Matrix) | L2 | P1 |
| fig-nb-02 | Why 96% of Notebooks Fail (Hidden State) | L1 | P1 |
| fig-nb-03 | Our Quarto Architecture | L2 | P1 |
| fig-nb-04 | Quarto Freeze -- Your CI Time Machine | L1 | P2 |
| fig-nb-05 | Notebook Testing Landscape | L3 | P2 |
| fig-nb-06 | MLSecOps for Notebooks | L3 | P2 |
| fig-nb-07 | From Notebook to Production | L2 | P1 |
| fig-nb-08 | Building Libraries from Notebooks | L2 | P3 |

## Key References

| Source | Citation | Used In |
|--------|----------|---------|
| Pimentel et al. 2019 | MSR 2019, 863K notebooks, 4.03% reproduce | fig-nb-01, fig-nb-02 |
| Pimentel et al. 2021 | Emp. Soft. Eng., 76.88% hidden state | fig-nb-02 |
| Rougier et al. 2014 | PLOS Comp Biol, 10 Simple Rules | fig-nb-07 |
| Google Security Research | CVE-2021-32797/98 (Jupyter RCE) | fig-nb-06 |
| JFrog 2025 | Picklescan zero-days, 89% bypass rate | fig-nb-06 |
| nbval paper | arXiv:2001.04808 | fig-nb-05 |
| fast.ai 2022 | nbdev2 + Quarto announcement | fig-nb-08 |
| Mine Cetinkaya-Rundel | Quarto tip-a-day (freeze) | fig-nb-04 |
| Netflix Tech Blog | Notebook Innovation (Papermill/Maestro) | fig-nb-07 |
| arXiv:2502.04184 (2025) | "Non-executable" notebooks restorable | fig-nb-02 |
| OWASP ML Security Top 10 | ML06 Supply Chain, ML10 Poisoning | fig-nb-06 |
| Yang et al. 2023 | Data Leakage in Notebooks (static analysis) | fig-nb-05 |
| Quaranta et al. 2022 | Notebook quality assessment | fig-nb-07 |

## Nikola Rougier's "Ten Simple Rules" Integration

Not Nikola T. Markov -- the correct reference is **Nicolas P. Rougier, Michael Droettboom,
and Philip E. Bourne** (2014, PLOS Computational Biology, DOI: 10.1371/journal.pcbi.1003833).

Rules most relevant to our notebook figures:

| Rule | Application |
|------|-------------|
| **#1 Know Your Audience** | README figures target data scientists evaluating the tool |
| **#3 Adapt to Medium** | GitHub constrains width; figures must be scannable at 50% zoom |
| **#5 Don't Trust Defaults** | Our setup_style() + COLORS dict = custom theming, not matplotlib defaults |
| **#8 Avoid Chartjunk** | Nano Banana Pro aesthetic: elegant medical illustration, no sci-fi |
| **#9 Message Trumps Beauty** | Each figure has ONE key message |

## Relationship to Existing Figures

```
EXISTING (reproducibility angle):
  fig-repro-02a → Jupyter fails (ELI5 - dependency focus)
  fig-repro-02b → Jupyter fails (Expert - error taxonomy)
  fig-repro-03  → Five horsemen (broader scope)
  fig-repro-14  → Lockfiles (dependency time machine)

NEW (notebook ecosystem angle):
  fig-nb-01 → Compare Jupyter/Quarto/Marimo (decision matrix)
  fig-nb-02 → Hidden state specifically (execution order, not deps)
  fig-nb-03 → Our Quarto architecture (project-specific)
  fig-nb-04 → Freeze mechanism (computation time machine)
  fig-nb-05 → Testing landscape (tools + our stack)
  fig-nb-06 → MLSecOps (CVEs + attack surface)
  fig-nb-07 → Notebook-to-production patterns
  fig-nb-08 → nbdev/fusen (library building)
```

No overlap with existing figures -- each addresses a distinct gap.

## Next Steps

1. Create individual `.md` files in `docs/repo-figures/figure-plans/`
2. Add semantic tags from STYLE-GUIDE.md to each plan
3. Generate PNGs via Nano Banana Pro (Gemini) using PROMPTING-INSTRUCTIONS.md
4. Update `notebooks/README.md` with figure references
