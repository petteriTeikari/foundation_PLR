# Notebooks

Interactive Quarto tutorials for the Foundation PLR biosignal analysis pipeline. These notebooks are **read-only walkthroughs** -- all computation lives in tested `src/` modules, and all data comes from a portable DuckDB database.

## Quick Start

```bash
# Prerequisites: Quarto >= 1.6 (https://quarto.org/docs/get-started/)

# From project root
uv sync --dev
source .venv/bin/activate
export QUARTO_PYTHON=.venv/bin/python

# Preview interactively (opens browser with live reload)
cd notebooks && quarto preview

# Or render to static HTML (output in notebooks/_output/)
quarto render
```

| Notebook | Audience | Topics |
|----------|----------|--------|
| `01-pipeline-walkthrough.qmd` | New researchers, PIs | Research question, 4-stage pipeline, DuckDB exploration, STRATOS metrics |
| `02-reproduce-and-extend.qmd` | Researchers reproducing results | Two-block architecture, reproducing numbers, custom queries, DCA curves |

**Contributing?** Copy `extensions/_template.qmd`, write your logic in `src/`, then `quarto render` your notebook. See `extensions/README.md` for the DuckDB table reference.

---

## Why Tutorials Only (Not Full Analysis Notebooks)

We deliberately share **thin tutorial notebooks**, not fat analysis notebooks. This is a conscious design choice rooted in hard-won experience: only **4.03% of computational notebooks reproduce successfully** (Pimentel et al. 2019, MSR, analyzing 863,878 notebooks), with fat notebook anti-patterns being the primary cause -- inline data loading with absolute paths, undeclared dependencies, hidden kernel state, and zero test coverage.

Our thin notebook pattern inverts this: notebooks are **orchestration layers** that import from tested `src/` modules, read from a portable DuckDB database, and use a shared style system. Every function called in a notebook cell has pytest coverage. Every SQL query runs against the same database schema. This means a notebook that renders today will render identically in five years, given the same `uv.lock` and `renv.lock`.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-07-notebook-to-production.jpg" alt="Comparison of four notebook-to-production patterns for data science and machine learning projects. Pattern 1 Thin Notebook recommended approach where notebook imports from tested src modules with pytest coverage making code production-ready. Pattern 2 nbdev literate programming where notebook IS the source code with hash-export directives extracting to installable Python package plus documentation plus tests good for library development like fastai. Pattern 3 Notebook-as-Orchestrator used by Databricks and Netflix where master notebook invokes worker notebooks in a DAG scheduling pipeline good for Spark and cloud platforms. Pattern 4 Fat Notebook anti-pattern where all logic is inline with 800 plus lines 60 cells no tests hardcoded paths and the result is code that cannot be tested reused or reproduced. Our implementation uses thin notebooks with 10 to 21 cells under 15 lines each reading from DuckDB with COLORS dict compared to typical fat notebooks with 50 to 100 plus cells 30 to 100 lines using inline pd.read_csv and hardcoded hex colors. Decision matrix shows building a reusable library use nbdev Python or fusen R and building an application use thin notebook plus tested src modules.">
</p>

The figure above contrasts four notebook-to-production patterns. The **Fat Notebook** (Pattern 4) is the default in most research -- and it is an anti-pattern. With 50-100+ cells, 30-100 lines each, inline `pd.read_csv()`, and hardcoded hex colors, these notebooks cannot be tested, reused, or reproduced outside their original author's machine. The **Thin Notebook** (Pattern 1, our approach) caps cells at 10-21 per notebook with under 15 lines each, importing all logic from `src/` modules. The **nbdev** (Pattern 2) and **Notebook-as-Orchestrator** (Pattern 3) patterns serve different use cases -- library development and Spark/cloud DAG scheduling, respectively.

| Metric | Our Notebooks | Typical Fat Notebooks |
|--------|---------------|------------------------|
| Cells per notebook | 10-21 | 50-100+ |
| Lines per cell | <15 | 30-100+ |
| Data source | DuckDB read-only | Inline `pd.read_csv()` |
| Colors | `COLORS` dict | Hardcoded hex values |
| Logic location | `src/` modules | Inline in cells |
| Test coverage | 2042 pytest tests on `src/` | None |

Concretely, our notebooks work as follows:

- **Logic** lives in `src/stats/`, `src/viz/` -- tested by 2042 pytest tests
- **Data** comes from DuckDB (`data/public/foundation_plr_results.db`) -- read-only queries
- **Style** uses `setup_style()` + `COLORS` dict from `plot_config.py` -- no hardcoded hex colors
- **Results** are never computed in cells -- only read from the pre-computed database

The notebooks exist to help researchers understand and reproduce the pipeline, not to run the analysis. The actual analysis is orchestrated by Prefect flows (`make reproduce`).

<details>
<summary>Alternative patterns: nbdev and fusen for library development</summary>

If your project builds a **reusable library** (not an analysis application), consider nbdev (Python, switched to Quarto for docs in 2022) or fusen (R, on CRAN since 2021) where the notebook IS the source code. In nbdev, `#| export` directives in notebook cells generate the installable Python package, and every cell without `#| export` becomes a test by default. nbdev also provides a custom git merge driver that solves `.ipynb` JSON merge conflicts. fastai v2, fastcore, and ghapi were all built this way. fusen provides the same workflow for R: labeled chunks in `.Rmd` files generate R source files, vignettes, and testthat tests via `fusen::inflate()` -- with the philosophy "if you can write RMarkdown, you can build a package."

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-08-nbdev-fusen.jpg" alt="Side-by-side comparison of nbdev Python library development framework by fast.ai and fusen R package development framework by ThinkR for building tested documented installable packages directly from notebooks. nbdev workflow shows notebook.ipynb with hash-export directives generating lib/module.py source code plus Quarto documentation site plus tests where every cell is a test by default with custom git merge driver solving ipynb JSON conflicts and success stories including fastai v2 fastcore and ghapi. fusen workflow shows dev_history.Rmd with labeled chunks for function examples and tests generating R source files vignettes and testthat tests via fusen inflate function available on CRAN since 2021 with philosophy if you can write RMarkdown you can build a package. Decision matrix shows building a library use nbdev or fusen where notebook is source and building an application use thin notebook pattern where notebook imports from tested src modules. Foundation PLR uses thin notebook because we build an analysis application not a reusable library.">
</p>

The key decision criterion is straightforward: if you are building **a library** (a reusable package that others `pip install` or `install.packages()`), use nbdev or fusen -- the notebook-as-source-code pattern gives you documentation, tests, and a distributable package from a single file. If you are building **an application** (an analysis pipeline with specific data, configs, and workflows), use the thin notebook pattern -- the notebook is a tutorial overlay on tested `src/` modules. Foundation PLR is an analysis application, not a reusable library, so we use thin notebooks.
</details>

---

## Why Quarto (Not Jupyter or Marimo)

This project uses **Quarto (`.qmd`) exclusively**. Jupyter (`.ipynb`) and marimo files are rejected by pre-commit hooks. This decision is based on three factors: **version control** (`.qmd` files are plain-text Markdown, producing clean git diffs -- unlike Jupyter's JSON binary format where a single cell output change can generate hundreds of diff lines), **reproducibility** (Quarto enforces top-to-bottom execution via `quarto render`, eliminating the hidden-state problem that plagues Jupyter), and **multi-language support** (a single `.qmd` document can contain Python, R, and JavaScript code cells with shared variables, essential for our mixed Python/R pipeline).

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-01-notebook-landscape.jpg" alt="Comprehensive comparison of three computational notebook platforms for data science Jupyter versus Quarto versus Marimo with feature matrix and decision flowchart. Jupyter uses ipynb JSON format with painful binary git diffs 4.03 percent reproducibility rate from Pimentel et al 2019 MSR study of 863K notebooks kernel-based multi-language support via ipywidgets interactivity mutable kernel execution model and massive ecosystem of 10M plus notebooks. Quarto uses qmd plain text markdown format with clean text-based git diffs high reproducibility via freeze mechanism native R plus Python plus JavaScript multi-language support in single document Shiny and OJS interactivity top-to-bottom execution model and built-in publishing to HTML PDF Word and ePub. Marimo uses py pure Python script format with clean text git diffs high reproducibility via DAG execution model Python-only language support native reactive UI interactivity and automatic DAG-based execution. Decision flowchart shows need R plus Python choose Quarto need reactive dashboard choose Marimo need massive ecosystem choose Jupyter with guards like nbstripout pre-commit and jupytext need publication or version control choose Quarto or Marimo. Key stat only 4.03 percent of 863878 Jupyter notebooks reproduce successfully.">
</p>

The platform comparison reveals a fundamental tradeoff. **Jupyter** has an unmatched ecosystem (10M+ notebooks, extensive widget library, JupyterHub for teams) but its JSON file format and mutable kernel state make version control and reproducibility extremely challenging. **Marimo** solves the reproducibility problem elegantly with reactive DAG execution (deleting a cell automatically invalidates all dependents) and pure `.py` file format, but is Python-only -- disqualifying it for our R-dependent statistical analyses (pminternal, dcurves). **Quarto** combines clean text-based diffs with native multi-language support and built-in publishing to HTML, PDF, Word, and ePub, making it the only platform that satisfies all three requirements simultaneously.

The decision flowchart in the figure provides a practical guide: if you need R + Python in the same document, Quarto is the only viable option. If you need reactive dashboards with pure Python, Marimo excels. If you need the massive ecosystem and don't mind the reproducibility tax, Jupyter with nbstripout + jupytext guards can work.

<details>
<summary>The Jupyter reproducibility problem</summary>

Jupyter notebooks store code, outputs, and metadata in a JSON format that produces noisy, often unreadable git diffs. More critically, **96% of published Jupyter notebooks fail to reproduce** (Pimentel et al. 2019, MSR, analyzing 863,878 notebooks) and **76.88% contain hidden execution-order dependencies** (Pimentel et al. 2021, Empirical Software Engineering). The hidden state problem is insidious: during interactive development, cells are executed out of order, deleted, and re-run. Variables from deleted cells persist in kernel memory, creating invisible dependencies that only surface when the notebook is re-run top-to-bottom -- typically by a reviewer or collaborator trying to reproduce results.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-02-hidden-state-problem.jpg" alt="Why 96 percent of Jupyter notebooks fail to reproduce illustrated as two-panel spot the difference showing the hidden state problem in computational notebooks. Left panel shows notebook during development where cells were executed out of order 1 5 3 7 2 with a deleted cell variable df still in kernel memory so everything appears to work correctly. Right panel shows the same notebook re-run top-to-bottom as cells 1 2 3 4 5 where cell 3 was deleted and cell 7 now fails with NameError because the variable df was defined in the deleted cell that no longer exists creating a cascade of failures. Key statistic 76.88 percent of Jupyter notebooks have at least one hidden state skip where cells depend on variables defined in cells that were executed out of order or no longer exist according to Pimentel et al 2021 Empirical Software Engineering. Bottom strip shows how Quarto prevents this with mandatory top-to-bottom execution via quarto render and how Marimo prevents this with reactive DAG execution where deleting a cell automatically invalidates all dependent cells.">
</p>

The figure illustrates this concretely: during development (left panel), cells executed out of order (1, 5, 3, 7, 2) appear to work because deleted-cell variables persist in kernel memory. When the same notebook is re-run top-to-bottom (right panel), `NameError` cascades through all dependent cells. Quarto prevents this with mandatory top-to-bottom execution via `quarto render`. Marimo prevents it with reactive DAG execution where deleting a cell automatically invalidates all dependent cells.
</details>

---

## Quarto Architecture

Our Quarto project is structured with three enforcement layers that catch progressively different classes of errors. Layer 1 (pre-commit) catches policy violations before code is even committed -- banned imports, hardcoded paths, hex color literals, and sensitive data patterns. Layer 2 (CI) catches runtime errors by executing `quarto render` with `error: false` in a fresh kernel. Layer 3 (freeze) enables fast CI by storing pre-computed results in `_freeze/` as JSON files committed to git, so CI only needs Quarto + Pandoc -- no Python, no DuckDB, no heavyweight compute.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-03-our-quarto-architecture.jpg" alt="Quarto project architecture for reproducible computational notebooks showing three enforcement layers for quality assurance. Project config in _quarto.yml sets error false to halt rendering on errors freeze auto to cache computation results theme cosmo for consistent styling and jupyter python3 engine. Two notebooks 01-pipeline-walkthrough.qmd with 10 code cells and 4 embedded Mermaid diagrams for architecture documentation and 02-reproduce-and-extend.qmd with 21 code cells for reproduction workflow plus extensions directory with contribution template. Three enforcement layers Layer 1 pre-commit runs AST-based check_notebook_format.py catching no hardcoded paths banned sklearn imports no plt.savefig calls and style violations before git commit. Layer 2 CI runs quarto render to html with fresh kernel and error false enforced catching runtime errors broken code and stale database queries. Layer 3 freeze stores pre-computed results in _freeze directory as JSON files committed to git enabling fast CI without Python environment. Data flow shows DuckDB database with 406 configurations queried via SQL by notebooks producing matplotlib figures displayed inline via plt.show with no savefig allowed. Import pattern uses src.viz.plot_config for COLORS setup_style and FIXED_CLASSIFIER constants.">
</p>

The architecture figure shows how these layers compose. The `_quarto.yml` configuration sets `error: false` (halt on any error), `freeze: auto` (cache results), and `jupyter: python3` (engine selection). The two notebooks are structured as thin tutorials: `01-pipeline-walkthrough.qmd` (10 code cells + 4 Mermaid architecture diagrams) and `02-reproduce-and-extend.qmd` (21 cells covering the reproduction workflow). Notebooks query the DuckDB database containing 406 preprocessing configurations (11 outlier methods x 8 imputation methods x 5 classifiers, excluding invalid combinations) via SQL, using the standard import pattern:

```python
from src.viz.plot_config import COLORS, setup_style, FIXED_CLASSIFIER
```

Figures are displayed inline with `plt.show()` -- `plt.savefig()` is banned by the AST-based pre-commit scanner, since notebooks are tutorials, not figure-generation scripts. The `extensions/` directory provides a contribution template (`_template.qmd`) for researchers adding new tutorials.

### Quarto Freeze: CI Without Compute

The freeze mechanism is Quarto's answer to the "CI needs a GPU / 200 packages / 20 minutes" problem. When a developer runs `quarto render` locally, the `freeze: auto` setting serializes all cell outputs (rendered HTML, generated PNGs, printed tables) into JSON files in the `_freeze/` directory. These files are committed to git alongside the source `.qmd` files, creating a version-controlled snapshot of computation results. In CI, `quarto render` detects that frozen results exist and skips re-execution entirely, producing the final HTML output in ~30 seconds using only Quarto + Pandoc -- no Python, no R, no DuckDB, no packages.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-04-quarto-freeze.jpg" alt="How Quarto freeze mechanism works as a CI time machine for computational notebooks enabling continuous integration without heavy compute environments. Four-step timeline Step 1 developer runs quarto render locally which requires full Python environment DuckDB matplotlib and all 200 plus packages installed. Step 2 freeze stores results in _freeze directory as JSON files containing rendered outputs plus generated PNG figures preserving exact computation state. Step 3 _freeze directory is committed to git alongside source qmd files creating a version-controlled snapshot of all computation results. Step 4 CI runs quarto render on frozen results needing only Quarto CLI plus Pandoc with no Python no R no DuckDB no GPU producing HTML output in approximately 30 seconds compared to 5 plus minutes without freeze. Three freeze modes auto re-executes only when source qmd changes recommended for most projects true never re-executes always uses frozen results and false always re-executes ignoring frozen results. Comparison without freeze CI needs Python 3.11 uv sync DuckDB matplotlib all packages taking 5 plus minutes with freeze CI needs only Quarto plus Pandoc taking 30 seconds. Complementary to lockfile-based dependency reproducibility where lockfiles are the dependency time machine and freeze is the computation time machine together making CI trivial.">
</p>

This is complementary to lockfile-based reproducibility: `uv.lock` and `renv.lock` are the **dependency** time machine (pinning exact package versions), while `_freeze/` is the **computation** time machine (preserving exact outputs). Together, they make CI trivial and full reproduction deterministic. Three freeze modes are available: `auto` (re-executes only when source `.qmd` changes -- recommended), `true` (never re-executes, always uses frozen results), and `false` (always re-executes, ignoring frozen results). We use `auto` so that editing a notebook cell triggers re-execution, but unrelated PRs skip computation entirely.

**Important caveat**: freeze only applies to project-level renders (`quarto render` in project root). Rendering individual files (`quarto render notebooks/01-pipeline-walkthrough.qmd`) always re-executes code regardless of the freeze setting.

---

## Notebook Security

Computational notebooks -- whether Jupyter, Quarto, R Markdown, or Marimo -- execute arbitrary code. This section documents the security landscape and our mitigations. The information below is relevant to **any project using computational notebooks**, not just this repository. The threat model is real: Qubitstrike (2023) was the first Jupyter-targeted cryptomining campaign, followed by Panamorfi (2024) DDoS attacks via exposed notebook instances, and the first Jupyter ransomware in 2024. NVD reports for Jupyter doubled between 2023 and 2024 (Jiang et al. 2025, arXiv:2507.18833).

### Jupyter Attack Surface (Where Most Research Exists)

The majority of notebook security research focuses on Jupyter because of its decade-long dominance, server-based architecture (JupyterHub exposes WebSocket connections, file browsers, and terminals), and JSON-based file format (which can embed executable HTML/JavaScript in cell outputs). Six primary attack vectors have been documented:

1. **Malicious Outputs**: XSS-to-RCE via Google Caja sandbox bypasses (CVE-2021-32797, CVE-2021-32798)
2. **Pickle Deserialization**: JFrog Security Research reports an 89% evasion rate for pickle scanning tools, enabling arbitrary code execution via crafted pickle files
3. **Hidden State**: Credentials, API keys, and tokens stored in notebook JSON metadata survive `Clear All Outputs` -- they remain in the `.ipynb` file
4. **Unsandboxed Render**: `quarto render` and `nbconvert --execute` run all code with full system privileges -- no sandbox, no isolation, no permission system
5. **Supply Chain**: Qubitstrike (2023, first Jupyter-targeted cryptomining), Panamorfi (2024, DDoS campaign via exposed instances), and the first Jupyter ransomware (2024)
6. **Exposed Instances**: Publicly accessible Jupyter servers enable crypto mining, NTLM hash leaks on Windows, and unauthorized compute access

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-06-mlsecops-notebooks.jpg" alt="MLSecOps attack surface map for Jupyter notebooks showing six security threat vectors with real CVE numbers and timeline of incidents relevant to any data science or machine learning team using notebooks. Vector 1 Malicious Outputs CVE-2021-32797 and CVE-2021-32798 cross-site scripting XSS leading to remote code execution RCE via Google Caja sandbox bypass in rendered notebook outputs discovered by Google Security Research. Vector 2 Pickle Deserialization CVE-2025-1716 picklescan bypass with JFrog Security Research reporting 89 percent evasion rate for pickle scanning tools enabling arbitrary code execution via crafted pickle files loaded in notebooks. Vector 3 Hidden State where credentials API keys and tokens stored in Jupyter notebook JSON metadata survive cell output clearing remaining in ipynb file even after user clears all outputs. Vector 4 Unsandboxed Render where quarto render and nbconvert execute all code with full system privileges no sandbox no isolation no permission system. Vector 5 Supply Chain attacks including Qubitstrike 2023 first Jupyter-targeted cryptomining campaign Panamorfi 2024 DDoS campaign via exposed notebook instances and first Jupyter ransomware in 2024. Vector 6 Exposed Instances where publicly accessible Jupyter servers enable crypto mining NTLM hash leaks on Windows and unauthorized access to compute resources. CVE timeline from 2021 to 2025 showing XSS to RCE in 2021 metadata credential leaks in 2022 Qubitstrike and exposed instance campaigns in 2023 Panamorfi DDoS ransomware and CVE doubling in 2024 and picklescan bypass zero-days in 2025. Defense tools include NB Defense by Protect AI ModelScan and Sigstore model signing with OWASP ML Security Top 10 reference ML06 AI Supply Chain Attacks.">
</p>

The figure maps all six vectors with CVE numbers and a timeline of real-world incidents from 2021 to 2025. Two vectors are particularly insidious: **Hidden State** (credentials, API keys, and tokens stored in Jupyter notebook JSON metadata survive even after `Clear All Outputs` -- they persist in the `.ipynb` file's metadata section) and **Supply Chain** (the Jupyter ecosystem's reliance on npm packages, pip dependencies, and container images creates a multi-layer attack surface). Defense tools like NB Defense (Protect AI) and ModelScan can mitigate some vectors, but the fundamental issue -- unsandboxed code execution -- remains inherent to the notebook paradigm.

### Quarto Security: What It Fixes, What Persists, What's New

Moving from Jupyter to Quarto is **not a security silver bullet**. Quarto's plain-text `.qmd` format eliminates several Jupyter-specific vectors (no embedded HTML/JS outputs, no JSON metadata injection, no server-side WebSocket attack surface), but code execution during `quarto render` remains fundamentally unsandboxed, and Quarto introduces its own attack surface through Lua filters (`os.execute`, `io.popen`), a weak extension trust model (Y/n prompt only, no code signing), and an expanded Deno/JavaScript dependency chain with its own CVE history.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-06b-quarto-security.jpg" alt="Quarto security analysis showing what Jupyter vulnerabilities are fixed what risks persist and what new attack vectors Quarto introduces organized as three-column comparison relevant to any team evaluating Quarto for computational notebooks. FIXED by Quarto column in green shows XSS-to-RCE via stored outputs eliminated because Quarto qmd files are plain text with no embedded HTML or JavaScript outputs unlike Jupyter ipynb JSON format which stored executable outputs triggering CVE-2021-32797 and CVE-2021-32798. Server-side attack surface eliminated because Quarto has no persistent kernel server no WebSocket connections no file browser no terminal unlike JupyterHub and Jupyter Server. JSON metadata injection eliminated because qmd is plain text not JSON binary format. Google Caja sanitizer dependency eliminated because Quarto does not need output sanitization. Hidden execution state eliminated because qmd files have no stored cell outputs or execution counts. PERSISTS column in gold shows unsandboxed code execution during quarto render where all code cells execute with full user privileges no sandbox no isolation same fundamental risk as jupyter nbconvert execute or knitr knit. Inherited Pandoc vulnerabilities including CVE-2023-35936 and CVE-2023-38745 arbitrary file write via crafted images and CVE-2025-51591 server-side request forgery SSRF via iframe src fetching internal URLs exploited to steal AWS IMDS credentials. Python and R package supply chain risks unchanged regardless of document format. PDF and LaTeX shell escape risk via LuaTeX CVE-2023-32700 shell command execution even with no-shell-escape flag in TeX Live 2017 through 2023. Fundamental reproducibility versus security tradeoff where executing someone elses code for reproducibility inherently requires trust. NEW RISK column in red shows Lua filter remote code execution where Quarto extensions include Pandoc Lua filters that can call os.execute and io.popen with full filesystem and system command access documented Pandoc behavior not a bug. Weak extension trust model with only a Y/n prompt at install time via quarto add with no code signing no permission system no sandboxing no centralized registry with security reviews no hash verification. Freeze directory tampering risk where _freeze JSON files committed to git have no cryptographic integrity verification allowing malicious modification of rendered outputs without changing source qmd files bypassing source-level code review. CVE-2024-38526 the only documented Quarto-specific CVE where polyfill.io CDN supply chain compromise via default MathJax template affected every Quarto HTML document including webpages reveal.js presentations and dashboards patched in Quarto 1.4.557 and 1.5.24. Expanded Deno and JavaScript npm dependency chain with Deno sandbox bypass CVEs including CVE-2022-24783 FFI bypass CVE-2024-34346 proc privilege escalation CVE-2024-32477 ANSI escape permission bypass plus npm supply chain attack surface. No SECURITY.md file and zero published security advisories on quarto-dev quarto-cli GitHub repository as of 2025. Dependency chain diagram shows qmd file flows through Quarto CLI to Pandoc with Lua filters and LuaTeX CVEs then to Deno with sandbox bypass CVEs then to Jupyter kernel with full Python R Julia system access then to Extensions with Lua plus JavaScript Y/n trust only then to CDN dependencies including polyfill.io CVE-2024-38526 and MathJax. Our mitigations table shows freeze-first CI so untrusted qmd files are not re-executed in CI no third-party extensions installed code review required on all freeze directory changes Quarto upgraded past 1.4.557 for polyfill.io fix uv.lock dependency pinning plus Dependabot alerts AST-based scanner for sensitive data patterns and permissions contents read in GitHub Actions workflows.">
</p>

The three-column comparison in the figure provides a complete risk assessment. **What Quarto fixes** (green): XSS-to-RCE via stored outputs eliminated (plain text, no embedded HTML/JS), server-side attack surface eliminated (no persistent kernel, no WebSocket, no file browser), JSON metadata injection eliminated, and hidden execution state eliminated. **What persists** (gold): unsandboxed code execution during `quarto render`, inherited Pandoc vulnerabilities (CVE-2023-35936, CVE-2023-38745 arbitrary file write; CVE-2025-51591 SSRF), Python/R supply chain risks, and PDF/LaTeX shell escape risk (CVE-2023-32700). **New risks** (red): Lua filter RCE via `os.execute`/`io.popen` in Quarto extensions, weak extension trust model (Y/n prompt only, no code signing, no sandboxing), `_freeze/` directory tampering (frozen JSON files rendered without re-execution could be maliciously modified to inject arbitrary HTML, bypassing source-level code review), expanded Deno/JavaScript dependency chain (CVE-2022-24783, CVE-2024-34346, CVE-2024-32477), and CVE-2024-38526 -- the only documented Quarto-specific CVE -- where the polyfill.io CDN supply chain compromise affected every Quarto HTML document via the default MathJax template (patched in Quarto 1.4.557 and 1.5.24).

As of 2025, the `quarto-dev/quarto-cli` GitHub repository has **zero published security advisories** and **no SECURITY.md file** for vulnerability reporting -- a governance gap worth monitoring.

<details>
<summary>Our mitigations</summary>

| Vector | Defense |
|--------|---------|
| Untrusted `.qmd` execution | `error: false` halts on errors; CI uses freeze (no re-execution) |
| Lua filter injection | No third-party Quarto extensions installed |
| `_freeze/` tampering | Code review required on all `_freeze/` changes |
| Supply chain (CDN) | Quarto >= 1.4.557 (polyfill.io patched) |
| Supply chain (Python) | `uv.lock` pinning + Dependabot alerts |
| Sensitive data leaks | AST-based scanner for patient IDs + absolute paths |
| CI privilege escalation | `permissions: contents: read` in workflows |
| `.ipynb` smuggling | Repo-wide scan, not just `notebooks/` |
</details>

### Testing Pyramid

Testing computational notebooks requires a fundamentally different approach than testing application code. Traditional unit tests verify function inputs/outputs, but notebooks introduce additional failure modes: stale database queries (the schema changed but the notebook wasn't re-rendered), policy violations (a collaborator added `sklearn.metrics` in a cell, breaking computation decoupling), and sensitive data leaks (a patient ID hardcoded during debugging). Our four-layer testing pyramid addresses each failure class with a dedicated tool.

<p align="center">
  <img src="../docs/repo-figures/assets/fig-nb-05-notebook-testing-landscape.jpg" alt="Four-layer testing pyramid for computational notebooks showing tools and strategies for ensuring notebook quality in data science and machine learning projects applicable to Jupyter Quarto and Marimo notebooks. Layer 1 base Static Analysis using AST-based checks like check_notebook_format.py which parses Python code cells as abstract syntax trees to detect banned imports such as sklearn.metrics and sklearn.calibration hardcoded hex color values rgb and rgba color literals and plt.savefig calls enforced via pre-commit hooks and CI pipeline catching style violations and policy breaches before code is committed. Layer 2 Smoke Testing using quarto render or nbmake equivalent which executes all cells top-to-bottom in a fresh kernel catching ImportError FileNotFoundError stale database queries undefined variables and execution order problems that cause 76 percent of notebook failures. Layer 3 Output Regression using nbval which compares stored notebook outputs against fresh execution results catching silent numerical drift and unexpected output changes though deliberately not used in this project because the thin notebook pattern means all computation lives in tested src modules making output regression redundant. Layer 4 top Unit Testing using testbook for testing individual functions within notebooks and pytest for testing src modules with this project running 2042 pytest tests on src modules catching logic errors and regressions in the actual computation code. Tool matrix shows check_notebook_format.py tests policy compliance used in pre-commit and CI quarto render tests execution used in CI on every PR pytest tests function logic used in CI and pre-commit nbval tests output regression not used because freeze handles this and testbook tests notebook functions not used because logic lives in src.">
</p>

The pyramid progresses from broad, fast checks (Layer 1: static analysis catches policy violations in milliseconds) to narrow, slow checks (Layer 4: 2042 pytest tests verify function logic in minutes). **Layer 1** uses AST parsing (not regex) to detect banned imports like `sklearn.metrics`, hardcoded hex colors, and `plt.savefig()` calls in code cells. **Layer 2** executes all cells top-to-bottom in a fresh kernel via `quarto render`, catching ImportError, FileNotFoundError, stale database queries, and undefined variables. **Layer 3** (output regression via nbval) and **Layer 4** (testbook for notebook-internal functions) are deliberately not used in this project because the thin notebook pattern pushes all testable logic into `src/` modules -- making notebook-level unit testing and output regression redundant.

Three layers enforce notebook quality:

| Layer | Tool | What It Catches |
|-------|------|-----------------|
| **Pre-commit** | `check_notebook_format.py` | `.ipynb` files, banned imports (AST), hex colors, `.savefig()`, sensitive data, marimo |
| **CI smoke test** | `quarto render` | ImportError, FileNotFound, stale DB queries, execution errors |
| **CI policy check** | Same script as pre-commit | Repo-wide enforcement on PRs and pushes to `main` |

The pre-commit hook uses **AST parsing** (not regex) to detect banned imports like `sklearn.metrics` and `sklearn.calibration` -- catching all import forms including `from sklearn import metrics`, `import sklearn.metrics as sm`, and aliased imports. Sensitive data patterns (patient IDs, absolute home paths) are scanned in code cells only.

Alternative notebook testing tools exist but are deliberately not used: **nbval** (output regression testing) is redundant since our thin notebook architecture pushes all computation to tested `src/` modules, **nbmake** duplicates `quarto render` functionality, and **testbook** (cell-level unit tests) is unnecessary since notebooks contain no testable logic -- the 2042 pytest tests on `src/` modules cover everything.

---

## Notebook Policy

| Rule | Enforcement |
|------|-------------|
| `.qmd` format only | Pre-commit hook + `.gitignore` + CI |
| No heavy logic in cells | Import from `src/` modules (thin notebook pattern) |
| Data via DuckDB read-only | AST-based import detection bans `sklearn.metrics` |
| No hardcoded colors/paths | Pre-commit regex + pattern checks |
| No `.ipynb` anywhere in repo | Repo-wide scan (not just `notebooks/`) |
| No marimo notebooks | `import marimo` detection in `.py` files |
| Must render in CI | `quarto render` in GitHub Actions |

## CI Workflow

The `.github/workflows/notebook-tests.yml` workflow:
1. Triggers on PRs touching `notebooks/`, `src/`, or validation scripts
2. Triggers on pushes to `main` touching `notebooks/` or `src/`
3. Runs `check_notebook_format.py` (repo-wide policy check)
4. Runs `quarto render --to html` (smoke test)
5. Uploads rendered output as artifact (7-day retention)

---

## References

### Notebook Reproducibility
- Pimentel JF, Murta L, Braganholo V, Freire J. "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks." *MSR 2019*. [DOI: 10.1109/MSR.2019.00077](https://doi.org/10.1109/MSR.2019.00077) -- 4.03% reproducibility rate across 863,878 notebooks.
- Pimentel JF, Murta L, Braganholo V, Freire J. "Understanding and Improving the Quality and Reproducibility of Jupyter Notebooks." *Empirical Software Engineering* 26, 65 (2021). [DOI: 10.1007/s10664-021-09961-9](https://doi.org/10.1007/s10664-021-09961-9) -- 76.88% of notebooks have hidden execution-order dependencies.

### Notebook Security
- CVE-2021-32797/32798: Jupyter Notebook XSS-to-RCE via Google Caja sandbox bypass. [NVD](https://nvd.nist.gov/vuln/detail/CVE-2021-32797)
- CVE-2024-38526: polyfill.io CDN compromise in Quarto MathJax template. [Posit Support](https://support.posit.co/hc/en-us/articles/24767859071895)
- CVE-2023-35936/38745: Pandoc arbitrary file write via crafted images. [GitHub Advisory](https://github.com/jgm/pandoc/security/advisories/GHSA-xj5q-fv23-575g)
- JFrog Security Research: picklescan bypass (89% evasion rate). [JFrog Blog](https://jfrog.com/blog/)
- OWASP ML Top 10: ML06 - AI Supply Chain Attacks. [owasp.org](https://owasp.org/www-project-machine-learning-security-top-10/)

### Quarto
- Quarto code execution and freeze. [quarto.org](https://quarto.org/docs/projects/code-execution.html)

### Notebook Development Frameworks
- Howard & Gugger 2022: nbdev v2 with Quarto. [fast.ai](https://www.fast.ai/posts/2022-07-28-nbdev2.html)
- fusen: Build R packages from Rmd. [CRAN](https://cran.r-project.org/package=fusen)
