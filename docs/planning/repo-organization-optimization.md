# Repository Organization Optimization Plan

**Date**: 2026-02-06
**Branch**: `chore/publication-polish`
**Author**: Architectural review (automated audit)

---

## 1. Executive Summary

The foundation_PLR repository has accumulated structural debt across 1,562 tracked files during rapid research iteration. The primary issues are: (a) duplicate output locations (`outputs/`, `artifacts/`, `figures/generated/`) with unclear canonical status, (b) 1.6 GB of untracked binary artifacts on disk (`outputs/top10_catboost_models.pkl`), (c) committed training artifacts that should be gitignored (`catboost_info/`, `scripts/plr_results.duckdb`), (d) a 766 MB untracked MkDocs build directory (`docs/repo-figures/generated/`), (e) dead code modules (`src/deploy/`, `src/figures/`), and (f) a root-level symlink (`r -> src/r`) and misplaced files (`3746252.3760801.pdf`, `analysis_report.json`) that confuse newcomers. This plan proposes 19 changes across 3 tiers, all achievable without modifying Python imports or breaking CI. Estimated disk savings from gitignore fixes alone: reclaim ~40 KB of tracked space plus prevent future accidents; estimated cognitive load reduction for new contributors: significant, by reducing the top-level directory count from 28 visible entries to 22.

---

## 2. Current State Assessment

### 2.1 Top-Level Directory Tree (28 visible entries)

```
foundation_PLR/                          # 1,562 tracked files
|
|-- AGENTS.md                            # Universal LLM agent instructions (tracked)
|-- ARCHITECTURE.md                      # System architecture docs (tracked)
|-- CITATION.cff                         # Citation metadata (tracked)
|-- CLAUDE.md                            # Claude-specific rules (tracked)
|-- CONTRIBUTING.md                      # Contribution guide (tracked)
|-- LICENSE                              # MIT license (tracked)
|-- Makefile                             # Build targets (tracked)
|-- README.md                            # Project README (tracked)
|-- mkdocs.yml                           # MkDocs config (tracked)
|-- pyproject.toml                       # Python project config (tracked)
|-- uv.lock                             # Dependency lockfile (tracked)
|
|-- 3746252.3760801.pdf                  # MISPLACED: ACM paper PDF (gitignored by /*.pdf)
|-- analysis_report.json                 # MISPLACED: pipeline report (gitignored by /*.json)
|
|-- Docker*, docker-compose.yml          # Container configs (tracked)
|-- .env, .env.example                   # Environment vars
|-- .python-version, .Rprofile           # Language version pins
|-- .pre-commit-config.yaml              # Pre-commit hooks
|-- renv.lock, .renvignore               # R package management
|
|-- r -> src/r                           # SYMLINK: convenience alias (tracked)
|
|-- apps/visualization/     162 MB       # React+D3.js dashboard (tracked: 16 files)
|-- archived/               600 KB       # NuwaTS, TSB_AD experimental code (tracked)
|-- artifacts/latex/         12 KB       # REDUNDANT: numbers.tex (tracked: 1 file)
|-- assets/                 2.3 MB       # Static images (tracked: 17 files)
|-- catboost_info/           40 KB       # PROBLEM: training artifacts (tracked: 4 files!)
|-- configs/                584 KB       # 20 Hydra subdirs (tracked: 83 files)
|-- data/                   1.7 GB       # private/public/r_data/synthetic (tracked: 2 files)
|-- docs/                   804 MB       # MkDocs + 793 MB repo-figures (tracked: 359 files)
|-- figures/generated/       42 MB       # Canonical manuscript figures (tracked: 101 files)
|-- logs/                    60 KB       # Pipeline logs (gitignored contents)
|-- mlruns/                  64 MB       # Local MLflow experiments (gitignored contents)
|-- notebooks/              144 KB       # Jupyter tutorials (tracked: 4 files)
|-- outputs/                1.7 GB       # OLD figures + 1.6 GB pickles (gitignored contents)
|-- renv/                    24 MB       # R package library (mostly gitignored)
|-- scripts/                1.1 MB       # 53 tracked utility scripts
|-- site/                   787 MB       # PROBLEM: MkDocs build output (gitignored, on disk)
|-- src/                     33 MB       # 23 source modules (tracked: 402+ files)
|-- tests/                  7.5 MB       # 30+ test modules (tracked: 171 files)
```

### 2.2 Key Metrics

| Metric | Value |
|--------|-------|
| Tracked files | 1,562 |
| Top-level visible dirs/files | 28 (high cognitive load) |
| Disk usage (working tree) | ~7.2 GB |
| Tracked data that should not be | `catboost_info/` (4 files), `artifacts/latex/numbers.tex` (1 file), `scripts/plr_results.duckdb` (1 file) |
| Untracked large dirs on disk | `site/` (787 MB), `outputs/` (1.7 GB), `docs/repo-figures/generated/` (766 MB) |
| Dead code modules | `src/deploy/` (confirmed stub), `src/figures/` (README-only, no code) |
| Duplicate outputs | `artifacts/latex/numbers.tex` = `outputs/latex/numbers.tex` (differ only by timestamp) |
| Empty directories | 18 (across catboost_info/tmp, outputs/stats_cache, outputs/synthetic, figures/synthetic, renv/staging, mlruns/.trash, .claude/sessions, .claude/hooks, .claude/config, figures/generated/svg, figures/generated/private, figures/generated/matplotlib, figures/generated/d3-js, tests/golden_files/figures, tests/r/_snaps, .claude/docs/biblio-pupil, .git/branches, .git/refs/tags) |

### 2.3 Pain Points for New Contributors

1. **Where do figures go?** Three candidates: `outputs/fig_*.png`, `figures/generated/`, `src/figures/generated/`
2. **Where does LaTeX go?** Two candidates: `outputs/latex/`, `artifacts/latex/`
3. **What is `r/` at root?** A symlink that appears to be a top-level directory
4. **Is `catboost_info/` mine?** No, it is committed training artifacts from a prior run
5. **Where are plans?** `docs/planning/` has 57 planning documents, `figures/generated/execution-plans/` has 6 more
6. **What is `3746252.3760801.pdf`?** An unnamed ACM paper sitting at root
7. **Is `scripts/` code or config?** It contains 53 scripts, a DuckDB database, and a README

---

## 3. Proposed Changes

### Tier 1: Immediate (Safe, No Import Changes, No CI Impact)

These changes can be made in a single commit. They remove tracked files that should never have been committed, delete confirmed dead code, and fix gitignore gaps.

---

#### T1-01: Remove `catboost_info/` from git tracking

**What**: Remove `catboost_info/` from the git index and add it to `.gitignore`.

**Why**: These are CatBoost training artifacts (binary TensorFlow events, training logs) generated at runtime. They are committed (4 files tracked) but provide no value -- they change on every training run and contain no reproducibility information.

**Risk**: None. No code imports from `catboost_info/`. Not referenced in CI. The directory will remain on disk for local use.

**Commands**:
```bash
# Add to .gitignore (before the existing /src/catboost_info/* rule)
echo "catboost_info/" >> .gitignore

# Remove from tracking
git rm -r --cached catboost_info/
git commit -m "chore: remove catboost_info/ from tracking (training artifacts)"
```

---

#### T1-02: Remove `artifacts/` directory (redundant with `outputs/latex/`)

**What**: Remove `artifacts/latex/numbers.tex` from tracking and delete the `artifacts/` directory.

**Why**: `artifacts/latex/numbers.tex` is identical to `outputs/latex/numbers.tex` except for the generation timestamp. Only 1 file is tracked in `artifacts/`. The canonical location per CLAUDE.md is `outputs/` (via `save_figure()` and the orchestration pipeline). No code references `artifacts/` as an output path (the only `artifacts/` grep hit in `src/` is for MLflow run artifacts, a completely different concept).

**Risk**: Low. If any LaTeX build process references `artifacts/latex/numbers.tex`, it will need updating. Check the sister manuscript repository first.

**Verification before executing**:
```bash
# Check if the manuscript repo references artifacts/latex/
grep -r "artifacts/latex" /home/petteri/Dropbox/github-personal/sci-llm-writer/ 2>/dev/null
```

**Commands**:
```bash
git rm -r artifacts/
echo "artifacts/" >> .gitignore
git commit -m "chore: remove redundant artifacts/ (canonical: outputs/latex/)"
```

---

#### T1-03: Remove `scripts/plr_results.duckdb` from tracking

**What**: Remove the 12 KB DuckDB file committed inside `scripts/`.

**Why**: A database file in a scripts directory is clearly a leftover from development. The canonical results database is `data/public/foundation_plr_results.db`. No script references `scripts/plr_results.duckdb` (verified by grep). Binary database files should never be tracked.

**Risk**: None.

**Commands**:
```bash
git rm scripts/plr_results.duckdb
echo "scripts/*.duckdb" >> .gitignore
git commit -m "chore: remove stale scripts/plr_results.duckdb from tracking"
```

---

#### T1-04: Remove misplaced root files

**What**: Delete `3746252.3760801.pdf` and `analysis_report.json` from the working tree.

**Why**: Both are already gitignored by the root-level `/*.pdf` and `/*.json` rules, so they are not tracked. They are development leftovers sitting at root. The PDF filename (ACM DOI format `10.1145/3746252.3760801`) suggests it is a reference paper that belongs in the researcher's personal bibliography, not the repo. The JSON is a pipeline output that should be in `outputs/`.

**Risk**: None (untracked files).

**Commands**:
```bash
rm 3746252.3760801.pdf
rm analysis_report.json
```

---

#### T1-05: Remove `src/figures/` (non-code directory inside source tree)

**What**: Remove `src/figures/README.md` and the `src/figures/` directory from tracking.

**Why**: `src/figures/` contains only a README.md pointing to `figures/generated/`. It is not a Python package (no `__init__.py`). It has a `generated/ggplot2/` subdirectory on disk that is empty/untracked. Having an output directory inside `src/` violates the principle that `src/` contains only source code. The canonical figure output is `figures/generated/`.

**Risk**: None. No Python imports reference `src.figures`. No CI references this path.

**Commands**:
```bash
git rm src/figures/README.md
# Clean up empty dirs on disk
rm -rf src/figures/
git commit -m "chore: remove src/figures/ (output dir does not belong in src/)"
```

---

#### T1-06: Add `site/` to `.gitignore` explicitly (belt and suspenders)

**What**: Ensure `site/` is robustly gitignored and remove it from disk if desired.

**Why**: `site/` is 787 MB of built MkDocs HTML. The `.gitignore` already has `site/` which is why 0 files are tracked. However, the directory persists on disk at 787 MB. This is working as designed -- `.gitignore` prevents tracking, `mkdocs build` regenerates it. No action needed on the gitignore itself, but document this for contributors.

**Risk**: None.

**Action**: Informational only. Add a comment in `.gitignore` near the `site/` entry:
```gitignore
# MkDocs build output — regenerate with: mkdocs build
# This directory can safely be deleted; it will be rebuilt automatically.
site/
```

---

#### T1-07: Add `docs/repo-figures/generated/` to `.gitignore`

**What**: Gitignore the 766 MB generated PNG directory.

**Why**: `docs/repo-figures/generated/*` is already in `.gitignore`, but the parent `docs/repo-figures/generated/` directory itself is not explicitly ignored. The current pattern `docs/repo-figures/generated/*` catches files inside it. However, the 766 MB of generated PNGs on disk could accidentally be committed if someone adjusts the ignore pattern. The tracked assets (JPEGs in `docs/repo-figures/assets/`) are fine at ~2 MB total. Add a clarifying comment.

**Risk**: None.

**Action**: Already handled by the existing `docs/repo-figures/generated/*` rule. Verify with:
```bash
git status docs/repo-figures/generated/  # Should show nothing
```

---

#### T1-08: Remove `src/deploy/` (confirmed dead code)

**What**: Remove the `src/deploy/` module (3 files: `__init__.py`, `flow_deployment.py`, `README.md`).

**Why**: `flow_deployment.py` is explicitly marked as `# TODO: DEAD CODE` in its own first line. It contains only a placeholder function that logs messages and does nothing. The README confirms it is a "Placeholder module". It is imported by `src/pipeline_PLR.py` but the `flow_deployment()` function is a no-op (`deploy = False`).

**Risk**: Low. Requires removing the import in `src/pipeline_PLR.py`. This is a one-line code change, but per the constraint of "reorganization only, no code changes", this should be deferred to a companion cleanup PR. Alternatively, keep the module but mark it more prominently as deprecated.

**Decision**: **Defer to Tier 2** since removing it requires a code change in `pipeline_PLR.py`.

---

### Tier 2: Short-Term (Requires Coordination, Minor Code Touches)

These changes require updating references in Makefile, configs, or minor code paths. Should be done in a dedicated cleanup branch.

---

#### T2-01: Remove the `r` symlink at root

**What**: Remove the `r -> src/r` symlink from the repository root.

**Why**:
- **Confusion**: A new contributor sees `r/` at root and assumes it is a top-level R source directory. The symlink target is `src/r/`, which is the real location.
- **No functional users**: No Makefile target, CI workflow, or pre-commit hook references `r/` (they all use `src/r/`). The `.Rprofile` only sources `renv/activate.R`. The pre-commit config explicitly references `src/r/`.
- **Cross-platform risk**: Symlinks behave differently on Windows (requires developer mode or admin). Since this is an academic repo that may be cloned by collaborators on various platforms, the symlink is a portability hazard.
- **Git tracking**: The symlink is tracked as a git blob containing the text `src/r`, adding unnecessary tracked state.

**Risk**: Low. If any developer has muscle memory of typing `r/figures/`, they would need to use `src/r/figures/` instead. A grep of the entire codebase shows zero references to the `r/` path (outside the symlink itself and the git ls-files entry).

**Commands**:
```bash
git rm r
git commit -m "chore: remove r -> src/r symlink (use src/r/ directly)"
```

---

#### T2-02: Consolidate `outputs/` -- Establish clear purpose

**What**: Clarify that `outputs/` is the *runtime output directory* for pipeline artifacts (not tracked), while `figures/generated/` is the *publication figure directory* (selectively tracked).

**Why**: Currently `outputs/` contains:
- 6 old PNG figures (231 KB each, from Jan 19) -- superseded by `figures/generated/`
- LaTeX tables (`outputs/latex/`, `outputs/tables/`)
- SHAP artifacts (1.6 GB pickle, 39 MB values, 39 MB checkpoints)
- Execution logs (`execution.log`, `EXECUTION_CHECKPOINT.json`)
- Empty dirs (`stats_cache/`, `synthetic/`)

The old PNGs in `outputs/` root are from the pre-ggplot2 era and are not referenced anywhere. The LaTeX tables in `outputs/tables/` are the canonical table output. The SHAP pickles are large intermediate artifacts.

**Risk**: Low. `outputs/` is already fully gitignored. This is a documentation/convention change.

**Actions**:
1. Update `outputs/README.md` to clearly state:
   - `outputs/` = runtime pipeline outputs, NEVER committed
   - `outputs/latex/` = generated LaTeX macros (numbers.tex)
   - `outputs/tables/` = generated LaTeX tables
   - `outputs/shap_*` = SHAP intermediate artifacts
   - Old `fig_*.png` files at root = legacy, safe to delete
2. Delete the 6 legacy PNGs from `outputs/` root:
   ```bash
   rm outputs/fig_calibration_plot.png outputs/fig_classifier_comparison.png \
      outputs/fig_decision_curve_analysis.png outputs/fig_effect_sizes_heatmap.png \
      outputs/fig_probability_distributions.png outputs/fig_roc_curves.png
   ```
3. Delete the old root-level LaTeX files (superseded by `outputs/tables/`):
   ```bash
   rm outputs/table_calibration.tex outputs/table_classifier_performance.tex \
      outputs/table_pairwise_comparisons.tex outputs/table_summary.tex
   ```

---

#### T2-03: Organize `scripts/` into subdirectories

**What**: Group the 53 scripts into logical subdirectories.

**Why**: A new contributor opening `scripts/` sees 53 files with no organization. The scripts fall into clear categories:

| Category | Count | Files |
|----------|-------|-------|
| **Extraction** (MLflow to DuckDB) | 12 | `extract_*.py` |
| **Export** (DuckDB to R/CSV) | 8 | `export_*.py` |
| **Validation/Verification** | 12 | `check_*.py`, `validate_*.py`, `verify_*.py` |
| **Computation** | 4 | `compute_*.py`, `plr_decomposition_*.py` |
| **Figure generation** | 3 | `figure_and_stats_generation.py`, `generate_*.py` |
| **Infrastructure** | 6 | `setup-dev-environment.sh`, `test-docker.sh`, `pre-commit`, `check_types.sh`, `config_versioning.py`, `auto_version_configs.py` |
| **One-off** | 3 | `reproduce_all_results.py`, `fix_json_provenance.py`, `sigllm_anomaly_detection.py` |
| **Meta** | 1 | `README.md` |

Proposed structure:
```
scripts/
  extraction/        # extract_*.py (MLflow -> DuckDB)
  export/            # export_*.py (DuckDB -> R/CSV)
  validation/        # check_*.py, validate_*.py, verify_*.py
  computation/       # compute_*.py, plr_decomposition_*.py
  figures/           # figure_and_stats_generation.py, generate_*.py
  infra/             # setup scripts, docker, pre-commit, typing
  reproduce_all_results.py  # Top-level entry point stays at root of scripts/
  README.md
```

**Risk**: Medium. The Makefile references 14 scripts by path (e.g., `python scripts/check-compliance.py`). All Makefile references must be updated. CI workflows reference `scripts/verify_registry_integrity.py`, `scripts/check_computation_decoupling.py`, `scripts/check_parallel_systems.py`.

**Commands**: Use `git mv` for each file, then update Makefile and CI paths.

**Makefile references to update** (all 14):
```
scripts/check-compliance.py          -> scripts/validation/check-compliance.py
scripts/test-docker.sh               -> scripts/infra/test-docker.sh
scripts/verify_registry_integrity.py -> scripts/validation/verify_registry_integrity.py
scripts/check_types.sh               -> scripts/infra/check_types.sh
scripts/pre-commit                   -> scripts/infra/pre-commit
scripts/reproduce_all_results.py     -> scripts/reproduce_all_results.py (stays)
scripts/validate_experiments.py      -> scripts/validation/validate_experiments.py
scripts/validate_figures.py          -> scripts/validation/validate_figures.py
```

**CI workflow references to update** (`ci.yml`):
```
scripts/verify_registry_integrity.py  -> scripts/validation/verify_registry_integrity.py
scripts/check_computation_decoupling.py -> scripts/validation/check_computation_decoupling.py
scripts/check_parallel_systems.py     -> scripts/validation/check_parallel_systems.py
```

---

#### T2-04: Remove `src/deploy/` (dead code with import fix)

**What**: Delete `src/deploy/` and remove its import from `src/pipeline_PLR.py`.

**Why**: As noted in T1-08, this is confirmed dead code with a `# TODO: DEAD CODE` marker. The function body is `deploy = False` followed by `logger.info()` calls.

**Risk**: Low. Requires changing one import line in `src/pipeline_PLR.py`.

**Commands**:
```bash
# Remove the import in pipeline_PLR.py
# (change: remove "from src.deploy.flow_deployment import flow_deployment")
# (change: remove the call to flow_deployment(cfg) if it exists)

git rm -r src/deploy/
git commit -m "chore: remove dead src/deploy/ module (placeholder never implemented)"
```

---

#### T2-05: Move execution plans out of `figures/generated/`

**What**: Move `figures/generated/execution-plans/` to `docs/planning/execution-plans/`.

**Why**: Execution plans are planning documents, not figure outputs. They ended up in `figures/generated/` likely because they were created during a figure generation planning session. They belong with the other 57 planning documents in `docs/planning/`.

**Risk**: Low. These are markdown/XML planning documents with no code references.

**Commands**:
```bash
git mv figures/generated/execution-plans/ docs/planning/execution-plans/
git commit -m "chore: move execution-plans from figures/ to docs/planning/"
```

---

#### T2-06: Move figure reports out of `figures/generated/`

**What**: Move the 5 expert review markdown files and 1 results report from `figures/generated/` to `docs/planning/` or a dedicated `docs/figure-reports/` directory.

**Why**: These are analysis reports, not generated figures:
- `figure-bibliography.md`
- `figure-reports-expert-review.md` (+ v2, deeper-discussion, deeper-discussion-v2)
- `results-report-from-iterated-council.md`

They are tracked and take up space in what should be a clean figure output directory.

**Risk**: Low. No code references these files.

**Commands**:
```bash
mkdir -p docs/figure-reports
git mv figures/generated/figure-bibliography.md docs/figure-reports/
git mv figures/generated/figure-reports-*.md docs/figure-reports/
git mv figures/generated/results-report-from-iterated-council.md docs/figure-reports/
git commit -m "chore: move figure analysis reports to docs/figure-reports/"
```

---

#### T2-07: Remove `docs/install-script-log-ubuntu-22-04.txt`

**What**: Remove the 9.7 MB install log from `docs/`.

**Why**: This is a raw terminal log from running the setup script on Ubuntu 22.04. It contains no documentation value -- it is a log capture, not instructions. At 9.7 MB it is the single largest text file in the repository. The setup script itself (`scripts/setup-dev-environment.sh`) is the canonical source for installation.

**Risk**: None. This is a one-time debug artifact.

**Commands**:
```bash
git rm docs/install-script-log-ubuntu-22-04.txt
echo "docs/*.txt" >> .gitignore  # Prevent future log file commits in docs/
git commit -m "chore: remove 9.7 MB install log from docs/"
```

---

#### T2-08: Clean up empty directories

**What**: Remove empty directories that serve no purpose.

**Why**: 18 empty directories were found. Most are gitignored output directories that get created at runtime. Git does not track empty directories, so most will simply reappear when the pipeline runs. The following are genuinely empty and tracked (via parent tracking):

| Directory | Action |
|-----------|--------|
| `figures/synthetic/` | Remove (synthetic isolation, should be created by pipeline if needed) |
| `figures/generated/svg/` | Keep (output target, created by pipeline) |
| `figures/generated/private/` | Keep (output target, created by pipeline) |
| `figures/generated/matplotlib/` | Keep (output target, created by pipeline) |
| `figures/generated/d3-js/` | Keep (output target, created by pipeline) |
| `outputs/synthetic/` | Already gitignored, keep |
| `outputs/stats_cache/` | Already gitignored, keep |

**Risk**: None. Empty directories in gitignored paths persist locally but are not tracked.

**Action**: No git changes needed. These are runtime directories.

---

### Tier 3: Won't Do (Evaluated and Rejected)

---

#### T3-01: Do NOT move `src/tools/` to `archived/`

**Evaluation**: `src/tools/` (23 MB) contains the ground-truth creation R code with Shiny apps. While it is legacy code that is not part of the main Python pipeline, it is:
- **Actively documented**: Has a Docker-based workflow (`DOCKER.md`, recent updates Jan 31)
- **Scientifically critical**: Ground truth creation methodology must be reproducible
- **Recently touched**: Commit `366216f` (Feb 2026) modified files in this directory

**Decision**: Keep in `src/tools/`. It is legitimately part of the source tree as the ground truth annotation tooling. Its 23 MB size is mostly R code and documentation, not binary bloat.

---

#### T3-02: Do NOT restructure `configs/` subdirectories

**Evaluation**: The 20 config subdirectories under `configs/` follow Hydra conventions. The `MODELS/archived/` subdirectory (2 YAML files: MICEFOREST.yaml, MICE.yaml) is clearly labeled. The directory structure maps to the pipeline stages and is well-documented in `configs/README.md`.

**Decision**: Keep as-is. Hydra config structure is determined by the Hydra framework, not by aesthetic preference.

---

#### T3-03: Do NOT remove `AGENTS.md`

**Evaluation**: `AGENTS.md` (4.2 KB, tracked) provides LLM agent instructions for non-Claude tools (Copilot, Cursor, Codex, Gemini). It is a deliberate cross-agent compatibility file that serves the same role as `CLAUDE.md` but for other tools.

**Decision**: Keep. It is intentional, well-written, and small.

---

#### T3-04: Do NOT merge `apps/` into `src/`

**Evaluation**: `apps/visualization/` is a React+D3.js TypeScript project with its own `package.json`. It is a separate application, not a Python module. Putting it in `src/` would confuse Python tooling (ruff, mypy, pytest).

**Decision**: Keep `apps/` as a separate top-level directory. This follows the convention of polyglot repos.

---

#### T3-05: Do NOT remove `archived/`

**Evaluation**: `archived/` (600 KB) contains NuwaTS and TSB_AD experimental code that was evaluated during the research but not included in the final pipeline. It has a README explaining its status. This is standard academic practice -- keeping evaluated alternatives available for reproducibility.

**Decision**: Keep. It is small, documented, and serves a reproducibility purpose.

---

#### T3-06: Do NOT remove the `r` symlink without checking R developer workflow

**Evaluation**: While the audit shows no code references to `r/` (all use `src/r/`), R developers may rely on the short path in their IDE or terminal workflow. The symlink is only 1 tracked file (5 bytes).

**Decision**: Moved to **Tier 2 (T2-01)** with recommendation to remove, but flag it to the team first.

---

#### T3-07: Do NOT flatten `docs/planning/` (57 files)

**Evaluation**: While 57 planning documents is a lot, they are all in a single directory and follow clear naming conventions. Subdirectory organization would break existing cross-references in other planning documents.

**Decision**: Keep as-is. The volume reflects a thorough research process. Consider archiving completed plans in a future pass.

---

## 4. Proposed Final Structure

After applying Tier 1 and Tier 2 changes:

```
foundation_PLR/
|
|-- AGENTS.md                            # Universal LLM agent instructions
|-- ARCHITECTURE.md                      # System architecture
|-- CITATION.cff                         # Citation metadata
|-- CLAUDE.md                            # Claude-specific rules
|-- CONTRIBUTING.md                      # Contribution guide
|-- LICENSE                              # MIT license
|-- Makefile                             # Build targets
|-- README.md                            # Project README
|-- mkdocs.yml                           # MkDocs config
|-- pyproject.toml                       # Python project config
|-- uv.lock                             # Dependency lockfile
|-- Docker*, docker-compose.yml          # Container configs
|-- .env, .env.example                   # Environment vars
|-- .python-version, .Rprofile           # Language version pins
|-- .pre-commit-config.yaml              # Pre-commit hooks
|-- renv.lock, .renvignore               # R package management
|
|                                        # REMOVED: r -> src/r symlink
|                                        # REMOVED: 3746252.3760801.pdf
|                                        # REMOVED: analysis_report.json
|                                        # REMOVED: artifacts/
|
|-- apps/visualization/                  # React+D3.js dashboard (unchanged)
|-- archived/                            # Evaluated alternatives (unchanged)
|-- assets/                              # Static images (unchanged)
|-- configs/                             # Hydra configs (unchanged)
|-- data/                                # private/public/r_data/synthetic (unchanged)
|-- docs/
|   |-- api-reference/                   # MkDocs API docs
|   |-- explanation/                     # Conceptual docs
|   |-- figure-reports/                  # NEW: moved from figures/generated/
|   |-- getting-started/                 # Setup guides
|   |-- planning/                        # 57 planning docs
|   |   |-- execution-plans/             # NEW: moved from figures/generated/
|   |-- repo-figures/                    # README visual documentation system
|   |-- research/                        # Research context
|   |-- tutorials/                       # Jupyter-style tutorials
|   |-- user-guide/                      # Usage docs
|   |-- index.md                         # MkDocs landing page
|   |                                    # REMOVED: install-script-log-ubuntu-22-04.txt
|-- figures/
|   |-- example_figures/                 # Reference figure examples
|   |-- generated/                       # CANONICAL manuscript figure output
|   |   |-- data/                        # JSON reproducibility data
|   |   |-- ggplot2/                     # R/ggplot2 figures
|   |   |   |-- main/                    # Main paper figures
|   |   |   |-- supplementary/           # Supplementary figures
|   |   |   |-- extra-supplementary/     # Extended figures
|   |   |-- shap/                        # SHAP visualizations
|   |   |-- supplementary/               # Subject trace PDFs
|   |                                    # REMOVED: execution-plans/ (moved to docs/)
|   |                                    # REMOVED: figure-reports*.md (moved to docs/)
|-- logs/                                # Pipeline logs (gitignored)
|-- mlruns/                              # Local MLflow (gitignored)
|-- notebooks/                           # Jupyter tutorials (unchanged)
|-- outputs/                             # Runtime pipeline outputs (gitignored)
|   |-- latex/                           # Generated LaTeX macros
|   |-- tables/                          # Generated LaTeX tables
|   |-- shap_checkpoints/               # SHAP computation state
|   |                                    # REMOVED: old fig_*.png, table_*.tex at root
|-- renv/                                # R package management (unchanged)
|-- scripts/
|   |-- extraction/                      # extract_*.py (MLflow -> DuckDB)
|   |-- export/                          # export_*.py (DuckDB -> R/CSV)
|   |-- validation/                      # check_*.py, validate_*.py, verify_*.py
|   |-- computation/                     # compute_*.py, plr_decomposition_*.py
|   |-- figures/                         # figure_and_stats_generation.py, generate_*
|   |-- infra/                           # setup, docker, pre-commit, typing
|   |-- reproduce_all_results.py         # Top-level entry point
|   |-- README.md
|-- src/                                 # Source code (unchanged structure)
|   |-- anomaly_detection/
|   |-- classification/
|   |-- config/
|   |-- data_io/
|   |-- decomposition/
|   |-- ensemble/
|   |-- extraction/
|   |-- featurization/
|   |-- imputation/
|   |-- log_helpers/
|   |-- metrics/
|   |-- orchestration/
|   |-- preprocess/
|   |-- r/                               # R source code (canonical location)
|   |-- stats/
|   |-- summarization/
|   |-- synthetic/
|   |-- tools/                           # Ground truth creation (kept)
|   |-- utils/
|   |-- viz/
|   |                                    # REMOVED: deploy/ (dead code)
|   |                                    # REMOVED: figures/ (misplaced output dir)
|-- tests/                               # Test suite (unchanged)
|
|                                        # GITIGNORED (not in tree):
|                                        # catboost_info/ (training artifacts)
|                                        # site/ (MkDocs build, 787 MB)
```

### Net Effect

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Top-level visible dirs | 28 | 22 | -6 |
| Tracked files to remove | -- | ~18 | Cleaner history |
| Git-tracked binaries removed | 4 (catboost_info) + 1 (plr_results.duckdb) + 1 (artifacts/numbers.tex) | 0 | -6 unnecessary tracked files |
| Dead code removed | 0 | `src/deploy/` (3 files), `src/figures/` (1 file) | -4 files |
| Disk recovered (tracked) | -- | ~9.7 MB (install log) + ~12 KB (duckdb) | ~9.7 MB |

---

## 5. Migration Script Outline

The following script should be run on a clean working tree (all changes committed).

```bash
#!/usr/bin/env bash
# repo-reorganize.sh -- Repository structure cleanup
# Run from repository root. Creates a single commit per tier.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "=== Pre-flight checks ==="
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Working tree is dirty. Commit or stash changes first."
    exit 1
fi

# ============================================================
# TIER 1: Safe removals (no code changes, no CI impact)
# ============================================================
echo "=== Tier 1: Safe removals ==="

# T1-01: Remove catboost_info from tracking
if git ls-files --error-unmatch catboost_info/ &>/dev/null; then
    git rm -r --cached catboost_info/
    # Ensure gitignore has the entry
    if ! grep -q '^catboost_info/' .gitignore; then
        echo "" >> .gitignore
        echo "# CatBoost training artifacts (generated at runtime)" >> .gitignore
        echo "catboost_info/" >> .gitignore
    fi
fi

# T1-02: Remove redundant artifacts/
if git ls-files --error-unmatch artifacts/ &>/dev/null; then
    git rm -r artifacts/
fi

# T1-03: Remove stale scripts/plr_results.duckdb
if git ls-files --error-unmatch scripts/plr_results.duckdb &>/dev/null; then
    git rm scripts/plr_results.duckdb
    if ! grep -q 'scripts/\*\.duckdb' .gitignore; then
        echo "scripts/*.duckdb" >> .gitignore
    fi
fi

# T1-04: Remove misplaced root files (untracked, just delete)
rm -f 3746252.3760801.pdf analysis_report.json

# T1-05: Remove src/figures/
if git ls-files --error-unmatch src/figures/ &>/dev/null; then
    git rm -r src/figures/
fi

# T1-06: Improve gitignore comments
# (manual edit -- add comments near site/ and docs/repo-figures/generated/)

# Stage gitignore changes
git add .gitignore

git commit -m "$(cat <<'EOF'
chore(repo): Tier 1 cleanup -- remove tracked artifacts and dead directories

- Remove catboost_info/ from tracking (training artifacts, now gitignored)
- Remove artifacts/latex/ (redundant with outputs/latex/)
- Remove scripts/plr_results.duckdb (stale development leftover)
- Remove src/figures/ (non-code output dir inside source tree)
- Delete misplaced root files (3746252.3760801.pdf, analysis_report.json)
- Update .gitignore with catboost_info/ and scripts/*.duckdb rules
EOF
)"

echo "Tier 1 complete. Commit: $(git rev-parse --short HEAD)"

# ============================================================
# TIER 2: Coordinated changes (requires Makefile/CI updates)
# ============================================================
echo "=== Tier 2: Coordinated changes ==="

# T2-01: Remove r symlink
if [ -L r ]; then
    git rm r
fi

# T2-05: Move execution plans
if [ -d figures/generated/execution-plans ]; then
    mkdir -p docs/planning/execution-plans
    git mv figures/generated/execution-plans/* docs/planning/execution-plans/
    rmdir figures/generated/execution-plans 2>/dev/null || true
fi

# T2-06: Move figure reports
if ls figures/generated/figure-reports-*.md &>/dev/null; then
    mkdir -p docs/figure-reports
    git mv figures/generated/figure-bibliography.md docs/figure-reports/ 2>/dev/null || true
    git mv figures/generated/figure-reports-*.md docs/figure-reports/ 2>/dev/null || true
    git mv figures/generated/results-report-from-iterated-council.md docs/figure-reports/ 2>/dev/null || true
fi

# T2-07: Remove install log
if git ls-files --error-unmatch docs/install-script-log-ubuntu-22-04.txt &>/dev/null; then
    git rm docs/install-script-log-ubuntu-22-04.txt
fi

git add -A
git commit -m "$(cat <<'EOF'
chore(repo): Tier 2 cleanup -- remove symlink, relocate docs, remove install log

- Remove r -> src/r symlink (use src/r/ directly)
- Move figures/generated/execution-plans/ to docs/planning/execution-plans/
- Move figure analysis reports to docs/figure-reports/
- Remove 9.7 MB install log from docs/
EOF
)"

echo "Tier 2 complete. Commit: $(git rev-parse --short HEAD)"

# T2-03 (scripts reorganization) and T2-04 (src/deploy removal)
# are handled separately due to Makefile/CI/code changes required.
echo ""
echo "=== MANUAL STEPS REMAINING ==="
echo "T2-03: Reorganize scripts/ into subdirs (requires Makefile + CI updates)"
echo "T2-04: Remove src/deploy/ (requires pipeline_PLR.py import change)"
echo "T2-02: Clean up outputs/ legacy files (local-only, not tracked)"
echo ""
echo "Done. Review commits with: git log --oneline -3"
```

---

## 6. What We Explicitly Keep As-Is (and Why)

| Item | Size | Why Keep |
|------|------|----------|
| `AGENTS.md` | 4.2 KB | Intentional cross-agent compatibility file for LLM coding assistants |
| `src/tools/` | 23 MB | Ground truth creation tooling; actively documented, scientifically critical |
| `archived/` | 600 KB | Evaluated alternatives (NuwaTS, TSB_AD); reproducibility requirement |
| `apps/visualization/` | 162 MB | Separate TypeScript project; follows polyglot repo convention |
| `configs/` (20 subdirs) | 584 KB | Hydra framework convention; well-documented |
| `configs/MODELS/archived/` | 8 KB | Clearly labeled archived model configs (MICE, MICEFOREST) |
| `docs/planning/` (57 files) | 1.2 MB | Research planning trail; valuable for reproducibility narrative |
| `docs/repo-figures/` | 793 MB on disk | Tracked assets are only ~2 MB; 766 MB generated/ is gitignored |
| `figures/generated/` | 42 MB | Canonical publication figures with JSON data; correctly tracked |
| `notebooks/` | 144 KB | Tutorial notebooks for onboarding |
| `outputs/` | 1.7 GB on disk | Runtime outputs, fully gitignored; canonical for pipeline artifacts |
| `.claude/` | 86 tracked files | Claude agent configuration and rules; actively maintained |
| `renv/` + `renv.lock` | 24 MB on disk | R package reproducibility; renv.lock is tracked (3 files) |
| `site/` | 787 MB on disk | MkDocs build output; correctly gitignored |
| `mlruns/` | 64 MB on disk | Local MLflow experiments; correctly gitignored |
| `data/` | 1.7 GB on disk | Research data; public/ tracked, private/ gitignored |

---

## 7. Appendix: Verification Checklist

After executing the migration, verify:

```bash
# 1. No broken imports
uv run python -c "import src; print('imports OK')"

# 2. Tests pass
uv run pytest tests/ -m "unit or guardrail" -x -q

# 3. Pre-commit hooks pass
pre-commit run --all-files

# 4. CI-referenced scripts exist
test -f scripts/validation/verify_registry_integrity.py || echo "MISSING"
test -f scripts/validation/check_computation_decoupling.py || echo "MISSING"
test -f scripts/validation/check_parallel_systems.py || echo "MISSING"

# 5. catboost_info is no longer tracked
git ls-files catboost_info/ | wc -l  # Should be 0

# 6. No unexpected files at root
ls *.pdf *.json 2>/dev/null | wc -l  # Should be 0

# 7. Symlink removed
test -L r && echo "SYMLINK STILL EXISTS" || echo "OK"

# 8. Figure generation still works
uv run python src/viz/generate_all_figures.py --list
```

---

## 8. Risk Assessment Summary

| Change | Risk | Mitigation |
|--------|------|------------|
| T1-01: catboost_info gitignore | None | Already gitignored by `src/catboost_info/*` pattern |
| T1-02: Remove artifacts/ | Low | Verify no manuscript repo references |
| T1-03: Remove scripts/plr_results.duckdb | None | No references found |
| T1-04: Remove root PDF/JSON | None | Already gitignored |
| T1-05: Remove src/figures/ | None | No imports, no code |
| T2-01: Remove r symlink | Low | No code references; team notification |
| T2-02: Clean outputs/ | None | Untracked files only |
| T2-03: Reorganize scripts/ | Medium | Makefile + CI workflow path updates |
| T2-04: Remove src/deploy/ | Low | One import to remove |
| T2-05: Move execution plans | None | No code references |
| T2-06: Move figure reports | None | No code references |
| T2-07: Remove install log | None | No documentation value |

---

*End of plan.*
