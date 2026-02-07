# CLAUDE.md - Foundation PLR Project Context

## Project Overview

**Repository**: `foundation_PLR` - Foundation Models for Pupillary Light Reflex (PLR) Analysis.
Evaluates whether generic time-series foundation models (MOMENT, UniTS, TimesNet, SAITS) improve biosignal preprocessing for glaucoma screening compared to traditional methods (LOF, SVM, linear interpolation).

## Research Question (Details: `.claude/rules/00-research-question.md`)

> How do preprocessing choices (outlier detection x imputation) affect ALL STRATOS-compliant downstream metrics when using handcrafted physiological features?

Fix the classifier (CatBoost). Vary the preprocessing. Measure effects on discrimination, calibration, and clinical utility. NOT about comparing classifiers. NOT about AUROC alone.

## The Pipeline

```
Raw PLR Signal → [1] Outlier Detection (11 methods) → [2] Imputation (8 methods)
    → [3] Featurization (FIXED: handcrafted) → [4] Classification (FIXED: CatBoost)
    → ALL STRATOS Metrics: AUROC, calibration slope/intercept/O:E, Brier, Net Benefit, DCA
```

Errors at each stage propagate downstream. FM embeddings underperform handcrafted features by 9pp.

## Critical Rules Summary

| # | Rule | Detail Location |
|---|------|-----------------|
| 1 | **Research question**: Fix classifier, vary preprocessing | `.claude/rules/00-research-question.md` |
| 2 | **Registry = single source of truth**: 11 outlier, 8 imputation, 5 classifier | `.claude/rules/05-registry-source-of-truth.md` |
| 3 | **Figure rules**: Load combos from YAML, max 4 curves, ground truth required | `.claude/rules/10-figures.md` |
| 4 | **STRATOS metrics**: ALL 5 domains, not AUROC-only | `.claude/rules/15-stratos-metrics.md` |
| 5 | **Package management**: uv only, conda/pip BANNED | `.claude/rules/20-package-management.md` |
| 6 | **No reimplementation**: Use verified libraries via interop | `.claude/rules/25-no-reimplementation.md` |
| 7 | **Zero tolerance figure QA**: `pytest tests/test_figure_qa/ -v` before any figure commit | `.claude/CLAUDE.md` |
| 8 | **Fix at source**: Data issues fixed in extraction (MLflow→DuckDB), never downstream | `.claude/CLAUDE.md` |
| 9 | **Computation decoupling**: `src/viz/` reads DuckDB only, never computes metrics | `.claude/CLAUDE.md` |
| 10 | **Anti-hardcoding**: No hex colors, literal paths, method names, or dimensions in code | `.claude/CLAUDE.md` |
| 11 | **No shortcuts**: Academic repo, full robustness only, read existing code first | `.claude/CLAUDE.md` |
| 12 | **grep/sed/awk BANNED** for structured data: Use AST/YAML parsers | See below |
| 13 | **DevEx**: Automate everything, no copy-paste walls, one command setup | See below |
| 14 | **Proactive behavior**: Plan before executing, propose alternatives, raise issues | See below |
| 15 | **Academic rigor over speed**: Correctness first, no lazy shortcuts for CI/build speed | See below |

## Figure QA (CRITICAL-FAILURE-001)

NEVER use synthetic data for scientific figures. Run `pytest tests/test_figure_qa/ -v` before any figure is finalized. Tests catch: data provenance fraud, statistical invalidity, rendering issues, publication standards, accessibility.

See: `.claude/docs/meta-learnings/CRITICAL-FAILURE-001-synthetic-data-in-figures.md`

## Fix Issues at Source (CRITICAL-FAILURE-002)

```
MLflow → DuckDB → CSV → R → Figure
  ↑ FIX HERE    ↑ NEVER   ↑ NEVER  ↑ NEVER
```

**Extraction validation**: No duplicates, only handcrafted features, ground truth AUROC = 0.911, total configs = 316.

See: `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md`

## Figure Reproducibility with JSON Data

Every figure MUST include a JSON data file for reproducibility. Use `save_figure(fig, 'name', data=data_dict)` from `plot_config.py`. JSON must contain all numeric data, summary statistics, sample sizes. Subject-level JSON files are PRIVATE (gitignored for patient privacy).

## Data Provenance

**Source**: Najjar et al. 2023, Br J Ophthalmol (DOI: 10.1136/bjophthalmol-2021-319938)

| Dataset | N | AUROC | Notes |
|---------|---|-------|-------|
| Najjar original | 322 | 0.94 | Full Singapore dataset (SNEC) |
| Our subset (classify) | 208 | 0.913 | 152 control + 56 glaucoma |
| Our subset (preprocess) | 507 | N/A | All with ground truth masks |

DO NOT compare our AUROC directly to Najjar's (different subset, different goal).

## Subject Counts Per Task

| Task | N | Breakdown |
|------|---|-----------|
| Outlier Detection | **507** | All subjects have ground truth outlier masks |
| Imputation | **507** | All subjects have ground truth denoised signals |
| Classification | **208** | 152 Control + 56 Glaucoma (labeled subset) |

299 subjects have outlier/imputation ground truth but no classification labels.

## Key Findings

| Finding | Value |
|---------|-------|
| Best AUROC | 0.913 (ground truth + CatBoost) |
| Preprocessing effect | eta-squared=0.15 |
| Handcrafted vs Embeddings | 0.830 vs 0.740 (9pp gap) |
| FM for preprocessing | Competitive with traditional |

## Critical Data Sources

| Source | Path |
|--------|------|
| Raw PLR + GT | `/home/petteri/Dropbox/github-personal/foundation-PLR/SERI_PLR_GLAUCOMA.db` |
| MLflow results | `/home/petteri/mlruns/` (542 pickles, 1000 bootstrap each) |
| Extracted metrics | `manuscripts/foundationPLR/data/cd_diagram_data.duckdb` |

## Sister Repositories

| Repo | Path |
|------|------|
| Manuscript | `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/latent-methods-results/` |
| Literature | `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/appendix-literature-review/` |
| Bibliography | `/home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-pupil/` |
| Master plan | `/home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR/planning/latent-method-results-update.md` |

## Dependencies

| Language | Manager | Notes |
|----------|---------|-------|
| Python 3.11+ | `uv` | conda/pip BANNED. See `.claude/rules/20-package-management.md` |
| R >= 4.4 | `install.packages()` | System R from CRAN. pminternal for stability analysis |
| TypeScript/JS | `npm` | Node.js 20 LTS for visualization |

Core Python: DuckDB, pandas, numpy, matplotlib, scipy, MLflow.

## Configuration Architecture

All configuration in `configs/` via Hydra. NEVER create parallel config dirs or hardcode config values.

| Config Path | Value | Description |
|-------------|-------|-------------|
| `CLS_EVALUATION.glaucoma_params.prevalence` | 0.0354 | Disease prevalence (Tham 2014) |
| `CLS_EVALUATION.BOOTSTRAP.n_iterations` | 1000 | Bootstrap iterations |
| `CLS_EVALUATION.BOOTSTRAP.alpha_CI` | 0.95 | Confidence interval |
| `VISUALIZATION.dpi` | 100 | Figure DPI |

## CD Diagrams: CORRECT Usage

Compare **preprocessing pipelines** with CatBoost fixed. NEVER compare classifiers.
1. Outlier method comparison (CatBoost + best imputation fixed)
2. Imputation method comparison (CatBoost + ground truth outliers fixed)
3. Full pipeline comparison (outlier x imputation, CatBoost fixed)

## grep/sed/awk BAN for Structured Data

| Tool | BANNED For | Use Instead |
|------|-----------|-------------|
| `grep`/`sed`/`awk` | Searching/editing Python/YAML/JSON | AST parsing, proper YAML parser |
| `import re` | Parsing Python/YAML/JSON | `ast.parse()` + `ast.walk()` |

See: `.claude/docs/meta-learnings/VIOLATION-001-regex-for-code-parsing.md`

## DevEx: Automate Everything

One command to set up: `sudo ./scripts/infra/setup-dev-environment.sh`. NEVER give copy-paste walls of manual commands. Put multi-step processes in scripts.

## Proactive Behavior

Plan before executing. Propose alternatives. Check existing code/config FIRST before creating anything new. Use TaskCreate for multi-step tasks. NEVER be a passive observer.

## Papers to Reference

| Topic | Key Paper |
|-------|-----------|
| PLR Benchmark | Najjar et al. 2023 Br J Ophthalmol |
| STRATOS Metrics | Van Calster et al. 2024 |
| TRIPOD+AI | Collins et al. 2024 |
| Model Stability | Riley 2023 BMC Medicine (pminternal) |
| Sample Size | Legha 2026 JCE |

## Advanced Analyses (Beyond STRATOS Basics)

| Analysis | Reference | Implementation |
|----------|-----------|----------------|
| Model instability | Riley 2023 (pminternal) | `src/stats/pminternal_wrapper.py` |
| Per-patient uncertainty | Kompa 2021 | Bootstrap prediction distributions |
| Selective classification | Barrenada 2025, Geifman 2017 | AURC, Risk-Coverage plots |

Metric sets defined in `src/viz/metric_registry.py` (STRATOS Core, Discrimination, Calibration, Clinical Utility, Outlier Detection, Imputation).

## Academic Rigor Over Speed

This repository is being frozen for academic publication. **Correctness and rigor always take priority over build speed or convenience.**

- **No `continue-on-error`**: CI failures must be genuinely fixed, never masked
- **No disabling features for speed**: If a tool (mkdocstrings, strict mode, pre-commit) fails, fix the root cause
- **Install real dependencies**: Don't skip deps to make builds faster if it means features don't work
- **Strict mode everywhere**: `mkdocs build --strict`, `pytest --strict-markers`, ruff checks — all enforced
- **Fix warnings, don't suppress them**: Every griffe warning, deprecation, or lint issue gets a proper fix
- **This is NOT a startup**: No "ship fast, fix later" mentality. Every commit is publication-ready

## Future Vision

End-to-end probabilistic PLR reconstruction: a single model that outputs mu(t), sigma-squared(t) per timepoint, incorporating pupil segmentation uncertainty (SAMv3), learned blink detection, and full uncertainty propagation. Self-supervised pretraining on 507 subjects (1M+ timepoints). Current discrete pipeline stages would be replaced by learned end-to-end reconstruction.
