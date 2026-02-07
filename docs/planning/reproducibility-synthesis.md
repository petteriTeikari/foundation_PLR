# Reproducibility Synthesis: Comprehensive Analysis & Action Plan

**Created**: 2026-01-29
**Status**: ACTIVE - Living Document
**Last Reviewer Round**: Initial Draft

---

## Executive Summary

This document synthesizes **23 planning documents**, **15 critical failure reports**, and current codebase state to provide a unified view of reproducibility enforcement in the Foundation PLR project. The analysis reveals a sophisticated multi-layered enforcement system that is **~70% implemented** but has significant gaps in R enforcement, visual regression testing, and computation decoupling.

### Key Metrics

| Dimension | Status | Evidence |
|-----------|--------|----------|
| Total Test Suite | **1127 tests** | `uv run pytest --collect-only` |
| Guardrail Tests | ~30 tests | `tests/test_guardrails/` (4 files) |
| R Hardcoding Tests | 20 tests | `tests/test_r_figures/test_hardcoding.py` |
| No-Hardcoding Tests | ~40 tests | `tests/test_no_hardcoding/` (6 files) |
| Figure QA Tests | ~30 tests | `tests/test_figure_qa/` (6 files) |
| Pre-commit Hooks | 5 active | ruff, ruff-format, registry-integrity, registry-validation, r-hardcoding |
| CRITICAL Failures | 1/4 active | FAILURE-003 (computation decoupling) |
| Planning Docs Synthesized | 23 | See cross-reference table |

---

## Part 1: Document Cross-Reference Matrix

### Source Documents Analyzed

#### Repository 1: `sci-llm-writer/manuscripts/foundationPLR/planning/`

| Doc ID | Document | Focus | Status |
|--------|----------|-------|--------|
| SLW-01 | `reproducible-mlflow-extraction-to-results.md` | Pipeline architecture | Implemented |
| SLW-02 | `figure-qa-check-plan.md` | Rendering artifacts | Partially done |
| SLW-03 | `methods-figures-planning.md` | Figure requirements | Reference only |

#### Repository 2: `foundation_PLR/docs/planning/`

| Doc ID | Document | Focus | Status |
|--------|----------|-------|--------|
| FPL-01 | `recurring-hardcoded-reproducability-failure-through-lens-of-LLM-paperology.md` | LLM failure analysis | Active |
| FPL-02 | `TDD-zero-hardcoding-plan.md` | Test specifications | **70% implemented** |
| FPL-03 | `hardcoding-guardrails-improvement.md` | Enforcement design | **80% implemented** |
| FPL-04 | `double-check-reproducibility.md` | Audit plan | **Phase 1-2 complete** |
| FPL-05 | `computation-doublecheck-plan-and-restriction-to-mlflow-duckdb-conversion.md` | Architecture | **Phase 1-2 complete** |
| FPL-06 | `audit-2026-01-27-reproducibility-fixes.md` | Implementation log | Complete |
| FPL-07 | `experiment-parameters-guardrail-improvements.md` | Registry system | **Complete** |
| FPL-08 | `figure-qa-third-or-nth-time.md` | Figure combinations | Pending |

#### Meta-Learnings: `.claude/docs/meta-learnings/`

| Doc ID | Document | Severity | Status |
|--------|----------|----------|--------|
| CF-001 | `CRITICAL-FAILURE-001-synthetic-data-in-figures.md` | CRITICAL | **Resolved** |
| CF-002 | `CRITICAL-FAILURE-002-hardcoding-despite-existing-systems.md` | CRITICAL | **Resolved** |
| CF-003 | `CRITICAL-FAILURE-003-computation-decoupling-violation.md` | CRITICAL | **Active** |
| CF-004 | `CRITICAL-FAILURE-004-r-figure-hardcoding.md` | CRITICAL | **Resolved** |
| V-001 | `VIOLATION-001-regex-for-code-parsing.md` | HIGH | **Resolved** |
| F-002 | `FAILURE-002-figure-aesthetic-inconsistency.md` | HIGH | **Active** |
| F-003 | `FAILURE-003-repeated-instruction-amnesia.md` | HIGH | **Resolved** |
| F-005 | `FAILURE-005-featurization-not-filtered.md` | HIGH | **Resolved** |

---

## Part 2: Architecture Analysis

### Current Enforcement Layers (5-Layer Model)

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 5: CI/CD (GitHub Actions)                     [NOT DEPLOYED]  │
│   - Repository-level enforcement                                    │
│   - Artifact generation and testing                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4: Pre-commit Hooks                           [80% COMPLETE]  │
│   ✅ ruff (Python linting)                                          │
│   ✅ ruff-format (Python formatting)                                │
│   ✅ registry-integrity (MLflow parameter validation)               │
│   ✅ r-hardcoding-check (R anti-hardcoding)                         │
│   ⚠️  Missing: TypeScript ESLint, visual regression                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: Test Suite                                 [85% COMPLETE]  │
│   ✅ tests/test_guardrails/ (20 Python tests)                       │
│   ✅ tests/test_r_figures/ (20 R hardcoding tests)                  │
│   ✅ tests/test_registry.py (53 registry tests)                     │
│   ⚠️  Missing: Visual regression, blank figure detection            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2: Configuration System                       [90% COMPLETE]  │
│   ✅ configs/mlflow_registry/ (SSOT for methods)                    │
│   ✅ configs/VISUALIZATION/ (figure specs, colors, combos)          │
│   ✅ src/r/figure_system/ (R config loading)                        │
│   ✅ src/data_io/registry.py (Python config loading)                │
│   ⚠️  Missing: data_filters.yaml standardization                    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Code Implementation                        [70% COMPLETE]  │
│   ✅ Python viz code (no hardcoding)                                │
│   ✅ R figure scripts (no hardcoded dims)                           │
│   ⚠️  Computation decoupling incomplete (Phase 3-4)                 │
│   ⚠️  Some viz code still computes metrics                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Two-Block Architecture (Target State)

```
═══════════════════════════════════════════════════════════════════════
BLOCK 1: EXTRACTION (Compute Once)            Status: 90% Complete
═══════════════════════════════════════════════════════════════════════

  MLflow Pickles                    DuckDB (Single Source of Truth)
  ─────────────────                 ──────────────────────────────────
  /home/petteri/mlruns/             foundation_plr_results.db
  └── 253031330985650090/            ├── essential_metrics (224 rows)
      └── artifacts/                 │   ├── auroc, auroc_ci_lo/hi
          └── bootstrap_metrics.pkl  │   ├── brier, scaled_brier
                                     │   ├── calibration_slope/intercept
                                     │   ├── o_e_ratio
                                     │   ├── net_benefit_5/10/15/20pct
                                     │   └── f1, sensitivity, specificity
                                     ├── roc_curves (per config)
                                     ├── calibration_curves (per config)
                                     └── dca_curves (per config)

  Scripts:
  ✅ scripts/extract_all_configs_to_duckdb.py (main extraction)
  ✅ scripts/export_*.py (14 export scripts for R)

═══════════════════════════════════════════════════════════════════════
BLOCK 2: VISUALIZATION (Read Only)            Status: 60% Complete
═══════════════════════════════════════════════════════════════════════

  DuckDB / JSON                     Figures
  ─────────────────                 ──────────────────────────────────
  data/r_data/*.json                figures/generated/
  data/r_data/*.csv                 ├── ggplot2/main/
                                    ├── ggplot2/supplementary/
  ⚠️ VIOLATIONS:                    └── matplotlib/
  - src/viz/retained_metric.py
    (computes metrics at runtime)
  - src/viz/calibration_plot.py
    (computes slope/intercept)
  - src/viz/dca_plot.py
    (computes net benefit)

  ✅ COMPLIANT:
  - src/r/figures/*.R
    (all read from JSON/CSV)
```

---

## Part 3: Gap Analysis

### Critical Gaps (Must Fix)

| Gap ID | Description | Source Doc | Impact | Effort | Status |
|--------|-------------|------------|--------|--------|--------|
| GAP-01 | Computation in viz layer (14 files) | CF-003 | Non-reproducible figures | High | PENDING |
| GAP-02 | No visual regression testing | F-002 | Silent visual bugs | Medium | PENDING |
| GAP-03 | Blank figure detection missing | CF-002 | Silent failures | Low | ✅ FIXED |
| GAP-04 | Legend size validation missing | CF-003 variant | 50% figure wasted | Low | PENDING |

### High Priority Gaps

| Gap ID | Description | Source Doc | Impact | Effort | Status |
|--------|-------------|------------|--------|--------|--------|
| GAP-05 | CI/CD pipeline not deployed | FPL-03 | No automated enforcement | Medium | PENDING |
| GAP-06 | TypeScript ESLint missing | FPL-03 | apps/visualization/ unguarded | Low | ✅ FIXED |
| GAP-07 | data_filters.yaml incomplete | F-005 | Filter duplication risk | Low | ✅ EXISTS

### Medium Priority Gaps

| Gap ID | Description | Source Doc | Impact | Effort |
|--------|-------------|------------|--------|--------|
| GAP-08 | Figure combinations not done | FPL-08 | 6 figures pending | Medium |
| GAP-09 | Rendering artifact check manual | SLW-02 | [cite:] tags slip through | Low |
| GAP-10 | R precommit package not installed | FPL-01 | No lintr/styler | Low |

---

## Part 4: Test Coverage Analysis

### Current Test Inventory

**Total: 1127 tests collected** (as of 2026-01-29)

```
tests/
├── test_guardrails/                          # ~30 tests (reproducibility guardrails)
│   ├── test_no_hardcoded_values.py          # Hardcoding detection
│   ├── test_json_provenance.py              # JSON metadata validation
│   ├── test_yaml_consistency.py             # YAML config consistency
│   └── test_data_location_policy.py         # Data file locations
│
├── test_no_hardcoding/                       # ~40 tests (TDD zero-hardcoding)
│   ├── test_r_no_hex_colors.py              # R hex color detection
│   ├── test_r_display_names.py              # R display name loading
│   ├── test_r_no_case_when_categorization.py # R categorization via YAML
│   ├── test_yaml_single_source.py           # YAML as SSOT
│   ├── test_method_abbreviations.py         # Method abbreviation loading
│   └── test_python_display_names.py         # Python display names
│
├── test_r_figures/                           # 20 tests (R figure compliance)
│   └── test_hardcoding.py                   # 4 test classes, 20 methods
│
├── test_figure_qa/                           # ~30 tests (figure quality)
│   ├── test_data_provenance.py              # Data source tracking
│   ├── test_statistical_validity.py         # Metric validity
│   ├── test_visual_rendering.py             # Visual checks
│   ├── test_publication_standards.py        # DPI, dimensions
│   ├── test_accessibility.py                # Colorblind safety
│   └── test_no_nan_ci.py                    # CI value validation
│
├── test_registry.py                          # ~50 tests (registry validation)
├── test_data_integrity.py                    # ~8 tests (data quality)
│
└── unit/                                     # ~800+ tests (core functionality)
    ├── test_metrics_utils.py
    ├── test_calibration_extended.py
    ├── test_clinical_utility.py
    ├── test_scaled_brier.py
    └── ... (40+ test files)
```

### Missing Test Coverage

| Category | Missing Tests | Priority | Notes |
|----------|---------------|----------|-------|
| Visual Regression | Compare PNG to golden images | HIGH | Requires golden baseline creation |
| Blank Detection | File size + pixel variance | HIGH | Can implement immediately |
| Legend Size | Max legend proportion 30% | MEDIUM | Requires image analysis |
| Computation Decoupling | No sklearn imports in viz plot code | HIGH | AST-based scanning needed |

### Existing Test Coverage (Already Implemented)

| Category | Test Location | Status |
|----------|---------------|--------|
| R Hardcoding | `tests/test_r_figures/test_hardcoding.py` | ✅ 20 tests |
| R Display Names | `tests/test_no_hardcoding/test_r_display_names.py` | ✅ |
| R Hex Colors | `tests/test_no_hardcoding/test_r_no_hex_colors.py` | ✅ |
| YAML SSOT | `tests/test_no_hardcoding/test_yaml_single_source.py` | ✅ |
| Python Hardcoding | `tests/test_guardrails/test_no_hardcoded_values.py` | ✅ |
| JSON Provenance | `tests/test_guardrails/test_json_provenance.py` | ✅ |
| Registry Validation | `tests/test_registry.py` | ✅ ~50 tests |
| Data Integrity | `tests/test_data_integrity.py` | ✅ ~8 tests |

---

## Part 5: Action Plan

### Phase 1: Computation Decoupling (CRITICAL)

**Timeline**: 2-3 hours
**Source**: CF-003, FPL-05

#### Tasks

- [ ] **1.1** Add pre-computed curves to DuckDB
  - `roc_curves` table: (config_id, fpr[], tpr[])
  - `calibration_curves` table: (config_id, bin_midpoints[], observed[])
  - `dca_curves` table: (config_id, thresholds[], nb_model[], nb_all[])

- [ ] **1.2** Refactor visualization code
  - `src/viz/retained_metric.py` → Read AURC from DB
  - `src/viz/calibration_plot.py` → Read curves from DB
  - `src/viz/dca_plot.py` → Read curves from DB

- [ ] **1.3** Add enforcement test
  ```python
  # tests/test_guardrails/test_computation_decoupling.py
  def test_no_sklearn_in_viz():
      """Viz code must not import sklearn metrics."""
      BANNED_IMPORTS = ['roc_auc_score', 'brier_score_loss', 'calibration_curve']
      # AST scan src/viz/*.py
  ```

- [ ] **1.4** Add pre-commit hook
  ```yaml
  - repo: local
    hooks:
      - id: computation-decoupling
        name: No metric computation in visualization
        entry: python scripts/check_computation_decoupling.py
        language: python
        files: ^src/viz/.*\.py$
  ```

### Phase 2: Visual Quality Enforcement (HIGH)

**Timeline**: 1-2 hours
**Source**: F-002, CF-002 variant

#### Tasks

- [x] **2.1** Implement blank figure detection ✅ (2026-01-29)
  ```python
  # tests/test_figure_qa/test_visual_rendering.py
  def test_figure_has_content_variance():  # IMPLEMENTED
      """All PNGs must have >10KB size and pixel variance."""
      MIN_VARIANCE = 100
      for png in FIGURE_DIR.glob("**/*.png"):
          variance = np.var(np.array(Image.open(png)))
          assert variance > MIN_VARIANCE
  ```

- [ ] **2.2** Implement legend size check
  ```python
  def test_legend_not_dominant():
      """Legend must not exceed 30% of figure width."""
      # Parse R script for legend.position settings
      # Or use image analysis to detect legend region
  ```

- [ ] **2.3** Create golden image baseline
  ```
  tests/golden_images/
  ├── fig_stratos_core.png
  ├── fig_multi_metric_raincloud.png
  └── ...
  ```

- [ ] **2.4** Visual regression test
  ```python
  def test_visual_regression():
      """Compare current figures to golden baselines."""
      for golden in GOLDEN_DIR.glob("*.png"):
          current = FIGURE_DIR / golden.name
          diff = image_diff(golden, current)
          assert diff < 0.01  # 1% pixel difference threshold
  ```

### Phase 3: CI/CD Deployment (MEDIUM)

**Timeline**: 1 hour
**Source**: FPL-03

#### Tasks

- [ ] **3.1** Create GitHub Actions workflow
  ```yaml
  # .github/workflows/quality-checks.yml
  name: Quality Checks
  on: [push, pull_request]
  jobs:
    tests:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - name: Install uv
          run: pip install uv
        - name: Install dependencies
          run: uv sync
        - name: Run tests
          run: uv run pytest
        - name: Run pre-commit
          run: uv run pre-commit run --all-files
  ```

- [ ] **3.2** Add figure generation job
  ```yaml
    figures:
      runs-on: ubuntu-latest
      steps:
        - name: Generate figures
          run: uv run python src/viz/generate_all_figures.py
        - name: Run visual regression
          run: uv run pytest tests/test_figure_qa/
  ```

### Phase 4: Remaining Fixes (LOW-MEDIUM)

**Timeline**: 1-2 hours
**Source**: Multiple

#### Tasks

- [ ] **4.1** Install R precommit package
  ```bash
  # In R console
  install.packages("precommit")
  precommit::use_precommit()
  ```

- [x] **4.2** Add TypeScript ESLint ✅ (2026-01-29)
  ```bash
  # IMPLEMENTED: apps/visualization/eslint.config.js (flat config format)
  # Added to package.json: eslint, typescript-eslint, eslint-plugin-react*
  npm install  # Run to install new dependencies
  npm run lint  # To run ESLint
  ```

- [ ] **4.3** Create figure combinations (6 figures)
  - `fig_prob_dist_combined` (1x2)
  - `fig_calibration_dca_combined` (1x2) ✅ Done
  - `fig_shap_importance_combined` (1x2) ✅ Done
  - `fig_vif_combined` (1x2) ✅ Done
  - `fig_selective_classification` (1x3) ✅ Done
  - `fig_roc_rc_combined` (1x2) ✅ Done

- [x] **4.4** Standardize data_filters.yaml ✅ (Already exists)
  ```yaml
  # configs/VISUALIZATION/data_filters.yaml (ALREADY EXISTS)
  defaults:
    featurization: "simple1.0"
    classifier: "CATBOOST"  # Note: uppercase as in DuckDB
  ```

---

## Part 6: Success Criteria

### Definition of Done

| Criterion | Metric | Target |
|-----------|--------|--------|
| Test Coverage | All tests pass | 100% (1127 tests) |
| Pre-commit | All hooks pass | 100% on clean commit |
| Computation Decoupling | No sklearn in src/viz/ plot code | 0 violations (see allowed list) |
| Visual Regression | All figures match golden | <1% pixel diff |
| CI/CD | GitHub Actions green | All jobs pass |
| R Compliance | No hardcoding in R | 0 violations |

**Allowed sklearn usage** (metric utility modules, NOT plot code):
- `src/viz/metric_registry.py` - defines metric computation functions
- `src/stats/*.py` - statistical computation modules

### Verification Commands

```bash
# Full verification suite
uv run pytest tests/ -v                    # All tests pass
uv run pre-commit run --all-files          # All hooks pass
uv run python scripts/check_r_hardcoding.py  # R compliance
uv run python scripts/verify_registry_integrity.py  # Registry valid

# Generate and verify figures
uv run python src/viz/generate_all_figures.py
uv run pytest tests/test_figure_qa/ -v
```

---

## Part 7: Reviewer Feedback Integration

### Round 1: Initial Synthesis

**Status**: Draft complete (2026-01-29)
**Gaps Identified**: 10 (4 critical, 3 high, 3 medium)
**Tests Defined**: 8 new tests needed

### Round 2: Technical Review (Complete)

**Reviewer**: Plan agent technical review
**Date**: 2026-01-29

**Corrections Applied**:

| Issue | Original | Corrected |
|-------|----------|-----------|
| Total test count | "113 tests" | **1127 tests** |
| Guardrail tests | "20 tests in 3 files" | ~30 tests in 4 files |
| Figure QA tests | "12 tests in 3 files" | ~30 tests in 6 files |
| Pre-commit hooks | "5 (list incomplete)" | 5 (corrected names) |
| Computation violations | "3 files" | 5+ files identified |

**Additional Gaps Identified**:

| Gap ID | Description | Severity |
|--------|-------------|----------|
| GAP-11 | `src/viz/metric_vs_cohort.py` computes metrics | MEDIUM |
| GAP-12 | `src/viz/prob_distribution.py` computes AUROC | MEDIUM |
| GAP-13 | Determine if `metric_registry.py` computation is acceptable | LOW |

**Architectural Clarification**:
- `src/viz/metric_registry.py` is a **utility module** defining metric functions - acceptable
- `src/stats/*.py` modules compute metrics - acceptable (extraction layer)
- Plot code in `src/viz/*_plot.py` should NOT compute - violations

### Round 3: Priority Review (Complete)

**Date**: 2026-01-29

**Quick Wins Implemented**:
- ✅ **2.1** Blank figure detection with pixel variance (`test_figure_has_content_variance`)
- ✅ **4.2** TypeScript ESLint config (`apps/visualization/eslint.config.js`)
- ✅ data_filters.yaml already existed (verified)

**Remaining Phases** (per priority):
- **Phase 1**: Computation decoupling (4-6 hours) - CRITICAL
- **Phase 2**: Visual quality enforcement (2-3 hours) - remaining tests
- **Phase 3**: CI/CD enhancement (30 min)

---

## Appendix A: File Reference

### Key Configuration Files

| File | Purpose | SSOT For |
|------|---------|----------|
| `configs/mlflow_registry/parameters/classification.yaml` | Method lists | 11 outliers, 8 imputations, 5 classifiers |
| `configs/VISUALIZATION/figure_registry.yaml` | Figure specs | 35 R figures, dimensions |
| `configs/VISUALIZATION/colors.yaml` | Color palette | All hex colors |
| `configs/VISUALIZATION/plot_hyperparam_combos.yaml` | Standard combos | 4 main + 5 extended |
| `configs/registry_canary.yaml` | Integrity check | Expected counts + SHA256 |

### Key Scripts

| Script | Purpose | Layer |
|--------|---------|-------|
| `scripts/extract_all_configs_to_duckdb.py` | MLflow → DuckDB | Extraction |
| `scripts/check_r_hardcoding.py` | R compliance check | Pre-commit |
| `scripts/verify_registry_integrity.py` | Registry validation | Pre-commit |
| `src/r/figure_system/save_figure.R` | R figure saving | Implementation |
| `src/r/figure_system/config_loader.R` | R config loading | Implementation |

---

## Appendix B: Historical Timeline

| Date | Event | Impact |
|------|-------|--------|
| 2026-01-22 | CRITICAL-FAILURE-001 discovered | Synthetic data in figures |
| 2026-01-24 | Registry guardrail system deployed | 53 tests, pre-commit hook |
| 2026-01-25 | Computation decoupling Phase 1-2 | STRATOS metrics in DB |
| 2026-01-27 | Audit fixes deployed | 20 Python tests passing |
| 2026-01-28 | CRITICAL-FAILURE-004 R hardcoding | Tests + pre-commit added |
| 2026-01-29 | R hardcoding remediation complete | 20 R tests passing |
| 2026-01-29 | This synthesis created | Unified action plan |

---

*Document maintained as part of CRITICAL-FAILURE-004 remediation and ongoing reproducibility enforcement.*
