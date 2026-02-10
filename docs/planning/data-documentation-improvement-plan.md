# Data Documentation Improvement Plan

**Created**: 2026-02-09
**Status**: In Progress
**Branch**: `chore/documentation-proofreading`

## Motivation

The `data/` directory contains the project's core artifacts (DuckDB databases, privacy-controlled lookups, synthetic test data, R exports) but has no README.md. There is also no documentation explaining how PLR data gets adapted to the common TSFM benchmark format (the `data_provider` pattern from UniTS/TimesNet), making onboarding new foundation models unnecessarily hard.

## Deliverables

### 1. `data/README.md`

Ten sections covering:
1. Overview (with fig-repo-79 ER diagram reference)
2. Directory layout (ASCII tree)
3. DuckDB schema reference (10 tables)
4. Data provenance (Najjar 2023, subject counts, privacy tiers)
5. TSFM data format requirements (per-model input specs)
6. The `data_provider` pattern (UniTS/Time-Series-Library common format)
7. How to onboard a new TSFM (7-step checklist)
8. Data versioning (SHA256 vs DVC vs Datalad)
9. Reproducibility commands
10. Cross-references

### 2. Four New Figure Plans

| ID | Title | Priority |
|----|-------|----------|
| fig-data-01 | TSFM Data Adaptation Pipeline | P1 |
| fig-data-02 | Common TS Benchmark Dataset Landscape | P2 |
| fig-data-03 | PLR Data Dictionary Structure | P1 |
| fig-data-04 | Data Versioning Decision Tree | P3 |

### 3. Existing Figures Referenced

The README reuses these existing figure plans rather than duplicating concepts:

| Figure | Topic | Location |
|--------|-------|----------|
| fig-repo-79 | DuckDB table relationships (ER diagram) | `docs/repo-figures/figure-plans/` |
| fig-repo-77 | Synthetic vs production isolation boundary | `docs/repo-figures/figure-plans/` |
| fig-repo-78 | Test data path resolution chain | `docs/repo-figures/figure-plans/` |
| fig-repo-23 | Data privacy classification | `docs/repo-figures/figure-plans/` |
| fig-repro-20 | Why DuckDB (single source of truth) | `docs/repo-figures/figure-plans/` |
| fig-repro-24 | Git LFS vs DuckDB comparison | `docs/repo-figures/figure-plans/` |
| fig-repro-17 | Bitwise vs functional reproducibility | `docs/repo-figures/figure-plans/` |
| fig-repro-22 | JSON sidecars for figure reproducibility | `docs/repo-figures/figure-plans/` |
| fig-repro-14 | Lockfiles as time machine | `docs/repo-figures/figure-plans/` |
| fig-repro-11 | Version pinning strategies | `docs/repo-figures/figure-plans/` |

## Files Created/Modified

| File | Action |
|------|--------|
| `data/README.md` | CREATE |
| `docs/planning/data-documentation-improvement-plan.md` | CREATE (this file) |
| `docs/repo-figures/figure-plans/fig-data-01-tsfm-data-adaptation-pipeline.md` | CREATE |
| `docs/repo-figures/figure-plans/fig-data-02-benchmark-dataset-landscape.md` | CREATE |
| `docs/repo-figures/figure-plans/fig-data-03-data-dictionary-structure.md` | CREATE |
| `docs/repo-figures/figure-plans/fig-data-04-data-versioning-decision-tree.md` | CREATE |

## Execution Order

1. Planning document (this file)
2. Four figure plans (parallel)
3. `data/README.md` (references figure plans)

## Key Reference Files

| File | Information Extracted |
|------|---------------------|
| `src/data_io/streaming_duckdb_export.py` | DuckDB table schemas |
| `src/data_io/ts_format.py` | Format conversion functions |
| `src/data_io/torch_data.py` | Windowing, padding, data selectors |
| `src/data_io/data_utils.py` | MOMENT-specific transforms |
| `configs/MODELS/` | Per-model YAML configs |
| `data/public/DATA_MANIFEST.yaml` | Public data manifest |
| `data/r_data/DATA_MANIFEST.yaml` | R export manifest |
| `docs/planning/reproducibility-and-mlsecops-improvements.md` | DVC deferral rationale |
