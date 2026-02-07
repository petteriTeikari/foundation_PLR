# Foundation PLR - Claude Code Behavior Contract

## Quick Reference

| Rule | Enforcement |
|------|-------------|
| **USE THE REGISTRY** | `configs/mlflow_registry/` is SINGLE SOURCE OF TRUTH |
| Use fixed hyperparam combos | `configs/VISUALIZATION/plot_hyperparam_combos.yaml` |
| Use demo subjects | `configs/demo_subjects.yaml` |
| Ground truth in every figure | Required |
| **ZERO TOLERANCE Figure QA** | `pytest tests/test_figure_qa/` before ANY figure commit |
| **NO HARDCODING** | Use config systems for colors, paths, dimensions, method names |
| **COMPUTE IN EXTRACTION ONLY** | `src/viz/` READS DuckDB, NEVER computes metrics |
| **NO SHORTCUTS** | Academic repo = FULL ROBUSTNESS ONLY |
| **EXPLORE EXISTING CODE FIRST** | READ related files BEFORE proposing new implementations |

**Cross-references:** Registry details in `rules/05-registry-source-of-truth.md` | STRATOS metrics in `rules/15-stratos-metrics.md` | Figures in `rules/10-figures.md` | Package management in `rules/20-package-management.md` | No reimplementation in `rules/25-no-reimplementation.md`

---

## CRITICAL: Figure QA (CRITICAL-FAILURE-001)

ALL figure validation failures are CRITICAL. After an incident where Claude generated calibration plots with SYNTHETIC data instead of real predictions, mandatory QA is enforced:

```bash
pytest tests/test_figure_qa/ -v  # MUST pass before any figure is committed
```

**Test categories:** P0=synthetic/fake data, P1=invalid metrics/overlapping elements, P2=DPI/dimensions/fonts, P3=accessibility. ZERO ERRORS ALLOWED across all categories.

NEVER: generate synthetic data for figures, skip QA tests, assume "it looks fine", use fixed seeds that make models identical.
ALWAYS: run QA tests, trace data provenance, verify models are distinguishable, include source hashes in metadata.

---

## CRITICAL: Visual Bug Priority (FAILURE-003)

When a visual bug is reported, FIX IT BEFORE ANY OTHER WORK.

- **Bug-First**: Fix visual bugs IMMEDIATELY. Infrastructure and planning wait.
- **Verify-Output**: After modifying figure code, VIEW the regenerated PNG with `Read` tool.
- **Count-Mentions**: Issue mentioned 2+ times = CRITICAL priority. Drop everything.
- **Compaction-Handoff**: After context compaction, first check for unfixed reported bugs.

---

## CRITICAL: Computation Decoupling (CRITICAL-FAILURE-003)

ALL metric computation happens in extraction. Visualization code READS DuckDB only.

```
[Extraction]                    [Visualization]
MLflow → DuckDB           →      DuckDB → Figures
ALL computation here              READ ONLY
```

**Available in DuckDB:** `calibration_slope`, `calibration_intercept`, `o_e_ratio`, `scaled_brier`, `net_benefit_5pct/10pct/15pct/20pct`

**BANNED in `src/viz/`:**
```python
from sklearn.metrics import roc_auc_score, brier_score_loss  # BANNED
from src.stats.calibration_extended import calibration_slope_intercept  # BANNED
from src.stats.clinical_utility import net_benefit  # BANNED
# CORRECT: read from DuckDB
df = conn.execute("SELECT auroc, calibration_slope FROM essential_metrics").fetchdf()
```

---

## CRITICAL: No Shortcuts (CRITICAL-FAILURE-006)

This is an ACADEMIC repository. "Quick wins" and shortcuts are FORBIDDEN.

**Pre-Implementation Checklist (MANDATORY):**
1. Search for existing implementations in `src/data_io/`, `src/extraction/`
2. Read related modules (e.g., `streaming_duckdb_export.py` has CheckpointManager, MemoryMonitor)
3. Check test files for usage patterns in `tests/`
4. If a reviewer agent recommends an approach, FOLLOW IT

**BANNED:** Proposing "quick wins", implementing without reading existing modules, panic coding, ignoring reviewer recommendations.

---

## CRITICAL: Anti-Hardcoding (CRITICAL-FAILURE-002 + CRITICAL-FAILURE-004)

**Self-check before EVERY code block:**
1. Any hex colors? Use `COLORS` dict / `resolve_color()` / `ECONOMIST_PALETTE`
2. Any literal paths? Use `save_figure()` / `save_publication_figure()`
3. Any method names? Load from YAML combos
4. Any dimensions? Get from figure config

If you cannot answer "NO" to all four, REVISE before presenting.

| If you wrote this | CHANGE TO |
|-------------------|-----------|
| `color = "#006BA2"` | `COLORS["name"]` or `resolve_color("--color-ref")` |
| `width = 14, height = 6` | `fig_config.dimensions.width` |
| `"figures/generated/..."` | `save_figure()` or `save_publication_figure()` |
| `outlier_method = "LOF"` | Load from `plot_hyperparam_combos.yaml` |
| `ggsave(path, ...)` | `save_publication_figure(plot, "name")` |

**Python:** Use `from src.viz.plot_config import setup_style, save_figure, COLORS` -- call `setup_style()` first, use `COLORS["name"]` not hex, use `save_figure()` not `plt.savefig()`.

**R (enforced by pre-commit hook `scripts/validation/check_r_hardcoding.py`):**
- Load colors from YAML via `load_color_definitions()`, NEVER write `"#RRGGBB"`
- Use `save_publication_figure()`, NEVER `ggsave()`
- Use `theme_foundation_plr()`, NEVER custom themes
- Load dimensions from config, NEVER hardcode width/height

---

## CRITICAL: STRATOS Metrics

NOT AUROC-ONLY. See `rules/15-stratos-metrics.md` for full details.

Use `src/viz/metric_registry.py` for all metric definitions. NEVER hardcode metric names. NEVER focus only on AUROC.

---

## Hyperparameter Combos

**NEVER hallucinate combos.** Before proposing ANY combination:
1. CHECK `configs/VISUALIZATION/plot_hyperparam_combos.yaml` for standard combos
2. CHECK `.claude/domains/mlflow-experiments.md` for documented combos
3. VERIFY method names exist in MLflow at `/home/petteri/mlruns`

**Standard 4 combos** (main figures) -- ALWAYS load from YAML:

| ID | Outlier | Imputation | AUROC |
|----|---------|------------|-------|
| ground_truth | pupil-gt | pupil-gt | 0.9110 |
| best_ensemble | Ensemble | CSDI | 0.9130 |
| best_single_fm | MOMENT-gt-finetune | SAITS | 0.9099 |
| traditional | LOF | SAITS | 0.8599 |

Extended combos (supplementary): `moment_full`, `lof_moment`, `timesnet_full`, `units_pipeline`, `simple_baseline` -- details in YAML.

NEVER: hardcode method names, forget ground_truth combo, invent combos without checking MLflow.

---

## Subject Selection & Data Privacy

Demo subjects: `configs/demo_subjects.yaml` -- 8 subjects (4 control H001-H004, 4 glaucoma G001-G004) stratified by outlier percentage.

- Private lookup `data/private/subject_lookup.yaml` maps Hxxx/Gxxx to PLRxxxx (gitignored)
- Public data uses Hxxx/Gxxx codes; original PLRxxxx codes are PRIVATE
- Subject-level JSON files are PRIVATE (excluded from git)

---

## Two-Block Pipeline

| Block | Entry Point | Purpose |
|-------|-------------|---------|
| Extraction | `src/orchestration/flows/extraction_flow.py` | MLflow to DuckDB with re-anonymization |
| Analysis | `src/orchestration/flows/analysis_flow.py` | DuckDB to figures/stats/LaTeX |

Commands: `make reproduce` (full) | `make reproduce-from-checkpoint` (analysis only) | `make extract` | `make analyze`

---

## Figure Generation

See `rules/10-figures.md` for constraints and patterns.

**Before generating ANY figure:** CHECK `figure_registry.yaml`, LOAD combos from YAML, call `setup_style()`, SAVE JSON data, CHECK privacy level.

**Commands:** `python src/viz/generate_all_figures.py [--figure R7|CD|...] [--list]`

---

## Context Awareness -- Prevent Amnesia

**Self-check before any task:**
1. "Am I only thinking about AUROC?" -- Use ALL STRATOS metrics
2. "Does Python already have this?" -- Check `src/stats/` before wrapping R
3. "Is this in the metric registry?" -- Check `MetricRegistry.has('name')`
4. "Am I reimplementing verified code?" -- Use existing libraries via interop
5. "Does this combo exist?" -- Verify in MLflow before using

**Context drift red flags:** Focusing only on AUROC, creating calibration code from scratch, hardcoding method names, comparing classifiers (WRONG QUESTION).

---

## Directory Structure (ENFORCED)

| Content Type | Location | BANNED Alternatives |
|--------------|----------|---------------------|
| Python source | `src/` | `src/scripts/`, root `.py` |
| R source | `src/r/` | `r/` in root |
| Configs | `configs/` | `config/`, root `.yaml` |
| Outputs | `outputs/` | `artifacts/`, `tables/` |
| Planning docs | `.claude/planning/` | `planning/` in root |
| Apps | `apps/` | Root-level app folders |
| Tests | `tests/` | `test/`, `src/tests/` |

Root allows: `README.md`, `Makefile`, `pyproject.toml`, `CLAUDE.md`, `ARCHITECTURE.md`, config dotfiles. Root bans: output files, source code, ad-hoc folders.

---

## MLflow Data Location

- Experiments: `/home/petteri/mlruns`
- Classification runs: `/home/petteri/mlruns/253031330985650090` (410 runs)
- Full docs: `.claude/domains/mlflow-experiments.md`

## Domain Context Files

| Task | Load |
|------|------|
| Figures | `configs/VISUALIZATION/figure_registry.yaml` |
| Visualization | `.claude/domains/visualization.md` |
| MLflow | `.claude/domains/mlflow-experiments.md` |
| Manuscript | `.claude/domains/manuscript.md` |

## Enforcement

Violations tracked via `.claude/scripts/combo_validator.py`
