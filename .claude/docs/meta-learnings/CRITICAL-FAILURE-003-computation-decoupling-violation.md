# CRITICAL-FAILURE-003: Computation Decoupling Violation

**Date Discovered**: 2026-01-27
**Severity**: HIGH
**Status**: Active - needs fixing

## The Incident

During planning for factorial design visualization and STRATOS metrics export, it was discovered that the codebase violates the fundamental two-block orchestration principle:

**Intended Architecture:**
```
[Block 1: Extraction]     [Block 2: Visualization]
   MLflow → DuckDB    →    DuckDB → Figures
   (ALL computation)       (READ ONLY)
```

**Actual Architecture:**
```
[Block 1: Extraction]     [Block 2: Visualization]
   MLflow → DuckDB    →    DuckDB → [COMPUTE] → Figures
   (partial metrics)       (STILL COMPUTING!)
                               ↑
                           VIOLATION
```

## What Went Wrong

1. **Extraction flow only exports partial metrics**: DuckDB contains `auroc`, `auroc_ci_lo`, `auroc_ci_hi`, `brier`, but MISSING:
   - `aurc` (Area Under Risk-Coverage)
   - `scaled_brier` (IPA)
   - `calibration_slope`, `calibration_intercept`, `o_e_ratio`
   - `net_benefit` at various thresholds

2. **Visualization code computes metrics on-the-fly**: 14+ files in `src/viz/` and `src/stats/` compute metrics from (y_true, y_prob) instead of reading from DuckDB.

3. **Direct MLflow access from viz layer**: `generate_instability_figures.py` directly loads pickle files from `/home/petteri/mlruns/` - completely bypassing the extraction layer.

## Why This Happened

1. **Incremental development**: Metrics were added to visualization code as needed, without enforcing the extraction-first principle
2. **No architectural enforcement**: No test or linter blocks compute calls from viz code
3. **Metrics functions are "convenient"**: `src/stats/calibration_extended.py` etc. provide clean APIs, making it easy to call them anywhere

## Files With Violations

### CRITICAL (compute metrics in viz layer)
| File | Metrics Computed |
|------|------------------|
| `src/viz/retained_metric.py` | AUROC, Brier, scaled Brier, NB, F1 |
| `src/viz/calibration_plot.py` | Calibration slope, intercept, ICI |
| `src/viz/dca_plot.py` | Net benefit, DCA curves |
| `src/viz/generate_instability_figures.py` | **DIRECT MLflow pickle loading** |
| `src/stats/pminternal_analysis.py` | MAPE, CII, bootstrap CI |

### HIGH (secondary compute calls)
| File | Metrics Computed |
|------|------------------|
| `src/viz/stratos_figures.py` | Multiple STRATOS metrics |
| `src/viz/prob_distribution.py` | AUROC, medians, means |

## Impact

1. **Non-reproducibility**: Same figure generated at different times may have different metrics if computation parameters change
2. **Performance**: Metrics recomputed every time figure is generated (wasteful)
3. **Data inconsistency**: DuckDB metrics may differ from on-the-fly computed metrics
4. **Testing difficulty**: Can't test viz layer without running actual computation

## Required Fix

### Phase 1: Extend Extraction Flow
Add to `essential_metrics` table in DuckDB:
- `aurc` (from `src/stats/uncertainty_quantification.py`)
- `scaled_brier` / `ipa` (from `src/stats/scaled_brier.py`)
- `calibration_slope`, `calibration_intercept`, `o_e_ratio` (from `src/stats/calibration_extended.py`)
- `net_benefit_5pct`, `net_benefit_10pct`, `net_benefit_15pct`, `net_benefit_20pct` (from `src/stats/clinical_utility.py`)

### Phase 2: Add DCA/Calibration Curve Tables
Create new DuckDB tables for curve data:
- `dca_curves` (threshold, nb_model, nb_all, nb_none)
- `calibration_curves` (bin_midpoint, observed, predicted)

### Phase 3: Refactor Visualization Code
Remove ALL metric computation from viz layer:
- `retained_metric.py` → Load pre-computed AUROC/Brier from DB
- `calibration_plot.py` → Load calibration metrics from DB
- `dca_plot.py` → Load net benefit values from DB
- `generate_instability_figures.py` → Add extraction task for bootstrap data

### Phase 4: Enforcement
1. Add test that greps for compute function calls in `src/viz/`
2. Add pre-commit hook to block new violations
3. Document in `.claude/CLAUDE.md`

## Prevention Rules (Add to CLAUDE.md)

```markdown
## CRITICAL: Computation Decoupling Rule

**NEVER compute metrics in visualization code.**

The pipeline has two blocks:
1. **Extraction Block** (`extraction_flow.py`): MLflow → DuckDB
   - ALL metric computation happens HERE
   - Calls functions from `src/stats/`
   - Writes to DuckDB tables

2. **Visualization Block** (`src/viz/`): DuckDB → Figures
   - READ ONLY from DuckDB
   - NEVER import or call `src/stats/` functions
   - NEVER load MLflow artifacts directly

**VIOLATIONS:**
- ❌ `from src.stats.calibration_extended import calibration_slope_intercept` in viz code
- ❌ `roc_auc_score(y_true, y_prob)` in viz code
- ❌ `pickle.load(mlruns_path / ...)` in viz code

**CORRECT:**
- ✅ `SELECT auroc, calibration_slope FROM essential_metrics` in viz code
```

## Lesson Learned

**Never assume metric availability based on function existence.**

The existence of `src/stats/scaled_brier.py` does NOT mean scaled Brier is in DuckDB. The extraction layer must explicitly call these functions and store results.

**The principle: Functions exist in `src/stats/` to be called by `extraction_flow.py`, NOT by `src/viz/`.**
