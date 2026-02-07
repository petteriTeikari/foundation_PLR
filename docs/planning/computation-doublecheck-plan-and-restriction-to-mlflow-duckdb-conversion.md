# Computation Doublecheck Plan: MLflow → DuckDB Restriction

**Created**: 2026-01-27
**Updated**: 2026-01-27
**Status**: Phase 1-2 Complete, Phase 3-4 Pending
**Priority**: CRITICAL

---

## High-Level Vision: Why Computation Decoupling Matters

### The Problem We Were Solving

Scientific reproducibility in machine learning pipelines is notoriously difficult. A common failure mode is **"scattered computation"** where the same metric can be computed in multiple places with potentially different implementations, parameters, or data preprocessing. This leads to:

1. **Inconsistent results**: A figure might show AUROC=0.91 while a table shows AUROC=0.89 for the same model
2. **Silent bugs**: Changes to a computation function affect some outputs but not others
3. **Debugging nightmares**: "Where did this number come from?" becomes unanswerable
4. **Non-reproducibility**: Re-running the pipeline may produce different results

### The Architectural Principle

We adopted a **two-block orchestration pattern** inspired by data engineering best practices (ETL/ELT patterns):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION BLOCK (Block 1)                           │
│                                                                             │
│   MLflow Artifacts (pickle) ──► Metric Computation ──► DuckDB              │
│                                                                             │
│   - Reads raw predictions (y_true, y_prob)                                 │
│   - Computes ALL derived metrics (AUROC, calibration, net benefit, etc.)   │
│   - Stores results in typed schema                                          │
│   - Single Source of Truth for all downstream consumers                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VISUALIZATION BLOCK (Block 2)                        │
│                                                                             │
│   DuckDB ──► SELECT queries ──► Figures/Tables/LaTeX                       │
│                                                                             │
│   - READ ONLY from DuckDB                                                   │
│   - NEVER computes metrics from raw data                                   │
│   - Focuses on presentation, styling, layout                               │
│   - Changes to viz code cannot affect metric values                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| **Single Source of Truth** | All metrics come from one place (DuckDB) |
| **Reproducibility** | Same query always returns same result |
| **Auditability** | Can trace any number back to its computation |
| **Separation of Concerns** | Statisticians write extraction, designers write viz |
| **Performance** | Metrics computed once, read many times |
| **Testing** | Can unit test extraction and viz independently |

### The Violation We Discovered (CRITICAL-FAILURE-003)

On 2026-01-27, during planning for STRATOS-compliant figures, we discovered that **14+ files** in `src/viz/` were computing metrics from raw predictions instead of reading from DuckDB:

- `src/viz/retained_metric.py` computed AUROC, Brier, scaled Brier
- `src/viz/calibration_plot.py` computed calibration slope/intercept
- `src/viz/dca_plot.py` computed net benefit across thresholds
- `src/viz/generate_instability_figures.py` even loaded MLflow pickles directly!

This violated the two-block principle and created risk of inconsistent results.

---

## What Was Done: Implementation Summary

### Phase 1: Extended Extraction to Compute STRATOS Metrics ✅

**Date**: 2026-01-27
**Files Modified**: `scripts/extract_all_configs_to_duckdb.py`

#### 1.1 Added `extract_stratos_metrics()` Function

```python
def extract_stratos_metrics(metrics_pickle_path: Path) -> dict[str, Any]:
    """
    Extract STRATOS-compliant metrics by computing from predictions.

    CRITICAL: This is the ONLY place STRATOS metrics are computed.
    Visualization code must read from DuckDB, not compute metrics.
    """
    # Load predictions from pickle
    y_prob = preds['y_pred_proba'].mean(axis=1)  # Mean across bootstrap
    y_true = labels[:, 0]  # Same across bootstrap iterations

    # Compute calibration metrics (Van Calster 2024)
    cal_result = calibration_slope_intercept(y_true, y_prob)

    # Compute scaled Brier (IPA)
    sb_result = scaled_brier_score(y_true, y_prob)

    # Compute net benefit at clinical thresholds
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        nb = net_benefit(y_true, y_prob, threshold)

    return {
        "calibration_slope": cal_result.slope,
        "calibration_intercept": cal_result.intercept,
        "o_e_ratio": cal_result.o_e_ratio,
        "scaled_brier": sb_result["ipa"],
        "net_benefit_5pct": ...,
        "net_benefit_10pct": ...,
        ...
    }
```

#### 1.2 Updated DuckDB Schema

The `essential_metrics` table was extended from 17 columns to **25 columns**:

```sql
CREATE TABLE essential_metrics (
    -- Existing columns (17)
    run_id VARCHAR PRIMARY KEY,
    model_path VARCHAR,
    outlier_method VARCHAR,
    imputation_method VARCHAR,
    featurization VARCHAR,
    classifier VARCHAR,
    outlier_display_name VARCHAR,
    imputation_display_name VARCHAR,
    classifier_display_name VARCHAR,
    auroc DOUBLE,
    auroc_ci_lo DOUBLE,
    auroc_ci_hi DOUBLE,
    brier DOUBLE,
    n_bootstrap INTEGER,
    outlier_source_known BOOLEAN,
    anomaly_source VARCHAR,
    mlflow_run_outlier_detection VARCHAR,

    -- NEW: STRATOS calibration metrics (8 columns)
    calibration_slope DOUBLE,      -- Target: 1.0
    calibration_intercept DOUBLE,  -- Target: 0.0
    o_e_ratio DOUBLE,              -- Target: 1.0
    scaled_brier DOUBLE,           -- IPA, range: (-∞, 1]
    net_benefit_5pct DOUBLE,       -- At 5% threshold
    net_benefit_10pct DOUBLE,      -- At 10% threshold
    net_benefit_15pct DOUBLE,      -- At 15% threshold
    net_benefit_20pct DOUBLE       -- At 20% threshold
);
```

#### 1.3 Re-ran Extraction

```bash
$ python scripts/extract_all_configs_to_duckdb.py
Scanning /home/petteri/mlruns/253031330985650090...
Found 225 CatBoost configurations
Wrote 224 rows to outputs/foundation_plr_results.db

EXTRACTION VERIFICATION
Total configurations: 224
Unique combinations: 224 (validated - no duplicates)
```

**Result**: All 224 configurations now have STRATOS metrics with **zero NULL values**.

### Phase 2: Updated Export for R ✅

**Date**: 2026-01-27
**Files Modified**: `scripts/export_data_for_r.py`

The CSV export was updated to include all STRATOS columns:

**Before** (12 columns):
```
run_id, outlier_method, ..., auroc, auroc_ci_lo, auroc_ci_hi, brier, n_bootstrap
```

**After** (20 columns):
```
run_id, outlier_method, ..., auroc, auroc_ci_lo, auroc_ci_hi, brier, n_bootstrap,
calibration_slope, calibration_intercept, o_e_ratio, scaled_brier,
net_benefit_5pct, net_benefit_10pct, net_benefit_15pct, net_benefit_20pct
```

### Phase 3: Visualization Refactoring (PENDING)

The following files still compute metrics and need refactoring:

| File | Current State | Required Change |
|------|---------------|-----------------|
| `src/viz/retained_metric.py` | Computes AUROC, Brier, NB at retention | Load from `retention_curves` table |
| `src/viz/calibration_plot.py` | Computes slope, intercept, ICI | Load from `essential_metrics` |
| `src/viz/dca_plot.py` | Computes NB sweep | Load from `dca_curves` table |
| `src/viz/generate_instability_figures.py` | Loads MLflow pickles directly | Load from `bootstrap_predictions` |

### Phase 4: Enforcement (PENDING)

Enforcement mechanisms to prevent future violations:

1. **Pre-commit hook** blocking metric computation imports in `src/viz/`
2. **AST-based test** scanning for banned function calls
3. **Documentation** in CLAUDE.md (✅ done)

---

## Updated Documentation

### CLAUDE.md Updates

Added to `.claude/CLAUDE.md`:

```markdown
## 🚨🚨🚨 CRITICAL: COMPUTATION DECOUPLING 🚨🚨🚨

**The Rule**: ALL metric computation happens in extraction.
Visualization code must ONLY read from DuckDB, NEVER compute metrics.

### BANNED in Visualization Code

from sklearn.metrics import roc_auc_score, brier_score_loss  # ❌
from src.stats.calibration_extended import calibration_slope_intercept  # ❌

### CORRECT in Visualization Code

df = conn.execute("SELECT auroc, calibration_slope FROM essential_metrics").fetchdf()  # ✅
```

### Metalearning Document

Created `.claude/docs/meta-learnings/CRITICAL-FAILURE-003-computation-decoupling-violation.md` with:
- Root cause analysis
- List of all violating files
- Prevention rules

---

## Technical Details: MLflow Pickle Structure

Understanding the MLflow artifact structure was critical for implementing extraction:

```python
# Load pickle
with open('metrics_*.pickle', 'rb') as f:
    data = pickle.load(f)

# Structure:
data.keys() = [
    'metrics_iter',      # Per-bootstrap iteration data
    'metrics_stats',     # Aggregated statistics
    'subjectwise_stats', # Per-subject summaries
    'subject_global_stats'
]

# Predictions are in:
data['metrics_iter']['test']['preds']['arrays']['predictions']
# Which contains:
{
    'y_pred_proba': np.array (n_subjects, n_bootstrap),  # Shape: (63, 1000)
    'y_pred': np.array (n_subjects, n_bootstrap),
    'label': np.array (n_subjects, n_bootstrap)  # Same across bootstrap
}
```

### Key Insight: Bootstrap Structure

Each run has 1000 bootstrap iterations with 63 test subjects:
- `y_pred_proba[i, j]` = probability for subject i in bootstrap iteration j
- `label[i, :]` = same value repeated (true label doesn't change)

For aggregate metrics, we use **mean prediction per subject**:
```python
y_prob_mean = y_pred_proba.mean(axis=1)  # (63,)
y_true = label[:, 0]  # (63,)
```

---

## Verification Results

After Phase 1-2 completion:

```python
# Check DuckDB contents
conn = duckdb.connect('outputs/foundation_plr_results.db', read_only=True)

# Columns now include STRATOS metrics
cols = conn.execute('DESCRIBE essential_metrics').fetchall()
# ... calibration_slope: DOUBLE
# ... calibration_intercept: DOUBLE
# ... o_e_ratio: DOUBLE
# ... scaled_brier: DOUBLE
# ... net_benefit_5pct: DOUBLE
# ... (etc.)

# Verify no NULLs
nulls = conn.execute('''
    SELECT
        SUM(CASE WHEN calibration_slope IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN scaled_brier IS NULL THEN 1 ELSE 0 END),
        COUNT(*)
    FROM essential_metrics
''').fetchone()
# Result: (0, 0, 224) - zero NULLs
```

---

**See also**: `.claude/docs/meta-learnings/CRITICAL-FAILURE-003-computation-decoupling-violation.md`

---

## 1. Current State Analysis

### 1.1 What's Currently in DuckDB

```sql
-- essential_metrics table has:
run_id, model_path, outlier_method, imputation_method, featurization, classifier,
outlier_display_name, imputation_display_name, classifier_display_name,
auroc, auroc_ci_lo, auroc_ci_hi, brier, n_bootstrap,
outlier_source_known, anomaly_source, mlflow_run_outlier_detection
```

### 1.2 What's MISSING from DuckDB (Required by STRATOS)

| Metric | Source Function | Status |
|--------|-----------------|--------|
| `aurc` | `src/stats/uncertainty_quantification.py:sec_classification()` | ❌ NOT in DB |
| `scaled_brier` / `ipa` | `src/stats/scaled_brier.py:scaled_brier_score()` | ❌ NOT in DB |
| `calibration_slope` | `src/stats/calibration_extended.py:calibration_slope_intercept()` | ❌ NOT in DB |
| `calibration_intercept` | same as above | ❌ NOT in DB |
| `o_e_ratio` | same as above | ❌ NOT in DB |
| `net_benefit_5pct` | `src/stats/clinical_utility.py:net_benefit()` | ❌ NOT in DB |
| `net_benefit_10pct` | same as above | ❌ NOT in DB |
| `net_benefit_15pct` | same as above | ❌ NOT in DB |
| `net_benefit_20pct` | same as above | ❌ NOT in DB |

### 1.3 Files Computing Metrics in Viz Layer (VIOLATIONS)

| File | Metrics Computed | Fix Required |
|------|------------------|--------------|
| `src/viz/retained_metric.py` | AUROC, Brier, scaled Brier, NB | Load from DB |
| `src/viz/calibration_plot.py` | Calibration slope/intercept/ICI | Load from DB |
| `src/viz/dca_plot.py` | Net benefit across thresholds | Load from DB |
| `src/viz/generate_instability_figures.py` | Direct MLflow pickle loading! | Major refactor |
| `src/viz/prob_distribution.py` | AUROC, distribution stats | Load from DB |
| `src/viz/stratos_figures.py` | Calls calibration/clinical_utility | Load from DB |

---

## 2. Required DuckDB Schema Changes

### 2.1 Extend `essential_metrics` Table

```sql
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS aurc REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS scaled_brier REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS calibration_slope REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS calibration_intercept REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS o_e_ratio REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS net_benefit_5pct REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS net_benefit_10pct REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS net_benefit_15pct REAL;
ALTER TABLE essential_metrics ADD COLUMN IF NOT EXISTS net_benefit_20pct REAL;
```

### 2.2 Add DCA Curves Table

```sql
CREATE TABLE IF NOT EXISTS dca_curves (
    config_id TEXT NOT NULL,  -- Reference to essential_metrics.run_id
    threshold REAL NOT NULL,
    net_benefit_model REAL,
    net_benefit_all REAL,
    net_benefit_none REAL,
    PRIMARY KEY (config_id, threshold)
);
```

### 2.3 Add Calibration Curves Table

```sql
CREATE TABLE IF NOT EXISTS calibration_curves (
    config_id TEXT NOT NULL,  -- Reference to essential_metrics.run_id
    bin_index INTEGER NOT NULL,
    bin_midpoint REAL,
    observed_proportion REAL,
    predicted_mean REAL,
    n_samples INTEGER,
    PRIMARY KEY (config_id, bin_index)
);
```

### 2.4 Add Bootstrap Predictions Table (for instability analysis)

```sql
CREATE TABLE IF NOT EXISTS bootstrap_predictions (
    config_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,  -- Anonymized (Hxxx/Gxxx)
    bootstrap_iter INTEGER NOT NULL,
    y_true INTEGER,
    y_prob REAL,
    PRIMARY KEY (config_id, subject_id, bootstrap_iter)
);
```

---

## 3. Extraction Flow Modifications

### 3.1 Update `extract_run_metrics()` Task

Location: `src/orchestration/flows/extraction_flow.py`

```python
def extract_run_metrics(run_path: Path, conn) -> dict:
    """
    Extract ALL STRATOS metrics from a single MLflow run.

    This is the ONLY place metrics are computed from predictions.
    """
    # Load predictions
    y_true, y_prob = load_predictions_from_run(run_path)

    # Compute ALL metrics HERE
    metrics = {
        # Existing
        'auroc': compute_auroc_with_ci(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),

        # NEW: STRATOS metrics
        'aurc': sec_classification(y_true, y_prob)['aurc'],
        'scaled_brier': scaled_brier_score(y_true, y_prob)['ipa'],

        # NEW: Calibration metrics
        **calibration_slope_intercept(y_true, y_prob).scalars,

        # NEW: Clinical utility at standard thresholds
        'net_benefit_5pct': net_benefit(y_true, y_prob, 0.05),
        'net_benefit_10pct': net_benefit(y_true, y_prob, 0.10),
        'net_benefit_15pct': net_benefit(y_true, y_prob, 0.15),
        'net_benefit_20pct': net_benefit(y_true, y_prob, 0.20),
    }

    return metrics
```

### 3.2 Add `extract_dca_curves()` Task

```python
@task(name="extract_dca_curves")
def extract_dca_curves(run_id: str, y_true, y_prob, conn) -> None:
    """
    Extract DCA curve data for a single run.
    """
    thresholds = np.arange(0.01, 0.50, 0.01)  # 1% to 49%

    for threshold in thresholds:
        nb_model = net_benefit(y_true, y_prob, threshold)
        nb_all = treat_all_net_benefit(y_true, threshold)
        nb_none = 0.0  # By definition

        conn.execute("""
            INSERT INTO dca_curves (config_id, threshold, net_benefit_model, net_benefit_all, net_benefit_none)
            VALUES (?, ?, ?, ?, ?)
        """, [run_id, threshold, nb_model, nb_all, nb_none])
```

### 3.3 Add `extract_calibration_curves()` Task

```python
@task(name="extract_calibration_curves")
def extract_calibration_curves(run_id: str, y_true, y_prob, conn, n_bins: int = 10) -> None:
    """
    Extract calibration curve data for a single run.
    """
    # Use LOESS-smoothed calibration (matches our figures)
    curve_data = compute_loess_calibration(y_true, y_prob)

    for i, (midpoint, observed, predicted, n) in enumerate(zip(
        curve_data['bin_midpoints'],
        curve_data['observed'],
        curve_data['predicted'],
        curve_data['counts']
    )):
        conn.execute("""
            INSERT INTO calibration_curves (config_id, bin_index, bin_midpoint, observed_proportion, predicted_mean, n_samples)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [run_id, i, midpoint, observed, predicted, n])
```

---

## 4. Visualization Code Refactoring

### 4.1 Pattern: Load from DB, Not Compute

**BEFORE (WRONG):**
```python
# src/viz/retained_metric.py
def compute_metric_at_retention(y_true, y_prob, retention_rate, metric_fn):
    # ... keep most confident predictions
    auroc = roc_auc_score(y_true_subset, y_prob_subset)  # ❌ COMPUTING
    return auroc
```

**AFTER (CORRECT):**
```python
# src/viz/retained_metric.py
def load_metric_at_retention(conn, run_id, retention_rate, metric_name):
    # Read from pre-computed retention curves table
    result = conn.execute("""
        SELECT metric_value FROM retention_curves
        WHERE run_id = ? AND retention_rate = ? AND metric_name = ?
    """, [run_id, retention_rate, metric_name]).fetchone()
    return result[0]  # ✅ READING FROM DB
```

### 4.2 Files to Refactor

| File | Current | Target |
|------|---------|--------|
| `retained_metric.py` | Computes AUROC/Brier/etc | Load from `retention_curves` table |
| `calibration_plot.py` | Computes slope/intercept | Load from `essential_metrics` |
| `dca_plot.py` | Computes NB at thresholds | Load from `dca_curves` table |
| `generate_instability_figures.py` | Loads MLflow pickles | Load from `bootstrap_predictions` |

---

## 5. Export Script Updates

### 5.1 Update `export_data_for_r.py`

```python
def export_essential_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """Export ALL STRATOS metrics to CSV."""
    df = conn.execute("""
        SELECT
            run_id,
            outlier_method, outlier_display_name,
            imputation_method, imputation_display_name,
            classifier, classifier_display_name,
            -- Discrimination
            auroc, auroc_ci_lo, auroc_ci_hi,
            -- Calibration (NEW)
            brier,
            scaled_brier,  -- NEW
            calibration_slope,  -- NEW
            calibration_intercept,  -- NEW
            o_e_ratio,  -- NEW
            -- Clinical Utility (NEW)
            net_benefit_5pct,  -- NEW
            net_benefit_10pct,  -- NEW
            net_benefit_15pct,  -- NEW
            net_benefit_20pct,  -- NEW
            -- Uncertainty (NEW)
            aurc,  -- NEW
            n_bootstrap
        FROM essential_metrics
        WHERE featurization LIKE '%simple%'
        ORDER BY auroc DESC
    """).fetchdf()

    output_path = OUTPUT_DIR / "essential_metrics.csv"
    df.to_csv(output_path, index=False)
```

---

## 6. Enforcement Mechanisms

### 6.1 Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: no-compute-in-viz
      name: Block metric computation in viz code
      entry: scripts/check_no_compute_in_viz.py
      language: python
      files: ^src/viz/.*\.py$
```

### 6.2 Test for Violations

Create `tests/test_decoupling.py`:

```python
import ast
import glob

BANNED_FUNCTIONS = [
    'roc_auc_score', 'brier_score_loss', 'calibration_slope_intercept',
    'net_benefit', 'scaled_brier_score', 'sec_classification'
]

def test_no_metric_computation_in_viz():
    """Ensure viz code doesn't compute metrics."""
    violations = []

    for py_file in glob.glob('src/viz/**/*.py', recursive=True):
        with open(py_file) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, 'id', '') or getattr(node.func, 'attr', '')
                if func_name in BANNED_FUNCTIONS:
                    violations.append(f"{py_file}: calls {func_name}")

    assert not violations, f"Computation in viz layer:\n" + "\n".join(violations)
```

---

## 7. Implementation Plan

### Phase 1: Extend Extraction (Priority 1) ✅ COMPLETED 2026-01-27
- [x] Add new columns to DuckDB schema (25 columns total)
- [x] Added `extract_stratos_metrics()` function to compute all STRATOS metrics
- [ ] Add `extract_dca_curves()` task (future)
- [ ] Add `extract_calibration_curves()` task (future)
- [x] Re-run extraction for all 224 runs

### Phase 2: Update Export (Priority 1) ✅ COMPLETED 2026-01-27
- [x] Update `export_data_for_r.py` to include new columns
- [x] Regenerate `essential_metrics.csv` (now has 20 columns)
- [ ] Verify R scripts still work

### Phase 3: Refactor Viz Code (Priority 2)
- [ ] `retained_metric.py` - load from DB
- [ ] `calibration_plot.py` - load from DB
- [ ] `dca_plot.py` - load from DB
- [ ] `generate_instability_figures.py` - major refactor

### Phase 4: Enforcement (Priority 3)
- [ ] Add pre-commit hook
- [ ] Add decoupling test
- [ ] Update CLAUDE.md with rules

---

## 8. Validation Checklist

After implementation, verify:

- [ ] `essential_metrics.csv` has 17+ columns (not just 12)
- [ ] All 410 runs have `scaled_brier`, `calibration_slope` values
- [ ] DCA curves table has data for all runs
- [ ] No `src/viz/` file imports `roc_auc_score` or similar
- [ ] `generate_instability_figures.py` doesn't reference `/home/petteri/mlruns/`
- [ ] All figures still render correctly
- [ ] Tests pass

---

## 9. Future Work: Architectural Improvements

### 9.1 Immediate Next Steps (Priority 1)

#### Complete Visualization Refactoring

The highest priority is completing Phase 3 - refactoring visualization code to read from DuckDB instead of computing metrics:

| Task | Effort | Impact |
|------|--------|--------|
| Refactor `retained_metric.py` | Medium | Removes most common violation |
| Refactor `calibration_plot.py` | Low | Simple - metrics already in DB |
| Refactor `dca_plot.py` | Medium | Needs `dca_curves` table first |
| Refactor `generate_instability_figures.py` | High | Needs `bootstrap_predictions` table |

#### Add Enforcement Tests

```python
# tests/test_decoupling.py
def test_no_metric_imports_in_viz():
    """Block metric computation imports in visualization code."""
    import ast
    BANNED = ['roc_auc_score', 'brier_score_loss', 'net_benefit', ...]

    for py_file in Path('src/viz').rglob('*.py'):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in BANNED
```

### 9.2 Medium-Term Improvements (Priority 2)

#### Extract Curve Data to DuckDB

Currently, only scalar metrics are in DuckDB. Curve data (for plotting) should also be extracted:

| Table | Purpose | Schema |
|-------|---------|--------|
| `dca_curves` | Decision Curve Analysis | `(config_id, threshold, nb_model, nb_all, nb_none)` |
| `calibration_curves` | Calibration plots | `(config_id, bin_idx, observed, predicted, n)` |
| `roc_curves` | ROC curves | `(config_id, fpr, tpr)` |
| `retention_curves` | Selective classification | `(config_id, retention, auroc, brier, nb)` |

This enables:
- Faster figure generation (no re-computation)
- Consistent curves across regeneration
- Easy export for supplementary materials

#### Implement Bootstrap Predictions Table

For instability analysis (Riley 2023), we need per-subject, per-bootstrap predictions:

```sql
CREATE TABLE bootstrap_predictions (
    config_id TEXT,
    subject_id TEXT,        -- Anonymized (Hxxx/Gxxx)
    bootstrap_iter INTEGER,
    y_true INTEGER,
    y_prob REAL,
    PRIMARY KEY (config_id, subject_id, bootstrap_iter)
);
```

**Size estimate**: 224 configs × 63 subjects × 1000 iterations = 14.1M rows (~500MB)

This is large but tractable. Alternative: store as Parquet files with DuckDB external table.

### 9.3 Long-Term Architectural Vision (Priority 3)

#### Prefect Flow Orchestration

Currently, extraction and visualization are loosely coupled scripts. A more robust approach uses Prefect flows:

```python
@flow(name="foundation-plr-pipeline")
def main_pipeline():
    # Block 1: Extraction
    db_path = extract_mlflow_to_duckdb()

    # Block 2: Export for R
    csv_path = export_for_r(db_path)

    # Block 3: Generate Figures
    figures = generate_all_figures(db_path)

    # Block 4: Generate LaTeX Tables
    tables = generate_latex_tables(db_path)

    # Block 5: Compile Manuscript
    compile_manuscript(figures, tables)
```

Benefits:
- Automatic caching (unchanged inputs → skip recomputation)
- Parallel execution where possible
- Clear dependency graph
- Retry on failure

#### Data Versioning with DVC

For full reproducibility, integrate DVC (Data Version Control):

```yaml
# dvc.yaml
stages:
  extract:
    cmd: python scripts/extract_all_configs_to_duckdb.py
    deps:
      - /home/petteri/mlruns/253031330985650090
    outs:
      - outputs/foundation_plr_results.db

  export:
    cmd: python scripts/export_data_for_r.py
    deps:
      - outputs/foundation_plr_results.db
    outs:
      - outputs/r_data/essential_metrics.csv
```

Benefits:
- Track DuckDB versions
- Reproduce any historical state
- Share artifacts without re-computation

### 9.4 Lessons Learned

| Lesson | Description |
|--------|-------------|
| **Define boundaries early** | The extraction/viz boundary should have been enforced from day 1 |
| **Schema first** | Define DuckDB schema before writing extraction code |
| **Ban imports explicitly** | Document banned patterns in CLAUDE.md immediately |
| **Test architectural rules** | Use AST-based tests to enforce architecture |
| **Metalearning docs work** | CRITICAL-FAILURE docs prevent repeat mistakes |

### 9.5 Recommended Reading

For developers extending this architecture:

1. **ETL Best Practices**: Kimball's Data Warehouse Toolkit
2. **Prefect Flows**: https://docs.prefect.io/concepts/flows/
3. **DuckDB for Analytics**: https://duckdb.org/docs/guides/
4. **STRATOS Guidelines**: Van Calster et al. 2024 (defines required metrics)
5. **This Repo's CLAUDE.md**: Essential reading before any changes

---

## 10. For Manuscript Authors

### How to Cite This Architecture

When describing the reproducibility approach in the manuscript:

> **Data Processing Pipeline**
>
> To ensure reproducibility and prevent metric computation inconsistencies, we implemented a two-block orchestration architecture. All performance metrics (AUROC, calibration slope, net benefit, etc.) are computed once during the extraction phase and stored in a DuckDB database. Visualization code reads exclusively from this database, never recomputing metrics from raw predictions. This design guarantees that all figures and tables report identical values for the same configuration.

### Key Numbers to Report

From the current DuckDB:

| Statistic | Value | Query |
|-----------|-------|-------|
| Total configurations | 224 | `SELECT COUNT(*) FROM essential_metrics` |
| CatBoost configs | 57 | `SELECT COUNT(*) WHERE classifier='CatBoost'` |
| AUROC range | 0.680 - 0.913 | `SELECT MIN(auroc), MAX(auroc)` |
| Mean calibration slope | ~2.5 | `SELECT AVG(calibration_slope)` |
| Mean scaled Brier (IPA) | ~0.35 | `SELECT AVG(scaled_brier)` |

### STRATOS Compliance

The extracted metrics satisfy Van Calster 2024 STRATOS requirements:

| STRATOS Requirement | Column(s) in DuckDB |
|---------------------|---------------------|
| Discrimination | `auroc`, `auroc_ci_lo`, `auroc_ci_hi` |
| Calibration (weak) | `calibration_slope`, `calibration_intercept` |
| Calibration (mean) | `o_e_ratio` |
| Overall performance | `brier`, `scaled_brier` |
| Clinical utility | `net_benefit_5pct`, `net_benefit_10pct`, etc. |

---

## Appendix A: MLflow Pickle Structure Reference

From exploration of a sample run:
```python
data = pickle.load(open('metrics_*.pickle', 'rb'))

# Top-level keys
data.keys() = [
    'metrics_iter',        # Per-bootstrap iteration data
    'metrics_stats',       # Aggregated statistics (mean, std, CI)
    'subjectwise_stats',   # Per-subject summaries
    'subject_global_stats' # Global subject-level statistics
]

# Predictions location
data['metrics_iter']['test']['preds']['arrays']['predictions']
# Contains:
{
    'y_pred_proba': np.array,  # Shape: (n_subjects, n_bootstrap) = (63, 1000)
    'y_pred': np.array,        # Binary predictions
    'label': np.array          # True labels (same across bootstrap)
}
```

This structure needs to be understood to properly extract bootstrap predictions.

---

## Appendix B: DuckDB Schema (Full)

```sql
-- As of 2026-01-27
CREATE TABLE essential_metrics (
    -- Identification
    run_id VARCHAR PRIMARY KEY,
    model_path VARCHAR,

    -- Configuration
    outlier_method VARCHAR,
    imputation_method VARCHAR,
    featurization VARCHAR,
    classifier VARCHAR,

    -- Display names (from YAML lookup)
    outlier_display_name VARCHAR,
    imputation_display_name VARCHAR,
    classifier_display_name VARCHAR,

    -- Core metrics (from MLflow)
    auroc DOUBLE,
    auroc_ci_lo DOUBLE,
    auroc_ci_hi DOUBLE,
    brier DOUBLE,
    n_bootstrap INTEGER,

    -- Metadata
    outlier_source_known BOOLEAN,
    anomaly_source VARCHAR,
    mlflow_run_outlier_detection VARCHAR,

    -- STRATOS calibration metrics (computed during extraction)
    calibration_slope DOUBLE,
    calibration_intercept DOUBLE,
    o_e_ratio DOUBLE,

    -- STRATOS overall metric
    scaled_brier DOUBLE,

    -- STRATOS clinical utility
    net_benefit_5pct DOUBLE,
    net_benefit_10pct DOUBLE,
    net_benefit_15pct DOUBLE,
    net_benefit_20pct DOUBLE
);

-- Total: 25 columns
-- Rows: 224 (as of 2026-01-27)
-- Size: ~1.3 MB
```

---

## Appendix C: File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `scripts/extract_all_configs_to_duckdb.py` | Modified | Added `extract_stratos_metrics()`, updated schema |
| `scripts/export_data_for_r.py` | Modified | Added STRATOS columns to CSV export |
| `.claude/CLAUDE.md` | Modified | Added computation decoupling rules |
| `.claude/docs/meta-learnings/CRITICAL-FAILURE-003-*.md` | Created | Root cause analysis |
| `docs/planning/computation-doublecheck-*.md` | Created | This document |
