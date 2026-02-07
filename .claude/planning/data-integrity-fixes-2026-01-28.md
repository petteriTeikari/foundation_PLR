# Data Integrity Fixes - 2026-01-28

## Session Summary

This document captures all changes made to fix data integrity issues, specifically
addressing FAILURE-005 (Ground Truth AUROC showing 0.850 instead of 0.911) and
FAILURE-006 (ambiguous annotations in figures).

**Root Cause**: Export scripts were not filtering by `featurization = 'simple1.0'`,
causing mixed handcrafted and embedding features to be included in exports.

---

## Changes Made

### 1. Created Single Source of Truth for Data Filtering

**File**: `configs/VISUALIZATION/data_filters.yaml`

```yaml
defaults:
  featurization: "simple1.0"  # Handcrafted amplitude bins + latency
  classifier: "CATBOOST"       # Uppercase as stored in DuckDB

expected_values:
  ground_truth:
    auroc: 0.911
    auroc_tolerance: 0.002
    n_predictions: 208
```

**Purpose**: Centralize all data extraction filters so export scripts use consistent values.

---

### 2. Created Python Module for Data Filters

**File**: `src/data_io/data_filters.py`

**Functions provided**:
- `get_default_featurization()` - Returns "simple1.0"
- `get_default_classifier()` - Returns "CATBOOST"
- `get_predictions_query()` - Builds SQL with proper filters
- `validate_auroc()` - Validates against expected values

**Purpose**: Programmatic access to filter values with validation helpers.

---

### 3. Created TDD Tests for Data Integrity

**File**: `tests/test_data_integrity.py`

**Tests**:
| Test | What it checks |
|------|----------------|
| `test_ground_truth_in_duckdb` | AUROC = 0.911 ± 0.002 in essential_metrics |
| `test_ground_truth_in_roc_rc_json` | AUROC correct in exported JSON |
| `test_ground_truth_computed_from_predictions` | AUROC computed from raw predictions |
| `test_prediction_count_ground_truth` | Reasonable count with correct class balance |
| `test_data_filters_yaml_exists` | Config file exists |
| `test_data_filters_has_correct_default` | Default featurization is simple1.0 |
| `test_calibration_json_exists` | Calibration data exported |

**Run with**: `pytest tests/test_data_integrity.py -v`

---

### 4. Fixed Export Scripts

#### 4.1 `scripts/export_roc_rc_data.py`

**Changes**:
- Added import: `from src.data_io.data_filters import get_default_featurization, get_default_classifier`
- Added `featurization` parameter to `get_predictions_for_combo()`
- Added featurization filter to SQL query
- Fixed YAML path: `plot_hyperparam_combos.yaml` → `combos.yaml`

**Before**:
```python
query = """
    SELECT y_true, y_prob FROM predictions
    WHERE outlier_method = ? AND imputation_method = ? AND classifier = ?
"""
```

**After**:
```python
if featurization is None:
    featurization = get_default_featurization()
query = """
    SELECT y_true, y_prob FROM predictions
    WHERE outlier_method = ? AND imputation_method = ?
      AND classifier = ? AND featurization = ?
"""
```

#### 4.2 `scripts/export_selective_classification_data.py`

**Changes**:
- Added import: `from src.data_io.data_filters import get_default_featurization`
- Added `featurization` parameter to `get_predictions()`
- Added featurization filter to SQL query
- Fixed YAML path: `plot_hyperparam_combos.yaml` → `combos.yaml`

#### 4.3 `scripts/export_predictions_for_r.py`

**Changes**:
- Added import: `from src.data_io.data_filters import get_default_featurization`
- Added featurization filter to `_load_predictions_from_db()`

---

### 5. Fixed Configuration

**File**: `configs/VISUALIZATION/data_filters.yaml`

**Change**: `classifier: "CatBoost"` → `classifier: "CATBOOST"`

**Reason**: Database stores classifier names in uppercase. Case mismatch caused
queries to return 0 rows.

---

### 6. Fixed Test Expectations

**File**: `tests/test_data_integrity.py`

**Change**: Updated `TestPredictionCounts` class

**Before**: Expected 208 predictions (full dataset)

**After**: Expects ~63 predictions (fold-0 only) with correct class balance ratio (~27% glaucoma)

**Reason**: The predictions table only stores one fold's test set data, not all 208 subjects.

---

### 7. Fixed Figure Annotation (FAILURE-006)

**File**: `src/r/figures/fig_calibration_dca_combined.R`

**Change**: Annotation now clearly labels which curve's metrics are shown

**Before**:
```r
annotation_text <- sprintf(
  "Slope: %.2f, O:E: %.2f\nBrier: %.3f, IPA: %.2f",
  ...
)
```

**After**:
```r
gt_idx <- which(cal_configs$name == "Ground Truth")
annotation_text <- sprintf(
  "Ground Truth:\nSlope: %.2f, O:E: %.2f\nBrier: %.3f, IPA: %.2f",
  ...
)
```

---

## Verification Results

### TDD Tests (All Pass)

```
tests/test_data_integrity.py::TestGroundTruthAUROC::test_ground_truth_in_duckdb PASSED
tests/test_data_integrity.py::TestGroundTruthAUROC::test_ground_truth_in_roc_rc_json PASSED
tests/test_data_integrity.py::TestGroundTruthAUROC::test_ground_truth_computed_from_predictions PASSED
tests/test_data_integrity.py::TestPredictionCounts::test_prediction_count_ground_truth PASSED
tests/test_data_integrity.py::TestFeaturizationFilter::test_data_filters_yaml_exists PASSED
tests/test_data_integrity.py::TestFeaturizationFilter::test_data_filters_has_correct_default PASSED
tests/test_data_integrity.py::TestFeaturizationFilter::test_no_mixed_featurization_in_exports PASSED
tests/test_data_integrity.py::TestCalibrationData::test_calibration_json_exists PASSED
```

### Export Script Output

**ROC+RC Export**:
```
Processing: ground_truth... OK (AUROC=0.9118, AURC=0.0507, with CI)
Processing: best_ensemble... OK (AUROC=0.9130, AURC=0.0456, with CI)
Processing: best_single_fm... OK (AUROC=0.9130, AURC=0.0509, with CI)
Processing: traditional... OK (AUROC=0.8606, AURC=0.0893, with CI)
```

**Selective Classification Export**:
```
Processing: ground_truth... OK (n=63, AUROC@100%=0.9118)
Processing: best_ensemble... OK (n=63, AUROC@100%=0.9130)
```

---

## Regenerated Figures

| Figure | Location | Status |
|--------|----------|--------|
| `fig_roc_rc_combined.png` | `figures/generated/ggplot2/main/` | ✅ Regenerated |
| `fig_calibration_dca_combined.png` | `figures/generated/ggplot2/main/` | ✅ Regenerated |
| `fig_selective_classification.png` | `figures/generated/ggplot2/supplementary/` | ✅ Regenerated |

---

## Expert Reviews Completed

Three background expert agents provided reviews:

### 1. Factorial ANOVA Review (ac94556)

**Key recommendations**:
- Use Aligned Rank Transform (ART) for non-parametric factorial analysis
- Two-stage hierarchical bootstrap for TOP-10 CI (not simple pooling)
- Report partial η² effect sizes for all effects
- Use Type III SS for unbalanced design

### 2. MCID Review (aa3f317)

**Key recommendations**:
- AUROC MCID should be **0.05** (not 0.02) - smaller differences undetectable with N=208
- Net Benefit threshold should be **0.10** (not 0.15) for glaucoma screening
- Add TOST equivalence testing for statistical vs clinical significance
- Power analysis shows MDD ≈ 0.054-0.074 for AUROC

### 3. STRATOS Visualization Review (a1f6e41)

**Key recommendations**:
- Change panel labels from (A,B,C,D) to lowercase (a,b,c,d) per Nature guidelines
- DCA threshold range should be 5%-30% with mandatory treat-all/treat-none references
- Add calibration metrics table to figures
- Specify LOESS parameters explicitly (frac=0.3, bootstrap n=200)

---

## Cross-References

- **FAILURE-005**: `.claude/docs/meta-learnings/FAILURE-005-featurization-not-filtered-again.md`
- **FAILURE-006**: `.claude/docs/meta-learnings/FAILURE-006-ambiguous-annotations-duplicate-legends.md`
- **CRITICAL-FAILURE-002**: `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md`

---

## Remaining Work

1. [ ] Apply expert review recommendations to figures
2. [ ] Update panel labels to lowercase (a,b,c,d)
3. [ ] Review DCA threshold range (consider 0.10 instead of 0.15)
4. [ ] Add TOST equivalence testing to statistical analysis
5. [ ] Audit remaining export scripts for any other missing filters
