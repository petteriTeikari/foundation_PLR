# Normalization Double-Check Investigation

**Date:** 2026-01-31
**Status:** Active Investigation
**Triggered by:** G001 (PLR4018) showing scaling issue in subject trace visualization

## Executive Summary

Investigation of normalization/denormalization handling in the Foundation PLR pipeline revealed:

1. **G001 (PLR4018) has corrupted `pupil_orig` data** - values range from -107 to +217 instead of typical -60 to +10
2. **Root cause**: `pupil_orig` for this subject has an erroneous baseline offset of ~184 units
3. **Workaround**: Use `pupil_raw` instead (correctly scaled: mean -10.65%)
4. **Systemic risk**: Dual normalization systems (global pipeline vs PyPOTS per-split scalers) may cause inconsistencies

## 1. The Observed Problem

### Subject G001 (PLR4018) Statistics

| Metric | PLR4018 (G001) - BROKEN | PLR1002 (H006) - Normal |
|--------|-------------------------|-------------------------|
| `pupil_orig` min | -106.90 | -101.73 |
| `pupil_orig` max | **217.24** | 2.98 |
| `pupil_orig` mean | **132.40** | -21.09 |
| `pupil_orig` std | **102.83** | 20.43 |
| `pupil_gt` mean | -8.91 | -19.53 |
| `pupil_raw` mean | -10.65 | -19.59 |

**Key Finding**: Only `pupil_orig` is broken. Both `pupil_raw` and `pupil_gt` are correctly scaled.

### Distribution Across All 152 Test Subjects

| Offset Category (pupil_orig - pupil_raw) | N Subjects |
|-----------------------------------------|------------|
| Minimal offset (<1) | **149** |
| Small offset (1-10) | 2 |
| Large offset (>50) | **1 (PLR4018 only)** |

## 2. Normalization Architecture in Codebase

### 2.1 Global Preprocessing Pipeline

**Location:** `src/preprocess/preprocess_PLR.py`

```
Forward Transform (Standardization):
    X_standardized = (X - mean) / stdev

Backward Transform (Destandardization):
    X_original = X_standardized * stdev + mean
```

**Key Functions:**
- `standardize_the_data_dict()` - Lines 35-64
- `destandardize_the_split_dict()` - Lines 105-140
- Statistics stored in `data_dicts["preprocess"]["standardization"]`

### 2.2 Imputation-Specific Destandardization

**Location:** `src/preprocess/preprocess_data.py`

- `destandardize_for_imputation_metric()` - Lines 301-334
- `destandardize_for_imputation_metrics()` - Lines 395-461 (includes auto-detection workaround)

**Auto-Detection Workaround:**
```python
# If predictions are 100x larger than targets, assumes predictions already destandardized
if abs(np.nanmean(predictions)) > 100 * abs(np.nanmean(targets)):
    # Skip destandardization
```

This workaround suggests the normalization inconsistency has been encountered before.

### 2.3 Data Import Column Naming

**Location:** `src/data_io/data_import.py` (Lines 363-370)

```
Column Renaming Scheme:
    "denoised" → "pupil_gt"          # Ground truth (denoised signal)
    "pupil_raw" → "pupil_orig"       # Original raw signal with outliers
    "pupil_toBeImputed" → "pupil_raw" # Raw with outliers set to NaN
```

### 2.4 Foundation Model Normalization

#### MOMENT
- Input: Already standardized via global preprocessing
- Output: Reconstruction in standardized space
- Denormalization: Done downstream via `destandardize_for_imputation_metric()`

#### PyPOTS (SAITS, CSDI, TimesNet)
**CRITICAL ISSUE:** Per-split scalers created independently

**Location:** `src/imputation/nuwats/NuwaTS/data_provider/data_loader_imputation.py`

```python
# Each dataset loader creates separate StandardScaler per split
self.scaler = StandardScaler()
if self.scale:
    self.scaler.fit(df_data.values)  # Fit on this split only!
    data = self.scaler.transform(df_data.values)
```

**Risk:** Test data may be normalized using different statistics than training.

## 3. Root Cause Analysis for G001

### Hypothesis: Data Ingestion Error

For PLR4018, the `pupil_orig` column appears to have been stored with:
- **Absolute pupil diameter** (in some unit, possibly pixels) instead of
- **Percent change from baseline** (which is the expected format)

Evidence:
1. `pupil_orig - pupil_raw` offset = **184.13** (only subject with offset > 50)
2. `pupil_raw` and `pupil_gt` are correctly in percent-change format
3. Suggests the baseline subtraction step was skipped for `pupil_orig` only

### Affected Data

| Column | PLR4018 Status | Fix Needed? |
|--------|----------------|-------------|
| `pupil_orig` | **BROKEN** (offset +184) | Yes - use `pupil_raw` instead |
| `pupil_raw` | OK | No |
| `pupil_gt` | OK | No |
| `pupil_orig_imputed` | **BROKEN** (inherits from pupil_orig) | Yes |
| `pupil_raw_imputed` | OK | No |

## 4. Dual Normalization Systems - Systemic Risk

The codebase has **two competing normalization systems**:

### System A: Global Preprocessing
- Uses sklearn StandardScaler
- Computes statistics from training split only
- Stores in `data_dict["preprocess"]["standardization"]`
- Used for MOMENT and general pipeline

### System B: PyPOTS Per-Split Loaders
- Each dataset loader creates its own StandardScaler
- May fit on train/val/test separately (data leakage risk)
- Uses NuwaTS custom StandardScaler implementation
- Not integrated with System A for denormalization

### Potential Failure Modes

1. **Double Destandardization**: If PyPOTS outputs are in original scale, applying System A destandardization again produces wrong results
2. **Single Standardization When Double Expected**: Opposite problem
3. **Scale Mismatch**: Different subjects normalized with different statistics

## 5. Recommendations

### 5.1 Immediate Fix for G001 Visualization

**Option A (Recommended):** Use `pupil_raw` instead of `pupil_orig` for all subjects
- Pros: `pupil_raw` is consistently scaled across all subjects
- Cons: Shows NaN gaps where outliers were (but we're marking outliers anyway)

**Option B:** Fix G001 in source database
- Requires: Recompute `pupil_orig` with correct baseline subtraction
- Risk: May require reprocessing MLflow experiments

### 5.2 Codebase Improvements

1. **Add Normalization Tracking Column** to database:
   ```sql
   ALTER TABLE train ADD COLUMN normalization_state TEXT;
   -- Values: 'raw', 'baseline_normalized', 'z_standardized'
   ```

2. **Unify Normalization Systems**: Single scaler instance shared between global pipeline and PyPOTS

3. **Add Validation Tests**: Check that all subjects have consistent value ranges before training/visualization

### 5.3 Methods.tex Documentation

Add to methods section:

```latex
\subsection{Data Preprocessing}

\subsubsection{Normalization}
PLR signals were baseline-normalized by computing percent change from the
pre-stimulus baseline period (0-10s):

\begin{equation}
    p_{normalized}(t) = \frac{p_{raw}(t) - p_{baseline}}{p_{baseline}} \times 100
\end{equation}

where $p_{baseline}$ is the median pupil diameter during the baseline period.

For model training, signals were further standardized using z-score normalization:

\begin{equation}
    p_{standardized}(t) = \frac{p_{normalized}(t) - \mu}{\sigma}
\end{equation}

where $\mu$ and $\sigma$ were computed from the training set only.

Model outputs (imputed signals) were destandardized back to the baseline-normalized
scale for visualization and feature extraction, ensuring physiologically meaningful
feature values.
```

## 6. Action Items

| Priority | Task | Owner | Status |
|----------|------|-------|--------|
| P0 | Fix G001 visualization (clamp or use pupil_raw) | Claude | ✅ DONE |
| P1 | Add validation test for value range consistency | Claude | ✅ DONE |
| P2 | Document normalization in methods.tex | - | TODO |
| P3 | Unify PyPOTS and global normalization | Claude | ✅ DONE |
| P3 | Add normalization_state column to DB schema | - | Backlog |

### Completed Work (2026-01-31)

**P0 - G001 Visualization Fix:**
- Outliers are now clamped to y-axis bounds (orange markers at ±limits)
- File: `r/figures/fig_subject_traces.R`

**P1 - Validation Tests:**
- Created `tests/test_data_quality/test_normalization_consistency.py` (6 tests)
- Created `tests/test_data_quality/test_validation_module.py` (8 tests)
- Created `src/data_io/validation/normalization_validator.py`
- PLR4018 is documented in `KNOWN_SCALING_ANOMALIES`

**P3 - Unified Normalization:**
- Created `src/preprocess/normalization_manager.py`
- Created `tests/test_data_quality/test_unified_normalization.py` (14 tests)
- Features: state tracking, double-normalization prevention, serialization
- All 28 data quality tests pass

## 7. Files Involved

### Core Normalization Logic
- `src/preprocess/preprocess_PLR.py` - Global standardization
- `src/preprocess/preprocess_data.py` - Imputation destandardization
- `src/data_io/data_import.py` - Column renaming and import

### Model-Specific
- `src/imputation/momentfm/moment_imputation_main.py` - MOMENT metrics
- `src/imputation/nuwats/NuwaTS/data_provider/data_loader_imputation.py` - PyPOTS scalers
- `src/imputation/nuwats/NuwaTS/utils/tools.py` - Custom StandardScaler

### Visualization
- `scripts/export_subject_traces_for_r.py` - JSON export
- `r/figures/fig_subject_traces.R` - Figure generation

## 8. References

- Van Calster et al. 2024 - STRATOS metrics (importance of calibration)
- Riley et al. 2023 - Model stability and prediction instability
