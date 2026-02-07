# FAILURE-005: Featurization Not Filtered (CRITICAL-FAILURE-002 Recurring)

## Severity: CRITICAL

## The Failure

`fig_roc_rc_combined.png` shows Ground Truth AUROC = 0.850 instead of the correct value 0.9110.

This is because `scripts/export_roc_rc_data.py` does NOT filter by featurization, causing it to mix:
- `simple1.0` (handcrafted features): AUROC = 0.9110 ✓
- `MOMENT-embedding-PCA`: AUROC = 0.7747 ✗
- `MOMENT-embedding`: AUROC = 0.7795 ✗

## Evidence

```sql
-- essential_metrics shows MULTIPLE featurizations:
pupil-gt + pupil-gt + CATBOOST | simple1.0: AUROC = 0.9110  ← CORRECT
pupil-gt + pupil-gt + CATBOOST | MOMENT-embedding-PCA: AUROC = 0.7747  ← WRONG
pupil-gt + pupil-gt + CATBOOST | MOMENT-embedding: AUROC = 0.7795  ← WRONG

-- predictions table shows MIXED featurizations:
MOMENT-embedding-PCA: N = 63
MOMENT-embedding: N = 63
simple1.0: N = 63
TOTAL: 189 (should be 208 for single featurization)
```

## Root Cause

`scripts/export_roc_rc_data.py` queries predictions WITHOUT filtering by featurization:

```python
# WRONG - no featurization filter
result = conn.execute("""
    SELECT y_true, y_prob
    FROM predictions
    WHERE outlier_method = ? AND imputation_method = ? AND classifier = 'CATBOOST'
""", ...)
```

Should be:

```python
# CORRECT - filter for handcrafted features only
result = conn.execute("""
    SELECT y_true, y_prob
    FROM predictions
    WHERE outlier_method = ? AND imputation_method = ? AND classifier = 'CATBOOST'
    AND featurization = 'simple1.0'
""", ...)
```

## This Is CRITICAL-FAILURE-002 Recurring

See: `.claude/docs/meta-learnings/CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md`

Despite documenting this exact failure previously, the bug was not fixed in ALL export scripts. The fix was applied to some scripts but not `export_roc_rc_data.py`.

## Impact

- Ground Truth shown as worst performer (AUROC 0.850) instead of best (0.911)
- Misleading figure that contradicts the paper's findings
- User had to discover this after "hours and hours of token wasting"

## Prevention Required

### 1. Single Source of Truth for Featurization Filter

Create a shared constant:

```python
# src/constants.py
FEATURIZATION_FILTER = "simple1.0"  # Handcrafted features ONLY
```

Import in ALL export scripts.

### 2. TDD for Data Provenance

Create tests that verify:
- Ground Truth AUROC = 0.911 ± 0.002 in ALL figures
- N predictions = 208 for classification figures
- Featurization = 'simple1.0' in all queries

### 3. Grep All Export Scripts

```bash
# Find ALL scripts that query predictions without featurization filter
grep -r "FROM predictions" scripts/ | grep -v "featurization"
```

## Files Affected

- `scripts/export_roc_rc_data.py` - MUST add featurization filter
- `data/r_data/roc_rc_data.json` - regenerate after fix
- `figures/generated/ggplot2/main/fig_roc_rc_combined.png` - wrong data

## Status: RESOLVED (2026-01-28)

- [x] Fix `export_roc_rc_data.py` to filter by `featurization = 'simple1.0'`
- [x] Regenerate `roc_rc_data.json`
- [x] Regenerate `fig_roc_rc_combined.png`
- [x] Verify Ground Truth AUROC = 0.9118 (within expected 0.911 ± 0.002)
- [x] Add TDD test for AUROC values (`tests/test_data_integrity.py`)
- [ ] Move figure to supplementary (per user request) - PENDING
- [x] Audit ALL other export scripts for same bug

## Resolution Details

See: `.claude/planning/data-integrity-fixes-2026-01-28.md` for full audit trail.

### Key Changes Made

1. **Created `configs/VISUALIZATION/data_filters.yaml`** - Single source of truth for featurization filter
2. **Created `src/data_io/data_filters.py`** - Python module to load filters
3. **Created `tests/test_data_integrity.py`** - TDD tests for data integrity
4. **Fixed export scripts**:
   - `scripts/export_roc_rc_data.py`
   - `scripts/export_selective_classification_data.py`
   - `scripts/export_predictions_for_r.py`

### Verification

All 8 TDD tests pass:
```
pytest tests/test_data_integrity.py -v
# Ground Truth AUROC = 0.9118 ✓
```
