# Pipeline Robustification Plan

**Created:** 2026-01-26
**Last Updated:** 2026-01-26 (Round 1 code review)
**Status:** READY FOR IMPLEMENTATION
**Priority:** CRITICAL - Scientific Integrity

---

## Executive Summary

The MLflow → DuckDB → R/ggplot2 figure pipeline has critical flaws that allowed hallucinated/incorrect metric values to appear in scientific figures. This document provides **concrete, TDD-compliant, production-grade fixes**.

**Code Review Status:** Round 1 complete - 5 reviewers identified gaps. This version addresses all findings.

---

## TDD Implementation Order

**CRITICAL:** Follow this EXACT order. Write tests FIRST, then implement.

```
┌─────────────────────────────────────────────────────────────────────┐
│ TDD CYCLE 1: Registry Integration (ROOT CAUSE FIX)                  │
│                                                                     │
│ 1. RED:   Write tests/integration/test_extraction_registry.py      │
│ 2. RED:   Run tests - confirm FAILURE                               │
│ 3. GREEN: Modify scripts/extract_all_configs_to_duckdb.py          │
│ 4. GREEN: Run tests - confirm PASS                                  │
│ 5. REFACTOR: Clean up, add logging                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TDD CYCLE 2: Statistical CI Aggregation Fix                         │
│                                                                     │
│ 1. RED:   Write tests/unit/test_ci_aggregation.py                  │
│ 2. RED:   Run tests - confirm FAILURE                               │
│ 3. GREEN: Modify src/r/figures/fig02_forest_outlier.R              │
│ 4. GREEN: Run tests - confirm PASS                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TDD CYCLE 3: Balanced Factorial Design Fix                          │
│                                                                     │
│ 1. RED:   Write tests/unit/test_balanced_subset.py                 │
│ 2. RED:   Run tests - confirm FAILURE                               │
│ 3. GREEN: Modify R aggregation to use balanced subset              │
│ 4. GREEN: Run tests - confirm PASS                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TDD CYCLE 4: E2E Pipeline Validation                                │
│                                                                     │
│ 1. RED:   Write tests/e2e/test_full_pipeline.py                    │
│ 2. Run full extraction + figure generation                          │
│ 3. GREEN: All validation passes                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## RESOLVED DECISIONS (No Ambiguity)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Keep redundant featurization filter in export? | **YES** | Defense-in-depth for scientific integrity |
| R paths: `here::here()` vs env variable? | **`here::here()`** | R convention, no shell setup needed |
| Duplicate run handling? | **Keep highest AUROC** | Current behavior is correct |
| Invalid method handling? | **Log warning + skip** | Don't fail entire extraction |
| CI aggregation method? | **Conservative bounds** | Guaranteed coverage, simple |
| Unbalanced design fix? | **Balanced 5-imputation subset** | Transparent, valid comparison |

---

## TDD CYCLE 1: Registry Integration (ROOT CAUSE)

### Step 1: RED - Write Failing Tests

Create `tests/integration/test_extraction_registry.py`:

```python
"""
Integration tests for extraction → registry validation.

TDD: Write these tests FIRST, then implement the fix.
Run: pytest tests/integration/test_extraction_registry.py -v
Expected: FAIL (extraction doesn't use registry yet)
"""

import pytest
import duckdb
from pathlib import Path

from src.data_io.registry import (
    get_valid_outlier_methods,
    get_valid_imputation_methods,
    get_valid_classifiers,
    EXPECTED_OUTLIER_COUNT,
    EXPECTED_IMPUTATION_COUNT,
    EXPECTED_CLASSIFIER_COUNT,
)

DB_PATH = Path("outputs/foundation_plr_results.db")


@pytest.fixture
def db_connection():
    """Connect to extraction database."""
    if not DB_PATH.exists():
        pytest.skip(f"Database not found: {DB_PATH}. Run extraction first.")
    return duckdb.connect(str(DB_PATH), read_only=True)


class TestExtractionUsesRegistry:
    """Verify extraction only includes registry-validated methods."""

    def test_all_outlier_methods_are_valid(self, db_connection):
        """Every extracted outlier method must be in the registry."""
        extracted = db_connection.execute(
            "SELECT DISTINCT outlier_method FROM essential_metrics"
        ).fetchall()
        extracted_methods = {row[0] for row in extracted}
        valid_methods = set(get_valid_outlier_methods())

        invalid = extracted_methods - valid_methods
        assert not invalid, (
            f"Extracted INVALID outlier methods: {invalid}. "
            f"These are not in the registry and must be excluded."
        )

    def test_all_imputation_methods_are_valid(self, db_connection):
        """Every extracted imputation method must be in the registry."""
        extracted = db_connection.execute(
            "SELECT DISTINCT imputation_method FROM essential_metrics"
        ).fetchall()
        extracted_methods = {row[0] for row in extracted}
        valid_methods = set(get_valid_imputation_methods())

        invalid = extracted_methods - valid_methods
        assert not invalid, (
            f"Extracted INVALID imputation methods: {invalid}. "
            f"These are not in the registry and must be excluded."
        )

    def test_outlier_count_not_exceeds_registry(self, db_connection):
        """Cannot have more outlier methods than registry defines."""
        count = db_connection.execute(
            "SELECT COUNT(DISTINCT outlier_method) FROM essential_metrics"
        ).fetchone()[0]

        assert count <= EXPECTED_OUTLIER_COUNT, (
            f"Extracted {count} outlier methods, registry defines {EXPECTED_OUTLIER_COUNT}. "
            f"Extraction is including invalid methods!"
        )

    def test_no_anomaly_method(self, db_connection):
        """'anomaly' is garbage and must NEVER appear."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics WHERE outlier_method = 'anomaly'"
        ).fetchone()[0]
        assert result == 0, "'anomaly' found in extraction - this is INVALID!"

    def test_no_exclude_method(self, db_connection):
        """'exclude' is garbage and must NEVER appear."""
        result = db_connection.execute(
            "SELECT COUNT(*) FROM essential_metrics WHERE outlier_method = 'exclude'"
        ).fetchone()[0]
        assert result == 0, "'exclude' found in extraction - this is INVALID!"


class TestGroundTruthAUROC:
    """Verify ground truth AUROC matches expected value."""

    EXPECTED_GT_AUROC = 0.911
    TOLERANCE = 0.002

    def test_ground_truth_config_exists(self, db_connection):
        """Ground truth (pupil-gt + pupil-gt + CatBoost) must exist."""
        count = db_connection.execute("""
            SELECT COUNT(*) FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND UPPER(classifier) = 'CATBOOST'
        """).fetchone()[0]

        assert count == 1, f"Expected 1 ground truth config, found {count}"

    def test_ground_truth_auroc_value(self, db_connection):
        """Ground truth AUROC must be 0.911 +/- 0.002."""
        auroc = db_connection.execute("""
            SELECT auroc FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND UPPER(classifier) = 'CATBOOST'
        """).fetchone()[0]

        assert abs(auroc - self.EXPECTED_GT_AUROC) < self.TOLERANCE, (
            f"Ground truth AUROC is {auroc:.4f}, expected {self.EXPECTED_GT_AUROC} +/- {self.TOLERANCE}. "
            f"DO NOT change expected value - find the extraction bug!"
        )
```

### Step 2: Run Tests - Confirm FAILURE

```bash
pytest tests/integration/test_extraction_registry.py -v
# Expected: FAILED - extraction includes 'anomaly', 'exclude', wrong counts
```

### Step 3: GREEN - Implement Registry Integration

Modify `scripts/extract_all_configs_to_duckdb.py`:

**Add imports after line 23:**
```python
from src.data_io.registry import (
    validate_outlier_method,
    validate_imputation_method,
    get_valid_classifiers,
    EXPECTED_OUTLIER_COUNT,
    EXPECTED_IMPUTATION_COUNT,
)

# Create case-insensitive classifier set
VALID_CLASSIFIERS_UPPER = frozenset(c.upper() for c in get_valid_classifiers())
```

**REMOVE line 32:**
```python
# DELETE THIS LINE:
EXCLUDED_OUTLIER_SOURCES = ["anomaly", "exclude"]
```

**Modify `scan_all_runs()` function - add validation after parsing (around line 247):**
```python
config = parse_run_name(model_file.stem)

# REGISTRY VALIDATION - skip invalid methods
if not validate_outlier_method(config["outlier"]):
    logger.warning(f"Skipping invalid outlier method: {config['outlier']} (not in registry)")
    continue

if not validate_imputation_method(config["imputation"]):
    logger.warning(f"Skipping invalid imputation method: {config['imputation']} (not in registry)")
    continue

if config["classifier"].upper() not in VALID_CLASSIFIERS_UPPER:
    logger.warning(f"Skipping invalid classifier: {config['classifier']} (not in registry)")
    continue

# Rest of processing...
```

**Add to `verify_extraction()` function:**
```python
def verify_extraction(output_path: Path) -> None:
    """Verify extraction matches registry expectations."""
    conn = duckdb.connect(str(output_path), read_only=True)

    # Verify no invalid methods
    invalid_outliers = conn.execute("""
        SELECT DISTINCT outlier_method FROM essential_metrics
        WHERE outlier_method IN ('anomaly', 'exclude')
    """).fetchall()
    if invalid_outliers:
        raise ValueError(f"INVALID outlier methods found: {invalid_outliers}")

    # Verify counts don't exceed registry
    n_outliers = conn.execute(
        "SELECT COUNT(DISTINCT outlier_method) FROM essential_metrics"
    ).fetchone()[0]
    if n_outliers > EXPECTED_OUTLIER_COUNT:
        raise ValueError(
            f"Extracted {n_outliers} outlier methods, registry defines {EXPECTED_OUTLIER_COUNT}"
        )

    n_imputations = conn.execute(
        "SELECT COUNT(DISTINCT imputation_method) FROM essential_metrics"
    ).fetchone()[0]
    if n_imputations > EXPECTED_IMPUTATION_COUNT:
        raise ValueError(
            f"Extracted {n_imputations} imputation methods, registry defines {EXPECTED_IMPUTATION_COUNT}"
        )

    # Verify ground truth AUROC
    gt_auroc = conn.execute("""
        SELECT auroc FROM essential_metrics
        WHERE outlier_method = 'pupil-gt'
          AND imputation_method = 'pupil-gt'
          AND UPPER(classifier) = 'CATBOOST'
    """).fetchone()

    if gt_auroc is None:
        raise ValueError("Ground truth config not found!")

    EXPECTED_GT_AUROC = 0.911
    if abs(gt_auroc[0] - EXPECTED_GT_AUROC) > 0.002:
        raise ValueError(
            f"Ground truth AUROC {gt_auroc[0]:.4f} outside expected range [0.909, 0.913]"
        )

    conn.close()
    print(f"Registry validation PASSED: {n_outliers} outliers, {n_imputations} imputations")
```

### Step 4: Run Tests - Confirm PASS

```bash
# Re-run extraction with fixes
python scripts/extract_all_configs_to_duckdb.py

# Run tests
pytest tests/integration/test_extraction_registry.py -v
# Expected: PASSED
```

---

## TDD CYCLE 2: Statistical CI Aggregation Fix

### Step 1: RED - Write Failing Tests

Create `tests/unit/test_ci_aggregation.py`:

```python
"""
Unit tests for CI aggregation - must use conservative bounds, NOT averaging.

TDD: Write FIRST, then fix R scripts.
"""

import pytest
import pandas as pd
import numpy as np


def aggregate_ci_conservative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CIs using conservative (envelope) bounds.

    This is the CORRECT method - averaging CI bounds is INVALID.
    """
    return df.groupby('outlier_method').agg(
        auroc_mean=('auroc', 'mean'),
        auroc_ci_lo=('auroc_ci_lo', 'min'),  # Conservative: minimum of lower bounds
        auroc_ci_hi=('auroc_ci_hi', 'max'),  # Conservative: maximum of upper bounds
        n_configs=('auroc', 'count')
    ).reset_index()


def aggregate_ci_wrong(df: pd.DataFrame) -> pd.DataFrame:
    """WRONG method - averaging CI bounds. DO NOT USE."""
    return df.groupby('outlier_method').agg(
        auroc_mean=('auroc', 'mean'),
        auroc_ci_lo=('auroc_ci_lo', 'mean'),  # WRONG!
        auroc_ci_hi=('auroc_ci_hi', 'mean'),  # WRONG!
    ).reset_index()


class TestCIAggregation:
    """Test that CI aggregation uses conservative bounds."""

    @pytest.fixture
    def sample_data(self):
        """Sample data with multiple configs per outlier method."""
        return pd.DataFrame({
            'outlier_method': ['LOF', 'LOF', 'LOF', 'pupil-gt', 'pupil-gt'],
            'imputation_method': ['SAITS', 'CSDI', 'TimesNet', 'SAITS', 'CSDI'],
            'auroc': [0.85, 0.87, 0.84, 0.91, 0.90],
            'auroc_ci_lo': [0.80, 0.82, 0.79, 0.88, 0.87],
            'auroc_ci_hi': [0.90, 0.92, 0.89, 0.94, 0.93],
        })

    def test_conservative_uses_min_for_lower_bound(self, sample_data):
        """Conservative CI uses MIN of lower bounds."""
        result = aggregate_ci_conservative(sample_data)
        lof_row = result[result['outlier_method'] == 'LOF'].iloc[0]

        # LOF has CI_lo values: 0.80, 0.82, 0.79 → min = 0.79
        assert lof_row['auroc_ci_lo'] == 0.79

    def test_conservative_uses_max_for_upper_bound(self, sample_data):
        """Conservative CI uses MAX of upper bounds."""
        result = aggregate_ci_conservative(sample_data)
        lof_row = result[result['outlier_method'] == 'LOF'].iloc[0]

        # LOF has CI_hi values: 0.90, 0.92, 0.89 → max = 0.92
        assert lof_row['auroc_ci_hi'] == 0.92

    def test_wrong_method_averages_bounds(self, sample_data):
        """Verify the WRONG method produces different (incorrect) results."""
        correct = aggregate_ci_conservative(sample_data)
        wrong = aggregate_ci_wrong(sample_data)

        lof_correct = correct[correct['outlier_method'] == 'LOF'].iloc[0]
        lof_wrong = wrong[wrong['outlier_method'] == 'LOF'].iloc[0]

        # Wrong method: mean([0.80, 0.82, 0.79]) = 0.803...
        # Correct method: min([0.80, 0.82, 0.79]) = 0.79
        assert lof_wrong['auroc_ci_lo'] != lof_correct['auroc_ci_lo']

    def test_conservative_ci_is_wider(self, sample_data):
        """Conservative CI should be WIDER than averaged CI."""
        correct = aggregate_ci_conservative(sample_data)
        wrong = aggregate_ci_wrong(sample_data)

        for method in ['LOF', 'pupil-gt']:
            correct_row = correct[correct['outlier_method'] == method].iloc[0]
            wrong_row = wrong[wrong['outlier_method'] == method].iloc[0]

            correct_width = correct_row['auroc_ci_hi'] - correct_row['auroc_ci_lo']
            wrong_width = wrong_row['auroc_ci_hi'] - wrong_row['auroc_ci_lo']

            assert correct_width >= wrong_width, (
                f"{method}: Conservative CI width ({correct_width:.3f}) should be >= "
                f"averaged width ({wrong_width:.3f})"
            )
```

### Step 2: GREEN - Fix R Scripts

Modify `src/r/figures/fig02_forest_outlier.R`:

**BEFORE (around lines 42-46):**
```r
outlier_summary <- metrics %>%
  group_by(outlier_method) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = mean(auroc_ci_lo, na.rm = TRUE),  # WRONG!
    auroc_ci_hi = mean(auroc_ci_hi, na.rm = TRUE),  # WRONG!
    n_configs = n(),
    .groups = "drop"
  )
```

**AFTER:**
```r
# CRITICAL: Use conservative bounds (min/max), NOT averaging
# Averaging CI bounds is STATISTICALLY INVALID
# See: docs/planning/pipeline-robustification-plan.md, TDD Cycle 2
outlier_summary <- metrics %>%
  group_by(outlier_method) %>%
  summarize(
    auroc_mean = mean(auroc, na.rm = TRUE),
    auroc_ci_lo = min(auroc_ci_lo, na.rm = TRUE),  # CONSERVATIVE: envelope lower
    auroc_ci_hi = max(auroc_ci_hi, na.rm = TRUE),  # CONSERVATIVE: envelope upper
    n_configs = n(),
    .groups = "drop"
  )
```

Apply same fix to `src/r/figures/fig03_forest_imputation.R`.

---

## TDD CYCLE 3: Balanced Factorial Design Fix

### Step 1: RED - Write Failing Tests

Create `tests/unit/test_balanced_subset.py`:

```python
"""
Unit tests for balanced factorial subset filtering.

The experimental design is unbalanced - different outlier methods have
different numbers of imputation methods tested. For valid comparison,
we must filter to a balanced subset.
"""

import pytest
import pandas as pd


def get_balanced_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to balanced factorial subset.

    Returns only (outlier, imputation) combinations where EVERY outlier
    method has been tested with EVERY imputation method in the subset.
    """
    # Find imputation methods available for ALL outlier methods
    imps_per_outlier = df.groupby('outlier_method')['imputation_method'].apply(set)
    common_imps = set.intersection(*imps_per_outlier.values)

    if not common_imps:
        raise ValueError("No common imputation methods across all outlier methods!")

    # Filter to balanced subset
    balanced = df[df['imputation_method'].isin(common_imps)].copy()

    # Verify balance
    counts = balanced.groupby('outlier_method').size()
    if counts.nunique() != 1:
        raise ValueError(f"Imbalanced result: {counts.to_dict()}")

    return balanced


class TestBalancedSubset:
    """Test balanced factorial subset filtering."""

    @pytest.fixture
    def unbalanced_data(self):
        """Unbalanced design: pupil-gt has extra imputation."""
        return pd.DataFrame({
            'outlier_method': [
                'pupil-gt', 'pupil-gt', 'pupil-gt', 'pupil-gt',  # 4 imputations
                'LOF', 'LOF', 'LOF',  # 3 imputations (missing pupil-gt imp)
                'MOMENT-gt-finetune', 'MOMENT-gt-finetune', 'MOMENT-gt-finetune',  # 3
            ],
            'imputation_method': [
                'SAITS', 'CSDI', 'TimesNet', 'pupil-gt',  # pupil-gt has extra
                'SAITS', 'CSDI', 'TimesNet',
                'SAITS', 'CSDI', 'TimesNet',
            ],
            'auroc': [0.91, 0.90, 0.89, 0.92, 0.85, 0.86, 0.84, 0.88, 0.87, 0.86],
        })

    def test_returns_balanced_counts(self, unbalanced_data):
        """All outlier methods should have same count after filtering."""
        balanced = get_balanced_subset(unbalanced_data)
        counts = balanced.groupby('outlier_method').size()

        assert counts.nunique() == 1, f"Unequal counts: {counts.to_dict()}"

    def test_excludes_non_common_imputations(self, unbalanced_data):
        """pupil-gt imputation should be excluded (not available for LOF)."""
        balanced = get_balanced_subset(unbalanced_data)

        assert 'pupil-gt' not in balanced['imputation_method'].values

    def test_keeps_common_imputations(self, unbalanced_data):
        """SAITS, CSDI, TimesNet should be kept (available for all)."""
        balanced = get_balanced_subset(unbalanced_data)
        kept = set(balanced['imputation_method'].unique())

        assert kept == {'SAITS', 'CSDI', 'TimesNet'}

    def test_balanced_has_fewer_rows(self, unbalanced_data):
        """Balanced subset should have fewer rows than original."""
        balanced = get_balanced_subset(unbalanced_data)

        assert len(balanced) < len(unbalanced_data)
        assert len(balanced) == 9  # 3 outliers × 3 imputations
```

### Step 2: GREEN - Fix R Scripts

Modify `src/r/figures/fig02_forest_outlier.R`:

**Add after loading data (around line 30):**
```r
# CRITICAL: Filter to balanced factorial subset
# Different outlier methods have different imputation coverage
# For valid comparison, use only common imputations
# See: docs/planning/pipeline-robustification-plan.md, TDD Cycle 3

# Find imputation methods available for ALL outlier methods
imp_coverage <- metrics %>%
  group_by(imputation_method) %>%
  summarize(n_outlier_methods = n_distinct(outlier_method), .groups = "drop")

max_coverage <- max(imp_coverage$n_outlier_methods)
common_imputations <- imp_coverage %>%
  filter(n_outlier_methods == max_coverage) %>%
  pull(imputation_method)

cat(sprintf("Using balanced subset: %d common imputation methods\n", length(common_imputations)))
cat(sprintf("Common imputations: %s\n", paste(common_imputations, collapse = ", ")))

# Filter to balanced subset
metrics_balanced <- metrics %>%
  filter(imputation_method %in% common_imputations)

# Verify balance
balance_check <- metrics_balanced %>%
  count(outlier_method) %>%
  pull(n)

stopifnot("Imbalanced design!" = length(unique(balance_check)) == 1)

# Use balanced metrics for aggregation
metrics <- metrics_balanced
```

---

## TDD CYCLE 4: E2E Pipeline Validation

### Step 1: RED - Write E2E Test

Create `tests/e2e/test_full_pipeline.py`:

```python
"""
End-to-end pipeline test: MLflow → DuckDB → CSV → R → Figure

This test validates the entire pipeline produces correct output.
"""

import pytest
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import duckdb
import pandas as pd


class TestFullPipeline:
    """E2E validation of the full pipeline."""

    DB_PATH = Path("outputs/foundation_plr_results.db")
    CSV_PATH = Path("outputs/r_data/essential_metrics.csv")
    FIGURE_PATH = Path("figures/generated/ggplot2/fig02_forest_outlier.pdf")

    def test_database_exists(self):
        """Extraction output exists."""
        assert self.DB_PATH.exists(), f"Database not found: {self.DB_PATH}"

    def test_csv_export_exists(self):
        """CSV export for R exists."""
        assert self.CSV_PATH.exists(), f"CSV not found: {self.CSV_PATH}"

    def test_csv_matches_database_count(self):
        """CSV row count matches database."""
        conn = duckdb.connect(str(self.DB_PATH), read_only=True)
        db_count = conn.execute("SELECT COUNT(*) FROM essential_metrics").fetchone()[0]
        conn.close()

        csv_df = pd.read_csv(self.CSV_PATH)

        assert len(csv_df) == db_count, (
            f"CSV has {len(csv_df)} rows, database has {db_count}"
        )

    def test_ground_truth_auroc_in_database(self):
        """Ground truth AUROC is 0.911 +/- 0.002."""
        conn = duckdb.connect(str(self.DB_PATH), read_only=True)
        auroc = conn.execute("""
            SELECT auroc FROM essential_metrics
            WHERE outlier_method = 'pupil-gt'
              AND imputation_method = 'pupil-gt'
              AND UPPER(classifier) = 'CATBOOST'
        """).fetchone()[0]
        conn.close()

        assert 0.909 <= auroc <= 0.913, f"GT AUROC {auroc} outside [0.909, 0.913]"

    def test_no_invalid_outlier_methods(self):
        """No 'anomaly' or 'exclude' in database."""
        conn = duckdb.connect(str(self.DB_PATH), read_only=True)
        invalid = conn.execute("""
            SELECT DISTINCT outlier_method FROM essential_metrics
            WHERE outlier_method IN ('anomaly', 'exclude')
        """).fetchall()
        conn.close()

        assert len(invalid) == 0, f"Invalid methods found: {invalid}"

    def test_figure_exists(self):
        """Forest plot figure exists."""
        assert self.FIGURE_PATH.exists(), f"Figure not found: {self.FIGURE_PATH}"

    def test_figure_not_stale(self):
        """Figure is newer than database (or within 1 hour)."""
        if not self.FIGURE_PATH.exists() or not self.DB_PATH.exists():
            pytest.skip("Files not found")

        fig_mtime = datetime.fromtimestamp(self.FIGURE_PATH.stat().st_mtime)
        db_mtime = datetime.fromtimestamp(self.DB_PATH.stat().st_mtime)

        # Figure should not be more than 1 hour older than database
        assert fig_mtime > db_mtime - timedelta(hours=1), (
            f"Figure is STALE: generated {fig_mtime}, database updated {db_mtime}"
        )
```

---

## Files To Modify (Complete List)

| File | Changes Required |
|------|------------------|
| `scripts/extract_all_configs_to_duckdb.py` | Add registry imports, remove hardcoded exclusions, add validation |
| `scripts/export_data_for_r.py` | Keep featurization filter (defense-in-depth), add logging |
| `src/r/figures/fig02_forest_outlier.R` | Fix CI aggregation, add balanced subset filter, use `here::here()` |
| `src/r/figures/fig03_forest_imputation.R` | Same fixes as fig02 |
| `tests/integration/test_extraction_registry.py` | CREATE - registry integration tests |
| `tests/unit/test_ci_aggregation.py` | CREATE - CI aggregation tests |
| `tests/unit/test_balanced_subset.py` | CREATE - balanced design tests |
| `tests/e2e/test_full_pipeline.py` | CREATE - E2E pipeline tests |

---

## R Script Path Fix (All R Files)

Modify ALL R scripts in `src/r/figures/` to use `here::here()`:

**BEFORE:**
```r
source("r/theme_foundation_plr.R")
source("r/color_palettes.R")
```

**AFTER:**
```r
library(here)
source(here("src", "r", "theme_foundation_plr.R"))
source(here("src", "r", "color_palettes.R"))
```

Ensure `here` package is installed:
```r
install.packages("here")
```

---

## Verification Commands (Run ALL Before Complete)

```bash
# 1. Registry integrity (anti-cheat)
make verify-registry-integrity

# 2. Registry unit tests (53 tests)
make test-registry

# 3. Run TDD Cycle 1 tests (extraction integration)
pytest tests/integration/test_extraction_registry.py -v

# 4. Run TDD Cycle 2+3 tests (statistics)
pytest tests/unit/test_ci_aggregation.py tests/unit/test_balanced_subset.py -v

# 5. Re-run extraction with fixes
python scripts/extract_all_configs_to_duckdb.py

# 6. Re-run export
python scripts/export_data_for_r.py

# 7. Regenerate figures
cd /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR
Rscript src/r/figures/fig02_forest_outlier.R
Rscript src/r/figures/fig03_forest_imputation.R

# 8. Run E2E tests
pytest tests/e2e/test_full_pipeline.py -v

# 9. Visual verification
# Open figures/generated/ggplot2/fig02_forest_outlier.pdf
# Verify ground truth (pupil-gt) shows AUROC ~0.91, NOT ~0.88
```

---

## Success Criteria

The pipeline is "production-grade" when ALL of these pass:

- [ ] `pytest tests/integration/test_extraction_registry.py` - PASSED
- [ ] `pytest tests/unit/test_ci_aggregation.py` - PASSED
- [ ] `pytest tests/unit/test_balanced_subset.py` - PASSED
- [ ] `pytest tests/e2e/test_full_pipeline.py` - PASSED
- [ ] `make check-registry` - PASSED
- [ ] Ground truth AUROC in figure = 0.91 (visually verified)
- [ ] No 'anomaly' or 'exclude' in any output
- [ ] Figure generated after database (not stale)

---

## Historical Issues (Reference)

The following issues from the original plan are now addressed:

| Issue | Original Status | New Status | Fix Location |
|-------|-----------------|------------|--------------|
| Issue 3: No expected value test | NOT FIXED | **FIXED** | TDD Cycle 1, `test_ground_truth_auroc_value` |
| Issue 4: R path fragility | NOT FIXED | **FIXED** | `here::here()` pattern |
| Issue 5: No E2E test | NOT FIXED | **FIXED** | TDD Cycle 4 |
| Issue 8: anomaly/exclude in data | NOT FIXED | **FIXED** | TDD Cycle 1, registry validation |
| Issue 9: Invalid CI aggregation | CRITICALLY BROKEN | **FIXED** | TDD Cycle 2, conservative bounds |
| Issue 10: Unbalanced design | CRITICALLY BROKEN | **FIXED** | TDD Cycle 3, balanced subset |
| Issue 11: Extraction ignores registry | ROOT CAUSE | **FIXED** | TDD Cycle 1, registry integration |
| Issue 12: Stale figures | NOT FIXED | **FIXED** | TDD Cycle 4, staleness test |

---

## Anti-Cheat Mechanism (Preserved)

The four-source verification remains in place:

1. `configs/mlflow_registry/parameters/classification.yaml` - Source of truth
2. `configs/registry_canary.yaml` - Canary with checksums
3. `src/data_io/registry.py` - Module constants
4. `tests/test_registry.py` - Hardcoded test assertions

Run `make verify-registry-integrity` to verify all four agree.
