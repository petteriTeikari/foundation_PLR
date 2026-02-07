# Pipeline Robustness Improvement Plan

## Executive Summary

During synthetic pipeline testing, we identified multiple issues that only surfaced at runtime. These issues could have been caught earlier with proper testing, validation, and fail-fast mechanisms. This document outlines a comprehensive plan to make the pipeline more robust and production-grade.

---

## Issues Discovered During Synthetic Pipeline Testing

### 1. MOMENT Model Handling in Ensemble (Critical)
- **File**: `src/ensemble/ensemble_utils.py`
- **Issue**: `get_best_moment_variant()` and related functions assumed MOMENT models always exist
- **Root Cause**: No defensive coding for missing model types
- **Impact**: Pipeline crash when running with non-MOMENT configurations

### 2. Light Timing Sort Logic (Critical)
- **File**: `src/featurization/feature_utils.py`
- **Issue**: `get_top1_of_col()` sorted by column value instead of time
- **Root Cause**: Incorrect assumption that sorting by light column value (all 1s) would differentiate rows
- **Impact**: Assertion failure with identical onset/offset times

### 3. Function Parameter Name Mismatch (High)
- **File**: `src/classification/xgboost_cls/xgboost_main.py`
- **Issue**: Called `cls_preprocess_cfg=` but function signature had `_cls_preprocess_cfg`
- **Root Cause**: Parameter name changed without updating all call sites
- **Impact**: TypeError at runtime

### 4. Division by Zero in Weight Normalization (Medium)
- **File**: `src/classification/weighing_utils.py`
- **Issue**: `normalize_to_unity()` divided by `nanmax()` without checking for zero
- **Root Cause**: No defensive coding for edge case data
- **Impact**: ZeroDivisionError with synthetic data

### 5. Missing None Guards in Logging (Medium)
- **File**: `src/classification/classifier_log_utils.py`
- **Issue**: Functions accessed metrics dicts without checking for None
- **Root Cause**: Assumed all metrics would always be computed
- **Impact**: AttributeError when optional metrics unavailable

---

## Proposed Solutions

### Phase 1: Static Analysis & Type Checking (Week 1)

#### 1.1 Enable Strict Type Checking
```bash
# Add to pyproject.toml
[tool.mypy]
strict = true
warn_unused_ignores = true
show_error_codes = true
```

**Files to prioritize:**
- `src/ensemble/ensemble_utils.py`
- `src/classification/classifier_log_utils.py`
- `src/classification/weighing_utils.py`
- `src/featurization/feature_utils.py`

#### 1.2 Add Function Signature Validation
Create pre-commit hook to validate parameter names match between calls and definitions:
```python
# scripts/check_parameter_names.py
"""
Use AST to validate function call parameter names match definitions.
"""
```

### Phase 2: Unit Tests for Edge Cases (Week 2)

#### 2.1 Test Cases for ensemble_utils.py
```python
# tests/unit/test_ensemble_utils.py
def test_get_best_moment_variant_no_moment_models():
    """Pipeline should handle configs without MOMENT models."""

def test_get_best_moment_variant_empty_dataframe():
    """Should return gracefully with empty input."""

def test_get_best_moments_per_source_mixed_models():
    """Test with mix of MOMENT and non-MOMENT models."""
```

#### 2.2 Test Cases for feature_utils.py
```python
# tests/unit/test_feature_utils.py
def test_light_timing_extraction():
    """Verify light onset < light offset."""

def test_light_timing_synthetic_data():
    """Test with synthetic data light protocol."""

def test_get_top1_of_col_sorts_by_time():
    """Ensure function sorts by time, not by column value."""
```

#### 2.3 Test Cases for weighing_utils.py
```python
# tests/unit/test_weighing_utils.py
def test_normalize_to_unity_zero_max():
    """Handle arrays where max is zero."""

def test_normalize_to_unity_all_nan():
    """Handle arrays that are all NaN."""

def test_normalize_to_unity_valid_data():
    """Normal case with valid data."""
```

### Phase 3: Integration Tests (Week 3)

#### 3.1 Synthetic Pipeline CI Test
```yaml
# .github/workflows/synthetic-pipeline.yml
name: Synthetic Pipeline CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
      - name: Install dependencies
        run: uv sync
      - name: Run synthetic pipeline
        run: |
          PYTHONPATH=. python src/pipeline_PLR.py --config-name=synthetic_run
```

#### 3.2 Pipeline Stage Validation Tests
```python
# tests/integration/test_pipeline_stages.py
def test_outlier_detection_stage():
    """Run outlier detection with synthetic data."""

def test_imputation_stage():
    """Run imputation with synthetic data."""

def test_featurization_stage():
    """Run featurization with synthetic data."""

def test_classification_stage():
    """Run classification with synthetic data."""
```

### Phase 4: Runtime Assertions & Fail-Fast (Week 4)

#### 4.1 Add Data Validation Assertions
```python
# src/data_io/validation/data_validators.py
def validate_light_stimuli(df: pl.DataFrame) -> None:
    """Validate light stimuli have proper timing."""
    red_on = df.filter(pl.col("Red") > 0).select("time").min()
    red_off = df.filter(pl.col("Red") > 0).select("time").max()
    assert red_on < red_off, f"Red light onset ({red_on}) must be before offset ({red_off})"

def validate_features(features: dict) -> None:
    """Validate extracted features are complete."""
    required_keys = ["amplitude_bins", "latency_features"]
    for key in required_keys:
        assert key in features, f"Missing required feature: {key}"
```

#### 4.2 Early Validation in Pipeline
```python
# src/pipeline_PLR.py - Add at pipeline start
def validate_pipeline_inputs(cfg: DictConfig) -> None:
    """Fail fast if configuration is invalid."""
    # Check data exists
    data_path = Path(cfg.DATA.data_path) / cfg.DATA.filename_DuckDB
    if not data_path.exists():
        raise FileNotFoundError(f"Database not found: {data_path}")

    # Validate database schema
    conn = duckdb.connect(str(data_path), read_only=True)
    required_cols = ["time", "pupil_gt", "Red", "Blue", "outlier_mask"]
    for col in required_cols:
        if col not in conn.execute("DESCRIBE train").df()["column_name"].values:
            raise ValueError(f"Missing required column: {col}")
```

### Phase 5: Pre-commit Hooks (Week 5)

#### 5.1 Add New Pre-commit Hooks
```yaml
# .pre-commit-config.yaml additions
repos:
  - repo: local
    hooks:
      - id: check-parameter-names
        name: Check parameter name consistency
        entry: python scripts/check_parameter_names.py
        language: python
        types: [python]

      - id: check-null-guards
        name: Check for missing null guards in dict access
        entry: python scripts/check_null_guards.py
        language: python
        types: [python]

      - id: check-division-guards
        name: Check for unguarded divisions
        entry: python scripts/check_division_guards.py
        language: python
        types: [python]
```

#### 5.2 Custom AST Checks
```python
# scripts/check_null_guards.py
"""
Check that dict.get() is used instead of dict[] when value might be None.
"""
import ast
import sys

class NullGuardChecker(ast.NodeVisitor):
    def visit_Subscript(self, node):
        # Check for metrics["key"] patterns without guards
        ...
```

---

## Implementation Priority Matrix

| Issue Category | Priority | Effort | Impact | Week |
|---------------|----------|--------|--------|------|
| Type checking | High | Medium | High | 1 |
| Unit tests | Critical | High | Critical | 2 |
| Integration tests | High | High | High | 3 |
| Runtime assertions | Medium | Low | Medium | 4 |
| Pre-commit hooks | Medium | Medium | Medium | 5 |

---

## Review Process

### Iteration 1: Technical Review
- **Reviewers**: Core maintainers
- **Focus**: Code correctness, test coverage
- **Timeline**: Week 1-2

### Iteration 2: Integration Review
- **Reviewers**: Pipeline users, ML engineers
- **Focus**: End-to-end testing, edge cases
- **Timeline**: Week 3-4

### Iteration 3: Production Review
- **Reviewers**: DevOps, CI/CD maintainers
- **Focus**: CI integration, deployment readiness
- **Timeline**: Week 5

---

## Success Metrics

1. **Test Coverage**: ≥80% coverage on critical pipeline modules
2. **CI Pass Rate**: Synthetic pipeline CI passes on every PR
3. **Runtime Failures**: Zero crashes from issues that could be caught statically
4. **Time to Detection**: Issues caught within 1 minute of commit (pre-commit)

---

## Appendix: Specific Code Changes

### A1: Add Type Stubs for Pipeline Functions

```python
# src/ensemble/ensemble_utils.pyi
from typing import Optional
import pandas as pd
from omegaconf import DictConfig

def get_best_moment_variant(
    best_runs_out: pd.DataFrame,
    best_metric_cfg: DictConfig,
    return_best_gt: bool
) -> pd.DataFrame: ...

def get_best_moment(
    best_metric_cfg: DictConfig,
    runs_moment: pd.DataFrame
) -> Optional[pd.DataFrame]: ...
```

### A2: Example Unit Test

```python
# tests/unit/test_ensemble_utils_edge_cases.py
import pytest
import pandas as pd
from omegaconf import OmegaConf
from src.ensemble.ensemble_utils import get_best_moment_variant

class TestGetBestMomentVariant:
    @pytest.fixture
    def empty_dataframe(self):
        return pd.DataFrame()

    @pytest.fixture
    def non_moment_only_dataframe(self):
        return pd.DataFrame({
            "tags.mlflow.runName": ["LOF_run1", "SVM_run1"],
            "metrics.test/f1": [0.85, 0.82]
        })

    def test_handles_empty_input(self, empty_dataframe):
        """Should return empty DataFrame, not crash."""
        cfg = OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})
        result = get_best_moment_variant(empty_dataframe, cfg, True)
        assert result.empty

    def test_handles_no_moment_models(self, non_moment_only_dataframe):
        """Should return non-MOMENT models when no MOMENT exists."""
        cfg = OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})
        result = get_best_moment_variant(non_moment_only_dataframe, cfg, True)
        assert len(result) == 2
        assert "MOMENT" not in result["tags.mlflow.runName"].values
```

---

## Progress Tracking

### Completed

- [x] **Phase 1 Static Analysis & Type Checking** - 2026-02-01
  - Configured mypy in `pyproject.toml` with practical settings
  - Added type stubs: `types-PyYAML`, `pandas-stubs` to dev dependencies
  - Created `scripts/check_types.sh` for CI reporting
  - Added `make type-check` target
  - **Baseline established: 218 errors (63 in critical modules)**
  - Fixed type annotation in `weighing_utils.py`
  - Critical modules: `ensemble_utils.py`, `classifier_log_utils.py`, `weighing_utils.py`, `feature_utils.py`

- [x] **Phase 2 Unit Tests (Partial)** - 2026-02-01
  - Created `tests/unit/test_ensemble_utils.py` (16 tests) - MOMENT model handling
  - Created `tests/unit/test_feature_utils.py` (14 tests) - Light timing extraction
  - Created `tests/unit/test_weighing_utils.py` (17 tests) - Division by zero handling
  - Created `tests/unit/test_classifier_log_utils.py` (10 tests) - None guards
  - **Total: 57 new tests, all passing**

- [x] **Bug Fix: Empty DataFrame in get_best_moment_variant**
  - Tests caught an additional edge case
  - Fixed by adding early return for empty/None DataFrame

- [x] **Phase 3 Integration Tests** - 2026-02-01
  - Created `tests/integration/test_synthetic_pipeline.py` (14 tests)
  - Tests cover: data loading, outlier detection, featurization, classification, ensemble utils
  - All tests use synthetic data (`SYNTH_PLR_DEMO.db`) - no real patient data needed
  - Added `make test-integration` target

- [x] **Phase 4 Runtime Assertions** - 2026-02-02
  - Created `src/data_io/validation/data_validators.py`
  - Validators: light stimuli, signal range, time monotonicity, features, database schema
  - Created `tests/unit/test_data_validators.py` (19 tests)
  - All tests passing

### Next Steps

1. [ ] Integrate validators into pipeline startup
2. [ ] Add pre-commit hooks for code quality (Phase 5)
3. [ ] Reduce type error count incrementally (ongoing)
