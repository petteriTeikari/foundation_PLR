# TDD Test Cases Expansion - Display Name Lookup
**Reference**: `docs/planning/lookup-model-names.md`
**Status**: Blueprint for RED phase
**Purpose**: Executable test cases needed before implementation

---

## COMPLETE TEST SUITE (25 Test Cases)

### Unit Test File: `tests/unit/test_display_names.py`

```python
"""
Unit tests for display names lookup table.

TDD Cycle:
1. RED: All tests below should FAIL (no implementation exists)
2. GREEN: Implement display_names.py to make tests PASS
3. REFACTOR: Optimize and clean up while tests stay GREEN

Grouping:
- File and schema tests (4)
- Getter tests (5)
- Validation tests (6)
- Caching tests (2)
- Error handling tests (4)
- Quality assurance tests (4)
"""

import pytest
from pathlib import Path
import tempfile
import yaml
import pandas as pd

# These will fail on import until display_names.py is created
# (That's expected during RED phase)
try:
    from src.data_io.display_names import (
        get_outlier_display_name,
        get_imputation_display_name,
        get_classifier_display_name,
        get_category_display_name,
        get_all_display_names,
        _load_display_names,
        DISPLAY_NAMES_PATH,
    )
    from src.data_io.registry import (
        get_valid_outlier_methods,
        get_valid_imputation_methods,
        get_valid_classifiers,
    )
except ImportError as e:
    pytest.skip(f"Module not ready: {e}", allow_module_level=True)


class TestDisplayNamesFile:
    """Tests for YAML file existence and structure. (P1 - RED Phase)"""

    def test_display_names_yaml_exists(self):
        """Display names YAML file must exist at canonical path."""
        assert DISPLAY_NAMES_PATH.exists(), \
            f"display_names.yaml not found at {DISPLAY_NAMES_PATH}"

    def test_display_names_yaml_is_readable(self):
        """YAML file must be readable and parseable."""
        with open(DISPLAY_NAMES_PATH) as f:
            names = yaml.safe_load(f)
        assert names is not None, "YAML file is empty or malformed"

    def test_display_names_schema_has_required_sections(self):
        """YAML must have outlier_methods, imputation_methods, classifiers."""
        names = _load_display_names()

        required_sections = [
            'outlier_methods',
            'imputation_methods',
            'classifiers',
        ]
        for section in required_sections:
            assert section in names, \
                f"Missing required section: {section}"
            assert isinstance(names[section], dict), \
                f"Section '{section}' is not a dict"

    def test_display_names_schema_has_categories(self):
        """YAML should have categories for grouping methods."""
        names = _load_display_names()

        assert 'categories' in names, "Missing 'categories' section"
        assert 'outlier' in names['categories'], \
            "Missing outlier category definitions"
        assert 'imputation' in names['categories'], \
            "Missing imputation category definitions"


class TestDisplayNamesGetters:
    """Tests for getter functions. (P1 - RED Phase)"""

    def test_get_outlier_display_name_valid(self):
        """Should return display name for valid outlier methods."""
        result = get_outlier_display_name("pupil-gt")
        assert result == "Ground Truth"

    def test_get_imputation_display_name_valid(self):
        """Should return display name for valid imputation methods."""
        result = get_imputation_display_name("pupil-gt")
        assert result == "Ground Truth"

    def test_get_classifier_display_name_valid(self):
        """Should return display name for valid classifiers."""
        result = get_classifier_display_name("CatBoost")
        assert result == "CatBoost"

    def test_get_all_display_names_returns_dict(self):
        """get_all_display_names() should return all mappings."""
        all_names = get_all_display_names()

        assert isinstance(all_names, dict)
        assert len(all_names) > 0
        # Should have entries from all categories
        assert "pupil-gt" in all_names
        assert "LOF" in all_names
        assert "CatBoost" in all_names

    def test_display_name_different_from_raw(self):
        """Display names should be user-friendly, not just raw values."""
        for method in ["pupil-gt", "MOMENT-gt-finetune", "OneClassSVM"]:
            display = get_outlier_display_name(method)
            # Not identical (unless method happens to already be user-friendly)
            # At minimum, pupil-gt → Ground Truth is different
            if method == "pupil-gt":
                assert display != method, \
                    f"Display name same as raw for: {method}"


class TestDisplayNamesValidation:
    """Tests for data quality and consistency. (P2)"""

    def test_no_missing_gt_abbreviation_in_display_names(self):
        """Display names should not contain '-gt' abbreviation."""
        all_names = get_all_display_names()

        for raw, display in all_names.items():
            assert "-gt" not in display.lower(), \
                f"'-gt' found in display name: '{raw}' → '{display}'"

    def test_no_missing_orig_abbreviation_in_display_names(self):
        """Display names should not contain '-orig' abbreviation."""
        all_names = get_all_display_names()

        for raw, display in all_names.items():
            assert "-orig" not in display.lower(), \
                f"'-orig' found in display name: '{raw}' → '{display}'"

    def test_no_duplicate_display_names(self):
        """No two methods should map to identical display names.

        Duplicate display names are ambiguous - can't tell methods apart in plots.
        """
        all_names = get_all_display_names()
        display_names = list(all_names.values())

        # Check for duplicates
        unique_count = len(set(display_names))
        total_count = len(display_names)

        assert unique_count == total_count, \
            f"Found {total_count - unique_count} duplicate display names"

    def test_display_names_start_with_capital(self):
        """All display names should start with capital letter."""
        all_names = get_all_display_names()

        for raw, display in all_names.items():
            assert display[0].isupper(), \
                f"'{display}' doesn't start with capital (from '{raw}')"

    def test_display_names_no_trailing_spaces(self):
        """Display names should not have leading/trailing whitespace."""
        all_names = get_all_display_names()

        for raw, display in all_names.items():
            assert display == display.strip(), \
                f"'{display}' has leading/trailing spaces (from '{raw}')"

    def test_display_names_no_double_hyphens(self):
        """Display names should not have double hyphens or malformed punctuation."""
        all_names = get_all_display_names()

        for raw, display in all_names.items():
            assert "--" not in display, \
                f"Double hyphen in '{display}' (from '{raw}')"
            assert not display.startswith('-'), \
                f"Leading hyphen in '{display}' (from '{raw}')"
            assert not display.endswith('-'), \
                f"Trailing hyphen in '{display}' (from '{raw}')"


class TestRegistryCrossPollination:
    """Tests that display_names and registry.py are in sync. (P1 CRITICAL)"""

    def test_all_outlier_methods_have_display_names(self):
        """Every outlier method in registry.yaml must have a display name.

        This catches when registry is updated but display_names.yaml is not.
        CRITICAL: Registry drift would cause visualization failures.
        """
        for method in get_valid_outlier_methods():
            display = get_outlier_display_name(method)
            assert display is not None, \
                f"Outlier method '{method}' missing from display_names.yaml"
            assert isinstance(display, str), \
                f"Display name for '{method}' is not a string: {display}"
            assert len(display) > 0, \
                f"Display name for '{method}' is empty string"

    def test_all_imputation_methods_have_display_names(self):
        """Every imputation method in registry.yaml must have a display name."""
        for method in get_valid_imputation_methods():
            display = get_imputation_display_name(method)
            assert display is not None, \
                f"Imputation method '{method}' missing from display_names.yaml"
            assert isinstance(display, str), \
                f"Display name for '{method}' is not a string"

    def test_all_classifiers_have_display_names(self):
        """Every classifier in registry.yaml must have a display name."""
        for clf in get_valid_classifiers():
            display = get_classifier_display_name(clf)
            assert display is not None, \
                f"Classifier '{clf}' missing from display_names.yaml"


class TestCaching:
    """Tests for @lru_cache performance optimization. (P1)"""

    def test_display_names_loaded_once(self):
        """_load_display_names() should cache results."""
        from src.data_io import display_names

        # Clear cache
        display_names._load_display_names.cache_clear()
        cache_info_before = display_names._load_display_names.cache_info()

        # First call - cache miss
        result1 = display_names._load_display_names()

        # Second call - cache hit
        result2 = display_names._load_display_names()

        # Check cache statistics
        cache_info = display_names._load_display_names.cache_info()

        assert cache_info.misses == 1, "First call should be cache miss"
        assert cache_info.hits >= 1, "Second call should be cache hit"
        assert result1 is result2, "Should return same cached object"

    def test_display_names_cache_survives_multiple_accessors(self):
        """All getter functions should use same cached YAML."""
        from src.data_io import display_names

        display_names._load_display_names.cache_clear()

        # Call through different getters
        get_outlier_display_name("pupil-gt")
        get_imputation_display_name("pupil-gt")
        get_classifier_display_name("CatBoost")

        # Should be 3 cache hits from shared source
        cache_info = display_names._load_display_names.cache_info()
        assert cache_info.hits >= 2, "Getters should share cache"


class TestErrorHandling:
    """Tests for error conditions and edge cases. (P1 CRITICAL)"""

    def test_missing_outlier_method_returns_none(self):
        """Missing methods should return None (graceful degradation).

        DECISION: Return None instead of raising, so caller can use raw name as fallback.
        This is important for visualization code that needs to display something.
        """
        result = get_outlier_display_name("nonexistent-method-xyz")
        assert result is None

    def test_missing_imputation_method_returns_none(self):
        """Missing imputation methods should return None."""
        result = get_imputation_display_name("nonexistent-method-xyz")
        assert result is None

    def test_missing_classifier_returns_none(self):
        """Missing classifiers should return None."""
        result = get_classifier_display_name("NonexistentClassifier")
        assert result is None

    def test_yaml_parse_error_raises_helpful_message(self):
        """If YAML is malformed, should raise helpful error."""
        from src.data_io import display_names
        import tempfile

        # Create malformed YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("{ invalid: yaml: structure [")
            temp_path = f.name

        try:
            original_path = display_names.DISPLAY_NAMES_PATH
            display_names.DISPLAY_NAMES_PATH = Path(temp_path)
            display_names._load_display_names.cache_clear()

            with pytest.raises(yaml.YAMLError):
                display_names._load_display_names()

            display_names.DISPLAY_NAMES_PATH = original_path
        finally:
            Path(temp_path).unlink()


class TestQualityAssurance:
    """Tests for publication-quality standards. (P2)"""

    def test_foundation_model_hyphenation_consistent(self):
        """Foundation models should use consistent hyphenation: Fine-tuned, Zero-shot."""
        assert "Fine-tuned" in get_outlier_display_name("MOMENT-gt-finetune")
        assert "Zero-shot" in get_outlier_display_name("MOMENT-gt-zeroshot")
        assert "Fine-tuned" in get_imputation_display_name("MOMENT-finetune")
        assert "Zero-shot" in get_imputation_display_name("MOMENT-zeroshot")

    def test_ensemble_names_are_distinct(self):
        """Different ensemble types should have clearly different display names."""
        full = get_outlier_display_name(
            "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"
        )
        thresholded = get_outlier_display_name(
            "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"
        )

        assert full != thresholded, "Ensemble names should be distinct"
        # Names should indicate difference
        assert "Full" in full or "All" in full or len(full) > len(thresholded)

    def test_acronyms_properly_capitalized(self):
        """Acronyms like LOF, SVM should be uppercase."""
        assert get_outlier_display_name("LOF") == "LOF"
        assert "SVM" in get_outlier_display_name("OneClassSVM")

    def test_category_display_names_consistent(self):
        """Category names should match those in registry."""
        outlier_cats = get_category_display_name("outlier", "foundation_model")
        assert outlier_cats is not None
        assert "Foundation" in outlier_cats or "foundation" in outlier_cats.lower()


class TestIntegrationWithRegistry:
    """Integration tests for cross-module functionality. (P1)"""

    def test_all_registry_methods_account_for_counts(self):
        """Number of display names should match registry counts."""
        from src.data_io.registry import (
            EXPECTED_OUTLIER_COUNT,
            EXPECTED_IMPUTATION_COUNT,
            EXPECTED_CLASSIFIER_COUNT,
        )

        names = _load_display_names()

        outlier_count = len(names.get("outlier_methods", {}))
        imputation_count = len(names.get("imputation_methods", {}))
        classifier_count = len(names.get("classifiers", {}))

        assert outlier_count == EXPECTED_OUTLIER_COUNT, \
            f"Outlier count mismatch: {outlier_count} vs {EXPECTED_OUTLIER_COUNT}"
        assert imputation_count == EXPECTED_IMPUTATION_COUNT, \
            f"Imputation count mismatch: {imputation_count} vs {EXPECTED_IMPUTATION_COUNT}"
        assert classifier_count == EXPECTED_CLASSIFIER_COUNT, \
            f"Classifier count mismatch: {classifier_count} vs {EXPECTED_CLASSIFIER_COUNT}"
```

---

### Integration Test File: `tests/integration/test_display_names_extraction.py`

```python
"""
Integration tests for display names in data extraction pipeline.

These tests verify that display names are correctly applied when
exporting data to DuckDB, CSV, and JSON formats.
"""

import pytest
import tempfile
from pathlib import Path
import duckdb
import pandas as pd
import json


class TestDisplayNamesInExtraction:
    """Tests that display names are included in extracted data. (P1)"""

    @pytest.fixture
    def sample_results_df(self):
        """Create sample MLflow results for testing."""
        return pd.DataFrame({
            'outlier_method': ['pupil-gt', 'LOF', 'MOMENT-gt-finetune'],
            'imputation_method': ['pupil-gt', 'SAITS', 'SAITS'],
            'classifier': ['CatBoost', 'CatBoost', 'CatBoost'],
            'auroc': [0.911, 0.850, 0.905],
            'auroc_ci_lo': [0.880, 0.810, 0.870],
            'auroc_ci_hi': [0.940, 0.890, 0.940],
        })

    def test_duckdb_export_includes_display_names(self, sample_results_df):
        """DuckDB export should include _display_name columns."""
        # This test assumes an export function exists
        # from src.data_io.extraction import export_results_to_duckdb

        # Mock for now - implement after display_names module exists
        from src.data_io.display_names import (
            get_outlier_display_name,
            get_imputation_display_name,
            get_classifier_display_name,
        )

        # Add display names to dataframe
        sample_results_df['outlier_display_name'] = \
            sample_results_df['outlier_method'].apply(get_outlier_display_name)
        sample_results_df['imputation_display_name'] = \
            sample_results_df['imputation_method'].apply(get_imputation_display_name)
        sample_results_df['classifier_display_name'] = \
            sample_results_df['classifier'].apply(get_classifier_display_name)

        # Verify columns exist
        assert 'outlier_display_name' in sample_results_df.columns
        assert 'imputation_display_name' in sample_results_df.columns
        assert 'classifier_display_name' in sample_results_df.columns

        # Verify values
        assert sample_results_df['outlier_display_name'].iloc[0] == 'Ground Truth'
        assert sample_results_df['imputation_display_name'].iloc[1] == 'SAITS'

    def test_csv_export_includes_display_names(self, sample_results_df):
        """CSV export should include display_name columns."""
        from src.data_io.display_names import (
            get_outlier_display_name,
            get_imputation_display_name,
        )

        # Prepare dataframe with display names
        sample_results_df['outlier_display_name'] = \
            sample_results_df['outlier_method'].apply(get_outlier_display_name)
        sample_results_df['imputation_display_name'] = \
            sample_results_df['imputation_method'].apply(get_imputation_display_name)

        # Write and read back
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            sample_results_df.to_csv(csv_path, index=False)

            # Verify on read
            read_df = pd.read_csv(csv_path)
            assert 'outlier_display_name' in read_df.columns
            assert read_df['outlier_display_name'].iloc[0] == 'Ground Truth'

    def test_json_export_includes_display_names(self, sample_results_df):
        """JSON export should include display_name fields."""
        from src.data_io.display_names import (
            get_outlier_display_name,
            get_imputation_display_name,
        )

        # Prepare dataframe
        sample_results_df['outlier_display_name'] = \
            sample_results_df['outlier_method'].apply(get_outlier_display_name)
        sample_results_df['imputation_display_name'] = \
            sample_results_df['imputation_method'].apply(get_imputation_display_name)

        # Convert to JSON
        json_data = sample_results_df.to_dict(orient='records')

        # Verify structure
        assert len(json_data) > 0
        assert 'outlier_display_name' in json_data[0]
        assert json_data[0]['outlier_display_name'] == 'Ground Truth'
```

---

## TEST RUNNING COMMANDS

### RED Phase (Expected: All Tests FAIL)

```bash
# Run all display name tests
pytest tests/unit/test_display_names.py -v

# Expected output:
# ERROR: cannot import name 'get_outlier_display_name' (module doesn't exist yet)

# Once we add fixture skips:
# tests/unit/test_display_names.py::TestDisplayNamesFile::test_display_names_yaml_exists FAILED
# tests/unit/test_display_names.py::TestDisplayNamesGetters::test_get_outlier_display_name_valid FAILED
# ... (25 failing tests)
```

### GREEN Phase (After implementing display_names.py)

```bash
# All tests should PASS
pytest tests/unit/test_display_names.py -v

# Expected output:
# tests/unit/test_display_names.py::TestDisplayNamesFile::test_display_names_yaml_exists PASSED
# ... (25 passing tests)

# Run with coverage
pytest tests/unit/test_display_names.py --cov=src.data_io.display_names --cov-report=html
```

---

## SUMMARY TABLE

| Test Group | Count | Status | Priority |
|------------|-------|--------|----------|
| File and Schema | 4 | To Write | P1 |
| Getters | 5 | To Write | P1 |
| Validation | 6 | To Write | P2 |
| Cross-Registry Sync | 3 | To Write | P1 CRITICAL |
| Caching | 2 | To Write | P1 |
| Error Handling | 4 | To Write | P1 |
| Quality Assurance | 4 | To Write | P2 |
| **Integration Tests** | 3 | To Write | P1 |
| **TOTAL** | **31** | **ALL MISSING** | **Mixed** |

---

## IMPLEMENTATION CHECKLIST

Use this checklist after tests are written:

```bash
# [ ] Write tests/unit/test_display_names.py (31 test cases)
# [ ] Write tests/integration/test_display_names_extraction.py (3 integration tests)
# [ ] Run tests - verify ALL FAIL (RED phase)
pytest tests/unit/test_display_names.py -v

# [ ] Create configs/mlflow_registry/display_names.yaml
# [ ] Create src/data_io/display_names.py
# [ ] Run tests - verify ALL PASS (GREEN phase)
pytest tests/unit/test_display_names.py -v

# [ ] Create src/r/load_display_names.R
# [ ] Create tests/integration/test_display_names_extraction.py
# [ ] Update extraction pipeline to use display names
# [ ] Run full test suite
pytest tests/unit/test_display_names.py tests/integration/test_display_names_extraction.py -v

# [ ] Refactor and optimize
# [ ] Verify all tests still pass
pytest tests/ -v

# [ ] Commit with message:
#     "feat: Add display names lookup (TDD: RED→GREEN→REFACTOR)"
```

