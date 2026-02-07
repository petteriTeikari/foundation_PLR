"""
Unit tests for display names lookup table.

TDD Cycle 1: RED - All tests should FAIL initially until implementation is complete.

These tests validate:
1. YAML file structure and existence
2. Coverage: all registry methods have display names
3. Naming conventions: no -gt, proper hyphenation
4. Python module API
5. Error handling: fallback behavior for unknown methods
"""

import pytest
from pathlib import Path


# =============================================================================
# Test Fixtures
# =============================================================================

DISPLAY_NAMES_PATH = (
    Path(__file__).parents[2] / "configs" / "mlflow_registry" / "display_names.yaml"
)


@pytest.fixture
def display_names_yaml():
    """Load display names YAML file."""
    import yaml

    assert (
        DISPLAY_NAMES_PATH.exists()
    ), f"display_names.yaml missing: {DISPLAY_NAMES_PATH}"

    with open(DISPLAY_NAMES_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def registry_outlier_methods():
    """Get outlier methods from registry."""
    from src.data_io.registry import get_valid_outlier_methods

    return get_valid_outlier_methods()


@pytest.fixture
def registry_imputation_methods():
    """Get imputation methods from registry."""
    from src.data_io.registry import get_valid_imputation_methods

    return get_valid_imputation_methods()


@pytest.fixture
def registry_classifiers():
    """Get classifiers from registry."""
    from src.data_io.registry import get_valid_classifiers

    return get_valid_classifiers()


# =============================================================================
# YAML Structure Tests (1-6)
# =============================================================================


class TestYAMLStructure:
    """Test YAML file structure and validity."""

    def test_display_names_yaml_exists(self):
        """Test 1: Display names YAML file must exist."""
        assert DISPLAY_NAMES_PATH.exists(), (
            f"display_names.yaml not found at {DISPLAY_NAMES_PATH}. "
            "Create this file to define publication-friendly display names."
        )

    def test_yaml_is_valid(self, display_names_yaml):
        """Test 2: YAML parses without error."""
        assert display_names_yaml is not None, "YAML file is empty or invalid"
        assert isinstance(display_names_yaml, dict), "YAML root must be a dict"

    def test_has_version(self, display_names_yaml):
        """Test 3: Version field exists."""
        assert "version" in display_names_yaml, "Missing 'version' field"
        assert display_names_yaml["version"], "Version cannot be empty"

    def test_has_outlier_methods_section(self, display_names_yaml):
        """Test 4: outlier_methods section exists."""
        assert (
            "outlier_methods" in display_names_yaml
        ), "Missing 'outlier_methods' section"
        assert isinstance(
            display_names_yaml["outlier_methods"], dict
        ), "outlier_methods must be a dict mapping raw names to display names"

    def test_has_imputation_methods_section(self, display_names_yaml):
        """Test 5: imputation_methods section exists."""
        assert (
            "imputation_methods" in display_names_yaml
        ), "Missing 'imputation_methods' section"
        assert isinstance(
            display_names_yaml["imputation_methods"], dict
        ), "imputation_methods must be a dict mapping raw names to display names"

    def test_has_classifiers_section(self, display_names_yaml):
        """Test 6: classifiers section exists."""
        assert "classifiers" in display_names_yaml, "Missing 'classifiers' section"
        assert isinstance(
            display_names_yaml["classifiers"], dict
        ), "classifiers must be a dict mapping raw names to display names"


# =============================================================================
# Coverage Tests (7-11)
# =============================================================================


class TestRegistryCoverage:
    """Test that all registry methods have display names."""

    def test_all_outlier_methods_covered(
        self, display_names_yaml, registry_outlier_methods
    ):
        """Test 7: All 11 registry outlier methods have display names."""
        yaml_outliers = set(display_names_yaml["outlier_methods"].keys())
        registry_outliers = set(registry_outlier_methods)

        missing = registry_outliers - yaml_outliers
        assert not missing, (
            f"Missing display names for outlier methods: {missing}. "
            f"Expected {len(registry_outliers)} methods, YAML has {len(yaml_outliers)}."
        )

    def test_all_imputation_methods_covered(
        self, display_names_yaml, registry_imputation_methods
    ):
        """Test 8: All 8 registry imputation methods have display names."""
        yaml_imputations = set(display_names_yaml["imputation_methods"].keys())
        registry_imputations = set(registry_imputation_methods)

        missing = registry_imputations - yaml_imputations
        assert not missing, (
            f"Missing display names for imputation methods: {missing}. "
            f"Expected {len(registry_imputations)} methods, YAML has {len(yaml_imputations)}."
        )

    def test_all_classifiers_covered(self, display_names_yaml, registry_classifiers):
        """Test 9: All 5 registry classifiers have display names."""
        yaml_classifiers = set(display_names_yaml["classifiers"].keys())
        registry_clfs = set(registry_classifiers)

        missing = registry_clfs - yaml_classifiers
        assert not missing, (
            f"Missing display names for classifiers: {missing}. "
            f"Expected {len(registry_clfs)} classifiers, YAML has {len(yaml_classifiers)}."
        )

    def test_no_extra_outlier_methods(
        self, display_names_yaml, registry_outlier_methods
    ):
        """Test 10: No display names for non-existent outlier methods."""
        yaml_outliers = set(display_names_yaml["outlier_methods"].keys())
        registry_outliers = set(registry_outlier_methods)

        extra = yaml_outliers - registry_outliers
        assert not extra, (
            f"Extra outlier methods in display_names.yaml not in registry: {extra}. "
            "Remove these or add them to the registry first."
        )

    def test_no_extra_imputation_methods(
        self, display_names_yaml, registry_imputation_methods
    ):
        """Test 11: No display names for non-existent imputation methods."""
        yaml_imputations = set(display_names_yaml["imputation_methods"].keys())
        registry_imputations = set(registry_imputation_methods)

        extra = yaml_imputations - registry_imputations
        assert not extra, (
            f"Extra imputation methods in display_names.yaml not in registry: {extra}. "
            "Remove these or add them to the registry first."
        )


# =============================================================================
# Naming Convention Tests (12-21)
# =============================================================================


class TestNamingConventions:
    """Test display name formatting rules."""

    def test_no_gt_in_display_names(self, display_names_yaml):
        """Test 12: No display name contains '-gt' abbreviation."""
        all_display_names = []
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["outlier_methods"].values()
        )
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["imputation_methods"].values()
        )
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["classifiers"].values()
        )

        violations = [name for name in all_display_names if "-gt" in name.lower()]
        assert not violations, (
            f"Display names containing '-gt': {violations}. "
            "'-gt' is an internal abbreviation and should not appear in publication names."
        )

    def test_no_orig_in_display_names(self, display_names_yaml):
        """Test 13: No display name contains '-orig' abbreviation."""
        all_display_names = []
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["outlier_methods"].values()
        )
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["imputation_methods"].values()
        )
        all_display_names.extend(
            v["display_name"] for v in display_names_yaml["classifiers"].values()
        )

        violations = [name for name in all_display_names if "-orig" in name.lower()]
        assert not violations, (
            f"Display names containing '-orig': {violations}. "
            "'-orig' is an internal abbreviation and should not appear in publication names."
        )

    def test_ground_truth_display(self, display_names_yaml):
        """Test 14: pupil-gt displays as 'Ground Truth'."""
        assert (
            display_names_yaml["outlier_methods"]["pupil-gt"]["display_name"]
            == "Ground Truth"
        )
        assert (
            display_names_yaml["imputation_methods"]["pupil-gt"]["display_name"]
            == "Ground Truth"
        )

    def test_moment_finetuned_display(self, display_names_yaml):
        """Test 15: MOMENT finetuned contains 'Fine-tuned' (hyphenated)."""
        outlier_moment = display_names_yaml["outlier_methods"]["MOMENT-gt-finetune"][
            "display_name"
        ]
        imputation_moment = display_names_yaml["imputation_methods"]["MOMENT-finetune"][
            "display_name"
        ]

        assert (
            "Fine-tuned" in outlier_moment
        ), f"Expected 'Fine-tuned' (hyphenated) in '{outlier_moment}'"
        assert (
            "Fine-tuned" in imputation_moment
        ), f"Expected 'Fine-tuned' (hyphenated) in '{imputation_moment}'"

    def test_moment_zeroshot_display(self, display_names_yaml):
        """Test 16: MOMENT zeroshot contains 'Zero-shot' (hyphenated)."""
        outlier_moment = display_names_yaml["outlier_methods"]["MOMENT-gt-zeroshot"][
            "display_name"
        ]
        imputation_moment = display_names_yaml["imputation_methods"]["MOMENT-zeroshot"][
            "display_name"
        ]

        assert (
            "Zero-shot" in outlier_moment
        ), f"Expected 'Zero-shot' (hyphenated) in '{outlier_moment}'"
        assert (
            "Zero-shot" in imputation_moment
        ), f"Expected 'Zero-shot' (hyphenated) in '{imputation_moment}'"

    def test_prophet_capitalization(self, display_names_yaml):
        """Test 17: PROPHET displays as 'Prophet' (proper noun)."""
        prophet_display = display_names_yaml["outlier_methods"]["PROPHET"][
            "display_name"
        ]
        assert (
            prophet_display == "Prophet"
        ), f"Expected 'Prophet' (proper noun capitalization), got '{prophet_display}'"

    def test_lof_uppercase(self, display_names_yaml):
        """Test 18: LOF remains uppercase (acronym)."""
        lof_display = display_names_yaml["outlier_methods"]["LOF"]["display_name"]
        assert lof_display == "LOF", f"Expected 'LOF' (acronym), got '{lof_display}'"

    def test_svm_expansion(self, display_names_yaml):
        """Test 19: OneClassSVM displays as 'One-Class SVM'."""
        svm_display = display_names_yaml["outlier_methods"]["OneClassSVM"][
            "display_name"
        ]
        assert (
            svm_display == "One-Class SVM"
        ), f"Expected 'One-Class SVM', got '{svm_display}'"

    def test_ensemble_full_name(self, display_names_yaml):
        """Test 20: Full ensemble displays as 'Ensemble (Full)'."""
        full_ensemble_key = (
            "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune"
        )
        full_display = display_names_yaml["outlier_methods"][full_ensemble_key][
            "display_name"
        ]
        assert (
            full_display == "Ensemble (Full)"
        ), f"Expected 'Ensemble (Full)', got '{full_display}'"

    def test_ensemble_thresholded_name(self, display_names_yaml):
        """Test 21: Thresholded ensemble displays as 'Ensemble (Thresholded)'."""
        thresholded_key = "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"
        thresholded_display = display_names_yaml["outlier_methods"][thresholded_key][
            "display_name"
        ]
        assert (
            thresholded_display == "Ensemble (Thresholded)"
        ), f"Expected 'Ensemble (Thresholded)', got '{thresholded_display}'"


# =============================================================================
# Python Module Tests (22-24)
# =============================================================================


class TestPythonModule:
    """Test Python display_names module API."""

    def test_python_module_loads(self):
        """Test 22: Python module imports successfully."""
        try:
            from src.data_io.display_names import (
                get_outlier_display_name,
                get_imputation_display_name,
                get_classifier_display_name,
            )
        except ImportError as e:
            pytest.fail(f"Failed to import display_names module: {e}")

        # Verify functions are callable
        assert callable(get_outlier_display_name)
        assert callable(get_imputation_display_name)
        assert callable(get_classifier_display_name)

    def test_fallback_returns_raw_name(self):
        """Test 23: Unknown method returns raw name (no crash)."""
        from src.data_io.display_names import get_outlier_display_name

        unknown_method = "NONEXISTENT-METHOD-xyz123"

        result = get_outlier_display_name(unknown_method)

        # Should return the raw name as fallback (never crash)
        assert (
            result == unknown_method
        ), f"Expected fallback to return raw name '{unknown_method}', got '{result}'"
        # Note: Also logs WARNING but loguru's output isn't captured by pytest.
        # The warning can be seen in "Captured stderr call" during test output.

    def test_get_all_display_names_returns_dict(self):
        """Test 24: get_all_display_names returns combined dict."""
        from src.data_io.display_names import get_all_display_names

        all_names = get_all_display_names()

        assert isinstance(all_names, dict), "Expected dict"

        # Should have at least 23 entries (11 outlier + 8 imputation + 5 classifiers - 1 shared)
        # Note: pupil-gt appears in both outlier and imputation, so dict has 23 unique keys
        assert (
            len(all_names) >= 23
        ), f"Expected at least 23 display names, got {len(all_names)}"

        # Verify some known entries
        assert "pupil-gt" in all_names, "Missing 'pupil-gt' in all display names"
        assert "CatBoost" in all_names, "Missing 'CatBoost' in all display names"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration with data pipeline."""

    def test_csv_export_has_display_name_column(self):
        """CSV export should have display_name columns after integration."""
        csv_path = (
            Path(__file__).parents[2] / "data" / "r_data" / "essential_metrics.csv"
        )
        assert (
            csv_path.exists()
        ), f"essential_metrics.csv missing: {csv_path}. Run: make analyze"

        import pandas as pd

        df = pd.read_csv(csv_path)

        # These columns should exist after display names integration
        expected_columns = [
            "outlier_display_name",
            "imputation_display_name",
            "classifier_display_name",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Column '{col}' missing from {csv_path}"
