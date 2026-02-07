"""
Registry Validation Tests - Quality Gate for Experiment Parameters.

These tests enforce that the MLflow registry is the SINGLE SOURCE OF TRUTH.
They use HARDCODED expected values because no more experiments will be run.

WARNING: If you need to expand the experimental design, you MUST update:
  1. configs/mlflow_registry/parameters/classification.yaml
  2. src/data_io/registry.py (EXPECTED_*_COUNT constants)
  3. This test file (all hardcoded values below)
  4. .claude/rules/05-registry-source-of-truth.md
  5. CLAUDE.md (both root and .claude/)

These tests are designed to FAIL LOUDLY if anyone changes the registry
without updating all dependent files.
"""

import pytest

from src.data_io.registry import (
    get_valid_outlier_methods,
    get_valid_imputation_methods,
    get_valid_classifiers,
    get_outlier_categories,
    get_imputation_categories,
    validate_outlier_method,
    validate_imputation_method,
    validate_classifier,
    get_expected_factorial_size,
    EXPECTED_OUTLIER_COUNT,
    EXPECTED_IMPUTATION_COUNT,
    EXPECTED_CLASSIFIER_COUNT,
    REGISTRY_PATH,
    CLASSIFICATION_PARAMS,
)

pytestmark = pytest.mark.guardrail


# ============================================================================
# HARDCODED EXPECTED VALUES - FINAL FOR PUBLICATION
# ============================================================================
# These are the EXACT method names that MUST be in the registry.
# If you see a test failure, check if someone modified the registry YAML
# without updating this test file.

EXPECTED_OUTLIER_METHODS = [
    "pupil-gt",  # Ground truth
    "MOMENT-gt-finetune",
    "MOMENT-gt-zeroshot",
    "UniTS-gt-finetune",
    "TimesNet-gt",
    "LOF",
    "OneClassSVM",
    "PROPHET",
    "SubPCA",
    "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
    "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune",
]

EXPECTED_IMPUTATION_METHODS = [
    "pupil-gt",  # Ground truth
    "MOMENT-finetune",
    "MOMENT-zeroshot",
    "CSDI",
    "SAITS",
    "TimesNet",
    "ensemble-CSDI-MOMENT-SAITS",
    "ensemble-CSDI-MOMENT-SAITS-TimesNet",
]

EXPECTED_CLASSIFIERS = [
    "CatBoost",
    "XGBoost",
    "TabPFN",
    "TabM",
    "LogisticRegression",
]

# INVALID methods - these MUST NOT be in the registry
INVALID_OUTLIER_METHODS = [
    "anomaly",  # Garbage placeholder from test runs
    "exclude",  # Garbage placeholder from test runs
    "MOMENT-orig-finetune",  # Not in registry (original variant)
    "UniTS-orig-finetune",  # Not in registry
    "UniTS-orig-zeroshot",  # Not in registry
    "TimesNet-orig",  # Not in registry
]


class TestRegistryExists:
    """Test that registry files exist and are valid YAML."""

    def test_registry_path_exists(self):
        """Registry directory must exist."""
        assert REGISTRY_PATH.exists(), f"Registry path not found: {REGISTRY_PATH}"

    def test_classification_params_exists(self):
        """Classification parameters file must exist."""
        assert (
            CLASSIFICATION_PARAMS.exists()
        ), f"Classification params not found: {CLASSIFICATION_PARAMS}"

    def test_classification_params_valid_yaml(self):
        """Classification parameters must be valid YAML."""
        import yaml

        with open(CLASSIFICATION_PARAMS) as f:
            params = yaml.safe_load(f)
        assert params is not None
        assert "pipeline" in params


class TestOutlierMethodCounts:
    """Test that outlier method counts match expected values."""

    def test_outlier_count_matches_expected(self):
        """Registry must return exactly 11 outlier methods."""
        methods = get_valid_outlier_methods()
        assert len(methods) == 11, (
            f"Expected 11 outlier methods, got {len(methods)}. "
            f"Registry might be corrupted or someone added/removed methods. "
            f"Methods found: {methods}"
        )

    def test_outlier_count_matches_constant(self):
        """Registry constant must match returned count."""
        methods = get_valid_outlier_methods()
        assert len(methods) == EXPECTED_OUTLIER_COUNT, (
            f"Registry module constant ({EXPECTED_OUTLIER_COUNT}) doesn't match "
            f"actual count ({len(methods)}). Update EXPECTED_OUTLIER_COUNT."
        )


class TestImputationMethodCounts:
    """Test that imputation method counts match expected values."""

    def test_imputation_count_matches_expected(self):
        """Registry must return exactly 8 imputation methods."""
        methods = get_valid_imputation_methods()
        assert len(methods) == 8, (
            f"Expected 8 imputation methods, got {len(methods)}. "
            f"Methods found: {methods}"
        )

    def test_imputation_count_matches_constant(self):
        """Registry constant must match returned count."""
        methods = get_valid_imputation_methods()
        assert len(methods) == EXPECTED_IMPUTATION_COUNT


class TestClassifierCounts:
    """Test that classifier counts match expected values."""

    def test_classifier_count_matches_expected(self):
        """Registry must return exactly 5 classifiers."""
        classifiers = get_valid_classifiers()
        assert len(classifiers) == 5, (
            f"Expected 5 classifiers, got {len(classifiers)}. "
            f"Classifiers found: {classifiers}"
        )

    def test_classifier_count_matches_constant(self):
        """Registry constant must match returned count."""
        classifiers = get_valid_classifiers()
        assert len(classifiers) == EXPECTED_CLASSIFIER_COUNT


class TestOutlierMethodContents:
    """Test that specific outlier methods exist in registry."""

    @pytest.mark.parametrize("method", EXPECTED_OUTLIER_METHODS)
    def test_expected_outlier_method_exists(self, method):
        """Each expected outlier method must be in the registry."""
        methods = get_valid_outlier_methods()
        assert method in methods, (
            f"Expected outlier method '{method}' not found in registry. "
            f"Available methods: {methods}"
        )

    @pytest.mark.parametrize("method", INVALID_OUTLIER_METHODS)
    def test_invalid_outlier_method_not_in_registry(self, method):
        """Invalid/garbage methods must NOT be in the registry."""
        methods = get_valid_outlier_methods()
        assert method not in methods, (
            f"INVALID outlier method '{method}' found in registry! "
            f"This is GARBAGE data that should have been removed. "
            f"Check configs/mlflow_registry/parameters/classification.yaml"
        )


class TestImputationMethodContents:
    """Test that specific imputation methods exist in registry."""

    @pytest.mark.parametrize("method", EXPECTED_IMPUTATION_METHODS)
    def test_expected_imputation_method_exists(self, method):
        """Each expected imputation method must be in the registry."""
        methods = get_valid_imputation_methods()
        assert method in methods, (
            f"Expected imputation method '{method}' not found in registry. "
            f"Available methods: {methods}"
        )


class TestClassifierContents:
    """Test that specific classifiers exist in registry."""

    @pytest.mark.parametrize("classifier", EXPECTED_CLASSIFIERS)
    def test_expected_classifier_exists(self, classifier):
        """Each expected classifier must be in the registry."""
        classifiers = get_valid_classifiers()
        assert classifier in classifiers, (
            f"Expected classifier '{classifier}' not found in registry. "
            f"Available classifiers: {classifiers}"
        )


class TestValidationFunctions:
    """Test the validation helper functions."""

    def test_validate_valid_outlier(self):
        """Valid outlier methods should pass validation."""
        assert validate_outlier_method("pupil-gt") is True
        assert validate_outlier_method("MOMENT-gt-finetune") is True
        assert validate_outlier_method("LOF") is True

    def test_validate_invalid_outlier(self):
        """Invalid outlier methods should fail validation."""
        assert validate_outlier_method("anomaly") is False
        assert validate_outlier_method("exclude") is False
        assert validate_outlier_method("MOMENT-orig-finetune") is False
        assert validate_outlier_method("nonexistent") is False
        assert validate_outlier_method("") is False

    def test_validate_valid_imputation(self):
        """Valid imputation methods should pass validation."""
        assert validate_imputation_method("pupil-gt") is True
        assert validate_imputation_method("SAITS") is True

    def test_validate_invalid_imputation(self):
        """Invalid imputation methods should fail validation."""
        assert validate_imputation_method("nonexistent") is False
        assert validate_imputation_method("linear") is False  # Not in registry!

    def test_validate_valid_classifier(self):
        """Valid classifiers should pass validation."""
        assert validate_classifier("CatBoost") is True
        assert validate_classifier("XGBoost") is True

    def test_validate_invalid_classifier(self):
        """Invalid classifiers should fail validation."""
        assert validate_classifier("RandomForest") is False
        assert validate_classifier("SVM") is False


class TestFactorialDesign:
    """Test factorial design calculations."""

    def test_expected_factorial_size(self):
        """Full factorial design should be 11 × 8 × 5 = 440."""
        expected_size = 11 * 8 * 5
        assert (
            get_expected_factorial_size() == expected_size
        ), f"Factorial size should be {expected_size} (11 outliers × 8 imputations × 5 classifiers)"

    def test_catboost_only_factorial_size(self):
        """CatBoost-only design should be 11 × 8 × 1 = 88."""
        expected_size = 11 * 8
        assert get_expected_factorial_size("CatBoost") == expected_size


class TestMethodCategories:
    """Test that method categories are properly defined."""

    def test_outlier_categories_exist(self):
        """Outlier categories should be defined."""
        categories = get_outlier_categories()
        assert categories is not None
        # Check for expected category types
        assert "ground_truth" in categories or "traditional" in categories

    def test_imputation_categories_exist(self):
        """Imputation categories should be defined."""
        categories = get_imputation_categories()
        assert categories is not None


class TestRegistryIntegrity:
    """Test overall registry integrity."""

    def test_no_duplicate_outlier_methods(self):
        """No duplicate outlier methods should exist."""
        methods = get_valid_outlier_methods()
        assert (
            len(methods) == len(set(methods))
        ), f"Duplicate outlier methods found: {[m for m in methods if methods.count(m) > 1]}"

    def test_no_duplicate_imputation_methods(self):
        """No duplicate imputation methods should exist."""
        methods = get_valid_imputation_methods()
        assert len(methods) == len(set(methods)), "Duplicate imputation methods found"

    def test_no_duplicate_classifiers(self):
        """No duplicate classifiers should exist."""
        classifiers = get_valid_classifiers()
        assert len(classifiers) == len(set(classifiers)), "Duplicate classifiers found"

    def test_ground_truth_in_both_outlier_and_imputation(self):
        """pupil-gt should be valid for both outlier detection and imputation."""
        assert validate_outlier_method("pupil-gt") is True
        assert validate_imputation_method("pupil-gt") is True


# ============================================================================
# GOLDEN VALUE TESTS - Expected AUROC values for key configurations
# ============================================================================
# These are used by integration tests to validate data extraction.
# See: tests/integration/test_data_extraction.py

EXPECTED_GROUND_TRUTH_AUROC = 0.911  # pupil-gt + pupil-gt + CatBoost
EXPECTED_BEST_ENSEMBLE_AUROC = 0.913  # ensemble + CSDI + CatBoost
EXPECTED_AUROC_TOLERANCE = 0.002


def get_expected_auroc_for_ground_truth():
    """Return expected AUROC for ground truth configuration."""
    return EXPECTED_GROUND_TRUTH_AUROC, EXPECTED_AUROC_TOLERANCE


def get_expected_auroc_for_best_ensemble():
    """Return expected AUROC for best ensemble configuration."""
    return EXPECTED_BEST_ENSEMBLE_AUROC, EXPECTED_AUROC_TOLERANCE
