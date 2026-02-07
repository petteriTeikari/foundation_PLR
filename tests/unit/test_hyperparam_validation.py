"""
Unit tests for hyperparameter validation.

TDD: Tests verify that hyperparameter configurations are valid and consistent.

Tests check:
1. All classifiers have corresponding hyperparam files
2. Hyperparam values are in valid ranges
3. Consistency across hyperparam definitions
"""

from pathlib import Path

import pytest
import yaml


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent / "configs"


def get_hyperparam_configs() -> list[Path]:
    """Get all hyperparameter YAML files."""
    hp_dir = get_configs_dir() / "CLS_HYPERPARAMS"
    return sorted(hp_dir.glob("*_hyperparam_space.yaml"))


def get_classifier_configs() -> list[Path]:
    """Get all classifier config files."""
    cls_dir = get_configs_dir() / "CLS_MODELS"
    return sorted(cls_dir.glob("*.yaml"))


class TestHyperparamFilesExist:
    """Test that hyperparameter files exist for all classifiers."""

    @pytest.mark.unit
    def test_all_classifiers_have_hyperparam_file(self):
        """Each classifier in CLS_MODELS should have a hyperparam file."""
        cls_configs = get_classifier_configs()
        hp_configs = get_hyperparam_configs()

        # Extract classifier names from filenames
        classifier_names = {p.stem for p in cls_configs}
        hp_classifier_names = {
            p.stem.replace("_hyperparam_space", "") for p in hp_configs
        }

        missing = classifier_names - hp_classifier_names
        assert not missing, "Classifiers missing hyperparameter files:\n" + "\n".join(
            f"  - {c}" for c in sorted(missing)
        )


class TestCatBoostHyperparams:
    """Test CatBoost hyperparameter validity."""

    @pytest.fixture
    def catboost_hp(self):
        """Load CatBoost hyperparameter config."""
        hp_path = (
            get_configs_dir() / "CLS_HYPERPARAMS" / "CATBOOST_hyperparam_space.yaml"
        )
        assert hp_path.exists(), f"Missing: {hp_path}"
        return yaml.safe_load(hp_path.read_text())

    @pytest.mark.unit
    def test_catboost_depth_positive(self, catboost_hp):
        """CatBoost depth values must be positive integers."""
        search_space = catboost_hp.get("CATBOOST", {}).get("SEARCH_SPACE", {})
        optuna_space = search_space.get("OPTUNA", {})

        depths = optuna_space.get("depth", [])
        assert depths, "No depth values defined in CATBOOST config"

        for depth in depths:
            assert isinstance(depth, int), f"Depth must be int, got {type(depth)}"
            assert depth > 0, f"Depth must be positive, got {depth}"
            assert depth <= 16, f"Depth {depth} exceeds typical max (16)"

    @pytest.mark.unit
    def test_catboost_learning_rate_valid(self, catboost_hp):
        """CatBoost learning rates must be in (0, 1]."""
        search_space = catboost_hp.get("CATBOOST", {}).get("SEARCH_SPACE", {})
        optuna_space = search_space.get("OPTUNA", {})

        lrs = optuna_space.get("lr", [])
        assert lrs, "No learning rate values defined in CATBOOST config"

        for lr in lrs:
            assert isinstance(lr, (int, float)), f"LR must be numeric, got {type(lr)}"
            assert lr > 0, f"LR must be positive, got {lr}"
            assert lr <= 1, f"LR {lr} exceeds 1 (unusual but technically valid)"


class TestXGBoostHyperparams:
    """Test XGBoost hyperparameter validity."""

    @pytest.fixture
    def xgboost_hp(self):
        """Load XGBoost hyperparameter config."""
        hp_path = (
            get_configs_dir() / "CLS_HYPERPARAMS" / "XGBOOST_hyperparam_space.yaml"
        )
        assert hp_path.exists(), f"Missing: {hp_path}"
        return yaml.safe_load(hp_path.read_text())

    @pytest.mark.unit
    def test_xgboost_max_depth_positive(self, xgboost_hp):
        """XGBoost max_depth values must be positive integers."""
        search_space = xgboost_hp.get("XGBOOST", {}).get("SEARCH_SPACE", {})

        # Check HYPEROPT space
        hyperopt_space = search_space.get("HYPEROPT", {})
        max_depth = hyperopt_space.get("max_depth", {})

        if max_depth:
            low = max_depth.get("low", 1)
            high = max_depth.get("high", 10)
            assert low > 0, f"max_depth low must be positive, got {low}"
            assert high > low, f"max_depth high ({high}) must exceed low ({low})"
            assert high <= 20, f"max_depth {high} exceeds typical max (20)"

    @pytest.mark.unit
    def test_xgboost_eta_valid(self, xgboost_hp):
        """XGBoost eta (learning rate) must be in (0, 1]."""
        search_space = xgboost_hp.get("XGBOOST", {}).get("SEARCH_SPACE", {})
        hyperopt_space = search_space.get("HYPEROPT", {})

        eta = hyperopt_space.get("eta", {})
        if eta:
            low = eta.get("low", 0.01)
            high = eta.get("high", 0.3)
            assert low > 0, f"eta low must be positive, got {low}"
            assert high <= 1, f"eta high {high} exceeds 1"


class TestLogisticRegressionHyperparams:
    """Test Logistic Regression hyperparameter validity."""

    @pytest.fixture
    def logreg_hp(self):
        """Load LogisticRegression hyperparameter config."""
        hp_path = (
            get_configs_dir()
            / "CLS_HYPERPARAMS"
            / "LogisticRegression_hyperparam_space.yaml"
        )
        assert hp_path.exists(), f"Missing: {hp_path}"
        return yaml.safe_load(hp_path.read_text())

    @pytest.mark.unit
    def test_logreg_c_positive(self, logreg_hp):
        """LogisticRegression C values must be positive."""
        search_space = logreg_hp.get("LogisticRegression", {}).get("SEARCH_SPACE", {})
        gridsearch = search_space.get("GRID", search_space.get("GRIDSEARCH", {}))

        c_values = gridsearch.get("C", [])
        assert c_values, "No C values defined in LogisticRegression config"

        for c in c_values:
            assert isinstance(c, (int, float)), f"C must be numeric, got {type(c)}"
            assert c > 0, f"C must be positive, got {c}"


class TestHyperparamConsistency:
    """Test consistency across hyperparameter definitions."""

    @pytest.mark.unit
    def test_all_hyperparam_files_have_version(self):
        """All hyperparameter files should have version field."""
        hp_configs = get_hyperparam_configs()
        missing_version = []

        for hp_path in hp_configs:
            content = yaml.safe_load(hp_path.read_text())
            if content is None or not isinstance(content, dict):
                continue

            has_version = (
                "_version" in content or "VERSION" in content or "version" in content
            )
            if not has_version:
                missing_version.append(hp_path.name)

        assert not missing_version, "Hyperparam files missing version:\n" + "\n".join(
            f"  - {f}" for f in missing_version
        )

    @pytest.mark.unit
    def test_hyperparam_files_have_search_space(self):
        """All hyperparameter files should define a SEARCH_SPACE."""
        hp_configs = get_hyperparam_configs()
        missing_search_space = []

        for hp_path in hp_configs:
            content = yaml.safe_load(hp_path.read_text())
            if content is None or not isinstance(content, dict):
                continue

            # Find the classifier key (e.g., CATBOOST, XGBOOST)
            classifier_key = hp_path.stem.replace("_hyperparam_space", "")
            classifier_config = content.get(classifier_key, {})

            if not isinstance(classifier_config, dict):
                continue

            if "SEARCH_SPACE" not in classifier_config:
                missing_search_space.append(hp_path.name)

        assert not missing_search_space, (
            "Hyperparam files missing SEARCH_SPACE:\n"
            + "\n".join(f"  - {f}" for f in missing_search_space)
        )
