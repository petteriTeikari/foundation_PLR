"""Unit tests for configuration utilities.

Tests functions from src/orchestration/hyperparameter_sweep_utils.py for
configuration management and hyperparameter setup.
"""

import pytest
from omegaconf import OmegaConf

from src.orchestration.hyperparameter_sweep_utils import (
    drop_other_models,
    pick_cfg_key,
    flatten_the_nested_dicts,
)


class TestDropOtherModels:
    """Tests for drop_other_models function."""

    @pytest.fixture
    def multi_model_outlier_cfg(self):
        """Config with multiple outlier detection models."""
        return OmegaConf.create(
            {
                "OUTLIER_MODELS": {
                    "LOF": {"MODEL": {"n_neighbors": 20}},
                    "OneClassSVM": {"MODEL": {"nu": 0.1}},
                    "PROPHET": {"MODEL": {"interval_width": 0.95}},
                }
            }
        )

    @pytest.fixture
    def multi_model_imputation_cfg(self):
        """Config with multiple imputation models."""
        return OmegaConf.create(
            {
                "MODELS": {
                    "MissForest": {"MODEL": {"max_iter": 10}},
                    "SAITS": {"MODEL": {"n_layers": 2}},
                }
            }
        )

    @pytest.fixture
    def multi_model_classification_cfg(self):
        """Config with multiple classification models."""
        return OmegaConf.create(
            {
                "CLS_MODELS": {
                    "XGBOOST": {"MODEL": {"n_estimators": 100}},
                    "CatBoost": {"MODEL": {"iterations": 100}},
                }
            }
        )

    @pytest.mark.unit
    def test_drop_other_models_keeps_specified(self, multi_model_outlier_cfg):
        """Test that only specified model is kept."""
        result = drop_other_models(
            multi_model_outlier_cfg, model="LOF", task="outlier_detection"
        )

        assert "LOF" in result["OUTLIER_MODELS"]
        assert len(result["OUTLIER_MODELS"]) == 1

    @pytest.mark.unit
    def test_drop_other_models_removes_others(self, multi_model_outlier_cfg):
        """Test that other models are removed."""
        result = drop_other_models(
            multi_model_outlier_cfg, model="LOF", task="outlier_detection"
        )

        assert "OneClassSVM" not in result["OUTLIER_MODELS"]
        assert "PROPHET" not in result["OUTLIER_MODELS"]

    @pytest.mark.unit
    def test_drop_other_models_imputation(self, multi_model_imputation_cfg):
        """Test dropping for imputation task."""
        result = drop_other_models(
            multi_model_imputation_cfg, model="MissForest", task="imputation"
        )

        assert "MissForest" in result["MODELS"]
        assert "SAITS" not in result["MODELS"]
        assert len(result["MODELS"]) == 1

    @pytest.mark.unit
    def test_drop_other_models_classification(self, multi_model_classification_cfg):
        """Test dropping for classification task."""
        result = drop_other_models(
            multi_model_classification_cfg, model="XGBOOST", task="classification"
        )

        assert "XGBOOST" in result["CLS_MODELS"]
        assert "CatBoost" not in result["CLS_MODELS"]
        assert len(result["CLS_MODELS"]) == 1

    @pytest.mark.unit
    def test_drop_other_models_invalid_task(self, multi_model_outlier_cfg):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            drop_other_models(multi_model_outlier_cfg, model="LOF", task="invalid_task")

    @pytest.mark.unit
    def test_drop_other_models_preserves_model_config(self, multi_model_outlier_cfg):
        """Test that model configuration is preserved."""
        result = drop_other_models(
            multi_model_outlier_cfg, model="LOF", task="outlier_detection"
        )

        assert result["OUTLIER_MODELS"]["LOF"]["MODEL"]["n_neighbors"] == 20


class TestPickCfgKey:
    """Tests for pick_cfg_key function."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal config for testing."""
        return OmegaConf.create(
            {
                "OUTLIER_MODELS": {},
                "MODELS": {},
                "CLS_MODELS": {},
            }
        )

    @pytest.mark.unit
    def test_pick_cfg_key_outlier_detection(self, minimal_config):
        """Test key selection for outlier detection."""
        key = pick_cfg_key(minimal_config, task="outlier_detection")
        assert key == "OUTLIER_MODELS"

    @pytest.mark.unit
    def test_pick_cfg_key_imputation(self, minimal_config):
        """Test key selection for imputation."""
        key = pick_cfg_key(minimal_config, task="imputation")
        assert key == "MODELS"

    @pytest.mark.unit
    def test_pick_cfg_key_classification(self, minimal_config):
        """Test key selection for classification."""
        key = pick_cfg_key(minimal_config, task="classification")
        assert key == "CLS_MODELS"

    @pytest.mark.unit
    def test_pick_cfg_key_invalid_task(self, minimal_config):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            pick_cfg_key(minimal_config, task="unknown_task")


class TestFlattenTheNestedDicts:
    """Tests for flatten_the_nested_dicts function."""

    @pytest.mark.unit
    def test_flatten_nested_single_model(self):
        """Test flattening with single model."""
        cfgs = {
            "LOF": {
                "LOF_config1": {"param": 1},
                "LOF_config2": {"param": 2},
            }
        }

        result = flatten_the_nested_dicts(cfgs)

        # Should return the inner dict of the first (and only) key
        assert "LOF_config1" in result
        assert "LOF_config2" in result

    @pytest.mark.unit
    def test_flatten_nested_preserves_configs(self):
        """Test that config values are preserved after flattening."""
        cfgs = {
            "Model1": {
                "config_a": {"value": 100},
                "config_b": {"value": 200},
            }
        }

        result = flatten_the_nested_dicts(cfgs)

        assert result["config_a"]["value"] == 100
        assert result["config_b"]["value"] == 200

    @pytest.mark.unit
    def test_flatten_nested_returns_dict(self):
        """Test that function returns a dictionary."""
        cfgs = {
            "Model": {
                "config1": {},
            }
        }

        result = flatten_the_nested_dicts(cfgs)

        assert isinstance(result, dict)
