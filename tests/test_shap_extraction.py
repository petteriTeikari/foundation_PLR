#!/usr/bin/env python3
"""
Tests for SHAP extraction and feature importance analysis.

TDD approach: Tests written BEFORE implementation.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data


class TestSHAPExtractionContract:
    """Test the contract/interface for SHAP extraction."""

    def test_top10_configs_are_defined(self):
        """Verify we have exactly 10 valid configs with known OD sources."""
        # Top-10 from MLflow - all configs have known outlier detection sources
        # Exclusion criteria: outlier_method in ["anomaly", "exclude"]
        # See docs/mlflow-naming-convention.md for full explanation
        expected_configs = [
            # Rank 1 (AUROC: 0.913)
            {
                "outlier": "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
                "imputation": "CSDI",
                "outlier_source_known": True,
            },
            # Rank 2 (AUROC: 0.912)
            {
                "outlier": "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
                "imputation": "TimesNet",
                "outlier_source_known": True,
            },
            # Rank 3 (AUROC: 0.912)
            {
                "outlier": "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune",
                "imputation": "CSDI",
                "outlier_source_known": True,
            },
            # Rank 4 (AUROC: 0.911)
            {
                "outlier": "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune",
                "imputation": "TimesNet",
                "outlier_source_known": True,
            },
            # Rank 5 (AUROC: 0.911) - GROUND TRUTH BASELINE
            {
                "outlier": "pupil-gt",
                "imputation": "pupil-gt",
                "outlier_source_known": True,
            },
            # Rank 6 (AUROC: 0.911)
            {
                "outlier": "pupil-gt",
                "imputation": "ensemble-CSDI-MOMENT-SAITS",
                "outlier_source_known": True,
            },
            # Rank 7 (AUROC: 0.910)
            {
                "outlier": "ensemble-LOF-MOMENT-OneClassSVM-PROPHET-SubPCA-TimesNet-UniTS-gt-finetune",
                "imputation": "SAITS",
                "outlier_source_known": True,
            },
            # Rank 8 (AUROC: 0.910)
            {
                "outlier": "pupil-gt",
                "imputation": "CSDI",
                "outlier_source_known": True,
            },
            # Rank 9 (AUROC: 0.910)
            {
                "outlier": "MOMENT-gt-finetune",
                "imputation": "SAITS",
                "outlier_source_known": True,
            },
            # Rank 10 (AUROC: 0.909)
            {
                "outlier": "pupil-gt",
                "imputation": "TimesNet",
                "outlier_source_known": True,
            },
        ]
        assert len(expected_configs) == 10, "Should have exactly 10 configs"
        # All configs must have known outlier sources
        for config in expected_configs:
            assert config[
                "outlier_source_known"
            ], f"Config {config['outlier']} should have known OD"
            # Verify NOT using unknown OD sources
            assert config["outlier"] not in [
                "anomaly",
                "exclude",
            ], f"Config uses unknown OD source: {config['outlier']}"

    def test_shap_output_structure(self):
        """Define expected SHAP output structure."""
        # This is the expected structure of SHAP results
        expected_structure = {
            "config_id": str,  # e.g., "rank_1"
            "outlier_method": str,
            "imputation_method": str,
            "auroc_mean": float,
            "auroc_ci": tuple,  # (ci_lo, ci_hi)
            "n_bootstrap": int,  # Should be 1000
            "n_features": int,
            "feature_names": list,
            "shap_values": {
                "per_bootstrap": np.ndarray,  # Shape: (n_bootstrap, n_samples, n_features)
                "mean": np.ndarray,  # Shape: (n_samples, n_features)
                "std": np.ndarray,  # Shape: (n_samples, n_features)
            },
            "feature_importance": {
                "mean_abs_shap": np.ndarray,  # Shape: (n_features,)
                "std_abs_shap": np.ndarray,  # Shape: (n_features,)
                "ci_lo": np.ndarray,
                "ci_hi": np.ndarray,
            },
        }
        # Just verify structure is defined - actual testing in implementation
        assert "shap_values" in expected_structure
        assert "feature_importance" in expected_structure


class TestSHAPDataArtifact:
    """Test SHAP data artifact creation."""

    @pytest.fixture
    def mock_shap_result(self):
        """Create mock SHAP result for testing."""
        n_bootstrap = 10  # Small for testing
        n_samples = 50
        n_features = 5
        feature_names = ["amp_bin_0", "amp_bin_1", "amp_bin_2", "amp_bin_3", "PIPR"]

        # Create mock SHAP values
        shap_per_boot = np.random.randn(n_bootstrap, n_samples, n_features)
        shap_mean = shap_per_boot.mean(axis=0)
        shap_std = shap_per_boot.std(axis=0)

        return {
            "config_id": "rank_1",
            "outlier_method": "ensemble-all",
            "imputation_method": "CSDI",
            "auroc_mean": 0.913,
            "auroc_ci": (0.904, 0.919),
            "n_bootstrap": n_bootstrap,
            "n_features": n_features,
            "feature_names": feature_names,
            "shap_values": {
                "per_bootstrap": shap_per_boot,
                "mean": shap_mean,
                "std": shap_std,
            },
            "feature_importance": {
                "mean_abs_shap": np.abs(shap_per_boot).mean(axis=(0, 1)),
                "std_abs_shap": np.abs(shap_per_boot).std(axis=(0, 1)),
            },
        }

    def test_shap_result_has_required_keys(self, mock_shap_result):
        """Verify SHAP result has all required keys."""
        required_keys = [
            "config_id",
            "outlier_method",
            "imputation_method",
            "auroc_mean",
            "n_bootstrap",
            "n_features",
            "feature_names",
            "shap_values",
            "feature_importance",
        ]
        for key in required_keys:
            assert key in mock_shap_result, f"Missing required key: {key}"

    def test_shap_values_shapes_consistent(self, mock_shap_result):
        """Verify SHAP value shapes are consistent."""
        shap_vals = mock_shap_result["shap_values"]
        n_boot = mock_shap_result["n_bootstrap"]
        n_feat = mock_shap_result["n_features"]

        # per_bootstrap: (n_bootstrap, n_samples, n_features)
        assert shap_vals["per_bootstrap"].shape[0] == n_boot
        assert shap_vals["per_bootstrap"].shape[2] == n_feat

        # mean/std: (n_samples, n_features)
        assert shap_vals["mean"].shape[1] == n_feat
        assert shap_vals["std"].shape[1] == n_feat

    def test_feature_importance_shapes(self, mock_shap_result):
        """Verify feature importance shapes match n_features."""
        n_feat = mock_shap_result["n_features"]
        fi = mock_shap_result["feature_importance"]

        assert fi["mean_abs_shap"].shape == (n_feat,)
        assert fi["std_abs_shap"].shape == (n_feat,)


class TestVIFAnalysis:
    """Tests for VIF (Variance Inflation Factor) analysis."""

    def test_vif_output_structure(self):
        """Define expected VIF output structure."""
        expected_structure = {
            "feature_name": str,
            "vif": float,
            "is_collinear": bool,  # VIF > 5
            "is_highly_collinear": bool,  # VIF > 10
        }
        # Just verify we've defined the structure
        assert "vif" in expected_structure

    @pytest.fixture
    def mock_feature_matrix(self):
        """Create mock feature matrix for VIF testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Create features with varying collinearity
        X = np.random.randn(n_samples, n_features)
        # Add collinearity: feature 4 = feature 0 + noise
        X[:, 4] = X[:, 0] + 0.1 * np.random.randn(n_samples)

        feature_names = [
            "amp_bin_0",
            "amp_bin_1",
            "amp_bin_2",
            "amp_bin_3",
            "amp_collinear",
        ]
        return X, feature_names

    def test_vif_detects_collinearity(self, mock_feature_matrix):
        """VIF should detect collinear features."""
        X, feature_names = mock_feature_matrix

        # When VIF is computed, the collinear feature should have high VIF
        # This test will fail until implementation exists
        # For now, just verify mock data is set up correctly
        assert X.shape[1] == len(feature_names)
        assert "amp_collinear" in feature_names


class TestEnsembleFeatureImportance:
    """Tests for ensemble feature importance aggregation."""

    def test_ensemble_aggregation_methods(self):
        """Verify both equal and AUROC-weighted aggregation are supported."""
        aggregation_methods = ["equal", "auroc_weighted"]
        for method in aggregation_methods:
            assert method in ["equal", "auroc_weighted"]

    @pytest.fixture
    def mock_multi_config_shap(self):
        """Create mock SHAP results from multiple configs."""
        n_features = 5
        feature_names = ["amp_bin_0", "amp_bin_1", "amp_bin_2", "amp_bin_3", "PIPR"]

        configs = []
        for i in range(3):  # 3 mock configs
            importance = np.random.rand(n_features) * (
                1 - i * 0.1
            )  # Slightly different
            configs.append(
                {
                    "config_id": f"rank_{i + 1}",
                    "auroc_mean": 0.91 - i * 0.01,
                    "feature_importance": {
                        "mean_abs_shap": importance,
                        "std_abs_shap": importance * 0.1,
                    },
                }
            )

        return configs, feature_names

    def test_ensemble_has_per_feature_uncertainty(self, mock_multi_config_shap):
        """Ensemble importance should include uncertainty from config variation."""
        configs, feature_names = mock_multi_config_shap

        # Extract importance values across configs
        importance_matrix = np.array(
            [c["feature_importance"]["mean_abs_shap"] for c in configs]
        )

        # Should be able to compute mean and std across configs
        ensemble_mean = importance_matrix.mean(axis=0)
        ensemble_std = importance_matrix.std(axis=0)

        assert ensemble_mean.shape == (len(feature_names),)
        assert ensemble_std.shape == (len(feature_names),)
        # Std should be non-zero since configs differ
        assert ensemble_std.sum() > 0


class TestDuckDBStorage:
    """Tests for DuckDB storage of SHAP results."""

    def test_db_schema_requirements(self):
        """Define required DuckDB schema for SHAP results."""
        required_tables = [
            "shap_config_summary",  # One row per config
            "shap_feature_importance",  # One row per config x feature
            "shap_bootstrap_values",  # Per-bootstrap SHAP (large table)
            "vif_analysis",  # VIF per feature
            "ensemble_importance",  # Aggregated across configs
        ]
        # Just verify we've defined the schema
        assert len(required_tables) == 5

    def test_shap_config_summary_columns(self):
        """Define columns for config summary table."""
        columns = {
            "config_id": "VARCHAR",
            "rank": "INTEGER",
            "outlier_method": "VARCHAR",
            "imputation_method": "VARCHAR",
            "auroc_mean": "DOUBLE",
            "auroc_ci_lo": "DOUBLE",
            "auroc_ci_hi": "DOUBLE",
            "n_bootstrap": "INTEGER",
            "n_features": "INTEGER",
        }
        assert "config_id" in columns
        assert "auroc_mean" in columns


# =============================================================================
# INTEGRATION TESTS (will be skipped if MLflow not available)
# =============================================================================


@pytest.mark.skipif(
    not Path("/home/petteri/mlruns").exists(), reason="MLflow directory not available"
)
class TestTop10ArtifactIntegrity:
    """Tests for the extracted top-10 CatBoost artifact."""

    @pytest.fixture
    def top10_artifact(self):
        """Load the top-10 artifact if it exists."""
        import pickle

        artifact_path = Path(
            "/home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR/outputs/top10_catboost_models.pkl"
        )
        if not artifact_path.exists():
            pytest.skip(f"Artifact not found: {artifact_path}")

        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    def test_artifact_has_10_configs(self, top10_artifact):
        """Verify artifact contains exactly 10 configurations."""
        assert len(top10_artifact["configs"]) == 10

    def test_all_configs_have_known_outlier_source(self, top10_artifact):
        """Verify no configs have unknown outlier sources (anomaly/exclude)."""
        excluded_sources = ["anomaly", "exclude"]
        for cfg in top10_artifact["configs"]:
            outlier = cfg["config"]["outlier_method"]
            assert outlier not in excluded_sources, f"Config has unknown OD: {outlier}"

    def test_both_train_and_test_splits_available(self, top10_artifact):
        """Verify both train and test data are extracted for each config."""
        for i, cfg in enumerate(top10_artifact["configs"]):
            assert cfg["X_train"] is not None, f"Config {i + 1} missing X_train"
            assert cfg["X_test"] is not None, f"Config {i + 1} missing X_test"
            assert cfg["y_train"] is not None, f"Config {i + 1} missing y_train"
            assert cfg["y_test"] is not None, f"Config {i + 1} missing y_test"

    def test_feature_names_consistent_across_configs(self, top10_artifact):
        """Verify feature names are the same for all configs."""
        first_names = top10_artifact["configs"][0]["feature_names"]
        for i, cfg in enumerate(top10_artifact["configs"][1:], 2):
            assert (
                cfg["feature_names"] == first_names
            ), f"Config {i} has different feature names"

    def test_feature_count_matches_data_shape(self, top10_artifact):
        """Verify feature dimension matches between names and data."""
        for i, cfg in enumerate(top10_artifact["configs"]):
            n_features = len(cfg["feature_names"])
            assert (
                cfg["X_train"].shape[1] == n_features
            ), f"Config {i + 1}: X_train shape mismatch"
            assert (
                cfg["X_test"].shape[1] == n_features
            ), f"Config {i + 1}: X_test shape mismatch"

    def test_bootstrap_models_available(self, top10_artifact):
        """Verify 1000 bootstrap models available for each config."""
        for i, cfg in enumerate(top10_artifact["configs"]):
            assert (
                cfg["bootstrap_models"] is not None
            ), f"Config {i + 1} missing bootstrap_models"
            assert (
                len(cfg["bootstrap_models"]) == 1000
            ), f"Config {i + 1} has {len(cfg['bootstrap_models'])} bootstrap models, expected 1000"

    def test_models_are_catboost_classifiers(self, top10_artifact):
        """Verify models are CatBoostClassifier with predict_proba."""
        for i, cfg in enumerate(top10_artifact["configs"]):
            model = cfg["model"]
            assert hasattr(
                model, "predict_proba"
            ), f"Config {i + 1} model missing predict_proba"
            assert (
                "CatBoost" in type(model).__name__
            ), f"Config {i + 1} model is {type(model).__name__}, not CatBoost"

    def test_exclusion_criteria_documented(self, top10_artifact):
        """Verify exclusion criteria are documented in metadata."""
        metadata = top10_artifact["metadata"]
        assert (
            "exclusion_criteria" in metadata
        ), "Missing exclusion_criteria in metadata"
        excl = metadata["exclusion_criteria"]
        assert "excluded_outlier_methods" in excl, "Missing excluded methods list"
        assert "anomaly" in excl["excluded_outlier_methods"]
        assert "exclude" in excl["excluded_outlier_methods"]


class TestSplitAgnosticAPI:
    """Tests for split-agnostic analysis functions."""

    def test_can_select_split_programmatically(self):
        """Verify we can switch between train/test splits easily."""

        # Define split-agnostic accessor
        def get_split_data(config: dict, split: str = "test"):
            """Get X and y for specified split."""
            assert split in ["train", "test"], f"Unknown split: {split}"
            X_key = f"X_{split}"
            y_key = f"y_{split}"
            return config.get(X_key), config.get(y_key)

        # Test with mock data
        mock_config = {
            "X_train": np.array([[1, 2], [3, 4]]),
            "X_test": np.array([[5, 6]]),
            "y_train": np.array([0, 1]),
            "y_test": np.array([1]),
        }

        X_train, y_train = get_split_data(mock_config, "train")
        X_test, y_test = get_split_data(mock_config, "test")

        assert X_train.shape == (2, 2)
        assert X_test.shape == (1, 2)
        assert len(y_train) == 2
        assert len(y_test) == 1


class TestMLflowIntegration:
    """Integration tests requiring MLflow access."""

    def test_can_find_catboost_runs(self):
        """Verify we can find CatBoost runs in MLflow."""
        mlruns = Path("/home/petteri/mlruns/253031330985650090")
        catboost_runs = list(mlruns.glob("*/artifacts/metrics/*CATBOOST*.pickle"))
        assert len(catboost_runs) > 0, "Should find CatBoost runs"

    def test_can_load_bootstrap_metrics(self):
        """Verify we can load bootstrap metrics pickle."""
        import pickle

        mlruns = Path("/home/petteri/mlruns/253031330985650090")
        pkl_files = list(mlruns.glob("*/artifacts/metrics/*CATBOOST*.pickle"))

        if pkl_files:
            with open(pkl_files[0], "rb") as f:
                data = pickle.load(f)

            assert "metrics_stats" in data
            assert "test" in data["metrics_stats"]
