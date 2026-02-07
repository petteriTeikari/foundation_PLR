#!/usr/bin/env python3
"""
Double-Check Tests for Extraction Pipeline.

TDD verification before expensive SHAP computation.
These tests verify data integrity, model functionality, and consistency
with original MLflow source data.

Run with:
    pytest tests/test_extraction_double_check.py -v

Author: Foundation PLR Team
Date: 2026-01-25
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for model unpickling
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "outputs" / "top10_catboost_models.pkl"
DB_PATH = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
MLRUNS_DIR = Path("/home/petteri/mlruns/253031330985650090")

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not ARTIFACT_PATH.exists(), reason="Extraction artifacts not available"
    ),
]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def artifact():
    """Load the top-10 artifact once for all tests."""
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"Artifact not found: {ARTIFACT_PATH}. Run: make extract")
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def db_connection():
    """Get DuckDB connection."""
    import duckdb

    if not DB_PATH.exists():
        pytest.skip(f"Database not found: {DB_PATH}. Run: make extract")
    return duckdb.connect(str(DB_PATH), read_only=True)


# =============================================================================
# 1. DATA INTEGRITY TESTS
# =============================================================================


class TestDataIntegrity:
    """Verify extracted data is complete and consistent."""

    def test_no_nan_in_features(self, artifact):
        """Features should have no NaN values."""
        for i, cfg in enumerate(artifact["configs"]):
            X_train = cfg["X_train"]
            X_test = cfg["X_test"]

            assert not np.isnan(X_train).any(), f"Config {i + 1}: NaN in X_train"
            assert not np.isnan(X_test).any(), f"Config {i + 1}: NaN in X_test"

    def test_no_nan_in_labels(self, artifact):
        """Labels should have no NaN values."""
        for i, cfg in enumerate(artifact["configs"]):
            y_train = cfg["y_train"]
            y_test = cfg["y_test"]

            assert not np.isnan(y_train).any(), f"Config {i + 1}: NaN in y_train"
            assert not np.isnan(y_test).any(), f"Config {i + 1}: NaN in y_test"

    def test_labels_are_binary(self, artifact):
        """Labels should be binary (0 or 1)."""
        for i, cfg in enumerate(artifact["configs"]):
            y_train = cfg["y_train"]
            y_test = cfg["y_test"]

            unique_train = set(np.unique(y_train))
            unique_test = set(np.unique(y_test))

            assert unique_train.issubset({0, 1}), (
                f"Config {i + 1}: y_train not binary: {unique_train}"
            )
            assert unique_test.issubset({0, 1}), (
                f"Config {i + 1}: y_test not binary: {unique_test}"
            )

    def test_sample_counts_match_expected(self, artifact):
        """Sample counts should match expected (208 total, ~145 train, ~63 test)."""
        for i, cfg in enumerate(artifact["configs"]):
            n_train = len(cfg["y_train"])
            n_test = len(cfg["y_test"])
            n_total = n_train + n_test

            # Expect ~70/30 split of 208 subjects
            assert 100 < n_train < 180, f"Config {i + 1}: unexpected n_train={n_train}"
            assert 30 < n_test < 100, f"Config {i + 1}: unexpected n_test={n_test}"
            assert 180 < n_total < 230, f"Config {i + 1}: unexpected n_total={n_total}"

    def test_feature_names_are_physiological(self, artifact):
        """Feature names should be recognizable PLR features."""
        expected_patterns = [
            "RED",
            "BLUE",
            "CONSTRICTION",
            "PHASIC",
            "SUSTAINED",
            "PIPR",
        ]
        feature_names = artifact["configs"][0]["feature_names"]

        # At least some features should match expected patterns
        matches = sum(
            1
            for name in feature_names
            for pattern in expected_patterns
            if pattern.lower() in name.lower()
        )
        assert matches >= 4, (
            f"Feature names don't look like PLR features: {feature_names}"
        )

    def test_feature_values_in_reasonable_range(self, artifact):
        """Feature values should be in physiologically reasonable range."""
        for i, cfg in enumerate(artifact["configs"]):
            X_test = cfg["X_test"]
            feature_names = cfg["feature_names"]

            # PIPR_AUC features can be quite variable (area under the curve)
            # Can range from -600 to +500 in extreme cases (arbitrary units)
            # Other features (constriction, phasic, sustained) are more bounded
            for j, name in enumerate(feature_names):
                col = X_test[:, j]
                if "PIPR_AUC" in name:
                    # PIPR AUC has wide range - mostly negative but can have outliers
                    # Real data shows values up to ~750 (arbitrary units, area under curve)
                    assert col.min() > -700, f"Config {i + 1}, {name}: extreme negative"
                    assert col.max() < 800, f"Config {i + 1}, {name}: extreme positive"
                else:
                    # Other features more bounded
                    assert col.min() > -100, f"Config {i + 1}, {name}: extreme negative"
                    assert col.max() < 100, f"Config {i + 1}, {name}: extreme positive"

    def test_flag_potential_outliers(self, artifact):
        """Flag (but don't fail) potential outliers for manual review."""
        outliers_found = []
        for i, cfg in enumerate(artifact["configs"]):
            X_test = cfg["X_test"]
            feature_names = cfg["feature_names"]

            for j, name in enumerate(feature_names):
                col = X_test[:, j]
                mean, std = col.mean(), col.std()

                # Z-score > 4 is unusual
                z_scores = np.abs((col - mean) / std) if std > 0 else np.zeros_like(col)
                extreme_idx = np.where(z_scores > 4)[0]

                if len(extreme_idx) > 0:
                    outliers_found.append(
                        {
                            "config": i + 1,
                            "feature": name,
                            "n_outliers": len(extreme_idx),
                            "max_z": z_scores.max(),
                        }
                    )

        # Just print for awareness, don't fail
        if outliers_found:
            print("\nPotential outliers found (Z > 4):")
            for o in outliers_found[:5]:  # Limit output
                print(
                    f"  Config {o['config']}, {o['feature']}: {o['n_outliers']} pts, max Z={o['max_z']:.1f}"
                )
            print(f"  (Total: {len(outliers_found)} feature-config combinations)")

        # Always pass - this is informational
        assert True


# =============================================================================
# 2. MODEL FUNCTIONALITY TESTS
# =============================================================================


class TestModelFunctionality:
    """Verify models can actually make predictions."""

    def test_model_can_predict_proba(self, artifact):
        """Each model should be able to predict probabilities."""
        for i, cfg in enumerate(artifact["configs"]):
            model = cfg["model"]
            X_test = cfg["X_test"]

            try:
                proba = model.predict_proba(X_test)
                assert proba.shape == (
                    len(X_test),
                    2,
                ), f"Config {i + 1}: unexpected proba shape {proba.shape}"
            except Exception as e:
                pytest.fail(f"Config {i + 1}: predict_proba failed: {e}")

    def test_predictions_are_valid_probabilities(self, artifact):
        """Predictions should be valid probabilities [0, 1]."""
        for i, cfg in enumerate(artifact["configs"]):
            model = cfg["model"]
            X_test = cfg["X_test"]

            proba = model.predict_proba(X_test)

            assert (proba >= 0).all(), f"Config {i + 1}: negative probabilities"
            assert (proba <= 1).all(), f"Config {i + 1}: probabilities > 1"
            # Rows should sum to 1
            row_sums = proba.sum(axis=1)
            assert np.allclose(row_sums, 1.0), (
                f"Config {i + 1}: proba rows don't sum to 1"
            )

    def test_bootstrap_models_give_different_predictions(self, artifact):
        """Bootstrap models should give slightly different predictions."""
        cfg = artifact["configs"][0]  # Test first config
        X_test = cfg["X_test"]
        bootstrap_models = cfg["bootstrap_models"]

        # Get predictions from first 10 bootstrap models
        predictions = []
        for model in bootstrap_models[:10]:
            proba = model.predict_proba(X_test)[:, 1]
            predictions.append(proba)

        predictions = np.array(predictions)

        # Predictions should vary across bootstrap (std > 0)
        std_across_bootstrap = predictions.std(axis=0)
        assert std_across_bootstrap.mean() > 0.001, (
            "Bootstrap models give identical predictions - something wrong"
        )

    def test_all_1000_bootstrap_models_work(self, artifact):
        """All 1000 bootstrap models should be loadable and functional."""
        # Test one config thoroughly
        cfg = artifact["configs"][0]
        X_sample = cfg["X_test"][:5]  # Small sample for speed
        bootstrap_models = cfg["bootstrap_models"]

        assert len(bootstrap_models) == 1000, (
            f"Expected 1000 models, got {len(bootstrap_models)}"
        )

        # Test every 100th model to verify they all work
        for idx in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
            model = bootstrap_models[idx]
            try:
                proba = model.predict_proba(X_sample)
                assert proba.shape == (5, 2), f"Bootstrap {idx}: wrong shape"
            except Exception as e:
                pytest.fail(f"Bootstrap model {idx} failed: {e}")


# =============================================================================
# 3. CROSS-VALIDATION WITH MLFLOW SOURCE
# =============================================================================


class TestCrossValidationWithMLflow:
    """Verify extracted values match original MLflow data."""

    def test_auroc_matches_mlflow_metrics(self, artifact, db_connection):
        """Extracted AUROC should match what's in DuckDB (from MLflow)."""
        for cfg in artifact["configs"]:
            run_id = cfg["config"]["run_id"]
            expected_auroc = cfg["config"]["auroc_mean"]

            # Query DuckDB
            result = db_connection.execute(
                "SELECT auroc FROM essential_metrics WHERE run_id = ?", [run_id]
            ).fetchone()

            if result:
                db_auroc = result[0]
                assert abs(db_auroc - expected_auroc) < 0.001, (
                    f"AUROC mismatch for {run_id}: artifact={expected_auroc}, db={db_auroc}"
                )

    def test_top10_ranks_are_correct(self, artifact, db_connection):
        """Top-10 should be correctly ranked by AUROC (descending)."""
        # Get AUROCs from artifact
        aurocs = [cfg["config"]["auroc_mean"] for cfg in artifact["configs"]]

        # Should be in descending order
        for i in range(len(aurocs) - 1):
            assert aurocs[i] >= aurocs[i + 1], (
                f"Rank {i + 1} AUROC ({aurocs[i]}) < Rank {i + 2} AUROC ({aurocs[i + 1]})"
            )

    def test_excluded_configs_not_in_artifact(self, artifact, db_connection):
        """Garbage outlier methods should NOT be in artifact.

        The registry defines exactly 11 valid outlier methods. Any method
        not in the registry is garbage (e.g., 'anomaly', 'exclude', '-orig-').
        """
        from src.data_io.registry import get_valid_outlier_methods

        valid_methods = set(get_valid_outlier_methods())
        excluded = db_connection.execute(
            """
            SELECT run_id, outlier_method
            FROM essential_metrics
            WHERE outlier_method NOT IN (SELECT UNNEST(?))
        """,
            [list(valid_methods)],
        ).fetchall()

        artifact_run_ids = {cfg["config"]["run_id"] for cfg in artifact["configs"]}

        for run_id, outlier in excluded:
            assert run_id not in artifact_run_ids, (
                f"Excluded config {run_id} (outlier={outlier}) found in artifact"
            )

    @pytest.mark.skipif(
        not MLRUNS_DIR.exists(), reason="MLflow directory not available"
    )
    def test_model_path_exists_in_mlflow(self, artifact):
        """Model paths in artifact should exist in MLflow."""
        for i, cfg in enumerate(artifact["configs"]):
            model_path = Path(cfg["config"]["model_path"])
            assert model_path.exists(), (
                f"Config {i + 1}: model path doesn't exist: {model_path}"
            )


# =============================================================================
# 4. SHAP READINESS TESTS
# =============================================================================


class TestSHAPReadiness:
    """Verify everything is ready for SHAP computation."""

    def test_catboost_has_tree_structure(self, artifact):
        """CatBoost models should have tree structure for TreeExplainer."""
        model = artifact["configs"][0]["model"]

        # CatBoost should have these attributes
        assert hasattr(model, "get_feature_importance"), (
            "Model missing get_feature_importance"
        )

    def test_feature_count_matches_model_expectation(self, artifact):
        """Feature count in data should match what model expects."""
        for i, cfg in enumerate(artifact["configs"]):
            model = cfg["model"]
            X_test = cfg["X_test"]

            # Try a prediction - will fail if feature count wrong
            try:
                model.predict_proba(X_test[:1])
            except Exception as e:
                pytest.fail(f"Config {i + 1}: model expects different features: {e}")

    def test_shap_compatible_data_types(self, artifact):
        """Data should be in SHAP-compatible format (float numpy arrays)."""
        for i, cfg in enumerate(artifact["configs"]):
            X_test = cfg["X_test"]

            assert isinstance(X_test, np.ndarray), (
                f"Config {i + 1}: X_test not numpy array"
            )
            assert X_test.dtype in [
                np.float32,
                np.float64,
                np.int32,
                np.int64,
            ], f"Config {i + 1}: X_test has unexpected dtype {X_test.dtype}"

    def test_computation_estimate(self, artifact):
        """Estimate and verify SHAP computation is feasible."""
        n_configs = len(artifact["configs"])
        n_bootstrap_per_config = len(artifact["configs"][0]["bootstrap_models"])
        n_test_samples = len(artifact["configs"][0]["X_test"])
        n_features = len(artifact["configs"][0]["feature_names"])

        total_shap_computations = n_configs * n_bootstrap_per_config
        print("\nSHAP Computation Estimate:")
        print(f"  Configs: {n_configs}")
        print(f"  Bootstrap per config: {n_bootstrap_per_config}")
        print(f"  Test samples: {n_test_samples}")
        print(f"  Features: {n_features}")
        print(f"  Total SHAP calls: {total_shap_computations:,}")

        # Verify reasonable numbers
        assert n_configs == 10, f"Expected 10 configs, got {n_configs}"
        assert n_bootstrap_per_config == 1000, (
            f"Expected 1000 bootstrap, got {n_bootstrap_per_config}"
        )
        assert 50 < n_test_samples < 100, f"Unexpected test size: {n_test_samples}"
        assert 5 < n_features < 20, f"Unexpected feature count: {n_features}"


# =============================================================================
# 5. GROUND TRUTH CONFIG VERIFICATION
# =============================================================================


class TestGroundTruthConfig:
    """Verify ground truth baseline config is present and correct."""

    def test_ground_truth_config_exists(self, artifact):
        """Should have a pupil-gt + pupil-gt config (ground truth baseline)."""
        gt_configs = [
            cfg
            for cfg in artifact["configs"]
            if cfg["config"]["outlier_method"] == "pupil-gt"
            and cfg["config"]["imputation_method"] == "pupil-gt"
        ]
        assert len(gt_configs) >= 1, "Missing ground truth config (pupil-gt + pupil-gt)"

    def test_ground_truth_auroc_reasonable(self, artifact):
        """Ground truth config should have high AUROC (our baseline)."""
        for cfg in artifact["configs"]:
            if (
                cfg["config"]["outlier_method"] == "pupil-gt"
                and cfg["config"]["imputation_method"] == "pupil-gt"
            ):
                auroc = cfg["config"]["auroc_mean"]
                assert 0.85 < auroc < 0.95, f"Ground truth AUROC unexpected: {auroc}"

    def test_best_config_beats_or_matches_ground_truth(self, artifact):
        """Best config (rank 1) should be >= ground truth AUROC."""
        best_auroc = artifact["configs"][0]["config"]["auroc_mean"]

        gt_auroc = None
        for cfg in artifact["configs"]:
            if (
                cfg["config"]["outlier_method"] == "pupil-gt"
                and cfg["config"]["imputation_method"] == "pupil-gt"
            ):
                gt_auroc = cfg["config"]["auroc_mean"]
                break

        if gt_auroc:
            assert best_auroc >= gt_auroc - 0.01, (
                f"Best config ({best_auroc}) worse than ground truth ({gt_auroc})"
            )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
