"""Unit tests for ensemble utilities.

Tests functions from src/ensemble/ensemble_utils.py for MOMENT model handling
and edge cases discovered during synthetic pipeline testing.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.ensemble.ensemble_utils import (
    get_best_moment,
    get_best_moment_variant,
    get_best_moments_per_source,
    get_non_moment_models,
)


class TestGetBestMoment:
    """Tests for get_best_moment function handling missing MOMENT models."""

    @pytest.fixture
    def best_metric_cfg(self):
        """Standard metric configuration for testing."""
        return OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})

    @pytest.mark.unit
    def test_handles_none_input(self, best_metric_cfg):
        """Should return None when runs_moment is None."""
        result = get_best_moment(best_metric_cfg, None)
        assert result is None

    @pytest.mark.unit
    def test_handles_empty_dataframe(self, best_metric_cfg):
        """Should return None when runs_moment is empty DataFrame."""
        empty_df = pd.DataFrame()
        result = get_best_moment(best_metric_cfg, empty_df)
        assert result is None

    @pytest.mark.unit
    def test_handles_valid_moment_runs_desc(self, best_metric_cfg):
        """Should return best MOMENT run when sorted DESC."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                ],
                "metrics.test/f1": [0.85, 0.80],
            }
        )
        result = get_best_moment(best_metric_cfg, runs)
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["metrics.test/f1"] == 0.85

    @pytest.mark.unit
    def test_handles_valid_moment_runs_asc(self):
        """Should return best MOMENT run when sorted ASC (for loss metrics)."""
        cfg = OmegaConf.create({"direction": "ASC", "split": "test", "string": "mae"})
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                ],
                "metrics.test/mae": [0.10, 0.15],  # Lower is better
            }
        )
        result = get_best_moment(cfg, runs)
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["metrics.test/mae"] == 0.10


class TestGetNonMomentModels:
    """Tests for get_non_moment_models function."""

    @pytest.mark.unit
    def test_filters_out_moment_models(self):
        """Should filter out runs with MOMENT in name."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "SAITS__pupil_gt_",
                    "MOMENT-finetune__pupil_gt_",
                    "CSDI__LOF",
                    "MOMENT-zeroshot__LOF",
                ],
                "metrics.test/f1": [0.80, 0.85, 0.82, 0.83],
            }
        )
        result = get_non_moment_models(runs)

        assert len(result) == 2
        assert "SAITS__pupil_gt_" in result["tags.mlflow.runName"].values
        assert "CSDI__LOF" in result["tags.mlflow.runName"].values
        assert "MOMENT-finetune__pupil_gt_" not in result["tags.mlflow.runName"].values

    @pytest.mark.unit
    def test_returns_empty_when_all_moment(self):
        """Should return empty DataFrame when all runs are MOMENT."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                ],
                "metrics.test/f1": [0.85, 0.80],
            }
        )
        result = get_non_moment_models(runs)
        assert result.empty

    @pytest.mark.unit
    def test_returns_all_when_no_moment(self):
        """Should return all runs when none are MOMENT."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "SAITS__pupil_gt_",
                    "CSDI__LOF",
                    "TimesNet__pupil_gt_",
                ],
                "metrics.test/f1": [0.80, 0.82, 0.81],
            }
        )
        result = get_non_moment_models(runs)
        assert len(result) == 3


class TestGetBestMomentsPerSource:
    """Tests for get_best_moments_per_source function."""

    @pytest.fixture
    def best_metric_cfg(self):
        """Standard metric configuration for testing."""
        return OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})

    @pytest.mark.unit
    def test_handles_no_moment_models(self, best_metric_cfg):
        """Should return None when no MOMENT models exist."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": ["SAITS__pupil_gt_", "CSDI__LOF"],
                "metrics.test/f1": [0.80, 0.82],
            }
        )
        result = get_best_moments_per_source(runs, best_metric_cfg)
        assert result is None

    @pytest.mark.unit
    def test_selects_best_moment_per_source(self, best_metric_cfg):
        """Should select best MOMENT model per unique source."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                    "MOMENT-finetune__LOF",
                    "SAITS__pupil_gt_",
                ],
                "metrics.test/f1": [0.85, 0.80, 0.83, 0.82],
            }
        )
        result = get_best_moments_per_source(runs, best_metric_cfg)

        assert result is not None
        # Should have 2 results: best for pupil_gt_ and best for LOF
        assert len(result) == 2


class TestGetBestMomentVariant:
    """Tests for get_best_moment_variant function - critical for pipeline robustness."""

    @pytest.fixture
    def best_metric_cfg(self):
        """Standard metric configuration for testing."""
        return OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})

    @pytest.mark.unit
    def test_handles_empty_input(self, best_metric_cfg):
        """Should return empty DataFrame when no models exist.

        This was the original bug - pipeline crashed with IndexError.
        """
        empty_df = pd.DataFrame()
        result = get_best_moment_variant(empty_df, best_metric_cfg, return_best_gt=True)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.unit
    def test_handles_only_non_moment_models(self, best_metric_cfg):
        """Should return non-MOMENT models when no MOMENT exists.

        This is the key edge case from synthetic pipeline testing.
        """
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": ["SAITS__pupil_gt_", "CSDI__pupil_gt_"],
                "metrics.test/f1": [0.80, 0.82],
            }
        )
        result = get_best_moment_variant(runs, best_metric_cfg, return_best_gt=True)

        assert len(result) == 2
        assert "SAITS__pupil_gt_" in result["tags.mlflow.runName"].values
        assert "CSDI__pupil_gt_" in result["tags.mlflow.runName"].values

    @pytest.mark.unit
    def test_handles_only_moment_models(self, best_metric_cfg):
        """Should return MOMENT model when no non-MOMENT exists."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                ],
                "metrics.test/f1": [0.85, 0.80],
            }
        )
        result = get_best_moment_variant(runs, best_metric_cfg, return_best_gt=True)

        # Should return just the best MOMENT
        assert len(result) == 1
        assert result.iloc[0]["metrics.test/f1"] == 0.85

    @pytest.mark.unit
    def test_handles_mixed_models(self, best_metric_cfg):
        """Should combine best MOMENT with all non-MOMENT models."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "SAITS__pupil_gt_",
                    "MOMENT-finetune__pupil_gt_",
                    "MOMENT-zeroshot__pupil_gt_",
                    "CSDI__pupil_gt_",
                ],
                "metrics.test/f1": [0.80, 0.85, 0.83, 0.82],
            }
        )
        result = get_best_moment_variant(runs, best_metric_cfg, return_best_gt=True)

        # Should have: SAITS, CSDI (non-MOMENT) + best MOMENT
        assert len(result) == 3
        run_names = result["tags.mlflow.runName"].values
        assert "SAITS__pupil_gt_" in run_names
        assert "CSDI__pupil_gt_" in run_names
        # Either finetune or zeroshot should be present (the best one)
        assert (
            "MOMENT-finetune__pupil_gt_" in run_names
            or "MOMENT-zeroshot__pupil_gt_" in run_names
        )

    @pytest.mark.unit
    def test_handles_return_best_gt_false(self, best_metric_cfg):
        """Should work correctly with return_best_gt=False."""
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": [
                    "SAITS__LOF",
                    "MOMENT-finetune__LOF",
                    "CSDI__OneClassSVM",
                ],
                "metrics.test/f1": [0.80, 0.85, 0.82],
            }
        )
        result = get_best_moment_variant(runs, best_metric_cfg, return_best_gt=False)

        # Should include non-MOMENT + best MOMENT per source
        assert not result.empty


class TestSyntheticPipelineEdgeCases:
    """Integration-style tests for edge cases discovered in synthetic pipeline."""

    @pytest.fixture
    def best_metric_cfg(self):
        return OmegaConf.create({"direction": "DESC", "split": "test", "string": "f1"})

    @pytest.mark.unit
    def test_synthetic_pipeline_no_moment_scenario(self, best_metric_cfg):
        """Simulate synthetic pipeline with LOF+SAITS only (no MOMENT).

        This exactly reproduces the bug that caused the pipeline crash.
        """
        # Synthetic pipeline only runs LOF (outlier) + SAITS (imputation)
        runs = pd.DataFrame(
            {
                "tags.mlflow.runName": ["SAITS__LOF"],
                "metrics.test/f1": [0.50],  # ~0.5 expected for random synthetic data
            }
        )

        # This should NOT crash
        result = get_best_moment_variant(runs, best_metric_cfg, return_best_gt=False)

        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["tags.mlflow.runName"] == "SAITS__LOF"

    @pytest.mark.unit
    def test_graceful_degradation_with_no_runs(self, best_metric_cfg):
        """Pipeline should handle case of zero runs gracefully."""
        empty_df = pd.DataFrame()

        # All these should return empty but not crash
        result1 = get_best_moment(best_metric_cfg, empty_df)
        result2 = get_non_moment_models(empty_df)
        result3 = get_best_moment_variant(
            empty_df, best_metric_cfg, return_best_gt=True
        )

        assert result1 is None
        assert result2.empty
        assert result3.empty
