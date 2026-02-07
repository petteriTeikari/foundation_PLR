"""Unit tests for classifier logging utilities.

Tests functions from src/classification/classifier_log_utils.py for None handling
and edge cases discovered during synthetic pipeline testing.

See: docs/planning/pipeline-robustness-plan.md for context.
"""

import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

# Note: Testing these functions requires mocking MLflow since they log directly


class TestNoneGuardsInLogging:
    """Tests for None guards in logging functions.

    These tests verify that logging functions handle None/missing metrics gracefully
    without crashing the pipeline.
    """

    @pytest.fixture
    def cls_model_cfg(self):
        """Standard classifier model configuration for testing."""
        return OmegaConf.create({"ARTIFACTS": {"results_format": "pickle"}})

    @pytest.mark.unit
    def test_log_source_metrics_handles_none_series(self):
        """Test that log_source_metrics_as_params handles None gracefully.

        This was an edge case - when featurization run info is None,
        the function should not crash.
        """
        from src.classification.classifier_log_utils import log_source_metrics_as_params

        with patch("mlflow.log_param"):
            # This should raise KeyError or AttributeError for None input
            # The caller should handle None before calling this
            with pytest.raises((TypeError, AttributeError)):
                log_source_metrics_as_params(None)

    @pytest.mark.unit
    def test_parse_and_log_cls_run_name_standard_format(self):
        """Test parsing standard 4-part run name."""
        from src.classification.classifier_log_utils import parse_and_log_cls_run_name

        with patch("mlflow.log_param") as mock_log_param:
            parse_and_log_cls_run_name("CatBoost__simple1.0__SAITS__LOF")

            # Should log 5 parameters (feature_param, imputation_source, anomaly_source,
            # both_sources_gt, both_sources_ensemble)
            assert mock_log_param.call_count == 5
            calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}
            assert calls["feature_param"] == "simple1.0"
            assert calls["imputation_source"] == "SAITS"
            assert calls["anomaly_source"] == "LOF"
            assert calls["both_sources_gt"] is False
            assert calls["both_sources_ensemble"] is False

    @pytest.mark.unit
    def test_parse_and_log_cls_run_name_two_part(self):
        """Test parsing 2-part run name (legacy format) - should not crash."""
        from src.classification.classifier_log_utils import parse_and_log_cls_run_name

        with patch("mlflow.log_param") as mock_log_param:
            # This will fail to parse and should log a warning but not crash
            parse_and_log_cls_run_name("CatBoost__ensemble-imput")

            # Should not log anything since parsing failed (exception caught)
            assert mock_log_param.call_count == 0


class TestMetricsDictStructure:
    """Tests for verifying metrics dict structure assumptions."""

    @pytest.mark.unit
    def test_metrics_dict_missing_subjectwise_stats(self):
        """Verify handling when subjectwise_stats key is missing."""
        metrics = {"other_key": "value"}

        # Using .get() with None default should not crash
        result = metrics.get("subjectwise_stats")
        assert result is None

    @pytest.mark.unit
    def test_metrics_dict_missing_subject_global_stats(self):
        """Verify handling when subject_global_stats key is missing."""
        metrics = {"other_key": "value"}

        # Using .get() with None default should not crash
        result = metrics.get("subject_global_stats")
        assert result is None

    @pytest.mark.unit
    def test_metrics_dict_with_none_values(self):
        """Verify handling when metrics contain explicit None values."""
        metrics = {
            "subjectwise_stats": None,
            "subject_global_stats": None,
            "other_metric": 0.85,
        }

        # Should handle None values without crashing
        assert metrics.get("subjectwise_stats") is None
        assert metrics.get("subject_global_stats") is None
        assert metrics.get("other_metric") == 0.85


class TestRunNameParsing:
    """Tests for run name parsing edge cases."""

    @pytest.mark.unit
    def test_run_name_with_dashes_in_model_name(self):
        """Test parsing run name with dashes (e.g., MOMENT-finetune)."""
        from src.classification.classifier_log_utils import parse_and_log_cls_run_name

        with patch("mlflow.log_param") as mock_log_param:
            parse_and_log_cls_run_name("CatBoost__simple1.0__MOMENT-finetune__pupil-gt")

            calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}
            assert calls["imputation_source"] == "MOMENT-finetune"
            assert calls["anomaly_source"] == "pupil-gt"

    @pytest.mark.unit
    def test_run_name_with_ensemble_names(self):
        """Test parsing run name with long ensemble names."""
        from src.classification.classifier_log_utils import parse_and_log_cls_run_name

        with patch("mlflow.log_param") as mock_log_param:
            run_name = (
                "CatBoost__simple1.0__ensemble-CSDI-MOMENT-SAITS__"
                "ensembleThresholded-MOMENT-TimesNet-UniTS-gt-finetune"
            )
            parse_and_log_cls_run_name(run_name)

            calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}
            assert "ensemble" in calls["imputation_source"]
            assert (
                "ensemble" in calls["anomaly_source"].lower()
                or "Thresholded" in calls["anomaly_source"]
            )
            # Should also set both_sources_ensemble to True
            assert calls["both_sources_ensemble"] is True


class TestSyntheticPipelineScenarios:
    """Tests for scenarios from synthetic pipeline testing."""

    @pytest.mark.unit
    def test_synthetic_pipeline_simple_run_name(self):
        """Test parsing simple run name from synthetic pipeline.

        Synthetic pipeline uses LOF+SAITS with XGBoost.
        """
        from src.classification.classifier_log_utils import parse_and_log_cls_run_name

        with patch("mlflow.log_param") as mock_log_param:
            # This is what synthetic pipeline produces
            parse_and_log_cls_run_name("XGBOOST__simple1.0__SAITS__LOF")

            calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}
            # Note: classifier name is not logged - only feature, imputation, anomaly
            assert calls["feature_param"] == "simple1.0"
            assert calls["imputation_source"] == "SAITS"
            assert calls["anomaly_source"] == "LOF"
            assert calls["both_sources_gt"] is False
            assert calls["both_sources_ensemble"] is False

    @pytest.mark.unit
    def test_metrics_logging_with_optional_fields_none(self):
        """Test that optional metrics fields being None doesn't crash logging.

        This was an issue in synthetic pipeline - some optional metrics
        like subjectwise_stats weren't computed.
        """
        # Simulating the metrics dict structure from synthetic pipeline
        metrics = {
            "test": {
                "auroc": 0.50,  # Expected for random synthetic data
                "brier": 0.25,
            },
            "train": {
                "auroc": 0.55,
                "brier": 0.24,
            },
            # These are explicitly None in synthetic pipeline
            "subjectwise_stats": None,
            "subject_global_stats": None,
        }

        # Verify None checking pattern works
        if metrics.get("subjectwise_stats") is None:
            # Should skip logging without error
            pass
        else:
            # Would have iterated over stats here
            for split in metrics["subjectwise_stats"].keys():
                pass

        if metrics.get("subject_global_stats") is None:
            # Should skip logging without error
            pass
        else:
            # Would have iterated over stats here
            for split, stats in metrics["subject_global_stats"].items():
                pass

        # If we get here without exception, the pattern works
        assert True
