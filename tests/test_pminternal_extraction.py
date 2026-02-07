"""
TDD Tests for pminternal Bootstrap Prediction Extraction.

These tests are written BEFORE the implementation (TDD approach).
Reference: Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"

Expected data structure in MLflow:
- Path: /home/petteri/mlruns/253031330985650090/*/artifacts/metrics/*.pickle
- y_pred_proba shape: (n_subjects, n_bootstrap) = (63, 1000) for test split
"""

from pathlib import Path

import pytest

# Constants for validation
MLFLOW_BASE = Path("/home/petteri/mlruns/253031330985650090")

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not MLFLOW_BASE.exists(), reason="MLflow data not available"),
]
EXPECTED_N_SUBJECTS_TEST = 63  # Test split subjects
EXPECTED_N_BOOTSTRAP = 1000  # Bootstrap iterations


class TestMLflowDataAvailability:
    """Tests to verify MLflow data is available and accessible."""

    def test_mlflow_base_exists(self):
        """MLflow experiment directory exists."""
        assert MLFLOW_BASE.exists(), f"MLflow base not found: {MLFLOW_BASE}"

    def test_mlflow_has_runs(self):
        """At least one run exists in MLflow experiment."""
        runs = [d for d in MLFLOW_BASE.iterdir() if d.is_dir()]
        assert len(runs) > 0, "No runs found in MLflow experiment"


class TestGroundTruthConfigExists:
    """Tests to verify ground truth config can be found."""

    def test_ground_truth_pickle_exists(self):
        """Ground truth config (pupil-gt + pupil-gt + CatBoost + simple) exists."""
        gt_pattern = "metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
        gt_files = list(MLFLOW_BASE.glob(f"*/artifacts/metrics/{gt_pattern}"))
        assert (
            len(gt_files) >= 1
        ), f"Ground truth pickle not found. Pattern: {gt_pattern}"

    def test_ground_truth_has_correct_structure(self):
        """Ground truth pickle has expected data structure."""
        import pickle

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        # Check required keys
        assert "metrics_iter" in data, "Missing 'metrics_iter' key"
        assert "test" in data["metrics_iter"], "Missing 'test' split"
        assert "preds" in data["metrics_iter"]["test"], "Missing 'preds' in test"
        assert "arrays" in data["metrics_iter"]["test"]["preds"], "Missing 'arrays'"
        assert (
            "predictions" in data["metrics_iter"]["test"]["preds"]["arrays"]
        ), "Missing 'predictions'"
        assert (
            "y_pred_proba"
            in data["metrics_iter"]["test"]["preds"]["arrays"]["predictions"]
        ), "Missing 'y_pred_proba'"


class TestBootstrapPredictionsShape:
    """Tests for bootstrap predictions matrix shape."""

    @pytest.fixture
    def gt_predictions(self):
        """Load ground truth predictions."""
        import pickle

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        return data["metrics_iter"]["test"]["preds"]["arrays"]["predictions"][
            "y_pred_proba"
        ]

    def test_predictions_is_2d_array(self, gt_predictions):
        """Predictions matrix is 2-dimensional."""
        import numpy as np

        assert isinstance(gt_predictions, np.ndarray), "Predictions not a numpy array"
        assert gt_predictions.ndim == 2, f"Expected 2D, got {gt_predictions.ndim}D"

    def test_predictions_shape_subjects(self, gt_predictions):
        """Predictions has expected number of subjects (test split)."""
        n_subjects = gt_predictions.shape[0]
        assert (
            n_subjects == EXPECTED_N_SUBJECTS_TEST
        ), f"Expected {EXPECTED_N_SUBJECTS_TEST} subjects, got {n_subjects}"

    def test_predictions_shape_bootstrap(self, gt_predictions):
        """Predictions has expected number of bootstrap iterations."""
        n_bootstrap = gt_predictions.shape[1]
        assert (
            n_bootstrap == EXPECTED_N_BOOTSTRAP
        ), f"Expected {EXPECTED_N_BOOTSTRAP} bootstrap iterations, got {n_bootstrap}"


class TestPredictionsValidRange:
    """Tests for prediction value validity."""

    @pytest.fixture
    def gt_predictions(self):
        """Load ground truth predictions."""
        import pickle

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        return data["metrics_iter"]["test"]["preds"]["arrays"]["predictions"][
            "y_pred_proba"
        ]

    def test_predictions_min_valid(self, gt_predictions):
        """All predictions >= 0 (valid probabilities)."""
        assert (
            gt_predictions.min() >= 0
        ), f"Invalid min probability: {gt_predictions.min()}"

    def test_predictions_max_valid(self, gt_predictions):
        """All predictions <= 1 (valid probabilities)."""
        assert (
            gt_predictions.max() <= 1
        ), f"Invalid max probability: {gt_predictions.max()}"

    def test_predictions_no_nan(self, gt_predictions):
        """No NaN values in predictions."""
        import numpy as np

        assert not np.isnan(gt_predictions).any(), "NaN values found in predictions"


class TestLabelsAvailable:
    """Tests for ground truth labels availability."""

    def test_labels_exist(self):
        """Ground truth labels are available in subjectwise_stats."""
        import pickle

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        assert "subjectwise_stats" in data, "Missing 'subjectwise_stats'"
        assert (
            "test" in data["subjectwise_stats"]
        ), "Missing 'test' in subjectwise_stats"
        assert "labels" in data["subjectwise_stats"]["test"], "Missing 'labels'"

    def test_labels_shape_matches_predictions(self):
        """Labels array length matches number of subjects in predictions."""
        import pickle

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        preds = data["metrics_iter"]["test"]["preds"]["arrays"]["predictions"][
            "y_pred_proba"
        ]
        labels = data["subjectwise_stats"]["test"]["labels"]

        assert (
            len(labels) == preds.shape[0]
        ), f"Labels length {len(labels)} != predictions subjects {preds.shape[0]}"

    def test_labels_binary(self):
        """Labels are binary (0 or 1)."""
        import pickle
        import numpy as np

        gt_files = list(
            MLFLOW_BASE.glob(
                "*/artifacts/metrics/metrics_CATBOOST*simple*pupil-gt__pupil-gt.pickle"
            )
        )
        if not gt_files:
            pytest.skip("Ground truth file not found")

        with open(gt_files[0], "rb") as f:
            data = pickle.load(f)

        labels = data["subjectwise_stats"]["test"]["labels"]
        unique_labels = np.unique(labels)

        assert set(unique_labels).issubset(
            {0, 1}
        ), f"Non-binary labels found: {unique_labels}"
