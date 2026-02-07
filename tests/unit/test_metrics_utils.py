"""Unit tests for metrics utilities.

Tests functions from src/metrics/metrics_utils.py for array validation
and metrics dictionary initialization.
"""

import numpy as np
import pytest

from src.metrics.metrics_utils import (
    check_array_triplet,
    get_subjectwise_arrays,
    init_metrics_dict,
)


class TestCheckArrayTriplet:
    """Tests for check_array_triplet function."""

    @pytest.mark.unit
    def test_check_triplet_valid(self):
        """Test that matching 3D shapes pass validation."""
        n_subjects, n_timepoints, n_features = 8, 1981, 1
        predictions = np.random.randn(n_subjects, n_timepoints, n_features)
        targets = np.random.randn(n_subjects, n_timepoints, n_features)
        masks = np.random.randint(0, 2, (n_subjects, n_timepoints, n_features))

        # Should not raise
        check_array_triplet(predictions, targets, masks)

    @pytest.mark.unit
    def test_check_triplet_mismatch_subjects(self):
        """Test that mismatched subject count raises assertion."""
        predictions = np.random.randn(8, 1981, 1)
        targets = np.random.randn(10, 1981, 1)  # Different subject count
        masks = np.random.randint(0, 2, (8, 1981, 1))

        with pytest.raises(AssertionError):
            check_array_triplet(predictions, targets, masks)

    @pytest.mark.unit
    def test_check_triplet_mismatch_timepoints(self):
        """Test that mismatched timepoints raises assertion."""
        predictions = np.random.randn(8, 1981, 1)
        targets = np.random.randn(8, 2000, 1)  # Different timepoints
        masks = np.random.randint(0, 2, (8, 1981, 1))

        with pytest.raises(AssertionError):
            check_array_triplet(predictions, targets, masks)

    @pytest.mark.unit
    def test_check_triplet_mismatch_features(self):
        """Test that mismatched features raises assertion."""
        predictions = np.random.randn(8, 1981, 1)
        targets = np.random.randn(8, 1981, 2)  # Different features
        masks = np.random.randint(0, 2, (8, 1981, 1))

        with pytest.raises(AssertionError):
            check_array_triplet(predictions, targets, masks)

    @pytest.mark.unit
    def test_check_triplet_2d_raises_error(self):
        """Test that 2D input raises ValueError."""
        predictions = np.random.randn(8, 1981)  # 2D
        targets = np.random.randn(8, 1981)
        masks = np.random.randint(0, 2, (8, 1981))

        with pytest.raises(ValueError, match="2D"):
            check_array_triplet(predictions, targets, masks)

    @pytest.mark.unit
    def test_check_triplet_single_subject(self):
        """Test with single subject."""
        predictions = np.random.randn(1, 1981, 1)
        targets = np.random.randn(1, 1981, 1)
        masks = np.random.randint(0, 2, (1, 1981, 1))

        # Should not raise
        check_array_triplet(predictions, targets, masks)


class TestGetSubjectwiseArrays:
    """Tests for get_subjectwise_arrays function."""

    @pytest.mark.unit
    def test_get_subjectwise_extracts_single(self):
        """Test extraction of single subject arrays."""
        n_subjects = 8
        predictions = np.arange(n_subjects * 100 * 1).reshape(n_subjects, 100, 1)
        targets = predictions * 2
        masks = np.ones_like(predictions)

        pred_i, target_i, mask_i = get_subjectwise_arrays(
            predictions, targets, masks, i=0
        )

        # Should have shape (1, timepoints, features)
        assert pred_i.shape == (1, 100, 1)
        assert target_i.shape == (1, 100, 1)
        assert mask_i.shape == (1, 100, 1)

    @pytest.mark.unit
    def test_get_subjectwise_correct_data(self):
        """Test that correct subject data is extracted."""
        n_subjects = 4
        # Create data where each subject has unique values
        predictions = np.zeros((n_subjects, 10, 1))
        for i in range(n_subjects):
            predictions[i, :, :] = i * 10

        targets = predictions.copy()
        masks = np.ones_like(predictions)

        for i in range(n_subjects):
            pred_i, _, _ = get_subjectwise_arrays(predictions, targets, masks, i=i)
            # Subject i should have values i*10
            assert np.all(pred_i == i * 10)

    @pytest.mark.unit
    def test_get_subjectwise_preserves_values(self):
        """Test that values are preserved correctly."""
        predictions = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        targets = predictions * 2
        masks = np.ones_like(predictions)

        pred_0, target_0, mask_0 = get_subjectwise_arrays(
            predictions, targets, masks, i=0
        )

        np.testing.assert_array_equal(pred_0, [[[1], [2], [3]]])
        np.testing.assert_array_equal(target_0, [[[2], [4], [6]]])

    @pytest.mark.unit
    def test_get_subjectwise_last_subject(self):
        """Test extraction of last subject."""
        n_subjects = 5
        predictions = np.arange(n_subjects * 10 * 1).reshape(n_subjects, 10, 1)
        targets = predictions.copy()
        masks = np.ones_like(predictions)

        pred_last, _, _ = get_subjectwise_arrays(
            predictions, targets, masks, i=n_subjects - 1
        )

        expected = predictions[n_subjects - 1 : n_subjects, :, :]
        np.testing.assert_array_equal(pred_last, expected)


class TestInitMetricsDict:
    """Tests for init_metrics_dict function."""

    @pytest.mark.unit
    def test_init_metrics_dict_structure(self):
        """Test that metrics dict has correct structure."""
        metrics = init_metrics_dict()

        assert "scalars" in metrics
        assert "arrays" in metrics
        assert "arrays_flat" in metrics

    @pytest.mark.unit
    def test_init_metrics_dict_scalars_is_dict(self):
        """Test that scalars is an empty dict."""
        metrics = init_metrics_dict()

        assert isinstance(metrics["scalars"], dict)
        assert len(metrics["scalars"]) == 0

    @pytest.mark.unit
    def test_init_metrics_dict_arrays_is_dict(self):
        """Test that arrays is an empty dict."""
        metrics = init_metrics_dict()

        assert isinstance(metrics["arrays"], dict)
        assert len(metrics["arrays"]) == 0

    @pytest.mark.unit
    def test_init_metrics_dict_arrays_flat_is_list(self):
        """Test that arrays_flat is an empty list."""
        metrics = init_metrics_dict()

        assert isinstance(metrics["arrays_flat"], list)
        assert len(metrics["arrays_flat"]) == 0

    @pytest.mark.unit
    def test_init_metrics_dict_independent_instances(self):
        """Test that multiple calls return independent instances."""
        metrics1 = init_metrics_dict()
        metrics2 = init_metrics_dict()

        metrics1["scalars"]["test"] = 1.0

        assert "test" not in metrics2["scalars"]
