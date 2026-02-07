# tests/test_data_quality/test_unified_normalization.py
"""
TDD tests for unified normalization system.

The unified normalization system should:
1. Maintain single scaler instance across pipeline
2. Track normalization state per array
3. Prevent double normalization/denormalization
4. Provide clear API for forward/backward transforms

Run: pytest tests/test_data_quality/test_unified_normalization.py -v
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestNormalizationManagerBasics:
    """Test basic NormalizationManager functionality."""

    def test_import_normalization_manager(self):
        """NormalizationManager should be importable."""
        from src.preprocess.normalization_manager import NormalizationManager

        assert NormalizationManager is not None

    def test_create_manager_with_data(self):
        """Should create manager and fit on data."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        manager = NormalizationManager()
        manager.fit(data)

        assert manager.is_fitted
        assert manager.mean is not None
        assert manager.std is not None

    def test_forward_transform(self):
        """Forward transform should standardize data."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        manager = NormalizationManager()
        manager.fit(data)

        transformed = manager.transform(data)

        # Z-score standardization: (x - mean) / std
        expected_mean = 0.0
        expected_std = 1.0

        assert np.isclose(np.mean(transformed), expected_mean, atol=1e-10)
        assert np.isclose(np.std(transformed), expected_std, atol=1e-10)

    def test_backward_transform(self):
        """Backward transform should restore original scale."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        manager = NormalizationManager()
        manager.fit(data)

        transformed = manager.transform(data)
        restored = manager.inverse_transform(transformed)

        np.testing.assert_allclose(restored, data, rtol=1e-10)

    def test_transform_new_data(self):
        """Should transform new data using fitted parameters."""
        from src.preprocess.normalization_manager import NormalizationManager

        train_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        test_data = np.array([15.0, 25.0, 35.0])

        manager = NormalizationManager()
        manager.fit(train_data)

        # Transform test data using training stats
        transformed = manager.transform(test_data)
        restored = manager.inverse_transform(transformed)

        np.testing.assert_allclose(restored, test_data, rtol=1e-10)


class TestStateTracking:
    """Test normalization state tracking to prevent double transforms."""

    def test_track_normalization_state(self):
        """Should track whether data has been normalized."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, 20.0, 30.0])
        manager = NormalizationManager()
        manager.fit(data)

        # Transform with tracking - assigns data_id
        transformed, data_id = manager.transform_tracked(data)

        # After transform, should be marked normalized
        assert manager.is_normalized(data_id)

        # After inverse transform, should be marked denormalized (raw)
        restored, _ = manager.inverse_transform_tracked(transformed, data_id)
        assert not manager.is_normalized(data_id)

    def test_prevent_double_normalization(self):
        """Should raise error on double normalization."""
        from src.preprocess.normalization_manager import (
            NormalizationManager,
            DoubleNormalizationError,
        )

        data = np.array([10.0, 20.0, 30.0])
        manager = NormalizationManager()
        manager.fit(data)

        # First transform OK
        transformed, data_id = manager.transform_tracked(data)

        # Second transform should raise
        with pytest.raises(DoubleNormalizationError):
            manager.transform_tracked(transformed, data_id)

    def test_prevent_double_denormalization(self):
        """Should raise error on double denormalization."""
        from src.preprocess.normalization_manager import (
            NormalizationManager,
            DoubleDenormalizationError,
        )

        data = np.array([10.0, 20.0, 30.0])
        manager = NormalizationManager()
        manager.fit(data)

        # Transform, then inverse
        transformed, data_id = manager.transform_tracked(data)
        restored, _ = manager.inverse_transform_tracked(transformed, data_id)

        # Second inverse should raise
        with pytest.raises(DoubleDenormalizationError):
            manager.inverse_transform_tracked(restored, data_id)


class TestNaNHandling:
    """Test handling of NaN values in normalization."""

    def test_fit_ignores_nan(self):
        """Fit should compute stats ignoring NaN values."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
        manager = NormalizationManager()
        manager.fit(data)

        # Mean and std should be computed from non-NaN values only
        expected_mean = np.nanmean(data)
        expected_std = np.nanstd(data)

        assert np.isclose(manager.mean, expected_mean)
        assert np.isclose(manager.std, expected_std)

    def test_transform_preserves_nan(self):
        """Transform should preserve NaN positions."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, np.nan, 30.0])
        manager = NormalizationManager()
        manager.fit(data)

        transformed = manager.transform(data)

        assert np.isnan(transformed[1])
        assert not np.isnan(transformed[0])
        assert not np.isnan(transformed[2])


class TestSerialization:
    """Test saving and loading normalization parameters."""

    def test_to_dict(self):
        """Should serialize parameters to dictionary."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([10.0, 20.0, 30.0])
        manager = NormalizationManager()
        manager.fit(data)

        params = manager.to_dict()

        assert "mean" in params
        assert "std" in params
        assert "is_fitted" in params

    def test_from_dict(self):
        """Should restore manager from dictionary."""
        from src.preprocess.normalization_manager import NormalizationManager

        original_data = np.array([10.0, 20.0, 30.0])
        manager1 = NormalizationManager()
        manager1.fit(original_data)

        params = manager1.to_dict()

        manager2 = NormalizationManager.from_dict(params)

        # Both managers should produce same transforms
        test_data = np.array([15.0, 25.0])
        t1 = manager1.transform(test_data)
        t2 = manager2.transform(test_data)

        np.testing.assert_allclose(t1, t2)


class TestIntegrationWithExisting:
    """Test integration with existing preprocessing functions."""

    def test_compatible_with_standardize_data_dict(self):
        """Manager should produce same results as existing standardize functions."""
        from src.preprocess.normalization_manager import NormalizationManager

        # Simulate the existing standardization behavior
        data = np.array([-20.0, -30.0, -40.0, -10.0, 0.0])

        # Using manager
        manager = NormalizationManager()
        manager.fit(data)
        transformed = manager.transform(data)

        # Direct computation (existing method)
        mean = np.nanmean(data)
        std = np.nanstd(data)
        expected = (data - mean) / std

        np.testing.assert_allclose(transformed, expected)

    def test_manager_uses_ddof_zero(self):
        """Manager should use ddof=0 for std (population std, matching numpy default)."""
        from src.preprocess.normalization_manager import NormalizationManager

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        manager = NormalizationManager()
        manager.fit(data)

        # numpy default is ddof=0
        expected_std = np.std(data, ddof=0)
        assert np.isclose(manager.std, expected_std)
