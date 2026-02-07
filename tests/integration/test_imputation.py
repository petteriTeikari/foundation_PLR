"""Integration tests for imputation functionality.

Tests imputation methods including MissForest on synthetic
and real data, verifying output quality and metrics computation.
"""

import numpy as np
import pandas as pd
import pytest


class TestMissForestImputation:
    """Tests for MissForest imputation method."""

    @pytest.fixture
    def missforest_available(self):
        """Check if MissForest is available."""
        try:
            from missforest import MissForest  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.fixture
    def data_with_missing(self):
        """Create synthetic data with missing values."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        # Create complete data
        data = np.random.randn(n_samples, n_features)

        # Add some structure (correlations)
        data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
        data[:, 2] = (
            data[:, 0] * 0.5 + data[:, 1] * 0.5 + np.random.randn(n_samples) * 0.2
        )

        # Store complete data for comparison
        data_complete = data.copy()

        # Introduce missing values (10% missing)
        mask = np.random.random((n_samples, n_features)) < 0.1
        data[mask] = np.nan

        # MissForest expects DataFrame
        columns = [f"feat_{i}" for i in range(n_features)]
        return (
            pd.DataFrame(data, columns=columns),
            pd.DataFrame(data_complete, columns=columns),
            mask,
        )

    @pytest.fixture
    def plr_data_with_missing(self, sample_plr_array):
        """Create PLR-like data with missing values."""
        data = sample_plr_array.copy()
        data_complete = data.copy()

        # Introduce 5% missing values
        np.random.seed(42)
        mask = np.random.random(data.shape) < 0.05
        data[mask] = np.nan

        return data, data_complete, mask

    @pytest.mark.integration
    def test_missforest_runs(self, missforest_available, data_with_missing):
        """Test that MissForest runs without errors."""
        if not missforest_available:
            pytest.skip("MissForest not installed")

        from missforest import MissForest

        data, _, _ = data_with_missing

        mf = MissForest(max_iter=3)
        imputed = mf.fit_transform(data)

        assert imputed is not None
        assert imputed.shape == data.shape

    @pytest.mark.integration
    def test_missforest_no_nans_output(self, missforest_available, data_with_missing):
        """Test that MissForest output contains no NaNs."""
        if not missforest_available:
            pytest.skip("MissForest not installed")

        from missforest import MissForest

        data, _, _ = data_with_missing

        # Verify input has NaNs
        assert np.any(np.isnan(data)), "Test data should have NaNs"

        mf = MissForest(max_iter=3)
        imputed = mf.fit_transform(data)

        # Output should have no NaNs
        assert not np.any(np.isnan(imputed)), "Imputed data should have no NaNs"

    @pytest.mark.integration
    def test_missforest_preserves_non_missing(
        self, missforest_available, data_with_missing
    ):
        """Test that MissForest preserves non-missing values."""
        if not missforest_available:
            pytest.skip("MissForest not installed")

        from missforest import MissForest

        data, data_complete, mask = data_with_missing

        mf = MissForest(max_iter=3)
        imputed = mf.fit_transform(data)

        # Check cell-by-cell: non-missing values should be unchanged
        # Use DataFrame index alignment (MissForest may reorder internally)
        for col in data.columns:
            col_mask = ~mask[:, data.columns.get_loc(col)]
            original = data_complete.loc[col_mask, col].values
            imputed_vals = imputed.loc[col_mask, col].values
            np.testing.assert_array_almost_equal(
                original,
                imputed_vals,
                decimal=5,
                err_msg=f"Non-missing values changed in column {col}",
            )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_missforest_imputation_quality(
        self, missforest_available, data_with_missing
    ):
        """Test that MissForest imputation is reasonably accurate."""
        if not missforest_available:
            pytest.skip("MissForest not installed")

        from missforest import MissForest

        data, data_complete, mask = data_with_missing

        mf = MissForest(max_iter=5)
        imputed = mf.fit_transform(data)

        # Compare imputed values to true values at missing positions
        imputed_at_missing = imputed.values[mask]
        true_at_missing = data_complete.values[mask]

        # Calculate MAE
        mae = np.mean(np.abs(imputed_at_missing - true_at_missing))

        # MAE should be reasonably low (data is standardized, so < 1 is good)
        assert mae < 1.5, f"Imputation MAE too high: {mae}"


class TestImputationMetrics:
    """Tests for imputation metrics computation."""

    @pytest.fixture
    def imputation_results(self):
        """Create mock imputation results."""
        np.random.seed(42)
        n_subjects = 8
        n_timepoints = 100
        n_features = 1

        # Ground truth
        targets = np.random.randn(n_subjects, n_timepoints, n_features)

        # Predictions with some error
        predictions = (
            targets + np.random.randn(n_subjects, n_timepoints, n_features) * 0.1
        )

        # Mask (where we had missing values)
        masks = (np.random.random((n_subjects, n_timepoints, n_features)) < 0.1).astype(
            int
        )

        return predictions, targets, masks

    @pytest.mark.integration
    def test_mse_computation(self, imputation_results):
        """Test MSE computation for imputation."""
        predictions, targets, masks = imputation_results

        # Compute MSE at masked positions
        mse_values = []
        for i in range(len(predictions)):
            mask_i = masks[i].astype(bool)
            if np.any(mask_i):
                pred_masked = predictions[i][mask_i]
                true_masked = targets[i][mask_i]
                mse = np.mean((pred_masked - true_masked) ** 2)
                mse_values.append(mse)

        avg_mse = np.mean(mse_values)

        # MSE should be positive and reasonable
        assert avg_mse > 0
        assert avg_mse < 1.0  # Since we added small noise

    @pytest.mark.integration
    def test_mae_computation(self, imputation_results):
        """Test MAE computation for imputation."""
        predictions, targets, masks = imputation_results

        # Compute MAE at masked positions
        masked_pred = predictions[masks.astype(bool)]
        masked_true = targets[masks.astype(bool)]

        mae = np.mean(np.abs(masked_pred - masked_true))

        assert mae > 0
        assert mae < 0.5  # Should be small given our noise level


class TestSimpleImputation:
    """Tests for simple imputation baselines."""

    @pytest.fixture
    def simple_missing_data(self):
        """Create simple data with known missing pattern."""
        data = np.array(
            [
                [1.0, 2.0, np.nan, 4.0, 5.0],
                [np.nan, 2.0, 3.0, 4.0, np.nan],
                [1.0, np.nan, 3.0, np.nan, 5.0],
            ]
        )
        return data

    @pytest.mark.integration
    def test_mean_imputation(self, simple_missing_data):
        """Test mean imputation baseline."""
        data = simple_missing_data.copy()

        # Compute column means ignoring NaN
        col_means = np.nanmean(data, axis=0)

        # Impute with column means
        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            data[mask, col] = col_means[col]

        assert not np.any(np.isnan(data))

    @pytest.mark.integration
    def test_interpolation_imputation(self, simple_missing_data):
        """Test linear interpolation imputation."""
        data = simple_missing_data.copy()

        # Interpolate each row
        for i in range(data.shape[0]):
            row = data[i, :]
            mask = np.isnan(row)
            if np.any(mask):
                x = np.arange(len(row))
                valid = ~mask
                row[mask] = np.interp(x[mask], x[valid], row[valid])

        assert not np.any(np.isnan(data))
