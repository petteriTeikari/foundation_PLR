"""
TDD Tests for PLR decomposition methods.

Tests verify:
1. Each method produces valid outputs
2. Components reconstruct original waveform (where applicable)
3. Expected number of components
4. Transformation can be saved and applied to new data
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# We'll import from the module once created
try:
    from src.decomposition import (
        TemplateFitting,
        StandardPCA,
        RotatedPCA,
        SparsePCADecomposition,
        GEDDecomposition,
    )

    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False


# Create synthetic PLR-like data for testing
@pytest.fixture
def synthetic_plr_data():
    """Generate synthetic PLR waveforms for testing."""
    np.random.seed(42)
    n_subjects = 50
    n_timepoints = 200

    # Time vector: 0-66 seconds
    time_vector = np.linspace(0, 66, n_timepoints)

    # Create synthetic PLR pattern
    # Baseline period
    baseline = 100  # mm baseline pupil size

    # Blue stimulus response (15.5-24.5s)
    blue_response = np.zeros(n_timepoints)
    blue_idx = (time_vector >= 15.5) & (time_vector <= 40.0)
    t_blue = time_vector[blue_idx] - 15.5
    blue_response[blue_idx] = -15 * (1 - np.exp(-t_blue / 2))

    # Red stimulus response (46.5-55.5s)
    red_response = np.zeros(n_timepoints)
    red_idx = time_vector >= 46.5
    t_red = time_vector[red_idx] - 46.5
    red_response[red_idx] = -10 * (1 - np.exp(-t_red / 2))

    # Mean waveform
    mean_wave = baseline + blue_response + red_response

    # Generate subjects with variation
    waveforms = np.zeros((n_subjects, n_timepoints))
    for i in range(n_subjects):
        # Individual variation in amplitude
        amp_var = 1 + 0.2 * np.random.randn()
        noise = 2 * np.random.randn(n_timepoints)
        waveforms[i] = mean_wave * amp_var + noise

    return {
        "waveforms": waveforms,
        "time_vector": time_vector,
        "n_subjects": n_subjects,
        "n_timepoints": n_timepoints,
    }


@pytest.mark.skipif(
    not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
)
class TestTemplateFitting:
    """Tests for Template Fitting decomposition."""

    def test_fit_returns_result(self, synthetic_plr_data):
        """Template fitting returns valid result."""
        tf = TemplateFitting()
        waveform = synthetic_plr_data["waveforms"][0]
        time_vector = synthetic_plr_data["time_vector"]

        result = tf.fit(waveform, time_vector)

        assert hasattr(result, "phasic")
        assert hasattr(result, "sustained")
        assert hasattr(result, "pipr")
        assert hasattr(result, "rmse")

    def test_components_same_length_as_input(self, synthetic_plr_data):
        """Components have same length as input waveform."""
        tf = TemplateFitting()
        waveform = synthetic_plr_data["waveforms"][0]
        time_vector = synthetic_plr_data["time_vector"]

        result = tf.fit(waveform, time_vector)

        assert len(result.phasic) == len(waveform)
        assert len(result.sustained) == len(waveform)
        assert len(result.pipr) == len(waveform)

    def test_reconstruction_error_reasonable(self, synthetic_plr_data):
        """Reconstruction RMSE is reasonable (< 20% of signal range)."""
        tf = TemplateFitting()
        waveform = synthetic_plr_data["waveforms"][0]
        time_vector = synthetic_plr_data["time_vector"]

        result = tf.fit(waveform, time_vector)

        signal_range = waveform.max() - waveform.min()
        assert result.rmse < 0.2 * signal_range, f"RMSE {result.rmse} too high"

    def test_batch_fit(self, synthetic_plr_data):
        """Batch fitting works for multiple subjects."""
        tf = TemplateFitting()
        waveforms = synthetic_plr_data["waveforms"][:10]
        time_vector = synthetic_plr_data["time_vector"]

        results = tf.fit_batch(waveforms, time_vector)

        assert len(results) == 10
        assert all(hasattr(r, "phasic") for r in results)


@pytest.mark.skipif(
    not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
)
class TestStandardPCA:
    """Tests for Standard PCA decomposition."""

    def test_fit_transform_returns_result(self, synthetic_plr_data):
        """PCA returns valid result with expected components."""
        pca = StandardPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        result = pca.fit_transform(waveforms)

        assert result.n_components == 3
        assert result.loadings.shape == (3, synthetic_plr_data["n_timepoints"])
        assert result.scores.shape == (synthetic_plr_data["n_subjects"], 3)

    def test_explained_variance_sums_to_less_than_one(self, synthetic_plr_data):
        """Explained variance ratios are valid (sum <= 1)."""
        pca = StandardPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        result = pca.fit_transform(waveforms)

        assert result.explained_variance.sum() <= 1.0
        assert all(result.explained_variance >= 0)
        assert all(result.explained_variance <= 1)

    def test_reconstruction(self, synthetic_plr_data):
        """PCA reconstruction approximates original."""
        pca = StandardPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        pca.fit(waveforms)
        result = pca.transform(waveforms)
        reconstructed = pca.inverse_transform(result.scores)

        # With 3 components, reconstruction should be close
        rmse = np.sqrt(np.mean((waveforms - reconstructed) ** 2))
        signal_std = waveforms.std()
        assert rmse < 0.5 * signal_std, f"Reconstruction RMSE {rmse} too high"


@pytest.mark.skipif(
    not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
)
class TestRotatedPCA:
    """Tests for Rotated PCA (Promax) decomposition."""

    def test_fit_transform_returns_result(self, synthetic_plr_data):
        """Rotated PCA returns valid result."""
        rpca = RotatedPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        result = rpca.fit_transform(waveforms)

        assert result.n_components == 3
        assert result.loadings.shape == (3, synthetic_plr_data["n_timepoints"])
        assert result.scores.shape == (synthetic_plr_data["n_subjects"], 3)

    def test_rotation_matrix_exists(self, synthetic_plr_data):
        """Rotation matrix is computed and valid."""
        rpca = RotatedPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        result = rpca.fit_transform(waveforms)

        assert result.rotation_matrix.shape == (3, 3)

    def test_factor_correlation_computed(self, synthetic_plr_data):
        """Factor correlation matrix is computed."""
        rpca = RotatedPCA(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]

        result = rpca.fit_transform(waveforms)

        assert result.factor_correlation.shape == (3, 3)
        # Diagonal should be close to 1
        assert np.allclose(np.diag(result.factor_correlation), 1.0, atol=0.1)


@pytest.mark.skipif(
    not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
)
class TestSparsePCA:
    """Tests for Sparse PCA decomposition."""

    def test_fit_transform_returns_result(self, synthetic_plr_data):
        """Sparse PCA returns valid result."""
        spca = SparsePCADecomposition(n_components=3, alpha=1.0)
        waveforms = synthetic_plr_data["waveforms"]

        result = spca.fit_transform(waveforms)

        assert result.n_components == 3
        assert result.loadings.shape == (3, synthetic_plr_data["n_timepoints"])
        assert result.scores.shape == (synthetic_plr_data["n_subjects"], 3)

    def test_sparsity_computed(self, synthetic_plr_data):
        """Sparsity measure is computed for each component."""
        spca = SparsePCADecomposition(n_components=3, alpha=1.0)
        waveforms = synthetic_plr_data["waveforms"]

        result = spca.fit_transform(waveforms)

        assert len(result.sparsity) == 3
        # Sparsity should be between 0 and 1
        assert all(0 <= s <= 1 for s in result.sparsity)


@pytest.mark.skipif(
    not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
)
class TestGED:
    """Tests for GED decomposition."""

    def test_fit_transform_returns_result(self, synthetic_plr_data):
        """GED returns valid result."""
        ged = GEDDecomposition(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]
        time_vector = synthetic_plr_data["time_vector"]

        result = ged.fit_transform(waveforms, time_vector)

        assert result.n_components == 3
        assert result.components.shape == (3, synthetic_plr_data["n_timepoints"])
        assert result.scores.shape == (synthetic_plr_data["n_subjects"], 3)

    def test_eigenvalues_positive(self, synthetic_plr_data):
        """GED eigenvalues should be positive (contrast ratios)."""
        ged = GEDDecomposition(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]
        time_vector = synthetic_plr_data["time_vector"]

        result = ged.fit_transform(waveforms, time_vector)

        assert all(result.eigenvalues > 0), "GED eigenvalues should be positive"

    def test_eigenvalues_sorted_descending(self, synthetic_plr_data):
        """GED eigenvalues should be sorted descending."""
        ged = GEDDecomposition(n_components=3)
        waveforms = synthetic_plr_data["waveforms"]
        time_vector = synthetic_plr_data["time_vector"]

        result = ged.fit_transform(waveforms, time_vector)

        # Check descending order
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] >= result.eigenvalues[i + 1]


class TestTransformationPersistence:
    """Tests that transformations can be saved and reapplied."""

    @pytest.mark.skipif(
        not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
    )
    def test_pca_can_transform_new_data(self, synthetic_plr_data):
        """PCA fitted on training data can transform test data."""
        pca = StandardPCA(n_components=3)

        # Split data
        train = synthetic_plr_data["waveforms"][:40]
        test = synthetic_plr_data["waveforms"][40:]

        # Fit on train
        pca.fit(train)

        # Transform test
        result = pca.transform(test)

        assert result.scores.shape == (10, 3)

    @pytest.mark.skipif(
        not DECOMPOSITION_AVAILABLE, reason="Decomposition module not available"
    )
    def test_template_fitting_uses_same_templates(self, synthetic_plr_data):
        """Template fitting uses consistent templates across subjects."""
        tf = TemplateFitting()
        time_vector = synthetic_plr_data["time_vector"]

        # Fit two different subjects
        result1 = tf.fit(synthetic_plr_data["waveforms"][0], time_vector)
        result2 = tf.fit(synthetic_plr_data["waveforms"][1], time_vector)

        # Templates (normalized) should be the same
        # But scaled components will differ based on amplitude
        phasic1_norm = result1.phasic / (np.abs(result1.phasic).max() + 1e-10)
        phasic2_norm = result2.phasic / (np.abs(result2.phasic).max() + 1e-10)

        # Correlation should be high (same shape)
        corr = np.corrcoef(phasic1_norm, phasic2_norm)[0, 1]
        assert abs(corr) > 0.99, f"Template shape should be consistent, got corr={corr}"
