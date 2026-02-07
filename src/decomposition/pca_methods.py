"""PCA-based Decomposition Methods for PLR Waveforms.

Provides three PCA variants:
1. Standard PCA - Orthogonal principal components
2. Rotated PCA (Promax) - Oblique rotation for correlated components
3. Sparse PCA - L1-penalized for interpretable sparse loadings

Reference: Bustos 2024 "Pupillary Manifolds", Zou 2006 "Sparse PCA"
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA, SparsePCA


@dataclass
class PCAResult:
    """Result from PCA-based decomposition."""

    loadings: NDArray[np.float64]  # Component loadings (n_components, n_timepoints)
    scores: NDArray[np.float64]  # Subject scores (n_subjects, n_components)
    explained_variance: NDArray[np.float64]  # Variance explained per component
    mean: NDArray[np.float64]  # Mean waveform
    n_components: int


class StandardPCA:
    """Standard Principal Component Analysis for PLR waveforms.

    Extracts orthogonal directions of maximum variance.

    Parameters
    ----------
    n_components : int
        Number of components to extract (default: 3)
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._pca = PCA(n_components=n_components)
        self._fitted = False

    def fit(self, waveforms: NDArray[np.float64]) -> "StandardPCA":
        """Fit PCA to waveforms.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)

        Returns
        -------
        self
        """
        self._pca.fit(waveforms)
        self._mean = waveforms.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, waveforms: NDArray[np.float64]) -> PCAResult:
        """Transform waveforms to component scores.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)

        Returns
        -------
        PCAResult
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        scores = self._pca.transform(waveforms)

        return PCAResult(
            loadings=self._pca.components_,
            scores=scores,
            explained_variance=self._pca.explained_variance_ratio_,
            mean=self._mean,
            n_components=self.n_components,
        )

    def fit_transform(self, waveforms: NDArray[np.float64]) -> PCAResult:
        """Fit and transform in one step."""
        self.fit(waveforms)
        return self.transform(waveforms)

    def inverse_transform(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct waveforms from scores."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before inverse_transform()")
        return self._pca.inverse_transform(scores)


@dataclass
class RotatedPCAResult:
    """Result from rotated PCA decomposition."""

    loadings: NDArray[np.float64]  # Rotated loadings (n_components, n_timepoints)
    scores: NDArray[np.float64]  # Rotated scores (n_subjects, n_components)
    explained_variance: NDArray[np.float64]  # Variance before rotation
    rotation_matrix: NDArray[np.float64]  # Rotation matrix
    factor_correlation: NDArray[np.float64]  # Correlation between factors
    mean: NDArray[np.float64]
    n_components: int


class RotatedPCA:
    """Rotated PCA with Promax oblique rotation.

    Allows correlated components, which is more physiologically realistic
    for PLR where transient, sustained, and PIPR share autonomic origins.

    Parameters
    ----------
    n_components : int
        Number of components (default: 3)
    power : float
        Promax power parameter (default: 4, higher = more oblique)

    Reference: Bustos 2024 "Pupillary Manifolds"
    """

    def __init__(self, n_components: int = 3, power: float = 4.0):
        self.n_components = n_components
        self.power = power
        self._pca = PCA(n_components=n_components)
        self._fitted = False

    def _promax_rotation(
        self, loadings: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply Promax oblique rotation to loadings.

        Promax is a two-step process:
        1. Varimax rotation (orthogonal)
        2. Oblique rotation toward simple structure
        """
        # Step 1: Varimax rotation
        varimax_loadings, varimax_rotation = self._varimax_rotation(loadings)

        # Step 2: Promax (oblique) rotation
        # Target matrix: raise varimax loadings to power
        target = np.sign(varimax_loadings) * np.abs(varimax_loadings) ** self.power

        # Normalize target rows
        target = target / np.sqrt((target**2).sum(axis=1, keepdims=True) + 1e-10)

        # Find rotation that maps varimax to target via least squares
        # L_promax = L_varimax @ T where T minimizes ||L_varimax @ T - Target||
        rotation, _, _, _ = np.linalg.lstsq(varimax_loadings, target, rcond=None)

        # Normalize rotation matrix columns
        rotation = rotation / np.sqrt((rotation**2).sum(axis=0, keepdims=True) + 1e-10)

        promax_loadings = varimax_loadings @ rotation

        return promax_loadings, rotation

    def _varimax_rotation(
        self, loadings: NDArray[np.float64], max_iter: int = 100, tol: float = 1e-6
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply Varimax orthogonal rotation."""
        n_features, n_components = loadings.shape
        rotation = np.eye(n_components)
        rotated = loadings.copy()

        for _ in range(max_iter):
            old_rotated = rotated.copy()

            for i in range(n_components):
                for j in range(i + 1, n_components):
                    # Compute optimal angle for (i,j) pair
                    x = rotated[:, i]
                    y = rotated[:, j]

                    u = x**2 - y**2
                    v = 2 * x * y

                    A = u.sum()
                    B = v.sum()
                    C = (u**2 - v**2).sum()
                    D = (2 * u * v).sum()

                    num = D - 2 * A * B / n_features
                    denom = C - (A**2 - B**2) / n_features

                    phi = 0.25 * np.arctan2(num, denom)

                    # Apply rotation
                    c, s = np.cos(phi), np.sin(phi)
                    rotated[:, [i, j]] = rotated[:, [i, j]] @ np.array(
                        [[c, -s], [s, c]]
                    )

                    # Update rotation matrix
                    rotation[:, [i, j]] = rotation[:, [i, j]] @ np.array(
                        [[c, -s], [s, c]]
                    )

            # Check convergence
            if np.abs(rotated - old_rotated).max() < tol:
                break

        return rotated, rotation

    def fit(self, waveforms: NDArray[np.float64]) -> "RotatedPCA":
        """Fit rotated PCA to waveforms."""
        # First fit standard PCA
        self._pca.fit(waveforms)
        self._mean = waveforms.mean(axis=0)

        # Get initial loadings (transpose to match factor analysis convention)
        initial_loadings = self._pca.components_.T  # (n_timepoints, n_components)

        # Apply Promax rotation
        self._rotated_loadings, self._rotation = self._promax_rotation(initial_loadings)

        # Compute factor correlation matrix (for oblique rotation)
        self._factor_correlation = self._rotation.T @ self._rotation

        self._fitted = True
        return self

    def transform(self, waveforms: NDArray[np.float64]) -> RotatedPCAResult:
        """Transform waveforms to rotated component scores."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Get PCA scores and rotate them
        pca_scores = self._pca.transform(waveforms)
        rotated_scores = pca_scores @ self._rotation

        return RotatedPCAResult(
            loadings=self._rotated_loadings.T,  # (n_components, n_timepoints)
            scores=rotated_scores,
            explained_variance=self._pca.explained_variance_ratio_,
            rotation_matrix=self._rotation,
            factor_correlation=self._factor_correlation,
            mean=self._mean,
            n_components=self.n_components,
        )

    def fit_transform(self, waveforms: NDArray[np.float64]) -> RotatedPCAResult:
        """Fit and transform in one step."""
        self.fit(waveforms)
        return self.transform(waveforms)


@dataclass
class SparsePCAResult:
    """Result from Sparse PCA decomposition."""

    loadings: NDArray[np.float64]  # Sparse component loadings
    scores: NDArray[np.float64]  # Subject scores
    n_components: int
    mean: NDArray[np.float64]
    sparsity: list[float]  # Fraction of zeros per component


class SparsePCADecomposition:
    """Sparse PCA for interpretable PLR decomposition.

    Applies L1 penalty to encourage sparse loadings, making each
    component driven by a subset of timepoints.

    Parameters
    ----------
    n_components : int
        Number of components (default: 3)
    alpha : float
        Sparsity controlling parameter (default: 1.0)

    Reference: Zou 2006 "Sparse Principal Component Analysis"
    """

    def __init__(self, n_components: int = 3, alpha: float = 1.0, max_iter: int = 50):
        """Initialize Sparse PCA.

        Parameters
        ----------
        n_components : int
            Number of components (default: 3)
        alpha : float
            Sparsity controlling parameter (default: 1.0)
        max_iter : int
            Maximum iterations for optimization (default: 100).
            Lower values = faster but potentially less accurate.
            Original sklearn default is 1000, reduced for speed.
        """
        self.n_components = n_components
        self.alpha = alpha
        # Note: SparsePCA is inherently slow due to iterative optimization.
        # For 1981 timepoints, max_iter=50 with CD method provides reasonable
        # accuracy while keeping computation feasible for bootstrap analysis.
        # method='cd' (coordinate descent) is ~3x faster than 'lars'.
        self._spca = SparsePCA(
            n_components=n_components,
            alpha=alpha,
            random_state=42,
            max_iter=max_iter,
            n_jobs=-1,  # Use all CPU cores
            method="cd",  # Coordinate descent - faster than default 'lars'
            tol=1e-4,  # Convergence tolerance
        )
        self._fitted = False

    def fit(self, waveforms: NDArray[np.float64]) -> "SparsePCADecomposition":
        """Fit Sparse PCA to waveforms."""
        self._spca.fit(waveforms)
        self._mean = waveforms.mean(axis=0)
        self._fitted = True
        return self

    def transform(self, waveforms: NDArray[np.float64]) -> SparsePCAResult:
        """Transform waveforms to sparse component scores."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        scores = self._spca.transform(waveforms)
        loadings = self._spca.components_

        # Compute sparsity (fraction of zeros)
        sparsity = [
            (np.abs(loadings[i]) < 1e-10).sum() / loadings.shape[1]
            for i in range(self.n_components)
        ]

        return SparsePCAResult(
            loadings=loadings,
            scores=scores,
            n_components=self.n_components,
            mean=self._mean,
            sparsity=sparsity,
        )

    def fit_transform(self, waveforms: NDArray[np.float64]) -> SparsePCAResult:
        """Fit and transform in one step."""
        self.fit(waveforms)
        return self.transform(waveforms)
