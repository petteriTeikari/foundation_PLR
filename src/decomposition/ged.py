"""Generalized Eigendecomposition (GED) for PLR Waveforms.

GED finds components that maximize contrast between two conditions,
e.g., stimulus period vs baseline. This is ideal for PLR where we
want components that respond maximally to light stimuli.

Reference: Cohen MX (2022) "A tutorial on generalized eigendecomposition
for denoising, contrast enhancement, and dimension reduction in
multichannel electrophysiology" NeuroImage 247:118809
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass
class GEDResult:
    """Result from GED decomposition."""

    components: NDArray[np.float64]  # GED components (n_components, n_timepoints)
    eigenvalues: NDArray[np.float64]  # Eigenvalues (contrast ratios)
    scores: NDArray[np.float64]  # Subject scores (n_subjects, n_components)
    mean: NDArray[np.float64]  # Mean waveform
    n_components: int


class GEDDecomposition:
    """Generalized Eigendecomposition for PLR analysis.

    GED maximizes: w^T S_signal w / w^T S_reference w

    where S_signal is the covariance during stimulus periods and
    S_reference is the covariance during baseline periods.

    This extracts components that are maximally driven by the stimulus.

    Parameters
    ----------
    n_components : int
        Number of components to extract (default: 3)
    blue_onset : float
        Blue stimulus onset time (seconds)
    blue_offset : float
        Blue stimulus offset time (seconds)
    red_onset : float
        Red stimulus onset time (seconds)
    red_offset : float
        Red stimulus offset time (seconds)
    baseline_end : float
        End of baseline period (seconds)
    regularization : float
        Regularization for covariance matrices (default: 0.01)
    """

    def __init__(
        self,
        n_components: int = 3,
        blue_onset: float = 15.5,
        blue_offset: float = 24.5,
        red_onset: float = 46.5,
        red_offset: float = 55.5,
        baseline_end: float = 14.0,
        regularization: float = 0.01,
    ):
        self.n_components = n_components
        self.blue_onset = blue_onset
        self.blue_offset = blue_offset
        self.red_onset = red_onset
        self.red_offset = red_offset
        self.baseline_end = baseline_end
        self.regularization = regularization
        self._fitted = False

    def _get_time_masks(
        self, time_vector: NDArray[np.float64]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Get masks for stimulus and baseline periods."""
        # Stimulus periods (blue + red)
        stim_mask = (
            (time_vector >= self.blue_onset) & (time_vector <= self.blue_offset)
        ) | ((time_vector >= self.red_onset) & (time_vector <= self.red_offset))

        # Baseline period (before first stimulus)
        baseline_mask = time_vector <= self.baseline_end

        return stim_mask, baseline_mask

    def _compute_weighted_covariance(
        self, waveforms: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute weighted covariance matrix over all timepoints.

        For single-channel PLR, we use temporal weighting instead of masks
        to ensure covariance matrices have the same dimensions.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)
        weights : ndarray
            Temporal weights of shape (n_timepoints,) indicating importance
            of each timepoint (e.g., 1.0 for stimulus, 0.0 for baseline)
        """
        # Center the data
        data_centered = waveforms - waveforms.mean(axis=0)

        # Apply temporal weights (sqrt for proper weighting in covariance)
        sqrt_weights = np.sqrt(weights + 1e-10)
        weighted_data = data_centered * sqrt_weights[np.newaxis, :]

        # Covariance (timepoints x timepoints)
        cov = weighted_data.T @ weighted_data / (waveforms.shape[0] - 1)

        return cov

    def fit(
        self, waveforms: NDArray[np.float64], time_vector: NDArray[np.float64]
    ) -> "GEDDecomposition":
        """Fit GED to waveforms.

        Uses weighted covariance approach for single-channel PLR data.
        Finds temporal patterns that maximize stimulus-to-baseline contrast.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)
        time_vector : ndarray
            Time points in seconds

        Returns
        -------
        self
        """
        self._time_vector = time_vector
        self._mean = waveforms.mean(axis=0)
        n_timepoints = len(time_vector)

        # Get time period masks
        stim_mask, baseline_mask = self._get_time_masks(time_vector)

        if stim_mask.sum() < 10 or baseline_mask.sum() < 10:
            raise ValueError(
                f"Insufficient timepoints: stimulus={stim_mask.sum()}, "
                f"baseline={baseline_mask.sum()}"
            )

        # Convert masks to weights (ensures same-sized covariance matrices)
        stim_weights = stim_mask.astype(np.float64)
        baseline_weights = baseline_mask.astype(np.float64)

        # Normalize weights to sum to 1
        stim_weights = stim_weights / (stim_weights.sum() + 1e-10)
        baseline_weights = baseline_weights / (baseline_weights.sum() + 1e-10)

        # Compute weighted covariance matrices (same size: n_timepoints x n_timepoints)
        S_stim = self._compute_weighted_covariance(waveforms, stim_weights)
        S_base = self._compute_weighted_covariance(waveforms, baseline_weights)

        # Regularize
        S_stim += self.regularization * np.eye(n_timepoints)
        S_base += self.regularization * np.eye(n_timepoints)

        # Solve generalized eigenvalue problem: S_stim @ w = lambda * S_base @ w
        eigenvalues, eigenvectors = linalg.eigh(S_stim, S_base)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top components (now full-length temporal filters)
        self._eigenvalues = eigenvalues[: self.n_components]
        self._eigenvectors = eigenvectors[:, : self.n_components]

        # Store masks for reference
        self._stim_mask = stim_mask
        self._baseline_mask = baseline_mask

        # Components are the eigenvectors (temporal patterns)
        # Normalize each component
        self._components = np.zeros((self.n_components, n_timepoints))
        for i in range(self.n_components):
            comp = self._eigenvectors[:, i]
            self._components[i] = comp / (np.linalg.norm(comp) + 1e-10)

        self._fitted = True
        return self

    def transform(self, waveforms: NDArray[np.float64]) -> GEDResult:
        """Transform waveforms to GED scores.

        Parameters
        ----------
        waveforms : ndarray
            Array of shape (n_subjects, n_timepoints)

        Returns
        -------
        GEDResult
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Center the data
        centered = waveforms - waveforms.mean(axis=0)

        # Project onto full-length eigenvectors
        scores = centered @ self._eigenvectors

        return GEDResult(
            components=self._components,
            eigenvalues=self._eigenvalues,
            scores=scores,
            mean=self._mean,
            n_components=self.n_components,
        )

    def fit_transform(
        self, waveforms: NDArray[np.float64], time_vector: NDArray[np.float64]
    ) -> GEDResult:
        """Fit and transform in one step."""
        self.fit(waveforms, time_vector)
        return self.transform(waveforms)


class GEDWithPIRP(GEDDecomposition):
    """GED variant that also extracts PIPR-focused component.

    Adds a third contrast: post-illumination period vs baseline.
    """

    def __init__(
        self,
        n_components: int = 3,
        pipr_start: float = 56.0,
        pipr_end: float = 66.0,
        **kwargs,
    ):
        super().__init__(n_components=n_components, **kwargs)
        self.pipr_start = pipr_start
        self.pipr_end = pipr_end

    def _get_pipr_mask(self, time_vector: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Get mask for PIPR period (after last stimulus)."""
        return (time_vector >= self.pipr_start) & (time_vector <= self.pipr_end)
