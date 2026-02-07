"""
Artifact injection for realistic PLR signal corruption.

This module adds realistic artifacts to clean PLR signals:
1. Blinks: Sharp drops to near-zero (100-300ms duration)
2. Segmentation noise: Random spikes from pupil detection errors
3. Missing data: Gaps where pupil could not be detected
4. Baseline noise: Continuous measurement noise

The outlier percentage is varied across subjects to test
outlier detection methods under different conditions.
"""

from typing import Tuple

import numpy as np


def inject_artifacts(
    pupil_gt: np.ndarray,
    outlier_pct: float,
    seed: int,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject realistic artifacts into a clean PLR signal.

    Parameters
    ----------
    pupil_gt : np.ndarray
        Clean ground truth pupil signal
    outlier_pct : float
        Fraction of timepoints to corrupt (0.02 to 0.40)
    seed : int
        Random seed for reproducibility
    fps : float
        Frames per second (for blink duration calculation)

    Returns
    -------
    pupil_orig : np.ndarray
        Corrupted signal with outliers (values present)
    pupil_raw : np.ndarray
        Signal with outliers as NaN (for downstream processing)
    outlier_mask : np.ndarray
        Binary mask where 1 = outlier, 0 = valid
    """
    rng = np.random.default_rng(seed)
    n = len(pupil_gt)

    # Start with clean signal + baseline noise
    pupil_orig = pupil_gt.copy()
    pupil_orig += rng.normal(0, 0.05, n)

    outlier_mask = np.zeros(n, dtype=np.int32)
    n_target_outliers = int(n * outlier_pct)

    if n_target_outliers == 0:
        pupil_raw = pupil_orig.copy()
        return pupil_orig, pupil_raw, outlier_mask

    # Distribute outliers across types
    n_blinks = max(1, int(n_target_outliers * 0.35))  # 35% blinks
    n_spikes = max(1, int(n_target_outliers * 0.45))  # 45% spikes
    n_gaps = max(1, int(n_target_outliers * 0.20 / 30))  # 20% in gap regions

    # Type 1: Blinks (sharp drops to near-zero)
    # Duration: 100-300ms at 30fps = 3-9 frames
    blink_starts = rng.choice(n - 15, size=min(n_blinks, n - 15), replace=False)
    for start in blink_starts:
        duration = rng.integers(3, 10)
        end = min(start + duration, n)
        # Blinks drop to near zero with some variability
        pupil_orig[start:end] = rng.uniform(0.0, 0.5, end - start)
        outlier_mask[start:end] = 1

    # Type 2: Segmentation spikes (random jumps from tracking errors)
    available_indices = np.where(outlier_mask == 0)[0]
    if len(available_indices) > n_spikes:
        spike_locs = rng.choice(available_indices, size=n_spikes, replace=False)
        # Spikes are multiplicative errors (0.7-1.4x)
        spike_factors = rng.uniform(0.7, 1.4, size=n_spikes)
        pupil_orig[spike_locs] *= spike_factors
        outlier_mask[spike_locs] = 1

    # Type 3: Missing data gaps (pupil completely lost)
    # Longer gaps (30-60 frames = 1-2 seconds)
    gap_starts = rng.choice(max(1, n - 60), size=min(n_gaps, n - 60), replace=False)
    for start in gap_starts:
        duration = rng.integers(30, 61)
        end = min(start + duration, n)
        # Complete loss of signal - random noise or NaN
        pupil_orig[start:end] = rng.uniform(0.0, 1.0, end - start)
        outlier_mask[start:end] = 1

    # Create pupil_raw with outliers as NaN
    pupil_raw = pupil_orig.copy()
    pupil_raw[outlier_mask == 1] = np.nan

    return pupil_orig, pupil_raw, outlier_mask


def create_imputed_signals(
    pupil_orig: np.ndarray,
    pupil_raw: np.ndarray,
    outlier_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create linearly interpolated versions of the signals.

    These are used as input to outlier detection and downstream processing.

    Parameters
    ----------
    pupil_orig : np.ndarray
        Corrupted signal with outliers present
    pupil_raw : np.ndarray
        Signal with outliers as NaN
    outlier_mask : np.ndarray
        Binary mask where 1 = outlier

    Returns
    -------
    pupil_orig_imputed : np.ndarray
        pupil_orig with linear interpolation over outliers
    pupil_raw_imputed : np.ndarray
        pupil_raw with linear interpolation over NaN
    imputation_mask : np.ndarray
        Boolean mask where True = imputed point
    """
    n = len(pupil_orig)
    indices = np.arange(n)

    # Impute pupil_orig (interpolate where outlier_mask == 1)
    valid_orig = outlier_mask == 0
    if np.sum(valid_orig) >= 2:
        pupil_orig_imputed = np.interp(
            indices, indices[valid_orig], pupil_orig[valid_orig]
        )
    else:
        pupil_orig_imputed = pupil_orig.copy()

    # Impute pupil_raw (interpolate where NaN)
    valid_raw = ~np.isnan(pupil_raw)
    if np.sum(valid_raw) >= 2:
        pupil_raw_imputed = np.interp(indices, indices[valid_raw], pupil_raw[valid_raw])
    else:
        pupil_raw_imputed = np.nan_to_num(pupil_raw, nan=4.0)  # Default to 4mm

    # Imputation mask (where we imputed)
    imputation_mask = outlier_mask.astype(bool)

    return pupil_orig_imputed, pupil_raw_imputed, imputation_mask
