"""
Core PLR curve generation using parametric physiological models.

This module generates synthetic pupillary light reflex (PLR) curves based on
physiological models from the literature. The curves are NOT derived from
real patient data - they are generated purely from parametric models.

References:
    - Najjar et al. 2023 Br J Ophthalmol (PLR physiology)
    - Park et al. 2011 Invest Ophthalmol Vis Sci (PIPR in glaucoma)
"""

from typing import Tuple

import numpy as np


def generate_plr_curve(
    class_label: str,
    seed: int,
    n_timepoints: int = 1981,
    duration_seconds: float = 66.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic PLR curve with physiologically plausible characteristics.

    The PLR response has distinct phases:
    1. Baseline (0-10s): Steady-state pupil diameter in darkness
    2. Red Light Response (10-25s): Rod-mediated constriction
    3. Recovery 1 (25-35s): Partial dilation after red light
    4. Blue Light Response (35-50s): Melanopsin-mediated constriction (PIPR)
    5. Recovery 2 (50-66s): Sustained constriction (pathological in glaucoma)

    PRIVACY NOTE: This function generates curves with significant random variation
    to ensure Pearson correlation with real data stays below 0.60. Variations include:
    - Random timing jitter (up to ±2s)
    - Non-linear trend components
    - Multiple frequency drift components
    - Random curve shape variations

    Parameters
    ----------
    class_label : str
        "control" or "glaucoma" - determines physiological parameters
    seed : int
        Random seed for reproducibility
    n_timepoints : int
        Number of timepoints (default 1981 at 30fps for 66s)
    duration_seconds : float
        Total duration in seconds

    Returns
    -------
    time : np.ndarray
        Time array in seconds
    pupil_gt : np.ndarray
        Ground truth pupil diameter in mm
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0, duration_seconds, n_timepoints)

    # Base physiological parameters differ by class (WIDER ranges for privacy)
    if class_label == "control":
        baseline = rng.uniform(3.5, 7.0)  # Wider range
        red_amplitude = rng.uniform(0.5, 2.0)  # Wider range
        blue_amplitude = rng.uniform(0.8, 2.5)  # Wider range
        pipr_sustained = rng.uniform(0.10, 0.35)  # Wider range
        recovery_tau = rng.uniform(3.0, 15.0)  # Wider range
    else:  # glaucoma
        baseline = rng.uniform(3.0, 6.0)  # Wider range
        red_amplitude = rng.uniform(0.3, 1.5)  # Wider range
        blue_amplitude = rng.uniform(0.5, 2.0)  # Wider range
        pipr_sustained = rng.uniform(0.05, 0.25)  # Wider range
        recovery_tau = rng.uniform(5.0, 20.0)  # Wider range

    # Add timing jitter (CRITICAL for privacy - breaks time alignment)
    timing_jitter = rng.uniform(-2.0, 2.0)  # Up to ±2s shift
    red_on = 10.0 + timing_jitter
    red_off = 25.0 + timing_jitter
    blue_on = 35.0 + timing_jitter + rng.uniform(-1.0, 1.0)  # Extra jitter
    blue_off = 50.0 + timing_jitter + rng.uniform(-1.0, 1.0)

    # Initialize with baseline
    pupil = np.ones_like(time) * baseline

    # Add MULTIPLE drift components (complex non-linear variation)
    for _ in range(rng.integers(2, 5)):  # 2-4 drift components
        drift_freq = rng.uniform(0.02, 0.25)
        drift_amp = rng.uniform(0.05, 0.25)
        drift_phase = rng.uniform(0, 2 * np.pi)
        pupil += drift_amp * np.sin(2 * np.pi * drift_freq * time + drift_phase)

    # Add random polynomial trend (quadratic or cubic)
    if rng.random() > 0.5:
        trend_coef = rng.uniform(-0.001, 0.001)
        pupil += trend_coef * (time - duration_seconds / 2) ** 2
    else:
        trend_coef = rng.uniform(-0.0001, 0.0001)
        pupil += trend_coef * (time - duration_seconds / 2) ** 3

    # Red light response (rod-mediated) with variable shape
    red_mask = (time >= red_on) & (time < red_off)
    constriction_rate = rng.uniform(0.3, 2.0)  # Wider rate range

    # Use different curve shapes (not always exponential)
    shape_type = rng.choice(["exp", "sigmoid", "linear_exp"])
    t_red = time[red_mask] - red_on
    if len(t_red) > 0:
        t_norm = t_red / (red_off - red_on)
        if shape_type == "exp":
            response = 1 - np.exp(-constriction_rate * t_red)
        elif shape_type == "sigmoid":
            midpoint = rng.uniform(0.3, 0.7)
            steepness = rng.uniform(5, 15)
            response = 1 / (1 + np.exp(-steepness * (t_norm - midpoint)))
        else:  # linear_exp blend
            linear_weight = rng.uniform(0.2, 0.5)
            exp_part = 1 - np.exp(-constriction_rate * t_red)
            linear_part = t_norm
            response = linear_weight * linear_part + (1 - linear_weight) * exp_part
        pupil[red_mask] -= red_amplitude * response

    # Recovery after red
    red_recovery_mask = (time >= red_off) & (time < blue_on)
    t_rec1 = time[red_recovery_mask] - red_off
    if len(t_rec1) > 0:
        min_after_red = (
            pupil[red_mask][-1] if np.any(red_mask) else baseline - red_amplitude
        )
        recovery_target = baseline * rng.uniform(0.85, 1.0)  # Doesn't fully recover
        pupil[red_recovery_mask] = min_after_red + (recovery_target - min_after_red) * (
            1 - np.exp(-t_rec1 / recovery_tau)
        )

    # Blue light response with different shape
    blue_mask = (time >= blue_on) & (time < blue_off)
    t_blue = time[blue_mask] - blue_on
    if len(t_blue) > 0:
        pre_blue = pupil[blue_mask][0] if np.any(blue_mask) else baseline
        blue_rate = constriction_rate * rng.uniform(0.8, 1.5)  # Variable rate
        pupil[blue_mask] = pre_blue - blue_amplitude * (1 - np.exp(-blue_rate * t_blue))

    # PIPR - sustained constriction after blue light
    pipr_mask = time >= blue_off
    t_pipr = time[pipr_mask] - blue_off
    if len(t_pipr) > 0:
        min_after_blue = (
            pupil[blue_mask][-1] if np.any(blue_mask) else baseline - blue_amplitude
        )
        sustained_level = baseline * (1 - pipr_sustained)
        # Add oscillation to PIPR recovery (hippus)
        hippus_freq = rng.uniform(0.1, 0.3)
        hippus_amp = rng.uniform(0.02, 0.08)
        hippus = hippus_amp * np.sin(2 * np.pi * hippus_freq * t_pipr)
        pupil[pipr_mask] = (
            min_after_blue
            + (sustained_level - min_after_blue)
            * (1 - np.exp(-t_pipr / (recovery_tau * rng.uniform(1.0, 2.5))))
            + hippus
        )

    # Add measurement noise
    measurement_noise = rng.normal(0, 0.05, n_timepoints)
    pupil += measurement_noise

    # Add random dropout/spike noise
    n_spikes = rng.integers(3, 12)
    spike_locs = rng.choice(
        n_timepoints, size=min(n_spikes, n_timepoints), replace=False
    )
    pupil[spike_locs] += rng.normal(0, 0.2, len(spike_locs))

    # Ensure physiologically valid range (2-8mm)
    pupil = np.clip(pupil, 2.0, 8.0)

    return time, pupil


def generate_light_stimuli(
    n_timepoints: int = 1981,
    duration_seconds: float = 66.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the light stimulus protocol.

    The protocol matches the SERI PLR study:
    - Red light: 10-25s
    - Blue light: 35-50s

    Parameters
    ----------
    n_timepoints : int
        Number of timepoints
    duration_seconds : float
        Total duration in seconds

    Returns
    -------
    red : np.ndarray
        Red light stimulus (0 or 1)
    blue : np.ndarray
        Blue light stimulus (0 or 1)
    light_stimuli : np.ndarray
        Combined stimulus (red + blue)
    """
    time = np.linspace(0, duration_seconds, n_timepoints)

    red = np.zeros(n_timepoints)
    blue = np.zeros(n_timepoints)

    # Red light: 10-25s
    red[(time >= 10) & (time < 25)] = 1.0

    # Blue light: 35-50s
    blue[(time >= 35) & (time < 50)] = 1.0

    light_stimuli = red + blue

    return red, blue, light_stimuli
