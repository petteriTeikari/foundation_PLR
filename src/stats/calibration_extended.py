"""
Extended calibration metrics (STRATOS-compliant).

Provides calibration slope, O:E ratio, and Brier decomposition.

Cross-references:
- planning/statistics-implementation.md (Section 2.6)
- appendix-literature-review/section-08-biostatistics.tex
- appendix-literature-review/section-13-calibration.tex

References:
- Van Calster et al. (2024). STRATOS guidance.
- Steyerberg et al. (2010). Assessing calibration.
- Murphy (1973). Brier score decomposition.

NOTE: O:E ratio = Observed/Expected (Van Calster 2024 standard)
- O:E > 1: Model underestimates risk (observed > predicted)
- O:E < 1: Model overestimates risk (observed < predicted)
- O:E ≈ 1: Well calibrated
"""

from typing import Union

import numpy as np
from sklearn.linear_model import LogisticRegression

from ._exceptions import ConvergenceError
from ._types import CalibrationResult, StatsResult
from ._validation import validate_binary_outcomes, validate_probabilities

__all__ = [
    "calibration_slope_intercept",
    "brier_decomposition",
]


def calibration_slope_intercept(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    eps: float = 1e-10,
) -> CalibrationResult:
    """
    Compute calibration slope, intercept, and O:E ratio.

    Fits logistic regression of outcomes on logit(predictions):
        logit(P(Y=1)) = α + β × logit(ŷ)

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1)
    y_prob : array-like
        Predicted probabilities
    eps : float, default 1e-10
        Small value for clipping probabilities

    Returns
    -------
    CalibrationResult
        Contains slope, intercept, o_e_ratio, brier_score

    Notes
    -----
    **Calibration slope interpretation:**
    - slope = 1.0: Perfect calibration
    - slope < 1.0: Overfitting (predictions too extreme)
    - slope > 1.0: Underfitting (predictions too conservative)

    **Observed:Expected ratio interpretation (Van Calster 2024):**
    - O:E ≈ 1.0: Well calibrated overall
    - O:E > 1.0: Under-prediction of risk (observed > predicted)
    - O:E < 1.0: Over-prediction of risk (observed < predicted)

    **Intercept (calibration-in-the-large):**
    - Should be near 0 for well-calibrated model
    - Positive: underestimate mean risk
    - Negative: overestimate mean risk

    Examples
    --------
    >>> y_true = [0, 0, 1, 1, 1]
    >>> y_prob = [0.1, 0.2, 0.7, 0.8, 0.9]
    >>> result = calibration_slope_intercept(y_true, y_prob)
    >>> print(f"Slope: {result.slope:.2f}, O:E: {result.o_e_ratio:.2f}")
    """
    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob, clip=True, eps=eps)

    n = len(y_true)

    # Convert probabilities to logits
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_prob = np.log(y_prob_clipped / (1 - y_prob_clipped))

    # Fit calibration model: outcome ~ logit(prediction)
    try:
        cal_model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000, warm_start=False
        )
        cal_model.fit(logit_prob.reshape(-1, 1), y_true)
        slope = cal_model.coef_[0][0]
        intercept = cal_model.intercept_[0]
    except Exception as e:
        raise ConvergenceError("Calibration model", 1000) from e

    # Observed:Expected ratio (Van Calster 2024 standard)
    # O:E = Observed / Expected = sum(y_true) / sum(y_prob)
    observed_rate = np.mean(y_true)
    predicted_rate = np.mean(y_prob)
    o_e_ratio = observed_rate / max(predicted_rate, eps)

    # Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)

    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        o_e_ratio=o_e_ratio,
        brier_score=brier_score,
        scalars={
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "o_e_ratio": o_e_ratio,
            "brier_score": brier_score,
        },
        metadata={
            "n_samples": n,
            "prevalence": float(observed_rate),
            "mean_predicted": float(predicted_rate),
        },
    )


def brier_decomposition(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    n_bins: int = 10,
) -> StatsResult:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.

    Brier = Reliability - Resolution + Uncertainty

    Where:
    - **Reliability** (calibration error): Measures how well predicted
      probabilities match observed frequencies. Lower is better.
    - **Resolution** (discrimination): Measures ability to separate
      different outcomes. Higher is better.
    - **Uncertainty**: Base rate uncertainty = p(1-p). Fixed for dataset.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes
    y_prob : array-like
        Predicted probabilities
    n_bins : int, default 10
        Number of probability bins

    Returns
    -------
    StatsResult
        Contains brier_score, reliability, resolution, uncertainty

    Notes
    -----
    The Murphy (1973) decomposition:

    BS = (1/N) Σ_k n_k (p_k - o_k)²    [Reliability]
       - (1/N) Σ_k n_k (o_k - ō)²      [Resolution]
       + ō(1 - ō)                       [Uncertainty]

    Where:
    - n_k: number of samples in bin k
    - p_k: mean predicted probability in bin k
    - o_k: observed frequency in bin k
    - ō: overall observed frequency (prevalence)

    Examples
    --------
    >>> result = brier_decomposition(y_true, y_prob)
    >>> print(f"Brier: {result.scalars['brier_score']:.4f}")
    >>> print(f"Reliability (lower better): {result.scalars['reliability']:.4f}")
    >>> print(f"Resolution (higher better): {result.scalars['resolution']:.4f}")
    """
    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob)

    n = len(y_true)
    prevalence = np.mean(y_true)

    # Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)

    # Uncertainty (base rate)
    uncertainty = prevalence * (1 - prevalence)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    reliability = 0.0
    resolution = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_k = np.sum(mask)

        if n_k > 0:
            p_k = np.mean(y_prob[mask])  # Mean predicted probability in bin
            o_k = np.mean(y_true[mask])  # Observed frequency in bin

            # Reliability: weighted squared difference between predicted and observed
            reliability += n_k * (p_k - o_k) ** 2

            # Resolution: weighted squared difference between observed and overall prevalence
            resolution += n_k * (o_k - prevalence) ** 2

    reliability /= n
    resolution /= n

    return StatsResult(
        scalars={
            "brier_score": brier_score,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
        },
        metadata={
            "n_bins": n_bins,
            "n_samples": n,
            "prevalence": prevalence,
        },
    )
