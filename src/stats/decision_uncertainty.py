"""
Decision Uncertainty (DU) metric for clinical prediction models.

Quantifies uncertainty about treatment decisions at a given threshold.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Appendix E.1)
- background-research/final-stats-research-before-implementation/literature-research.md

References:
- Barrenada et al. (2025). The fundamental problem of providing uncertainty
  in individual risk predictions using clinical prediction models. BMJ Medicine.

DU Formula:
    DU = min(P(p > threshold), P(p < threshold))

where probabilities are computed across bootstrap samples.

Interpretation:
- DU = 0: Complete certainty about treatment decision
- DU = 0.5: Maximum uncertainty (50% chance of crossing threshold)
- Higher DU values indicate patients for whom the model is uncertain
  about whether they fall above or below the decision threshold.
"""

from typing import Dict, Union

import numpy as np

from ._exceptions import ValidationError

__all__ = [
    "decision_uncertainty",
    "decision_uncertainty_per_subject",
    "decision_uncertainty_summary",
]


def decision_uncertainty(
    bootstrap_samples: Union[np.ndarray, list],
    threshold: float,
) -> float:
    """
    Compute Decision Uncertainty for a single subject.

    DU = min(P(p > threshold), P(p < threshold))

    Parameters
    ----------
    bootstrap_samples : array-like
        Predicted probabilities across bootstrap iterations
        Shape: (n_bootstrap,)
    threshold : float
        Decision threshold (e.g., 0.1 for 10% risk cutoff)

    Returns
    -------
    float
        Decision Uncertainty in [0, 0.5]
        - 0: complete certainty
        - 0.5: maximum uncertainty

    Examples
    --------
    >>> samples = np.array([0.55, 0.58, 0.60, 0.62, 0.65])
    >>> du = decision_uncertainty(samples, threshold=0.5)
    >>> print(f"Decision Uncertainty: {du:.2f}")
    Decision Uncertainty: 0.00  # All samples above threshold
    """
    bootstrap_samples = np.asarray(bootstrap_samples)

    if bootstrap_samples.size == 0:
        raise ValidationError(
            parameter="bootstrap_samples",
            expected="non-empty array",
            actual="empty array",
        )

    if not 0 <= threshold <= 1:
        raise ValidationError(
            parameter="threshold", expected="value in [0, 1]", actual=str(threshold)
        )

    # Compute proportion above and below threshold
    p_above = np.mean(bootstrap_samples > threshold)
    p_below = np.mean(bootstrap_samples < threshold)

    # DU = min(P(above), P(below))
    du = min(p_above, p_below)

    return float(du)


def decision_uncertainty_per_subject(
    bootstrap_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Compute Decision Uncertainty for multiple subjects.

    Parameters
    ----------
    bootstrap_matrix : np.ndarray
        Predicted probabilities across bootstrap iterations
        Shape: (n_subjects, n_bootstrap)
    threshold : float
        Decision threshold

    Returns
    -------
    np.ndarray
        Decision Uncertainty for each subject
        Shape: (n_subjects,)

    Examples
    --------
    >>> # 50 subjects, 1000 bootstrap samples each
    >>> predictions = np.random.uniform(0, 1, (50, 1000))
    >>> du_per_subject = decision_uncertainty_per_subject(predictions, threshold=0.1)
    >>> print(f"Mean DU: {du_per_subject.mean():.3f}")
    """
    bootstrap_matrix = np.asarray(bootstrap_matrix)

    if bootstrap_matrix.ndim != 2:
        raise ValidationError(
            parameter="bootstrap_matrix",
            expected="2D array (n_subjects, n_bootstrap)",
            actual=f"shape={bootstrap_matrix.shape}",
        )

    if not 0 <= threshold <= 1:
        raise ValidationError(
            parameter="threshold", expected="value in [0, 1]", actual=str(threshold)
        )

    n_subjects = bootstrap_matrix.shape[0]
    du_values = np.zeros(n_subjects)

    for i in range(n_subjects):
        du_values[i] = decision_uncertainty(bootstrap_matrix[i], threshold)

    return du_values


def decision_uncertainty_summary(
    bootstrap_matrix: np.ndarray,
    threshold: float,
    du_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Compute summary statistics for Decision Uncertainty.

    Parameters
    ----------
    bootstrap_matrix : np.ndarray
        Predicted probabilities across bootstrap iterations
        Shape: (n_subjects, n_bootstrap)
    threshold : float
        Decision threshold for treatment/no-treatment
    du_threshold : float, default 0.3
        Threshold for classifying subjects as "high uncertainty"

    Returns
    -------
    dict
        Summary statistics:
        - mean_du: Mean Decision Uncertainty across subjects
        - median_du: Median Decision Uncertainty
        - std_du: Standard deviation of DU
        - min_du: Minimum DU
        - max_du: Maximum DU
        - pct_above_threshold: % of subjects with DU > du_threshold
        - n_subjects: Number of subjects
        - n_uncertain: Number of subjects with high uncertainty

    Examples
    --------
    >>> predictions = np.random.uniform(0, 1, (100, 1000))
    >>> summary = decision_uncertainty_summary(predictions, threshold=0.1)
    >>> print(f"{summary['pct_above_threshold']:.1f}% of subjects have high DU")
    """
    du_per_subject = decision_uncertainty_per_subject(bootstrap_matrix, threshold)

    n_uncertain = np.sum(du_per_subject > du_threshold)
    pct_uncertain = 100 * n_uncertain / len(du_per_subject)

    return {
        "mean_du": float(np.mean(du_per_subject)),
        "median_du": float(np.median(du_per_subject)),
        "std_du": float(np.std(du_per_subject)),
        "min_du": float(np.min(du_per_subject)),
        "max_du": float(np.max(du_per_subject)),
        "pct_above_threshold": float(pct_uncertain),
        "n_subjects": len(du_per_subject),
        "n_uncertain": int(n_uncertain),
        "du_threshold_used": du_threshold,
        "decision_threshold_used": threshold,
    }
