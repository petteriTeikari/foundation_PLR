"""
Scaled Brier Score (Index of Prediction Accuracy).

Provides a skill score that compares model performance to a null model.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Appendix E.1)

References:
- Steyerberg et al. (2010). Assessing calibration.
- Van Calster et al. (2019). Calibration: the Achilles heel.

IPA Formula:
    IPA = 1 - Brier/Brier_null

where:
    Brier = mean((y_prob - y_true)²)
    Brier_null = prevalence × (1 - prevalence)

Interpretation:
- IPA = 1: Perfect predictions
- IPA > 0: Better than null model (positive skill)
- IPA = 0: Same as null model (no skill)
- IPA < 0: Worse than null model (negative skill, harmful)
"""

from typing import Dict, Optional, Union

import numpy as np

from ._defaults import DEFAULT_CI_LEVEL, DEFAULT_N_BOOTSTRAP
from ._exceptions import InsufficientDataError, SingleClassError, ValidationError
from ._validation import validate_binary_outcomes, validate_probabilities

__all__ = [
    "scaled_brier_score",
    "scaled_brier_score_with_ci",
    "interpret_ipa",
]


def scaled_brier_score(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
) -> Dict[str, float]:
    """
    Compute scaled Brier score (Index of Prediction Accuracy).

    IPA = 1 - Brier/Brier_null

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1)
    y_prob : array-like
        Predicted probabilities

    Returns
    -------
    dict
        Contains:
        - brier: Brier score
        - brier_null: Null model Brier score (prevalence × (1 - prevalence))
        - ipa: Index of Prediction Accuracy
        - prevalence: Observed prevalence
        - interpretation: Qualitative interpretation

    Raises
    ------
    SingleClassError
        If only one class is present (Brier_null = 0)
    InsufficientDataError
        If arrays are empty
    ValidationError
        If arrays have different lengths

    Examples
    --------
    >>> y_true = [0, 0, 1, 1]
    >>> y_prob = [0.1, 0.2, 0.8, 0.9]
    >>> result = scaled_brier_score(y_true, y_prob)
    >>> print(f"IPA: {result['ipa']:.3f} ({result['interpretation']})")
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Validation
    if len(y_true) == 0:
        raise InsufficientDataError(
            required=1,
            actual=0,
            context="scaled Brier score requires at least one sample",
        )

    if len(y_true) != len(y_prob):
        raise ValidationError(
            parameter="y_true, y_prob",
            expected="arrays of equal length",
            actual=f"lengths {len(y_true)}, {len(y_prob)}",
        )

    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob)

    # Check for single class
    prevalence = np.mean(y_true)
    if prevalence == 0 or prevalence == 1:
        class_counts = {0: int(np.sum(y_true == 0)), 1: int(np.sum(y_true == 1))}
        raise SingleClassError(class_counts)

    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    # Null model Brier score
    # The null model predicts prevalence for everyone
    # Brier_null = mean((prevalence - y_true)²)
    #            = prevalence × (1 - prevalence)² + (1 - prevalence) × prevalence²
    #            = prevalence × (1 - prevalence)
    brier_null = prevalence * (1 - prevalence)

    # Index of Prediction Accuracy (IPA)
    ipa = 1 - brier / brier_null

    interpretation = interpret_ipa(ipa)

    return {
        "brier": float(brier),
        "brier_null": float(brier_null),
        "ipa": float(ipa),
        "prevalence": float(prevalence),
        "interpretation": interpretation,
        "n_samples": len(y_true),
    }


def scaled_brier_score_with_ci(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute scaled Brier score with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1)
    y_prob : array-like
        Predicted probabilities
    n_bootstrap : int, default 1000
        Number of bootstrap samples
    ci_level : float, default 0.95
        Confidence level (0-1)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Contains all fields from scaled_brier_score plus:
        - ipa_ci_lower: Lower bound of CI
        - ipa_ci_upper: Upper bound of CI
        - brier_ci_lower: Lower bound of Brier CI
        - brier_ci_upper: Upper bound of Brier CI

    Examples
    --------
    >>> result = scaled_brier_score_with_ci(y_true, y_prob, n_bootstrap=500)
    >>> print(f"IPA: {result['ipa']:.3f} ({result['ipa_ci_lower']:.3f}-{result['ipa_ci_upper']:.3f})")
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Get point estimate
    result = scaled_brier_score(y_true, y_prob)

    # Bootstrap
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    bootstrap_ipa = []
    bootstrap_brier = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.integers(0, n, n)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        # Skip if single class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        prevalence_boot = np.mean(y_true_boot)
        brier_boot = np.mean((y_prob_boot - y_true_boot) ** 2)
        brier_null_boot = prevalence_boot * (1 - prevalence_boot)
        ipa_boot = 1 - brier_boot / brier_null_boot

        bootstrap_ipa.append(ipa_boot)
        bootstrap_brier.append(brier_boot)

    bootstrap_ipa = np.array(bootstrap_ipa)
    bootstrap_brier = np.array(bootstrap_brier)

    # Percentile CI
    alpha = 1 - ci_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    result["ipa_ci_lower"] = float(np.percentile(bootstrap_ipa, lower_pct))
    result["ipa_ci_upper"] = float(np.percentile(bootstrap_ipa, upper_pct))
    result["brier_ci_lower"] = float(np.percentile(bootstrap_brier, lower_pct))
    result["brier_ci_upper"] = float(np.percentile(bootstrap_brier, upper_pct))
    result["n_bootstrap"] = len(bootstrap_ipa)
    result["ci_level"] = ci_level

    return result


def interpret_ipa(ipa: float) -> str:
    """
    Provide qualitative interpretation of IPA.

    Based on guidance from Steyerberg et al. and Van Calster et al.

    Parameters
    ----------
    ipa : float
        Index of Prediction Accuracy

    Returns
    -------
    str
        Qualitative interpretation

    Interpretation thresholds:
    - IPA < 0: 'harmful' (worse than prevalence model)
    - IPA ≈ 0: 'useless' (same as prevalence model)
    - IPA ≈ 0.1-0.2: 'weak' prediction
    - IPA ≈ 0.2-0.4: 'moderate' prediction
    - IPA ≈ 0.4-0.6: 'good' prediction
    - IPA > 0.6: 'excellent' prediction

    Examples
    --------
    >>> interpret_ipa(0.35)
    'moderate'
    >>> interpret_ipa(-0.1)
    'harmful'
    """
    if ipa < 0:
        return "harmful"
    elif ipa < 0.05:
        return "useless"
    elif ipa < 0.2:
        return "weak"
    elif ipa < 0.4:
        return "moderate"
    elif ipa < 0.6:
        return "good"
    else:
        return "excellent"
