"""
Clinical utility metrics: Net Benefit and Decision Curve Analysis.

Following STRATOS initiative recommendations for clinical model evaluation.

Cross-references:
- planning/statistics-implementation.md (Section 2.7)
- appendix-literature-review/section-11-clinical-model-evaluation.tex

References:
- Vickers & Elkin (2006). Decision curve analysis.
- Van Calster et al. (2024). STRATOS guidance.

Glaucoma-specific context:
- Screening: 1-10% threshold range (high sensitivity preferred)
- Diagnosis: 10-30% threshold range (balanced)
- Prevalence: ~3.5% in general population
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._exceptions import ValidationError
from ._validation import (
    validate_binary_outcomes,
    validate_in_range,
    validate_probabilities,
)

__all__ = [
    "net_benefit",
    "decision_curve_analysis",
    "standardized_net_benefit",
    "optimal_threshold_cost_sensitive",
]


# Glaucoma-specific constants
GLAUCOMA_THRESHOLD_RANGE = (0.01, 0.30)  # 1% to 30%
GLAUCOMA_PREVALENCE = 0.035  # ~3.5% in general population


def net_benefit(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float,
) -> float:
    """
    Compute net benefit at a single threshold.

    Net Benefit quantifies the clinical utility of a prediction model
    by weighing true positives against false positives at a given
    decision threshold.

    Formula:
        NB = TP/n - FP/n × (p_t / (1 - p_t))

    Where p_t / (1 - p_t) represents the odds at threshold p_t,
    which serves as the exchange rate between benefits (TP) and harms (FP).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1)
    y_prob : array-like
        Predicted probabilities
    threshold : float
        Decision threshold (probability cutoff), must be in (0, 1)

    Returns
    -------
    float
        Net benefit value. Can be negative if FP harm outweighs TP benefit.

    Notes
    -----
    Interpretation:
    - NB > 0: Model has clinical utility at this threshold
    - NB = 0: Equivalent to treat-none strategy
    - NB < 0: Model causes net harm (too many false positives)

    The threshold represents the probability above which treatment/intervention
    would be initiated. It implicitly encodes the cost-benefit ratio:
        threshold = Cost_FP / (Cost_FP + Benefit_TP)

    Examples
    --------
    >>> y_true = [0, 0, 1, 1, 1]
    >>> y_prob = [0.1, 0.3, 0.6, 0.8, 0.9]
    >>> nb = net_benefit(y_true, y_prob, threshold=0.5)
    >>> print(f"Net Benefit at 50%: {nb:.3f}")
    """
    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob, clip=False)
    validate_in_range(threshold, "threshold", 0.0, 1.0, inclusive="neither")

    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Odds at threshold (exchange rate)
    odds = threshold / (1 - threshold)

    # Net benefit
    nb = (tp / n) - (fp / n) * odds
    return float(nb)


def decision_curve_analysis(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    thresholds: Optional[np.ndarray] = None,
    threshold_range: Tuple[float, float] = GLAUCOMA_THRESHOLD_RANGE,
    n_thresholds: int = 30,
) -> pd.DataFrame:
    """
    Decision Curve Analysis for clinical utility assessment.

    Computes net benefit across a range of threshold probabilities and
    compares the model to treat-all and treat-none strategies.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes
    y_prob : array-like
        Predicted probabilities
    thresholds : array-like, optional
        Specific thresholds to evaluate. If None, uses threshold_range.
    threshold_range : tuple, default (0.01, 0.30)
        Min and max threshold for glaucoma screening context
    n_thresholds : int, default 30
        Number of thresholds to evaluate

    Returns
    -------
    pd.DataFrame
        Columns: threshold, nb_model, nb_all, nb_none, sensitivity,
        specificity, ppv, npv, model_useful

    Notes
    -----
    For glaucoma screening:
    - Low thresholds (1-5%): prioritize not missing disease
    - Medium thresholds (5-15%): balance sensitivity/specificity
    - Higher thresholds (15-30%): reduce false positives

    A model provides clinical utility when:
        nb_model > max(nb_all, nb_none)

    The area between model curve and max(treat-all, treat-none)
    represents the clinical benefit.

    Examples
    --------
    >>> dca = decision_curve_analysis(y_true, y_prob)
    >>> useful_range = dca[dca['model_useful']]['threshold']
    >>> print(f"Model useful at thresholds: {useful_range.min():.0%} - {useful_range.max():.0%}")
    """
    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob)

    if thresholds is None:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    n = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n - n_pos
    prevalence = n_pos / n

    results = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        odds = pt / (1 - pt)

        # Net benefit of model
        nb_model = (tp / n) - (fp / n) * odds

        # Treat-all strategy: predict positive for everyone
        # TP = all positives, FP = all negatives
        nb_all = prevalence - (1 - prevalence) * odds
        nb_all = max(nb_all, 0.0)  # Cap at 0 for interpretability

        # Treat-none strategy: always 0
        nb_none = 0.0

        # Additional metrics
        sensitivity = tp / max(n_pos, 1)
        specificity = tn / max(n_neg, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        results.append(
            {
                "threshold": pt,
                "nb_model": nb_model,
                "nb_all": nb_all,
                "nb_none": nb_none,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "n_positive_predictions": int(tp + fp),
            }
        )

    df = pd.DataFrame(results)

    # Add clinical utility indicator
    df["model_useful"] = df["nb_model"] > np.maximum(df["nb_all"], df["nb_none"])

    return df


def standardized_net_benefit(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float,
) -> float:
    """
    Standardized net benefit (sNB) scaled to [0, 1].

    Formula:
        sNB = (NB_model - NB_none) / (NB_perfect - NB_none)

    Where NB_perfect = prevalence (perfect model has no false positives).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes
    y_prob : array-like
        Predicted probabilities
    threshold : float
        Decision threshold

    Returns
    -------
    float
        Standardized net benefit in [0, 1]

    Notes
    -----
    sNB = 0: No better than treat-none
    sNB = 1: Perfect model performance
    """
    y_true = validate_binary_outcomes(y_true)
    y_prob = validate_probabilities(y_prob)
    validate_in_range(threshold, "threshold", 0.0, 1.0, inclusive="neither")

    prevalence = np.mean(y_true)

    nb_model = net_benefit(y_true, y_prob, threshold)
    nb_none = 0.0
    nb_perfect = prevalence  # Perfect sensitivity, no false positives

    if nb_perfect - nb_none < 1e-10:
        return 0.0

    snb = (nb_model - nb_none) / (nb_perfect - nb_none)
    return float(np.clip(snb, 0.0, 1.0))


def optimal_threshold_cost_sensitive(
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> float:
    """
    Compute optimal threshold based on cost ratio.

    Formula:
        p_t* = C_FP / (C_FP + C_FN)

    Parameters
    ----------
    cost_fp : float, default 1.0
        Cost of false positive (e.g., unnecessary follow-up testing)
    cost_fn : float, default 1.0
        Cost of false negative (e.g., missed glaucoma, potential vision loss)

    Returns
    -------
    float
        Optimal decision threshold

    Notes
    -----
    For glaucoma screening where missing disease is costly:
    - cost_fn > cost_fp → lower threshold (higher sensitivity)
    - Example: cost_fn = 5, cost_fp = 1 → p_t* = 1/6 ≈ 0.17

    Common clinical scenarios:
    - Screening (high sensitivity): cost_fn >> cost_fp → low threshold
    - Diagnosis (balanced): cost_fn ≈ cost_fp → threshold ≈ 0.5
    - Limited resources (high specificity): cost_fp >> cost_fn → high threshold

    Examples
    --------
    >>> # Glaucoma screening: missing disease 5x worse than false alarm
    >>> threshold = optimal_threshold_cost_sensitive(cost_fp=1.0, cost_fn=5.0)
    >>> print(f"Optimal threshold: {threshold:.1%}")
    Optimal threshold: 16.7%
    """
    if cost_fp < 0 or cost_fn < 0:
        raise ValidationError(
            "costs", "non-negative values", f"cost_fp={cost_fp}, cost_fn={cost_fn}"
        )
    if cost_fp + cost_fn == 0:
        raise ValidationError("costs", "at least one positive cost", "both zero")

    return cost_fp / (cost_fp + cost_fn)
