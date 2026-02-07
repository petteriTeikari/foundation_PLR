"""
Multiple comparison correction methods.

Provides Benjamini-Hochberg FDR, Bonferroni, and Holm-Bonferroni corrections.

Cross-references:
- planning/statistics-implementation.md (Section 2.2)
- appendix-literature-review/section-08-biostatistics.tex

References:
- Benjamini & Hochberg (1995). Controlling the False Discovery Rate.
- Holm (1979). A Simple Sequentially Rejective Multiple Test Procedure.
"""

from typing import Union

import numpy as np
from statsmodels.stats.multitest import multipletests

from ._exceptions import ValidationError
from ._types import FDRResult
from ._validation import validate_array

__all__ = [
    "benjamini_hochberg",
    "bonferroni",
    "holm",
    "apply_fdr_correction",
]


def benjamini_hochberg(
    p_values: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> FDRResult:
    """
    Benjamini-Hochberg FDR correction.

    Controls the False Discovery Rate (expected proportion of false
    discoveries among rejected hypotheses).

    Algorithm:
        1. Order p-values: p(1) <= p(2) <= ... <= p(m)
        2. Find largest k such that p(k) <= (k/m) × α
        3. Reject all H(i) for i = 1, ..., k

    Adjusted p-value: p_adj(i) = min(p(i) × m/rank(i), 1)

    Parameters
    ----------
    p_values : array-like
        Original p-values (must be in [0, 1])
    alpha : float, default 0.05
        Significance level for controlling FDR

    Returns
    -------
    FDRResult
        Contains p_values, p_adjusted, reject, method, alpha, n_rejected

    Raises
    ------
    ValidationError
        If p-values are empty or outside [0, 1]

    Notes
    -----
    - Controls FDR at level α (expected proportion of false positives)
    - More powerful than FWER methods (Bonferroni, Holm) for many tests
    - Recommended when many tests are expected to be truly positive
    - For factorial analysis with 13 comparisons, FDR is appropriate

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.02, 0.04, 0.05, 0.1, 0.5]
    >>> result = benjamini_hochberg(p_values)
    >>> print(f"Rejected: {result.n_rejected} of {len(p_values)}")
    Rejected: 5 of 7
    """
    p_values = _validate_p_values(p_values)

    # Use statsmodels for robust implementation
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    return FDRResult(
        p_values=p_values,
        p_adjusted=p_adjusted,
        reject=reject,
        method="benjamini-hochberg",
        alpha=alpha,
        n_rejected=int(np.sum(reject)),
    )


def bonferroni(
    p_values: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> FDRResult:
    """
    Bonferroni correction for multiple comparisons.

    Controls the Family-Wise Error Rate (FWER) - probability of at least
    one false positive among all tests.

    Formula: p_adj = p × m (capped at 1.0)

    Parameters
    ----------
    p_values : array-like
        Original p-values
    alpha : float, default 0.05
        Significance level for controlling FWER

    Returns
    -------
    FDRResult
        Contains corrected p-values and rejection decisions

    Notes
    -----
    - Very conservative, especially for many tests
    - Controls FWER at level α (probability of ANY false positive)
    - Use when false positives are very costly
    - For 13 comparisons at α=0.05: effective α = 0.0038

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.05]
    >>> result = bonferroni(p_values)
    >>> result.p_adjusted  # [0.003, 0.03, 0.15]
    """
    p_values = _validate_p_values(p_values)

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="bonferroni")

    return FDRResult(
        p_values=p_values,
        p_adjusted=p_adjusted,
        reject=reject,
        method="bonferroni",
        alpha=alpha,
        n_rejected=int(np.sum(reject)),
    )


def holm(
    p_values: Union[np.ndarray, list],
    alpha: float = 0.05,
) -> FDRResult:
    """
    Holm-Bonferroni step-down procedure.

    A sequentially rejective method that is uniformly more powerful
    than Bonferroni while still controlling FWER.

    Algorithm:
        1. Order p-values: p(1) <= p(2) <= ... <= p(m)
        2. For k = 1 to m:
           - If p(k) > α/(m-k+1), accept H(k) and all remaining
           - Else reject H(k) and continue

    Parameters
    ----------
    p_values : array-like
        Original p-values
    alpha : float, default 0.05
        Significance level for controlling FWER

    Returns
    -------
    FDRResult
        Contains corrected p-values and rejection decisions

    Notes
    -----
    - Controls FWER like Bonferroni but more powerful
    - Always rejects at least as many hypotheses as Bonferroni
    - Good middle ground when FWER control is needed but Bonferroni
      is too conservative

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.04, 0.05]
    >>> result = holm(p_values)
    >>> result.n_rejected  # More than Bonferroni
    """
    p_values = _validate_p_values(p_values)

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="holm")

    return FDRResult(
        p_values=p_values,
        p_adjusted=p_adjusted,
        reject=reject,
        method="holm",
        alpha=alpha,
        n_rejected=int(np.sum(reject)),
    )


def apply_fdr_correction(
    p_values: Union[np.ndarray, list],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> FDRResult:
    """
    Apply multiple comparison correction with specified method.

    Convenience function that dispatches to appropriate correction method.

    Parameters
    ----------
    p_values : array-like
        Original p-values
    method : str, default 'fdr_bh'
        Correction method:
        - 'fdr_bh': Benjamini-Hochberg FDR
        - 'bonferroni': Bonferroni FWER
        - 'holm': Holm-Bonferroni FWER
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    FDRResult
        Corrected p-values and decisions

    Examples
    --------
    >>> result = apply_fdr_correction([0.001, 0.01, 0.05], method='fdr_bh')
    """
    method_map = {
        "fdr_bh": benjamini_hochberg,
        "benjamini-hochberg": benjamini_hochberg,
        "bonferroni": bonferroni,
        "holm": holm,
    }

    if method.lower() not in method_map:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(method_map.keys())}"
        )

    return method_map[method.lower()](p_values, alpha=alpha)


def _validate_p_values(p_values: Union[np.ndarray, list]) -> np.ndarray:
    """Validate and convert p-values array."""
    p_values = validate_array(p_values, name="p_values")

    if len(p_values) == 0:
        raise ValidationError("p_values", "non-empty array", "empty array")

    if np.any(p_values < 0) or np.any(p_values > 1):
        min_p = np.min(p_values)
        max_p = np.max(p_values)
        raise ValidationError(
            "p_values", "values in [0, 1]", f"values in [{min_p:.4f}, {max_p:.4f}]"
        )

    return p_values
