"""
Effect size computations with confidence intervals.

STRATOS/TRIPOD requirement: Report effect sizes alongside p-values.

Cross-references:
- planning/statistics-implementation.md (Section 4.1)
- appendix-literature-review/section-08-biostatistics.tex
- methods-10-statistical-analysis.tex

References:
- Cohen (1988). Statistical Power Analysis for the Behavioral Sciences
- Hedges & Olkin (1985). Statistical Methods for Meta-Analysis
"""

from typing import Optional, Union

import numpy as np
from scipy import stats as scipy_stats

from ._types import EffectSizeResult
from ._validation import validate_array, validate_min_samples

__all__ = [
    "cohens_d",
    "hedges_g",
    "partial_eta_squared",
    "generalized_eta_squared",
    "omega_squared",
    "cohens_f",
]


def cohens_d(
    group1: Union[np.ndarray, list],
    group2: Union[np.ndarray, list],
    hedges: bool = True,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> EffectSizeResult:
    """
    Compute Cohen's d (or Hedges' g) with confidence interval.

    Cohen's d is the standardized mean difference:
        d = (M1 - M2) / s_pooled

    where s_pooled is the pooled standard deviation.

    Parameters
    ----------
    group1 : array-like
        First group values
    group2 : array-like
        Second group values
    hedges : bool, default True
        Apply Hedges' correction for small sample bias.
        Recommended for n < 50 per group.
    ci_method : str, default 'bootstrap'
        Method for confidence interval:
        - 'bootstrap': Bootstrap CI (recommended)
        - 'analytical': Normal approximation CI
        - 'none': No CI computed
    n_bootstrap : int, default 2000
        Number of bootstrap samples for CI
    alpha : float, default 0.05
        Significance level for CI (0.05 = 95% CI)
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    EffectSizeResult
        Contains effect_size (d or g), ci_lower, ci_upper, interpretation

    Raises
    ------
    InsufficientDataError
        If either group has fewer than 2 observations

    Notes
    -----
    Interpretation (Cohen, 1988):
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large

    Hedges' correction:
        g = d × (1 - 3 / (4×df - 1))
    where df = n1 + n2 - 2. This reduces upward bias in small samples.

    Examples
    --------
    >>> group1 = [1, 2, 3, 4, 5]
    >>> group2 = [2, 3, 4, 5, 6]
    >>> result = cohens_d(group1, group2)
    >>> print(f"d = {result.effect_size:.3f} ({result.interpretation})")
    d = -0.632 (medium)
    """
    # Validate inputs
    group1 = validate_array(group1, name="group1")
    group2 = validate_array(group2, name="group2")
    validate_min_samples(group1, min_n=2, name="group1")
    validate_min_samples(group2, min_n=2, name="group2")

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    mean_diff = mean1 - mean2

    # Pooled standard deviation
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    # Handle zero variance case
    if pooled_std < 1e-15:
        if abs(mean_diff) < 1e-15:
            d = 0.0
        else:
            d = np.inf * np.sign(mean_diff)
    else:
        d = mean_diff / pooled_std

    # Hedges' correction for small sample bias
    if hedges and np.isfinite(d):
        df = n1 + n2 - 2
        correction_factor = 1 - 3 / (4 * df - 1)
        d *= correction_factor

    # Confidence interval
    ci_lower, ci_upper = np.nan, np.nan

    if ci_method == "bootstrap" and np.isfinite(d):
        ci_lower, ci_upper = _bootstrap_ci_cohens_d(
            group1, group2, hedges, n_bootstrap, alpha, random_state
        )
    elif ci_method == "analytical" and np.isfinite(d):
        # Approximate SE for Cohen's d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        z = scipy_stats.norm.ppf(1 - alpha / 2)
        ci_lower = d - z * se_d
        ci_upper = d + z * se_d

    # Interpretation based on Cohen's guidelines
    interpretation = _interpret_cohens_d(d)

    return EffectSizeResult(
        effect_size=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
        scalars={
            "cohens_d": d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        metadata={
            "hedges_correction": hedges,
            "ci_method": ci_method,
            "n1": n1,
            "n2": n2,
            "mean1": float(mean1),
            "mean2": float(mean2),
            "pooled_std": float(pooled_std),
            "alpha": alpha,
        },
    )


def hedges_g(
    group1: Union[np.ndarray, list],
    group2: Union[np.ndarray, list],
    **kwargs,
) -> EffectSizeResult:
    """
    Compute Hedges' g (bias-corrected Cohen's d).

    Alias for cohens_d with hedges=True.

    See Also
    --------
    cohens_d : Full documentation
    """
    return cohens_d(group1, group2, hedges=True, **kwargs)


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> EffectSizeResult:
    """
    Compute partial eta-squared for ANOVA.

    η²_p = SS_effect / (SS_effect + SS_error)

    This represents the proportion of variance in the DV explained
    by the effect, controlling for other effects.

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect of interest
    ss_error : float
        Sum of squares for error (residual)

    Returns
    -------
    EffectSizeResult
        Contains effect_size (η²_p) and interpretation

    Raises
    ------
    ValueError
        If sum of squares values are negative

    Notes
    -----
    Interpretation (Cohen, 1988):
        η²_p < 0.01: negligible
        0.01 ≤ η²_p < 0.06: small
        0.06 ≤ η²_p < 0.14: medium
        η²_p ≥ 0.14: large

    Warning: η²_p tends to overestimate population effect size,
    especially with small samples. Consider using omega² instead.

    Examples
    --------
    >>> result = partial_eta_squared(ss_effect=10.0, ss_error=90.0)
    >>> print(f"η²_p = {result.effect_size:.3f} ({result.interpretation})")
    η²_p = 0.100 (medium)
    """
    if ss_effect < 0 or ss_error < 0:
        raise ValueError("Sum of squares must be non-negative")

    total = ss_effect + ss_error
    if total < 1e-15:
        eta_sq = 0.0
    else:
        eta_sq = ss_effect / total

    interpretation = _interpret_eta_squared(eta_sq)

    return EffectSizeResult(
        effect_size=eta_sq,
        ci_lower=np.nan,
        ci_upper=np.nan,
        interpretation=interpretation,
        scalars={"partial_eta_squared": eta_sq},
        metadata={"ss_effect": ss_effect, "ss_error": ss_error},
    )


def generalized_eta_squared(
    ss_effect: float,
    ss_error: float,
    ss_subjects: float = 0.0,
) -> EffectSizeResult:
    """
    Compute generalized eta-squared (Olejnik & Algina, 2003).

    η²_G = SS_effect / (SS_effect + SS_error + SS_subjects)

    More comparable across studies than partial η².

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_error : float
        Sum of squares for error
    ss_subjects : float, default 0.0
        Sum of squares for subjects (for repeated measures)

    Returns
    -------
    EffectSizeResult
        Contains generalized eta-squared

    Notes
    -----
    For between-subjects designs, ss_subjects = 0 and
    generalized η² equals partial η².
    """
    if ss_effect < 0 or ss_error < 0 or ss_subjects < 0:
        raise ValueError("Sum of squares must be non-negative")

    total = ss_effect + ss_error + ss_subjects
    if total < 1e-15:
        eta_sq = 0.0
    else:
        eta_sq = ss_effect / total

    interpretation = _interpret_eta_squared(eta_sq)

    return EffectSizeResult(
        effect_size=eta_sq,
        ci_lower=np.nan,
        ci_upper=np.nan,
        interpretation=interpretation,
        scalars={"generalized_eta_squared": eta_sq},
        metadata={
            "ss_effect": ss_effect,
            "ss_error": ss_error,
            "ss_subjects": ss_subjects,
        },
    )


def omega_squared(
    ss_effect: float,
    ss_error: float,
    df_effect: int,
    ms_error: float,
) -> EffectSizeResult:
    """
    Compute omega-squared (less biased than eta-squared).

    ω² = (SS_effect - df_effect × MS_error) / (SS_total + MS_error)

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect
    ss_error : float
        Sum of squares for error
    df_effect : int
        Degrees of freedom for the effect
    ms_error : float
        Mean square error

    Returns
    -------
    EffectSizeResult
        Contains omega-squared (clipped to 0 if negative)

    Notes
    -----
    Omega-squared provides a less biased estimate of population
    effect size than eta-squared, especially for small samples.

    Can be negative for very small effects; clipped to 0.

    Examples
    --------
    >>> result = omega_squared(ss_effect=10, ss_error=90, df_effect=2, ms_error=1.8)
    >>> print(f"ω² = {result.effect_size:.3f}")
    """
    if ss_effect < 0 or ss_error < 0 or ms_error < 0:
        raise ValueError("Sum of squares and MS_error must be non-negative")

    ss_total = ss_effect + ss_error
    numerator = ss_effect - df_effect * ms_error
    denominator = ss_total + ms_error

    if denominator < 1e-15:
        omega_sq = 0.0
    else:
        omega_sq = numerator / denominator

    # Clip negative values to 0
    omega_sq = max(0.0, omega_sq)

    interpretation = _interpret_eta_squared(omega_sq)

    return EffectSizeResult(
        effect_size=omega_sq,
        ci_lower=np.nan,
        ci_upper=np.nan,
        interpretation=interpretation,
        scalars={"omega_squared": omega_sq},
        metadata={
            "ss_effect": ss_effect,
            "ss_error": ss_error,
            "df_effect": df_effect,
            "ms_error": ms_error,
        },
    )


def cohens_f(eta_squared: float) -> float:
    """
    Convert eta-squared to Cohen's f.

    f = sqrt(η² / (1 - η²))

    Parameters
    ----------
    eta_squared : float
        Eta-squared or partial eta-squared value

    Returns
    -------
    float
        Cohen's f value

    Notes
    -----
    Interpretation (Cohen, 1988):
        f ≈ 0.10: small
        f ≈ 0.25: medium
        f ≈ 0.40: large

    Examples
    --------
    >>> cohens_f(0.06)  # Medium effect
    0.2526...
    """
    if eta_squared >= 1.0:
        return np.inf
    if eta_squared <= 0.0:
        return 0.0

    return np.sqrt(eta_squared / (1 - eta_squared))


# --------------------------------------------------------------------------
# Internal helper functions
# --------------------------------------------------------------------------


def _interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d using standard guidelines.

    Cohen (1988) conventions:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    abs_d = abs(d)
    if not np.isfinite(abs_d):
        return "undefined"
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_eta_squared(eta_sq: float) -> str:
    """
    Interpret eta-squared (or omega-squared) using standard guidelines.

    Cohen (1988) conventions:
        η² < 0.01: negligible
        0.01 ≤ η² < 0.06: small
        0.06 ≤ η² < 0.14: medium
        η² ≥ 0.14: large
    """
    if not np.isfinite(eta_sq):
        return "undefined"
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


def _bootstrap_ci_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    hedges: bool,
    n_bootstrap: int,
    alpha: float,
    random_state: Optional[int] = None,
) -> tuple:
    """
    Compute bootstrap CI for Cohen's d using percentile method.

    For BCa method, use the bootstrap module.
    """
    rng = np.random.default_rng(random_state)
    n1, n2 = len(group1), len(group2)

    boot_d = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample each group independently
        boot_g1 = rng.choice(group1, size=n1, replace=True)
        boot_g2 = rng.choice(group2, size=n2, replace=True)

        # Compute d for bootstrap sample (no CI to avoid recursion)
        try:
            result = cohens_d(boot_g1, boot_g2, hedges=hedges, ci_method="none")
            boot_d[i] = result.effect_size
        except Exception:
            boot_d[i] = np.nan

    # Remove NaN values
    boot_d = boot_d[np.isfinite(boot_d)]

    if len(boot_d) < n_bootstrap * 0.5:
        # Too many failed bootstrap samples
        return np.nan, np.nan

    # Percentile CI
    ci_lower = np.percentile(boot_d, 100 * alpha / 2)
    ci_upper = np.percentile(boot_d, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper
