"""
Bootstrap inference methods.

Provides percentile and BCa (bias-corrected and accelerated) bootstrap CIs.

Cross-references:
- planning/statistics-implementation.md (Section 2.4)
- appendix-literature-review/section-08-biostatistics.tex

References:
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
- DiCiccio & Efron (1996). Bootstrap confidence intervals.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy import stats as scipy_stats

from ._types import BootstrapResult
from ._validation import validate_array, validate_min_samples, validate_positive

__all__ = [
    "bca_bootstrap_ci",
    "percentile_bootstrap_ci",
    "stratified_bootstrap_sample",
    "bootstrap_se",
]


def bca_bootstrap_ci(
    data: Union[np.ndarray, list],
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa corrects for:
    1. Bias: Systematic difference between bootstrap distribution and true distribution
    2. Acceleration: Rate of change of SE with parameter value (skewness correction)

    Parameters
    ----------
    data : array-like
        Original data sample
    statistic_fn : callable
        Function that computes statistic from sample: statistic_fn(data) -> float
    n_bootstrap : int, default 2000
        Number of bootstrap samples
    alpha : float, default 0.05
        Significance level (0.05 = 95% CI)
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    BootstrapResult
        Contains point_estimate, ci_lower, ci_upper, se, bias, method, n_bootstrap

    Raises
    ------
    InsufficientDataError
        If data has fewer than 3 observations

    Notes
    -----
    BCa is preferred over percentile method when:
    - The statistic has non-constant variance
    - The sampling distribution is skewed
    - Accurate coverage probability is important

    The BCa interval has second-order accuracy, meaning coverage error
    is O(n^-1) rather than O(n^-0.5) for percentile method.

    Examples
    --------
    >>> data = np.random.normal(loc=5.0, scale=2.0, size=50)
    >>> result = bca_bootstrap_ci(data, np.mean)
    >>> print(f"Mean: {result.point_estimate:.3f} "
    ...       f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    """
    data = validate_array(data, name="data")
    validate_min_samples(
        data, min_n=3, name="data", context="BCa bootstrap requires n >= 3"
    )
    validate_positive(n_bootstrap, "n_bootstrap")

    rng = np.random.default_rng(random_state)
    n = len(data)

    # Original statistic
    theta_hat = statistic_fn(data)

    # Bootstrap distribution
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(boot_sample)

    # Bootstrap SE and bias
    se = np.std(boot_stats, ddof=1)
    bias = np.mean(boot_stats) - theta_hat

    # Bias correction factor z0
    # z0 = Φ^(-1)(proportion of bootstrap estimates < original estimate)
    prop_below = np.mean(boot_stats < theta_hat)
    # Handle edge cases
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = scipy_stats.norm.ppf(prop_below)

    # Acceleration factor a (using jackknife)
    jackknife_stats = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats[i] = statistic_fn(jackknife_sample)

    theta_dot = np.mean(jackknife_stats)
    diff = theta_dot - jackknife_stats

    numerator = np.sum(diff**3)
    denominator = 6 * (np.sum(diff**2) ** 1.5)

    if denominator > 1e-10:
        a = numerator / denominator
    else:
        a = 0.0

    # Adjusted percentiles
    z_alpha_lower = scipy_stats.norm.ppf(alpha / 2)
    z_alpha_upper = scipy_stats.norm.ppf(1 - alpha / 2)

    def adjusted_percentile(z_alpha):
        """Compute BCa-adjusted percentile."""
        numerator = z0 + z_alpha
        denominator = 1 - a * numerator
        if abs(denominator) < 1e-10:
            return 0.5  # Fallback
        adjusted_z = z0 + numerator / denominator
        return scipy_stats.norm.cdf(adjusted_z)

    alpha1 = adjusted_percentile(z_alpha_lower)
    alpha2 = adjusted_percentile(z_alpha_upper)

    # Ensure valid percentiles
    alpha1 = np.clip(alpha1, 1e-6, 1 - 1e-6)
    alpha2 = np.clip(alpha2, 1e-6, 1 - 1e-6)

    ci_lower = np.percentile(boot_stats, 100 * alpha1)
    ci_upper = np.percentile(boot_stats, 100 * alpha2)

    return BootstrapResult(
        point_estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        bias=bias,
        method="bca",
        n_bootstrap=n_bootstrap,
        scalars={
            "point_estimate": theta_hat,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se": se,
            "bias": bias,
        },
        arrays={"bootstrap_distribution": boot_stats},
        metadata={
            "alpha": alpha,
            "z0": z0,
            "acceleration": a,
            "n_data": n,
        },
    )


def percentile_bootstrap_ci(
    data: Union[np.ndarray, list],
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """
    Percentile bootstrap confidence interval.

    Simple method that uses quantiles of bootstrap distribution directly.

    Parameters
    ----------
    data : array-like
        Original data sample
    statistic_fn : callable
        Function that computes statistic from sample
    n_bootstrap : int, default 2000
        Number of bootstrap samples
    alpha : float, default 0.05
        Significance level (0.05 = 95% CI)
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    BootstrapResult
        Contains point_estimate, ci_lower, ci_upper, se

    Notes
    -----
    The percentile method is simpler but may have poor coverage when:
    - The statistic is biased
    - The sampling distribution is skewed

    Consider using BCa method for better accuracy.

    Examples
    --------
    >>> data = np.random.normal(size=50)
    >>> result = percentile_bootstrap_ci(data, np.mean)
    >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    """
    data = validate_array(data, name="data")
    validate_min_samples(data, min_n=2, name="data")
    validate_positive(n_bootstrap, "n_bootstrap")

    rng = np.random.default_rng(random_state)
    n = len(data)

    # Original statistic
    theta_hat = statistic_fn(data)

    # Bootstrap distribution
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(boot_sample)

    # SE and bias
    se = np.std(boot_stats, ddof=1)
    bias = np.mean(boot_stats) - theta_hat

    # Percentile CI
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se=se,
        bias=bias,
        method="percentile",
        n_bootstrap=n_bootstrap,
        scalars={
            "point_estimate": theta_hat,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "se": se,
            "bias": bias,
        },
        arrays={"bootstrap_distribution": boot_stats},
        metadata={
            "alpha": alpha,
            "n_data": n,
        },
    )


def stratified_bootstrap_sample(
    strata: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate stratified bootstrap sample indices.

    Maintains exact class proportions in each bootstrap sample.
    Essential for classification problems with imbalanced classes.

    Parameters
    ----------
    strata : array-like
        Stratum labels (e.g., class labels for classification)
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    np.ndarray
        Bootstrap sample indices

    Notes
    -----
    For binary classification with class imbalance, stratified bootstrap
    ensures each bootstrap sample has the same class ratio as the original.

    Examples
    --------
    >>> y_true = np.array([0, 0, 0, 0, 1, 1])  # 4:2 ratio
    >>> indices = stratified_bootstrap_sample(y_true)
    >>> y_boot = y_true[indices]  # Should have ~4:2 ratio
    """
    if rng is None:
        rng = np.random.default_rng()

    strata = np.asarray(strata)
    len(strata)

    unique_strata = np.unique(strata)
    indices = []

    for stratum in unique_strata:
        stratum_mask = strata == stratum
        stratum_indices = np.where(stratum_mask)[0]
        n_stratum = len(stratum_indices)

        # Sample with replacement within stratum
        boot_indices = rng.choice(stratum_indices, size=n_stratum, replace=True)
        indices.extend(boot_indices)

    return np.array(indices)


def bootstrap_se(
    data: Union[np.ndarray, list],
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    random_state: Optional[int] = None,
) -> float:
    """
    Compute bootstrap standard error of a statistic.

    Parameters
    ----------
    data : array-like
        Original data sample
    statistic_fn : callable
        Function that computes statistic from sample
    n_bootstrap : int, default 2000
        Number of bootstrap samples
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    float
        Bootstrap standard error

    Examples
    --------
    >>> data = np.random.normal(size=50)
    >>> se = bootstrap_se(data, np.median)
    """
    data = validate_array(data, name="data")
    rng = np.random.default_rng(random_state)
    n = len(data)

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(boot_sample)

    return np.std(boot_stats, ddof=1)
