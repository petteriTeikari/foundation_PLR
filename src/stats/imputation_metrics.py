"""
Reporting metrics for imputation/reconstruction.

Imputation is NOT "just preprocessing" - it's a statistical model with its own
uncertainty that must be reported per MICE guidelines (Van Buuren, 2018).

Cross-references:
- planning/biostatistics-implementation-plan.md (Section 2.0.1)
- appendix-literature-review/section-08-biostatistics.tex

Required metrics for reporting:
1. Reconstruction accuracy (MAE, RMSE)
2. Coverage probability of prediction intervals
3. Fraction missing information (FMI)
4. Imputation variance contribution
5. Downstream sensitivity to imputation choice

References:
- Van Buuren, S. (2018). Flexible Imputation of Missing Data. CRC Press.
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

from ._defaults import DEFAULT_N_BOOTSTRAP
from ._types import StatsResult
from ._validation import validate_array

__all__ = [
    "ImputationMetricsResult",
    "compute_reconstruction_metrics",
    "compute_coverage_probability",
    "compute_fraction_missing_information",
    "compute_imputation_variance_ratio",
    "imputation_report",
]


@dataclass
class ImputationMetricsResult(StatsResult):
    """
    Comprehensive imputation quality metrics.

    Attributes
    ----------
    rmse : float
        Root Mean Squared Error of reconstruction
    mae : float
        Mean Absolute Error of reconstruction
    coverage_90 : float
        Coverage probability at 90% prediction interval
    coverage_95 : float
        Coverage probability at 95% prediction interval
    fmi : float
        Fraction of Missing Information
    imputation_variance_ratio : float
        Ratio of imputation variance to total variance
    """

    rmse: float = 0.0
    mae: float = 0.0
    coverage_90: float = 0.0
    coverage_95: float = 0.0
    fmi: float = 0.0
    imputation_variance_ratio: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ImputationMetricsResult(RMSE={self.rmse:.4f}, MAE={self.mae:.4f}, "
            f"Coverage95={self.coverage_95:.1%})"
        )


def compute_reconstruction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute reconstruction accuracy metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (original before masking)
    y_pred : np.ndarray
        Imputed/reconstructed values
    mask : np.ndarray, optional
        Boolean mask where True = missing (to evaluate only imputed values)
        If None, evaluates all positions

    Returns
    -------
    Dict[str, float]
        Contains: rmse, mae, mse, mape (if applicable)

    Notes
    -----
    When reporting imputation quality in a publication:
    - Report RMSE and MAE with 95% CIs (via bootstrap)
    - Report separately for different missingness patterns if applicable
    - Compare against baseline (e.g., mean imputation, LOCF)
    """
    y_true = validate_array(y_true, name="y_true")
    y_pred = validate_array(y_pred, name="y_pred")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    # Apply mask if provided
    if mask is not None:
        mask = validate_array(mask, name="mask").astype(bool)
        if mask.shape != y_true.shape:
            raise ValueError(f"mask shape {mask.shape} != y_true shape {y_true.shape}")
        y_true_eval = y_true[mask]
        y_pred_eval = y_pred[mask]
    else:
        y_true_eval = y_true.flatten()
        y_pred_eval = y_pred.flatten()

    if len(y_true_eval) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mse": np.nan, "n_evaluated": 0}

    errors = y_pred_eval - y_true_eval

    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # MAPE only if no zeros in true values
    if np.all(np.abs(y_true_eval) > 1e-10):
        mape = np.mean(np.abs(errors / y_true_eval)) * 100
    else:
        mape = np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "mape": float(mape),
        "n_evaluated": len(y_true_eval),
    }


def compute_coverage_probability(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: List[float] = [0.90, 0.95],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute coverage probability of prediction intervals.

    A well-calibrated imputation model should have:
    - 90% of true values within 90% prediction interval
    - 95% of true values within 95% prediction interval

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred_mean : np.ndarray
        Mean of imputation (point estimate)
    y_pred_std : np.ndarray
        Standard deviation of imputation
    confidence_levels : List[float]
        Confidence levels to evaluate
    mask : np.ndarray, optional
        Boolean mask for imputed positions

    Returns
    -------
    Dict[str, float]
        Coverage probability at each confidence level

    Notes
    -----
    Coverage probability is a key calibration metric for imputation:
    - Coverage < nominal: prediction intervals too narrow (overconfident)
    - Coverage > nominal: prediction intervals too wide (underconfident)
    - Coverage ≈ nominal: well-calibrated uncertainty estimates

    Ideally report as: "95% PI coverage: 94.2% (ideal: 95%)"
    """
    y_true = validate_array(y_true, name="y_true")
    y_pred_mean = validate_array(y_pred_mean, name="y_pred_mean")
    y_pred_std = validate_array(y_pred_std, name="y_pred_std")

    # Apply mask
    if mask is not None:
        mask = validate_array(mask, name="mask").astype(bool)
        y_true = y_true[mask]
        y_pred_mean = y_pred_mean[mask]
        y_pred_std = y_pred_std[mask]
    else:
        y_true = y_true.flatten()
        y_pred_mean = y_pred_mean.flatten()
        y_pred_std = y_pred_std.flatten()

    if len(y_true) == 0:
        return {f"coverage_{int(c * 100)}": np.nan for c in confidence_levels}

    coverage = {}
    for conf in confidence_levels:
        # Z-score for confidence level (two-sided)
        z = scipy_stats.norm.ppf((1 + conf) / 2)

        lower = y_pred_mean - z * y_pred_std
        upper = y_pred_mean + z * y_pred_std

        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage[f"coverage_{int(conf * 100)}"] = float(np.mean(in_interval))

    return coverage


def compute_fraction_missing_information(
    within_imputation_variance: float,
    between_imputation_variance: float,
    n_imputations: int,
) -> float:
    """
    Compute Fraction of Missing Information (FMI).

    FMI quantifies how much information is lost due to missing data.
    Used in Rubin's rules for combining multiply imputed estimates.

    Parameters
    ----------
    within_imputation_variance : float
        Average variance within each imputation (Ū)
    between_imputation_variance : float
        Variance of point estimates across imputations (B)
    n_imputations : int
        Number of imputations (m)

    Returns
    -------
    float
        FMI value in [0, 1]
        - FMI ≈ 0: Very little information lost
        - FMI ≈ 1: Nearly all information lost

    Notes
    -----
    FMI = (B + B/m) / T

    where T = Ū + B + B/m (total variance)

    A high FMI indicates:
    - More uncertainty due to missingness
    - Need for more imputations
    - Potentially problematic missingness mechanism (MNAR)

    Reference: Rubin (1987), Section 3.4
    """
    if n_imputations < 2:
        raise ValueError("Need at least 2 imputations to compute FMI")

    B = between_imputation_variance
    U_bar = within_imputation_variance

    # Between-imputation variance contribution with finite-m correction
    B_corrected = B + B / n_imputations

    # Total variance
    T = U_bar + B_corrected

    if T < 1e-10:
        return 0.0

    # Relative increase in variance due to nonresponse (r)
    r = B_corrected / U_bar if U_bar > 1e-10 else np.inf

    # FMI with finite-m correction
    # γ_m = (r + 2/(ν+3)) / (r + 1)
    # Simplified for large degrees of freedom: γ ≈ r / (r + 1)
    fmi = r / (r + 1) if np.isfinite(r) else 1.0

    return float(np.clip(fmi, 0, 1))


def compute_imputation_variance_ratio(
    estimates_per_imputation: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ratio of imputation variance to total variance.

    Parameters
    ----------
    estimates_per_imputation : np.ndarray
        Array of estimates from each imputation (m,) or (m, k) for k parameters

    Returns
    -------
    Dict[str, float]
        Contains:
        - between_variance: Variance across imputations
        - pooled_variance: Pooled estimate variance (if within-variance provided)
        - ratio: Between / (Between + Within)

    Notes
    -----
    A high ratio indicates that imputation method choice strongly affects
    downstream estimates - consider reporting sensitivity analysis.
    """
    estimates = validate_array(estimates_per_imputation, name="estimates")

    if estimates.ndim == 1:
        estimates = estimates.reshape(-1, 1)

    n_imputations, n_params = estimates.shape

    if n_imputations < 2:
        return {
            "between_variance": 0.0,
            "n_imputations": n_imputations,
            "n_params": n_params,
        }

    # Between-imputation variance
    between_var = np.var(estimates, axis=0, ddof=1)

    return {
        "between_variance": float(np.mean(between_var)),
        "between_variance_per_param": between_var.tolist()
        if n_params > 1
        else float(between_var[0]),
        "n_imputations": n_imputations,
        "n_params": n_params,
    }


def imputation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_std: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_state: Optional[int] = None,
) -> ImputationMetricsResult:
    """
    Generate comprehensive imputation quality report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Imputed values (point estimates)
    y_pred_std : np.ndarray, optional
        Standard deviation of imputation (for coverage)
    mask : np.ndarray, optional
        Boolean mask where True = missing/imputed
    n_bootstrap : int, default DEFAULT_N_BOOTSTRAP
        Bootstrap iterations for CIs
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    ImputationMetricsResult
        Comprehensive metrics with CIs

    Examples
    --------
    >>> result = imputation_report(y_true, y_imputed, y_std, mask=missing_mask)
    >>> print(f"RMSE: {result.rmse:.4f}")
    >>> print(f"95% PI Coverage: {result.coverage_95:.1%} (ideal: 95%)")

    For publication, format as:
        "Reconstruction RMSE was 0.045 ± 0.003 (95% CI: [0.039, 0.051]).
         95% prediction interval coverage was 93.2% (ideal: 95%),
         indicating slightly overconfident uncertainty estimates."
    """
    from .bootstrap import percentile_bootstrap_ci

    # Basic reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(y_true, y_pred, mask)

    # Coverage if uncertainty provided
    if y_pred_std is not None:
        coverage = compute_coverage_probability(
            y_true, y_pred, y_pred_std, confidence_levels=[0.90, 0.95], mask=mask
        )
        coverage_90 = coverage.get("coverage_90", np.nan)
        coverage_95 = coverage.get("coverage_95", np.nan)
    else:
        coverage_90 = np.nan
        coverage_95 = np.nan

    # Bootstrap CIs for reconstruction metrics
    np.random.default_rng(random_state)

    if mask is not None:
        mask = mask.astype(bool)
        indices = np.where(mask.flatten())[0]
    else:
        indices = np.arange(y_true.size)

    def rmse_statistic(idx_sample):
        yt = y_true.flatten()[idx_sample]
        yp = y_pred.flatten()[idx_sample]
        return np.sqrt(np.mean((yt - yp) ** 2))

    def mae_statistic(idx_sample):
        yt = y_true.flatten()[idx_sample]
        yp = y_pred.flatten()[idx_sample]
        return np.mean(np.abs(yt - yp))

    if len(indices) >= 10:
        rmse_boot = percentile_bootstrap_ci(
            indices, rmse_statistic, n_bootstrap, random_state=random_state
        )
        mae_boot = percentile_bootstrap_ci(
            indices, mae_statistic, n_bootstrap, random_state=random_state
        )

        rmse_ci = (rmse_boot.ci_lower, rmse_boot.ci_upper)
        mae_ci = (mae_boot.ci_lower, mae_boot.ci_upper)
    else:
        rmse_ci = (np.nan, np.nan)
        mae_ci = (np.nan, np.nan)

    return ImputationMetricsResult(
        rmse=recon_metrics["rmse"],
        mae=recon_metrics["mae"],
        coverage_90=coverage_90,
        coverage_95=coverage_95,
        fmi=np.nan,  # Requires multiple imputations
        imputation_variance_ratio=np.nan,  # Requires multiple imputations
        scalars={
            "rmse": recon_metrics["rmse"],
            "mae": recon_metrics["mae"],
            "mse": recon_metrics["mse"],
            "rmse_ci_lower": rmse_ci[0],
            "rmse_ci_upper": rmse_ci[1],
            "mae_ci_lower": mae_ci[0],
            "mae_ci_upper": mae_ci[1],
            "coverage_90": coverage_90,
            "coverage_95": coverage_95,
            "n_evaluated": recon_metrics["n_evaluated"],
        },
        metadata={
            "n_bootstrap": n_bootstrap,
            "has_uncertainty": y_pred_std is not None,
        },
    )
