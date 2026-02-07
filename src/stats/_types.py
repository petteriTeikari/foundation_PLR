"""
Standard types for statistical computations.

Provides consistent return types across all statistics modules.

Cross-references:
- planning/statistics-implementation.md (Section 1.2)
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class StatsResult:
    """
    Standard return type for all statistical computations.

    Attributes
    ----------
    scalars : dict[str, float]
        Single numeric values (e.g., effect_size, p_value)
    arrays : dict[str, np.ndarray]
        Numpy arrays (e.g., bootstrap_samples, ci_bounds)
    metadata : dict[str, Any]
        Additional information (e.g., n_samples, method, warnings)

    Examples
    --------
    >>> result = StatsResult(
    ...     scalars={'cohens_d': 0.5, 'p_value': 0.02},
    ...     arrays={'bootstrap_samples': np.array([0.4, 0.5, 0.6])},
    ...     metadata={'n1': 50, 'n2': 50, 'method': 'hedges'}
    ... )
    >>> result.scalars['cohens_d']
    0.5
    """

    scalars: Dict[str, float] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dictionary.

        Converts numpy arrays to Python lists for JSON compatibility.

        Returns
        -------
        dict
            Dictionary with keys 'scalars', 'arrays', and 'metadata'.
            Arrays are converted to nested lists.
        """
        return {
            "scalars": self.scalars,
            "arrays": {k: v.tolist() for k, v in self.arrays.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StatsResult":
        """
        Reconstruct a StatsResult from a dictionary.

        Inverse of `to_dict()`. Converts lists back to numpy arrays.

        Parameters
        ----------
        d : dict
            Dictionary with keys 'scalars', 'arrays', and 'metadata'.
            Arrays should be nested lists (as produced by `to_dict()`).

        Returns
        -------
        StatsResult
            Reconstructed StatsResult instance.
        """
        return cls(
            scalars=d.get("scalars", {}),
            arrays={k: np.array(v) for k, v in d.get("arrays", {}).items()},
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the StatsResult.

        Returns
        -------
        str
            Formatted string showing all scalar values.
        """
        scalar_str = ", ".join(f"{k}={v:.4f}" for k, v in self.scalars.items())
        return f"StatsResult({scalar_str})"


@dataclass
class EffectSizeResult(StatsResult):
    """
    Effect size with confidence interval.

    Attributes
    ----------
    effect_size : float
        Point estimate of effect size
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    interpretation : str
        Qualitative interpretation (e.g., "small", "medium", "large")
    """

    effect_size: float = 0.0
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    interpretation: str = ""

    def __repr__(self) -> str:
        """
        Return a string representation of the EffectSizeResult.

        Returns
        -------
        str
            Formatted string showing effect size, 95% CI, and interpretation.
        """
        return (
            f"EffectSizeResult(d={self.effect_size:.3f}, "
            f"95%CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
            f"interpretation='{self.interpretation}')"
        )


@dataclass
class ANOVAResult(StatsResult):
    """
    ANOVA or mixed-effects model result.

    Attributes
    ----------
    f_statistic : float
        F-statistic for the test
    p_value : float
        p-value (uncorrected)
    df_between : int
        Degrees of freedom between groups
    df_within : int
        Degrees of freedom within groups (error)
    partial_eta_sq : float
        Partial eta-squared effect size
    omega_sq : float
        Omega-squared effect size (less biased)
    """

    f_statistic: float = 0.0
    p_value: float = 1.0
    df_between: int = 0
    df_within: int = 0
    partial_eta_sq: float = 0.0
    omega_sq: float = 0.0

    def __repr__(self) -> str:
        """
        Return a string representation of the ANOVAResult.

        Returns
        -------
        str
            Formatted string showing F-statistic, p-value, and partial eta-squared.
        """
        return (
            f"ANOVAResult(F={self.f_statistic:.2f}, "
            f"p={self.p_value:.4f}, "
            f"η²_p={self.partial_eta_sq:.3f})"
        )


@dataclass
class CalibrationResult(StatsResult):
    """
    Calibration assessment result (STRATOS-compliant).

    Attributes
    ----------
    slope : float
        Calibration slope (target = 1.0)
        - slope < 1: overfitting (predictions too extreme)
        - slope > 1: underfitting (predictions too conservative)
    intercept : float
        Calibration intercept (calibration-in-the-large)
    o_e_ratio : float
        Observed:Expected ratio (Van Calster 2024 standard, target = 1.0)
        - O:E > 1: under-prediction of risk (observed > predicted)
        - O:E < 1: over-prediction of risk (observed < predicted)
    brier_score : float
        Brier score (overall calibration + discrimination)
    """

    slope: float = 0.0
    intercept: float = 0.0
    o_e_ratio: float = 0.0
    brier_score: float = 0.0

    def __repr__(self) -> str:
        """
        Return a string representation of the CalibrationResult.

        Returns
        -------
        str
            Formatted string showing calibration slope, O:E ratio, and Brier score.
        """
        return (
            f"CalibrationResult(slope={self.slope:.3f}, "
            f"O:E={self.o_e_ratio:.3f}, "
            f"Brier={self.brier_score:.4f})"
        )


@dataclass
class FDRResult:
    """
    Multiple comparison correction result.

    Attributes
    ----------
    p_values : np.ndarray
        Original p-values
    p_adjusted : np.ndarray
        Adjusted p-values
    reject : np.ndarray
        Boolean array indicating which hypotheses to reject
    method : str
        Correction method used
    alpha : float
        Significance level
    n_rejected : int
        Number of rejected hypotheses
    """

    p_values: np.ndarray = field(default_factory=lambda: np.array([]))
    p_adjusted: np.ndarray = field(default_factory=lambda: np.array([]))
    reject: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = ""
    alpha: float = 0.05
    n_rejected: int = 0

    def __repr__(self) -> str:
        """
        Return a string representation of the FDRResult.

        Returns
        -------
        str
            Formatted string showing correction method, number of tests,
            and number of rejected hypotheses.
        """
        return (
            f"FDRResult(method='{self.method}', "
            f"n_tests={len(self.p_values)}, "
            f"n_rejected={self.n_rejected})"
        )


@dataclass
class BootstrapResult(StatsResult):
    """
    Bootstrap inference result.

    Attributes
    ----------
    point_estimate : float
        Original sample statistic
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    se : float
        Bootstrap standard error
    bias : float
        Bootstrap bias estimate
    method : str
        CI method used (e.g., 'bca', 'percentile')
    n_bootstrap : int
        Number of bootstrap samples
    """

    point_estimate: float = 0.0
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    se: float = np.nan
    bias: float = np.nan
    method: str = ""
    n_bootstrap: int = 0

    def __repr__(self) -> str:
        """
        Return a string representation of the BootstrapResult.

        Returns
        -------
        str
            Formatted string showing point estimate, 95% CI, and CI method.
        """
        return (
            f"BootstrapResult(est={self.point_estimate:.4f}, "
            f"95%CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"method='{self.method}')"
        )
