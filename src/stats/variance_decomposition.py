"""
Variance decomposition via factorial ANOVA.

Computes the proportion of variance explained by each factor
in a multi-factor experimental design.

REVIEWER CORRECTION: Use standard factorial ANOVA (Type III SS), NOT mixed-effects.
The 409 experimental runs represent configuration-level aggregates where each run
already averages over subjects via cross-validation. The random subject effect is
integrated out through the CV procedure.

Cross-references:
- planning/figure-and-stats-creation-plan.md (STEP 2.1)
- appendix-literature-review/section-08-biostatistics.tex

References:
- Cohen (1988). Statistical Power Analysis for the Behavioral Sciences
- Maxwell & Delaney (2004). Designing Experiments and Analyzing Data
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ._defaults import DEFAULT_N_BOOTSTRAP
from ._exceptions import (
    InsufficientDataError,
    StatsError,
)
from ._types import StatsResult
from ._validation import (
    check_variance,
    validate_dataframe,
    validate_factors,
)

__all__ = [
    "FactorialANOVAResult",
    "AssumptionTestResult",
    "factorial_anova",
    "test_anova_assumptions",
    "compute_effect_size_ci",
]


@dataclass
class AssumptionTestResult:
    """
    Result of ANOVA assumption tests.

    Attributes
    ----------
    normality_statistic : float
        Shapiro-Wilk W statistic on residuals
    normality_pvalue : float
        p-value for normality test
    normality_passed : bool
        Whether normality assumption is met (p > 0.05)
    homogeneity_statistic : float
        Levene's test statistic
    homogeneity_pvalue : float
        p-value for homogeneity test
    homogeneity_passed : bool
        Whether homogeneity assumption is met (p > 0.05)
    warnings : list
        List of assumption violation warnings
    """

    normality_statistic: float = 0.0
    normality_pvalue: float = 0.0
    normality_passed: bool = True
    homogeneity_statistic: float = 0.0
    homogeneity_pvalue: float = 0.0
    homogeneity_passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = (
            "OK" if (self.normality_passed and self.homogeneity_passed) else "VIOLATED"
        )
        return f"AssumptionTestResult(status={status}, normality_p={self.normality_pvalue:.4f}, homogeneity_p={self.homogeneity_pvalue:.4f})"


@dataclass
class FactorialANOVAResult(StatsResult):
    """
    Comprehensive factorial ANOVA result.

    Attributes
    ----------
    anova_table : pd.DataFrame
        ANOVA table with SS, df, MS, F, p for each factor
    effect_sizes : dict
        Partial eta-squared and omega-squared for each factor
    effect_size_ci : dict
        90% CI for partial eta-squared (per convention)
    assumptions : AssumptionTestResult
        Results of assumption tests
    n_observations : int
        Total number of observations
    r_squared : float
        Overall R-squared (proportion of variance explained by model)
    """

    anova_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    effect_sizes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    effect_size_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    assumptions: Optional[AssumptionTestResult] = None
    n_observations: int = 0
    r_squared: float = 0.0

    def __repr__(self) -> str:
        factors = list(self.effect_sizes.keys())
        return f"FactorialANOVAResult(factors={factors}, R²={self.r_squared:.3f})"

    def to_latex(self, caption: str = "Variance Decomposition") -> str:
        """Generate LaTeX table for ANOVA results."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Factor & SS & df & MS & F & p & $\eta^2_p$ & $\omega^2$ \\",
            r"\midrule",
        ]

        for factor, sizes in self.effect_sizes.items():
            if factor in self.anova_table.index:
                row = self.anova_table.loc[factor]
                # Format p-value
                p_val = row.get("PR(>F)", row.get("p-value", np.nan))
                if pd.notna(p_val):
                    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
                else:
                    p_str = "-"

                ss = row.get("sum_sq", row.get("SS", 0))
                df = row.get("df", 0)
                ms = ss / df if df > 0 else 0
                f_val = row.get("F", 0)

                lines.append(
                    f"{factor} & {ss:.2f} & {int(df)} & {ms:.4f} & "
                    f"{f_val:.2f} & {p_str} & "
                    f"{sizes['partial_eta_sq']:.3f} & {sizes['omega_sq']:.3f} \\\\"
                )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)


def factorial_anova(
    data: pd.DataFrame,
    dv: str,
    factors: List[str],
    ss_type: int = 3,
    include_interactions: bool = True,
    compute_assumptions: bool = True,
    alpha: float = 0.05,
) -> FactorialANOVAResult:
    """
    Perform factorial ANOVA with Type III sum of squares.

    Parameters
    ----------
    data : pd.DataFrame
        Data with dependent variable and factor columns
    dv : str
        Name of the dependent variable column
    factors : list of str
        Names of factor columns (independent variables)
    ss_type : int, default 3
        Type of sum of squares (1, 2, or 3). Type III recommended for
        unbalanced designs.
    include_interactions : bool, default True
        Whether to include all two-way and three-way interactions
    compute_assumptions : bool, default True
        Whether to test ANOVA assumptions
    alpha : float, default 0.05
        Significance level for effect size CI (90% CI by convention)

    Returns
    -------
    FactorialANOVAResult
        Comprehensive ANOVA results including effect sizes and assumptions

    Raises
    ------
    ValidationError
        If data is invalid or missing required columns
    InsufficientDataError
        If insufficient observations for the model

    Notes
    -----
    Effect size interpretation (Cohen, 1988):
        Partial η² < 0.01: negligible
        0.01 ≤ η² < 0.06: small
        0.06 ≤ η² < 0.14: medium
        η² ≥ 0.14: large

    Omega-squared (ω²) provides a less biased estimate than η².

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'auroc': [0.85, 0.87, 0.82, 0.88, 0.90, 0.86],
    ...     'outlier': ['IQR', 'IQR', 'MAD', 'MAD', 'ZScore', 'ZScore'],
    ...     'imputation': ['Mean', 'SAITS', 'Mean', 'SAITS', 'Mean', 'SAITS'],
    ... })
    >>> result = factorial_anova(df, 'auroc', ['outlier', 'imputation'])
    >>> print(result.effect_sizes['outlier']['partial_eta_sq'])
    """
    # Validate inputs
    required_cols = [dv] + factors
    data = validate_dataframe(data, required_cols, name="data")
    validate_factors(data, factors)

    # Check variance in DV
    check_variance(data[dv].values, name=dv)

    # Need minimum observations per cell
    n_obs = len(data)
    min_obs = 2 * (len(factors) + 1)  # Rule of thumb
    if n_obs < min_obs:
        raise InsufficientDataError(
            required=min_obs,
            actual=n_obs,
            context=f"Factorial ANOVA with {len(factors)} factors",
        )

    # Build model formula
    formula = _build_formula(dv, factors, include_interactions)

    # Perform ANOVA using statsmodels
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        raise ImportError("statsmodels is required for factorial ANOVA")

    # Fit OLS model
    try:
        model = ols(formula, data=data).fit()
    except Exception as e:
        raise StatsError(f"OLS model fitting failed: {e}")

    # Compute ANOVA table with specified SS type
    try:
        anova_table = anova_lm(model, typ=ss_type)
    except Exception as e:
        raise StatsError(f"ANOVA computation failed: {e}")

    # Compute effect sizes for each factor
    effect_sizes = {}
    effect_size_ci = {}

    # Get residual SS and df
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    df_residual = anova_table.loc["Residual", "df"]
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    ss_total = anova_table["sum_sq"].sum()

    for term in anova_table.index:
        if term == "Residual":
            continue

        ss_effect = anova_table.loc[term, "sum_sq"]
        df_effect = anova_table.loc[term, "df"]

        # Partial eta-squared: SS_effect / (SS_effect + SS_error)
        partial_eta_sq = (
            ss_effect / (ss_effect + ss_residual)
            if (ss_effect + ss_residual) > 0
            else 0
        )

        # Omega-squared: (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
        omega_sq = (ss_effect - df_effect * ms_residual) / (ss_total + ms_residual)
        omega_sq = max(0.0, omega_sq)  # Clip negative values

        effect_sizes[term] = {
            "partial_eta_sq": partial_eta_sq,
            "omega_sq": omega_sq,
            "ss": ss_effect,
            "df": int(df_effect),
            "f_value": anova_table.loc[term, "F"],
            "p_value": anova_table.loc[term, "PR(>F)"],
        }

        # Bootstrap CI for partial eta-squared
        ci_lower, ci_upper = compute_effect_size_ci(
            data,
            dv,
            factors,
            term,
            n_bootstrap=1000,
            ci_level=0.90,  # 90% CI per convention for effect sizes
        )
        effect_size_ci[term] = (ci_lower, ci_upper)

    # Compute overall R-squared
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    # Test assumptions if requested
    assumptions = None
    if compute_assumptions:
        residuals = model.resid
        assumptions = test_anova_assumptions(data, dv, factors, residuals)

    return FactorialANOVAResult(
        anova_table=anova_table,
        effect_sizes=effect_sizes,
        effect_size_ci=effect_size_ci,
        assumptions=assumptions,
        n_observations=n_obs,
        r_squared=r_squared,
        scalars={
            "r_squared": r_squared,
            "ss_residual": ss_residual,
            "df_residual": df_residual,
            "n_observations": n_obs,
        },
        metadata={
            "dv": dv,
            "factors": factors,
            "ss_type": ss_type,
            "include_interactions": include_interactions,
            "formula": formula,
        },
    )


def test_anova_assumptions(
    data: pd.DataFrame,
    dv: str,
    factors: List[str],
    residuals: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> AssumptionTestResult:
    """
    Test ANOVA assumptions: normality and homogeneity of variance.

    Parameters
    ----------
    data : pd.DataFrame
        Data with dependent variable and factor columns
    dv : str
        Name of the dependent variable column
    factors : list of str
        Names of factor columns
    residuals : np.ndarray, optional
        Model residuals. If None, computed from group means.
    alpha : float, default 0.05
        Significance level for assumption tests

    Returns
    -------
    AssumptionTestResult
        Results of normality and homogeneity tests

    Notes
    -----
    Normality is tested on residuals using Shapiro-Wilk test.
    For n > 5000, uses D'Agostino-Pearson test instead.

    Homogeneity is tested using Levene's test (median-based, robust).
    """
    warnings_list = []

    # Compute residuals if not provided
    if residuals is None:
        # Simple residuals from group means
        groups = data.groupby(factors)[dv].transform("mean")
        residuals = data[dv].values - groups.values

    residuals = np.asarray(residuals)
    n_resid = len(residuals)

    # Test normality on residuals
    if n_resid <= 5000:
        # Shapiro-Wilk test
        norm_stat, norm_pvalue = scipy_stats.shapiro(residuals)
    else:
        # D'Agostino-Pearson for large samples
        norm_stat, norm_pvalue = scipy_stats.normaltest(residuals)

    normality_passed = norm_pvalue > alpha
    if not normality_passed:
        warnings_list.append(
            f"Normality assumption violated (p={norm_pvalue:.4f}). "
            "ANOVA is robust to moderate violations with balanced designs."
        )

    # Test homogeneity of variance using Levene's test
    # Group data by the first factor for simplicity
    # For full factorial, could use all factor combinations
    primary_factor = factors[0]
    groups = [group[dv].values for _, group in data.groupby(primary_factor)]

    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        lev_stat, lev_pvalue = scipy_stats.levene(*groups, center="median")
    else:
        lev_stat, lev_pvalue = np.nan, 1.0
        warnings_list.append(
            "Could not compute Levene's test: insufficient group sizes"
        )

    homogeneity_passed = lev_pvalue > alpha if np.isfinite(lev_pvalue) else True
    if not homogeneity_passed:
        warnings_list.append(
            f"Homogeneity assumption violated (Levene p={lev_pvalue:.4f}). "
            "Consider Welch's ANOVA or non-parametric alternatives."
        )

    return AssumptionTestResult(
        normality_statistic=float(norm_stat),
        normality_pvalue=float(norm_pvalue),
        normality_passed=normality_passed,
        homogeneity_statistic=float(lev_stat) if np.isfinite(lev_stat) else 0.0,
        homogeneity_pvalue=float(lev_pvalue) if np.isfinite(lev_pvalue) else 1.0,
        homogeneity_passed=homogeneity_passed,
        warnings=warnings_list,
    )


def compute_effect_size_ci(
    data: pd.DataFrame,
    dv: str,
    factors: List[str],
    term: str,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = 0.90,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap CI for partial eta-squared.

    Parameters
    ----------
    data : pd.DataFrame
        Original data
    dv : str
        Dependent variable
    factors : list of str
        Factor names
    term : str
        Term to compute CI for
    n_bootstrap : int, default DEFAULT_N_BOOTSTRAP
        Number of bootstrap samples
    ci_level : float, default 0.90
        Confidence level (90% is convention for effect sizes)
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (ci_lower, ci_upper) for partial eta-squared
    """
    # Import once outside the loop for performance
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    n = len(data)
    formula = _build_formula(dv, factors, include_interactions=True)

    boot_eta_sq = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(n, size=n, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)

        try:
            # Compute partial eta-squared for this bootstrap sample
            model = ols(formula, data=boot_data).fit()
            anova_table = anova_lm(model, typ=3)

            if term in anova_table.index:
                ss_effect = anova_table.loc[term, "sum_sq"]
                ss_residual = anova_table.loc["Residual", "sum_sq"]
                eta_sq = (
                    ss_effect / (ss_effect + ss_residual)
                    if (ss_effect + ss_residual) > 0
                    else 0
                )
                boot_eta_sq.append(eta_sq)
        except Exception:
            # Skip failed bootstrap iterations
            continue

    if len(boot_eta_sq) < n_bootstrap * 0.5:
        # Too many failures
        return (np.nan, np.nan)

    boot_eta_sq = np.array(boot_eta_sq)
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_eta_sq, 100 * alpha / 2)
    ci_upper = np.percentile(boot_eta_sq, 100 * (1 - alpha / 2))

    return (ci_lower, ci_upper)


def _build_formula(
    dv: str,
    factors: List[str],
    include_interactions: bool = True,
) -> str:
    """
    Build formula string for statsmodels.

    Parameters
    ----------
    dv : str
        Dependent variable name
    factors : list of str
        Factor names
    include_interactions : bool
        Whether to include interactions

    Returns
    -------
    str
        Formula string (e.g., "auroc ~ C(outlier) + C(imputation) + C(outlier):C(imputation)")
    """
    # Wrap factors in C() for categorical
    categorical_factors = [f"C({f})" for f in factors]

    if include_interactions:
        # Full factorial: main effects + all interactions
        # Use * operator which expands to all terms
        rhs = " * ".join(categorical_factors)
    else:
        # Main effects only
        rhs = " + ".join(categorical_factors)

    return f"{dv} ~ {rhs}"
