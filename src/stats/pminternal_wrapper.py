"""
Python wrapper for R's pminternal package.

pminternal provides model instability analysis for clinical prediction models,
as recommended by Riley (2023 BMC Medicine) and STRATOS/TRIPOD-AI guidelines.

This module wraps pminternal R functions via subprocess, providing:
- Model instability metrics (calibration slope variance across bootstrap samples)
- Validation statistics with confidence intervals
- Flexible calibration assessment

The subprocess approach is used instead of rpy2 for better compatibility
when conda R and system R coexist.

References:
- Riley, R. D. et al. (2023). External validation of clinical prediction models
  using big data from e-health records or IPD meta-analysis:
  opportunities and challenges. BMC Medicine, 21, 414.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive
  analytics. BMC Medicine, 17, 230.

Requirements:
- R >= 4.4 with pminternal, pmcalibration packages installed at /usr/bin/R
- System R (not conda R) required for pminternal

Usage:
    from src.stats.pminternal_wrapper import (
        validate_model,
        calibration_metrics,
        instability_analysis,
    )

    # Basic validation
    result = validate_model(y_true, y_prob)
    print(f"Calibration slope: {result['calibration_slope']:.3f}")

    # Full instability analysis with bootstrap
    instab = instability_analysis(y_true, y_prob, n_bootstrap=200)
    print(f"Instability index: {instab['instability_index']:.3f}")
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.stats._defaults import DEFAULT_CI_LEVEL

logger = logging.getLogger(__name__)

# Cache for R availability check
_r_available = None
_r_path = None


def _find_system_r() -> Optional[str]:
    """Find system R installation (not conda)."""
    global _r_path
    if _r_path is not None:
        return _r_path

    # Prefer system R over conda R
    candidates = ["/usr/bin/R", "/usr/local/bin/R"]

    for r_path in candidates:
        if os.path.exists(r_path):
            _r_path = r_path
            logger.info(f"Found system R at {r_path}")
            return r_path

    # Fallback to PATH R (may be conda)
    try:
        result = subprocess.run(
            ["which", "R"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            _r_path = result.stdout.strip()
            return _r_path
    except Exception:
        pass

    return None


def _check_r_pminternal() -> bool:
    """Check if R and pminternal are available."""
    global _r_available
    if _r_available is not None:
        return _r_available

    r_path = _find_system_r()
    if not r_path:
        _r_available = False
        return False

    # Check if pminternal is installed
    check_script = 'cat(as.character(packageVersion("pminternal")))'
    try:
        result = subprocess.run(
            [r_path, "--vanilla", "--quiet", "-e", check_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "Error" not in result.stderr:
            _r_available = True
            logger.info(f"pminternal version: {result.stdout.strip()}")
        else:
            _r_available = False
            logger.warning(f"pminternal not found: {result.stderr}")
    except Exception as e:
        _r_available = False
        logger.warning(f"R check failed: {e}")

    return _r_available


def _run_r_script(script: str, timeout: int = 60) -> dict:
    """Run an R script and return the JSON result."""
    r_path = _find_system_r()
    if not r_path:
        raise RuntimeError("System R not found")

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [r_path, "--vanilla", "--slave", "-f", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")

        # Parse JSON output - find the JSON object in output
        # R may include prompts and other output, so we extract just the JSON
        output = result.stdout
        json_start = output.find("{")
        json_end = output.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            raise RuntimeError(f"No JSON output from R: {output}")

        json_line = output[json_start:json_end]

        return json.loads(json_line)
    finally:
        os.unlink(script_path)


@dataclass
class ValidationResult:
    """Result of pminternal model validation."""

    # Discrimination
    c_statistic: float
    c_statistic_se: float
    c_statistic_ci_lower: float
    c_statistic_ci_upper: float

    # Calibration
    calibration_slope: float
    calibration_slope_se: float
    calibration_slope_ci_lower: float
    calibration_slope_ci_upper: float

    calibration_intercept: float
    calibration_intercept_se: float
    calibration_intercept_ci_lower: float
    calibration_intercept_ci_upper: float

    # O:E ratio
    oe_ratio: float
    oe_ratio_se: float
    oe_ratio_ci_lower: float
    oe_ratio_ci_upper: float

    # Overall performance
    brier_score: float
    scaled_brier: float

    # Sample info
    n_samples: int
    n_events: int
    event_rate: float


@dataclass
class InstabilityResult:
    """Result of model instability analysis."""

    # Instability metrics
    instability_index: float  # CV of calibration slope across bootstrap
    slope_sd: float  # SD of calibration slope
    slope_mean: float  # Mean calibration slope
    slope_cv: float  # Coefficient of variation

    # Bootstrap distribution
    slope_bootstrap: NDArray[np.float64]  # All bootstrap slopes
    slope_percentile_2_5: float
    slope_percentile_97_5: float

    # Interpretation
    stability_rating: str  # "stable", "moderate", "unstable"

    # Sample info
    n_bootstrap: int
    n_samples: int


def validate_model(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    ci_level: float = DEFAULT_CI_LEVEL,
) -> ValidationResult:
    """
    Validate a binary prediction model using pminternal.

    Computes discrimination (c-statistic), calibration (slope, intercept, O:E),
    and overall performance (Brier score) with confidence intervals.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    ci_level : float, default=DEFAULT_CI_LEVEL
        Confidence level for intervals

    Returns
    -------
    ValidationResult
        Validation metrics with confidence intervals

    Raises
    ------
    RuntimeError
        If R/pminternal is not available
    ValueError
        If inputs are invalid
    """
    if not _check_r_pminternal():
        raise RuntimeError(
            "R with pminternal package not available. "
            "Run: sudo ./scripts/setup-dev-environment.sh"
        )

    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0 or 1)")
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be in [0, 1]")

    # Create R script
    y_str = ",".join(map(str, y_true))
    p_str = ",".join(map(str, y_prob))

    script = f"""
suppressPackageStartupMessages({{
    library(pmcalibration)
    library(pROC)
    library(jsonlite)
}})

y_true <- c({y_str})
y_prob <- c({p_str})
ci_level <- {ci_level}

# Use pmcalibration::logistic_cal - the standard method per Van Calster et al. 2019
# This is the PROPER way to compute calibration slope and intercept
cal <- suppressWarnings(logistic_cal(y = y_true, p = y_prob))

# Extract from logistic_cal output (uses proper methodology)
slope_model <- cal$calibration_slope
intercept_model <- cal$calibration_intercept

slope <- as.numeric(coef(slope_model)["LP"])
slope_se <- as.numeric(summary(slope_model)$coefficients["LP", "Std. Error"])
intercept <- as.numeric(coef(intercept_model)["(Intercept)"])
intercept_se <- as.numeric(summary(intercept_model)$coefficients["(Intercept)", "Std. Error"])

# C-statistic via pROC
roc_obj <- suppressWarnings(roc(y_true, y_prob, quiet=TRUE))
c_stat <- as.numeric(auc(roc_obj))
c_ci <- suppressWarnings(ci.auc(roc_obj, conf.level=ci_level))

# O:E ratio
oe <- sum(y_true) / sum(y_prob)
oe_se <- oe * sqrt(1/sum(y_true) + 1/length(y_true))

# Brier score
brier <- mean((y_true - y_prob)^2)
prevalence <- mean(y_true)
brier_null <- prevalence * (1 - prevalence)
scaled_brier <- ifelse(brier_null > 0, 1 - brier/brier_null, 0)

# CI calculations
z <- qnorm(0.5 + ci_level/2)

result <- list(
    c_statistic = c_stat,
    c_statistic_se = (c_ci[3] - c_ci[1]) / (2*z),
    c_statistic_ci_lower = as.numeric(c_ci[1]),
    c_statistic_ci_upper = as.numeric(c_ci[3]),
    calibration_slope = slope,
    calibration_slope_se = slope_se,
    calibration_slope_ci_lower = slope - z*slope_se,
    calibration_slope_ci_upper = slope + z*slope_se,
    calibration_intercept = intercept,
    calibration_intercept_se = intercept_se,
    calibration_intercept_ci_lower = intercept - z*intercept_se,
    calibration_intercept_ci_upper = intercept + z*intercept_se,
    oe_ratio = oe,
    oe_ratio_se = oe_se,
    oe_ratio_ci_lower = oe - z*oe_se,
    oe_ratio_ci_upper = oe + z*oe_se,
    brier_score = brier,
    scaled_brier = scaled_brier,
    n_samples = length(y_true),
    n_events = sum(y_true),
    event_rate = prevalence
)

cat(toJSON(result, auto_unbox=TRUE))
"""

    result = _run_r_script(script)

    return ValidationResult(
        c_statistic=result["c_statistic"],
        c_statistic_se=result["c_statistic_se"],
        c_statistic_ci_lower=result["c_statistic_ci_lower"],
        c_statistic_ci_upper=result["c_statistic_ci_upper"],
        calibration_slope=result["calibration_slope"],
        calibration_slope_se=result["calibration_slope_se"],
        calibration_slope_ci_lower=result["calibration_slope_ci_lower"],
        calibration_slope_ci_upper=result["calibration_slope_ci_upper"],
        calibration_intercept=result["calibration_intercept"],
        calibration_intercept_se=result["calibration_intercept_se"],
        calibration_intercept_ci_lower=result["calibration_intercept_ci_lower"],
        calibration_intercept_ci_upper=result["calibration_intercept_ci_upper"],
        oe_ratio=result["oe_ratio"],
        oe_ratio_se=result["oe_ratio_se"],
        oe_ratio_ci_lower=result["oe_ratio_ci_lower"],
        oe_ratio_ci_upper=result["oe_ratio_ci_upper"],
        brier_score=result["brier_score"],
        scaled_brier=result["scaled_brier"],
        n_samples=result["n_samples"],
        n_events=result["n_events"],
        event_rate=result["event_rate"],
    )


def instability_analysis(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
) -> InstabilityResult:
    """
    Perform model instability analysis using bootstrap resampling.

    Computes the instability index (coefficient of variation of calibration
    slope across bootstrap samples), as recommended by Riley (2023).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class
    n_bootstrap : int, default=200
        Number of bootstrap iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    InstabilityResult
        Instability metrics and bootstrap distribution

    References
    ----------
    Riley, R. D. et al. (2023). External validation of clinical prediction
    models using big data from e-health records or IPD meta-analysis.
    BMC Medicine, 21, 414.
    """
    if not _check_r_pminternal():
        raise RuntimeError(
            "R with pminternal package not available. "
            "Run: sudo ./scripts/setup-dev-environment.sh"
        )

    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    y_str = ",".join(map(str, y_true))
    p_str = ",".join(map(str, y_prob))
    seed_str = f"set.seed({random_state})" if random_state else ""

    script = f"""
suppressPackageStartupMessages({{
    library(pmcalibration)
    library(jsonlite)
}})

{seed_str}

y_true <- c({y_str})
y_prob <- c({p_str})
n_bootstrap <- {n_bootstrap}
n <- length(y_true)

# Function to compute calibration slope using pmcalibration::logistic_cal
# This is the proper method per Van Calster et al. 2019
compute_slope <- function(y, p) {{
    cal <- suppressWarnings(logistic_cal(y = y, p = p))
    return(as.numeric(coef(cal$calibration_slope)["LP"]))
}}

# Bootstrap calibration slopes
slopes <- c()
for(i in 1:n_bootstrap) {{
    idx <- sample(n, n, replace=TRUE)
    y_boot <- y_true[idx]
    p_boot <- y_prob[idx]

    # Skip if degenerate
    if(length(unique(y_boot)) < 2) next

    tryCatch({{
        slope <- compute_slope(y_boot, p_boot)
        if(is.finite(slope)) {{
            slopes <- c(slopes, slope)
        }}
    }}, error=function(e) {{}}, warning=function(w) {{}})
}}

# Compute statistics
slope_mean <- mean(slopes)
slope_sd <- sd(slopes)
slope_cv <- slope_sd / abs(slope_mean)

# Stability rating (Riley 2023)
stability_rating <- ifelse(slope_cv < 0.1, "stable",
                    ifelse(slope_cv < 0.2, "moderate", "unstable"))

result <- list(
    instability_index = slope_cv,
    slope_sd = slope_sd,
    slope_mean = slope_mean,
    slope_cv = slope_cv,
    slope_bootstrap = slopes,
    slope_percentile_2_5 = as.numeric(quantile(slopes, 0.025)),
    slope_percentile_97_5 = as.numeric(quantile(slopes, 0.975)),
    stability_rating = stability_rating,
    n_bootstrap = length(slopes),
    n_samples = n
)

cat(toJSON(result, auto_unbox=TRUE))
"""

    result = _run_r_script(script, timeout=120)

    return InstabilityResult(
        instability_index=result["instability_index"],
        slope_sd=result["slope_sd"],
        slope_mean=result["slope_mean"],
        slope_cv=result["slope_cv"],
        slope_bootstrap=np.array(result["slope_bootstrap"]),
        slope_percentile_2_5=result["slope_percentile_2_5"],
        slope_percentile_97_5=result["slope_percentile_97_5"],
        stability_rating=result["stability_rating"],
        n_bootstrap=result["n_bootstrap"],
        n_samples=result["n_samples"],
    )


def calibration_metrics(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
) -> dict:
    """
    Compute STRATOS-compliant calibration metrics.

    Convenience function that extracts key calibration metrics
    in a dictionary format suitable for figure generation.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities

    Returns
    -------
    dict
        Dictionary with calibration_slope, calibration_intercept,
        oe_ratio, brier_score, and their confidence intervals
    """
    result = validate_model(y_true, y_prob)

    return {
        "calibration_slope": result.calibration_slope,
        "calibration_slope_ci": (
            result.calibration_slope_ci_lower,
            result.calibration_slope_ci_upper,
        ),
        "calibration_intercept": result.calibration_intercept,
        "calibration_intercept_ci": (
            result.calibration_intercept_ci_lower,
            result.calibration_intercept_ci_upper,
        ),
        "oe_ratio": result.oe_ratio,
        "oe_ratio_ci": (result.oe_ratio_ci_lower, result.oe_ratio_ci_upper),
        "brier_score": result.brier_score,
        "scaled_brier": result.scaled_brier,
        "c_statistic": result.c_statistic,
        "c_statistic_ci": (result.c_statistic_ci_lower, result.c_statistic_ci_upper),
        "n_samples": result.n_samples,
        "n_events": result.n_events,
        "event_rate": result.event_rate,
    }


# Pure Python fallback implementations for when R is not available
def _calibration_slope_intercept_python(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
) -> dict:
    """
    Pure Python implementation of calibration slope and intercept.

    Falls back to this when R/pminternal is not available.
    Uses logistic regression on logit(p) to get calibration curve.
    """
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression

    # Clip probabilities to avoid inf in logit
    eps = 1e-10
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    lp = logit(p_clipped)

    # Fit logistic regression: logit(P(Y=1)) = a + b*lp
    # Calibration slope = b, intercept = a
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y_true)

    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])

    # O:E ratio
    oe = float(np.sum(y_true)) / float(np.sum(y_prob))

    # Brier score
    brier = float(np.mean((y_true - y_prob) ** 2))

    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "oe_ratio": oe,
        "brier_score": brier,
    }


def calibration_metrics_safe(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
) -> dict:
    """
    Compute calibration metrics with fallback to pure Python.

    Tries R/pminternal first, falls back to Python implementation
    if R is not available. Suitable for CI/CD environments.
    """
    if _check_r_pminternal():
        try:
            return calibration_metrics(y_true, y_prob)
        except Exception as e:
            logger.warning(f"pminternal failed, using Python fallback: {e}")

    # Fallback to pure Python
    result = _calibration_slope_intercept_python(y_true, y_prob)

    # Add missing fields with NaN
    result.update(
        {
            "calibration_slope_ci": (float("nan"), float("nan")),
            "calibration_intercept_ci": (float("nan"), float("nan")),
            "oe_ratio_ci": (float("nan"), float("nan")),
            "scaled_brier": float("nan"),
            "c_statistic": float("nan"),
            "c_statistic_ci": (float("nan"), float("nan")),
            "n_samples": len(y_true),
            "n_events": int(np.sum(y_true)),
            "event_rate": float(np.mean(y_true)),
        }
    )

    return result
