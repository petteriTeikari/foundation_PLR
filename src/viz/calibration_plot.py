"""Calibration plot visualization module.

Implements STRATOS-compliant smoothed calibration curves with LOESS smoothing.
Based on Van Calster et al. 2024 guidelines.

COMPUTATION DECOUPLING: This module performs visualization ONLY.
- LOESS smoothing and bootstrap CI are visualization rendering (acceptable).
- Calibration metrics (slope, intercept, Brier, O:E) come from DuckDB.
- The *_from_db functions read pre-computed metrics from DuckDB.
- NO sklearn imports. NO src.stats imports.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate

from src.viz.plot_config import save_figure, COLORS
from src.viz.figure_dimensions import get_dimensions


def compute_loess_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, frac: float = 0.3, n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LOESS-smoothed calibration curve.

    This is visualization rendering (smoothing for display), NOT metric computation.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    frac : float
        Fraction of data used for LOESS smoothing (default 0.3)
    n_points : int
        Number of points for output curve

    Returns
    -------
    x_smooth : ndarray
        Sorted probability values
    y_smooth : ndarray
        Smoothed calibration values (observed frequencies)
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        # LOESS smoothing
        smoothed = lowess(y_true, y_prob, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        # Fallback: binned calibration
        return _compute_binned_calibration(y_true, y_prob, n_bins=n_points // 5)


def _compute_binned_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback binned calibration if statsmodels not available."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(y_true[mask].mean())

    return np.array(bin_centers), np.array(bin_means)


def compute_calibration_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 200,
    frac: float = 0.3,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for calibration curve.

    This is visualization rendering (CI bands for display), NOT metric computation.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bootstrap : int
        Number of bootstrap iterations
    frac : float
        LOESS smoothing fraction
    alpha : float
        Significance level for CI (default 0.05 for 95% CI)

    Returns
    -------
    x_vals : ndarray
        Common x-axis values
    y_lower : ndarray
        Lower confidence bound
    y_upper : ndarray
        Upper confidence bound
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    # Common x-axis for interpolation
    x_common = np.linspace(0.05, 0.95, 50)
    bootstrap_curves = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t_boot = y_true[idx]
        y_p_boot = y_prob[idx]

        try:
            x_smooth, y_smooth = compute_loess_calibration(
                y_t_boot, y_p_boot, frac=frac
            )
            if len(x_smooth) > 1:
                # Interpolate to common x-axis
                f = interpolate.interp1d(
                    x_smooth, y_smooth, bounds_error=False, fill_value="extrapolate"
                )
                bootstrap_curves.append(f(x_common))
        except (ValueError, RuntimeError):
            continue

    if len(bootstrap_curves) < 10:
        # Not enough bootstrap samples, return wide CI
        return x_common, np.zeros_like(x_common), np.ones_like(x_common)

    bootstrap_array = np.array(bootstrap_curves)
    y_lower = np.percentile(bootstrap_array, alpha / 2 * 100, axis=0)
    y_upper = np.percentile(bootstrap_array, (1 - alpha / 2) * 100, axis=0)

    # Clip to valid range
    y_lower = np.clip(y_lower, 0, 1)
    y_upper = np.clip(y_upper, 0, 1)

    return x_common, y_lower, y_upper


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    show_ci: bool = True,
    show_rug: bool = True,
    show_metrics: bool = True,
    metrics: Optional[Dict[str, float]] = None,
    frac: float = 0.3,
    ci_alpha: float = 0.2,
    n_bootstrap: int = 200,
    save_path: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot smoothed calibration curve with LOESS.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    label : str, optional
        Legend label for the model
    color : str, optional
        Line color
    show_ci : bool
        Whether to show confidence intervals
    show_rug : bool
        Whether to show histogram rug at bottom
    show_metrics : bool
        Whether to annotate with calibration metrics
    metrics : dict, optional
        Pre-computed calibration metrics from DuckDB. Expected keys:
        'calibration_slope' (or 'slope'), 'calibration_intercept' (or 'intercept').
        If show_metrics is True and metrics is None, the annotation is skipped.
    frac : float
        LOESS smoothing fraction
    ci_alpha : float
        Alpha for CI shading
    n_bootstrap : int
        Number of bootstrap samples for CI
    save_path : str, optional
        If provided, saves JSON data for reproducibility

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    try:
        from src.viz.plot_config import setup_style
    except ImportError:
        from plot_config import setup_style
    setup_style()

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("calibration_small"))
    else:
        fig = ax.get_figure()

    # Reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Compute LOESS smoothed curve (visualization rendering)
    x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob, frac=frac)

    # Plot main curve
    ax.plot(x_smooth, y_smooth, color=color, label=label or "Model", linewidth=2)

    # Confidence intervals (visualization rendering)
    if show_ci:
        x_ci, y_lower, y_upper = compute_calibration_ci(
            y_true, y_prob, n_bootstrap=n_bootstrap, frac=frac
        )
        ax.fill_between(x_ci, y_lower, y_upper, alpha=ci_alpha, color=color)

    # Histogram rug
    if show_rug:
        # Show distribution of predictions at bottom
        ax.scatter(
            y_prob[y_true == 0],
            np.full(sum(y_true == 0), -0.02),
            marker="|",
            alpha=0.3,
            color=COLORS["control"],
            s=30,
            label="Controls",
        )
        ax.scatter(
            y_prob[y_true == 1],
            np.full(sum(y_true == 1), -0.04),
            marker="|",
            alpha=0.3,
            color=COLORS["glaucoma"],
            s=30,
            label="Cases",
        )

    # Calibration metrics annotation (from pre-computed metrics, NOT computed here)
    if show_metrics and metrics is not None:
        # Support both naming conventions (DuckDB uses calibration_slope,
        # older code used slope)
        slope = metrics.get("calibration_slope", metrics.get("slope"))
        intercept = metrics.get("calibration_intercept", metrics.get("intercept"))

        parts = []
        if slope is not None:
            parts.append(f"Slope: {slope:.2f}")
        if intercept is not None:
            parts.append(f"Intercept: {intercept:.2f}")

        if parts:
            metrics_text = "\n".join(parts)
            ax.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=COLORS["background"], alpha=0.8),
            )

    # Labels and limits
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.02)
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")

    # Save JSON data for reproducibility
    if save_path:
        json_path = Path(save_path).with_suffix(".json")
        json_data = {
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
            "loess_frac": frac,
            "x_smooth": x_smooth.tolist(),
            "y_smooth": y_smooth.tolist(),
        }
        if show_ci:
            json_data["x_ci"] = x_ci.tolist()
            json_data["y_lower"] = y_lower.tolist()
            json_data["y_upper"] = y_upper.tolist()

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fig, ax


def plot_calibration_multi_model(
    models_data: Dict[str, Dict],
    ax: Optional["plt.Axes"] = None,
    show_ci: bool = False,
    colors: Optional[List[str]] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot calibration curves for multiple models.

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to {'y_true': ..., 'y_prob': ...}
    ax : matplotlib.axes.Axes, optional
    show_ci : bool
        Whether to show confidence intervals
    colors : list, optional
        Colors for each model

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("calibration_small"))
    else:
        fig = ax.get_figure()

    # Reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    for (model_name, data), color in zip(models_data.items(), colors):
        y_true = np.asarray(data["y_true"])
        y_prob = np.asarray(data["y_prob"])

        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)
        ax.plot(x_smooth, y_smooth, label=model_name, color=color, linewidth=2)

        if show_ci:
            x_ci, y_lower, y_upper = compute_calibration_ci(y_true, y_prob)
            ax.fill_between(x_ci, y_lower, y_upper, alpha=0.15, color=color)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    return fig, ax


# ============================================================================
# STRATOS-COMPLIANT CALIBRATION EXPORT (DuckDB-based)
# ============================================================================
#
# CRITICAL: These functions read from DuckDB instead of computing metrics.
# All metric computation happens during extraction (scripts/extract_*_to_duckdb.py).
# See: CRITICAL-FAILURE-003-computation-decoupling-violation.md


def save_calibration_extended_json_from_db(
    run_id: str,
    output_path: str,
    db_path: Optional[str] = None,
) -> dict:
    """
    Save extended calibration metrics to JSON by reading from DuckDB.

    CRITICAL: This function reads PRE-COMPUTED metrics from DuckDB.
    It does NOT compute metrics - all computation happens during extraction.

    Parameters
    ----------
    run_id : str
        Run ID to load metrics for
    output_path : str
        Path to save JSON file
    db_path : str, optional
        Path to DuckDB file

    Returns
    -------
    dict
        The JSON data structure
    """
    import duckdb

    # Find database
    if db_path is None:
        db_paths = [
            Path("data/public/foundation_plr_results.db"),
            Path("data/foundation_plr_results.db"),
        ]
        for p in db_paths:
            if p.exists():
                db_path = str(p)
                break
        if db_path is None:
            raise FileNotFoundError("DuckDB not found")

    conn = duckdb.connect(db_path, read_only=True)

    # Read pre-computed scalar metrics
    metrics_row = conn.execute(
        """
        SELECT
            auroc, brier, scaled_brier,
            calibration_slope, calibration_intercept, o_e_ratio
        FROM essential_metrics
        WHERE run_id = ?
    """,
        [run_id],
    ).fetchone()

    if metrics_row is None:
        conn.close()
        raise ValueError(f"Run {run_id} not found in essential_metrics")

    auroc, brier, scaled_brier, cal_slope, cal_intercept, o_e_ratio = metrics_row

    # Read pre-computed calibration curve
    curve_row = conn.execute(
        """
        SELECT x_smooth, y_smooth, ci_lower, ci_upper
        FROM calibration_curves
        WHERE run_id = ?
    """,
        [run_id],
    ).fetchone()

    conn.close()

    if curve_row is None:
        raise ValueError(f"Run {run_id} not found in calibration_curves")

    # Parse JSON arrays
    x_smooth = json.loads(curve_row[0])
    y_smooth = json.loads(curve_row[1])
    ci_lower = json.loads(curve_row[2])
    ci_upper = json.loads(curve_row[3])

    # Build JSON structure
    json_data = {
        "run_id": run_id,
        # STRATOS required metrics (pre-computed)
        "stratos_metrics": {
            "auroc": auroc,
            "brier_score": brier,
            "scaled_brier": scaled_brier,
            "calibration_slope": cal_slope,
            "calibration_intercept": cal_intercept,
            "o_e_ratio": o_e_ratio,
        },
        # Calibration curve for plotting (pre-computed)
        "calibration_curve": {
            "x_smooth": x_smooth,
            "y_smooth": y_smooth,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return json_data


def save_calibration_multi_combo_json_from_db(
    run_ids: List[str],
    output_path: str,
    db_path: Optional[str] = None,
) -> dict:
    """
    Save calibration metrics for multiple runs to single JSON.

    CRITICAL: Reads from DuckDB, does NOT compute metrics.

    Parameters
    ----------
    run_ids : list of str
        Run IDs to include
    output_path : str
        Path to save JSON file
    db_path : str, optional
        Path to DuckDB file

    Returns
    -------
    dict
        The JSON data structure with all combos
    """
    import duckdb

    # Find database
    if db_path is None:
        db_paths = [
            Path("data/public/foundation_plr_results.db"),
            Path("data/foundation_plr_results.db"),
        ]
        for p in db_paths:
            if p.exists():
                db_path = str(p)
                break
        if db_path is None:
            raise FileNotFoundError("DuckDB not found")

    conn = duckdb.connect(db_path, read_only=True)

    all_combos = {}
    for run_id in run_ids:
        # Read scalar metrics
        metrics_row = conn.execute(
            """
            SELECT
                outlier_method, imputation_method, classifier,
                calibration_slope, calibration_intercept, o_e_ratio, brier
            FROM essential_metrics
            WHERE run_id = ?
        """,
            [run_id],
        ).fetchone()

        if metrics_row is None:
            continue

        outlier, imputation, classifier, slope, intercept, oe, brier = metrics_row

        # Read curve data
        curve_row = conn.execute(
            """
            SELECT x_smooth, y_smooth
            FROM calibration_curves
            WHERE run_id = ?
        """,
            [run_id],
        ).fetchone()

        combo_key = f"{outlier}+{imputation}"
        all_combos[combo_key] = {
            "run_id": run_id,
            "outlier_method": outlier,
            "imputation_method": imputation,
            "classifier": classifier,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "o_e_ratio": oe,
            "brier_score": brier,
        }

        if curve_row:
            all_combos[combo_key]["curve"] = {
                "x": json.loads(curve_row[0]),
                "y": json.loads(curve_row[1]),
            }

    conn.close()

    json_data = {
        "n_combos": len(all_combos),
        "combos": all_combos,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return json_data


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================


def generate_calibration_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
    filename: str = "fig_calibration_smoothed",
) -> Tuple[str, str]:
    """
    Generate calibration plot and save to file.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    metrics : dict, optional
        Pre-computed calibration metrics from DuckDB (e.g. calibration_slope,
        calibration_intercept). If None, metrics annotation is skipped.
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename (without extension)

    Returns
    -------
    png_path, json_path : paths to generated files
    """
    import matplotlib.pyplot as plt

    # LOESS curve for JSON data (visualization rendering)
    x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

    json_data = {
        "curve": {"x": x_smooth.tolist(), "y": y_smooth.tolist()},
    }
    if metrics is not None:
        json_data["metrics"] = metrics

    fig, ax = plot_calibration_curve(
        y_true,
        y_prob,
        show_ci=True,
        show_rug=True,
        show_metrics=metrics is not None,
        metrics=metrics,
    )

    # Save using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    return str(png_path), str(json_path)
