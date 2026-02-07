"""Uncertainty scatter visualization module.

Shows relationship between predicted probability and uncertainty.
Based on Filos et al. 2019 visualization approach.
"""

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import COLORS, save_figure


def compute_uncertainty_correlation(
    y_prob: np.ndarray, uncertainty: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between probability and uncertainty.

    Parameters
    ----------
    y_prob : array-like
        Predicted probabilities
    uncertainty : array-like
        Uncertainty estimates

    Returns
    -------
    correlation, p_value : tuple of floats
    """
    y_prob = np.asarray(y_prob)
    uncertainty = np.asarray(uncertainty)

    # Remove NaN values
    valid_mask = ~(np.isnan(y_prob) | np.isnan(uncertainty))
    if valid_mask.sum() < 3:
        return np.nan, np.nan

    corr, p_val = spearmanr(y_prob[valid_mask], uncertainty[valid_mask])
    return float(corr), float(p_val)


def plot_uncertainty_scatter(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    ax=None,
    color_by_outcome: bool = True,
    case_color: Optional[str] = None,
    control_color: Optional[str] = None,
    case_label: str = "Glaucoma",
    control_label: str = "Control",
    alpha: float = 0.6,
    show_regression: bool = False,
    show_correlation: bool = True,
    save_json_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot uncertainty vs predicted probability scatter.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    uncertainty : array-like
        Uncertainty estimates
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    color_by_outcome : bool
        Whether to color points by true outcome
    case_color : str, optional
        Color for cases. Defaults to COLORS["glaucoma"].
    control_color : str, optional
        Color for controls. Defaults to COLORS["control"].
    case_label : str
        Label for cases
    control_label : str
        Label for controls
    alpha : float
        Point transparency
    show_regression : bool
        Whether to show regression/LOESS line
    show_correlation : bool
        Whether to annotate with correlation coefficient
    save_json_path : str, optional
        If provided, saves JSON data

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    try:
        from src.viz.plot_config import setup_style
    except ImportError:
        from plot_config import setup_style
    setup_style()

    # Resolve colors from COLORS dict if not provided
    if case_color is None:
        case_color = COLORS["glaucoma"]
    if control_color is None:
        control_color = COLORS["control"]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    uncertainty = np.asarray(uncertainty)

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    if color_by_outcome:
        # Split by outcome
        cases_mask = y_true == 1
        controls_mask = y_true == 0

        ax.scatter(
            y_prob[controls_mask],
            uncertainty[controls_mask],
            c=control_color,
            alpha=alpha,
            label=f"{control_label}",
            edgecolors="none",
            s=50,
        )
        ax.scatter(
            y_prob[cases_mask],
            uncertainty[cases_mask],
            c=case_color,
            alpha=alpha,
            label=f"{case_label}",
            edgecolors="none",
            s=50,
        )
        ax.legend()
    else:
        ax.scatter(
            y_prob, uncertainty, alpha=alpha, c="steelblue", edgecolors="none", s=50
        )

    # Regression line
    if show_regression:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(uncertainty, y_prob, frac=0.3, return_sorted=True)
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                "k-",
                linewidth=2,
                alpha=0.8,
                label="LOESS trend",
            )
        except ImportError:
            # Fallback to linear regression
            z = np.polyfit(y_prob, uncertainty, 1)
            p = np.poly1d(z)
            x_line = np.linspace(y_prob.min(), y_prob.max(), 100)
            ax.plot(
                x_line, p(x_line), "k--", linewidth=1.5, alpha=0.7, label="Linear trend"
            )

    # Correlation annotation
    if show_correlation:
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)
        if not np.isnan(corr):
            p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            corr_text = f"Spearman r = {corr:.3f} ({p_str})"
            ax.text(
                0.05,
                0.95,
                corr_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    # Labels
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Uncertainty")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Uncertainty vs Predicted Probability")

    # Grid
    ax.grid(True, alpha=0.3)

    # Save JSON data
    if save_json_path:
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)
        json_data = {
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
            "uncertainty": uncertainty.tolist(),
            "correlation": {
                "spearman_r": float(corr) if not np.isnan(corr) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
            },
        }
        with open(save_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fig, ax


def plot_uncertainty_by_correctness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[plt.Figure, Any]:
    """
    Plot uncertainty scatter split by prediction correctness.

    Two panels: Correct predictions vs Incorrect predictions.
    Based on Filos et al. 2019 Figure 4 style.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    uncertainty : array-like
        Uncertainty estimates
    threshold : float
        Decision threshold for correctness
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    uncertainty = np.asarray(uncertainty)

    # Determine correctness
    y_pred = (y_prob >= threshold).astype(int)
    correct = y_pred == y_true

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Correct predictions
    ax0 = axes[0]
    ax0.scatter(
        y_prob[correct],
        uncertainty[correct],
        alpha=0.5,
        c="green",
        edgecolors="none",
        s=50,
    )
    ax0.set_title(f"Correct Predictions (n={correct.sum()})")
    ax0.set_xlabel("P(Glaucoma)")
    ax0.set_ylabel("Uncertainty")
    ax0.set_xlim(-0.02, 1.02)
    ax0.grid(True, alpha=0.3)

    # Incorrect predictions
    ax1 = axes[1]
    ax1.scatter(
        y_prob[~correct],
        uncertainty[~correct],
        alpha=0.5,
        c="red",
        edgecolors="none",
        s=50,
    )
    ax1.set_title(f"Incorrect Predictions (n={(~correct).sum()})")
    ax1.set_xlabel("P(Glaucoma)")
    ax1.set_xlim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Annotate mean uncertainty
    if correct.sum() > 0:
        mean_unc_correct = uncertainty[correct].mean()
        ax0.text(
            0.05,
            0.95,
            f"Mean unc: {mean_unc_correct:.3f}",
            transform=ax0.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    if (~correct).sum() > 0:
        mean_unc_incorrect = uncertainty[~correct].mean()
        ax1.text(
            0.05,
            0.95,
            f"Mean unc: {mean_unc_incorrect:.3f}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    return fig, axes


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================


def generate_uncertainty_scatter_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    output_dir: Optional[Path] = None,
    filename: str = "fig_uncertainty_vs_prob",
) -> Tuple[str, str]:
    """
    Generate uncertainty scatter plot and save to file.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    uncertainty : array-like
        Uncertainty estimates
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename (without extension)

    Returns
    -------
    png_path, json_path : paths to generated files
    """
    # Compute correlation for JSON data
    corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

    # Prepare data for JSON export
    json_data = {
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "uncertainty": uncertainty.tolist(),
        "correlation": {"spearman_rho": corr, "p_value": p_val},
    }

    fig, ax = plot_uncertainty_scatter(
        y_true,
        y_prob,
        uncertainty,
        color_by_outcome=True,
        show_correlation=True,
    )

    # Use save_figure for proper output handling
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)

    json_path = png_path.parent / "data" / f"{filename}.json"
    return str(png_path), str(json_path)


if __name__ == "__main__":
    from src.viz.data_loader import MockDataLoader

    print("Generating uncertainty scatter plot with mock data...")

    loader = MockDataLoader(n_groups=1, n_iterations=1, seed=42)
    mock_df = loader.load_raw(["y_true", "y_prob", "uncertainty"], limit=200)

    y_true = mock_df["y_true"].values
    y_prob = mock_df["y_prob"].values
    uncertainty = mock_df["uncertainty"].values

    png_path, json_path = generate_uncertainty_scatter_figure(
        y_true,
        y_prob,
        uncertainty,
        filename="fig_uncertainty_vs_prob",
    )

    print(f"Generated: {png_path}")
    print(f"JSON data: {json_path}")
