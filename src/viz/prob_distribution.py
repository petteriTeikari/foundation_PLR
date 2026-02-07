"""Probability distribution visualization module.

Visualizes predicted probability distributions stratified by outcome class.
Shows discrimination between cases and controls.

All metric computation happens in extraction (Block 1).
This module reads pre-computed stats from DuckDB (Block 2: READ ONLY).
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import numpy as np

from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import COLORS, save_figure


def load_distribution_stats_from_db(
    config_id: int,
    db_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load pre-computed distribution statistics from DuckDB.

    Parameters
    ----------
    config_id : int
        Configuration ID to load stats for
    db_path : str, optional
        Path to DuckDB file

    Returns
    -------
    dict with keys:
        - auroc: Area Under ROC Curve
        - median_cases: Median predicted probability for cases
        - median_controls: Median predicted probability for controls
        - mean_cases: Mean predicted probability for cases
        - mean_controls: Mean predicted probability for controls
        - n_cases: Number of case samples
        - n_controls: Number of control samples
    """
    import duckdb

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
            raise FileNotFoundError("DuckDB not found (run extraction first)")

    conn = duckdb.connect(db_path, read_only=True)

    try:
        row = conn.execute(
            """
            SELECT auroc, median_cases, median_controls,
                   mean_cases, mean_controls, n_cases, n_controls
            FROM distribution_stats
            WHERE config_id = ?
        """,
            [config_id],
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {
            "auroc": np.nan,
            "median_cases": np.nan,
            "median_controls": np.nan,
            "mean_cases": np.nan,
            "mean_controls": np.nan,
            "n_cases": 0,
            "n_controls": 0,
        }

    return {
        "auroc": row[0] if row[0] is not None else np.nan,
        "median_cases": row[1] if row[1] is not None else np.nan,
        "median_controls": row[2] if row[2] is not None else np.nan,
        "mean_cases": row[3] if row[3] is not None else np.nan,
        "mean_controls": row[4] if row[4] is not None else np.nan,
        "n_cases": row[5] or 0,
        "n_controls": row[6] or 0,
    }


def _compute_stats_from_arrays(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute basic distribution stats from arrays (no sklearn).

    Only uses numpy for summary statistics. AUROC is read from DB
    or passed in via stats parameter. This function does NOT compute AUROC.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    cases_mask = y_true == 1
    controls_mask = y_true == 0
    prob_cases = y_prob[cases_mask]
    prob_controls = y_prob[controls_mask]

    return {
        "auroc": np.nan,  # Must be read from DB
        "median_cases": float(np.median(prob_cases)) if len(prob_cases) > 0 else np.nan,
        "median_controls": float(np.median(prob_controls))
        if len(prob_controls) > 0
        else np.nan,
        "mean_cases": float(np.mean(prob_cases)) if len(prob_cases) > 0 else np.nan,
        "mean_controls": float(np.mean(prob_controls))
        if len(prob_controls) > 0
        else np.nan,
        "n_cases": len(prob_cases),
        "n_controls": len(prob_controls),
    }


def plot_probability_distributions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    plot_type: Literal["histogram", "density", "violin", "box"] = "histogram",
    bins: int = 30,
    alpha: float = 0.6,
    show_threshold: bool = False,
    threshold: float = 0.5,
    show_stats: bool = True,
    stats: Optional[Dict[str, float]] = None,
    case_label: str = "Glaucoma",
    control_label: str = "Control",
    case_color: Optional[str] = None,
    control_color: Optional[str] = None,
    save_json_path: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot predicted probability distributions by outcome class.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0=control, 1=case)
    y_prob : array-like
        Predicted probabilities
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    plot_type : str
        Type of plot: 'histogram', 'density', 'violin', or 'box'
    bins : int
        Number of histogram bins
    alpha : float
        Transparency for histogram bars
    show_threshold : bool
        Whether to show vertical line at decision threshold
    threshold : float
        Decision threshold to mark
    show_stats : bool
        Whether to annotate with statistics
    stats : dict, optional
        Pre-computed stats dict (from load_distribution_stats_from_db).
        If None, computes basic stats from arrays (without AUROC).
    case_label : str
        Label for cases in legend
    control_label : str
        Label for controls in legend
    case_color : str, optional
        Color for cases. Defaults to COLORS["glaucoma"].
    control_color : str, optional
        Color for controls. Defaults to COLORS["control"].
    save_json_path : str, optional
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

    # Resolve colors from COLORS dict if not provided
    if case_color is None:
        case_color = COLORS["glaucoma"]
    if control_color is None:
        control_color = COLORS["control"]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    # Split by class
    cases_mask = y_true == 1
    controls_mask = y_true == 0
    prob_cases = y_prob[cases_mask]
    prob_controls = y_prob[controls_mask]

    # Use provided stats or compute basic ones (no sklearn)
    if stats is None:
        stats = _compute_stats_from_arrays(y_true, y_prob)

    if plot_type == "histogram":
        # Overlapping histograms
        if len(prob_controls) > 0:
            ax.hist(
                prob_controls,
                bins=bins,
                alpha=alpha,
                color=control_color,
                label=f"{control_label} (n={len(prob_controls)})",
                density=True,
            )
        if len(prob_cases) > 0:
            ax.hist(
                prob_cases,
                bins=bins,
                alpha=alpha,
                color=case_color,
                label=f"{case_label} (n={len(prob_cases)})",
                density=True,
            )
        ax.set_ylabel("Density")

    elif plot_type == "density":
        # KDE density plots
        from scipy.stats import gaussian_kde

        x_grid = np.linspace(0, 1, 200)

        if len(prob_controls) > 1:
            kde_controls = gaussian_kde(prob_controls, bw_method="scott")
            ax.plot(
                x_grid,
                kde_controls(x_grid),
                color=control_color,
                linewidth=2,
                label=f"{control_label} (n={len(prob_controls)})",
            )
            ax.fill_between(
                x_grid, kde_controls(x_grid), alpha=0.3, color=control_color
            )

        if len(prob_cases) > 1:
            kde_cases = gaussian_kde(prob_cases, bw_method="scott")
            ax.plot(
                x_grid,
                kde_cases(x_grid),
                color=case_color,
                linewidth=2,
                label=f"{case_label} (n={len(prob_cases)})",
            )
            ax.fill_between(x_grid, kde_cases(x_grid), alpha=0.3, color=case_color)

        ax.set_ylabel("Density")

    elif plot_type == "violin":
        # Violin plot
        parts = ax.violinplot(
            [prob_controls, prob_cases],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )

        # Color the violin parts
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([control_label, case_label])
        ax.set_ylabel("P(Glaucoma)")

    elif plot_type == "box":
        # Box plot
        box_data = [prob_controls, prob_cases]
        bp = ax.boxplot(box_data, labels=[control_label, case_label], patch_artist=True)

        # Color boxes
        bp["boxes"][0].set_facecolor(control_color)
        bp["boxes"][1].set_facecolor(case_color)
        for box in bp["boxes"]:
            box.set_alpha(0.6)

        ax.set_ylabel("P(Glaucoma)")

    # Show threshold line
    if show_threshold and plot_type in ["histogram", "density"]:
        ax.axvline(
            x=threshold,
            color=COLORS["text_primary"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Threshold ({threshold})",
        )

    # Add statistics annotation
    if show_stats and not np.isnan(stats.get("auroc", np.nan)):
        stats_text = f"AUROC: {stats['auroc']:.3f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=COLORS["background"], alpha=0.8),
        )

    # Labels
    if plot_type in ["histogram", "density"]:
        ax.set_xlabel("Predicted Probability")
        ax.legend(loc="upper center")
        ax.set_xlim(-0.02, 1.02)

    ax.set_title("Predicted Probability Distribution by Outcome")

    # Save JSON data
    if save_json_path:
        json_data = {
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
            "y_prob_cases": prob_cases.tolist(),
            "y_prob_controls": prob_controls.tolist(),
            "plot_type": plot_type,
            "statistics": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in stats.items()
            },
        }
        with open(save_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fig, ax


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================


def generate_probability_distribution_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Optional[Path] = None,
    filename: str = "fig_prob_dist_by_outcome",
    plot_type: str = "histogram",
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[str, str]:
    """
    Generate probability distribution plot and save to file.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename (without extension)
    plot_type : str
        Type of plot
    stats : dict, optional
        Pre-computed stats from load_distribution_stats_from_db.

    Returns
    -------
    png_path, json_path : paths to generated files
    """
    import matplotlib.pyplot as plt

    # Prepare data for JSON export
    if stats is None:
        stats = _compute_stats_from_arrays(y_true, y_prob)

    cases_mask = y_true == 1
    controls_mask = y_true == 0

    json_data = {
        "y_prob_cases": y_prob[cases_mask].tolist(),
        "y_prob_controls": y_prob[controls_mask].tolist(),
        "plot_type": plot_type,
        "statistics": {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in stats.items()
        },
    }

    fig, ax = plot_probability_distributions(
        y_true,
        y_prob,
        plot_type=plot_type,
        show_stats=True,
        stats=stats,
    )

    # Use save_figure for proper output handling
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)

    # Return paths
    json_path = png_path.parent / "data" / f"{filename}.json"
    return str(png_path), str(json_path)
