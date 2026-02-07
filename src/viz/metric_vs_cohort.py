"""Metric vs cohort visualization module.

Shows how metrics change when selecting subsets based on uncertainty.
Based on Dohopolski et al. 2022 visualization approach.

All metric computation happens in extraction (Block 1).
This module reads pre-computed cohort metrics from DuckDB (Block 2: READ ONLY).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import save_figure


def load_cohort_curve_from_db(
    config_id: int,
    metric_name: str = "auroc",
    db_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed cohort curve from DuckDB.

    Parameters
    ----------
    config_id : int
        Configuration ID to load data for
    metric_name : str
        Metric name (auroc, brier, scaled_brier)
    db_path : str, optional
        Path to DuckDB file

    Returns
    -------
    fractions, metric_values : tuple of ndarrays
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
        rows = conn.execute(
            """
            SELECT cohort_fraction, metric_value
            FROM cohort_metrics
            WHERE config_id = ? AND metric_name = ?
            ORDER BY cohort_fraction
        """,
            [config_id, metric_name],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return np.array([]), np.array([])

    fractions = np.array([r[0] for r in rows])
    values = np.array([r[1] for r in rows])

    return fractions, values


# Metric display labels (no computation, just names)
METRIC_LABELS = {
    "auroc": "AUROC",
    "brier": "Negative Brier Score",
    "scaled_brier": "Scaled Brier (IPA)",
    "accuracy": "Accuracy",
}


def plot_metric_vs_cohort(
    fractions: np.ndarray,
    metric_values: np.ndarray,
    ax=None,
    metric: str = "auroc",
    show_baseline: bool = True,
    show_improvement: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    save_json_path: Optional[str] = None,
):
    """
    Plot metric value vs cohort fraction (percentage in certain cohort).

    Parameters
    ----------
    fractions : array-like
        Cohort fractions (0-1 range)
    metric_values : array-like
        Pre-computed metric values at each fraction (from DuckDB)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    metric : str
        Metric name (for labels)
    show_baseline : bool
        Whether to show baseline (100% retention) as reference
    show_improvement : bool
        Whether to annotate improvement at 50% retention
    color : str, optional
        Line color
    label : str, optional
        Line label
    save_json_path : str, optional
        If provided, saves JSON data

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    fractions = np.asarray(fractions)
    metric_values = np.asarray(metric_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    # Convert fractions to percentages
    cohort_pct = fractions * 100

    # Plot curve
    ax.plot(
        cohort_pct,
        metric_values,
        "-o",
        color=color,
        label=label,
        linewidth=2,
        markersize=4,
    )

    # Baseline reference
    if show_baseline and len(metric_values) > 0:
        baseline = metric_values[-1]  # 100% retention
        ax.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Baseline ({metric.upper()}: {baseline:.3f})",
        )

    # Improvement annotation
    if show_improvement and len(fractions) > 0 and len(metric_values) > 0:
        idx_50 = np.argmin(np.abs(fractions - 0.5))
        metric_50 = metric_values[idx_50]
        baseline = metric_values[-1]
        improvement = (
            (metric_50 - baseline) / abs(baseline) * 100 if baseline != 0 else 0
        )

        ax.text(
            0.05,
            0.95,
            f"At 50% cohort: {metric_50:.3f}\nImprovement: {improvement:+.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Labels
    ax.set_xlabel("% Patients in Certain Cohort")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()))
    ax.set_xlim(15, 105)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Cohort Size")

    if label or show_baseline:
        ax.legend(loc="lower right")

    ax.grid(True, alpha=0.3)

    # Save JSON data
    if save_json_path:
        json_data = {
            "cohort_fractions": fractions.tolist(),
            "cohort_percentages": cohort_pct.tolist(),
            "metric_values": metric_values.tolist(),
            "metric": metric,
            "baseline": float(metric_values[-1]) if len(metric_values) > 0 else None,
        }
        with open(save_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fig, ax


def plot_multi_metric_vs_cohort(
    cohort_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot multiple metrics vs cohort fraction as subplots.

    Parameters
    ----------
    cohort_data : dict
        Maps metric name to (fractions, metric_values) tuples.
        These come from load_cohort_curve_from_db.
    figsize : tuple
        Figure size

    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
    """
    import matplotlib.pyplot as plt

    metrics = list(cohort_data.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        fractions, values = cohort_data[metric]
        plot_metric_vs_cohort(
            fractions,
            values,
            ax=ax,
            metric=metric,
            show_baseline=True,
            show_improvement=True,
        )

    plt.tight_layout()
    return fig, axes


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================


def generate_metric_vs_cohort_figure(
    fractions: np.ndarray,
    metric_values: np.ndarray,
    output_dir: Optional[Path] = None,
    filename: str = "fig_metric_change_vs_certain",
    metric: str = "auroc",
) -> Tuple[str, str]:
    """
    Generate metric vs cohort plot and save to file.

    Parameters
    ----------
    fractions : array-like
        Cohort fractions (from DuckDB)
    metric_values : array-like
        Pre-computed metric values (from DuckDB)
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename (without extension)
    metric : str
        Metric name (for labels)

    Returns
    -------
    png_path, json_path : paths to generated files
    """
    import matplotlib.pyplot as plt

    fractions = np.asarray(fractions)
    metric_values = np.asarray(metric_values)

    baseline = float(metric_values[-1]) if len(metric_values) > 0 else None

    json_data = {
        "metric": metric,
        "cohort_fractions": fractions.tolist(),
        "cohort_metrics": metric_values.tolist(),
        "baseline": baseline,
    }

    fig, ax = plot_metric_vs_cohort(
        fractions,
        metric_values,
        metric=metric,
        show_baseline=True,
        show_improvement=True,
    )

    # Save using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    return str(png_path), str(json_path)
