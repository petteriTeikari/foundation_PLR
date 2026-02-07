"""
Figure generation for model instability analysis (Riley 2023).

Creates:
- Prediction instability plot (Riley 2023 Fig 2 style)
- Per-patient uncertainty distributions (Kompa 2021 style)
- MAPE histogram comparison

Cross-references:
- planning/latent-method-results-update.md (Section 26.3)
- Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"
- Kompa B et al. (2021) "Second opinion needed: communicating uncertainty in medical ML"
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import gaussian_kde

from src.stats.pminternal_analysis import (
    BootstrapPredictionData,
    compute_prediction_instability_metrics,
    create_prediction_instability_plot_data,
)
from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import COLORS, save_figure


def plot_prediction_instability(
    data: BootstrapPredictionData,
    ax: Optional[plt.Axes] = None,
    subsample: int = 200,
    alpha: float = 0.1,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Create Riley 2023 style prediction instability scatter plot.

    X-axis: Original model prediction
    Y-axis: Bootstrap model predictions
    Diagonal: Perfect agreement
    Dashed lines: 2.5th and 97.5th percentiles

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    ax : plt.Axes, optional
        Axes to plot on. Creates new figure if None.
    subsample : int, default 200
        Number of bootstrap samples to show
    alpha : float, default 0.1
        Point transparency
    title : str, optional
        Plot title

    Returns
    -------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    plot_data : dict
        Data used for plotting (for JSON export)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("square_small"))
    else:
        fig = ax.get_figure()

    # Get plot data
    plot_data = create_prediction_instability_plot_data(data, subsample=subsample)

    # Scatter plot
    ax.scatter(
        plot_data["scatter"]["x"],
        plot_data["scatter"]["y"],
        alpha=alpha,
        s=2,
        c=COLORS["text_secondary"],
        rasterized=True,  # For smaller PDF files
    )

    # Diagonal line (perfect agreement)
    ax.plot([0, 1], [0, 1], "k-", linewidth=1.5, label="Perfect agreement")

    # Percentile lines
    x = plot_data["percentile_lines"]["x"]
    p_2_5 = plot_data["percentile_lines"]["p_2_5"]
    p_97_5 = plot_data["percentile_lines"]["p_97_5"]

    ax.plot(x, p_2_5, "k--", linewidth=1, label="2.5th/97.5th percentile")
    ax.plot(x, p_97_5, "k--", linewidth=1)

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Estimated risk from developed model")
    ax.set_ylabel("Estimated risk from bootstrap models")
    ax.set_aspect("equal")

    if title:
        ax.set_title(title)

    ax.legend(loc="upper left", frameon=False)

    return fig, ax, plot_data


def plot_per_patient_uncertainty(
    data: BootstrapPredictionData,
    patient_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Create Kompa 2021 style per-patient uncertainty distributions.

    Shows bootstrap prediction distributions for selected patients.

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    patient_indices : list of int, optional
        Indices of patients to show. Auto-selects interesting patients if None.
    ax : plt.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axes
    plot_data : dict
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    y_bootstrap = data.y_pred_proba_bootstrap

    # Auto-select patients: find ones with different uncertainty levels
    if patient_indices is None:
        sds = np.std(y_bootstrap, axis=0)
        means = np.mean(y_bootstrap, axis=0)

        # Find patients with similar mean but different SD
        # Low uncertainty (narrow distribution)
        mid_range = (means > 0.3) & (means < 0.7)
        if np.any(mid_range):
            low_sd_idx = np.where(mid_range)[0][np.argmin(sds[mid_range])]
            high_sd_idx = np.where(mid_range)[0][np.argmax(sds[mid_range])]
            patient_indices = [low_sd_idx, high_sd_idx]
        else:
            # Fallback: just take first two
            patient_indices = [0, 1]

    # Plot KDE for each patient
    x = np.linspace(0, 1, 200)
    colors = plt.cm.tab10.colors
    plot_data = {"patients": []}

    for i, idx in enumerate(patient_indices):
        preds = y_bootstrap[:, idx]
        mean_pred = np.mean(preds)
        sd_pred = np.std(preds)

        # KDE - handle edge case of zero variance
        if sd_pred < 1e-10:
            logger.warning(f"Patient {idx} has near-zero variance, skipping KDE")
            continue
        try:
            kde = gaussian_kde(preds)
            y = kde(x)
        except np.linalg.LinAlgError:
            logger.warning(f"KDE failed for patient {idx}, skipping")
            continue

        color = colors[i % len(colors)]
        label = f"Patient {idx + 1} (Mean={mean_pred:.2f}, SD={sd_pred:.3f})"

        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.axvline(mean_pred, color=color, linestyle="--", alpha=0.7)

        plot_data["patients"].append(
            {
                "index": int(idx),
                "mean_pred": float(mean_pred),
                "sd_pred": float(sd_pred),
                "y_true": int(data.y_true[idx]),
                "kde_x": x.tolist(),
                "kde_y": y.tolist(),
            }
        )

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_xlabel("Risk of Glaucoma")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=False)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Predictive Uncertainty for Individual Patients")

    return fig, ax, plot_data


def plot_mape_histogram(
    data_list: List[Tuple[str, BootstrapPredictionData]],
    ax: Optional[plt.Axes] = None,
    bins: int = 20,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Create histogram comparing MAPE across preprocessing methods.

    Parameters
    ----------
    data_list : list of (name, BootstrapPredictionData)
        List of named bootstrap data to compare
    ax : plt.Axes, optional
        Axes to plot on
    bins : int, default 20
        Number of histogram bins

    Returns
    -------
    fig, ax, plot_data
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10.colors
    plot_data = {"methods": []}

    for i, (name, data) in enumerate(data_list):
        metrics = compute_prediction_instability_metrics(data)
        mape = metrics.mape

        color = colors[i % len(colors)]
        ax.hist(
            mape,
            bins=bins,
            alpha=0.6,
            color=color,
            label=f"{name} (mean={metrics.mean_mape:.3f})",
        )

        plot_data["methods"].append(
            {
                "name": name,
                "mean_mape": float(metrics.mean_mape),
                "median_mape": float(np.median(mape)),
                "mape_values": mape.tolist(),
            }
        )

    ax.set_xlabel("MAPE (Mean Absolute Prediction Error)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Prediction Instability by Preprocessing Method")
    ax.legend(loc="upper right", frameon=False)

    return fig, ax, plot_data


def plot_instability_comparison(
    data_list: List[Tuple[str, BootstrapPredictionData]],
    figsize: Tuple[float, float] = (15, 5),
    subsample: int = 200,
) -> Tuple[plt.Figure, List[plt.Axes], Dict[str, Any]]:
    """
    Create 3-panel comparison of instability across methods.

    Panel layout:
    1. Ground truth instability
    2. FM pipeline instability
    3. Traditional pipeline instability

    Parameters
    ----------
    data_list : list of (name, BootstrapPredictionData)
        List of named bootstrap data (should be 3 items)
    figsize : tuple
        Figure size
    subsample : int
        Bootstrap samples per plot

    Returns
    -------
    fig, axes, plot_data
    """
    n_panels = len(data_list)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    if n_panels == 1:
        axes = [axes]

    plot_data = {"panels": []}

    for ax, (name, data) in zip(axes, data_list):
        metrics = compute_prediction_instability_metrics(data)
        _, _, panel_data = plot_prediction_instability(
            data,
            ax=ax,
            subsample=subsample,
            title=f"{name}\n(n={data.n_subjects}, Mean MAPE={metrics.mean_mape:.3f})",
        )
        panel_data["name"] = name
        panel_data["mean_mape"] = float(metrics.mean_mape)
        panel_data["mean_cii"] = float(metrics.mean_cii)
        plot_data["panels"].append(panel_data)

    plt.tight_layout()

    return fig, axes, plot_data


def save_figure_with_data(
    fig: plt.Figure,
    output_name: str,
    output_dir: Path,
    plot_data: Dict[str, Any],
    dpi: int = 300,
) -> None:
    """
    Save figure as PNG/PDF and data as JSON.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    output_name : str
        Base name for output files (without extension)
    output_dir : Path
        Output directory
    plot_data : dict
        Data to save as JSON
    dpi : int
        Resolution for PNG
    """
    output_dir_path = Path(output_dir) if output_dir else None

    # Use save_figure for proper output handling
    saved_path = save_figure(
        fig, output_name, data=plot_data, output_dir=output_dir_path
    )
    logger.info(f"Saved: {saved_path}")
