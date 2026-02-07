"""
Heatmap visualizations for methodological sensitivity analysis.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Fig 5)

Creates heatmaps showing performance (AUROC) as a function of:
- Outlier detection method (rows)
- Imputation method (columns)

With optional annotation of:
- Cell values
- Significance indicators
- Benchmark achievement
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

__all__ = [
    "draw_sensitivity_heatmap",
    "heatmap_from_pivot",
    "annotated_heatmap",
    "sensitivity_heatmap_grid",
]


def draw_sensitivity_heatmap(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str = "Sensitivity Analysis Heatmap",
    xlabel: str = "Columns",
    ylabel: str = "Rows",
    cmap: str = "RdYlGn",
    output_path: Optional[str] = None,
    save_data_path: Optional[str] = None,
    figure_id: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    annotate: bool = True,
    fmt: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    highlight_best: bool = True,
    highlight_threshold: Optional[float] = None,
    cbar_label: str = "Value",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a sensitivity analysis heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with row, column, and value columns
    row_col : str
        Column name for row categories (e.g., 'outlier_method')
    col_col : str
        Column name for column categories (e.g., 'imputation_method')
    value_col : str
        Column name for values (e.g., 'auroc')
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    cmap : str
        Matplotlib colormap name
    output_path : str, optional
        If provided, save figure to this path
    save_data_path : str, optional
        If provided, save underlying data as JSON to this path
    figure_id : str, optional
        Figure identifier for data export (e.g., "fig05")
    figsize : tuple
        Figure size (width, height)
    annotate : bool
        Whether to annotate cells with values
    fmt : str
        Format string for annotations
    vmin, vmax : float, optional
        Color scale limits
    highlight_best : bool
        Whether to highlight the best cell
    highlight_threshold : float, optional
        If provided, add border to cells exceeding this threshold
    cbar_label : str
        Colorbar label

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Pivot data to matrix format
    pivot = data.pivot(index=row_col, columns=col_col, values=value_col)

    return heatmap_from_pivot(
        pivot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        output_path=output_path,
        save_data_path=save_data_path,
        figure_id=figure_id,
        figsize=figsize,
        annotate=annotate,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        highlight_best=highlight_best,
        highlight_threshold=highlight_threshold,
        cbar_label=cbar_label,
    )


def heatmap_from_pivot(
    pivot: pd.DataFrame,
    title: str = "Heatmap",
    xlabel: str = "Columns",
    ylabel: str = "Rows",
    cmap: str = "RdYlGn",
    output_path: Optional[str] = None,
    save_data_path: Optional[str] = None,
    figure_id: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    annotate: bool = True,
    fmt: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    highlight_best: bool = True,
    highlight_threshold: Optional[float] = None,
    cbar_label: str = "Value",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw heatmap from pre-pivoted DataFrame.

    Parameters
    ----------
    pivot : pd.DataFrame
        Pivoted data with row index and column names
    ... (same as draw_sensitivity_heatmap)
    """
    # Use constrained_layout instead of tight_layout (works better with colorbars)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Get data as array
    data_array = pivot.values

    # Set color limits
    if vmin is None:
        vmin = np.nanmin(data_array)
    if vmax is None:
        vmax = np.nanmax(data_array)

    # Draw heatmap
    im = ax.imshow(data_array, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Annotate cells
    if annotate:
        # Determine text color threshold
        mid_val = (vmin + vmax) / 2

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = data_array[i, j]
                if np.isnan(val):
                    continue

                # Choose text color based on background
                text_color = "white" if val < mid_val else "black"

                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    # Highlight best cell
    if highlight_best:
        best_idx = np.unravel_index(np.nanargmax(data_array), data_array.shape)
        rect = Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="gold",
            linewidth=3,
        )
        ax.add_patch(rect)

    # Highlight cells above threshold
    if highlight_threshold is not None:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if data_array[i, j] >= highlight_threshold:
                    rect = Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="lime",
                        linewidth=2,
                        linestyle="--",
                    )
                    ax.add_patch(rect)

    # Note: Using constrained_layout instead of tight_layout (set in subplots)

    # Save figure if path provided
    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    # Save data as JSON if path provided
    if save_data_path:
        from .figure_data import save_figure_data

        data_dict = {
            "row_labels": list(pivot.index),
            "col_labels": list(pivot.columns),
            "values": pivot.values.tolist(),
            "row_axis_label": ylabel,
            "col_axis_label": xlabel,
            "value_label": cbar_label,
        }

        save_figure_data(
            figure_id=figure_id or "heatmap",
            figure_title=title,
            data=data_dict,
            output_path=save_data_path,
            metadata={
                "highlight_threshold": highlight_threshold,
                "cmap": cmap,
            },
        )

    return fig, ax


def annotated_heatmap(
    pivot: pd.DataFrame,
    ci_lower: pd.DataFrame,
    ci_upper: pd.DataFrame,
    title: str = "Heatmap with Confidence Intervals",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    cmap: str = "RdYlGn",
    show_ci: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw heatmap with confidence interval annotations.

    Parameters
    ----------
    pivot : pd.DataFrame
        Point estimates (pivoted)
    ci_lower : pd.DataFrame
        Lower CI bounds (same shape as pivot)
    ci_upper : pd.DataFrame
        Upper CI bounds (same shape as pivot)
    title : str
        Plot title
    output_path : str, optional
        Save path
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    show_ci : bool
        Whether to show CI in annotation

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    data_array = pivot.values
    vmin, vmax = np.nanmin(data_array), np.nanmax(data_array)

    # Draw heatmap
    im = ax.imshow(data_array, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Colorbar
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    # Annotate with point estimate and CI
    mid_val = (vmin + vmax) / 2

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data_array[i, j]
            if np.isnan(val):
                continue

            text_color = "white" if val < mid_val else "black"

            if show_ci:
                lo = ci_lower.iloc[i, j]
                hi = ci_upper.iloc[i, j]
                text = f"{val:.2f}\n[{lo:.2f},{hi:.2f}]"
            else:
                text = f"{val:.3f}"

            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    return fig, ax


def sensitivity_heatmap_grid(
    data: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    facet_col: str,
    title: str = "Sensitivity Analysis by Classifier",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    cmap: str = "RdYlGn",
    shared_colorbar: bool = True,
) -> plt.Figure:
    """
    Create a grid of heatmaps, one per facet (e.g., classifier).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data
    row_col : str
        Row category column (e.g., 'outlier_method')
    col_col : str
        Column category column (e.g., 'imputation_method')
    value_col : str
        Value column (e.g., 'auroc')
    facet_col : str
        Faceting column (e.g., 'classifier')
    title : str
        Overall title
    output_path : str, optional
        Save path
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    shared_colorbar : bool
        Whether to use shared color scale across facets

    Returns
    -------
    fig : matplotlib figure
    """
    facets = data[facet_col].unique()
    n_facets = len(facets)

    # Determine grid layout
    n_cols = min(3, n_facets)
    n_rows = int(np.ceil(n_facets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    # Get global min/max for shared colorbar
    if shared_colorbar:
        vmin = data[value_col].min()
        vmax = data[value_col].max()
    else:
        vmin, vmax = None, None

    for idx, facet in enumerate(facets):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]

        facet_data = data[data[facet_col] == facet]
        pivot = facet_data.pivot(index=row_col, columns=col_col, values=value_col)

        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{facet}", fontsize=10)

    # Hide unused axes
    for idx in range(n_facets, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].axis("off")

    # Shared colorbar
    if shared_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    return fig
