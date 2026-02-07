"""
Forest plots for displaying effect sizes with confidence intervals.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Figs 2-4)

A forest plot shows:
1. Point estimates (e.g., mean AUROC)
2. Confidence intervals as horizontal lines
3. Methods sorted by performance
4. Optional reference line (e.g., benchmark AUROC)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "draw_forest_plot",
    "forest_plot_from_dataframe",
    "grouped_forest_plot",
]


def draw_forest_plot(
    methods: List[str],
    point_estimates: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    title: str = "Forest Plot",
    xlabel: str = "Metric",
    reference_line: Optional[float] = None,
    reference_label: str = "Reference",
    output_path: Optional[str] = None,
    save_data_path: Optional[str] = None,
    figure_id: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    color: str = "steelblue",
    sort_by_estimate: bool = True,
    highlight_top_n: int = 0,
    significance_threshold: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a forest plot.

    Parameters
    ----------
    methods : list of str
        Method/classifier names
    point_estimates : list of float
        Point estimates (e.g., mean AUROC)
    ci_lower : list of float
        Lower confidence interval bounds
    ci_upper : list of float
        Upper confidence interval bounds
    title : str
        Plot title
    xlabel : str
        X-axis label
    reference_line : float, optional
        Value for vertical reference line (e.g., benchmark AUROC 0.93)
    reference_label : str
        Label for reference line
    output_path : str, optional
        If provided, save figure to this path
    save_data_path : str, optional
        If provided, save underlying data as JSON to this path
    figure_id : str, optional
        Figure identifier for data export (e.g., "fig02")
    figsize : tuple
        Figure size (width, height)
    color : str
        Color for points and error bars
    sort_by_estimate : bool
        Whether to sort methods by point estimate (descending)
    highlight_top_n : int
        Number of top methods to highlight in different color
    significance_threshold : float, optional
        If provided, draw a dashed line at this threshold

    Returns
    -------
    fig, ax : matplotlib figure and axes

    Examples
    --------
    >>> methods = ['CatBoost', 'XGBoost', 'LogReg']
    >>> aurocs = [0.91, 0.88, 0.82]
    >>> ci_low = [0.87, 0.84, 0.77]
    >>> ci_high = [0.94, 0.91, 0.86]
    >>> fig, ax = draw_forest_plot(methods, aurocs, ci_low, ci_high,
    ...                            title='Classifier Comparison',
    ...                            reference_line=0.93,
    ...                            reference_label='Najjar 2021')
    """
    # Convert to arrays
    methods = np.array(methods)
    point_estimates = np.array(point_estimates)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Sort by estimate if requested
    if sort_by_estimate:
        sort_idx = np.argsort(point_estimates)[::-1]  # Descending
        methods = methods[sort_idx]
        point_estimates = point_estimates[sort_idx]
        ci_lower = ci_lower[sort_idx]
        ci_upper = ci_upper[sort_idx]

    n_methods = len(methods)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Y positions
    y_pos = np.arange(n_methods)

    # Draw reference line(s)
    if reference_line is not None:
        ax.axvline(
            x=reference_line,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=reference_label,
        )

    if significance_threshold is not None:
        ax.axvline(
            x=significance_threshold,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

    # Draw error bars and points
    for i, (method, est, lo, hi) in enumerate(
        zip(methods, point_estimates, ci_lower, ci_upper)
    ):
        # Determine color
        if highlight_top_n > 0 and i < highlight_top_n:
            point_color = "gold"
            edge_color = "darkorange"
        else:
            point_color = color
            edge_color = "darkblue"

        # Error bar (CI)
        ax.hlines(y=i, xmin=lo, xmax=hi, color=point_color, linewidth=2, alpha=0.7)

        # Point estimate
        ax.scatter(
            [est],
            [i],
            c=point_color,
            s=100,
            zorder=5,
            edgecolors=edge_color,
            linewidth=1.5,
        )

        # CI endpoints
        ax.scatter(
            [lo, hi], [i, i], c=point_color, s=30, marker="|", zorder=4, alpha=0.8
        )

    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add value annotations on right
    x_max = max(ci_upper) * 1.05
    for i, (est, lo, hi) in enumerate(zip(point_estimates, ci_lower, ci_upper)):
        text = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
        ax.text(x_max, i, text, va="center", fontsize=8)

    # Adjust x-axis
    x_min = min(ci_lower) * 0.95
    ax.set_xlim(x_min, x_max * 1.15)

    # Grid
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    if reference_line is not None:
        ax.legend(loc="lower right")

    # Invert y-axis so best is at top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save figure if path provided (in PNG, SVG, EPS formats)
    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    # Save data as JSON if path provided
    if save_data_path:
        from .figure_data import save_figure_data

        data = {
            "methods": methods.tolist(),
            "estimates": point_estimates.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "xlabel": xlabel,
        }

        reference_lines = None
        if reference_line is not None:
            reference_lines = {
                "reference": {"value": reference_line, "label": reference_label}
            }

        save_figure_data(
            figure_id=figure_id or "forest_plot",
            figure_title=title,
            data=data,
            output_path=save_data_path,
            reference_lines=reference_lines,
        )

    return fig, ax


def forest_plot_from_dataframe(
    df: pd.DataFrame,
    method_col: str,
    estimate_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create forest plot from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with method names, estimates, and CIs
    method_col : str
        Column name for method names
    estimate_col : str
        Column name for point estimates
    ci_lower_col : str
        Column name for CI lower bounds
    ci_upper_col : str
        Column name for CI upper bounds
    **kwargs
        Additional arguments passed to draw_forest_plot

    Returns
    -------
    fig, ax : matplotlib figure and axes

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'classifier': ['CatBoost', 'XGBoost', 'LogReg'],
    ...     'auroc': [0.91, 0.88, 0.82],
    ...     'auroc_ci_lower': [0.87, 0.84, 0.77],
    ...     'auroc_ci_upper': [0.94, 0.91, 0.86]
    ... })
    >>> fig, ax = forest_plot_from_dataframe(
    ...     df, 'classifier', 'auroc', 'auroc_ci_lower', 'auroc_ci_upper',
    ...     title='AUROC by Classifier'
    ... )
    """
    return draw_forest_plot(
        methods=df[method_col].tolist(),
        point_estimates=df[estimate_col].tolist(),
        ci_lower=df[ci_lower_col].tolist(),
        ci_upper=df[ci_upper_col].tolist(),
        **kwargs,
    )


def grouped_forest_plot(
    df: pd.DataFrame,
    method_col: str,
    group_col: str,
    estimate_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str = "Grouped Forest Plot",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    group_colors: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped forest plot with multiple methods per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with method names, groups, estimates, and CIs
    method_col : str
        Column name for method names
    group_col : str
        Column name for group identifier
    estimate_col : str
        Column name for point estimates
    ci_lower_col : str
        Column name for CI lower bounds
    ci_upper_col : str
        Column name for CI upper bounds
    title : str
        Plot title
    output_path : str, optional
        Save path
    figsize : tuple
        Figure size
    group_colors : dict, optional
        Mapping of group names to colors

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    groups = df[group_col].unique()

    if group_colors is None:
        cmap = plt.cm.get_cmap("tab10")
        group_colors = {g: cmap(i) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = 0
    y_positions = []
    y_labels = []
    group_boundaries = []

    for group in groups:
        group_data = df[df[group_col] == group].sort_values(
            estimate_col, ascending=False
        )

        color = group_colors.get(group, "steelblue")

        for _, row in group_data.iterrows():
            # Draw error bar
            ax.hlines(
                y=y_pos,
                xmin=row[ci_lower_col],
                xmax=row[ci_upper_col],
                color=color,
                linewidth=2,
                alpha=0.7,
            )

            # Draw point
            ax.scatter(
                [row[estimate_col]],
                [y_pos],
                c=[color],
                s=80,
                zorder=5,
                edgecolors="black",
                linewidth=1,
            )

            y_labels.append(row[method_col])
            y_positions.append(y_pos)
            y_pos += 1

        # Add group separator
        group_boundaries.append(y_pos - 0.5)
        y_pos += 0.5  # Space between groups

    # Draw group boundaries
    for boundary in group_boundaries[:-1]:
        ax.axhline(y=boundary, color="gray", linewidth=0.5, alpha=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    return fig, ax
