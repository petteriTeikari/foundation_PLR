"""
Specification Curve visualization for methodological sensitivity analysis.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Fig 6)

References:
- Simonsohn et al. (2020). Specification Curve Analysis.

A specification curve shows:
1. Top panel: All point estimates sorted by value
2. Bottom panel: Which specifications were used for each estimate
   - Dots indicate which method was used at each pipeline stage
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

__all__ = [
    "draw_specification_curve",
    "specification_curve_from_dataframe",
    "simple_specification_curve",
]


def draw_specification_curve(
    estimates: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    specifications: Dict[str, List[str]],
    title: str = "Specification Curve Analysis",
    ylabel: str = "AUROC",
    output_path: Optional[str] = None,
    save_data_path: Optional[str] = None,
    figure_id: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
    reference_line: Optional[float] = None,
    reference_label: str = "Reference",
    highlight_range: Optional[Tuple[float, float]] = None,
    show_ci: bool = True,
    ci_alpha: float = 0.3,
) -> plt.Figure:
    """
    Draw a specification curve with method indicators.

    Parameters
    ----------
    estimates : list of float
        Point estimates for each specification
    ci_lower : list of float
        Lower confidence interval bounds
    ci_upper : list of float
        Upper confidence interval bounds
    specifications : dict
        Dictionary mapping factor name to list of specification values
        e.g., {'outlier': ['IQR', 'MAD', ...], 'imputation': ['SAITS', 'Mean', ...]}
    title : str
        Plot title
    ylabel : str
        Y-axis label for estimates panel
    output_path : str, optional
        If provided, save figure to this path
    save_data_path : str, optional
        If provided, save underlying data as JSON to this path
    figure_id : str, optional
        Figure identifier for data export (e.g., "fig06")
    figsize : tuple
        Figure size
    reference_line : float, optional
        Horizontal reference line value
    reference_label : str
        Label for reference line
    highlight_range : tuple of (low, high), optional
        Highlight estimates within this range (e.g., benchmark CI)
    show_ci : bool
        Whether to show confidence intervals
    ci_alpha : float
        Alpha for CI shading

    Returns
    -------
    fig : matplotlib figure
    """
    n_specs = len(estimates)

    # Sort by estimate
    sort_idx = np.argsort(estimates)[::-1]  # Descending
    estimates = np.array(estimates)[sort_idx]
    ci_lower = np.array(ci_lower)[sort_idx]
    ci_upper = np.array(ci_upper)[sort_idx]

    sorted_specs = {}
    for factor, values in specifications.items():
        sorted_specs[factor] = np.array(values)[sort_idx]

    # Get unique values for each factor (for y-axis in bottom panel)
    factor_names = list(sorted_specs.keys())
    factor_levels = {f: list(set(v)) for f, v in sorted_specs.items()}

    # Calculate height ratios - make bottom panel taller for readability
    n_factors = len(factor_names)
    total_levels = sum(len(factor_levels[f]) for f in factor_names)
    estimate_height = 3
    spec_height = max(n_factors * 1.2, total_levels * 0.3)  # Taller bottom panel

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[estimate_height, spec_height], hspace=0.08)

    # Top panel: Estimates
    ax_estimates = fig.add_subplot(gs[0])

    x = np.arange(n_specs)

    # Draw confidence intervals
    if show_ci:
        ax_estimates.fill_between(
            x, ci_lower, ci_upper, alpha=ci_alpha, color="steelblue"
        )

    # Draw point estimates
    ax_estimates.plot(
        x, estimates, "o-", markersize=3, color="steelblue", linewidth=0.5
    )

    # Reference line
    if reference_line is not None:
        ax_estimates.axhline(
            y=reference_line,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=reference_label,
        )
        ax_estimates.legend(loc="lower right")

    # Highlight range
    if highlight_range is not None:
        low, high = highlight_range
        ax_estimates.axhspan(
            low,
            high,
            alpha=0.15,
            color="green",
            label=f"Target range [{low:.2f}, {high:.2f}]",
        )

    ax_estimates.set_ylabel(ylabel)
    ax_estimates.set_title(title, fontsize=12, fontweight="bold")
    ax_estimates.set_xlim(-0.5, n_specs - 0.5)
    ax_estimates.tick_params(labelbottom=False)

    # Add summary stats annotation
    median_est = np.median(estimates)
    range_est = estimates.max() - estimates.min()
    annotation = f"Median: {median_est:.3f}, Range: {range_est:.3f}"
    ax_estimates.text(
        0.02,
        0.95,
        annotation,
        transform=ax_estimates.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Bottom panel: Specification indicators
    ax_specs = fig.add_subplot(gs[1], sharex=ax_estimates)

    # Create y positions for each factor
    y_positions = {}
    cumulative_y = 0
    for factor in factor_names:
        n_levels = len(factor_levels[factor])
        y_positions[factor] = {
            level: cumulative_y + i for i, level in enumerate(factor_levels[factor])
        }
        cumulative_y += n_levels + 0.5  # Add gap between factors

    # Draw specification indicators
    colors = plt.cm.Set2(np.linspace(0, 1, n_factors))

    for factor_idx, factor in enumerate(factor_names):
        color = colors[factor_idx]
        for i, spec_value in enumerate(sorted_specs[factor]):
            y = y_positions[factor][spec_value]
            ax_specs.scatter([i], [y], c=[color], s=20, marker="s")

    # Draw factor labels and level labels
    for factor_idx, factor in enumerate(factor_names):
        # Factor label on the left
        y_center = np.mean([y_positions[factor][lvl] for lvl in factor_levels[factor]])
        ax_specs.text(
            -n_specs * 0.02,
            y_center,
            factor,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=colors[factor_idx],
        )

        # Level labels on the right
        for level in factor_levels[factor]:
            y = y_positions[factor][level]
            ax_specs.text(
                n_specs + n_specs * 0.01,
                y,
                level,
                ha="left",
                va="center",
                fontsize=7,
                color=colors[factor_idx],
            )

    ax_specs.set_xlim(-n_specs * 0.15, n_specs + n_specs * 0.15)
    ax_specs.set_ylim(-0.5, cumulative_y - 0.5)
    ax_specs.set_xlabel("Specification (sorted by estimate)")
    ax_specs.set_yticks([])

    # Draw horizontal separators between factors
    for factor in factor_names[:-1]:
        max_y = max(y_positions[factor].values())
        separator_y = max_y + 0.25
        ax_specs.axhline(
            y=separator_y, color="gray", linestyle="-", linewidth=0.5, alpha=0.5
        )

    # Remove spines
    ax_specs.spines["top"].set_visible(False)
    ax_specs.spines["right"].set_visible(False)
    ax_specs.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    # Save data as JSON if path provided
    if save_data_path:
        from .figure_data import save_figure_data

        data_dict = {
            "estimates": estimates.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "specifications": {k: v.tolist() for k, v in sorted_specs.items()},
            "specification_order": sort_idx.tolist(),
        }

        reference_lines = None
        if reference_line is not None:
            reference_lines = {
                "reference": {"value": reference_line, "label": reference_label}
            }

        save_figure_data(
            figure_id=figure_id or "specification_curve",
            figure_title=title,
            data=data_dict,
            output_path=save_data_path,
            reference_lines=reference_lines,
            metadata={
                "n_specifications": n_specs,
                "median_estimate": float(median_est),
                "estimate_range": float(range_est),
            },
        )

    return fig


def specification_curve_from_dataframe(
    df: pd.DataFrame,
    estimate_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    spec_cols: List[str],
    **kwargs,
) -> plt.Figure:
    """
    Create specification curve from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with estimates and specification columns
    estimate_col : str
        Column name for point estimates
    ci_lower_col : str
        Column name for CI lower bounds
    ci_upper_col : str
        Column name for CI upper bounds
    spec_cols : list of str
        Column names for specification factors (e.g., ['outlier_method', 'imputation_method'])
    **kwargs
        Additional arguments passed to draw_specification_curve

    Returns
    -------
    fig : matplotlib figure

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'auroc': [0.91, 0.88, 0.82, 0.85],
    ...     'auroc_ci_lower': [0.87, 0.84, 0.77, 0.81],
    ...     'auroc_ci_upper': [0.94, 0.91, 0.86, 0.88],
    ...     'outlier': ['IQR', 'MAD', 'None', 'ZScore'],
    ...     'imputation': ['SAITS', 'SAITS', 'Mean', 'KNN'],
    ...     'classifier': ['CatBoost', 'CatBoost', 'LogReg', 'XGBoost']
    ... })
    >>> fig = specification_curve_from_dataframe(
    ...     df, 'auroc', 'auroc_ci_lower', 'auroc_ci_upper',
    ...     ['outlier', 'imputation', 'classifier'],
    ...     title='Pipeline Specification Curve'
    ... )
    """
    specifications = {col: df[col].tolist() for col in spec_cols}

    return draw_specification_curve(
        estimates=df[estimate_col].tolist(),
        ci_lower=df[ci_lower_col].tolist(),
        ci_upper=df[ci_upper_col].tolist(),
        specifications=specifications,
        **kwargs,
    )


def simple_specification_curve(
    df: pd.DataFrame,
    estimate_col: str,
    title: str = "Specification Curve",
    ylabel: str = "Estimate",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    reference_line: Optional[float] = None,
    color: str = "steelblue",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a simple specification curve without bottom panel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with estimates
    estimate_col : str
        Column name for estimates
    title : str
        Plot title
    ylabel : str
        Y-axis label
    output_path : str, optional
        Save path
    figsize : tuple
        Figure size
    reference_line : float, optional
        Reference line value
    color : str
        Point/line color

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    estimates = df[estimate_col].sort_values(ascending=False).values
    n = len(estimates)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(range(n), estimates, "o-", markersize=2, color=color, linewidth=0.5)

    if reference_line is not None:
        ax.axhline(
            y=reference_line, color="red", linestyle="--", linewidth=1.5, alpha=0.7
        )

    ax.set_xlabel("Specification (sorted)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Summary stats
    ax.text(
        0.98,
        0.02,
        f"Range: {estimates.max():.3f} - {estimates.min():.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    return fig, ax
