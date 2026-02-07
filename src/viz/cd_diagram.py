"""
Critical Difference Diagram for statistical method comparison.

Implements Demšar (2006) CD diagrams using Friedman test + Nemenyi post-hoc.

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Figs 8-11)

References:
- Demšar (2006). Statistical comparisons of classifiers.
- Nemenyi (1963). Distribution-free multiple comparisons.

The diagram shows:
1. Methods ranked by average performance
2. Cliques (groups) of methods NOT significantly different
3. Critical Difference (CD) bar showing minimum significant difference
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.viz.plot_config import COLORS

__all__ = [
    "friedman_nemenyi_test",
    "compute_critical_difference",
    "draw_cd_diagram",
    "identify_cliques",
    "prepare_cd_data",
]


def friedman_nemenyi_test(
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict:
    """
    Perform Friedman test with Nemenyi post-hoc analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with rows as "datasets" (e.g., preprocessing configs)
        and columns as "methods" (e.g., classifiers).
        Values are performance metrics (e.g., AUROC).
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Contains:
        - friedman_statistic: Chi-square statistic
        - friedman_pvalue: p-value for Friedman test
        - average_ranks: Dict of method -> average rank
        - critical_difference: CD value for Nemenyi test
        - pairwise_significant: Dict of (method1, method2) -> bool
        - cliques: List of sets of methods NOT significantly different

    Examples
    --------
    >>> # Rows = configs, columns = classifiers
    >>> data = pd.DataFrame({
    ...     'CatBoost': [0.91, 0.89, 0.93],
    ...     'XGBoost': [0.88, 0.87, 0.90],
    ...     'LogReg': [0.82, 0.81, 0.84]
    ... })
    >>> result = friedman_nemenyi_test(data)
    >>> print(f"Friedman p={result['friedman_pvalue']:.4f}")
    """
    n_datasets, n_methods = data.shape
    methods = data.columns.tolist()

    # Compute ranks for each row (dataset)
    # Higher performance = lower rank (rank 1 = best)
    ranks = data.rank(axis=1, ascending=False)
    average_ranks = ranks.mean().to_dict()

    # Friedman test
    # Using scipy's implementation
    friedman_stat, friedman_pvalue = stats.friedmanchisquare(
        *[data[col] for col in methods]
    )

    # Critical Difference (Nemenyi)
    cd = compute_critical_difference(n_methods, n_datasets, alpha)

    # Pairwise comparisons
    pairwise_significant = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            rank_diff = abs(average_ranks[m1] - average_ranks[m2])
            pairwise_significant[(m1, m2)] = rank_diff > cd
            pairwise_significant[(m2, m1)] = rank_diff > cd

    # Identify cliques
    cliques = identify_cliques(average_ranks, cd)

    return {
        "friedman_statistic": float(friedman_stat),
        "friedman_pvalue": float(friedman_pvalue),
        "average_ranks": average_ranks,
        "critical_difference": float(cd),
        "pairwise_significant": pairwise_significant,
        "cliques": cliques,
        "n_datasets": n_datasets,
        "n_methods": n_methods,
        "alpha": alpha,
    }


def compute_critical_difference(
    n_methods: int,
    n_datasets: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute Nemenyi critical difference.

    CD = q_α × sqrt(k(k+1) / (6N))

    where:
    - q_α: critical value from Studentized range distribution
    - k: number of methods
    - N: number of datasets

    Parameters
    ----------
    n_methods : int
        Number of methods being compared (k)
    n_datasets : int
        Number of datasets/configurations (N)
    alpha : float
        Significance level

    Returns
    -------
    float
        Critical difference value
    """
    # Critical values for Nemenyi test (Studentized range / sqrt(2))
    # Values from Demšar (2006) Table 5
    # For α = 0.05
    q_alpha_05 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
        16: 3.426,
        17: 3.458,
        18: 3.489,
        19: 3.517,
        20: 3.544,
    }
    # For α = 0.10
    q_alpha_10 = {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
        11: 2.978,
        12: 3.030,
        13: 3.077,
        14: 3.120,
        15: 3.159,
        16: 3.196,
        17: 3.230,
        18: 3.261,
        19: 3.291,
        20: 3.319,
    }

    if alpha == 0.05:
        q_alpha = q_alpha_05
    elif alpha == 0.10:
        q_alpha = q_alpha_10
    else:
        # Approximate using alpha=0.05 table
        q_alpha = q_alpha_05

    # Get q value (use max if n_methods > 20)
    k = min(n_methods, 20)
    q = q_alpha.get(k, q_alpha[20])

    cd = q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd


def identify_cliques(
    average_ranks: Dict[str, float],
    cd: float,
) -> List[List[str]]:
    """
    Identify cliques (groups of methods not significantly different).

    A clique is a maximal set of methods where all pairs differ by < CD.

    Parameters
    ----------
    average_ranks : dict
        Method -> average rank mapping
    cd : float
        Critical difference

    Returns
    -------
    list of lists
        Each inner list is a clique of method names
    """
    methods = list(average_ranks.keys())
    n = len(methods)

    # Sort by rank
    sorted_methods = sorted(methods, key=lambda m: average_ranks[m])

    # Find cliques using greedy approach
    cliques = []
    i = 0
    while i < n:
        clique = [sorted_methods[i]]
        j = i + 1

        # Extend clique while within CD
        while j < n:
            rank_diff = abs(
                average_ranks[sorted_methods[j]] - average_ranks[sorted_methods[i]]
            )
            if rank_diff <= cd:
                clique.append(sorted_methods[j])
                j += 1
            else:
                break

        # Only add if clique has 2+ methods
        if len(clique) >= 2:
            # Check if this clique is subsumed by existing
            is_new = True
            for existing in cliques:
                if set(clique).issubset(set(existing)):
                    is_new = False
                    break
            if is_new:
                cliques.append(clique)

        i += 1

    return cliques


def draw_cd_diagram(
    data: Union[pd.DataFrame, Dict],
    title: str = "Critical Difference Diagram",
    output_path: Optional[str] = None,
    save_data_path: Optional[str] = None,
    figure_id: Optional[str] = None,
    alpha: float = 0.05,
    figsize: Tuple[float, float] = (10, 5),
    text_fontsize: int = 10,
    line_width: float = 2.5,
    marker_size: int = 100,
    highlight_best: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a Critical Difference diagram.

    Parameters
    ----------
    data : pd.DataFrame or dict
        If DataFrame: rows = datasets, columns = methods, values = metrics
        If dict: output from friedman_nemenyi_test()
    title : str
        Plot title
    output_path : str, optional
        If provided, save figure to this path
    save_data_path : str, optional
        If provided, save underlying data as JSON to this path
    figure_id : str, optional
        Figure identifier for data export (e.g., "fig08")
    alpha : float
        Significance level for Nemenyi test
    figsize : tuple
        Figure size (width, height)
    text_fontsize : int
        Font size for method names
    line_width : float
        Width of clique bars
    marker_size : int
        Size of rank markers
    highlight_best : bool
        Whether to highlight the best method

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Run statistical test if data is DataFrame
    if isinstance(data, pd.DataFrame):
        result = friedman_nemenyi_test(data, alpha=alpha)
    else:
        result = data

    avg_ranks = result["average_ranks"]
    cd = result["critical_difference"]
    cliques = result["cliques"]
    n_methods = result["n_methods"]

    # Sort methods by average rank
    sorted_methods = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up axes
    rank_min = 1
    rank_max = n_methods
    ax.set_xlim(rank_min - 0.5, rank_max + 0.5)
    ax.set_ylim(-0.5, n_methods + 1)

    # Draw axis line
    ax.axhline(y=n_methods + 0.3, color="black", linewidth=1)

    # Draw tick marks and labels on top
    for rank in range(1, n_methods + 1):
        ax.plot([rank, rank], [n_methods + 0.2, n_methods + 0.4], "k-", lw=1)
        ax.text(
            rank,
            n_methods + 0.6,
            str(rank),
            ha="center",
            va="bottom",
            fontsize=text_fontsize - 2,
        )

    # Draw CD bar on right side
    cd_start = 1
    cd_end = 1 + cd
    ax.plot([cd_start, cd_end], [n_methods + 0.8, n_methods + 0.8], "k-", lw=2)
    ax.plot([cd_start, cd_start], [n_methods + 0.7, n_methods + 0.9], "k-", lw=2)
    ax.plot([cd_end, cd_end], [n_methods + 0.7, n_methods + 0.9], "k-", lw=2)
    ax.text(
        (cd_start + cd_end) / 2,
        n_methods + 1.0,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=text_fontsize - 1,
    )

    # Draw methods with their ranks
    y_positions = {}
    for i, method in enumerate(sorted_methods):
        rank = avg_ranks[method]
        y = n_methods - i - 0.5

        # Draw marker at rank position
        color = "gold" if (i == 0 and highlight_best) else "steelblue"
        ax.scatter([rank], [y], c=color, s=marker_size, zorder=5, edgecolors="black")

        # Draw line from method name to marker
        if rank < (n_methods + 1) / 2:
            # Left side - name on left
            ax.plot([rank_min - 0.3, rank], [y, y], "k-", lw=0.5, alpha=0.5)
            ax.text(
                rank_min - 0.4,
                y,
                method,
                ha="right",
                va="center",
                fontsize=text_fontsize,
            )
        else:
            # Right side - name on right
            ax.plot([rank, rank_max + 0.3], [y, y], "k-", lw=0.5, alpha=0.5)
            ax.text(
                rank_max + 0.4,
                y,
                method,
                ha="left",
                va="center",
                fontsize=text_fontsize,
            )

        # Store y position for clique bars
        y_positions[method] = y

    # Draw clique bars (horizontal lines connecting methods not significantly different)
    clique_colors = [
        COLORS["cd_rank1"],
        COLORS["cd_rank2"],
        COLORS["cd_rank3"],
        COLORS["cd_rank4"],
        COLORS["cd_rank5"],
    ]

    for idx, clique in enumerate(cliques):
        if len(clique) < 2:
            continue

        # Find y range for this clique
        clique_ranks = [avg_ranks[m] for m in clique]
        bar_left = min(clique_ranks) - 0.1
        bar_right = max(clique_ranks) + 0.1

        # Position bar slightly below the methods
        bar_y = min([y_positions[m] for m in clique]) - 0.3 - (idx * 0.15)

        color = clique_colors[idx % len(clique_colors)]

        # Draw horizontal bar
        ax.plot(
            [bar_left, bar_right],
            [bar_y, bar_y],
            color=color,
            linewidth=line_width,
            solid_capstyle="round",
        )

    # Style
    ax.set_title(title, fontsize=text_fontsize + 2, pad=20)
    ax.text(
        (rank_min + rank_max) / 2,
        n_methods + 1.3,
        "← better",
        ha="center",
        va="bottom",
        fontsize=text_fontsize - 2,
        style="italic",
    )

    # Remove frame
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Save figure if path provided
    if output_path:
        from .figure_export import save_figure_all_formats

        save_figure_all_formats(fig, output_path)
        plt.close(fig)

    # Save data as JSON if path provided
    if save_data_path:
        from .figure_data import save_figure_data

        # Convert cliques to serializable format
        cliques_serializable = [list(c) for c in cliques]

        data_dict = {
            "methods": sorted_methods,
            "average_ranks": [avg_ranks[m] for m in sorted_methods],
            "critical_difference": cd,
            "cliques": cliques_serializable,
            "friedman_statistic": result["friedman_statistic"],
            "friedman_pvalue": result["friedman_pvalue"],
        }

        save_figure_data(
            figure_id=figure_id or "cd_diagram",
            figure_title=title,
            data=data_dict,
            output_path=save_data_path,
            metadata={
                "n_datasets": result["n_datasets"],
                "n_methods": result["n_methods"],
                "alpha": alpha,
            },
        )

    return fig, ax


def prepare_cd_data(
    df: pd.DataFrame,
    config_col: str,
    method_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Prepare data for CD diagram from long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with config, method, and value columns
    config_col : str
        Column name for configuration/dataset identifier
    method_col : str
        Column name for method identifier
    value_col : str
        Column name for performance metric

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame suitable for friedman_nemenyi_test

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'config': ['A', 'A', 'B', 'B'],
    ...     'classifier': ['Cat', 'XGB', 'Cat', 'XGB'],
    ...     'auroc': [0.9, 0.85, 0.88, 0.83]
    ... })
    >>> wide_df = prepare_cd_data(df, 'config', 'classifier', 'auroc')
    """
    return df.pivot(index=config_col, columns=method_col, values=value_col)
