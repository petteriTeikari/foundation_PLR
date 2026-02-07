#!/usr/bin/env python
"""
featurization_comparison.py - Figure R7: Featurization Method Comparison

Generates a side-by-side comparison of handcrafted physiological features
vs MOMENT embeddings for glaucoma classification.

Key finding: Handcrafted features (0.83) substantially outperform
MOMENT embeddings (0.74) by ~9 percentage points.

Usage:
    python src/viz/featurization_comparison.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_dimensions import get_dimensions
from plot_config import (
    COLORS,
    KEY_STATS,
    add_benchmark_line,
    get_connection,
    save_figure,
    setup_style,
)


def fetch_featurization_data() -> pd.DataFrame:
    """Fetch featurization performance from DuckDB."""
    conn = get_connection()

    # Get mean AUROC and CI by featurization method
    query = """
    SELECT
        featurization,
        AVG(auroc) as mean_auroc,
        STDDEV(auroc) as std_auroc,
        COUNT(*) as n_configs,
        MIN(auroc) as min_auroc,
        MAX(auroc) as max_auroc
    FROM essential_metrics
    WHERE auroc IS NOT NULL
      AND featurization IS NOT NULL
      AND featurization != 'Unknown'
    GROUP BY featurization
    ORDER BY mean_auroc DESC
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    return df


def fetch_bootstrap_ci() -> Optional[pd.DataFrame]:
    """Fetch bootstrap confidence intervals if available."""
    conn = get_connection()

    # Try to get percentile-based CIs
    query = """
    SELECT
        featurization,
        AVG(auroc) as mean_auroc,
        PERCENTILE_CONT(0.025) WITHIN GROUP (ORDER BY auroc) as ci_lower,
        PERCENTILE_CONT(0.975) WITHIN GROUP (ORDER BY auroc) as ci_upper,
        COUNT(*) as n
    FROM essential_metrics
    WHERE auroc IS NOT NULL
      AND featurization IS NOT NULL
      AND featurization != 'Unknown'
    GROUP BY featurization
    """

    try:
        df = conn.execute(query).fetchdf()
    except Exception:
        # Fallback: use standard error
        df = None

    conn.close()
    return df


def create_figure() -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create the featurization comparison figure."""
    setup_style()

    # Fetch data
    df = fetch_featurization_data()
    ci_df = fetch_bootstrap_ci()

    # Create figure
    fig, ax = plt.subplots(figsize=get_dimensions("single_narrow"))

    # Data
    methods = df["featurization"].tolist()
    means = df["mean_auroc"].tolist()

    # Map method names to display names
    display_names = {
        "handcrafted": "Handcrafted\nPhysiological",
        "moment_embeddings": "MOMENT\nEmbeddings",
        "embeddings": "MOMENT\nEmbeddings",
    }
    labels = [display_names.get(m, m) for m in methods]

    # Colors based on performance
    colors = []
    for m in methods:
        if "handcrafted" in m.lower():
            colors.append(COLORS["handcrafted"])
        else:
            colors.append(COLORS["embeddings"])

    # Calculate error bars
    if ci_df is not None and "ci_lower" in ci_df.columns:
        # Use actual CIs
        errors = []
        for i, method in enumerate(methods):
            row = ci_df[ci_df["featurization"] == method]
            if not row.empty:
                lower = means[i] - row["ci_lower"].values[0]
                upper = row["ci_upper"].values[0] - means[i]
                errors.append([lower, upper])
            else:
                errors.append([df.iloc[i]["std_auroc"], df.iloc[i]["std_auroc"]])
        errors = np.array(errors).T
    else:
        # Fallback to standard deviation
        errors = df["std_auroc"].values

    # Bar positions
    x = np.arange(len(methods))
    width = 0.6

    # Create bars
    bars = ax.bar(
        x,
        means,
        width,
        color=colors,
        edgecolor=COLORS["text_primary"],
        linewidth=1,
        yerr=errors if isinstance(errors, np.ndarray) and errors.ndim == 1 else errors,
        capsize=5,
        error_kw={"linewidth": 1.5},
    )

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add benchmark line
    add_benchmark_line(ax, KEY_STATS["benchmark_auroc"], "Najjar 2021 (0.93)")

    # Add gap annotation
    if len(means) >= 2:
        gap = abs(means[0] - means[1])
        gap_pp = int(round(gap * 100))
        mid_y = (means[0] + means[1]) / 2

        # Draw arrow between bars
        ax.annotate(
            "",
            xy=(0.15, means[1] + 0.01),
            xytext=(0.15, means[0] - 0.01),
            arrowprops=dict(arrowstyle="<->", color=COLORS["text_primary"], lw=1.5),
        )
        ax.text(
            -0.15,
            mid_y,
            f"{gap_pp}pp\ngap",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLORS["bad"],
        )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean AUROC", fontweight="bold")
    ax.set_ylim(0.65, 1.0)
    ax.set_title(
        "Featurization Method Comparison\nHandcrafted Features vs Foundation Model Embeddings",
        fontweight="bold",
        pad=15,
    )

    # Add legend
    ax.legend(loc="upper right")

    # Add sample size annotation
    for i, (m, n) in enumerate(zip(methods, df["n_configs"])):
        ax.text(
            i,
            0.67,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["text_secondary"],
        )

    plt.tight_layout()

    return fig, {
        "methods": methods,
        "mean_auroc": means,
        "n_configs": df["n_configs"].tolist(),
        "std_auroc": df["std_auroc"].tolist(),
        "gap_pp": int(round(abs(means[0] - means[1]) * 100))
        if len(means) >= 2
        else None,
    }


def main() -> None:
    """Generate and save the figure."""
    print("Generating Figure R7: Featurization Comparison...")

    fig, data = create_figure()
    save_figure(fig, "fig_R7_featurization_comparison", data=data)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
