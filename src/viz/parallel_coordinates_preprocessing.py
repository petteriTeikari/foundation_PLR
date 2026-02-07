#!/usr/bin/env python3
"""
parallel_coordinates_preprocessing.py - Parallel coordinates plot for preprocessing analysis.

Shows the relationship between:
- Axis 1: Outlier Detection F1 Score
- Axis 2: Imputation MAE
- Color: CatBoost AUROC

Each line represents one (outlier_method, imputation_method) configuration.
This visualizes how preprocessing quality (outlier F1 and imputation MAE)
correlates with downstream classification performance (CatBoost AUROC).

Research Question: How do preprocessing choices affect classification performance?
Key Rule: FIX classifier = CatBoost, VARY preprocessing.

Usage:
    python src/viz/parallel_coordinates_preprocessing.py
"""

# Import shared config
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).parent))
from figure_dimensions import get_dimensions
from plot_config import COLORS, MANUSCRIPT_ROOT, save_figure, setup_style


def load_data() -> pd.DataFrame:
    """Load preprocessing correlation data."""
    data_path = MANUSCRIPT_ROOT / "data" / "preprocessing_correlation_data.csv"
    df = pd.read_csv(data_path)
    return df


def create_figure() -> tuple[plt.Figure, dict]:
    """
    Create parallel coordinates plot showing preprocessing → AUROC relationship.

    Returns:
        (fig, data_dict) tuple
    """
    setup_style()

    # Load data
    df = load_data()

    # Filter to rows that have both outlier F1 and imputation MAE
    plot_df = df.dropna(
        subset=["outlier_f1", "imputation_mae", "catboost_auroc"]
    ).copy()

    if len(plot_df) < 5:
        # If not enough complete rows, use partial data
        plot_df = df.dropna(subset=["catboost_auroc"]).copy()
        # Fill missing values with mean for visualization
        plot_df["outlier_f1"] = plot_df["outlier_f1"].fillna(
            plot_df["outlier_f1"].mean()
        )
        plot_df["imputation_mae"] = plot_df["imputation_mae"].fillna(
            plot_df["imputation_mae"].mean()
        )

    print(f"Plotting {len(plot_df)} configurations")

    # Create figure
    fig, ax = plt.subplots(figsize=get_dimensions("single_wide"))

    # Define axes and their ranges
    axes_data = {
        "Outlier F1": {
            "values": plot_df["outlier_f1"].values,
            "min": 0,
            "max": max(plot_df["outlier_f1"].max() * 1.1, 0.5),
            "better": "higher",
        },
        "Imputation MAE": {
            "values": plot_df["imputation_mae"].values,
            "min": 0,
            "max": plot_df["imputation_mae"].max() * 1.1,
            "better": "lower",
        },
        "CatBoost AUROC": {
            "values": plot_df["catboost_auroc"].values,
            "min": 0.7,  # Start at 0.7 to show differences better
            "max": 0.95,
            "better": "higher",
        },
    }

    n_axes = len(axes_data)
    axis_names = list(axes_data.keys())

    # Normalize values to [0, 1] for each axis
    normalized = {}
    for name, info in axes_data.items():
        values = info["values"]
        min_val, max_val = info["min"], info["max"]

        # For MAE, invert so lower is better (higher position)
        if info["better"] == "lower":
            normalized[name] = 1 - (values - min_val) / (max_val - min_val)
        else:
            normalized[name] = (values - min_val) / (max_val - min_val)

    # Clip to [0, 1]
    for name in normalized:
        normalized[name] = np.clip(normalized[name], 0, 1)

    # Create colormap based on AUROC
    auroc_values = plot_df["catboost_auroc"].values
    norm = mcolors.Normalize(vmin=auroc_values.min(), vmax=auroc_values.max())
    cmap = plt.cm.RdYlGn  # Red (low) to Green (high)

    # Draw lines
    x_positions = np.arange(n_axes)

    for i in range(len(plot_df)):
        y_values = [normalized[name][i] for name in axis_names]
        color = cmap(norm(auroc_values[i]))

        # Draw line segments
        points = np.array([x_positions, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, colors=[color], linewidths=1.5, alpha=0.6)
        ax.add_collection(lc)

    # Draw vertical axes
    for i, (name, info) in enumerate(axes_data.items()):
        ax.axvline(x=i, color=COLORS["text_primary"], linewidth=1.5, zorder=10)

        # Add tick labels
        min_val, max_val = info["min"], info["max"]
        for j, val in enumerate([min_val, (min_val + max_val) / 2, max_val]):
            if info["better"] == "lower":
                y_pos = 1 - (val - min_val) / (max_val - min_val)
            else:
                y_pos = (val - min_val) / (max_val - min_val)
            ax.plot([i - 0.03, i + 0.03], [y_pos, y_pos], "k-", linewidth=0.5)
            ax.text(i - 0.1, y_pos, f"{val:.2f}", ha="right", va="center", fontsize=8)

    # Set axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_names, fontsize=10, fontweight="bold")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("CatBoost AUROC", fontsize=10)

    # Set limits
    ax.set_xlim(-0.5, n_axes - 0.5)
    ax.set_ylim(-0.05, 1.05)

    # Remove default y-axis
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add title
    ax.set_title(
        "Preprocessing Quality → Classification Performance\n"
        "(Each line = one preprocessing configuration)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    # Add annotation for interpretation
    ax.text(
        0.02,
        -0.12,
        "Note: Higher position = better (for MAE, this means lower values)",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color=COLORS["text_secondary"],
    )

    # Highlight best configuration
    best_idx = plot_df["catboost_auroc"].idxmax()
    best_row = plot_df.loc[best_idx]

    ax.text(
        0.02,
        0.98,
        f"Best: {best_row['outlier_method'][:30]}... + {best_row['imputation_method']}\n"
        f"AUROC: {best_row['catboost_auroc']:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Prepare data dict for JSON export
    data_dict = {
        "n_configurations": len(plot_df),
        "axes": axis_names,
        "best_config": {
            "outlier_method": best_row["outlier_method"],
            "imputation_method": best_row["imputation_method"],
            "auroc": float(best_row["catboost_auroc"]),
            "outlier_f1": float(best_row["outlier_f1"]),
            "imputation_mae": float(best_row["imputation_mae"]),
        },
        "correlations": {
            "auroc_vs_f1": float(plot_df["catboost_auroc"].corr(plot_df["outlier_f1"])),
            "auroc_vs_mae": float(
                plot_df["catboost_auroc"].corr(plot_df["imputation_mae"])
            ),
        },
        "auroc_range": {
            "min": float(auroc_values.min()),
            "max": float(auroc_values.max()),
        },
    }

    return fig, data_dict


def main() -> None:
    """Generate and save parallel coordinates figure."""
    print("Creating parallel coordinates plot...")

    fig, data = create_figure()

    # Save figure
    save_figure(fig, "fig_parallel_preprocessing", data=data, formats=("png", "pdf"))

    print("\nData summary:")
    print(f"  Configurations: {data['n_configurations']}")
    print(
        f"  AUROC range: {data['auroc_range']['min']:.3f} - {data['auroc_range']['max']:.3f}"
    )
    print(f"  Correlation (AUROC vs F1): r = {data['correlations']['auroc_vs_f1']:.3f}")
    print(
        f"  Correlation (AUROC vs MAE): r = {data['correlations']['auroc_vs_mae']:.3f}"
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
