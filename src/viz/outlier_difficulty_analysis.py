#!/usr/bin/env python3
"""
outlier_difficulty_analysis.py - Visualize easy vs hard outlier classification.

Shows that:
- EASY outliers (blinks, >3 SD from ground truth): 16.5% of all outliers
- HARD outliers (subtle, ≤3 SD from ground truth): 83.5% of all outliers

Key insight: Most outliers are subtle deviations marked by human annotators
that are near the signal, not obvious artifacts like blinks. Methods that
catch all EASY outliers are often sufficient for good downstream performance.

Usage:
    python src/viz/outlier_difficulty_analysis.py
"""

# Import shared config
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figure_dimensions import get_dimensions
from plot_config import COLORS, MANUSCRIPT_ROOT, save_figure, setup_style


def load_data() -> pd.DataFrame:
    """Load outlier difficulty analysis data."""
    data_path = MANUSCRIPT_ROOT / "data" / "outlier_difficulty_analysis.csv"
    df = pd.read_csv(data_path)
    return df


def create_figure() -> tuple:
    """
    Create visualization of easy vs hard outlier distribution.

    Returns:
        (fig, data_dict) tuple
    """
    setup_style()

    # Load data
    df = load_data()

    # Create figure with multiple panels
    fig = plt.figure(figsize=get_dimensions("double_tall"))

    # =========================================================================
    # Panel A: Overall pie chart of EASY vs HARD outliers
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)

    total_easy = df["easy_count"].sum()
    total_hard = df["hard_count"].sum()
    total_outliers = total_easy + total_hard

    sizes = [total_easy, total_hard]
    labels = [
        f"EASY\n(blinks)\n{total_easy:,}\n({total_easy / total_outliers * 100:.1f}%)",
        f"HARD\n(subtle)\n{total_hard:,}\n({total_hard / total_outliers * 100:.1f}%)",
    ]
    colors = [COLORS["good"], COLORS["bad"]]
    explode = (0.05, 0)

    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax1.set_title(
        "A. Overall Outlier Difficulty Distribution", fontsize=11, fontweight="bold"
    )

    # =========================================================================
    # Panel B: Comparison by class (control vs glaucoma)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    classes = ["control", "glaucoma"]
    class_names = ["Control", "Glaucoma"]
    x = np.arange(len(classes))
    width = 0.35

    easy_by_class = []
    hard_by_class = []

    for cls in classes:
        cls_df = df[df["class_label"] == cls]
        easy_by_class.append(cls_df["easy_pct"].mean())
        hard_by_class.append(cls_df["hard_pct"].mean())

    bars1 = ax2.bar(
        x - width / 2, easy_by_class, width, label="EASY (>3 SD)", color=COLORS["good"]
    )
    bars2 = ax2.bar(
        x + width / 2, hard_by_class, width, label="HARD (≤3 SD)", color=COLORS["bad"]
    )

    ax2.set_ylabel("Mean Outlier % per Subject")
    ax2.set_xlabel("Class")
    ax2.set_title("B. Outlier Difficulty by Class", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # =========================================================================
    # Panel C: Distribution of outlier percentage by subject
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    # Histogram of total outlier percentage
    control_df = df[df["class_label"] == "control"]
    glaucoma_df = df[df["class_label"] == "glaucoma"]

    bins = np.linspace(0, max(df["outlier_pct"].max(), 40), 20)

    ax3.hist(
        control_df["outlier_pct"],
        bins=bins,
        alpha=0.6,
        label=f"Control (n={len(control_df)})",
        color=COLORS["primary"],
    )
    ax3.hist(
        glaucoma_df["outlier_pct"],
        bins=bins,
        alpha=0.6,
        label=f"Glaucoma (n={len(glaucoma_df)})",
        color=COLORS["secondary"],
    )

    ax3.set_xlabel("Total Outlier Percentage per Subject")
    ax3.set_ylabel("Number of Subjects")
    ax3.set_title("C. Distribution of Outlier Rates", fontsize=11, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Add median lines
    control_median = control_df["outlier_pct"].median()
    glaucoma_median = glaucoma_df["outlier_pct"].median()
    ax3.axvline(
        control_median,
        color=COLORS["primary"],
        linestyle="--",
        linewidth=2,
        label=f"Control median: {control_median:.1f}%",
    )
    ax3.axvline(
        glaucoma_median,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=2,
        label=f"Glaucoma median: {glaucoma_median:.1f}%",
    )

    # =========================================================================
    # Panel D: EASY vs HARD ratio across subjects
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    # Calculate ratio of EASY to total for each subject (where there are outliers)
    df_with_outliers = df[df["total_outliers"] > 0].copy()
    df_with_outliers["easy_ratio"] = (
        df_with_outliers["easy_count"] / df_with_outliers["total_outliers"] * 100
    )

    control_ratios = df_with_outliers[df_with_outliers["class_label"] == "control"][
        "easy_ratio"
    ]
    glaucoma_ratios = df_with_outliers[df_with_outliers["class_label"] == "glaucoma"][
        "easy_ratio"
    ]

    data_to_plot = [control_ratios, glaucoma_ratios]
    positions = [1, 2]

    bp = ax4.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

    colors_bp = [COLORS["primary"], COLORS["secondary"]]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(["Control", "Glaucoma"])
    ax4.set_ylabel("% of Outliers that are EASY")
    ax4.set_title("D. EASY Outlier Ratio per Subject", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Add mean annotation
    for i, data in enumerate(data_to_plot):
        mean_val = data.mean()
        ax4.annotate(
            f"μ={mean_val:.1f}%",
            xy=(positions[i], mean_val),
            xytext=(0.3, 0),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    # Add overall title
    fig.suptitle(
        "Outlier Difficulty Analysis: EASY (Blinks) vs HARD (Subtle Deviations)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Prepare data dict
    data_dict = {
        "total_outliers": int(total_outliers),
        "easy_count": int(total_easy),
        "hard_count": int(total_hard),
        "easy_pct": float(total_easy / total_outliers * 100),
        "hard_pct": float(total_hard / total_outliers * 100),
        "by_class": {
            "control": {
                "n_subjects": int(len(control_df)),
                "mean_outlier_pct": float(control_df["outlier_pct"].mean()),
                "mean_easy_pct": float(control_df["easy_pct"].mean()),
                "mean_hard_pct": float(control_df["hard_pct"].mean()),
            },
            "glaucoma": {
                "n_subjects": int(len(glaucoma_df)),
                "mean_outlier_pct": float(glaucoma_df["outlier_pct"].mean()),
                "mean_easy_pct": float(glaucoma_df["easy_pct"].mean()),
                "mean_hard_pct": float(glaucoma_df["hard_pct"].mean()),
            },
        },
        "interpretation": (
            "Most outliers (83.5%) are HARD (subtle deviations ≤3 SD from ground truth). "
            "Only 16.5% are EASY (obvious artifacts like blinks). "
            "Glaucoma subjects have higher overall outlier rates (12.3% vs 6.3% for controls)."
        ),
    }

    return fig, data_dict


def main():
    """Generate and save outlier difficulty figure."""
    print("Creating outlier difficulty analysis figure...")

    fig, data = create_figure()

    # Save figure
    save_figure(fig, "fig_outlier_easy_vs_hard", data=data, formats=("png", "pdf"))

    print("\n=== Summary ===")
    print(f"Total outliers: {data['total_outliers']:,}")
    print(f"EASY (blinks, >3 SD): {data['easy_count']:,} ({data['easy_pct']:.1f}%)")
    print(f"HARD (subtle, ≤3 SD): {data['hard_count']:,} ({data['hard_pct']:.1f}%)")
    print(
        f"\nControl: {data['by_class']['control']['n_subjects']} subjects, "
        f"{data['by_class']['control']['mean_outlier_pct']:.1f}% mean outlier rate"
    )
    print(
        f"Glaucoma: {data['by_class']['glaucoma']['n_subjects']} subjects, "
        f"{data['by_class']['glaucoma']['mean_outlier_pct']:.1f}% mean outlier rate"
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
