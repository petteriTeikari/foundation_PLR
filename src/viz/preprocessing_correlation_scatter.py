#!/usr/bin/env python3
"""
preprocessing_correlation_scatter.py - Scatter plots showing preprocessing → AUROC correlation.

Two-panel figure:
- Panel A: CatBoost AUROC vs Outlier Detection F1
- Panel B: CatBoost AUROC vs Imputation MAE

Shows how preprocessing quality (measured independently) correlates with
downstream classification performance.

Research Question: How do preprocessing choices affect classification performance?
Key Rule: FIX classifier = CatBoost, VARY preprocessing.

Usage:
    python src/viz/preprocessing_correlation_scatter.py
"""

# Import shared config
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
    Create two-panel scatter plot.

    Returns:
        (fig, data_dict) tuple
    """
    setup_style()

    # Load data
    df = load_data()

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=get_dimensions("wide"))

    # =========================================================================
    # Panel A: AUROC vs Outlier F1
    # =========================================================================
    ax1 = axes[0]

    # Filter to rows with valid outlier F1
    df_f1 = df.dropna(subset=["outlier_f1", "catboost_auroc"]).copy()

    if len(df_f1) > 0:
        x = df_f1["outlier_f1"].values
        y = df_f1["catboost_auroc"].values

        # Scatter plot
        ax1.scatter(
            x,
            y,
            c=COLORS["primary"],
            alpha=0.7,
            s=60,
            edgecolors=COLORS["background"],
            linewidths=0.5,
        )

        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax1.plot(
            x_line, y_line, color=COLORS["bad"], linewidth=2, linestyle="--", alpha=0.8
        )

        # Correlation annotation
        ax1.text(
            0.05,
            0.95,
            f"r = {r_value:.3f}\np = {p_value:.3e}\nn = {len(df_f1)}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor=COLORS["background"],
                alpha=0.9,
                edgecolor=COLORS["text_secondary"],
            ),
        )

        f1_corr = r_value
        f1_p = p_value
        f1_n = len(df_f1)
    else:
        f1_corr = np.nan
        f1_p = np.nan
        f1_n = 0

    ax1.set_xlabel("Outlier Detection F1 Score", fontsize=11)
    ax1.set_ylabel("CatBoost AUROC", fontsize=11)
    ax1.set_title(
        "A. Outlier Detection Quality vs Classification", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: AUROC vs Imputation MAE
    # =========================================================================
    ax2 = axes[1]

    # Filter to rows with valid imputation MAE
    df_mae = df.dropna(subset=["imputation_mae", "catboost_auroc"]).copy()

    if len(df_mae) > 0:
        x = df_mae["imputation_mae"].values
        y = df_mae["catboost_auroc"].values

        # Scatter plot
        ax2.scatter(
            x,
            y,
            c=COLORS["secondary"],
            alpha=0.7,
            s=60,
            edgecolors=COLORS["background"],
            linewidths=0.5,
        )

        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax2.plot(
            x_line, y_line, color=COLORS["bad"], linewidth=2, linestyle="--", alpha=0.8
        )

        # Correlation annotation
        ax2.text(
            0.05,
            0.95,
            f"r = {r_value:.3f}\np = {p_value:.3f}\nn = {len(df_mae)}",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor=COLORS["background"],
                alpha=0.9,
                edgecolor=COLORS["text_secondary"],
            ),
        )

        mae_corr = r_value
        mae_p = p_value
        mae_n = len(df_mae)
    else:
        mae_corr = np.nan
        mae_p = np.nan
        mae_n = 0

    ax2.set_xlabel("Imputation MAE", fontsize=11)
    ax2.set_ylabel("CatBoost AUROC", fontsize=11)
    ax2.set_title(
        "B. Imputation Quality vs Classification", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Overall styling
    # =========================================================================
    plt.tight_layout()

    # Add overall title
    fig.suptitle(
        "Preprocessing Quality → Classification Performance (CatBoost)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Prepare data dict
    data_dict = {
        "panel_a": {
            "metric": "outlier_f1",
            "correlation_r": float(f1_corr) if not np.isnan(f1_corr) else None,
            "correlation_p": float(f1_p) if not np.isnan(f1_p) else None,
            "n_points": f1_n,
        },
        "panel_b": {
            "metric": "imputation_mae",
            "correlation_r": float(mae_corr) if not np.isnan(mae_corr) else None,
            "correlation_p": float(mae_p) if not np.isnan(mae_p) else None,
            "n_points": mae_n,
        },
        "interpretation": {
            "f1_correlation": "Moderate positive correlation - better outlier detection → better AUROC",
            "mae_correlation": "Weak correlation - imputation quality has less direct impact on AUROC",
        },
    }

    return fig, data_dict


def main() -> None:
    """Generate and save scatter plots."""
    print("Creating preprocessing correlation scatter plots...")

    fig, data = create_figure()

    # Save figure
    save_figure(fig, "fig_preprocessing_vs_auroc", data=data, formats=("png", "pdf"))

    print("\n=== Correlation Summary ===")
    print(
        f"Outlier F1 vs AUROC: r = {data['panel_a']['correlation_r']:.3f} (n = {data['panel_a']['n_points']})"
    )
    print(
        f"Imputation MAE vs AUROC: r = {data['panel_b']['correlation_r']:.3f} (n = {data['panel_b']['n_points']})"
    )

    print("\n=== Interpretation ===")
    print(data["interpretation"]["f1_correlation"])
    print(data["interpretation"]["mae_correlation"])

    plt.close(fig)


if __name__ == "__main__":
    main()
