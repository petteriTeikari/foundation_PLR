#!/usr/bin/env python
"""
foundation_model_dashboard.py - Figure R8: Foundation Model Performance Dashboard

Multi-panel figure showing foundation model performance across pipeline stages:
- Panel A: Outlier detection (MOMENT, UniTS vs traditional)
- Panel B: Imputation (zero-shot vs fine-tuned vs trained)
- Panel C: Featurization (embeddings vs handcrafted)

Key finding: Foundation models are competitive for preprocessing but
substantially underperform for feature extraction.

Usage:
    python src/viz/foundation_model_dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_dimensions import get_dimensions
from plot_config import (
    COLORS,
    get_category_display_name,
    get_connection,
    save_figure,
    setup_style,
)


def fetch_outlier_performance() -> pd.DataFrame:
    """Fetch outlier detection method performance."""
    conn = get_connection()

    query = """
    SELECT
        outlier_method,
        AVG(auroc) as mean_auroc,
        STDDEV(auroc) as std_auroc,
        COUNT(*) as n
    FROM essential_metrics
    WHERE auroc IS NOT NULL
      AND outlier_method IS NOT NULL
      AND outlier_method != 'Unknown'
    GROUP BY outlier_method
    ORDER BY mean_auroc DESC
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df


def fetch_imputation_performance() -> pd.DataFrame:
    """Fetch imputation method performance."""
    conn = get_connection()

    query = """
    SELECT
        imputation_method,
        AVG(auroc) as mean_auroc,
        STDDEV(auroc) as std_auroc,
        COUNT(*) as n
    FROM essential_metrics
    WHERE auroc IS NOT NULL
      AND imputation_method IS NOT NULL
      AND imputation_method != 'Unknown'
    GROUP BY imputation_method
    ORDER BY mean_auroc DESC
    """

    df = conn.execute(query).fetchdf()
    conn.close()
    return df


def fetch_featurization_performance() -> pd.DataFrame:
    """Fetch featurization method performance."""
    conn = get_connection()

    query = """
    SELECT
        featurization,
        AVG(auroc) as mean_auroc,
        STDDEV(auroc) as std_auroc,
        COUNT(*) as n
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


def categorize_method(method_name: str, task: str) -> str:
    """Categorize a method as foundation model, traditional, or other."""
    method_lower = method_name.lower()

    # Foundation model indicators
    fm_keywords = ["moment", "units", "chronos", "timesnet", "saits", "csdi"]

    # Traditional method indicators
    trad_keywords = ["lof", "svm", "isolation", "prophet", "linear", "mean", "median"]

    # Ensemble indicators
    ensemble_keywords = ["ensemble", "combined", "meta"]

    if any(kw in method_lower for kw in ensemble_keywords):
        return "ensemble"
    elif any(kw in method_lower for kw in fm_keywords):
        return "foundation_model"
    elif any(kw in method_lower for kw in trad_keywords):
        return "traditional"
    elif "ground" in method_lower or "gt" in method_lower or "pupil-gt" in method_lower:
        return "ground_truth"
    elif "handcrafted" in method_lower:
        return "handcrafted"
    elif "embed" in method_lower:
        return "embeddings"
    else:
        return "other"


def get_color_for_category(category: str) -> str:
    """Get color for a method category."""
    color_map = {
        "foundation_model": COLORS["foundation_model"],
        "traditional": COLORS["traditional"],
        "ensemble": COLORS["ensemble"],
        "ground_truth": COLORS["neutral"],
        "handcrafted": COLORS["handcrafted"],
        "embeddings": COLORS["embeddings"],
        "other": COLORS["neutral"],
    }
    return color_map.get(category, COLORS["neutral"])


def create_horizontal_bar_panel(
    ax: plt.Axes, df: pd.DataFrame, task: str, title: str, show_ylabel: bool = True
) -> list[str]:
    """Create a horizontal bar chart for one panel."""
    # Determine the method column name based on what's in the dataframe
    if "outlier_method" in df.columns:
        method_col = "outlier_method"
    elif "imputation_method" in df.columns:
        method_col = "imputation_method"
    elif "featurization" in df.columns:
        method_col = "featurization"
    else:
        raise ValueError(f"Unknown columns in df: {df.columns.tolist()}")

    methods = df[method_col].tolist()
    means = df["mean_auroc"].tolist()
    stds = df["std_auroc"].tolist()

    # Limit to top 8 methods for clarity
    if len(methods) > 8:
        methods = methods[:8]
        means = means[:8]
        stds = stds[:8]

    # Shorten method names for display
    display_names = []
    for m in methods:
        name = m.replace("ensemble-", "Ens:").replace("MOMENT-", "MOM-")
        name = name.replace("moment_", "MOM-").replace(
            "pupil-gt", get_category_display_name("ground_truth")
        )
        name = name.replace("handcrafted", "Handcrafted")
        if len(name) > 20:
            name = name[:18] + "..."
        display_names.append(name)

    # Colors by category
    categories = [categorize_method(m, task) for m in methods]
    colors = [get_color_for_category(c) for c in categories]

    # Plot
    y_pos = np.arange(len(methods))
    bars = ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        height=0.7,
    )

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(mean + 0.01, i, f"{mean:.3f}", va="center", fontsize=8)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlim(0.65, 0.95)
    ax.set_xlabel("Mean AUROC")
    ax.set_title(title, fontweight="bold", fontsize=10)

    if not show_ylabel:
        ax.set_yticklabels([])

    # Invert y-axis so best is at top
    ax.invert_yaxis()

    return categories


def create_figure() -> tuple[plt.Figure, dict]:
    """Create the 3-panel foundation model dashboard."""
    setup_style()

    # Fetch all data
    outlier_df = fetch_outlier_performance()
    imputation_df = fetch_imputation_performance()
    feature_df = fetch_featurization_performance()

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=get_dimensions("dashboard"))

    # Panel A: Outlier Detection
    create_horizontal_bar_panel(
        axes[0], outlier_df, "outlier", "A. Outlier Detection\n(Downstream AUROC)"
    )

    # Panel B: Imputation
    create_horizontal_bar_panel(
        axes[1],
        imputation_df,
        "imputation",
        "B. Imputation\n(Downstream AUROC)",
        show_ylabel=True,
    )

    # Panel C: Featurization
    create_horizontal_bar_panel(
        axes[2],
        feature_df,
        "featurization",
        "C. Featurization\n(Classification AUROC)",
        show_ylabel=True,
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=COLORS["foundation_model"],
            edgecolor="black",
            label=get_category_display_name("foundation_model"),
        ),
        Patch(
            facecolor=COLORS["traditional"],
            edgecolor="black",
            label=get_category_display_name("traditional"),
        ),
        Patch(
            facecolor=COLORS["ensemble"],
            edgecolor="black",
            label=get_category_display_name("ensemble"),
        ),
        Patch(facecolor=COLORS["handcrafted"], edgecolor="black", label="Handcrafted"),
        Patch(facecolor=COLORS["embeddings"], edgecolor="black", label="Embeddings"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    # Main title
    fig.suptitle(
        "Foundation Model Performance Across Pipeline Stages",
        fontweight="bold",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()

    return fig, {
        "outlier": {
            "methods": outlier_df["outlier_method"].tolist()[:8],
            "auroc": outlier_df["mean_auroc"].tolist()[:8],
        },
        "imputation": {
            "methods": imputation_df["imputation_method"].tolist()[:8],
            "auroc": imputation_df["mean_auroc"].tolist()[:8],
        },
        "featurization": {
            "methods": feature_df["featurization"].tolist(),
            "auroc": feature_df["mean_auroc"].tolist(),
        },
    }


def main() -> None:
    """Generate and save the figure."""
    print("Generating Figure R8: Foundation Model Dashboard...")

    fig, data = create_figure()
    save_figure(fig, "fig_R8_foundation_model_dashboard", data=data)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
