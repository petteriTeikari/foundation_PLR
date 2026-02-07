#!/usr/bin/env python
"""
factorial_matrix.py - Figure M3: Factorial Design Matrix

Visualizes the factorial experimental design structure:
- 2 featurization methods
- 7+ outlier detection methods (including ensembles)
- 5+ imputation methods (including ensembles)
- 5 classifiers

Total: 407 unique configurations

Usage:
    python src/viz/factorial_matrix.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from figure_dimensions import get_dimensions
from plot_config import (
    COLORS,
    get_category_display_name,
    get_connection,
    save_figure,
    setup_style,
)


def fetch_factorial_counts() -> Tuple[Dict[str, List[str]], int]:
    """Fetch counts of configurations per factor combination."""
    conn = get_connection()

    # Get unique values per factor
    queries = {
        "outlier": "SELECT DISTINCT outlier_method FROM essential_metrics WHERE outlier_method IS NOT NULL AND outlier_method != 'Unknown'",
        "imputation": "SELECT DISTINCT imputation_method FROM essential_metrics WHERE imputation_method IS NOT NULL AND imputation_method != 'Unknown'",
        "featurization": "SELECT DISTINCT featurization FROM essential_metrics WHERE featurization IS NOT NULL AND featurization != 'Unknown'",
        "classifier": "SELECT DISTINCT classifier FROM essential_metrics WHERE classifier IS NOT NULL AND classifier != 'Unknown'",
    }

    factors = {}
    for name, query in queries.items():
        result = conn.execute(query).fetchall()
        factors[name] = [r[0] for r in result if r[0]]

    # Get total configurations
    total = conn.execute(
        "SELECT COUNT(*) FROM essential_metrics WHERE auroc IS NOT NULL"
    ).fetchone()[0]

    conn.close()

    return factors, total


def create_figure() -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create the factorial design visualization."""
    setup_style()

    factors, total_configs = fetch_factorial_counts()

    # Create figure
    fig, ax = plt.subplots(figsize=get_dimensions("matrix"))

    # Design as hierarchical boxes
    # Layout: centered pipeline stages from left to right

    box_height = 0.12
    stage_x = [0.1, 0.35, 0.6, 0.85]  # X positions for 4 stages
    stage_labels = [
        "Outlier\nDetection",
        "Imputation",
        "Featurization",
        "Classification",
    ]
    stage_keys = ["outlier", "imputation", "featurization", "classifier"]

    # Colors for different method types
    def get_method_color(method: str) -> str:
        method_lower = method.lower()
        if "ensemble" in method_lower:
            return COLORS["ensemble"]
        elif any(
            kw in method_lower
            for kw in ["moment", "units", "chronos", "timesnet", "saits", "csdi"]
        ):
            return COLORS["foundation_model"]
        elif "handcrafted" in method_lower:
            return COLORS["handcrafted"]
        elif "embed" in method_lower:
            return COLORS["embeddings"]
        elif "catboost" in method_lower:
            return COLORS["catboost"]
        elif "xgboost" in method_lower:
            return COLORS["xgboost"]
        elif "tabpfn" in method_lower:
            return COLORS["tabpfn"]
        else:
            return COLORS["traditional"]

    # Draw each stage
    for i, (x, label, key) in enumerate(zip(stage_x, stage_labels, stage_keys)):
        methods = factors.get(key, [])
        n_methods = len(methods)

        # Stage header box
        header_rect = mpatches.FancyBboxPatch(
            (x - 0.08, 0.85),
            0.16,
            0.1,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=COLORS["grid_lines"],
            edgecolor=COLORS["text_primary"],
            linewidth=2,
        )
        ax.add_patch(header_rect)
        ax.text(
            x, 0.90, label, ha="center", va="center", fontweight="bold", fontsize=10
        )
        ax.text(x, 0.86, f"({n_methods} methods)", ha="center", va="center", fontsize=8)

        # Method boxes (show up to 6, then "...")
        max_show = 6
        methods_to_show = methods[:max_show]
        if len(methods) > max_show:
            methods_to_show.append(f"... +{len(methods) - max_show} more")

        y_start = 0.75
        for j, method in enumerate(methods_to_show):
            y = y_start - j * (box_height + 0.02)

            # Shorten long names
            display_name = method
            if len(display_name) > 15:
                display_name = display_name[:13] + ".."

            color = get_method_color(method) if "..." not in method else "white"

            rect = mpatches.FancyBboxPatch(
                (x - 0.07, y - box_height / 2),
                0.14,
                box_height,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor=color,
                edgecolor=COLORS["text_primary"],
                linewidth=1,
                alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(x, y, display_name, ha="center", va="center", fontsize=7)

    # Draw connecting arrows
    arrow_y = 0.90
    for i in range(len(stage_x) - 1):
        ax.annotate(
            "",
            xy=(stage_x[i + 1] - 0.09, arrow_y),
            xytext=(stage_x[i] + 0.09, arrow_y),
            arrowprops=dict(arrowstyle="->", color=COLORS["text_primary"], lw=2),
        )

    # Add multiplication symbols and counts
    mult_y = 0.78
    for i in range(len(stage_x) - 1):
        ax.text(
            (stage_x[i] + stage_x[i + 1]) / 2,
            mult_y,
            "×",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

    # Total configurations box
    total_rect = mpatches.FancyBboxPatch(
        (0.3, 0.05),
        0.4,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=COLORS["accent"],
        edgecolor=COLORS["text_primary"],
        linewidth=2,
        alpha=0.3,
    )
    ax.add_patch(total_rect)
    ax.text(
        0.5,
        0.11,
        f"Total: {total_configs} configurations",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=12,
    )
    ax.text(
        0.5,
        0.07,
        f"({len(factors['outlier'])} × {len(factors['imputation'])} × {len(factors['featurization'])} × {len(factors['classifier'])})",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Legend - load display names from config (not hardcoded)
    legend_elements = [
        mpatches.Patch(
            facecolor=COLORS["foundation_model"],
            edgecolor=COLORS["text_primary"],
            label=get_category_display_name("foundation_model"),
        ),
        mpatches.Patch(
            facecolor=COLORS["traditional"],
            edgecolor=COLORS["text_primary"],
            label=get_category_display_name("traditional"),
        ),
        mpatches.Patch(
            facecolor=COLORS["ensemble"],
            edgecolor=COLORS["text_primary"],
            label=get_category_display_name("ensemble"),
        ),
        mpatches.Patch(
            facecolor=COLORS["handcrafted"],
            edgecolor=COLORS["text_primary"],
            label="Handcrafted",  # This is a featurization type, not a category
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)

    # Title
    ax.set_title(
        "Factorial Experimental Design: Pipeline Configuration Space",
        fontweight="bold",
        fontsize=12,
        pad=20,
    )

    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()

    return fig, {
        "factors": {k: len(v) for k, v in factors.items()},
        "total_configurations": total_configs,
        "methods": factors,
    }


def main() -> None:
    """Generate and save the figure."""
    print("Generating Figure M3: Factorial Design Matrix...")

    fig, data = create_figure()
    save_figure(fig, "fig_M3_factorial_matrix", data=data)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
