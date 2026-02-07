#!/usr/bin/env python
"""
utility_matrix.py - Figure C3: Foundation Model Utility Matrix

Simple 2x3 matrix visualization showing where foundation models are useful
vs where they underperform:

                    Foundation Models
Task                Useful          Not Useful
─────────────────────────────────────────────
Outlier Detection     ✓
Imputation            ✓
Featurization                          ✗

Key finding: Foundation models competitive for preprocessing,
but substantially underperform for feature extraction.

Usage:
    python src/viz/utility_matrix.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from figure_dimensions import get_dimensions
from plot_config import (
    COLORS,
    FIXED_CLASSIFIER,
    KEY_STATS,
    get_connection,
    save_figure,
    setup_style,
)

# ============================================================================
# UTILITY ASSESSMENT DATA LOADING
# ============================================================================

# Load comparison methods from YAML combos (single source of truth).
# FM representative: best_single_fm combo's outlier method
# Traditional baseline: traditional combo's outlier and imputation methods
# FM imputation: lof_moment combo's imputation method (MOMENT zero-shot)
_COMBOS_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "VISUALIZATION"
    / "plot_hyperparam_combos.yaml"
)


def _load_comparison_methods() -> Dict[str, str]:
    """Load comparison method names from YAML combos."""
    with open(_COMBOS_PATH) as f:
        config = yaml.safe_load(f)

    combos_by_id = {}
    for combo in config.get("standard_combos", []):
        combos_by_id[combo["id"]] = combo
    for combo in config.get("extended_combos", []):
        combos_by_id[combo["id"]] = combo

    return {
        "outlier_fm": combos_by_id["best_single_fm"]["outlier_method"],
        "outlier_baseline": combos_by_id["traditional"]["outlier_method"],
        "imputation_fm": combos_by_id["lof_moment"]["imputation_method"],
        "imputation_baseline": combos_by_id["traditional"]["imputation_method"],
    }


_COMPARISON_METHODS = _load_comparison_methods()
_OUTLIER_FM = _COMPARISON_METHODS["outlier_fm"]
_OUTLIER_BASELINE = _COMPARISON_METHODS["outlier_baseline"]
_IMPUTATION_FM = _COMPARISON_METHODS["imputation_fm"]
_IMPUTATION_BASELINE = _COMPARISON_METHODS["imputation_baseline"]


def get_utility_data() -> Dict[str, Dict[str, Any]]:
    """
    Load utility assessment data from database.

    Compares FM methods vs traditional methods for each pipeline stage.
    All comparisons use the fixed classifier (research design).
    """
    conn = get_connection()

    # Outlier Detection: FM vs traditional
    outlier_df = conn.execute(f"""
        SELECT outlier_method, AVG(auroc) as mean_auroc
        FROM essential_metrics
        WHERE classifier = '{FIXED_CLASSIFIER}'
        AND outlier_method IN ('{_OUTLIER_FM}', '{_OUTLIER_BASELINE}')
        GROUP BY outlier_method
    """).fetchdf()

    fm_outlier = outlier_df[outlier_df["outlier_method"] == _OUTLIER_FM][
        "mean_auroc"
    ].values[0]
    baseline_outlier = outlier_df[outlier_df["outlier_method"] == _OUTLIER_BASELINE][
        "mean_auroc"
    ].values[0]

    # Imputation: FM vs trained DL
    imputation_df = conn.execute(f"""
        SELECT imputation_method, AVG(auroc) as mean_auroc
        FROM essential_metrics
        WHERE classifier = '{FIXED_CLASSIFIER}'
        AND imputation_method IN ('{_IMPUTATION_FM}', '{_IMPUTATION_BASELINE}')
        GROUP BY imputation_method
    """).fetchdf()

    fm_imputation = imputation_df[imputation_df["imputation_method"] == _IMPUTATION_FM][
        "mean_auroc"
    ].values[0]
    baseline_imputation = imputation_df[
        imputation_df["imputation_method"] == _IMPUTATION_BASELINE
    ]["mean_auroc"].values[0]

    conn.close()

    return {
        "Outlier Detection": {
            "useful": True,
            "fm_performance": round(fm_outlier, 3),  # MOMENT-gt-finetune
            "baseline_performance": round(baseline_outlier, 3),  # LOF
            "note": "MOMENT competitive with traditional",
        },
        "Imputation": {
            "useful": True,
            "fm_performance": round(fm_imputation, 3),  # MOMENT zero-shot
            "baseline_performance": round(baseline_imputation, 3),  # SAITS (trained)
            "note": "Zero-shot matches trained methods",
        },
        "Featurization": {
            "useful": False,
            "fm_performance": KEY_STATS["embeddings_mean_auroc"],  # 0.740
            "baseline_performance": KEY_STATS["handcrafted_mean_auroc"],  # 0.830
            "note": "9pp deficit vs handcrafted",
        },
    }


def create_figure() -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create the utility matrix figure."""
    setup_style()

    # Load utility data from database
    UTILITY_DATA = get_utility_data()

    fig, ax = plt.subplots(figsize=get_dimensions("single"))

    # Grid setup
    tasks = list(UTILITY_DATA.keys())

    # Cell dimensions
    cell_width = 0.35
    cell_height = 0.18
    x_start = 0.15
    y_start = 0.75

    # Column headers
    ax.text(
        x_start + cell_width / 2,
        y_start + 0.12,
        "Useful",
        ha="center",
        fontweight="bold",
        fontsize=11,
        color=COLORS["good"],
    )
    ax.text(
        x_start + cell_width * 1.5 + 0.05,
        y_start + 0.12,
        "Not Useful",
        ha="center",
        fontweight="bold",
        fontsize=11,
        color=COLORS["bad"],
    )

    # Row labels and cells
    for i, task in enumerate(tasks):
        y = y_start - i * (cell_height + 0.05)
        data = UTILITY_DATA[task]

        # Task label (left side)
        ax.text(
            x_start - 0.02,
            y,
            task,
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        # Determine cell positions
        if data["useful"]:
            cell_x = x_start
            cell_color = COLORS["good"]
            symbol = "✓"
        else:
            cell_x = x_start + cell_width + 0.05
            cell_color = COLORS["bad"]
            symbol = "✗"

        # Draw the filled cell
        rect = mpatches.FancyBboxPatch(
            (cell_x, y - cell_height / 2),
            cell_width,
            cell_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=cell_color,
            edgecolor=COLORS["text_primary"],
            linewidth=1.5,
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Symbol
        ax.text(
            cell_x + cell_width / 2,
            y + 0.02,
            symbol,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=COLORS["background"],
        )

        # Performance values
        fm_perf = data["fm_performance"]
        baseline_perf = data["baseline_performance"]
        ax.text(
            cell_x + cell_width / 2,
            y - 0.04,
            f"FM: {fm_perf:.2f} vs {baseline_perf:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["background"],
        )

        # Draw empty cell for the other column
        other_x = x_start + cell_width + 0.05 if data["useful"] else x_start
        rect_empty = mpatches.FancyBboxPatch(
            (other_x, y - cell_height / 2),
            cell_width,
            cell_height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=COLORS["background_neutral"],
            edgecolor=COLORS["grid_lines"],
            linewidth=1,
            alpha=0.5,
        )
        ax.add_patch(rect_empty)

    # Add note at bottom
    ax.text(
        0.5,
        0.08,
        "FM = Foundation Model performance (AUROC)\n"
        "Baseline = Traditional/handcrafted method performance",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
        color=COLORS["text_secondary"],
        transform=ax.transAxes,
    )

    # Main title
    ax.set_title(
        "Foundation Model Utility by Pipeline Stage\n"
        "Task-Dependent Performance Patterns",
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
        "tasks": tasks,
        "utility_assessment": {k: v["useful"] for k, v in UTILITY_DATA.items()},
        "fm_performance": {k: v["fm_performance"] for k, v in UTILITY_DATA.items()},
        "baseline_performance": {
            k: v["baseline_performance"] for k, v in UTILITY_DATA.items()
        },
    }


def main() -> None:
    """Generate and save the figure."""
    print("Generating Figure C3: Foundation Model Utility Matrix...")

    fig, data = create_figure()
    save_figure(fig, "fig_C3_utility_matrix", data=data)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
