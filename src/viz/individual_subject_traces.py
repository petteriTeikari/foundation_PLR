#!/usr/bin/env python3
"""
individual_subject_traces.py - Plot PLR signal traces for individual subjects.

Shows PLR signal at different preprocessing stages for 12 demo subjects:
- 6 Control subjects (2 high outlier, 2 average, 2 low outlier)
- 6 Glaucoma subjects (2 high outlier, 2 average, 2 low outlier)

For each subject shows:
- Raw signal with outliers marked
- Ground truth signal

This demonstrates the signal quality variation across subjects and
the impact of different outlier percentages.

Usage:
    python src/viz/individual_subject_traces.py
"""

# Import shared config
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import COLORS, get_category_display_name, setup_style

# Database path - use centralized path utilities
from src.utils.paths import get_seri_db_path, get_visualization_config_dir

DB_PATH = get_seri_db_path()


def load_demo_subjects() -> dict:
    """Load demo subjects configuration."""
    # Use centralized config path utility
    config_path = get_visualization_config_dir() / "demo_subjects.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"demo_subjects.yaml not found at: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_subject_data(subject_code: str) -> pd.DataFrame:
    """Load signal data for a specific subject."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # Try train table first, then test
    for table in ["train", "test"]:
        df = conn.execute(
            f"""
            SELECT
                time,
                pupil_orig,
                pupil_gt,
                outlier_mask,
                light_stimuli
            FROM {table}
            WHERE subject_code = ?
            ORDER BY time
        """,
            [subject_code],
        ).fetchdf()

        if len(df) > 0:
            break

    conn.close()
    return df


def plot_subject_traces(subjects_list: list, class_label: str) -> tuple:
    """
    Plot signal traces for a list of subjects.

    Args:
        subjects_list: List of subject dicts with 'code', 'outlier_pct', 'note'
        class_label: 'control' or 'glaucoma'

    Returns:
        (fig, data_dict) tuple
    """
    n_subjects = len(subjects_list)

    fig, axes = plt.subplots(n_subjects, 1, figsize=(14, 2.5 * n_subjects), sharex=True)
    if n_subjects == 1:
        axes = [axes]

    data_records = []

    for idx, subject_info in enumerate(subjects_list):
        ax = axes[idx]
        subject_code = subject_info["code"]
        outlier_pct = subject_info["outlier_pct"]
        note = subject_info["note"]

        # Load data
        df = get_subject_data(subject_code)

        if len(df) == 0:
            ax.text(
                0.5,
                0.5,
                f"No data for {subject_code}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        time = df["time"].values
        pupil_orig = df["pupil_orig"].values
        pupil_gt = df["pupil_gt"].values
        outlier_mask = df["outlier_mask"].values == 1
        light = df["light_stimuli"].values

        # Plot ground truth (reference)
        ax.plot(
            time,
            pupil_gt,
            color=COLORS["good"],
            linewidth=1.5,
            label=get_category_display_name("ground_truth"),
            alpha=0.8,
        )

        # Plot original signal with outliers
        ax.plot(
            time,
            pupil_orig,
            color=COLORS["neutral"],
            linewidth=0.8,
            label="Raw",
            alpha=0.6,
        )

        # Mark outliers
        outlier_times = time[outlier_mask]
        outlier_values = pupil_orig[outlier_mask]
        ax.scatter(
            outlier_times,
            outlier_values,
            color=COLORS["bad"],
            s=10,
            alpha=0.6,
            label=f"Outliers ({outlier_pct:.1f}%)",
            zorder=5,
        )

        # Add light stimulus bar at top
        light_on = light > 0.5
        if np.any(light_on):
            light_starts = np.where(np.diff(light_on.astype(int)) == 1)[0]
            light_ends = np.where(np.diff(light_on.astype(int)) == -1)[0]
            for start, end in zip(light_starts, light_ends):
                ax.axvspan(time[start], time[end], alpha=0.2, color=COLORS["highlight"])

        # Labels
        ax.set_ylabel("Pupil Size\n(normalized)")
        ax.set_title(f"{subject_code} - {note}", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

        # Record data
        data_records.append(
            {
                "subject_code": subject_code,
                "class": class_label,
                "outlier_pct": outlier_pct,
                "n_samples": len(df),
                "n_outliers": int(np.sum(outlier_mask)),
            }
        )

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    return fig, data_records


def create_figures() -> tuple:
    """
    Create control and glaucoma trace figures.

    Returns:
        ((fig_control, data_control), (fig_glaucoma, data_glaucoma))
    """
    setup_style()

    # Load configuration
    config = load_demo_subjects()
    demo_subjects = config["demo_subjects"]

    # Flatten subject lists
    control_subjects = []
    for category in ["high_outlier", "average_outlier", "low_outlier"]:
        control_subjects.extend(demo_subjects["control"][category])

    glaucoma_subjects = []
    for category in ["high_outlier", "average_outlier", "low_outlier"]:
        glaucoma_subjects.extend(demo_subjects["glaucoma"][category])

    # Create figures
    print("Creating control subject traces...")
    fig_control, data_control = plot_subject_traces(control_subjects, "control")
    fig_control.suptitle(
        "Control Subjects - PLR Signal Quality", fontsize=14, fontweight="bold", y=1.01
    )

    print("Creating glaucoma subject traces...")
    fig_glaucoma, data_glaucoma = plot_subject_traces(glaucoma_subjects, "glaucoma")
    fig_glaucoma.suptitle(
        "Glaucoma Subjects - PLR Signal Quality", fontsize=14, fontweight="bold", y=1.01
    )

    return (fig_control, data_control), (fig_glaucoma, data_glaucoma)


def main():
    """Generate and save subject trace figures."""
    from plot_config import save_figure

    print("Creating individual subject trace plots...")

    (fig_control, data_control), (fig_glaucoma, data_glaucoma) = create_figures()

    # Prepare data for JSON export
    data_control_dict = {
        "subjects": data_control,
        "n_subjects": len(data_control),
        "class": "control",
    }
    data_glaucoma_dict = {
        "subjects": data_glaucoma,
        "n_subjects": len(data_glaucoma),
        "class": "glaucoma",
    }

    # Save using standard save_figure (PNG + SVG, no PDF)
    save_figure(fig_control, "fig_subject_traces_control", data=data_control_dict)
    save_figure(fig_glaucoma, "fig_subject_traces_glaucoma", data=data_glaucoma_dict)

    print("\n=== Summary ===")
    print(f"Control subjects: {len(data_control)}")
    print(f"Glaucoma subjects: {len(data_glaucoma)}")

    plt.close(fig_control)
    plt.close(fig_glaucoma)


if __name__ == "__main__":
    main()
