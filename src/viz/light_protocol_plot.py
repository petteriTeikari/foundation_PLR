"""
Light Protocol Visualization

Plot the chromatic pupillometry light exposure protocol from the SERI PLR database.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import matplotlib.pyplot as plt

from src.viz.plot_config import COLORS, save_figure

logger = logging.getLogger(__name__)


def plot_light_protocol(
    db_path: str,
    output_path: str,
    subject_code: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 3),
    dpi: int = 300,
) -> str:
    """
    Plot the chromatic pupillometry light exposure protocol.

    The protocol consists of:
    - Baseline (dark adaptation): ~15.5s
    - Blue light ramp: ~15.5-24.7s (intensity 0→231)
    - Recovery 1 (PIPR window): ~24.7-46.5s
    - Red light ramp: ~46.5-55.6s (intensity 0→231)
    - Recovery 2: ~55.6-66s

    Args:
        db_path: Path to SERI_PLR_GLAUCOMA.db
        output_path: Path for output figure (without extension)
        subject_code: Specific subject to use, or None for first subject
        figsize: Figure size in inches
        dpi: DPI for PNG output

    Returns:
        Base path of saved figures
    """
    # Connect to database
    conn = duckdb.connect(db_path, read_only=True)

    # Get subject code if not specified
    if subject_code is None:
        subject_code = conn.execute(
            "SELECT DISTINCT subject_code FROM train LIMIT 1"
        ).fetchone()[0]

    # Get data for one subject
    df = conn.execute(
        f"""
        SELECT time, Red, Blue, pupil_raw, pupil_gt
        FROM train
        WHERE subject_code = '{subject_code}'
        ORDER BY time
        """
    ).fetchdf()
    conn.close()

    # Create single figure showing light protocol
    fig, ax = plt.subplots(figsize=figsize)

    # Plot light stimulus
    ax.fill_between(
        df["time"],
        df["Blue"],
        alpha=0.7,
        color=COLORS["blue_stimulus"],
        label="Blue (469 nm)",
    )
    ax.fill_between(
        df["time"],
        df["Red"],
        alpha=0.7,
        color=COLORS["red_stimulus"],
        label="Red (640 nm)",
    )
    ax.set_ylabel("Light Intensity (device units)", fontsize=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 260)
    ax.set_xlim(0, 66)

    # Add phase annotations
    phases = [
        (0, 15.5, "Baseline\n(dark)", COLORS["background_light"]),
        (15.5, 24.7, "Blue\nramp", COLORS["blue_zone"]),
        (24.7, 46.5, "Recovery\n(PIPR)", COLORS["background_light"]),
        (46.5, 55.6, "Red\nramp", COLORS["red_zone"]),
        (55.6, 66, "Recovery", COLORS["background_light"]),
    ]

    for start, end, label, color in phases:
        ax.axvspan(start, end, alpha=0.3, color=color, zorder=0)
        ax.text((start + end) / 2, 245, label, ha="center", va="top", fontsize=8)

    ax.set_title(
        "Chromatic Pupillometry Light Protocol", fontsize=11, fontweight="bold"
    )

    # Tight layout
    plt.tight_layout()

    # Save using figure system
    output_base = Path(output_path)
    filename = output_base.stem if output_base.suffix else output_base.name

    # Prepare data for JSON export
    data = {
        "subject_code": subject_code,
        "phases": [{"start": s, "end": e, "label": lbl} for s, e, lbl, _ in phases],
        "time_range": [float(df["time"].min()), float(df["time"].max())],
    }

    saved_path = save_figure(
        fig,
        filename,
        data=data,
        output_dir=output_base.parent if output_base.parent != Path(".") else None,
    )

    plt.close(fig)

    return str(saved_path.with_suffix(""))


if __name__ == "__main__":
    # Test the function
    import os

    logging.basicConfig(level=logging.INFO)

    # Use environment variable or default relative path
    db_path = os.environ.get(
        "SERI_PLR_DB_PATH",
        str(Path(__file__).parent.parent.parent.parent / "SERI_PLR_GLAUCOMA.db"),
    )
    output_path = "fig_light_protocol"

    result = plot_light_protocol(db_path, output_path)
    print(f"Saved figures to: {result}")
