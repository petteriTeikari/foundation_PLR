import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_io.data_utils import get_list_of_unique_subjects
from src.utils import get_artifacts_dir
from src.viz.plot_config import save_figure
from src.viz.viz_styling_utils import blank_subplot
from src.viz.viz_subplots import viz_input_subplot
from src.viz.viz_utils import get_font_scaler


def pick_input_viz_numpy_arrays(df_subject: pl.DataFrame) -> dict:
    """Extract time and pupil columns from a subject DataFrame as NumPy arrays.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Single-subject Polars DataFrame containing ``time``, ``pupil_orig``,
        ``pupil_raw``, and ``gt`` columns.

    Returns
    -------
    dict
        Nested dict with ``"x"`` (time array) and ``"y"`` (orig, raw, gt arrays).
    """
    # Get the numpy arrays for the visualization
    return {
        "x": {
            "time": df_subject["time"].to_numpy(),
        },
        "y": {
            "orig": df_subject["pupil_orig"].to_numpy(),
            "raw": df_subject["pupil_raw"].to_numpy(),
            "gt": df_subject["gt"].to_numpy(),
        },
    }


def compute_PLR_residuals(data_to_plot):
    """Compute residuals between original/raw signals and ground truth.

    Parameters
    ----------
    data_to_plot : dict
        Nested dict produced by :func:`pick_input_viz_numpy_arrays` with
        ``"x"`` and ``"y"`` sub-dicts.

    Returns
    -------
    dict
        Nested dict with ``"x"`` (time) and ``"y"`` containing
        ``"orig-gt"`` and ``"raw-gt"`` residual arrays.
    """
    return {
        "x": {
            "time": data_to_plot["x"]["time"],
        },
        "y": {
            "orig-gt": data_to_plot["y"]["orig"] - data_to_plot["y"]["gt"],
            "raw-gt": data_to_plot["y"]["raw"] - data_to_plot["y"]["gt"],
        },
    }


def visualize_input_per_subject(df_subject, code, cfg, viz_cfg):
    """Generate a 2-row figure showing PLR signals and residuals for one subject.

    Row 0 plots the original, raw, and ground-truth signals. Row 1 plots
    the corresponding residuals against ground truth.

    Parameters
    ----------
    df_subject : pl.DataFrame
        Single-subject Polars DataFrame with PLR signal columns.
    code : str
        Subject identifier used as filename and in the figure title.
    cfg : DictConfig
        Full Hydra config (unused directly but forwarded for consistency).
    viz_cfg : DictConfig
        Visualization sub-config with ``SNS``, ``dpi``, and font settings.

    Returns
    -------
    str
        Absolute path to the saved figure PNG.
    """
    # Get the data as numpy arrays in dicts
    data_to_plot = pick_input_viz_numpy_arrays(df_subject)
    residuals_to_plot = compute_PLR_residuals(data_to_plot)

    n_cols = 3
    n_rows = 2

    output_dir = get_artifacts_dir(service_name="figures_debug")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style(style=viz_cfg["SNS"]["style"])
    sns.set_context("notebook", rc={"lines.linewidth": 2})
    sns.set_palette(viz_cfg["SNS"]["palette"])

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(19.2, 10.8),
        dpi=viz_cfg["dpi"],
        tight_layout=True,
    )
    missingness_ratio = 100 * (df_subject["no_outliers"][0] / df_subject.shape[0])
    fig.suptitle(
        f"{code} | missingness percentage = {missingness_ratio:.2f}%",
        fontsize=int(get_font_scaler(viz_cfg) * 12),
    )

    # Plot the original data
    r = 0
    for col, col_name in enumerate(data_to_plot["y"].keys()):
        ax = axes[r, col]
        viz_input_subplot(
            ax=ax,
            x=data_to_plot["x"]["time"],
            y=data_to_plot["y"][col_name],
            col_name=col_name,
            viz_cfg=viz_cfg,
            title_prefix="PLR",
        )

    # Plot the Residuals
    r = 1
    for col, col_name in enumerate(residuals_to_plot["y"].keys()):
        ax = axes[r, col]
        viz_input_subplot(
            ax=ax,
            x=residuals_to_plot["x"]["time"],
            y=residuals_to_plot["y"][col_name],
            col_name=col_name,
            viz_cfg=viz_cfg,
            title_prefix="Residuals",
            y_lims=(-10, 10),
        )
    blank_subplot(ax_in=axes[r, col + 1], viz_cfg=viz_cfg)

    fig = plt.gcf()
    output_dir_path = Path(output_dir) if output_dir else None
    saved_path = save_figure(fig, code, output_dir=output_dir_path)
    return str(saved_path)


# @task(
#     log_prints=True,
#     name="Visualize the imported data",
#     description="Examine that the data quality is good",
# )
def visualize_input_data(df: pl.DataFrame, cfg: DictConfig, PLR_length: int = 1981):
    """Generate per-subject input visualization figures for all subjects.

    Iterate over every unique subject in *df*, assert the expected PLR
    length, and produce a diagnostic figure via
    :func:`visualize_input_per_subject`.

    Parameters
    ----------
    df : pl.DataFrame
        Multi-subject Polars DataFrame containing a ``subject_code``
        column and the standard PLR signal columns.
    cfg : DictConfig
        Full Hydra configuration (``VISUALIZATION`` sub-key is used).
    PLR_length : int, optional
        Expected number of rows per subject. Default is ``1981``.

    Returns
    -------
    list[str]
        Paths to the saved figure PNGs, one per subject.

    Raises
    ------
    AssertionError
        If any subject's row count does not equal *PLR_length*.
    """
    unique_subjects = sorted(get_list_of_unique_subjects(df))
    paths_out = []
    for i, code in enumerate(
        tqdm(unique_subjects, desc="Visualizing data", total=len(unique_subjects))
    ):
        df_subject = df.filter(pl.col("subject_code") == code)
        assert df_subject.shape[0] == PLR_length, (
            f"Subject {code} has {df_subject.shape[0]} rows, expected {PLR_length}"
        )
        paths_out.append(
            visualize_input_per_subject(
                df_subject, code, cfg, viz_cfg=cfg["VISUALIZATION"]
            )
        )

    return paths_out
