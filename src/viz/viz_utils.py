from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.data_io.data_utils import get_unique_polars_rows


def pick_single_feature_with_code_from_df(
    df: pl.DataFrame,
    feature_name: str,
    code_col: str = "subject_code",
    metadata_prefix: str = "metadata_",
) -> pl.DataFrame:
    """
    Extract a single feature column along with subject code and metadata columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame containing features and metadata.
    feature_name : str
        Name of the feature column to extract.
    code_col : str, optional
        Name of the subject code column (default: "subject_code").
    metadata_prefix : str, optional
        Prefix for metadata columns to include (default: "metadata_").

    Returns
    -------
    pl.DataFrame
        DataFrame with subject code, the selected feature, and metadata columns.
    """
    features_to_keep = [code_col] + [feature_name] + [metadata_prefix]
    cols_to_keep = []
    df_out = None
    for col_to_keep in features_to_keep:
        # find substring matches in df.columns
        matches = [col for col in df.columns if col_to_keep in col]
        for match_str in matches:
            cols_to_keep.append(match_str)
        df_out = df.select(set(cols_to_keep))
    df_out = df_out.select(sorted(df_out.columns))

    return df_out


def clean_feature_name_for_title(feature_name: str) -> str:
    """
    Format a feature name as a human-readable title with color prefix.

    Parameters
    ----------
    feature_name : str
        Raw feature name with underscore-separated components (e.g., "red_amplitude_max").

    Returns
    -------
    str
        Formatted title string (e.g., "red: amplitude max").
    """
    color = feature_name.split("_")[0]
    feature_name = " ".join(feature_name.split("_")[1:])
    return f"{color}: {feature_name}"


def get_font_scaler(viz_cfg: Dict[str, Any]) -> float:
    """
    Compute font scaling factor based on DPI configuration.

    Parameters
    ----------
    viz_cfg : dict
        Visualization configuration dictionary containing "dpi" key.

    Returns
    -------
    float
        Scaling factor relative to 100 DPI baseline.
    """
    return viz_cfg["dpi"] / 100


def get_baseline_key(model: str) -> str:
    """
    Extract baseline key from model name based on suffix.

    Parameters
    ----------
    model : str
        Model name containing either "_RAW" or "_GT" suffix.

    Returns
    -------
    str
        Baseline key: "raw" or "gt".

    Raises
    ------
    ValueError
        If model name does not contain a recognized baseline suffix.
    """
    if "_RAW" in model:
        return "raw"
    elif "_GT" in model:
        return "gt"
    else:
        logger.error(f"Could not determine the baseline key for the model {model}")
        raise ValueError(f"Could not determine the baseline key for the model {model}")


def filter_imputation_df_for_orig(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_name: str = "SAITS",  # Legacy default; callers should pass explicitly
    split_key: str = "gt",
    split: str = "val",
) -> pl.DataFrame:
    """
    Filter imputation DataFrame by model, split key, and split type.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with imputation results.
    cfg : DictConfig
        Configuration containing DATA.PLR_length for validation.
    model_name : str, optional
        Name of imputation model to filter (default: "SAITS").
    split_key : str, optional
        Split key to filter ("gt" or "raw") (default: "gt").
    split : str, optional
        Data split to filter ("train", "val", "test") (default: "val").

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame matching specified criteria.

    Raises
    ------
    AssertionError
        If DataFrame row count is not a multiple of PLR_length at any stage.
    """
    assert (
        df.shape[0] % cfg["DATA"]["PLR_length"] == 0
    ), "Dataframe should have multiples of PLR recording length"
    df = df.filter(pl.col("model") == model_name)
    assert (
        df.shape[0] % cfg["DATA"]["PLR_length"] == 0
    ), "Dataframe should have multiples of PLR recording length"
    df = df.filter(pl.col("split_key") == split_key)
    assert (
        df.shape[0] % cfg["DATA"]["PLR_length"] == 0
    ), "Dataframe should have multiples of PLR recording length"
    df = df.filter(pl.col("split") == split)
    assert (
        df.shape[0] % cfg["DATA"]["PLR_length"] == 0
    ), "Dataframe should have multiples of PLR recording length"
    return df


def compute_stats_for_shaded_lineseries(
    df_to_plot: pl.DataFrame,
    cfg: DictConfig,
    y_cols: Tuple[str, ...] = ("mean",),
    error_cols: Tuple[str, ...] = ("ci_lower", "ci_upper"),
    stats_on_col: str = "time",
) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
    """
    Compute summary statistics across subjects for shaded line series plots.

    Parameters
    ----------
    df_to_plot : pl.DataFrame
        Input DataFrame with columns for time, subject_code, and values.
    cfg : DictConfig
        Configuration containing DATA.PLR_length for validation.
    y_cols : tuple, optional
        Column names to compute statistics on (default: ("mean",)).
    error_cols : tuple, optional
        Column names for error bounds (default: ("ci_lower", "ci_upper")).
    stats_on_col : str, optional
        Column to use as the x-axis/time dimension (default: "time").

    Returns
    -------
    dict
        Dictionary mapping each y_col to a dict containing:
        - n: number of subjects
        - t: time vector
        - mean: mean across subjects per timepoint
        - std: standard deviation across subjects per timepoint
        - ci_lower: 2.5th percentile
        - ci_upper: 97.5th percentile
    """

    def check_for_null_times(time_rows):
        null_rows = time_rows.filter(pl.col("time").is_null())
        if null_rows.shape[0] > 0:
            logger.warning("Found NULL times in the dataframe, why is this happening?")
            logger.warning(null_rows)
            logger.warning("Dropping these NULL times")
            time_rows = time_rows.filter(pl.col("time").is_not_null())
        return time_rows

    def check_pre_stats(df):
        time_rows = get_unique_polars_rows(df, "time")
        time_rows = check_for_null_times(time_rows)
        if time_rows.shape[0] != cfg["DATA"]["PLR_length"]:
            logger.error(
                "Expected {} unique time points, but got {} unique time points".format(
                    cfg["DATA"]["PLR_length"], time_rows.shape[0]
                )
            )
            logger.error(
                "Minimum time = {}s, Maximum time = {}s".format(
                    time_rows["time"].min(), time_rows["time"].max()
                )
            )
            ideal_time = np.linspace(0, 1980, 1981) / 30
            logger.error(
                "At 30 fps, you should get time between {} and {} seconds".format(
                    ideal_time[0], ideal_time[-1]
                )
            )
            raise ValueError(
                "Time points are not as expected, cannot visualize these, "
                "and there is something fishy about the data?"
            )

    def stats_per_column(y_col, error_cols, df):
        assert (
            df.shape[0] % cfg["DATA"]["PLR_length"] == 0
        ), "Dataframe should have multiples of PLR recording length"
        # Go to wide (spreadsheet) format
        check_pre_stats(df)
        df_wide = df.pivot(
            on="subject_code", index=stats_on_col, values=y_col
        )  # same as "melt" in pandas/ggplo2 in R
        time_vec = df_wide[stats_on_col].to_numpy()
        assert (
            time_vec.shape[0] == cfg["DATA"]["PLR_length"]
        ), f"Expected {cfg['DATA']['PLR_length']} rows, got {time_vec.shape[0]} rows"
        df_wide = df_wide.drop(stats_on_col)  # drop the time column
        no_unique_subjects = df_wide.shape[1]  # time is the first column
        assert (
            df_wide.shape[0] == cfg["DATA"]["PLR_length"]
        ), f"Expected {cfg['DATA']['PLR_length']} rows, got {df_wide.shape[0]} rows"

        # Numpy stats
        numpy_array = df_wide.to_numpy()

        return {
            "n": no_unique_subjects,
            "t": time_vec,
            "mean": np.mean(numpy_array, axis=1),
            "std": np.std(numpy_array, axis=1),
            "ci_lower": np.percentile(numpy_array, 2.5, axis=1),
            "ci_upper": np.percentile(numpy_array, 97.5, axis=1),
        }

    plot_stats = {}
    for y_col in y_cols:
        plot_stats[y_col] = stats_per_column(
            y_col=y_col, error_cols=error_cols, df=df_to_plot
        )

    return plot_stats


# @task(
#     log_prints=True,
#     name="Create MP4 Video from Figures",
#     description="Create a video from the figures on disk",
# )
def create_video_from_figures_on_disk(fig_paths: List[str], cfg: DictConfig) -> None:
    """
    Create an MP4 video from a sequence of figure images.

    Parameters
    ----------
    fig_paths : list of str
        Paths to figure images to combine into video.
    cfg : DictConfig
        Configuration with video generation settings.

    Returns
    -------
    None
        Placeholder function - video creation not yet implemented.
    """
    # moviepy, or what: https://stackoverflow.com/a/62434934/6412152
    logger.warning("Video creation placeholder")


def get_subjectwise_df_metrics(
    metrics_subjectwise: Dict[str, Dict[str, Any]], best_metric_cfg: DictConfig
) -> Tuple[pd.DataFrame, str]:
    """
    Convert subject-wise metrics dictionary to a pandas DataFrame.

    Parameters
    ----------
    metrics_subjectwise : dict
        Dictionary mapping subject_code to metric values dict.
    best_metric_cfg : DictConfig
        Configuration specifying which metric to extract (first key used).

    Returns
    -------
    tuple
        (pd.DataFrame, str): DataFrame with subject_code, missing_rate, and
        the selected metric; and the metric name as a string.
    """
    metric_name = list(best_metric_cfg.keys())[0]
    best_value_key = best_metric_cfg[metric_name]["string"]
    dict_tmp = {}
    for i, (subject_code, metrics) in enumerate(metrics_subjectwise.items()):
        value = metrics_subjectwise[subject_code][best_value_key]
        if i == 0:
            dict_tmp = {
                "subject_code": [subject_code],
                "missing_rate": [metrics_subjectwise[subject_code]["missing_rate"]],
                metric_name: [value],
            }
        else:
            dict_tmp["subject_code"].append(subject_code)
            dict_tmp["missing_rate"].append(
                metrics_subjectwise[subject_code]["missing_rate"]
            )
            dict_tmp[metric_name].append(value)

    return pd.DataFrame(dict_tmp), metric_name


# ============================================================================
# FIGURE EXPORT UTILITIES
# ============================================================================


def save_figure_all_formats(
    fig: Figure,
    output_path: str,
    dpi: int = 300,
    formats: Tuple[str, ...] = ("png", "svg", "eps"),
) -> str:
    """
    Save a matplotlib figure in multiple formats.

    NOTE: This is a backward-compatibility wrapper around save_figure().
    Prefer using save_figure() from plot_config.py directly.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str
        Base output path (with or without extension). Extension will be replaced.
    dpi : int
        Resolution for PNG (default 300 for A4 print quality)
    formats : tuple
        Formats to save (default: png, svg, eps)

    Returns
    -------
    str
        Base filename (without extension) for logging
    """
    from src.viz.plot_config import save_figure

    path = Path(output_path)
    base_name = path.stem
    parent = path.parent if path.parent != Path(".") else None

    # Delegate to save_figure from plot_config
    save_figure(fig, base_name, formats=list(formats), output_dir=parent)
    logger.info(f"Saved: {base_name} (via save_figure)")

    return base_name
