import pandas as pd
from loguru import logger
import numpy as np
from omegaconf import DictConfig
import polars as pl

from src.data_io.data_utils import get_unique_polars_rows


def pick_single_feature_with_code_from_df(
    df, feature_name, code_col="subject_code", metadata_prefix="metadata_"
):
    """
    Pick a single feature from a DataFrame with a feature name
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


def clean_feature_name_for_title(feature_name):
    """
    Clean the feature name for a title
    """
    color = feature_name.split("_")[0]
    feature_name = " ".join(feature_name.split("_")[1:])
    return f"{color}: {feature_name}"


def get_font_scaler(viz_cfg):
    return viz_cfg["dpi"] / 100


def get_baseline_key(model):
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
    model_name: str = "SAITS",
    split_key: str = "gt",
    split: str = "val",
):
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
    y_cols: tuple = ("mean",),
    error_cols: tuple = ("ci_lower", "ci_upper"),
    stats_on_col="time",
):
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
def create_video_from_figures_on_disk(fig_paths, cfg):
    # moviepy, or what: https://stackoverflow.com/a/62434934/6412152
    logger.warning("Video creation placeholder")


def get_subjectwise_df_metrics(metrics_subjectwise: dict, best_metric_cfg: DictConfig):
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
