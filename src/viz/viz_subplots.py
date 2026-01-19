from loguru import logger

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import polars as pl
import seaborn as sns

from src.viz.viz_styling_utils import style_timeseries_ax, style_distribution_plot
from src.viz.viz_utils import (
    clean_feature_name_for_title,
    get_font_scaler,
    compute_stats_for_shaded_lineseries,
)


def feature_subplot(
    fig,
    r,
    c,
    ax_in,
    viz_cfg,
    df_to_plot,
    model,
    split,
    split_key,
    feature_name,
    hue="metadata_class_label",
    y_stat="value",
):
    """
    Plot a single feature on a subplot
    """
    df = df_to_plot.to_pandas()  # Convert to pandas for seaborn
    font_scaler = get_font_scaler(viz_cfg)

    if viz_cfg["FEATURES"]["type"] == "violin":
        sns.violinplot(
            data=df,
            split=True,
            inner="points",
            hue=hue,
            y=f"{feature_name}_{y_stat}",  # TODO! there is _std as well so for each patient there are 2 points
            ax=ax_in,
        )
        # https://seaborn.pydata.org/tutorial/aesthetics.html#removing-axes-spines
        try:
            sns.despine(ax=ax_in, offset=10, trim=True)
        except Exception as e:
            logger.error(f"Error in using Seaborn despine: {e}")

    else:
        logger.error("Unknown plot type = {}".format(viz_cfg["FEATURES"]["type"]))
        raise NotImplementedError(
            "Unknown plot type = {}".format(viz_cfg["FEATURES"]["type"])
        )

    # Title
    feature_row = "{} | {}".format(clean_feature_name_for_title(feature_name), split)
    model_name_row = f"{model}"
    if r == 0:
        # Feature name on the first row
        ax_in.set_title(
            feature_row + "\n" + model_name_row,
            y=1.0,
            loc="left",
            fontsize=int(9 * font_scaler),
        )
    else:
        ax_in.set_title(
            model_name_row, y=0.95, loc="left", fontsize=int(9 * font_scaler)
        )

    # Style
    ax_in.set_xlabel("")
    ax_in.set_ylabel("")
    ax_in.set_xticks([])  # for major ticks
    ax_in.set_xticks([], minor=True)  # for minor ticks
    ax_in.legend(loc="best", fontsize=int(7 * font_scaler), framealpha=0.0)


def viz_timeseries_subplot(
    fig: plt.Figure,
    r: int,
    c: int,
    linear_idx: int,  # Use for coloring each model differently
    ax_in: plt.Axes,
    viz_cfg: DictConfig,
    cfg: DictConfig,
    df_to_plot: pl.DataFrame,
    y_col: str = "mean",
    y_gt: str = "gt",
    title_str: str = None,
    hue: str = None,
    split: str = None,
    x_col: str = "time",
    plot_gt_with_imputations: bool = True,
    use_class_labels: bool = False,
    original_data: bool = False,
    unique_subjects: list = None,
    metrics_dict: dict = None,
    y_lims: tuple = (-80, 20),
):
    font_scaler = get_font_scaler(viz_cfg)

    if use_class_labels:
        logger.info("Visualizing time series with the class labels")
        no_samples_in = df_to_plot.shape[0]
        df_to_plot = df_to_plot.filter(
            ~pl.all_horizontal(pl.col("class_label").is_null())
        )
        # hue = "class_label"
        no_samples_out = df_to_plot.shape[0]
        logger.info("Visualizing a total of {} samples".format(no_samples_out))
        if no_samples_out < no_samples_in:
            logger.info(
                f"Filtered out {no_samples_in - no_samples_out} samples with missing class labels"
            )
    else:
        # When not using class labels, you can compare how train/val splits look like?
        # hue = None
        # no_samples_out = df_to_plot.shape[0]
        logger.debug("Groping all the labels together")

    if plot_gt_with_imputations:
        df_stats = compute_stats_for_shaded_lineseries(
            df_to_plot, y_cols=(y_col, y_gt), cfg=cfg
        )
    else:
        df_stats = compute_stats_for_shaded_lineseries(
            df_to_plot, y_cols=(y_col,), cfg=cfg
        )

    for y_col in df_stats.keys():
        plt.sca(ax_in)
        gt_overlay = False
        if not original_data:
            if y_col == "gt":
                color = "b"
                gt_overlay = True
            else:
                color = "k"
        else:
            color = "b"  # get from cfg with linear_idx

        plt.plot(
            df_stats[y_col]["t"], df_stats[y_col]["mean"], label=y_col, color=color
        )

        if not gt_overlay:
            # When plotting the ground truth, do not plot the shading, so that the imputation is visible better
            ax_in.fill_between(
                df_stats[y_col]["t"],
                df_stats[y_col]["mean"] - df_stats[y_col]["std"],
                df_stats[y_col]["mean"] + df_stats[y_col]["std"],
                alpha=viz_cfg["TIMSERIES"]["shading_alpha"],
                color=color,
            )

    # TODO! Plot when the light was on

    style_timeseries_ax(
        ax_in,
        title_str,
        y_lims,
        legend_on=not plot_gt_with_imputations,
        font_scaler=font_scaler,
    )


def viz_input_subplot(
    ax, x, y, col_name, title_prefix, viz_cfg, y_lims: tuple = (-80, 20)
):
    ax.plot(x, y, label=col_name)
    title_str = f"{title_prefix} | {col_name}"
    # TODO! Add light timing
    # TODO! Annotate the missing value
    style_timeseries_ax(
        ax, title_str, y_lims, legend_on=False, font_scaler=get_font_scaler(viz_cfg)
    )


def viz_individual_metric_distribution_subplot(
    ax_in, df_to_plot, metric_name, title_str, split, split_key, viz_cfg
):
    try:
        sns.histplot(df_to_plot, x=metric_name, bins=20, ax=ax_in)
        style_distribution_plot(ax_in, title_str, viz_cfg)
    except Exception as e:
        logger.error(f"Error in plotting the distribution: {e}")
        logger.error(f"Dataframe to plot: {df_to_plot}")
        logger.error("You have an empty subplot now")
