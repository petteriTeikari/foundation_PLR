from loguru import logger
import os

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from omegaconf import DictConfig
from src.data_io.data_utils import get_unique_polars_rows
from src.utils import get_artifacts_dir
from src.viz.viz_subplots import (
    viz_timeseries_subplot,
    viz_individual_metric_distribution_subplot,
)
from src.viz.viz_utils import (
    get_font_scaler,
    filter_imputation_df_for_orig,
    get_subjectwise_df_metrics,
)


# @task(
#     log_prints=True,
#     name="Imputation Visualization",
#     description="Imputation Visualization",
# )
def visualize_imputations(imputation_artifacts: dict, cfg: DictConfig):
    logger.info("Visualizing the imputed data")
    models = list(imputation_artifacts["metrics"].keys())
    fig_paths = {}

    for split in imputation_artifacts["metrics"][models[0]].keys():
        path_output_dir = visualize_imputation_quality(
            metrics=imputation_artifacts["metrics"],
            df=imputation_artifacts["df"],  # TODO! This is not yet in the artifacts
            cfg=cfg,
            viz_cfg=cfg["VISUALIZATION"],
            use_class_labels=False,
            split=split,
            y_lims=(-80, 20),
        )

        if path_output_dir is not None:
            fig_paths[f"global_{split}"] = path_output_dir
        else:
            logger.warning("Could not save the feature visualization to MLflow")

    return fig_paths


def visualize_imputation_quality(
    metrics: dict,
    df: pl.DataFrame,
    cfg: DictConfig,
    viz_cfg: DictConfig,
    use_class_labels: bool = True,
    plot_gt_with_imputations: bool = True,
    split: str = "val",
    split_key="gt",
    compare_splits: bool = False,
    y_lims: tuple = (-80, 20),
):
    # These are the actual PLR recorded ('raw') and the manually supervised denoising
    orig_keys = {"Raw": "pupil_raw", "Denoised Grouth Truth": "gt"}
    n_cols = 2
    n_rows = 1 + np.ceil(len(metrics.keys()) / n_cols).astype(int)
    model_names = list(metrics.keys())
    split_keys = get_unique_polars_rows(df, "split_key")["split_key"].to_numpy()

    output_dir = get_artifacts_dir(service_name="figures")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style(style=viz_cfg["SNS"]["style"])
    sns.set_context("paper")
    sns.set_palette(viz_cfg["SNS"]["palette"])

    # Init the Matplotlib figure layout
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(viz_cfg["col_unit"] * n_cols, viz_cfg["row_unit"] * n_rows),
        dpi=viz_cfg["dpi"],
        tight_layout=True,
    )
    fig.suptitle(
        "imputation quality",
        fontsize=int(get_font_scaler(viz_cfg) * 12),
    )

    # The first row is for the original data
    r = 0
    for c, orig_key in enumerate(orig_keys.keys()):
        # Original data is the same for each imputation model
        # so just pick the first one
        df_to_plot = filter_imputation_df_for_orig(
            df, model_name=model_names[0], split_key=split_keys[0], split=split, cfg=cfg
        )
        unique_subjects = list(
            get_unique_polars_rows(df_to_plot, "subject_code")[
                "subject_code"
            ].to_numpy()
        )

        viz_timeseries_subplot(
            fig=fig,
            r=r,
            c=c,
            linear_idx=None,
            ax_in=axes[r, c],
            viz_cfg=viz_cfg,
            cfg=cfg,
            df_to_plot=df_to_plot,
            y_col=orig_keys[orig_key],
            y_gt="gt",
            title_str=orig_key,
            split=split,
            use_class_labels=use_class_labels,
            plot_gt_with_imputations=plot_gt_with_imputations,
            original_data=True,
            unique_subjects=unique_subjects,
            y_lims=y_lims,
        )

    # The rest of the rows are for the imputed data
    r_offset = 1
    best_metric_name = "MAE"
    for j, model_name in enumerate(model_names):
        best_value = metrics[model_name][split][split_key]["global"][
            best_metric_name.lower()
        ]
        r = (j // n_cols) + r_offset
        c = j - (r * n_cols)
        metrics_dict = metrics[model_name][split][split_key]["global"]

        df_to_plot = filter_imputation_df_for_orig(
            df, model_name=model_name, split_key=split_keys[0], split=split, cfg=cfg
        )
        unique_subjects = list(
            get_unique_polars_rows(df_to_plot, "subject_code")[
                "subject_code"
            ].to_numpy()
        )

        viz_timeseries_subplot(
            fig=fig,
            r=r,
            c=c,
            linear_idx=j,
            ax_in=axes[r, c],
            cfg=cfg,
            viz_cfg=viz_cfg,
            df_to_plot=df_to_plot,
            y_col="mean",
            y_gt="gt",
            title_str=f"{model_name} | {best_metric_name} = {best_value:.2f}",
            split=split,
            use_class_labels=use_class_labels,
            plot_gt_with_imputations=plot_gt_with_imputations,
            original_data=False,
            unique_subjects=unique_subjects,
            metrics_dict=metrics_dict,
            y_lims=y_lims,
        )
        # TODO! You are visualizing the whole dataset (both inliers and outliers), and might not be so obvious
        #  how the imputation quality differs, think of a way to visualize only the outliers

    # Plot the metric residuals subjectwise
    # TODO! add some switch so that you can use this both after single model,
    #  and in some summarization viz after all the models
    # At the moment working just for the single model
    c = 1
    ax_in = axes[r, c]
    metrics_subjectwise = metrics[model_name][split][split_key]["subjectwise"]
    df_subjectwise, metric_name_best = get_subjectwise_df_metrics(
        metrics_subjectwise, best_metric_cfg=cfg["IMPUTATION_METRICS"]["best_metric"]
    )
    # TODO! df_to_plot contains "mising_rate", you could plot MAE as a function of that?
    viz_individual_metric_distribution_subplot(
        ax_in,
        df_subjectwise,
        metric_name_best,
        f"{metric_name_best}: Subjectwise Distribution",
        split,
        split_key,
        viz_cfg,
    )

    path_output_dir = os.path.join(output_dir, f"imputations_{split}_{split_key}.png")
    logger.info(f"Saving the feature visualization to {path_output_dir}")
    fig.savefig(path_output_dir)

    return path_output_dir
