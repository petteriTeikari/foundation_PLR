from loguru import logger
import os

import seaborn as sns
from omegaconf import DictConfig

from src.featurization.feature_utils import (
    get_feature_names,
)
import matplotlib.pyplot as plt

from src.utils import get_artifacts_dir
from src.viz.viz_subplots import feature_subplot
from src.viz.viz_utils import (
    pick_single_feature_with_code_from_df,
    get_font_scaler,
)


# @task(
#     log_prints=True,
#     name="Visualize PLR",
#     description="Visualize the PLR data, and the derived features for publication and for 'graphical unit testing' of the data",
# )
def visualize_features(features: dict, cfg: DictConfig):
    logger.info("Visualizing the PLR features")
    sources = list(features.keys())
    feature_names = get_feature_names(features)
    split_keys = [
        "gt",
        "raw",
    ]  # hard-coded now, "BASELINE" ones do not have the both atm # get_split_keys(features)
    viz_cfg = cfg["VISUALIZATION"]
    colors = ["Blue", "Red"]
    splits = list(features[sources[0]]["data"].keys())

    output_dir = get_artifacts_dir(service_name="figures")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style(style=viz_cfg["SNS"]["style"])
    sns.set_context("paper")
    sns.set_palette(viz_cfg["SNS"]["palette"])

    paths_out = {}
    for split_key in split_keys:
        for color in colors:
            logger.debug(f"VIZ FEATURES | color = {color}, split_key = {split_key}")
            # "train/val_gt" was when you tried to impute the missing values from denoised data (synthetic missingess)
            # -- this should perform better obviously
            # "train/val_raw" was when you tried to impute the missing values from noisy data (real missingess)
            # -- this should perform worse obviously, but if performs okay, easier to use in real life
            paths_out[f"features_{color}_{split_key}"] = viz_features_per_color(
                features,
                sources,
                feature_names,
                splits,
                split_key,
                color=color,
                viz_cfg=viz_cfg,
                output_dir=output_dir,
            )

    return paths_out


def viz_features_per_color(
    features,
    sources,
    feature_names,
    splits,
    split_key,
    color: str,
    viz_cfg,
    output_dir=None,
):
    # Keep only the feature names with the desired color substring
    features_per_color = [f for f in feature_names if color in f]
    n_cols = len(features_per_color) * len(splits)
    n_rows = len(sources)

    # Init the Matplotlib figure layout
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(viz_cfg["col_unit"] * n_cols, viz_cfg["row_unit"] * n_rows),
        dpi=viz_cfg["dpi"],
        tight_layout=True,
    )
    fig.suptitle(
        "PLR Features ({})".format(split_key),
        fontsize=int(get_font_scaler(viz_cfg) * 12),
    )

    for row, model in enumerate(sources):  # Loop through the sources
        for i, feature_name in enumerate(
            features_per_color
        ):  # Loop through the features
            for j, split in enumerate(splits):  # Loop through the splits
                col_plot_idx = i * len(splits) + j
                linear_idx = row * n_cols + col_plot_idx
                logger.debug(
                    f"row = {row}, col_plot_idx = {col_plot_idx}, linear_idx = {linear_idx}"
                )
                try:
                    if "BASELINE" in model:
                        if "GT" in model:
                            split_key_df = "gt"
                        elif "Raw" in model:
                            split_key_df = "raw"
                        else:
                            logger.error(f"Unknown baseline key in model name: {model}")
                            raise ValueError(
                                f"Unknown baseline key in model name: {model}"
                            )
                    else:
                        split_key_df = split_key
                    df_to_plot = pick_single_feature_with_code_from_df(
                        df=features[model]["data"][split][split_key_df],
                        feature_name=feature_name,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not pick the feature {feature_name} from the dataframe (BASELINE?)"
                    )
                    logger.warning(
                        "model = {}, split = {}, split_key = {}".format(
                            model, split, split_key
                        )
                    )
                    logger.warning(e)
                    return

                feature_subplot(
                    fig,
                    r=row,
                    c=col_plot_idx,
                    ax_in=axes[row, col_plot_idx],
                    viz_cfg=viz_cfg,
                    df_to_plot=df_to_plot,
                    model=model,
                    split=split,
                    split_key=split_key,
                    feature_name=feature_name,
                )

    path_output_dir = os.path.join(
        output_dir, f"features_{color}_{split}_{split_key}.png"
    )
    logger.debug(f"Saving the feature visualization to {path_output_dir}")
    fig.savefig(path_output_dir)

    return path_output_dir
