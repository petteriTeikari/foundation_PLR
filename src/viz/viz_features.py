import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from omegaconf import DictConfig

from src.featurization.feature_utils import (
    get_feature_names,
)
from src.utils import get_artifacts_dir
from src.viz.plot_config import save_figure
from src.viz.viz_subplots import feature_subplot
from src.viz.viz_utils import (
    get_font_scaler,
    pick_single_feature_with_code_from_df,
)


# @task(
#     log_prints=True,
#     name="Visualize PLR",
#     description="Visualize the PLR data, and the derived features for publication and for 'graphical unit testing' of the data",
# )
def visualize_features(features: dict, cfg: DictConfig):
    """Generate feature-distribution figures for all color/split combinations.

    Iterate over split keys (``"gt"``, ``"raw"``) and LED colors
    (``"Blue"``, ``"Red"``), producing one figure per combination via
    :func:`viz_features_per_color`.

    Parameters
    ----------
    features : dict
        Nested feature dict keyed by source model, then ``"data"`` ->
        split -> split_key -> DataFrame.
    cfg : DictConfig
        Full Hydra configuration (``VISUALIZATION`` sub-key is used).

    Returns
    -------
    dict[str, str]
        Mapping of ``"features_{color}_{split_key}"`` to saved figure
        paths.
    """
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
    """Plot feature distributions for a single LED color and split key.

    Create a grid figure with one row per source model and columns for
    each (feature x split) combination, then save to disk.

    Parameters
    ----------
    features : dict
        Nested feature dict keyed by source model.
    sources : list[str]
        Model/source names (determines row count).
    feature_names : list[str]
        All available feature names; filtered by *color* substring.
    splits : list[str]
        Data splits (e.g. ``["train", "val"]``).
    split_key : str
        Split key label (``"gt"`` or ``"raw"``).
    color : str
        LED color substring to filter features (``"Blue"`` or ``"Red"``).
    viz_cfg : DictConfig
        Visualization config with layout and style settings.
    output_dir : str or None, optional
        Directory for the saved figure. Default is ``None`` (auto-resolved).

    Returns
    -------
    str or None
        Path to the saved figure, or ``None`` if a feature could not be
        extracted.
    """
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

    output_dir_path = Path(output_dir) if output_dir else None
    filename = f"features_{color}_{split}_{split_key}"
    saved_path = save_figure(fig, filename, output_dir=output_dir_path)
    logger.debug(f"Saved the feature visualization to {saved_path}")

    return str(saved_path)
