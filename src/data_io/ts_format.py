import os

import pandas as pd
from loguru import logger
import numpy as np

from src.data_io.data_utils import transform_data_for_momentfm
from src.data_io.data_wrangler import convert_df_to_dict
from src.data_io.torch_data import pick_splits_from_data_dict_to_ts
from src.utils import get_data_dir


def concatenate_subjects(X, y):
    """
    Check out a bit the documentation for this, can you provide an array of targets
    https://github.com/aeon-toolkit/aeon/discussions/2328
    https://github.com/aeon-toolkit/aeon-tutorials/blob/main/ECML-2024/Notebooks/part6_anomaly_detection.ipynb
    """
    X = X.flatten()[np.newaxis, np.newaxis, :]  # (n_cases, n_channels, n_timepoints)
    y = y.flatten()
    return X, y


def write_as_numpy_for_vanilla_dataloader(
    X, y, split, data_dir, data_name, dataset_cfg, train_on: str
):
    subdir = os.path.join(data_dir, data_name)
    os.makedirs(subdir, exist_ok=True)

    if dataset_cfg["trim_to_size"] is not None:
        # This takes fixed windows, the Class-based sliding window sampler might be better in the future
        # if MOMENT shows some promise
        X, y, _ = transform_data_for_momentfm(X, y, dataset_cfg, "UniTS")

    df = pd.DataFrame(X)
    df_labels = pd.DataFrame(y)

    if train_on == "pupil_gt":
        suffix = "-gt"
    else:
        suffix = ""
    logger.info(f"Writing {split} data to {subdir}")
    logger.info(f"X shape = {df.shape}")  # X shape = (355, 1981) or (152,1981)
    df.to_csv(os.path.join(subdir, f"{split}{suffix}.csv"), index=False)
    logger.info(
        f"Labels shape = {df_labels.shape}"
    )  # Labels shape = (355, 1981) or (152,1981)
    df_labels.to_csv(os.path.join(subdir, f"{split}{suffix}_label.csv"), index=False)


def write_as_PLR(X, y, split, data_dir, data_name):
    subdir = os.path.join(data_dir, data_name)
    os.makedirs(subdir, exist_ok=True)

    # kick out first sample to have some nice integer division to seq_len
    X = X[:, 1:]
    y = y[:, 1:]

    df = pd.DataFrame(X)
    df_labels = pd.DataFrame(y)

    # Train split shape: (703255, 16), unique subjects = 355 (9.77% outliers)
    # Test split shape: (301112, 16), unique subjects = 152 (9.67% outliers)
    logger.info(f"Writing {split} data to {subdir}")
    logger.info(f"X shape = {df.shape}")  # X shape = (355, 1981) or (152,1981)
    df.to_csv(os.path.join(subdir, f"{split}.csv"), index=False)
    logger.info(
        f"Labels shape = {df_labels.shape}"
    )  # Labels shape = (355, 1981) or (152,1981)
    df_labels.to_csv(os.path.join(subdir, f"{split}_label.csv"), index=False)


def write_as_psm(X, y, split, data_dir, data_name):
    subdir = os.path.join(data_dir, data_name)
    os.makedirs(subdir, exist_ok=True)

    linear_time = np.linspace(1, len(y), len(y))
    dict = {"time": linear_time, "pupil": np.squeeze(X)}
    df: pd.DataFrame = pd.DataFrame(dict)
    df_labels: pd.DataFrame = pd.DataFrame({"time": linear_time, "label": y})

    logger.info(f"Writing {split} data to {subdir}")
    df.to_csv(os.path.join(subdir, f"{split}.csv"), index=False)
    df_labels.to_csv(os.path.join(subdir, f"{split}_label.csv"), index=False)


def export_df_to_ts_format(
    df, cfg, model_cfg, task: str = "outlier_detection", write_as: str = "numpy"
):
    """
    https://github.com/mims-harvard/UniTS/blob/main/Tutorial.md
    https://www.aeon-toolkit.org/en/latest/examples/datasets/data_loading.html

    The dataset should contain newdata_TRAIN.ts and newdata_TEST.ts files.
    """
    data_dict = convert_df_to_dict(data_df=df, cfg=cfg)
    train_on = model_cfg["MODEL"]["train_on"]
    data_splits = pick_splits_from_data_dict_to_ts(
        data_dict_df=data_dict["df"], model_cfg=model_cfg, train_on=train_on
    )

    data_dir = get_data_dir()
    data_name = cfg["DATA"]["filename_DuckDB"].replace(".db", "")
    filenames = {"train": f"{data_name}_TRAIN.ts", "test": f"{data_name}_TEST.ts"}

    for split, data in data_splits.items():
        path_out = os.path.join(data_dir, filenames[split])
        X = data["X"]
        y = data["y"].astype(int)
        y_sum = np.sum(y)
        y_percentage = 100 * (y_sum / len(y))
        try:
            if write_as == "PSM":
                # Custom format, for example used by UniTS
                X, y = concatenate_subjects(X, y)
                write_as_psm(X, y, split, data_dir, data_name)
            elif write_as == "PLR":
                write_as_PLR(X, y, split, data_dir, data_name)
            elif write_as == "numpy":
                write_as_numpy_for_vanilla_dataloader(
                    X,
                    y,
                    split,
                    data_dir,
                    data_name,
                    dataset_cfg=model_cfg["TORCH"]["DATASET"],
                    train_on=train_on,
                )
            elif write_as == ".ts":
                X, y = concatenate_subjects(X, y)
                from aeon.datasets import write_to_tsfile

                logger.info(f"Writing to {path_out}")
                logger.info(
                    f"with shape {X.shape} and outlier percentage = {y_percentage:.2f}%"
                )
                write_to_tsfile(
                    X,
                    path=path_out,
                    y=y,
                    problem_name=data_name,
                    header=None,
                    regression=False,
                )
        except Exception as e:
            logger.error(f"Error writing to {path_out}: {e}")
            logger.error(
                "Please install the aeon library (which had some glitches with uv)"
            )
            raise e
