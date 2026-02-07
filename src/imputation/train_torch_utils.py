from typing import Literal

from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data_io.torch_data import create_dataset_from_numpy

_TASK_TYPES = Literal["imputation", "outlier_detection"]


def create_torch_dataloader(
    data_dict_df: dict,
    task: str,
    model_cfg: DictConfig,
    split: str,
    cfg: DictConfig,
    model_name: str = None,
):
    """Create a PyTorch DataLoader for a specific data split.

    Creates a TensorDataset from numpy arrays and wraps it in a DataLoader
    with the specified configuration.

    Parameters
    ----------
    data_dict_df : dict
        Data dictionary containing arrays per split.
    task : str
        Task type ('imputation' or 'outlier_detection').
    model_cfg : DictConfig
        Model configuration with TORCH.DATASET and TORCH.DATALOADER settings.
    split : str
        Split name ('train', 'test', 'outlier_train', 'outlier_test').
    cfg : DictConfig
        Full Hydra configuration.
    model_name : str, optional
        Model name for dataset creation. Default is None.

    Returns
    -------
    DataLoader
        PyTorch DataLoader configured for the specified split.

    Raises
    ------
    NotImplementedError
        If dataset_type is 'class' (not yet implemented).
    ValueError
        If dataset_type is unknown.
    """
    # Create the dataset
    if model_cfg["TORCH"]["DATASET"]["dataset_type"] == "numpy":
        dataset = create_dataset_from_numpy(
            data_dict_df=data_dict_df,
            dataset_cfg=model_cfg["TORCH"]["DATASET"],
            model_cfg=model_cfg,
            split=split,
            task=task,
            model_name=model_name,
        )
    elif model_cfg["TORCH"]["DATASET"]["dataset_type"] == "class":
        raise NotImplementedError(
            "Class based dataset creation is not implemented yet, please use numpy"
        )
        # dataset = AnomalyDetectionPLRDataset(
        #     split, model_artifacts, dataset_cfg=model_cfg["TORCH"]["DATASET"]
        # )
    else:
        logger.error(
            "Unknown Torch Dataset creation method = {}".format(
                model_cfg["TORCH"]["DATASET"]["dataset_type"]
            )
        )
        raise ValueError("Unknown Torch Dataset creation method")

    # Create the dataloder from the dataset
    # Compare to moment-research/moment/data/dataloader.py#L98
    dataloader = DataLoader(
        dataset, **model_cfg["TORCH"]["DATALOADER"]
    )  # create your dataloader

    return dataloader


def create_torch_dataloaders(
    task: str,
    model_name: str,
    data_dict_df: dict,
    model_cfg: DictConfig,
    cfg: DictConfig,
    create_outlier_dataloaders: bool = True,
):
    """Create PyTorch DataLoaders for all required data splits.

    Creates train and test dataloaders, with optional outlier-specific
    dataloaders for anomaly detection tasks.

    Parameters
    ----------
    task : str
        Task type ('imputation' or 'outlier_detection').
    model_name : str
        Model name for dataset creation.
    data_dict_df : dict
        Data dictionary containing arrays per split.
    model_cfg : DictConfig
        Model configuration with TORCH settings.
    cfg : DictConfig
        Full Hydra configuration.
    create_outlier_dataloaders : bool, optional
        Whether to create outlier_train and outlier_test dataloaders.
        Default is True.

    Returns
    -------
    dict
        Dictionary mapping split names to DataLoaders. Contains 'train'
        and 'test', plus 'outlier_train' and 'outlier_test' if requested.
    """
    # Create torch dataloader for zero-shot imputation
    logger.info("Creating torch dataloaders")
    train_dataloader = create_torch_dataloader(
        data_dict_df,
        task=task,
        model_cfg=model_cfg,
        split="train",
        cfg=cfg,
        model_name=model_name,
    )
    test_dataloader = create_torch_dataloader(
        data_dict_df,
        task=task,
        model_cfg=model_cfg,
        split="test",
        cfg=cfg,
        model_name=model_name,
    )
    if create_outlier_dataloaders:
        # if model_cfg["MODEL"]["train_on"] != 'pupil_orig_imputed':
        # No need for separate outlier dataloaders if the actual test/train are the same
        # A bit unconventional naming here, we are going to be training with the clean data
        # unsupervised, and MOMENT hopefully learns to reconstruct the denoised signal, and pick
        # up the anomalies from this outlier dataloader
        outlier_train_dataloader = create_torch_dataloader(
            data_dict_df,
            task=task,
            model_cfg=model_cfg,
            split="outlier_train",
            cfg=cfg,
            model_name=model_name,
        )

        outlier_test_dataloader = create_torch_dataloader(
            data_dict_df,
            task=task,
            model_cfg=model_cfg,
            split="outlier_test",
            cfg=cfg,
            model_name=model_name,
        )

        return {
            "train": train_dataloader,
            "test": test_dataloader,
            "outlier_train": outlier_train_dataloader,
            "outlier_test": outlier_test_dataloader,
        }
        # else:
        #     return {"train": train_dataloader, "test": test_dataloader}

    else:
        return {"train": train_dataloader, "test": test_dataloader}
