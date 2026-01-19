from omegaconf import DictConfig
from loguru import logger
from torch.utils.data import DataLoader
from typing import Literal

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
    """
    Create torch dataloader for both zero-shot imputation and fine-tuning
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
