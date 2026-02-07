import importlib
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig

# from src.anomaly_detection.units.units_train import units_train_script
from src.data_io.ts_format import export_df_to_ts_format


def units_model_wrapper(
    configs: DictConfig, path: str, device, model_cfg: DictConfig, cfg: DictConfig
):
    """
    Initialize and configure UniTS model for outlier detection.

    Parameters
    ----------
    configs : DictConfig
        Model configuration with model name and task config.
    path : str
        Path for logging and checkpoints.
    device : str
        Device to load model onto.
    model_cfg : DictConfig
        Model-specific configuration.
    cfg : DictConfig
        Full Hydra configuration.
    """
    # Define the model
    model = build_model(
        model_name=configs.model,
        device=device,
        task_data_config_list=[configs.task_data_config_list],
    )

    # Load pretrained weights
    model = get_units_outlier_model(path, "", model)

    # Model param check
    units_model_param_check(model, path)


def build_model(model_name: str, device, task_data_config_list: list):
    """
    Build UniTS model from module.

    Parameters
    ----------
    model_name : str
        Name of the model module to import.
    device : str
        Device to load model onto.
    task_data_config_list : list
        List of task data configurations.

    Returns
    -------
    torch.nn.Module
        Initialized UniTS model.

    References
    ----------
    https://github.com/mims-harvard/UniTS/blob/0e0281482864017cac8832b2651906ff5375a34e/exp/exp_pretrain.py#L83
    """
    # https://github.com/mims-harvard/UniTS/blob/0e0281482864017cac8832b2651906ff5375a34e/exp/exp_pretrain.py#L83
    module = importlib.import_module("models." + model_name)
    model = module.Model(task_data_config_list, pretrain=True).to(device)
    return model.to(device)


def units_model_param_check(model, path: str):
    """
    Log UniTS model parameter counts.

    Separates prompt-related parameters from core model parameters
    for accurate parameter counting.

    Parameters
    ----------
    model : torch.nn.Module
        UniTS model to analyze.
    path : str
        Path for logging context.
    """
    # Model param check
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Parameters number for all {} M".format(pytorch_total_params / 1e6),
        folder=path,
    )
    model_param = []
    for name, param in model.named_parameters():
        if (
            ("prompts" in name and "prompt2forecat" not in name)
            or "prompt_token" in name
            or "mask_prompt" in name
            or "cls_prompt" in name
            or "mask_token" in name
            or "cls_token" in name
            or "category_token" in name
        ):
            print("skip this:", name)
        else:
            model_param.append(param.numel())
    model_total_params = sum(model_param)
    logger.info(
        "Parameters number for UniTS {} M".format(model_total_params / 1e6),
        folder=path,
    )


def get_units_outlier_model(path: str, pretrained_weight: str, model):
    """
    Load pretrained weights into UniTS model.

    Parameters
    ----------
    path : str
        Base path for checkpoint files.
    pretrained_weight : str
        Path to pretrained weights, 'auto' for default, or empty to skip.
    model : torch.nn.Module
        UniTS model to load weights into.

    Returns
    -------
    torch.nn.Module
        Model with loaded weights.

    References
    ----------
    https://github.com/mims-harvard/UniTS/blob/0e0281482864017cac8832b2651906ff5375a34e/exp/exp_sup.py#L275C9-L294C41
    """
    if pretrained_weight is not None:
        if pretrained_weight == "auto":
            pretrain_weight_path = str(Path(path) / "pretrain_checkpoint.pth")
        else:
            pretrain_weight_path = pretrained_weight
        logger.info("loading pretrained model:", pretrain_weight_path, folder=path)
        if "pretrain_checkpoint.pth" in pretrain_weight_path:
            state_dict = torch.load(pretrain_weight_path, map_location="cpu")["student"]
            ckpt = {}
            for k, v in state_dict.items():
                if "cls_prompts" not in k:
                    ckpt[k] = v
        else:
            ckpt = torch.load(pretrain_weight_path, map_location="cpu")
        _ = model.load_state_dict(ckpt, strict=False)

    return model


def units_outlier_wrapper(
    df: pl.DataFrame,
    cfg: DictConfig,
    model_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    task="outlier_detection",
    model_name: str = "UniTS",
    just_export_data_to_ts_format: bool = True,
):
    """
    Run UniTS-based outlier detection on PLR data.

    Currently primarily exports data to .ts format for external processing.
    Full integration with training is in progress.

    Parameters
    ----------
    df : pl.DataFrame
        Input PLR data.
    cfg : DictConfig
        Full Hydra configuration.
    model_cfg : DictConfig
        UniTS model configuration.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow run name.
    task : str, optional
        Task name. Default is 'outlier_detection'.
    model_name : str, optional
        Model name. Default is 'UniTS'.
    just_export_data_to_ts_format : bool, optional
        If True, only export data to .ts format. Default is True.

    Returns
    -------
    tuple
        A tuple containing (None, None) if just exporting,
        or (artifacts, model) when training is implemented.

    References
    ----------
    https://github.com/mims-harvard/UniTS
    https://github.com/mims-harvard/UniTS/blob/main/scripts/few_shot_anomaly_detection/UniTS_finetune_few_shot_anomaly_detection.sh
    """

    # args are contained in model_cfg['PARAMS'], we rename this to configs
    configs = model_cfg["PARAMS"]
    device = cfg["DEVICE"]["device"]

    if just_export_data_to_ts_format:
        # Simpler to test the model by exporting our data in the .ts format
        # and run the tutorial without the integration work to our codebase yet
        # This is the first step to integrate the UniTS model
        export_df_to_ts_format(df, cfg, model_cfg)
        return None, None

    else:
        path = ""
        _ = units_model_wrapper(
            configs=configs, path=path, device=device, model_cfg=model_cfg, cfg=cfg
        )

        # # Training
        # units_train_script(
        #     configs=configs,
        #     device=device,
        #     model=model,
        #     model_cfg=model_cfg,
        #     cfg=cfg,
        #     path=path,
        # )
