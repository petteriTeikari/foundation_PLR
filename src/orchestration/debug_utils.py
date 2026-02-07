from typing import Any

from loguru import logger
from omegaconf import DictConfig, open_dict


def debug_classification_macro(cfg: DictConfig) -> DictConfig:
    """Reduce bootstrap iterations for faster debugging.

    Modifies the configuration to use only 50 bootstrap iterations
    instead of the default (typically 1000).

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary to modify.

    Returns
    -------
    DictConfig
        Modified configuration with reduced bootstrap iterations.
    """
    with open_dict(cfg):
        cfg["CLS_EVALUATION"]["BOOTSTRAP"]["n_iterations"] = 50
        logger.info(
            "Setting number of bootstrap iterations to {} to speed up debugging".format(
                cfg["CLS_EVALUATION"]["BOOTSTRAP"]["n_iterations"]
            )
        )
    return cfg


def pick_one_model(cfg: DictConfig, model_name: str = "SAITS") -> DictConfig:
    """Keep only one model in configuration for testing.

    Reduces the model dictionary to contain only the specified model.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with MODELS dictionary.
    model_name : str, optional
        Name of the model to keep. Default is 'SAITS'.

    Returns
    -------
    DictConfig
        Modified configuration with single model.
    """
    logger.warning("Picking just one model for testing purposes: {}".format(model_name))
    cfg["MODELS"] = {model_name: cfg["MODELS"][model_name]}
    return cfg


def debug_train_only_for_one_epoch(cfg: DictConfig) -> DictConfig:
    """Reduce all epoch counts to 1 for quick debugging.

    Recursively searches configuration for epoch-related keys and
    sets them to 1 for fast iteration during development.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary to modify.

    Returns
    -------
    DictConfig
        Modified configuration with all epoch counts set to 1.

    Notes
    -----
    Modifies keys: 'epochs', 'max_epoch', 'train_epochs'.
    """

    def replace_item(obj: DictConfig, key: str, replace_value: Any) -> DictConfig:
        for k, v in obj.items():
            if isinstance(v, DictConfig):
                obj[k] = replace_item(v, key, replace_value)
        if key in obj:
            old_value = obj[key]
            if isinstance(obj[key], int):
                obj[key] = replace_value
                logger.warning(
                    "Replacing old value '{}={}' with '{}={}'".format(
                        key, old_value, key, obj[key]
                    )
                )
        return obj.copy()

    # Would it be robust enough to just search for the "epoch" substring?
    cfg_modified = replace_item(cfg.copy(), "epochs", replace_value=1)
    cfg_modified = replace_item(cfg_modified.copy(), "max_epoch", replace_value=1)
    cfg_modified = replace_item(cfg_modified.copy(), "train_epochs", replace_value=1)

    return cfg_modified


def fix_tree_learners_for_debug(cfg: DictConfig, model_name: str) -> DictConfig:
    """Reduce tree-based model iterations for debugging.

    Specifically handles MissForest by reducing max_iter to 2.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    model_name : str
        Name of the model to check.

    Returns
    -------
    DictConfig
        Modified configuration if MissForest, otherwise unchanged.
    """
    if "MISSFOREST" in model_name:
        logger.warning("DEBUG | Setting the MissForest max_iter to 2")
        with open_dict(cfg):
            cfg["MODELS"][model_name]["MODEL"]["max_iter"] = 2

    return cfg
