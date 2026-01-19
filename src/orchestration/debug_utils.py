from omegaconf import DictConfig, open_dict
from loguru import logger


def debug_classification_macro(cfg):
    with open_dict(cfg):
        cfg["CLS_EVALUATION"]["BOOTSTRAP"]["n_iterations"] = 50
        logger.info(
            "Setting number of bootstrap iterations to {} to speed up debugging".format(
                cfg["CLS_EVALUATION"]["BOOTSTRAP"]["n_iterations"]
            )
        )
    return cfg


def pick_one_model(cfg: DictConfig, model_name: str = "SAITS"):
    logger.warning("Picking just one model for testing purposes: {}".format(model_name))
    cfg["MODELS"] = {model_name: cfg["MODELS"][model_name]}
    return cfg


def debug_train_only_for_one_epoch(cfg):
    def replace_item(obj, key, replace_value):
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


def fix_tree_learners_for_debug(cfg, model_name):
    if "MISSFOREST" in model_name:
        logger.warning("DEBUG | Setting the MissForest max_iter to 2")
        with open_dict(cfg):
            cfg["MODELS"][model_name]["MODEL"]["max_iter"] = 2

    return cfg
