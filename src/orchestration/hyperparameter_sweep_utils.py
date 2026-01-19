from omegaconf import DictConfig
from loguru import logger
from omegaconf import open_dict

from src.log_helpers.log_naming_uris_and_dirs import update_outlier_detection_run_name
from src.orchestration.hyperparamer_list_utils import (
    define_list_hyperparam_combos,
    create_name_from_model_params,
    define_grid_hyperparam_combos,
)


def flatten_the_nested_dicts(cfgs, delimiter="_"):
    cfgs_flat = cfgs[list(cfgs.keys())[0]].copy()
    logger.debug(
        "HYPERPARAMETER SEARCH | {} model architectures".format(len(cfgs.keys()))
    )
    logger.info(
        "HYPERPARAMETER SEARCH | {} | {} hyperparameter sets".format(
            list(cfgs.keys())[0], len(cfgs_flat.keys())
        )
    )

    return cfgs_flat


def drop_other_models(cfg_model, model, task: str):
    """
    Drop all models from the config except the one specified
    """
    if task == "outlier_detection":
        model_cfg_key = "OUTLIER_MODELS"
    elif task == "imputation":
        model_cfg_key = "MODELS"
    elif task == "classification":
        model_cfg_key = "CLS_MODELS"
    else:
        logger.error(f"Task {task} not recognized")
        raise ValueError(f"Task {task} not recognized")

    cfg_out = cfg_model.copy()
    for model_name in cfg_model[model_cfg_key]:
        if model_name != model:
            logger.debug(f"dropping model {model_name} from the config")
            with open_dict(cfg_out):
                del cfg_out[model_cfg_key][model_name]

    # as you copy the config for each hyperparameter combo, only
    # one model per cfg is allowed
    assert (
        len(cfg_out[model_cfg_key]) == 1
    ), "Only one model per cfg is allowed, you had {}".format(
        list(cfg_out[model_cfg_key].keys())
    )

    return cfg_out


def pick_cfg_key(cfg: DictConfig, task: str):
    if task == "outlier_detection":
        cfg_key = "OUTLIER_MODELS"
    elif task == "imputation":
        cfg_key = "MODELS"
    elif task == "classification":
        cfg_key = "CLS_MODELS"
    else:
        raise ValueError(f"Task {task} not recognized")
    return cfg_key


def define_hyperparameter_search(cfg: DictConfig, task: str, cfg_key: str) -> dict:
    cfgs = {}
    no_models = len(cfg[cfg_key].keys())  # dict containing the Hydra DictConfigs
    logger.debug("HYPERPARAMETER SEARCH | {} model architectures".format(no_models))
    logger.debug(list(cfg[cfg_key].keys()))
    for model in cfg[cfg_key]:  # e.g. SAITS
        # Get the preferred search method for this particular model, "LIST" or "GRID",
        # hyperopt/optuna Bayesian optimization done "inside the cfg"
        cfgs[model] = cfg.copy()
        cfgs[model] = drop_other_models(cfg_model=cfgs[model], model=model, task=task)
        if "HYPERPARAMS" in cfg[cfg_key][model]:
            method = cfg[cfg_key][model]["HYPERPARAMS"]["method"]  # e.g. LIST
            if method in cfg[cfg_key][model]["SEARCH_SPACE"]:
                logger.info(
                    f"HYPERPARAMETER SEARCH | method {method} found for model {model}"
                )
                if method == "LIST":
                    cfgs[model] = define_list_hyperparam_combos(
                        cfg_model=cfgs[model],
                        param_dict=cfg[cfg_key][model]["SEARCH_SPACE"][method],
                        model=model,
                        task=task,
                        cfg_key=cfg_key,
                    )
                    # Harmonize maybe a bit this, as there is the extra "model_name" nesting
                    # cfgs = flatten_the_nested_dicts(cfgs)

                elif method == "GRID":
                    cfgs[model] = define_grid_hyperparam_combos(
                        cfg_model=cfgs[model],
                        param_dict=cfg[cfg_key][model]["SEARCH_SPACE"][method],
                        model=model,
                        task=task,
                        cfg_key=cfg_key,
                    )
                else:
                    logger.error(
                        'Method not recognized, must be "LIST" or "GRID", not {}'.format(
                            method
                        )
                    )
                    raise ValueError(
                        'Method not recognized, must be "LIST" or "GRID", not {}'.format(
                            method
                        )
                    )
            else:
                logger.error(
                    f"No params defined for search method {method} not for model {model}"
                )
                raise ValueError(
                    f"No params defined for search method {method} not for model {model}"
                )
        else:
            logger.info(
                f"No HYPERPARAMS key found for model {model}, training just with the default hyperparameters"
            )
            method = None

    if method == "LIST":
        cfgs_tmp = cfgs.copy()
        cfgs = {}
        # TODO! Add some checks if you don't have any hyperparams, you need to add extra key then, see TimesNet below
        for model in cfgs_tmp:
            # if model == 'TimesNet':  # quick fix
            #     logger.warning('Manual fix for TimesNet')
            #     cfgs['TimesNet'] = cfg
            for cfg_name, cfg in cfgs_tmp[model].items():
                cfgs[cfg_name] = cfg

    return cfgs


def define_hyperparam_group(cfg: DictConfig, task: str) -> dict:
    cfg_key = pick_cfg_key(cfg=cfg, task=task)
    if cfg["EXPERIMENT"]["hyperparam_search"]:
        try:
            cfgs = define_hyperparameter_search(cfg, task=task, cfg_key=cfg_key)
            logger.info("HYPERPARAMETER SEARCH |total of {} configs".format(len(cfgs)))
        except Exception as e:
            logger.error(f"Error in defining the hyperparameter search experiment: {e}")
            raise e
    else:
        logger.info(
            "Skipping hyperparameter search, "
            "running just one config (single model+single set of hyperparameters)"
        )
        model_name = list(cfg[cfg_key].keys())
        if len(model_name) != 1:
            logger.error(
                "You have multiple models defined in your config, but hyperparameter search is disabled"
            )
            logger.error("Model names: {}".format(model_name))
            raise ValueError(
                "You have multiple models defined in your config, but hyperparameter search is disabled"
            )
        else:
            model_name = model_name[0]
            if task == "outlier_detection":
                cfg_name = update_outlier_detection_run_name(cfg)

            elif task == "imputation":
                cfg_name = create_name_from_model_params(
                    model_name=model_name, param_cfg=cfg[cfg_key][model_name]["MODEL"]
                )
            elif task == "classification":
                logger.error("Implement classification naming")
                raise NotImplementedError("Implement classification naming")
            else:
                logger.error(f"Task {task} not recognized")
                raise ValueError(f"Task {task} not recognized")
            logger.info(f"SINGLE RUN | Model name: {model_name}, cfg_name: {cfg_name}")

        cfgs = {cfg_name: cfg}

    # Flatten the nested dicts
    cfgs_out = {}
    for cfg_name, cfg in cfgs.items():
        # e.g. LOF, PROPHET, TimesNet
        nested_dict_first_key = list(cfg.keys())[0]
        if cfg_name in nested_dict_first_key:
            # If for example one of your models had hyperparams combinations
            # you would have all those nested inside the dictionary, and the first key
            # would contain the model name
            for hyperparam_key in cfg.keys():
                # e.g. LOF-1, LOF-2, LOF-3
                cfgs_out[hyperparam_key] = cfg[hyperparam_key]
        else:
            cfgs_out[cfg_name] = cfg

    return cfgs_out
