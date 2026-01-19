import itertools

from omegaconf import DictConfig
from loguru import logger

from src.log_helpers.log_naming_uris_and_dirs import (
    update_outlier_detection_run_name,
    update_imputation_run_name,
)


def clean_param(param):
    fields = param.split("_")
    if len(fields) > 1:
        # e.g. d_ffn -> dFfn
        for i, string in enumerate(fields):
            if i == 0:
                param = string
            else:
                param += string.title()
    return param


def create_hyperparam_name(
    param: str,
    value_from_list,
    i: int = 0,
    j: int = 0,
    n_params: int = 1,
    value_key_delimiter: str = "",
    param_delimiter: str = "_",
):
    try:
        param = clean_param(param)
    except Exception:
        logger.error(f"Error in cleaning the parameter {param}")
        raise ValueError(f"Error in cleaning the parameter {param}")
    param_key = param + value_key_delimiter + str(value_from_list)
    if j + 1 != n_params:
        param_key += param_delimiter
    return param_key


def create_name_from_model_params(model_name: str, param_cfg: DictConfig) -> str:
    cfg_name = f"{model_name}_"
    for j, (param, value) in enumerate(param_cfg.items()):
        string = f"{create_hyperparam_name(param, value, j=j, n_params=len(param_cfg.keys()))}"
        cfg_name += string
    return cfg_name


def define_list_hyperparam_combos(
    cfg_model: DictConfig, param_dict: DictConfig, model: str, task: str, cfg_key: str
) -> dict:
    def check_no_of_values(param_dict):
        # all params must have the same number of values in the LIST method
        no_of_values_per_param = []
        for i, param in enumerate(param_dict.keys()):
            no_of_values_per_param.append(len(param_dict[param]))
            if i > 0:
                assert (
                    no_of_values_per_param[i] == no_of_values_per_param[i - 1]
                ), "All parameters must have the same number of values"
        return no_of_values_per_param

    cfg_params = {}
    no_of_values_per_param = check_no_of_values(param_dict)[0]
    for i in range(
        no_of_values_per_param
    ):  # e.g, 3 values for each param 'len(list) = 3'
        cfg_tmp = cfg_model.copy()
        param_combo_key = ""
        for j, param in enumerate(
            param_dict.keys()
        ):  # how many hyperparams you wanted to vary
            if param in cfg_tmp[cfg_key][model]["MODEL"]:
                value_from_list = param_dict[param][i]
                # create a name for the hyperparameter, this will be used then in the run names (MLflows)
                param_combo_key += f"{create_hyperparam_name(param, value_from_list, i, j, n_params=len(param_dict.keys()))}"
                cfg_tmp[cfg_key][model]["MODEL"][param] = value_from_list
            else:
                logger.error(
                    f"Parameter {param} not found in the model {model} (typo in your search_space?"
                )
                logger.error(
                    f'Possible param keys for assignment = {cfg_tmp[cfg_key][model]["MODEL"].keys()}'
                )
                raise ValueError(
                    f"Parameter {param} not found in the model {model} (typo in your search_space?"
                )
        cfg_params[f"{model}_{param_combo_key}"] = cfg_tmp

    return cfg_params


def define_grid_hyperparam_combos(
    cfg_model: DictConfig, param_dict: DictConfig, model: str, task: str, cfg_key: str
) -> dict:
    choice_keys = list(param_dict.keys())
    choices = list(itertools.product(*param_dict.values()))

    cfgs = {}
    for i, choice in enumerate(choices):
        cfg_tmp = cfg_model.copy()
        for j, key in enumerate(choice_keys):
            logger.debug("Setting {} to {}".format(key, choice[j]))
            cfg_tmp[cfg_key][model]["MODEL"][key] = choice[j]
        if task == "outlier_detection":
            cfg_name = update_outlier_detection_run_name(cfg_tmp)
        elif task == "imputation":
            cfg_name = update_imputation_run_name(cfg_tmp)
        else:
            logger.error("Define some name for the run, task = {}".format(task))
            raise ValueError("Define some name for the run, task = {}".format(task))

        cfgs[cfg_name] = cfg_tmp

    return cfgs
