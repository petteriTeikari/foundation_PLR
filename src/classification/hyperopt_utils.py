import numpy as np
from omegaconf import DictConfig
from loguru import logger
from hyperopt import hp


def parse_hyperopt_search_space(
    hyperopt_cfg: DictConfig, return_only_static: bool = False
):
    def parse_choice(param_name, param_dict):
        return hp.choice(
            param_name,
            np.arange(param_dict["min"], param_dict["max"], param_dict["step"]),
        )

    def parse_uniform(param_name, param_dict):
        return hp.uniform(param_name, param_dict["min"], param_dict["max"])

    def parse_value(param_name, param_dict):
        return param_dict["value"]

    params = {}
    logger.info("Parsing the hyperopt search space for XGBoost")
    for param_name, param_dict in hyperopt_cfg.items():
        if return_only_static:
            # When not optimizing, training the model after hyperopt
            if param_dict["hp_func"] is None:
                params[param_name] = parse_value(param_name, param_dict)
        else:
            # When optimizing
            if param_dict["hp_func"] == "choice":
                params[param_name] = parse_choice(param_name, param_dict)
            elif param_dict["hp_func"] == "uniform":
                params[param_name] = parse_uniform(param_name, param_dict)
            elif param_dict["hp_func"] is None:
                params[param_name] = parse_value(param_name, param_dict)
            else:
                logger.error(f"Unknown hyperopt function: {param_dict['hp_func']}")
                raise ValueError(f"Unknown hyperopt function: {param_dict['hp_func']}")
    logger.debug("Hyperopt search space parsed successfully, {}".format(params))
    return params
