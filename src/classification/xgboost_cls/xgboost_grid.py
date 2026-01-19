import itertools
from copy import deepcopy
from loguru import logger
from omegaconf import DictConfig, open_dict


def do_grid_search(hparam_cfg: DictConfig):
    if "GRID" in hparam_cfg["SEARCH_SPACE"]:
        run_grid = hparam_cfg["SEARCH_SPACE"]["GRID"]["run_grid_hyperparam_search"]
        if run_grid:
            grid_contains = hparam_cfg["SEARCH_SPACE"]["GRID"]["grid_contains"]
            return True, grid_contains
        else:
            return False, None
    else:
        return False, None


def get_weight_choice_name(choice, params):
    choice_name = "w"
    for i, key in enumerate(params.keys()):
        if "features" in key:
            str_choice = "-featW" if choice[i] else ""
        elif "classes" in key:
            str_choice = "-classW" if choice[i] else ""
        elif "samples" in key:
            str_choice = "-sampleW" if choice[i] else ""
        choice_name += str_choice
    if choice_name == "choice":
        choice_name = "noW"
    return choice_name


def define_weighing_grid(xgboost_cfg, key1="MODEL", key2="WEIGHING"):
    # Maybe there is some easier way to combine grid with hyperopt?
    params = {
        "weigh_the_samples": [True, False],
        "weigh_the_classes": [True, False],
        "weigh_the_features": [True, False],
    }

    # Get all the 8 possible permutations with 3 parameters each with 2 possible values
    choices = list(itertools.product(*params.values()))

    cfgs = {}
    for choice in choices:
        choice_name = get_weight_choice_name(choice, params)
        cfg_tmp = deepcopy(xgboost_cfg)
        for i, key in enumerate(params.keys()):
            with open_dict(cfg_tmp):
                cfg_tmp[key1][key2][key] = choice[i]
        cfgs[choice_name] = cfg_tmp

    return cfgs


def define_grid_cfgs(xgboost_cfg, grid_contains):
    for key in grid_contains:
        if key == "weights":
            grid_cfgs = define_weighing_grid(xgboost_cfg)
        else:
            # Obviously gets quite tricky if you have some unorthodox combos here, but for now it's fine
            raise ValueError(f"Unknown grid key: {key}")

    return grid_cfgs


def define_xgboost_grid_search_space(
    hparam_cfg: DictConfig, xgboost_cfg: DictConfig, cfg: DictConfig
):
    run_grid, grid_contains = do_grid_search(hparam_cfg)

    if run_grid:
        grid_cfgs = define_grid_cfgs(xgboost_cfg, grid_contains)
        logger.info(
            "Grid search for XGBoost HPO, {} configurations".format(len(grid_cfgs))
        )
    else:
        grid_cfgs = {"no_grid": deepcopy(xgboost_cfg)}
        logger.info("No grid search for XGBoost HPO")

    return grid_cfgs
