import itertools
from copy import deepcopy

from loguru import logger
from omegaconf import DictConfig, open_dict


def do_grid_search(hparam_cfg: DictConfig):
    """
    Check if grid search is enabled in the hyperparameter configuration.

    Parameters
    ----------
    hparam_cfg : DictConfig
        Hyperparameter configuration containing search space settings.

    Returns
    -------
    tuple[bool, list | None]
        A tuple of (run_grid, grid_contains) where run_grid indicates if grid
        search should be run, and grid_contains lists the parameters to grid over.
    """
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
    """
    Generate a descriptive name for a weighing choice combination.

    Parameters
    ----------
    choice : tuple
        A tuple of boolean values indicating which weights are enabled.
    params : dict
        Dictionary mapping parameter names to their possible values.

    Returns
    -------
    str
        A string identifier for the weighing choice (e.g., 'w-featW-classW').
    """
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
    """
    Create grid of XGBoost configs with all combinations of weighing options.

    Generates 8 configurations covering all permutations of sample, class,
    and feature weighing (3 binary parameters = 2^3 combinations).

    Parameters
    ----------
    xgboost_cfg : DictConfig
        Base XGBoost configuration to modify.
    key1 : str, optional
        First-level config key for weighing settings, by default "MODEL".
    key2 : str, optional
        Second-level config key for weighing settings, by default "WEIGHING".

    Returns
    -------
    dict[str, DictConfig]
        Dictionary mapping choice names to modified XGBoost configurations.
    """
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
    """
    Generate grid configurations based on the specified grid parameters.

    Parameters
    ----------
    xgboost_cfg : DictConfig
        Base XGBoost configuration to expand into a grid.
    grid_contains : list[str]
        List of parameter groups to include in grid search (e.g., ["weights"]).

    Returns
    -------
    dict[str, DictConfig]
        Dictionary mapping configuration names to XGBoost configs.

    Raises
    ------
    ValueError
        If an unknown grid key is specified in grid_contains.
    """
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
    """
    Define the complete XGBoost grid search space based on configuration.

    Parameters
    ----------
    hparam_cfg : DictConfig
        Hyperparameter configuration with grid search settings.
    xgboost_cfg : DictConfig
        Base XGBoost model configuration.
    cfg : DictConfig
        Overall experiment configuration (unused but kept for interface).

    Returns
    -------
    dict[str, DictConfig]
        Dictionary of XGBoost configurations to evaluate. Contains either
        multiple grid configurations or a single "no_grid" configuration.
    """
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
