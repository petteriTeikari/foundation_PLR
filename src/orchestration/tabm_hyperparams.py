import itertools

import numpy as np
from loguru import logger

from src.log_helpers.log_naming_uris_and_dirs import get_eval_metric_name


def get_metric_of_runs(metrics, eval_metric, split="test"):
    metric_list = []
    for _, metric in metrics.items():
        scalars = metric["metrics_stats"][split]["metrics"]["scalars"]
        metric_list.append(scalars[eval_metric.upper()]["mean"])
    return np.array(metric_list)


def pick_the_best_hyperparam_metrics(metrics, hparam_cfg, model_cfgs, cfg):
    eval_metric = get_eval_metric_name("TabM", cfg)
    metric_of_runs = get_metric_of_runs(metrics, eval_metric)
    choice_keys, choices = get_grid_choices(hparam_cfg)
    assert len(choices) == len(metric_of_runs), (
        f"Number of choices {len(choices)} "
        f"does not match number of metrics {len(metric_of_runs)}"
    )

    best_idx = np.argmax(metric_of_runs)
    best_metrics = metrics[best_idx]
    best_choice = {
        "choice": choices[best_idx],
        "choice_keys": choice_keys,
        "model_cfg": model_cfgs[best_idx],
    }
    return best_metrics, best_choice


def get_grid_choices(hparam_cfg):
    grid_search_space = hparam_cfg["SEARCH_SPACE"]["GRID"]
    choice_keys = list(grid_search_space.keys())
    choices = list(itertools.product(*grid_search_space.values()))
    return choice_keys, choices


def create_tabm_grid_experiment(hparam_cfg, cls_model_cfg):
    # e.g. with 50 bootstrap iterations
    # src.classification.bootstrap_evaluation:bootstrap_evaluator:245 - Bootstrap evaluation in 32.43 seconds
    # Bootstrap iterations:   2%|▏ | 1/50 [00:00<00:27,  1.78it/s]-
    # Best epoch: 0, val score = 0.4107, test score: 0.4250, train score: 0.4000
    choice_keys, choices = get_grid_choices(hparam_cfg)
    cfgs = []
    for ii, choice in enumerate(choices):
        cfg_tmp = cls_model_cfg.copy()
        for i, key in enumerate(choice_keys):
            cfg_tmp[key] = choice[i]
        cfgs.append(cfg_tmp)
    logger.info(f"Created {len(cfgs)} hyperparameter configurations")
    return cfgs


def create_tabm_hyperparam_experiment(run_name, hparam_cfg, cls_model_cfg):
    if hparam_cfg["HYPERPARAMS"]["run_grid_hyperparam_search"]:
        if hparam_cfg["HYPERPARAMS"]["run_only_on_ground_truth"]:
            if "pupil-gt__pupil-gt" in run_name:
                if "LIST" in hparam_cfg["SEARCH_SPACE"].keys():
                    logger.error("List not implemented yet")
                    raise NotImplementedError("List not implemented yet")
                elif "GRID" in hparam_cfg["SEARCH_SPACE"].keys():
                    cfgs = create_tabm_grid_experiment(hparam_cfg, cls_model_cfg)
                    return cfgs
                else:
                    logger.error(f'Unknown search space, {hparam_cfg["SEARCH_SPACE"]}')
                    raise ValueError(
                        f'Unknown search space, {hparam_cfg["SEARCH_SPACE"]}'
                    )
            else:
                return [cls_model_cfg]
        else:
            return [cls_model_cfg]

    else:
        return [cls_model_cfg]
