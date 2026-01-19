# Some of the classifications runs had 100 iterations instead of the "commonly accepted 1000 runs", so batch-delete
# them all and re-run this configs with the correct value
import os
import shutil

import mlflow
from loguru import logger

from src.ensemble.ensemble_classification import import_metrics_iter
from src.scripts.sigllm_anomaly_detection import get_repo_root

EXPERIMENT_NAME = "PLR_Classification"


def clear_incorrect_cls_runs(
    experiment_name: str,
    correct_iters: int,
    incorrect_iters: int,
    delete_folder: bool = True,
):
    mlruns_dir = os.path.join(get_repo_root(), "mlruns")
    mlflow.set_tracking_uri(mlruns_dir)
    mlruns = mlflow.search_runs(experiment_names=[experiment_name])
    if len(mlruns) == 0:
        logger.error(
            'No runs found from "{}", mlruns in = {}'.format(
                experiment_name, mlruns_dir
            )
        )
        raise ValueError(
            'No runs found from "{}", mlruns in = {}'.format(
                experiment_name, mlruns_dir
            )
        )

    no_erroneous_runs = 0
    for idx, mlrun in mlruns.iterrows():
        exp_id = mlrun["experiment_id"]
        run_id = mlrun["run_id"]
        run_name = mlrun["tags.mlflow.runName"]
        model_name = mlrun["params.model_name"]
        if "ensemble-" not in run_name:
            try:
                metrics_iter = import_metrics_iter(
                    run_id, run_name, model_name, subdir="metrics"
                )
                n = metrics_iter["test"]["preds"]["arrays"]["predictions"][
                    "y_pred_proba"
                ].shape[1]
                if n != correct_iters:
                    logger.info(f"n = {n}: {model_name} / {run_name}")
                    no_erroneous_runs += 1
                    if delete_folder:
                        folder_path = os.path.join(mlruns_dir, exp_id, run_id)
                        if os.path.exists(folder_path):
                            shutil.rmtree(folder_path)
                            if os.path.exists(folder_path):
                                logger.error(f"Problem deleting {folder_path}")
                            else:
                                logger.info(f"Deleted {folder_path}")
                        else:
                            logger.warning(
                                f"error in path, could not delete! ({folder_path})"
                            )
            except Exception:
                logger.error(
                    f"Problem with run_id = {run_id}, run_name = {run_name}, model_name = {model_name}"
                )

    logger.info(f"Removed a total of {no_erroneous_runs} runs")


if __name__ == "__main__":
    clear_incorrect_cls_runs(
        experiment_name=EXPERIMENT_NAME, correct_iters=1000, incorrect_iters=100
    )
