import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_utils import get_artifact
from src.ensemble.ensemble_utils import get_results_from_mlflow_for_ensembling
from src.log_helpers.local_artifacts import load_results_dict


def import_cls_artifacts(mlflow_runs: pd.DataFrame, cfg: DictConfig):
    """Import classification artifacts from MLflow runs.

    Downloads bootstrap metrics and baseline model results for each
    classification run.

    Parameters
    ----------
    mlflow_runs : pd.DataFrame
        DataFrame containing MLflow run metadata with columns
        'run_id', 'tags.mlflow.runName', 'params.model_name'.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    dict
        Dictionary mapping run names to their metrics:
        - 'bootstrap': Bootstrap evaluation results
        - 'baseline': Baseline model results (if available)
    """
    metrics = {}

    for idx, mlflow_run in tqdm(
        mlflow_runs.iterrows(),
        desc="Importing classification artifacts",
        total=len(mlflow_runs),
    ):
        run_id = mlflow_run["run_id"]
        run_name = mlflow_run["tags.mlflow.runName"]
        model_name = mlflow_run["params.model_name"]
        metrics[run_name] = {}

        # bootstrap resaults
        artifact_path = get_artifact(run_id, run_name, model_name, subdir="metrics")
        metrics[run_name]["bootstrap"] = load_results_dict(artifact_path)

        # baseline results
        artifact_path = get_artifact(
            run_id, run_name, model_name, subdir="baseline_model"
        )
        if artifact_path is not None:
            metrics[run_name]["baseline"] = load_results_dict(artifact_path)

    return metrics


def get_classification_summary_data(cfg: DictConfig, experiment_name: str):
    """Get summary data for classification experiment.

    Retrieves all MLflow runs from the classification experiment and
    imports their artifacts (bootstrap metrics, baseline results).

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary.
    experiment_name : str
        Name of the classification experiment in MLflow.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data_df': Placeholder DataFrame
        - 'mlflow_runs': DataFrame of all classification runs
        - 'artifacts_dict_summary': Metrics dictionary per run
    """
    # this is now same as importing data for the classification ensemble
    mlflow_runs_dict = get_results_from_mlflow_for_ensembling(
        experiment_name=experiment_name, cfg=cfg, task="classification"
    )

    mlflow_runs = pd.DataFrame()
    for source_name, mlflow_run in mlflow_runs_dict.items():
        mlflow_runs = pd.concat([mlflow_runs, mlflow_run], axis=0)

    metrics = import_cls_artifacts(mlflow_runs, cfg)

    data = {
        "data_df": pd.DataFrame({"placeholder": [0]}),
        "mlflow_runs": mlflow_runs,
        "artifacts_dict_summary": metrics,
    }

    return data
