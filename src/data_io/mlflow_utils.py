import argparse
import pickle
from pathlib import Path

import mlflow


def init_mlflow(tracking_uri: str, use_quickndirty_abs_check: bool = True):
    """Initialize MLflow with the specified tracking URI.

    Uses centralized path utilities for portable paths.

    Parameters
    ----------
    tracking_uri : str
        The MLflow tracking URI path (used when use_quickndirty_abs_check is False).
    use_quickndirty_abs_check : bool, optional
        If True, use centralized path utilities to get mlruns directory.
        If False, resolve path relative to current working directory. Default is True.

    Returns
    -------
    None
        Sets MLflow tracking URI as a side effect.

    Raises
    ------
    FileNotFoundError
        If the resolved tracking URI path does not exist.
    """
    from src.utils.paths import get_mlruns_dir

    # Use centralized path utilities instead of hardcoded paths
    if use_quickndirty_abs_check:
        mlruns_dir = get_mlruns_dir()
        abs_path = mlruns_dir
    else:
        abs_path = (Path.cwd().parent.parent / tracking_uri).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"MLflow tracking URI does not exist: {abs_path}")

    mlflow.set_tracking_uri(str(abs_path))
    print(f"Init ML | tracking_uri = {abs_path}")


def init_mlflow_experiment(experiment_name: str):
    """Set the active MLflow experiment by name.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to activate.

    Returns
    -------
    None
        Sets the active experiment as a side effect.
    """
    mlflow.set_experiment(experiment_name)


def log_mlflow_params(args):
    """Log all arguments from a namespace object as MLflow parameters.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace containing parameters to log.

    Returns
    -------
    None
        Logs parameters to MLflow and prints them to stdout.
    """
    print("Input arguments:")
    for key, value in vars(args).items():
        print(f" {key}: {value}")
        mlflow.log_param(key, value)


def mlflow_log_metrics(metrics_dict: dict):
    """Log metrics from a nested dictionary to MLflow.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with split names as keys, each containing a 'scalars' dict
        with metric name-value pairs. None values are skipped.

    Returns
    -------
    None
        Logs metrics to MLflow with format '{split}/{metric_name}'.
    """
    for split in metrics_dict.keys():
        split_out = split.replace("outlier_", "")
        scalars = metrics_dict[split]["scalars"]
        for key, value in scalars.items():
            if value is not None:
                mlflow.log_metric(f"{split_out}/{key}", value)


def mlflow_log_model(checkpoint_path: str):
    """Log a model checkpoint file as an MLflow artifact.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint file to log.

    Returns
    -------
    None
        Logs the checkpoint to MLflow under 'model' artifact path.
    """
    mlflow.log_artifact(checkpoint_path, artifact_path="model")


def mlflow_log_results(artifacts, checkpoint_path: str, args: argparse.Namespace):
    """Serialize and log outlier detection results as an MLflow artifact.

    Parameters
    ----------
    artifacts : dict
        Dictionary containing results to serialize and log.
    checkpoint_path : str
        Path to the model checkpoint (used to determine output directory).
    args : argparse.Namespace
        Arguments containing mlflow_run name for the output filename.

    Returns
    -------
    None
        Saves pickle file and logs to MLflow under 'outlier_detection' path.
    """
    base_dir = Path(checkpoint_path).parent
    model_name = args.mlflow_run
    results_path = base_dir / f"outlierDetection_{model_name}.pickle"
    with open(results_path, "wb") as handle:
        pickle.dump(artifacts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact(str(results_path), artifact_path="outlier_detection")


def mlflow_log_experiment(
    epoch_losses: list, artifacts: dict, checkpoint_path: str, args
):
    """Log complete experiment results including metrics, model, and artifacts.

    Parameters
    ----------
    epoch_losses : list
        List of training losses per epoch.
    artifacts : dict
        Dictionary containing 'results' metrics and 'best_epoch' index.
    checkpoint_path : str
        Path to the saved model checkpoint.
    args : argparse.Namespace
        Arguments containing task_data_config_path and mlflow_run name.

    Returns
    -------
    None
        Logs metrics, model checkpoint, results pickle, and data config to MLflow.
    """
    mlflow_log_metrics(metrics_dict=artifacts["results"])
    mlflow_log_model(checkpoint_path=checkpoint_path)
    mlflow_log_results(artifacts, checkpoint_path=checkpoint_path, args=args)

    # Best train loss
    loss = epoch_losses[artifacts["best_epoch"]]
    mlflow.log_metric("train/loss", loss)

    # copy the data .yaml to MLflow, make the path absolute before
    data_yaml = Path(args.task_data_config_path).resolve()
    if data_yaml.exists():
        mlflow.log_artifact(str(data_yaml), artifact_path="data_provider")
