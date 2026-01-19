from loguru import logger
import mlflow
from omegaconf import DictConfig


def get_run_ids_from_infos(mlflow_infos):
    run_ids = {}
    for name in mlflow_infos.keys():
        run_ids[name] = mlflow_infos[name]["run_info"]["run_id"]
    return run_ids


# @task(
#     log_prints=True,
#     name="Export visualizations as artifacts",
#     description="Export artifacts to MLflow (and Prefect?)",
# )
def export_viz_as_artifacts(
    fig_paths: dict,
    flow_type: str,
    cfg: DictConfig,
    mlflow_run_ids: dict = None,
    mlflow_infos: dict = None,
):
    logger.info(f"Logging the {flow_type} visualizations as artifacts")
    if mlflow_run_ids is None:
        if mlflow_infos is not None:
            mlflow_run_ids = get_run_ids_from_infos(mlflow_infos)
        else:
            logger.error("Need some information about the MLflow run")
            raise ValueError("Need some information about the MLflow run")

    for fig_name, path_output_dir in fig_paths.items():
        logger.debug(f"Logging the {fig_name} as artifact from {path_output_dir}")
        for model_name, run_id in mlflow_run_ids.items():
            # Note! This is not run-specific plots as it aggregates all the models (i.e. various MLflow runs)
            # Logging now to every run separately, PNGs are not that massive in the end
            try:
                with mlflow.start_run(run_id):
                    logger.debug(
                        f"MLFLOW Artifact Log | model_name = {model_name}, run_id = {run_id}"
                    )
                    mlflow.log_artifact(path_output_dir, "figures")
            except Exception as e:
                logger.error(
                    f"Could not save the {flow_type} visualization to MLflow: {e}"
                )

        # Save these to Prefect artifacts as well?
