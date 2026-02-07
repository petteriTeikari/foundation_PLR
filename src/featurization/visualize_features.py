from loguru import logger
from omegaconf import DictConfig

from src.log_helpers.viz_log_utils import export_viz_as_artifacts
from src.viz.viz_features import visualize_features


def visualize_features_of_all_sources(
    features: dict, mlflow_infos: dict, cfg: DictConfig
):
    """Generate and export feature visualizations for all data sources.

    Creates visualizations combining features from multiple sources and
    logs them as MLflow artifacts.

    Parameters
    ----------
    features : dict
        Dictionary of features keyed by data source.
    mlflow_infos : dict
        MLflow run information for artifact logging.
    cfg : DictConfig
        Configuration with VISUALIZATION settings.
    """
    if cfg["PLR_FEATURIZATION"]["VISUALIZATION"]["visualize_features"]:
        # Visualize features from all the sources in one figure
        fig_paths = visualize_features(features=features, cfg=cfg)

        # Note! Now that we had all different sources in one figure, we cannot log image to a specific run,
        # so as a compromise, we write the same figure to each of the runs (sources, models) visualized
        export_viz_as_artifacts(
            fig_paths,
            flow_type="featurization",
            cfg=cfg,
            mlflow_infos=mlflow_infos,
        )

        # If you want to loop and visualize source (model) by source
        # for data_source in features.keys():
        #     data_features = features[data_source]["data"]
        #     mlflow_run = features[data_source]["mlflow_run"]

    else:
        logger.info("Skipping the visualization of the features")
