from loguru import logger
from omegaconf import DictConfig

from src.classification.classifier_log_utils import retrain_classifier
from src.classification.classifier_utils import (
    cls_train_on_this_combo,
    drop_useless_cols,
    keep_only_labeled_subjects,
)
from src.classification.train_classifier import train_classifier
from src.ensemble.tasks_ensembling import task_ensemble
from src.featurization.embedding.dim_reduction import (
    apply_dimensionality_reduction_for_feature_sources,
)
from src.featurization.feature_log import import_features_from_mlflow
from src.log_helpers.log_naming_uris_and_dirs import (
    update_cls_run_name,
)
from src.orchestration.debug_utils import debug_classification_macro


# @task(
#     log_prints=True,
#     name="Get hand-crafted features",
#     description="Get hand-crafted features",
# )
def get_the_features(cfg: DictConfig, experiment_name: str) -> dict:
    """
    Load and prepare features for classification.

    Loads precomputed features from MLflow, filters to labeled subjects,
    removes metadata columns, and applies dimensionality reduction if
    configured for embeddings.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    experiment_name : str
        MLflow experiment name for featurization.

    Returns
    -------
    dict
        Features dictionary with data per source and split.
    """
    # Task) Load the precomputed features from MLflow
    features = import_features_from_mlflow(cfg, experiment_name=experiment_name)

    # Task) Select subjects for the classification task
    # You have now the features of all the subjects, but not all have labels, so if you want to do supervised
    # classification, just keep the ones with a label. If you have some semi-supervised idea, you can skip the drop
    features = keep_only_labeled_subjects(features=features, cfg=cfg)

    # Drop "useless columns" like "outlier_mask"
    features = drop_useless_cols(features, cfg)

    # Dimensionality reduction for the embeddings (if desired)
    features = apply_dimensionality_reduction_for_feature_sources(
        features=features, cfg=cfg
    )

    # Task) Create a flat dataframe
    # -> export to DuckDB

    # sort the sources
    features = dict(reversed(sorted(features.items())))

    return features


# @flow(
#     log_prints=True,
#     name="PLR Classification",
#     description="Classify Glaucoma from the PLR features",
# )
def flow_feature_classification(cfg: DictConfig, prev_experiment_name: str) -> None:
    """
    Classification subflow for hand-crafted features and embeddings.

    Iterates over all preprocessing sources and classifier models,
    training and evaluating each combination. Skips certain combinations
    based on filtering logic (e.g., experimental models on non-GT data).

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with CLS_MODELS and settings.
    prev_experiment_name : str
        MLflow experiment name for featurization.
    """
    # Get the features from MLflow
    features = get_the_features(cfg=cfg, experiment_name=prev_experiment_name)

    if cfg["EXPERIMENT"]["debug"]:
        cfg = debug_classification_macro(cfg)

    # PLR Classification
    no_of_runs = len(features) * len(cfg["CLS_MODELS"])
    run_idx = 0
    for i, (source_name, features_per_source) in enumerate(features.items()):
        for j, cls_model_name in enumerate(cfg["CLS_MODELS"]):
            logger.info(f"Source #{i + 1}/{len(features)}: {source_name}")
            logger.info(
                f"Running pipeline for hyperparameter group #{j + 1}/{len(cfg['CLS_MODELS'])}: {cls_model_name}"
            )

            # Define run name
            run_name = update_cls_run_name(
                cls_model_name,
                source_name,
                model_cfg=cfg["CLS_MODELS"][cls_model_name],
                hparam_cfg=cfg["CLS_HYPERPARAMS"][cls_model_name],
                cfg=cfg,
            )

            if cls_train_on_this_combo(
                run_name=run_name, cls_model_name=cls_model_name
            ):
                logger.info(f"Run name #{run_idx + 1}/{no_of_runs}: {run_name}")

                # Train the classifier
                if retrain_classifier(run_name, cfg):
                    train_classifier(
                        source_name,
                        features_per_source,
                        cls_model_name,
                        run_name,
                        cfg,
                    )
            else:
                logger.info("Skipping run name = {}".format(run_name))

            run_idx += 1

    # The ensembling part will fetch the trained imputation models from MLflow
    task_ensemble(cfg=cfg, task="classification", sources=features)
