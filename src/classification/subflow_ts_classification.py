from omegaconf import DictConfig
from loguru import logger

from src.classification.classification_ts.train_ts_classifier import train_ts_classifier
from src.classification.classifier_log_utils import retrain_classifier
from src.classification.classifier_utils import (
    keep_only_labeled_subjects_from_source,
    cls_is_on_ground_truth,
)
from src.data_io.define_sources_for_flow import define_sources_for_flow
from src.log_helpers.log_naming_uris_and_dirs import get_moment_cls_run_name


def prepare_sources_for_classifcation(sources: dict, cfg: DictConfig):
    """
    sources: dict
        pupil-gt__pupil-gt: dict
            df: dict
                test: dict
                    time: dict
                        ndarrays
                    data: dict
                        X: np.ndarray e.g. (152,1981) (no_samples, no_timepoints)
                        X_GT: np.ndarray
                        mask: np.ndarray
                        CI_pos: np.ndarray
                        CI_neg: np.ndarray
                    labels: dict
                        ndarrays
                    light: dict
                        ndarrays
                    metadata: dict
                        ndarrays
                train
                    same as test
            preprocess: dict
            mlflow: ?
    """
    logger.info("Dropping samples without a class_label (e.g. control vs glaucoma)")
    for source in sources.keys():
        sources[source]["df"] = keep_only_labeled_subjects_from_source(
            data_dicts=sources[source]["df"], cfg=cfg
        )

    return sources


def flow_ts_classification(cfg: DictConfig, prev_experiment_name: str) -> None:
    # Get the data sources (from imputation, and from original ground truth DuckDB database)
    sources = define_sources_for_flow(
        cfg=cfg, prev_experiment_name=prev_experiment_name, task="imputation"
    )

    # Prepare sources for classification (i.e. drop samples without a class label)
    # as the function below is a generic one, used also with outlier detection/imputation
    sources = prepare_sources_for_classifcation(sources, cfg=cfg)

    # PLR Classification
    no_of_runs = len(sources) * len(cfg["CLS_TS_MODELS"])
    run_idx = 0
    for i, (source_name, features_per_source) in enumerate(sources.items()):
        for j, cls_model_name in enumerate(cfg["CLS_TS_MODELS"]):
            cls_model_cfg = cfg["CLS_TS_MODELS"][cls_model_name]
            logger.info(f"Source #{i+1}/{len(sources)}: {source_name}")
            logger.info(
                f"Running pipeline for hyperparameter group #{j+1}/{len(cfg['CLS_MODELS'])}: {cls_model_name}"
            )
            run_name = f"{get_moment_cls_run_name(cls_model_name, cls_model_cfg)}__{source_name}"
            logger.info(f"Run name #{run_idx+1}/{no_of_runs}: {run_name}")
            run_idx += 1

            # Train the classifier
            if cls_is_on_ground_truth(run_name=run_name):
                # this does not seem to perform so well atm so no point in wasting time for all the combos
                if retrain_classifier(
                    run_name, cfg, cfg_key="CLASSIFICATION_TS_SETTINGS"
                ):
                    train_ts_classifier(
                        source_name,
                        features_per_source,
                        cls_model_name,
                        run_name,
                        cfg,
                        cls_model_cfg,
                    )
            else:
                logger.info("Skipping run name = {}".format(run_name))
