import os
import time
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from omegaconf import DictConfig

from src.imputation.impute_with_models import pypots_imputer_wrapper
from src.imputation.pypots.pypots_utils import (
    create_dataset_dicts_for_pypots,
    define_pypots_outputs,
)


def pypots_wrapper(
    source_data: Dict[str, Any],
    model_cfg: DictConfig,
    cfg: DictConfig,
    model_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Wrapper function for the PyPOTS models
    https://github.com/WenjieDu/PyPOTS
    """
    output_dir, model_fname, model_artifacts_path = define_pypots_outputs(
        model_name=model_name, artifact_type="model"
    )

    # PyPots expects this format, so put the numpy arrays to the dataset dicts
    dataset_dicts = create_dataset_dicts_for_pypots(source_data)

    # Train the model
    model, model_artifacts = pypots_model_wrapper(
        dataset_train=dataset_dicts["train"],
        dataset_test=dataset_dicts["test"],
        cfg=cfg,
        model_params=model_cfg["MODEL"],
        artifact_dir=output_dir,
        model_name=model_name,
    )

    # Use the imputer to impute the data
    model_artifacts["imputation"] = pypots_imputer_wrapper(
        model, model_name, dataset_dicts, source_data, cfg
    )

    # Compute imputation metrics
    # atm the computation is done after the whole flow, see
    # "metrics = compute_metrics_by_model(model_name, imputation_artifacts, cfg)"

    return model, model_artifacts


def saits_model_wrapper(
    dataset: Dict[str, Any],
    cfg: DictConfig,
    model_params: DictConfig,
    artifact_dir: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    from pypots.imputation import SAITS

    n_steps = dataset["X"].shape[1]
    n_features = dataset["X"].shape[2]

    if "timing" not in results:
        results["timing"] = {}

    # Initialize the SAITS model
    saits = SAITS(
        n_steps=n_steps, n_features=n_features, saving_path=artifact_dir, **model_params
    )

    results["train"] = {}
    # TODO! Add the Tensorboard loss logging, i.e. "training curves",
    #  or extract them from the disk as PyPOTS saves these

    logger.info("Start training the SAITS model")
    start_time = time.time()
    saits.fit(dataset)  # train the model on the dataset
    results["timing"]["train"] = time.time() - start_time
    logger.info("SAITS model training finished!")

    return saits, results


def pypots_model_wrapper(
    dataset_train: Dict[str, Any],
    dataset_test: Dict[str, Any],
    cfg: DictConfig,
    model_params: DictConfig,
    artifact_dir: str,
    model_name: str,
) -> Tuple[Any, Dict[str, Any]]:
    from pypots.imputation import (
        CSDI,
        SAITS,
        TimesNet,
    )  # will import the annoying ASCII logo

    n_steps = dataset_train["X"].shape[1]
    n_features = dataset_train["X"].shape[2]

    model_artifacts = {}
    model_artifacts["timing"] = {}

    # Initialize the PyPOTS model
    if model_name == "SAITS":
        model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            saving_path=artifact_dir,
            **model_params,
        )
    elif model_name == "CSDI":
        model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            saving_path=artifact_dir,
            **model_params,
        )
    elif model_name == "TimesNet":
        model = TimesNet(
            n_steps=n_steps,
            n_features=n_features,
            saving_path=artifact_dir,
            **model_params,
        )
    else:
        logger.error('Model "{}" not implemented!'.format(model_name))
        raise NotImplementedError('Model "{}" not implemented!'.format(model_name))

    model_artifacts["model_info"] = {
        "num_params": model.num_params,
        "saving_path": model.saving_path,
        "tensorboard_dir": model.summary_writer.log_dir,
        "tensorboar_filename_suffix": model.summary_writer.filename_suffix,
        "PyPOTS": True,
    }

    # Add the Tensorboard loss logging? Get this in the summary flow?
    logger.info("Start training the {} model".format(model_name))
    start_time = time.time()
    model.fit(
        train_set=dataset_train, val_set=dataset_test
    )  # train the model on the dataset
    model_artifacts["timing"]["train"] = time.time() - start_time
    logger.info("{} model training finished!".format(model_name))

    return model, model_artifacts


def export_pypots_model(
    model: Any, model_path: str, results: Dict[str, Any], debug_load: bool = True
) -> Dict[str, Any]:
    artifact_dir, fname = os.path.split(model_path)
    if not os.path.exists(artifact_dir):
        logger.warning(
            "Artifact directory does not exist, cannot save the model! {}".format(
                artifact_dir
            )
        )
    else:
        logger.info("Saving the {} model to disk: {}".format(fname, model_path))
        start_time = time.time()
        model.save(model_path)  # save the model for future use
        results["timing"]["model_save"] = time.time() - start_time
        if debug_load:
            start_time = time.time()
            model.load(model_path)
            results["timing"]["model_load"] = time.time() - start_time
            logger.info(
                "{} model loaded from disk: {} (in {:.2f} ms)".format(
                    fname, model_path, 1000 * results["timing"]["model_load"]
                )
            )

    return results
