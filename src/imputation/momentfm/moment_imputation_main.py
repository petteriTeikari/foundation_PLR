import time

import mlflow
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from src.anomaly_detection.momentfm_outlier.moment_outlier_zeroshot import (
    momentfm_outlier_zeroshot,
)
from src.imputation.momentfm.moment_utils import (
    import_moment_model,
    init_torch_training,
)
from src.imputation.train_utils import create_imputation_dict_from_moment
from src.metrics.evaluate_imputation_metrics import compute_imputation_metrics
from src.metrics.metrics_utils import check_array_triplet
from src.preprocess.preprocess_data import (
    destandardize_for_imputation_metric,
)


def log_moment_mlflow_params(model, model_cfg: DictConfig):
    """Log MOMENT model parameters to MLflow.

    Parameters
    ----------
    model : MOMENTPipeline
        MOMENT model instance to extract parameter count from.
    model_cfg : DictConfig
        Model configuration containing pretrained model path and kwargs.
    """
    mlflow.log_param("model_name", model_cfg["MODEL"]["pretrained_model_name_or_path"])
    mlflow.log_param("num_params", sum(p.numel() for p in model.encoder.parameters()))
    for k, v in model_cfg["MODEL"]["model_kwargs"].items():
        mlflow.log_param(k, v)


def compute_moment_imputation_metrics(
    imputation_results,
    data_dict,
    cfg,
    imputation_time,
    use_training_data_as_gt=False,
    checks_on: bool = False,
):
    """Compute imputation metrics from MOMENT model outputs.

    Calculates MSE, MAE metrics comparing imputed values to ground truth,
    with optional destandardization for interpretable units.

    Parameters
    ----------
    imputation_results : dict
        Dictionary of imputation results keyed by split name, containing
        'results_dict' with 'split_results' arrays.
    data_dict : dict
        Data dictionary with 'preprocess' standardization stats and 'df'
        containing ground truth data per split.
    cfg : DictConfig
        Configuration for metric computation.
    imputation_time : float
        Time taken for imputation in seconds.
    use_training_data_as_gt : bool, optional
        If True, use training input as ground truth (noisy). If False,
        use human-annotated denoised ground truth. Default is False.
    checks_on : bool, optional
        If True, perform additional validation checks. Default is False.

    Returns
    -------
    tuple
        (metrics, imputation_dict) where metrics is a dict of metrics per
        split and imputation_dict is in PyPOTS-compatible format.

    Raises
    ------
    ValueError
        If split names between imputation results and data don't match.
    """
    stdz_dict = data_dict["preprocess"]["standardization"]
    metrics = {}
    imputation_dict = {}
    # sort the dictionary keys to make sure that the splits are in the same order
    imputation_results = dict(sorted(imputation_results.items()))
    data_dict["df"] = dict(sorted(data_dict["df"].items()))
    for imputation_split, data_split in zip(
        imputation_results.items(), data_dict["df"].items()
    ):
        if imputation_split[0] != data_split[0]:
            # hacky fixes, TODO! harmonize split naming
            if imputation_split[0] == "outlier_test" and data_split[0] == "test":
                split_name = data_split[0]
            elif imputation_split[0] == "outlier_train" and data_split[0] == "train":
                split_name = data_split[0]
            else:
                logger.error(
                    f"Even with accepted exception: Imputation split {imputation_split[0]} "
                    f"does not match {data_split[0]}"
                )
                raise ValueError(
                    f"Even with accepted exception: Imputation split {imputation_split[0]} "
                    f"does not match {data_split[0]}"
                )
        else:
            logger.error(
                f"Imputation split {imputation_split[0]} does not match {data_split[0]}"
            )
            raise ValueError(
                f"Imputation split {imputation_split[0]} does not match {data_split[0]}"
            )

        result_arrays = imputation_split[1]["results_dict"]["split_results"]["arrays"]

        # PyPOTS metrics need the following arrays, with the features as the 3rd dimension (we have only pupil size)
        if use_training_data_as_gt:
            # NOTE! this "trues" now refer to the signal used to train the imputer, not the ground truth necessarily,
            #       as in this was the output from the outlier detection model(s) and can be possibly quite noisy
            # You could obviously evaluate on this noisy data, but we will at the moment use the real human-annotated
            # ground truth (and denoised) for the evaluation
            targets = result_arrays["trues"][
                :, :, np.newaxis
            ]  # the input pupil data that you used to train the imputer
        else:
            # The denoised ground truth
            targets = data_dict["df"][split_name]["data"]["X_GT"][:, :, np.newaxis]

        predictions = result_arrays["preds"][:, :, np.newaxis]  # same as reconstruction
        masks = result_arrays["labels"][:, :, np.newaxis]  # the missing values
        check_array_triplet(predictions=predictions, targets=targets, masks=masks)

        # Get the metrics (MSE, MAE) in input pupil size units (well normalized to baseline) so that they
        # are more intuitive to grasp
        targets_metrics, predictions_metrics = destandardize_for_imputation_metric(
            targets, predictions, stdz_dict
        )

        # Compute the metrics (global, and subject-wise), the same function as for PyPOTS models
        # if your indicating_mask is non-None, error is computed only from the missing values, and not the
        # reconstruction error of the whole PLR signal
        metrics[split_name] = compute_imputation_metrics(
            targets=targets_metrics,
            predictions=predictions_metrics,
            indicating_mask=masks,
            cfg=cfg,
            metadata_dict=data_split[1]["metadata"],
            checks_on=checks_on,
        )

        # Create imputation dict so that the output is the same as in PyPOTS models
        imputation_dict[split_name] = create_imputation_dict_from_moment(
            imputation_mean=predictions,
            indicating_mask=masks,
            imputation_time=imputation_time,
        )

    return metrics, imputation_dict


def moment_imputation_wrapper(
    model,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    data_dict: dict,
    cfg: DictConfig,
    run_name: str,
):
    """Execute MOMENT imputation with zero-shot or fine-tuned model.

    Applies the MOMENT model for time series imputation, leveraging the
    reconstruction capability trained during outlier detection.

    Parameters
    ----------
    model : MOMENTPipeline
        MOMENT model (pretrained or fine-tuned).
    dataloaders : dict[str, DataLoader]
        Dictionary of PyTorch DataLoaders keyed by split name.
    data_dict : dict
        Data dictionary with preprocessing stats and metadata.
    cfg : DictConfig
        Full Hydra configuration.
    run_name : str
        MLflow run name for logging.

    Returns
    -------
    dict
        Dictionary containing:
        - 'imputation_results': Raw model outputs per split
        - 'metrics': Computed imputation metrics
        - 'imputation': PyPOTS-compatible imputation dictionary
    """
    # Re-using the same outlier detection function for zero-shot evaluation
    # This is now "zeroshot", it is either
    # 1) zeroshot, with the model coming from Moment people
    # 2) zeroshot for the finetuned model in the anomaly detection, so the actual time was spent on anomaly detection
    start_time = time.time()
    detection_type = cfg["MODELS"]["MOMENT"]["MODEL"]["detection_type"]
    imputation_results, _, _ = momentfm_outlier_zeroshot(
        model,
        dataloaders,
        data_dict,
        cfg,
        outlier_model_cfg=cfg["MODELS"]["MOMENT"],
        run_name=run_name,
        task_name="imputation",
        tqdm_string="Imputation Zero-Shot ({}) ".format(detection_type),
    )

    # Compute imputation metrics
    metrics, imputation_dict = compute_moment_imputation_metrics(
        imputation_results, data_dict, cfg, imputation_time=time.time() - start_time
    )

    return {
        "imputation_results": imputation_results,
        "metrics": metrics,
        "imputation": imputation_dict,
    }


def moment_imputation_main(
    data_dict: dict,
    model_cfg: DictConfig,
    cfg: DictConfig,
    model_name: str = None,
    run_name: str = None,
):
    """
    See the original Moment imputation tutorial:
    https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/imputation.ipynb

    See moment-research/moment/tasks/imputation_finetune.py
    https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/moment/tasks/imputation_finetune.py
    https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/scripts/finetuning/imputation.py
    https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/configs/imputation/linear_probing.yaml
    """

    # init stuff
    dataloaders = init_torch_training(
        data_dict,
        cfg,
        model_cfg,
        run_name=run_name,
        task="imputation",
        create_outlier_dataloaders=False,
    )

    # Import the pretrained model
    model = import_moment_model(model_cfg, task="imputation", cfg=cfg)
    model = model.to(cfg["DEVICE"]["device"]).float()

    # Log the model parameters to MLflow
    log_moment_mlflow_params(model, model_cfg)

    # Either zero-shot imputation (no training) or fine-tuning (with some training)
    # Now in contrast to the outlier detection, we don't finetune anymore per se
    # We loaded the finetuned (reconstruction) model ("init_torch_training") and the following evaluation
    # is always zero-shot, and the "real zero-shot" is for the Moment standard pretrained model from HuggingFace
    model_artifacts = moment_imputation_wrapper(
        model, dataloaders, data_dict, cfg, run_name=run_name
    )

    return model, model_artifacts
