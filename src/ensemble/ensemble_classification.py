import warnings

import numpy as np
from loguru import logger
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.anomaly_detection.anomaly_utils import get_artifact
from src.classification.bootstrap_evaluation import get_ensemble_stats
from src.classification.stats_metric_utils import bootstrap_metrics_per_split
from src.classification.xgboost_cls.xgboost_utils import encode_labels_to_integers
from src.ensemble.ensemble_utils import are_codes_the_same
from src.log_helpers.local_artifacts import load_results_dict


def import_model_metrics(run_id, run_name, model_name, subdir: str = "baseline_model"):
    artifacts_path = get_artifact(run_id, run_name, model_name, subdir=subdir)
    artifacts = load_results_dict(artifacts_path)
    return artifacts


def get_preds_and_labels_from_artifacts(artifacts):
    # NOTE! to make things simpler, we get the data from the stats subdict which are averaged from n bootstrap iters
    # this does not account the case in which you did not use same number of bootstrap iters (if you care), assuming
    # that all the models used the same number of iterations
    y_pred_proba, y_pred_proba_var, label = {}, {}, {}
    for split in artifacts["subjectwise_stats"].keys():
        y_pred_proba[split] = artifacts["subjectwise_stats"][split]["preds"][
            "y_pred_proba"
        ]["mean"]
        y_pred_proba_var[split] = (
            artifacts["subjectwise_stats"][split]["preds"]["y_pred_proba"]["std"] ** 2
        )
        label[split] = artifacts["subjectwise_stats"][split]["preds"]["label"]["mean"]

    return y_pred_proba, y_pred_proba_var, label


def import_model_preds_and_labels(
    run_id, run_name, model_name, subdir: str = "metrics"
):
    artifacts = import_model_metrics(run_id, run_name, model_name, subdir=subdir)
    y_pred_proba, y_pred_proba_var, label = get_preds_and_labels_from_artifacts(
        artifacts
    )
    return y_pred_proba, y_pred_proba_var, label


def import_metrics_iter(run_id, run_name, model_name, subdir: str = "metrics"):
    artifacts = import_model_metrics(run_id, run_name, model_name, subdir=subdir)
    return artifacts["metrics_iter"]


def concentate_one_var(
    array_out: dict[str, np.ndarray], array_per_submodel: dict[str, np.ndarray]
):
    if array_out is None:
        array_out = {}
        for split in array_per_submodel.keys():
            array_out[split] = array_per_submodel[split][np.newaxis, :]
    else:
        for split in array_out.keys():
            array_out[split] = np.concatenate(
                [array_out[split], array_per_submodel[split][np.newaxis, :]], axis=0
            )

    return array_out


def concatenate_arrays(
    preds_out, preds_var_out, labels_out, y_pred_proba, y_pred_proba_var, label
):
    preds_out = concentate_one_var(preds_out, y_pred_proba)
    preds_var_out = concentate_one_var(preds_var_out, y_pred_proba_var)
    # should be the same, thus no need for this
    # labels_out = concentate_one_var(labels_out, label)

    return preds_out, preds_var_out, label


def check_dicts(preds_out, preds_var_out, labels_out, no_submodel_runs):
    for split in preds_out.keys():
        assert preds_out[split].shape[0] == no_submodel_runs, (
            f"preds_out[split].shape[0]: "
            f"{preds_out[split].shape[0]}, "
            f"no_submodel_runs: {no_submodel_runs}"
        )
        assert preds_var_out[split].shape[0] == no_submodel_runs, (
            f"preds_var_out[split].shape[0]: "
            f"{preds_var_out[split].shape[0]}, "
            f"no_submodel_runs: {no_submodel_runs}"
        )


def compute_stats(preds_out, preds_var_out):
    preds = {}
    preds_std = {}
    preds_meanstd = {}
    for split in preds_out.keys():
        preds[split] = np.mean(preds_out[split], axis=0)
        preds_std[split] = np.std(preds_out[split], axis=0)
        preds_meanstd[split] = np.mean(preds_var_out[split], axis=0) ** 0.5

    return preds, preds_std, preds_meanstd


def aggregate_pred_dict(preds_out, preds_per_submodel, ensemble: bool = False):
    for var in preds_per_submodel:  # e.g. y_pred_proba, y_pred, label
        assert isinstance(preds_per_submodel[var], dict), (
            f"preds_per_submodel[var] is not a dict, "
            f"but {type(preds_per_submodel[var])}"
        )
        unique_codes_out = sorted(list(preds_out[var].keys()))
        unique_codes_in = sorted(list(preds_per_submodel[var].keys()))
        if ensemble:
            assert unique_codes_out == unique_codes_in, (
                "You do not have to have the same subjects in all splits? \n"
                "As in you ran some MLflow with runs with certain subjects,\n"
                "And later redefined the splits?"
            )
        for code in preds_per_submodel[var]:
            list_of_preds = preds_per_submodel[var][code]
            # no_of_bootstrap_iters = len(list_of_preds)
            preds_out[var][code] += list_of_preds

    return preds_out


def aggregate_preds(preds_out, preds_per_submodel):
    for split in preds_per_submodel:
        preds_out[split] = np.concatenate(
            [preds_out[split], preds_per_submodel[split]], axis=1
        )
    return preds_out


def check_metrics_iter_preds_dict(dict_arrays: dict[str, dict[str, list]]):
    # (no_subjects, no_of_bootstrap_iters)
    if "labels" in dict_arrays:
        # TODO! why with CATBOOST? fix this eventually so that only one key
        assert (
            len(dict_arrays["y_pred_proba"]) == len(dict_arrays["labels"])
        ), f'you have {len(dict_arrays["y_pred_proba"])} y_pred_proba and {len(dict_arrays["labels"])} labels'
    elif "label" in dict_arrays:
        assert (
            len(dict_arrays["y_pred_proba"]) == len(dict_arrays["label"])
        ), f'you have {len(dict_arrays["y_pred_proba"])} y_pred_proba and {len(dict_arrays["label"])} labels'
    assert len(dict_arrays["y_pred_proba"]) == len(dict_arrays["y_pred"])


def check_metrics_iter_preds(dict_arrays: dict[str, np.ndarray]):
    # (no_subjects, no_of_bootstrap_iters)
    if "label" in dict_arrays:
        assert (
            dict_arrays["y_pred_proba"].shape[0] == dict_arrays["label"].shape[0]
        ), f'you have {dict_arrays["y_pred_proba"].shape[0]} y_pred_proba and {dict_arrays["label"].shape[0]} labels'
    elif "labels" in dict_arrays:
        # TODO! why with CATBOOST? fix this eventually so that only one key
        assert (
            dict_arrays["y_pred_proba"].shape[0] == dict_arrays["labels"].shape[0]
        ), f'you have {dict_arrays["y_pred_proba"].shape[0]} y_pred_proba and {dict_arrays["labels"].shape[0]} labels'
    assert dict_arrays["y_pred_proba"].shape[0] == dict_arrays["y_pred"].shape[0]


def check_metrics_iter_shapes(iter_split):
    if "preds_dict" in iter_split:
        check_metrics_iter_preds_dict(dict_arrays=iter_split["preds_dict"]["arrays"])
    else:
        check_metrics_iter_preds(
            dict_arrays=iter_split["preds"]["arrays"]["predictions"]
        )


def check_subjects_in_splits(metrics_iter):
    if metrics_iter is not None:
        subjects = {}
        for split in metrics_iter.keys():
            if "preds_dict" in metrics_iter[split]:
                subjects[split] = sorted(
                    list(
                        metrics_iter[split]["preds_dict"]["arrays"][
                            "y_pred_proba"
                        ].keys()
                    )
                )
            elif "preds" in metrics_iter[split]:
                no_subjects_preds = len(
                    metrics_iter[split]["preds"]["arrays"]["predictions"][
                        "y_pred_proba"
                    ]
                )
                logger.debug(f"{no_subjects_preds} in test split")
            else:
                logger.error("Where are the preds?")
                raise ValueError("Where are the preds?")

        if "val" in subjects:
            assert (
                subjects["train"] == subjects["val"]
            ), "Your train and val codes do not match?"

        return subjects
    else:
        return None


def check_compare_subjects_for_aggregation(
    subject_codes, subject_codes_model, run_name, i, split_to_check: str = "train"
):
    error_run = []
    if subject_codes is not None and subject_codes_model is not None:
        for split in subject_codes.keys():
            # these come from whole bootstrap experiment, so train and val should have the same codes
            if split == split_to_check:
                if subject_codes[split] != subject_codes_model[split]:
                    logger.error(
                        f"Submodel #{i+1}: {run_name} seem to have different subjects in splits than in previous submodels"
                    )
                    error_run = [run_name]
                    if len(subject_codes[split]) != len(subject_codes_model[split]):
                        logger.error(
                            "Lengths of splits do not even seem to match, ensemble n = {}, submodel n = {}".format(
                                len(subject_codes[split]),
                                len(subject_codes_model[split]),
                            )
                        )
                        # raise ValueError('Your ensemble seem to come from different splits, did you redefine the splits\n'
                        #                  'while running the experiments? Need to delete the runs with old split definitions\n'
                        #                  'and rerun those to get this ensembling working?')
                    else:
                        logger.debug("Ensemble codes | Model Codes")
                        for code_ens, code in zip(
                            subject_codes[split], subject_codes_model[split]
                        ):
                            logger.debug(f"{code_ens} | {code}")
                        # raise ValueError('Your ensemble seem to come from different splits, did you redefine the splits\n'
                        #                  'while running the experiments? Need to delete the runs with old split definitions\n'
                        #                  'and rerun those to get this ensembling working?')

    return error_run


def aggregate_metric_iter(
    metrics_iter, metrics_iter_model, run_name: str, ensemble: bool = False
):
    if metrics_iter is None:
        metrics_iter = {}
        for split in metrics_iter_model.keys():
            metrics_iter[split] = metrics_iter_model[split]
            metrics_iter[split].pop("metrics")

    else:
        for split in metrics_iter.keys():
            metrics_iter_model[split].pop("metrics")
            if "preds_dict" in metrics_iter_model[split]:
                # train/val
                preds: dict[str, dict] = metrics_iter_model[split]["preds_dict"][
                    "arrays"
                ]
                metrics_iter[split]["preds_dict"]["arrays"] = aggregate_pred_dict(
                    preds_out=metrics_iter[split]["preds_dict"]["arrays"],
                    preds_per_submodel=preds,
                    ensemble=ensemble,
                )
                check_metrics_iter_preds_dict(
                    dict_arrays=metrics_iter[split]["preds_dict"]["arrays"]
                )

            else:
                # test
                preds: dict[str, np.ndarray] = metrics_iter_model[split]["preds"][
                    "arrays"
                ]["predictions"]
                metrics_iter[split]["preds"]["arrays"]["predictions"] = aggregate_preds(
                    preds_out=metrics_iter[split]["preds"]["arrays"]["predictions"],
                    preds_per_submodel=preds,
                )
                check_metrics_iter_preds(
                    dict_arrays=metrics_iter[split]["preds"]["arrays"]["predictions"]
                )

    return metrics_iter


def get_label_array(label_dict: dict[str, np.ndarray]):
    label_array = []
    for code in label_dict.keys():
        label_array.append(label_dict[code][0])
    label_array = np.array(label_array)
    assert label_array.shape[0] == len(label_dict), (
        f"label_array.shape[0]: {label_array.shape[0]}, "
        f"len(label_dict): {len(label_dict)}"
    )
    return label_array


def get_preds_array(preds_dict: dict[str, np.ndarray]):
    def get_min_bootstrap_iters_from_subjects(preds_dict):
        lengths = []
        for code in preds_dict.keys():
            lengths.append(len(preds_dict[code]))
        return min(lengths)

    array_iter_no = get_min_bootstrap_iters_from_subjects(preds_dict)
    preds_array = np.zeros((len(preds_dict), array_iter_no))
    for i, code in enumerate(preds_dict.keys()):
        preds_array[i] = preds_dict[code][:array_iter_no]

    return preds_array


def recompute_ensemble_metrics(metrics_iter, sources: dict, cfg: DictConfig):
    warnings.simplefilter("ignore")
    # skip "val" for now
    splits = ["train", "test"]
    for split in splits:  # metrics_iter.keys():
        if "preds_dict" in metrics_iter[split]:
            y_true = get_label_array(
                label_dict=metrics_iter[split]["preds_dict"]["arrays"]["label"]
            )
            preds_array = get_preds_array(
                preds_dict=metrics_iter[split]["preds_dict"]["arrays"]["y_pred_proba"]
            )

        elif "preds" in metrics_iter[split]:
            y_true = metrics_iter[split]["preds"]["arrays"]["predictions"]["label"][
                :, 0
            ]
            preds_array = metrics_iter[split]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ]

        else:
            logger.error(
                "Where are the predictions? {}".format(metrics_iter[split].keys())
            )
            raise ValueError(
                "Where are the predictions? {}".format(metrics_iter[split].keys())
            )

        method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]
        dict_arrays_compact = get_compact_dict_arrays(sources)
        codes_per_split = dict_arrays_compact[f"subject_codes_{split}"]

        for idx in tqdm(
            range(preds_array.shape[1]),
            desc=f"Recomputing ensemble metrics for {split}",
        ):
            preds = create_pred_dict(split_preds=preds_array[:, idx], y_true=y_true)
            metrics_iter[split] = bootstrap_metrics_per_split(
                None,
                y_true,
                preds,
                None,
                model_name="ensemble",
                metrics_per_split=metrics_iter[split],
                codes_per_split=codes_per_split,
                method_cfg=method_cfg,
                cfg=cfg,
                split=split,
                skip_mlflow=True,
                recompute_for_ensemble=True,
            )
            check_metrics_iter_shapes(iter_split=metrics_iter[split])

    warnings.resetwarnings()

    return metrics_iter


def get_cls_preds_from_artifact(
    run, i, no_submodel_runs, aggregate_preds: bool = False
):
    run_id = run["run_id"]
    run_name = run["tags.mlflow.runName"]
    model_name = run["params.model_name"]
    if aggregate_preds:
        logger.info(
            f"{i + 1}/{no_submodel_runs}: Ensembling model: {model_name}, run_id: {run_id}, run_name: {run_name}"
        )
    # Baseline (as in no bootstrapping)
    # preds_baseline, labels_baseline = import_model_preds_and_labels(run_id, run_name, model_name, subdir='baseline_model')

    # Bootstrapped model
    # y_pred_proba, y_pred_proba_var, label = (
    #     import_model_preds_and_labels(run_id, run_name, model_name, subdir='metrics'))
    # preds_out, preds_var_out, labels_out = concatenate_arrays(
    #     preds_out, preds_var_out, labels_out, y_pred_proba, y_pred_proba_var, label
    # )
    metrics_iter_model = import_metrics_iter(
        run_id, run_name, model_name, subdir="metrics"
    )
    n = metrics_iter_model["test"]["preds"]["arrays"]["predictions"][
        "y_pred_proba"
    ].shape[1]
    if aggregate_preds:
        logger.info("Submodel consists of {} bootstrap iterations".format(n))

    return metrics_iter_model


def aggregate_submodels(
    ensemble_model_runs,
    no_submodel_runs,
    aggregate_preds: bool = True,
    split_to_check="train",
    ensemble_codes: pd.DataFrame = None,
):
    metrics_iter = None
    subject_codes_out = {}
    error_runs = []

    for i, (idx, run) in enumerate(ensemble_model_runs.iterrows()):
        metrics_iter_model = get_cls_preds_from_artifact(
            run, i, no_submodel_runs, aggregate_preds=aggregate_preds
        )
        subject_codes = check_subjects_in_splits(metrics_iter)
        subject_codes_model = check_subjects_in_splits(metrics_iter_model)
        error_runs += check_compare_subjects_for_aggregation(
            subject_codes, subject_codes_model, run["tags.mlflow.runName"], i
        )
        subject_codes_out[run["tags.mlflow.runName"]] = subject_codes_model[
            split_to_check
        ]
        df_out = pd.DataFrame()

        if aggregate_preds:
            metrics_iter = aggregate_metric_iter(
                metrics_iter,
                metrics_iter_model,
                run_name=run["tags.mlflow.runName"],
                ensemble=True,
            )

    for run_name, list_of_codes in subject_codes_out.items():
        df_out[run_name] = list_of_codes
    all_submodels_have_same_codes = are_codes_the_same(df_out)

    if not aggregate_preds:
        if len(error_runs) > 0:
            # well assuming that the first submodel of the ensemble is correct
            logger.error(
                "These runs seem to be done with a different set of subjects than others?"
            )
            for run in error_runs:
                logger.error(run)
            raise ValueError(
                "Your ensemble seem to come from different splits, did you redefine the splits\n"
                "while running the experiments? Need to delete the runs with old split definitions\n"
                "and rerun those to get this ensembling working?"
            )

    return metrics_iter, df_out, all_submodels_have_same_codes


def get_classification_preds(ensemble_model_runs, sources: dict, cfg: DictConfig):
    no_submodel_runs = ensemble_model_runs.shape[0]
    if no_submodel_runs > 0:
        _, ensemble_codes, same_codes = aggregate_submodels(
            ensemble_model_runs, no_submodel_runs, aggregate_preds=False
        )
        if same_codes:
            metrics_iter, _, _ = aggregate_submodels(
                ensemble_model_runs,
                no_submodel_runs,
                aggregate_preds=True,
                ensemble_codes=ensemble_codes,
            )
            # This metrics_iter is now the same that you would get from the "normal bootstrap" with just all the
            # iterations aggregated together
            n = metrics_iter["test"]["preds"]["arrays"]["predictions"][
                "y_pred_proba"
            ].shape[1]
            logger.info(
                "Ensemble consists a total of {} bootstrap iterations".format(n)
            )

            # compute the metrics for the ensemble
            metrics_iter = recompute_ensemble_metrics(metrics_iter, sources, cfg)
        else:
            logger.warning(
                "The codes used to train different submodels do not seem to be the same!"
            )
            metrics_iter = None
    else:
        metrics_iter = None

    return metrics_iter


def create_pred_dict(split_preds: np.ndarray, y_true: np.ndarray):
    preds = {
        "y_pred": (split_preds > 0.5).astype(int),
        "y_pred_proba": split_preds,
        "labels": y_true,
    }
    return preds


def get_compact_dict_arrays(sources):
    """
    see e.g. bootstrap_compute_subject_stats()
             get_labels_and_codes()
    for where these are needed
    """
    first_feature_source = list(sources.keys())[0]
    data = sources[first_feature_source]["data"]
    dict_arrays_compact = {}
    for split in data.keys():
        dict_arrays_compact[f"subject_codes_{split}"] = data[split][
            "subject_code"
        ].to_numpy()
        # this is now as a string, e.g. "control" vs. "glaucoma"
        dict_arrays_compact[f"y_{split}"] = data[split][
            "metadata_class_label"
        ].to_numpy()
        # to integers
        dict_arrays_compact[f"y_{split}"] = encode_labels_to_integers(
            dict_arrays_compact[f"y_{split}"]
        )

    return dict_arrays_compact


def compute_cls_ensemble_metrics(metrics_iter: dict, sources: dict, cfg: DictConfig):
    method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]
    dict_arrays_compact = get_compact_dict_arrays(sources)
    metrics_stats, subjectwise_stats, subject_global_stats = get_ensemble_stats(
        metrics_iter,
        dict_arrays_compact,
        method_cfg,
        call_from="classification_ensemble",
    )

    metrics = {
        "metrics_iter": metrics_iter,
        "metrics_stats": metrics_stats,
        "subjectwise_stats": subjectwise_stats,
        "subject_global_stats": subject_global_stats,
    }

    return metrics


def ensemble_classification(
    ensemble_model_runs: pd.DataFrame,
    cfg: DictConfig,
    sources: dict,
    ensemble_name: str,
):
    # Get imputation mask and labels for each model
    metrics_iter = get_classification_preds(ensemble_model_runs, sources, cfg=cfg)

    # Compute the metrics for the ensemble
    if metrics_iter is not None:
        metrics = compute_cls_ensemble_metrics(metrics_iter, sources, cfg)
    else:
        metrics = None

    return metrics
