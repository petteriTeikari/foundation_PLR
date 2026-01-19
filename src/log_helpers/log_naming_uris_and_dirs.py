import os

from loguru import logger
from omegaconf import DictConfig

from src.utils import get_data_dir, get_artifacts_dir


def get_feature_pickle_artifact_uri(run, source, cfg, subdir="features"):
    return f'runs:/{run["run_id"]}/{subdir}/{get_feature_pickle_base(source)}'


def get_feature_pickle_base(run_name):
    return f"{run_name}.pickle"


def get_features_pickle_fname(data_source: str) -> str:
    return get_feature_pickle_base(data_source)


def get_baseline_names():
    return ["BASELINE_DenoisedGT", "BASELINE_OutlierRemovedRaw"]


def get_feature_name_from_cfg(cfg):
    return (
        f'{cfg["PLR_FEATURIZATION"]["FEATURES_METADATA"]["name"]}'
        f'{cfg["PLR_FEATURIZATION"]["FEATURES_METADATA"]["version"]}'
    )


def define_featurization_run_name_from_base(base_name: str, cfg: DictConfig):
    return f"features-{get_feature_name_from_cfg(cfg)}_{base_name}"


def xgboost_variant_run_name(
    run_name: str, xgboost_cfg: DictConfig, model_name: str = "XGBOOST"
) -> str:
    variant_name = xgboost_cfg["variant_name"]
    if len(variant_name) > 0:
        return run_name.replace(model_name, f"{model_name}_{variant_name}")
    else:
        return run_name


def get_pypots_model_path(results_path, ext_out: str = ".pypots"):
    basedir, fname = os.path.split(results_path)
    fname, ext = os.path.splitext(fname)
    fname = fname.replace("results", "model")
    return os.path.join(basedir, fname + ext_out)


def get_mlflow_metric_name(split, metric_key):
    return f"{split}/{metric_key}"


def get_outlier_pickle_name(model_name):
    return f"outlierDetection_{model_name}.pickle"


def get_outlier_csv_name(model_name, split, key):
    base_fname = get_outlier_pickle_name(model_name).replace(".pickle", "")
    return f"{base_fname}_{split}_{key}.csv"


def get_duckdb_file(data_cfg,
                    use_demo_data: bool = False,
                    demo_db_file: str = 'PLR_demo_data.db'):
    data_dir = get_data_dir(data_path=data_cfg["data_path"])
    if use_demo_data:
        logger.warning(f'Using the demo data ({demo_db_file}) for testing the pipeline')
        db_path = os.path.join(data_dir, demo_db_file)
    else:
        db_path = os.path.join(data_dir, data_cfg["filename_DuckDB"])
    if not os.path.isfile(db_path):
        logger.error(f"File {db_path} does not exist")
        raise FileNotFoundError(db_path)

    return db_path


def update_outlier_detection_run_name(cfg: DictConfig):
    if len(cfg["OUTLIER_MODELS"].keys()) == 1:
        model_name = list(cfg["OUTLIER_MODELS"].keys())[0]
    else:
        logger.error("Only one model should be used for outlier detection")
        raise ValueError("Only one model should be used for outlier detection")
    if model_name == "MOMENT":
        # finetune or zeroshot
        detection_type = cfg["OUTLIER_MODELS"][model_name]["MODEL"][
            "detection_type"
        ].replace("-", "")
        # large, base, or small
        model_variant = cfg["OUTLIER_MODELS"][model_name]["MODEL"][
            "pretrained_model_name_or_path"
        ]
        model_variant = model_variant.split("/")[-1]
        # train on denoised gt, or noisier pupil_raw_imputed
        train_on = cfg["OUTLIER_MODELS"][model_name]["MODEL"]["train_on"]
        if train_on != "gt":
            if train_on == "pupil_raw_imputed":
                # shorter name
                suffix = "_raw"
            else:
                suffix = "_" + train_on
        else:
            suffix = ""

        run_name = f"{model_name}_{detection_type}_{model_variant}{suffix}"
    elif model_name == "NuwaTS":
        detection_type = cfg["OUTLIER_MODELS"][model_name]["MODEL"][
            "detection_type"
        ].replace("-", "")
        run_name = f"{model_name}_{detection_type}"
    else:
        logger.warning("No fancy run name for the model = {}".format(model_name))
        run_name = model_name
        logger.warning("Using the model name as the run name: {}".format(run_name))

    return run_name


def update_imputation_run_name(cfg: DictConfig):
    if len(cfg["MODELS"].keys()) == 1:
        model_name = list(cfg["MODELS"].keys())[0]
    else:
        logger.error("Only one model should be used for outlier detection")
        raise ValueError("Only one model should be used for outlier detection")
    if model_name == "MOMENT":
        # finetune or zeroshot
        detection_type = cfg["MODELS"][model_name]["MODEL"]["detection_type"].replace(
            "-", ""
        )
        # large, base, or small
        model_variant = cfg["MODELS"][model_name]["MODEL"][
            "pretrained_model_name_or_path"
        ]
        model_variant = model_variant.split("/")[-1]
        # train on denoised gt, or noisier pupil_raw_imputed
        train_on = cfg["MODELS"][model_name]["MODEL"]["train_on"]
        if train_on != "gt":
            if train_on == "pupil_raw_imputed":
                # shorter name
                suffix = "_raw"
            else:
                suffix = "_" + train_on
        else:
            suffix = ""

        run_name = f"{model_name}_{detection_type}_{model_variant}{suffix}"
    else:
        logger.warning("No fancy run name for the model = {}".format(model_name))
        run_name = model_name
        logger.warning("Using the model name as the run name: {}".format(run_name))

    return run_name


def get_torch_model_name(run_name: str):
    # e.g. MOMENT_finetune_MOMENT-1-large_pupil_gt_model.pth
    return f"{run_name}_model.pth"


def get_debug_string_to_add() -> str:
    return "__DEBUG_"


def get_demo_string_to_add() -> str:
    return "__DEMODATA_"

def if_runname_is_debug(run_name: str) -> bool:
    return get_debug_string_to_add() in run_name


def experiment_name_wrapper(experiment_name: str, cfg: DictConfig) -> str:
    if cfg["EXPERIMENT"]["use_demo_data"]:
        experiment_name = get_demo_string_to_add() + experiment_name
    if cfg["EXPERIMENT"]["debug"]:
        experiment_name = get_debug_string_to_add() + experiment_name
    return experiment_name


def get_outlier_detection_experiment_name(cfg):
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["OUTLIER_DETECTION"], cfg=cfg
    )
    return experiment_name


def get_model_name_from_run_name(run_name: str, task: str) -> str:
    model_name = run_name.split("_")[0]
    if "MOMENT" in run_name:
        model_key = (
            run_name.replace("MOMENT-1", "")
            .replace("-large", "")
            .replace("-base", "")
            .replace("-small", "")
            .replace("pupil", "")
        )
    else:
        model_key = model_name
    return model_name, model_key


def get_foundation_model_names():
    return ["MOMENT", "UniTS"]


def get_simple_outlier_detectors():
    return ["LOF", "OneClassSVM", "PROPHET"]


def get_eval_metric_name(cls_model_name: str, cfg: DictConfig):
    hparam_cfg = cfg["CLS_HYPERPARAMS"][cls_model_name]
    if "metric_val" in hparam_cfg["HYPERPARAMS"]:
        # XGBoost, CatBoost, TabM
        eval_metric = hparam_cfg["HYPERPARAMS"]["metric_val"]
    elif "fit_params" in hparam_cfg["HYPERPARAMS"]:
        # Logistic regression
        eval_metric = hparam_cfg["HYPERPARAMS"]["fit_params"]["scoring"]
    else:
        logger.error("Where is your eval_metric defined? ({})".format(cls_model_name))
        raise ValueError(
            "Where is your eval_metric defined? ({})".format(cls_model_name)
        )
    return eval_metric


def get_train_loss_name(cfg: DictConfig):
    return cfg["CLASSIFICATION_SETTINGS"]["loss"]


def update_cls_run_name(
    cls_model_name: str,
    source_name: str,
    model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
) -> str:
    # train_loss = get_train_loss_name(cfg)
    eval_metric = get_eval_metric_name(cls_model_name, cfg)
    return f"{cls_model_name}_eval-{eval_metric}__{source_name}"


def get_embedding_npy_fname(model_name, split):
    return f"{model_name}_embedding_{split}.npy"


def get_moment_cls_run_name(cls_model_name, cls_model_cfg):
    model_variant = (
        cls_model_cfg["MODEL"]["pretrained_model_name_or_path"]
        .split("/")[-1]
        .split("-")[-1]
    )
    detection_type = cls_model_cfg["MODEL"]["detection_type"]
    weighing_string = "_w" if cls_model_cfg["MODEL"]["use_weighed_loss"] else ""
    cls_run_name = f"{cls_model_name}-{model_variant}_{detection_type}{weighing_string}"
    return cls_run_name


def get_imputation_pickle_name(model_name: str):
    return f"imputation_{model_name}.pickle"


def get_summary_fname(experiment_name: str):
    return f'summary_{experiment_name.replace("PLR_", "")}.db'


def get_summary_fpath(experiment_name: str):
    dir_out = get_artifacts_dir("dataframes")
    db_fname = get_summary_fname(experiment_name)
    db_path = os.path.join(dir_out, db_fname)
    if os.path.exists(db_path):
        os.remove(db_path)
    return db_path


def get_summary_artifacts_fname(experiment_name: str):
    return f'summary_artifacts_{experiment_name.replace("PLR_", "")}.pickle'


def get_summary_artifacts_fpath(experiment_name: str):
    dir_out = get_artifacts_dir("artifacts")
    fname = get_summary_artifacts_fname(experiment_name)
    fpath = os.path.join(dir_out, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
    return fpath


def parse_task_from_exp_name(experiment_name):
    # You could as well use the cfg hard-coded names?
    if "OutlierDetection" in experiment_name:
        task = "outlier_detection"
    elif "Imputation" in experiment_name:
        task = "imputation"
    elif "Classification" in experiment_name:
        task = "classification"
    elif "Featurization" in experiment_name:
        task = "featurization"
    return task
