from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger
from omegaconf import DictConfig

from src.utils import get_artifacts_dir, get_data_dir


def get_feature_pickle_artifact_uri(
    run: Dict[str, Any], source: str, cfg: DictConfig, subdir: str = "features"
) -> str:
    """Construct MLflow artifact URI for feature pickle files.

    Parameters
    ----------
    run : dict
        MLflow run dictionary containing 'run_id'.
    source : str
        Data source name used for filename generation.
    cfg : DictConfig
        Configuration object (currently unused but kept for API consistency).
    subdir : str, default "features"
        Subdirectory within the MLflow artifact store.

    Returns
    -------
    str
        MLflow artifact URI in format 'runs:/{run_id}/{subdir}/{filename}'.
    """
    return f"runs:/{run['run_id']}/{subdir}/{get_feature_pickle_base(source)}"


def get_feature_pickle_base(run_name: str) -> str:
    """Generate base filename for feature pickle files.

    Parameters
    ----------
    run_name : str
        Name of the run to use as the base filename.

    Returns
    -------
    str
        Filename with .pickle extension.
    """
    return f"{run_name}.pickle"


def get_features_pickle_fname(data_source: str) -> str:
    """Generate pickle filename for feature data.

    Parameters
    ----------
    data_source : str
        Name of the data source.

    Returns
    -------
    str
        Filename with .pickle extension.
    """
    return get_feature_pickle_base(data_source)


def get_baseline_names() -> List[str]:
    """Get list of baseline method names for PLR preprocessing.

    Returns
    -------
    list of str
        Baseline method names: denoised ground truth and outlier-removed raw.
    """
    return ["BASELINE_DenoisedGT", "BASELINE_OutlierRemovedRaw"]


def get_feature_name_from_cfg(cfg: DictConfig) -> str:
    """Extract feature name and version from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing PLR_FEATURIZATION.FEATURES_METADATA with
        'name' and 'version' keys.

    Returns
    -------
    str
        Combined feature name and version string.
    """
    return (
        f"{cfg['PLR_FEATURIZATION']['FEATURES_METADATA']['name']}"
        f"{cfg['PLR_FEATURIZATION']['FEATURES_METADATA']['version']}"
    )


def define_featurization_run_name_from_base(base_name: str, cfg: DictConfig) -> str:
    """Construct featurization run name from base name and configuration.

    Parameters
    ----------
    base_name : str
        Base name to append to the run name.
    cfg : DictConfig
        Configuration containing feature metadata.

    Returns
    -------
    str
        Run name in format 'features-{feature_name}{version}_{base_name}'.
    """
    return f"features-{get_feature_name_from_cfg(cfg)}_{base_name}"


def xgboost_variant_run_name(
    run_name: str, xgboost_cfg: DictConfig, model_name: str = "XGBOOST"
) -> str:
    """Modify run name to include XGBoost variant suffix.

    Parameters
    ----------
    run_name : str
        Original run name containing the model name.
    xgboost_cfg : DictConfig
        XGBoost configuration containing 'variant_name'.
    model_name : str, default "XGBOOST"
        Model name string to find and replace in run_name.

    Returns
    -------
    str
        Modified run name with variant suffix, or original if no variant.
    """
    variant_name = xgboost_cfg["variant_name"]
    if len(variant_name) > 0:
        return run_name.replace(model_name, f"{model_name}_{variant_name}")
    else:
        return run_name


def get_pypots_model_path(results_path: str, ext_out: str = ".pypots") -> str:
    """Convert results path to PyPOTS model path.

    Parameters
    ----------
    results_path : str
        Path to results file.
    ext_out : str, default ".pypots"
        Extension for the output model file.

    Returns
    -------
    str
        Path to PyPOTS model file with 'results' replaced by 'model'.
    """
    results_path = Path(results_path)
    fname = results_path.stem.replace("results", "model")
    return str(results_path.parent / (fname + ext_out))


def get_mlflow_metric_name(split: str, metric_key: str) -> str:
    """Construct MLflow metric name from split and metric key.

    Parameters
    ----------
    split : str
        Data split name (e.g., 'train', 'test', 'val').
    metric_key : str
        Metric identifier (e.g., 'auroc', 'mae').

    Returns
    -------
    str
        MLflow metric name in format '{split}/{metric_key}'.
    """
    return f"{split}/{metric_key}"


def get_outlier_pickle_name(model_name: str) -> str:
    """Generate pickle filename for outlier detection results.

    Parameters
    ----------
    model_name : str
        Name of the outlier detection model.

    Returns
    -------
    str
        Filename in format 'outlierDetection_{model_name}.pickle'.
    """
    return f"outlierDetection_{model_name}.pickle"


def get_outlier_csv_name(model_name: str, split: str, key: str) -> str:
    """Generate CSV filename for outlier detection data export.

    Parameters
    ----------
    model_name : str
        Name of the outlier detection model.
    split : str
        Data split name (e.g., 'train', 'test').
    key : str
        Data key identifier.

    Returns
    -------
    str
        Filename in format 'outlierDetection_{model_name}_{split}_{key}.csv'.
    """
    base_fname = get_outlier_pickle_name(model_name).replace(".pickle", "")
    return f"{base_fname}_{split}_{key}.csv"


def get_duckdb_file(
    data_cfg: DictConfig,
    use_demo_data: bool = False,
    demo_db_file: str = "PLR_demo_data.db",
    use_synthetic_data: bool = False,
) -> str:
    """Get path to DuckDB database file.

    Parameters
    ----------
    data_cfg : DictConfig
        Data configuration containing 'data_path' and 'filename_DuckDB'.
    use_demo_data : bool, default False
        If True, use demo database for testing.
    demo_db_file : str, default 'PLR_demo_data.db'
        Filename of demo database.
    use_synthetic_data : bool, default False
        If True, use synthetic database (SYNTH_PLR_DEMO.db) for CI/testing.
        This takes precedence over use_demo_data.

    Returns
    -------
    str
        Absolute path to the DuckDB file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    # Check for synthetic data (highest priority - for CI/testing)
    if use_synthetic_data:
        from src.utils.paths import get_synthetic_db_path

        db_path = get_synthetic_db_path()
        logger.info(f"Using SYNTHETIC data for testing: {db_path}")
        if not db_path.is_file():
            logger.error(f"Synthetic database not found: {db_path}")
            logger.error("Run: python -m src.synthetic.demo_dataset to generate it")
            raise FileNotFoundError(str(db_path))
        return str(db_path)

    # Check for demo data
    if use_demo_data:
        data_dir = get_data_dir(data_path=data_cfg["data_path"])
        logger.warning(f"Using the demo data ({demo_db_file}) for testing the pipeline")
        db_path = data_dir / demo_db_file
        if not db_path.is_file():
            logger.error(f"File {db_path} does not exist")
            raise FileNotFoundError(str(db_path))
        return str(db_path)

    # Default: use configured database
    # Check if it's a synthetic path (data/synthetic/...)
    if "synthetic" in data_cfg.get("data_path", ""):
        from src.utils.paths import PROJECT_ROOT

        db_path = PROJECT_ROOT / data_cfg["data_path"] / data_cfg["filename_DuckDB"]
        logger.info(f"Using synthetic database: {db_path}")
    else:
        data_dir = get_data_dir(data_path=data_cfg["data_path"])
        db_path = data_dir / data_cfg["filename_DuckDB"]

    if not db_path.is_file():
        logger.error(f"File {db_path} does not exist")
        raise FileNotFoundError(str(db_path))

    return str(db_path)


def update_outlier_detection_run_name(cfg: DictConfig) -> str:
    """Generate descriptive run name for outlier detection based on configuration.

    Creates a run name that encodes the model type, detection method, variant,
    and training data source. For MOMENT models, includes finetune/zeroshot mode,
    model size (large/base/small), and training data type.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing OUTLIER_MODELS with model-specific settings.

    Returns
    -------
    str
        Descriptive run name encoding model configuration.

    Raises
    ------
    ValueError
        If more than one model is specified in OUTLIER_MODELS.
    """
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


def update_imputation_run_name(cfg: DictConfig) -> str:
    """Generate descriptive run name for imputation based on configuration.

    Creates a run name that encodes the model type, detection method, variant,
    and training data source. For MOMENT models, includes finetune/zeroshot mode,
    model size (large/base/small), and training data type.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing MODELS with model-specific settings.

    Returns
    -------
    str
        Descriptive run name encoding model configuration.

    Raises
    ------
    ValueError
        If more than one model is specified in MODELS.
    """
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


def get_torch_model_name(run_name: str) -> str:
    """Generate PyTorch model filename from run name.

    Parameters
    ----------
    run_name : str
        Name of the training run.

    Returns
    -------
    str
        Model filename with .pth extension (e.g., 'MOMENT_finetune_large_model.pth').
    """
    # e.g. MOMENT_finetune_MOMENT-1-large_pupil_gt_model.pth
    return f"{run_name}_model.pth"


def get_debug_string_to_add() -> str:
    """Get prefix string for debug experiment names.

    Returns
    -------
    str
        Debug prefix '__DEBUG_'.
    """
    return "__DEBUG_"


def get_demo_string_to_add() -> str:
    """Get prefix string for demo data experiment names.

    Returns
    -------
    str
        Demo data prefix '__DEMODATA_'.
    """
    return "__DEMODATA_"


def get_synthetic_string_to_add() -> str:
    """Get prefix string for synthetic data experiment names.

    Part of the 4-gate isolation architecture. See src/utils/data_mode.py.

    Returns
    -------
    str
        Synthetic data prefix 'synth_'.
    """
    from src.utils.data_mode import SYNTHETIC_EXPERIMENT_PREFIX

    return SYNTHETIC_EXPERIMENT_PREFIX


def if_runname_is_debug(run_name: str) -> bool:
    """Check if run name indicates a debug run.

    Parameters
    ----------
    run_name : str
        Name of the run to check.

    Returns
    -------
    bool
        True if run name contains the debug prefix.
    """
    return get_debug_string_to_add() in run_name


def experiment_name_wrapper(experiment_name: str, cfg: DictConfig) -> str:
    """Add prefixes to experiment name based on configuration flags.

    Prepends demo data, debug, and/or synthetic prefixes to the experiment name
    if the corresponding configuration flags are set.

    Part of the 4-gate isolation architecture. See src/utils/data_mode.py.

    Priority order (applied in reverse so first prefix appears first):
    1. synthetic (synth_) - from EXPERIMENT.is_synthetic or data_mode detection
    2. demo data (__DEMODATA_) - from EXPERIMENT.use_demo_data
    3. debug (__DEBUG_) - from EXPERIMENT.debug

    Parameters
    ----------
    experiment_name : str
        Base experiment name.
    cfg : DictConfig
        Configuration with EXPERIMENT.use_demo_data, EXPERIMENT.debug,
        and EXPERIMENT.is_synthetic flags.

    Returns
    -------
    str
        Experiment name with appropriate prefixes.
    """
    from src.utils.data_mode import is_synthetic_from_config

    if cfg["EXPERIMENT"]["use_demo_data"]:
        experiment_name = get_demo_string_to_add() + experiment_name
    if cfg["EXPERIMENT"]["debug"]:
        experiment_name = get_debug_string_to_add() + experiment_name

    # Add synthetic prefix if detected from config
    # This includes EXPERIMENT.is_synthetic=true, experiment_prefix="synth_",
    # or DATA.data_path contains "synthetic"
    if is_synthetic_from_config(cfg):
        experiment_name = get_synthetic_string_to_add() + experiment_name

    return experiment_name


def get_outlier_detection_experiment_name(cfg: DictConfig) -> str:
    """Get experiment name for outlier detection from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing PREFECT.FLOW_NAMES.OUTLIER_DETECTION.

    Returns
    -------
    str
        Experiment name with appropriate prefixes applied.
    """
    experiment_name = experiment_name_wrapper(
        experiment_name=cfg["PREFECT"]["FLOW_NAMES"]["OUTLIER_DETECTION"], cfg=cfg
    )
    return experiment_name


def get_model_name_from_run_name(run_name: str, task: str) -> Tuple[str, str]:
    """Extract model name and key from run name.

    For MOMENT models, strips version and size information to create a
    normalized key. For other models, the key equals the model name.

    Parameters
    ----------
    run_name : str
        Full run name containing model information.
    task : str
        Task type (currently unused, reserved for future use).

    Returns
    -------
    tuple of str
        Tuple of (model_name, model_key) where model_key is normalized.
    """
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


def get_foundation_model_names() -> List[str]:
    """Get list of supported foundation model names.

    Returns
    -------
    list of str
        Names of foundation models: MOMENT and UniTS.
    """
    return ["MOMENT", "UniTS"]


def get_simple_outlier_detectors() -> List[str]:
    """Get list of traditional outlier detection method names.

    Returns
    -------
    list of str
        Names of simple outlier detectors: LOF, OneClassSVM, PROPHET.
    """
    return ["LOF", "OneClassSVM", "PROPHET"]


def get_eval_metric_name(cls_model_name: str, cfg: DictConfig) -> str:
    """Extract evaluation metric name from classifier configuration.

    Looks for metric_val in HYPERPARAMS (XGBoost, CatBoost, TabM) or
    fit_params.scoring (Logistic Regression).

    Parameters
    ----------
    cls_model_name : str
        Name of the classifier model.
    cfg : DictConfig
        Configuration containing CLS_HYPERPARAMS for the model.

    Returns
    -------
    str
        Name of the evaluation metric.

    Raises
    ------
    ValueError
        If eval_metric cannot be found in the configuration.
    """
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


def get_train_loss_name(cfg: DictConfig) -> str:
    """Get training loss function name from configuration.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing CLASSIFICATION_SETTINGS.loss.

    Returns
    -------
    str
        Name of the loss function.
    """
    return cfg["CLASSIFICATION_SETTINGS"]["loss"]


def update_cls_run_name(
    cls_model_name: str,
    source_name: str,
    model_cfg: DictConfig,
    hparam_cfg: DictConfig,
    cfg: DictConfig,
) -> str:
    """Construct classification run name from model and source information.

    Parameters
    ----------
    cls_model_name : str
        Name of the classifier model.
    source_name : str
        Name of the data source/preprocessing pipeline.
    model_cfg : DictConfig
        Model configuration (currently unused).
    hparam_cfg : DictConfig
        Hyperparameter configuration (currently unused).
    cfg : DictConfig
        Full configuration for extracting eval metric.

    Returns
    -------
    str
        Run name in format '{model}_eval-{metric}__{source}'.
    """
    # train_loss = get_train_loss_name(cfg)
    eval_metric = get_eval_metric_name(cls_model_name, cfg)
    return f"{cls_model_name}_eval-{eval_metric}__{source_name}"


def get_embedding_npy_fname(model_name: str, split: str) -> str:
    """Generate filename for embedding numpy array.

    Parameters
    ----------
    model_name : str
        Name of the model that generated embeddings.
    split : str
        Data split name (e.g., 'train', 'test').

    Returns
    -------
    str
        Filename in format '{model_name}_embedding_{split}.npy'.
    """
    return f"{model_name}_embedding_{split}.npy"


def get_moment_cls_run_name(cls_model_name: str, cls_model_cfg: DictConfig) -> str:
    """Generate classification run name for MOMENT model.

    Encodes model variant, detection type, and loss weighting in the name.

    Parameters
    ----------
    cls_model_name : str
        Base classifier model name.
    cls_model_cfg : DictConfig
        MOMENT model configuration with MODEL settings.

    Returns
    -------
    str
        Run name in format '{model}-{variant}_{detection_type}[_w]'.
    """
    model_variant = (
        cls_model_cfg["MODEL"]["pretrained_model_name_or_path"]
        .split("/")[-1]
        .split("-")[-1]
    )
    detection_type = cls_model_cfg["MODEL"]["detection_type"]
    weighing_string = "_w" if cls_model_cfg["MODEL"]["use_weighed_loss"] else ""
    cls_run_name = f"{cls_model_name}-{model_variant}_{detection_type}{weighing_string}"
    return cls_run_name


def get_imputation_pickle_name(model_name: str) -> str:
    """Generate pickle filename for imputation results.

    Parameters
    ----------
    model_name : str
        Name of the imputation model.

    Returns
    -------
    str
        Filename in format 'imputation_{model_name}.pickle'.
    """
    return f"imputation_{model_name}.pickle"


def get_summary_fname(experiment_name: str) -> str:
    """Generate summary database filename from experiment name.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    str
        Filename with 'PLR_' prefix removed and .db extension.
    """
    return f"summary_{experiment_name.replace('PLR_', '')}.db"


def get_summary_fpath(experiment_name: str) -> str:
    """Get full path for summary database, removing existing file if present.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    str
        Full path to summary database file.

    Notes
    -----
    Deletes existing file at the path before returning.
    """
    dir_out = get_artifacts_dir("dataframes")
    db_fname = get_summary_fname(experiment_name)
    db_path = dir_out / db_fname
    if db_path.exists():
        db_path.unlink()
    return str(db_path)


def get_summary_artifacts_fname(experiment_name: str) -> str:
    """Generate summary artifacts pickle filename from experiment name.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    str
        Filename with 'PLR_' prefix removed and .pickle extension.
    """
    return f"summary_artifacts_{experiment_name.replace('PLR_', '')}.pickle"


def get_summary_artifacts_fpath(experiment_name: str) -> str:
    """Get full path for summary artifacts pickle, removing existing file if present.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    Returns
    -------
    str
        Full path to summary artifacts pickle file.

    Notes
    -----
    Deletes existing file at the path before returning.
    """
    dir_out = get_artifacts_dir("artifacts")
    fname = get_summary_artifacts_fname(experiment_name)
    fpath = dir_out / fname
    if fpath.exists():
        fpath.unlink()
    return str(fpath)


def parse_task_from_exp_name(experiment_name: str) -> str:
    """Parse task type from experiment name string.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment containing task identifier.

    Returns
    -------
    str
        Task type: 'outlier_detection', 'imputation', 'classification',
        or 'featurization'.
    """
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
