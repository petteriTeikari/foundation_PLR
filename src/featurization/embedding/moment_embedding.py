from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import polars as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data_io.data_wrangler import (
    convert_object_type,
    convert_subject_dict_of_arrays_to_df,
    fix_pl_schema,
)
from src.featurization.feature_log import featurization_mlflow_metrics_and_params
from src.imputation.momentfm.moment_utils import (
    import_moment_from_mlflow,
    init_torch_training,
)
from src.log_helpers.local_artifacts import save_results_dict
from src.log_helpers.log_naming_uris_and_dirs import (
    get_embedding_npy_fname,
    get_features_pickle_fname,
)
from src.utils import get_artifacts_dir


def log_embeddings_to_mlflow(
    embeddings: dict[str, Any],
    run_name: str,
    model_name: str,
    source_name: str,
    save_as_numpy: bool = True,
) -> None:
    """Save embeddings to disk and log as MLflow artifacts.

    Parameters
    ----------
    embeddings : dict
        Dictionary with 'data' containing DataFrames per split.
    run_name : str
        Run name for file naming.
    model_name : str
        Model name for file naming.
    source_name : str
        Source name (currently unused).
    save_as_numpy : bool, optional
        If True, save per-split numpy arrays, by default True.
    """
    dir_out = Path(get_artifacts_dir("embeddings"))

    # as numpy arrays
    if save_as_numpy:
        for split, df in embeddings["data"].items():
            embedding_fname = get_embedding_npy_fname(model_name, split)
            path_out = dir_out / embedding_fname
            if path_out.exists():
                path_out.unlink()
            np.save(str(path_out), df.to_numpy())  # e.g. (16, 1024)
            mlflow.log_artifact(str(path_out), "embeddings")

    # as single pickled thing
    # e.g. MOMENT-embedding__pupil-gt__pupil-gt.pickle
    path_out = dir_out / get_features_pickle_fname(run_name)
    save_results_dict(embeddings, str(path_out))
    mlflow.log_artifact(str(path_out), "embeddings")


def get_dataframe_from_dict(
    split_dict_subject: dict[str, dict[str, np.ndarray]],
    cfg: DictConfig,
    drop_col_wildcard: str = "mask",
) -> pl.DataFrame:
    """Convert subject dictionary to metadata DataFrame.

    Parameters
    ----------
    split_dict_subject : dict
        Subject dictionary with metadata and labels.
    cfg : DictConfig
        Configuration dictionary (currently unused).
    drop_col_wildcard : str, optional
        Wildcard for columns to drop, by default 'mask'.

    Returns
    -------
    pl.DataFrame
        DataFrame with metadata columns prefixed with 'metadata_'.
    """
    df = convert_subject_dict_of_arrays_to_df(
        split_dict_subject, wildcard_categories=["metadata", "labels"]
    )
    # drop the mask columns (that do not make any sense as a single row)
    drop_cols = [i for i in df.columns if drop_col_wildcard in i]
    df = df.drop(drop_cols)
    df = df.rename(lambda column_name: "metadata_" + column_name)
    return df


def create_pseudo_embedding_std(
    embeddings_out: np.ndarray | pl.DataFrame,
) -> pl.DataFrame:
    """Create placeholder standard deviation columns for embeddings.

    Creates zero-filled columns to match handcrafted feature format.

    Parameters
    ----------
    embeddings_out : np.ndarray or pl.DataFrame
        Embedding array with shape (n_samples, n_features).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns 'embedding{i}_std' filled with zeros.
    """
    df_stdev = pl.DataFrame(
        np.zeros_like(embeddings_out),
        schema=[f"embedding{i}_std" for i in range(embeddings_out.shape[1])],
    )
    return df_stdev


def create_embeddings_df(embeddings_out: np.ndarray) -> pl.DataFrame:
    """Create DataFrame from embedding array.

    Parameters
    ----------
    embeddings_out : np.ndarray
        Embedding array with shape (n_samples, n_features).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns 'embedding{i}_value'.
    """
    df_embeddings = pl.DataFrame(
        embeddings_out,
        schema=[f"embedding{i}_value" for i in range(embeddings_out.shape[1])],
    )
    return df_embeddings


def create_split_embedding_df(
    embeddings_out: np.ndarray,
    subject_codes: list[str],
    df_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Create complete embedding DataFrame with metadata and codes.

    Parameters
    ----------
    embeddings_out : np.ndarray
        Embedding array with shape (n_samples, n_features).
    subject_codes : list
        List of subject identifiers.
    df_metadata : pl.DataFrame
        Metadata DataFrame for subjects.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with subject_code, embeddings, std, and metadata.
    """
    df_codes = pl.DataFrame(subject_codes, schema=["subject_code"])
    df_embeddings = create_embeddings_df(embeddings_out)
    # we don't have no stdev for the embeddings, but to make downstream code manage with less exceptions, let's add them
    df_stdev = create_pseudo_embedding_std(df_embeddings)

    df = pl.concat([df_codes, df_embeddings, df_stdev, df_metadata], how="horizontal")

    return df


def get_subject_dict_for_df(
    embeddings_out: np.ndarray,
    split_dict: dict[str, dict[str, np.ndarray]],
    cfg: DictConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Prepare subject dictionary for DataFrame conversion.

    Extracts first timepoint from arrays for scalar metadata.

    Parameters
    ----------
    embeddings_out : np.ndarray
        Embedding array for validation of subject count.
    split_dict : dict
        Split dictionary with metadata arrays.
    cfg : DictConfig
        Configuration dictionary (currently unused).

    Returns
    -------
    dict
        Subject dictionary with scalar values per category.

    Raises
    ------
    AssertionError
        If embedding and input subject counts don't match.
    """
    no_of_embedding_subjects = embeddings_out.shape[0]
    no_of_input_subjects = split_dict["metadata"]["subject_code"].shape[0]
    assert no_of_input_subjects == no_of_embedding_subjects
    split_dict_subject = deepcopy(split_dict)
    for category in split_dict:
        for variable in split_dict[category]:
            array = split_dict[category][variable][:, 0]
            array = convert_object_type(array)
            split_dict_subject[category][variable] = array
    return split_dict_subject


def get_subject_codes(
    split_dict: dict[str, dict[str, np.ndarray]],
) -> list[str]:
    """Extract subject codes from split dictionary.

    Parameters
    ----------
    split_dict : dict
        Split dictionary with metadata containing subject_code array.

    Returns
    -------
    list
        List of subject code strings.
    """
    subject_codes = []
    for i in range(split_dict["metadata"]["subject_code"].shape[0]):
        subject_codes.append(split_dict["metadata"]["subject_code"][i, 0])
    return subject_codes


def combine_embeddings_with_metadata_for_df(
    embeddings_out: np.ndarray,
    split_dict: dict[str, dict[str, np.ndarray]],
    cfg: DictConfig,
) -> pl.DataFrame:
    """Combine embedding array with subject metadata into DataFrame.

    Parameters
    ----------
    embeddings_out : np.ndarray
        Embedding array with shape (n_samples, n_features).
    split_dict : dict
        Split dictionary with metadata.
    cfg : DictConfig
        Configuration dictionary.

    Returns
    -------
    pl.DataFrame
        Combined DataFrame with embeddings and metadata.

    See Also
    --------
    compute_features_from_dict : Similar function for handcrafted features.
    """
    subject_codes = get_subject_codes(split_dict)
    split_dict_subject = get_subject_dict_for_df(embeddings_out, split_dict, cfg)
    df_metadata = get_dataframe_from_dict(split_dict_subject, cfg)
    df_metadata = fix_pl_schema(df_metadata)

    # Combine 4 "sub-dataframes" into one
    df = create_split_embedding_df(embeddings_out, subject_codes, df_metadata)

    return df


def get_embeddings_per_split(
    model: torch.nn.Module,
    dataloader: DataLoader,
    split_dict: dict[str, dict[str, np.ndarray]],
    model_cfg: DictConfig,
    cfg: DictConfig,
) -> pl.DataFrame:
    """Compute embeddings for all batches in a data split.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model for embedding extraction.
    dataloader : DataLoader
        PyTorch dataloader for the split.
    split_dict : dict
        Split dictionary with metadata.
    model_cfg : DictConfig
        Model configuration.
    cfg : DictConfig
        Main configuration with DEVICE settings.

    Returns
    -------
    pl.DataFrame
        DataFrame with embeddings and metadata.

    Raises
    ------
    AssertionError
        If embeddings are None (model initialization issue).
    """
    embeddings_out = None
    for i, (batch_x, labels, input_masks) in enumerate(dataloader):
        x = batch_x.unsqueeze(1)
        # x = torch.randn(16, 1, 1981)
        outputs = model(x_enc=x.to(cfg["DEVICE"]["device"]))
        embeddings = outputs.embeddings.detach().cpu().numpy()
        assert (
            embeddings is not None
        ), "Embeddings are None, problem with initalizing the model?"
        if embeddings_out is None:
            embeddings_out = embeddings
        else:
            embeddings_out = np.concatenate((embeddings_out, embeddings), axis=0)

    # combine metadata with embeddings, and return a Polars dataframe so that this matches the handcrafed features
    # and you can do downstream classification more easily
    embeddings_df = combine_embeddings_with_metadata_for_df(
        embeddings_out, split_dict, cfg
    )

    return embeddings_df


def get_embeddings(
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    source_data: dict[str, Any],
    model_cfg: DictConfig,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Compute embeddings for all data splits.

    Parameters
    ----------
    model : torch.nn.Module
        MOMENT model for embedding extraction.
    dataloaders : dict
        Dictionary of dataloaders keyed by split name.
    source_data : dict
        Source data with 'df' and 'mlflow' keys.
    model_cfg : DictConfig
        Model configuration.
    cfg : DictConfig
        Main configuration.

    Returns
    -------
    dict
        Dictionary with 'data' (embeddings per split) and 'mlflow_run'.
    """
    embeddings = {}
    for split, dataloader in dataloaders.items():
        logger.info(f"Computing embeddings for split {split}")
        embeddings[split] = get_embeddings_per_split(
            model,
            dataloader,
            split_dict=source_data["df"][split],
            model_cfg=model_cfg,
            cfg=cfg,
        )

    features = {"data": embeddings, "mlflow_run": source_data["mlflow"]}

    return features


def import_moment_embedder(cfg: DictConfig, model_cfg: DictConfig) -> torch.nn.Module:
    """Import MOMENT model configured for embedding extraction.

    Parameters
    ----------
    cfg : DictConfig
        Main configuration with DEVICE settings.
    model_cfg : DictConfig
        Model configuration for MOMENT.

    Returns
    -------
    torch.nn.Module
        MOMENT model ready for embedding extraction.

    See Also
    --------
    https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb
    """
    model_kwargs = {"task_name": "embedding"}
    # This is the same load as with imputation thus the task name a bit confusing
    model = import_moment_from_mlflow(
        model_cfg=model_cfg, cfg=cfg, task="embedding", model_kwargs=model_kwargs
    )
    model = model.to(cfg["DEVICE"]["device"]).float()
    return model


def moment_embedder(
    source_data: dict[str, Any],
    source_name: str,
    model_cfg: DictConfig,
    cfg: DictConfig,
    run_name: str,
    # artifacts_dir: str,
    model_name: str,
    pre_embedding_cfg: Optional[DictConfig],
    # experiment_name: str
) -> None:
    """Extract MOMENT embeddings for a data source with MLflow tracking.

    Imports model, computes embeddings, optionally applies PCA, and logs
    results to MLflow.

    Parameters
    ----------
    source_data : dict
        Source data dictionary with 'df' and 'mlflow' keys.
    source_name : str
        Name of the data source.
    model_cfg : DictConfig
        MOMENT model configuration.
    cfg : DictConfig
        Main configuration dictionary.
    run_name : str
        MLflow run name.
    model_name : str
        Model name for logging.
    pre_embedding_cfg : DictConfig or None
        Pre-embedding configuration (e.g., PCA).

    Raises
    ------
    ValueError
        If pre_embedding_cfg has unknown preprocessing method.
    """
    # Get the model
    model = import_moment_embedder(cfg, model_cfg)

    # init stuff
    dataloaders = init_torch_training(
        data_dict=source_data,
        cfg=cfg,
        model_cfg=model_cfg,
        run_name=run_name,
        task="imputation",
        create_outlier_dataloaders=False,
    )

    with mlflow.start_run(run_name=run_name):
        # Log params and metrics to MLflow
        featurization_mlflow_metrics_and_params(
            mlflow_run=source_data["mlflow"], source_name=source_name, cfg=cfg
        )

        mlflow.log_param("model_name", model_name)
        mlflow.log_param(
            "pretrained_model_name_or_path",
            model_cfg["MODEL"]["pretrained_model_name_or_path"],
        )

        # Get the embeddings
        embeddings = get_embeddings(model, dataloaders, source_data, model_cfg, cfg)

        # Possible preprocessing (well preprocessing for classifier, post-processing for embedding)
        if pre_embedding_cfg is not None:
            if len(pre_embedding_cfg.keys()) == 1:
                if list(pre_embedding_cfg.keys())[0] == "PCA":
                    from src.featurization.embedding.dim_reduction import (
                        apply_PCA_for_embedding,
                    )

                    embeddings = apply_PCA_for_embedding(
                        embeddings, pca_config=pre_embedding_cfg["PCA"]
                    )
                else:
                    logger.error(
                        "Unknown preprocessing method = {}".format(
                            list(pre_embedding_cfg.keys())[0]
                        )
                    )
                    raise ValueError(
                        "Unknown preprocessing method {}".format(
                            list(pre_embedding_cfg.keys())[0]
                        )
                    )
            else:
                logger.error(
                    "Only one key allowed here, pre_embedding_cfg = {}".format(
                        pre_embedding_cfg
                    )
                )
                raise ValueError(
                    "Only one key allowed here, pre_embedding_cfg = {}".format(
                        pre_embedding_cfg
                    )
                )

        # Log to Mlflow as an artifact
        log_embeddings_to_mlflow(embeddings, run_name, model_name, source_name)
        mlflow.end_run()
