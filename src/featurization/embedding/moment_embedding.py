import os
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig
import polars as pl
import mlflow
from loguru import logger

from src.data_io.data_wrangler import (
    convert_subject_dict_of_arrays_to_df,
    convert_object_type,
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
    embeddings, run_name, model_name, source_name, save_as_numpy: bool = True
):
    dir_out = get_artifacts_dir("embeddings")

    # as numpy arrays
    if save_as_numpy:
        for split, df in embeddings["data"].items():
            embedding_fname = get_embedding_npy_fname(model_name, split)
            path_out = os.path.join(dir_out, embedding_fname)
            if os.path.exists(path_out):
                os.remove(path_out)
            np.save(path_out, df.to_numpy())  # e.g. (16, 1024)
            mlflow.log_artifact(path_out, "embeddings")

    # as single pickled thing
    # e.g. MOMENT-embedding__pupil-gt__pupil-gt.pickle
    path_out = os.path.join(dir_out, get_features_pickle_fname(run_name))
    save_results_dict(embeddings, path_out)
    mlflow.log_artifact(path_out, "embeddings")


def get_dataframe_from_dict(split_dict_subject, cfg, drop_col_wildcard: str = "mask"):
    df = convert_subject_dict_of_arrays_to_df(
        split_dict_subject, wildcard_categories=["metadata", "labels"]
    )
    # drop the mask columns (that do not make any sense as a single row)
    drop_cols = [i for i in df.columns if drop_col_wildcard in i]
    df = df.drop(drop_cols)
    df = df.rename(lambda column_name: "metadata_" + column_name)
    return df


def create_pseudo_embedding_std(embeddings_out):
    df_stdev = pl.DataFrame(
        np.zeros_like(embeddings_out),
        schema=[f"embedding{i}_std" for i in range(embeddings_out.shape[1])],
    )
    return df_stdev


def create_embeddings_df(embeddings_out):
    df_embeddings = pl.DataFrame(
        embeddings_out,
        schema=[f"embedding{i}_value" for i in range(embeddings_out.shape[1])],
    )
    return df_embeddings


def create_split_embedding_df(embeddings_out, subject_codes, df_metadata):
    df_codes = pl.DataFrame(subject_codes, schema=["subject_code"])
    df_embeddings = create_embeddings_df(embeddings_out)
    # we don't have no stdev for the embeddings, but to make downstream code manage with less exceptions, let's add them
    df_stdev = create_pseudo_embedding_std(df_embeddings)

    df = pl.concat([df_codes, df_embeddings, df_stdev, df_metadata], how="horizontal")

    return df


def get_subject_dict_for_df(embeddings_out, split_dict, cfg):
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


def get_subject_codes(split_dict):
    subject_codes = []
    for i in range(split_dict["metadata"]["subject_code"].shape[0]):
        subject_codes.append(split_dict["metadata"]["subject_code"][i, 0])
    return subject_codes


def combine_embeddings_with_metadata_for_df(embeddings_out, split_dict, cfg):
    """
    See compute_features_from_dict() in src/featurization/featurize_PLR.py
    """
    subject_codes = get_subject_codes(split_dict)
    split_dict_subject = get_subject_dict_for_df(embeddings_out, split_dict, cfg)
    df_metadata = get_dataframe_from_dict(split_dict_subject, cfg)
    df_metadata = fix_pl_schema(df_metadata)

    # Combine 4 "sub-dataframes" into one
    df = create_split_embedding_df(embeddings_out, subject_codes, df_metadata)

    return df


def get_embeddings_per_split(model, dataloader, split_dict, model_cfg, cfg):
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
    model, dataloaders, source_data, model_cfg, cfg
) -> dict[str, pl.DataFrame]:
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


def import_moment_embedder(cfg: DictConfig, model_cfg: DictConfig):
    """
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
    source_data: dict,
    source_name: str,
    model_cfg: DictConfig,
    cfg: DictConfig,
    run_name: str,
    # artifacts_dir: str,
    model_name: str,
    pre_embedding_cfg: DictConfig,
    # experiment_name: str
):
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
