from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from loguru import logger
import polars as pl
import umap
from sklearn import preprocessing
from tqdm import tqdm
import mlflow

from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA

from src.featurization.embedding.moment_embedding import (
    create_pseudo_embedding_std,
    create_embeddings_df,
)


def cap_dimensionality_of_PCA(train_pcs, test_pcs, max_dim: int = 96):
    """
    TabPFN cannot handle more than 100 values so that is the extreme limit,
    and if clean data gives 28, we don't want to deviate too much from this,
    even
    """
    max_dim_in = max(train_pcs.shape[1], test_pcs.shape[1])
    if max_dim_in > max_dim:
        logger.warning(
            f"capping dimensions from max_dim_in {max_dim_in} > max_dim {max_dim}"
        )
        if train_pcs.shape[1] > max_dim:
            train_pcs = train_pcs[:, :max_dim]
        if test_pcs.shape[1] > max_dim:
            test_pcs = test_pcs[:, :max_dim]

    return train_pcs, test_pcs


def apply_PCA_for_embedding(embeddings, pca_config: DictConfig):
    # Get just the features
    train_df: pd.DataFrame = get_df_features(df=embeddings["data"]["train"]).to_pandas()
    test_df: pd.DataFrame = get_df_features(df=embeddings["data"]["test"]).to_pandas()

    # Standardize
    scalar = StandardScaler()
    scalar.fit(train_df)
    train_df_scaled = pd.DataFrame(scalar.transform(train_df))
    test_df_scaled = pd.DataFrame(scalar.transform(test_df))

    # PCA
    pca = PCA(n_components=pca_config["explained_variance"])
    pca.fit(train_df_scaled)  # data: (n_samples, n_features)
    # print(pca.explained_variance_ratio_)
    # print(np.cumsum(pca.explained_variance_ratio_))
    train_pcs = pca.transform(train_df_scaled)
    test_pcs = pca.transform(test_df_scaled)
    assert (
        train_pcs.shape[1] == test_pcs.shape[1]
    ), "number of features (PCs) does not match"

    logger.info(
        f"PCA kept {train_pcs.shape[1]} components with explained variance"
        f" = {np.cumsum(pca.explained_variance_ratio_)[-1]:.3f}"
    )

    # pupil-gt__pupil-gt ground truth got down to 21 principal components
    # what if some outlier+imputation combo gives larger dimensionality
    train_pcs, test_pcs = cap_dimensionality_of_PCA(
        train_pcs, test_pcs, max_dim=pca_config["max_dim"]
    )

    for param, value in pca_config.items():
        mlflow.log_param(param, value)
    mlflow.log_param("no_of_PCs", train_pcs.shape[1])

    # assign to the input dataframe with some metadata as well
    embeddings = assign_features_back_to_full_df(embeddings, train_pcs, test_pcs)

    return embeddings


def assign_features_back_to_full_df(embeddings, train_pcs, test_pcs):
    def get_metadata_cols(df_out):
        cols = df_out.columns
        cols_to_drop = [i for i in cols if "embedding" in i]
        df = df_out.drop(cols_to_drop)
        return df

    def assign_per_df(pcs, df_out):
        df_meta = get_metadata_cols(df_out).to_pandas()
        for idx in range(pcs.shape[1]):
            df_meta["embedding{}_value".format(idx)] = pcs[:, idx]
            df_meta["embedding{}_std".format(idx)] = 0  # placeholder
        return df_meta

    embeddings["data"]["train"] = assign_per_df(
        pcs=train_pcs, df_out=embeddings["data"]["train"]
    )
    embeddings["data"]["test"] = assign_per_df(
        pcs=test_pcs, df_out=embeddings["data"]["test"]
    )

    return embeddings


def apply_dimensionality_reduction_for_feature_sources(features: dict, cfg: DictConfig):
    dim_cfg = cfg["CLASSIFICATION_SETTINGS"]["DIM_REDUCTION"]
    if dim_cfg["enable"]:
        logger.info("Applying dimensionality reduction for embedding feature sources")
        logger.info(dim_cfg)

        start_time = datetime.now()
        n = 0
        for source_name, source_data in tqdm(
            features.items(), desc="Applying dimensionality reduction"
        ):
            if "embedding" in source_name:
                n += 1
                if dim_cfg["enable"]:
                    features[source_name]["data"] = embedding_dim_reduction_wrapper(
                        embeddings=source_data["data"],
                        dim_cfg=dim_cfg,
                        source_name=source_name,
                    )

        dim_time_per_sample = ((datetime.now() - start_time).total_seconds()) / n
        logger.info(
            f"Dimensionality reduction done in {dim_time_per_sample:.2f} seconds per source"
        )

    return features


def torch_to_numpy(torch_tensor):
    print("todo!")


def get_df_features(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    cols_to_keep = [
        i for i in cols if "_value" in i
    ]  # from MOMENT this should be length of 1024
    df = df.select(cols_to_keep)

    return df


def get_feature_embedding_df(
    df: pl.DataFrame,
    label_col: str = "metadata_class_label",
    return_classes_as_int: bool = True,
):
    """
    drop the _std and metadata_ and subject_code cols
    df
      (145,2052) for train
    """
    cols = df.columns
    df_in = deepcopy(df)
    labels = df[label_col].to_numpy()
    assert len(np.unique(labels)) == 2, "we are doing a binary classification"
    if return_classes_as_int:
        # convert string labels to integers
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)  # 0: control, 1: glaucoma
    cols_to_keep = [
        i for i in cols if "_value" in i
    ]  # from MOMENT this should be length of 1024
    cols_to_drop = [
        i for i in cols if "_std" in i
    ]  # from MOMENT this should be length of 1024
    df = df.select(cols_to_keep)
    df_out = df_in.drop(cols_to_drop + cols_to_keep)  # the metadata and code, etc

    return df, labels, df_out


def combine_cols_to_out(embedding_train, embedding_test, train_df_out, test_df_out):
    def combine_per_split(embeddings: np.ndarray, df_out: pl.DataFrame):
        assert embeddings.shape[0] == df_out.shape[0]
        df_stdev = create_pseudo_embedding_std(embeddings)
        df_embeddings = create_embeddings_df(embeddings)
        df = pl.concat([df_embeddings, df_stdev, df_out], how="horizontal")
        assert df.shape[0] == df_out.shape[0]
        return df

    df_train = combine_per_split(embeddings=embedding_train, df_out=train_df_out)
    df_test = combine_per_split(embeddings=embedding_test, df_out=test_df_out)

    return df_train, df_test


def umap_wrapper(embeddings, dim_cfg, source_name):
    """
    Obviously this becomes its own HPO experiment if you would like to give this a good fair assessment?
    Now quick'n'dirty dimensional reduction with some "default values" and see how this performs without tweaking
    embeddings: dict
      - test
          pl.DataFrame (63,2052) -> pl.DataFrame (63,20) when n_components=8
      - train
          pl.DataFrame (145,2052) -> pl.DataFrame (145,20) when n_components=8
    """

    train_df, train_labels, train_df_out = get_feature_embedding_df(
        df=embeddings["train"]
    )
    test_df, test_labels, test_df_out = get_feature_embedding_df(df=embeddings["test"])
    assert (
        train_df.shape[1] == test_df.shape[1]
    ), "Number of features do not match between test and train"

    # unsupervised baseline
    # https://umap-learn.readthedocs.io/en/latest/supervised.html#umap-on-fashion-mnist
    mapper = umap.UMAP(
        n_neighbors=dim_cfg["n_neighbors"],
        n_components=dim_cfg["n_components"],
        random_state=dim_cfg["random_state"],
        transform_seed=dim_cfg["transform_seed"],
        n_jobs=1,
    )

    if dim_cfg["supervised"]:
        # https://umap-learn.readthedocs.io/en/latest/supervised.html#training-with-labels-and-embedding-unlabelled-test-data-metric-learning-with-umap
        mapper = mapper.fit(train_df.to_numpy(), np.array(train_labels))

    embedding_train = mapper.fit_transform(
        train_df.to_numpy()
    )  # e.g. (145,1024) -> (145,8)
    embedding_test = mapper.fit_transform(test_df.to_numpy())

    # combine with the metadata and create the pseudostdev again
    embeddings["train"], embeddings["test"] = combine_cols_to_out(
        embedding_train, embedding_test, train_df_out, test_df_out
    )

    return embeddings


def embedding_dim_reduction_wrapper(embeddings, dim_cfg, source_name):
    """
    Go down from 1024 -> to 2D/3D for visualization (and for classification)
    """
    if dim_cfg["method"] == "UMAP":
        embeddings = umap_wrapper(embeddings, dim_cfg, source_name)
    else:
        logger.error("Method {} not implemented! Typo?".format(dim_cfg["method"]))
        raise NotImplementedError(
            "Method {} not implemented!".format(dim_cfg["method"])
        )

    return embeddings
