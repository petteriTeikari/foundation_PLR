import pandas as pd
from loguru import logger
from omegaconf import DictConfig
import mlflow

from src.featurization.embedding.moment_embedding import moment_embedder


def embedding_script(
    cfg: DictConfig,
    source_name: str,
    source_data: dict,
    model_name: str,
    embedding_cfg: DictConfig,
    run_name: str,
    pre_embedding_cfg: DictConfig,
):
    if model_name == "MOMENT":
        moment_embedder(
            source_data=source_data,
            source_name=source_name,
            model_cfg=embedding_cfg,
            cfg=cfg,
            run_name=run_name,
            model_name=model_name,
            pre_embedding_cfg=pre_embedding_cfg,
        )

    else:
        logger.error("Model {} not implemented! Typo?".format(model_name))
        raise NotImplementedError("Model {} not implemented!".format(model_name))


def if_embedding_not_done(run_name, experiment_name, cfg):
    mlflow_runs: pd.DataFrame = mlflow.search_runs(experiment_names=[experiment_name])
    run_names = list(mlflow_runs["tags.mlflow.runName"])
    if run_name not in run_names:
        return True
    else:
        logger.info("Embedding featurization already done!")
        return False


# @flow(
#     log_prints=True,
#     name="PLR Embedding",
#     description="  s",
# )
def flow_embedding(
    cfg: DictConfig, sources: dict, experiment_name: str, prev_experiment_name: str
):
    embedding_cfgs = {"MOMENT": cfg["PLR_EMBEDDING"]}

    preprocessing_cfgs = {"HighDim": None, "PCA": cfg["EMBEDDING"]["PREPROCESSING"]}

    no_of_runs = len(sources) * len(embedding_cfgs) * len(preprocessing_cfgs)
    run_idx = 0
    # Featurize (or skip if previous results found from MLflow)
    for source_idx, (source_name, source_data) in enumerate(sources.items()):
        for idx, (model_name, embedding_cfg) in enumerate(embedding_cfgs.items()):
            for pre_idx, (preproc_name, pre_embedding_cfg) in enumerate(
                preprocessing_cfgs.items()
            ):
                logger.info(f"Source #{source_idx+1}/{len(sources)}: {source_name}")
                logger.info(
                    f"Running pipeline for embedding method #{idx+1}/{len(embedding_cfgs)}: {model_name}"
                )

                if pre_embedding_cfg is None:
                    run_name = f"{model_name}-embedding__{source_name}"
                else:
                    run_name = f"{model_name}-embedding-{preproc_name}__{source_name}"
                logger.info(f"Run name #{run_idx+1}/{no_of_runs}: {run_name}")
                run_idx += 1

                if if_embedding_not_done(run_name, experiment_name, cfg):
                    embedding_script(
                        cfg=cfg,
                        source_name=source_name,
                        source_data=source_data,
                        model_name=model_name,
                        embedding_cfg=embedding_cfg[model_name],
                        run_name=run_name,
                        pre_embedding_cfg=pre_embedding_cfg,
                    )
                else:
                    logger.info("Embedding already done, skipping now")
