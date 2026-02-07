import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import torch
from loguru import logger

from src.classification.tabm.tabm_reference import Model, make_parameter_groups


def create_tabm_model(
    device,
    arch_type: str,
    data: dict,
    d_block: int = 512,
    dropout: float = 0.1,
    d_embedding: int = 16,
    k: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
    compile_model: bool = False,
):
    n_cont_features = data["train"]["x_cont"].shape[1]
    n_classes = len(torch.unique(data["train"]["y"]))
    assert (
        n_classes == 2
    ), "Only binary classification is supported. You had {} classes.".format(n_classes)

    if arch_type == "tabm-mini":
        # TabM-mini with the piecewise-linear embeddings.
        try:
            bins = rtdl_num_embeddings.compute_bins(
                data["train"]["x_cont"]
            )  # astype(np.float32)
        # Elaborate try/except just for the demo model, and very small datasets in general
        except Exception as e:
            try:
                logger.warning(e)
                logger.warning(
                    "Trying a smaller bin number, e.g. when you are running the demo data"
                )
                bins = rtdl_num_embeddings.compute_bins(
                    data["train"]["x_cont"], n_bins=2
                )  # astype(np.float32)
            except Exception as e2:
                try:
                    logger.warning(e)
                    logger.warning(
                        "Smaller bin number did not work out, duplicating the train data"
                    )
                    data["train"]["x_cont"] = data["train"]["x_cont"].repeat(2, 1)
                    data["train"]["y"] = data["train"]["y"].repeat(2)
                    bins = rtdl_num_embeddings.compute_bins(
                        data["train"]["x_cont"], n_bins=2
                    )  # astype(np.float32)

                except Exception:
                    logger.error(
                        "Failed to compute bins for TabM, error: {}".format(e2)
                    )
                    raise ValueError(
                        "Failed to compute bins for TabM, error: {}".format(e2)
                    )
    else:
        logger.error(
            f"Unknown architecture type: {arch_type}, only tabm-mini is supported now."
        )
        raise ValueError(
            f"Unknown architecture type: {arch_type}, only tabm-mini is supported now."
        )

    model = Model(
        n_num_features=n_cont_features,
        cat_cardinalities=None,
        n_classes=n_classes,
        backbone={
            "type": "MLP",
            "n_blocks": 3 if bins is None else 2,
            "d_block": d_block,
            "dropout": dropout,
        },
        bins=bins,
        num_embeddings=(
            None
            if bins is None
            else {
                "type": "PiecewiseLinearEmbeddings",
                "d_embedding": d_embedding,
                "activation": False,
                "version": "B",
            }
        ),
        arch_type=arch_type,
        k=k,
    ).to(device)

    optimizer = torch.optim.AdamW(
        make_parameter_groups(model), lr=lr, weight_decay=weight_decay
    )

    if compile_model:
        # NOTE
        # `torch.compile` is intentionally called without the `mode` argument
        # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
        model = torch.compile(model)
        evaluation_mode = torch.no_grad
    else:
        evaluation_mode = torch.inference_mode

    return model, optimizer, evaluation_mode
