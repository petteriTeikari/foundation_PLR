import os
import torch
from omegaconf import DictConfig
from loguru import logger

from src.log_helpers.log_naming_uris_and_dirs import get_torch_model_name


def save_model_to_disk(
    model,
    optimizer,
    scaler,
    checkpoint_path: str,
    cfg: DictConfig,
    best_epoch: int,
    device: str,
    first_save: bool,
    run_name: str,
):
    # https://github.com/moment-timeseries-foundation-model/moment-research/blob/main/moment/tasks/base.py#L169
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        best_epoch: best_epoch,
    }
    fname = get_torch_model_name(run_name)
    checkpoint_file = os.path.join(checkpoint_path, fname)
    if cfg["EXPERIMENT"]["debug"]:
        logger.info("Saving model to disk: {}".format(checkpoint_file))
    with open(checkpoint_file, "wb") as f:
        torch.save(checkpoint, f)

    if first_save:
        state_dict_out = model.state_dict().__str__()
        # load the model back, and make sure that the state_dict is the same
        model_debug = load_model_from_disk(
            model, checkpoint_file, device, cfg, task="outlier_detection"
        )
        state_dict_in = model_debug.state_dict().__str__()
        compare_state_dicts(state_dict_out, state_dict_in)
        if cfg["EXPERIMENT"]["debug"]:
            logger.info("Debug load of the saved model worked fine!")

    return checkpoint_file


def compare_state_dicts(state_dict_out, state_dict_in, same_ok: bool = True):
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
    if not same_ok:
        assert state_dict_out != state_dict_in, (
            "State dicts are the same! "
            "(your finetuned weights are the same as standard HuggingFace pretrained weights)"
        )
    else:
        assert state_dict_out == state_dict_in, (
            "State dicts are not the same! "
            "(you did not read the same weights back from the disk as you saved)"
        )


def remove_head_from_state_dict(checkpoint):
    # drop the head (that came from anomaly detection pretraining, and we are just re-using the model now)
    # check if this okay as we get this error without the drop.
    # "Unexpected key(s) in state_dict: "head.linear.weight", "head.linear.bias"."
    # But as we only trained the head, if we drop the head, the model will be as the pretrained model
    # from HuggingFace
    logger.info(
        'Pop "head.linear.weight" and "head.linear.bias" from '
        "pretrained anomaly detection MOMENT that we use for embedding"
    )
    if "head.linear.weight" in checkpoint["model_state_dict"].keys():
        checkpoint["model_state_dict"].pop("head.linear.weight")
    if "head.linear.bias" in checkpoint["model_state_dict"].keys():
        checkpoint["model_state_dict"].pop("head.linear.bias")

    return checkpoint


def load_state_dict(model, checkpoint, task):
    # if task == 'embedding':
    #     checkpoint = remove_head_from_state_dict(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_model_from_disk(
    model,
    checkpoint_file: str,
    device: str,
    cfg: DictConfig,
    task: str,
    load_to_cpu_if_fails_with_gpu: bool = True,
):
    # https://github.com/moment-timeseries-foundation-model/moment-research/blob/3ab637e413f35f2c317573c0ace280d825c558de/moment/tasks/base.py#L206
    if device == "cpu":
        # You are using `torch.load` with `weights_only=False` (the current default value), which uses the default
        # pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary
        # code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
        # for more details). In a future release, the default value for `weights_only` will be flipped to `True`.
        # This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be
        # allowed to be loaded via this mode unless they are explicitly allowlisted by the user via
        # `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True`
        # for any use case where you don't have full control of the loaded file.
        # Please open an issue on GitHub for any issues related to this experimental feature.
        checkpoint = torch.load(checkpoint_file, map_location=device)
    else:
        # device += ":0"  # TODO! hack, use maybe torch.device(device) later?
        device = 0  # 'cuda'
        try:
            checkpoint = torch.load(
                checkpoint_file, map_location=lambda storage, loc: storage.cuda(device)
            )
        except Exception as e:
            logger.error("Problem with loading the model, {}".format(e))
            if task == "imputation":
                # with imputation task, we are not finetuning anymore anything, so the time with the CPU
                # does not get all crazy. As in we are just loading the finetuned model from the outlier detection task
                # and then doing zero-shot inference
                logger.warning("Trying to load the model to CPU")
                device = "cpu"
                try:
                    checkpoint = torch.load(checkpoint_file, map_location=device)
                except Exception as e:
                    logger.error("Even the CPU loading failed!")
                    logger.error("Problem with loading the model, {}".format(e))
                    raise e

    return load_state_dict(model, checkpoint, task)
