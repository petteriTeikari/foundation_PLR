import numpy as np
import torch
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
from torch import optim

from src.imputation.momentfm.moment_imputation_main import (
    compute_moment_imputation_metrics,
)
from src.imputation.momentfm.moment_utils import (
    init_torch_training,
    reshape_array_to_original_shape,
)
from src.imputation.nuwats.NuwaTS.exp.exp_imputation import Exp_Imputation
from src.imputation.nuwats.get_parser import get_parser
from src.imputation.nuwats.nuwats_tools import (
    download_pretrained_nuwats_weights,
)


def mask_x(x: torch.Tensor, y: torch.Tensor):
    """"""
    batch_x_inp = x.masked_fill(y == 1, 0)
    # no_masked_points = torch.sum(batch_x_inp == 0).sum().item()
    return batch_x_inp


def check_dataloader_for_nuwats(dataloader):
    masked_points = 0
    for i, (batch_x_gt, batch_y, _) in enumerate(dataloader):
        no_of_nans_in_x = torch.isnan(batch_x_gt).sum().item()
        assert (
            no_of_nans_in_x == 0
        ), "You are not supposed to have NaNs in your input data"
        masked_points += torch.sum(batch_y == 1).sum().item()
        # if your trim and batch size is small, you might have batches that do not have any masked points

    # So assess only after the whole dataloader
    assert masked_points > 0, (
        "You need to have some missing points masked, "
        "otherwise no idea what imputation to assess"
    )


def nuwats_forward(model, batch_x_gt, batch_y, device):
    batch_x_gt = batch_x_gt.float().to(device)
    batch_y = batch_y.long().to(device)
    batch_x_inp = mask_x(x=batch_x_gt, y=batch_y).unsqueeze(2)
    batch_x_gt = batch_x_gt.unsqueeze(2)  # e.g. (16,96) -> (16,96,1)
    batch_y = batch_y.unsqueeze(2)

    # imputation
    outputs, _ = model(batch_x_inp, None, None, None, batch_y)
    # outputs = fill_missing_data_ON(inp, mask)

    return outputs, batch_x_gt, batch_y, batch_x_inp


def nuwats_evaluate(
    model, dataloader, device, split: str, features: str = "S", criterion=None
):
    """
    https://github.com/Chengyui/NuwaTS/blob/2b7850d3f5271e6d6bc9bd82e8dc6237c1fe64ef/exp/exp_imputation.py#L249
    """
    check_dataloader_for_nuwats(dataloader)
    preds = []
    trues = []
    masks = []
    total_loss = []
    model.eval()
    for i, (batch_x_gt, batch_y, _) in enumerate(
        dataloader
    ):  # total=len(dataloader), desc=f'Evaluate on "{split}"')
        samples_in = batch_x_gt.shape[0] * batch_x_gt.shape[1]
        with torch.no_grad():
            outputs, batch_x_gt, batch_y, batch_x_inp = nuwats_forward(
                model, batch_x_gt, batch_y, device
            )
            samples_out = outputs.shape[0] * outputs.shape[1]
            assert samples_in == samples_out, (
                "Output samples from NuwaTS do not match the number of input samples,ņ"
                "Maybe you tried something else prediction_length than 96?"
            )

        # eval
        f_dim = -1 if features == "MS" else 0
        B, T, N = batch_x_gt.shape  # in original Colab, (1 96 1), now example (1,16,96)
        outputs = outputs[:, -T:, f_dim:]
        outputs = outputs.detach().cpu().numpy()
        pred = outputs
        no_of_nans_in_pred = np.sum(np.isnan(pred))
        if no_of_nans_in_pred > 0:
            logger.error(
                "NaNs found in the prediction outputs, {:.2f}% out of {} samples".format(
                    100 * (no_of_nans_in_pred / pred.size), pred.size
                )
            )
            logger.error("Abort training, and you need to examine why was this?")
            return None, None, None, None

        true = batch_x_gt.detach().cpu().numpy()
        mask = batch_y.detach().cpu().numpy()
        if criterion is not None:
            loss = criterion(pred[mask == 0], true[mask == 0])
            total_loss.append(loss)
        preds.append(pred)
        trues.append(true)
        masks.append(mask)

    total_loss = np.average(total_loss)
    preds = np.concatenate(preds, 0)[:, :, 0]
    trues = np.concatenate(trues, 0)[:, :, 0]
    masks = np.concatenate(masks, 0)[:, :, 0]
    model.train()

    return preds, trues, masks, total_loss


def reshape_eval_to_output(preds, trues, masks, model_cfg, cfg):
    """
    reshape back to input shape, e.g. from (7455, 96) -> (355, 1981)
    and match the dict to the output of other methods, see e.g. "eval_moment_outlier_finetune()"
    """
    preds = reshape_array_to_original_shape(preds, cfg, model_cfg, dim=2)
    trues = reshape_array_to_original_shape(trues, cfg, model_cfg, dim=2)
    masks = reshape_array_to_original_shape(masks, cfg, model_cfg, dim=2)
    assert preds.shape == trues.shape == masks.shape

    split_results = {}
    split_results["arrays"] = {
        "trues": trues,  # e.g. (355, 1981)
        "preds": preds,
        "labels": masks,
    }
    split_results["arrays_flat"] = {
        "trues_valid": split_results["arrays"]["trues"].flatten(),  # e.g. (703255,)
        "preds_valid": split_results["arrays"]["preds"].flatten(),
        "labels_valid": split_results["arrays"]["labels"].flatten(),
    }

    return split_results


def dataloader_eval_wrapper(model, dataloader, device, split, model_cfg, cfg):
    # Zero-shot imputation (reconstruction)
    preds, trues, masks, _ = nuwats_evaluate(
        model, dataloader, device, split, features=model_cfg["MODEL"]["features"]
    )

    # Shape back to input space, e.g. (355, 1981)
    imputation_results = reshape_eval_to_output(preds, trues, masks, model_cfg, cfg)

    return imputation_results


def dataloaders_eval(dataloaders, model, device, model_cfg, cfg):
    imputation_results = {}
    for split, dataloader in dataloaders.items():
        # use extra nesting keys to have these match "older" models
        imputation_results[split] = {"results_dict": {}}
        imputation_results[split]["results_dict"]["split_results"] = (
            dataloader_eval_wrapper(model, dataloader, device, split, model_cfg, cfg)
        )

    return imputation_results


def nuwats_zeroshot_wrapper(model, dataloaders, data_dict, device, model_cfg, cfg):
    logger.info("NuwaTS Zeroshot Imputation")
    imputation_results = dataloaders_eval(dataloaders, model, device, model_cfg, cfg)

    # Compute imputation metrics
    metrics, imputation_dict = compute_moment_imputation_metrics(
        imputation_results=imputation_results,
        data_dict=data_dict,
        cfg=cfg,
        imputation_time=None,
        checks_on=True,
    )

    return {
        "imputation_results": imputation_results,
        "metrics": metrics,
        "imputation": imputation_dict,
    }


def adjust_learning_rate(optimizer, epoch, lr, lradj):
    if lradj == "type1":
        lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # logger.info("Updating learning rate to {}".format(lr))
    return lr


def nuwats_finetune_train(
    model, dataloaders, device, model_cfg, cfg, features: str = "S"
):
    """
    See Simple_try in the .zip file of the tutorial notebook for guidance, and even a bit more:
    def train(self,..) in exp_imputation.py
    """
    criterion = nn.MSELoss()
    model_optim = optim.Adam(model.parameters(), lr=model_cfg["MODEL"]["lr"])

    for i, (name, param) in enumerate(model.named_parameters()):
        if "prefix" in name:  # or 'mlp' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_steps = len(dataloaders["train"])
    logger.info("train steps:", train_steps)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Total params = {total_params}")
    logger.info(f"Trainable params = {total_trainable_params}")
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            logger.info(f"{name} | requires_grad = True")

    train_losses = []
    best_loss = np.inf
    model_path = None
    imputation_results = None
    for epoch in (
        pbar := tqdm(
            range(model_cfg["MODEL"]["no_epochs"]),
            total=model_cfg["MODEL"]["no_epochs"],
        )
    ):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x_gt, batch_y, _) in enumerate(dataloaders["train"]):
            iter_count += 1
            model_optim.zero_grad()
            # Petteri: Need for representation (if needed/desired?)
            # batch_x_mark = ? # see e.g. "__getitem__(self, index)" of "Dataset_ETT_minute"
            outputs, batch_x_gt, batch_y, batch_x_inp = nuwats_forward(
                model, batch_x_gt, batch_y, device
            )

            f_dim = -1 if features == "MS" else 0
            B, T, N = batch_x_gt.shape
            outputs = outputs[:, -T:, f_dim:]
            loss = criterion(outputs, batch_x_gt)

            # loss = criterion(outputs[mask == 0], batch_x[mask == 0]) + con_loss * self.args.con_weight
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        train_losses.append(train_loss)
        _, _, _, test_loss = nuwats_evaluate(
            model,
            dataloader=dataloaders["test"],
            device=device,
            split="test",
            features=features,
            criterion=criterion,
        )

        if test_loss is None:
            return None, None, None

        if test_loss < best_loss:
            best_loss = test_loss
            imputation_results = dataloaders_eval(
                dataloaders, model, device, model_cfg, cfg
            )
            # TODO! save model to disk
            model_path = None

        # scheduler.step(epoch)
        lr = adjust_learning_rate(
            model_optim, epoch + 1, lr=model_cfg["lr"], lradj=model_cfg["lradj"]
        )

        pbar.set_description(
            f"NuwaTS Finetune | Epoch = {epoch + 1}/{model_cfg['no_epochs']}, "
            f"Train loss = {train_loss:.4f}, "
            f"test_loss = {test_loss:.4f}, "
            f"lr = {lr:.5f}"
        )

        del model
        torch.cuda.empty_cache()

    return model_path, imputation_results, train_losses


def nuwats_finetune_wrapper(model, dataloaders, data_dict, device, model_cfg, cfg):
    model_path, imputation_results, train_losses = nuwats_finetune_train(
        model, dataloaders, device, model_cfg, cfg
    )

    if imputation_results is None:
        return None, None

    else:
        # Compute imputation metrics
        metrics, imputation_dict = compute_moment_imputation_metrics(
            imputation_results=imputation_results,
            data_dict=data_dict,
            cfg=cfg,
            imputation_time=None,
            checks_on=True,
        )

        model_artifacts = {
            "imputation_results": imputation_results,
            "metrics": metrics,
            "imputation": imputation_dict,
        }

        return model_path, model_artifacts


def nuwats_imputation_main(
    data_dict: dict,
    model_cfg: DictConfig,
    cfg: DictConfig,
    model_name: str = None,
    run_name: str = None,
):
    """
    https://github.com/Chengyui/NuwaTS
    https://colab.research.google.com/drive/1jjM6g4N7AqyHjYawZWJdbFgNY7p4ZtGY?usp=sharing
    https://arxiv.org/pdf/2405.15317
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        logger.error("Not working with CPU debugging")
        raise NotImplementedError("Not working with CPU debugging")

    # Create the dataloaders
    dataloaders = init_torch_training(
        data_dict=data_dict,
        cfg=cfg,
        model_cfg=model_cfg,
        model_name="NuwaTS",
        run_name=run_name,
        task="imputation",
        create_outlier_dataloaders=False,
    )

    model_path = download_pretrained_nuwats_weights(
        weights_fname="NuwaTS_math_finetuned.pth"
    )

    # https://colab.research.google.com/drive/1jjM6g4N7AqyHjYawZWJdbFgNY7p4ZtGY?usp=sharing
    Exp = Exp_Imputation
    # TODO! put some of these to the .yaml if you decide that NuwaTS is useful for PLR
    args = get_parser()
    exp = Exp(args)

    model = exp.model
    model.load_state_dict(torch.load(model_path, map_location=device))

    if model_cfg["MODEL"]["detection_type"] == "zero-shot":
        model_artifacts = nuwats_zeroshot_wrapper(
            model, dataloaders, data_dict, device, model_cfg, cfg
        )
    elif model_cfg["MODEL"]["detection_type"] == "fine-tune":
        model_path, model_artifacts = nuwats_finetune_wrapper(
            model, dataloaders, data_dict, device, model_cfg, cfg
        )
    else:
        logger.error(
            "Unknown detection type = {}".format(model_cfg["MODEL"]["detection_type"])
        )
        raise NotImplementedError(
            "Unknown detection type = {}".format(model_cfg["MODEL"]["detection_type"])
        )

    return model_path, model_artifacts
