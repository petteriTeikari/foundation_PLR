import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.anomaly_detection.extra_eval.eval_outlier_detection import (
    eval_outlier_detection,
)
from src.anomaly_detection.ts_library.time_series_lib_utils import (
    adjust_learning_rate,
)


def init_training(model, outlier_model_cfg):
    if outlier_model_cfg["PARAMS"]["optimizer"] == "Adam":
        model_optim = torch.optim.Adam(
            model.parameters(), lr=outlier_model_cfg["PARAMS"]["lr"]
        )
    else:
        logger.error(
            f"Optimizer {outlier_model_cfg['PARAMS']['optimizer']} not implemented yet"
        )
        raise NotImplementedError(
            f"Optimizer {outlier_model_cfg['PARAMS']['optimizer']} not implemented yet"
        )

    if outlier_model_cfg["PARAMS"]["criterion"] == "MSE":
        criterion = torch.nn.MSELoss()
    else:
        logger.error(
            f"Criterion {outlier_model_cfg['PARAMS']['criterion']} not implemented yet"
        )
        raise NotImplementedError(
            f"Criterion {outlier_model_cfg['PARAMS']['criterion']} not implemented yet"
        )

    return model_optim, criterion


def timesnet_forward(
    batch_x, model, device, criterion, features, return_score: bool = False
):
    """
    args:
        batch_x: torch.Tensor
            expected size (batch_size, seq_len, num_features), e.g. (128,100,25) for the demo PSM dataset
        batch_y: torch.Tensor
            expected size (batch_size, seq_len, 1), e.g. (128,100,1) for the demo PSM dataset
    """
    batch_x = batch_x.float().to(device)  # e.g (16,100)
    batch_x = batch_x.unsqueeze(2)  # e.g. (16,100,1)
    # batch_y = batch_y.float().to(device) # e.g (16,100)
    # batch_y = batch_y.unsqueeze(2) # e.g. (16,100,1)
    outputs = model(batch_x, None, None, None)  # e.g. (16,100,1)
    if return_score:
        score = torch.mean(criterion(batch_x, outputs), dim=-1)
        score = score.detach().cpu().numpy()
        loss = score
    else:
        f_dim = -1 if features == "MS" else 0
        outputs = outputs[:, :, f_dim:]  # e.g. (16,100,1)
        loss = criterion(outputs, batch_x)
    return outputs, loss


def timesnet_train(
    model,
    device,
    outlier_model_cfg,
    cfg,
    run_name,
    experiment_name,
    train_loader,
    test_loader,
    outlier_train_loader,
    outlier_test_loader,
    recon_on_outliers: bool = False,
):
    """
    https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_anomaly_detection.py#L20
    """

    # train_steps = len(train_loader)
    model_optim, criterion = init_training(model, outlier_model_cfg)
    configs = outlier_model_cfg["PARAMS"]
    best_val_loss = np.inf
    test_metrics = {}
    losses = {"train": [], "test": []}

    if recon_on_outliers:
        logger.warning('Training the reconstruction on the noisy "pupil_orig" data')
        train_loader = outlier_train_loader
        test_loader = outlier_test_loader

    for epoch in (pbar := tqdm(range(configs.train_epochs), desc="TimesNet Training")):
        iter_count = 0
        train_loss = []

        model.train()
        for i, (batch_x, batch_y, _) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            outputs, loss = timesnet_forward(
                batch_x, model, device, criterion, configs.features
            )
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        test_loss = timesnet_val(
            model, test_loader, criterion, device, configs.features
        )
        pbar.set_description(
            f"{epoch + 1}/{configs.train_epochs} TimesNet Training | "
            f"Train loss = {np.mean(train_loss):.5f}, Test loss = {test_loss:.5f}"
        )
        losses["train"].append(train_loss)
        losses["test"].append(test_loss)
        logger.debug(test_loss)
        adjust_learning_rate(
            model_optim, epoch + 1, configs.lradj, configs.lr, configs.train_epochs
        )

        # Test
        # NOTE! This does not now affect the model saving or validation loss tracking, so you could
        # evaluate this less frequently and not on every epoch, if this gets a bit slow?
        metrics_test = eval_outlier_detection(
            model=model,
            train_loader=outlier_train_loader,
            test_loader=outlier_test_loader,
            device=device,
            features=configs.features,
            anomaly_ratio=configs.anomaly_ratio,
            window_size=outlier_model_cfg["TORCH"]["DATASET"]["trim_to_size"],
            length_PLR=cfg["DATA"]["PLR_length"],
        )
        test_metrics = append_keys(metrics_test, test_metrics)

        if test_loss < best_val_loss:
            # See e.g. "get_arrays_for_splits_from_outlier_artifacts()" of what the imputation flow
            # expects to find from the output
            best_epoch = epoch
            logger.debug(
                f"New best epoch: {best_epoch}, loss improved from {best_val_loss:.5f} to {test_loss:.5f}"
            )
            best_val_loss = test_loss
            best_arrays = get_best_arrays(metrics_test)

    metadata = get_best_metadata(losses, test_metrics, best_epoch)
    results_best, best_arrays = combine_results(
        best_arrays, losses, cfg, outlier_model_cfg
    )
    metrics = get_best_metrics(test_metrics, best_epoch)
    # e.g. results_best[split]["results_dict"]["split_results"]["arrays"]["preds"]
    metrics_out = {
        "results_best": results_best,
        "metrics": metrics,
        "metadata": metadata,
        "best_arrays": best_arrays,  # this same is in results_best (deep there), refactor eventually all these models
    }

    return metrics_out, model


def get_best_metrics(test_metrics, best_epoch):
    metrics = {}
    for split in test_metrics:
        metrics[split] = {}
        for key, value_list in test_metrics[split]["scalars"].items():
            if isinstance(value_list, list):
                value = value_list[best_epoch]
                metrics[split][key] = value
            else:
                logger.error("Is not a list, but {}".format(type(value_list)))
                raise ValueError

    return metrics


def combine_results(best_arrays, losses, cfg, outlier_model_cfg):
    results_best = {}
    # e.g. results_best[split]["results_dict"]["split_results"]["arrays"]["preds"]
    for split in losses:
        results_best[split] = {}
        results_best[split]["losses"] = losses[split]

    for split in best_arrays:
        results_best[split] = {}
        results_best[split]["results_dict"] = {}
        results_best[split]["results_dict"]["split_results"] = {}
        results_best[split]["results_dict"]["split_results"]["arrays"] = {}
        for key, array in best_arrays[split].items():
            results_best[split]["results_dict"]["split_results"]["arrays"][key] = array

    return results_best, best_arrays


def get_best_arrays(metrics_test):
    # If you need so save up some RAM, no need to save necessarily all the values on every epoch?
    best_arrays = {}
    for split in metrics_test:
        best_arrays[split] = {}
        arrays_flat = metrics_test[split]["arrays_flat"]
        for key, array in arrays_flat.items():
            best_arrays[split][key] = array
        arrays = metrics_test[split]["arrays"]
        for key, array in arrays.items():
            best_arrays[split][key] = array

    return best_arrays


def get_best_metadata(losses, test_metrics, best_epoch):
    metadata = {
        "best_epoch": best_epoch,
        "best_loss_train": losses["train"][best_epoch],
        "best_loss_test": losses["test"][best_epoch],
        # as in outlier_test
        "best_metric_string": "f1",
        "best_outlier_test_metric": test_metrics["outlier_test"]["scalars"]["global"][
            best_epoch
        ]["f1"],
        "best_outlier_train_metric": test_metrics["outlier_train"]["scalars"]["global"][
            best_epoch
        ]["f1"],
    }

    return metadata


def append_keys(metrics_test, test_metrics):
    for split in metrics_test.keys():
        if split not in test_metrics:
            test_metrics[split] = {}
        for var_type in metrics_test[split]:
            if var_type == "scalars":
                if var_type not in test_metrics[split]:
                    test_metrics[split][var_type] = {}
                for var in metrics_test[split][var_type]:
                    if var not in test_metrics[split][var_type]:
                        test_metrics[split][var_type][var] = [
                            metrics_test[split][var_type][var]
                        ]
                    else:
                        test_metrics[split][var_type][var].append(
                            metrics_test[split][var_type][var]
                        )

    return test_metrics


def timesnet_val(model, loader, criterion, device, features: str):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, _, _) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            outputs, loss = timesnet_forward(
                batch_x, model, device, criterion, features
            )
            total_loss.append(loss.cpu().numpy())
    total_loss = np.average(total_loss)
    model.train()
    return total_loss
