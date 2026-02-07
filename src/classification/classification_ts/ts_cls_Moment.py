import os
import random
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import torch
from loguru import logger
from momentfm import MOMENTPipeline
from omegaconf import DictConfig
from scipy.special import softmax
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.bootstrap_evaluation import get_ensemble_stats
from src.classification.classifier_log_utils import (
    classifier_log_cls_evaluation_to_mlflow,
)
from src.classification.stats_metric_utils import bootstrap_metrics
from src.data_io.data_wrangler import convert_datadict_to_dict_arrays
from src.imputation.momentfm.moment_utils import init_torch_training


def train_epoch(
    model: MOMENTPipeline,
    device: str,
    train_dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    reduction: str = "mean",
) -> float:
    """
    Train only classification head for one epoch.

    Parameters
    ----------
    model : MOMENTPipeline
        The MOMENT model with classification head.
    device : str
        Device to run training on ('cuda' or 'cpu').
    train_dataloader : DataLoader
        PyTorch DataLoader containing training batches.
    criterion : torch.nn.Module
        Loss function (e.g., CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    reduction : str, optional
        Reduction method for output logits. Default is 'mean'.

    Returns
    -------
    float
        Average training loss over all batches.
    """
    model.to(device)
    model.train()
    losses = []

    for batch_x, batch_labels, _ in train_dataloader:
        optimizer.zero_grad()
        batch_x = batch_x.to(device).float().unsqueeze(1)
        batch_labels = batch_labels.to(device).long()
        output = model(x_enc=batch_x, reduction=reduction)
        loss = criterion(output.logits, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss


def evaluate_epoch(
    dataloader: DataLoader,
    model: MOMENTPipeline,
    criterion: torch.nn.Module,
    device: str,
    phase: str = "val",
    reduction: str = "mean",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch DataLoader containing evaluation batches.
    model : MOMENTPipeline
        The MOMENT model with classification head.
    criterion : torch.nn.Module
        Loss function for computing evaluation loss.
    device : str
        Device to run evaluation on ('cuda' or 'cpu').
    phase : str, optional
        Evaluation phase name (e.g., 'val', 'test'). Default is 'val'.
    reduction : str, optional
        Reduction method for output logits. Default is 'mean'.

    Returns
    -------
    tuple
        Tuple of (avg_loss, accuracy, outputs, labels) where:
        - avg_loss : float - Average loss over all batches
        - accuracy : float - Classification accuracy
        - outputs : np.ndarray - Model logits for all samples
        - labels : np.ndarray - Ground truth labels for all samples
    """
    model.eval()
    model.to(device)
    total_loss, total_correct = 0, 0

    outputs = None
    labels = None
    with torch.no_grad():
        for batch_x, batch_labels, _ in dataloader:
            batch_x = batch_x.to(device).float().unsqueeze(1)
            batch_labels = batch_labels.to(device).long()
            output = model(x_enc=batch_x, reduction=reduction)
            if outputs is None:
                outputs = output.logits.detach().cpu().numpy()
                labels = batch_labels.detach().cpu().numpy()
            else:
                outputs = np.concatenate(
                    (outputs, output.logits.detach().cpu().numpy()), axis=0
                )
                labels = np.concatenate(
                    (labels, batch_labels.detach().cpu().numpy()), axis=0
                )

            loss = criterion(output.logits, batch_labels)
            total_loss += loss.item()
            total_correct += (output.logits.argmax(dim=1) == batch_labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy, outputs, labels


def get_labels_from_loader(train_loader: DataLoader) -> np.ndarray:
    """
    Extract all labels from a DataLoader.

    Parameters
    ----------
    train_loader : DataLoader
        PyTorch DataLoader to extract labels from.

    Returns
    -------
    np.ndarray
        Concatenated array of all labels from the DataLoader.
    """
    labels = None
    for _, batch_labels, _ in train_loader:
        if labels is None:
            labels = batch_labels.detach().cpu().numpy()
        else:
            labels = np.concatenate(
                (labels, batch_labels.detach().cpu().numpy()), axis=0
            )
    return labels


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute balanced class weights from training data.

    Parameters
    ----------
    train_loader : DataLoader
        PyTorch DataLoader containing training data.

    Returns
    -------
    torch.Tensor
        Tensor of class weights for weighted loss computation.
    """
    labels = get_labels_from_loader(train_loader)
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)


def get_preds_from_logits(
    logits: np.ndarray, labels: np.ndarray, split: str
) -> Dict[str, np.ndarray]:
    """
    Convert model logits to probability predictions.

    Parameters
    ----------
    logits : np.ndarray
        Raw model output logits of shape (n_samples, n_classes).
    labels : np.ndarray
        Ground truth labels (unused but kept for API consistency).
    split : str
        Data split name (unused but kept for API consistency).

    Returns
    -------
    dict
        Dictionary with 'pred' key containing softmax probabilities.
    """
    probs = softmax(logits)  # (n_samples, n_classes), e.g. (63,2)
    # probs_class1 = probs[:, 1] # (n_samples,)
    # preds = probs_class1 > 0.5
    return {"pred": probs}


def get_dict_split_from_dataloaders(
    dataloaders: Dict[str, DataLoader],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert PyTorch DataLoaders to dictionary format for metric computation.

    Parameters
    ----------
    dataloaders : dict
        Dictionary mapping split names to DataLoader objects.

    Returns
    -------
    dict
        Dictionary mapping split names to data dictionaries, where each
        data dictionary contains:
        - X : np.ndarray - Feature array
        - y : np.ndarray - Label array
        - w : np.ndarray - Sample weights (ones)
        - codes : np.ndarray - Sample indices
    """

    def get_dict_per_split(dataloader: DataLoader, split: str) -> Dict[str, np.ndarray]:
        x, y = None, None
        for batch_x, batch_labels, _ in dataloader:
            if y is None:
                y = batch_labels.detach().cpu().numpy()
                x = batch_x.detach().cpu().numpy()
            else:
                y = np.concatenate((y, batch_labels.detach().cpu().numpy()), axis=0)
                x = np.concatenate((x, batch_x.detach().cpu().numpy()), axis=0)
        codes = np.linspace(1, len(y), len(y))
        w = np.ones((len(y)))
        return {"X": x, "y": y, "w": w, "codes": codes}

    dict_splits = {}
    for split, dataloader in dataloaders.items():
        dict_splits[split] = get_dict_per_split(dataloader, split)

    return dict_splits


def moment_ts_train_script(
    dataloaders: Dict[str, DataLoader],
    device: str,
    cfg: DictConfig,
    cls_model_cfg: DictConfig,
) -> Tuple[List[str], Dict[str, Any], Dict[int, Dict[str, List[float]]]]:
    """
    Train an ensemble of MOMENT classifiers on time series data.

    Parameters
    ----------
    dataloaders : dict
        Dictionary with 'train' and 'test' DataLoaders.
    device : str
        Device to run training on ('cuda' or 'cpu').
    cfg : DictConfig
        Main Hydra configuration object.
    cls_model_cfg : DictConfig
        Classifier-specific model configuration.

    Returns
    -------
    tuple
        Tuple of (models, metrics_iter, train_losses) where:
        - models : list - Trained model paths or placeholders
        - metrics_iter : dict - Per-iteration metrics from bootstrap evaluation
        - train_losses : dict - Training and test losses per submodel
    """
    # Not that computationally expensive with our small dataset, but reduce maybe to 5 if you larger dataset
    no_of_submodels_in_ensemble = cls_model_cfg["MODEL"]["no_submodels"]
    epoch = cls_model_cfg["MODEL"]["no_epochs"]

    train_losses = {}

    metrics_iter = {}
    models = []
    preds = {}

    train_msg = (
        f"Moment for classification ({cls_model_cfg['MODEL']['detection_type']})"
    )
    for submodel in (pbar := tqdm(range(no_of_submodels_in_ensemble), desc=train_msg)):
        # Initialize the Moment classifier model
        model = init_moment_cls_model(cls_model_cfg, device)

        losses = {"train": [], "test": []}
        if cls_model_cfg["MODEL"]["use_weighed_loss"]:
            class_weights = get_class_weights(train_loader=dataloaders["train"]).to(
                device
            )
            criterion = torch.nn.CrossEntropyLoss(
                weight=class_weights, reduction="mean"
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=cls_model_cfg["MODEL"]["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cls_model_cfg["MODEL"]["max_lr"],
            total_steps=(epoch * len(dataloaders["train"])),
        )
        control_randomness(random.randint(0, cls_model_cfg["MODEL"]["max_random_seed"]))

        for i in range(epoch):
            train_loss = train_epoch(
                model, device, dataloaders["train"], criterion, optimizer, scheduler
            )
            test_loss, test_accuracy, _, _ = evaluate_epoch(
                dataloaders["test"], model, criterion, device, phase="test"
            )
            pbar.set_description(
                f"{train_msg} | Epoch = {i + 1}/{epoch}, Train loss = {train_loss:.4f}, "
                f"test_loss = {test_loss:.4f}, test_acc = {test_accuracy:.2f}"
            )
            # lr=1e-4
            # Epoch = 50, Train loss = 0.5270, test_loss = 0.5304, test_acc = 0.73:  0/2 [05:28<?, ?it/s]
            losses["train"].append(train_loss)
            losses["test"].append(test_loss)

        # Here you could dump the losses to Tensorboard or to disk if desired
        train_losses[submodel] = losses

        # Evaluate
        _, _, test_logits, test_labels = evaluate_epoch(
            dataloaders["test"], model, criterion, device, phase="test"
        )
        preds["test"] = get_preds_from_logits(
            logits=test_logits, labels=test_labels, split="test"
        )

        _, _, train_logits, train_labels = evaluate_epoch(
            dataloaders["train"], model, criterion, device, phase="test"
        )
        preds["train"] = get_preds_from_logits(
            logits=train_logits, labels=train_labels, split="train"
        )

        # Same function as for bootstrapping now for submodel of an ensemble
        metrics_iter = bootstrap_metrics(
            i=submodel,
            model=model,
            dict_splits=get_dict_split_from_dataloaders(dataloaders),
            metrics=metrics_iter,
            results_per_iter=preds,
            method_cfg=cfg["CLS_EVALUATION"]["BOOTSTRAP"],
            cfg=cfg,
        )

        # You could save the best model to disk, and append them to a list
        # models.append(model.to("cpu"))
        models.append("dummy_path.pt")
        del model
        torch.cuda.empty_cache()
        logger.debug(
            f"CUDA Memory allocated (1): {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    return models, metrics_iter, train_losses


def control_randomness(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int, optional
        Random seed value. Default is 42.

    Returns
    -------
    None
    """
    logger.debug("Random seed = {}".format(seed))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_moment_cls_model(cls_model_cfg: DictConfig, device: str) -> MOMENTPipeline:
    """
    Initialize the MOMENT classifier model from pretrained weights.

    Parameters
    ----------
    cls_model_cfg : DictConfig
        Classifier model configuration containing model_kwargs.
    device : str
        Device to load the model on ('cuda' or 'cpu').

    Returns
    -------
    MOMENTPipeline
        Initialized MOMENT model ready for classification.

    Raises
    ------
    Exception
        If model cannot be moved to the specified device.
    """
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs=dict(cls_model_cfg["MODEL"]["model_kwargs"]),
    )
    model.init()
    try:
        model.to(device)
    except Exception as e:
        # cfg["DEVICE"]["device"] = 'cuda' gives:
        # RuntimeError: CUDA error: device-side assert triggered
        # Examine why? Luckily we only train for a couple of epochs, and can live with CPU training
        logger.error(e)
        raise e
    return model


def log_moment_cls_mlflow_params(cls_model_cfg: DictConfig) -> None:
    """
    Log MOMENT classifier parameters to MLflow.

    Parameters
    ----------
    cls_model_cfg : DictConfig
        Classifier model configuration to log.

    Returns
    -------
    None
    """
    mlflow.log_param("data_source", "time_series")
    for key, value in cls_model_cfg["MODEL"].items():
        if key != "model_kwargs":
            mlflow.log_param(key, value)
        else:
            for subkey, value2 in value.items():
                mlflow.log_param(subkey, value2)


def ts_cls_moment_main(
    source_name: str,
    features_per_source: Dict[str, Any],
    cls_model_name: str,
    cls_model_cfg: DictConfig,
    run_name: str,
    cfg: DictConfig,
) -> None:
    """
    Main entry point for MOMENT time series classification pipeline.

    Trains an ensemble of MOMENT classifiers, evaluates performance using
    bootstrap metrics, and logs all results to MLflow.

    Parameters
    ----------
    source_name : str
        Name of the data source being processed.
    features_per_source : dict
        Dictionary containing feature data, with 'df' key holding the DataFrame.
    cls_model_name : str
        Name of the classifier model (should be 'MOMENT').
    cls_model_cfg : DictConfig
        Classifier-specific model configuration.
    run_name : str
        Name for the MLflow run.
    cfg : DictConfig
        Main Hydra configuration object.

    Returns
    -------
    None
        Results are logged to MLflow rather than returned.

    See Also
    --------
    https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/ptbxl_classification.ipynb
    """
    with mlflow.start_run(run_name=run_name):
        log_moment_cls_mlflow_params(cls_model_cfg)
        device = cfg["DEVICE"]["device"]

        # Create the dataloaders
        dataloaders = init_torch_training(
            data_dict=features_per_source,
            cfg=cfg,
            model_cfg=cls_model_cfg,
            run_name=run_name,
            task="ts_cls",
            create_outlier_dataloaders=False,
        )

        # Train the model
        # NOTE! If you don't do full finetuning, this is basically what we did from the classification from embeddings
        #       just with a linear probe rather than with something more powerful like XGBoost/CatBoost
        models, metrics_iter, train_losses = moment_ts_train_script(
            dataloaders, device=device, cfg=cfg, cls_model_cfg=cls_model_cfg
        )

        # convert data_dict -> dict_arrays
        dict_arrays = convert_datadict_to_dict_arrays(
            data_dict=features_per_source["df"], cls_model_cfg=cls_model_cfg
        )

        # Get ensemble/bootstrap stats
        method_cfg = cfg["CLS_EVALUATION"]["BOOTSTRAP"]
        metrics_stats, subjectwise_stats, subject_global_stats = get_ensemble_stats(
            metrics_iter,
            dict_arrays=dict_arrays,
            method_cfg=method_cfg,
            call_from="ts_ensemble",
            sort_list=False,
        )

        metrics_out = {
            "metrics_iter": metrics_iter,
            "metrics_stats": metrics_stats,
            "subjectwise_stats": subjectwise_stats,
            "subject_global_stats": subject_global_stats,
            "train_losses": train_losses,
        }

        classifier_log_cls_evaluation_to_mlflow(
            None,
            None,
            models,
            metrics_out,
            dict_arrays=dict_arrays,
            cls_model_cfg=cls_model_cfg,
            run_name=run_name,
            model_name="MOMENT",
            log_manual_model=False,
        )

        mlflow.end_run()
