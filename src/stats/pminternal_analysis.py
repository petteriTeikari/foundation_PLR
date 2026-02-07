"""
pminternal-style Model Instability Analysis (Riley 2023).

Provides:
- Bootstrap prediction matrix export for pminternal R package
- Prediction instability plots (Riley 2023 Fig 2)
- MAPE (Mean Absolute Prediction Error) per subject
- Classification Instability Index (CII)
- Per-patient uncertainty distributions (Kompa 2021)

Cross-references:
- planning/latent-method-results-update.md (Section 26)
- Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"
- Kompa B et al. (2021) "Second opinion needed: communicating uncertainty in medical ML"

NOTE: This module provides Python-native implementations.
For R/pminternal integration, use src/stats/pminternal_wrapper.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger

__all__ = [
    "BootstrapPredictionData",
    "InstabilityMetrics",
    "load_bootstrap_predictions_from_mlflow",
    "compute_prediction_instability_metrics",
    "compute_per_patient_uncertainty",
    "export_pminternal_data",
]


@dataclass
class BootstrapPredictionData:
    """
    Bootstrap prediction data for pminternal analysis.

    Attributes
    ----------
    combo_id : str
        Unique identifier for the preprocessing+classifier combination
    y_true : np.ndarray
        True binary outcomes (n_subjects,)
    y_pred_proba_original : np.ndarray
        Original model predictions (n_subjects,)
    y_pred_proba_bootstrap : np.ndarray
        Bootstrap predictions (n_bootstrap, n_subjects)
    n_subjects : int
        Number of subjects
    n_bootstrap : int
        Number of bootstrap samples
    metadata : dict, optional
        Additional metadata (outlier_method, imputation_method, classifier)
    """

    combo_id: str
    y_true: np.ndarray
    y_pred_proba_original: np.ndarray
    y_pred_proba_bootstrap: np.ndarray
    n_subjects: int
    n_bootstrap: int
    metadata: Optional[Dict[str, Any]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "combo_id": self.combo_id,
            "n_subjects": self.n_subjects,
            "n_bootstrap": self.n_bootstrap,
            "y_true": self.y_true.tolist(),
            "y_pred_proba_original": self.y_pred_proba_original.tolist(),
            "y_pred_proba_bootstrap": self.y_pred_proba_bootstrap.tolist(),
            "metadata": self.metadata or {},
        }


@dataclass
class InstabilityMetrics:
    """
    Instability metrics per subject.

    Attributes
    ----------
    mape : np.ndarray
        Mean Absolute Prediction Error per subject (n_subjects,)
    cii : np.ndarray
        Classification Instability Index per subject at threshold (n_subjects,)
    prediction_sd : np.ndarray
        Standard deviation of predictions per subject (n_subjects,)
    ci_lower : np.ndarray
        2.5th percentile of predictions (n_subjects,)
    ci_upper : np.ndarray
        97.5th percentile of predictions (n_subjects,)
    threshold : float
        Classification threshold used for CII
    """

    mape: np.ndarray
    cii: np.ndarray
    prediction_sd: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    threshold: float

    @property
    def mean_mape(self) -> float:
        """Mean MAPE across all subjects."""
        return float(np.mean(self.mape))

    @property
    def mean_cii(self) -> float:
        """Mean CII across all subjects."""
        return float(np.mean(self.cii))


def load_bootstrap_predictions_from_mlflow(
    mlruns_dir: Union[str, Path],
    experiment_id: str,
    run_name: str,
) -> Optional[BootstrapPredictionData]:
    """
    Load bootstrap predictions from MLflow artifacts.

    Parameters
    ----------
    mlruns_dir : str or Path
        Path to mlruns directory
    experiment_id : str
        MLflow experiment ID
    run_name : str
        Run name pattern to match

    Returns
    -------
    BootstrapPredictionData or None
        Bootstrap data if found, None otherwise
    """
    import pickle

    mlruns_path = Path(mlruns_dir) / experiment_id

    if not mlruns_path.exists():
        logger.warning(f"Experiment directory not found: {mlruns_path}")
        return None

    # Find matching run
    for run_dir in mlruns_path.iterdir():
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists():
            continue

        try:
            import yaml

            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            if meta.get("run_name", "") != run_name:
                continue
        except Exception:
            continue

        # Found matching run - load artifacts
        arrays_path = run_dir / "artifacts" / "dict_arrays"
        if not arrays_path.exists():
            continue

        pickle_files = list(arrays_path.glob("*.pickle"))
        if not pickle_files:
            continue

        try:
            with open(pickle_files[0], "rb") as f:
                arrays_data = pickle.load(f)

            y_true = arrays_data.get("y_test", np.array([]))
            # Bootstrap predictions: try different naming conventions
            y_bootstrap = None
            for key in ["y_pred_proba_bootstrap", "y_pred_proba", "predictions"]:
                if key in arrays_data:
                    y_bootstrap = arrays_data[key]
                    break

            y_mean = arrays_data.get("y_pred_proba_mean", None)

            if len(y_true) == 0 or y_bootstrap is None:
                continue

            # Ensure correct shape: (n_bootstrap, n_subjects)
            if y_bootstrap.ndim == 1:
                # Single prediction, reshape
                y_bootstrap = y_bootstrap.reshape(1, -1)
            elif y_bootstrap.shape[0] < y_bootstrap.shape[1]:
                # Likely (n_subjects, n_bootstrap), transpose
                y_bootstrap = y_bootstrap.T

            n_bootstrap, n_subjects = y_bootstrap.shape

            # Original prediction: use mean if available, else first bootstrap
            if y_mean is not None:
                y_original = y_mean
            else:
                y_original = np.mean(y_bootstrap, axis=0)

            return BootstrapPredictionData(
                combo_id=run_name,
                y_true=y_true,
                y_pred_proba_original=y_original,
                y_pred_proba_bootstrap=y_bootstrap,
                n_subjects=n_subjects,
                n_bootstrap=n_bootstrap,
                metadata={"run_id": run_dir.name},
            )

        except Exception as e:
            logger.warning(f"Failed to load {pickle_files[0]}: {e}")
            continue

    return None


def compute_prediction_instability_metrics(
    data: BootstrapPredictionData,
    threshold: float = 0.5,
) -> InstabilityMetrics:
    """
    Compute prediction instability metrics (Riley 2023).

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    threshold : float, default 0.5
        Classification threshold for CII computation

    Returns
    -------
    InstabilityMetrics
        Per-subject instability metrics
    """
    y_original = data.y_pred_proba_original
    y_bootstrap = data.y_pred_proba_bootstrap

    # MAPE: Mean Absolute Prediction Error per subject
    # MAPE_i = mean(|p_bootstrap - p_original|) across bootstrap samples
    mape = np.mean(np.abs(y_bootstrap - y_original), axis=0)

    # Standard deviation of predictions per subject
    prediction_sd = np.std(y_bootstrap, axis=0)

    # Confidence intervals (2.5th and 97.5th percentiles)
    ci_lower = np.percentile(y_bootstrap, 2.5, axis=0)
    ci_upper = np.percentile(y_bootstrap, 97.5, axis=0)

    # Classification Instability Index (CII)
    # CII_i = proportion of bootstrap samples with different classification
    y_class_original = (y_original >= threshold).astype(int)
    y_class_bootstrap = (y_bootstrap >= threshold).astype(int)
    different_class = (y_class_bootstrap != y_class_original).astype(float)
    cii = np.mean(different_class, axis=0)

    return InstabilityMetrics(
        mape=mape,
        cii=cii,
        prediction_sd=prediction_sd,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        threshold=threshold,
    )


def compute_per_patient_uncertainty(
    data: BootstrapPredictionData,
) -> Dict[str, Any]:
    """
    Compute per-patient uncertainty summary (Kompa 2021).

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data

    Returns
    -------
    dict
        Per-patient uncertainty summary including:
        - patient_ids: list of patient indices
        - mean_pred: mean prediction per patient
        - sd_pred: standard deviation per patient
        - ci_lower: 2.5th percentile
        - ci_upper: 97.5th percentile
        - y_true: true outcome
    """
    y_bootstrap = data.y_pred_proba_bootstrap

    return {
        "n_patients": data.n_subjects,
        "n_bootstrap": data.n_bootstrap,
        "patient_ids": list(range(data.n_subjects)),
        "mean_pred": np.mean(y_bootstrap, axis=0).tolist(),
        "sd_pred": np.std(y_bootstrap, axis=0).tolist(),
        "ci_lower": np.percentile(y_bootstrap, 2.5, axis=0).tolist(),
        "ci_upper": np.percentile(y_bootstrap, 97.5, axis=0).tolist(),
        "y_true": data.y_true.tolist(),
    }


def export_pminternal_data(
    data: BootstrapPredictionData,
    output_path: Union[str, Path],
) -> None:
    """
    Export bootstrap data in pminternal-compatible JSON format.

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    output_path : str or Path
        Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data.to_json_dict(), f, indent=2)

    logger.info(f"Exported pminternal data to {output_path}")


def create_prediction_instability_plot_data(
    data: BootstrapPredictionData,
    subsample: int = 200,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Create data for Riley 2023 style prediction instability plot.

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    subsample : int, default 200
        Number of bootstrap samples to include in plot (for performance)
    random_seed : int, default 42
        Random seed for reproducible subsampling

    Returns
    -------
    dict
        Plot data including:
        - scatter: dict with 'x' (original predictions) and 'y' (bootstrap predictions)
        - percentile_lines: dict with 'x', 'p_2_5', 'p_97_5' for percentile lines
        - diagonal: dict with 'min', 'max' for perfect agreement line
        - metadata: dict with 'n_subjects', 'n_bootstrap', 'n_points_plotted'
    """
    y_original = data.y_pred_proba_original
    y_bootstrap = data.y_pred_proba_bootstrap

    # Subsample bootstrap iterations for plotting (with reproducible seed)
    if data.n_bootstrap > subsample:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(data.n_bootstrap, subsample, replace=False)
        y_bootstrap_sub = y_bootstrap[indices, :]
    else:
        y_bootstrap_sub = y_bootstrap

    # Create scatter data efficiently using numpy broadcasting
    n_bootstrap_sub = y_bootstrap_sub.shape[0]
    x_scatter = np.repeat(y_original, n_bootstrap_sub).tolist()
    y_scatter = y_bootstrap_sub.T.flatten().tolist()

    # Compute percentile lines
    p_2_5 = np.percentile(y_bootstrap, 2.5, axis=0)
    p_97_5 = np.percentile(y_bootstrap, 97.5, axis=0)

    # Sort by original prediction for line plotting
    sort_idx = np.argsort(y_original)

    return {
        "scatter": {
            "x": x_scatter,
            "y": y_scatter,
        },
        "percentile_lines": {
            "x": y_original[sort_idx].tolist(),
            "p_2_5": p_2_5[sort_idx].tolist(),
            "p_97_5": p_97_5[sort_idx].tolist(),
        },
        "diagonal": {
            "min": 0.0,
            "max": 1.0,
        },
        "metadata": {
            "n_subjects": data.n_subjects,
            "n_bootstrap": data.n_bootstrap,
            "n_points_plotted": len(x_scatter),
        },
    }
