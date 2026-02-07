#!/usr/bin/env python
"""
Generate prediction instability figures from MLflow data.

Generates:
- Riley 2023 style prediction instability scatter plots
- Kompa 2021 style per-patient uncertainty distributions
- MAPE histogram comparisons

Cross-references:
- Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"
- Kompa B et al. (2021) "Second opinion needed: communicating uncertainty in medical ML"
- planning/latent-method-results-update.md (Section 26)
"""

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless backend - must be before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Add viz to path
sys.path.insert(0, str(Path(__file__).parent))

from fig_instability_plots import (  # noqa: E402
    plot_instability_comparison,
    plot_mape_histogram,
    plot_per_patient_uncertainty,
)
from plot_config import FIGURES_DIR, FIXED_CLASSIFIER, save_figure, setup_style  # noqa: E402

from src.stats.pminternal_analysis import (  # noqa: E402
    BootstrapPredictionData,
)

# MLflow experiment configuration - use centralized path utilities
from src.utils.paths import get_classification_experiment_id, get_mlruns_dir  # noqa: E402

MLRUNS_DIR = get_mlruns_dir()
EXPERIMENT_ID = get_classification_experiment_id()


def _load_standard_combos() -> dict:
    """Load standard combos from YAML config.

    Maps YAML fields to MLflow param names:
    - outlier_method -> anomaly_source
    - imputation_method -> imputation_source
    - classifier -> model_name
    """
    import yaml

    combos_path = (
        project_root / "configs" / "VISUALIZATION" / "plot_hyperparam_combos.yaml"
    )
    if not combos_path.exists():
        raise FileNotFoundError(f"Combos config not found: {combos_path}")

    with open(combos_path) as f:
        config = yaml.safe_load(f)

    # Build combos dict from standard_combos in YAML
    result = {}
    for combo in config.get("standard_combos", []):
        combo_id = combo["id"]
        result[combo_id] = {
            "anomaly_source": combo["outlier_method"],
            "imputation_source": combo["imputation_method"],
            "model_name": combo.get("classifier", FIXED_CLASSIFIER),
            "display_name": combo["name"],
        }
    return result


# Lazy load combos from config
_standard_combos_cache = None


def get_standard_combos() -> dict:
    """Get standard combos (lazy loaded from config)."""
    global _standard_combos_cache
    if _standard_combos_cache is None:
        _standard_combos_cache = _load_standard_combos()
    return _standard_combos_cache


# Backward compatibility alias
STANDARD_COMBOS = get_standard_combos()


def find_run_by_combo(
    experiment_dir: Path,
    anomaly_source: str,
    imputation_source: str,
    model_name: str,
) -> Optional[Path]:
    """
    Find MLflow run directory matching the given combo.

    Searches through run directories to find one matching the specified
    outlier method, imputation method, and classifier.

    Parameters
    ----------
    experiment_dir : Path
        Path to MLflow experiment directory
    anomaly_source : str
        Outlier detection method name (MLflow param name: anomaly_source)
    imputation_source : str
        Imputation method name (MLflow param name: imputation_source)
    model_name : str
        Classifier name (MLflow param name: model_name)

    Returns
    -------
    Path or None
        Path to matching run directory, or None if not found
    """
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # MLflow stores params as individual files in a params/ directory
        params_dir = run_dir / "params"
        if not params_dir.exists():
            continue

        # Check for individual param files
        outlier_path = params_dir / "anomaly_source"
        imputation_path = params_dir / "imputation_source"
        classifier_path = params_dir / "model_name"

        if not all(
            p.exists() for p in [outlier_path, imputation_path, classifier_path]
        ):
            continue

        try:
            with open(outlier_path) as f:
                run_outlier = f.read().strip()
            with open(imputation_path) as f:
                run_imputation = f.read().strip()
            with open(classifier_path) as f:
                run_classifier = f.read().strip()

            if (
                run_outlier == anomaly_source
                and run_imputation == imputation_source
                and run_classifier == model_name
            ):
                return run_dir

        except Exception:
            continue

    return None


def load_bootstrap_data_from_run(run_dir: Path) -> Optional[BootstrapPredictionData]:
    """
    Load bootstrap prediction data from an MLflow run directory.

    Parameters
    ----------
    run_dir : Path
        Path to MLflow run directory

    Returns
    -------
    BootstrapPredictionData or None
        Loaded bootstrap data, or None if not found
    """
    # Find metrics pickle file
    metrics_dir = run_dir / "artifacts" / "metrics"
    if not metrics_dir.exists():
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return None

    pickle_files = list(metrics_dir.glob("metrics_*.pickle"))
    if not pickle_files:
        logger.warning(f"No metrics pickle files found in {metrics_dir}")
        return None

    try:
        with open(pickle_files[0], "rb") as f:
            data = pickle.load(f)

        test_data = data.get("metrics_iter", {}).get("test", {})
        predictions = (
            test_data.get("preds", {}).get("arrays", {}).get("predictions", {})
        )

        y_pred_proba = predictions.get("y_pred_proba")
        y_true = predictions.get("label")

        if y_pred_proba is None or y_true is None:
            logger.warning(f"Missing predictions in {pickle_files[0]}")
            return None

        # Shape: (n_subjects, n_bootstrap) -> need to transpose to (n_bootstrap, n_subjects)
        y_pred_proba = y_pred_proba.T
        n_bootstrap, n_subjects = y_pred_proba.shape

        # True labels (just need one column, they're all the same)
        y_true_arr = y_true[:, 0]

        # Original prediction: mean across bootstrap samples
        y_original = np.mean(y_pred_proba, axis=0)

        # Extract combo info from pickle filename
        filename = pickle_files[0].stem
        combo_id = filename.replace("metrics_", "")

        return BootstrapPredictionData(
            combo_id=combo_id,
            y_true=y_true_arr,
            y_pred_proba_original=y_original,
            y_pred_proba_bootstrap=y_pred_proba,
            n_subjects=n_subjects,
            n_bootstrap=n_bootstrap,
            metadata={"run_dir": str(run_dir)},
        )

    except Exception as e:
        logger.error(f"Failed to load data from {pickle_files[0]}: {e}")
        return None


def load_standard_combos() -> Dict[str, BootstrapPredictionData]:
    """
    Load bootstrap data for all standard combos.

    Returns
    -------
    dict
        Mapping from combo name to BootstrapPredictionData
    """
    experiment_dir = MLRUNS_DIR / EXPERIMENT_ID
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return {}

    loaded_data = {}

    for combo_name, combo_config in STANDARD_COMBOS.items():
        logger.info(f"Loading combo: {combo_name}")

        run_dir = find_run_by_combo(
            experiment_dir,
            combo_config["anomaly_source"],
            combo_config["imputation_source"],
            combo_config["model_name"],
        )

        if run_dir is None:
            logger.warning(f"No run found for combo: {combo_name}")
            continue

        data = load_bootstrap_data_from_run(run_dir)
        if data is not None:
            data.metadata["display_name"] = combo_config["display_name"]
            loaded_data[combo_name] = data
            logger.info(f"  Loaded: n={data.n_subjects}, B={data.n_bootstrap}")
        else:
            logger.warning(f"  Failed to load data for: {combo_name}")

    return loaded_data


def generate_instability_comparison_figure(
    data_dict: Dict[str, BootstrapPredictionData],
    combo_names: List[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Generate multi-panel instability comparison figure.

    Parameters
    ----------
    data_dict : dict
        Mapping from combo name to BootstrapPredictionData
    combo_names : list of str, optional
        Which combos to include (default: all)

    Returns
    -------
    fig : plt.Figure
    plot_data : dict
    """
    if combo_names is None:
        combo_names = list(data_dict.keys())

    # Prepare data list for plotting
    data_list = []
    for name in combo_names:
        if name in data_dict:
            display_name = data_dict[name].metadata.get("display_name", name)
            data_list.append((display_name, data_dict[name]))

    if not data_list:
        raise ValueError("No data available for plotting")

    fig, axes, plot_data = plot_instability_comparison(
        data_list,
        figsize=(5 * len(data_list), 5),
        subsample=200,
    )

    return fig, plot_data


def generate_per_patient_uncertainty_figure(
    data: BootstrapPredictionData,
    patient_indices: Optional[List[int]] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Generate per-patient uncertainty distribution figure.

    Parameters
    ----------
    data : BootstrapPredictionData
        Bootstrap prediction data
    patient_indices : list of int, optional
        Specific patients to show

    Returns
    -------
    fig : plt.Figure
    plot_data : dict
    """
    fig, ax, plot_data = plot_per_patient_uncertainty(
        data,
        patient_indices=patient_indices,
    )
    return fig, plot_data


def generate_mape_comparison_figure(
    data_dict: Dict[str, BootstrapPredictionData],
    combo_names: List[str] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Generate MAPE histogram comparison figure.

    Parameters
    ----------
    data_dict : dict
        Mapping from combo name to BootstrapPredictionData
    combo_names : list of str, optional
        Which combos to include

    Returns
    -------
    fig : plt.Figure
    plot_data : dict
    """
    if combo_names is None:
        combo_names = list(data_dict.keys())

    data_list = []
    for name in combo_names:
        if name in data_dict:
            display_name = data_dict[name].metadata.get("display_name", name)
            data_list.append((display_name, data_dict[name]))

    if not data_list:
        raise ValueError("No data available for plotting")

    fig, ax, plot_data = plot_mape_histogram(data_list)
    return fig, plot_data


def main():
    """Generate all instability figures."""
    print("\n" + "=" * 60)
    print("Generating Prediction Instability Figures (Riley 2023 / Kompa 2021)")
    print("=" * 60)

    # Setup
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for standard combos
    print("\n[1/4] Loading bootstrap data from MLflow...")
    data_dict = load_standard_combos()

    if not data_dict:
        print("ERROR: No data loaded. Generating with mock data for testing.")
        # Generate mock data for testing
        np.random.seed(42)
        n_subjects = 63
        n_bootstrap = (
            1000  # TODO: load from config (CLS_EVALUATION.BOOTSTRAP.n_iterations)
        )

        for name, config in list(STANDARD_COMBOS.items())[:3]:
            y_true = np.random.binomial(1, 0.3, n_subjects)
            y_original = np.clip(
                y_true * 0.7
                + (1 - y_true) * 0.3
                + np.random.normal(0, 0.1, n_subjects),
                0.01,
                0.99,
            )
            y_bootstrap = np.clip(
                y_original + np.random.normal(0, 0.08, (n_bootstrap, n_subjects)),
                0.01,
                0.99,
            )

            data_dict[name] = BootstrapPredictionData(
                combo_id=name,
                y_true=y_true,
                y_pred_proba_original=y_original,
                y_pred_proba_bootstrap=y_bootstrap,
                n_subjects=n_subjects,
                n_bootstrap=n_bootstrap,
                metadata={"display_name": config["display_name"]},
            )

    print(f"  Loaded {len(data_dict)} pipeline configurations")

    # Generate figures
    results = {"success": [], "failed": []}

    # Figure 1: Multi-panel instability comparison
    print("\n[2/4] Generating instability comparison figure...")
    try:
        fig, plot_data = generate_instability_comparison_figure(data_dict)
        save_figure(fig, "fig_instability_comparison", data=plot_data)
        plt.close(fig)
        results["success"].append("instability_comparison")
        print("  ✓ Saved: fig_instability_comparison")
    except Exception as e:
        logger.error(f"Failed to generate instability comparison: {e}")
        results["failed"].append(("instability_comparison", str(e)))

    # Figure 2: Per-patient uncertainty (for ground truth combo)
    print("\n[3/4] Generating per-patient uncertainty figure...")
    try:
        if "ground_truth" in data_dict:
            data = data_dict["ground_truth"]
        else:
            data = list(data_dict.values())[0]

        fig, plot_data = generate_per_patient_uncertainty_figure(data)
        save_figure(fig, "fig_per_patient_uncertainty", data=plot_data)
        plt.close(fig)
        results["success"].append("per_patient_uncertainty")
        print("  ✓ Saved: fig_per_patient_uncertainty")
    except Exception as e:
        logger.error(f"Failed to generate per-patient uncertainty: {e}")
        results["failed"].append(("per_patient_uncertainty", str(e)))

    # Figure 3: MAPE histogram comparison
    print("\n[4/4] Generating MAPE histogram comparison...")
    try:
        fig, plot_data = generate_mape_comparison_figure(data_dict)
        save_figure(fig, "fig_mape_comparison", data=plot_data)
        plt.close(fig)
        results["success"].append("mape_comparison")
        print("  ✓ Saved: fig_mape_comparison")
    except Exception as e:
        logger.error(f"Failed to generate MAPE comparison: {e}")
        results["failed"].append(("mape_comparison", str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("INSTABILITY FIGURE GENERATION SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(results['success'])}/3")
    for name in results["success"]:
        print(f"  ✓ {name}")
    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for name, error in results["failed"]:
            print(f"  ✗ {name}: {error[:50]}...")

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
