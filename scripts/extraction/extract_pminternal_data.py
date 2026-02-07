#!/usr/bin/env python3
"""
Extract bootstrap predictions from MLflow for pminternal instability plots.

This script extracts per-patient bootstrap predictions from MLflow pickle files
and exports them to JSON format for R consumption.

Reference: Riley RD et al. (2023) "Clinical prediction models and the multiverse of madness"

Output: data/r_data/pminternal_bootstrap_predictions.json
"""

import json
import pickle

# Constants - use centralized path utilities
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.paths import get_classification_experiment_id, get_mlruns_dir

MLFLOW_BASE = get_mlruns_dir() / get_classification_experiment_id()
OUTPUT_PATH = Path("data/r_data/pminternal_bootstrap_predictions.json")
COMBOS_PATH = Path("configs/VISUALIZATION/plot_hyperparam_combos.yaml")


def load_target_configs() -> dict[str, dict]:
    """
    Load target configurations from plot_hyperparam_combos.yaml.

    Uses standard_combos + deep_learning from extended_combos.
    Fixes classifier to CATBOOST for consistency in comparing preprocessing
    effects (per CLAUDE.md: "FIX the classifier").

    Returns:
        Dictionary mapping combo_id to config dict with keys:
        outlier, imputation, classifier, featurization
    """
    with open(COMBOS_PATH) as f:
        yaml_data = yaml.safe_load(f)

    configs = {}

    # Load standard combos (ground_truth, best_ensemble, best_single_fm, traditional)
    for combo in yaml_data["standard_combos"]:
        configs[combo["id"]] = {
            "outlier": combo["outlier_method"],
            "imputation": combo["imputation_method"],
            # Fix classifier to CATBOOST for consistent preprocessing comparison
            "classifier": "CATBOOST",
            "featurization": "simple1.0",
        }

    # Add deep_learning from extended_combos (timesnet_full)
    for combo in yaml_data["extended_combos"]:
        if combo["id"] == "timesnet_full":
            configs["deep_learning"] = {
                "outlier": combo["outlier_method"],
                "imputation": combo["imputation_method"],
                "classifier": "CATBOOST",
                "featurization": "simple1.0",
            }
            break

    return configs


def find_pickle_file(
    outlier: str, imputation: str, classifier: str, featurization: str
) -> Path | None:
    """
    Find MLflow pickle file matching the given configuration.

    Filename pattern: metrics_{classifier}_eval-auc__{featurization}__{imputation}__{outlier}.pickle
    """
    # Build pattern - note the double underscores
    pattern = f"metrics_{classifier}_eval-auc__{featurization}__{imputation}__{outlier}.pickle"

    # Search all runs
    matches = list(MLFLOW_BASE.glob(f"*/artifacts/metrics/{pattern}"))

    if matches:
        return matches[0]

    # Try with flexible pattern if exact match fails
    flexible_pattern = (
        f"metrics_{classifier}*{featurization}*{imputation}*{outlier}.pickle"
    )
    matches = list(MLFLOW_BASE.glob(f"*/artifacts/metrics/{flexible_pattern}"))

    if matches:
        return matches[0]

    return None


def extract_bootstrap_predictions(pickle_path: Path) -> dict[str, Any]:
    """
    Extract bootstrap predictions from a single pickle file.

    Returns:
        dict with keys:
        - y_true: ground truth labels (n_patients,)
        - y_prob_original: mean predictions (n_patients,)
        - y_prob_bootstrap: bootstrap predictions (n_bootstrap, n_patients)
        - n_patients: number of patients
        - n_bootstrap: number of bootstrap iterations
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Extract predictions: shape (n_patients, n_bootstrap)
    preds = data["metrics_iter"]["test"]["preds"]["arrays"]["predictions"][
        "y_pred_proba"
    ]

    # Extract labels
    labels = data["subjectwise_stats"]["test"]["labels"]

    # Compute original model prediction as mean across bootstraps
    y_prob_original = preds.mean(axis=1)

    # Transpose predictions to (n_bootstrap, n_patients) for R
    y_prob_bootstrap = preds.T

    return {
        "y_true": labels.tolist(),
        "y_prob_original": y_prob_original.tolist(),
        "y_prob_bootstrap": y_prob_bootstrap.tolist(),
        "n_patients": int(preds.shape[0]),
        "n_bootstrap": int(preds.shape[1]),
    }


def export_to_json(configs_data: dict[str, dict], output_path: Path) -> None:
    """
    Export extracted data to JSON for R consumption.

    Schema:
    {
        "metadata": {...},
        "configs": {
            "config_id": {...},
            ...
        }
    }
    """
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "generator": "scripts/extract_pminternal_data.py",
            "data_source": {
                "mlflow_experiment": str(MLFLOW_BASE),
                "description": "Bootstrap predictions for pminternal instability plots",
            },
            "reference": "Riley RD et al. (2023) BMC Medicine 21:502",
        },
        "configs": {},
    }

    for config_id, data in configs_data.items():
        output["configs"][config_id] = {
            "config_id": config_id,
            **data,
        }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported to {output_path}")
    print(f"  Configs: {list(output['configs'].keys())}")
    for config_id, data in output["configs"].items():
        print(
            f"  - {config_id}: {data['n_patients']} patients, {data['n_bootstrap']} bootstraps"
        )


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("pminternal Bootstrap Prediction Extraction")
    print("=" * 60)

    # Load combos from YAML (single source of truth)
    target_configs = load_target_configs()
    print(f"Loaded {len(target_configs)} combos from {COMBOS_PATH}")

    extracted_configs = {}

    for config_id, config in target_configs.items():
        print(f"\nProcessing: {config_id}")
        print(f"  Outlier: {config['outlier']}")
        print(f"  Imputation: {config['imputation']}")
        print(f"  Classifier: {config['classifier']}")

        pickle_path = find_pickle_file(
            outlier=config["outlier"],
            imputation=config["imputation"],
            classifier=config["classifier"],
            featurization=config["featurization"],
        )

        if pickle_path is None:
            print("  WARNING: Pickle file not found, skipping")
            continue

        print(f"  Found: {pickle_path.name}")

        try:
            data = extract_bootstrap_predictions(pickle_path)
            extracted_configs[config_id] = data
            print(
                f"  Extracted: {data['n_patients']} patients × {data['n_bootstrap']} bootstraps"
            )

            # Validate
            assert all(0 <= p <= 1 for p in data["y_prob_original"]), (
                "Invalid probabilities"
            )
            print("  Validated: probabilities in [0, 1]")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not extracted_configs:
        print("\nERROR: No configs extracted!")
        return 1

    # Export to JSON
    print("\n" + "=" * 60)
    print("Exporting to JSON")
    print("=" * 60)

    export_to_json(extracted_configs, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
