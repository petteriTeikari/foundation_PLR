#!/usr/bin/env python3
"""
Extract SYNTHETIC configurations from MLflow to DuckDB.

This is the synthetic-data-only extraction script.
Part of the 4-gate isolation architecture.

CRITICAL: This script ONLY extracts synthetic runs.
Production extraction uses extract_all_configs_to_duckdb.py.

Synthetic runs are identified by ANY of:
1. Run name prefix: __SYNTHETIC_
2. Experiment name prefix: synth_
3. Tag: is_synthetic=true or data_source=synthetic

Output:
    outputs/synthetic/synthetic_foundation_plr_results.db

Usage:
    python scripts/extract_synthetic_to_duckdb.py

Author: Foundation PLR Team
Date: 2026-02-02
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import data mode utilities
from src.utils.data_mode import (  # noqa: E402
    SYNTHETIC_RUN_PREFIX,
    get_synthetic_output_dir,
    is_synthetic_experiment_name,
    is_synthetic_run_name,
)

# Import display names lookup
from src.data_io.display_names import (  # noqa: E402
    get_classifier_display_name,
    get_imputation_display_name,
    get_outlier_display_name,
)

# Import registry validation (CRITICAL: validates parsed method names)
from src.data_io.registry import (  # noqa: E402
    validate_classifier,
    validate_imputation_method,
    validate_outlier_method,
)

# Configuration - use centralized path utilities
from src.utils.paths import get_mlruns_dir  # noqa: E402

MLRUNS_DIR = get_mlruns_dir()
OUTPUT_DIR = get_synthetic_output_dir()
OUTPUT_DB = OUTPUT_DIR / "synthetic_foundation_plr_results.db"

# Classifier name mapping: MLflow names → Registry names
CLASSIFIER_NAME_MAP = {
    "CATBOOST": "CatBoost",
    "XGBOOST": "XGBoost",
    "TABPFN": "TabPFN",
    "TABM": "TabM",
    "LogisticRegression": "LogisticRegression",
}


def parse_run_name(run_name: str) -> dict[str, str]:
    """
    Parse MLflow run name to extract configuration.

    Handles both synthetic-prefixed and non-prefixed names.

    Parameters
    ----------
    run_name : str
        The MLflow run name (may have __SYNTHETIC_ prefix).

    Returns
    -------
    dict[str, str]
        Dictionary with classifier, featurization, imputation, outlier keys.
    """
    # Remove synthetic prefix if present
    if run_name.startswith(SYNTHETIC_RUN_PREFIX):
        run_name = run_name[len(SYNTHETIC_RUN_PREFIX) :]

    # Remove other common prefixes
    name = run_name.replace("model_", "").replace(".pickle", "")

    # Split by double underscore
    parts = name.split("__")

    result = {
        "classifier": "",
        "featurization": "",
        "imputation": "",
        "outlier": "",
    }

    if len(parts) >= 1:
        classifier_parts = parts[0].split("_")
        raw_classifier = classifier_parts[0]
        result["classifier"] = CLASSIFIER_NAME_MAP.get(raw_classifier, raw_classifier)

    if len(parts) >= 2:
        result["featurization"] = parts[1]

    if len(parts) >= 3:
        result["imputation"] = parts[2]

    if len(parts) >= 4:
        result["outlier"] = parts[3]

    # Validate parsed methods against registry (log warnings for invalid)
    if result["outlier"] and not validate_outlier_method(result["outlier"]):
        print(
            f"    Warning: Unregistered outlier method '{result['outlier']}' in: {name}"
        )

    if result["imputation"] and not validate_imputation_method(result["imputation"]):
        print(
            f"    Warning: Unregistered imputation method '{result['imputation']}' in: {name}"
        )

    if result["classifier"] and not validate_classifier(result["classifier"]):
        print(
            f"    Warning: Unregistered classifier '{result['classifier']}' in: {name}"
        )

    return result


def get_run_params(run_dir: Path) -> dict[str, Any]:
    """Read MLflow params from run directory."""
    params = {}
    params_dir = run_dir / "params"

    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                try:
                    params[param_file.name] = param_file.read_text().strip()
                except Exception:
                    pass

    return params


def get_run_tags(run_dir: Path) -> dict[str, str]:
    """Read MLflow tags from run directory."""
    tags = {}
    tags_dir = run_dir / "tags"

    if tags_dir.exists():
        for tag_file in tags_dir.iterdir():
            if tag_file.is_file():
                try:
                    tags[tag_file.name] = tag_file.read_text().strip()
                except Exception:
                    pass

    return tags


def is_synthetic_run(run_name: str, experiment_name: str, tags: dict[str, str]) -> bool:
    """
    Check if a run is synthetic.

    Parameters
    ----------
    run_name : str
        MLflow run name.
    experiment_name : str
        MLflow experiment name.
    tags : dict
        MLflow run tags.

    Returns
    -------
    bool
        True if run is synthetic.
    """
    # Check run name prefix
    if is_synthetic_run_name(run_name):
        return True

    # Check experiment name
    if is_synthetic_experiment_name(experiment_name):
        return True

    # Check tags
    if tags.get("is_synthetic", "").lower() == "true":
        return True
    if tags.get("data_source", "").lower() == "synthetic":
        return True

    return False


def extract_bootstrap_metrics(metrics_pickle_path: Path) -> dict[str, Any]:
    """Extract bootstrap metrics from a metrics pickle file."""
    result = {
        "auroc_mean": None,
        "auroc_std": None,
        "auroc_ci_lo": None,
        "auroc_ci_hi": None,
        "n_bootstrap": None,
        "brier_mean": None,
        "brier_std": None,
    }

    if not metrics_pickle_path.exists():
        return result

    try:
        with open(metrics_pickle_path, "rb") as f:
            data = pickle.load(f)

        if "metrics_stats" in data and "test" in data["metrics_stats"]:
            test_stats = data["metrics_stats"]["test"]

            if "metrics" in test_stats and "scalars" in test_stats["metrics"]:
                scalars = test_stats["metrics"]["scalars"]

                if "AUROC" in scalars:
                    auroc_stats = scalars["AUROC"]
                    result["auroc_mean"] = float(auroc_stats.get("mean", 0))
                    result["auroc_std"] = float(auroc_stats.get("std", 0))
                    result["n_bootstrap"] = int(auroc_stats.get("n", 1000))

                    ci = auroc_stats.get("ci")
                    if ci is not None and len(ci) >= 2:
                        result["auroc_ci_lo"] = float(ci[0])
                        result["auroc_ci_hi"] = float(ci[1])

                if "Brier" in scalars:
                    brier_stats = scalars["Brier"]
                    result["brier_mean"] = float(brier_stats.get("mean", 0))
                    result["brier_std"] = float(brier_stats.get("std", 0))

    except Exception as e:
        print(f"    Warning: Could not parse metrics pickle: {e}")

    return result


def find_synthetic_experiments(mlruns_dir: Path) -> list[tuple[str, str]]:
    """
    Find all synthetic experiments in MLflow.

    Returns
    -------
    list[tuple[str, str]]
        List of (experiment_id, experiment_name) tuples.
    """
    experiments = []

    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        # Check for meta.yaml to get experiment name
        meta_file = exp_dir / "meta.yaml"
        if not meta_file.exists():
            continue

        try:
            import yaml

            with open(meta_file) as f:
                meta = yaml.safe_load(f)
            exp_name = meta.get("name", "")

            if is_synthetic_experiment_name(exp_name):
                experiments.append((exp_dir.name, exp_name))

        except Exception:
            continue

    return experiments


def scan_synthetic_runs(mlruns_dir: Path) -> list[dict[str, Any]]:
    """
    Scan all synthetic runs in MLflow.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries with run metadata and metrics.
    """
    # Find synthetic experiments
    synthetic_experiments = find_synthetic_experiments(mlruns_dir)
    print(f"Found {len(synthetic_experiments)} synthetic experiments")

    runs = []
    run_count = 0
    valid_count = 0

    for experiment_id, experiment_name in synthetic_experiments:
        experiment_dir = mlruns_dir / experiment_id
        print(f"\nScanning {experiment_name} ({experiment_dir})...")

        for run_dir in sorted(experiment_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue

            # Check for model artifacts
            model_dir = run_dir / "artifacts" / "model"
            metrics_dir = run_dir / "artifacts" / "metrics"

            if not model_dir.exists():
                continue

            # Find any model pickle file
            model_pickles = list(model_dir.glob("model_*.pickle"))
            if not model_pickles:
                continue

            model_file = model_pickles[0]
            run_count += 1

            # Get run tags
            tags = get_run_tags(run_dir)

            # Get run name from tags or infer from model file
            run_name = tags.get("mlflow.runName", model_file.stem)

            # Verify this is indeed a synthetic run
            if not is_synthetic_run(run_name, experiment_name, tags):
                continue

            # Parse configuration from filename
            config = parse_run_name(run_name)

            # Read params
            params = get_run_params(run_dir)

            # Find and parse metrics pickle
            metrics_data = {
                "auroc_mean": None,
                "auroc_ci_lo": None,
                "auroc_ci_hi": None,
                "n_bootstrap": None,
                "brier_mean": None,
            }

            if metrics_dir.exists():
                metrics_pickles = list(metrics_dir.glob("metrics_*.pickle"))
                if metrics_pickles:
                    metrics_data = extract_bootstrap_metrics(metrics_pickles[0])

            runs.append(
                {
                    "run_id": run_dir.name,
                    "model_path": str(model_file),
                    "experiment_name": experiment_name,
                    "run_name": run_name,
                    "outlier_method": config["outlier"],
                    "imputation_method": config["imputation"],
                    "featurization": config["featurization"],
                    "classifier": config["classifier"],
                    "outlier_display_name": get_outlier_display_name(config["outlier"]),
                    "imputation_display_name": get_imputation_display_name(
                        config["imputation"]
                    ),
                    "classifier_display_name": get_classifier_display_name(
                        config["classifier"]
                    ),
                    "auroc": metrics_data["auroc_mean"],
                    "auroc_ci_lo": metrics_data["auroc_ci_lo"],
                    "auroc_ci_hi": metrics_data["auroc_ci_hi"],
                    "brier": metrics_data["brier_mean"],
                    "n_bootstrap": metrics_data["n_bootstrap"],
                    "is_synthetic": True,  # Always true for this script
                    "anomaly_source": params.get("anomaly_source", config["outlier"]),
                }
            )

            valid_count += 1

    print(f"\nScanned {run_count} total runs, {valid_count} valid synthetic runs")
    return runs


def create_synthetic_database(runs: list[dict[str, Any]], output_db: Path) -> None:
    """
    Create DuckDB database for synthetic results.

    Parameters
    ----------
    runs : list[dict[str, Any]]
        List of run dictionaries.
    output_db : Path
        Output database path.
    """
    output_db.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if output_db.exists():
        output_db.unlink()

    conn = duckdb.connect(str(output_db))

    # Create table
    conn.execute(
        """
        CREATE TABLE synthetic_metrics (
            run_id VARCHAR,
            model_path VARCHAR,
            experiment_name VARCHAR,
            run_name VARCHAR,
            outlier_method VARCHAR,
            imputation_method VARCHAR,
            featurization VARCHAR,
            classifier VARCHAR,
            outlier_display_name VARCHAR,
            imputation_display_name VARCHAR,
            classifier_display_name VARCHAR,
            auroc DOUBLE,
            auroc_ci_lo DOUBLE,
            auroc_ci_hi DOUBLE,
            brier DOUBLE,
            n_bootstrap INTEGER,
            is_synthetic BOOLEAN,
            anomaly_source VARCHAR,
            extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insert runs
    for run in runs:
        conn.execute(
            """
            INSERT INTO synthetic_metrics (
                run_id, model_path, experiment_name, run_name,
                outlier_method, imputation_method, featurization, classifier,
                outlier_display_name, imputation_display_name, classifier_display_name,
                auroc, auroc_ci_lo, auroc_ci_hi, brier, n_bootstrap,
                is_synthetic, anomaly_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run["run_id"],
                run["model_path"],
                run["experiment_name"],
                run["run_name"],
                run["outlier_method"],
                run["imputation_method"],
                run["featurization"],
                run["classifier"],
                run["outlier_display_name"],
                run["imputation_display_name"],
                run["classifier_display_name"],
                run["auroc"],
                run["auroc_ci_lo"],
                run["auroc_ci_hi"],
                run["brier"],
                run["n_bootstrap"],
                run["is_synthetic"],
                run["anomaly_source"],
            ),
        )

    # Add metadata table
    conn.execute(
        """
        CREATE TABLE _metadata (
            key VARCHAR,
            value VARCHAR
        )
    """
    )

    conn.execute(
        "INSERT INTO _metadata VALUES (?, ?)",
        ("extraction_timestamp", datetime.now().isoformat()),
    )
    conn.execute("INSERT INTO _metadata VALUES (?, ?)", ("is_synthetic", "true"))
    conn.execute(
        "INSERT INTO _metadata VALUES (?, ?)",
        ("source", "extract_synthetic_to_duckdb.py"),
    )
    conn.execute("INSERT INTO _metadata VALUES (?, ?)", ("total_runs", str(len(runs))))

    conn.close()

    print(f"\nCreated: {output_db}")
    print(f"  Total runs: {len(runs)}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SYNTHETIC DATA EXTRACTION")
    print("=" * 60)
    print(f"\nMLflow directory: {MLRUNS_DIR}")
    print(f"Output database: {OUTPUT_DB}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan synthetic runs
    runs = scan_synthetic_runs(MLRUNS_DIR)

    if not runs:
        print("\nNo synthetic runs found!")
        print("To create synthetic runs, use:")
        print("  python src/pipeline_PLR.py --config-name=synthetic_run")
        return

    # Create database
    create_synthetic_database(runs, OUTPUT_DB)

    print("\nSynthetic extraction complete!")


if __name__ == "__main__":
    main()
