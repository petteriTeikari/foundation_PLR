#!/usr/bin/env python3
"""
Extract Top-10 CatBoost Models with Full Artifacts for SHAP Analysis.

This is Phase 1.2 of the end-to-end visualization pipeline.
Extracts models, features, and metadata for the top-10 CatBoost configurations.

EXCLUSION CRITERIA:
- Configs where outlier_method in ["anomaly", "exclude"] are excluded
- These represent unknown/legacy outlier detection sources
- See docs/mlflow-naming-convention.md for full explanation

MODEL STRUCTURE:
- MLflow model pickles contain: list of 1000 bootstrap `ClassificationEnsembleSGLB`
- Each ensemble contains: list of 2 `CatBoostClassifier` models
- For SHAP, we extract the inner CatBoostClassifier from each bootstrap

Usage:
    python scripts/extract_top10_models_with_artifacts.py

Output:
    outputs/top10_catboost_models.pkl

Author: Foundation PLR Team
Date: 2026-01-25
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

# Add src to path for unpickling the custom model class
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
OUTPUT_PKL = Path("outputs/top10_catboost_models.pkl")
DB_PATH = Path("outputs/foundation_plr_results.db")

# Exclusion criteria - configs with these outlier methods are excluded
EXCLUDED_OUTLIER_METHODS = ["anomaly", "exclude"]


def get_top10_configs(db_path: Path) -> list[dict[str, Any]]:
    """
    Get top-10 CatBoost configurations from DuckDB.

    Uses the pre-computed top10_catboost view which excludes
    configs with unknown outlier sources.
    """
    conn = duckdb.connect(str(db_path), read_only=True)

    # Get top 10 from pre-filtered view
    rows = conn.execute("""
        SELECT
            rank,
            run_id,
            outlier_method,
            imputation_method,
            auroc,
            auroc_ci_lo,
            auroc_ci_hi
        FROM top10_catboost
        ORDER BY rank
    """).fetchall()

    # Also get model_path for each
    configs = []
    for row in rows:
        rank, run_id, outlier, imputation, auroc, ci_lo, ci_hi = row

        # Get full row including model_path
        full_row = conn.execute(
            """
            SELECT model_path, featurization, n_bootstrap
            FROM essential_metrics
            WHERE run_id = ?
        """,
            [run_id],
        ).fetchone()

        if full_row:
            model_path, featurization, n_bootstrap = full_row
            configs.append(
                {
                    "rank": rank,
                    "run_id": run_id,
                    "outlier_method": outlier,
                    "imputation_method": imputation,
                    "featurization": featurization,
                    "auroc_mean": auroc,
                    "auroc_ci_lo": ci_lo,
                    "auroc_ci_hi": ci_hi,
                    "n_bootstrap": n_bootstrap,
                    "model_path": model_path,
                }
            )

    conn.close()
    return configs


def extract_catboost_from_ensemble(ensemble_model: Any) -> Any:
    """
    Extract the inner CatBoostClassifier from a ClassificationEnsembleSGLB.

    The custom ensemble class wraps 2 CatBoostClassifier models.
    We return the first one for SHAP analysis (they're similar).
    """
    if hasattr(ensemble_model, "ensemble") and len(ensemble_model.ensemble) > 0:
        return ensemble_model.ensemble[0]  # Return first CatBoostClassifier
    return ensemble_model  # Already a CatBoost or unknown structure


def load_model_and_features(config: dict[str, Any]) -> dict[str, Any]:
    """
    Load model and feature arrays for a configuration.

    MODEL STRUCTURE:
    - MLflow pickle contains: list of 1000 bootstrap ClassificationEnsembleSGLB
    - Each ensemble contains: list of 2 CatBoostClassifier
    - We extract: list of 1000 CatBoostClassifier (one per bootstrap)

    Returns dictionary with:
    - bootstrap_models: list of 1000 CatBoostClassifier (for per-bootstrap SHAP)
    - model: First CatBoostClassifier (for quick testing)
    - X_train, X_test, y_train, y_test: Feature arrays
    - feature_names: List of feature names
    """
    result = {
        "config": config,
        "bootstrap_models": None,  # List of 1000 CatBoostClassifier
        "model": None,  # First model for quick testing
        "n_bootstrap": 0,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "feature_names": None,
        "errors": [],
    }

    model_path = Path(config["model_path"])

    # Load model
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model_list = pickle.load(f)

            # model_list is a list of 1000 bootstrap ClassificationEnsembleSGLB
            if isinstance(model_list, list):
                result["n_bootstrap"] = len(model_list)

                # Extract inner CatBoostClassifier from each ensemble
                catboost_models = []
                for ensemble_model in model_list:
                    inner = extract_catboost_from_ensemble(ensemble_model)
                    catboost_models.append(inner)

                result["bootstrap_models"] = catboost_models
                result["model"] = catboost_models[0]  # First for quick access
            else:
                result["model"] = model_list

        except Exception as e:
            result["errors"].append(f"Model load failed: {e}")
    else:
        result["errors"].append(f"Model file not found: {model_path}")

    # Find and load feature arrays
    # They're in artifacts/dict_arrays/ directory
    run_dir = model_path.parent.parent.parent  # model_path is in artifacts/model/
    arrays_dir = run_dir / "artifacts" / "dict_arrays"

    if arrays_dir.exists():
        # Find the matching pickle file
        model_stem = model_path.stem.replace("model_", "dictArrays_")
        array_files = list(arrays_dir.glob(f"{model_stem}*.pickle"))
        if not array_files:
            array_files = list(arrays_dir.glob("*.pickle"))

        if array_files:
            try:
                with open(array_files[0], "rb") as f:
                    data = pickle.load(f)

                # Extract arrays (keys are lowercase in the pickle)
                # Use explicit key checking to avoid numpy array truth value issues
                result["X_train"] = (
                    data.get("x_train") if "x_train" in data else data.get("X_train")
                )
                result["X_test"] = (
                    data.get("x_test") if "x_test" in data else data.get("X_test")
                )
                result["y_train"] = data.get("y_train")
                result["y_test"] = data.get("y_test")
                result["feature_names"] = data.get("feature_names")

            except Exception as e:
                result["errors"].append(f"Arrays load failed: {e}")
        else:
            result["errors"].append("No dict_arrays pickle found")
    else:
        result["errors"].append(f"dict_arrays dir not found: {arrays_dir}")

    return result


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1.2: Extract Top-10 CatBoost Models with Artifacts")
    print("=" * 60)

    # Check database exists
    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        print("Run Phase 1.1 first: python scripts/extract_all_configs_to_duckdb.py")
        return

    # Get top-10 configs
    print("\nLoading top-10 configurations from DuckDB...")
    configs = get_top10_configs(DB_PATH)
    print(f"Found {len(configs)} top-10 CatBoost configs")

    # Display configs
    print("\nTop-10 configurations:")
    for cfg in configs:
        od_short = (
            cfg["outlier_method"][:35] + "..."
            if len(cfg["outlier_method"]) > 35
            else cfg["outlier_method"]
        )
        print(
            f"  Rank {cfg['rank']:2d}: AUROC={cfg['auroc_mean']:.3f} | {od_short} + {cfg['imputation_method']}"
        )

    # Load models and features
    print("\nLoading models and feature arrays...")
    loaded_models = []
    for cfg in configs:
        print(f"  Loading rank {cfg['rank']}...", end=" ")
        loaded = load_model_and_features(cfg)
        loaded_models.append(loaded)

        if loaded["errors"]:
            print(f"ERRORS: {loaded['errors']}")
        else:
            model_info = f"model={type(loaded['model']).__name__}"
            bootstrap_info = f"bootstrap={loaded['n_bootstrap']}"
            feat_info = f"features={loaded['X_test'].shape if loaded['X_test'] is not None else 'N/A'}"
            print(f"OK ({model_info}, {bootstrap_info}, {feat_info})")

    # Compile artifact
    artifact = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "source_db": str(DB_PATH),
            "n_configs": len(loaded_models),
            "exclusion_criteria": {
                "excluded_outlier_methods": EXCLUDED_OUTLIER_METHODS,
                "reason": "Unknown/legacy outlier detection sources",
                "documentation": "docs/mlflow-naming-convention.md",
            },
        },
        "configs": loaded_models,
    }

    # Save
    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved artifact to: {OUTPUT_PKL}")
    print(f"File size: {OUTPUT_PKL.stat().st_size / 1024 / 1024:.1f} MB")

    # Summarize
    n_with_model = sum(1 for m in loaded_models if m["model"] is not None)
    n_with_features = sum(1 for m in loaded_models if m["X_test"] is not None)
    print("\nSummary:")
    print(f"  Models loaded: {n_with_model}/10")
    print(f"  Feature arrays loaded: {n_with_features}/10")

    if n_with_model < 10 or n_with_features < 10:
        print("\nWARNING: Some models or features failed to load!")
        for m in loaded_models:
            if m["errors"]:
                print(f"  Rank {m['config']['rank']}: {m['errors']}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Verify the artifact:
   python -c "import pickle; d=pickle.load(open('outputs/top10_catboost_models.pkl','rb')); print(len(d['configs']))"

2. Proceed to Phase 2.1: Compute SHAP values
   python scripts/compute_shap_values.py
""")


if __name__ == "__main__":
    main()
