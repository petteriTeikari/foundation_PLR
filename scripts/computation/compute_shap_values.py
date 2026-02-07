#!/usr/bin/env python3
"""
Compute SHAP Values for Top-10 CatBoost Models.

This is Phase 2 of the end-to-end visualization pipeline.
Computes SHAP values for all 1000 bootstrap models per configuration,
enabling uncertainty quantification in feature importance.

COMPUTATION SCALE:
- 10 configurations × 1000 bootstrap models = 10,000 SHAP computations
- ~63 test samples × ~10 features per computation
- Estimated time: 30-60 minutes depending on hardware

PARALLELIZATION:
- Uses joblib for parallel processing across bootstrap models
- Configurable n_jobs (default: -1 = all cores)

OUTPUT:
- outputs/shap_values_top10.pkl: Full SHAP values for all bootstrap models
- outputs/shap_summary_top10.pkl: Aggregated statistics (mean, std, CI)

Usage:
    python scripts/compute_shap_values.py [--n-jobs N] [--resume]

Author: Foundation PLR Team
Date: 2026-01-25
"""

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to path for model unpickling
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import SHAP
try:
    import shap
except ImportError:
    print("ERROR: shap not installed. Run: uv pip install shap")
    sys.exit(1)

# Configuration
ARTIFACT_PATH = Path("outputs/top10_catboost_models.pkl")
OUTPUT_SHAP_FULL = Path("outputs/shap_values_top10.pkl")
OUTPUT_SHAP_SUMMARY = Path("outputs/shap_summary_top10.pkl")
CHECKPOINT_DIR = Path("outputs/shap_checkpoints")


def compute_shap_for_model(
    model: Any,
    X: np.ndarray,
    model_idx: int,
) -> dict[str, Any]:
    """
    Compute SHAP values for a single model.

    Parameters
    ----------
    model : CatBoostClassifier
        The trained model
    X : np.ndarray
        Feature matrix (test set)
    model_idx : int
        Index of this model (for tracking)

    Returns
    -------
    dict with:
        - shap_values: np.ndarray of shape (n_samples, n_features)
        - expected_value: float (base value)
        - model_idx: int
    """
    try:
        # Create TreeExplainer for CatBoost
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values
        # For binary classification, we get values for both classes
        # We use class 1 (positive/glaucoma) SHAP values
        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification returns [class0_values, class1_values]
            shap_values = shap_values[1]  # Use class 1 (positive class)
            expected_value = explainer.expected_value[1]
        else:
            # Single array (some versions)
            expected_value = explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = (
                    expected_value[1] if len(expected_value) > 1 else expected_value[0]
                )

        return {
            "shap_values": shap_values,
            "expected_value": float(expected_value),
            "model_idx": model_idx,
            "success": True,
        }

    except Exception as e:
        return {
            "shap_values": None,
            "expected_value": None,
            "model_idx": model_idx,
            "success": False,
            "error": str(e),
        }


def compute_shap_for_config(
    config_data: dict[str, Any],
    config_idx: int,
    n_jobs: int = -1,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Compute SHAP values for all bootstrap models in a configuration.

    Parameters
    ----------
    config_data : dict
        Configuration data with bootstrap_models, X_test, etc.
    config_idx : int
        Index of this configuration (1-10)
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    checkpoint_dir : Path, optional
        Directory for checkpoints

    Returns
    -------
    dict with aggregated SHAP statistics
    """
    bootstrap_models = config_data["bootstrap_models"]
    X_test = config_data["X_test"]
    feature_names = config_data["feature_names"]
    config_info = config_data["config"]

    n_bootstrap = len(bootstrap_models)
    n_samples = X_test.shape[0]
    n_features = X_test.shape[1]

    print(
        f"\n  Config {config_idx}: {config_info['outlier_method']} + {config_info['imputation_method']}"
    )
    print(f"    Bootstrap models: {n_bootstrap}")
    print(f"    Test samples: {n_samples}")
    print(f"    Features: {n_features}")

    # Check for checkpoint
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / f"config_{config_idx}_shap.pkl"
        if checkpoint_file.exists():
            print(f"    Loading from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "rb") as f:
                return pickle.load(f)

    # Compute SHAP values in parallel
    print(f"    Computing SHAP values (n_jobs={n_jobs})...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(compute_shap_for_model)(model, X_test, idx)
        for idx, model in enumerate(
            tqdm(bootstrap_models, desc=f"    Config {config_idx}")
        )
    )

    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f}s ({elapsed / n_bootstrap:.3f}s per model)")

    # Check for failures
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"    WARNING: {len(failures)} models failed SHAP computation")
        for f in failures[:3]:
            print(f"      Model {f['model_idx']}: {f.get('error', 'unknown')}")

    # Collect successful SHAP values
    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        raise RuntimeError(f"All {n_bootstrap} SHAP computations failed!")

    # Stack SHAP values: shape (n_successful, n_samples, n_features)
    all_shap_values = np.stack([r["shap_values"] for r in successful_results])
    all_expected_values = np.array([r["expected_value"] for r in successful_results])

    # Compute statistics across bootstrap
    result = {
        "config_idx": config_idx,
        "config_info": config_info,
        "feature_names": feature_names,
        "n_bootstrap": n_bootstrap,
        "n_successful": len(successful_results),
        "n_samples": n_samples,
        "n_features": n_features,
        # Full SHAP values (for detailed analysis)
        "shap_values_all": all_shap_values,  # (n_bootstrap, n_samples, n_features)
        "expected_values_all": all_expected_values,  # (n_bootstrap,)
        # Aggregated statistics
        "shap_values_mean": all_shap_values.mean(axis=0),  # (n_samples, n_features)
        "shap_values_std": all_shap_values.std(axis=0),  # (n_samples, n_features)
        "shap_values_ci_lo": np.percentile(all_shap_values, 2.5, axis=0),
        "shap_values_ci_hi": np.percentile(all_shap_values, 97.5, axis=0),
        # Feature importance (mean |SHAP| across samples)
        "feature_importance_mean": np.abs(all_shap_values).mean(
            axis=(0, 1)
        ),  # (n_features,)
        "feature_importance_std": np.abs(all_shap_values)
        .mean(axis=1)
        .std(axis=0),  # (n_features,)
        # Per-sample uncertainty
        "sample_shap_uncertainty": all_shap_values.std(axis=0).mean(
            axis=1
        ),  # (n_samples,)
        "computation_time_s": elapsed,
    }

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(result, f)
        print(f"    Saved checkpoint: {checkpoint_file}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for top-10 CatBoost models"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all cores)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Config indices to process (e.g., '1,2,3' or 'all')",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: Compute SHAP Values for Top-10 CatBoost Models")
    print("=" * 60)

    # Load artifact
    if not ARTIFACT_PATH.exists():
        print(f"ERROR: Artifact not found: {ARTIFACT_PATH}")
        print(
            "Run Phase 1.2 first: python scripts/extract_top10_models_with_artifacts.py"
        )
        return

    print(f"\nLoading artifact from {ARTIFACT_PATH}...")
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    configs = artifact["configs"]
    print(f"Loaded {len(configs)} configurations")

    # Determine which configs to process
    if args.configs == "all":
        config_indices = list(range(len(configs)))
    else:
        config_indices = [int(x.strip()) - 1 for x in args.configs.split(",")]

    print(f"\nProcessing configs: {[i + 1 for i in config_indices]}")

    # Setup checkpoint directory
    checkpoint_dir = CHECKPOINT_DIR if args.resume else None
    if args.resume:
        print(f"Resume mode: checkpoints in {CHECKPOINT_DIR}")

    # Compute SHAP values for each configuration
    all_results = []
    total_start = time.time()

    for idx in config_indices:
        config_data = configs[idx]
        result = compute_shap_for_config(
            config_data,
            config_idx=idx + 1,
            n_jobs=args.n_jobs,
            checkpoint_dir=checkpoint_dir,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Total computation time: {total_elapsed / 60:.1f} minutes")

    # Save full results
    OUTPUT_SHAP_FULL.parent.mkdir(parents=True, exist_ok=True)

    full_output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "source_artifact": str(ARTIFACT_PATH),
            "n_configs": len(all_results),
            "computation_time_minutes": total_elapsed / 60,
        },
        "results": all_results,
    }

    print(f"\nSaving full SHAP values to {OUTPUT_SHAP_FULL}...")
    with open(OUTPUT_SHAP_FULL, "wb") as f:
        pickle.dump(full_output, f)
    print(f"File size: {OUTPUT_SHAP_FULL.stat().st_size / 1024 / 1024:.1f} MB")

    # Save summary (without full bootstrap arrays - smaller file)
    summary_results = []
    for r in all_results:
        summary = {
            k: v
            for k, v in r.items()
            if k not in ["shap_values_all", "expected_values_all"]
        }
        summary_results.append(summary)

    summary_output = {
        "metadata": full_output["metadata"],
        "results": summary_results,
    }

    print(f"Saving summary to {OUTPUT_SHAP_SUMMARY}...")
    with open(OUTPUT_SHAP_SUMMARY, "wb") as f:
        pickle.dump(summary_output, f)
    print(f"File size: {OUTPUT_SHAP_SUMMARY.stat().st_size / 1024 / 1024:.1f} MB")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    for r in all_results:
        cfg = r["config_info"]
        print(
            f"\nConfig {r['config_idx']}: {cfg['outlier_method'][:20]} + {cfg['imputation_method']}"
        )
        print(f"  Successful bootstrap models: {r['n_successful']}/{r['n_bootstrap']}")
        print("  Top features by importance:")
        importance = list(
            zip(
                r["feature_names"],
                r["feature_importance_mean"],
                r["feature_importance_std"],
            )
        )
        importance.sort(key=lambda x: -x[1])
        for name, mean, std in importance[:5]:
            print(f"    {name}: {mean:.4f} ± {std:.4f}")

    print(f"\n{'=' * 60}")
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Verify SHAP computation:
   python -c "import pickle; d=pickle.load(open('outputs/shap_summary_top10.pkl','rb')); print(len(d['results']))"

2. Proceed to Phase 3: Generate visualizations
   python scripts/generate_shap_figures.py
""")


if __name__ == "__main__":
    main()
