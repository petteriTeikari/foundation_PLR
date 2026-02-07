#!/usr/bin/env python3
"""
Compute Variance Inflation Factor (VIF) for feature collinearity analysis.

Phase 1.3 of ggplot2 visualization migration.
Computes VIF for all features in top-10 CatBoost models.

VIF Interpretation (with physiological context from OPT-13):
- Temporal window features (same stimulus): VIF > 20 = High, > 10 = Moderate
- Cross-stimulus features: VIF > 5 = High, > 3 = Moderate
- General threshold: VIF > 5 suggests multicollinearity

Output:
    outputs/r_data/vif_analysis.json - VIF scores with concern levels

Usage:
    python scripts/compute_vif.py

Author: Foundation PLR Team
Date: 2026-01-25
"""

import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration - Canonical data locations
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "outputs" / "top10_catboost_models.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "r_data"


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:12]


def create_metadata(generator: str, source_path: Path) -> dict:
    """Create standard metadata wrapper."""
    return {
        "created": datetime.now().isoformat(),
        "schema_version": "1.0",
        "generator": generator,
        "r_script": "src/r/figures/fig_vif_analysis.R",
        "data_source": {
            "file": str(source_path),
            "hash": compute_file_hash(source_path) if source_path.exists() else "N/A",
        },
    }


def get_feature_category(feature_name: str) -> str:
    """Categorize feature by physiological source."""
    if feature_name.startswith("Blue_") or feature_name.startswith("amp_bin_"):
        # Check if it's a Blue bin (0-4) or Red bin (5-9)
        if feature_name.startswith("amp_bin_"):
            bin_num = int(feature_name.split("_")[-1])
            if bin_num < 5:
                return "Blue_temporal"
            else:
                return "Red_temporal"
        return "Blue_temporal"
    elif feature_name.startswith("Red_"):
        return "Red_temporal"
    elif feature_name in ["PIPR", "MEDFA", "MAX_CONSTRICTION"]:
        return "Latency"
    else:
        return "Other"


def get_concern_level(vif: float, category: str) -> str:
    """
    Determine VIF concern level with physiological context.

    From OPT-13 (Statistical Rigor Reviewer):
    - Temporal window features (same stimulus): VIF > 20 = High, > 10 = Moderate
    - Cross-stimulus features: VIF > 5 = High, > 3 = Moderate
    """
    if category in ["Blue_temporal", "Red_temporal"]:
        # Same-stimulus temporal features: higher threshold
        if vif >= 20:
            return "High"
        elif vif >= 10:
            return "Moderate"
        else:
            return "OK"
    else:
        # Cross-stimulus or latency features: standard threshold
        if vif >= 10:
            return "High"
        elif vif >= 5:
            return "Moderate"
        else:
            return "OK"


def compute_vif_for_config(X: np.ndarray, feature_names: list) -> list:
    """Compute VIF for all features in a config."""
    # Add constant for VIF computation
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])

    vif_data = []
    for i, name in enumerate(feature_names):
        try:
            # Note: statsmodels VIF uses column index in augmented matrix
            # So feature i is at index i+1 (after the constant)
            vif = variance_inflation_factor(X_with_const, i + 1)
        except Exception:
            vif = np.nan

        category = get_feature_category(name)
        concern = get_concern_level(vif, category) if np.isfinite(vif) else "Unknown"

        vif_data.append(
            {
                "feature": name,
                "VIF": float(vif) if np.isfinite(vif) else None,
                "category": category,
                "concern": concern,
            }
        )

    return vif_data


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1.3: Compute VIF Analysis")
    print("=" * 60)

    # Check input file
    if not ARTIFACT_PATH.exists():
        print(f"ERROR: Artifact not found: {ARTIFACT_PATH}")
        print(
            "Run Phase 1.2 first: python scripts/extract_top10_models_with_artifacts.py"
        )
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifact
    print(f"\nLoading: {ARTIFACT_PATH}")
    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    configs = artifact["configs"]
    print(f"Loaded {len(configs)} configurations")

    # Compute VIF for each config
    all_vif_results = []

    for i, cfg in enumerate(configs):
        config_info = cfg["config"]
        X_train = cfg["X_train"]
        feature_names = cfg["feature_names"]

        print(
            f"\nConfig {i + 1}: {config_info['outlier_method'][:20]} + {config_info['imputation_method']}"
        )
        print(f"  Features: {len(feature_names)}, Samples: {X_train.shape[0]}")

        vif_data = compute_vif_for_config(X_train, feature_names)

        # Sort by VIF descending
        vif_data.sort(key=lambda x: -(x["VIF"] if x["VIF"] else 0))

        all_vif_results.append(
            {
                "config_idx": i + 1,
                "name": f"{config_info['outlier_method'][:15]}+{config_info['imputation_method'][:10]}",
                "outlier_method": config_info["outlier_method"],
                "imputation_method": config_info["imputation_method"],
                "n_samples": X_train.shape[0],
                "vif_scores": vif_data,
            }
        )

        # Print top 5 VIF
        print("  Top VIF scores:")
        for v in vif_data[:5]:
            concern_emoji = {"High": "!", "Moderate": "~", "OK": " "}[v["concern"]]
            vif_str = f"{v['VIF']:.2f}" if v["VIF"] else "N/A"
            print(
                f"    {concern_emoji} {v['feature']:<20} VIF={vif_str:<8} ({v['category']})"
            )

    # Compute aggregate VIF (mean across configs)
    print("\n" + "-" * 40)
    print("Computing aggregate VIF across configs...")

    feature_names = configs[0]["feature_names"]
    n_features = len(feature_names)
    n_configs = len(configs)

    # Collect VIF values
    vif_matrix = np.zeros((n_configs, n_features))
    for i, result in enumerate(all_vif_results):
        for j, name in enumerate(feature_names):
            # Find this feature in the config's VIF results
            for v in result["vif_scores"]:
                if v["feature"] == name:
                    vif_matrix[i, j] = v["VIF"] if v["VIF"] else np.nan
                    break

    # Compute mean, std, min, max
    aggregate_vif = []
    for j, name in enumerate(feature_names):
        vif_vals = vif_matrix[:, j]
        valid_vals = vif_vals[~np.isnan(vif_vals)]

        if len(valid_vals) > 0:
            mean_vif = float(np.mean(valid_vals))
            std_vif = float(np.std(valid_vals))
            min_vif = float(np.min(valid_vals))
            max_vif = float(np.max(valid_vals))
        else:
            mean_vif = std_vif = min_vif = max_vif = None

        category = get_feature_category(name)
        concern = get_concern_level(mean_vif, category) if mean_vif else "Unknown"

        aggregate_vif.append(
            {
                "feature": name,
                "category": category,
                "VIF_mean": mean_vif,
                "VIF_std": std_vif,
                "VIF_min": min_vif,
                "VIF_max": max_vif,
                "concern": concern,
            }
        )

    # Sort by mean VIF
    aggregate_vif.sort(key=lambda x: -(x["VIF_mean"] if x["VIF_mean"] else 0))

    # Create output
    output = {
        "metadata": create_metadata("scripts/compute_vif.py", ARTIFACT_PATH),
        "data": {
            "n_configs": n_configs,
            "n_features": n_features,
            "interpretation": {
                "temporal_threshold_high": 20,
                "temporal_threshold_moderate": 10,
                "general_threshold_high": 10,
                "general_threshold_moderate": 5,
                "note": "Blue stimulus features show expected correlation due to shared pupillary dynamics",
            },
            "aggregate": aggregate_vif,
            "per_config": all_vif_results,
        },
    }

    output_path = OUTPUT_DIR / "vif_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VIF ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nAggregate VIF (mean across configs):")
    print(f"{'Feature':<20} {'Mean VIF':<10} {'Concern':<10} {'Category'}")
    print("-" * 55)
    for v in aggregate_vif:
        vif_str = f"{v['VIF_mean']:.2f}" if v["VIF_mean"] else "N/A"
        print(f"{v['feature']:<20} {vif_str:<10} {v['concern']:<10} {v['category']}")

    # Count concerns
    high_count = sum(1 for v in aggregate_vif if v["concern"] == "High")
    mod_count = sum(1 for v in aggregate_vif if v["concern"] == "Moderate")
    print(
        f"\nSummary: {high_count} High concern, {mod_count} Moderate concern, "
        f"{n_features - high_count - mod_count} OK"
    )

    print("\nNext steps:")
    print("  1. Open R and run: source('src/r/figures/fig_vif_analysis.R')")


if __name__ == "__main__":
    main()
