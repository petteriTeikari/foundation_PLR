#!/usr/bin/env python3
"""
Export SHAP data to R-friendly JSON format.

Phase 1.2 of ggplot2 visualization migration.
Exports SHAP values with proper structure for ggplot2 visualization.

Output Files:
    outputs/r_data/shap_feature_importance.json - Per-feature importance with CI
    outputs/r_data/shap_per_sample.json - Sample-level SHAP for beeswarm
    outputs/r_data/shap_ensemble_aggregated.json - Aggregated across top-10

Usage:
    python scripts/export_shap_for_r.py

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

# Configuration - Canonical data locations
PROJECT_ROOT = Path(__file__).parent.parent.parent
SHAP_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "shap_summary_top10.pkl"
SHAP_FULL_PATH = PROJECT_ROOT / "outputs" / "shap_values_top10.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "r_data"
VIF_PATH = PROJECT_ROOT / "data" / "r_data" / "vif_analysis.json"


def check_vif_multicollinearity(
    vif_path: Path, threshold: float = 10.0
) -> tuple[bool, list, dict]:
    """
    Check VIF values and warn if multicollinearity is severe.

    SHAP values are unreliable when features are highly correlated (VIF > 10).
    This pre-check warns users before they interpret SHAP rankings.

    Args:
        vif_path: Path to vif_analysis.json
        threshold: VIF threshold for warning (default 10.0)

    Returns:
        (is_safe, problematic_features, vif_summary)
    """
    if not vif_path.exists():
        print(f"  Note: VIF file not found ({vif_path.name})")
        print("  Run: python scripts/compute_vif.py first for multicollinearity check")
        return True, [], {}

    with open(vif_path) as f:
        vif_data = json.load(f)

    problematic = []
    vif_summary = {}

    for feat in vif_data["data"]["aggregate"]:
        vif_mean = feat["VIF_mean"]
        if vif_mean is not None:
            vif_summary[feat["feature"]] = {
                "vif": round(vif_mean, 1),
                "concern": feat["concern"],
            }
            if vif_mean > threshold:
                problematic.append(
                    {
                        "feature": feat["feature"],
                        "vif": round(vif_mean, 1),
                        "concern": feat["concern"],
                    }
                )

    is_safe = len(problematic) == 0
    return is_safe, problematic, vif_summary


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
        "r_script": "src/r/figures/fig_shap_*.R",
        "data_source": {
            "file": str(source_path),
            "hash": compute_file_hash(source_path) if source_path.exists() else "N/A",
        },
    }


def export_feature_importance(shap_data: dict) -> None:
    """Export per-feature importance with bootstrap uncertainty and VIF annotations."""
    print("\n1. Exporting shap_feature_importance.json...")

    # Get VIF data if available
    vif_summary = shap_data.get("vif_summary", {})
    vif_warning = shap_data.get("vif_warning", False)

    results = shap_data["results"]
    configs = []

    for r in results:
        config_info = r["config_info"]
        feature_names = r["feature_names"]
        importance_mean = r["feature_importance_mean"]
        importance_std = r["feature_importance_std"]

        # Compute CI from mean and std (approximate 95% CI)
        ci_lo = importance_mean - 1.96 * importance_std
        ci_hi = importance_mean + 1.96 * importance_std

        # Create feature list with VIF annotations
        features = []
        for i, name in enumerate(feature_names):
            feat_dict = {
                "feature": name,
                "mean_abs_shap": float(importance_mean[i]),
                "std": float(importance_std[i]),
                "ci_lo": float(max(0, ci_lo[i])),  # SHAP importance can't be negative
                "ci_hi": float(ci_hi[i]),
            }
            # Add VIF annotation if available
            if name in vif_summary:
                feat_dict["vif"] = vif_summary[name]["vif"]
                feat_dict["vif_concern"] = vif_summary[name]["concern"]
            features.append(feat_dict)

        # Sort by importance
        features.sort(key=lambda x: -x["mean_abs_shap"])

        configs.append(
            {
                "config_idx": r["config_idx"],
                "name": f"{config_info['outlier_method'][:20]} + {config_info['imputation_method']}",
                "outlier_method": config_info["outlier_method"],
                "imputation_method": config_info["imputation_method"],
                "n_bootstrap": r["n_successful"],
                "n_samples": r["n_samples"],
                "feature_importance": features,
            }
        )

    # Build metadata with VIF warning
    metadata = create_metadata("scripts/export_shap_for_r.py", SHAP_SUMMARY_PATH)
    if vif_warning:
        metadata["vif_warning"] = {
            "has_multicollinearity": True,
            "message": "Features with VIF > 10 detected. SHAP values may be unreliable.",
            "recommendation": "Interpret correlated features as groups, not individually.",
        }

    output = {
        "metadata": metadata,
        "data": {
            "n_configs": len(configs),
            "n_features": len(results[0]["feature_names"]),
            "feature_names": list(results[0]["feature_names"]),
            "vif_summary": vif_summary,  # Include VIF for all features
            "configs": configs,
        },
    }

    output_path = OUTPUT_DIR / "shap_feature_importance.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved: {output_path}")


def export_per_sample_shap(shap_data: dict) -> None:
    """Export sample-level SHAP values for beeswarm plots."""
    print("\n2. Exporting shap_per_sample.json...")

    results = shap_data["results"]

    # For beeswarm, we need per-sample SHAP values
    # Use mean SHAP across bootstrap as point estimate
    # This keeps file size manageable

    samples = []
    for r in results:
        config_info = r["config_info"]
        feature_names = r["feature_names"]
        shap_mean = r["shap_values_mean"]  # (n_samples, n_features)

        config_name = f"{config_info['outlier_method'][:15]}+{config_info['imputation_method'][:10]}"

        # Add each sample
        for sample_idx in range(shap_mean.shape[0]):
            for feat_idx, feat_name in enumerate(feature_names):
                samples.append(
                    {
                        "config": config_name,
                        "config_idx": r["config_idx"],
                        "sample_idx": sample_idx,
                        "feature": feat_name,
                        "shap_value": float(shap_mean[sample_idx, feat_idx]),
                    }
                )

    output = {
        "metadata": create_metadata("scripts/export_shap_for_r.py", SHAP_SUMMARY_PATH),
        "data": {
            "n_samples_total": len(samples),
            "n_configs": len(results),
            "samples": samples,
        },
    }

    output_path = OUTPUT_DIR / "shap_per_sample.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved: {output_path} ({len(samples)} sample-feature pairs)")


def compute_hierarchical_variance(shap_data: dict) -> dict:
    """
    Compute hierarchical variance decomposition for ensemble SHAP.

    From Statistical Rigor Reviewer (OPT-1):
    Var_total = Var_within (pooled within-config) + Var_between (across 10 configs)
    """
    results = shap_data["results"]
    n_configs = len(results)
    n_features = len(results[0]["feature_names"])
    feature_names = results[0]["feature_names"]

    # Collect importance means across configs
    config_means = np.zeros((n_configs, n_features))
    config_stds = np.zeros((n_configs, n_features))

    for i, r in enumerate(results):
        config_means[i] = r["feature_importance_mean"]
        config_stds[i] = r["feature_importance_std"]

    # Compute hierarchical variance
    # Between-config variance
    var_between = np.var(config_means, axis=0)

    # Within-config variance (pooled)
    var_within = np.mean(config_stds**2, axis=0)

    # Total variance
    var_total = var_between + var_within

    # Ensemble mean (equal-weighted)
    ensemble_mean = np.mean(config_means, axis=0)
    ensemble_std = np.sqrt(var_total)

    # Effective sample size (Kish's formula approximation)
    # n_eff = (sum of weights)^2 / sum of weights^2
    # With equal weights, n_eff = n_configs
    n_eff = n_configs

    return {
        "feature_names": list(feature_names),
        "ensemble_mean": ensemble_mean.tolist(),
        "ensemble_std": ensemble_std.tolist(),
        "var_within": var_within.tolist(),
        "var_between": var_between.tolist(),
        "var_total": var_total.tolist(),
        "n_eff": n_eff,
        "n_configs": n_configs,
    }


def export_ensemble_aggregated(shap_data: dict) -> None:
    """Export ensemble-aggregated SHAP with hierarchical variance."""
    print("\n3. Exporting shap_ensemble_aggregated.json...")

    # Compute hierarchical variance
    ensemble = compute_hierarchical_variance(shap_data)

    # Create feature list sorted by importance
    features = []
    for i, name in enumerate(ensemble["feature_names"]):
        mean_val = ensemble["ensemble_mean"][i]
        std_val = ensemble["ensemble_std"][i]
        features.append(
            {
                "feature": name,
                "ensemble_mean": mean_val,
                "ensemble_std": std_val,
                "ci_lo": max(0, mean_val - 1.96 * std_val),
                "ci_hi": mean_val + 1.96 * std_val,
                "var_within": ensemble["var_within"][i],
                "var_between": ensemble["var_between"][i],
                "var_total": ensemble["var_total"][i],
                "pct_var_between": (
                    ensemble["var_between"][i] / ensemble["var_total"][i] * 100
                    if ensemble["var_total"][i] > 0
                    else 0
                ),
            }
        )

    features.sort(key=lambda x: -x["ensemble_mean"])

    output = {
        "metadata": create_metadata("scripts/export_shap_for_r.py", SHAP_SUMMARY_PATH),
        "data": {
            "aggregation": "equal_weighted",
            "n_configs": ensemble["n_configs"],
            "n_eff": ensemble["n_eff"],
            "variance_decomposition": {
                "method": "hierarchical",
                "formula": "Var_total = Var_within + Var_between",
                "interpretation": "pct_var_between shows cross-config variability",
            },
            "features": features,
        },
    }

    output_path = OUTPUT_DIR / "shap_ensemble_aggregated.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved: {output_path}")

    # Print summary
    print("\n   Variance decomposition summary:")
    print(f"   {'Feature':<20} {'Mean':<8} {'Std':<8} {'%Between':<10}")
    print("   " + "-" * 50)
    for f in features[:5]:
        print(
            f"   {f['feature']:<20} {f['ensemble_mean']:.4f}   "
            f"{f['ensemble_std']:.4f}   {f['pct_var_between']:.1f}%"
        )


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1.2: Export SHAP Data for R")
    print("=" * 60)

    # Check input files
    if not SHAP_SUMMARY_PATH.exists():
        print(f"ERROR: SHAP summary not found: {SHAP_SUMMARY_PATH}")
        print("Run Phase 2 first: python scripts/compute_shap_values.py")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # VIF PRE-CHECK: Warn about multicollinearity before SHAP interpretation
    # =========================================================================
    print("\n[VIF Pre-check] Checking for multicollinearity...")
    is_safe, problematic, vif_summary = check_vif_multicollinearity(VIF_PATH)

    if not is_safe:
        print("\n" + "=" * 60)
        print("⚠️  MULTICOLLINEARITY WARNING")
        print("=" * 60)
        print("The following features have VIF > 10:")
        for p in problematic:
            symbol = "🔴" if p["concern"] == "High" else "🟡"
            print(f"  {symbol} {p['feature']}: VIF = {p['vif']} ({p['concern']})")
        print("\n" + "-" * 60)
        print("SHAP values for these features should be interpreted with CAUTION.")
        print("When features are correlated, SHAP arbitrarily splits importance")
        print("between them. Consider interpreting as feature GROUPS rather than")
        print("individual features.")
        print("\nReferences:")
        print("  - Molnar (2019) Interpretable ML Book, Section 9.5.3.1")
        print("  - Chen et al. (2020) 'True to Model or True to Data?'")
        print("=" * 60 + "\n")
    else:
        print("  ✓ No severe multicollinearity detected (all VIF < 10)")

    # Load SHAP data
    print(f"\nLoading: {SHAP_SUMMARY_PATH}")
    with open(SHAP_SUMMARY_PATH, "rb") as f:
        shap_data = pickle.load(f)

    # Store VIF summary for downstream use
    shap_data["vif_summary"] = vif_summary
    shap_data["vif_warning"] = not is_safe

    print(f"Configs: {len(shap_data['results'])}")
    print(f"Features: {len(shap_data['results'][0]['feature_names'])}")

    # Export data
    export_feature_importance(shap_data)
    export_per_sample_shap(shap_data)
    export_ensemble_aggregated(shap_data)

    print("\n" + "=" * 60)
    print("SHAP EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Files created:")
    for f in OUTPUT_DIR.glob("shap_*.json"):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")

    print("\nNext steps:")
    print("  1. Run: python scripts/compute_vif.py")
    print("  2. Open R and run: source('src/r/figures/fig_shap_importance.R')")


if __name__ == "__main__":
    main()
