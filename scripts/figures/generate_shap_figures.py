#!/usr/bin/env python3
"""
Generate SHAP Visualizations for Top-10 CatBoost Models.

This is Phase 3 of the end-to-end visualization pipeline.
Creates publication-ready SHAP figures with uncertainty quantification.

FIGURES GENERATED:
1. Feature importance bar plots with bootstrap CI
2. SHAP beeswarm plots per configuration
3. Feature importance comparison across preprocessing methods
4. SHAP value heatmap (samples × features)

Usage:
    python scripts/generate_shap_figures.py [--config N] [--all]

Author: Foundation PLR Team
Date: 2026-01-25
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load_category_display_name(category_id: str) -> str:
    """Load a category display name from config."""
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "mlflow_registry"
        / "display_names.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    categories = config.get("categories", {})
    return categories.get(category_id, {}).get("display_name", category_id)


# Configuration
SHAP_SUMMARY_PATH = Path("outputs/shap_summary_top10.pkl")
SHAP_FULL_PATH = Path("outputs/shap_values_top10.pkl")
ARTIFACT_PATH = Path("outputs/top10_catboost_models.pkl")
OUTPUT_DIR = Path("figures/generated/shap")

# Plot styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# Load colors from centralized config (NO HARDCODING!)
def _load_colors_from_yaml() -> dict:
    """Load colors from centralized colors.yaml config."""
    colors_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "VISUALIZATION"
        / "colors.yaml"
    )
    try:
        with open(colors_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load colors.yaml: {e}")
        return {}


_COLORS_CONFIG = _load_colors_from_yaml()

# Color scheme for features - loaded from config
FEATURE_COLORS = {
    "Blue": _COLORS_CONFIG.get("feature_blue", "#1f77b4"),
    "Red": _COLORS_CONFIG.get("feature_red", "#d62728"),
    "default": _COLORS_CONFIG.get("feature_default", "#7f7f7f"),
}


def get_feature_color(feature_name: str) -> str:
    """Get color based on feature wavelength (Blue/Red)."""
    if feature_name.startswith("Blue_"):
        return FEATURE_COLORS["Blue"]
    elif feature_name.startswith("Red_"):
        return FEATURE_COLORS["Red"]
    return FEATURE_COLORS["default"]


def shorten_feature_name(name: str) -> str:
    """Shorten feature name for display."""
    # Remove wavelength prefix for cleaner display
    short = name.replace("Blue_", "B:").replace("Red_", "R:")
    short = short.replace("_value", "")
    return short


def plot_feature_importance_with_ci(
    result: dict,
    output_path: Path,
    top_n: int = 8,
) -> dict:
    """
    Create feature importance bar plot with bootstrap confidence intervals.

    Parameters
    ----------
    result : dict
        SHAP result for one configuration
    output_path : Path
        Where to save the figure
    top_n : int
        Number of top features to show

    Returns
    -------
    dict : Data for JSON export
    """
    feature_names = result["feature_names"]
    importance_mean = result["feature_importance_mean"]
    importance_std = result["feature_importance_std"]
    config_info = result["config_info"]

    # Sort by importance
    sorted_idx = np.argsort(importance_mean)[::-1][:top_n]

    names = [feature_names[i] for i in sorted_idx]
    means = importance_mean[sorted_idx]
    stds = importance_std[sorted_idx]
    colors = [get_feature_color(n) for n in names]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(names))
    ax.barh(
        y_pos,
        means,
        xerr=stds * 1.96,
        capsize=3,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_feature_name(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (feature importance)")
    ax.set_title(
        f"Feature Importance: {config_info['outlier_method'][:25]} + {config_info['imputation_method']}\n"
        f"AUROC: {config_info['auroc_mean']:.3f} | Bootstrap: {result['n_successful']}/1000"
    )

    # Add legend for wavelength colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=FEATURE_COLORS["Blue"], label="Blue (469nm)"),
        Patch(facecolor=FEATURE_COLORS["Red"], label="Red (640nm)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    # Return data for JSON
    return {
        "config_idx": result["config_idx"],
        "outlier_method": config_info["outlier_method"],
        "imputation_method": config_info["imputation_method"],
        "auroc": config_info["auroc_mean"],
        "n_bootstrap": result["n_successful"],
        "features": [
            {
                "name": names[i],
                "importance_mean": float(means[i]),
                "importance_std": float(stds[i]),
                "importance_ci_lo": float(means[i] - 1.96 * stds[i]),
                "importance_ci_hi": float(means[i] + 1.96 * stds[i]),
            }
            for i in range(len(names))
        ],
    }


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    config_info: dict,
    output_path: Path,
) -> dict:
    """
    Create SHAP beeswarm plot showing value distribution.

    Parameters
    ----------
    shap_values : np.ndarray
        Mean SHAP values (n_samples, n_features)
    X : np.ndarray
        Feature values (n_samples, n_features)
    feature_names : list
        Feature names
    config_info : dict
        Configuration metadata
    output_path : Path
        Where to save

    Returns
    -------
    dict : Data for JSON export
    """
    n_samples, n_features = shap_values.shape

    # Sort features by mean absolute SHAP
    importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, feat_idx in enumerate(sorted_idx):
        feat_shap = shap_values[:, feat_idx]
        feat_values = X[:, feat_idx]

        # Normalize feature values for coloring
        feat_norm = (feat_values - feat_values.min()) / (
            feat_values.max() - feat_values.min() + 1e-8
        )

        # Add jitter to y position
        y_jitter = i + np.random.uniform(-0.3, 0.3, n_samples)

        scatter = ax.scatter(
            feat_shap,
            y_jitter,
            c=feat_norm,
            cmap="coolwarm",
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([shorten_feature_name(feature_names[i]) for i in sorted_idx])
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(
        f"SHAP Beeswarm: {config_info['outlier_method'][:25]} + {config_info['imputation_method']}"
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("Feature value (normalized)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    return {
        "config": config_info,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def plot_feature_importance_comparison(
    all_results: list[dict],
    output_path: Path,
) -> dict:
    """
    Compare feature importance across all configurations.

    Parameters
    ----------
    all_results : list
        List of SHAP results for all configs
    output_path : Path
        Where to save

    Returns
    -------
    dict : Data for JSON export
    """
    # Get all unique features
    all_features = all_results[0]["feature_names"]
    n_configs = len(all_results)
    n_features = len(all_features)

    # Build importance matrix
    importance_matrix = np.zeros((n_configs, n_features))
    config_labels = []

    for i, result in enumerate(all_results):
        importance_matrix[i, :] = result["feature_importance_mean"]
        cfg = result["config_info"]
        # Create short label
        od_short = cfg["outlier_method"][:15]
        imp_short = cfg["imputation_method"][:10]
        config_labels.append(f"{od_short}\n+{imp_short}")

    # Sort features by mean importance across configs
    mean_importance = importance_matrix.mean(axis=0)
    sorted_feat_idx = np.argsort(mean_importance)[::-1]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        importance_matrix[:, sorted_feat_idx],
        aspect="auto",
        cmap="YlOrRd",
    )

    # Labels
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(
        [shorten_feature_name(all_features[i]) for i in sorted_feat_idx],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(config_labels, fontsize=8)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Configuration (Outlier + Imputation)")
    ax.set_title("Feature Importance Comparison Across Preprocessing Methods")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean |SHAP value|")

    # Add AUROC annotations
    for i, result in enumerate(all_results):
        auroc = result["config_info"]["auroc_mean"]
        ax.text(n_features + 0.3, i, f"AUROC: {auroc:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    # Return data for JSON
    return {
        "features": all_features,
        "configs": [
            {
                "idx": r["config_idx"],
                "outlier": r["config_info"]["outlier_method"],
                "imputation": r["config_info"]["imputation_method"],
                "auroc": r["config_info"]["auroc_mean"],
                "importance": r["feature_importance_mean"].tolist(),
            }
            for r in all_results
        ],
    }


def plot_uncertainty_by_feature(
    all_results: list[dict],
    output_path: Path,
) -> dict:
    """
    Show how feature importance uncertainty varies across configs.

    Parameters
    ----------
    all_results : list
        List of SHAP results
    output_path : Path
        Where to save

    Returns
    -------
    dict : Data for JSON export
    """
    features = all_results[0]["feature_names"]
    n_features = len(features)
    n_configs = len(all_results)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_features)
    width = 0.8 / n_configs

    for i, result in enumerate(all_results):
        means = result["feature_importance_mean"]
        stds = result["feature_importance_std"]

        offset = (i - n_configs / 2 + 0.5) * width
        cfg = result["config_info"]

        # Color based on config type
        if (
            cfg["outlier_method"] == "pupil-gt"
            and cfg["imputation_method"] == "pupil-gt"
        ):
            color = "gold"
            label = _load_category_display_name("ground_truth")
        elif "ensemble" in cfg["outlier_method"].lower():
            color = "steelblue"
            label = f"{_load_category_display_name('ensemble')} (Rank {result['config_idx']})"
        else:
            color = "gray"
            label = f"Rank {result['config_idx']}"

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=color,
            label=label,
            alpha=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [shorten_feature_name(f) for f in features], rotation=45, ha="right"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean |SHAP value| (± std)")
    ax.set_title(
        "Feature Importance with Bootstrap Uncertainty\nAcross Top-10 Configurations"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    return {"n_configs": n_configs, "n_features": n_features}


def plot_ground_truth_vs_ensemble(
    all_results: list[dict],
    output_path: Path,
) -> dict:
    """
    Compare ground truth config vs best ensemble config.

    Parameters
    ----------
    all_results : list
        List of SHAP results
    output_path : Path
        Where to save

    Returns
    -------
    dict : Data for JSON export
    """
    # Find ground truth and best ensemble configs
    gt_result = None
    ensemble_result = None

    for r in all_results:
        cfg = r["config_info"]
        if (
            cfg["outlier_method"] == "pupil-gt"
            and cfg["imputation_method"] == "pupil-gt"
        ):
            gt_result = r
        elif r["config_idx"] == 1:  # Best config (rank 1)
            ensemble_result = r

    if gt_result is None or ensemble_result is None:
        print(
            "  WARNING: Could not find ground truth or ensemble config for comparison"
        )
        return {}

    features = gt_result["feature_names"]
    n_features = len(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by ground truth importance
    gt_importance = gt_result["feature_importance_mean"]
    sorted_idx = np.argsort(gt_importance)[::-1]

    # Left: Ground Truth
    ax = axes[0]
    y_pos = np.arange(n_features)
    colors = [get_feature_color(features[i]) for i in sorted_idx]

    ax.barh(
        y_pos,
        gt_importance[sorted_idx],
        xerr=gt_result["feature_importance_std"][sorted_idx] * 1.96,
        capsize=3,
        color=colors,
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_feature_name(features[i]) for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Ground Truth (pupil-gt + pupil-gt)\nAUROC: {gt_result['config_info']['auroc_mean']:.3f}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Best Ensemble
    ax = axes[1]
    ens_importance = ensemble_result["feature_importance_mean"]

    ax.barh(
        y_pos,
        ens_importance[sorted_idx],
        xerr=ensemble_result["feature_importance_std"][sorted_idx] * 1.96,
        capsize=3,
        color=colors,
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_feature_name(features[i]) for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    cfg = ensemble_result["config_info"]
    ax.set_title(
        f"Best {_load_category_display_name('ensemble')} ({cfg['outlier_method'][:20]}...)\n"
        f"AUROC: {cfg['auroc_mean']:.3f}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=FEATURE_COLORS["Blue"], label="Blue (469nm)"),
        Patch(facecolor=FEATURE_COLORS["Red"], label="Red (640nm)"),
    ]
    fig.legend(
        handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02)
    )

    plt.suptitle("Feature Importance: Ground Truth vs Best Automated Pipeline", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

    # Compute correlation
    correlation = np.corrcoef(gt_importance, ens_importance)[0, 1]

    return {
        "ground_truth": {
            "auroc": gt_result["config_info"]["auroc_mean"],
            "importance": gt_importance.tolist(),
        },
        "best_ensemble": {
            "outlier": cfg["outlier_method"],
            "imputation": cfg["imputation_method"],
            "auroc": cfg["auroc_mean"],
            "importance": ens_importance.tolist(),
        },
        "importance_correlation": float(correlation),
        "features": features,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate SHAP visualizations")
    parser.add_argument("--config", type=int, help="Generate for specific config only")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3: Generate SHAP Visualizations")
    print("=" * 60)

    # Check inputs exist
    if not SHAP_SUMMARY_PATH.exists():
        print(f"ERROR: SHAP summary not found: {SHAP_SUMMARY_PATH}")
        print("Run Phase 2 first: python scripts/compute_shap_values.py")
        return

    # Load SHAP summary
    print(f"\nLoading SHAP summary from {SHAP_SUMMARY_PATH}...")
    with open(SHAP_SUMMARY_PATH, "rb") as f:
        shap_data = pickle.load(f)

    all_results = shap_data["results"]
    print(f"Loaded {len(all_results)} configurations")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Store all JSON data
    all_json_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "source": str(SHAP_SUMMARY_PATH),
            "n_configs": len(all_results),
        },
        "figures": {},
    }

    # 1. Individual feature importance plots
    print("\n1. Generating individual feature importance plots...")
    for result in all_results:
        idx = result["config_idx"]
        if args.config and idx != args.config:
            continue

        output_path = OUTPUT_DIR / f"feature_importance_config_{idx}.png"
        print(f"   Config {idx}...", end=" ")

        json_data = plot_feature_importance_with_ci(result, output_path)
        all_json_data["figures"][f"feature_importance_config_{idx}"] = json_data
        print("done")

    # 2. Feature importance comparison heatmap
    print("\n2. Generating feature importance comparison heatmap...")
    output_path = OUTPUT_DIR / "feature_importance_comparison.png"
    json_data = plot_feature_importance_comparison(all_results, output_path)
    all_json_data["figures"]["feature_importance_comparison"] = json_data
    print("   done")

    # 3. Uncertainty comparison
    print("\n3. Generating uncertainty comparison plot...")
    output_path = OUTPUT_DIR / "feature_importance_uncertainty.png"
    json_data = plot_uncertainty_by_feature(all_results, output_path)
    all_json_data["figures"]["feature_importance_uncertainty"] = json_data
    print("   done")

    # 4. Ground truth vs ensemble comparison
    print("\n4. Generating ground truth vs ensemble comparison...")
    output_path = OUTPUT_DIR / "gt_vs_ensemble_comparison.png"
    json_data = plot_ground_truth_vs_ensemble(all_results, output_path)
    all_json_data["figures"]["gt_vs_ensemble_comparison"] = json_data
    print("   done")

    # 5. Beeswarm plots (need full SHAP values and feature data)
    if SHAP_FULL_PATH.exists() and ARTIFACT_PATH.exists():
        print("\n5. Generating SHAP beeswarm plots...")

        with open(SHAP_FULL_PATH, "rb") as f:
            shap_full = pickle.load(f)
        with open(ARTIFACT_PATH, "rb") as f:
            artifact = pickle.load(f)

        for i, result in enumerate(shap_full["results"]):
            idx = result["config_idx"]
            if args.config and idx != args.config:
                continue

            output_path = OUTPUT_DIR / f"shap_beeswarm_config_{idx}.png"
            print(f"   Config {idx}...", end=" ")

            # Get mean SHAP values and feature data
            shap_mean = result["shap_values_mean"]
            X_test = artifact["configs"][i]["X_test"]
            feature_names = result["feature_names"]
            config_info = result["config_info"]

            json_data = plot_shap_beeswarm(
                shap_mean, X_test, feature_names, config_info, output_path
            )
            all_json_data["figures"][f"shap_beeswarm_config_{idx}"] = json_data
            print("done")
    else:
        print("\n5. Skipping beeswarm plots (need full SHAP values)")

    # Save JSON data
    json_path = OUTPUT_DIR / "shap_figures_data.json"
    with open(json_path, "w") as f:
        json.dump(all_json_data, f, indent=2, default=str)
    print(f"\nSaved JSON data to {json_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated figures in: {OUTPUT_DIR}")
    print(f"  - {len(all_results)} individual feature importance plots")
    print("  - 1 feature importance comparison heatmap")
    print("  - 1 uncertainty comparison plot")
    print("  - 1 ground truth vs ensemble comparison")
    if SHAP_FULL_PATH.exists() and ARTIFACT_PATH.exists():
        print(f"  - {len(all_results)} SHAP beeswarm plots")

    # List files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
