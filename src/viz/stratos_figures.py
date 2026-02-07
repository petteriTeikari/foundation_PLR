"""
STRATOS-compliant figure generation.

Generates all STRATOS/TRIPOD-AI required figures:
1. Calibration plots (smoothed curves with slope/intercept)
2. Decision Curve Analysis (net benefit curves)
3. Calibration metrics scatter (slope vs O:E ratio)
4. Probability distribution histograms
5. Discrimination plots (ROC curves)

References:
- Van Calster et al. (2024). STRATOS guidance.
- Collins et al. (2024). TRIPOD-AI reporting guidelines.

Usage:
    python -m src.viz.stratos_figures generate --db PATH_TO_DB
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from .config_loader import get_config_loader

try:
    from .figure_dimensions import get_dimensions
    from .plot_config import COLORS, FIXED_CLASSIFIER, save_figure, setup_style
except ImportError:
    # Absolute import for standalone usage
    from src.viz.figure_dimensions import get_dimensions
    from src.viz.plot_config import COLORS, FIXED_CLASSIFIER, save_figure, setup_style


__all__ = [
    "load_stratos_data",
    "generate_calibration_stratos_figure",
    "generate_dca_stratos_figure",
    "generate_calibration_metrics_scatter",
    "generate_probability_distribution",
    "generate_all_stratos_figures",
]

# Classifier name → MLflow run name format mapping.
# Single source: avoid duplicating this in each function.
# TODO: Move to configs/mlflow_registry/ or src/data_io/registry.py
_CLASSIFIER_TO_MLFLOW = {
    "CatBoost": "CATBOOST",
    "XGBoost": "XGBOOST",
    "TabPFN": "TabPFN",
    "TabM": "TabM",
    "LogisticRegression": "LogisticRegression",
}


# ============================================================================
# DATA LOADING
# ============================================================================


def load_stratos_data(
    db_path: str,
    combo_ids: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Load predictions and STRATOS metrics from DuckDB.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    combo_ids : list of str, optional
        Specific combo IDs to load. If None, loads standard combos from config.

    Returns
    -------
    dict
        Maps combo_id to {
            'y_true': array, 'y_prob': array,
            'calibration_slope': float, 'calibration_intercept': float,
            'o_e_ratio': float, 'net_benefit_5pct': float, ...
        }
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Load standard combos from config if not specified
    if combo_ids is None:
        config = get_config_loader()
        standard_combos = config.get_standard_hyperparam_combos()
        combo_ids = [c["id"] for c in standard_combos]

    conn = duckdb.connect(str(db_path), read_only=True)

    combos_data = {}

    try:
        # Get unique source_names that match our combo patterns
        all_sources = (
            conn.execute("SELECT DISTINCT source_name FROM predictions")
            .fetchdf()["source_name"]
            .tolist()
        )

        logger.info(f"Found {len(all_sources)} unique source_names in database")

        # Load predictions for each combo
        for combo_id in combo_ids:
            # Load combo config to get method names
            config = get_config_loader()
            try:
                combo_config = config.get_combo_config(combo_id)
            except Exception:
                logger.warning(f"Combo config not found for {combo_id}")
                continue

            outlier = combo_config.get("outlier_method", "")
            imputation = combo_config.get("imputation_method", "")
            classifier = combo_config.get("classifier", FIXED_CLASSIFIER)

            # Map classifier names to MLflow format
            clf_mlflow = _CLASSIFIER_TO_MLFLOW.get(classifier, classifier.upper())

            # Find matching source_name by exact matching of outlier and imputation
            matching_sources = []
            for s in all_sources:
                s_lower = s.lower()
                # Pattern: {clf}_eval-auc__simple1.0__{imputation}__{outlier}
                # Check if all components match
                outlier_match = outlier.lower() in s_lower
                imputation_match = imputation.lower() in s_lower
                classifier_match = clf_mlflow.lower() in s_lower

                if outlier_match and imputation_match and classifier_match:
                    # Extra check: make sure the outlier is at the END (after imputation)
                    parts = s.split("__")
                    if len(parts) >= 4:
                        s_outlier = parts[-1].lower()
                        s_imputation = parts[-2].lower()
                        if (
                            outlier.lower() == s_outlier
                            and imputation.lower() == s_imputation
                        ):
                            matching_sources.append(s)

            if not matching_sources:
                # Try less strict matching
                for s in all_sources:
                    s_lower = s.lower()
                    if (
                        outlier.lower() in s_lower
                        and imputation.lower() in s_lower
                        and clf_mlflow.lower() in s_lower
                    ):
                        matching_sources.append(s)

            if not matching_sources:
                logger.warning(
                    f"No matching source for combo {combo_id}: {clf_mlflow}/{imputation}/{outlier}"
                )
                continue

            source_name = matching_sources[0]
            logger.info(f"Loading combo {combo_id} from {source_name}")

            # Load predictions
            pred_df = conn.execute(f"""
                SELECT y_true, y_prob
                FROM predictions
                WHERE source_name = '{source_name}'
            """).fetchdf()

            if len(pred_df) == 0:
                logger.warning(f"No predictions found for {source_name}")
                continue

            # Load STRATOS metrics from metrics_per_fold
            metrics_df = conn.execute(f"""
                SELECT
                    AVG(calibration_slope) as calibration_slope,
                    AVG(calibration_intercept) as calibration_intercept,
                    AVG(e_o_ratio) as o_e_ratio,
                    AVG(auroc) as auroc,
                    AVG(brier_score) as brier_score,
                    AVG(net_benefit_5pct) as net_benefit_5pct,
                    AVG(net_benefit_10pct) as net_benefit_10pct,
                    AVG(net_benefit_20pct) as net_benefit_20pct
                FROM metrics_per_fold
                WHERE source_name = '{source_name}'
            """).fetchdf()

            combos_data[combo_id] = {
                "source_name": source_name,
                "y_true": pred_df["y_true"].values,
                "y_prob": pred_df["y_prob"].values,
                "n_samples": len(pred_df),
                "calibration_slope": float(metrics_df["calibration_slope"].iloc[0])
                if not metrics_df.empty
                else None,
                "calibration_intercept": float(
                    metrics_df["calibration_intercept"].iloc[0]
                )
                if not metrics_df.empty
                else None,
                "o_e_ratio": float(metrics_df["o_e_ratio"].iloc[0])
                if not metrics_df.empty
                else None,
                "auroc": float(metrics_df["auroc"].iloc[0])
                if not metrics_df.empty
                else None,
                "brier_score": float(metrics_df["brier_score"].iloc[0])
                if not metrics_df.empty
                else None,
                "net_benefit_5pct": float(metrics_df["net_benefit_5pct"].iloc[0])
                if not metrics_df.empty
                else None,
                "net_benefit_10pct": float(metrics_df["net_benefit_10pct"].iloc[0])
                if not metrics_df.empty
                else None,
                "net_benefit_20pct": float(metrics_df["net_benefit_20pct"].iloc[0])
                if not metrics_df.empty
                else None,
            }

    finally:
        conn.close()

    logger.info(f"Loaded data for {len(combos_data)} combos")
    return combos_data


def load_dca_curves_from_db(
    db_path: str,
    combo_ids: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load pre-computed DCA curves from DuckDB.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    combo_ids : list of str, optional
        Specific combo IDs to load

    Returns
    -------
    dict
        Maps combo_id to DataFrame with DCA curve data
    """
    db_path = Path(db_path)
    conn = duckdb.connect(str(db_path), read_only=True)

    dca_data = {}

    try:
        # Get unique source_names
        all_sources = (
            conn.execute("SELECT DISTINCT source_name FROM dca_curves")
            .fetchdf()["source_name"]
            .tolist()
        )

        # Load standard combos from config if not specified
        if combo_ids is None:
            config = get_config_loader()
            standard_combos = config.get_standard_hyperparam_combos()
            combo_ids = [c["id"] for c in standard_combos]

        for combo_id in combo_ids:
            config = get_config_loader()
            try:
                combo_config = config.get_combo_config(combo_id)
            except Exception:
                continue

            outlier = combo_config.get("outlier_method", "")
            imputation = combo_config.get("imputation_method", "")
            classifier = combo_config.get("classifier", FIXED_CLASSIFIER)

            # Map classifier names
            clf_mlflow = _CLASSIFIER_TO_MLFLOW.get(classifier, classifier.upper())

            # Find matching source
            matching_sources = []
            for s in all_sources:
                if (
                    outlier.lower() in s.lower()
                    and imputation.lower() in s.lower()
                    and clf_mlflow.lower() in s.lower()
                ):
                    matching_sources.append(s)
                    break

            if not matching_sources:
                continue

            source_name = matching_sources[0]

            dca_df = conn.execute(f"""
                SELECT
                    threshold,
                    net_benefit_model,
                    net_benefit_all,
                    net_benefit_none,
                    sensitivity,
                    specificity
                FROM dca_curves
                WHERE source_name = '{source_name}'
                ORDER BY threshold
            """).fetchdf()

            if len(dca_df) > 0:
                dca_data[combo_id] = dca_df

    finally:
        conn.close()

    return dca_data


# ============================================================================
# FIGURE GENERATION
# ============================================================================


def generate_calibration_stratos_figure(
    combos_data: Dict[str, Dict],
    output_dir: Optional[Path] = None,
    filename: str = "fig_calibration_stratos",
) -> Tuple[str, str]:
    """
    Generate STRATOS-compliant calibration figure with multiple combos.

    Parameters
    ----------
    combos_data : dict
        Output from load_stratos_data()
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename

    Returns
    -------
    png_path, json_path
    """
    setup_style()

    fig, ax = plt.subplots(figsize=get_dimensions("calibration"))

    # Reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration", linewidth=1)

    # Load colors from config
    config = get_config_loader()
    colors = config.get_colors().get("combo_colors", {})

    # Plot each combo
    for combo_id, data in combos_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        color = colors.get(combo_id, None)

        # Compute LOESS curve
        from .calibration_plot import compute_loess_calibration

        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob, frac=0.3)

        # Get display name
        try:
            combo_config = config.get_combo_config(combo_id)
            display_name = combo_config.get("display_name", combo_id)
        except Exception:
            display_name = combo_id

        # Add metrics to label
        slope = data.get("calibration_slope")
        if slope is not None:
            label = f"{display_name} (slope={slope:.2f})"
        else:
            label = display_name

        ax.plot(x_smooth, y_smooth, label=label, color=color, linewidth=2)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Calibration Curves (STRATOS-compliant)", fontsize=14)

    # Add metrics annotation
    metrics_text = "STRATOS Metrics:\n"
    for combo_id, data in combos_data.items():
        slope = data.get("calibration_slope")
        oe = data.get("o_e_ratio")
        if slope is not None and oe is not None:
            metrics_text += f"  {combo_id}: slope={slope:.2f}, O:E={oe:.2f}\n"

    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=COLORS["background"], alpha=0.8),
    )

    # Prepare JSON data
    json_data = {
        "combos": {
            combo_id: {
                "n_samples": int(data["n_samples"]),
                "calibration_slope": data.get("calibration_slope"),
                "calibration_intercept": data.get("calibration_intercept"),
                "o_e_ratio": data.get("o_e_ratio"),
                "y_true": data["y_true"].tolist(),
                "y_prob": data["y_prob"].tolist(),
            }
            for combo_id, data in combos_data.items()
        }
    }

    # Save figure using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Generated calibration figure: {png_path}")
    return str(png_path), str(json_path)


def generate_dca_stratos_figure(
    combos_data: Dict[str, Dict],
    output_dir: Optional[Path] = None,
    filename: str = "fig_dca_stratos",
    threshold_range: Tuple[float, float] = (0.01, 0.30),
) -> Tuple[str, str]:
    """
    Generate STRATOS-compliant DCA figure with multiple combos.

    Parameters
    ----------
    combos_data : dict
        Output from load_stratos_data()
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename
    threshold_range : tuple
        (min, max) threshold range

    Returns
    -------
    png_path, json_path
    """
    setup_style()

    fig, ax = plt.subplots(figsize=get_dimensions("dca"))

    # Get colors
    config = get_config_loader()
    colors = config.get_colors().get("combo_colors", {})

    # Compute thresholds
    n_thresholds = 50
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    # Get prevalence from first combo
    first_data = list(combos_data.values())[0]
    prevalence = np.mean(first_data["y_true"])

    # Reference lines
    from .dca_plot import compute_treat_all_nb, compute_treat_none_nb

    nb_all = np.array([compute_treat_all_nb(prevalence, t) for t in thresholds])
    nb_none = np.array([compute_treat_none_nb(t) for t in thresholds])

    ax.plot(
        thresholds,
        nb_all,
        "--",
        color=COLORS["text_secondary"],
        alpha=0.7,
        label="Treat All",
        linewidth=1.5,
    )
    ax.plot(
        thresholds,
        nb_none,
        ":",
        color=COLORS["text_primary"],
        alpha=0.5,
        label="Treat None",
        linewidth=1.5,
    )

    # Plot each combo
    dca_curves = {}
    from .dca_plot import compute_net_benefit

    for combo_id, data in combos_data.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        color = colors.get(combo_id, None)

        nb_model = np.array(
            [compute_net_benefit(y_true, y_prob, t) for t in thresholds]
        )
        dca_curves[combo_id] = nb_model

        # Get display name
        try:
            combo_config = config.get_combo_config(combo_id)
            display_name = combo_config.get("display_name", combo_id)
        except Exception:
            display_name = combo_id

        ax.plot(thresholds, nb_model, label=display_name, color=color, linewidth=2)

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_xlim(threshold_range[0] - 0.01, threshold_range[1] + 0.01)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=COLORS["grid_lines"], linestyle="-", alpha=0.3)
    ax.set_title("Decision Curve Analysis (STRATOS-compliant)", fontsize=14)

    # Set reasonable y-axis limits
    all_nb = []
    for combo_id, nb in dca_curves.items():
        all_nb.extend(nb.tolist())
    all_nb.extend(nb_all[nb_all > -0.5].tolist())
    if all_nb:
        y_min = max(-0.1, np.nanmin(all_nb) - 0.02)
        y_max = np.nanmax(all_nb) + 0.02
        ax.set_ylim(y_min, y_max)

    # Prepare JSON data
    json_data = {
        "threshold_range": list(threshold_range),
        "thresholds": thresholds.tolist(),
        "prevalence": float(prevalence),
        "nb_all": nb_all.tolist(),
        "nb_none": nb_none.tolist(),
        "combos": {
            combo_id: {
                "net_benefit": dca_curves[combo_id].tolist(),
                "net_benefit_10pct": combos_data[combo_id].get("net_benefit_10pct"),
            }
            for combo_id in combos_data
        },
    }

    # Save using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    logger.info(f"Generated DCA figure: {png_path}")
    return str(png_path), str(json_path)


def generate_calibration_metrics_scatter(
    combos_data: Dict[str, Dict],
    output_dir: Optional[Path] = None,
    filename: str = "fig_calibration_metrics_scatter",
) -> Tuple[str, str]:
    """
    Generate calibration metrics scatter plot (slope vs O:E ratio).

    This STRATOS-recommended plot shows both calibration slope and O:E ratio
    to assess calibration quality.

    Parameters
    ----------
    combos_data : dict
        Output from load_stratos_data()
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename

    Returns
    -------
    png_path, json_path
    """
    setup_style()

    fig, ax = plt.subplots(figsize=get_dimensions("calibration"))

    # Get colors
    config = get_config_loader()
    colors = config.get_colors().get("combo_colors", {})

    # Plot each combo
    scatter_data = []
    for combo_id, data in combos_data.items():
        slope = data.get("calibration_slope")
        oe = data.get("o_e_ratio")

        if slope is None or oe is None:
            continue

        color = colors.get(combo_id, None)

        # Get display name
        try:
            combo_config = config.get_combo_config(combo_id)
            display_name = combo_config.get("display_name", combo_id)
        except Exception:
            display_name = combo_id

        ax.scatter(
            slope,
            oe,
            s=150,
            color=color,
            label=display_name,
            alpha=0.8,
            edgecolors="black",
        )
        scatter_data.append(
            {
                "combo_id": combo_id,
                "display_name": display_name,
                "calibration_slope": slope,
                "o_e_ratio": oe,
            }
        )

    # Reference lines (ideal = 1.0 for both)
    ax.axvline(
        x=1.0,
        color=COLORS["text_secondary"],
        linestyle="--",
        alpha=0.5,
        label="Ideal slope",
    )
    ax.axhline(
        y=1.0,
        color=COLORS["text_secondary"],
        linestyle="--",
        alpha=0.5,
        label="Ideal O:E",
    )

    # Add "ideal" point
    ax.scatter(
        [1.0],
        [1.0],
        s=200,
        marker="*",
        color=COLORS["highlight"],
        edgecolors=COLORS["text_primary"],
        label="Perfect calibration",
        zorder=10,
    )

    ax.set_xlabel("Calibration Slope", fontsize=12)
    ax.set_ylabel("Observed:Expected Ratio (O:E)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Calibration Metrics Scatter (STRATOS)", fontsize=14)

    # Set axis limits with some padding
    all_slopes = [d["calibration_slope"] for d in scatter_data]
    all_oe = [d["o_e_ratio"] for d in scatter_data]
    if all_slopes and all_oe:
        x_min, x_max = min(all_slopes + [1.0]) - 0.1, max(all_slopes + [1.0]) + 0.1
        y_min, y_max = min(all_oe + [1.0]) - 0.1, max(all_oe + [1.0]) + 0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Prepare JSON data
    json_data = {
        "ideal_slope": 1.0,
        "ideal_oe_ratio": 1.0,
        "combos": scatter_data,
    }

    # Save using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    logger.info(f"Generated calibration metrics scatter: {png_path}")
    return str(png_path), str(json_path)


def generate_probability_distribution(
    combos_data: Dict[str, Dict],
    output_dir: Optional[Path] = None,
    filename: str = "fig_prob_distribution_stratos",
) -> Tuple[str, str]:
    """
    Generate probability distribution histograms for each combo.

    STRATOS recommends showing the distribution of predicted probabilities
    to assess model behavior.

    Parameters
    ----------
    combos_data : dict
        Output from load_stratos_data()
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename

    Returns
    -------
    png_path, json_path
    """
    setup_style()

    n_combos = len(combos_data)
    n_cols = min(4, n_combos)
    n_rows = (n_combos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )

    config = get_config_loader()

    hist_data = {}

    for idx, (combo_id, data) in enumerate(combos_data.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        y_true = data["y_true"]
        y_prob = data["y_prob"]

        # Get display name
        try:
            combo_config = config.get_combo_config(combo_id)
            display_name = combo_config.get("display_name", combo_id)
        except Exception:
            display_name = combo_id

        # Histogram for positive and negative cases
        bins = np.linspace(0, 1, 21)

        ax.hist(
            y_prob[y_true == 0],
            bins=bins,
            alpha=0.6,
            color=COLORS["control"],
            label="Controls",
            density=True,
        )
        ax.hist(
            y_prob[y_true == 1],
            bins=bins,
            alpha=0.6,
            color=COLORS["glaucoma"],
            label="Cases",
            density=True,
        )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title(display_name, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

        # Store histogram data
        hist_data[combo_id] = {
            "bins": bins.tolist(),
            "controls_probs": y_prob[y_true == 0].tolist(),
            "cases_probs": y_prob[y_true == 1].tolist(),
        }

    # Hide empty subplots
    for idx in range(len(combos_data), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("Probability Distributions by Outcome (STRATOS)", fontsize=14, y=1.02)
    fig.tight_layout()

    # Save using figure system
    png_path = save_figure(fig, filename, data=hist_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    logger.info(f"Generated probability distribution: {png_path}")
    return str(png_path), str(json_path)


def generate_all_stratos_figures(
    db_path: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Generate all STRATOS-compliant figures.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    output_dir : Path, optional
        Output directory (default: uses save_figure default)

    Returns
    -------
    dict
        Maps figure name to (png_path, json_path)
    """
    logger.info(f"Generating STRATOS figures from {db_path}")

    # Load data
    combos_data = load_stratos_data(db_path)

    if not combos_data:
        logger.error("No combo data loaded. Cannot generate figures.")
        return {}

    results = {}

    # Generate each figure
    try:
        results["calibration"] = generate_calibration_stratos_figure(
            combos_data, output_dir, "fig_calibration_stratos"
        )
    except Exception as e:
        logger.error(f"Failed to generate calibration figure: {e}")

    try:
        results["dca"] = generate_dca_stratos_figure(
            combos_data, output_dir, "fig_dca_stratos"
        )
    except Exception as e:
        logger.error(f"Failed to generate DCA figure: {e}")

    try:
        results["calibration_scatter"] = generate_calibration_metrics_scatter(
            combos_data, output_dir, "fig_calibration_metrics_scatter"
        )
    except Exception as e:
        logger.error(f"Failed to generate calibration scatter: {e}")

    try:
        results["prob_distribution"] = generate_probability_distribution(
            combos_data, output_dir, "fig_prob_distribution_stratos"
        )
    except Exception as e:
        logger.error(f"Failed to generate probability distribution: {e}")

    logger.info(f"Generated {len(results)} STRATOS figures")
    return results


# ============================================================================
# CLI
# ============================================================================


def main():
    """Command-line interface for STRATOS figure generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate STRATOS-compliant figures")
    parser.add_argument(
        "command", choices=["generate", "list"], help="Command: generate or list"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: uses save_figure default)",
    )
    parser.add_argument(
        "--figure",
        choices=["all", "calibration", "dca", "scatter", "distribution"],
        default="all",
        help="Which figure to generate",
    )

    args = parser.parse_args()

    if args.command == "list":
        # List available combos in database
        combos_data = load_stratos_data(args.db)
        print(f"Available combos ({len(combos_data)}):")
        for combo_id, data in combos_data.items():
            print(f"  {combo_id}: {data['n_samples']} samples")
        return

    if args.command == "generate":
        if args.figure == "all":
            results = generate_all_stratos_figures(args.db, args.output_dir)
            for name, paths in results.items():
                print(f"Generated {name}: {paths[0]}")
        else:
            combos_data = load_stratos_data(args.db)
            if args.figure == "calibration":
                paths = generate_calibration_stratos_figure(
                    combos_data, args.output_dir
                )
            elif args.figure == "dca":
                paths = generate_dca_stratos_figure(combos_data, args.output_dir)
            elif args.figure == "scatter":
                paths = generate_calibration_metrics_scatter(
                    combos_data, args.output_dir
                )
            elif args.figure == "distribution":
                paths = generate_probability_distribution(combos_data, args.output_dir)
            print(f"Generated: {paths[0]}")


if __name__ == "__main__":
    main()
