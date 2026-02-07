"""
Retained Metric Visualization - Selective Classification Analysis.

# AIDEV-NOTE: This is a KEY figure for the manuscript (fig_retained_multi_metric).
# ARCHITECTURE: Config-driven, no hardcoding!
# - COMBOS: Load from configs/VISUALIZATION/plot_hyperparam_combos.yaml (standard=4, extended=8)
# - METRICS: Load from configs/VISUALIZATION/metrics.yaml (clinical, ml_standard, uncertainty, etc.)
# - Main vs Supplementary: controlled by config, not code
#
# RULES:
# 1. Load combos from YAML - NEVER hardcode combo names
# 2. Load metrics from YAML - NEVER hardcode metric lists
# 3. MUST include ground_truth in every comparison figure
# 4. Call setup_style() before any matplotlib operations
#
# COMPUTATION DECOUPLING (CRITICAL-FAILURE-003):
# This module is READ-ONLY for metrics. All retention curve data is pre-computed
# during extraction and stored in the DuckDB `retention_metrics` table.
# This module reads from DuckDB and plots. It NEVER computes metrics.

Plots metrics (AUROC, Brier, Net Benefit) at different retention rates based on
uncertainty estimates. This implements the "rejection curve" or "risk-coverage
curve" pattern from:

- Galil et al. 2023 (AURC - Area Under Risk-Coverage)
- Leibig et al. 2017 (Uncertainty in diabetic retinopathy)
- Filos et al. 2019 (BDL Benchmarks)
- STRATOS Initiative (Van Calster et al. 2024)

The key insight: by rejecting uncertain predictions and keeping only
confident ones, metrics typically improve on the retained subset.

Usage:
    >>> from src.viz.retained_metric import load_retention_curve_from_db
    >>> rates, values = load_retention_curve_from_db("ground_truth__pupil-gt", "auroc")
    >>> fig, ax = plot_retention_curve(rates, values, metric_name="auroc")

    # Multiple combos from DB
    >>> fig, axes = generate_multi_combo_retention_figure(db_path="/path/to/db")
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from src.viz.config_loader import ConfigLoader

import numpy as np

from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import save_figure

__all__ = [
    # DB loading
    "load_retention_curve_from_db",
    "load_all_retention_curves_from_db",
    # Plotting
    "plot_retention_curve",
    "plot_multi_metric_retention",
    "plot_multi_model_retention",
    # Figure generation
    "generate_retention_figures",
    "generate_multi_combo_retention_figure",
    # Config helpers
    "get_metric_combo",
    "get_metric_label",
    "load_combos_from_yaml",
    # Display labels
    "METRIC_LABELS",
]


# ============================================================================
# DATABASE LOADING - Read pre-computed retention curves from DuckDB
# ============================================================================


def load_retention_curve_from_db(
    config_id: str,
    metric_name: str,
    db_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pre-computed retention curve from the DuckDB retention_metrics table.

    Parameters
    ----------
    config_id : str
        The config identifier (e.g., "ground_truth__pupil-gt", "LOF__SAITS")
    metric_name : str
        The metric name (e.g., "auroc", "scaled_brier", "net_benefit")
    db_path : str, optional
        Path to DuckDB database. If None, uses the default from plot_config.

    Returns
    -------
    retention_rates : ndarray
        Array of retention rate values (ascending)
    metric_values : ndarray
        Array of metric values at each retention rate

    Raises
    ------
    ValueError
        If no data found for the given config_id and metric_name
    """
    import duckdb

    if db_path is not None:
        conn = duckdb.connect(str(db_path), read_only=True)
    else:
        from src.viz.plot_config import get_connection

        conn = get_connection()

    try:
        df = conn.execute(
            """
            SELECT retention_rate, metric_value
            FROM retention_metrics
            WHERE config_id = ? AND metric_name = ?
            ORDER BY retention_rate ASC
            """,
            [config_id, metric_name],
        ).fetchdf()
    finally:
        conn.close()

    if df.empty:
        raise ValueError(
            f"No retention data found for config_id='{config_id}', "
            f"metric_name='{metric_name}'. "
            f"Check that the retention_metrics table has been populated by extraction."
        )

    return df["retention_rate"].values, df["metric_value"].values


def load_all_retention_curves_from_db(
    metric_name: str,
    db_path: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all retention curves for a given metric from DuckDB.

    Parameters
    ----------
    metric_name : str
        The metric name (e.g., "auroc", "scaled_brier")
    db_path : str, optional
        Path to DuckDB database. If None, uses the default from plot_config.

    Returns
    -------
    dict
        Mapping of config_id -> (retention_rates, metric_values)
    """
    import duckdb

    if db_path is not None:
        conn = duckdb.connect(str(db_path), read_only=True)
    else:
        from src.viz.plot_config import get_connection

        conn = get_connection()

    try:
        df = conn.execute(
            """
            SELECT config_id, retention_rate, metric_value
            FROM retention_metrics
            WHERE metric_name = ?
            ORDER BY config_id, retention_rate ASC
            """,
            [metric_name],
        ).fetchdf()
    finally:
        conn.close()

    result = {}
    for config_id, group_df in df.groupby("config_id"):
        result[config_id] = (
            group_df["retention_rate"].values,
            group_df["metric_value"].values,
        )

    return result


# ============================================================================
# CONFIG HELPERS
# ============================================================================


def _get_loader() -> "ConfigLoader":
    """Get the centralized config loader with caching.

    Uses configs/VISUALIZATION/ (consolidated from old config/ directory).
    """
    try:
        from src.viz.config_loader import get_config_loader
    except ImportError:
        from config_loader import get_config_loader
    return get_config_loader()


def get_metric_combo(combo_name: Optional[str] = None) -> List[str]:
    """Get list of metrics for a named combo from config.

    Uses centralized config loader with caching.
    """
    loader = _get_loader()
    return loader.get_metric_combo(combo_name)


def get_metric_label(metric_name: str) -> str:
    """Get display label for a metric from config.

    Uses centralized config loader with caching.
    """
    loader = _get_loader()
    label = loader.get_metric_label(metric_name)
    if label == metric_name:
        # Fallback to hardcoded labels for backwards compatibility
        return METRIC_LABELS.get(metric_name, metric_name)
    return label


# Display labels only (not computation) - backwards compatibility
METRIC_LABELS: Dict[str, str] = {
    "auroc": "AUROC",
    "brier": "Negative Brier Score",
    "scaled_brier": "Scaled Brier (IPA)",
    "net_benefit": "Net Benefit",
    "f1": "F1 Score",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def load_combos_from_yaml(combo_set: str = "standard") -> List[Dict]:
    """Load combos from configs/VISUALIZATION/plot_hyperparam_combos.yaml.

    Uses centralized config loader with caching.

    Parameters
    ----------
    combo_set : str
        'standard' for 4 main figure combos, 'extended' for supplementary,
        'all' for both.
    """
    loader = _get_loader()

    if combo_set == "standard":
        return loader.get_standard_hyperparam_combos()
    elif combo_set == "extended":
        return loader.get_extended_hyperparam_combos()
    elif combo_set == "all":
        return (
            loader.get_standard_hyperparam_combos()
            + loader.get_extended_hyperparam_combos()
        )
    else:
        raise ValueError(
            f"Unknown combo_set: {combo_set}. Use 'standard', 'extended', or 'all'."
        )


# ============================================================================
# PLOTTING - Accepts pre-computed data, NEVER computes metrics
# ============================================================================


def plot_retention_curve(
    retention_rates: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str = "auroc",
    ax: Optional["plt.Axes"] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    show_baseline: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot metric vs retention rate curve from pre-computed data.

    Parameters
    ----------
    retention_rates : array-like
        Retention rate values (x-axis)
    metric_values : array-like
        Metric values at each retention rate (y-axis)
    metric_name : str
        Metric name for axis label lookup
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    label : str, optional
        Legend label
    color : str, optional
        Line color
    show_baseline : bool
        Whether to show baseline (100% retention) line

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    plot_kwargs = {"label": label}
    if color is not None:
        plot_kwargs["color"] = color

    ax.plot(retention_rates, metric_values, **plot_kwargs)

    # Baseline (full dataset)
    if show_baseline and len(metric_values) > 0:
        baseline = metric_values[-1]  # 100% retention
        if np.isfinite(baseline):
            ax.axhline(
                y=baseline,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Baseline (100%)",
            )

    ax.set_xlabel("Fraction of Data Retained")
    ax.set_ylabel(METRIC_LABELS.get(metric_name, str(metric_name)))
    ax.set_xlim(0, 1.05)

    if label:
        ax.legend()

    return fig, ax


def plot_multi_metric_retention(
    retention_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["plt.Figure", np.ndarray]:
    """
    Create subplot grid with retention curves for multiple metrics.

    Parameters
    ----------
    retention_data : dict
        Mapping of metric_name -> (retention_rates, metric_values).
        Each key should be a metric name with its pre-computed curve data.
    metrics : list of str, optional
        Metrics to plot (subset of retention_data keys).
        Default: all keys in retention_data, or clinical combo from config.
    figsize : tuple, optional
        Figure size. Default: (5*n_metrics, 5)

    Returns
    -------
    fig : Figure
    axes : array of Axes
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        if retention_data:
            metrics = list(retention_data.keys())
        else:
            metrics = get_metric_combo("clinical")

    n_metrics = len(metrics)
    if figsize is None:
        figsize = (5 * n_metrics, 5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in retention_data:
            rates, values = retention_data[metric]
            plot_retention_curve(
                rates,
                values,
                metric_name=metric,
                ax=ax,
                show_baseline=True,
            )
        ax.set_title(f"Retained vs {get_metric_label(metric)}")

    plt.tight_layout()
    return fig, axes


def plot_multi_model_retention(
    data: Dict[str, Dict],
    metric_name: str = "auroc",
    ax: Optional["plt.Axes"] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot retention curves for multiple models on same axes.

    Parameters
    ----------
    data : dict
        Dictionary mapping model names to their data dicts.
        Each dict should have keys: 'retention_rates', 'metric_values'
        and optionally 'color'.
    metric_name : str
        Metric name for axis label
    ax : Axes, optional
        Axes to plot on

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for (model_name, model_data), default_color in zip(data.items(), colors):
        color = model_data.get("color", default_color)
        plot_retention_curve(
            np.asarray(model_data["retention_rates"]),
            np.asarray(model_data["metric_values"]),
            metric_name=metric_name,
            ax=ax,
            label=model_name,
            color=color,
            show_baseline=False,
        )

    ax.legend()
    ax.set_title(f"Retained vs {get_metric_label(metric_name)}")

    return fig, ax


# ============================================================================
# FIGURE GENERATION FUNCTIONS - Read from DuckDB, plot, save
# ============================================================================


def generate_retention_figures(
    combo_retention_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate retention curve figures from pre-computed data and save to files.

    Parameters
    ----------
    combo_retention_data : dict
        Nested dict: combo_id -> metric_name -> (retention_rates, metric_values).
        Pre-computed data loaded from DuckDB.
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    metrics : list of str, optional
        Metrics to generate figures for. Default: clinical combo from config.

    Returns
    -------
    dict
        Mapping of figure names to paths
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = get_metric_combo("clinical")

    paths = {}

    # Generate individual metric figures (aggregate across combos)
    for metric in metrics:
        filename = f"fig_retained_{metric.replace('_', '')}"

        fig, ax = plt.subplots(figsize=get_dimensions("single"))

        for combo_id, metric_data in combo_retention_data.items():
            if metric in metric_data:
                rates, values = metric_data[metric]
                plot_retention_curve(
                    rates,
                    values,
                    metric_name=metric,
                    ax=ax,
                    label=combo_id,
                    show_baseline=False,
                )

        ax.set_title(f"Retained vs {get_metric_label(metric)}")
        ax.legend(loc="lower right", fontsize=8)

        # Build JSON data for reproducibility
        json_data = {
            "metric": metric,
            "combos": {},
        }
        for combo_id, metric_data in combo_retention_data.items():
            if metric in metric_data:
                rates, values = metric_data[metric]
                json_data["combos"][combo_id] = {
                    "retention_rates": rates.tolist(),
                    "metric_values": [v if not np.isnan(v) else None for v in values],
                }

        png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
        paths[filename] = str(png_path)
        plt.close(fig)

    # Generate multi-metric figure
    filename = "fig_retained_multi_metric"
    multi_json = {"metrics": {}}

    # Aggregate: for the multi-metric view, pick first combo or merge
    # Use first combo as representative for single-combo multi-metric view
    first_combo_id = next(iter(combo_retention_data), None)
    if first_combo_id is not None:
        retention_data_for_plot = {}
        for metric in metrics:
            if metric in combo_retention_data[first_combo_id]:
                rates, values = combo_retention_data[first_combo_id][metric]
                retention_data_for_plot[metric] = (rates, values)
                multi_json["metrics"][metric] = {
                    "retention_rates": rates.tolist(),
                    "metric_values": [v if not np.isnan(v) else None for v in values],
                }

        fig, axes = plot_multi_metric_retention(
            retention_data_for_plot, metrics=metrics
        )
        png_path = save_figure(fig, filename, data=multi_json, output_dir=output_dir)
        paths[filename] = str(png_path)
        plt.close(fig)

    return paths


def generate_multi_combo_retention_figure(
    combo_retention_data: Optional[Dict[str, Dict]] = None,
    output_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate retention curves for multiple combos (4 standard combos).

    Reads pre-computed retention curve data from DuckDB.

    Parameters
    ----------
    combo_retention_data : dict, optional
        Pre-loaded data: combo_id -> {
            'retention_rates': {metric_name: array},
            'metric_values': {metric_name: array},
            'color': str,
            'label': str
        }
        If None, loads from DuckDB using standard combos.
    output_dir : Path, optional
        Output directory. If None, uses save_figure default.
    metrics : list of str, optional
        Metrics to plot. Default: clinical combo from config.
    db_path : str, optional
        Path to DuckDB database. Used when combo_retention_data is None.

    Returns
    -------
    dict
        Mapping of figure names to output paths
    """
    import matplotlib.pyplot as plt

    try:
        from src.viz.plot_config import get_combo_color, save_figure, setup_style
    except ImportError:
        from plot_config import get_combo_color, save_figure, setup_style

    setup_style()

    if metrics is None:
        metrics = get_metric_combo("clinical")

    # If no pre-loaded data, load from DB using standard combos
    if combo_retention_data is None:
        combos = load_combos_from_yaml("standard")
        combo_retention_data = {}

        for combo in combos:
            combo_id = combo["id"]
            # Build config_id from combo parameters
            outlier = combo.get("outlier_method", "")
            imputation = combo.get("imputation_method", "")
            config_id = f"{outlier}__{imputation}" if imputation else outlier

            entry = {
                "color": get_combo_color(combo_id),
                "label": combo.get("name", combo_id),
                "retention_rates": {},
                "metric_values": {},
            }

            for metric in metrics:
                try:
                    rates, values = load_retention_curve_from_db(
                        config_id, metric, db_path=db_path
                    )
                    entry["retention_rates"][metric] = rates
                    entry["metric_values"][metric] = values
                except ValueError:
                    # No data for this config_id/metric combination
                    pass

            if entry["retention_rates"]:
                combo_retention_data[combo_id] = entry

    # Create multi-metric figure with all combos
    n_metrics = len(metrics)
    base_width, base_height = get_dimensions("double_short")
    fig, axes = plt.subplots(1, n_metrics, figsize=(base_width, base_height))

    if n_metrics == 1:
        axes = [axes]

    # Plot each combo on each metric subplot
    for ax, metric in zip(axes, metrics):
        for combo_id, data in combo_retention_data.items():
            rates = data.get("retention_rates", {}).get(metric)
            values = data.get("metric_values", {}).get(metric)
            if rates is None or values is None:
                continue

            color = data.get("color", get_combo_color(combo_id))
            label = data.get("label", combo_id)

            ax.plot(
                rates,
                values,
                color=color,
                label=label,
                linewidth=1.5,
            )

        ax.set_xlabel("Fraction of Data Retained")
        ax.set_ylabel(get_metric_label(metric))
        ax.set_xlim(0.1, 1.05)
        ax.set_title(f"{get_metric_label(metric)}")
        ax.legend(loc="lower right", fontsize=8)

        # Style adjustments
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # Build JSON data for reproducibility
    json_data = {
        "combos": {},
        "metrics": metrics,
    }
    for combo_id, data in combo_retention_data.items():
        combo_json = {
            "label": data.get("label", combo_id),
        }
        for metric in metrics:
            rates = data.get("retention_rates", {}).get(metric)
            values = data.get("metric_values", {}).get(metric)
            if rates is not None and values is not None:
                combo_json[metric] = {
                    "retention_rates": (
                        rates.tolist() if hasattr(rates, "tolist") else list(rates)
                    ),
                    "metric_values": [
                        v if not np.isnan(v) else None
                        for v in (
                            values.tolist()
                            if hasattr(values, "tolist")
                            else list(values)
                        )
                    ],
                }
        json_data["combos"][combo_id] = combo_json

    # Save figure and JSON using centralized save_figure
    filename = "fig_retained_multi_metric"
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)

    return {filename: str(png_path)}
