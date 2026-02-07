"""Decision Curve Analysis (DCA) visualization module.

Implements STRATOS-compliant DCA plots for clinical utility assessment.
Based on Vickers & Elkin 2006 and Van Calster et al. 2024 guidelines.

Architecture (CRITICAL-FAILURE-003 compliant):
-----------------------------------------------
- Pure-math net benefit formulas (compute_net_benefit, compute_treat_all_nb,
  compute_treat_none_nb, compute_dca_curves) are ACCEPTABLE: they are simple
  TP/FP arithmetic with NO sklearn or src.stats imports.
- DCA curve data for production figures is loaded from DuckDB via
  load_dca_curves_from_db(). All metric computation happens in extraction.
- NO imports from src.stats. NO sklearn imports.

See: https://github.com/petteriTeikari/foundation_PLR/issues/13
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

import numpy as np

from src.viz.figure_dimensions import get_dimensions
from src.viz.plot_config import COLORS, save_figure

__all__ = [
    # Net benefit computation (pure math, no sklearn)
    "compute_net_benefit",
    "compute_treat_all_nb",
    "compute_treat_none_nb",
    "compute_dca_curves",
    # DB loading
    "load_dca_curves_from_db",
    # Plotting
    "plot_dca",
    "plot_dca_multi_model",
    "plot_dca_from_db",
    # Figure generation
    "generate_dca_figure",
]


def compute_net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> float:
    """
    Compute net benefit at a given threshold probability.

    Net Benefit = TP/n - FP/n * (pt / (1-pt))

    Where pt is the threshold probability.

    This is a pure-math formula (no sklearn, no src.stats). Acceptable in viz.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_prob : array-like
        Predicted probabilities
    threshold : float
        Decision threshold probability

    Returns
    -------
    float
        Net benefit at the given threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    if n == 0:
        return np.nan

    # Avoid division by zero
    if threshold >= 1.0:
        return 0.0
    if threshold <= 0.0:
        threshold = 1e-10

    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    odds = threshold / (1 - threshold)
    nb = tp / n - fp / n * odds

    return nb


def compute_treat_all_nb(prevalence: float, threshold: float) -> float:
    """
    Compute net benefit for treat-all strategy.

    NB(treat-all) = prevalence - (1 - prevalence) * (pt / (1-pt))

    Parameters
    ----------
    prevalence : float
        Disease prevalence in the population
    threshold : float
        Decision threshold probability

    Returns
    -------
    float
        Net benefit for treat-all strategy
    """
    if threshold >= 1.0:
        return -np.inf
    if threshold <= 0.0:
        threshold = 1e-10

    odds = threshold / (1 - threshold)
    return prevalence - (1 - prevalence) * odds


def compute_treat_none_nb(threshold: float) -> float:
    """
    Compute net benefit for treat-none strategy.

    NB(treat-none) = 0 (always)

    Parameters
    ----------
    threshold : float
        Decision threshold probability (unused, for interface consistency)

    Returns
    -------
    float
        Always returns 0.0
    """
    return 0.0


def compute_dca_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    threshold_range: Tuple[float, float] = (0.01, 0.30),
    n_thresholds: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute DCA curves for model, treat-all, and treat-none strategies.

    Uses only pure-math net benefit formulas (no sklearn, no src.stats).

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    thresholds : array-like, optional
        Specific thresholds to evaluate. If None, uses threshold_range.
    threshold_range : tuple
        (min, max) threshold range (default: 1-30% for glaucoma)
    n_thresholds : int
        Number of threshold points to evaluate

    Returns
    -------
    dict with keys:
        - thresholds: array of threshold probabilities
        - nb_model: net benefit for model at each threshold
        - nb_all: net benefit for treat-all at each threshold
        - nb_none: net benefit for treat-none at each threshold
        - prevalence: disease prevalence
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if thresholds is None:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    prevalence = y_true.mean()

    nb_model = np.array([compute_net_benefit(y_true, y_prob, t) for t in thresholds])
    nb_all = np.array([compute_treat_all_nb(prevalence, t) for t in thresholds])
    nb_none = np.array([compute_treat_none_nb(t) for t in thresholds])

    return {
        "thresholds": thresholds,
        "nb_model": nb_model,
        "nb_all": nb_all,
        "nb_none": nb_none,
        "prevalence": prevalence,
    }


# ============================================================================
# DB LOADING (COMPUTATION DECOUPLING COMPLIANT)
# ============================================================================


def load_dca_curves_from_db(
    db_path: str,
    combo_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load pre-computed DCA curves from DuckDB.

    Supports two DuckDB schemas:
    1. Streaming schema (config_id + per-row thresholds): joins essential_metrics
       to find matching configs by outlier/imputation/classifier.
    2. Curve extraction schema (run_id + JSON arrays): joins essential_metrics
       by run_id to resolve combo matching.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    combo_ids : list of str, optional
        Specific combo IDs to load. If None, loads standard combos from config.

    Returns
    -------
    dict
        Maps combo_id to dict with keys:
        - thresholds: np.ndarray of threshold values
        - nb_model: np.ndarray of model net benefits
        - nb_all: np.ndarray of treat-all net benefits
        - nb_none: np.ndarray of treat-none net benefits
    """
    import duckdb

    from src.viz.config_loader import get_config_loader

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)

    dca_data = {}

    try:
        # Determine schema by checking column names
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'dca_curves'"
            ).fetchall()
        }

        # Load standard combos from config if not specified
        if combo_ids is None:
            config = get_config_loader()
            standard_combos = config.get_standard_hyperparam_combos()
            combo_ids = [c["id"] for c in standard_combos]

        if "config_id" in columns and "threshold" in columns:
            # Streaming schema: one row per threshold per config_id
            dca_data = _load_dca_streaming_schema(conn, combo_ids)
        elif "run_id" in columns and "thresholds" in columns:
            # Curve extraction schema: JSON arrays per run_id
            dca_data = _load_dca_json_schema(conn, combo_ids)
        else:
            raise ValueError(f"Unrecognized dca_curves schema. Columns: {columns}")

    finally:
        conn.close()

    return dca_data


def _load_dca_streaming_schema(
    conn,
    combo_ids: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load DCA data from streaming schema (config_id + per-row thresholds)."""
    from src.viz.config_loader import get_config_loader
    from src.viz.plot_config import FIXED_CLASSIFIER

    dca_data = {}

    for combo_id in combo_ids:
        config = get_config_loader()
        try:
            combo_config = config.get_combo_config(combo_id)
        except Exception:
            continue

        outlier = combo_config.get("outlier_method", "")
        imputation = combo_config.get("imputation_method", "")
        classifier = combo_config.get("classifier", FIXED_CLASSIFIER)

        # Find config_id matching this combo
        result = conn.execute(
            """
            SELECT config_id FROM essential_metrics
            WHERE outlier_method = ? AND imputation_method = ? AND classifier = ?
            LIMIT 1
            """,
            [outlier, imputation, classifier],
        ).fetchone()

        if result is None:
            continue

        config_id = result[0]

        # Load DCA curve data for this config
        dca_df = conn.execute(
            """
            SELECT threshold, net_benefit_model, net_benefit_all, net_benefit_none
            FROM dca_curves
            WHERE config_id = ?
            ORDER BY threshold
            """,
            [config_id],
        ).fetchdf()

        if len(dca_df) == 0:
            continue

        dca_data[combo_id] = {
            "thresholds": dca_df["threshold"].values,
            "nb_model": dca_df["net_benefit_model"].values,
            "nb_all": dca_df["net_benefit_all"].values,
            "nb_none": dca_df["net_benefit_none"].values,
        }

    return dca_data


def _load_dca_json_schema(
    conn,
    combo_ids: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load DCA data from curve extraction schema (run_id + JSON arrays)."""
    from src.viz.config_loader import get_config_loader
    from src.viz.plot_config import FIXED_CLASSIFIER

    dca_data = {}

    for combo_id in combo_ids:
        config = get_config_loader()
        try:
            combo_config = config.get_combo_config(combo_id)
        except Exception:
            continue

        outlier = combo_config.get("outlier_method", "")
        imputation = combo_config.get("imputation_method", "")
        classifier = combo_config.get("classifier", FIXED_CLASSIFIER)

        # Find run_id matching this combo via essential_metrics
        result = conn.execute(
            """
            SELECT em.run_id FROM essential_metrics em
            WHERE em.outlier_method = ? AND em.imputation_method = ? AND em.classifier = ?
            LIMIT 1
            """,
            [outlier, imputation, classifier],
        ).fetchone()

        if result is None:
            continue

        run_id = result[0]

        # Load DCA curve data (JSON arrays)
        dca_row = conn.execute(
            "SELECT thresholds, nb_model, nb_all, nb_none FROM dca_curves WHERE run_id = ?",
            [run_id],
        ).fetchone()

        if dca_row is None:
            continue

        dca_data[combo_id] = {
            "thresholds": np.array(json.loads(dca_row[0])),
            "nb_model": np.array(json.loads(dca_row[1])),
            "nb_all": np.array(json.loads(dca_row[2])),
            "nb_none": np.array(json.loads(dca_row[3])),
        }

    return dca_data


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    threshold_range: Tuple[float, float] = (0.01, 0.30),
    n_thresholds: int = 50,
    model_label: str = "Model",
    model_color: Optional[str] = None,
    show_treat_all: bool = True,
    show_treat_none: bool = True,
    save_json_path: Optional[str] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot Decision Curve Analysis from raw predictions.

    Uses pure-math compute_dca_curves (no sklearn, no src.stats).

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    threshold_range : tuple
        (min, max) threshold range (default: 1-30% for glaucoma screening)
    n_thresholds : int
        Number of threshold points
    model_label : str
        Label for model in legend
    model_color : str, optional
        Color for model line
    show_treat_all : bool
        Whether to show treat-all reference line
    show_treat_none : bool
        Whether to show treat-none reference line
    save_json_path : str, optional
        If provided, saves JSON data for reproducibility

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    try:
        from src.viz.plot_config import setup_style
    except ImportError:
        from plot_config import setup_style
    setup_style()

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    # Compute DCA curves (pure math only)
    dca_data = compute_dca_curves(
        y_true, y_prob, threshold_range=threshold_range, n_thresholds=n_thresholds
    )

    thresholds = dca_data["thresholds"]
    nb_model = dca_data["nb_model"]
    nb_all = dca_data["nb_all"]
    nb_none = dca_data["nb_none"]

    # Plot model curve
    ax.plot(thresholds, nb_model, label=model_label, color=model_color, linewidth=2)

    # Plot treat-all reference
    if show_treat_all:
        ax.plot(
            thresholds,
            nb_all,
            "--",
            color=COLORS["text_secondary"],
            alpha=0.7,
            label="Treat All",
            linewidth=1.5,
        )

    # Plot treat-none reference
    if show_treat_none:
        ax.plot(
            thresholds,
            nb_none,
            ":",
            color=COLORS["text_primary"],
            alpha=0.5,
            label="Treat None",
            linewidth=1.5,
        )

    # Labels and formatting
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_xlim(threshold_range[0] - 0.01, threshold_range[1] + 0.01)

    # Set reasonable y-axis limits
    all_nb = np.concatenate([nb_model, nb_all[nb_all > -0.5]])
    if len(all_nb) > 0:
        y_min = max(-0.1, np.nanmin(all_nb) - 0.02)
        y_max = np.nanmax(all_nb) + 0.02
        ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper right")
    ax.axhline(y=0, color=COLORS["grid_lines"], linestyle="-", alpha=0.3)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save JSON data for reproducibility
    if save_json_path:
        json_data = {
            "thresholds": thresholds.tolist(),
            "nb_model": nb_model.tolist(),
            "nb_all": nb_all.tolist(),
            "nb_none": nb_none.tolist(),
            "prevalence": float(dca_data["prevalence"]),
            "threshold_range": list(threshold_range),
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
        }
        with open(save_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return fig, ax


def plot_dca_multi_model(
    models_data: Dict[str, Dict],
    ax: Optional["plt.Axes"] = None,
    threshold_range: Tuple[float, float] = (0.01, 0.30),
    n_thresholds: int = 50,
    colors: Optional[List[str]] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot DCA for multiple models on same axes from raw predictions.

    Uses pure-math compute_net_benefit (no sklearn, no src.stats).

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to {'y_true': ..., 'y_prob': ...}
    ax : matplotlib.axes.Axes, optional
    threshold_range : tuple
        (min, max) threshold range
    n_thresholds : int
        Number of threshold points
    colors : list, optional
        Colors for each model

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("single"))
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    # Get prevalence from first model
    first_data = list(models_data.values())[0]
    y_true = np.asarray(first_data["y_true"])
    prevalence = y_true.mean()

    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    # Plot each model
    for (model_name, data), color in zip(models_data.items(), colors):
        y_t = np.asarray(data["y_true"])
        y_p = np.asarray(data["y_prob"])

        nb_model = np.array([compute_net_benefit(y_t, y_p, t) for t in thresholds])
        ax.plot(thresholds, nb_model, label=model_name, color=color, linewidth=2)

    # Plot references
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

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_xlim(threshold_range[0] - 0.01, threshold_range[1] + 0.01)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_dca_from_db(
    db_path: str,
    combo_ids: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    output_dir: Optional[Path] = None,
    filename: str = "fig_dca_curves",
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plot DCA curves from pre-computed data in DuckDB.

    This is the PREFERRED method for production figures. Reads pre-computed
    DCA curves from the database (no on-the-fly computation).

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    combo_ids : list of str, optional
        Specific combo IDs to plot. If None, loads standard combos.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    output_dir : Path, optional
        Output directory for saving figure
    filename : str
        Base filename for saving

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    import matplotlib.pyplot as plt

    try:
        from src.viz.config_loader import get_config_loader
        from src.viz.plot_config import setup_style
    except ImportError:
        from config_loader import get_config_loader
        from plot_config import setup_style

    setup_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=get_dimensions("dca"))
    else:
        fig = ax.get_figure()

    # Load pre-computed DCA curves from DB
    dca_data = load_dca_curves_from_db(db_path, combo_ids=combo_ids)

    if not dca_data:
        ax.text(
            0.5,
            0.5,
            "No DCA data found in database",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return fig, ax

    # Get colors from config
    config = get_config_loader()
    colors = config.get_colors().get("combo_colors", {})

    # Plot each combo
    for combo_id, curves in dca_data.items():
        color = colors.get(combo_id, None)

        # Get display name
        try:
            combo_config = config.get_combo_config(combo_id)
            display_name = combo_config.get("display_name", combo_id)
        except Exception:
            display_name = combo_id

        ax.plot(
            curves["thresholds"],
            curves["nb_model"],
            label=display_name,
            color=color,
            linewidth=2,
        )

    # Plot treat-all and treat-none from the first combo's data
    first_curves = list(dca_data.values())[0]
    ax.plot(
        first_curves["thresholds"],
        first_curves["nb_all"],
        "--",
        color=COLORS["text_secondary"],
        alpha=0.7,
        label="Treat All",
        linewidth=1.5,
    )
    ax.plot(
        first_curves["thresholds"],
        first_curves["nb_none"],
        ":",
        color=COLORS["text_primary"],
        alpha=0.5,
        label="Treat None",
        linewidth=1.5,
    )

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    thresholds = first_curves["thresholds"]
    ax.set_xlim(thresholds.min() - 0.01, thresholds.max() + 0.01)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color=COLORS["grid_lines"], linestyle="-", alpha=0.3)

    # Set reasonable y-axis limits
    all_nb = []
    for curves in dca_data.values():
        all_nb.extend(curves["nb_model"].tolist())
    nb_all_vals = first_curves["nb_all"]
    all_nb.extend(nb_all_vals[nb_all_vals > -0.5].tolist())
    if all_nb:
        y_min = max(-0.1, np.nanmin(all_nb) - 0.02)
        y_max = np.nanmax(all_nb) + 0.02
        ax.set_ylim(y_min, y_max)

    # Prepare JSON data for reproducibility
    json_data = {
        "thresholds": thresholds.tolist(),
        "nb_all": nb_all_vals.tolist(),
        "nb_none": first_curves["nb_none"].tolist(),
        "combos": {
            combo_id: {
                "net_benefit": curves["nb_model"].tolist(),
            }
            for combo_id, curves in dca_data.items()
        },
    }

    # Save using figure system
    save_figure(fig, filename, data=json_data, output_dir=output_dir)

    return fig, ax


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================


def generate_dca_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Optional[Path] = None,
    filename: str = "fig_dca_curves",
    threshold_range: Tuple[float, float] = (0.01, 0.30),
) -> Tuple[str, str]:
    """
    Generate DCA plot from raw predictions and save to file.

    For production figures, prefer plot_dca_from_db() which reads
    pre-computed data from DuckDB.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    output_dir : Path, optional
        Output directory (default: uses save_figure default)
    filename : str
        Base filename (without extension)
    threshold_range : tuple
        (min, max) threshold range

    Returns
    -------
    png_path, json_path : paths to generated files
    """
    import matplotlib.pyplot as plt

    # Compute DCA curves for JSON data (pure math only)
    dca_curves_data = compute_dca_curves(
        y_true, y_prob, threshold_range=threshold_range
    )

    json_data = {
        "threshold_range": list(threshold_range),
        "thresholds": dca_curves_data["thresholds"].tolist(),
        "nb_model": dca_curves_data["nb_model"].tolist(),
        "nb_all": dca_curves_data["nb_all"].tolist(),
        "nb_none": dca_curves_data["nb_none"].tolist(),
    }

    fig, ax = plot_dca(y_true, y_prob, threshold_range=threshold_range)

    # Save using figure system
    png_path = save_figure(fig, filename, data=json_data, output_dir=output_dir)
    plt.close(fig)
    json_path = png_path.parent / "data" / f"{filename}.json"

    return str(png_path), str(json_path)
