"""
Metric Registry - Centralized metric definitions for visualization.

This module provides:
- MetricDefinition: Dataclass for metric specifications (display name, format, etc.)
- MetricRegistry: Central registry for all metrics
- Standard metrics pre-registered (AUROC, MAE, Brier, etc.)

ARCHITECTURE NOTE (CRITICAL-FAILURE-003):
-----------------------------------------
This is a DUAL-USE module:

1. **For Visualization (src/viz/)**: Use MetricRegistry.get() to access metric
   metadata (display_name, format_str, higher_is_better). Do NOT use compute_fn.

2. **For Extraction (scripts/extract_*.py)**: The compute_fn fields provide
   metric computation functions that are used ONLY during extraction to compute
   metrics and store them in DuckDB.

The compute functions (_compute_auroc, _compute_brier, etc.) import sklearn.metrics
which is BANNED in visualization code. These functions should ONLY be called
by extraction scripts, never by visualization modules.

Example usage (VISUALIZATION - READ metadata only):
    >>> from src.viz.metric_registry import MetricRegistry
    >>> metric = MetricRegistry.get('auroc')
    >>> print(metric.display_name)  # "AUROC"
    >>> print(metric.higher_is_better)  # True
    >>> # Do NOT call: metric.compute_fn(y_true, y_prob)  # BANNED in viz!

Example usage (EXTRACTION - compute and store):
    >>> # In extraction scripts only:
    >>> metric = MetricRegistry.get('auroc')
    >>> value = metric.compute_fn(y_true, y_prob)  # OK in extraction
    >>> conn.execute("INSERT INTO metrics (auroc) VALUES (?)", [value])
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class MetricDefinition:
    """
    Definition of a metric for visualization.

    Attributes
    ----------
    name : str
        Internal name (e.g., 'auroc', 'brier')
    display_name : str
        Human-readable name for plots (e.g., 'AUROC', 'Brier Score')
    higher_is_better : bool
        True if higher values are better (for coloring/ranking)
    unit : str
        Unit string if applicable (e.g., '%', 'ms')
    format_str : str
        Format string for display (e.g., '.3f', '.1%')
    value_range : tuple
        Expected (min, max) range for validation
    compute_fn : callable, optional
        Function(y_true, y_prob, **kwargs) -> float
    """

    name: str
    display_name: str
    higher_is_better: Optional[bool] = True
    unit: str = ""
    format_str: str = ".3f"
    value_range: Tuple[float, float] = (0.0, 1.0)
    compute_fn: Optional[Callable[..., float]] = None

    def format_value(self, value: float) -> str:
        """Format a value for display."""
        if self.unit == "%":
            return f"{value * 100:{self.format_str}}{self.unit}"
        return f"{value:{self.format_str}}{self.unit}"

    def is_better(self, a: float, b: float) -> bool:
        """Return True if a is better than b."""
        if self.higher_is_better:
            return a > b
        return a < b


class MetricRegistry:
    """
    Registry of all metrics used in visualizations.

    This is a singleton-like class with class methods for registration
    and retrieval. Metrics are registered once at import time.

    Example
    -------
    >>> MetricRegistry.register(MetricDefinition(
    ...     name='my_metric',
    ...     display_name='My Custom Metric',
    ...     higher_is_better=True
    ... ))
    >>> metric = MetricRegistry.get('my_metric')
    """

    _metrics: Dict[str, MetricDefinition] = {}

    @classmethod
    def register(cls, metric: MetricDefinition) -> None:
        """Register a metric definition."""
        cls._metrics[metric.name] = metric

    @classmethod
    def get(cls, name: str) -> MetricDefinition:
        """
        Get metric by name.

        Raises KeyError if metric not found.
        """
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise KeyError(f"Unknown metric: '{name}'. Available metrics: {available}")
        return cls._metrics[name]

    @classmethod
    def get_or_default(cls, name: str) -> MetricDefinition:
        """
        Get metric by name, or create a default one if not found.

        Useful for ad-hoc metrics from database columns.
        """
        if name in cls._metrics:
            return cls._metrics[name]
        # Create default definition
        return MetricDefinition(
            name=name,
            display_name=name.replace("_", " ").title(),
            higher_is_better=True,
        )

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names."""
        return sorted(cls._metrics.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a metric is registered."""
        return name in cls._metrics


# =============================================================================
# STANDARD METRIC COMPUTE FUNCTIONS
# =============================================================================
#
# CRITICAL-FAILURE-003 NOTE:
# These functions are for EXTRACTION USE ONLY. They import sklearn.metrics
# which is BANNED in visualization code. The viz layer should read pre-computed
# metrics from DuckDB, not call these functions.
#
# These functions are attached to MetricDefinition.compute_fn for use by:
# - scripts/extract_all_configs_to_duckdb.py
# - src/orchestration/flows/extraction_flow.py
# - Unit tests
#
# NEVER import or call these from src/viz/ modules!
# =============================================================================


def _compute_auroc(y_true: NDArray[np.floating], y_prob: NDArray[np.floating]) -> float:
    """Compute AUROC (Area Under ROC Curve).

    NOTE: For EXTRACTION use only. Viz code should read from DuckDB.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def _compute_brier(y_true: NDArray[np.floating], y_prob: NDArray[np.floating]) -> float:
    """Compute Brier score (lower is better)."""
    from sklearn.metrics import brier_score_loss

    return brier_score_loss(y_true, y_prob)


def _compute_scaled_brier(
    y_true: NDArray[np.floating], y_prob: NDArray[np.floating]
) -> float:
    """
    Compute scaled Brier score (IPA - Index of Prediction Accuracy).

    0 = null model (predicting prevalence), 1 = perfect model.
    Can be negative for worse-than-null models.
    """
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(y_true, y_prob)
    prevalence = y_true.mean()
    brier_null = prevalence * (1 - prevalence)
    if brier_null == 0:
        return np.nan
    return 1 - brier / brier_null


def _compute_net_benefit(
    y_true: NDArray[np.floating], y_prob: NDArray[np.floating], threshold: float = 0.15
) -> float:
    """
    Compute Net Benefit at given threshold probability.

    NB = TP/n - FP/n * (pt / (1-pt))
    """
    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / n - fp / n * threshold / (1 - threshold)


def _compute_mae(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Compute Mean Absolute Error (lower is better)."""
    return np.mean(np.abs(y_true - y_pred))


def _compute_f1(
    y_true: NDArray[np.floating], y_prob: NDArray[np.floating], threshold: float = 0.5
) -> float:
    """Compute F1 score at given threshold."""
    from sklearn.metrics import f1_score

    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred)


def _compute_sensitivity(
    y_true: NDArray[np.floating], y_prob: NDArray[np.floating], threshold: float = 0.5
) -> float:
    """Compute sensitivity (recall) at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if (tp + fn) == 0:
        return np.nan
    return tp / (tp + fn)


def _compute_specificity(
    y_true: NDArray[np.floating], y_prob: NDArray[np.floating], threshold: float = 0.5
) -> float:
    """Compute specificity at given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    if (tn + fp) == 0:
        return np.nan
    return tn / (tn + fp)


# =============================================================================
# REGISTER STANDARD METRICS
# =============================================================================

# Classification metrics
MetricRegistry.register(
    MetricDefinition(
        name="auroc",
        display_name="AUROC",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
        compute_fn=_compute_auroc,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="brier",
        display_name="Brier Score",
        higher_is_better=False,
        format_str=".4f",
        value_range=(0.0, 1.0),
        compute_fn=_compute_brier,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="scaled_brier",
        display_name="Scaled Brier (IPA)",
        higher_is_better=True,
        format_str=".3f",
        value_range=(-1.0, 1.0),
        compute_fn=_compute_scaled_brier,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="net_benefit",
        display_name="Net Benefit",
        higher_is_better=True,
        format_str=".3f",
        value_range=(-1.0, 1.0),
        compute_fn=_compute_net_benefit,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="f1",
        display_name="F1 Score",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
        compute_fn=_compute_f1,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="sensitivity",
        display_name="Sensitivity",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
        compute_fn=_compute_sensitivity,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="specificity",
        display_name="Specificity",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
        compute_fn=_compute_specificity,
    )
)

# Regression/imputation metrics
MetricRegistry.register(
    MetricDefinition(
        name="mae",
        display_name="MAE",
        higher_is_better=False,
        format_str=".4f",
        value_range=(0.0, float("inf")),
        compute_fn=_compute_mae,
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="rmse",
        display_name="RMSE",
        higher_is_better=False,
        format_str=".4f",
        value_range=(0.0, float("inf")),
    )
)

# Outlier detection metrics
MetricRegistry.register(
    MetricDefinition(
        name="outlier_f1",
        display_name="Outlier F1",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="outlier_precision",
        display_name="Outlier Precision",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="outlier_recall",
        display_name="Outlier Recall",
        higher_is_better=True,
        format_str=".3f",
        value_range=(0.0, 1.0),
    )
)

# Calibration metrics (STRATOS-compliant)
MetricRegistry.register(
    MetricDefinition(
        name="calibration_slope",
        display_name="Calibration Slope",
        higher_is_better=None,  # Target is 1.0, not higher/lower
        format_str=".3f",
        value_range=(0.0, 2.0),
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="calibration_intercept",
        display_name="Calibration Intercept",
        higher_is_better=None,  # Target is 0.0, not higher/lower
        format_str=".3f",
        value_range=(-1.0, 1.0),
    )
)

MetricRegistry.register(
    MetricDefinition(
        name="o_e_ratio",
        display_name="O:E Ratio",
        higher_is_better=None,  # Target is 1.0, not higher/lower
        format_str=".3f",
        value_range=(0.0, 3.0),
    )
)

# =============================================================================
# STRATOS METRIC SETS (Van Calster 2024)
# =============================================================================
# These sets define which metrics to report together per STRATOS guidelines

STRATOS_METRIC_SETS = {
    # Core set: MANDATORY for all reporting
    "stratos_core": [
        "auroc",  # Discrimination
        "brier",  # Overall performance
        "scaled_brier",  # Overall performance (interpretable)
        "calibration_slope",  # Calibration (weak)
        "calibration_intercept",  # Calibration (mean)
        "o_e_ratio",  # Calibration (mean)
        "net_benefit",  # Clinical utility
    ],
    # Discrimination only
    "discrimination": [
        "auroc",
        "sensitivity",
        "specificity",
    ],
    # Calibration only
    "calibration": [
        "brier",
        "scaled_brier",
        "calibration_slope",
        "calibration_intercept",
        "o_e_ratio",
    ],
    # Clinical utility only
    "clinical_utility": [
        "net_benefit",
    ],
    # Outlier detection
    "outlier_detection": [
        "outlier_f1",
        "outlier_precision",
        "outlier_recall",
    ],
    # Imputation/reconstruction
    "imputation": [
        "mae",
        "rmse",
    ],
    # Full STRATOS for manuscript figures
    "manuscript_full": [
        "auroc",
        "brier",
        "scaled_brier",
        "calibration_slope",
        "o_e_ratio",
        "net_benefit",
        "sensitivity",
        "specificity",
    ],
}


def get_metric_set(set_name: str) -> List[MetricDefinition]:
    """
    Get a predefined set of metrics by name.

    Parameters
    ----------
    set_name : str
        Name of the metric set (e.g., 'stratos_core', 'calibration')

    Returns
    -------
    list of MetricDefinition
        The metrics in the requested set

    Raises
    ------
    KeyError
        If set_name is not a valid metric set
    """
    if set_name not in STRATOS_METRIC_SETS:
        available = list(STRATOS_METRIC_SETS.keys())
        raise KeyError(f"Unknown metric set: '{set_name}'. Available: {available}")

    return [
        MetricRegistry.get_or_default(name) for name in STRATOS_METRIC_SETS[set_name]
    ]


def list_metric_sets() -> List[str]:
    """List all available metric set names."""
    return list(STRATOS_METRIC_SETS.keys())
