"""
Figure data export utilities.

Saves underlying data for each generated figure as JSON,
enabling re-plotting without re-running the full pipeline.

Data format: JSON (universal, human-readable, Excel-compatible)
- Single consistent format for ALL figures
- Hierarchical structure via nested dicts
- Arrays stored as lists

Cross-references:
- planning/figure-and-stats-creation-plan.md (Figure Data Export Convention)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

__all__ = [
    "save_figure_data",
    "load_figure_data",
    "FigureDataExporter",
]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_figure_data(
    figure_id: str,
    figure_title: str,
    data: Dict[str, Any],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    reference_lines: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Save figure data as JSON.

    Parameters
    ----------
    figure_id : str
        Unique identifier for the figure (e.g., "fig01", "fig02")
    figure_title : str
        Human-readable title for the figure
    data : dict
        The underlying data for the figure. Structure varies by figure type:
        - Forest plots: {"methods": [...], "estimates": [...], "ci_lower": [...], "ci_upper": [...]}
        - Heatmaps: {"row_labels": [...], "col_labels": [...], "values": [[...]]}
        - CD diagrams: {"methods": [...], "ranks": [...], "cliques": [...]}
    output_path : str or Path
        Path to save the JSON file
    metadata : dict, optional
        Additional metadata (e.g., n_observations, parameters used)
    reference_lines : dict, optional
        Reference lines to include in the plot
        Format: {"line_name": {"value": float, "label": str}}

    Returns
    -------
    Path
        Path to the saved JSON file

    Examples
    --------
    >>> save_figure_data(
    ...     figure_id="fig02",
    ...     figure_title="Outlier Method Comparison",
    ...     data={
    ...         "methods": ["IQR", "MAD", "ZScore"],
    ...         "estimates": [0.89, 0.87, 0.85],
    ...         "ci_lower": [0.85, 0.83, 0.81],
    ...         "ci_upper": [0.92, 0.90, 0.88]
    ...     },
    ...     output_path="figures/generated/data/fig02_forest_outlier_data.json",
    ...     reference_lines={"najjar_2021": {"value": 0.93, "label": "Najjar 2021"}}
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_data = {
        "figure_id": figure_id,
        "figure_title": figure_title,
        "generated_at": datetime.now().isoformat(),
        "data": data,
    }

    if metadata:
        figure_data["metadata"] = metadata

    if reference_lines:
        figure_data["reference_lines"] = reference_lines

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(figure_data, f, indent=2, cls=NumpyEncoder)

    return output_path


def load_figure_data(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load figure data from JSON.

    Parameters
    ----------
    input_path : str or Path
        Path to the JSON file

    Returns
    -------
    dict
        Figure data dictionary

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file contains invalid JSON

    Examples
    --------
    >>> data = load_figure_data("figures/generated/data/fig02_forest_outlier_data.json")
    >>> methods = data["data"]["methods"]
    >>> estimates = data["data"]["estimates"]
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Figure data file not found: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in figure data file {input_path}: {e}")


class FigureDataExporter:
    """
    Helper class for consistent figure data export.

    Manages output directory and provides convenient methods
    for saving different figure types.

    Examples
    --------
    >>> exporter = FigureDataExporter("figures/generated/data/")
    >>> exporter.save_forest_plot(
    ...     "fig02", "Outlier Method Comparison",
    ...     methods=["IQR", "MAD"], estimates=[0.89, 0.87],
    ...     ci_lower=[0.85, 0.83], ci_upper=[0.92, 0.90]
    ... )
    """

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize exporter.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save all figure data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, figure_id: str, suffix: str = "") -> Path:
        """Generate output path for a figure."""
        filename = f"{figure_id}{suffix}_data.json"
        return self.output_dir / filename

    def save_forest_plot(
        self,
        figure_id: str,
        figure_title: str,
        methods: List[str],
        estimates: List[float],
        ci_lower: List[float],
        ci_upper: List[float],
        xlabel: str = "AUROC",
        reference_line: Optional[float] = None,
        reference_label: str = "Reference",
        **metadata,
    ) -> Path:
        """
        Save forest plot data.

        Parameters
        ----------
        figure_id : str
            Figure identifier (e.g., "fig02")
        figure_title : str
            Human-readable title
        methods : list of str
            Method names
        estimates : list of float
            Point estimates
        ci_lower : list of float
            Lower CI bounds
        ci_upper : list of float
            Upper CI bounds
        xlabel : str
            Metric name
        reference_line : float, optional
            Reference line value
        reference_label : str
            Reference line label
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file

        Raises
        ------
        ValueError
            If input arrays have mismatched lengths
        """
        # Validate input lengths
        n = len(methods)
        if len(estimates) != n or len(ci_lower) != n or len(ci_upper) != n:
            raise ValueError(
                f"Input arrays have mismatched lengths: methods={len(methods)}, "
                f"estimates={len(estimates)}, ci_lower={len(ci_lower)}, ci_upper={len(ci_upper)}"
            )

        data = {
            "methods": list(methods),
            "estimates": list(estimates),
            "ci_lower": list(ci_lower),
            "ci_upper": list(ci_upper),
            "xlabel": xlabel,
        }

        reference_lines = None
        if reference_line is not None:
            reference_lines = {
                "reference": {"value": reference_line, "label": reference_label}
            }

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
            reference_lines=reference_lines,
        )

    def save_heatmap(
        self,
        figure_id: str,
        figure_title: str,
        row_labels: List[str],
        col_labels: List[str],
        values: List[List[float]],
        row_axis_label: str = "",
        col_axis_label: str = "",
        value_label: str = "Value",
        **metadata,
    ) -> Path:
        """
        Save heatmap data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        row_labels : list of str
            Row labels
        col_labels : list of str
            Column labels
        values : list of list of float
            2D array of values (rows x cols)
        row_axis_label : str
            Row axis label
        col_axis_label : str
            Column axis label
        value_label : str
            Label for values
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file

        Raises
        ------
        ValueError
            If values dimensions don't match labels
        """
        # Validate dimensions
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        if len(values) != n_rows:
            raise ValueError(
                f"Values has {len(values)} rows but {n_rows} row_labels provided"
            )
        for i, row in enumerate(values):
            if len(row) != n_cols:
                raise ValueError(
                    f"Row {i} has {len(row)} values but {n_cols} col_labels provided"
                )

        data = {
            "row_labels": list(row_labels),
            "col_labels": list(col_labels),
            "values": [list(row) for row in values],
            "row_axis_label": row_axis_label,
            "col_axis_label": col_axis_label,
            "value_label": value_label,
        }

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
        )

    def save_cd_diagram(
        self,
        figure_id: str,
        figure_title: str,
        methods: List[str],
        average_ranks: List[float],
        critical_difference: float,
        cliques: List[List[str]],
        friedman_statistic: float,
        friedman_pvalue: float,
        **metadata,
    ) -> Path:
        """
        Save CD diagram data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        methods : list of str
            Method names
        average_ranks : list of float
            Average ranks for each method
        critical_difference : float
            CD value
        cliques : list of list of str
            Groups of methods not significantly different
        friedman_statistic : float
            Friedman test statistic
        friedman_pvalue : float
            Friedman test p-value
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file
        """
        data = {
            "methods": list(methods),
            "average_ranks": list(average_ranks),
            "critical_difference": critical_difference,
            "cliques": [list(c) for c in cliques],
            "friedman_statistic": friedman_statistic,
            "friedman_pvalue": friedman_pvalue,
        }

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
        )

    def save_specification_curve(
        self,
        figure_id: str,
        figure_title: str,
        estimates: List[float],
        ci_lower: List[float],
        ci_upper: List[float],
        specifications: Dict[str, List[str]],
        specification_order: List[int],
        reference_line: Optional[float] = None,
        **metadata,
    ) -> Path:
        """
        Save specification curve data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        estimates : list of float
            Point estimates for each specification
        ci_lower : list of float
            Lower CI bounds
        ci_upper : list of float
            Upper CI bounds
        specifications : dict
            Dictionary mapping specification names to lists of values
            e.g., {"outlier": ["IQR", "MAD", ...], "imputation": [...]}
        specification_order : list of int
            Indices for sorted order (by estimate)
        reference_line : float, optional
            Reference line value
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file
        """
        data = {
            "estimates": list(estimates),
            "ci_lower": list(ci_lower),
            "ci_upper": list(ci_upper),
            "specifications": {k: list(v) for k, v in specifications.items()},
            "specification_order": list(specification_order),
        }

        reference_lines = None
        if reference_line is not None:
            reference_lines = {
                "reference": {"value": reference_line, "label": "Reference"}
            }

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
            reference_lines=reference_lines,
        )

    def save_variance_decomposition(
        self,
        figure_id: str,
        figure_title: str,
        factors: List[str],
        partial_eta_sq: List[float],
        omega_sq: List[float],
        partial_eta_sq_ci_lower: Optional[List[float]] = None,
        partial_eta_sq_ci_upper: Optional[List[float]] = None,
        **metadata,
    ) -> Path:
        """
        Save variance decomposition bar chart data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        factors : list of str
            Factor names
        partial_eta_sq : list of float
            Partial eta-squared values
        omega_sq : list of float
            Omega-squared values
        partial_eta_sq_ci_lower : list of float, optional
            Lower CI bounds for partial eta-squared
        partial_eta_sq_ci_upper : list of float, optional
            Upper CI bounds for partial eta-squared
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file
        """
        data = {
            "factors": list(factors),
            "partial_eta_sq": list(partial_eta_sq),
            "omega_sq": list(omega_sq),
        }

        if partial_eta_sq_ci_lower is not None:
            data["partial_eta_sq_ci_lower"] = list(partial_eta_sq_ci_lower)
        if partial_eta_sq_ci_upper is not None:
            data["partial_eta_sq_ci_upper"] = list(partial_eta_sq_ci_upper)

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
        )

    def save_calibration_curves(
        self,
        figure_id: str,
        figure_title: str,
        bin_midpoints: List[float],
        observed_proportions: Dict[str, List[float]],
        predicted_proportions: Optional[Dict[str, List[float]]] = None,
        n_samples: Optional[Dict[str, List[int]]] = None,
        calibration_slopes: Optional[Dict[str, float]] = None,
        **metadata,
    ) -> Path:
        """
        Save calibration curve data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        bin_midpoints : list of float
            Probability bin midpoints
        observed_proportions : dict
            Mapping of config names to observed proportions per bin
        predicted_proportions : dict, optional
            Mapping of config names to predicted proportions per bin
        n_samples : dict, optional
            Mapping of config names to sample counts per bin
        calibration_slopes : dict, optional
            Mapping of config names to calibration slopes
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file
        """
        data = {
            "bin_midpoints": list(bin_midpoints),
            "observed_proportions": {
                k: list(v) for k, v in observed_proportions.items()
            },
        }

        if predicted_proportions is not None:
            data["predicted_proportions"] = {
                k: list(v) for k, v in predicted_proportions.items()
            }

        if n_samples is not None:
            data["n_samples"] = {k: list(v) for k, v in n_samples.items()}

        if calibration_slopes is not None:
            data["calibration_slopes"] = calibration_slopes

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
        )

    def save_dca_curves(
        self,
        figure_id: str,
        figure_title: str,
        thresholds: List[float],
        net_benefits: Dict[str, List[float]],
        treat_all_benefit: List[float],
        treat_none_benefit: List[float],
        **metadata,
    ) -> Path:
        """
        Save decision curve analysis data.

        Parameters
        ----------
        figure_id : str
            Figure identifier
        figure_title : str
            Human-readable title
        thresholds : list of float
            Decision thresholds
        net_benefits : dict
            Mapping of config names to net benefit values at each threshold
        treat_all_benefit : list of float
            Net benefit for "treat all" strategy
        treat_none_benefit : list of float
            Net benefit for "treat none" strategy (typically 0)
        **metadata
            Additional metadata

        Returns
        -------
        Path
            Path to saved file
        """
        data = {
            "thresholds": list(thresholds),
            "net_benefits": {k: list(v) for k, v in net_benefits.items()},
            "treat_all_benefit": list(treat_all_benefit),
            "treat_none_benefit": list(treat_none_benefit),
        }

        return save_figure_data(
            figure_id=figure_id,
            figure_title=figure_title,
            data=data,
            output_path=self._get_output_path(figure_id),
            metadata=metadata or None,
        )
