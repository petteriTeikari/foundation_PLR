"""
Visualization module for foundation-PLR pipeline analysis.

This module provides publication-quality visualizations for:
- Critical Difference diagrams (Demšar 2006)
- Forest plots with confidence intervals
- Heatmaps for sensitivity analysis
- Specification curves (Simonsohn 2020)

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md
"""

# Critical Difference diagrams
# Calibration plots (STRATOS-compliant)
from .calibration_plot import (
    compute_calibration_ci,
    compute_loess_calibration,
    generate_calibration_figure,
    plot_calibration_curve,
    plot_calibration_multi_model,
)
from .cd_diagram import (
    compute_critical_difference,
    draw_cd_diagram,
    friedman_nemenyi_test,
    identify_cliques,
    prepare_cd_data,
)

# Data loading abstraction
from .data_loader import (
    DataLoader,
    DataQuery,
    DuckDBLoader,
    MockDataLoader,
    create_loader,
)

# Decision Curve Analysis (DCA)
from .dca_plot import (
    compute_dca_curves,
    compute_net_benefit,
    compute_treat_all_nb,
    compute_treat_none_nb,
    generate_dca_figure,
    load_dca_curves_from_db,
    plot_dca,
    plot_dca_from_db,
    plot_dca_multi_model,
)

# Figure data export (JSON)
from .figure_data import (
    FigureDataExporter,
    load_figure_data,
    save_figure_data,
)

# Forest plots
from .forest_plot import (
    draw_forest_plot,
    forest_plot_from_dataframe,
    grouped_forest_plot,
)

# Sensitivity heatmaps
from .heatmap_sensitivity import (
    annotated_heatmap,
    draw_sensitivity_heatmap,
    heatmap_from_pivot,
    sensitivity_heatmap_grid,
)

# Metric registry
from .metric_registry import (
    MetricDefinition,
    MetricRegistry,
)

# Metric vs cohort plots
from .metric_vs_cohort import (
    generate_metric_vs_cohort_figure,
    load_cohort_curve_from_db,
    plot_metric_vs_cohort,
    plot_multi_metric_vs_cohort,
)

# Probability distribution plots
from .prob_distribution import (
    generate_probability_distribution_figure,
    load_distribution_stats_from_db,
    plot_probability_distributions,
)

# Retained metric (uncertainty visualization)
from .retained_metric import (
    generate_multi_combo_retention_figure,
    generate_retention_figures,
    load_all_retention_curves_from_db,
    load_retention_curve_from_db,
    plot_multi_metric_retention,
    plot_multi_model_retention,
    plot_retention_curve,
)

# Specification curves
from .specification_curve import (
    draw_specification_curve,
    simple_specification_curve,
    specification_curve_from_dataframe,
)

# Uncertainty scatter plots
from .uncertainty_scatter import (
    compute_uncertainty_correlation,
    generate_uncertainty_scatter_figure,
    plot_uncertainty_by_correctness,
    plot_uncertainty_scatter,
)

__all__ = [
    # CD diagrams
    "friedman_nemenyi_test",
    "compute_critical_difference",
    "draw_cd_diagram",
    "identify_cliques",
    "prepare_cd_data",
    # Forest plots
    "draw_forest_plot",
    "forest_plot_from_dataframe",
    "grouped_forest_plot",
    # Heatmaps
    "draw_sensitivity_heatmap",
    "heatmap_from_pivot",
    "annotated_heatmap",
    "sensitivity_heatmap_grid",
    # Specification curves
    "draw_specification_curve",
    "specification_curve_from_dataframe",
    "simple_specification_curve",
    # Figure data export
    "save_figure_data",
    "load_figure_data",
    "FigureDataExporter",
    # Data loading
    "DataLoader",
    "DuckDBLoader",
    "MockDataLoader",
    "DataQuery",
    "create_loader",
    # Metric registry
    "MetricDefinition",
    "MetricRegistry",
    # Retained metric (uncertainty visualization)
    "load_retention_curve_from_db",
    "load_all_retention_curves_from_db",
    "plot_retention_curve",
    "plot_multi_metric_retention",
    "plot_multi_model_retention",
    "generate_retention_figures",
    "generate_multi_combo_retention_figure",
    # Calibration plots (STRATOS-compliant)
    "compute_loess_calibration",
    "compute_calibration_ci",
    "plot_calibration_curve",
    "plot_calibration_multi_model",
    "generate_calibration_figure",
    # Decision Curve Analysis (DCA)
    "compute_net_benefit",
    "compute_treat_all_nb",
    "compute_treat_none_nb",
    "compute_dca_curves",
    "load_dca_curves_from_db",
    "plot_dca",
    "plot_dca_from_db",
    "plot_dca_multi_model",
    "generate_dca_figure",
    # Probability distribution plots
    "load_distribution_stats_from_db",
    "plot_probability_distributions",
    "generate_probability_distribution_figure",
    # Uncertainty scatter plots
    "compute_uncertainty_correlation",
    "plot_uncertainty_scatter",
    "plot_uncertainty_by_correctness",
    "generate_uncertainty_scatter_figure",
    # Metric vs cohort plots
    "load_cohort_curve_from_db",
    "plot_metric_vs_cohort",
    "plot_multi_metric_vs_cohort",
    "generate_metric_vs_cohort_figure",
]
