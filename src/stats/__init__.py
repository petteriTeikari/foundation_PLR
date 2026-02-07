"""
Statistics module for foundation-PLR pipeline analysis.

This module provides:
- Effect size computations with confidence intervals
- Multiple comparison corrections (FDR, Bonferroni)
- Factorial ANOVA for pipeline comparison
- Calibration metrics (STRATOS-compliant)
- Clinical utility metrics (Decision Curve Analysis)
- Bootstrap inference (BCa method)

Cross-references:
- planning/statistics-implementation.md
- appendix-literature-review/section-08-biostatistics.tex
"""

# Lazy imports to avoid loading mlflow-dependent modules during testing
# Import these submodules directly when needed:
# from src.stats.classifier_metrics import get_classifier_metrics
# from src.stats.calibration_metrics import brier_score, expected_calibration_error

# Core types and exceptions are always available
from ._exceptions import (
    CheckpointError,
    ConvergenceError,
    DegenerateCaseError,
    InsufficientDataError,
    SingleClassError,
    StatsError,
    ValidationError,
)
from ._types import (
    ANOVAResult,
    BootstrapResult,
    CalibrationResult,
    EffectSizeResult,
    FDRResult,
    StatsResult,
)

# Bootstrap inference
from .bootstrap import (
    bca_bootstrap_ci,
    bootstrap_se,
    percentile_bootstrap_ci,
    stratified_bootstrap_sample,
)

# Extended calibration (STRATOS-compliant)
from .calibration_extended import (
    brier_decomposition,
    calibration_slope_intercept,
)

# Clinical utility (Decision Curve Analysis)
from .clinical_utility import (
    decision_curve_analysis,
    net_benefit,
    optimal_threshold_cost_sensitive,
    standardized_net_benefit,
)

# Decision Uncertainty (Barrenada et al. 2025)
from .decision_uncertainty import (
    decision_uncertainty,
    decision_uncertainty_per_subject,
    decision_uncertainty_summary,
)

# Effect sizes (no external dependencies)
from .effect_sizes import (
    cohens_d,
    cohens_f,
    generalized_eta_squared,
    hedges_g,
    omega_squared,
    partial_eta_squared,
)

# FDR correction (uses statsmodels)
from .fdr_correction import (
    apply_fdr_correction,
    benjamini_hochberg,
    bonferroni,
    holm,
)

# Imputation metrics (reporting beyond preprocessing)
from .imputation_metrics import (
    ImputationMetricsResult,
    compute_coverage_probability,
    compute_fraction_missing_information,
    compute_reconstruction_metrics,
    imputation_report,
)

# pminternal analysis (Python-native Riley 2023 instability metrics)
from .pminternal_analysis import (
    BootstrapPredictionData,
    InstabilityMetrics,
    compute_per_patient_uncertainty,
    compute_prediction_instability_metrics,
    create_prediction_instability_plot_data,
    export_pminternal_data,
    load_bootstrap_predictions_from_mlflow,
)

# pminternal wrapper (R package for model instability - Riley 2023)
from .pminternal_wrapper import (
    InstabilityResult,
    ValidationResult,
    calibration_metrics,
    calibration_metrics_safe,
    instability_analysis,
    validate_model,
)

# Scaled Brier Score (IPA)
from .scaled_brier import (
    interpret_ipa,
    scaled_brier_score,
    scaled_brier_score_with_ci,
)

# Uncertainty propagation (Monte Carlo analysis)
from .uncertainty_propagation import (
    DecisionStabilityResult,
    SensitivityResult,
    UncertaintyResult,
    clinical_decision_stability,
    compute_required_accuracy,
    monte_carlo_classifier_uncertainty,
    sensitivity_analysis_delta,
)

# Variance Decomposition (Factorial ANOVA)
from .variance_decomposition import (
    AssumptionTestResult,
    FactorialANOVAResult,
    compute_effect_size_ci,
    factorial_anova,
    test_anova_assumptions,
)

__all__ = [
    # Types
    "StatsResult",
    "EffectSizeResult",
    "ANOVAResult",
    "CalibrationResult",
    "FDRResult",
    "BootstrapResult",
    # Exceptions
    "StatsError",
    "InsufficientDataError",
    "SingleClassError",
    "ConvergenceError",
    "CheckpointError",
    "ValidationError",
    "DegenerateCaseError",
    # Effect sizes
    "cohens_d",
    "hedges_g",
    "partial_eta_squared",
    "generalized_eta_squared",
    "omega_squared",
    "cohens_f",
    # FDR correction
    "benjamini_hochberg",
    "bonferroni",
    "holm",
    "apply_fdr_correction",
    # Bootstrap
    "bca_bootstrap_ci",
    "percentile_bootstrap_ci",
    "stratified_bootstrap_sample",
    "bootstrap_se",
    # Extended calibration
    "calibration_slope_intercept",
    "brier_decomposition",
    # Clinical utility
    "net_benefit",
    "decision_curve_analysis",
    "standardized_net_benefit",
    "optimal_threshold_cost_sensitive",
    # Uncertainty propagation
    "UncertaintyResult",
    "SensitivityResult",
    "DecisionStabilityResult",
    "monte_carlo_classifier_uncertainty",
    "clinical_decision_stability",
    "sensitivity_analysis_delta",
    "compute_required_accuracy",
    # Imputation metrics
    "ImputationMetricsResult",
    "compute_reconstruction_metrics",
    "compute_coverage_probability",
    "compute_fraction_missing_information",
    "imputation_report",
    # Decision Uncertainty
    "decision_uncertainty",
    "decision_uncertainty_per_subject",
    "decision_uncertainty_summary",
    # Scaled Brier Score
    "scaled_brier_score",
    "scaled_brier_score_with_ci",
    "interpret_ipa",
    # Variance Decomposition
    "FactorialANOVAResult",
    "AssumptionTestResult",
    "factorial_anova",
    "test_anova_assumptions",
    "compute_effect_size_ci",
    # pminternal wrapper (R package)
    "ValidationResult",
    "InstabilityResult",
    "validate_model",
    "instability_analysis",
    "calibration_metrics",
    "calibration_metrics_safe",
    # pminternal analysis (Python-native)
    "BootstrapPredictionData",
    "InstabilityMetrics",
    "load_bootstrap_predictions_from_mlflow",
    "compute_prediction_instability_metrics",
    "compute_per_patient_uncertainty",
    "export_pminternal_data",
    "create_prediction_instability_plot_data",
]


def __getattr__(name):
    """Lazy loading for mlflow-dependent modules."""
    if name == "get_classifier_metrics":
        from .classifier_metrics import get_classifier_metrics

        return get_classifier_metrics
    elif name == "brier_score":
        from .calibration_metrics import brier_score

        return brier_score
    elif name == "expected_calibration_error":
        from .calibration_metrics import expected_calibration_error

        return expected_calibration_error
    elif name == "isotonic_calibration":
        from .classifier_calibration import isotonic_calibration

        return isotonic_calibration
    raise AttributeError(f"module 'stats' has no attribute '{name}'")
