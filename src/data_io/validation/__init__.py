# src/data_io/validation/__init__.py
"""Data validation modules for Foundation PLR."""

from .normalization_validator import (
    NormalizationValidator,
    ScalingAnomaly,
    get_known_anomalies,
    validate_subject_scaling,
)

from .data_validators import (
    DataValidationError,
    validate_light_stimuli,
    validate_signal_range,
    validate_time_monotonic,
    validate_features,
    validate_database_schema,
    validate_subject_count,
    validate_pipeline_inputs,
)

__all__ = [
    # Normalization validators
    "NormalizationValidator",
    "ScalingAnomaly",
    "validate_subject_scaling",
    "get_known_anomalies",
    # Data validators
    "DataValidationError",
    "validate_light_stimuli",
    "validate_signal_range",
    "validate_time_monotonic",
    "validate_features",
    "validate_database_schema",
    "validate_subject_count",
    "validate_pipeline_inputs",
]
