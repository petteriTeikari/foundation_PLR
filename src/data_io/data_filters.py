"""
Data Filters Module - Single Source of Truth for Data Extraction
================================================================

This module loads filter configuration from YAML and provides
query helpers for all export scripts.

CRITICAL: ALL export scripts MUST use this module to ensure
consistent featurization filtering.

Created: 2026-01-28
Reason: CRITICAL-FAILURE-002 and FAILURE-005 - mixed featurization
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


# Find project root
def _find_project_root() -> Path:
    markers = ["pyproject.toml", "CLAUDE.md", ".git"]
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = _find_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "VISUALIZATION" / "data_filters.yaml"


def load_data_filters() -> Dict[str, Any]:
    """Load data filters from YAML config."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Data filters config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_default_featurization() -> str:
    """Get default featurization filter (handcrafted features)."""
    config = load_data_filters()
    return config["defaults"]["featurization"]


def get_default_classifier() -> str:
    """Get default classifier filter."""
    config = load_data_filters()
    return config["defaults"]["classifier"]


def get_expected_ground_truth_auroc() -> Tuple[float, float]:
    """Get expected Ground Truth AUROC and tolerance."""
    config = load_data_filters()
    gt = config["expected_values"]["ground_truth"]
    return gt["auroc"], gt["auroc_tolerance"]


def get_expected_n_predictions() -> int:
    """Get expected number of predictions per config."""
    config = load_data_filters()
    return config["expected_values"]["ground_truth"]["n_predictions"]


def get_predictions_query(
    outlier_method: str,
    imputation_method: str,
    classifier: Optional[str] = None,
    featurization: Optional[str] = None,
) -> str:
    """
    Build SQL query for predictions with proper filters.

    This is the ONLY way export scripts should query predictions.
    """
    config = load_data_filters()

    if classifier is None:
        classifier = config["defaults"]["classifier"]
    if featurization is None:
        featurization = config["defaults"]["featurization"]

    return f"""
        SELECT y_true, y_prob, subject_id
        FROM predictions
        WHERE outlier_method = '{outlier_method}'
          AND imputation_method = '{imputation_method}'
          AND classifier = '{classifier}'
          AND featurization = '{featurization}'
    """


def get_metrics_query(
    classifier: Optional[str] = None,
    featurization: Optional[str] = None,
) -> str:
    """
    Build SQL query for essential_metrics with proper filters.
    """
    config = load_data_filters()

    if classifier is None:
        classifier = config["defaults"]["classifier"]
    if featurization is None:
        featurization = config["defaults"]["featurization"]

    return f"""
        SELECT *
        FROM essential_metrics
        WHERE classifier = '{classifier}'
          AND featurization = '{featurization}'
    """


def validate_auroc(auroc: float, config_name: str = "ground_truth") -> bool:
    """
    Validate AUROC is within expected range.

    Raises AssertionError if out of range.
    """
    config = load_data_filters()
    expected = config["expected_values"].get(config_name)

    if expected is None:
        return True  # No validation for unknown configs

    expected_auroc = expected["auroc"]
    tolerance = expected["auroc_tolerance"]

    if not (expected_auroc - tolerance <= auroc <= expected_auroc + tolerance):
        raise AssertionError(
            f"{config_name} AUROC {auroc:.4f} not in expected range "
            f"[{expected_auroc - tolerance:.3f}, {expected_auroc + tolerance:.3f}]"
        )

    return True


def validate_n_predictions(n: int) -> bool:
    """
    Validate number of predictions is correct.

    Raises AssertionError if wrong.
    """
    expected = get_expected_n_predictions()

    if n != expected:
        raise AssertionError(
            f"Got {n} predictions, expected {expected} (152 control + 56 glaucoma)"
        )

    return True


# Convenience: print config on import for debugging
if __name__ == "__main__":
    config = load_data_filters()
    print("=== Data Filters Config ===")
    print(f"Default featurization: {config['defaults']['featurization']}")
    print(f"Default classifier: {config['defaults']['classifier']}")
    print(f"Expected GT AUROC: {config['expected_values']['ground_truth']['auroc']}")
    print(
        f"Expected N predictions: {config['expected_values']['ground_truth']['n_predictions']}"
    )
