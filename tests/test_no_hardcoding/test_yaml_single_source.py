"""
TDD Test: Verify YAML files are the ONLY source of truth.
"""

from pathlib import Path

import pytest
import yaml


def get_project_root() -> Path:
    """Find project root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def test_combos_yaml_has_all_required_fields() -> None:
    """plot_hyperparam_combos.yaml must have all required fields for each combo."""
    project_root = get_project_root()
    combos_path = project_root / "configs/VISUALIZATION/plot_hyperparam_combos.yaml"

    if not combos_path.exists():
        pytest.fail(
            "plot_hyperparam_combos.yaml not found at configs/VISUALIZATION/plot_hyperparam_combos.yaml"
        )

    config = yaml.safe_load(combos_path.read_text())

    required_fields = [
        "id",
        "name",
        "short_name",
        "outlier_method",
        "imputation_method",
        "classifier",
    ]

    # Check standard combos
    if "standard_combos" in config:
        for combo in config["standard_combos"]:
            for field in required_fields:
                assert field in combo, (
                    f"Standard combo {combo.get('id', '?')} missing {field}"
                )

    # Check extended combos
    if "extended_combos" in config:
        for combo in config["extended_combos"]:
            for field in required_fields:
                assert field in combo, (
                    f"Extended combo {combo.get('id', '?')} missing {field}"
                )


def test_display_names_yaml_exists_and_complete() -> None:
    """display_names.yaml must exist and define all methods."""
    project_root = get_project_root()
    display_path = project_root / "configs/mlflow_registry/display_names.yaml"

    if not display_path.exists():
        pytest.fail(
            "display_names.yaml not found!\n"
            "Create at: configs/mlflow_registry/display_names.yaml"
        )

    config = yaml.safe_load(display_path.read_text())

    # Expected minimum counts from registry
    assert "outlier_methods" in config, (
        "display_names.yaml missing outlier_methods section"
    )
    assert "imputation_methods" in config, (
        "display_names.yaml missing imputation_methods section"
    )
    assert "classifiers" in config, "display_names.yaml missing classifiers section"

    # Count methods (new format uses nested objects)
    outlier_count = len(config["outlier_methods"])
    imputation_count = len(config["imputation_methods"])
    classifier_count = len(config["classifiers"])

    assert outlier_count >= 11, f"Only {outlier_count} outlier methods defined, need 11"
    assert imputation_count >= 7, (
        f"Only {imputation_count} imputation methods defined, need 7"
    )
    assert classifier_count >= 5, f"Only {classifier_count} classifiers defined, need 5"


def test_no_duplicate_combo_definitions() -> None:
    """There should be only ONE combos file, not two."""
    project_root = get_project_root()
    viz_config = project_root / "configs/VISUALIZATION"

    if not viz_config.exists():
        pytest.skip("configs/VISUALIZATION not found")

    combo_files = list(viz_config.glob("*combo*.yaml"))

    # After consolidation, should only be plot_hyperparam_combos.yaml (not plot_hyperparam_plot_hyperparam_combos.yaml too)
    assert len(combo_files) == 1, (
        f"Found {len(combo_files)} combo files: {[f.name for f in combo_files]}.\n"
        f"Should only have plot_hyperparam_combos.yaml! Delete duplicates."
    )


def test_category_mapping_yaml_exists() -> None:
    """category_mapping.yaml must exist for method categorization."""
    project_root = get_project_root()
    category_path = project_root / "configs/mlflow_registry/category_mapping.yaml"

    if not category_path.exists():
        pytest.fail(
            "category_mapping.yaml not found!\n"
            "Create at: configs/mlflow_registry/category_mapping.yaml\n"
            "This replaces all case_when() categorization in R files."
        )

    config = yaml.safe_load(category_path.read_text())

    # Must have outlier and imputation categories
    assert "outlier_method_categories" in config, "Missing outlier_method_categories"
    assert "imputation_method_categories" in config, (
        "Missing imputation_method_categories"
    )


def test_method_abbreviations_yaml_exists() -> None:
    """method_abbreviations.yaml must exist for CD diagram labels."""
    project_root = get_project_root()
    abbrev_path = project_root / "configs/mlflow_registry/method_abbreviations.yaml"

    if not abbrev_path.exists():
        pytest.fail(
            "method_abbreviations.yaml not found!\n"
            "Create at: configs/mlflow_registry/method_abbreviations.yaml\n"
            "This replaces 73 hardcoded abbreviations in cd_diagram.R."
        )

    config = yaml.safe_load(abbrev_path.read_text())

    # Must have abbreviation sections
    assert "outlier_method_abbreviations" in config, (
        "Missing outlier_method_abbreviations"
    )
    assert "imputation_method_abbreviations" in config, (
        "Missing imputation_method_abbreviations"
    )
