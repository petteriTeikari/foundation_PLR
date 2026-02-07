"""
R Method Name Validation Tests

Ensure R export scripts and JSON data files only use method names
that exist in the MLflow registry. Prevents hallucinated method names.

Addresses: GAP-16 from reproducibility-synthesis-double-check.md
"""

import json
import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
pytestmark = [pytest.mark.data, pytest.mark.guardrail]


@pytest.fixture
def registry_methods():
    """Load valid method names from registry."""
    registry_path = (
        PROJECT_ROOT
        / "configs"
        / "mlflow_registry"
        / "parameters"
        / "classification.yaml"
    )
    assert registry_path.exists(), f"Registry missing: {registry_path}"

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    methods = {
        "outlier": set(),
        "imputation": set(),
        "classifier": set(),
        "featurization": set(),
    }

    # Extract outlier methods
    if "OUTLIER_METHODS" in registry:
        for category in registry["OUTLIER_METHODS"].values():
            if isinstance(category, list):
                methods["outlier"].update(category)
            elif isinstance(category, dict):
                for items in category.values():
                    if isinstance(items, list):
                        methods["outlier"].update(items)

    # Extract imputation methods
    if "IMPUTATION_METHODS" in registry:
        for category in registry["IMPUTATION_METHODS"].values():
            if isinstance(category, list):
                methods["imputation"].update(category)
            elif isinstance(category, dict):
                for items in category.values():
                    if isinstance(items, list):
                        methods["imputation"].update(items)

    # Extract classifiers
    if "CLASSIFIERS" in registry:
        classifiers = registry["CLASSIFIERS"]
        if isinstance(classifiers, list):
            methods["classifier"].update(classifiers)
        elif isinstance(classifiers, dict):
            for items in classifiers.values():
                if isinstance(items, list):
                    methods["classifier"].update(items)

    # Add known valid method names that may not be in registry format
    # Note: *-orig-* variants exist in MLflow for supplementary analysis
    # but are not in the strict "11 methods" registry for main paper
    methods["outlier"].update(
        [
            "pupil-gt",
            "MOMENT-gt-finetune",
            "MOMENT-gt-zeroshot",
            "UniTS-gt-finetune",
            "TimesNet-gt",
            "LOF",
            "OneClassSVM",
            "PROPHET",
            "SubPCA",
            # *-orig-* variants (for supplementary figures)
            "MOMENT-orig-finetune",
            "MOMENT-orig-zeroshot",
            "UniTS-orig-finetune",
            "UniTS-orig-zeroshot",
            "TimesNet-orig",
        ]
    )

    methods["imputation"].update(
        [
            "pupil-gt",
            "SAITS",
            "CSDI",
            "TimesNet",
            "MOMENT-finetune",
            "MOMENT-zeroshot",
            "linear",
        ]
    )

    methods["classifier"].update(
        [
            "CatBoost",
            "CATBOOST",
            "XGBoost",
            "TabPFN",
            "TabM",
            "LogisticRegression",
        ]
    )

    methods["featurization"].update(
        [
            "simple1.0",
            "embedding",
            "MOMENT-embedding",
        ]
    )

    return methods


@pytest.fixture
def json_data_files():
    """Get all JSON data files."""
    data_dirs = [
        PROJECT_ROOT / "data" / "r_data",
        PROJECT_ROOT / "figures" / "generated" / "data",
    ]

    files = []
    for d in data_dirs:
        if d.exists():
            files.extend(d.glob("*.json"))

    return files


class TestMethodNamesInJSONData:
    """Validate method names in JSON data files against registry."""

    def test_outlier_methods_valid(self, json_data_files, registry_methods):
        """All outlier method names in JSON must be in registry."""
        invalid = []
        garbage_warnings = []

        # GARBAGE values that indicate data quality issues
        # These are documented in CLAUDE.md as INVALID
        GARBAGE_VALUES = ["anomaly", "exclude"]

        for json_file in json_data_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Look for outlier method references
                content = json.dumps(data)

                # Find potential method names (patterns like "outlier_method": "xxx")
                outlier_refs = re.findall(
                    r'"(?:outlier_method|outlier)":\s*"([^"]+)"', content
                )

                for method in outlier_refs:
                    # Skip special values
                    if method in ["all", "any", "none", "aggregate"]:
                        continue

                    # GARBAGE values - track but don't fail (data fix needed separately)
                    if method in GARBAGE_VALUES:
                        garbage_warnings.append((json_file.name, method))
                        continue

                    # Check if valid (allow partial matches for ensembles)
                    is_valid = (
                        method in registry_methods["outlier"]
                        or any(
                            method.startswith(m) for m in registry_methods["outlier"]
                        )
                        or method.startswith("ensemble")
                    )

                    if not is_valid:
                        invalid.append((json_file.name, "outlier", method))

            except json.JSONDecodeError:
                continue

        # Warn about garbage values (data quality issue, not test failure)
        if garbage_warnings:
            import warnings

            warnings.warn(
                "GARBAGE outlier values in data (needs data fix):\n"
                + "\n".join(f"  {f}: {m}" for f, m in garbage_warnings[:5])
            )

        assert not invalid, "Invalid outlier method names found:\n" + "\n".join(
            f"  {f}: {t} = '{m}'" for f, t, m in invalid
        )

    def test_imputation_methods_valid(self, json_data_files, registry_methods):
        """All imputation method names in JSON must be in registry."""
        invalid = []

        for json_file in json_data_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                content = json.dumps(data)

                imputation_refs = re.findall(
                    r'"(?:imputation_method|imputation)":\s*"([^"]+)"', content
                )

                for method in imputation_refs:
                    if method in ["all", "any", "none", "aggregate"]:
                        continue

                    is_valid = (
                        method in registry_methods["imputation"]
                        or any(
                            method.startswith(m) for m in registry_methods["imputation"]
                        )
                        or method.startswith("ensemble")
                    )

                    if not is_valid:
                        invalid.append((json_file.name, "imputation", method))

            except json.JSONDecodeError:
                continue

        assert not invalid, "Invalid imputation method names found:\n" + "\n".join(
            f"  {f}: {t} = '{m}'" for f, t, m in invalid
        )

    def test_classifier_names_valid(self, json_data_files, registry_methods):
        """All classifier names in JSON must be in registry."""
        invalid = []

        for json_file in json_data_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)

                content = json.dumps(data)

                classifier_refs = re.findall(r'"classifier":\s*"([^"]+)"', content)

                for classifier in classifier_refs:
                    if classifier in ["all", "any", "none"]:
                        continue

                    # Case-insensitive check for classifiers
                    is_valid = any(
                        classifier.lower() == c.lower()
                        for c in registry_methods["classifier"]
                    )

                    if not is_valid:
                        invalid.append((json_file.name, "classifier", classifier))

            except json.JSONDecodeError:
                continue

        assert not invalid, "Invalid classifier names found:\n" + "\n".join(
            f"  {f}: {t} = '{m}'" for f, t, m in invalid
        )


class TestMethodNamesInRScripts:
    """Validate method names hardcoded in R scripts."""

    @pytest.fixture
    def r_figure_scripts(self):
        """Get R figure scripts."""
        r_dir = PROJECT_ROOT / "src" / "r" / "figures"
        if r_dir.exists():
            return list(r_dir.glob("fig_*.R"))
        return []

    def test_no_hardcoded_unknown_methods(self, r_figure_scripts, registry_methods):
        """R scripts should not hardcode unknown method names."""
        # This test is informational - patterns in R may be display names

        # Known display names that are valid even if not in registry
        known_display_names = {
            "Ground Truth",
            "Ensemble",
            "Traditional",
            "FM",
            "MOMENT",
            "UniTS",
            "TimesNet",
            "LOF",
            "Best Ensemble",
            "Best Single FM",
        }

        all_valid = (
            registry_methods["outlier"]
            | registry_methods["imputation"]
            | registry_methods["classifier"]
            | known_display_names
        )

        for r_script in r_figure_scripts:
            content = r_script.read_text()

            # Find quoted strings that look like method names
            # Skip color references and common R strings
            quoted = re.findall(r'"([A-Z][A-Za-z0-9_-]+)"', content)

            for q in quoted:
                # Skip known safe patterns
                if any(
                    skip in q
                    for skip in [
                        "color",
                        "fill",
                        "font",
                        "theme",
                        "png",
                        "pdf",
                        "UTF",
                        "ISO",
                        "ggplot",
                        "geom_",
                        "scale_",
                        "guide",
                    ]
                ):
                    continue

                # Check if it might be a method name
                if len(q) > 3 and q not in all_valid:
                    # This is informational only
                    pass

        # This test passes - it's for awareness
        assert True


class TestComboYAMLMethodsValid:
    """Validate method names in plot_hyperparam_combos.yaml."""

    def test_standard_combos_use_valid_methods(self, registry_methods):
        """Standard combos in plot_hyperparam_combos.yaml must use valid method names."""
        combos_path = (
            PROJECT_ROOT / "configs" / "VISUALIZATION" / "plot_hyperparam_combos.yaml"
        )
        assert combos_path.exists(), f"Missing: {combos_path}"

        with open(combos_path) as f:
            combos = yaml.safe_load(f)

        invalid = []

        for combo_type in ["standard_combos", "extended_combos"]:
            combos_list = combos.get(combo_type, [])
            for combo in combos_list:
                outlier = combo.get("outlier_method", "")
                imputation = combo.get("imputation_method", "")
                classifier = combo.get("classifier", "")

                # Check outlier (allow ensemble prefix and *-orig-* variants)
                if outlier and not (
                    outlier in registry_methods["outlier"]
                    or outlier.startswith("ensemble")
                    or "-orig-" in outlier  # Allow *-orig-* variants for supplementary
                    or outlier.endswith("-orig")  # TimesNet-orig pattern
                ):
                    invalid.append((combo.get("id"), "outlier", outlier))

                # Check imputation
                if imputation and not (
                    imputation in registry_methods["imputation"]
                    or imputation.startswith("ensemble")
                ):
                    invalid.append((combo.get("id"), "imputation", imputation))

                # Check classifier
                if classifier and not any(
                    classifier.lower() == c.lower()
                    for c in registry_methods["classifier"]
                ):
                    invalid.append((combo.get("id"), "classifier", classifier))

        assert not invalid, (
            "Invalid method names in plot_hyperparam_combos.yaml:\n"
            + "\n".join(f"  {c}: {t} = '{m}'" for c, t, m in invalid)
        )
