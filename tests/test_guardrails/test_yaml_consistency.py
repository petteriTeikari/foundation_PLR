"""
Guardrail Tests: YAML Configuration Consistency

Ensure YAML configuration files are internally consistent and follow rules.
"""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
VIZ_CONFIG_DIR = PROJECT_ROOT / "configs" / "VISUALIZATION"


class TestYAMLConsistency:
    """Test YAML configuration consistency."""

    def test_color_definitions_exist(self):
        """Check that color_definitions section exists in combos YAML."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())

        assert "color_definitions" in config, (
            "Missing color_definitions section in plot_hyperparam_combos.yaml"
        )

        color_defs = config["color_definitions"]
        assert len(color_defs) > 0, "color_definitions is empty"

    def test_all_color_refs_defined(self):
        """Check that all color_ref values reference existing definitions."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())
        color_defs = config.get("color_definitions", {})

        violations = []

        # Check standard_combos
        for combo in config.get("standard_combos", []):
            color_var = combo.get("color_var")
            if color_var and color_var not in color_defs:
                violations.append(
                    {
                        "section": "standard_combos",
                        "combo_id": combo.get("id"),
                        "color_var": color_var,
                    }
                )

        # Check extended_combos
        for combo in config.get("extended_combos", []):
            color_var = combo.get("color_var")
            if color_var and color_var not in color_defs:
                violations.append(
                    {
                        "section": "extended_combos",
                        "combo_id": combo.get("id"),
                        "color_var": color_var,
                    }
                )

        if violations:
            msg = "GUARDRAIL VIOLATION: color_var references undefined colors!\n\n"
            for v in violations:
                msg += f"  {v['section']}.{v['combo_id']}: {v['color_var']} not in color_definitions\n"
            pytest.fail(msg)

    def test_all_combos_have_required_fields(self):
        """Check that all combos have required fields."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())
        required_fields = ["id", "name", "outlier_method", "imputation_method"]

        violations = []

        for section in ["standard_combos", "extended_combos"]:
            for combo in config.get(section, []):
                for field in required_fields:
                    if field not in combo:
                        violations.append(
                            {
                                "section": section,
                                "combo_id": combo.get("id", "MISSING_ID"),
                                "missing_field": field,
                            }
                        )

        if violations:
            msg = "GUARDRAIL VIOLATION: Combos missing required fields!\n\n"
            for v in violations:
                msg += f"  {v['section']}.{v['combo_id']}: missing '{v['missing_field']}'\n"
            pytest.fail(msg)

    def test_no_duplicate_combo_ids(self):
        """Check that combo IDs are unique across all sections."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())

        all_ids = []
        for section in ["standard_combos", "extended_combos"]:
            for combo in config.get(section, []):
                combo_id = combo.get("id")
                if combo_id:
                    all_ids.append((section, combo_id))

        seen = {}
        duplicates = []
        for section, combo_id in all_ids:
            if combo_id in seen:
                duplicates.append(
                    {
                        "id": combo_id,
                        "first": seen[combo_id],
                        "second": section,
                    }
                )
            else:
                seen[combo_id] = section

        if duplicates:
            msg = "GUARDRAIL VIOLATION: Duplicate combo IDs!\n\n"
            for d in duplicates:
                msg += f"  '{d['id']}' appears in both {d['first']} and {d['second']}\n"
            pytest.fail(msg)

    def test_figure_categories_reference_valid_figures(self):
        """Check that figure_categories only list valid figures.

        NOTE: This test is currently a warning only, not a failure.
        The figure_categories section can reference:
        1. Composed figures defined in the 'figures' section (full layout definitions)
        2. Standalone figure scripts (R or Python) that don't need layout definitions

        A stricter version would require all figures to be defined, but that's
        overly prescriptive for standalone scripts.
        """
        layouts_path = VIZ_CONFIG_DIR / "figure_layouts.yaml"
        if not layouts_path.exists():
            pytest.skip("figure_layouts.yaml not found")

        config = yaml.safe_load(layouts_path.read_text())

        # Get all defined figures (composed multi-panel figures)
        defined_figures = set(config.get("figures", {}).keys())

        # Get all categorized figures
        categories = config.get("figure_categories", {})
        undefined = []

        for cat_name, cat_info in categories.items():
            for fig_id in cat_info.get("figures", []):
                if fig_id not in defined_figures:
                    undefined.append(
                        {
                            "category": cat_name,
                            "figure": fig_id,
                        }
                    )

        # Just log a warning, don't fail
        # Many figures are standalone scripts that don't need layout definitions
        if undefined:
            import warnings

            msg = (
                f"Note: {len(undefined)} figures in figure_categories are not "
                f"defined in figures section. This is OK for standalone scripts.\n"
                f"Defined composed figures: {sorted(defined_figures)}"
            )
            warnings.warn(msg)

    def test_no_misleading_baseline_names(self):
        """Check that combo names accurately describe their configurations."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())

        # Check that no combo called "simple_baseline" exists (was renamed)
        for section in ["standard_combos", "extended_combos"]:
            for combo in config.get(section, []):
                combo_id = combo.get("id", "")
                imputation = combo.get("imputation_method", "")

                # Check for misleading "baseline" or "simple" in names that use FM
                uses_fm = any(
                    fm in imputation for fm in ["MOMENT", "SAITS", "CSDI", "TimesNet"]
                )

                if uses_fm and (
                    "simple" in combo_id.lower() or "baseline" in combo_id.lower()
                ):
                    pytest.fail(
                        f"GUARDRAIL VIOLATION: Combo '{combo_id}' has misleading name!\n\n"
                        f"  imputation_method: {imputation} (foundation model)\n"
                        f"  But ID suggests 'simple' or 'baseline' (traditional methods)\n\n"
                        f"FIX: Rename to accurately describe (e.g., 'hybrid_*')."
                    )


class TestPresetGroups:
    """Test that preset groups reference valid combos."""

    def test_preset_groups_reference_valid_combos(self):
        """Check that preset_groups only reference existing combo IDs."""
        yaml_path = VIZ_CONFIG_DIR / "plot_hyperparam_combos.yaml"
        if not yaml_path.exists():
            pytest.skip("plot_hyperparam_combos.yaml not found")

        config = yaml.safe_load(yaml_path.read_text())

        # Collect all valid combo IDs
        valid_ids = set()
        for section in ["standard_combos", "extended_combos"]:
            for combo in config.get(section, []):
                if combo.get("id"):
                    valid_ids.add(combo["id"])

        # Check preset_groups
        presets = config.get("preset_groups", {})
        violations = []

        for preset_name, preset_info in presets.items():
            for combo_id in preset_info.get("combos", []):
                if combo_id not in valid_ids:
                    violations.append(
                        {
                            "preset": preset_name,
                            "invalid_id": combo_id,
                        }
                    )

        if violations:
            msg = "GUARDRAIL VIOLATION: preset_groups reference invalid combo IDs!\n\n"
            for v in violations:
                msg += f"  {v['preset']}: '{v['invalid_id']}' not defined\n"
            msg += f"\nValid IDs: {sorted(valid_ids)}"
            pytest.fail(msg)
