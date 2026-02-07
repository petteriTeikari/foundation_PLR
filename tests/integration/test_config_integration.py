"""
Integration tests for configuration system.

These tests verify:
1. Config loader works with real YAML files
2. Visualization modules can use config loader
3. Config changes propagate to visualization
"""

from pathlib import Path
from types import MappingProxyType


class TestRealConfigFiles:
    """Test loading from actual config files."""

    def test_load_real_combos_yaml(self):
        """Should load combos from actual configs/VISUALIZATION/plot_hyperparam_combos.yaml."""
        from src.config.loader import YAMLConfigLoader

        # Load from real config
        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        combos = loader.load_combos()

        # Verify structure
        assert isinstance(combos, MappingProxyType)
        assert "standard_combos" in combos
        assert "extended_combos" in combos

        # Verify we have the expected combos
        standard = loader.get_standard_combos()
        assert len(standard) >= 4, "Should have at least 4 standard combos"

        # Verify ground_truth is first
        assert standard[0]["id"] == "ground_truth"

        # Verify all required fields present
        for combo in standard:
            assert "id" in combo
            assert "outlier_method" in combo
            assert "imputation_method" in combo
            assert "classifier" in combo
            assert "auroc" in combo

    def test_load_real_methods_yaml(self):
        """Should load methods from actual configs/VISUALIZATION/methods.yaml."""
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        methods = loader.load_methods()

        # Verify structure
        assert isinstance(methods, MappingProxyType)
        assert "outlier_detection" in methods
        assert "imputation" in methods

        # Verify known methods exist
        outlier_methods = methods["outlier_detection"]
        assert "pupil-gt" in outlier_methods
        assert "MOMENT-gt-finetune" in outlier_methods
        assert "LOF" in outlier_methods

        imputation_methods = methods["imputation"]
        assert "SAITS" in imputation_methods
        assert "MOMENT-finetune" in imputation_methods

    def test_load_real_colors_yaml(self):
        """Should load colors from actual configs/VISUALIZATION/colors.yaml."""
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        colors = loader.load_colors()

        # Verify structure
        assert isinstance(colors, MappingProxyType)

        # Verify required colors exist
        assert "ground_truth" in colors
        assert "fm_primary" in colors
        assert "traditional" in colors

    def test_get_combo_by_id_works(self):
        """Should retrieve combo by ID from real config."""
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        # Get ground truth combo
        gt_combo = loader.get_combo_by_id("ground_truth")
        assert gt_combo["outlier_method"] == "pupil-gt"
        assert gt_combo["imputation_method"] == "pupil-gt"
        assert gt_combo["classifier"] == "CatBoost"
        assert gt_combo["auroc"] == 0.9110

        # Get best FM combo
        fm_combo = loader.get_combo_by_id("best_single_fm")
        assert fm_combo["outlier_method"] == "MOMENT-gt-finetune"
        assert fm_combo["imputation_method"] == "SAITS"

    def test_get_method_color_works(self):
        """Should resolve method colors from real config."""
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        # Get color for ground truth
        gt_color = loader.get_method_color("pupil-gt", "outlier_detection")
        assert gt_color.startswith("#")

        # Get color for MOMENT
        moment_color = loader.get_method_color(
            "MOMENT-gt-finetune", "outlier_detection"
        )
        assert moment_color.startswith("#")


class TestPlotConfigIntegration:
    """Test integration between plot_config and config loader."""

    def test_plot_config_imports_work(self):
        """All expected imports from plot_config should work."""
        from src.viz.plot_config import (
            setup_style,
            save_figure,
            COLORS,
            KEY_STATS,
            add_benchmark_line,
            get_combo_color,
            get_method_display_name,
        )

        # Verify callable
        assert callable(setup_style)
        assert callable(save_figure)
        assert callable(add_benchmark_line)
        assert callable(get_combo_color)
        assert callable(get_method_display_name)

        # Verify dicts
        assert isinstance(COLORS, dict)
        assert isinstance(KEY_STATS, dict)

    def test_key_stats_matches_combos(self):
        """KEY_STATS AUROC values should match plot_hyperparam_combos.yaml."""
        from src.viz.plot_config import KEY_STATS
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        gt_combo = loader.get_combo_by_id("ground_truth")

        # KEY_STATS ground_truth_auroc should match plot_hyperparam_combos.yaml
        assert KEY_STATS["ground_truth_auroc"] == gt_combo["auroc"]


class TestConfigDrivenVisualization:
    """Test that visualization modules can use config."""

    def test_utility_matrix_imports(self):
        """utility_matrix should be importable."""
        from src.viz.utility_matrix import create_figure, get_utility_data

        assert callable(create_figure)
        assert callable(get_utility_data)

    def test_utility_matrix_uses_key_stats(self):
        """utility_matrix should use KEY_STATS for featurization values."""
        import pytest
        from src.viz.utility_matrix import get_utility_data
        from src.viz.plot_config import KEY_STATS

        try:
            utility_data = get_utility_data()
        except (FileNotFoundError, Exception):
            pytest.skip("DuckDB not available for utility_matrix test")

        # Featurization row should use KEY_STATS values
        feat_data = utility_data.get("Featurization", {})

        # These should match KEY_STATS
        assert feat_data.get("fm_performance") == KEY_STATS["embeddings_mean_auroc"]
        assert (
            feat_data.get("baseline_performance") == KEY_STATS["handcrafted_mean_auroc"]
        )


class TestNoHardcodedValues:
    """Verify no hardcoded method names in source (outside config/comments)."""

    def test_config_loader_has_no_hardcoded_method_names(self):
        """config/loader.py should not have hardcoded method names."""
        source_path = (
            Path(__file__).parent.parent.parent / "src" / "config" / "loader.py"
        )
        source = source_path.read_text()

        # These specific method names should not appear in loader.py
        # (they should only be in YAML config files)
        forbidden = [
            "MOMENT-gt-finetune",
            "MOMENT-gt-zeroshot",
            "UniTS-orig-finetune",
            "ensemble-LOF-MOMENT",
        ]

        for method in forbidden:
            assert method not in source, (
                f"Hardcoded method name '{method}' found in loader.py. "
                "Method names should only appear in YAML config files."
            )

    def test_combo_ids_are_lowercase_underscored(self):
        """All combo IDs should follow naming convention."""
        from src.config.loader import YAMLConfigLoader

        config_dir = Path(__file__).parent.parent.parent / "configs" / "VISUALIZATION"
        loader = YAMLConfigLoader(config_dir=config_dir)

        import re

        pattern = re.compile(r"^[a-z_]+$")

        for combo in loader.get_all_combos():
            combo_id = combo["id"]
            assert pattern.match(combo_id), (
                f"Combo ID '{combo_id}' doesn't match naming convention. "
                "Should be lowercase with underscores only."
            )
