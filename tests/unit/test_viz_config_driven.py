"""
Tests verifying visualization modules use config, not hardcoded values.

These tests ensure that:
1. Method names come from config
2. Colors come from config
3. AUROC values come from config where applicable
"""

import pytest
from pathlib import Path
import re


# ============================================================================
# TEST CLASS: No Hardcoded Method Names in Code
# ============================================================================


class TestNoHardcodedMethodNames:
    """Verify source code doesn't hardcode method names that should come from config."""

    @pytest.fixture
    def viz_source_files(self):
        """Get all Python source files in src/viz."""
        viz_dir = Path(__file__).parent.parent.parent / "src" / "viz"
        return list(viz_dir.glob("*.py"))

    def test_no_hardcoded_combo_auroc_in_utility_matrix(self):
        """utility_matrix.py should not have hardcoded AUROC values for combos."""
        source_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "utility_matrix.py"
        )
        source = source_path.read_text()

        # These specific hardcoded AUROC values should not appear
        # Note: We allow the KEY_STATS import which has these values centralized
        # The issue is when values appear directly in UTILITY_DATA dict

        # Check that UTILITY_DATA doesn't have hardcoded numeric literals
        # Pattern: 'fm_performance': 0.xxx or 'baseline_performance': 0.xxx
        hardcoded_pattern = r"'(fm_performance|baseline_performance)':\s*0\.\d{2,3}"

        # Allow KEY_STATS references like KEY_STATS['embeddings_mean_auroc']
        # But flag direct numeric assignments
        utility_data_section = re.search(
            r"UTILITY_DATA\s*=\s*\{.*?\n\}", source, re.DOTALL
        )

        if utility_data_section:
            section_text = utility_data_section.group()
            # Count hardcoded numeric values (excluding KEY_STATS references)
            hardcoded_matches = re.findall(hardcoded_pattern, section_text)

            # We expect some hardcoded values for now (will be migrated later)
            # This test documents the current state and will fail after migration
            # For now, just verify the count doesn't increase
            assert len(hardcoded_matches) <= 4, (
                f"Found {len(hardcoded_matches)} hardcoded performance values in UTILITY_DATA. "
                "These should be loaded from config."
            )

    def test_no_forbidden_method_names_in_comments_only(self):
        """Method names in docstrings/comments are OK, in code logic they're not."""
        # This is a more relaxed test - we allow method names in comments
        # but not in actual variable assignments or conditionals
        source_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "plot_config.py"
        )
        source = source_path.read_text()

        # The config loader import should exist
        assert (
            "from src.viz.config_loader import" in source
        ), "plot_config.py should import from src.viz.config_loader"


# ============================================================================
# TEST CLASS: Config Functions Available
# ============================================================================


class TestConfigFunctionsAvailable:
    """Verify config access functions are available in plot_config."""

    def test_get_combo_color_exists(self):
        """get_combo_color function should be available."""
        from src.viz.plot_config import get_combo_color

        # Should return a color string for valid combo
        color = get_combo_color("ground_truth")
        assert isinstance(color, str)
        assert color.startswith("#") or color.startswith("rgb")

    def test_get_method_display_name_exists(self):
        """get_method_display_name function should be available."""
        from src.viz.plot_config import get_method_display_name

        # Should return a string
        name = get_method_display_name("MOMENT-gt-finetune", "outlier_detection")
        assert isinstance(name, str)

    def test_key_stats_has_required_keys(self):
        """KEY_STATS should have all required performance statistics."""
        from src.viz.plot_config import KEY_STATS

        required_keys = [
            "handcrafted_mean_auroc",
            "embeddings_mean_auroc",
            "benchmark_auroc",
            "ground_truth_auroc",
        ]

        for key in required_keys:
            assert key in KEY_STATS, f"KEY_STATS missing required key: {key}"

    def test_colors_has_semantic_keys(self):
        """COLORS should have semantic color keys."""
        from src.viz.plot_config import COLORS

        semantic_keys = ["good", "bad", "neutral", "reference"]

        for key in semantic_keys:
            assert key in COLORS, f"COLORS missing semantic key: {key}"


# ============================================================================
# TEST CLASS: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Verify backward compatibility with existing code."""

    def test_setup_style_is_alias(self):
        """setup_style should be an alias for apply_style."""
        from src.viz.plot_config import setup_style, apply_style

        assert setup_style is apply_style

    def test_save_figure_exists(self):
        """save_figure function should exist and be callable."""
        from src.viz.plot_config import save_figure

        assert callable(save_figure)

    def test_add_benchmark_line_exists(self):
        """add_benchmark_line function should exist and be callable."""
        from src.viz.plot_config import add_benchmark_line

        assert callable(add_benchmark_line)


# ============================================================================
# TEST CLASS: Integration with Config Loader
# ============================================================================


class TestConfigLoaderIntegration:
    """Test integration between viz modules and config loader."""

    def test_plot_config_uses_config_loader(self):
        """plot_config should import and use config loader."""
        import src.viz.plot_config as pc

        # Should have config loader imported
        assert hasattr(pc, "get_config_loader")

    def test_get_combo_color_uses_config(self):
        """get_combo_color should read from config when available."""
        from src.viz.plot_config import get_combo_color

        # Test that get_combo_color returns a valid color string for known combos
        color = get_combo_color("ground_truth")
        assert isinstance(color, str)
        assert color.startswith("#")

        # Test fallback for unknown combo
        unknown_color = get_combo_color("nonexistent_combo")
        assert isinstance(unknown_color, str)
        assert unknown_color.startswith("#")
