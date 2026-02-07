"""Tests for probability distribution figure generation.

Tests the probability distribution visualization pipeline:
1. Module can be imported
2. Distribution statistics are computed correctly
3. Multiple plot types (histogram, density, violin, box) work
4. Production figures (when present) meet quality standards
5. No hardcoded colors in source code
"""

import json
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"


class TestProbDistributionModuleStructure:
    """Tests for prob_distribution module structure and imports."""

    def test_module_importable(self):
        """Test that the prob_distribution module can be imported."""
        from src.viz import prob_distribution

        assert prob_distribution is not None

    def test_has_stats_function(self):
        """Test that distribution statistics function exists."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        assert callable(_compute_stats_from_arrays)

    def test_has_db_loading_function(self):
        """Test that DB loading function exists."""
        from src.viz.prob_distribution import load_distribution_stats_from_db

        assert callable(load_distribution_stats_from_db)

    def test_has_plotting_function(self):
        """Test that main plotting function exists."""
        from src.viz.prob_distribution import plot_probability_distributions

        assert callable(plot_probability_distributions)

    def test_has_figure_generation_function(self):
        """Test that figure generation function exists."""
        from src.viz.prob_distribution import generate_probability_distribution_figure

        assert callable(generate_probability_distribution_figure)

    def test_imports_colors_from_plot_config(self):
        """Test that COLORS is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "prob_distribution.py"
        content = module_path.read_text()

        assert "from src.viz.plot_config import" in content
        assert "COLORS" in content


class TestProbDistributionStatsComputation:
    """Tests for distribution statistics computation."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.27, n)
        # Good discrimination: cases have higher probabilities
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.2 + np.random.normal(0, 0.1, n), 0, 1
        )
        return y_true, y_prob

    def test_stats_returns_dict(self, mock_data):
        """Test that compute_distribution_stats returns dict with expected keys."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true, y_prob = mock_data
        stats = _compute_stats_from_arrays(y_true, y_prob)

        assert isinstance(stats, dict)
        expected_keys = [
            "auroc",
            "median_cases",
            "median_controls",
            "mean_cases",
            "mean_controls",
            "n_cases",
            "n_controls",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_auroc_is_nan(self, mock_data):
        """Test AUROC is NaN (must come from DuckDB, not computed here)."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true, y_prob = mock_data
        stats = _compute_stats_from_arrays(y_true, y_prob)

        assert np.isnan(stats["auroc"]), (
            "AUROC should be NaN from _compute_stats_from_arrays "
            "(must be read from DuckDB instead)"
        )

    def test_stats_medians_in_valid_range(self, mock_data):
        """Test medians are in [0, 1] range."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true, y_prob = mock_data
        stats = _compute_stats_from_arrays(y_true, y_prob)

        assert 0 <= stats["median_cases"] <= 1
        assert 0 <= stats["median_controls"] <= 1

    def test_stats_counts_match_data(self, mock_data):
        """Test that case/control counts match data."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true, y_prob = mock_data
        stats = _compute_stats_from_arrays(y_true, y_prob)

        expected_cases = np.sum(y_true == 1)
        expected_controls = np.sum(y_true == 0)

        assert stats["n_cases"] == expected_cases
        assert stats["n_controls"] == expected_controls

    def test_good_discrimination_cases_higher(self, mock_data):
        """Test that with good discrimination, cases have higher mean probability."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true, y_prob = mock_data
        stats = _compute_stats_from_arrays(y_true, y_prob)

        # With good discrimination, cases should have higher mean
        assert stats["mean_cases"] > stats["mean_controls"], (
            "Cases should have higher mean probability than controls"
        )

    def test_stats_handles_single_class(self):
        """Test stats handle single class gracefully."""
        from src.viz.prob_distribution import _compute_stats_from_arrays

        y_true = np.zeros(50)  # All controls
        y_prob = np.random.random(50)

        stats = _compute_stats_from_arrays(y_true, y_prob)

        # AUROC is always NaN from this function (must come from DB)
        assert np.isnan(stats["auroc"])
        # Case stats should be NaN with no cases
        assert np.isnan(stats["median_cases"])
        assert stats["n_cases"] == 0


class TestProbDistributionPlotTypes:
    """Tests for different plot type options."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing."""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1
        )
        return y_true, y_prob

    def test_histogram_plot_type(self, mock_data):
        """Test histogram plot type works."""
        from src.viz.prob_distribution import plot_probability_distributions
        import matplotlib.pyplot as plt

        y_true, y_prob = mock_data
        fig, ax = plot_probability_distributions(y_true, y_prob, plot_type="histogram")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_density_plot_type(self, mock_data):
        """Test density plot type works."""
        from src.viz.prob_distribution import plot_probability_distributions
        import matplotlib.pyplot as plt

        y_true, y_prob = mock_data
        fig, ax = plot_probability_distributions(y_true, y_prob, plot_type="density")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_violin_plot_type(self, mock_data):
        """Test violin plot type works."""
        from src.viz.prob_distribution import plot_probability_distributions
        import matplotlib.pyplot as plt

        y_true, y_prob = mock_data
        fig, ax = plot_probability_distributions(y_true, y_prob, plot_type="violin")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_box_plot_type(self, mock_data):
        """Test box plot type works."""
        from src.viz.prob_distribution import plot_probability_distributions
        import matplotlib.pyplot as plt

        y_true, y_prob = mock_data
        fig, ax = plot_probability_distributions(y_true, y_prob, plot_type="box")

        assert fig is not None
        assert ax is not None
        plt.close(fig)


class TestProbDistributionNoHardcoding:
    """Tests to verify no hardcoded values in source code."""

    def test_uses_colors_dict_for_case_control(self):
        """Verify case/control colors come from COLORS dict."""
        module_path = PROJECT_ROOT / "src" / "viz" / "prob_distribution.py"
        content = module_path.read_text()

        # Should reference COLORS["glaucoma"] or COLORS["control"]
        assert 'COLORS["glaucoma"]' in content or "COLORS['glaucoma']" in content
        assert 'COLORS["control"]' in content or "COLORS['control']" in content

    def test_no_hardcoded_hex_colors_in_plotting(self):
        """Verify no hardcoded hex colors in main plotting code."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "prob_distribution.py"
        content = module_path.read_text()

        problematic_patterns = []
        in_function = False

        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()

            # Track if we're in the main plotting function
            if "def plot_probability_distributions" in line:
                in_function = True
            elif line.startswith("def ") and in_function:
                in_function = False

            if not in_function:
                continue

            # Skip comments
            if stripped.startswith("#"):
                continue
            # Skip COLORS references
            if "COLORS[" in line or "COLORS.get(" in line:
                continue
            # Skip parameter defaults
            if "case_color" in line or "control_color" in line:
                continue

            # Check for hardcoded hex colors
            if re.search(r'color\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', line):
                problematic_patterns.append((i, stripped))

        assert len(problematic_patterns) == 0, (
            "Found hardcoded hex colors in plot function:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in problematic_patterns)
        )

    def test_imports_save_figure_from_plot_config(self):
        """Verify save_figure is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "prob_distribution.py"
        content = module_path.read_text()

        assert "from src.viz.plot_config import" in content
        assert "save_figure" in content


class TestProbDistributionProductionFigure:
    """Tests for production figure (combined ggplot2 probability distribution)."""

    # After R migration, individual prob dist figures were merged into
    # a composite ggplot2 figure. JSON data is in data/r_data/.
    COMBINED_FIGURE = FIGURES_DIR / "ggplot2" / "main" / "fig_prob_dist_combined.png"
    PREDICTIONS_JSON = PROJECT_ROOT / "data" / "r_data" / "predictions_top4.json"

    @pytest.fixture(autouse=True)
    def _require_production_data(self):
        """Skip all tests in this class if production data is not available."""
        if not self.COMBINED_FIGURE.exists():
            pytest.skip(
                f"Production data not found: {self.COMBINED_FIGURE}. Run: make analyze"
            )

    def test_production_figure_not_blank(self):
        """Test production figure has visible content."""
        img = Image.open(self.COMBINED_FIGURE)
        arr = np.array(img)

        if len(arr.shape) == 3:
            std_vals = [arr[:, :, i].std() for i in range(min(3, arr.shape[2]))]
            assert max(std_vals) > 10, "Figure appears blank"
        else:
            assert arr.std() > 10, "Figure appears blank"

    def test_production_json_has_distribution_data(self):
        """Test production JSON has distribution data."""
        if not self.PREDICTIONS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.PREDICTIONS_JSON}. Run: make analyze"
            )
        with open(self.PREDICTIONS_JSON) as f:
            data = json.load(f)

        json_str = json.dumps(data)
        assert (
            "y_prob" in json_str or "predictions" in json_str or "prob" in json_str
        ), "Predictions JSON missing probability data"

    def test_production_json_has_statistics(self):
        """Test production JSON includes prediction data for each config."""
        if not self.PREDICTIONS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.PREDICTIONS_JSON}. Run: make analyze"
            )
        with open(self.PREDICTIONS_JSON) as f:
            data = json.load(f)

        configs = data.get("data", {}).get("configs", [])
        assert len(configs) > 0, "JSON has no config data"
        for cfg in configs:
            assert "y_true" in cfg, f"Config {cfg.get('name')} missing y_true"
            assert "y_prob" in cfg, f"Config {cfg.get('name')} missing y_prob"

    def test_production_json_not_synthetic(self):
        """Test production JSON is not marked as synthetic."""
        if not self.PREDICTIONS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.PREDICTIONS_JSON}. Run: make analyze"
            )
        with open(self.PREDICTIONS_JSON) as f:
            data = json.load(f)

        assert data.get("synthetic") is not True, (
            "CRITICAL: Production figure marked as synthetic!"
        )

    def test_production_distributions_separable(self):
        """Test that case and control distributions are separable."""
        if not self.PREDICTIONS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.PREDICTIONS_JSON}. Run: make analyze"
            )
        with open(self.PREDICTIONS_JSON) as f:
            data = json.load(f)

        # Check for separability in various data structures
        if "y_prob_cases" in data and "y_prob_controls" in data:
            cases = np.array(data["y_prob_cases"])
            controls = np.array(data["y_prob_controls"])

            if len(cases) > 0 and len(controls) > 0:
                assert np.mean(cases) > np.mean(controls), (
                    "Cases should have higher mean probability than controls"
                )
        elif "statistics" in data:
            stats = data["statistics"]
            if "mean_cases" in stats and "mean_controls" in stats:
                assert stats["mean_cases"] > stats["mean_controls"]
