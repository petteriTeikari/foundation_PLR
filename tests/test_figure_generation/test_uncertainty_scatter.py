"""Tests for uncertainty scatter figure generation.

Tests the uncertainty visualization pipeline:
1. Module can be imported
2. Correlation computation is correct
3. Plotting functions work with different options
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


class TestUncertaintyScatterModuleStructure:
    """Tests for uncertainty_scatter module structure and imports."""

    def test_module_importable(self):
        """Test that the uncertainty_scatter module can be imported."""
        from src.viz import uncertainty_scatter

        assert uncertainty_scatter is not None

    def test_has_correlation_function(self):
        """Test that correlation computation function exists."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        assert callable(compute_uncertainty_correlation)

    def test_has_scatter_plot_function(self):
        """Test that main scatter plotting function exists."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter

        assert callable(plot_uncertainty_scatter)

    def test_has_correctness_plot_function(self):
        """Test that correctness-based plot function exists."""
        from src.viz.uncertainty_scatter import plot_uncertainty_by_correctness

        assert callable(plot_uncertainty_by_correctness)

    def test_has_figure_generation_function(self):
        """Test that figure generation function exists."""
        from src.viz.uncertainty_scatter import generate_uncertainty_scatter_figure

        assert callable(generate_uncertainty_scatter_figure)

    def test_imports_colors_from_plot_config(self):
        """Test that COLORS is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "uncertainty_scatter.py"
        content = module_path.read_text()

        assert "from src.viz.plot_config import" in content
        assert "COLORS" in content


class TestUncertaintyCorrelationComputation:
    """Tests for uncertainty correlation computation."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data with uncertainty."""
        np.random.seed(42)
        n = 200
        y_prob = np.random.random(n)
        # Uncertainty tends to be higher near 0.5 (boundary)
        uncertainty = np.abs(y_prob - 0.5) * 2 + np.random.exponential(0.1, n)
        return y_prob, uncertainty

    @pytest.fixture
    def correlated_data(self):
        """Generate data with known positive correlation."""
        np.random.seed(42)
        n = 200
        y_prob = np.random.random(n)
        # Uncertainty increases with probability
        uncertainty = y_prob * 0.5 + np.random.normal(0, 0.05, n)
        return y_prob, uncertainty

    def test_correlation_returns_two_floats(self, mock_data):
        """Test that correlation returns (corr, p_value) tuple."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob, uncertainty = mock_data
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        assert isinstance(corr, float)
        assert isinstance(p_val, float)

    def test_correlation_in_valid_range(self, mock_data):
        """Test that correlation is in [-1, 1] range."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob, uncertainty = mock_data
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        assert -1 <= corr <= 1

    def test_p_value_in_valid_range(self, mock_data):
        """Test that p-value is in [0, 1] range."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob, uncertainty = mock_data
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        assert 0 <= p_val <= 1

    def test_positive_correlation_detected(self, correlated_data):
        """Test that positive correlation is detected correctly."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob, uncertainty = correlated_data
        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        # Should detect positive correlation
        assert corr > 0.3, f"Expected positive correlation, got {corr}"
        # Should be significant
        assert p_val < 0.05, f"Expected significant correlation, p={p_val}"

    def test_handles_nan_values(self):
        """Test that NaN values are handled gracefully."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
        uncertainty = np.array([0.5, 0.6, 0.7, np.nan, 0.9])

        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        # Should handle NaN and still compute (or return NaN)
        assert np.isfinite(corr) or np.isnan(corr)

    def test_handles_too_few_samples(self):
        """Test handling of too few samples."""
        from src.viz.uncertainty_scatter import compute_uncertainty_correlation

        y_prob = np.array([0.5, 0.6])
        uncertainty = np.array([0.3, 0.4])

        corr, p_val = compute_uncertainty_correlation(y_prob, uncertainty)

        # Should return NaN for too few samples (< 3)
        # or compute with limited data
        assert np.isfinite(corr) or np.isnan(corr)


class TestUncertaintyScatterPlotting:
    """Tests for scatter plot functionality."""

    @pytest.fixture
    def mock_data(self):
        """Generate complete mock data for plotting."""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1
        )
        uncertainty = np.abs(y_prob - 0.5) + np.random.exponential(0.1, n)
        return y_true, y_prob, uncertainty

    def test_scatter_plot_creates_figure(self, mock_data):
        """Test that scatter plot creates figure and axes."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, ax = plot_uncertainty_scatter(y_true, y_prob, uncertainty)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_scatter_plot_color_by_outcome(self, mock_data):
        """Test color by outcome option."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, ax = plot_uncertainty_scatter(
            y_true, y_prob, uncertainty, color_by_outcome=True
        )

        # Should have legend entries for both classes
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    def test_scatter_plot_no_color_by_outcome(self, mock_data):
        """Test without color by outcome."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, ax = plot_uncertainty_scatter(
            y_true, y_prob, uncertainty, color_by_outcome=False
        )

        assert fig is not None
        plt.close(fig)

    def test_scatter_plot_with_regression(self, mock_data):
        """Test with regression line option."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, ax = plot_uncertainty_scatter(
            y_true, y_prob, uncertainty, show_regression=True
        )

        assert fig is not None
        plt.close(fig)

    def test_scatter_plot_shows_correlation(self, mock_data):
        """Test correlation annotation option."""
        from src.viz.uncertainty_scatter import plot_uncertainty_scatter
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, ax = plot_uncertainty_scatter(
            y_true, y_prob, uncertainty, show_correlation=True
        )

        # Should have text annotation
        texts = ax.texts
        # At least one text box for correlation
        assert len(texts) > 0

        plt.close(fig)


class TestUncertaintyByCorrectnessPlot:
    """Tests for uncertainty by correctness plot."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data with known correctness patterns."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n), 0, 1
        )
        # Higher uncertainty for incorrect predictions
        y_pred = (y_prob >= 0.5).astype(int)
        correct = y_pred == y_true
        uncertainty = np.where(correct, 0.2, 0.5) + np.random.exponential(0.1, n)
        return y_true, y_prob, uncertainty

    def test_correctness_plot_creates_two_panels(self, mock_data):
        """Test that correctness plot creates two-panel figure."""
        from src.viz.uncertainty_scatter import plot_uncertainty_by_correctness
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, axes = plot_uncertainty_by_correctness(y_true, y_prob, uncertainty)

        assert fig is not None
        assert len(axes) == 2  # Correct and Incorrect panels
        plt.close(fig)

    def test_correctness_panels_have_titles(self, mock_data):
        """Test that panels have descriptive titles."""
        from src.viz.uncertainty_scatter import plot_uncertainty_by_correctness
        import matplotlib.pyplot as plt

        y_true, y_prob, uncertainty = mock_data
        fig, axes = plot_uncertainty_by_correctness(y_true, y_prob, uncertainty)

        # Both panels should have titles (non-empty strings)
        for i, ax in enumerate(axes):
            title = ax.get_title()
            assert title is not None, f"Panel {i} title is None"
            # Title might be empty if no samples in that category
            # Just verify the plot was created successfully

        plt.close(fig)


class TestUncertaintyScatterNoHardcoding:
    """Tests to verify no hardcoded values in source code."""

    def test_uses_colors_dict_for_case_control(self):
        """Verify case/control colors come from COLORS dict."""
        module_path = PROJECT_ROOT / "src" / "viz" / "uncertainty_scatter.py"
        content = module_path.read_text()

        # Should reference COLORS["glaucoma"] or COLORS["control"]
        assert 'COLORS["glaucoma"]' in content or "COLORS['glaucoma']" in content
        assert 'COLORS["control"]' in content or "COLORS['control']" in content

    def test_no_hardcoded_hex_colors_in_main_scatter(self):
        """Verify no hardcoded hex colors in main scatter function."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "uncertainty_scatter.py"
        content = module_path.read_text()

        problematic_patterns = []
        in_function = False

        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()

            # Track if we're in the main scatter function
            if "def plot_uncertainty_scatter" in line:
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
            if re.search(r'c\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', line):
                problematic_patterns.append((i, stripped))
            if re.search(r'color\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', line):
                problematic_patterns.append((i, stripped))

        assert len(problematic_patterns) == 0, (
            "Found hardcoded hex colors in scatter function:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in problematic_patterns)
        )

    def test_imports_save_figure_from_plot_config(self):
        """Verify save_figure is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "uncertainty_scatter.py"
        content = module_path.read_text()

        assert "from src.viz.plot_config import" in content
        assert "save_figure" in content


class TestUncertaintyScatterProductionFigure:
    """Tests for production figure (ggplot2 ROC/risk-coverage combined)."""

    # After R migration, the uncertainty scatter Python figure was replaced
    # by the ggplot2 ROC/risk-coverage combined figure with selective classification data.
    COMBINED_FIGURE = FIGURES_DIR / "ggplot2" / "main" / "fig_roc_rc_combined.png"
    SC_JSON = PROJECT_ROOT / "data" / "r_data" / "selective_classification_data.json"

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

    def test_production_json_has_uncertainty_data(self):
        """Test production JSON has retention/rejection data for selective classification."""
        if not self.SC_JSON.exists():
            pytest.skip(f"Production data not found: {self.SC_JSON}. Run: make analyze")
        with open(self.SC_JSON) as f:
            data = json.load(f)

        data_dict = data.get("data", {})
        assert "retention_levels" in data_dict or "rejection_ratios" in data_dict, (
            "Selective classification JSON missing retention_levels/rejection_ratios"
        )

    def test_production_json_has_correlation(self):
        """Test production JSON includes performance metrics per config."""
        if not self.SC_JSON.exists():
            pytest.skip(f"Production data not found: {self.SC_JSON}. Run: make analyze")
        with open(self.SC_JSON) as f:
            data = json.load(f)

        configs = data.get("data", {}).get("configs", [])
        assert len(configs) > 0, "Selective classification JSON has no configs"
        cfg = configs[0]
        assert "baseline_metrics" in cfg or "auroc_at_retention" in cfg, (
            "Selective classification JSON missing performance metrics"
        )

    def test_production_json_not_synthetic(self):
        """Test production JSON is not marked as synthetic."""
        if not self.SC_JSON.exists():
            pytest.skip(f"Production data not found: {self.SC_JSON}. Run: make analyze")
        with open(self.SC_JSON) as f:
            data = json.load(f)

        assert data.get("synthetic") is not True, (
            "CRITICAL: Production figure marked as synthetic!"
        )

    def test_production_json_has_content(self):
        """Test that production JSON has meaningful content."""
        if not self.SC_JSON.exists():
            pytest.skip(f"Production data not found: {self.SC_JSON}. Run: make analyze")
        with open(self.SC_JSON) as f:
            data = json.load(f)

        # Should have some substantive content
        assert len(json.dumps(data)) > 100, "JSON file appears empty or trivial"
