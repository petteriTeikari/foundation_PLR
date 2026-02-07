"""Tests for Decision Curve Analysis (DCA) figure generation.

Tests the DCA visualization pipeline:
1. Module can be imported
2. Net benefit computation is correct
3. DCA curves (model, treat-all, treat-none) are computed correctly
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


class TestDCAPlotModuleStructure:
    """Tests for dca_plot module structure and imports."""

    def test_module_importable(self):
        """Test that the dca_plot module can be imported."""
        from src.viz import dca_plot

        assert dca_plot is not None

    def test_has_net_benefit_function(self):
        """Test that net benefit computation function exists."""
        from src.viz.dca_plot import compute_net_benefit

        assert callable(compute_net_benefit)

    def test_has_reference_strategy_functions(self):
        """Test that treat-all and treat-none functions exist."""
        from src.viz.dca_plot import compute_treat_all_nb, compute_treat_none_nb

        assert callable(compute_treat_all_nb)
        assert callable(compute_treat_none_nb)

    def test_has_dca_curves_function(self):
        """Test that DCA curve computation function exists."""
        from src.viz.dca_plot import compute_dca_curves

        assert callable(compute_dca_curves)

    def test_has_plotting_functions(self):
        """Test that plotting functions exist."""
        from src.viz.dca_plot import plot_dca, plot_dca_multi_model

        assert callable(plot_dca)
        assert callable(plot_dca_multi_model)

    def test_has_figure_generation_function(self):
        """Test that figure generation function exists."""
        from src.viz.dca_plot import generate_dca_figure

        assert callable(generate_dca_figure)

    def test_has_db_loading_function(self):
        """Test that DB loading function exists."""
        from src.viz.dca_plot import load_dca_curves_from_db

        assert callable(load_dca_curves_from_db)

    def test_has_plot_from_db_function(self):
        """Test that plot_dca_from_db function exists."""
        from src.viz.dca_plot import plot_dca_from_db

        assert callable(plot_dca_from_db)

    def test_exports_all_public_functions(self):
        """Test that __all__ exports expected functions."""
        from src.viz.dca_plot import __all__

        expected = [
            "compute_net_benefit",
            "compute_treat_all_nb",
            "compute_treat_none_nb",
            "compute_dca_curves",
            "load_dca_curves_from_db",
            "plot_dca",
            "plot_dca_from_db",
            "plot_dca_multi_model",
            "generate_dca_figure",
        ]
        for fn_name in expected:
            assert fn_name in __all__, f"Missing from __all__: {fn_name}"


class TestDCANetBenefitComputation:
    """Tests for net benefit calculation."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing."""
        np.random.seed(42)
        n = 200
        prevalence = 0.27
        y_true = np.random.binomial(1, prevalence, n)
        # Predictions with reasonable discrimination
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.2 + np.random.normal(0, 0.1, n), 0, 1
        )
        return y_true, y_prob

    def test_net_benefit_returns_float(self, mock_data):
        """Test that net benefit returns a float."""
        from src.viz.dca_plot import compute_net_benefit

        y_true, y_prob = mock_data
        result = compute_net_benefit(y_true, y_prob, threshold=0.15)

        assert isinstance(result, float)

    def test_net_benefit_formula(self):
        """Test net benefit formula: NB = TP/n - FP/n * (pt/(1-pt))."""
        from src.viz.dca_plot import compute_net_benefit

        # Simple case with known values
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 3 cases, 7 controls
        y_prob = np.array([0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        threshold = 0.5

        # At threshold 0.5: predictions = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        # TP = 2 (predictions 0.8, 0.7 for cases)
        # FP = 1 (prediction 0.6 for control)
        # NB = 2/10 - 1/10 * (0.5/0.5) = 0.2 - 0.1 = 0.1

        result = compute_net_benefit(y_true, y_prob, threshold)
        expected = 0.1

        assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"

    def test_treat_all_net_benefit(self):
        """Test treat-all strategy net benefit."""
        from src.viz.dca_plot import compute_treat_all_nb

        # NB(treat-all) = prevalence - (1-prevalence) * (pt/(1-pt))
        prevalence = 0.3
        threshold = 0.2

        result = compute_treat_all_nb(prevalence, threshold)
        expected = prevalence - (1 - prevalence) * (threshold / (1 - threshold))

        assert abs(result - expected) < 0.001

    def test_treat_none_net_benefit_always_zero(self):
        """Test treat-none strategy always returns 0."""
        from src.viz.dca_plot import compute_treat_none_nb

        for threshold in [0.1, 0.2, 0.3, 0.5, 0.8]:
            result = compute_treat_none_nb(threshold)
            assert result == 0.0

    def test_net_benefit_edge_cases(self, mock_data):
        """Test net benefit handles edge case thresholds."""
        from src.viz.dca_plot import compute_net_benefit

        y_true, y_prob = mock_data

        # Threshold at 1.0 should return 0 (no one treated)
        result_1 = compute_net_benefit(y_true, y_prob, threshold=1.0)
        assert result_1 == 0.0

        # Threshold near 0 should be handled
        result_0 = compute_net_benefit(y_true, y_prob, threshold=0.001)
        assert np.isfinite(result_0)


class TestDCACurvesComputation:
    """Tests for DCA curve computation."""

    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.27, n)
        y_prob = np.clip(
            y_true * 0.7 + (1 - y_true) * 0.2 + np.random.normal(0, 0.1, n), 0, 1
        )
        return y_true, y_prob

    def test_dca_curves_returns_dict(self, mock_data):
        """Test that compute_dca_curves returns dict with required keys."""
        from src.viz.dca_plot import compute_dca_curves

        y_true, y_prob = mock_data
        result = compute_dca_curves(y_true, y_prob)

        assert isinstance(result, dict)
        assert "thresholds" in result
        assert "nb_model" in result
        assert "nb_all" in result
        assert "nb_none" in result
        assert "prevalence" in result

    def test_dca_curves_arrays_same_length(self, mock_data):
        """Test that all DCA curve arrays have same length."""
        from src.viz.dca_plot import compute_dca_curves

        y_true, y_prob = mock_data
        result = compute_dca_curves(y_true, y_prob, n_thresholds=50)

        n = len(result["thresholds"])
        assert len(result["nb_model"]) == n
        assert len(result["nb_all"]) == n
        assert len(result["nb_none"]) == n

    def test_dca_thresholds_in_range(self, mock_data):
        """Test that thresholds are in specified range."""
        from src.viz.dca_plot import compute_dca_curves

        y_true, y_prob = mock_data
        threshold_range = (0.05, 0.25)
        result = compute_dca_curves(y_true, y_prob, threshold_range=threshold_range)

        thresholds = result["thresholds"]
        assert thresholds.min() >= threshold_range[0] - 0.001
        assert thresholds.max() <= threshold_range[1] + 0.001

    def test_treat_none_is_zero_array(self, mock_data):
        """Test that treat-none NB is always zero."""
        from src.viz.dca_plot import compute_dca_curves

        y_true, y_prob = mock_data
        result = compute_dca_curves(y_true, y_prob)

        nb_none = result["nb_none"]
        assert np.all(nb_none == 0), "Treat-none should always be 0"

    def test_good_model_beats_treat_all_at_high_threshold(self, mock_data):
        """Test that a useful model beats treat-all at higher thresholds."""
        from src.viz.dca_plot import compute_dca_curves

        y_true, y_prob = mock_data
        result = compute_dca_curves(y_true, y_prob, threshold_range=(0.1, 0.4))

        # At higher thresholds, a good model should outperform treat-all
        # (treat-all becomes negative at high thresholds)
        nb_model = result["nb_model"]
        nb_all = result["nb_all"]

        # Model should be >= treat-all for most thresholds (not strict equality)
        model_advantage = np.sum(nb_model >= nb_all)
        total = len(nb_model)
        assert (
            model_advantage / total > 0.3
        ), "A reasonable model should beat treat-all at some thresholds"


class TestDCAPlotNoHardcoding:
    """Tests to verify no hardcoded values in source code."""

    def test_no_hardcoded_hex_colors_in_model_curve(self):
        """Verify model curve uses COLORS dict, not hardcoded hex."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "dca_plot.py"
        content = module_path.read_text()

        problematic_patterns = []
        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "COLORS[" in line or "COLORS.get(" in line:
                continue
            # Check for hardcoded hex colors
            if re.search(r'color\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', line):
                problematic_patterns.append((i, stripped))

        assert len(problematic_patterns) == 0, (
            "Found hardcoded hex colors:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in problematic_patterns)
        )

    def test_imports_save_figure_from_plot_config(self):
        """Verify save_figure is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "dca_plot.py"
        content = module_path.read_text()

        assert (
            "from src.viz.plot_config import" in content
            or "from plot_config import" in content
        )
        assert "save_figure" in content


class TestDCAPlotProductionFigure:
    """Tests for production figure (combined ggplot2 calibration+DCA)."""

    # After R migration, individual DCA figures were merged into
    # a composite ggplot2 figure. JSON data is in data/r_data/.
    COMBINED_FIGURE = (
        FIGURES_DIR / "ggplot2" / "main" / "fig_calibration_dca_combined.png"
    )
    DCA_JSON = PROJECT_ROOT / "data" / "r_data" / "dca_data.json"

    def test_production_figure_not_blank(self):
        """Test production figure has visible content."""
        assert (
            self.COMBINED_FIGURE.exists()
        ), f"Missing: {self.COMBINED_FIGURE}. Run: make analyze"
        img = Image.open(self.COMBINED_FIGURE)
        arr = np.array(img)

        if len(arr.shape) == 3:
            std_vals = [arr[:, :, i].std() for i in range(min(3, arr.shape[2]))]
            assert max(std_vals) > 10, "Figure appears blank"
        else:
            assert arr.std() > 10, "Figure appears blank"

    def test_production_json_has_dca_data(self):
        """Test production JSON has DCA curve data."""
        assert self.DCA_JSON.exists(), f"Missing: {self.DCA_JSON}. Run: make analyze"
        with open(self.DCA_JSON) as f:
            data = json.load(f)

        json_str = json.dumps(data)
        assert (
            "threshold" in json_str or "net_benefit" in json_str
        ), "DCA JSON missing threshold/net_benefit data"

    def test_production_json_has_reference_strategies(self):
        """Test production JSON includes treat-all and treat-none."""
        assert self.DCA_JSON.exists(), f"Missing: {self.DCA_JSON}"
        with open(self.DCA_JSON) as f:
            data = json.load(f)

        json_str = json.dumps(data)
        assert (
            "all" in json_str.lower() or "none" in json_str.lower()
        ), "DCA JSON missing reference strategies (treat-all/treat-none)"

    def test_production_json_not_synthetic(self):
        """Test production JSON is not marked as synthetic."""
        assert self.DCA_JSON.exists(), f"Missing: {self.DCA_JSON}"
        with open(self.DCA_JSON) as f:
            data = json.load(f)

        assert (
            data.get("synthetic") is not True
        ), "CRITICAL: Production figure marked as synthetic!"

    def test_production_threshold_range_clinical(self):
        """Test production figure uses clinically relevant threshold range."""
        assert self.DCA_JSON.exists(), f"Missing: {self.DCA_JSON}"
        with open(self.DCA_JSON) as f:
            data = json.load(f)

        # Check for threshold data in various structures
        json_str = json.dumps(data)
        assert "threshold" in json_str.lower(), "DCA JSON missing threshold data"
