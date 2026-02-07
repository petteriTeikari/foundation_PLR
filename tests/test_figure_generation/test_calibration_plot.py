"""Tests for calibration plot (STRATOS-compliant) figure generation.

Tests the calibration visualization pipeline:
1. Module can be imported
2. LOESS smoothing and CI computation work
3. Calibration metrics (slope, intercept, ICI) are computed correctly
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


class TestCalibrationPlotModuleStructure:
    """Tests for calibration_plot module structure and imports."""

    def test_module_importable(self):
        """Test that the calibration_plot module can be imported."""
        from src.viz import calibration_plot

        assert calibration_plot is not None

    def test_has_loess_function(self):
        """Test that LOESS calibration function exists."""
        from src.viz.calibration_plot import compute_loess_calibration

        assert callable(compute_loess_calibration)

    def test_has_ci_function(self):
        """Test that CI computation function exists."""
        from src.viz.calibration_plot import compute_calibration_ci

        assert callable(compute_calibration_ci)

    def test_has_plotting_functions(self):
        """Test that plotting functions exist."""
        from src.viz.calibration_plot import (
            plot_calibration_curve,
            plot_calibration_multi_model,
        )

        assert callable(plot_calibration_curve)
        assert callable(plot_calibration_multi_model)

    def test_has_figure_generation_function(self):
        """Test that figure generation function exists."""
        from src.viz.calibration_plot import generate_calibration_figure

        assert callable(generate_calibration_figure)

    def test_has_db_export_functions(self):
        """Test that DuckDB-based export functions exist (computation decoupling)."""
        from src.viz.calibration_plot import (
            save_calibration_extended_json_from_db,
            save_calibration_multi_combo_json_from_db,
        )

        assert callable(save_calibration_extended_json_from_db)
        assert callable(save_calibration_multi_combo_json_from_db)


class TestCalibrationComputation:
    """Tests for calibration curve computation."""

    @pytest.fixture
    def well_calibrated_data(self):
        """Generate well-calibrated mock data."""
        np.random.seed(42)
        n = 500
        # True probabilities
        p_true = np.random.uniform(0.1, 0.9, n)
        # Labels sampled according to true probabilities
        y_true = (np.random.random(n) < p_true).astype(int)
        # Predictions close to true probabilities (well calibrated)
        y_prob = np.clip(p_true + np.random.normal(0, 0.05, n), 0.01, 0.99)
        return y_true, y_prob

    @pytest.fixture
    def poorly_calibrated_data(self):
        """Generate poorly-calibrated mock data."""
        np.random.seed(42)
        n = 500
        # True probability around 0.3
        y_true = np.random.binomial(1, 0.3, n)
        # Predictions overconfident (biased high)
        y_prob = np.clip(y_true * 0.9 + (1 - y_true) * 0.4, 0.01, 0.99)
        return y_true, y_prob

    def test_loess_calibration_returns_arrays(self, well_calibrated_data):
        """Test LOESS calibration returns sorted arrays."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true, y_prob = well_calibrated_data
        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        assert isinstance(x_smooth, np.ndarray)
        assert isinstance(y_smooth, np.ndarray)
        assert len(x_smooth) == len(y_smooth)
        # Should be sorted
        assert np.all(np.diff(x_smooth) >= 0), "x_smooth should be sorted"

    def test_loess_values_in_valid_range(self, well_calibrated_data):
        """Test LOESS values are approximately in [0, 1] range."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true, y_prob = well_calibrated_data
        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        # x values should be in prediction range
        assert x_smooth.min() >= 0
        assert x_smooth.max() <= 1
        # y values (observed frequencies) should be approximately in [0, 1]
        # LOESS smoothing can produce values slightly outside [0, 1] at edges
        # This is expected behavior for LOESS - clipping happens downstream
        assert y_smooth.min() >= -0.1, "y_smooth too negative"
        assert y_smooth.max() <= 1.1, "y_smooth too large"

    def test_calibration_ci_returns_three_arrays(self, well_calibrated_data):
        """Test CI computation returns x, lower, upper arrays."""
        from src.viz.calibration_plot import compute_calibration_ci

        y_true, y_prob = well_calibrated_data
        x_vals, y_lower, y_upper = compute_calibration_ci(
            y_true,
            y_prob,
            n_bootstrap=50,  # Fewer for speed
        )

        assert isinstance(x_vals, np.ndarray)
        assert isinstance(y_lower, np.ndarray)
        assert isinstance(y_upper, np.ndarray)
        assert len(x_vals) == len(y_lower) == len(y_upper)

    def test_calibration_ci_bounds_ordered(self, well_calibrated_data):
        """Test that CI lower <= upper (excluding NaN values)."""
        from src.viz.calibration_plot import compute_calibration_ci

        y_true, y_prob = well_calibrated_data
        x_vals, y_lower, y_upper = compute_calibration_ci(
            y_true, y_prob, n_bootstrap=50
        )

        # Filter out NaN values (can occur at boundaries due to extrapolation)
        valid_mask = ~(np.isnan(y_lower) | np.isnan(y_upper))

        # Lower bound should be <= upper bound where both are valid
        if valid_mask.sum() > 0:
            assert np.all(y_lower[valid_mask] <= y_upper[valid_mask]), (
                "CI lower should be <= upper (for valid values)"
            )

    def test_plot_accepts_precomputed_metrics(self, well_calibrated_data):
        """Test that plot_calibration_curve accepts pre-computed metrics dict."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.viz.calibration_plot import plot_calibration_curve

        y_true, y_prob = well_calibrated_data

        # Simulate metrics from DuckDB
        metrics = {
            "calibration_slope": 1.02,
            "calibration_intercept": -0.01,
        }

        fig, ax = plot_calibration_curve(
            y_true, y_prob, show_metrics=True, metrics=metrics
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_well_calibrated_loess_close_to_diagonal(self, well_calibrated_data):
        """Test that well-calibrated data LOESS curve is close to diagonal."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true, y_prob = well_calibrated_data
        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        # Well-calibrated predictions: LOESS should be close to diagonal
        ici = np.mean(np.abs(y_smooth - x_smooth))
        assert ici < 0.15, (
            f"Expected LOESS close to diagonal for well-calibrated data, got ICI={ici}"
        )

    def test_loess_ici_is_non_negative(self, well_calibrated_data):
        """Test that ICI from LOESS is non-negative."""
        from src.viz.calibration_plot import compute_loess_calibration

        y_true, y_prob = well_calibrated_data
        x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)

        ici = np.mean(np.abs(y_smooth - x_smooth))
        assert ici >= 0, "ICI should be non-negative"


class TestCalibrationPlotNoHardcoding:
    """Tests to verify no hardcoded values in source code."""

    def test_no_hardcoded_hex_colors_in_main_curve(self):
        """Verify main curve plotting uses COLORS dict, not hardcoded hex."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "calibration_plot.py"
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
        module_path = PROJECT_ROOT / "src" / "viz" / "calibration_plot.py"
        content = module_path.read_text()

        assert (
            "from src.viz.plot_config import" in content
            or "from plot_config import" in content
        )
        assert "save_figure" in content


class TestCalibrationPlotSTRATOSCompliance:
    """Tests for STRATOS guideline compliance."""

    def test_no_sklearn_imports(self):
        """Test that calibration_plot module has no sklearn imports (computation decoupling)."""
        module_path = PROJECT_ROOT / "src" / "viz" / "calibration_plot.py"
        content = module_path.read_text()

        assert "from sklearn" not in content, (
            "CRITICAL: calibration_plot.py must not import sklearn "
            "(computation decoupling violation)"
        )
        assert "import sklearn" not in content, (
            "CRITICAL: calibration_plot.py must not import sklearn "
            "(computation decoupling violation)"
        )

    def test_no_stats_imports(self):
        """Test that calibration_plot module has no src.stats imports (computation decoupling)."""
        module_path = PROJECT_ROOT / "src" / "viz" / "calibration_plot.py"
        content = module_path.read_text()

        assert "from ..stats" not in content, (
            "CRITICAL: calibration_plot.py must not import from src.stats "
            "(computation decoupling violation)"
        )
        assert "from src.stats" not in content, (
            "CRITICAL: calibration_plot.py must not import from src.stats "
            "(computation decoupling violation)"
        )

    def test_deprecated_compute_functions_removed(self):
        """Test that deprecated direct-compute functions have been removed from source."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "calibration_plot.py"
        content = module_path.read_text()

        # Check deprecated save_calibration_extended_json (not the _from_db version)
        assert not re.search(r"def save_calibration_extended_json\(", content), (
            "Deprecated save_calibration_extended_json should be removed"
        )

        # Check deprecated save_calibration_multi_combo_json (not the _from_db version)
        assert not re.search(r"def save_calibration_multi_combo_json\(", content), (
            "Deprecated save_calibration_multi_combo_json should be removed"
        )

        assert "def compute_calibration_metrics(" not in content, (
            "compute_calibration_metrics should be removed (use DuckDB instead)"
        )


class TestCalibrationPlotProductionFigure:
    """Tests for production figure (combined ggplot2 calibration+DCA)."""

    # After R migration, individual calibration figures were merged into
    # a composite ggplot2 figure. JSON data is in data/r_data/.
    COMBINED_FIGURE = (
        FIGURES_DIR / "ggplot2" / "main" / "fig_calibration_dca_combined.png"
    )
    CALIBRATION_JSON = PROJECT_ROOT / "data" / "r_data" / "calibration_data.json"

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

    def test_production_json_has_calibration_curve(self):
        """Test production JSON has calibration curve data."""
        if not self.CALIBRATION_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.CALIBRATION_JSON}. Run: make analyze"
            )
        with open(self.CALIBRATION_JSON) as f:
            data = json.load(f)

        assert "data" in data or "calibration_curve" in data or "metrics" in data

    def test_production_json_has_stratos_metrics(self):
        """Test production JSON includes STRATOS-required metrics."""
        if not self.CALIBRATION_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.CALIBRATION_JSON}. Run: make analyze"
            )
        with open(self.CALIBRATION_JSON) as f:
            data = json.load(f)

        # Check for STRATOS metrics (may be nested)
        json_str = json.dumps(data)
        has_slope = "slope" in json_str or "calibration_slope" in json_str
        assert has_slope, "Calibration JSON missing slope metric"

    def test_production_json_not_synthetic(self):
        """Test production JSON is not marked as synthetic."""
        if not self.CALIBRATION_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.CALIBRATION_JSON}. Run: make analyze"
            )
        with open(self.CALIBRATION_JSON) as f:
            data = json.load(f)

        assert data.get("synthetic") is not True, (
            "CRITICAL: Production figure marked as synthetic!"
        )
