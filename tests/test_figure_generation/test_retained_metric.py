"""Tests for retained metric (retention/risk-coverage curves) figure generation.

Tests the retention curve visualization pipeline after CRITICAL-FAILURE-003
refactoring (computation decoupling):
1. Module can be imported
2. Core functions exist and are callable
3. No sklearn imports (computation decoupling enforced)
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


class TestRetainedMetricModuleStructure:
    """Tests for retained_metric module structure and imports."""

    def test_module_importable(self):
        """Test that the retained_metric module can be imported."""
        from src.viz import retained_metric

        assert retained_metric is not None

    def test_has_db_loading_functions(self):
        """Test that DB loading functions exist."""
        from src.viz.retained_metric import (
            load_retention_curve_from_db,
            load_all_retention_curves_from_db,
        )

        assert callable(load_retention_curve_from_db)
        assert callable(load_all_retention_curves_from_db)

    def test_has_plotting_functions(self):
        """Test that plotting functions exist."""
        from src.viz.retained_metric import (
            plot_retention_curve,
            plot_multi_metric_retention,
            plot_multi_model_retention,
        )

        assert callable(plot_retention_curve)
        assert callable(plot_multi_metric_retention)
        assert callable(plot_multi_model_retention)

    def test_has_figure_generation_functions(self):
        """Test that figure generation functions exist."""
        from src.viz.retained_metric import (
            generate_retention_figures,
            generate_multi_combo_retention_figure,
        )

        assert callable(generate_retention_figures)
        assert callable(generate_multi_combo_retention_figure)

    def test_has_config_helpers(self):
        """Test that config helper functions exist."""
        from src.viz.retained_metric import (
            get_metric_combo,
            get_metric_label,
            load_combos_from_yaml,
        )

        assert callable(get_metric_combo)
        assert callable(get_metric_label)
        assert callable(load_combos_from_yaml)

    def test_has_metric_labels(self):
        """Test that METRIC_LABELS display dict is exported."""
        from src.viz.retained_metric import METRIC_LABELS

        assert isinstance(METRIC_LABELS, dict)
        assert "auroc" in METRIC_LABELS
        assert "brier" in METRIC_LABELS
        assert "net_benefit" in METRIC_LABELS

    def test_no_metric_registry(self):
        """Test that METRIC_REGISTRY (compute function mapping) is removed."""
        from src.viz import retained_metric

        assert not hasattr(retained_metric, "METRIC_REGISTRY"), (
            "METRIC_REGISTRY should be removed - compute functions belong in extraction."
        )

    def test_no_sklearn_imports(self):
        """Test that no sklearn imports exist (CRITICAL-FAILURE-003)."""
        module_path = PROJECT_ROOT / "src" / "viz" / "retained_metric.py"
        content = module_path.read_text()

        assert "sklearn" not in content, (
            "CRITICAL-FAILURE-003: Found sklearn import in viz module."
        )

    def test_no_compute_functions(self):
        """Test that on-the-fly compute functions are removed."""
        from src.viz import retained_metric

        # These should NOT exist
        assert not hasattr(retained_metric, "compute_metric_at_retention")
        assert not hasattr(retained_metric, "compute_retention_curve")
        assert not hasattr(retained_metric, "compute_aurc")
        assert not hasattr(retained_metric, "metric_auroc")
        assert not hasattr(retained_metric, "metric_brier")
        assert not hasattr(retained_metric, "metric_scaled_brier")
        assert not hasattr(retained_metric, "metric_net_benefit")


class TestRetainedMetricNoHardcoding:
    """Tests to verify no hardcoded values in source code."""

    def test_no_hardcoded_hex_colors_in_plotting(self):
        """Verify plotting code uses COLORS dict, not hardcoded hex."""
        import re

        module_path = PROJECT_ROOT / "src" / "viz" / "retained_metric.py"
        content = module_path.read_text()

        problematic_patterns = []
        for i, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "COLORS[" in line or "COLORS.get(" in line:
                continue
            if re.search(r'color\s*=\s*["\']#[0-9A-Fa-f]{6}["\']', line):
                problematic_patterns.append((i, stripped))

        assert len(problematic_patterns) == 0, (
            "Found hardcoded hex colors:\n"
            + "\n".join(f"  Line {ln}: {code}" for ln, code in problematic_patterns)
        )

    def test_imports_save_figure_from_plot_config(self):
        """Verify save_figure is imported from plot_config."""
        module_path = PROJECT_ROOT / "src" / "viz" / "retained_metric.py"
        content = module_path.read_text()

        assert (
            "from src.viz.plot_config import" in content
            or "from plot_config import" in content
        )
        assert "save_figure" in content


class TestRetainedMetricProductionFigure:
    """Tests for production figure (ggplot2 multi-metric raincloud)."""

    # After R migration, the retained_metric Python figure was replaced
    # by the ggplot2 multi-metric raincloud figure.
    COMBINED_FIGURE = (
        FIGURES_DIR / "ggplot2" / "main" / "fig_multi_metric_raincloud.png"
    )
    METRICS_JSON = PROJECT_ROOT / "data" / "r_data" / "catboost_metrics.json"

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

    def test_production_json_has_required_fields(self):
        """Test production JSON has required structure."""
        if not self.METRICS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.METRICS_JSON}. Run: make analyze"
            )
        with open(self.METRICS_JSON) as f:
            data = json.load(f)

        json_str = json.dumps(data)
        assert "auroc" in json_str.lower() or "metric" in json_str.lower(), (
            "Metrics JSON missing metric data"
        )

    def test_production_json_not_synthetic(self):
        """Test production JSON is not marked as synthetic."""
        if not self.METRICS_JSON.exists():
            pytest.skip(
                f"Production data not found: {self.METRICS_JSON}. Run: make analyze"
            )
        with open(self.METRICS_JSON) as f:
            data = json.load(f)

        assert data.get("synthetic") is not True, (
            "CRITICAL: Production figure marked as synthetic! "
            "Using fake data in publication is FORBIDDEN."
        )
