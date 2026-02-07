"""Unit tests for retained_metric visualization module.

Tests the refactored module that reads pre-computed retention curves from
DuckDB and plots them. All metric computation has been moved to extraction
per CRITICAL-FAILURE-003 (computation decoupling).

Tests cover:
- DB loading functions (with mock DuckDB)
- Plotting functions (accept pre-computed arrays)
- Config helpers
- Edge cases
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt


class TestLoadRetentionCurveFromDB:
    """Tests for loading pre-computed retention curves from DuckDB."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a temporary DuckDB with retention_metrics table."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("""
            CREATE TABLE retention_metrics (
                config_id VARCHAR,
                retention_rate DOUBLE,
                metric_name VARCHAR,
                metric_value DOUBLE
            )
        """)

        # Insert test data
        rates = np.linspace(0.1, 1.0, 10)
        for rate in rates:
            # AUROC improves at lower retention
            auroc_val = 0.85 + 0.1 * (1 - rate)
            conn.execute(
                "INSERT INTO retention_metrics VALUES (?, ?, ?, ?)",
                ["gt__pupil-gt", rate, "auroc", auroc_val],
            )
            # Scaled Brier
            brier_val = 0.3 + 0.2 * (1 - rate)
            conn.execute(
                "INSERT INTO retention_metrics VALUES (?, ?, ?, ?)",
                ["gt__pupil-gt", rate, "scaled_brier", brier_val],
            )
            # Second config
            conn.execute(
                "INSERT INTO retention_metrics VALUES (?, ?, ?, ?)",
                ["LOF__SAITS", rate, "auroc", auroc_val - 0.05],
            )

        conn.close()
        return db_path

    def test_load_returns_arrays(self, mock_db):
        """Test that loading returns numpy arrays."""
        from src.viz.retained_metric import load_retention_curve_from_db

        rates, values = load_retention_curve_from_db(
            "gt__pupil-gt", "auroc", db_path=mock_db
        )

        assert isinstance(rates, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(rates) == len(values)
        assert len(rates) == 10

    def test_load_rates_sorted_ascending(self, mock_db):
        """Test that retention rates are sorted ascending."""
        from src.viz.retained_metric import load_retention_curve_from_db

        rates, _ = load_retention_curve_from_db(
            "gt__pupil-gt", "auroc", db_path=mock_db
        )

        assert np.all(np.diff(rates) >= 0), "Retention rates should be ascending"

    def test_load_raises_on_missing_config(self, mock_db):
        """Test that missing config_id raises ValueError."""
        from src.viz.retained_metric import load_retention_curve_from_db

        with pytest.raises(ValueError, match="No retention data found"):
            load_retention_curve_from_db(
                "nonexistent__config", "auroc", db_path=mock_db
            )

    def test_load_raises_on_missing_metric(self, mock_db):
        """Test that missing metric_name raises ValueError."""
        from src.viz.retained_metric import load_retention_curve_from_db

        with pytest.raises(ValueError, match="No retention data found"):
            load_retention_curve_from_db(
                "gt__pupil-gt", "nonexistent_metric", db_path=mock_db
            )

    def test_load_different_metrics(self, mock_db):
        """Test loading different metrics for same config."""
        from src.viz.retained_metric import load_retention_curve_from_db

        _, auroc_vals = load_retention_curve_from_db(
            "gt__pupil-gt", "auroc", db_path=mock_db
        )
        _, brier_vals = load_retention_curve_from_db(
            "gt__pupil-gt", "scaled_brier", db_path=mock_db
        )

        # Values should differ between metrics
        assert not np.allclose(auroc_vals, brier_vals)


class TestLoadAllRetentionCurvesFromDB:
    """Tests for loading all retention curves for a metric."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a temporary DuckDB with multiple configs."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("""
            CREATE TABLE retention_metrics (
                config_id VARCHAR,
                retention_rate DOUBLE,
                metric_name VARCHAR,
                metric_value DOUBLE
            )
        """)

        for config_id in ["config_a", "config_b", "config_c"]:
            for rate in [0.2, 0.5, 0.8, 1.0]:
                conn.execute(
                    "INSERT INTO retention_metrics VALUES (?, ?, ?, ?)",
                    [config_id, rate, "auroc", 0.8 + np.random.random() * 0.1],
                )
        conn.close()
        return db_path

    def test_load_all_returns_dict(self, mock_db):
        """Test that load_all returns a dict of config_id -> (rates, values)."""
        from src.viz.retained_metric import load_all_retention_curves_from_db

        result = load_all_retention_curves_from_db("auroc", db_path=mock_db)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert "config_a" in result
        assert "config_b" in result
        assert "config_c" in result

    def test_load_all_values_are_array_tuples(self, mock_db):
        """Test that each value is a tuple of (rates, values) arrays."""
        from src.viz.retained_metric import load_all_retention_curves_from_db

        result = load_all_retention_curves_from_db("auroc", db_path=mock_db)

        for config_id, (rates, values) in result.items():
            assert isinstance(rates, np.ndarray)
            assert isinstance(values, np.ndarray)
            assert len(rates) == 4  # 4 rates per config

    def test_load_all_empty_for_unknown_metric(self, mock_db):
        """Test that unknown metric returns empty dict."""
        from src.viz.retained_metric import load_all_retention_curves_from_db

        result = load_all_retention_curves_from_db(
            "nonexistent_metric", db_path=mock_db
        )

        assert result == {}


class TestRetentionCurvePlotting:
    """Tests for plotting functions with pre-computed data."""

    @pytest.fixture
    def sample_curve(self):
        """Generate sample pre-computed retention curve data."""
        rates = np.linspace(0.1, 1.0, 50)
        # AUROC improves at lower retention
        values = (
            0.85 + 0.1 * (1 - rates) + np.random.default_rng(42).normal(0, 0.01, 50)
        )
        return rates, values

    def test_plot_retention_curve_returns_figure(self, sample_curve):
        """Test that plot function returns valid figure."""
        from src.viz.retained_metric import plot_retention_curve

        rates, values = sample_curve
        fig, ax = plot_retention_curve(rates, values, metric_name="auroc")

        assert isinstance(fig, plt.Figure)
        assert ax is not None
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_curve, tmp_path):
        """Test that figure can be saved to file."""
        from src.viz.retained_metric import plot_retention_curve

        rates, values = sample_curve
        fig, ax = plot_retention_curve(rates, values)

        output_path = tmp_path / "test_retained_curve.pdf"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_plot_with_existing_axes(self, sample_curve):
        """Test plotting to existing axes."""
        from src.viz.retained_metric import plot_retention_curve

        rates, values = sample_curve
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_retention_curve(rates, values, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_plot_with_label_and_color(self, sample_curve):
        """Test plot with custom label and color."""
        from src.viz.retained_metric import plot_retention_curve

        rates, values = sample_curve
        fig, ax = plot_retention_curve(
            rates, values, label="Ground Truth", color="blue"
        )

        # Check legend was created
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_plot_without_baseline(self, sample_curve):
        """Test plot with baseline disabled."""
        from src.viz.retained_metric import plot_retention_curve

        rates, values = sample_curve
        fig, ax = plot_retention_curve(rates, values, show_baseline=False)

        # Should have only 1 line (no baseline)
        assert len(ax.get_lines()) == 1
        plt.close(fig)


class TestMultiMetricRetention:
    """Tests for multi-metric subplot grid."""

    def test_plot_multi_metric(self):
        """Test multi-metric subplot grid."""
        from src.viz.retained_metric import plot_multi_metric_retention

        rates = np.linspace(0.1, 1.0, 20)
        retention_data = {
            "auroc": (rates, 0.85 + 0.1 * (1 - rates)),
            "scaled_brier": (rates, 0.3 + 0.2 * (1 - rates)),
        }

        fig, axes = plot_multi_metric_retention(retention_data)

        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_multi_metric_single(self):
        """Test multi-metric with single metric."""
        from src.viz.retained_metric import plot_multi_metric_retention

        rates = np.linspace(0.1, 1.0, 20)
        retention_data = {"auroc": (rates, 0.85 + 0.1 * (1 - rates))}

        fig, axes = plot_multi_metric_retention(retention_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMultiModelRetention:
    """Tests for multi-model overlay plot."""

    def test_plot_multi_model(self):
        """Test multi-model overlay plot."""
        from src.viz.retained_metric import plot_multi_model_retention

        rates = np.linspace(0.1, 1.0, 20)
        data = {
            "Model A": {
                "retention_rates": rates,
                "metric_values": 0.85 + 0.1 * (1 - rates),
            },
            "Model B": {
                "retention_rates": rates,
                "metric_values": 0.80 + 0.1 * (1 - rates),
            },
        }

        fig, ax = plot_multi_model_retention(data, metric_name="auroc")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestConfigHelpers:
    """Tests for config helper functions."""

    def test_metric_labels_dict_exists(self):
        """Test that METRIC_LABELS dict is exported."""
        from src.viz.retained_metric import METRIC_LABELS

        assert isinstance(METRIC_LABELS, dict)
        assert "auroc" in METRIC_LABELS
        assert "brier" in METRIC_LABELS
        assert "net_benefit" in METRIC_LABELS

    def test_get_metric_label_returns_string(self):
        """Test that get_metric_label returns a display string."""
        from src.viz.retained_metric import get_metric_label

        label = get_metric_label("auroc")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_load_combos_from_yaml(self):
        """Test that combos can be loaded from YAML."""
        from src.viz.retained_metric import load_combos_from_yaml

        combos = load_combos_from_yaml("standard")
        assert isinstance(combos, list)
        assert len(combos) > 0

    def test_load_combos_invalid_set_raises(self):
        """Test that invalid combo set raises ValueError."""
        from src.viz.retained_metric import load_combos_from_yaml

        with pytest.raises(ValueError, match="Unknown combo_set"):
            load_combos_from_yaml("nonexistent")


class TestNoComputationViolations:
    """Tests to verify the module has zero computation decoupling violations."""

    def test_no_sklearn_imports(self):
        """Test that no sklearn imports exist in the module."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "retained_metric.py"
        )
        content = module_path.read_text()

        assert "sklearn" not in content, (
            "CRITICAL-FAILURE-003: Found sklearn import in viz module. "
            "All metric computation must happen in extraction."
        )

    def test_no_metric_compute_functions(self):
        """Test that no metric_* compute functions exist."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "retained_metric.py"
        )
        content = module_path.read_text()

        # These functions should NOT exist
        banned_functions = [
            "def metric_auroc",
            "def metric_brier",
            "def metric_scaled_brier",
            "def metric_net_benefit",
            "def metric_f1",
            "def metric_accuracy",
            "def metric_sensitivity",
            "def metric_specificity",
            "def compute_metric_at_retention",
            "def compute_retention_curve",
            "def compute_aurc",
        ]

        for func in banned_functions:
            assert func not in content, (
                f"Found banned compute function '{func}' in retained_metric.py. "
                "Metric computation belongs in extraction, not visualization."
            )

    def test_no_metric_registry_dict(self):
        """Test that the METRIC_REGISTRY function-mapping dict is removed."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "retained_metric.py"
        )
        content = module_path.read_text()

        assert "METRIC_REGISTRY" not in content, (
            "METRIC_REGISTRY (function mapping) should be removed. "
            "Use METRIC_LABELS for display labels only."
        )

    def test_no_main_function(self):
        """Test that the main() function with mock data is removed."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "retained_metric.py"
        )
        content = module_path.read_text()

        assert (
            "def main(" not in content
        ), "main() function with mock data generation should be removed."
        assert "__main__" not in content, "__main__ block should be removed."

    def test_no_hardcoded_hex_colors(self):
        """Verify plotting code uses COLORS dict, not hardcoded hex."""
        import re
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "retained_metric.py"
        )
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
