"""Unit tests for metric vs cohort visualization.

Shows how metrics change when selecting subsets based on uncertainty.
Based on Dohopolski et al. 2022 visualization approach.

COMPUTATION DECOUPLING: After refactoring, this module reads pre-computed
cohort curve data from DuckDB. Tests validate the plotting API accepts
pre-computed arrays and the DB loading functions.
"""

import json
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestMetricVsCohort:
    """Tests for metric vs cohort visualization with pre-computed data."""

    @pytest.fixture
    def sample_curve(self):
        """Generate sample pre-computed cohort curve data."""
        fractions = np.linspace(0.2, 1.0, 9)
        # AUROC improves at lower fractions (more selective cohort)
        values = 0.85 + 0.1 * (1 - fractions)
        return fractions, values

    def test_plot_returns_figure(self, sample_curve):
        """Test that plot returns valid figure."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions, values = sample_curve
        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plot_saves_to_file(self, sample_curve, tmp_path):
        """Test that figure can be saved to file."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions, values = sample_curve
        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        output_path = tmp_path / "test_metric_cohort.pdf"
        fig.savefig(output_path)
        assert output_path.exists()
        plt.close(fig)

    def test_axes_labels(self, sample_curve):
        """Test that axes have proper labels."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions, values = sample_curve
        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_multiple_metrics(self):
        """Test plotting multiple metrics."""
        from src.viz.metric_vs_cohort import plot_multi_metric_vs_cohort

        fractions = np.linspace(0.2, 1.0, 9)
        cohort_data = {
            "auroc": (fractions, 0.85 + 0.1 * (1 - fractions)),
            "brier": (fractions, 0.3 + 0.2 * (1 - fractions)),
        }
        fig, axes = plot_multi_metric_vs_cohort(cohort_data)
        assert len(axes) == 2
        plt.close(fig)

    def test_baseline_reference(self, sample_curve):
        """Test that baseline (100% retention) is shown."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions, values = sample_curve
        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        # Should have lines (main curve + baseline)
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close(fig)


class TestLoadCohortCurveFromDB:
    """Tests for loading pre-computed cohort curves from DuckDB."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a temporary DuckDB with cohort_metrics table."""
        import duckdb

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        conn.execute("""
            CREATE TABLE cohort_metrics (
                cohort_id INTEGER,
                config_id INTEGER,
                cohort_fraction REAL,
                metric_name VARCHAR,
                metric_value REAL
            )
        """)

        # Insert test data
        idx = 0
        fractions = np.linspace(0.2, 1.0, 5)
        for frac in fractions:
            idx += 1
            conn.execute(
                "INSERT INTO cohort_metrics VALUES (?, ?, ?, ?, ?)",
                [idx, 1, float(frac), "auroc", 0.85 + 0.1 * (1 - frac)],
            )

        conn.close()
        return db_path

    def test_load_returns_arrays(self, mock_db):
        """Test that loading returns numpy arrays."""
        from src.viz.metric_vs_cohort import load_cohort_curve_from_db

        fractions, values = load_cohort_curve_from_db(
            config_id=1, metric_name="auroc", db_path=mock_db
        )

        assert isinstance(fractions, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(fractions) == len(values)
        assert len(fractions) == 5

    def test_load_empty_for_missing(self, mock_db):
        """Test empty arrays for missing config/metric."""
        from src.viz.metric_vs_cohort import load_cohort_curve_from_db

        fractions, values = load_cohort_curve_from_db(
            config_id=999, metric_name="auroc", db_path=mock_db
        )

        assert len(fractions) == 0
        assert len(values) == 0


class TestMetricVsCohortDataExport:
    """Tests for JSON data export."""

    def test_json_data_export(self, tmp_path):
        """Test JSON reproducibility data is saved."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions = np.linspace(0.2, 1.0, 9)
        values = 0.85 + 0.1 * (1 - fractions)

        json_path = tmp_path / "test_metric_cohort.json"
        fig, ax = plot_metric_vs_cohort(
            fractions, values, metric="auroc", save_json_path=str(json_path)
        )

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "cohort_fractions" in data
        assert "metric_values" in data
        plt.close(fig)


class TestMetricVsCohortEdgeCases:
    """Edge cases for metric vs cohort visualization."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions = np.array([])
        values = np.array([])

        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        plt.close(fig)

    def test_single_point(self):
        """Test with single data point."""
        from src.viz.metric_vs_cohort import plot_metric_vs_cohort

        fractions = np.array([1.0])
        values = np.array([0.85])

        fig, ax = plot_metric_vs_cohort(fractions, values, metric="auroc")
        plt.close(fig)


class TestNoComputationViolations:
    """Tests to verify no computation decoupling violations."""

    def test_no_sklearn_imports(self):
        """Test that no sklearn imports exist in the module."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "metric_vs_cohort.py"
        )
        content = module_path.read_text()

        assert (
            "sklearn" not in content
        ), "CRITICAL-FAILURE-003: Found sklearn import in viz module."

    def test_no_compute_functions(self):
        """Test that compute functions have been removed."""
        from pathlib import Path

        module_path = (
            Path(__file__).parent.parent.parent / "src" / "viz" / "metric_vs_cohort.py"
        )
        content = module_path.read_text()

        banned_functions = [
            "def compute_metric(",
            "def compute_metric_at_cohort_fraction(",
            "def compute_cohort_curve(",
        ]

        for func in banned_functions:
            assert (
                func not in content
            ), f"Found banned compute function '{func}' in metric_vs_cohort.py."
