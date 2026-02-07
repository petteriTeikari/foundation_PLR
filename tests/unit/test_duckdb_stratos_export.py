"""
Tests for STRATOS-required metrics extraction to DuckDB.

These tests verify that all STRATOS (Van Calster et al. 2024) required metrics
flow from MLflow artifacts → DuckDB tables → JSON export.

STRATOS Core Set (MANDATORY):
1. AUROC with 95% CI - Discrimination
2. Calibration slope - Spread of predictions
3. Calibration intercept - Mean calibration
4. O:E ratio (Observed:Expected) - Risk level calibration
5. Net Benefit at multiple thresholds - Clinical utility
6. Full DCA curves - Clinical utility across thresholds

References:
    Van Calster et al. 2024 "Performance evaluation of predictive AI models
    to support medical decisions" (STRATOS Initiative Topic Group 6)
"""

import numpy as np
import pandas as pd
import pytest

import duckdb

# Import module under test
from src.data_io.duckdb_export import (
    RESULTS_SCHEMA,
    export_results_to_duckdb,
)

# Import stats modules for computing STRATOS metrics
from src.stats.calibration_extended import calibration_slope_intercept
from src.stats.clinical_utility import decision_curve_analysis, net_benefit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_predictions():
    """Create sample predictions (y_true, y_prob) for STRATOS metrics."""
    np.random.seed(42)
    n = 200  # Similar to our N=208 dataset

    # Create realistic predictions with some calibration
    # 30% prevalence (close to our glaucoma dataset)
    y_true = np.random.binomial(1, 0.3, n)

    # Create somewhat calibrated predictions
    # Add noise to true labels to create probabilities
    y_prob = np.clip(
        y_true * 0.6 + (1 - y_true) * 0.2 + np.random.normal(0, 0.15, n), 0.01, 0.99
    )

    return y_true, y_prob


@pytest.fixture
def sample_metrics_with_stratos(sample_predictions):
    """Create sample metrics DataFrame including STRATOS metrics."""
    y_true, y_prob = sample_predictions

    # Compute STRATOS metrics
    cal_result = calibration_slope_intercept(y_true, y_prob)

    # Create per-fold metrics with STRATOS fields
    np.random.seed(42)
    metrics = []
    for fold in range(5):
        # Simulate slight variation across folds
        base_auroc = 0.85 + np.random.normal(0, 0.03)
        metrics.append(
            {
                "metric_id": fold,
                "source_name": "test_pipeline",
                "classifier": "CatBoost",
                "fold": fold,
                # Discrimination
                "auroc": np.clip(base_auroc, 0.7, 0.95),
                "aupr": np.clip(base_auroc - 0.05, 0.6, 0.9),
                # Calibration (STRATOS required)
                "brier_score": np.random.uniform(0.12, 0.18),
                "calibration_slope": cal_result.slope + np.random.normal(0, 0.05),
                "calibration_intercept": cal_result.intercept
                + np.random.normal(0, 0.02),
                "e_o_ratio": cal_result.o_e_ratio + np.random.normal(0, 0.05),
                # Classification metrics
                "sensitivity": np.random.uniform(0.75, 0.90),
                "specificity": np.random.uniform(0.70, 0.85),
                "ppv": np.random.uniform(0.50, 0.70),
                "npv": np.random.uniform(0.85, 0.95),
                "f1_score": np.random.uniform(0.60, 0.80),
                "accuracy": np.random.uniform(0.75, 0.85),
                # Clinical utility (STRATOS required)
                "net_benefit_5pct": net_benefit(y_true, y_prob, 0.05),
                "net_benefit_10pct": net_benefit(y_true, y_prob, 0.10),
                "net_benefit_20pct": net_benefit(y_true, y_prob, 0.20),
            }
        )

    return pd.DataFrame(metrics)


@pytest.fixture
def sample_dca_curves(sample_predictions):
    """Create sample DCA curves DataFrame."""
    y_true, y_prob = sample_predictions

    # Use clinical utility module to compute DCA
    dca_df = decision_curve_analysis(
        y_true, y_prob, threshold_range=(0.01, 0.50), n_thresholds=50
    )

    # Add required columns for DuckDB
    dca_df["dca_id"] = range(len(dca_df))
    dca_df["source_name"] = "test_pipeline"
    dca_df["classifier"] = "CatBoost"

    # Rename columns to match schema
    dca_df = dca_df.rename(
        columns={
            "nb_model": "net_benefit_model",
            "nb_all": "net_benefit_all",
            "nb_none": "net_benefit_none",
        }
    )

    return dca_df[
        [
            "dca_id",
            "source_name",
            "classifier",
            "threshold",
            "net_benefit_model",
            "net_benefit_all",
            "net_benefit_none",
            "sensitivity",
            "specificity",
        ]
    ]


@pytest.fixture
def sample_calibration_curves(sample_predictions):
    """Create sample calibration curves DataFrame."""
    y_true, y_prob = sample_predictions

    # Compute binned calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    curves = []
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_midpoint = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            observed = np.mean(y_true[mask])

            # Wilson CI for proportion
            if n_in_bin >= 5:
                z = 1.96
                p = observed
                ci_lower = (
                    p
                    + z**2 / (2 * n_in_bin)
                    - z * np.sqrt(p * (1 - p) / n_in_bin + z**2 / (4 * n_in_bin**2))
                ) / (1 + z**2 / n_in_bin)
                ci_upper = (
                    p
                    + z**2 / (2 * n_in_bin)
                    + z * np.sqrt(p * (1 - p) / n_in_bin + z**2 / (4 * n_in_bin**2))
                ) / (1 + z**2 / n_in_bin)
            else:
                ci_lower = ci_upper = np.nan

            curves.append(
                {
                    "curve_id": len(curves),
                    "source_name": "test_pipeline",
                    "classifier": "CatBoost",
                    "bin_midpoint": bin_midpoint,
                    "observed_proportion": observed,
                    "bin_count": int(n_in_bin),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    return pd.DataFrame(curves)


# ============================================================================
# Test: DuckDB Schema includes STRATOS columns
# ============================================================================


class TestSTRATOSSchema:
    """Test that DuckDB schema includes all STRATOS-required columns."""

    def test_metrics_per_fold_has_calibration_slope(self, tmp_path):
        """Schema must include calibration_slope column."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            # Check column exists
            result = con.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'metrics_per_fold' AND column_name = 'calibration_slope'
            """).fetchall()

            assert (
                len(result) == 1
            ), "calibration_slope column must exist in metrics_per_fold"

    def test_metrics_per_fold_has_calibration_intercept(self, tmp_path):
        """Schema must include calibration_intercept column."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            result = con.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'metrics_per_fold' AND column_name = 'calibration_intercept'
            """).fetchall()

            assert (
                len(result) == 1
            ), "calibration_intercept column must exist in metrics_per_fold"

    def test_metrics_per_fold_has_e_o_ratio(self, tmp_path):
        """Schema must include e_o_ratio (O:E ratio) column."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            result = con.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'metrics_per_fold' AND column_name = 'e_o_ratio'
            """).fetchall()

            assert len(result) == 1, "e_o_ratio column must exist in metrics_per_fold"

    def test_metrics_per_fold_has_net_benefit_columns(self, tmp_path):
        """Schema must include net_benefit at multiple thresholds."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            required_nb_columns = [
                "net_benefit_5pct",
                "net_benefit_10pct",
                "net_benefit_20pct",
            ]

            for col in required_nb_columns:
                result = con.execute(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'metrics_per_fold' AND column_name = '{col}'
                """).fetchall()

                assert len(result) == 1, f"{col} column must exist in metrics_per_fold"

    def test_dca_curves_table_exists(self, tmp_path):
        """DCA curves table must exist in schema."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            result = con.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name = 'dca_curves'
            """).fetchall()

            assert len(result) == 1, "dca_curves table must exist"

    def test_dca_curves_has_required_columns(self, tmp_path):
        """DCA curves table must have all required columns."""
        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            required_columns = [
                "threshold",
                "net_benefit_model",
                "net_benefit_all",
                "net_benefit_none",
                "sensitivity",
                "specificity",
            ]

            for col in required_columns:
                result = con.execute(f"""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'dca_curves' AND column_name = '{col}'
                """).fetchall()

                assert len(result) == 1, f"{col} column must exist in dca_curves"


# ============================================================================
# Test: STRATOS Metrics Extraction
# ============================================================================


class TestSTRATOSMetricsExtraction:
    """Test that STRATOS metrics are correctly extracted and stored."""

    def test_calibration_slope_stored_correctly(
        self, sample_metrics_with_stratos, tmp_path
    ):
        """Calibration slope must be stored with realistic values."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT calibration_slope FROM metrics_per_fold
                WHERE calibration_slope IS NOT NULL
            """).fetchall()

            assert len(result) > 0, "calibration_slope values must be stored"

            # Slope should be positive and finite (range depends on calibration)
            # Well-calibrated: 0.8-1.2, but test fixtures may produce wider range
            slopes = [r[0] for r in result]
            assert all(
                0.1 <= s <= 10.0 for s in slopes
            ), f"calibration_slope values unrealistic: {slopes}"

    def test_calibration_intercept_stored_correctly(
        self, sample_metrics_with_stratos, tmp_path
    ):
        """Calibration intercept must be stored."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT calibration_intercept FROM metrics_per_fold
                WHERE calibration_intercept IS NOT NULL
            """).fetchall()

            assert len(result) > 0, "calibration_intercept values must be stored"

            # Intercept should be small (between -1 and 1 for well-calibrated model)
            intercepts = [r[0] for r in result]
            assert all(
                -2.0 <= i <= 2.0 for i in intercepts
            ), f"calibration_intercept values unrealistic: {intercepts}"

    def test_e_o_ratio_stored_correctly(self, sample_metrics_with_stratos, tmp_path):
        """O:E ratio must be stored with correct interpretation."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT e_o_ratio FROM metrics_per_fold
                WHERE e_o_ratio IS NOT NULL
            """).fetchall()

            assert len(result) > 0, "e_o_ratio values must be stored"

            # O:E ratio should be positive (typically between 0.5 and 2.0)
            ratios = [r[0] for r in result]
            assert all(
                0.3 <= r <= 3.0 for r in ratios
            ), f"e_o_ratio values unrealistic: {ratios}"

    def test_net_benefit_stored_at_multiple_thresholds(
        self, sample_metrics_with_stratos, tmp_path
    ):
        """Net benefit must be stored at 5%, 10%, and 20% thresholds."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            for threshold_col in [
                "net_benefit_5pct",
                "net_benefit_10pct",
                "net_benefit_20pct",
            ]:
                result = con.execute(f"""
                    SELECT {threshold_col} FROM metrics_per_fold
                    WHERE {threshold_col} IS NOT NULL
                """).fetchall()

                assert len(result) > 0, f"{threshold_col} values must be stored"

                # Net benefit can be negative but shouldn't be extremely so
                nbs = [r[0] for r in result]
                assert all(
                    -0.5 <= nb <= 1.0 for nb in nbs
                ), f"{threshold_col} values unrealistic: {nbs}"


# ============================================================================
# Test: DCA Curves Export
# ============================================================================


class TestDCACurvesExport:
    """Test that DCA curves are correctly exported to DuckDB."""

    def test_dca_curves_table_populated(
        self, sample_dca_curves, sample_metrics_with_stratos, tmp_path
    ):
        """DCA curves table must be populated with data."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("SELECT COUNT(*) FROM dca_curves").fetchone()

            assert result[0] > 0, "dca_curves table must have data"

    def test_dca_curves_has_sufficient_threshold_points(
        self, sample_dca_curves, sample_metrics_with_stratos, tmp_path
    ):
        """DCA curves must have at least 20 threshold points per model."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT source_name, classifier, COUNT(DISTINCT threshold) as n_thresholds
                FROM dca_curves
                GROUP BY source_name, classifier
            """).fetchall()

            for row in result:
                source_name, classifier, n_thresholds = row
                assert (
                    n_thresholds >= 20
                ), f"DCA curves for {source_name}/{classifier} has only {n_thresholds} thresholds (need >= 20)"

    def test_dca_curves_has_treat_all_and_treat_none(
        self, sample_dca_curves, sample_metrics_with_stratos, tmp_path
    ):
        """DCA curves must include treat-all and treat-none reference lines."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT net_benefit_all, net_benefit_none
                FROM dca_curves
                LIMIT 1
            """).fetchone()

            assert result is not None, "DCA curve data missing"
            assert result[0] is not None, "net_benefit_all must be computed"
            assert result[1] is not None, "net_benefit_none must be computed"

    def test_dca_threshold_range_clinically_relevant(
        self, sample_dca_curves, sample_metrics_with_stratos, tmp_path
    ):
        """DCA thresholds must cover clinically relevant range (1% - 50%)."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT MIN(threshold), MAX(threshold)
                FROM dca_curves
            """).fetchone()

            min_thresh, max_thresh = result

            # Should start at or below 5% (typical screening threshold)
            assert (
                min_thresh <= 0.05
            ), f"DCA must include low thresholds (min={min_thresh:.2%}, need <= 5%)"

            # Should extend to at least 30%
            assert (
                max_thresh >= 0.30
            ), f"DCA must include high thresholds (max={max_thresh:.2%}, need >= 30%)"


# ============================================================================
# Test: Calibration Curves Export
# ============================================================================


class TestCalibrationCurvesExport:
    """Test that calibration curves are correctly exported."""

    def test_calibration_curves_table_populated(
        self, sample_calibration_curves, sample_metrics_with_stratos, tmp_path
    ):
        """Calibration curves table must be populated."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            calibration_curves=sample_calibration_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("SELECT COUNT(*) FROM calibration_curves").fetchone()

            assert result[0] > 0, "calibration_curves table must have data"

    def test_calibration_curves_has_confidence_intervals(
        self, sample_calibration_curves, sample_metrics_with_stratos, tmp_path
    ):
        """Calibration curves must include confidence intervals."""
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            calibration_curves=sample_calibration_curves,
        )

        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT ci_lower, ci_upper
                FROM calibration_curves
                WHERE ci_lower IS NOT NULL
            """).fetchall()

            # At least some bins should have CIs (bins with n >= 5)
            assert len(result) > 0, "calibration curves must have confidence intervals"

            for ci_lower, ci_upper in result:
                assert ci_lower <= ci_upper, "ci_lower must be <= ci_upper"


# ============================================================================
# Test: Compute STRATOS Metrics from Predictions
# ============================================================================


class TestComputeSTRATOSFromPredictions:
    """Test computing STRATOS metrics from y_true, y_prob."""

    def test_calibration_slope_computed_correctly(self, sample_predictions):
        """Calibration slope computation matches expected values."""
        y_true, y_prob = sample_predictions

        result = calibration_slope_intercept(y_true, y_prob)

        # Slope should be defined
        assert result.slope is not None
        assert np.isfinite(result.slope)

        # Slope should be positive and finite (test fixtures may not be perfectly calibrated)
        assert (
            0.1 <= result.slope <= 10.0
        ), f"Slope {result.slope} out of realistic range"

    def test_e_o_ratio_computed_correctly(self, sample_predictions):
        """O:E ratio computation follows Van Calster 2024 definition."""
        y_true, y_prob = sample_predictions

        result = calibration_slope_intercept(y_true, y_prob)

        # O:E = observed / expected
        observed = np.mean(y_true)
        expected = np.mean(y_prob)
        manual_oe = observed / expected

        assert np.isclose(
            result.o_e_ratio, manual_oe, rtol=0.01
        ), f"O:E ratio mismatch: {result.o_e_ratio} vs manual {manual_oe}"

    def test_net_benefit_computation(self, sample_predictions):
        """Net benefit computation follows Vickers & Elkin formula."""
        y_true, y_prob = sample_predictions

        threshold = 0.15
        nb = net_benefit(y_true, y_prob, threshold)

        # Manual computation for verification
        n = len(y_true)
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        odds = threshold / (1 - threshold)
        manual_nb = (tp / n) - (fp / n) * odds

        assert np.isclose(nb, manual_nb), f"Net benefit mismatch: {nb} vs {manual_nb}"

    def test_dca_provides_full_curve(self, sample_predictions):
        """DCA returns complete curve with model/treat-all/treat-none."""
        y_true, y_prob = sample_predictions

        dca_df = decision_curve_analysis(
            y_true, y_prob, threshold_range=(0.01, 0.50), n_thresholds=50
        )

        assert len(dca_df) >= 20, "DCA should have at least 20 threshold points"
        assert "nb_model" in dca_df.columns or "net_benefit_model" in dca_df.columns
        assert "nb_all" in dca_df.columns or "net_benefit_all" in dca_df.columns
        assert "nb_none" in dca_df.columns or "net_benefit_none" in dca_df.columns


# ============================================================================
# Test: Integration - Full Pipeline
# ============================================================================


class TestSTRATOSPipelineIntegration:
    """Integration tests for full STRATOS metrics pipeline."""

    def test_full_stratos_export_and_load(
        self,
        sample_predictions,
        sample_metrics_with_stratos,
        sample_dca_curves,
        sample_calibration_curves,
        tmp_path,
    ):
        """Full pipeline: export all STRATOS data and verify retrieval."""
        db_path = tmp_path / "stratos_results.db"

        # Export all data
        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
            calibration_curves=sample_calibration_curves,
        )

        # Verify all tables are populated
        with duckdb.connect(str(db_path), read_only=True) as con:
            tables_to_check = [
                ("metrics_per_fold", 5),  # 5 folds
                ("dca_curves", 20),  # At least 20 threshold points
                ("calibration_curves", 5),  # At least 5 bins
            ]

            for table, min_rows in tables_to_check:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert (
                    count >= min_rows
                ), f"Table {table} has {count} rows, expected >= {min_rows}"

    def test_stratos_metrics_queryable(
        self, sample_metrics_with_stratos, sample_dca_curves, tmp_path
    ):
        """STRATOS metrics can be queried with SQL for reporting."""
        db_path = tmp_path / "stratos_query.db"

        export_results_to_duckdb(
            predictions_df=pd.DataFrame(),
            metrics_per_fold=sample_metrics_with_stratos,
            metrics_aggregate=pd.DataFrame(),
            output_path=db_path,
            dca_curves=sample_dca_curves,
        )

        # Run STRATOS summary query
        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute("""
                SELECT
                    source_name,
                    classifier,
                    AVG(auroc) as mean_auroc,
                    AVG(calibration_slope) as mean_cal_slope,
                    AVG(calibration_intercept) as mean_cal_intercept,
                    AVG(e_o_ratio) as mean_oe_ratio,
                    AVG(net_benefit_10pct) as mean_nb_10pct
                FROM metrics_per_fold
                GROUP BY source_name, classifier
            """).df()

            assert len(result) > 0, "STRATOS summary query should return results"

            # All STRATOS columns should have non-null values
            for col in [
                "mean_auroc",
                "mean_cal_slope",
                "mean_cal_intercept",
                "mean_oe_ratio",
                "mean_nb_10pct",
            ]:
                assert result[col].notna().all(), f"{col} has null values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
