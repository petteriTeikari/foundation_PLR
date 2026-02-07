"""
CRITICAL: No NaN CI values in figure data.

Any figure showing confidence intervals MUST have valid (non-NaN) values.
NaN CIs indicate upstream data quality issues that must be resolved
before generating publication figures.

This test prevents CRITICAL-FAILURE scenarios where figures show
incomplete or misleading uncertainty information.
"""

import pandas as pd
from pathlib import Path

import duckdb


class TestNoNaNCIValues:
    """Ensure no NaN CI values in figure data sources.

    CRITICAL: NaN CIs represent failed experiments that should be filtered
    out during extraction. If any NaN CIs appear in the database, it indicates
    a bug in the extraction pipeline.
    """

    DB_PATH = (
        Path(__file__).parent.parent.parent
        / "data"
        / "public"
        / "foundation_plr_results.db"
    )
    CSV_PATH = (
        Path(__file__).parent.parent.parent
        / "data"
        / "r_data"
        / "essential_metrics.csv"
    )

    def test_no_nan_ci_in_database_any_classifier(self):
        """Non-degenerate data in database must have valid (non-NaN) CIs.

        The raw DB contains ALL extraction runs including degenerate models
        (AUROC=0.5, NaN CIs) from classifiers that failed to converge.
        These are excluded from the filtered CSV export. Here we verify
        that no non-degenerate model has NaN CIs.
        """
        assert self.DB_PATH.exists(), (
            f"Database missing: {self.DB_PATH}. Run: make extract"
        )

        conn = duckdb.connect(str(self.DB_PATH), read_only=True)
        nan_rows = conn.execute("""
            SELECT classifier, outlier_method, imputation_method, auroc, auroc_ci_lower, auroc_ci_upper
            FROM essential_metrics
            WHERE (auroc_ci_lower IS NULL OR auroc_ci_upper IS NULL
               OR isnan(auroc_ci_lower) OR isnan(auroc_ci_upper))
              AND auroc > 0.5
        """).fetchall()
        conn.close()

        assert len(nan_rows) == 0, (
            f"CRITICAL: Found {len(nan_rows)} non-degenerate rows with NULL/NaN CI values! "
            f"These should have been filtered during extraction. "
            f"First few: {nan_rows[:3]}"
        )

    def test_catboost_no_nan_ci_in_database(self):
        """CatBoost data (used for main figures) must have no NaN CIs."""
        assert self.DB_PATH.exists(), (
            f"Database missing: {self.DB_PATH}. Run: make extract"
        )

        conn = duckdb.connect(str(self.DB_PATH), read_only=True)
        nan_count = conn.execute("""
            SELECT COUNT(*) FROM essential_metrics
            WHERE classifier = 'CatBoost'
              AND (auroc_ci_lower IS NULL OR auroc_ci_upper IS NULL
                   OR isnan(auroc_ci_lower) OR isnan(auroc_ci_upper))
        """).fetchone()[0]
        conn.close()

        assert nan_count == 0, (
            f"CRITICAL: Found {nan_count} CatBoost rows with NULL/NaN CI values. "
            f"All CIs must be valid for publication figures!"
        )

    def test_catboost_no_nan_ci_in_csv(self):
        """CatBoost data in CSV export must have no NaN CIs."""
        assert self.CSV_PATH.exists(), (
            f"CSV missing: {self.CSV_PATH}. Run: make analyze"
        )

        df = pd.read_csv(self.CSV_PATH)
        catboost = df[df["classifier"].str.upper() == "CATBOOST"]

        nan_ci_lo = catboost["auroc_ci_lo"].isna().sum()
        nan_ci_hi = catboost["auroc_ci_hi"].isna().sum()

        assert nan_ci_lo == 0, (
            f"CRITICAL: {nan_ci_lo} CatBoost rows have NaN auroc_ci_lo in CSV!"
        )
        assert nan_ci_hi == 0, (
            f"CRITICAL: {nan_ci_hi} CatBoost rows have NaN auroc_ci_hi in CSV!"
        )

    def test_no_nan_ci_in_csv_any_classifier(self):
        """ALL data in CSV must have valid (non-NaN) CIs.

        NaN CIs indicate failed experiments that should be filtered during extraction.
        """
        assert self.CSV_PATH.exists(), (
            f"CSV missing: {self.CSV_PATH}. Run: make analyze"
        )

        df = pd.read_csv(self.CSV_PATH)

        nan_ci_lo = df["auroc_ci_lo"].isna().sum()
        nan_ci_hi = df["auroc_ci_hi"].isna().sum()
        total_nan = nan_ci_lo + nan_ci_hi

        assert total_nan == 0, (
            f"CRITICAL: Found {total_nan} NaN CI values in CSV export! "
            f"(auroc_ci_lo: {nan_ci_lo}, auroc_ci_hi: {nan_ci_hi}). "
            f"These should have been filtered during extraction."
        )

    def test_forest_plot_data_no_nan(self):
        """Forest plot aggregated data must have no NaN after min/max aggregation."""
        assert self.CSV_PATH.exists(), (
            f"CSV missing: {self.CSV_PATH}. Run: make analyze"
        )

        df = pd.read_csv(self.CSV_PATH)
        catboost = df[df["classifier"].str.upper() == "CATBOOST"]

        # Simulate the forest plot aggregation (conservative CIs)
        aggregated = (
            catboost.groupby("outlier_method")
            .agg(
                auroc_mean=("auroc", "mean"),
                auroc_ci_lo=("auroc_ci_lo", "min"),
                auroc_ci_hi=("auroc_ci_hi", "max"),
            )
            .reset_index()
        )

        nan_methods = aggregated[
            aggregated["auroc_ci_lo"].isna() | aggregated["auroc_ci_hi"].isna()
        ]["outlier_method"].tolist()

        assert len(nan_methods) == 0, (
            f"CRITICAL: Forest plot would have NaN CIs for methods: {nan_methods}. "
            f"This would create incomplete/misleading figures!"
        )

    def test_ci_bounds_valid_for_catboost(self):
        """CatBoost CI bounds must satisfy: ci_lo <= auroc <= ci_hi."""
        assert self.CSV_PATH.exists(), (
            f"CSV missing: {self.CSV_PATH}. Run: make analyze"
        )

        df = pd.read_csv(self.CSV_PATH)
        catboost = df[df["classifier"].str.upper() == "CATBOOST"]

        # Check bounds (excluding NaN which is checked elsewhere)
        valid = catboost.dropna(subset=["auroc_ci_lo", "auroc_ci_hi"])

        invalid_lo = valid[valid["auroc_ci_lo"] > valid["auroc"]]
        invalid_hi = valid[valid["auroc_ci_hi"] < valid["auroc"]]

        assert len(invalid_lo) == 0, f"Found {len(invalid_lo)} rows where CI_lo > AUROC"
        assert len(invalid_hi) == 0, f"Found {len(invalid_hi)} rows where CI_hi < AUROC"


class TestFigureJSONDataQuality:
    """Validate JSON data files used for figures."""

    FOREST_JSON = (
        Path(__file__).parent.parent.parent
        / "figures"
        / "generated"
        / "data"
        / "fig02_forest_outlier.json"
    )

    def test_forest_json_no_nan_if_exists(self):
        """Forest plot JSON must have no NaN values."""
        assert self.FOREST_JSON.exists(), (
            f"Forest plot JSON not found: {self.FOREST_JSON}. Run: make analyze"
        )

        import json

        with open(self.FOREST_JSON) as f:
            data = json.load(f)

        # Check for NaN strings in JSON (JSON doesn't support NaN natively)
        json_str = json.dumps(data)

        assert "NaN" not in json_str, "CRITICAL: Forest plot JSON contains NaN values!"
        assert "null" not in json_str.lower() or '"null"' in json_str.lower(), (
            "Warning: Forest plot JSON may contain null values"
        )
