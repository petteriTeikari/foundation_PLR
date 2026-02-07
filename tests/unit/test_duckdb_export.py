"""
Tests for DuckDB export functionality.

Tests cover:
1. MLflow artifact extraction
2. Memory-efficient data loading
3. DuckDB schema creation and population
4. Data integrity verification
"""

import gc
import pickle

import numpy as np
import pandas as pd
import pytest

# Import module under test
from src.data_io.duckdb_export import (
    FEATURES_SCHEMA,
    RESULTS_SCHEMA,
    concat_dataframes_efficient,
    export_features_to_duckdb,
    export_results_to_duckdb,
    load_features_from_duckdb,
    load_results_from_duckdb,
    load_artifact_safe,
    iter_artifacts_chunked,
    DuckDBAnalysisPipeline,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_features_data():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_subjects = 50

    features = pd.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n_subjects)],
            "eye": np.random.choice(["OD", "OS"], n_subjects),
            "baseline_diameter": np.random.normal(5.0, 0.5, n_subjects),
            "constriction_amplitude": np.random.normal(1.5, 0.3, n_subjects),
            "constriction_amplitude_rel": np.random.normal(0.3, 0.05, n_subjects),
            "max_constriction_diameter": np.random.normal(3.5, 0.4, n_subjects),
            "latency_to_constriction": np.random.normal(0.2, 0.05, n_subjects),
            "latency_75pct": np.random.normal(0.3, 0.05, n_subjects),
            "time_to_redilation": np.random.normal(2.0, 0.3, n_subjects),
            "max_constriction_velocity": np.random.normal(3.0, 0.5, n_subjects),
            "mean_constriction_velocity": np.random.normal(2.0, 0.3, n_subjects),
            "max_redilation_velocity": np.random.normal(1.5, 0.3, n_subjects),
            "pipr_6s": np.random.normal(0.8, 0.1, n_subjects),
            "pipr_10s": np.random.normal(0.75, 0.1, n_subjects),
            "recovery_time": np.random.normal(4.0, 0.5, n_subjects),
            "constriction_duration": np.random.normal(0.8, 0.1, n_subjects),
        }
    )

    metadata = pd.DataFrame(
        {
            "subject_id": features["subject_id"],
            "eye": features["eye"],
            "split": np.random.choice(
                ["train", "val", "test"], n_subjects, p=[0.6, 0.2, 0.2]
            ),
            "source_name": "test_pipeline",
            "has_glaucoma": np.random.randint(0, 2, n_subjects),
        }
    )

    return {"test_pipeline": features}, metadata


@pytest.fixture
def sample_predictions_data():
    """Create sample predictions data for testing."""
    np.random.seed(42)
    n_predictions = 200

    predictions = pd.DataFrame(
        {
            "prediction_id": range(n_predictions),
            "subject_id": [f"S{i % 50:03d}" for i in range(n_predictions)],
            "eye": np.random.choice(["OD", "OS"], n_predictions),
            "fold": np.random.randint(0, 5, n_predictions),
            "bootstrap_iter": np.zeros(n_predictions, dtype=int),
            "outlier_method": np.random.choice(
                ["MOMENT", "LOF", "OneClassSVM"], n_predictions
            ),
            "imputation_method": np.random.choice(
                ["SAITS", "TimesNet", "MOMENT"], n_predictions
            ),
            "featurization": "simple1.0",
            "classifier": np.random.choice(
                ["XGBOOST", "LogisticRegression", "CatBoost"], n_predictions
            ),
            "source_name": "test_pipeline",
            "y_true": np.random.randint(0, 2, n_predictions),
            "y_pred": np.random.randint(0, 2, n_predictions),
            "y_prob": np.random.uniform(0, 1, n_predictions),
            "mlflow_run_id": "test_run_123",
        }
    )

    return predictions


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics per fold and aggregate data."""
    np.random.seed(42)
    classifiers = ["XGBOOST", "LogisticRegression", "CatBoost"]
    folds = list(range(5))

    # Per-fold metrics
    metrics_per_fold = []
    metric_id = 0
    for clf in classifiers:
        for fold in folds:
            metrics_per_fold.append(
                {
                    "metric_id": metric_id,
                    "source_name": "test_pipeline",
                    "classifier": clf,
                    "fold": fold,
                    "auroc": np.random.uniform(0.7, 0.95),
                    "aupr": np.random.uniform(0.6, 0.9),
                    "brier_score": np.random.uniform(0.1, 0.25),
                    "calibration_slope": np.random.uniform(0.8, 1.2),
                    "calibration_intercept": np.random.uniform(-0.1, 0.1),
                    "e_o_ratio": np.random.uniform(0.9, 1.1),
                    "sensitivity": np.random.uniform(0.7, 0.95),
                    "specificity": np.random.uniform(0.7, 0.95),
                    "ppv": np.random.uniform(0.6, 0.9),
                    "npv": np.random.uniform(0.8, 0.95),
                    "f1_score": np.random.uniform(0.7, 0.9),
                    "accuracy": np.random.uniform(0.75, 0.9),
                    "net_benefit_5pct": np.random.uniform(0.01, 0.05),
                    "net_benefit_10pct": np.random.uniform(0.02, 0.08),
                    "net_benefit_20pct": np.random.uniform(0.03, 0.1),
                }
            )
            metric_id += 1

    metrics_per_fold_df = pd.DataFrame(metrics_per_fold)

    # Aggregate metrics
    metrics_aggregate = []
    aggregate_id = 0
    for clf in classifiers:
        for metric_name in ["auroc", "aupr", "brier_score"]:
            vals = np.random.uniform(0.7, 0.9, 5)
            metrics_aggregate.append(
                {
                    "aggregate_id": aggregate_id,
                    "source_name": "test_pipeline",
                    "classifier": clf,
                    "metric_name": metric_name,
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "ci_lower": np.percentile(vals, 2.5),
                    "ci_upper": np.percentile(vals, 97.5),
                    "median": np.median(vals),
                    "q25": np.percentile(vals, 25),
                    "q75": np.percentile(vals, 75),
                    "n_observations": 5,
                }
            )
            aggregate_id += 1

    metrics_aggregate_df = pd.DataFrame(metrics_aggregate)

    return metrics_per_fold_df, metrics_aggregate_df


@pytest.fixture
def temp_artifact_dir(tmp_path):
    """Create temporary directory with mock artifacts."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    # Create mock pickle files
    for i in range(5):
        data = {
            "predictions": np.random.rand(100),
            "metrics": {"auroc": 0.85 + i * 0.01},
        }
        with open(artifact_dir / f"artifact_{i}.pickle", "wb") as f:
            pickle.dump(data, f)

    return artifact_dir


# ============================================================================
# Test: Memory-efficient utilities
# ============================================================================


class TestMemoryEfficiency:
    """Tests for memory-efficient data loading utilities."""

    def test_concat_dataframes_efficient_empty_list(self):
        """concat_dataframes_efficient handles empty list."""
        result = concat_dataframes_efficient([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_concat_dataframes_efficient_single_df(self):
        """concat_dataframes_efficient handles single DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = concat_dataframes_efficient([df])
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_concat_dataframes_efficient_multiple_dfs(self):
        """concat_dataframes_efficient correctly concatenates multiple DataFrames."""
        dfs = [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
            pd.DataFrame({"a": [9, 10], "b": [11, 12]}),
        ]
        result = concat_dataframes_efficient(dfs)
        assert len(result) == 6
        assert list(result["a"]) == [1, 2, 5, 6, 9, 10]

    def test_load_artifact_safe_context_manager(self, temp_artifact_dir):
        """load_artifact_safe properly loads and cleans up artifacts."""
        artifact_path = temp_artifact_dir / "artifact_0.pickle"

        with load_artifact_safe(artifact_path) as artifact:
            assert "predictions" in artifact
            assert "metrics" in artifact
            assert isinstance(artifact["predictions"], np.ndarray)

        # After context exit, artifact should be cleaned up
        gc.collect()

    def test_iter_artifacts_chunked(self, temp_artifact_dir):
        """iter_artifacts_chunked processes artifacts in batches."""
        artifact_paths = list(temp_artifact_dir.glob("*.pickle"))

        batches = list(iter_artifacts_chunked(artifact_paths, batch_size=2))

        assert len(batches) == 3  # 5 artifacts / 2 batch_size = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1


# ============================================================================
# Test: DuckDB Export Functions
# ============================================================================


class TestDuckDBExport:
    """Tests for DuckDB export functionality."""

    def test_export_features_to_duckdb(self, sample_features_data, tmp_path):
        """export_features_to_duckdb creates valid DuckDB file."""
        features_data, metadata = sample_features_data
        output_path = tmp_path / "features.db"

        result = export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=output_path,
        )

        assert result.exists()
        assert result.stat().st_size > 0

        # Verify we can read the data back
        import duckdb

        with duckdb.connect(str(output_path), read_only=True) as con:
            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            assert "plr_features" in table_names
            assert "feature_metadata" in table_names

            # Check row counts
            features_count = con.execute(
                "SELECT COUNT(*) FROM plr_features"
            ).fetchone()[0]
            metadata_count = con.execute(
                "SELECT COUNT(*) FROM feature_metadata"
            ).fetchone()[0]

            assert features_count == 50
            assert metadata_count == 50

    def test_export_results_to_duckdb(
        self, sample_predictions_data, sample_metrics_data, tmp_path
    ):
        """export_results_to_duckdb creates valid DuckDB file."""
        metrics_per_fold, metrics_aggregate = sample_metrics_data
        output_path = tmp_path / "results.db"

        result = export_results_to_duckdb(
            predictions_df=sample_predictions_data,
            metrics_per_fold=metrics_per_fold,
            metrics_aggregate=metrics_aggregate,
            output_path=output_path,
        )

        assert result.exists()
        assert result.stat().st_size > 0

        # Verify data
        import duckdb

        with duckdb.connect(str(output_path), read_only=True) as con:
            pred_count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            fold_count = con.execute(
                "SELECT COUNT(*) FROM metrics_per_fold"
            ).fetchone()[0]
            agg_count = con.execute(
                "SELECT COUNT(*) FROM metrics_aggregate"
            ).fetchone()[0]

            assert pred_count == 200
            assert fold_count == 15  # 3 classifiers * 5 folds
            assert agg_count == 9  # 3 classifiers * 3 metrics

    def test_export_handles_optional_tables(self, sample_predictions_data, tmp_path):
        """export_results_to_duckdb handles optional tables gracefully."""
        output_path = tmp_path / "results_minimal.db"

        # Export with minimal data (no calibration curves, DCA curves, etc.)
        result = export_results_to_duckdb(
            predictions_df=sample_predictions_data,
            metrics_per_fold=pd.DataFrame(),
            metrics_aggregate=pd.DataFrame(),
            output_path=output_path,
        )

        assert result.exists()


# ============================================================================
# Test: DuckDB Load Functions
# ============================================================================


class TestDuckDBLoad:
    """Tests for loading data from DuckDB."""

    def test_load_features_from_duckdb(self, sample_features_data, tmp_path):
        """load_features_from_duckdb correctly loads feature data."""
        features_data, metadata = sample_features_data
        output_path = tmp_path / "features.db"

        # First export
        export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=output_path,
        )

        # Then load
        X, y, feature_names = load_features_from_duckdb(output_path)

        assert X.shape[0] == 50  # n_subjects
        assert X.shape[1] > 0  # has features
        assert y.shape[0] == 50
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_load_features_with_split_filter(self, sample_features_data, tmp_path):
        """load_features_from_duckdb filters by split correctly."""
        features_data, metadata = sample_features_data
        output_path = tmp_path / "features.db"

        export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=output_path,
        )

        # Load only train split
        X_train, y_train, _ = load_features_from_duckdb(output_path, split="train")

        # Should be fewer than total
        assert X_train.shape[0] <= 50
        assert X_train.shape[0] > 0

    def test_load_results_from_duckdb(
        self, sample_predictions_data, sample_metrics_data, tmp_path
    ):
        """load_results_from_duckdb correctly loads result data."""
        metrics_per_fold, metrics_aggregate = sample_metrics_data
        output_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=sample_predictions_data,
            metrics_per_fold=metrics_per_fold,
            metrics_aggregate=metrics_aggregate,
            output_path=output_path,
        )

        # Load different tables
        df_predictions = load_results_from_duckdb(output_path, table="predictions")
        df_metrics = load_results_from_duckdb(output_path, table="metrics_aggregate")

        assert len(df_predictions) == 200
        assert len(df_metrics) == 9


# ============================================================================
# Test: Analysis Pipeline
# ============================================================================


class TestDuckDBAnalysisPipeline:
    """Tests for the analysis pipeline."""

    def test_pipeline_from_features(self, sample_features_data, tmp_path):
        """Pipeline correctly initializes from features database."""
        features_data, metadata = sample_features_data
        db_path = tmp_path / "features.db"

        export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=db_path,
        )

        pipeline = DuckDBAnalysisPipeline.from_features(db_path)

        assert pipeline.can_run_classification()
        assert pipeline._features is not None
        assert pipeline._labels is not None

    def test_pipeline_from_results(
        self, sample_predictions_data, sample_metrics_data, tmp_path
    ):
        """Pipeline correctly initializes from results database."""
        metrics_per_fold, metrics_aggregate = sample_metrics_data
        db_path = tmp_path / "results.db"

        export_results_to_duckdb(
            predictions_df=sample_predictions_data,
            metrics_per_fold=metrics_per_fold,
            metrics_aggregate=metrics_aggregate,
            output_path=db_path,
        )

        pipeline = DuckDBAnalysisPipeline.from_results(db_path)

        assert pipeline.can_run_statistics()
        assert pipeline._predictions_df is not None

    def test_pipeline_run_classification(self, sample_features_data, tmp_path):
        """Pipeline can run classification from features."""
        features_data, metadata = sample_features_data
        db_path = tmp_path / "features.db"

        export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=db_path,
        )

        pipeline = DuckDBAnalysisPipeline.from_features(db_path)
        results = pipeline.run_classification(
            classifiers=["LogisticRegression"],
            n_folds=3,
        )

        assert len(results) > 0
        assert "y_true" in results.columns
        assert "y_pred" in results.columns
        assert "y_prob" in results.columns

    def test_pipeline_export_for_reproduction(self, sample_features_data, tmp_path):
        """Pipeline can export results for reproduction."""
        features_data, metadata = sample_features_data
        features_db = tmp_path / "features.db"
        results_db = tmp_path / "results.db"

        export_features_to_duckdb(
            features_data=features_data,
            metadata=metadata,
            output_path=features_db,
        )

        pipeline = DuckDBAnalysisPipeline.from_features(features_db)
        pipeline.run_classification(classifiers=["LogisticRegression"], n_folds=2)

        output_path = pipeline.export_for_reproduction(results_db)

        assert output_path.exists()


# ============================================================================
# Test: Schema Validation
# ============================================================================


class TestSchemaValidation:
    """Tests for DuckDB schema correctness."""

    def test_features_schema_creates_valid_tables(self, tmp_path):
        """FEATURES_SCHEMA creates valid DuckDB tables."""
        import duckdb

        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(FEATURES_SCHEMA)

            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            expected_tables = [
                "feature_metadata",
                "plr_features",
                "feature_provenance",
                "feature_statistics",
            ]
            for table in expected_tables:
                assert table in table_names

    def test_results_schema_creates_valid_tables(self, tmp_path):
        """RESULTS_SCHEMA creates valid DuckDB tables."""
        import duckdb

        db_path = tmp_path / "test_schema.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(RESULTS_SCHEMA)

            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]

            expected_tables = [
                "predictions",
                "metrics_per_fold",
                "metrics_aggregate",
                "calibration_curves",
                "dca_curves",
                "mlflow_runs",
            ]
            for table in expected_tables:
                assert table in table_names


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_export_empty_features(self, tmp_path):
        """Handles empty feature data gracefully."""
        output_path = tmp_path / "empty_features.db"

        result = export_features_to_duckdb(
            features_data={},
            metadata=pd.DataFrame(
                columns=["subject_id", "eye", "split", "source_name", "has_glaucoma"]
            ),
            output_path=output_path,
        )

        assert result.exists()

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Loading nonexistent file raises appropriate error."""
        with pytest.raises(Exception):  # duckdb raises various errors
            load_features_from_duckdb(tmp_path / "nonexistent.db")

    def test_pipeline_without_features_cannot_classify(self, tmp_path):
        """Pipeline without features cannot run classification."""
        pipeline = DuckDBAnalysisPipeline()

        assert not pipeline.can_run_classification()

        with pytest.raises(ValueError, match="Cannot run classification"):
            pipeline.run_classification()

    def test_pipeline_without_results_cannot_run_stats(self, tmp_path):
        """Pipeline without results cannot run statistics."""
        pipeline = DuckDBAnalysisPipeline()

        assert not pipeline.can_run_statistics()

        with pytest.raises(ValueError, match="Cannot run statistics"):
            pipeline.run_statistics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
