"""Unit tests for data loader module."""

import numpy as np
import pandas as pd
import pytest

from src.viz.data_loader import (
    DuckDBLoader,
    MockDataLoader,
    DataQuery,
    create_loader,
)


class TestMockDataLoader:
    """Tests for MockDataLoader class."""

    def test_load_aggregated_metrics_default(self):
        """Test loading aggregated metrics with default synthetic data."""
        loader = MockDataLoader(n_groups=3)
        df = loader.load_aggregated_metrics("method", "auroc")

        assert len(df) == 3
        assert "method" in df.columns
        assert "mean_auroc" in df.columns
        assert "std_auroc" in df.columns
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert "n" in df.columns

    def test_load_aggregated_metrics_custom_data(self):
        """Test loading aggregated metrics with custom data."""
        custom_df = pd.DataFrame(
            {
                "model": ["A", "B"],
                "mean_accuracy": [0.9, 0.8],
                "std_accuracy": [0.02, 0.03],
                "ci_lower": [0.88, 0.77],
                "ci_upper": [0.92, 0.83],
                "n": [100, 100],
            }
        )
        loader = MockDataLoader(mock_data={"agg_model_accuracy": custom_df})
        df = loader.load_aggregated_metrics("model", "accuracy")

        assert len(df) == 2
        assert df["mean_accuracy"].iloc[0] == 0.9

    def test_load_per_iteration_default(self):
        """Test loading per-iteration data with default synthetic data."""
        loader = MockDataLoader(n_groups=3, n_iterations=50)
        df = loader.load_per_iteration("method", "iteration", "auroc")

        assert len(df) == 50
        assert len(df.columns) == 3

    def test_load_raw_default(self):
        """Test loading raw data with default synthetic data."""
        loader = MockDataLoader()
        df = loader.load_raw(["y_prob", "y_true", "uncertainty"])

        assert len(df) == 1000  # Default
        assert "y_prob" in df.columns
        assert "y_true" in df.columns
        assert "uncertainty" in df.columns
        # Check value ranges
        assert (df["y_prob"] >= 0).all() and (df["y_prob"] <= 1).all()
        assert set(df["y_true"].unique()).issubset({0, 1})

    def test_load_raw_with_limit(self):
        """Test loading raw data with row limit."""
        loader = MockDataLoader()
        df = loader.load_raw(["x", "y"], limit=50)

        assert len(df) == 50

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data."""
        loader1 = MockDataLoader(seed=42)
        loader2 = MockDataLoader(seed=42)

        df1 = loader1.load_aggregated_metrics("method", "auroc")
        df2 = loader2.load_aggregated_metrics("method", "auroc")

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        loader1 = MockDataLoader(seed=42)
        loader2 = MockDataLoader(seed=123)

        df1 = loader1.load_aggregated_metrics("method", "auroc")
        df2 = loader2.load_aggregated_metrics("method", "auroc")

        assert not df1["mean_auroc"].equals(df2["mean_auroc"])


class TestDuckDBLoader:
    """Tests for DuckDBLoader class."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary DuckDB database for testing."""
        import duckdb

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))

        # Create test table
        conn.execute("""
            CREATE TABLE test_metrics (
                method VARCHAR,
                iteration INTEGER,
                auroc DOUBLE,
                brier DOUBLE
            )
        """)

        # Insert test data
        methods = ["MethodA", "MethodB", "MethodC"]
        for method in methods:
            base_auroc = 0.7 + methods.index(method) * 0.05
            for i in range(100):
                conn.execute(
                    "INSERT INTO test_metrics VALUES (?, ?, ?, ?)",
                    [
                        method,
                        i,
                        base_auroc + np.random.randn() * 0.02,
                        0.2 - methods.index(method) * 0.03 + np.random.randn() * 0.01,
                    ],
                )

        conn.close()
        return db_path

    def test_init_with_nonexistent_db_raises(self, tmp_path):
        """Test that initializing with nonexistent DB raises error."""
        with pytest.raises(FileNotFoundError):
            DuckDBLoader(tmp_path / "nonexistent.duckdb")

    def test_load_aggregated_metrics(self, temp_db):
        """Test loading aggregated metrics from DuckDB."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        df = loader.load_aggregated_metrics("method", "auroc")

        assert len(df) == 3
        assert "method" in df.columns
        assert "mean_auroc" in df.columns
        assert df["mean_auroc"].iloc[0] > df["mean_auroc"].iloc[2]  # Sorted descending

    def test_load_aggregated_metrics_with_filter(self, temp_db):
        """Test loading with filter."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        df = loader.load_aggregated_metrics(
            "method", "auroc", filters={"method": "MethodA"}
        )

        assert len(df) == 1
        assert df["method"].iloc[0] == "MethodA"

    def test_load_per_iteration(self, temp_db):
        """Test loading per-iteration data."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        df = loader.load_per_iteration("method", "iteration", "auroc")

        assert len(df) == 100  # Number of iterations
        assert len(df.columns) == 3  # Number of methods

    def test_load_raw(self, temp_db):
        """Test loading raw data."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        df = loader.load_raw(["method", "auroc"], limit=10)

        assert len(df) == 10
        assert "method" in df.columns
        assert "auroc" in df.columns

    def test_get_available_columns(self, temp_db):
        """Test getting available columns."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        columns = loader.get_available_columns()

        assert "method" in columns
        assert "iteration" in columns
        assert "auroc" in columns
        assert "brier" in columns

    def test_get_unique_values(self, temp_db):
        """Test getting unique values."""
        loader = DuckDBLoader(temp_db, table_name="test_metrics")
        methods = loader.get_unique_values("method")

        assert len(methods) == 3
        assert "MethodA" in methods


class TestDataQuery:
    """Tests for DataQuery dataclass."""

    def test_basic_query(self):
        """Test creating a basic query."""
        query = DataQuery(group_col="method", value_col="auroc")
        assert query.group_col == "method"
        assert query.value_col == "auroc"
        assert query.filters is None

    def test_query_with_filters(self):
        """Test query with filters."""
        query = DataQuery(
            group_col="method", value_col="auroc", filters={"classifier": "CatBoost"}
        )
        assert query.filters == {"classifier": "CatBoost"}


class TestCreateLoader:
    """Tests for create_loader factory function."""

    def test_create_mock_loader_from_dict(self):
        """Test creating mock loader from dictionary."""
        mock_data = {"test": pd.DataFrame({"x": [1, 2, 3]})}
        loader = create_loader(mock_data)
        assert isinstance(loader, MockDataLoader)

    def test_create_duckdb_loader(self, tmp_path):
        """Test creating DuckDB loader."""
        import duckdb

        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE essential_metrics (x INTEGER)")
        conn.close()

        loader = create_loader(db_path)
        assert isinstance(loader, DuckDBLoader)

    def test_create_loader_unknown_type_raises(self, tmp_path):
        """Test that unknown file type raises error."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()

        with pytest.raises(ValueError, match="Unknown data source type"):
            create_loader(unknown_file)


class TestDataLoaderInterface:
    """Tests for DataLoader interface compliance."""

    @pytest.fixture(params=["mock", "duckdb"])
    def loader(self, request, tmp_path):
        """Parametrized fixture for different loader types."""
        if request.param == "mock":
            return MockDataLoader()
        else:
            import duckdb

            db_path = tmp_path / "test.duckdb"
            conn = duckdb.connect(str(db_path))
            conn.execute("""
                CREATE TABLE essential_metrics (
                    method VARCHAR,
                    iteration INTEGER,
                    auroc DOUBLE
                )
            """)
            for method in ["A", "B", "C"]:
                for i in range(50):
                    conn.execute(
                        "INSERT INTO essential_metrics VALUES (?, ?, ?)",
                        [method, i, 0.8 + np.random.randn() * 0.05],
                    )
            conn.close()
            return DuckDBLoader(db_path)

    def test_load_aggregated_returns_dataframe(self, loader):
        """Test that load_aggregated_metrics returns DataFrame."""
        result = loader.load_aggregated_metrics("method", "auroc")
        assert isinstance(result, pd.DataFrame)

    def test_load_per_iteration_returns_dataframe(self, loader):
        """Test that load_per_iteration returns DataFrame."""
        result = loader.load_per_iteration("method", "iteration", "auroc")
        assert isinstance(result, pd.DataFrame)

    def test_load_raw_returns_dataframe(self, loader):
        """Test that load_raw returns DataFrame."""
        result = loader.load_raw(["method", "auroc"], limit=10)
        assert isinstance(result, pd.DataFrame)
