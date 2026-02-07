"""
Data Loader - Abstracts data loading for visualizations.

Enables testing with mock data and switching between data sources.
This module provides a clean separation between data fetching and visualization.

Example usage:
    >>> from src.viz.data_loader import DuckDBLoader, MockDataLoader
    >>> loader = DuckDBLoader('/path/to/db.duckdb', 'metrics_table')
    >>> df = loader.load_aggregated_metrics('method', 'auroc')
    >>> print(df.columns)  # ['method', 'mean_auroc', 'std_auroc', 'ci_lower', 'ci_upper', 'n']
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class DataQuery:
    """
    Specification for a data query.

    Attributes
    ----------
    group_col : str
        Column to group by (e.g., 'method', 'classifier')
    value_col : str
        Column containing metric values
    filters : dict, optional
        Column -> value filters to apply
    order_by : str, optional
        Column to order results by (default: descending by mean value)
    limit : int, optional
        Maximum number of groups to return
    """

    group_col: str
    value_col: str
    filters: Optional[Dict[str, Any]] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None


class DataLoader(ABC):
    """
    Abstract base class for data loading.

    Subclasses implement specific data sources (DuckDB, CSV, mock data).
    """

    @abstractmethod
    def load_aggregated_metrics(
        self,
        group_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
        ci_method: str = "percentile",
        ci_alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Load aggregated metrics with confidence intervals.

        Parameters
        ----------
        group_col : str
            Column to group by
        value_col : str
            Column containing metric values
        filters : dict, optional
            Filters to apply {column: value}
        ci_method : str
            CI method: 'percentile', 'normal', or 'none'
        ci_alpha : float
            Significance level for CI (default 0.05 for 95% CI)

        Returns
        -------
        pd.DataFrame
            Columns: [group_col, f'mean_{value_col}', f'std_{value_col}',
                      'ci_lower', 'ci_upper', 'n']
        """
        pass

    @abstractmethod
    def load_per_iteration(
        self,
        group_col: str,
        iteration_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Load per-iteration data for statistical tests.

        Parameters
        ----------
        group_col : str
            Column identifying methods/configurations
        iteration_col : str
            Column identifying bootstrap iterations
        value_col : str
            Column containing metric values

        Returns
        -------
        pd.DataFrame
            Wide format: index=iteration, columns=methods, values=metric
        """
        pass

    @abstractmethod
    def load_raw(
        self,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load raw data with specified columns.

        Parameters
        ----------
        columns : list of str
            Columns to select
        filters : dict, optional
            Filters to apply
        limit : int, optional
            Maximum rows to return

        Returns
        -------
        pd.DataFrame
        """
        pass


class DuckDBLoader(DataLoader):
    """
    Load data from DuckDB database.

    Parameters
    ----------
    db_path : str or Path
        Path to DuckDB database file
    table_name : str
        Default table name to query
    """

    def __init__(
        self, db_path: Union[str, Path], table_name: str = "essential_metrics"
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def _get_connection(self):
        """Get a read-only DuckDB connection."""
        import duckdb

        return duckdb.connect(str(self.db_path), read_only=True)

    def _build_where_clause(
        self,
        filters: Optional[Dict[str, Any]],
        required_cols: List[str],
    ) -> str:
        """Build SQL WHERE clause from filters."""
        clauses = [f"{col} IS NOT NULL" for col in required_cols]

        if filters:
            for col, val in filters.items():
                if isinstance(val, str):
                    clauses.append(f"{col} = '{val}'")
                elif isinstance(val, (list, tuple)):
                    val_str = ", ".join(
                        f"'{v}'" if isinstance(v, str) else str(v) for v in val
                    )
                    clauses.append(f"{col} IN ({val_str})")
                elif val is None:
                    clauses.append(f"{col} IS NULL")
                else:
                    clauses.append(f"{col} = {val}")

        return " AND ".join(clauses)

    def load_aggregated_metrics(
        self,
        group_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
        ci_method: str = "percentile",
        ci_alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Load aggregated metrics with confidence intervals."""
        conn = self._get_connection()

        where_clause = self._build_where_clause(filters, [group_col, value_col])

        # Compute CI bounds based on method
        if ci_method == "percentile":
            ci_lower_expr = (
                f"PERCENTILE_CONT({ci_alpha / 2}) WITHIN GROUP (ORDER BY {value_col})"
            )
            ci_upper_expr = f"PERCENTILE_CONT({1 - ci_alpha / 2}) WITHIN GROUP (ORDER BY {value_col})"
        elif ci_method == "normal":
            # Use normal approximation: mean ± z * std / sqrt(n)
            z = 1.96  # For 95% CI
            ci_lower_expr = (
                f"AVG({value_col}) - {z} * STDDEV({value_col}) / SQRT(COUNT(*))"
            )
            ci_upper_expr = (
                f"AVG({value_col}) + {z} * STDDEV({value_col}) / SQRT(COUNT(*))"
            )
        else:
            ci_lower_expr = "NULL"
            ci_upper_expr = "NULL"

        query = f"""
        SELECT
            {group_col},
            AVG({value_col}) as mean_{value_col},
            STDDEV({value_col}) as std_{value_col},
            {ci_lower_expr} as ci_lower,
            {ci_upper_expr} as ci_upper,
            COUNT(*) as n
        FROM {self.table_name}
        WHERE {where_clause}
        GROUP BY {group_col}
        ORDER BY mean_{value_col} DESC
        """

        try:
            df = conn.execute(query).fetchdf()
        finally:
            conn.close()

        return df

    def load_per_iteration(
        self,
        group_col: str,
        iteration_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Load per-iteration data in wide format for statistical tests."""
        conn = self._get_connection()

        where_clause = self._build_where_clause(
            filters, [group_col, iteration_col, value_col]
        )

        query = f"""
        SELECT {group_col}, {iteration_col}, {value_col}
        FROM {self.table_name}
        WHERE {where_clause}
        """

        try:
            df = conn.execute(query).fetchdf()
        finally:
            conn.close()

        # Pivot to wide format
        df_wide = df.pivot(index=iteration_col, columns=group_col, values=value_col)

        return df_wide

    def load_raw(
        self,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load raw data with specified columns."""
        conn = self._get_connection()

        cols_str = ", ".join(columns)
        where_clause = self._build_where_clause(filters, [])

        query = f"SELECT {cols_str} FROM {self.table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        try:
            df = conn.execute(query).fetchdf()
        finally:
            conn.close()

        return df

    def get_available_columns(self) -> List[str]:
        """Get list of columns in the table."""
        conn = self._get_connection()
        try:
            result = conn.execute(f"DESCRIBE {self.table_name}").fetchdf()
            return result["column_name"].tolist()
        finally:
            conn.close()

    def get_unique_values(self, column: str) -> List[Any]:
        """Get unique values in a column."""
        conn = self._get_connection()
        try:
            result = conn.execute(
                f"SELECT DISTINCT {column} FROM {self.table_name} WHERE {column} IS NOT NULL"
            ).fetchdf()
            return result[column].tolist()
        finally:
            conn.close()


class MockDataLoader(DataLoader):
    """
    Mock data loader for testing.

    Parameters
    ----------
    mock_data : dict, optional
        Pre-defined mock data {key: DataFrame}
    n_groups : int
        Number of groups to generate for synthetic data
    n_iterations : int
        Number of iterations for per-iteration data
    seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        mock_data: Optional[Dict[str, pd.DataFrame]] = None,
        n_groups: int = 5,
        n_iterations: int = 100,
        seed: int = 42,
    ):
        self.mock_data = mock_data or {}
        self.n_groups = n_groups
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(seed)

    def load_aggregated_metrics(
        self,
        group_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
        ci_method: str = "percentile",
        ci_alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic aggregated metrics."""
        key = f"agg_{group_col}_{value_col}"
        if key in self.mock_data:
            return self.mock_data[key]

        # Generate synthetic data
        groups = [f"Method_{chr(65 + i)}" for i in range(self.n_groups)]
        means = self.rng.uniform(0.7, 0.9, self.n_groups)
        stds = self.rng.uniform(0.02, 0.08, self.n_groups)

        # Sort by mean (descending)
        sort_idx = np.argsort(-means)
        groups = [groups[i] for i in sort_idx]
        means = means[sort_idx]
        stds = stds[sort_idx]

        return pd.DataFrame(
            {
                group_col: groups,
                f"mean_{value_col}": means,
                f"std_{value_col}": stds,
                "ci_lower": means - 1.96 * stds / np.sqrt(self.n_iterations),
                "ci_upper": means + 1.96 * stds / np.sqrt(self.n_iterations),
                "n": [self.n_iterations] * self.n_groups,
            }
        )

    def load_per_iteration(
        self,
        group_col: str,
        iteration_col: str,
        value_col: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic per-iteration data."""
        key = f"iter_{group_col}_{value_col}"
        if key in self.mock_data:
            return self.mock_data[key]

        groups = [f"Method_{chr(65 + i)}" for i in range(self.n_groups)]

        data = {}
        for i, group in enumerate(groups):
            mean = 0.7 + i * 0.05
            data[group] = self.rng.normal(mean, 0.05, self.n_iterations)

        return pd.DataFrame(data, index=range(self.n_iterations))

    def load_raw(
        self,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic raw data."""
        key = f"raw_{'_'.join(columns)}"
        if key in self.mock_data:
            df = self.mock_data[key]
            return df.head(limit) if limit else df

        n_rows = limit or 1000
        data = {}
        for col in columns:
            if "prob" in col.lower():
                data[col] = self.rng.uniform(0, 1, n_rows)
            elif "true" in col.lower() or "label" in col.lower():
                data[col] = self.rng.binomial(1, 0.3, n_rows)
            elif "uncertainty" in col.lower() or "std" in col.lower():
                data[col] = self.rng.uniform(0.01, 0.2, n_rows)
            else:
                data[col] = self.rng.standard_normal(n_rows)

        return pd.DataFrame(data)


def create_loader(
    source: Union[str, Path, Dict],
    table_name: str = "essential_metrics",
) -> DataLoader:
    """
    Factory function to create appropriate data loader.

    Parameters
    ----------
    source : str, Path, or dict
        - If str/Path ending in .duckdb: DuckDBLoader
        - If str/Path ending in .csv: Load as DataFrame and use MockDataLoader
        - If dict: MockDataLoader with pre-defined data
    table_name : str
        Table name for DuckDB

    Returns
    -------
    DataLoader
    """
    if isinstance(source, dict):
        return MockDataLoader(mock_data=source)

    source = Path(source)

    if source.suffix == ".duckdb":
        return DuckDBLoader(source, table_name)
    elif source.suffix == ".csv":
        df = pd.read_csv(source)
        return MockDataLoader(mock_data={"raw": df})
    else:
        raise ValueError(f"Unknown data source type: {source}")
