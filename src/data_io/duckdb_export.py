"""
Memory-efficient export of features and classifier results to DuckDB.

This module creates shareable DuckDB databases that enable:
1. Reproducibility without raw clinical data access
2. Memory-efficient analysis (target: <16GB RAM)
3. Continuation from intermediate artifacts

Cross-references:
- planning/share-features-and-classifier-outputs.md
- planning/statistics-implementation.md (Memory Optimization section)

Output Files:
- foundation_plr_features.db: Hand-crafted PLR features (shareable)
- foundation_plr_results.db: All classifier outputs (shareable)

Usage:
    # Export from mlruns
    python -m src.data_io.duckdb_export export --mlruns ./mlruns

    # Continue analysis from features.db
    python -m src.data_io.duckdb_export analyze --from-features features.db

    # Continue analysis from results.db (re-run only statistics)
    python -m src.data_io.duckdb_export analyze --from-results results.db
"""

import gc
import json
import pickle
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

try:
    import mlflow
except ImportError:
    mlflow = None


__all__ = [
    "export_features_to_duckdb",
    "export_results_to_duckdb",
    "load_features_from_duckdb",
    "load_results_from_duckdb",
    "DuckDBAnalysisPipeline",
]


# ============================================================================
# Schema Definitions
# ============================================================================

FEATURES_SCHEMA = """
-- Feature metadata (subject-level)
CREATE TABLE IF NOT EXISTS feature_metadata (
    subject_id TEXT NOT NULL,
    eye TEXT NOT NULL CHECK (eye IN ('OD', 'OS')),
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    source_name TEXT NOT NULL,
    has_glaucoma INTEGER NOT NULL CHECK (has_glaucoma IN (0, 1)),
    PRIMARY KEY (subject_id, eye, source_name)
);

-- Hand-crafted PLR features
CREATE TABLE IF NOT EXISTS plr_features (
    subject_id TEXT NOT NULL,
    eye TEXT NOT NULL,
    source_name TEXT NOT NULL,
    -- Amplitude features
    baseline_diameter REAL,
    constriction_amplitude REAL,
    constriction_amplitude_rel REAL,
    max_constriction_diameter REAL,
    -- Latency features
    latency_to_constriction REAL,
    latency_75pct REAL,
    time_to_redilation REAL,
    -- Velocity features
    max_constriction_velocity REAL,
    mean_constriction_velocity REAL,
    max_redilation_velocity REAL,
    -- PIPR features
    pipr_6s REAL,
    pipr_10s REAL,
    -- Derived features
    recovery_time REAL,
    constriction_duration REAL,
    PRIMARY KEY (subject_id, eye, source_name)
);

-- Feature provenance (how features were computed)
CREATE TABLE IF NOT EXISTS feature_provenance (
    source_name TEXT PRIMARY KEY,
    outlier_method TEXT,
    imputation_method TEXT,
    featurization_method TEXT,
    computation_date TEXT,
    mlflow_run_id TEXT,
    config_hash TEXT
);

-- Feature statistics (for uncertainty analysis)
CREATE TABLE IF NOT EXISTS feature_statistics (
    subject_id TEXT NOT NULL,
    eye TEXT NOT NULL,
    source_name TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    mean REAL,
    std REAL,
    ci_lower REAL,
    ci_upper REAL,
    n_bootstrap INTEGER,
    PRIMARY KEY (subject_id, eye, source_name, feature_name)
);

-- Indices for efficient queries
CREATE INDEX IF NOT EXISTS idx_features_source ON plr_features(source_name);
CREATE INDEX IF NOT EXISTS idx_metadata_split ON feature_metadata(split);
CREATE INDEX IF NOT EXISTS idx_metadata_label ON feature_metadata(has_glaucoma);
"""

RESULTS_SCHEMA = """
-- Predictions from all models
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    subject_id TEXT,
    eye TEXT,
    fold INTEGER,
    bootstrap_iter INTEGER DEFAULT 0,
    -- Configuration
    outlier_method TEXT,
    imputation_method TEXT,
    featurization TEXT,
    classifier TEXT,
    source_name TEXT,
    -- Predictions
    y_true INTEGER,
    y_pred INTEGER,
    y_prob REAL,
    -- Provenance
    mlflow_run_id TEXT
);

-- Metrics per fold
CREATE TABLE IF NOT EXISTS metrics_per_fold (
    metric_id INTEGER PRIMARY KEY,
    source_name TEXT,
    classifier TEXT,
    fold INTEGER,
    -- Discrimination
    auroc REAL,
    aupr REAL,
    -- Calibration
    brier_score REAL,
    calibration_slope REAL,
    calibration_intercept REAL,
    e_o_ratio REAL,
    -- Classification metrics
    sensitivity REAL,
    specificity REAL,
    ppv REAL,
    npv REAL,
    f1_score REAL,
    accuracy REAL,
    -- Clinical utility
    net_benefit_5pct REAL,
    net_benefit_10pct REAL,
    net_benefit_20pct REAL
);

-- Aggregate metrics (mean, CI)
CREATE TABLE IF NOT EXISTS metrics_aggregate (
    aggregate_id INTEGER PRIMARY KEY,
    source_name TEXT,
    classifier TEXT,
    metric_name TEXT,
    mean REAL,
    std REAL,
    ci_lower REAL,
    ci_upper REAL,
    median REAL,
    q25 REAL,
    q75 REAL,
    n_observations INTEGER
);

-- Calibration curve data
CREATE TABLE IF NOT EXISTS calibration_curves (
    curve_id INTEGER PRIMARY KEY,
    source_name TEXT,
    classifier TEXT,
    bin_midpoint REAL,
    observed_proportion REAL,
    bin_count INTEGER,
    ci_lower REAL,
    ci_upper REAL
);

-- Decision curve analysis data
CREATE TABLE IF NOT EXISTS dca_curves (
    dca_id INTEGER PRIMARY KEY,
    source_name TEXT,
    classifier TEXT,
    threshold REAL,
    net_benefit_model REAL,
    net_benefit_all REAL,
    net_benefit_none REAL,
    sensitivity REAL,
    specificity REAL
);

-- MLflow run metadata
CREATE TABLE IF NOT EXISTS mlflow_runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT,
    run_name TEXT,
    status TEXT,
    start_time TEXT,
    end_time TEXT,
    params_json TEXT,
    metrics_json TEXT,
    tags_json TEXT
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_predictions_config ON predictions(
    outlier_method, imputation_method, featurization, classifier
);
CREATE INDEX IF NOT EXISTS idx_metrics_source ON metrics_per_fold(source_name);
CREATE INDEX IF NOT EXISTS idx_aggregate_metric ON metrics_aggregate(metric_name);
"""


# ============================================================================
# Memory-efficient artifact loading
# ============================================================================


@contextmanager
def load_artifact_safe(artifact_path: Path) -> Generator[Any, None, None]:
    """
    Context manager for safe artifact loading with cleanup.

    Usage:
        with load_artifact_safe(path) as artifact:
            # Use artifact
        # Artifact is automatically deleted and garbage collected
    """
    artifact = None
    try:
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        yield artifact
    finally:
        del artifact
        gc.collect()


def iter_artifacts_chunked(
    artifact_paths: List[Path], batch_size: int = 5
) -> Generator[List[Any], None, None]:
    """
    Iterate over artifacts in batches with explicit cleanup.

    Prevents memory accumulation when processing many artifacts.
    """
    for i in range(0, len(artifact_paths), batch_size):
        batch_paths = artifact_paths[i : i + batch_size]
        batch_artifacts = []

        for path in batch_paths:
            with open(path, "rb") as f:
                batch_artifacts.append(pickle.load(f))

        yield batch_artifacts

        # Explicit cleanup
        del batch_artifacts
        gc.collect()
        logger.debug(f"Processed batch {i // batch_size + 1}, memory cleaned")


def concat_dataframes_efficient(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Efficiently concatenate multiple DataFrames.

    Uses pandas concat with copy=False for O(n) performance instead of O(n^2)
    that would occur with iterative concatenation.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of DataFrames to concatenate.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with reset index.
    """
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, copy=False)


# ============================================================================
# Export Functions
# ============================================================================


def export_features_to_duckdb(
    features_data: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_path: Union[str, Path],
    provenance: Optional[Dict[str, Dict]] = None,
    chunk_size: int = 1000,
) -> Path:
    """
    Export hand-crafted features to DuckDB.

    Parameters
    ----------
    features_data : Dict[str, pd.DataFrame]
        Mapping from source_name to features DataFrame
    metadata : pd.DataFrame
        Subject metadata (subject_id, eye, split, has_glaucoma)
    output_path : str or Path
        Output .db file path
    provenance : Dict[str, Dict], optional
        Mapping from source_name to provenance info
    chunk_size : int, default 1000
        Rows per insert batch

    Returns
    -------
    Path
        Path to created database
    """
    output_path = Path(output_path)
    logger.info(f"Exporting features to {output_path}")

    # Remove existing file if present
    if output_path.exists():
        output_path.unlink()

    with duckdb.connect(str(output_path)) as con:
        # Create schema
        con.execute(FEATURES_SCHEMA)

        # Insert metadata
        if len(metadata) > 0:
            logger.info(f"Inserting {len(metadata)} metadata rows")
            # Register DataFrame and insert
            con.register("metadata_df", metadata)
            con.execute("INSERT INTO feature_metadata SELECT * FROM metadata_df")
            con.unregister("metadata_df")

        # Get expected column order from schema
        features_columns = [
            "subject_id",
            "eye",
            "source_name",
            "baseline_diameter",
            "constriction_amplitude",
            "constriction_amplitude_rel",
            "max_constriction_diameter",
            "latency_to_constriction",
            "latency_75pct",
            "time_to_redilation",
            "max_constriction_velocity",
            "mean_constriction_velocity",
            "max_redilation_velocity",
            "pipr_6s",
            "pipr_10s",
            "recovery_time",
            "constriction_duration",
        ]

        # Insert features
        total_rows = 0
        for source_name, df in features_data.items():
            if len(df) == 0:
                continue

            df = df.copy()
            df["source_name"] = source_name

            # Reorder columns to match schema, adding missing columns as NULL
            ordered_df = pd.DataFrame()
            for col in features_columns:
                if col in df.columns:
                    ordered_df[col] = df[col]
                else:
                    ordered_df[col] = None

            # Insert in chunks
            for i in range(0, len(ordered_df), chunk_size):
                chunk = ordered_df.iloc[i : i + chunk_size]
                con.register("chunk_df", chunk)
                con.execute("INSERT INTO plr_features SELECT * FROM chunk_df")
                con.unregister("chunk_df")

            total_rows += len(df)
            gc.collect()

            logger.debug(f"  Inserted {len(df)} rows for {source_name}")

        # Insert provenance
        if provenance:
            prov_df = pd.DataFrame(
                [{"source_name": k, **v} for k, v in provenance.items()]
            )
            con.register("prov_df", prov_df)
            con.execute("INSERT INTO feature_provenance SELECT * FROM prov_df")
            con.unregister("prov_df")

    file_size = output_path.stat().st_size / 1024 / 1024
    logger.info(
        f"Features export complete: {output_path} ({file_size:.1f} MB, {total_rows} rows)"
    )
    return output_path


def export_results_to_duckdb(
    predictions_df: pd.DataFrame,
    metrics_per_fold: pd.DataFrame,
    metrics_aggregate: pd.DataFrame,
    output_path: Union[str, Path],
    calibration_curves: Optional[pd.DataFrame] = None,
    dca_curves: Optional[pd.DataFrame] = None,
    mlflow_runs: Optional[pd.DataFrame] = None,
    chunk_size: int = 1000,
) -> Path:
    """
    Export classifier results to DuckDB.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        All predictions with columns matching schema
    metrics_per_fold : pd.DataFrame
        Metrics per fold
    metrics_aggregate : pd.DataFrame
        Aggregated metrics (mean, CI)
    output_path : str or Path
        Output .db file path
    calibration_curves : pd.DataFrame, optional
        Calibration curve data
    dca_curves : pd.DataFrame, optional
        Decision curve analysis data
    mlflow_runs : pd.DataFrame, optional
        MLflow run metadata
    chunk_size : int, default 1000
        Rows per insert batch

    Returns
    -------
    Path
        Path to created database
    """
    output_path = Path(output_path)
    logger.info(f"Exporting results to {output_path}")

    if output_path.exists():
        output_path.unlink()

    with duckdb.connect(str(output_path)) as con:
        con.execute(RESULTS_SCHEMA)

        # Insert predictions in chunks
        if predictions_df is not None and len(predictions_df) > 0:
            logger.info(f"Inserting {len(predictions_df)} predictions")
            for i in range(0, len(predictions_df), chunk_size):
                chunk = predictions_df.iloc[i : i + chunk_size]
                con.register("chunk_df", chunk)
                con.execute("INSERT INTO predictions SELECT * FROM chunk_df")
                con.unregister("chunk_df")
            gc.collect()

        # Insert metrics
        if metrics_per_fold is not None and len(metrics_per_fold) > 0:
            logger.info(f"Inserting {len(metrics_per_fold)} fold metrics")
            con.register("fold_df", metrics_per_fold)
            con.execute("INSERT INTO metrics_per_fold SELECT * FROM fold_df")
            con.unregister("fold_df")

        if metrics_aggregate is not None and len(metrics_aggregate) > 0:
            logger.info(f"Inserting {len(metrics_aggregate)} aggregate metrics")
            con.register("agg_df", metrics_aggregate)
            con.execute("INSERT INTO metrics_aggregate SELECT * FROM agg_df")
            con.unregister("agg_df")

        # Insert optional tables
        if calibration_curves is not None and len(calibration_curves) > 0:
            con.register("cal_df", calibration_curves)
            con.execute("INSERT INTO calibration_curves SELECT * FROM cal_df")
            con.unregister("cal_df")

        if dca_curves is not None and len(dca_curves) > 0:
            con.register("dca_df", dca_curves)
            con.execute("INSERT INTO dca_curves SELECT * FROM dca_df")
            con.unregister("dca_df")

        if mlflow_runs is not None and len(mlflow_runs) > 0:
            con.register("runs_df", mlflow_runs)
            con.execute("INSERT INTO mlflow_runs SELECT * FROM runs_df")
            con.unregister("runs_df")

    file_size = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Results export complete: {output_path} ({file_size:.1f} MB)")
    return output_path


# ============================================================================
# Load Functions (for continuation)
# ============================================================================


def load_features_from_duckdb(
    db_path: Union[str, Path],
    source_name: Optional[str] = None,
    split: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load features from DuckDB for classification.

    Parameters
    ----------
    db_path : str or Path
        Path to features.db
    source_name : str, optional
        Filter for specific pipeline configuration
    split : str, optional
        Filter for 'train', 'val', or 'test'

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    feature_names : List[str]
        Feature column names
    """
    db_path = Path(db_path)
    logger.info(f"Loading features from {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        query = """
            SELECT f.*, m.has_glaucoma, m.split
            FROM plr_features f
            JOIN feature_metadata m
                ON f.subject_id = m.subject_id AND f.eye = m.eye
            WHERE 1=1
        """

        if source_name:
            query += f" AND f.source_name = '{source_name}'"
        if split:
            query += f" AND m.split = '{split}'"

        df = con.execute(query).df()

    # Feature columns (exclude metadata)
    metadata_cols = ["subject_id", "eye", "source_name", "has_glaucoma", "split"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].values
    y = df["has_glaucoma"].values.astype(int)

    logger.info(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_cols


def load_results_from_duckdb(
    db_path: Union[str, Path],
    table: str = "metrics_aggregate",
) -> pd.DataFrame:
    """
    Load results from DuckDB.

    Parameters
    ----------
    db_path : str or Path
        Path to results.db
    table : str, default "metrics_aggregate"
        Table to load: "predictions", "metrics_per_fold", "metrics_aggregate",
        "calibration_curves", "dca_curves", "mlflow_runs"

    Returns
    -------
    pd.DataFrame
        Requested data
    """
    db_path = Path(db_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {table}").df()

    logger.info(f"Loaded {len(df)} rows from {table}")
    return df


# ============================================================================
# Analysis Pipeline (continuation from artifacts)
# ============================================================================


@dataclass
class DuckDBAnalysisPipeline:
    """
    Pipeline for running analysis from DuckDB artifacts.

    Supports continuation from:
    1. features.db - re-run classification + statistics
    2. results.db - re-run only statistics

    Usage:
        # From features (re-run classification)
        pipeline = DuckDBAnalysisPipeline.from_features("features.db")
        pipeline.run_classification()
        pipeline.run_statistics()

        # From results (re-run only statistics)
        pipeline = DuckDBAnalysisPipeline.from_results("results.db")
        pipeline.run_statistics()
    """

    features_db: Optional[Path] = None
    results_db: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs/analysis"))

    # Loaded data
    _features: Optional[np.ndarray] = field(default=None, repr=False)
    _labels: Optional[np.ndarray] = field(default=None, repr=False)
    _feature_names: Optional[List[str]] = field(default=None, repr=False)
    _predictions_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _metrics_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    @classmethod
    def from_features(
        cls,
        features_db: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DuckDBAnalysisPipeline":
        """Create pipeline from features database (will run classification)."""
        pipeline = cls(
            features_db=Path(features_db),
            output_dir=Path(output_dir) if output_dir else Path("outputs/analysis"),
        )
        pipeline._load_features()
        return pipeline

    @classmethod
    def from_results(
        cls, results_db: Union[str, Path], output_dir: Optional[Union[str, Path]] = None
    ) -> "DuckDBAnalysisPipeline":
        """Create pipeline from results database (statistics only)."""
        pipeline = cls(
            results_db=Path(results_db),
            output_dir=Path(output_dir) if output_dir else Path("outputs/analysis"),
        )
        pipeline._load_results()
        return pipeline

    def _load_features(self) -> None:
        """Load features from DuckDB."""
        if self.features_db is None:
            raise ValueError("features_db not set")

        self._features, self._labels, self._feature_names = load_features_from_duckdb(
            self.features_db
        )
        logger.info(
            f"Loaded features: {self._features.shape[0]} samples, "
            f"{self._features.shape[1]} features"
        )

    def _load_results(self) -> None:
        """Load results from DuckDB."""
        if self.results_db is None:
            raise ValueError("results_db not set")

        self._predictions_df = load_results_from_duckdb(
            self.results_db, table="predictions"
        )
        self._metrics_df = load_results_from_duckdb(
            self.results_db, table="metrics_aggregate"
        )
        logger.info(
            f"Loaded results: {len(self._predictions_df)} predictions, "
            f"{len(self._metrics_df)} aggregate metrics"
        )

    def can_run_classification(self) -> bool:
        """Check if classification can be run (requires features)."""
        return self._features is not None

    def can_run_statistics(self) -> bool:
        """Check if statistics can be run (requires results)."""
        return self._metrics_df is not None or self._predictions_df is not None

    def run_classification(
        self,
        classifiers: Optional[List[str]] = None,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Run classification on loaded features.

        Parameters
        ----------
        classifiers : List[str], optional
            Classifier names to use (default: LogReg, XGBoost, CatBoost)
        n_folds : int, default 5
            Number of CV folds

        Returns
        -------
        pd.DataFrame
            Classification results
        """
        if not self.can_run_classification():
            raise ValueError(
                "Cannot run classification: features not loaded. "
                "Use from_features() to create pipeline."
            )

        logger.info("Running classification...")

        # Import here to avoid circular deps
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        try:
            from xgboost import XGBClassifier

            has_xgboost = True
        except ImportError:
            has_xgboost = False
            logger.warning("XGBoost not available")

        if classifiers is None:
            classifiers = ["LogisticRegression"]
            if has_xgboost:
                classifiers.append("XGBoost")

        results = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for clf_name in classifiers:
            logger.info(f"  Training {clf_name}...")

            for fold, (train_idx, test_idx) in enumerate(
                skf.split(self._features, self._labels)
            ):
                X_train = self._features[train_idx]
                X_test = self._features[test_idx]
                y_train = self._labels[train_idx]
                y_test = self._labels[test_idx]

                # Initialize classifier
                if clf_name == "LogisticRegression":
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                elif clf_name == "XGBoost" and has_xgboost:
                    clf = XGBClassifier(
                        n_estimators=100,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=42,
                    )
                else:
                    logger.warning(f"Unknown classifier: {clf_name}, skipping")
                    continue

                # Train and predict
                clf.fit(X_train, y_train)
                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                # Compute AUROC (stored but not used directly - predictions are logged below)
                _ = roc_auc_score(y_test, y_prob)

                for i, (idx, prob, pred, true) in enumerate(
                    zip(test_idx, y_prob, y_pred, y_test)
                ):
                    results.append(
                        {
                            "prediction_id": len(results),
                            "subject_id": f"S{idx:03d}",
                            "eye": "OD",  # Placeholder
                            "fold": fold,
                            "bootstrap_iter": 0,
                            "outlier_method": "unknown",
                            "imputation_method": "unknown",
                            "featurization": "unknown",
                            "classifier": clf_name,
                            "source_name": "duckdb_pipeline",
                            "y_true": int(true),
                            "y_pred": int(pred),
                            "y_prob": float(prob),
                            "mlflow_run_id": None,
                        }
                    )

        self._predictions_df = pd.DataFrame(results)
        logger.info(f"Classification complete: {len(self._predictions_df)} predictions")

        # Compute aggregate metrics
        self._compute_aggregate_metrics()

        return self._predictions_df

    def _compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics from predictions."""
        if self._predictions_df is None:
            return

        from sklearn.metrics import roc_auc_score

        agg_results = []
        aggregate_id = 0
        for clf_name, group in self._predictions_df.groupby("classifier"):
            aurocs_per_fold = group.groupby("fold").apply(
                lambda x: roc_auc_score(x["y_true"], x["y_prob"]),
                include_groups=False,
            )

            agg_results.append(
                {
                    "aggregate_id": aggregate_id,
                    "source_name": "duckdb_pipeline",
                    "classifier": clf_name,
                    "metric_name": "auroc",
                    "mean": aurocs_per_fold.mean(),
                    "std": aurocs_per_fold.std(),
                    "ci_lower": np.percentile(aurocs_per_fold, 2.5),
                    "ci_upper": np.percentile(aurocs_per_fold, 97.5),
                    "median": np.median(aurocs_per_fold),
                    "q25": np.percentile(aurocs_per_fold, 25),
                    "q75": np.percentile(aurocs_per_fold, 75),
                    "n_observations": len(aurocs_per_fold),
                }
            )
            aggregate_id += 1

        self._metrics_df = pd.DataFrame(agg_results)

    def run_statistics(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run statistical analysis on loaded/computed results.

        Parameters
        ----------
        output_dir : Path, optional
            Override output directory

        Returns
        -------
        Dict[str, Any]
            Statistical results
        """
        if not self.can_run_statistics():
            raise ValueError(
                "Cannot run statistics: results not loaded. "
                "Use from_results() or run_classification() first."
            )

        output_dir = output_dir or self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running statistical analysis...")

        results = {
            "metrics_summary": self._metrics_df.to_dict()
            if self._metrics_df is not None
            else {},
            "n_predictions": len(self._predictions_df)
            if self._predictions_df is not None
            else 0,
        }

        # Add calibration analysis if predictions available
        if self._predictions_df is not None:
            from ..stats.calibration_extended import calibration_slope_intercept

            for clf_name, group in self._predictions_df.groupby("classifier"):
                try:
                    cal_result = calibration_slope_intercept(
                        group["y_true"].values, group["y_prob"].values
                    )
                    results[f"calibration_{clf_name}"] = {
                        "slope": cal_result.slope,
                        "intercept": cal_result.intercept,
                        "e_o_ratio": cal_result.e_o_ratio,
                        "brier_score": cal_result.brier_score,
                    }
                except Exception as e:
                    logger.warning(f"Calibration failed for {clf_name}: {e}")

        # Save results
        results_path = output_dir / "statistics_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Statistics saved to {results_path}")
        return results

    def export_for_reproduction(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """Export current state to DuckDB for future reproduction."""
        output_path = Path(output_path)

        if self._predictions_df is None:
            raise ValueError("No predictions to export. Run classification first.")

        return export_results_to_duckdb(
            predictions_df=self._predictions_df,
            metrics_per_fold=None,  # Could compute from predictions
            metrics_aggregate=self._metrics_df,
            output_path=output_path,
        )


# ============================================================================
# MLflow Artifact Extraction
# ============================================================================


def _compute_stratos_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Compute STRATOS-required metrics from predictions.

    Computes:
    - calibration_slope: Slope from logistic calibration model
    - calibration_intercept: Intercept from logistic calibration model
    - e_o_ratio: Observed/Expected ratio (Van Calster 2024)
    - net_benefit_5pct, _10pct, _20pct: Net benefit at clinical thresholds

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcomes (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities

    Returns
    -------
    Dict[str, Optional[float]]
        STRATOS metrics dictionary
    """
    result = {
        "calibration_slope": None,
        "calibration_intercept": None,
        "e_o_ratio": None,
        "net_benefit_5pct": None,
        "net_benefit_10pct": None,
        "net_benefit_20pct": None,
    }

    try:
        # Import stats modules
        from ..stats.calibration_extended import calibration_slope_intercept
        from ..stats.clinical_utility import net_benefit

        # Compute calibration metrics
        cal_result = calibration_slope_intercept(y_true, y_prob)
        result["calibration_slope"] = float(cal_result.slope)
        result["calibration_intercept"] = float(cal_result.intercept)
        result["e_o_ratio"] = float(cal_result.o_e_ratio)

        # Compute net benefit at clinical thresholds
        result["net_benefit_5pct"] = float(net_benefit(y_true, y_prob, 0.05))
        result["net_benefit_10pct"] = float(net_benefit(y_true, y_prob, 0.10))
        result["net_benefit_20pct"] = float(net_benefit(y_true, y_prob, 0.20))

    except Exception as e:
        logger.warning(f"Failed to compute STRATOS metrics: {e}")

    return result


def extract_mlflow_classification_runs(
    mlruns_dir: Union[str, Path],
    experiment_id: Optional[str] = None,
    batch_size: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract classification results from MLflow runs.

    Extracts all available metrics and computes STRATOS-required metrics:
    - Calibration slope, intercept, O:E ratio
    - Net benefit at 5%, 10%, 20% thresholds
    - Full DCA curves (50 threshold points from 1% to 50%)

    Parameters
    ----------
    mlruns_dir : str or Path
        Path to mlruns directory
    experiment_id : str, optional
        Specific experiment ID (defaults to classification experiment)
    batch_size : int
        Number of runs to process per batch

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (predictions_df, metrics_per_fold_df, metrics_aggregate_df, dca_curves_df, mlflow_runs_df)
    """
    import yaml

    mlruns_dir = Path(mlruns_dir)

    # Find classification experiment if not specified
    if experiment_id is None:
        # Look for experiment with most runs (likely classification)
        experiment_id = _find_classification_experiment(mlruns_dir)

    exp_dir = mlruns_dir / experiment_id
    if not exp_dir.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    logger.info(f"Extracting from experiment {experiment_id}")

    # Collect all run directories
    run_dirs = [
        d for d in exp_dir.iterdir() if d.is_dir() and (d / "meta.yaml").exists()
    ]
    logger.info(f"Found {len(run_dirs)} runs")

    all_predictions = []
    all_metrics = []
    all_aggregate = []
    all_dca_curves = []
    mlflow_runs = []

    prediction_id = 0
    metric_id = 0
    dca_id = 0

    # Process in batches
    for batch_start in range(0, len(run_dirs), batch_size):
        batch_dirs = run_dirs[batch_start : batch_start + batch_size]

        for run_dir in batch_dirs:
            try:
                # Load run metadata
                with open(run_dir / "meta.yaml") as f:
                    meta = yaml.safe_load(f)

                run_id = meta.get("run_id", "")
                run_name = meta.get("run_name", "")

                # Skip non-classification runs (case-insensitive matching)
                run_name_upper = run_name.upper()
                if not any(
                    clf.upper() in run_name_upper
                    for clf in [
                        "XGBOOST",
                        "LogisticRegression",
                        "CatBoost",
                        "TabM",
                        "TabPFN",
                    ]
                ):
                    continue

                # Parse run configuration from name
                config = _parse_run_name(run_name)

                # Store MLflow run info
                mlflow_runs.append(
                    {
                        "run_id": run_id,
                        "experiment_name": f"exp_{experiment_id}",
                        "run_name": run_name,
                        "status": meta.get("status", ""),
                        "start_time": meta.get("start_time", ""),
                        "end_time": meta.get("end_time", ""),
                        "params_json": json.dumps(config),
                        "metrics_json": "{}",
                        "tags_json": "{}",
                    }
                )

                # Load metrics pickle
                metrics_path = _find_artifact(run_dir, "metrics", "*.pickle")
                if metrics_path:
                    with load_artifact_safe(metrics_path) as metrics_data:
                        # Extract aggregate metrics
                        if "metrics_stats" in metrics_data:
                            for split_name, split_data in metrics_data[
                                "metrics_stats"
                            ].items():
                                if (
                                    "metrics" not in split_data
                                    or "scalars" not in split_data["metrics"]
                                ):
                                    continue

                                scalars = split_data["metrics"]["scalars"]
                                for metric_name, metric_vals in scalars.items():
                                    if (
                                        isinstance(metric_vals, dict)
                                        and "mean" in metric_vals
                                    ):
                                        all_aggregate.append(
                                            {
                                                "aggregate_id": len(all_aggregate),
                                                "source_name": config.get(
                                                    "source_name", run_name
                                                ),
                                                "classifier": config.get(
                                                    "classifier", "unknown"
                                                ),
                                                "metric_name": f"{split_name}/{metric_name}",
                                                "mean": metric_vals.get("mean", np.nan),
                                                "std": metric_vals.get("std", np.nan),
                                                "ci_lower": metric_vals.get(
                                                    "ci", [np.nan, np.nan]
                                                )[0]
                                                if isinstance(
                                                    metric_vals.get("ci"),
                                                    (list, np.ndarray),
                                                )
                                                else np.nan,
                                                "ci_upper": metric_vals.get(
                                                    "ci", [np.nan, np.nan]
                                                )[1]
                                                if isinstance(
                                                    metric_vals.get("ci"),
                                                    (list, np.ndarray),
                                                )
                                                else np.nan,
                                                "median": np.nan,
                                                "q25": np.nan,
                                                "q75": np.nan,
                                                "n_observations": metric_vals.get(
                                                    "n", 0
                                                ),
                                            }
                                        )

                        # Extract per-iteration metrics (all available, not just AUROC)
                        if "metrics_iter" in metrics_data:
                            for split_name, split_data in metrics_data[
                                "metrics_iter"
                            ].items():
                                if (
                                    "metrics" not in split_data
                                    or "scalars" not in split_data["metrics"]
                                ):
                                    continue

                                scalars = split_data["metrics"]["scalars"]

                                # Map MLflow metric names to schema column names
                                METRIC_MAPPING = {
                                    "AUROC": "auroc",
                                    "AUPR": "aupr",
                                    "Brier": "brier_score",
                                    "sensitivity": "sensitivity",
                                    "specificity": "specificity",
                                    "PPV": "ppv",
                                    "NPV": "npv",
                                    "F1": "f1_score",
                                    "accuracy": "accuracy",
                                }

                                # Determine number of folds from AUROC (or any available metric)
                                n_folds = 0
                                for metric_name in scalars:
                                    vals = scalars[metric_name]
                                    if isinstance(vals, (list, np.ndarray)):
                                        n_folds = min(len(vals), 10)  # Cap at 10
                                        break

                                # Extract all metrics per fold
                                for fold in range(n_folds):
                                    fold_metrics = {
                                        "metric_id": metric_id,
                                        "source_name": config.get(
                                            "source_name", run_name
                                        ),
                                        "classifier": config.get(
                                            "classifier", "unknown"
                                        ),
                                        "fold": fold,
                                        # Initialize all columns
                                        "auroc": None,
                                        "aupr": None,
                                        "brier_score": None,
                                        "calibration_slope": None,
                                        "calibration_intercept": None,
                                        "e_o_ratio": None,
                                        "sensitivity": None,
                                        "specificity": None,
                                        "ppv": None,
                                        "npv": None,
                                        "f1_score": None,
                                        "accuracy": None,
                                        "net_benefit_5pct": None,
                                        "net_benefit_10pct": None,
                                        "net_benefit_20pct": None,
                                    }

                                    # Extract available metrics
                                    for (
                                        mlflow_name,
                                        schema_col,
                                    ) in METRIC_MAPPING.items():
                                        if mlflow_name in scalars:
                                            vals = scalars[mlflow_name]
                                            if isinstance(
                                                vals, (list, np.ndarray)
                                            ) and fold < len(vals):
                                                val = vals[fold]
                                                fold_metrics[schema_col] = (
                                                    float(val)
                                                    if np.isfinite(val)
                                                    else None
                                                )

                                    all_metrics.append(fold_metrics)
                                    metric_id += 1

                # Load dict_arrays pickle for predictions
                arrays_path = _find_artifact(run_dir, "dict_arrays", "*.pickle")
                if arrays_path:
                    with load_artifact_safe(arrays_path) as arrays_data:
                        # Extract subject codes if available
                        subject_codes_test = arrays_data.get("subject_codes_test", [])
                        y_test = arrays_data.get("y_test", np.array([]))

                        # Note: Full predictions require subjectwise_stats from metrics
                        # For now, store aggregated info
                        if metrics_path and "subjectwise_stats" in metrics_data:
                            subj_stats = metrics_data.get("subjectwise_stats", {})
                            if "test" in subj_stats and "preds" in subj_stats["test"]:
                                preds = subj_stats["test"]["preds"]
                                y_pred_proba = preds.get("y_pred_proba", {})
                                y_pred_mean = y_pred_proba.get("mean", np.array([]))

                                for i, (code, y_true_val) in enumerate(
                                    zip(subject_codes_test, y_test)
                                ):
                                    prob = (
                                        y_pred_mean[i] if i < len(y_pred_mean) else 0.5
                                    )
                                    all_predictions.append(
                                        {
                                            "prediction_id": prediction_id,
                                            "subject_id": str(code),
                                            "eye": "OD",  # Default
                                            "fold": 0,  # Bootstrap aggregated
                                            "bootstrap_iter": 0,
                                            "outlier_method": config.get(
                                                "outlier_method", ""
                                            ),
                                            "imputation_method": config.get(
                                                "imputation_method", ""
                                            ),
                                            "featurization": config.get(
                                                "featurization", ""
                                            ),
                                            "classifier": config.get("classifier", ""),
                                            "source_name": config.get(
                                                "source_name", run_name
                                            ),
                                            "y_true": int(y_true_val),
                                            "y_pred": int(prob >= 0.5),
                                            "y_prob": float(prob),
                                            "mlflow_run_id": run_id,
                                        }
                                    )
                                    prediction_id += 1

                                # Compute STRATOS metrics from predictions
                                y_true_arr = np.array(y_test)
                                y_prob_arr = (
                                    np.array(y_pred_mean)
                                    if len(y_pred_mean) > 0
                                    else np.array([])
                                )

                                if len(y_true_arr) > 10 and len(y_prob_arr) == len(
                                    y_true_arr
                                ):
                                    stratos_metrics = _compute_stratos_metrics(
                                        y_true_arr, y_prob_arr
                                    )

                                    # Update the last fold metric with STRATOS values
                                    # (Store as aggregate since we have one y_true/y_prob set)
                                    if all_metrics:
                                        # Find metrics for this run and update them
                                        source = config.get("source_name", run_name)
                                        for m in all_metrics:
                                            if m["source_name"] == source:
                                                m["calibration_slope"] = (
                                                    stratos_metrics.get(
                                                        "calibration_slope"
                                                    )
                                                )
                                                m["calibration_intercept"] = (
                                                    stratos_metrics.get(
                                                        "calibration_intercept"
                                                    )
                                                )
                                                m["e_o_ratio"] = stratos_metrics.get(
                                                    "e_o_ratio"
                                                )
                                                m["net_benefit_5pct"] = (
                                                    stratos_metrics.get(
                                                        "net_benefit_5pct"
                                                    )
                                                )
                                                m["net_benefit_10pct"] = (
                                                    stratos_metrics.get(
                                                        "net_benefit_10pct"
                                                    )
                                                )
                                                m["net_benefit_20pct"] = (
                                                    stratos_metrics.get(
                                                        "net_benefit_20pct"
                                                    )
                                                )

                                    # Compute DCA curves
                                    try:
                                        from ..stats.clinical_utility import (
                                            decision_curve_analysis,
                                        )

                                        dca_df = decision_curve_analysis(
                                            y_true_arr,
                                            y_prob_arr,
                                            threshold_range=(0.01, 0.50),
                                            n_thresholds=50,
                                        )

                                        source = config.get("source_name", run_name)
                                        classifier = config.get("classifier", "unknown")

                                        for _, row in dca_df.iterrows():
                                            all_dca_curves.append(
                                                {
                                                    "dca_id": dca_id,
                                                    "source_name": source,
                                                    "classifier": classifier,
                                                    "threshold": float(
                                                        row["threshold"]
                                                    ),
                                                    "net_benefit_model": float(
                                                        row["nb_model"]
                                                    ),
                                                    "net_benefit_all": float(
                                                        row["nb_all"]
                                                    ),
                                                    "net_benefit_none": float(
                                                        row["nb_none"]
                                                    ),
                                                    "sensitivity": float(
                                                        row["sensitivity"]
                                                    ),
                                                    "specificity": float(
                                                        row["specificity"]
                                                    ),
                                                }
                                            )
                                            dca_id += 1

                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to compute DCA curves: {e}"
                                        )

            except Exception as e:
                logger.warning(f"Error processing run {run_dir.name}: {e}")
                continue

        # Cleanup after each batch
        gc.collect()
        logger.info(
            f"Processed {min(batch_start + batch_size, len(run_dirs))}/{len(run_dirs)} runs"
        )

    predictions_df = (
        pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()
    )
    metrics_per_fold_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    metrics_aggregate_df = (
        pd.DataFrame(all_aggregate) if all_aggregate else pd.DataFrame()
    )
    dca_curves_df = pd.DataFrame(all_dca_curves) if all_dca_curves else pd.DataFrame()
    mlflow_runs_df = pd.DataFrame(mlflow_runs) if mlflow_runs else pd.DataFrame()

    logger.info(
        f"Extracted {len(predictions_df)} predictions, "
        f"{len(metrics_per_fold_df)} fold metrics, "
        f"{len(metrics_aggregate_df)} aggregate metrics, "
        f"{len(dca_curves_df)} DCA curve points"
    )

    return (
        predictions_df,
        metrics_per_fold_df,
        metrics_aggregate_df,
        dca_curves_df,
        mlflow_runs_df,
    )


def _find_classification_experiment(mlruns_dir: Path) -> str:
    """Find the experiment ID with most runs (likely classification).

    Parameters
    ----------
    mlruns_dir : Path
        Path to the mlruns directory.

    Returns
    -------
    str
        Experiment ID with the most runs.

    Raises
    ------
    ValueError
        If no experiments are found in the directory.
    """
    max_runs = 0
    best_exp = None

    for exp_dir in mlruns_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in (".trash", "models", "0"):
            run_count = sum(
                1
                for d in exp_dir.iterdir()
                if d.is_dir() and (d / "meta.yaml").exists()
            )
            if run_count > max_runs:
                max_runs = run_count
                best_exp = exp_dir.name

    if best_exp is None:
        raise ValueError("No experiments found")

    logger.info(f"Selected experiment {best_exp} with {max_runs} runs")
    return best_exp


def _parse_run_name(run_name: str) -> Dict[str, str]:
    """Parse configuration from MLflow run name.

    Extracts classifier, featurization, imputation, and outlier method
    from the run name format: classifier_metric__featurization__imputation__outlier.

    Parameters
    ----------
    run_name : str
        MLflow run name string.

    Returns
    -------
    Dict[str, str]
        Dictionary containing source_name, classifier, featurization,
        imputation_method, and outlier_method.
    """
    config = {"source_name": run_name}

    parts = run_name.split("__")
    if len(parts) >= 1:
        # First part is typically classifier_metric
        clf_part = parts[0].split("_")
        config["classifier"] = clf_part[0]

    if len(parts) >= 2:
        config["featurization"] = parts[1]

    if len(parts) >= 3:
        config["imputation_method"] = parts[2]

    if len(parts) >= 4:
        config["outlier_method"] = parts[3]

    return config


def _find_artifact(run_dir: Path, subdir: str, pattern: str) -> Optional[Path]:
    """Find artifact file in MLflow run directory.

    Parameters
    ----------
    run_dir : Path
        Path to the MLflow run directory.
    subdir : str
        Subdirectory within artifacts (e.g., "metrics", "dict_arrays").
    pattern : str
        Glob pattern for matching files (e.g., "*.pickle").

    Returns
    -------
    Optional[Path]
        Path to the first matching artifact file, or None if not found.
    """
    artifacts_dir = run_dir / "artifacts" / subdir
    if not artifacts_dir.exists():
        return None

    matches = list(artifacts_dir.glob(pattern))
    return matches[0] if matches else None


def export_mlflow_to_duckdb(
    mlruns_dir: Union[str, Path],
    output_path: Union[str, Path],
    experiment_id: Optional[str] = None,
) -> Path:
    """
    Export MLflow classification results to DuckDB.

    Parameters
    ----------
    mlruns_dir : str or Path
        Path to mlruns directory
    output_path : str or Path
        Output .db file path
    experiment_id : str, optional
        Specific experiment ID

    Returns
    -------
    Path
        Path to created database
    """
    (
        predictions_df,
        metrics_per_fold_df,
        metrics_aggregate_df,
        dca_curves_df,
        mlflow_runs_df,
    ) = extract_mlflow_classification_runs(mlruns_dir, experiment_id)

    return export_results_to_duckdb(
        predictions_df=predictions_df,
        metrics_per_fold=metrics_per_fold_df,
        metrics_aggregate=metrics_aggregate_df,
        output_path=output_path,
        dca_curves=dca_curves_df,
        mlflow_runs=mlflow_runs_df,
    )


# ============================================================================
# CLI
# ============================================================================


def main():
    """Command-line interface for DuckDB export/analysis.

    Supports two main commands:
    - export: Export MLflow runs to a DuckDB database
    - analyze: Run analysis from existing features or results databases

    Returns
    -------
    None
        Executes CLI commands and writes output to files.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Export features/results to DuckDB or run analysis from artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export from mlruns")
    export_parser.add_argument(
        "--mlruns", required=True, help="Path to mlruns directory"
    )
    export_parser.add_argument("--experiment-id", help="Specific experiment ID")
    export_parser.add_argument(
        "--output", default="foundation_plr_results.db", help="Output database file"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run analysis from artifacts"
    )
    analyze_parser.add_argument("--from-features", help="Path to features.db")
    analyze_parser.add_argument("--from-results", help="Path to results.db")
    analyze_parser.add_argument("--output-dir", default="outputs/analysis")

    args = parser.parse_args()

    if args.command == "export":
        logger.info(f"Export from {args.mlruns}")
        output_path = export_mlflow_to_duckdb(
            args.mlruns,
            args.output,
            args.experiment_id,
        )
        logger.info(f"Exported to {output_path}")

    elif args.command == "analyze":
        if args.from_features:
            pipeline = DuckDBAnalysisPipeline.from_features(
                args.from_features, output_dir=args.output_dir
            )
            pipeline.run_classification()
            pipeline.run_statistics()

        elif args.from_results:
            pipeline = DuckDBAnalysisPipeline.from_results(
                args.from_results, output_dir=args.output_dir
            )
            pipeline.run_statistics()

        else:
            logger.error("Specify --from-features or --from-results")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
