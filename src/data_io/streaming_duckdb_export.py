"""
Memory-efficient streaming export to DuckDB with checkpointing.

Fixes the 62GB RAM crash by:
1. Streaming extraction - write directly to DuckDB per run
2. Progress checkpointing - resume from any crash point
3. Memory monitoring - explicit warnings and throttling
4. Two-tier storage - essential vs supplementary metrics

Cross-references:
- planning/remaining-duckdb-stats-viz-tasks-plan.md (Part 1, 2B)
- src/data_io/duckdb_export.py (original implementation)

Usage:
    exporter = StreamingDuckDBExporter("mlruns", "results.db")
    exporter.export()  # Can be interrupted and resumed
"""

import gc
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb
import numpy as np
import psutil
from loguru import logger

# Timing thresholds for performance monitoring
SLOW_RUN_THRESHOLD_SECONDS = 30.0  # Warn if single run takes > 30s
EXPECTED_RUNS_PER_MINUTE = 10.0  # Warn if slower than this rate

__all__ = [
    "StreamingDuckDBExporter",
    "CheckpointManager",
    "MemoryMonitor",
]


# ============================================================================
# Schema for Essential Metrics (Van Calster 2024 STRATOS compliant)
# ============================================================================

ESSENTIAL_METRICS_SCHEMA = """
-- Essential metrics table (Van Calster 2024 standard)
CREATE TABLE IF NOT EXISTS essential_metrics (
    config_id INTEGER PRIMARY KEY,
    -- Configuration
    outlier_method TEXT NOT NULL,
    imputation_method TEXT NOT NULL,
    classifier TEXT NOT NULL,
    source_name TEXT NOT NULL,
    mlflow_run_id TEXT,
    -- Discrimination
    auroc REAL,
    auroc_ci_lower REAL,
    auroc_ci_upper REAL,
    -- Calibration
    calibration_slope REAL,
    calibration_intercept REAL,
    o_e_ratio REAL,
    brier_score REAL,
    -- Clinical utility
    net_benefit_5pct REAL,
    net_benefit_10pct REAL,
    net_benefit_15pct REAL,
    net_benefit_20pct REAL,
    -- Sample info
    n_test INTEGER,
    n_events INTEGER,
    prevalence REAL,
    -- Extraction metadata
    extracted_at TEXT
);

-- Supplementary metrics (improper per Van Calster, but kept for STRATOS appendix)
CREATE TABLE IF NOT EXISTS supplementary_metrics (
    config_id INTEGER PRIMARY KEY,
    -- These are NOT primary outcomes per STRATOS
    f1_score REAL,
    accuracy REAL,
    aupr REAL,
    sensitivity REAL,
    specificity REAL,
    ppv REAL,
    npv REAL,
    -- Reference: essential_metrics.config_id
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Calibration curves (10 bins per config) - auto-increment ID
CREATE SEQUENCE IF NOT EXISTS seq_calibration_curve_id START 1;
CREATE TABLE IF NOT EXISTS calibration_curves (
    curve_id INTEGER PRIMARY KEY DEFAULT nextval('seq_calibration_curve_id'),
    config_id INTEGER NOT NULL,
    bin_index INTEGER NOT NULL,
    bin_midpoint REAL,
    observed_proportion REAL,
    predicted_mean REAL,
    n_samples INTEGER,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Probability distributions (STRATOS required) - auto-increment ID
CREATE SEQUENCE IF NOT EXISTS seq_prob_dist_id START 1;
CREATE TABLE IF NOT EXISTS probability_distributions (
    dist_id INTEGER PRIMARY KEY DEFAULT nextval('seq_prob_dist_id'),
    config_id INTEGER NOT NULL,
    outcome INTEGER NOT NULL CHECK (outcome IN (0, 1)),
    bin_index INTEGER NOT NULL,
    bin_start REAL,
    bin_end REAL,
    count INTEGER,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Decision Curve Analysis (STRATOS clinical utility) - auto-increment ID
CREATE SEQUENCE IF NOT EXISTS seq_dca_id START 1;
CREATE TABLE IF NOT EXISTS dca_curves (
    dca_id INTEGER PRIMARY KEY DEFAULT nextval('seq_dca_id'),
    config_id INTEGER NOT NULL,
    threshold REAL NOT NULL,
    net_benefit_model REAL,
    net_benefit_all REAL,
    net_benefit_none REAL,
    sensitivity REAL,
    specificity REAL,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Subject-level predictions (for per-patient uncertainty analysis)
-- Subject codes may be re-anonymized (Hxxx/Gxxx) for privacy
CREATE SEQUENCE IF NOT EXISTS seq_pred_id START 1;
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY DEFAULT nextval('seq_pred_id'),
    config_id INTEGER NOT NULL,
    subject_code TEXT NOT NULL,
    y_true INTEGER NOT NULL CHECK (y_true IN (0, 1)),
    y_prob REAL NOT NULL,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Retention metrics (selective classification curves)
-- Pre-computed metric values at various retention rates for visualization
CREATE SEQUENCE IF NOT EXISTS seq_retention_id START 1;
CREATE TABLE IF NOT EXISTS retention_metrics (
    retention_id INTEGER PRIMARY KEY DEFAULT nextval('seq_retention_id'),
    config_id INTEGER NOT NULL,
    retention_rate REAL NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Cohort metrics (metric vs cohort fraction curves)
-- Pre-computed metric values at various cohort fractions
CREATE SEQUENCE IF NOT EXISTS seq_cohort_id START 1;
CREATE TABLE IF NOT EXISTS cohort_metrics (
    cohort_id INTEGER PRIMARY KEY DEFAULT nextval('seq_cohort_id'),
    config_id INTEGER NOT NULL,
    cohort_fraction REAL NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Distribution statistics (summary stats per config for prob_distribution viz)
CREATE TABLE IF NOT EXISTS distribution_stats (
    config_id INTEGER PRIMARY KEY,
    auroc REAL,
    median_cases REAL,
    median_controls REAL,
    mean_cases REAL,
    mean_controls REAL,
    n_cases INTEGER,
    n_controls INTEGER,
    FOREIGN KEY (config_id) REFERENCES essential_metrics(config_id)
);

-- Checkpoint tracking (run_id is the primary key)
-- NOTE: No index on status to avoid DuckDB ON CONFLICT limitations
CREATE TABLE IF NOT EXISTS extraction_checkpoints (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT
);

-- Indices for efficient queries
CREATE INDEX IF NOT EXISTS idx_essential_config ON essential_metrics(
    outlier_method, imputation_method, classifier
);
CREATE INDEX IF NOT EXISTS idx_calibration_config ON calibration_curves(config_id);
CREATE INDEX IF NOT EXISTS idx_prob_dist_config ON probability_distributions(config_id);
CREATE INDEX IF NOT EXISTS idx_dca_config ON dca_curves(config_id);
CREATE INDEX IF NOT EXISTS idx_predictions_config ON predictions(config_id);
CREATE INDEX IF NOT EXISTS idx_predictions_subject ON predictions(subject_code);
CREATE INDEX IF NOT EXISTS idx_retention_config ON retention_metrics(config_id);
CREATE INDEX IF NOT EXISTS idx_retention_metric ON retention_metrics(config_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_cohort_config ON cohort_metrics(config_id);
CREATE INDEX IF NOT EXISTS idx_cohort_metric ON cohort_metrics(config_id, metric_name);
"""


# ============================================================================
# Memory Monitoring
# ============================================================================


@dataclass
class MemoryMonitor:
    """
    Monitor memory usage and warn/throttle if necessary.

    Attributes
    ----------
    warning_threshold_gb : float
        Warn when memory exceeds this (default: 12GB)
    critical_threshold_gb : float
        Force GC and pause when exceeding this (default: 14GB)
    """

    warning_threshold_gb: float = 12.0
    critical_threshold_gb: float = 14.0
    _warned: bool = field(default=False, repr=False)

    def check(self) -> Tuple[float, str]:
        """
        Check current memory usage.

        Returns
        -------
        tuple of (usage_gb, status)
            status is 'ok', 'warning', or 'critical'
        """
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 / 1024 / 1024

        if mem_gb >= self.critical_threshold_gb:
            return mem_gb, "critical"
        elif mem_gb >= self.warning_threshold_gb:
            return mem_gb, "warning"
        else:
            return mem_gb, "ok"

    def enforce(self) -> None:
        """
        Check memory and take action if necessary.

        At warning level: Log warning
        At critical level: Force garbage collection
        """
        mem_gb, status = self.check()

        if status == "critical":
            logger.warning(f"CRITICAL: Memory at {mem_gb:.2f}GB. Forcing GC...")
            gc.collect()
            # Re-check after GC
            mem_gb, status = self.check()
            if status == "critical":
                logger.error(
                    f"Memory still at {mem_gb:.2f}GB after GC. Consider reducing batch size."
                )

        elif status == "warning" and not self._warned:
            logger.warning(
                f"Memory at {mem_gb:.2f}GB (threshold: {self.warning_threshold_gb}GB)"
            )
            self._warned = True

        elif status == "ok":
            self._warned = False

    def get_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.check()[0]


# ============================================================================
# Checkpoint Management
# ============================================================================


@dataclass
class CheckpointManager:
    """
    Manage extraction checkpoints for crash recovery.

    Tracks which runs have been processed and allows resumption.
    """

    db_path: Path
    _completed_runs: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self._load_completed_runs()

    def _load_completed_runs(self) -> None:
        """Load set of completed run IDs from database."""
        if not self.db_path.exists():
            return

        try:
            with duckdb.connect(str(self.db_path)) as con:
                result = con.execute("""
                    SELECT run_id FROM extraction_checkpoints
                    WHERE status = 'completed'
                """).fetchall()
                self._completed_runs = {row[0] for row in result}
        except Exception:
            # Table may not exist yet
            pass

    def is_completed(self, run_id: str) -> bool:
        """Check if a run has already been extracted."""
        return run_id in self._completed_runs

    def mark_started(self, run_id: str, con: duckdb.DuckDBPyConnection) -> None:
        """Mark a run as started (in progress).

        Uses INSERT ... ON CONFLICT DO UPDATE for upsert semantics.
        Note: Status index was removed from schema to allow this pattern.
        """
        now = datetime.now().isoformat()
        con.execute(
            """
            INSERT INTO extraction_checkpoints (run_id, status, started_at)
            VALUES (?, 'started', ?)
            ON CONFLICT (run_id) DO UPDATE SET
                status = 'started',
                started_at = excluded.started_at
        """,
            [run_id, now],
        )

    def mark_completed(self, run_id: str, con: duckdb.DuckDBPyConnection) -> None:
        """Mark a run as successfully completed.

        Uses INSERT ... ON CONFLICT DO UPDATE for upsert semantics.
        """
        now = datetime.now().isoformat()
        con.execute(
            """
            INSERT INTO extraction_checkpoints (run_id, status, started_at, completed_at)
            VALUES (?, 'completed', ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                status = 'completed',
                completed_at = excluded.completed_at
        """,
            [run_id, now, now],
        )
        self._completed_runs.add(run_id)

    def mark_failed(
        self, run_id: str, error: str, con: duckdb.DuckDBPyConnection
    ) -> None:
        """Mark a run as failed with error message.

        Uses INSERT ... ON CONFLICT DO UPDATE for upsert semantics.
        """
        now = datetime.now().isoformat()
        con.execute(
            """
            INSERT INTO extraction_checkpoints
            (run_id, status, started_at, completed_at, error_message)
            VALUES (?, 'failed', ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                status = 'failed',
                completed_at = excluded.completed_at,
                error_message = excluded.error_message
        """,
            [run_id, now, now, error],
        )

    def get_progress(self) -> Dict[str, int]:
        """Get extraction progress statistics."""
        if not self.db_path.exists():
            return {"completed": 0, "failed": 0, "in_progress": 0}

        with duckdb.connect(str(self.db_path)) as con:
            result = con.execute("""
                SELECT status, COUNT(*) as cnt
                FROM extraction_checkpoints
                GROUP BY status
            """).fetchall()

        stats = {"completed": 0, "failed": 0, "started": 0}
        for status, cnt in result:
            stats[status] = cnt

        return stats


# ============================================================================
# Streaming Exporter
# ============================================================================


class StreamingDuckDBExporter:
    """
    Memory-efficient streaming export to DuckDB.

    Writes directly to DuckDB per run instead of accumulating in memory.

    Parameters
    ----------
    mlruns_dir : str or Path
        Path to mlruns directory
    output_path : str or Path
        Output .db file path
    experiment_id : str, optional
        Specific experiment ID (auto-detects if not provided)
    memory_threshold_gb : float, default 12.0
        Warning threshold for memory usage
    subject_mapping : dict, optional
        Mapping from original subject codes to anonymized codes (e.g., PLRxxxx -> Hxxx).
        If provided, subject codes in predictions table will be re-anonymized.
    extract_predictions : bool, default True
        Whether to extract subject-level predictions (for per-patient analysis).

    Examples
    --------
    >>> exporter = StreamingDuckDBExporter("mlruns", "results.db")
    >>> exporter.export()
    >>> # Can interrupt and resume:
    >>> exporter.export()  # Skips already-processed runs
    >>>
    >>> # With re-anonymization:
    >>> mapping = {"PLR001": "H001", "PLR002": "G001"}
    >>> exporter = StreamingDuckDBExporter("mlruns", "results.db", subject_mapping=mapping)
    >>> exporter.export()  # Predictions use Hxxx/Gxxx codes
    """

    def __init__(
        self,
        mlruns_dir: Union[str, Path],
        output_path: Union[str, Path],
        experiment_id: Optional[str] = None,
        memory_threshold_gb: float = 12.0,
        subject_mapping: Optional[Dict[str, str]] = None,
        extract_predictions: bool = True,
    ):
        self.mlruns_dir = Path(mlruns_dir)
        self.output_path = Path(output_path)
        self.experiment_id = experiment_id
        self.subject_mapping = subject_mapping or {}
        self.extract_predictions = extract_predictions

        self.memory_monitor = MemoryMonitor(
            warning_threshold_gb=memory_threshold_gb,
            critical_threshold_gb=memory_threshold_gb + 2.0,
        )
        self.checkpoint_manager = CheckpointManager(self.output_path)

        # Auto-detect experiment if not provided
        if self.experiment_id is None:
            self.experiment_id = self._find_classification_experiment()

    def _find_classification_experiment(self) -> str:
        """Find the experiment ID with most runs."""
        max_runs = 0
        best_exp = None

        for exp_dir in self.mlruns_dir.iterdir():
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
            raise ValueError("No experiments found in mlruns directory")

        logger.info(f"Selected experiment {best_exp} with {max_runs} runs")
        return best_exp

    def _init_database(self) -> None:
        """Initialize database schema if needed."""
        with duckdb.connect(str(self.output_path)) as con:
            con.execute(ESSENTIAL_METRICS_SCHEMA)

    def _get_run_dirs(self) -> List[Path]:
        """Get list of run directories to process."""
        exp_dir = self.mlruns_dir / self.experiment_id

        run_dirs = []
        for d in exp_dir.iterdir():
            if d.is_dir() and (d / "meta.yaml").exists():
                run_dirs.append(d)

        return sorted(run_dirs, key=lambda x: x.name)

    def _parse_run_name(self, run_name: str) -> Dict[str, str]:
        """Parse configuration from run name."""
        config = {"source_name": run_name}

        parts = run_name.split("__")
        if len(parts) >= 1:
            clf_part = parts[0].split("_")
            config["classifier"] = clf_part[0]

        if len(parts) >= 2:
            config["featurization"] = parts[1]

        if len(parts) >= 3:
            config["imputation_method"] = parts[2]

        if len(parts) >= 4:
            config["outlier_method"] = parts[3]

        return config

    def _extract_run(
        self,
        run_dir: Path,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> bool:
        """
        Extract a single run and write directly to DuckDB.

        Returns True if successful, False otherwise.
        """
        run_id = run_dir.name

        # Parse meta.yaml for run info
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists():
            return False

        try:
            import yaml

            with open(meta_path) as f:
                meta = yaml.safe_load(f)
        except Exception:
            meta = {}

        run_name = meta.get("run_name", run_dir.name)
        config = self._parse_run_name(run_name)

        # Find metrics pickle
        metrics_path = self._find_artifact(run_dir, "metrics", "*.pickle")
        if not metrics_path:
            logger.debug(f"No metrics found for {run_name}")
            return False

        # Load metrics (this is the only large object we load)
        try:
            with open(metrics_path, "rb") as f:
                metrics_data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metrics for {run_name}: {e}")
            return False

        # Extract essential metrics
        essential = self._extract_essential_metrics(metrics_data, config)
        essential["config_id"] = config_id
        essential["mlflow_run_id"] = run_id
        essential["source_name"] = config.get("source_name", run_name)
        essential["outlier_method"] = config.get("outlier_method", "unknown")
        essential["imputation_method"] = config.get("imputation_method", "unknown")
        essential["classifier"] = config.get("classifier", "unknown")
        essential["extracted_at"] = datetime.now().isoformat()

        # Insert essential metrics
        self._insert_essential_metrics(essential, con)

        # Extract supplementary metrics
        supplementary = self._extract_supplementary_metrics(metrics_data, config)
        supplementary["config_id"] = config_id
        self._insert_supplementary_metrics(supplementary, con)

        # Extract calibration curve data
        self._extract_calibration_curve(metrics_data, config_id, con)

        # Extract probability distribution and compute net benefit if missing
        nb_from_raw = self._extract_probability_distribution(run_dir, config_id, con)

        # Update metrics if they were computed from raw predictions
        # and not already in the essential metrics (includes STRATOS calibration)
        if nb_from_raw:
            update_fields = [
                "net_benefit_5pct",
                "net_benefit_10pct",
                "net_benefit_15pct",
                "net_benefit_20pct",
                "calibration_slope",
                "calibration_intercept",
                "o_e_ratio",
            ]

            update_needed = False
            for key in update_fields:
                if essential.get(key) is None and nb_from_raw.get(key) is not None:
                    update_needed = True
                    break

            if update_needed:
                con.execute(
                    """
                    UPDATE essential_metrics
                    SET net_benefit_5pct = COALESCE(net_benefit_5pct, ?),
                        net_benefit_10pct = COALESCE(net_benefit_10pct, ?),
                        net_benefit_15pct = COALESCE(net_benefit_15pct, ?),
                        net_benefit_20pct = COALESCE(net_benefit_20pct, ?),
                        calibration_slope = COALESCE(calibration_slope, ?),
                        calibration_intercept = COALESCE(calibration_intercept, ?),
                        o_e_ratio = COALESCE(o_e_ratio, ?)
                    WHERE config_id = ?
                """,
                    [
                        nb_from_raw.get("net_benefit_5pct"),
                        nb_from_raw.get("net_benefit_10pct"),
                        nb_from_raw.get("net_benefit_15pct"),
                        nb_from_raw.get("net_benefit_20pct"),
                        nb_from_raw.get("calibration_slope"),
                        nb_from_raw.get("calibration_intercept"),
                        nb_from_raw.get("o_e_ratio"),
                        config_id,
                    ],
                )

        # Extract subject-level predictions with optional re-anonymization
        self._extract_predictions(metrics_data, config_id, con)

        # Cleanup
        del metrics_data
        gc.collect()

        return True

    def _extract_essential_metrics(
        self,
        metrics_data: Dict,
        config: Dict[str, str],
    ) -> Dict[str, Any]:
        """Extract Van Calster 2024 essential metrics."""
        result = {
            "auroc": None,
            "auroc_ci_lower": None,
            "auroc_ci_upper": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "o_e_ratio": None,
            "brier_score": None,
            "net_benefit_5pct": None,
            "net_benefit_10pct": None,
            "net_benefit_15pct": None,
            "net_benefit_20pct": None,
            "n_test": None,
            "n_events": None,
            "prevalence": None,
        }

        # Navigate to test metrics
        if "metrics_stats" not in metrics_data:
            return result

        test_stats = metrics_data["metrics_stats"].get("test", {})
        if "metrics" not in test_stats or "scalars" not in test_stats["metrics"]:
            return result

        scalars = test_stats["metrics"]["scalars"]

        # AUROC
        if "AUROC" in scalars:
            auroc_data = scalars["AUROC"]
            result["auroc"] = auroc_data.get("mean")
            ci = auroc_data.get("ci", [None, None])
            if isinstance(ci, (list, np.ndarray)) and len(ci) >= 2:
                result["auroc_ci_lower"] = ci[0]
                result["auroc_ci_upper"] = ci[1]

        # Brier score
        if "Brier" in scalars:
            result["brier_score"] = scalars["Brier"].get("mean")

        # Calibration metrics (if computed)
        if "CalibrationSlope" in scalars:
            result["calibration_slope"] = scalars["CalibrationSlope"].get("mean")
        if "CalibrationIntercept" in scalars:
            result["calibration_intercept"] = scalars["CalibrationIntercept"].get(
                "mean"
            )
        if "OE_Ratio" in scalars:
            result["o_e_ratio"] = scalars["OE_Ratio"].get("mean")

        # Sample info
        if "N_Test" in scalars:
            result["n_test"] = int(scalars["N_Test"].get("mean", 0))
        if "N_Events" in scalars:
            result["n_events"] = int(scalars["N_Events"].get("mean", 0))
        if "Prevalence" in scalars:
            result["prevalence"] = scalars["Prevalence"].get("mean")

        # Net benefit at different thresholds
        # Try to extract from pre-computed values first
        for threshold, key in [
            (5, "net_benefit_5pct"),
            (10, "net_benefit_10pct"),
            (15, "net_benefit_15pct"),
            (20, "net_benefit_20pct"),
        ]:
            nb_key = f"NetBenefit_{threshold}pct"
            if nb_key in scalars:
                result[key] = scalars[nb_key].get("mean")
            else:
                # Alternative naming conventions
                alt_keys = [
                    f"NB_{threshold}",
                    f"net_benefit_{threshold}",
                    f"NetBenefit{threshold}",
                ]
                for alt_key in alt_keys:
                    if alt_key in scalars:
                        result[key] = scalars[alt_key].get("mean")
                        break

        return result

    def _extract_supplementary_metrics(
        self,
        metrics_data: Dict,
        config: Dict[str, str],
    ) -> Dict[str, Any]:
        """Extract supplementary (improper per STRATOS) metrics."""
        result = {
            "f1_score": None,
            "accuracy": None,
            "aupr": None,
            "sensitivity": None,
            "specificity": None,
            "ppv": None,
            "npv": None,
        }

        if "metrics_stats" not in metrics_data:
            return result

        test_stats = metrics_data["metrics_stats"].get("test", {})
        if "metrics" not in test_stats or "scalars" not in test_stats["metrics"]:
            return result

        scalars = test_stats["metrics"]["scalars"]

        metric_mapping = {
            "F1": "f1_score",
            "Accuracy": "accuracy",
            "AUPR": "aupr",
            "Sensitivity": "sensitivity",
            "Recall": "sensitivity",
            "Specificity": "specificity",
            "PPV": "ppv",
            "Precision": "ppv",
            "NPV": "npv",
        }

        for src_name, dst_name in metric_mapping.items():
            if src_name in scalars:
                result[dst_name] = scalars[src_name].get("mean")

        return result

    def _extract_calibration_curve(
        self,
        metrics_data: Dict,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Extract calibration curve data (10 bins)."""
        # Check if calibration data exists in metrics
        if "metrics_stats" not in metrics_data:
            return

        test_stats = metrics_data["metrics_stats"].get("test", {})
        if "metrics" not in test_stats:
            return

        metrics = test_stats["metrics"]

        # Check for calibration curve arrays
        calibration_data = None

        # Try different possible locations for calibration data
        if "arrays" in metrics:
            arrays = metrics["arrays"]
            # Look for calibration bins data
            if "calibration_bins" in arrays:
                calibration_data = arrays["calibration_bins"]
            elif "CalibrationBins" in arrays:
                calibration_data = arrays["CalibrationBins"]

        # Alternative: Check for calibration in scalars format
        if calibration_data is None and "scalars" in metrics:
            scalars = metrics["scalars"]
            # Construct from individual bin entries if they exist
            bin_midpoints = []
            observed = []
            predicted = []
            n_samples = []

            for i in range(10):
                mid_key = f"cal_bin_{i}_midpoint"
                obs_key = f"cal_bin_{i}_observed"
                pred_key = f"cal_bin_{i}_predicted"
                n_key = f"cal_bin_{i}_n"

                if mid_key in scalars:
                    bin_midpoints.append(scalars[mid_key].get("mean", (i + 0.5) / 10))
                    observed.append(scalars.get(obs_key, {}).get("mean"))
                    predicted.append(scalars.get(pred_key, {}).get("mean"))
                    n_samples.append(int(scalars.get(n_key, {}).get("mean", 0)))

            if bin_midpoints:
                calibration_data = {
                    "bin_midpoints": bin_midpoints,
                    "observed": observed,
                    "predicted": predicted,
                    "n_samples": n_samples,
                }

        if calibration_data is None:
            return

        # Insert calibration curve data
        try:
            if isinstance(calibration_data, dict):
                midpoints = calibration_data.get("bin_midpoints", [])
                observed = calibration_data.get("observed", [])
                predicted = calibration_data.get("predicted", [])
                n_samples = calibration_data.get("n_samples", [])

                for i, (mid, obs, pred, n) in enumerate(
                    zip(midpoints, observed, predicted, n_samples)
                ):
                    if mid is not None:
                        con.execute(
                            """
                            INSERT INTO calibration_curves
                            (config_id, bin_index, bin_midpoint, observed_proportion,
                             predicted_mean, n_samples)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """,
                            [config_id, i, mid, obs, pred, int(n) if n else None],
                        )
        except Exception as e:
            logger.debug(f"Could not insert calibration data: {e}")

    def _compute_net_benefit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Compute net benefit at a given threshold.

        Net benefit = (TP/N) - (FP/N) × (threshold / (1 - threshold))

        Parameters
        ----------
        y_true : np.ndarray
            True binary outcomes
        y_prob : np.ndarray
            Predicted probabilities
        threshold : float
            Decision threshold (e.g., 0.05, 0.10, 0.20)

        Returns
        -------
        float
            Net benefit value
        """
        n = len(y_true)
        if n == 0:
            return 0.0

        # Make predictions at threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate TP and FP
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Net benefit formula
        if threshold <= 0.0 or threshold >= 1.0:
            return 0.0

        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        return float(net_benefit)

    def _extract_probability_distribution(
        self,
        run_dir: Path,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> Optional[Dict[str, float]]:
        """
        Extract probability distribution by outcome class.

        Also returns net benefit values computed from raw predictions.
        """
        # Find arrays pickle for predictions
        arrays_path = self._find_artifact(run_dir, "dict_arrays", "*.pickle")
        if not arrays_path:
            return None

        try:
            with open(arrays_path, "rb") as f:
                arrays_data = pickle.load(f)
        except Exception:
            return None

        y_test = arrays_data.get("y_test", np.array([]))
        y_prob = arrays_data.get("y_pred_proba_mean", np.array([]))

        if len(y_test) == 0 or len(y_prob) == 0:
            del arrays_data
            return None

        # Compute histogram for each outcome class
        bin_edges = np.linspace(0, 1, 11)  # 10 bins

        for outcome in [0, 1]:
            mask = y_test == outcome
            if not np.any(mask):
                continue

            probs = y_prob[mask]
            counts, _ = np.histogram(probs, bins=bin_edges)

            for bin_idx, count in enumerate(counts):
                con.execute(
                    """
                    INSERT INTO probability_distributions
                    (config_id, outcome, bin_index, bin_start, bin_end, count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        config_id,
                        outcome,
                        bin_idx,
                        bin_edges[bin_idx],
                        bin_edges[bin_idx + 1],
                        int(count),
                    ],
                )

        # Compute net benefit at clinical thresholds
        net_benefits = {
            "net_benefit_5pct": self._compute_net_benefit(y_test, y_prob, 0.05),
            "net_benefit_10pct": self._compute_net_benefit(y_test, y_prob, 0.10),
            "net_benefit_15pct": self._compute_net_benefit(y_test, y_prob, 0.15),
            "net_benefit_20pct": self._compute_net_benefit(y_test, y_prob, 0.20),
        }

        # Compute STRATOS calibration metrics from raw predictions
        stratos_metrics = self._compute_stratos_metrics_from_raw(y_test, y_prob)
        net_benefits.update(stratos_metrics)

        # Extract DCA curves (STRATOS clinical utility)
        self._extract_dca_curves(y_test, y_prob, config_id, con)

        # Extract distribution statistics
        self._extract_distribution_stats(y_test, y_prob, config_id, con)

        # Extract retention metrics (selective classification curves)
        uncertainty = arrays_data.get("y_pred_proba_std", None)
        if uncertainty is not None and len(uncertainty) > 0:
            self._extract_retention_metrics(y_test, y_prob, uncertainty, config_id, con)
            self._extract_cohort_metrics(y_test, y_prob, uncertainty, config_id, con)

        del arrays_data
        gc.collect()

        return net_benefits

    def _extract_dca_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """
        Extract Decision Curve Analysis data.

        Computes DCA at multiple thresholds for clinical utility assessment.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes
        y_prob : np.ndarray
            Predicted probabilities
        config_id : int
            Configuration ID for database
        con : duckdb.DuckDBPyConnection
            Database connection
        """
        from src.stats.clinical_utility import decision_curve_analysis

        try:
            # Compute DCA curve with clinically relevant threshold range
            dca_df = decision_curve_analysis(
                y_true, y_prob, threshold_range=(0.01, 0.50), n_thresholds=50
            )

            # Insert DCA curve data
            for idx, row in dca_df.iterrows():
                # Handle different column naming conventions
                nb_model = row.get(
                    "nb_model", row.get("net_benefit_model", row.get("net_benefit"))
                )
                nb_all = row.get("nb_all", row.get("net_benefit_all"))
                nb_none = row.get("nb_none", row.get("net_benefit_none", 0.0))

                con.execute(
                    """
                    INSERT INTO dca_curves
                    (config_id, threshold, net_benefit_model, net_benefit_all,
                     net_benefit_none, sensitivity, specificity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        config_id,
                        row.get("threshold"),
                        nb_model,
                        nb_all,
                        nb_none,
                        row.get("sensitivity"),
                        row.get("specificity"),
                    ],
                )

        except Exception as e:
            logger.warning(f"Failed to compute DCA curves: {e}")

    def _compute_stratos_metrics_from_raw(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """
        Compute STRATOS-required calibration metrics from raw predictions.

        Computes calibration slope, intercept, and O:E ratio using
        the calibration_slope_intercept function from src.stats.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes (0 or 1)
        y_prob : np.ndarray
            Predicted probabilities

        Returns
        -------
        dict
            Contains calibration_slope, calibration_intercept, o_e_ratio
        """
        from src.stats.calibration_extended import calibration_slope_intercept

        result = {
            "calibration_slope": None,
            "calibration_intercept": None,
            "o_e_ratio": None,
        }

        try:
            cal_result = calibration_slope_intercept(y_true, y_prob)
            result["calibration_slope"] = cal_result.slope
            result["calibration_intercept"] = cal_result.intercept
            result["o_e_ratio"] = cal_result.o_e_ratio
        except Exception as e:
            logger.warning(f"Failed to compute calibration metrics: {e}")

        return result

    def _extract_distribution_stats(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """
        Extract summary statistics for probability distributions.

        Stores per-class summary stats so visualization can annotate
        distribution plots without importing sklearn.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes (0 or 1)
        y_prob : np.ndarray
            Predicted probabilities
        config_id : int
            Configuration ID for database
        con : duckdb.DuckDBPyConnection
            Database connection
        """
        try:
            from sklearn.metrics import roc_auc_score

            cases_mask = y_true == 1
            controls_mask = y_true == 0

            prob_cases = y_prob[cases_mask]
            prob_controls = y_prob[controls_mask]

            auroc = None
            if len(np.unique(y_true)) >= 2:
                auroc = float(roc_auc_score(y_true, y_prob))

            con.execute(
                """
                INSERT INTO distribution_stats
                (config_id, auroc, median_cases, median_controls,
                 mean_cases, mean_controls, n_cases, n_controls)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    config_id,
                    auroc,
                    float(np.median(prob_cases)) if len(prob_cases) > 0 else None,
                    float(np.median(prob_controls)) if len(prob_controls) > 0 else None,
                    float(np.mean(prob_cases)) if len(prob_cases) > 0 else None,
                    float(np.mean(prob_controls)) if len(prob_controls) > 0 else None,
                    int(len(prob_cases)),
                    int(len(prob_controls)),
                ],
            )
        except Exception as e:
            logger.debug(f"Could not insert distribution stats: {e}")

    def _extract_retention_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        uncertainty: np.ndarray,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """
        Extract retention (selective classification) metrics at various rates.

        Pre-computes metrics at 20 retention rates (0.05 to 1.0 in steps of 0.05)
        for AUROC, Brier, scaled Brier, and net benefit.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes
        y_prob : np.ndarray
            Predicted probabilities
        uncertainty : np.ndarray
            Prediction uncertainty (lower = more confident)
        config_id : int
            Configuration ID for database
        con : duckdb.DuckDBPyConnection
            Database connection
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score

        retention_rates = np.arange(0.05, 1.05, 0.05)
        metrics_to_compute = ["auroc", "brier", "scaled_brier", "net_benefit"]

        try:
            for rate in retention_rates:
                n_retain = max(10, int(len(y_true) * rate))
                n_retain = min(n_retain, len(y_true))

                sorted_idx = np.argsort(uncertainty)[:n_retain]
                y_t = y_true[sorted_idx]
                y_p = y_prob[sorted_idx]

                if len(np.unique(y_t)) < 2:
                    continue

                for metric_name in metrics_to_compute:
                    value = None
                    try:
                        if metric_name == "auroc":
                            value = float(roc_auc_score(y_t, y_p))
                        elif metric_name == "brier":
                            value = float(-brier_score_loss(y_t, y_p))
                        elif metric_name == "scaled_brier":
                            brier = brier_score_loss(y_t, y_p)
                            prevalence = y_t.mean()
                            brier_null = prevalence * (1 - prevalence)
                            if brier_null > 0:
                                value = float(1 - brier / brier_null)
                        elif metric_name == "net_benefit":
                            n = len(y_t)
                            threshold = 0.15
                            y_pred = (y_p >= threshold).astype(int)
                            tp = np.sum((y_pred == 1) & (y_t == 1))
                            fp = np.sum((y_pred == 1) & (y_t == 0))
                            value = float(tp / n - fp / n * threshold / (1 - threshold))
                    except Exception:
                        pass

                    if value is not None:
                        con.execute(
                            """
                            INSERT INTO retention_metrics
                            (config_id, retention_rate, metric_name, metric_value)
                            VALUES (?, ?, ?, ?)
                        """,
                            [config_id, float(rate), metric_name, value],
                        )
        except Exception as e:
            logger.debug(f"Could not insert retention metrics: {e}")

    def _extract_cohort_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        uncertainty: np.ndarray,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """
        Extract cohort-based metrics at various fractions.

        Pre-computes metrics at 20 cohort fractions (0.05 to 1.0)
        based on uncertainty-sorted subsets.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes
        y_prob : np.ndarray
            Predicted probabilities
        uncertainty : np.ndarray
            Prediction uncertainty (lower = more certain)
        config_id : int
            Configuration ID for database
        con : duckdb.DuckDBPyConnection
            Database connection
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score

        fractions = np.arange(0.05, 1.05, 0.05)
        metrics_to_compute = ["auroc", "brier", "scaled_brier"]

        try:
            for fraction in fractions:
                n_retain = max(10, int(len(y_true) * fraction))
                n_retain = min(n_retain, len(y_true))

                sorted_idx = np.argsort(uncertainty)[:n_retain]
                y_t = y_true[sorted_idx]
                y_p = y_prob[sorted_idx]

                if len(np.unique(y_t)) < 2:
                    continue

                for metric_name in metrics_to_compute:
                    value = None
                    try:
                        if metric_name == "auroc":
                            value = float(roc_auc_score(y_t, y_p))
                        elif metric_name == "brier":
                            value = float(-brier_score_loss(y_t, y_p))
                        elif metric_name == "scaled_brier":
                            brier = brier_score_loss(y_t, y_p)
                            prevalence = y_t.mean()
                            brier_null = prevalence * (1 - prevalence)
                            if brier_null > 0:
                                value = float(1 - brier / brier_null)
                    except Exception:
                        pass

                    if value is not None:
                        con.execute(
                            """
                            INSERT INTO cohort_metrics
                            (config_id, cohort_fraction, metric_name, metric_value)
                            VALUES (?, ?, ?, ?)
                        """,
                            [config_id, float(fraction), metric_name, value],
                        )
        except Exception as e:
            logger.debug(f"Could not insert cohort metrics: {e}")

    def _insert_essential_metrics(
        self,
        metrics: Dict[str, Any],
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Insert essential metrics into database."""
        cols = list(metrics.keys())
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        con.execute(
            f"""
            INSERT INTO essential_metrics ({col_names})
            VALUES ({placeholders})
        """,
            [metrics[c] for c in cols],
        )

    def _insert_supplementary_metrics(
        self,
        metrics: Dict[str, Any],
        con: duckdb.DuckDBPyConnection,
    ) -> None:
        """Insert supplementary metrics into database."""
        cols = list(metrics.keys())
        placeholders = ", ".join(["?" for _ in cols])
        col_names = ", ".join(cols)

        con.execute(
            f"""
            INSERT INTO supplementary_metrics ({col_names})
            VALUES ({placeholders})
        """,
            [metrics[c] for c in cols],
        )

    def _extract_predictions(
        self,
        metrics_data: Dict,
        config_id: int,
        con: duckdb.DuckDBPyConnection,
    ) -> int:
        """
        Extract subject-level predictions with optional re-anonymization.

        Parameters
        ----------
        metrics_data : dict
            Loaded metrics pickle data
        config_id : int
            Configuration ID for database
        con : duckdb.DuckDBPyConnection
            Database connection

        Returns
        -------
        int
            Number of predictions extracted
        """
        if not self.extract_predictions:
            return 0

        # Look for subjectwise_stats in metrics data
        if "subjectwise_stats" not in metrics_data:
            return 0

        subj_stats = metrics_data["subjectwise_stats"]
        if "test" not in subj_stats:
            return 0

        test_data = subj_stats["test"]

        # Extract required fields
        subject_codes = test_data.get("subject_code", [])
        y_true = test_data.get("labels", [])
        preds = test_data.get("preds", {})
        y_prob_data = preds.get("y_pred_proba", {})
        y_prob_mean = y_prob_data.get("mean", [])

        # Handle numpy arrays
        if hasattr(subject_codes, "tolist"):
            subject_codes = subject_codes.tolist()
        if hasattr(y_true, "tolist"):
            y_true = y_true.tolist()
        if hasattr(y_prob_mean, "tolist"):
            y_prob_mean = y_prob_mean.tolist()

        if not subject_codes or not y_true:
            return 0

        n_extracted = 0
        for i, (code, y_t) in enumerate(zip(subject_codes, y_true)):
            prob = y_prob_mean[i] if i < len(y_prob_mean) else 0.5

            # Re-anonymize subject code if mapping provided
            anon_code = code
            if self.subject_mapping and code in self.subject_mapping:
                anon_code = self.subject_mapping[code]

            con.execute(
                """
                INSERT INTO predictions (config_id, subject_code, y_true, y_prob)
                VALUES (?, ?, ?, ?)
            """,
                [config_id, anon_code, int(y_t), float(prob)],
            )
            n_extracted += 1

        return n_extracted

    def _find_artifact(
        self,
        run_dir: Path,
        subdir: str,
        pattern: str,
    ) -> Optional[Path]:
        """Find artifact file in run directory."""
        artifacts_dir = run_dir / "artifacts" / subdir
        if not artifacts_dir.exists():
            return None

        matches = list(artifacts_dir.glob(pattern))
        return matches[0] if matches else None

    def export(self) -> Dict[str, int]:
        """
        Run streaming export with checkpointing.

        Returns
        -------
        dict
            Statistics: completed, skipped, failed counts
        """
        self._init_database()

        run_dirs = self._get_run_dirs()
        total_runs = len(run_dirs)

        logger.info(f"Found {total_runs} runs to process")

        # Check existing progress
        progress = self.checkpoint_manager.get_progress()
        logger.info(f"Previous progress: {progress}")

        stats = {"completed": 0, "skipped": 0, "failed": 0}

        # Timing tracking for performance monitoring
        extraction_start = time.time()
        run_times: List[float] = []
        slow_run_warnings = 0

        with duckdb.connect(str(self.output_path)) as con:
            config_id = self._get_next_config_id(con)

            for idx, run_dir in enumerate(run_dirs):
                run_id = run_dir.name

                # Skip if already completed
                if self.checkpoint_manager.is_completed(run_id):
                    stats["skipped"] += 1
                    continue

                # Check memory
                self.memory_monitor.enforce()

                # Process run with timing
                run_start = time.time()
                try:
                    self.checkpoint_manager.mark_started(run_id, con)

                    success = self._extract_run(run_dir, config_id, con)

                    if success:
                        self.checkpoint_manager.mark_completed(run_id, con)
                        stats["completed"] += 1
                        config_id += 1
                    else:
                        self.checkpoint_manager.mark_failed(
                            run_id, "No metrics found", con
                        )
                        stats["failed"] += 1

                except Exception as e:
                    self.checkpoint_manager.mark_failed(run_id, str(e), con)
                    stats["failed"] += 1
                    logger.warning(f"Error processing {run_id}: {e}")

                # Track run timing
                run_duration = time.time() - run_start
                run_times.append(run_duration)

                # PERFORMANCE WARNING: Single run taking too long
                if run_duration > SLOW_RUN_THRESHOLD_SECONDS:
                    slow_run_warnings += 1
                    logger.warning(
                        f"⚠️ SLOW RUN: {run_id} took {run_duration:.1f}s "
                        f"(threshold: {SLOW_RUN_THRESHOLD_SECONDS}s). "
                        f"Possible implementation issue if this persists!"
                    )

                # Progress logging with timing
                if (idx + 1) % 10 == 0 or idx == total_runs - 1:
                    mem_gb = self.memory_monitor.get_usage_gb()
                    elapsed = time.time() - extraction_start
                    runs_processed = stats["completed"] + stats["failed"]
                    rate = runs_processed / (elapsed / 60) if elapsed > 0 else 0

                    logger.info(
                        f"Progress: {idx + 1}/{total_runs} "
                        f"(completed={stats['completed']}, skipped={stats['skipped']}, "
                        f"failed={stats['failed']}, mem={mem_gb:.1f}GB, "
                        f"rate={rate:.1f} runs/min)"
                    )

                    # PERFORMANCE WARNING: Rate too slow
                    if runs_processed >= 10 and rate < EXPECTED_RUNS_PER_MINUTE:
                        logger.warning(
                            f"⚠️ SLOW EXTRACTION: {rate:.1f} runs/min "
                            f"(expected: >{EXPECTED_RUNS_PER_MINUTE} runs/min). "
                            f"Check for memory accumulation or implementation issues!"
                        )

        # Final timing summary
        total_time = time.time() - extraction_start
        avg_time = sum(run_times) / len(run_times) if run_times else 0

        logger.info(
            f"Export complete: {stats} | "
            f"Total time: {total_time:.1f}s | "
            f"Avg per run: {avg_time:.2f}s | "
            f"Slow run warnings: {slow_run_warnings}"
        )

        # Final health check
        if slow_run_warnings > 5:
            logger.warning(
                f"🚨 {slow_run_warnings} runs exceeded {SLOW_RUN_THRESHOLD_SECONDS}s. "
                "Review implementation for memory leaks or performance issues!"
            )

        return stats

    def _get_next_config_id(self, con: duckdb.DuckDBPyConnection) -> int:
        """Get next available config_id."""
        result = con.execute(
            "SELECT COALESCE(MAX(config_id), 0) + 1 FROM essential_metrics"
        ).fetchone()
        return result[0]


def main():
    """CLI for streaming export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Memory-efficient streaming export to DuckDB"
    )
    parser.add_argument("mlruns", help="Path to mlruns directory")
    parser.add_argument("output", help="Output .db file path")
    parser.add_argument(
        "--experiment", help="Specific experiment ID (auto-detects if not provided)"
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=12.0,
        help="Memory warning threshold in GB (default: 12)",
    )

    args = parser.parse_args()

    exporter = StreamingDuckDBExporter(
        args.mlruns,
        args.output,
        experiment_id=args.experiment,
        memory_threshold_gb=args.memory_threshold,
    )

    stats = exporter.export()
    print(f"Export complete: {stats}")


if __name__ == "__main__":
    main()
