"""
Block 1: Extraction Flow - MLflow to DuckDB with Privacy Separation.

This flow extracts MLflow experiment results and creates:
1. PUBLIC: foundation_plr_results.db with re-anonymized subject codes (Hxxx/Gxxx)
2. PRIVATE: subject_lookup.yaml mapping anonymized → original codes
3. PRIVATE: demo_subjects_traces.pkl with raw PLR traces for visualization

Uses StreamingDuckDBExporter for:
- Checkpoint/resume capability (crash recovery)
- Memory monitoring with automatic GC
- Progress logging every 10 runs
- STRATOS-compliant schema (Van Calster 2024)

Usage:
    python -m src.orchestration.flows.extraction_flow
    # Or via Prefect:
    prefect deployment run extraction-flow/default
"""

import gc
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import numpy as np
import psutil
import yaml
from loguru import logger

# Import StreamingDuckDBExporter for robust extraction
from src.data_io.streaming_duckdb_export import StreamingDuckDBExporter
from src.utils.paths import (
    PROJECT_ROOT,
    get_classification_experiment_id,
    get_mlruns_dir,
    get_results_db_path,
    get_seri_db_path,
)

# Prefect compatibility layer (uses importlib for static-analysis safety)
from src.orchestration._prefect_compat import flow, task


# ============================================================================
# Configuration
# ============================================================================

# Use centralized path utilities (env-aware, portable)
MLRUNS_DIR = get_mlruns_dir()
SERI_DB_PATH = get_seri_db_path()

# Output paths
PUBLIC_DB_PATH = get_results_db_path()
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
SUBJECT_LOOKUP_PATH = PRIVATE_DIR / "subject_lookup.yaml"
DEMO_TRACES_PATH = PRIVATE_DIR / "demo_subjects_traces.pkl"

# MLflow experiment ID for classification
CLASSIFICATION_EXP_ID = get_classification_experiment_id()


# ============================================================================
# Subject Re-anonymization
# ============================================================================


@task(name="generate_subject_mapping", retries=1)
def generate_subject_mapping(
    seri_db_path: Path = SERI_DB_PATH,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Generate re-anonymization mapping for all subjects.

    Maps original PLRxxxx codes to anonymized Hxxx (healthy) / Gxxx (glaucoma) codes.

    Returns
    -------
    Tuple[Dict[str, str], Dict[str, str]]
        (original_to_anon, anon_to_original) mapping dictionaries
    """
    logger.info("Generating subject re-anonymization mapping...")

    conn = duckdb.connect(str(seri_db_path), read_only=True)

    # Get all unique subjects with their labels from both train and test
    subjects = conn.execute("""
        SELECT DISTINCT subject_code, class_label
        FROM (
            SELECT subject_code, class_label FROM train
            UNION
            SELECT subject_code, class_label FROM test
        )
        WHERE class_label IS NOT NULL
        ORDER BY class_label, subject_code
    """).fetchall()

    conn.close()

    original_to_anon = {}
    anon_to_original = {}

    healthy_counter = 1
    glaucoma_counter = 1

    for subject_code, class_label in subjects:
        if class_label == "control":
            anon_code = f"H{healthy_counter:03d}"
            healthy_counter += 1
        elif class_label == "glaucoma":
            anon_code = f"G{glaucoma_counter:03d}"
            glaucoma_counter += 1
        else:
            continue  # Skip unlabeled subjects

        original_to_anon[subject_code] = anon_code
        anon_to_original[anon_code] = subject_code

    logger.info(
        f"Generated mapping for {len(original_to_anon)} subjects "
        f"({healthy_counter - 1} healthy, {glaucoma_counter - 1} glaucoma)"
    )

    return original_to_anon, anon_to_original


@task(name="save_private_lookup", retries=1)
def save_private_lookup(
    anon_to_original: Dict[str, str],
    output_path: Path = SUBJECT_LOOKUP_PATH,
) -> Path:
    """
    Save the private lookup table (anonymized → original codes).

    This file is gitignored and should only exist locally.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lookup_data = {
        "lookup": anon_to_original,
        "reverse_lookup": {v: k for k, v in anon_to_original.items()},
        "metadata": {
            "created": datetime.now().isoformat(),
            "source_database": str(SERI_DB_PATH),
            "n_healthy": sum(1 for k in anon_to_original if k.startswith("H")),
            "n_glaucoma": sum(1 for k in anon_to_original if k.startswith("G")),
            "WARNING": "PRIVATE FILE - DO NOT SHARE OR COMMIT",
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(lookup_data, f, default_flow_style=False)

    logger.info(f"Saved private lookup table to {output_path}")
    return output_path


# ============================================================================
# MLflow Extraction - StreamingDuckDBExporter (RECOMMENDED)
# ============================================================================


@task(name="extract_with_streaming_exporter", retries=2, retry_delay_seconds=30)
def extract_with_streaming_exporter(
    mlruns_dir: Path = MLRUNS_DIR,
    experiment_id: str = CLASSIFICATION_EXP_ID,
    output_path: Path = PUBLIC_DB_PATH,
    memory_threshold_gb: float = 12.0,
    subject_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Extract MLflow runs using StreamingDuckDBExporter (RECOMMENDED).

    This is the robust extraction method that provides:
    - Checkpoint/resume capability (survives crashes)
    - Memory monitoring with automatic GC
    - Progress logging every 10 runs
    - STRATOS-compliant schema (Van Calster 2024)
    - Optional subject code re-anonymization

    Parameters
    ----------
    mlruns_dir : Path
        Path to MLflow runs directory
    experiment_id : str
        MLflow experiment ID
    output_path : Path
        Output DuckDB file path
    memory_threshold_gb : float
        Memory warning threshold
    subject_mapping : dict, optional
        Mapping from original subject codes to anonymized codes (PLRxxxx -> Hxxx/Gxxx)

    Returns dictionary with:
    - completed: Number of runs successfully extracted
    - skipped: Number of runs already in checkpoint
    - failed: Number of runs that failed extraction
    - output_path: Path to output DuckDB file
    """
    logger.info("=" * 60)
    logger.info("STREAMING EXTRACTION (Checkpoint/Resume Enabled)")
    logger.info("=" * 60)
    logger.info(f"MLruns dir: {mlruns_dir}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Memory threshold: {memory_threshold_gb} GB")
    if subject_mapping:
        logger.info(f"Subject re-anonymization: {len(subject_mapping)} mappings")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize streaming exporter with optional re-anonymization
    exporter = StreamingDuckDBExporter(
        mlruns_dir=mlruns_dir,
        output_path=output_path,
        experiment_id=experiment_id,
        memory_threshold_gb=memory_threshold_gb,
        subject_mapping=subject_mapping,
    )

    # Run export with checkpoint/resume
    stats = exporter.export()

    logger.info("=" * 60)
    logger.info("STREAMING EXTRACTION COMPLETE")
    logger.info(f"  Completed: {stats['completed']}")
    logger.info(f"  Skipped (already done): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info("=" * 60)

    # Get database size
    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

    return {
        "completed": stats["completed"],
        "skipped": stats["skipped"],
        "failed": stats["failed"],
        "output_path": output_path,
        "size_mb": size_mb,
    }


# ============================================================================
# MLflow Extraction - Legacy (DEPRECATED, kept for compatibility)
# ============================================================================


@task(name="extract_mlflow_runs_legacy", retries=2, retry_delay_seconds=30)
def extract_mlflow_runs(
    mlruns_dir: Path = MLRUNS_DIR,
    experiment_id: str = CLASSIFICATION_EXP_ID,
    original_to_anon: Optional[Dict[str, str]] = None,
    output_path: Path = PUBLIC_DB_PATH,
) -> Dict[str, Any]:
    """
    Extract classification runs from MLflow with STREAMING INSERTS.

    CRITICAL FIX (2026-02-01): Changed from batch accumulation to streaming inserts.
    Previous version accumulated all predictions in memory before writing,
    causing OOM after ~300 runs. Now writes per-run immediately.

    Returns dictionary with:
    - n_predictions: Count of prediction records written
    - n_aggregates: Count of aggregate metrics written
    - n_runs: Count of runs processed
    - output_path: Path to DuckDB file
    """

    logger.info(f"Extracting MLflow runs from experiment {experiment_id}...")
    logger.info(f"Output: {output_path}")

    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Get all run directories
    run_dirs = [
        d for d in experiment_dir.iterdir() if d.is_dir() and (d / "meta.yaml").exists()
    ]
    total_runs = len(run_dirs)

    logger.info(f"Found {total_runs} runs to process")

    # Create output directory and database
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    # Create tables upfront
    conn.execute("""
        CREATE TABLE predictions (
            prediction_id INTEGER,
            subject_code VARCHAR,
            y_true INTEGER,
            y_prob DOUBLE,
            run_id VARCHAR,
            classifier VARCHAR,
            featurization VARCHAR,
            imputation_method VARCHAR,
            outlier_method VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE metrics_aggregate (
            run_id VARCHAR,
            source_name VARCHAR,
            split VARCHAR,
            classifier VARCHAR,
            featurization VARCHAR,
            imputation_method VARCHAR,
            outlier_method VARCHAR,
            AUROC_mean DOUBLE,
            AUROC_std DOUBLE,
            AUROC_ci_lo DOUBLE,
            AUROC_ci_hi DOUBLE,
            Brier_mean DOUBLE,
            Brier_std DOUBLE,
            sensitivity_mean DOUBLE,
            specificity_mean DOUBLE
        )
    """)

    # Tracking counters
    n_predictions = 0
    n_aggregates = 0
    n_runs_processed = 0
    start_time = time.time()
    last_heartbeat = start_time

    for i, run_dir in enumerate(run_dirs):
        try:
            run_data = _process_single_run(run_dir, original_to_anon)
            if run_data:
                # STREAMING INSERT: Write predictions immediately
                preds = run_data.get("predictions", [])
                if preds:
                    for pred in preds:
                        conn.execute(
                            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                n_predictions,
                                pred.get("subject_code"),
                                pred.get("y_true"),
                                pred.get("y_prob"),
                                pred.get("run_id"),
                                pred.get("classifier"),
                                pred.get("featurization"),
                                pred.get("imputation_method"),
                                pred.get("outlier_method"),
                            ],
                        )
                        n_predictions += 1

                # STREAMING INSERT: Write aggregate immediately
                agg = run_data.get("aggregate")
                if agg:
                    conn.execute(
                        "INSERT INTO metrics_aggregate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            agg.get("run_id"),
                            agg.get("source_name"),
                            agg.get("split"),
                            agg.get("classifier"),
                            agg.get("featurization"),
                            agg.get("imputation_method"),
                            agg.get("outlier_method"),
                            agg.get("AUROC_mean"),
                            agg.get("AUROC_std"),
                            agg.get("AUROC_ci_lo"),
                            agg.get("AUROC_ci_hi"),
                            agg.get("Brier_mean"),
                            agg.get("Brier_std"),
                            agg.get("sensitivity_mean"),
                            agg.get("specificity_mean"),
                        ],
                    )
                    n_aggregates += 1

                n_runs_processed += 1

        except Exception as e:
            logger.warning(f"Error processing run {run_dir.name}: {e}")
            continue

        # HEARTBEAT: Log progress every 30 seconds or every 10 runs
        current_time = time.time()
        if current_time - last_heartbeat > 30 or (i + 1) % 10 == 0:
            elapsed = current_time - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = total_runs - (i + 1)
            eta_seconds = remaining / rate if rate > 0 else 0
            mem_gb = psutil.Process().memory_info().rss / (1024**3)

            logger.info(
                f"[PROGRESS] {i + 1}/{total_runs} ({100 * (i + 1) / total_runs:.1f}%) | "
                f"Rate: {rate:.2f}/s | ETA: {eta_seconds / 60:.1f}min | "
                f"Mem: {mem_gb:.2f}GB | Preds: {n_predictions}"
            )
            last_heartbeat = current_time

        # Memory cleanup every 50 runs
        if (i + 1) % 50 == 0:
            gc.collect()

    # Create essential_metrics view
    conn.execute("""
        CREATE VIEW essential_metrics AS
        SELECT
            run_id,
            source_name,
            split,
            classifier,
            featurization,
            imputation_method,
            outlier_method,
            AUROC_mean as auroc,
            AUROC_std as auroc_std,
            AUROC_ci_lo as auroc_ci_lower,
            AUROC_ci_hi as auroc_ci_upper,
            Brier_mean as brier,
            Brier_std as brier_std,
            sensitivity_mean as sensitivity,
            specificity_mean as specificity
        FROM metrics_aggregate
    """)

    conn.commit()
    conn.close()

    elapsed_total = time.time() - start_time
    size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"EXTRACTION COMPLETE: {n_predictions} predictions, "
        f"{n_aggregates} aggregates from {n_runs_processed} runs in {elapsed_total / 60:.1f}min"
    )
    logger.info(f"Output file: {output_path} ({size_mb:.2f} MB)")

    return {
        "n_predictions": n_predictions,
        "n_aggregates": n_aggregates,
        "n_runs": n_runs_processed,
        "output_path": output_path,
    }


def _process_single_run(
    run_dir: Path,
    original_to_anon: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single MLflow run directory."""

    # Load run metadata
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    run_id = meta.get("run_id", run_dir.name)
    run_name = meta.get("run_name", "")

    # Skip non-classification runs
    classifiers = ["XGBOOST", "CATBOOST", "TABPFN", "TABM", "LOGISTICREGRESSION"]
    if not any(clf in run_name.upper() for clf in classifiers):
        return None

    # Parse configuration from run name
    config = _parse_run_name(run_name)

    result = {
        "predictions": [],
        "aggregate": None,
        "metadata": {
            "run_id": run_id,
            "run_name": run_name,
            "status": meta.get("status", ""),
            "start_time": meta.get("start_time", ""),
            "config": config,
        },
    }

    # Load metrics pickle if available
    metrics_path = run_dir / "artifacts" / "metrics"
    metrics_files = list(metrics_path.glob("*.pickle")) if metrics_path.exists() else []

    if metrics_files:
        try:
            with open(metrics_files[0], "rb") as f:
                metrics_data = pickle.load(f)

            # Extract aggregate metrics
            if "metrics_stats" in metrics_data:
                for split_name, split_data in metrics_data["metrics_stats"].items():
                    if (
                        "metrics" not in split_data
                        or "scalars" not in split_data["metrics"]
                    ):
                        continue

                    scalars = split_data["metrics"]["scalars"]
                    agg = {
                        "run_id": run_id,
                        "source_name": run_name,
                        "split": split_name,
                        **config,
                    }

                    # Extract all available metrics
                    for metric_name, metric_vals in scalars.items():
                        if isinstance(metric_vals, dict) and "mean" in metric_vals:
                            agg[f"{metric_name}_mean"] = metric_vals.get("mean")
                            agg[f"{metric_name}_std"] = metric_vals.get("std")
                            ci = metric_vals.get("ci", [None, None])
                            # Handle numpy arrays, lists, and tuples
                            if (
                                isinstance(ci, (list, tuple, np.ndarray))
                                and len(ci) >= 2
                            ):
                                # Convert numpy scalars to Python floats
                                agg[f"{metric_name}_ci_lo"] = (
                                    float(ci[0]) if ci[0] is not None else None
                                )
                                agg[f"{metric_name}_ci_hi"] = (
                                    float(ci[1]) if ci[1] is not None else None
                                )

                    result["aggregate"] = agg

            # Extract per-subject predictions
            if "subjectwise_stats" in metrics_data:
                subj_stats = metrics_data.get("subjectwise_stats", {})
                if "test" in subj_stats:
                    test_data = subj_stats["test"]
                    # Correct structure: labels and subject_code are at test level, not inside preds
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

                    for i, (code, y_t) in enumerate(zip(subject_codes, y_true)):
                        prob = y_prob_mean[i] if i < len(y_prob_mean) else 0.5

                        # Re-anonymize subject code
                        anon_code = code
                        if original_to_anon and code in original_to_anon:
                            anon_code = original_to_anon[code]

                        result["predictions"].append(
                            {
                                "subject_code": anon_code,
                                "y_true": int(y_t),
                                "y_prob": float(prob),
                                "run_id": run_id,
                                **config,
                            }
                        )

        except Exception as e:
            logger.debug(f"Error loading metrics for {run_id}: {e}")

    return result


def _parse_run_name(run_name: str) -> dict[str, str]:
    """Parse configuration from MLflow run name."""
    config = {
        "classifier": "Unknown",
        "featurization": "Unknown",
        "imputation_method": "Unknown",
        "outlier_method": "Unknown",
    }

    if not run_name:
        return config

    parts = run_name.split("__")

    if parts:
        clf_part = parts[0].split("_")[0].upper()
        classifier_map = {
            "XGBOOST": "XGBoost",
            "CATBOOST": "CatBoost",
            "TABPFN": "TabPFN",
            "TABM": "TabM",
            "LOGISTICREGRESSION": "LogisticRegression",
        }
        config["classifier"] = classifier_map.get(clf_part, clf_part)

    if len(parts) > 1:
        config["featurization"] = parts[1]
    if len(parts) > 2:
        config["imputation_method"] = parts[2]
    if len(parts) > 3:
        config["outlier_method"] = parts[3]

    return config


# ============================================================================
# DuckDB Export
# ============================================================================


@task(name="export_to_duckdb", retries=1)
def export_to_duckdb(
    extracted_data: Dict[str, Any],
    output_path: Path = PUBLIC_DB_PATH,
) -> Path:
    """
    Export extracted data to public DuckDB database.

    The database contains re-anonymized subject codes and is safe to share.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.exists():
        output_path.unlink()

    logger.info(f"Exporting to DuckDB: {output_path}")

    conn = duckdb.connect(str(output_path))

    # Create predictions table
    conn.execute("""
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY,
            subject_code VARCHAR,
            y_true INTEGER,
            y_prob DOUBLE,
            run_id VARCHAR,
            classifier VARCHAR,
            featurization VARCHAR,
            imputation_method VARCHAR,
            outlier_method VARCHAR
        )
    """)

    # Insert predictions
    predictions = extracted_data.get("predictions", [])
    if predictions:
        for i, pred in enumerate(predictions):
            conn.execute(
                """
                INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    i,
                    pred.get("subject_code"),
                    pred.get("y_true"),
                    pred.get("y_prob"),
                    pred.get("run_id"),
                    pred.get("classifier"),
                    pred.get("featurization"),
                    pred.get("imputation_method"),
                    pred.get("outlier_method"),
                ],
            )

    # Create aggregate metrics table
    conn.execute("""
        CREATE TABLE metrics_aggregate (
            run_id VARCHAR PRIMARY KEY,
            source_name VARCHAR,
            split VARCHAR,
            classifier VARCHAR,
            featurization VARCHAR,
            imputation_method VARCHAR,
            outlier_method VARCHAR,
            AUROC_mean DOUBLE,
            AUROC_std DOUBLE,
            AUROC_ci_lo DOUBLE,
            AUROC_ci_hi DOUBLE,
            Brier_mean DOUBLE,
            Brier_std DOUBLE,
            sensitivity_mean DOUBLE,
            specificity_mean DOUBLE
        )
    """)

    # Insert aggregate metrics
    aggregates = extracted_data.get("metrics_aggregate", [])
    for agg in aggregates:
        conn.execute(
            """
            INSERT INTO metrics_aggregate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                agg.get("run_id"),
                agg.get("source_name"),
                agg.get("split"),
                agg.get("classifier"),
                agg.get("featurization"),
                agg.get("imputation_method"),
                agg.get("outlier_method"),
                agg.get("AUROC_mean"),
                agg.get("AUROC_std"),
                agg.get("AUROC_ci_lo"),
                agg.get("AUROC_ci_hi"),
                agg.get("Brier_mean"),
                agg.get("Brier_std"),
                agg.get("sensitivity_mean"),
                agg.get("specificity_mean"),
            ],
        )

    conn.commit()

    # Create essential_metrics view for compatibility with viz modules
    # This view provides lowercase column names that the viz infrastructure expects
    conn.execute("""
        CREATE VIEW essential_metrics AS
        SELECT
            run_id,
            source_name,
            split,
            classifier,
            featurization,
            imputation_method,
            outlier_method,
            AUROC_mean as auroc,
            AUROC_std as auroc_std,
            AUROC_ci_lo as auroc_ci_lower,
            AUROC_ci_hi as auroc_ci_upper,
            Brier_mean as brier,
            Brier_std as brier_std,
            sensitivity_mean as sensitivity,
            specificity_mean as specificity
        FROM metrics_aggregate
    """)

    # Log summary
    n_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_aggs = conn.execute("SELECT COUNT(*) FROM metrics_aggregate").fetchone()[0]
    logger.info(f"Exported {n_preds} predictions, {n_aggs} aggregate metrics")
    logger.info("Created essential_metrics view for viz module compatibility")

    conn.close()

    # Log file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"DuckDB file size: {size_mb:.2f} MB")

    return output_path


# ============================================================================
# Demo Subject Traces
# ============================================================================


@task(name="extract_demo_traces", retries=1)
def extract_demo_traces(
    seri_db_path: Path = SERI_DB_PATH,
    demo_config_path: Optional[Path] = None,
    anon_to_original: Optional[Dict[str, str]] = None,
    output_path: Path = DEMO_TRACES_PATH,
) -> Optional[Path]:
    """
    Extract raw PLR traces for demo subjects (PRIVATE).

    This creates a pickle file with individual PLR traces that cannot be shared.
    """
    if demo_config_path is None:
        demo_config_path = PROJECT_ROOT / "config" / "demo_subjects.yaml"

    if not demo_config_path.exists():
        logger.warning(f"Demo subjects config not found: {demo_config_path}")
        return None

    # Load demo subject codes
    with open(demo_config_path) as f:
        demo_config = yaml.safe_load(f)

    demo_codes = demo_config.get("all_demo_subjects", [])
    if not demo_codes:
        logger.warning("No demo subjects defined in config")
        return None

    logger.info(f"Extracting traces for {len(demo_codes)} demo subjects...")

    # Map anonymized codes to original
    if anon_to_original is None:
        # Load from private lookup if available
        lookup_path = PRIVATE_DIR / "subject_lookup.yaml"
        if lookup_path.exists():
            with open(lookup_path) as f:
                lookup_data = yaml.safe_load(f)
            anon_to_original = lookup_data.get("lookup", {})

    if not anon_to_original:
        logger.error("Cannot extract traces without subject lookup table")
        return None

    # Extract traces from database
    conn = duckdb.connect(str(seri_db_path), read_only=True)

    traces = {}
    for anon_code in demo_codes:
        original_code = anon_to_original.get(anon_code)
        if not original_code:
            logger.warning(f"No mapping found for {anon_code}")
            continue

        try:
            # Check both train and test tables
            result = conn.execute(
                f"""
                SELECT time, pupil_gt, pupil_raw, outlier_mask,
                       Red, Blue, class_label
                FROM (
                    SELECT * FROM train WHERE subject_code = '{original_code}'
                    UNION ALL
                    SELECT * FROM test WHERE subject_code = '{original_code}'
                )
                ORDER BY time
            """
            ).fetchall()

            if result:
                traces[anon_code] = {
                    "time": [r[0] for r in result],
                    "pupil_gt": [r[1] for r in result],
                    "pupil_raw": [r[2] for r in result],
                    "outlier_mask": [r[3] for r in result],
                    "red_stimulus": [r[4] for r in result],
                    "blue_stimulus": [r[5] for r in result],
                    "class_label": result[0][6] if result else None,
                    "original_code": original_code,  # Keep for internal reference
                }
                logger.debug(f"Extracted {len(result)} timepoints for {anon_code}")
        except Exception as e:
            logger.warning(f"Error extracting trace for {anon_code}: {e}")

    conn.close()

    if not traces:
        logger.warning("No traces extracted")
        return None

    # Save to private pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "traces": traces,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "n_subjects": len(traces),
                    "WARNING": "PRIVATE FILE - DO NOT SHARE",
                },
            },
            f,
        )

    logger.info(f"Saved {len(traces)} demo traces to {output_path}")
    return output_path


# ============================================================================
# Main Flow
# ============================================================================


@flow(name="extraction-flow", log_prints=True)
def extraction_flow(
    mlruns_dir: Path = MLRUNS_DIR,
    seri_db_path: Path = SERI_DB_PATH,
    experiment_id: str = CLASSIFICATION_EXP_ID,
    use_streaming_exporter: bool = True,
    memory_threshold_gb: float = 12.0,
) -> Dict[str, Path]:
    """
    Block 1: Extract MLflow results to DuckDB with privacy separation.

    Creates:
    - PUBLIC: data/public/foundation_plr_results.db (shareable)
    - PRIVATE: data/private/subject_lookup.yaml (gitignored)
    - PRIVATE: data/private/demo_subjects_traces.pkl (gitignored)

    Parameters
    ----------
    mlruns_dir : Path
        Path to MLflow runs directory
    seri_db_path : Path
        Path to SERI PLR database
    experiment_id : str
        MLflow experiment ID
    use_streaming_exporter : bool
        If True (default), use StreamingDuckDBExporter with:
        - Checkpoint/resume capability
        - Memory monitoring
        - STRATOS-compliant schema
        If False, use legacy extraction (no checkpoint, simpler schema)
    memory_threshold_gb : float
        Memory warning threshold in GB (default: 12.0)
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1: EXTRACTION FLOW")
    logger.info("=" * 60)

    # Step 1: Generate subject re-anonymization mapping
    original_to_anon, anon_to_original = generate_subject_mapping(seri_db_path)

    # Step 2: Save private lookup table
    lookup_path = save_private_lookup(anon_to_original)

    # Step 3: Extract MLflow runs
    if use_streaming_exporter:
        # Use StreamingDuckDBExporter (RECOMMENDED)
        # Provides: checkpoint/resume, memory monitoring, STRATOS schema
        # Re-anonymization integrated: subject codes become Hxxx/Gxxx
        extraction_result = extract_with_streaming_exporter(
            mlruns_dir=mlruns_dir,
            experiment_id=experiment_id,
            output_path=PUBLIC_DB_PATH,
            memory_threshold_gb=memory_threshold_gb,
            subject_mapping=original_to_anon,  # Re-anonymize PLRxxxx -> Hxxx/Gxxx
        )
        public_db_path = extraction_result["output_path"]
        logger.info(
            f"Streaming extraction complete: "
            f"{extraction_result['completed']} completed, "
            f"{extraction_result['skipped']} skipped, "
            f"{extraction_result['failed']} failed"
        )
    else:
        # Use legacy extraction (DEPRECATED)
        # Kept for compatibility, no checkpoint support
        extraction_result = extract_mlflow_runs(
            mlruns_dir, experiment_id, original_to_anon, PUBLIC_DB_PATH
        )
        public_db_path = extraction_result["output_path"]
        logger.info(
            f"Legacy extraction complete: {extraction_result['n_predictions']} predictions, "
            f"{extraction_result['n_aggregates']} aggregates"
        )

    # Step 4: Extract demo subject traces (private)
    demo_traces_path = extract_demo_traces(
        seri_db_path=seri_db_path,
        anon_to_original=anon_to_original,
    )

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"  Public DB: {public_db_path}")
    logger.info(f"  Private lookup: {lookup_path}")
    logger.info(f"  Demo traces: {demo_traces_path}")
    logger.info("=" * 60)

    return {
        "public_db": public_db_path,
        "private_lookup": lookup_path,
        "demo_traces": demo_traces_path,
    }


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> None:
    """Run extraction flow from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Block 1: MLflow extraction flow")
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=MLRUNS_DIR,
        help="Path to mlruns directory",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=CLASSIFICATION_EXP_ID,
        help="MLflow experiment ID",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy extraction (no checkpoint, simpler schema). "
        "Default uses StreamingDuckDBExporter with checkpoint/resume.",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=12.0,
        help="Memory warning threshold in GB (default: 12.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN - would execute:")
        logger.info(f"  MLruns dir: {args.mlruns_dir}")
        logger.info(f"  Experiment ID: {args.experiment_id}")
        logger.info(f"  Output public DB: {PUBLIC_DB_PATH}")
        logger.info(f"  Output private dir: {PRIVATE_DIR}")
        logger.info(f"  Use streaming exporter: {not args.legacy}")
        logger.info(f"  Memory threshold: {args.memory_threshold} GB")
        return

    result = extraction_flow(
        mlruns_dir=args.mlruns_dir,
        experiment_id=args.experiment_id,
        use_streaming_exporter=not args.legacy,
        memory_threshold_gb=args.memory_threshold,
    )

    print("\nExtraction complete!")
    print(f"  Public DB: {result['public_db']}")
    print(f"  Private lookup: {result['private_lookup']}")


if __name__ == "__main__":
    main()
