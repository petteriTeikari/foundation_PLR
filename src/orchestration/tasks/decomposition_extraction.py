"""Prefect task for decomposition signal extraction.

This is an OPTIONAL, time-consuming extraction (~11+ hours for full dataset).
Should only be run when decomposition analysis is needed.

Usage in Prefect flow:
    from src.orchestration.tasks.decomposition_extraction import extract_decomposition_signals_task

    @flow
    def my_flow(include_decomposition: bool = False):
        if include_decomposition:
            extract_decomposition_signals_task()
"""

import pickle
import re
from pathlib import Path
from typing import Any, Optional

import duckdb
import yaml

# Registry validation - SINGLE SOURCE OF TRUTH
from src.data_io.registry import validate_imputation_method, validate_outlier_method

# Prefect compatibility layer (uses importlib for static-analysis safety)
from src.orchestration._prefect_compat import PREFECT_AVAILABLE, get_run_logger, task


# Project paths
from src.utils.paths import (
    get_mlflow_registry_dir,
    get_mlruns_dir,
    get_preprocessed_signals_db_path,
)

MLFLOW_ROOT = get_mlruns_dir()
IMPUTATION_EXPERIMENT = "940304421003085572"
DEFAULT_OUTPUT_DB = get_preprocessed_signals_db_path()
CATEGORY_MAPPING = get_mlflow_registry_dir() / "category_mapping.yaml"

# Invalid method names to skip
GARBAGE_METHODS = {"exclude", "anomaly"}


def load_category_mapping() -> dict[str, Any]:
    """Load category mapping configuration."""
    with open(CATEGORY_MAPPING) as f:
        return yaml.safe_load(f)


def get_outlier_category(method: str, category_config: dict[str, Any]) -> str:
    """Map outlier method to preprocessing category."""
    exact = category_config["outlier_method_categories"]["exact"]
    patterns = category_config["outlier_method_categories"]["patterns"]

    if method in exact:
        return exact[method]

    for p in patterns:
        if re.search(p["pattern"], method):
            return p["category"]

    return "Unknown"


def is_garbage_method(method: str) -> bool:
    """Check if method is a garbage placeholder."""
    return any(g in method.lower() for g in GARBAGE_METHODS)


def normalize_outlier_method(raw: str) -> str:
    """Normalize outlier method name to canonical form."""
    raw = raw.replace("_gt", "-gt").replace("_orig", "-orig")

    if raw.startswith("MOMENT_"):
        mode = raw.split("_")[1]
        if "___gt" in raw:
            return f"MOMENT-gt-{mode}"
        elif "___orig" in raw:
            return f"MOMENT-orig-{mode}"
        else:
            return f"MOMENT-{mode}"

    return raw


def parse_imputation_run_name(run_name: str) -> tuple[Optional[str], Optional[str]]:
    """Parse imputation and outlier method from imputation experiment run name.

    This is for imputation experiment runs (experiment 940304421003085572),
    which have a different format than classification runs.

    Format: {imputation}_{variant}__{outlier}_impPLR_v0.1-Outlier

    Parameters
    ----------
    run_name : str
        Imputation experiment run name

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (imputation_method, outlier_method) if valid and in registry, else (None, None)
    """
    if "__" not in run_name:
        return None, None

    parts = run_name.split("__")
    imp_method = parts[0].split("_")[0]
    outlier_part = parts[1].replace("_impPLR_v0.1", "").replace("-Outlier", "")
    outlier_method = normalize_outlier_method(outlier_part)

    # Validate against registry - SINGLE SOURCE OF TRUTH
    if not validate_imputation_method(imp_method):
        return None, None

    if not validate_outlier_method(outlier_method):
        return None, None

    return imp_method, outlier_method


def extract_signals_from_pickle(pickle_path: Path) -> list[dict[str, Any]]:
    """Extract per-subject signals from a pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    source_df = data["source_data"]["df"]
    imputation = data["model_artifacts"]["imputation"]

    results = []

    for split in ["train", "test"]:
        if split not in imputation:
            continue

        split_data = source_df[split]
        imp_data = imputation[split]["imputation_dict"]["imputation"]

        subject_codes = split_data["metadata"]["subject_code"]
        time_array = split_data["time"]["time"]
        mean_signals = imp_data["mean"]

        n_subjects = mean_signals.shape[0]

        for i in range(n_subjects):
            results.append(
                {
                    "subject_code": subject_codes[i, 0],
                    "signal": mean_signals[i, :, 0],
                    "time_vector": time_array[i, :],
                }
            )

    return results


def check_extraction_needed(output_db: Path, force: bool = False) -> bool:
    """Check if extraction is needed based on existing DB."""
    if force:
        return True

    if not output_db.exists():
        return True

    # Check if DB has expected structure and data
    try:
        conn = duckdb.connect(str(output_db), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM preprocessed_signals").fetchone()[0]
        conn.close()

        # If we have substantial data, skip extraction
        if count > 50000:  # Expected ~68K rows
            return False
    except Exception:
        return True

    return True


@task(
    name="extract_decomposition_signals",
    description="Extract per-subject preprocessed signals to DuckDB for decomposition analysis",
    tags=["decomposition", "extraction", "long-running"],
    retries=0,  # Don't retry - too expensive
    timeout_seconds=86400,  # 24 hour timeout
)
def extract_decomposition_signals_task(
    output_db: Optional[Path] = None,
    force: bool = False,
    chunk_size: int = 1000,
) -> Path:
    """
    Extract per-subject preprocessed signals from MLflow imputation runs.

    This is a LONG-RUNNING task (~11+ hours for full dataset).

    Parameters
    ----------
    output_db : Path, optional
        Output DuckDB path. Default: data/private/preprocessed_signals_per_subject.db
    force : bool
        Force re-extraction even if DB exists
    chunk_size : int
        Batch insert chunk size (default: 1000)

    Returns
    -------
    Path
        Path to the created/existing DuckDB
    """
    if output_db is None:
        output_db = DEFAULT_OUTPUT_DB

    # Check if extraction needed
    if not check_extraction_needed(output_db, force):
        if PREFECT_AVAILABLE:
            logger = get_run_logger()
            logger.info(f"Skipping extraction - DB exists with data: {output_db}")
        else:
            print(f"Skipping extraction - DB exists with data: {output_db}")
        return output_db

    # Load category mapping
    category_config = load_category_mapping()

    # Find pickle files
    exp_path = MLFLOW_ROOT / IMPUTATION_EXPERIMENT
    pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))

    if PREFECT_AVAILABLE:
        logger = get_run_logger()
        logger.info(f"Found {len(pickle_files)} imputation pickle files")
    else:
        print(f"Found {len(pickle_files)} imputation pickle files")

    # Ensure output directory
    output_db.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing if force
    if output_db.exists():
        output_db.unlink()

    # Connect to DuckDB
    conn = duckdb.connect(str(output_db))

    # Create table
    conn.execute("""
        CREATE TABLE preprocessed_signals (
            config_id INTEGER,
            outlier_method VARCHAR,
            imputation_method VARCHAR,
            preprocessing_category VARCHAR,
            subject_code VARCHAR,
            signal DOUBLE[],
            time_vector DOUBLE[]
        )
    """)

    # Process pickles
    config_id = 0
    total_subjects = 0
    all_rows = []

    for idx, pickle_path in enumerate(pickle_files):
        if idx % 20 == 0:
            msg = f"Processing {idx + 1}/{len(pickle_files)}..."
            if PREFECT_AVAILABLE:
                logger.info(msg)
            else:
                print(msg, flush=True)

        # Get run name
        run_dir = pickle_path.parent.parent.parent
        tags_file = run_dir / "tags" / "mlflow.runName"

        if not tags_file.exists():
            continue

        with open(tags_file) as f:
            run_name = f.read().strip()

        imp_method, outlier_method = parse_imputation_run_name(run_name)

        # parse_imputation_run_name returns (None, None) for:
        # - Invalid format (no "__" separator)
        # - Methods not in registry (validates against SINGLE SOURCE OF TRUTH)
        if not imp_method or not outlier_method:
            continue

        # Note: is_garbage_method check is now redundant since registry validation
        # already rejects invalid methods like "anomaly" and "exclude", but we
        # keep it as defense-in-depth
        if is_garbage_method(outlier_method):
            continue

        category = get_outlier_category(outlier_method, category_config)

        if category == "Unknown":
            continue

        try:
            signals = extract_signals_from_pickle(pickle_path)
        except Exception as e:
            if PREFECT_AVAILABLE:
                logger.warning(f"Error extracting {pickle_path.name}: {e}")
            continue

        for sig in signals:
            all_rows.append(
                (
                    config_id,
                    outlier_method,
                    imp_method,
                    category,
                    sig["subject_code"],
                    sig["signal"].tolist(),
                    sig["time_vector"].tolist(),
                )
            )
            total_subjects += 1

        config_id += 1

    # Batch insert
    msg = f"Inserting {len(all_rows)} rows..."
    if PREFECT_AVAILABLE:
        logger.info(msg)
    else:
        print(msg, flush=True)

    for i in range(0, len(all_rows), chunk_size):
        chunk = all_rows[i : i + chunk_size]
        conn.executemany(
            "INSERT INTO preprocessed_signals VALUES (?, ?, ?, ?, ?, ?, ?)",
            chunk,
        )

    # Create indexes
    if PREFECT_AVAILABLE:
        logger.info("Creating indexes...")
    else:
        print("Creating indexes...", flush=True)

    conn.execute(
        "CREATE INDEX idx_config ON preprocessed_signals(outlier_method, imputation_method)"
    )
    conn.execute(
        "CREATE INDEX idx_category ON preprocessed_signals(preprocessing_category)"
    )
    conn.execute("CREATE INDEX idx_subject ON preprocessed_signals(subject_code)")

    conn.close()

    msg = f"Extraction complete: {config_id} configs, {total_subjects} subject-signals"
    if PREFECT_AVAILABLE:
        logger.info(msg)
    else:
        print(msg, flush=True)

    return output_db


# Standalone execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract decomposition signals to DuckDB"
    )
    parser.add_argument("--force", action="store_true", help="Force re-extraction")
    parser.add_argument("--output", type=Path, help="Output DB path")

    args = parser.parse_args()

    result = extract_decomposition_signals_task(
        output_db=args.output,
        force=args.force,
    )
    print(f"Output: {result}")
