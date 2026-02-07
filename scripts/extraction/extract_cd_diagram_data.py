#!/usr/bin/env python3
"""
extract_cd_diagram_data.py - Extract per-iteration bootstrap metrics for CD diagrams.

This script:
1. Scans MLflow artifacts for metrics pickle files
2. Extracts per-iteration AUROC values (1000 iterations per run)
3. Saves to a separate DuckDB for CD diagram generation

Output: cd_diagram_data.duckdb with tables:
- per_iteration_metrics: All 1000 AUROC values per run
- aggregated_for_cd: Pivoted data ready for Friedman test

CRITICAL NOTE (2026-02-01):
This script was identified as having CRITICAL memory issues - it accumulated
542K records (542 pickles × 1000 iterations) in memory before any DB write.
Fixed to use streaming inserts with gc.collect() after each pickle.
See: CRITICAL-FAILURE-005 and docs/planning/robustify-mlruns-extraction.md

Usage:
    python scripts/extract_cd_diagram_data.py
    python scripts/extract_cd_diagram_data.py --output /path/to/output.duckdb
"""

import argparse
import gc
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb

# MLflow runs location - use centralized path utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data_io.registry import parse_run_name as registry_parse_run_name
from src.utils.paths import get_mlruns_dir

MLRUNS_PATH = get_mlruns_dir()

# Output database location
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent
    / "sci-llm-writer"
    / "manuscripts"
    / "foundationPLR"
    / "data"
    / "cd_diagram_data.duckdb"
)


def parse_run_name(run_name: str) -> Dict[str, str] | None:
    """
    Parse run name to extract configuration with registry validation.

    Example run names:
    - XGBOOST_eval-auc__simple1.0__SAITS__ensemble-LOF-MOMENT-...
    - LogisticRegression_eval-roc_auc__simple1.0__MOMENT-zeroshot__OneClassSVM
    - CATBOOST_eval-auc__embeddings__SAITS__MOMENT-gt

    Returns dict with: classifier, featurization, imputation_method, outlier_method
    Returns None if the run contains invalid methods (not in registry).

    CRITICAL: Uses registry validation to skip orphan/test runs.
    See: .claude/rules/05-registry-source-of-truth.md
    """
    parts = run_name.split("__")

    # First part: CLASSIFIER_eval-metric
    classifier_part = parts[0] if parts else ""
    classifier_match = re.match(r"^([A-Za-z]+)", classifier_part)
    classifier = classifier_match.group(1) if classifier_match else "Unknown"

    # Map classifier names
    classifier_map = {
        "XGBOOST": "XGBoost",
        "CATBOOST": "CatBoost",
        "LogisticRegression": "LogisticRegression",
        "TabPFN": "TabPFN",
    }
    classifier = classifier_map.get(classifier, classifier)

    # Second part: featurization (simple1.0, embeddings, handcrafted)
    featurization = parts[1] if len(parts) > 1 else "Unknown"

    # Third part: imputation method
    imputation = parts[2] if len(parts) > 2 else "Unknown"

    # Fourth part: outlier method
    outlier = parts[3] if len(parts) > 3 else "Unknown"

    # Validate against registry using the canonical run name format
    # The registry expects: signal__outlier__imputation__classifier
    # But pickle files use: CLASSIFIER_metric__featurization__imputation__outlier
    # We need to validate outlier and imputation against the registry
    canonical_run_name = f"pupil-full__{outlier}__{imputation}__{classifier}"
    parsed = registry_parse_run_name(
        canonical_run_name, require_valid=True, log_invalid=False
    )

    if parsed is None:
        # Run contains invalid methods - skip it
        return None

    return {
        "classifier": classifier,
        "featurization": featurization,
        "imputation_method": imputation,
        "outlier_method": outlier,
        "run_name": run_name,
    }


def find_metrics_pickles(mlruns_path: Path) -> List[Path]:
    """Find all metrics pickle files in MLflow runs."""
    pickles = []
    for exp_dir in mlruns_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in [".trash", "models", "0"]:
            continue

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metrics_dir = run_dir / "artifacts" / "metrics"
            if metrics_dir.exists():
                for pickle_file in metrics_dir.glob("metrics_*.pickle"):
                    pickles.append(pickle_file)

    return pickles


def extract_per_iteration_auroc(
    pickle_path: Path,
) -> Tuple[Dict[str, str] | None, List[float]]:
    """
    Extract per-iteration AUROC values from a metrics pickle.

    Returns:
        (config_dict, auroc_list) where auroc_list has 1000 values.
        config_dict is None if the run contains invalid methods (not in registry).
    """
    with open(pickle_path, "rb") as f:
        metrics = pickle.load(f)

    # Parse run name from filename
    run_name = pickle_path.stem.replace("metrics_", "")
    config = parse_run_name(run_name)

    # If config is None, the run contains invalid methods - skip it
    if config is None:
        return None, []

    # Extract per-iteration AUROC
    auroc_list = []
    try:
        auroc_list = metrics["metrics_iter"]["test"]["metrics"]["scalars"]["AUROC"]
    except (KeyError, TypeError):
        print(f"Warning: Could not extract AUROC from {pickle_path.name}")

    return config, auroc_list


def create_cd_database(
    pickles: List[Path], output_path: Path, verbose: bool = True
) -> None:
    """
    Create DuckDB database with per-iteration metrics for CD diagrams.

    FIXED (2026-02-01): Uses STREAMING inserts instead of accumulating all
    542K records in memory. Each pickle's data is inserted immediately and
    memory is freed via gc.collect().
    """
    # Create DuckDB FIRST, then stream inserts
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file if it exists
    if output_path.exists():
        output_path.unlink()

    conn = duckdb.connect(str(output_path))

    # Create main table with per-iteration data
    conn.execute("""
        CREATE TABLE per_iteration_metrics (
            classifier VARCHAR,
            featurization VARCHAR,
            imputation_method VARCHAR,
            outlier_method VARCHAR,
            run_name VARCHAR,
            iteration INTEGER,
            auroc DOUBLE
        )
    """)

    # STREAMING INSERT: Process each pickle and insert immediately
    # This prevents accumulating 542K records in memory
    total_records = 0
    successful_pickles = 0
    unique_classifiers = set()
    unique_outliers = set()
    unique_imputations = set()

    skipped_invalid = 0
    for i, pickle_path in enumerate(pickles):
        if verbose and i % 50 == 0:
            import psutil

            mem_gb = psutil.Process().memory_info().rss / (1024**3)
            print(
                f"Processing {i + 1}/{len(pickles)}: {pickle_path.name[:40]}... (mem: {mem_gb:.2f}GB)"
            )

        try:
            config, auroc_list = extract_per_iteration_auroc(pickle_path)

            # Skip runs with invalid methods (not in registry)
            if config is None:
                skipped_invalid += 1
                continue

            if auroc_list:
                # Create records for THIS pickle only
                records = [
                    (
                        config["classifier"],
                        config["featurization"],
                        config["imputation_method"],
                        config["outlier_method"],
                        config["run_name"],
                        iter_idx,
                        auroc,
                    )
                    for iter_idx, auroc in enumerate(auroc_list)
                ]

                # STREAM INSERT: Insert immediately, don't accumulate
                conn.executemany(
                    """
                    INSERT INTO per_iteration_metrics
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )

                total_records += len(records)
                successful_pickles += 1
                unique_classifiers.add(config["classifier"])
                unique_outliers.add(config["outlier_method"])
                unique_imputations.add(config["imputation_method"])

                # Free memory immediately
                del records, auroc_list
                gc.collect()

        except Exception as e:
            print(f"Error processing {pickle_path.name}: {e}")

    if total_records == 0:
        print("No records extracted!")
        conn.close()
        return

    print(f"\nExtracted {total_records} records from {successful_pickles} pickle files")
    print(
        f"Skipped {skipped_invalid} pickle files with invalid methods (not in registry)"
    )
    print(f"Unique classifiers: {len(unique_classifiers)}")
    print(f"Unique outlier methods: {len(unique_outliers)}")
    print(f"Unique imputation methods: {len(unique_imputations)}")

    # Create aggregated view for CD diagrams
    # This creates the wide-format data needed for Friedman test
    conn.execute("""
        CREATE TABLE aggregated_by_config AS
        SELECT
            outlier_method || '__' || imputation_method as config,
            classifier,
            featurization,
            AVG(auroc) as mean_auroc,
            STDDEV(auroc) as std_auroc,
            MIN(auroc) as min_auroc,
            MAX(auroc) as max_auroc,
            COUNT(*) as n_iterations
        FROM per_iteration_metrics
        GROUP BY outlier_method, imputation_method, classifier, featurization
        ORDER BY config, classifier
    """)

    # Create pivot table for classifier comparison (rows=configs, cols=classifiers)
    conn.execute("""
        CREATE TABLE cd_classifier_pivot AS
        SELECT
            config,
            MAX(CASE WHEN classifier = 'CatBoost' THEN mean_auroc END) as CatBoost,
            MAX(CASE WHEN classifier = 'XGBoost' THEN mean_auroc END) as XGBoost,
            MAX(CASE WHEN classifier = 'TabPFN' THEN mean_auroc END) as TabPFN,
            MAX(CASE WHEN classifier = 'LogisticRegression' THEN mean_auroc END) as LogisticRegression
        FROM aggregated_by_config
        WHERE featurization = 'simple1.0'
        GROUP BY config
        HAVING COUNT(DISTINCT classifier) >= 2
    """)

    # Summary statistics
    print("\n=== DATABASE SUMMARY ===")
    print(f"Output: {output_path}")

    result = conn.execute("SELECT COUNT(*) FROM per_iteration_metrics").fetchone()
    print(f"Total records: {result[0]}")

    result = conn.execute(
        "SELECT COUNT(DISTINCT config) FROM aggregated_by_config"
    ).fetchone()
    print(f"Unique configs: {result[0]}")

    result = conn.execute("SELECT * FROM cd_classifier_pivot LIMIT 5").fetchdf()
    print("\nSample CD pivot data:")
    print(result.to_string())

    conn.close()
    print(f"\nDatabase created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract CD diagram data from MLflow")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output DuckDB path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mlruns",
        "-m",
        type=Path,
        default=MLRUNS_PATH,
        help=f"MLflow runs path (default: {MLRUNS_PATH})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    # Find all metrics pickles
    print(f"Scanning {args.mlruns} for metrics pickles...")
    pickles = find_metrics_pickles(args.mlruns)
    print(f"Found {len(pickles)} pickle files")

    if not pickles:
        print("No pickle files found!")
        sys.exit(1)

    # Create database
    create_cd_database(pickles, args.output, args.verbose)


if __name__ == "__main__":
    main()
