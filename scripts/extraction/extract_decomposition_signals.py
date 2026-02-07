#!/usr/bin/env python3
"""
Extract per-subject preprocessed signals to DuckDB for decomposition analysis.

This script:
1. Iterates through MLflow imputation pickle files
2. Parses run names to get (outlier_method, imputation_method)
3. Maps to preprocessing category using category_mapping.yaml
4. Stores per-subject signals in DuckDB

Output: data/private/preprocessed_signals_per_subject.db

Usage:
    uv run python scripts/extract_decomposition_signals.py

Expected Input Data (as of 2026-02-01):
=====================================
The imputation experiment (MLflow ID: 940304421003085572) contains 136 pickle files.

These represent ALL combinations of (imputation_method × outlier_method) tested:

| Imputation Method | Runs | Notes |
|-------------------|------|-------|
| MOMENT            | 90   | 6 variants (finetune/zeroshot × small/base/large) × ~15 outlier methods |
| SAITS             | 15   | 1 variant × 15 outlier methods |
| CSDI              | 15   | 1 variant × 15 outlier methods |
| TimesNet          | 15   | 1 variant × 15 outlier methods |
| ensemble          | 1    | Special combination |
| TOTAL             | 136  | |

Each pickle contains reconstructed signals for 507 subjects.
Total extracted: 136 × 507 = 68,952 subject-config combinations.

IMPORTANT: This is BEFORE classification/featurization.
The 500+ classification runs come from: 136 × (classifiers) × (featurization types).
"""

import gc
import pickle
import re

# Project paths - use centralized path utilities
import sys
from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_io.registry import validate_imputation_method, validate_outlier_method  # noqa: E402
from src.utils.paths import (  # noqa: E402
    get_mlflow_registry_dir,
    get_mlruns_dir,
    get_preprocessed_signals_db_path,
)

MLFLOW_ROOT = get_mlruns_dir()
IMPUTATION_EXPERIMENT = "940304421003085572"
OUTPUT_DB = get_preprocessed_signals_db_path()
CATEGORY_MAPPING = get_mlflow_registry_dir() / "category_mapping.yaml"

# Invalid/garbage method names to skip (per CLAUDE.md)
GARBAGE_METHODS = {"exclude", "anomaly"}


def _get_preprocessing_categories() -> list[str]:
    """Load preprocessing category display names from config."""
    display_names_path = get_mlflow_registry_dir() / "display_names.yaml"
    with open(display_names_path) as f:
        config = yaml.safe_load(f)
    categories = config.get("categories", {})
    order = [
        "ground_truth",
        "foundation_model",
        "deep_learning",
        "traditional",
        "ensemble",
    ]
    return [categories.get(cat_id, {}).get("display_name", cat_id) for cat_id in order]


def load_category_mapping() -> dict:
    """Load category mapping configuration."""
    with open(CATEGORY_MAPPING) as f:
        return yaml.safe_load(f)


def get_outlier_category(method: str, category_config: dict) -> str:
    """Map outlier method to category."""
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
    # Handle pupil_gt -> pupil-gt
    raw = raw.replace("_gt", "-gt").replace("_orig", "-orig")

    # Handle MOMENT_finetune___gt -> MOMENT-gt-finetune
    if raw.startswith("MOMENT_"):
        mode = raw.split("_")[1]  # finetune or zeroshot
        if "___gt" in raw:
            return f"MOMENT-gt-{mode}"
        elif "___orig" in raw:
            return f"MOMENT-orig-{mode}"
        else:
            return f"MOMENT-{mode}"

    return raw


def parse_run_name(run_name: str) -> tuple[str, str] | None:
    """Parse imputation and outlier method from run name with registry validation.

    Format: {IMPUTATION}_{params}__{OUTLIER}_impPLR_v0.1

    Returns
    -------
    tuple[str, str] | None
        Tuple of (imputation_method, outlier_method) if both are valid according
        to the registry, None if format is invalid or methods are not in registry.

    Note
    ----
    This is specific to imputation experiment run names, which differ from
    classification run names (signal__outlier__imputation__classifier).
    Registry validation ensures only the 11 valid outlier methods and 8 valid
    imputation methods are processed.
    """
    if "__" not in run_name:
        return None

    parts = run_name.split("__")
    if len(parts) < 2:
        return None

    # Imputation: first part before underscore in first segment
    imp_method = parts[0].split("_")[0]

    # Outlier: second segment, remove _impPLR_v0.1 suffix and -Outlier
    outlier_part = parts[1].replace("_impPLR_v0.1", "").replace("-Outlier", "")
    outlier_method = normalize_outlier_method(outlier_part)

    # Registry validation - skip runs with methods not in the registry
    if not validate_imputation_method(imp_method):
        return None
    if not validate_outlier_method(outlier_method):
        return None

    return imp_method, outlier_method


def extract_signals_from_pickle(pickle_path: Path) -> list[dict]:
    """Extract per-subject signals from a pickle file.

    Returns list of dicts with keys:
        - subject_code
        - signal (numpy array)
        - time_vector (numpy array)
    """
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

        # Get arrays
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


def main():
    """Main extraction function."""

    print("=" * 70, flush=True)
    print("Extracting Per-Subject Preprocessed Signals to DuckDB", flush=True)
    print("=" * 70, flush=True)

    # Load category mapping
    category_config = load_category_mapping()

    # Find all imputation pickles
    exp_path = MLFLOW_ROOT / IMPUTATION_EXPERIMENT
    pickle_files = list(exp_path.glob("*/artifacts/imputation/*.pickle"))
    print(f"Found {len(pickle_files)} imputation pickle files", flush=True)

    # Ensure output directory exists
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB (overwrite existing)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    conn = duckdb.connect(str(OUTPUT_DB))

    # Create table
    conn.execute(
        """
        CREATE TABLE preprocessed_signals (
            config_id INTEGER,
            outlier_method VARCHAR,
            imputation_method VARCHAR,
            preprocessing_category VARCHAR,
            subject_code VARCHAR,
            signal DOUBLE[],
            time_vector DOUBLE[]
        )
    """
    )

    # Process each pickle with STREAMING inserts (memory-efficient)
    # NOTE: Previous version collected all rows in memory causing 13GB+ RAM usage
    # and severe swap thrashing. This version inserts after each pickle file.
    config_id = 0
    total_subjects = 0
    skipped_garbage = 0
    skipped_parse_error = 0

    for idx, pickle_path in enumerate(pickle_files):
        # Progress indicator for every file
        print(
            f"Processing {idx + 1}/{len(pickle_files)}: {pickle_path.name}...",
            flush=True,
        )

        # Get run name from tags
        run_dir = pickle_path.parent.parent.parent
        tags_file = run_dir / "tags" / "mlflow.runName"

        if not tags_file.exists():
            skipped_parse_error += 1
            continue

        with open(tags_file) as f:
            run_name = f.read().strip()

        # Parse run name with registry validation
        parsed = parse_run_name(run_name)

        if parsed is None:
            skipped_parse_error += 1
            continue

        imp_method, outlier_method = parsed

        # Skip garbage methods (additional check for methods like "exclude", "anomaly")
        if is_garbage_method(outlier_method):
            skipped_garbage += 1
            continue

        # Get category
        category = get_outlier_category(outlier_method, category_config)

        if category == "Unknown":
            print(f"  Skipping unknown category for {outlier_method}", flush=True)
            continue

        # Extract signals
        try:
            signals = extract_signals_from_pickle(pickle_path)
        except Exception as e:
            print(f"  Error extracting from {pickle_path.name}: {e}", flush=True)
            continue

        # STREAMING INSERT: Insert this pickle's data immediately
        # Don't accumulate in memory across all pickles
        rows = [
            (
                config_id,
                outlier_method,
                imp_method,
                category,
                sig["subject_code"],
                sig["signal"].tolist(),
                sig["time_vector"].tolist(),
            )
            for sig in signals
        ]

        if rows:
            conn.executemany(
                """
                INSERT INTO preprocessed_signals
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                rows,
            )
            total_subjects += len(rows)
            config_id += 1
            print(
                f"  Inserted {len(rows)} subjects ({outlier_method} + {imp_method})",
                flush=True,
            )

        # Explicit garbage collection to free memory after each pickle
        del signals, rows
        gc.collect()

    print("Creating indexes...", flush=True)

    # Create indexes
    conn.execute(
        "CREATE INDEX idx_config ON preprocessed_signals(outlier_method, imputation_method)"
    )
    conn.execute(
        "CREATE INDEX idx_category ON preprocessed_signals(preprocessing_category)"
    )
    conn.execute("CREATE INDEX idx_subject ON preprocessed_signals(subject_code)")

    # Print summary
    print()
    print("=" * 70)
    print("Extraction Summary")
    print("=" * 70)
    print(f"Total configs extracted: {config_id}")
    print(f"Total subject-config rows: {total_subjects}")
    print(f"Skipped (garbage methods): {skipped_garbage}")
    print(f"Skipped (parse errors): {skipped_parse_error}")

    # Show category distribution
    print()
    print("Category Distribution:")
    result = conn.execute(
        """
        SELECT preprocessing_category, COUNT(DISTINCT config_id) as n_configs,
               COUNT(DISTINCT subject_code) as n_subjects
        FROM preprocessed_signals
        GROUP BY preprocessing_category
        ORDER BY n_configs DESC
    """
    ).fetchall()

    for cat, n_configs, n_subjects in result:
        print(f"  {cat}: {n_configs} configs, {n_subjects} unique subjects")

    # Show sample methods per category
    print()
    print("Sample methods per category:")
    for cat in _get_preprocessing_categories():
        methods = conn.execute(
            """
            SELECT DISTINCT outlier_method
            FROM preprocessed_signals
            WHERE preprocessing_category = ?
            LIMIT 3
        """,
            [cat],
        ).fetchall()
        method_list = [m[0] for m in methods]
        print(f"  {cat}: {', '.join(method_list)}")

    conn.close()
    print()
    print(f"Output saved to: {OUTPUT_DB}")


if __name__ == "__main__":
    main()
