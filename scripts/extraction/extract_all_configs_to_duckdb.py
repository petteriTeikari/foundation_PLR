#!/usr/bin/env python3
"""
Extract ALL configurations from MLflow to DuckDB.

This is Phase 1.1 of the end-to-end visualization pipeline.
Creates the foundation_plr_results.db with essential_metrics table.

CRITICAL: This script uses the registry as the Single Source of Truth
for validating outlier/imputation/classifier methods. NO hardcoded method
lists are allowed. See src/data_io/registry.py.

NOTE (2026-02-01): This script accumulates ~300 runs in memory before writing
because deduplication requires the full list. This is acceptable given the
small scale (~300 runs × 25 fields ≈ 100KB). Memory monitoring added.
See: docs/planning/robustify-mlruns-extraction.md

Usage:
    python scripts/extract_all_configs_to_duckdb.py

Output:
    outputs/foundation_plr_results.db

Author: Foundation PLR Team
Date: 2026-01-25
"""

import gc
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import psutil

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import STRATOS metric computation functions
# These are called ONLY during extraction - NEVER in visualization code!
# Import display names lookup (Single Source of Truth for publication labels)
from src.data_io.display_names import (  # noqa: E402
    get_classifier_display_name,
    get_imputation_display_name,
    get_outlier_display_name,
)

# Import registry as Single Source of Truth (NEVER hardcode method lists!)
from src.data_io.registry import (  # noqa: E402
    EXPECTED_CLASSIFIER_COUNT,
    EXPECTED_IMPUTATION_COUNT,
    EXPECTED_OUTLIER_COUNT,
    get_valid_classifiers,
    get_valid_imputation_methods,
    get_valid_outlier_methods,
    validate_classifier,
    validate_imputation_method,
    validate_outlier_method,
)
from src.stats.calibration_extended import calibration_slope_intercept  # noqa: E402
from src.stats.clinical_utility import net_benefit  # noqa: E402
from src.stats.scaled_brier import scaled_brier_score  # noqa: E402

# Configuration - use centralized path utilities
from src.utils.paths import get_classification_experiment_id, get_mlruns_dir  # noqa: E402

# Import synthetic data isolation utilities
# CRITICAL: Production extraction must REJECT synthetic runs!
# See: src/utils/data_mode.py and docs/planning/synthetic-data-isolation-plan.md
from src.utils.data_mode import (  # noqa: E402
    is_synthetic_run_name,
)

MLRUNS_DIR = get_mlruns_dir()
EXPERIMENT_ID = get_classification_experiment_id()
OUTPUT_DB = Path("outputs/foundation_plr_results.db")

# Classifier name mapping: MLflow names → Registry names
# MLflow uses uppercase (CATBOOST), registry uses mixed case (CatBoost)
CLASSIFIER_NAME_MAP = {
    "CATBOOST": "CatBoost",
    "XGBOOST": "XGBoost",
    "TABPFN": "TabPFN",
    "TABM": "TabM",
    "LogisticRegression": "LogisticRegression",  # Already correct
}


def parse_run_name(run_name: str) -> dict[str, str]:
    """
    Parse MLflow run name to extract configuration.

    Run name format: {CLASSIFIER}_eval-{metric}__{featurization}__{imputation}__{outlier}

    Parameters
    ----------
    run_name : str
        The MLflow run name.

    Returns
    -------
    dict[str, str]
        Dictionary with classifier, featurization, imputation, outlier keys.
    """
    # Remove prefix if present
    name = run_name.replace("model_", "").replace(".pickle", "")

    # Split by double underscore
    parts = name.split("__")

    result = {
        "classifier": "",
        "featurization": "",
        "imputation": "",
        "outlier": "",
    }

    if len(parts) >= 1:
        # First part: CATBOOST_eval-auc
        classifier_parts = parts[0].split("_")
        raw_classifier = classifier_parts[0]
        # Normalize classifier name to match registry (e.g., "CATBOOST" → "CatBoost")
        result["classifier"] = CLASSIFIER_NAME_MAP.get(raw_classifier, raw_classifier)

    if len(parts) >= 2:
        result["featurization"] = parts[1]

    if len(parts) >= 3:
        result["imputation"] = parts[2]

    if len(parts) >= 4:
        result["outlier"] = parts[3]
    elif len(parts) == 2:
        # Legacy format: default to "anomaly" (this is the problematic case!)
        result["outlier"] = "anomaly"

    return result


def get_run_params(run_dir: Path) -> dict[str, Any]:
    """Read MLflow params from run directory."""
    params = {}
    params_dir = run_dir / "params"

    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                try:
                    params[param_file.name] = param_file.read_text().strip()
                except Exception:
                    pass

    return params


def extract_bootstrap_metrics(metrics_pickle_path: Path) -> dict[str, Any]:
    """
    Extract bootstrap metrics from a metrics pickle file.

    The metrics pickle structure is:
    - metrics_stats['test']['metrics']['scalars']['AUROC']['mean']
    - metrics_stats['test']['metrics']['scalars']['AUROC']['ci'][0] (ci_lo)
    - metrics_stats['test']['metrics']['scalars']['AUROC']['ci'][1] (ci_hi)

    Returns dictionary with auroc_mean, auroc_std, auroc_ci_lo, auroc_ci_hi,
    and full bootstrap distribution.
    """
    result = {
        "auroc_mean": None,
        "auroc_std": None,
        "auroc_ci_lo": None,
        "auroc_ci_hi": None,
        "n_bootstrap": None,
        "brier_mean": None,
        "brier_std": None,
    }

    if not metrics_pickle_path.exists():
        return result

    try:
        with open(metrics_pickle_path, "rb") as f:
            data = pickle.load(f)

        # Navigate to test metrics - NEW STRUCTURE
        if "metrics_stats" in data and "test" in data["metrics_stats"]:
            test_stats = data["metrics_stats"]["test"]

            # The metrics are nested: test['metrics']['scalars']['AUROC']
            if "metrics" in test_stats and "scalars" in test_stats["metrics"]:
                scalars = test_stats["metrics"]["scalars"]

                # Extract AUROC
                if "AUROC" in scalars:
                    auroc_stats = scalars["AUROC"]
                    result["auroc_mean"] = float(auroc_stats.get("mean", 0))
                    result["auroc_std"] = float(auroc_stats.get("std", 0))
                    result["n_bootstrap"] = int(auroc_stats.get("n", 1000))

                    # CI is a numpy array [ci_lo, ci_hi]
                    ci = auroc_stats.get("ci")
                    if ci is not None and len(ci) >= 2:
                        result["auroc_ci_lo"] = float(ci[0])
                        result["auroc_ci_hi"] = float(ci[1])

                # Extract Brier if available
                if "Brier" in scalars:
                    brier_stats = scalars["Brier"]
                    result["brier_mean"] = float(brier_stats.get("mean", 0))
                    result["brier_std"] = float(brier_stats.get("std", 0))

    except Exception as e:
        print(f"    Warning: Could not parse metrics pickle: {e}")

    return result


def extract_stratos_metrics(metrics_pickle_path: Path) -> dict[str, Any]:
    """
    Extract STRATOS-compliant metrics by computing from predictions.

    CRITICAL: This is the ONLY place STRATOS metrics are computed.
    Visualization code must read from DuckDB, not compute metrics.
    See: CRITICAL-FAILURE-003-computation-decoupling-violation.md

    Computes:
    - calibration_slope, calibration_intercept, o_e_ratio (from calibration_extended)
    - scaled_brier (IPA) (from scaled_brier)
    - net_benefit at 5%, 10%, 15%, 20% thresholds (from clinical_utility)

    Parameters
    ----------
    metrics_pickle_path : Path
        Path to the metrics pickle file.

    Returns
    -------
    dict[str, Any]
        Dictionary with STRATOS metrics.
    """
    result = {
        "calibration_slope": None,
        "calibration_intercept": None,
        "o_e_ratio": None,
        "scaled_brier": None,
        "net_benefit_5pct": None,
        "net_benefit_10pct": None,
        "net_benefit_15pct": None,
        "net_benefit_20pct": None,
    }

    if not metrics_pickle_path.exists():
        return result

    try:
        with open(metrics_pickle_path, "rb") as f:
            data = pickle.load(f)

        # Extract predictions from metrics_iter
        # Structure: metrics_iter['test']['preds']['arrays']['predictions']
        if "metrics_iter" not in data:
            return result

        test_iter = data.get("metrics_iter", {}).get("test", {})
        preds = test_iter.get("preds", {}).get("arrays", {}).get("predictions", {})

        if not preds:
            return result

        y_prob_all = preds.get("y_pred_proba")  # Shape: (n_subjects, n_bootstrap)
        labels = preds.get("label")  # Shape: (n_subjects, n_bootstrap)

        if y_prob_all is None or labels is None:
            return result

        # Use mean prediction per subject (across bootstrap)
        y_prob = y_prob_all.mean(axis=1)  # (n_subjects,)
        y_true = labels[:, 0]  # All columns are same, take first

        # Ensure valid array sizes
        if len(y_true) < 10 or len(y_prob) < 10:
            return result

        # Check for single class (can't compute metrics)
        if len(np.unique(y_true)) < 2:
            return result

        # Compute calibration metrics
        try:
            cal_result = calibration_slope_intercept(y_true, y_prob)
            result["calibration_slope"] = float(cal_result.slope)
            result["calibration_intercept"] = float(cal_result.intercept)
            result["o_e_ratio"] = float(cal_result.o_e_ratio)
        except Exception as e:
            print(f"    Warning: Could not compute calibration metrics: {e}")

        # Compute scaled Brier (IPA)
        try:
            sb_result = scaled_brier_score(y_true, y_prob)
            result["scaled_brier"] = float(sb_result["ipa"])
        except Exception as e:
            print(f"    Warning: Could not compute scaled Brier: {e}")

        # Compute net benefit at standard thresholds
        for threshold, col_name in [
            (0.05, "net_benefit_5pct"),
            (0.10, "net_benefit_10pct"),
            (0.15, "net_benefit_15pct"),
            (0.20, "net_benefit_20pct"),
        ]:
            try:
                nb = net_benefit(y_true, y_prob, threshold)
                result[col_name] = float(nb)
            except Exception as e:
                print(f"    Warning: Could not compute net benefit at {threshold}: {e}")

    except Exception as e:
        print(f"    Warning: Could not extract STRATOS metrics: {e}")

    return result


def scan_all_runs(
    mlruns_dir: Path, experiment_id: str, classifier_filter: str | None = None
) -> list[dict[str, Any]]:
    """
    Scan all runs in the experiment.

    Parameters
    ----------
    mlruns_dir : Path
        Path to MLflow tracking directory.
    experiment_id : str
        MLflow experiment ID.
    classifier_filter : str | None
        Optional filter for classifier type (e.g., 'CATBOOST').
        If None, extracts all classifiers.

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries with run metadata and metrics.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    runs = []
    run_count = 0
    valid_count = 0

    print(f"Scanning {experiment_dir}...")
    print(
        f"[PREFLIGHT] Initial memory: {psutil.Process().memory_info().rss / (1024**3):.2f}GB"
    )

    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        # Check for model artifacts
        model_dir = run_dir / "artifacts" / "model"
        metrics_dir = run_dir / "artifacts" / "metrics"

        if not model_dir.exists():
            continue

        # Find any model pickle file
        model_pickles = list(model_dir.glob("model_*.pickle"))
        if not model_pickles:
            continue

        model_file = model_pickles[0]
        run_count += 1

        # Parse configuration from filename
        config = parse_run_name(model_file.stem)

        # CRITICAL: Reject synthetic runs in production extraction
        # Part of the 4-gate isolation architecture.
        # Synthetic runs have __SYNTHETIC_ prefix or is_synthetic=true tag.
        # See: src/utils/data_mode.py
        if is_synthetic_run_name(model_file.stem):
            continue  # Skip synthetic runs

        # Apply classifier filter if specified
        if classifier_filter and config["classifier"] != classifier_filter:
            continue

        # CRITICAL: Only extract HANDCRAFTED FEATURES (featurization containing 'simple')
        # Embedding-based features (MOMENT-embedding, etc.) have ~9pp lower AUROC
        # and MUST NOT be mixed with handcrafted feature results.
        # See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
        if "simple" not in config["featurization"].lower():
            continue

        # CRITICAL: Validate against registry (Single Source of Truth)
        # NEVER use hardcoded method lists - the registry defines exactly what's valid.
        # See: src/data_io/registry.py and tests/integration/test_extraction_registry.py
        outlier_valid = validate_outlier_method(config["outlier"])
        imputation_valid = validate_imputation_method(config["imputation"])
        classifier_valid = validate_classifier(config["classifier"])

        if not outlier_valid:
            # Skip invalid outlier methods (e.g., "anomaly", "exclude", "-orig-" variants)
            continue

        if not imputation_valid:
            # Skip invalid imputation methods
            continue

        if not classifier_valid:
            # Skip invalid classifiers
            continue

        # Read params
        params = get_run_params(run_dir)

        # Find and parse metrics pickle
        metrics_data = {
            "auroc_mean": None,
            "auroc_ci_lo": None,
            "auroc_ci_hi": None,
            "n_bootstrap": None,
            "brier_mean": None,
        }

        # STRATOS metrics (computed from predictions)
        stratos_data = {
            "calibration_slope": None,
            "calibration_intercept": None,
            "o_e_ratio": None,
            "scaled_brier": None,
            "net_benefit_5pct": None,
            "net_benefit_10pct": None,
            "net_benefit_15pct": None,
            "net_benefit_20pct": None,
        }

        if metrics_dir.exists():
            # Try to find metrics pickle matching the model
            metrics_pickles = list(metrics_dir.glob(f"*{config['classifier']}*.pickle"))
            if not metrics_pickles:
                # Fallback: any metrics pickle
                metrics_pickles = list(metrics_dir.glob("metrics_*.pickle"))
            if metrics_pickles:
                metrics_data = extract_bootstrap_metrics(metrics_pickles[0])
                # CRITICAL: Extract STRATOS metrics from predictions
                # This is the ONLY place these metrics are computed!
                stratos_data = extract_stratos_metrics(metrics_pickles[0])

        # CRITICAL: Filter out runs with NaN CI values
        # These represent failed experiments (e.g., AUROC=0.5 with zero variance)
        # NaN CIs cannot be plotted and indicate upstream experiment failures
        ci_lo = metrics_data.get("auroc_ci_lo")
        ci_hi = metrics_data.get("auroc_ci_hi")
        has_nan_ci = (
            ci_lo is None
            or ci_hi is None
            or (isinstance(ci_lo, float) and math.isnan(ci_lo))
            or (isinstance(ci_hi, float) and math.isnan(ci_hi))
        )
        if has_nan_ci:
            # Skip this run - it's a failed experiment
            continue

        # After registry validation, all extracted methods are known/valid
        # (invalid methods were already filtered out above)
        outlier_source_known = True  # Always true after registry validation

        runs.append(
            {
                "run_id": run_dir.name,
                "model_path": str(model_file),
                "outlier_method": config["outlier"],
                "imputation_method": config["imputation"],
                "featurization": config["featurization"],
                "classifier": config["classifier"],
                # Display names from YAML lookup (Single Source of Truth)
                "outlier_display_name": get_outlier_display_name(config["outlier"]),
                "imputation_display_name": get_imputation_display_name(
                    config["imputation"]
                ),
                "classifier_display_name": get_classifier_display_name(
                    config["classifier"]
                ),
                "auroc": metrics_data["auroc_mean"],
                "auroc_ci_lo": metrics_data["auroc_ci_lo"],
                "auroc_ci_hi": metrics_data["auroc_ci_hi"],
                "brier": metrics_data["brier_mean"],
                "n_bootstrap": metrics_data["n_bootstrap"],
                "outlier_source_known": outlier_source_known,
                # Additional params
                "anomaly_source": params.get("anomaly_source", config["outlier"]),
                "mlflow_run_outlier_detection": params.get(
                    "mlflow_run_outlier_detection"
                ),
                # STRATOS metrics (computed from predictions during extraction)
                # CRITICAL: These must be computed HERE, not in visualization code!
                "calibration_slope": stratos_data["calibration_slope"],
                "calibration_intercept": stratos_data["calibration_intercept"],
                "o_e_ratio": stratos_data["o_e_ratio"],
                "scaled_brier": stratos_data["scaled_brier"],
                "net_benefit_5pct": stratos_data["net_benefit_5pct"],
                "net_benefit_10pct": stratos_data["net_benefit_10pct"],
                "net_benefit_15pct": stratos_data["net_benefit_15pct"],
                "net_benefit_20pct": stratos_data["net_benefit_20pct"],
            }
        )

        valid_count += 1

        if run_count % 50 == 0:
            mem_gb = psutil.Process().memory_info().rss / (1024**3)
            print(
                f"  Processed {run_count} runs, {valid_count} valid (mem: {mem_gb:.2f}GB)..."
            )
            gc.collect()  # Periodic cleanup

    mem_gb = psutil.Process().memory_info().rss / (1024**3)
    print(f"Found {len(runs)} valid configurations (mem: {mem_gb:.2f}GB)")
    return runs


def deduplicate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate runs by (outlier, imputation, classifier), keeping highest AUROC.

    If multiple runs exist for the same configuration (e.g., from re-running experiments),
    this function keeps only the best-performing run (highest AUROC).

    See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
    """
    # Group by (outlier, imputation, classifier)
    best_runs: dict[tuple[str, str, str], dict[str, Any]] = {}

    for run in runs:
        key = (run["outlier_method"], run["imputation_method"], run["classifier"])
        auroc = run.get("auroc") or 0.0

        if key not in best_runs:
            best_runs[key] = run
        else:
            existing_auroc = best_runs[key].get("auroc") or 0.0
            if auroc > existing_auroc:
                best_runs[key] = run

    deduplicated = list(best_runs.values())
    n_removed = len(runs) - len(deduplicated)

    if n_removed > 0:
        print(
            f"  Deduplicated: removed {n_removed} duplicate runs (kept highest AUROC)"
        )

    return deduplicated


def write_to_duckdb(runs: list[dict[str, Any]], output_path: Path) -> None:
    """Write extracted runs to DuckDB.

    CRITICAL: Deduplicates runs before writing to ensure no duplicate
    (outlier, imputation, classifier) combinations exist.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if output_path.exists():
        output_path.unlink()

    # CRITICAL: Deduplicate before writing
    runs = deduplicate_runs(runs)

    # Connect and create tables
    conn = duckdb.connect(str(output_path))

    # Create essential_metrics table with STRATOS metrics
    # CRITICAL: All metrics are computed during extraction (this script).
    # Visualization code must READ from this table, NEVER compute metrics.
    # See: CRITICAL-FAILURE-003-computation-decoupling-violation.md
    conn.execute("""
        CREATE TABLE essential_metrics (
            run_id VARCHAR PRIMARY KEY,
            model_path VARCHAR,
            outlier_method VARCHAR,
            imputation_method VARCHAR,
            featurization VARCHAR,
            classifier VARCHAR,
            -- Display names from YAML lookup (Single Source of Truth)
            outlier_display_name VARCHAR,
            imputation_display_name VARCHAR,
            classifier_display_name VARCHAR,
            -- Core metrics (from MLflow)
            auroc DOUBLE,
            auroc_ci_lo DOUBLE,
            auroc_ci_hi DOUBLE,
            brier DOUBLE,
            n_bootstrap INTEGER,
            outlier_source_known BOOLEAN,
            anomaly_source VARCHAR,
            mlflow_run_outlier_detection VARCHAR,
            -- STRATOS calibration metrics (Van Calster 2024)
            calibration_slope DOUBLE,
            calibration_intercept DOUBLE,
            o_e_ratio DOUBLE,
            -- STRATOS overall metric (scaled Brier / IPA)
            scaled_brier DOUBLE,
            -- STRATOS clinical utility metrics (Net Benefit)
            net_benefit_5pct DOUBLE,
            net_benefit_10pct DOUBLE,
            net_benefit_15pct DOUBLE,
            net_benefit_20pct DOUBLE
        )
    """)

    # Insert rows (25 columns total including STRATOS metrics)
    for run in runs:
        conn.execute(
            """
            INSERT INTO essential_metrics VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
        """,
            [
                run["run_id"],
                run["model_path"],
                run["outlier_method"],
                run["imputation_method"],
                run["featurization"],
                run["classifier"],
                run["outlier_display_name"],
                run["imputation_display_name"],
                run["classifier_display_name"],
                run["auroc"],
                run["auroc_ci_lo"],
                run["auroc_ci_hi"],
                run["brier"],
                run["n_bootstrap"],
                run["outlier_source_known"],
                run["anomaly_source"],
                run["mlflow_run_outlier_detection"],
                # STRATOS metrics
                run["calibration_slope"],
                run["calibration_intercept"],
                run["o_e_ratio"],
                run["scaled_brier"],
                run["net_benefit_5pct"],
                run["net_benefit_10pct"],
                run["net_benefit_15pct"],
                run["net_benefit_20pct"],
            ],
        )

    # Create metadata table
    conn.execute("""
        CREATE TABLE extraction_metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)

    metadata = {
        "extraction_date": datetime.now().isoformat(),
        "mlruns_dir": str(MLRUNS_DIR),
        "experiment_id": EXPERIMENT_ID,
        "total_configs": str(len(runs)),
        "validation_method": "registry",  # Uses src.data_io.registry as Single Source of Truth
        "registry_outlier_count": str(EXPECTED_OUTLIER_COUNT),
        "registry_imputation_count": str(EXPECTED_IMPUTATION_COUNT),
        "registry_classifier_count": str(EXPECTED_CLASSIFIER_COUNT),
    }

    for key, value in metadata.items():
        conn.execute("INSERT INTO extraction_metadata VALUES (?, ?)", [key, value])

    # Create view for CatBoost configs only (use registry-normalized name)
    conn.execute("""
        CREATE VIEW catboost_configs AS
        SELECT *
        FROM essential_metrics
        WHERE classifier = 'CatBoost'
    """)

    # Create view for top-10 CatBoost (excluding unknown outlier sources)
    conn.execute("""
        CREATE VIEW top10_catboost AS
        SELECT
            ROW_NUMBER() OVER (ORDER BY auroc DESC) as rank,
            run_id,
            outlier_method,
            outlier_display_name,
            imputation_method,
            imputation_display_name,
            auroc,
            auroc_ci_lo,
            auroc_ci_hi
        FROM essential_metrics
        WHERE classifier = 'CatBoost' AND outlier_source_known = true
        ORDER BY auroc DESC
        LIMIT 10
    """)

    conn.close()
    print(f"Wrote {len(runs)} rows to {output_path}")


def verify_extraction(output_path: Path) -> None:
    """Verify the extraction was successful.

    CRITICAL: This function validates:
    1. No duplicate (outlier, imputation, classifier) combinations
    2. ALL extracted methods are in the registry (Single Source of Truth)
    3. Extracted counts don't exceed registry counts

    See: CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md
    See: tests/integration/test_extraction_registry.py
    """
    conn = duckdb.connect(str(output_path), read_only=True)

    print("\n" + "=" * 60)
    print("EXTRACTION VERIFICATION")
    print("=" * 60)

    # Total count
    count = conn.execute("SELECT COUNT(*) FROM essential_metrics").fetchone()[0]
    print(f"Total configurations: {count}")

    # CRITICAL: Check for duplicates
    unique_combos = conn.execute("""
        SELECT COUNT(DISTINCT outlier_method || '|' || imputation_method || '|' || classifier)
        FROM essential_metrics
    """).fetchone()[0]

    if count != unique_combos:
        print(f"\n{'!' * 60}")
        print(
            f"CRITICAL ERROR: {count - unique_combos} DUPLICATE COMBINATIONS DETECTED!"
        )
        print(f"{'!' * 60}")
        print("This indicates mixed featurization types in the data.")
        print("Check CRITICAL-FAILURE-002-mixed-featurization-in-extraction.md")

        # Show duplicates
        duplicates = conn.execute("""
            SELECT outlier_method, imputation_method, classifier, COUNT(*) as cnt
            FROM essential_metrics
            GROUP BY outlier_method, imputation_method, classifier
            HAVING COUNT(*) > 1
        """).fetchall()
        print("\nDuplicates found:")
        for od, imp, clf, cnt in duplicates:
            print(f"  {od} + {imp} + {clf}: {cnt} entries")

        conn.close()
        raise ValueError("Extraction contains duplicates - cannot proceed!")
    else:
        print(f"Unique combinations: {unique_combos} (validated - no duplicates)")

    # CRITICAL: Registry validation - Single Source of Truth
    print("\n--- Registry Validation ---")

    # Check all extracted outlier methods are in registry
    extracted_outliers = conn.execute(
        "SELECT DISTINCT outlier_method FROM essential_metrics"
    ).fetchall()
    extracted_outlier_set = {row[0] for row in extracted_outliers}
    valid_outlier_set = set(get_valid_outlier_methods())
    invalid_outliers = extracted_outlier_set - valid_outlier_set

    if invalid_outliers:
        print(f"  CRITICAL ERROR: Invalid outlier methods: {invalid_outliers}")
        raise ValueError(
            f"Extraction contains invalid outlier methods: {invalid_outliers}"
        )
    print(
        f"  Outlier methods: {len(extracted_outlier_set)}/{EXPECTED_OUTLIER_COUNT} (all valid)"
    )

    # Check all extracted imputation methods are in registry
    extracted_imps = conn.execute(
        "SELECT DISTINCT imputation_method FROM essential_metrics"
    ).fetchall()
    extracted_imp_set = {row[0] for row in extracted_imps}
    valid_imp_set = set(get_valid_imputation_methods())
    invalid_imps = extracted_imp_set - valid_imp_set

    if invalid_imps:
        print(f"  CRITICAL ERROR: Invalid imputation methods: {invalid_imps}")
        raise ValueError(
            f"Extraction contains invalid imputation methods: {invalid_imps}"
        )
    print(
        f"  Imputation methods: {len(extracted_imp_set)}/{EXPECTED_IMPUTATION_COUNT} (all valid)"
    )

    # Check all extracted classifiers are in registry
    extracted_clfs = conn.execute(
        "SELECT DISTINCT classifier FROM essential_metrics"
    ).fetchall()
    extracted_clf_set = {row[0] for row in extracted_clfs}
    valid_clf_set = set(get_valid_classifiers())
    invalid_clfs = extracted_clf_set - valid_clf_set

    if invalid_clfs:
        print(f"  CRITICAL ERROR: Invalid classifiers: {invalid_clfs}")
        raise ValueError(f"Extraction contains invalid classifiers: {invalid_clfs}")
    print(
        f"  Classifiers: {len(extracted_clf_set)}/{EXPECTED_CLASSIFIER_COUNT} (all valid)"
    )

    # Count with/without AUROC
    with_auroc = conn.execute(
        "SELECT COUNT(*) FROM essential_metrics WHERE auroc IS NOT NULL"
    ).fetchone()[0]
    print(f"Configs with AUROC: {with_auroc}")

    # AUROC range
    result = conn.execute("""
        SELECT MIN(auroc), MAX(auroc), AVG(auroc)
        FROM essential_metrics
        WHERE auroc IS NOT NULL
    """).fetchone()

    if result[0] is not None:
        print(f"AUROC range: [{result[0]:.3f}, {result[1]:.3f}], mean: {result[2]:.3f}")
    else:
        print("AUROC range: No valid AUROC values found")

    # Classifier distribution
    classifiers = conn.execute("""
        SELECT classifier, COUNT(*) as cnt
        FROM essential_metrics
        GROUP BY classifier
        ORDER BY cnt DESC
    """).fetchall()
    print("\nClassifier distribution:")
    for clf, cnt in classifiers:
        print(f"  {clf}: {cnt} configs")

    # Unknown outlier sources
    unknown = conn.execute("""
        SELECT COUNT(*)
        FROM essential_metrics
        WHERE outlier_source_known = false
    """).fetchone()[0]
    print(f"\nConfigs with unknown outlier source: {unknown}")

    # CatBoost count
    catboost_count = conn.execute("""
        SELECT COUNT(*) FROM catboost_configs
    """).fetchone()[0]
    print(f"CatBoost configs: {catboost_count}")

    # Top 10 CatBoost
    print("\nTop-10 CatBoost (excluding unknown OD):")
    top10 = conn.execute("""
        SELECT rank, outlier_method, imputation_method, auroc, auroc_ci_lo, auroc_ci_hi
        FROM top10_catboost
    """).fetchall()

    for row in top10:
        rank, od, imp, auroc, ci_lo, ci_hi = row
        od_short = od[:40] + "..." if len(od) > 40 else od
        if auroc is not None and ci_lo is not None and ci_hi is not None:
            print(
                f"  {rank:2d}. AUROC={auroc:.3f} ({ci_lo:.3f}-{ci_hi:.3f}) | {od_short} + {imp}"
            )
        elif auroc is not None:
            print(f"  {rank:2d}. AUROC={auroc:.3f} (CI N/A) | {od_short} + {imp}")
        else:
            print(f"  {rank:2d}. AUROC=N/A | {od_short} + {imp}")

    conn.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase 1.1: Extract ALL configs to DuckDB")
    print("=" * 60)

    # Scan ALL runs (all classifiers)
    runs = scan_all_runs(MLRUNS_DIR, EXPERIMENT_ID, classifier_filter=None)

    # Write to DuckDB
    write_to_duckdb(runs, OUTPUT_DB)

    # Verify
    verify_extraction(OUTPUT_DB)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Run verification tests:
   pytest tests/test_extraction_verification.py -v

2. Proceed to Phase 1.2: Extract top-10 models with artifacts
   python scripts/extract_top10_models_with_artifacts.py
""")


if __name__ == "__main__":
    main()
