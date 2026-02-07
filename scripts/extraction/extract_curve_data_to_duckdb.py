#!/usr/bin/env python3
"""
Extract curve data from MLflow to DuckDB.

This script adds pre-computed curve data to the existing DuckDB:
- predictions: raw (y_true, y_prob) arrays for reproducibility
- roc_curves: FPR, TPR, thresholds for ROC plotting
- calibration_curves: smoothed calibration curve with CI
- dca_curves: Decision Curve Analysis (Net Benefit vs threshold)

CRITICAL: All curve computation happens HERE during extraction.
Visualization code must READ from these tables, NEVER compute curves.
See: CRITICAL-FAILURE-003-computation-decoupling-violation.md

Usage:
    python scripts/extract_curve_data_to_duckdb.py

Requires:
    - Existing foundation_plr_results.db (from extract_all_configs_to_duckdb.py)
    - Access to MLflow pickle files

Author: Foundation PLR Team
Date: 2026-01-29
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_curve  # noqa: E402

# Configuration - use centralized path utilities
from src.utils.paths import get_classification_experiment_id, get_mlruns_dir  # noqa: E402

MLRUNS_DIR = get_mlruns_dir()
EXPERIMENT_ID = get_classification_experiment_id()
DB_PATHS = [
    Path("data/public/foundation_plr_results.db"),
    Path("data/foundation_plr_results.db"),
    Path("outputs/foundation_plr_results.db"),
]


def find_db_path() -> Path:
    """Find existing DuckDB file."""
    for p in DB_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"DuckDB not found in any of: {DB_PATHS}")


def compute_loess_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LOESS-smoothed calibration curve.

    Returns (x_smooth, y_smooth) arrays.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(y_true, y_prob, frac=frac, return_sorted=True)
        return smoothed[:, 0], smoothed[:, 1]
    except ImportError:
        # Fallback: binned calibration
        n_bins = 20
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means.append(y_true[mask].mean())
        return np.array(bin_centers), np.array(bin_means)


def compute_calibration_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 100,
    frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap CI for calibration curve.

    Returns (x_common, ci_lower, ci_upper).
    """
    from scipy import interpolate

    n = len(y_true)
    x_common = np.linspace(0.05, 0.95, 50)
    bootstrap_curves = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            x_smooth, y_smooth = compute_loess_calibration(
                y_true[idx], y_prob[idx], frac=frac
            )
            if len(x_smooth) > 1:
                f = interpolate.interp1d(
                    x_smooth, y_smooth, bounds_error=False, fill_value="extrapolate"
                )
                bootstrap_curves.append(f(x_common))
        except (ValueError, RuntimeError):
            continue

    if len(bootstrap_curves) < 10:
        return x_common, np.zeros_like(x_common), np.ones_like(x_common)

    bootstrap_array = np.array(bootstrap_curves)
    ci_lower = np.clip(np.percentile(bootstrap_array, 2.5, axis=0), 0, 1)
    ci_upper = np.clip(np.percentile(bootstrap_array, 97.5, axis=0), 0, 1)

    return x_common, ci_lower, ci_upper


def compute_dca_curves(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """
    Compute Decision Curve Analysis curves.

    Returns dict with thresholds, nb_model, nb_all, nb_none.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)

    n = len(y_true)
    prevalence = np.mean(y_true)

    nb_model = []
    nb_all = []
    nb_none = []

    for threshold in thresholds:
        # Net benefit for "treat all"
        nb_all.append(prevalence - (1 - prevalence) * threshold / (1 - threshold))

        # Net benefit for "treat none"
        nb_none.append(0.0)

        # Net benefit for model
        pred_pos = y_prob >= threshold
        tp = np.sum((pred_pos) & (y_true == 1)) / n
        fp = np.sum((pred_pos) & (y_true == 0)) / n
        nb = tp - fp * threshold / (1 - threshold)
        nb_model.append(nb)

    return {
        "thresholds": thresholds,
        "nb_model": np.array(nb_model),
        "nb_all": np.array(nb_all),
        "nb_none": np.array(nb_none),
    }


def extract_predictions_from_pickle(pickle_path: Path) -> dict[str, Any] | None:
    """
    Extract (y_true, y_prob) from metrics pickle.

    Returns dict with y_true, y_prob arrays or None if extraction fails.
    """
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        if "metrics_iter" not in data:
            return None

        test_iter = data.get("metrics_iter", {}).get("test", {})
        preds = test_iter.get("preds", {}).get("arrays", {}).get("predictions", {})

        if not preds:
            return None

        y_prob_all = preds.get("y_pred_proba")  # (n_subjects, n_bootstrap)
        labels = preds.get("label")  # (n_subjects, n_bootstrap)

        if y_prob_all is None or labels is None:
            return None

        y_prob = y_prob_all.mean(axis=1)  # Mean across bootstrap
        y_true = labels[:, 0]  # First column (all same)

        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            return None

        return {"y_true": y_true, "y_prob": y_prob}

    except Exception as e:
        print(f"    Warning: Could not extract predictions: {e}")
        return None


def get_runs_needing_curves(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Get runs from essential_metrics that need curve extraction."""
    # Get runs that have AUROC (valid runs)
    result = conn.execute("""
        SELECT run_id, outlier_method, imputation_method, classifier
        FROM essential_metrics
        WHERE auroc IS NOT NULL
    """).fetchall()

    return [
        {
            "run_id": row[0],
            "outlier_method": row[1],
            "imputation_method": row[2],
            "classifier": row[3],
        }
        for row in result
    ]


def find_metrics_pickle(run_id: str, classifier: str) -> Path | None:
    """Find metrics pickle for a run."""
    run_dir = MLRUNS_DIR / EXPERIMENT_ID / run_id
    if not run_dir.exists():
        return None

    metrics_dir = run_dir / "artifacts" / "metrics"
    if not metrics_dir.exists():
        return None

    # Try to find matching metrics pickle
    patterns = [
        f"*{classifier}*.pickle",
        "metrics_*.pickle",
    ]
    for pattern in patterns:
        pickles = list(metrics_dir.glob(pattern))
        if pickles:
            return pickles[0]

    return None


def create_curve_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create curve tables if they don't exist."""
    # predictions table - stores raw arrays as JSON
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            run_id VARCHAR PRIMARY KEY,
            y_true VARCHAR,  -- JSON array
            y_prob VARCHAR   -- JSON array
        )
    """)

    # roc_curves table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS roc_curves (
            run_id VARCHAR PRIMARY KEY,
            fpr VARCHAR,         -- JSON array
            tpr VARCHAR,         -- JSON array
            thresholds VARCHAR   -- JSON array
        )
    """)

    # calibration_curves table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_curves (
            run_id VARCHAR PRIMARY KEY,
            x_smooth VARCHAR,    -- JSON array
            y_smooth VARCHAR,    -- JSON array
            ci_lower VARCHAR,    -- JSON array
            ci_upper VARCHAR     -- JSON array
        )
    """)

    # dca_curves table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dca_curves (
            run_id VARCHAR PRIMARY KEY,
            thresholds VARCHAR,  -- JSON array
            nb_model VARCHAR,    -- JSON array
            nb_all VARCHAR,      -- JSON array
            nb_none VARCHAR      -- JSON array
        )
    """)


def extract_all_curves(db_path: Path) -> None:
    """
    Extract curve data for all runs in the database.

    CRITICAL: This is the ONLY place curve data is computed.
    Visualization code must READ from these tables.
    """
    conn = duckdb.connect(str(db_path))

    # Create tables
    create_curve_tables(conn)

    # Get runs needing curves
    runs = get_runs_needing_curves(conn)
    print(f"Processing {len(runs)} runs for curve extraction...")

    extracted = 0
    skipped = 0

    for i, run_info in enumerate(runs):
        run_id = run_info["run_id"]
        classifier = run_info["classifier"]

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(runs)}")

        # Check if already extracted
        existing = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        if existing > 0:
            skipped += 1
            continue

        # Find and extract from pickle
        pickle_path = find_metrics_pickle(run_id, classifier)
        if pickle_path is None:
            continue

        preds = extract_predictions_from_pickle(pickle_path)
        if preds is None:
            continue

        y_true = preds["y_true"]
        y_prob = preds["y_prob"]

        try:
            # Store predictions
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?)",
                [
                    run_id,
                    json.dumps(y_true.tolist()),
                    json.dumps(y_prob.tolist()),
                ],
            )

            # Compute and store ROC curve
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
            conn.execute(
                "INSERT INTO roc_curves VALUES (?, ?, ?, ?)",
                [
                    run_id,
                    json.dumps(fpr.tolist()),
                    json.dumps(tpr.tolist()),
                    json.dumps(thresholds_roc.tolist()),
                ],
            )

            # Compute and store calibration curve
            x_smooth, y_smooth = compute_loess_calibration(y_true, y_prob)
            x_ci, ci_lower, ci_upper = compute_calibration_ci(y_true, y_prob)
            conn.execute(
                "INSERT INTO calibration_curves VALUES (?, ?, ?, ?, ?)",
                [
                    run_id,
                    json.dumps(x_smooth.tolist()),
                    json.dumps(y_smooth.tolist()),
                    json.dumps(ci_lower.tolist()),
                    json.dumps(ci_upper.tolist()),
                ],
            )

            # Compute and store DCA curve
            dca = compute_dca_curves(y_true, y_prob)
            conn.execute(
                "INSERT INTO dca_curves VALUES (?, ?, ?, ?, ?)",
                [
                    run_id,
                    json.dumps(dca["thresholds"].tolist()),
                    json.dumps(dca["nb_model"].tolist()),
                    json.dumps(dca["nb_all"].tolist()),
                    json.dumps(dca["nb_none"].tolist()),
                ],
            )

            extracted += 1

        except Exception as e:
            print(f"    Error processing {run_id}: {e}")
            continue

    conn.close()
    print(f"\nExtracted curve data for {extracted} runs ({skipped} already existed)")


def verify_extraction(db_path: Path) -> None:
    """Verify curve tables have data."""
    conn = duckdb.connect(str(db_path), read_only=True)

    print("\n" + "=" * 60)
    print("CURVE DATA VERIFICATION")
    print("=" * 60)

    tables = ["predictions", "roc_curves", "calibration_curves", "dca_curves"]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    # Check essential_metrics count for comparison
    em_count = conn.execute(
        "SELECT COUNT(*) FROM essential_metrics WHERE auroc IS NOT NULL"
    ).fetchone()[0]
    print(f"\n  essential_metrics (valid): {em_count} rows")

    conn.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Extracting curve data to DuckDB")
    print("=" * 60)

    try:
        db_path = find_db_path()
        print(f"Using database: {db_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run extract_all_configs_to_duckdb.py first.")
        sys.exit(1)

    extract_all_curves(db_path)
    verify_extraction(db_path)

    print("\n" + "=" * 60)
    print("DONE - Curve tables populated")
    print("=" * 60)


if __name__ == "__main__":
    main()
