#!/usr/bin/env python
"""
Extract Top-10 CatBoost Configs per Preprocessing Category.

This script implements the "Top-10 pooling" approach to avoid winner-takes-all
effects when comparing preprocessing methods (Azad et al. 2026, Varoquaux & Cheplygina 2022).

Categories:
- Ground Truth: pupil-gt outlier detection (baseline)
- Ensemble FM: ensemble-based outlier detection
- Single FM: MOMENT, UniTS, TimesNet (non-ensemble)
- Traditional+DL: LOF, OneClassSVM, PROPHET, SubPCA

Output: data/r_data/top10_by_category.json

Reference:
- Azad et al. (2026) "Beyond Leaderboards: Why Current ML Evaluation Fails"
- Varoquaux & Cheplygina (2022) "Machine learning for medical imaging"
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import registry for method categories (single source of truth)
from src.data_io.registry import get_outlier_categories  # noqa: E402


def _build_category_queries() -> Dict[str, str]:
    """
    Build SQL queries for each preprocessing category using the registry.

    This ensures method names come from the single source of truth
    (configs/mlflow_registry/parameters/classification.yaml).
    """
    categories = get_outlier_categories()

    # Build list literals for SQL IN clauses
    ground_truth_methods = categories.get("ground_truth", ["pupil-gt"])
    ensemble_methods = categories.get("ensemble", [])
    fm_methods = categories.get("foundation_model", [])
    dl_methods = categories.get("deep_learning", [])
    traditional_methods = categories.get("traditional", [])

    # Single FM = foundation_model + deep_learning (non-ensemble)
    single_fm_methods = fm_methods + dl_methods

    # Format for SQL IN clause
    def sql_list(methods: List[str]) -> str:
        return ", ".join(f"'{m}'" for m in methods)

    return {
        "ground_truth": f"""
            SELECT outlier_method, imputation_method, auroc
            FROM essential_metrics
            WHERE classifier = 'CATBOOST'
              AND outlier_method IN ({sql_list(ground_truth_methods)})
            ORDER BY auroc DESC
            LIMIT 10
        """,
        "ensemble_fm": f"""
            SELECT outlier_method, imputation_method, auroc
            FROM essential_metrics
            WHERE classifier = 'CATBOOST'
              AND outlier_method IN ({sql_list(ensemble_methods)})
            ORDER BY auroc DESC
            LIMIT 10
        """,
        "single_fm": f"""
            SELECT outlier_method, imputation_method, auroc
            FROM essential_metrics
            WHERE classifier = 'CATBOOST'
              AND outlier_method IN ({sql_list(single_fm_methods)})
            ORDER BY auroc DESC
            LIMIT 10
        """,
        "traditional_dl": f"""
            SELECT outlier_method, imputation_method, auroc
            FROM essential_metrics
            WHERE classifier = 'CATBOOST'
              AND outlier_method IN ({sql_list(traditional_methods)})
            ORDER BY auroc DESC
            LIMIT 10
        """,
    }


# Build queries at module load time (uses registry)
CATEGORY_QUERIES = _build_category_queries()

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================


def extract_top10_for_category(
    conn: duckdb.DuckDBPyConnection, category: str
) -> List[Dict[str, Any]]:
    """Extract top-10 configs for a preprocessing category.

    Args:
        conn: DuckDB connection
        category: One of the defined category keys

    Returns:
        List of dicts with outlier_method, imputation_method, auroc
    """
    if category not in CATEGORY_QUERIES:
        raise ValueError(f"Unknown category: {category}")

    query = CATEGORY_QUERIES[category]
    results = conn.execute(query).fetchall()

    configs = [
        {
            "outlier_method": r[0],
            "imputation_method": r[1],
            "auroc": float(r[2]) if r[2] else None,
        }
        for r in results
    ]

    # Defensive: ensure sorted by AUROC descending (in case SQL order is lost)
    configs.sort(key=lambda x: x["auroc"] or 0, reverse=True)

    return configs


def get_predictions_for_config(
    conn: duckdb.DuckDBPyConnection,
    outlier_method: str,
    imputation_method: str,
    classifier: str = "CATBOOST",
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions for a specific config.

    Args:
        conn: DuckDB connection
        outlier_method: Outlier detection method
        imputation_method: Imputation method
        classifier: Classifier (default CATBOOST)

    Returns:
        Tuple of (y_true, y_prob) arrays, deduplicated by subject_id
    """
    query = """
        WITH dedup AS (
            SELECT subject_id, y_true, y_prob,
                   ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY mlflow_run_id) as rn
            FROM predictions
            WHERE outlier_method = ?
              AND imputation_method = ?
              AND classifier = ?
        )
        SELECT y_true, y_prob
        FROM dedup
        WHERE rn = 1
        ORDER BY subject_id
    """

    results = conn.execute(
        query, [outlier_method, imputation_method, classifier]
    ).fetchall()

    if not results:
        return None, None

    y_true = np.array([r[0] for r in results])
    y_prob = np.array([r[1] for r in results])

    return y_true, y_prob


def pool_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Pool predictions from multiple configs.

    Args:
        predictions: Dict mapping config_id to {"y_true": array, "y_prob": array}

    Returns:
        Dict with pooled "y_true" and "y_prob" arrays
    """
    all_y_true = []
    all_y_prob = []

    for config_id, pred in predictions.items():
        if pred["y_true"] is not None:
            all_y_true.extend(pred["y_true"])
            all_y_prob.extend(pred["y_prob"])

    return {
        "y_true": np.array(all_y_true),
        "y_prob": np.array(all_y_prob),
    }


def compute_pooled_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute ROC curve from pooled predictions.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities

    Returns:
        Dict with fpr, tpr, auroc
    """
    if len(np.unique(y_true)) < 2:
        return {"fpr": [0, 1], "tpr": [0, 1], "auroc": None}

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auroc": round(auroc, 4),
    }


def compute_pooled_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration curve from pooled predictions.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        Dict with predicted, observed, counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    observed_proportions = []
    counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_midpoints.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            observed_proportions.append(y_true[mask].mean())
            counts.append(int(mask.sum()))

    return {
        "predicted": bin_midpoints,
        "observed": observed_proportions,
        "counts": counts,
    }


def create_output_structure(category_data: Dict[str, Dict]) -> Dict[str, Any]:
    """Create the output JSON structure.

    Args:
        category_data: Dict mapping category names to their data

    Returns:
        Complete output structure with metadata
    """
    return {
        "metadata": {
            "created": datetime.now().isoformat(),
            "generator": "scripts/extract_top10_by_category.py",
            "schema_version": "1.0",
            "description": "Top-10 CatBoost configs per preprocessing category with pooled curves",
            "references": [
                "Azad et al. (2026) - Avoiding winner-takes-all in ML evaluation",
                "Varoquaux & Cheplygina (2022) - Machine learning for medical imaging",
            ],
        },
        "categories": category_data,
    }


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file for provenance."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Extract top-10 configs per category and compute pooled curves."""
    print("=" * 60)
    print("Extracting Top-10 Configs Per Preprocessing Category")
    print("=" * 60)

    # Connect to database
    db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"\n[1/4] Connecting to database: {db_path.name}")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Process each category
    print("\n[2/4] Extracting top-10 configs per category...")
    category_data = {}

    for category in CATEGORY_QUERIES.keys():
        print(f"\n  Category: {category}")

        # Get top-10 configs
        configs = extract_top10_for_category(conn, category)
        print(f"    Found {len(configs)} configs")

        if not configs:
            print(f"    WARNING: No configs found for {category}")
            category_data[category] = {
                "configs": [],
                "n_configs": 0,
                "pooled_roc": None,
                "pooled_calibration": None,
            }
            continue

        # Get predictions for each config
        predictions = {}
        for i, cfg in enumerate(configs):
            y_true, y_prob = get_predictions_for_config(
                conn, cfg["outlier_method"], cfg["imputation_method"]
            )
            if y_true is not None:
                predictions[f"config_{i}"] = {"y_true": y_true, "y_prob": y_prob}
                cfg["n_predictions"] = len(y_true)
            else:
                cfg["n_predictions"] = 0

        print(f"    Loaded predictions for {len(predictions)} configs")

        # Pool predictions and compute curves
        if predictions:
            pooled = pool_predictions(predictions)
            print(f"    Pooled: {len(pooled['y_true'])} predictions")

            pooled_roc = compute_pooled_roc(pooled["y_true"], pooled["y_prob"])
            pooled_cal = compute_pooled_calibration(pooled["y_true"], pooled["y_prob"])

            print(f"    Pooled AUROC: {pooled_roc['auroc']:.4f}")
        else:
            pooled_roc = None
            pooled_cal = None

        category_data[category] = {
            "configs": configs,
            "n_configs": len(configs),
            "n_pooled_predictions": len(pooled["y_true"]) if predictions else 0,
            "pooled_roc": pooled_roc,
            "pooled_calibration": pooled_cal,
        }

    # Create output
    print("\n[3/4] Creating output structure...")
    output = create_output_structure(category_data)

    # Add data source provenance
    output["metadata"]["data_source"] = {
        "file": str(db_path.name),
        "hash": compute_file_hash(db_path),
    }

    # Save
    print("\n[4/4] Saving JSON output...")
    output_path = PROJECT_ROOT / "data" / "r_data" / "top10_by_category.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print("\nCategory Summary:")
    for cat, data in category_data.items():
        pooled_auroc = data["pooled_roc"]["auroc"] if data["pooled_roc"] else "N/A"
        print(f"  {cat}: {data['n_configs']} configs, pooled AUROC = {pooled_auroc}")

    conn.close()


if __name__ == "__main__":
    main()
