#!/usr/bin/env python
"""
Export selective classification data for visualization.

Generates LIBRARY-AGNOSTIC JSON data showing how metrics change as uncertain
predictions are rejected. Can be consumed by matplotlib, ggplot2, D3.js, etc.

Samples are ranked by uncertainty (AURC-style: most confident retained first).

Data Source: outputs/foundation_plr_results_stratos.db
Output: outputs/r_data/selective_classification_data.json

Cross-references:
- Geifman & El-Yaniv (2017) "Selective Classification for DNNs"
- configs/VISUALIZATION/plot_hyperparam_combos.yaml (combo definitions)
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import data filters (SINGLE SOURCE OF TRUTH for featurization)
from src.data_io.data_filters import get_default_featurization  # noqa: E402


def compute_file_hash(filepath: Path) -> str:
    """Compute xxhash64-style hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_standard_combos() -> List[Dict]:
    """Load the 5 standard preprocessing categories from YAML.

    Only exports the 5 standard preprocessing categories:
    - ground_truth, best_ensemble, best_single_fm, deep_learning, traditional

    Returns:
        combos: List of combo configs with pipeline details
    """
    combos_path = (
        PROJECT_ROOT / "configs" / "VISUALIZATION" / "plot_hyperparam_combos.yaml"
    )
    if not combos_path.exists():
        raise FileNotFoundError(f"YAML config not found: {combos_path}")

    with open(combos_path) as f:
        config = yaml.safe_load(f)

    combos = []

    # Standard combos - use category_name and category_color_ref from YAML
    for combo in config.get("standard_combos", []):
        combo_id = combo["id"]
        combos.append(
            {
                "id": combo_id,
                "name": combo.get("category_name", combo["name"]),
                "short_name": combo.get("short_name", combo["id"]),
                "outlier_method": combo["outlier_method"],
                "imputation_method": combo["imputation_method"],
                "classifier": combo.get("classifier", "CatBoost"),
                "color_ref": combo.get("category_color_ref", combo.get("color_ref")),
            }
        )

    return combos


def get_predictions(
    conn: duckdb.DuckDBPyConnection,
    outlier_method: str,
    imputation_method: str,
    classifier: str = "CATBOOST",
    featurization: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions for a specific combo WITH DEDUPLICATION.

    CRITICAL: Uses featurization filter from data_filters.yaml to avoid
    mixing handcrafted (simple1.0) with embedding features.

    Note: Ground truth (pupil-gt + pupil-gt) has 3 MLflow runs = 3x predictions.
    We deduplicate by taking first occurrence per subject_id.
    """
    # Use default featurization from YAML config if not specified
    if featurization is None:
        featurization = get_default_featurization()

    query = """
        WITH dedup AS (
            SELECT subject_id, y_true, y_prob,
                   ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY mlflow_run_id) as rn
            FROM predictions
            WHERE outlier_method = ?
              AND imputation_method = ?
              AND classifier = ?
              AND featurization = ?
        )
        SELECT y_true, y_prob
        FROM dedup
        WHERE rn = 1
        ORDER BY subject_id
    """
    result = conn.execute(
        query, [outlier_method, imputation_method, classifier, featurization]
    ).fetchall()
    if not result:
        return None, None
    y_true = np.array([r[0] for r in result])
    y_prob = np.array([r[1] for r in result])
    return y_true, y_prob


def compute_uncertainty(y_prob: np.ndarray) -> np.ndarray:
    """Compute prediction uncertainty (lower = more confident).

    Uses distance from 0.5 as confidence proxy (higher distance = more confident).
    Uncertainty = 1 - confidence.
    """
    confidence = np.abs(y_prob - 0.5) * 2  # Scale to [0, 1]
    return 1 - confidence


def compute_net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.15
) -> float:
    """Compute Net Benefit at a given threshold.

    Net Benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))
    """
    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Net benefit formula
    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return float(nb)


def compute_scaled_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Scaled Brier Score (IPA).

    IPA = 1 - (Brier / Brier_null)
    Where Brier_null = prevalence * (1 - prevalence)

    Higher is better (1 = perfect, 0 = no skill, negative = worse than null).
    """
    brier = brier_score_loss(y_true, y_prob)
    prevalence = np.mean(y_true)
    brier_null = prevalence * (1 - prevalence)

    if brier_null == 0:
        return 0.0

    ipa = 1 - (brier / brier_null)
    return float(ipa)


def compute_metrics_at_retention_levels(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    retention_levels: List[float],
    nb_threshold: float = 0.15,
) -> Dict[str, List[float]]:
    """Compute metrics at each retention level.

    Samples are sorted by confidence (most confident first).
    At each retention level, we keep only the most confident samples.
    """
    uncertainty = compute_uncertainty(y_prob)
    sorted_indices = np.argsort(
        uncertainty
    )  # Ascending uncertainty = descending confidence

    n_samples = len(y_true)
    results = {
        "auroc": [],
        "net_benefit": [],
        "scaled_brier": [],
    }

    for retention in retention_levels:
        n_retain = int(n_samples * retention)
        if n_retain < 5:  # Need minimum samples for meaningful metrics
            results["auroc"].append(None)
            results["net_benefit"].append(None)
            results["scaled_brier"].append(None)
            continue

        # Select most confident samples
        retained_idx = sorted_indices[:n_retain]
        y_true_ret = y_true[retained_idx]
        y_prob_ret = y_prob[retained_idx]

        # Check if we have both classes
        if len(np.unique(y_true_ret)) < 2:
            results["auroc"].append(None)
            results["net_benefit"].append(None)
            results["scaled_brier"].append(None)
            continue

        # Compute metrics
        try:
            auroc = roc_auc_score(y_true_ret, y_prob_ret)
        except ValueError:
            auroc = None

        nb = compute_net_benefit(y_true_ret, y_prob_ret, nb_threshold)
        ipa = compute_scaled_brier(y_true_ret, y_prob_ret)

        results["auroc"].append(round(auroc, 4) if auroc else None)
        results["net_benefit"].append(round(nb, 4))
        results["scaled_brier"].append(round(ipa, 4))

    return results


def main():
    """Export selective classification data for 5 standard preprocessing categories."""
    print("=" * 60)
    print("Exporting Selective Classification Data for R")
    print("=" * 60)

    # Load combos
    print("\n[1/4] Loading 5 standard combos from YAML...")
    combos = load_standard_combos()
    print(f"  Loaded {len(combos)} combos: {[c['id'] for c in combos]}")

    # Connect to database
    db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
    print(f"\n[2/4] Connecting to database: {db_path.name}")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Define retention levels (from 10% to 100% in 5% increments)
    retention_levels = [
        round(x * 0.05, 2) for x in range(2, 21)
    ]  # 0.10, 0.15, ..., 1.00
    print(f"\n[3/4] Computing metrics at {len(retention_levels)} retention levels...")

    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "generator": "scripts/export_selective_classification_for_r.py",
            "data_source": {
                "file": db_path.name,
                "db_hash": compute_file_hash(db_path),
            },
            "net_benefit_threshold": 0.15,
        },
        "data": {
            "retention_levels": retention_levels,
            "rejection_ratios": [round(1 - r, 2) for r in retention_levels],
            "configs": [],
        },
    }

    for combo in combos:
        combo_id = combo["id"]
        print(f"  Processing: {combo_id}...", end=" ")

        y_true, y_prob = get_predictions(
            conn,
            combo["outlier_method"],
            combo["imputation_method"],
            "CATBOOST",
        )

        if y_true is None:
            print("SKIP (no data)")
            continue

        metrics = compute_metrics_at_retention_levels(
            y_true, y_prob, retention_levels, nb_threshold=0.15
        )

        # Full-data baseline metrics
        auroc_full = roc_auc_score(y_true, y_prob)
        nb_full = compute_net_benefit(y_true, y_prob, 0.15)
        ipa_full = compute_scaled_brier(y_true, y_prob)

        output["data"]["configs"].append(
            {
                "id": combo_id,
                "name": combo["name"],
                "short_name": combo["short_name"],
                "color_ref": combo["color_ref"],
                "n_predictions": len(y_true),
                "baseline_metrics": {
                    "auroc": round(auroc_full, 4),
                    "net_benefit": round(nb_full, 4),
                    "scaled_brier": round(ipa_full, 4),
                },
                "auroc_at_retention": metrics["auroc"],
                "net_benefit_at_retention": metrics["net_benefit"],
                "scaled_brier_at_retention": metrics["scaled_brier"],
            }
        )
        print(f"OK (n={len(y_true)}, AUROC@100%={auroc_full:.4f})")

    # Save
    print("\n[4/4] Saving JSON output...")
    output_path = (
        PROJECT_ROOT / "data" / "r_data" / "selective_classification_data.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Configs: {len(output['data']['configs'])}/{len(combos)}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
