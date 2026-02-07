#!/usr/bin/env python
"""
Export ROC and RC (Risk-Coverage) curves for visualization.

Generates LIBRARY-AGNOSTIC JSON data for the ROC + RC combined figure.
Can be consumed by matplotlib (Python), ggplot2 (R), D3.js, or any other viz library.

Uses YAML combos as SINGLE SOURCE OF TRUTH for which configs to include.

Data Source: outputs/foundation_plr_results_stratos.db
Output: outputs/r_data/roc_rc_data.json

Cross-references:
- configs/VISUALIZATION/plot_hyperparam_combos.yaml (combo definitions)
- .claude/planning/figure-qa-and-improvements.md (specifications)
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import data filters (SINGLE SOURCE OF TRUTH for featurization)
from src.data_io.data_filters import (  # noqa: E402
    get_default_classifier,
    get_default_featurization,
)


def compute_roc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 200,
    ci_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute ROC curve with bootstrap confidence intervals.

    Returns interpolated ROC curve on a common FPR grid with CI bands.
    """
    # Common FPR grid for interpolation
    fpr_grid = np.linspace(0, 1, 101)

    # Original ROC curve
    fpr_orig, tpr_orig, _ = roc_curve(y_true, y_prob)
    auroc_orig = roc_auc_score(y_true, y_prob)

    # Interpolate original to grid
    tpr_interp_orig = np.interp(fpr_grid, fpr_orig, tpr_orig)

    # Bootstrap resampling
    n_samples = len(y_true)
    rng = np.random.default_rng(42)
    tpr_boots = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        # Skip if only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_prob_boot)
            tpr_interp = np.interp(fpr_grid, fpr_boot, tpr_boot)
            tpr_boots.append(tpr_interp)
        except ValueError:
            continue

    if len(tpr_boots) < 10:
        # Not enough successful bootstraps, return without CI
        return {
            "fpr": fpr_grid.tolist(),
            "tpr": tpr_interp_orig.tolist(),
            "auroc": round(auroc_orig, 4),
            "has_ci": False,
        }

    tpr_boots = np.array(tpr_boots)

    # Compute percentile CIs
    tpr_lo = np.percentile(tpr_boots, 100 * ci_alpha / 2, axis=0)
    tpr_hi = np.percentile(tpr_boots, 100 * (1 - ci_alpha / 2), axis=0)

    return {
        "fpr": fpr_grid.tolist(),
        "tpr": tpr_interp_orig.tolist(),
        "tpr_ci_lo": tpr_lo.tolist(),
        "tpr_ci_hi": tpr_hi.tolist(),
        "auroc": round(auroc_orig, 4),
        "has_ci": True,
        "n_bootstrap": len(tpr_boots),
    }


def compute_file_hash(filepath: Path) -> str:
    """Compute xxhash64-style hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # First 16 chars for brevity


def load_yaml_combos() -> Tuple[List[Dict], Dict]:
    """Load combo definitions from YAML - SINGLE SOURCE OF TRUTH.

    Only exports the 5 standard preprocessing categories:
    - ground_truth, best_ensemble, best_single_fm, deep_learning, traditional

    Returns:
        combos: List of combo configs with pipeline details
        color_defs: Dict of color definitions
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
                "is_aggregate": False,
            }
        )

    return combos, config.get("color_definitions", {})


def get_predictions_for_combo(
    conn: duckdb.DuckDBPyConnection,
    outlier_method: str,
    imputation_method: str,
    classifier: str = None,
    featurization: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query predictions from DuckDB for a specific combo WITH DEDUPLICATION.

    CRITICAL: Uses featurization filter from data_filters.yaml to avoid
    mixing handcrafted (simple1.0) with embedding features.

    Note: Ground truth (pupil-gt + pupil-gt) has 3 MLflow runs = 3x predictions.
    We deduplicate by taking first occurrence per subject_id.
    """
    # Use defaults from YAML config if not specified
    if classifier is None:
        classifier = get_default_classifier()
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


def compute_rc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[List[float], List[float]]:
    """
    Compute Risk-Coverage curve.

    - Sort samples by confidence (descending)
    - At each coverage level, compute error rate (risk)

    Risk = 1 - Accuracy at each coverage level
    Coverage = fraction of samples retained (most confident first)
    """
    # Confidence = how close to 0 or 1
    confidence = np.abs(y_prob - 0.5)
    sorted_indices = np.argsort(-confidence)  # Descending by confidence

    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = (y_prob[sorted_indices] > 0.5).astype(int)

    n_samples = len(y_true)
    coverage = []
    risk = []

    for i in range(1, n_samples + 1):
        cov = i / n_samples
        errors = np.sum(y_true_sorted[:i] != y_pred_sorted[:i])
        r = errors / i  # Error rate at this coverage
        coverage.append(cov)
        risk.append(r)

    return coverage, risk


def compute_aurc(coverage: List[float], risk: List[float]) -> float:
    """Compute Area Under Risk-Coverage curve (lower is better)."""
    return float(np.trapz(risk, coverage))


def get_top10_configs(conn: duckdb.DuckDBPyConnection) -> List[Tuple[str, str]]:
    """Get top 10 CatBoost configs by AUROC from database.

    First tries the top10_catboost view, then computes AUROC properly as fallback.
    """
    # First check if there's a view
    try:
        result = conn.execute("""
            SELECT outlier_method, imputation_method
            FROM top10_catboost
            ORDER BY rank
            LIMIT 10
        """).fetchall()
        return [(r[0], r[1]) for r in result]
    except Exception:
        pass

    # Fallback: compute AUROC properly for each combo
    print("    (Computing Top-10 from predictions - no view found)")

    # Get defaults from YAML config
    classifier = get_default_classifier()
    featurization = get_default_featurization()

    # Get unique combos - MUST filter by featurization to avoid mixing
    combos = conn.execute(f"""
        SELECT DISTINCT outlier_method, imputation_method
        FROM predictions
        WHERE classifier = '{classifier}'
          AND featurization = '{featurization}'
    """).fetchall()

    auroc_scores = []
    for outlier, imputation in combos:
        y_true, y_prob = get_predictions_for_combo(
            conn, outlier, imputation
        )  # Uses defaults from YAML
        if y_true is not None and len(np.unique(y_true)) == 2:
            try:
                auroc = roc_auc_score(y_true, y_prob)
                auroc_scores.append((outlier, imputation, auroc))
            except ValueError:
                pass

    # Sort by AUROC descending and return top 10
    auroc_scores.sort(key=lambda x: x[2], reverse=True)
    return [(o, i) for o, i, _ in auroc_scores[:10]]


def compute_aggregate_curves(
    conn: duckdb.DuckDBPyConnection,
    configs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Compute aggregate ROC and RC curves across multiple configs.

    Aggregation method: Pool all predictions, then compute curves.
    """
    all_y_true = []
    all_y_prob = []

    for outlier, imputation in configs:
        y_true, y_prob = get_predictions_for_combo(conn, outlier, imputation)
        if y_true is not None:
            all_y_true.extend(y_true)
            all_y_prob.extend(y_prob)

    if not all_y_true:
        return None

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    # RC curve
    coverage, risk = compute_rc_curve(y_true, y_prob)
    aurc = compute_aurc(coverage, risk)

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auroc": round(auroc, 4),
        },
        "rc": {
            "coverage": coverage,
            "risk": risk,
            "aurc": round(aurc, 4),
        },
        "n_predictions": len(y_true),
    }


def main():
    """Export ROC and RC curve data for all YAML combos."""
    print("=" * 60)
    print("Exporting ROC + RC Curve Data for R")
    print("=" * 60)

    # Load YAML combos
    print("\n[1/4] Loading combo definitions from YAML...")
    combos, color_defs = load_yaml_combos()
    print(f"  Loaded {len(combos)} combos (including Top-10 Mean aggregate)")

    # Connect to database
    db_path = PROJECT_ROOT / "data" / "public" / "foundation_plr_results_stratos.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"\n[2/4] Connecting to database: {db_path.name}")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Process each combo
    print("\n[3/4] Computing ROC and RC curves...")
    output = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "generator": "scripts/export_roc_rc_for_r.py",
            "data_source": {
                "file": str(db_path.name),
                "db_hash": compute_file_hash(db_path),
            },
        },
        "data": {
            "n_configs": 0,
            "configs": [],
        },
    }

    success_count = 0
    for combo in combos:
        combo_id = combo["id"]
        print(f"  Processing: {combo_id}...", end=" ")

        if combo.get("is_aggregate"):
            # Handle aggregate (Top-10 Mean)
            top10_configs = get_top10_configs(conn)
            curves = compute_aggregate_curves(conn, top10_configs)
            if curves:
                output["data"]["configs"].append(
                    {
                        "id": combo_id,
                        "name": combo["name"],
                        "short_name": combo["short_name"],
                        "color_ref": combo["color_ref"],
                        "is_aggregate": True,
                        "n_source_configs": len(top10_configs),
                        **curves,
                    }
                )
                success_count += 1
                print(f"OK (pooled {curves['n_predictions']} predictions)")
            else:
                print("SKIP (no data)")
        else:
            # Regular combo
            y_true, y_prob = get_predictions_for_combo(
                conn,
                combo["outlier_method"],
                combo["imputation_method"],
                "CATBOOST",
            )

            if y_true is None:
                print("SKIP (no predictions)")
                continue

            # ROC curve with bootstrap CI
            roc_data = compute_roc_with_ci(y_true, y_prob, n_bootstrap=200)
            auroc = roc_data["auroc"]

            # RC curve
            coverage, risk = compute_rc_curve(y_true, y_prob)
            aurc = compute_aurc(coverage, risk)

            output["data"]["configs"].append(
                {
                    "id": combo_id,
                    "name": combo["name"],
                    "short_name": combo["short_name"],
                    "color_ref": combo["color_ref"],
                    "outlier_method": combo["outlier_method"],
                    "imputation_method": combo["imputation_method"],
                    "is_aggregate": False,
                    "roc": roc_data,
                    "rc": {
                        "coverage": coverage,
                        "risk": risk,
                        "aurc": round(aurc, 4),
                    },
                    "n_predictions": len(y_true),
                }
            )
            success_count += 1
            ci_status = "with CI" if roc_data.get("has_ci") else "no CI"
            print(f"OK (AUROC={auroc:.4f}, AURC={aurc:.4f}, {ci_status})")

    output["data"]["n_configs"] = success_count

    # Save output
    print("\n[4/4] Saving JSON output...")
    output_path = PROJECT_ROOT / "data" / "r_data" / "roc_rc_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved: {output_path}")
    print(f"  Configs: {success_count}/{len(combos)}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
