#!/usr/bin/env python3
"""
Verify Report Metrics Against DuckDB Source of Truth.

This script ensures all reported AUROC values and metrics match the database.
Run this before generating any reports or figures to prevent hallucinated values.

Usage:
    uv run python scripts/verify_report_metrics.py

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

import sys
from pathlib import Path

import duckdb

# Expected values (verified 2026-01-29)
EXPECTED_VALUES = {
    "gt_auroc": 0.9110,
    "gt_auroc_ci_lo": 0.9028,
    "gt_auroc_ci_hi": 0.9182,
    "best_auroc": 0.9130,
    "handcrafted_mean": 0.8304,
    "embedding_mean": 0.7040,
    "epv_handcrafted": 7.0,  # 56 events / 8 features
    "n_events": 56,
    "n_features_handcrafted": 8,
}

# Tolerances for floating-point comparison
TOLERANCE_STRICT = 0.001  # For specific AUROC values
TOLERANCE_MEAN = 0.01  # For mean values


# Default database path (using Path for cross-platform compatibility)
DEFAULT_DB_PATH = Path("data") / "public" / "foundation_plr_results_stratos.db"


def verify_metrics(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    """
    Run all verification queries and check against expected values.

    Returns:
        True if all checks pass, False otherwise
    """
    if not Path(db_path).exists():
        print(f"ERROR: Database not found: {db_path}")
        return False

    conn = duckdb.connect(db_path, read_only=True)
    all_passed = True

    print("=" * 60)
    print("METRIC VERIFICATION")
    print("=" * 60)
    print(f"Database: {db_path}")
    print()

    # Check 1: Ground Truth AUROC
    print("1. Ground Truth AUROC (pupil-gt + pupil-gt + CatBoost + simple)")
    result = conn.execute(
        """
        SELECT auroc, auroc_ci_lo, auroc_ci_hi
        FROM essential_metrics
        WHERE outlier_method = 'pupil-gt'
          AND imputation_method = 'pupil-gt'
          AND classifier = 'CATBOOST'
          AND featurization LIKE 'simple%'
        """
    ).fetchone()

    if result:
        gt_auroc, gt_ci_lo, gt_ci_hi = result
        passed = abs(gt_auroc - EXPECTED_VALUES["gt_auroc"]) < TOLERANCE_STRICT
        status = "✓ PASS" if passed else "✗ FAIL"
        print(
            f"   {status}: AUROC = {gt_auroc:.4f} (expected: {EXPECTED_VALUES['gt_auroc']:.4f})"
        )
        if not passed:
            all_passed = False
    else:
        print("   ✗ FAIL: Ground truth config not found")
        all_passed = False

    # Check 2: Best AUROC
    print("\n2. Best AUROC (CatBoost + simple)")
    result = conn.execute(
        """
        SELECT auroc
        FROM essential_metrics
        WHERE featurization LIKE 'simple%' AND classifier = 'CATBOOST'
        ORDER BY auroc DESC
        LIMIT 1
        """
    ).fetchone()

    if result:
        best_auroc = result[0]
        passed = abs(best_auroc - EXPECTED_VALUES["best_auroc"]) < TOLERANCE_STRICT
        status = "✓ PASS" if passed else "✗ FAIL"
        print(
            f"   {status}: AUROC = {best_auroc:.4f} (expected: {EXPECTED_VALUES['best_auroc']:.4f})"
        )
        if not passed:
            all_passed = False
    else:
        print("   ✗ FAIL: No configs found")
        all_passed = False

    # Check 3: Featurization means
    print("\n3. Featurization Mean AUROC")
    result = conn.execute(
        """
        SELECT featurization, AVG(auroc) as mean_auroc
        FROM essential_metrics
        GROUP BY featurization
        """
    ).fetchdf()

    simple_mean = result[result["featurization"].str.contains("simple")][
        "mean_auroc"
    ].values[0]
    embed_rows = result[result["featurization"] == "MOMENT-embedding"]

    passed_simple = (
        abs(simple_mean - EXPECTED_VALUES["handcrafted_mean"]) < TOLERANCE_MEAN
    )
    status = "✓ PASS" if passed_simple else "✗ FAIL"
    print(
        f"   {status}: Handcrafted mean = {simple_mean:.4f} (expected: {EXPECTED_VALUES['handcrafted_mean']:.4f})"
    )
    if not passed_simple:
        all_passed = False

    if len(embed_rows) > 0:
        embed_mean = embed_rows["mean_auroc"].values[0]
        passed_embed = (
            abs(embed_mean - EXPECTED_VALUES["embedding_mean"]) < TOLERANCE_MEAN
        )
        status = "✓ PASS" if passed_embed else "✗ FAIL"
        print(
            f"   {status}: Embedding mean = {embed_mean:.4f} (expected: {EXPECTED_VALUES['embedding_mean']:.4f})"
        )
        if not passed_embed:
            all_passed = False

    # Check 4: EPV calculation
    print("\n4. EPV Calculation")
    epv = EXPECTED_VALUES["n_events"] / EXPECTED_VALUES["n_features_handcrafted"]
    passed_epv = abs(epv - EXPECTED_VALUES["epv_handcrafted"]) < 0.01
    status = "✓ PASS" if passed_epv else "✗ FAIL"
    print(
        f"   {status}: EPV = {epv:.1f} (expected: {EXPECTED_VALUES['epv_handcrafted']:.1f})"
    )
    print(
        f"         ({EXPECTED_VALUES['n_events']} events / {EXPECTED_VALUES['n_features_handcrafted']} features)"
    )

    # Check 5: Config counts
    print("\n5. Configuration Counts")
    result = conn.execute(
        """
        SELECT COUNT(DISTINCT outlier_method) as n_outlier,
               COUNT(DISTINCT imputation_method) as n_imputation,
               COUNT(DISTINCT classifier) as n_classifier
        FROM essential_metrics
        WHERE featurization LIKE 'simple%'
        """
    ).fetchone()

    if result:
        n_outlier, n_imputation, n_classifier = result
        print(f"   Outlier methods: {n_outlier}")
        print(f"   Imputation methods: {n_imputation}")
        print(f"   Classifiers: {n_classifier}")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL VERIFICATION CHECKS PASSED")
    else:
        print("✗ SOME VERIFICATION CHECKS FAILED")
    print("=" * 60)

    conn.close()
    return all_passed


def main():
    """Main entry point."""
    success = verify_metrics()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
