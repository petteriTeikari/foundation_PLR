#!/usr/bin/env python3
"""
Computation Decoupling Check Script.

Pre-commit hook that verifies visualization code does NOT compute metrics.
All metrics must be pre-computed in extraction layer and stored in DuckDB.

Architecture:
  Block 1: Extraction (MLflow → DuckDB) - ALL computation happens here
  Block 2: Visualization (DuckDB → Figures) - READ ONLY

Addresses: GAP-01, CF-003 (CRITICAL-FAILURE-003-computation-decoupling-violation)

Usage:
    python scripts/check_computation_decoupling.py [files...]

    If files are provided, only checks those files.
    If no files provided, checks all src/viz/*.py files.

Exit codes:
    0: No violations found
    1: Violations found (computation in viz code)
"""

import ast
import sys
from pathlib import Path

# Files that ARE allowed to compute metrics (utilities, not viz)
ALLOWED_COMPUTATION_FILES = {
    "metric_registry.py",  # Defines metric functions (compute_fn for extraction use)
    "metrics_utils.py",  # Metric utilities
    "__init__.py",  # Package init
}

# Banned sklearn imports (any module)
BANNED_SKLEARN_IMPORTS = {
    "roc_auc_score",
    "brier_score_loss",
    "calibration_curve",
    "roc_curve",
    "precision_recall_curve",
    "f1_score",
    "accuracy_score",
    "confusion_matrix",
    "LogisticRegression",
}

# Banned sklearn modules (full module imports)
BANNED_SKLEARN_MODULES = {
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.calibration",
}

# Banned imports from our stats modules
BANNED_STATS_IMPORTS = {
    "calibration_slope_intercept",
    "scaled_brier_score",
    "net_benefit",
    "compute_dca",
    "compute_aurc",
    "decision_curve_analysis",
    "brier_decomposition",
}


class ImportChecker(ast.NodeVisitor):
    """AST visitor to find banned imports."""

    def __init__(self):
        self.violations = []

    def visit_ImportFrom(self, node):
        module = node.module or ""

        for alias in node.names:
            name = alias.name

            # Check any sklearn module for banned names
            if "sklearn" in module and name in BANNED_SKLEARN_IMPORTS:
                self.violations.append(
                    {
                        "type": "sklearn_import",
                        "name": name,
                        "lineno": node.lineno,
                        "module": module,
                    }
                )

            # Check our stats modules
            if "stats" in module and name in BANNED_STATS_IMPORTS:
                self.violations.append(
                    {
                        "type": "stats_import",
                        "name": name,
                        "lineno": node.lineno,
                        "module": module,
                    }
                )

        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if any(banned in alias.name for banned in BANNED_SKLEARN_MODULES):
                self.violations.append(
                    {
                        "type": "sklearn_module",
                        "name": alias.name,
                        "lineno": node.lineno,
                        "module": alias.name,
                    }
                )
        self.generic_visit(node)


def check_file(filepath: Path) -> list:
    """Check a single file for violations.

    Supports `# noqa: computation-decoupling` comments to ignore specific lines.
    ALL src/viz/*.py files are checked equally (no more plot-file-only strictness).
    """
    # Skip allowed files
    if filepath.name in ALLOWED_COMPUTATION_FILES:
        return []

    try:
        content = filepath.read_text()
        lines = content.splitlines()
        tree = ast.parse(content)

        checker = ImportChecker()
        checker.visit(tree)

        # Filter out violations on lines with noqa comment
        filtered_violations = []
        for v in checker.violations:
            line = lines[v["lineno"] - 1] if v["lineno"] <= len(lines) else ""
            if "noqa: computation-decoupling" not in line:
                filtered_violations.append(v)

        return [(filepath, v) for v in filtered_violations]

    except SyntaxError:
        return []


def main():
    """Main entry point."""
    # Get files to check
    if len(sys.argv) > 1:
        # Check specific files (from pre-commit)
        files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
        # Only check files in src/viz/
        files = [f for f in files if "src/viz" in str(f) or "src\\viz" in str(f)]
    else:
        # Check ALL src/viz/*.py files (not just plot patterns)
        viz_dir = Path("src/viz")
        if not viz_dir.exists():
            print("No src/viz directory found")
            return 0

        files = list(viz_dir.glob("*.py"))
        files = [f for f in files if f.name not in ALLOWED_COMPUTATION_FILES]

    if not files:
        return 0

    # Check all files
    all_violations = []
    for filepath in files:
        violations = check_file(filepath)
        all_violations.extend(violations)

    # Report
    if all_violations:
        print("COMPUTATION DECOUPLING VIOLATION")
        print("=" * 50)
        print("Visualization code must NOT compute metrics.")
        print("All metrics must be pre-computed in extraction layer.")
        print()

        for filepath, v in all_violations:
            print(f"  {filepath.name}:{v['lineno']}")
            print(f"    Imports {v['name']} from {v['module']}")
            print()

        print("FIX: Read pre-computed metrics from DuckDB instead.")
        print("See: CRITICAL-FAILURE-003-computation-decoupling-violation.md")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
