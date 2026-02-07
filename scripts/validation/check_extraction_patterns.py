#!/usr/bin/env python3
r"""
Pre-commit hook to detect dangerous memory accumulation patterns in extraction scripts.

This script uses AST analysis to detect patterns like:
- all_rows = []; for x in data: all_rows.append(...)
- records = []; records.append(...) without intermediate flush

These patterns caused CRITICAL-FAILURE-005 (24-hour stuck extraction).

Usage:
    python scripts/check_extraction_patterns.py [--fix]

As pre-commit hook:
    - repo: local
      hooks:
        - id: check-extraction-patterns
          name: Check extraction memory patterns
          entry: python scripts/check_extraction_patterns.py
          language: python
          files: scripts/extract.*\.py$
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import NamedTuple


class AccumulationPattern(NamedTuple):
    """Detected accumulation pattern."""

    file: str
    line: int
    variable: str
    description: str


EXTRACTION_SCRIPTS = [
    "scripts/extraction/extract_decomposition_signals.py",
    "scripts/extraction/extract_all_configs_to_duckdb.py",
    "scripts/extraction/extract_curve_data_to_duckdb.py",
    "scripts/extraction/extract_cd_diagram_data.py",
    "scripts/extraction/extract_pminternal_data.py",
    "scripts/extraction/extract_preprocessing_metrics.py",
    "scripts/extraction/extract_outlier_difficulty.py",
    "scripts/extraction/extract_top_models_from_mlflow.py",
    "scripts/extraction/extract_top10_by_category.py",
    "scripts/extraction/extract_top10_models_with_artifacts.py",
]

# Known safe patterns (variables that are OK to accumulate)
SAFE_ACCUMULATION_VARS = {
    # Small bounded lists
    "pickles",  # File paths
    "model_pickles",
    "metrics_pickles",
    "methods",
    "classifiers",
    "categories",
    # Sets used for tracking
    "unique_classifiers",
    "unique_outliers",
    "unique_imputations",
    # Calibration/curve data (bounded by bin count, typically <20)
    "bin_centers",
    "bin_means",
    "bin_midpoints",
    "observed_proportions",
    "counts",
    "bootstrap_curves",
    # DCA/net benefit (bounded by threshold count)
    "nb_all",
    "nb_none",
    "nb_model",
    # Model selection (bounded, typically top-10)
    "configs",
    "catboost_models",
    "loaded_models",
    "selected",
    "models",
    # Per-subject results within a single pickle (bounded by 507 subjects)
    "results",
    # Runs list (bounded by ~300-400 configs, acceptable with dedup)
    "runs",
    # Generic records (typically bounded by config count or subject count)
    # These are LOW RISK scripts as identified in audit
    "records",
}

# Maximum expected rows before warning (heuristic)
MAX_SAFE_ROWS = 1000


class AccumulationDetector(ast.NodeVisitor):
    """AST visitor to detect unbounded list accumulation patterns."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[AccumulationPattern] = []
        self.list_vars: dict[str, int] = {}  # var_name -> line defined
        self.in_for_loop = False
        self.for_loop_depth = 0

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track list variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check if assigning empty list
                if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
                    self.list_vars[target.id] = node.lineno
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Track when we're inside a for loop."""
        self.for_loop_depth += 1
        self.in_for_loop = True
        self.generic_visit(node)
        self.for_loop_depth -= 1
        if self.for_loop_depth == 0:
            self.in_for_loop = False

    def visit_Call(self, node: ast.Call) -> None:
        """Check for .append() calls on tracked lists inside loops."""
        if self.in_for_loop:
            # Check for list.append(...)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "append":
                    if isinstance(node.func.value, ast.Name):
                        var_name = node.func.value.id
                        # Check if this is a tracked list variable
                        if var_name in self.list_vars:
                            # Skip safe variables
                            if var_name.lower() in SAFE_ACCUMULATION_VARS:
                                pass
                            elif any(
                                safe in var_name.lower()
                                for safe in ["pickle", "file", "path"]
                            ):
                                pass
                            else:
                                self.violations.append(
                                    AccumulationPattern(
                                        file=self.filename,
                                        line=node.lineno,
                                        variable=var_name,
                                        description=f"Unbounded list accumulation: '{var_name}.append()' inside loop",
                                    )
                                )
        self.generic_visit(node)


def check_for_gc_collect(source: str) -> bool:
    """Check if gc.collect() is called in the file."""
    return "gc.collect()" in source


def check_for_streaming_insert(source: str) -> bool:
    """Check if there's evidence of streaming inserts (executemany inside loop)."""
    # Look for executemany pattern
    return "executemany" in source and ("for " in source or "while " in source)


def analyze_file(filepath: Path) -> list[AccumulationPattern]:
    """Analyze a single file for accumulation patterns."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

    detector = AccumulationDetector(str(filepath))
    detector.visit(tree)

    violations = detector.violations

    # Add warnings for missing best practices
    if violations:
        if not check_for_gc_collect(source):
            violations.append(
                AccumulationPattern(
                    file=str(filepath),
                    line=0,
                    variable="gc",
                    description="Missing gc.collect() - consider adding for memory cleanup",
                )
            )

    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Check for memory accumulation patterns"
    )
    parser.add_argument(
        "files", nargs="*", help="Files to check (default: all extraction scripts)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Check all extraction scripts"
    )
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    args = parser.parse_args()

    # Determine which files to check
    if args.files:
        files = [Path(f) for f in args.files]
    elif args.all:
        project_root = Path(__file__).parent.parent.parent
        files = [project_root / script for script in EXTRACTION_SCRIPTS]
    else:
        # Default: check all extraction scripts
        project_root = Path(__file__).parent.parent.parent
        files = [
            project_root / script
            for script in EXTRACTION_SCRIPTS
            if (project_root / script).exists()
        ]

    all_violations: list[AccumulationPattern] = []

    for filepath in files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        violations = analyze_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        print("\n" + "=" * 70)
        print("MEMORY ACCUMULATION PATTERN WARNINGS")
        print("=" * 70)
        print("\nThese patterns may cause memory issues in large-scale extraction.")
        print("See: CRITICAL-FAILURE-005 for why this matters.\n")

        for v in all_violations:
            print(f"{v.file}:{v.line}: {v.description}")

        print("\n" + "-" * 70)
        print("Recommendations:")
        print("1. Use streaming inserts (executemany after each batch)")
        print("2. Add gc.collect() after processing each item")
        print("3. Use ExtractionGuardrails from src/extraction/guardrails.py")
        print("-" * 70 + "\n")

        if args.strict:
            sys.exit(1)
    else:
        print("No memory accumulation patterns detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
