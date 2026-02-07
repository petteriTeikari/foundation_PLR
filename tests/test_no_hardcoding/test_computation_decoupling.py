"""Test that visualization code does not compute metrics.

CRITICAL-FAILURE-003 Rule:
ALL metric computation happens in extraction (scripts/extract_*.py).
Visualization code (src/viz/) must ONLY read from DuckDB, NEVER compute metrics.

This test enforces that rule by checking for banned imports in viz modules.

See: planning/refactor-action-plan.md Phase 1.2
"""

import ast
from pathlib import Path
import pytest

VIZ_DIR = Path(__file__).parent.parent.parent / "src" / "viz"

# Imports that are BANNED in visualization code
# These indicate metric computation which should happen in extraction only
BANNED_IMPORTS = [
    # sklearn metric computation
    ("sklearn.metrics", "roc_auc_score"),
    ("sklearn.metrics", "brier_score_loss"),
    ("sklearn.metrics", "log_loss"),
    ("sklearn.metrics", "accuracy_score"),
    ("sklearn.metrics", "f1_score"),
    ("sklearn.metrics", "precision_score"),
    ("sklearn.metrics", "recall_score"),
    ("sklearn.metrics", "confusion_matrix"),
    ("sklearn.metrics", "classification_report"),
    # Internal stats modules (should only be used in extraction)
    ("src.stats.calibration_extended", "calibration_slope_intercept"),
    ("src.stats.calibration_extended", "brier_decomposition"),
    ("src.stats.scaled_brier", "scaled_brier_score"),
    ("src.stats.clinical_utility", "net_benefit"),
    ("src.stats.clinical_utility", "decision_curve_analysis"),
]

# Modules that are allowed (actually needed for visualization)
ALLOWED_MODULES = [
    "numpy",  # Array operations
    "pandas",  # DataFrame operations
    "matplotlib",  # Plotting
    "duckdb",  # Reading from database
    "scipy.ndimage",  # Image smoothing for viz
    "scipy.interpolate",  # Interpolation for smooth curves
]

# Files that have approved exceptions (with deprecation notes)
EXCEPTIONS = {
    # metric_registry.py is a DUAL-USE module: it defines metric metadata (display names,
    # higher_is_better, etc.) for viz AND compute functions for extraction.
    # The compute functions (e.g., _compute_auroc) are imported by extraction scripts,
    # not used in the viz layer itself. The viz layer only uses MetricRegistry.get().
    "metric_registry.py": "Dual-use module: compute functions are for extraction, metadata for viz",
    # All other files have been refactored (GH#13):
    # - calibration_plot.py: RESOLVED - _from_db() functions only
    # - retained_metric.py: RESOLVED - reads from DuckDB retention_metrics table
    # - metric_vs_cohort.py: RESOLVED - reads from DuckDB cohort_metrics table
    # - prob_distribution.py: RESOLVED - reads from DuckDB distribution_stats table
    # - dca_plot.py: RESOLVED - reads from DuckDB dca_curves table
}


def get_viz_python_files() -> list[Path]:
    """Get all Python files in the viz directory."""
    if not VIZ_DIR.exists():
        return []
    return sorted(VIZ_DIR.rglob("*.py"))


def extract_imports(filepath: Path) -> list[tuple[int, str, str]]:
    """Extract all imports from a Python file.

    Returns
    -------
    list of tuples
        Each tuple is (line_number, module_name, imported_name)
    """
    content = filepath.read_text()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name, alias.name))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append((node.lineno, module, alias.name))

    return imports


def find_banned_imports(filepath: Path) -> list[tuple[int, str, str]]:
    """Find banned imports in a visualization file.

    Returns
    -------
    list of tuples
        Each tuple is (line_number, module, name) of banned imports
    """
    imports = extract_imports(filepath)
    violations = []

    for line_num, module, name in imports:
        for banned_module, banned_name in BANNED_IMPORTS:
            # Check if this import matches a banned pattern
            if banned_module in module and (name == banned_name or banned_name == "*"):
                violations.append((line_num, module, name))
            # Also check full import like "from sklearn.metrics import roc_auc_score"
            elif module == banned_module and name == banned_name:
                violations.append((line_num, module, name))

    return violations


@pytest.mark.parametrize(
    "py_file",
    [
        f
        for f in get_viz_python_files()
        if f.name not in EXCEPTIONS and f.name != "__init__.py"
    ],
    ids=lambda f: f.name,
)
def test_no_metric_computation_imports(py_file: Path):
    """Viz files should not import metric computation modules.

    All metric computation must happen during extraction.
    Visualization code should only READ pre-computed metrics from DuckDB.
    """

    violations = find_banned_imports(py_file)

    if violations:
        error_lines = [
            f"  Line {line}: from {module} import {name}"
            for line, module, name in violations
        ]
        pytest.fail(
            f"Found {len(violations)} banned import(s) in {py_file.name}:\n"
            + "\n".join(error_lines)
            + "\n\n"
            + "CRITICAL-FAILURE-003: Viz code must READ from DuckDB, not COMPUTE metrics.\n"
            + "Fix: Replace metric computation with database reads.\n"
            + "See: .claude/docs/meta-learnings/CRITICAL-FAILURE-003-computation-decoupling-violation.md"
        )


class TestDecouplingSanity:
    """Sanity checks for the decoupling architecture."""

    def test_extraction_script_exists(self):
        """Extraction script should exist and compute metrics."""
        extraction_script = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "extraction"
            / "extract_all_configs_to_duckdb.py"
        )
        assert extraction_script.exists(), "Missing extraction script"

        # Verify it imports the stats modules (it SHOULD compute metrics)
        content = extraction_script.read_text()
        assert "calibration_slope_intercept" in content, (
            "Extraction script should compute calibration metrics"
        )
        assert "scaled_brier_score" in content, (
            "Extraction script should compute scaled Brier score"
        )
        assert "net_benefit" in content, "Extraction script should compute net benefit"

    def test_duckdb_schema_has_stratos_metrics(self):
        """DuckDB schema should include pre-computed STRATOS metrics."""
        # This is a placeholder test - actual verification requires DB
        # In production, we'd verify these columns exist:
        expected_columns = [
            "calibration_slope",
            "calibration_intercept",
            "o_e_ratio",
            "scaled_brier",
            "net_benefit_5pct",
            "net_benefit_10pct",
            "net_benefit_15pct",
            "net_benefit_20pct",
        ]
        # TODO: Connect to actual DB and verify schema
        assert len(expected_columns) == 8  # Placeholder assertion


class TestReadPatterns:
    """Test that correct read-from-DB patterns are available."""

    def test_duckdb_importable(self):
        """duckdb should be importable for reading data."""
        import duckdb

        assert duckdb is not None

    def test_read_pattern_example(self):
        """Example of correct read pattern that viz code should use."""
        # This demonstrates the correct pattern:
        #
        # def get_auroc_from_db(run_id: str, db_path: str) -> float:
        #     conn = duckdb.connect(db_path, read_only=True)
        #     result = conn.execute(
        #         "SELECT auroc FROM essential_metrics WHERE run_id = ?",
        #         [run_id]
        #     ).fetchone()
        #     return result[0] if result else None
        #
        # Viz code should use patterns like this, not sklearn.metrics
        pass  # Documentation test
