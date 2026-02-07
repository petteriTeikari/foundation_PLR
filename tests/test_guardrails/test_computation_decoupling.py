"""
Computation Decoupling Enforcement Tests

Ensure that visualization code (src/viz/*.py) does NOT compute metrics.
Metrics must be pre-computed in extraction layer and stored in DuckDB.

Architecture:
  Block 1: Extraction (MLflow → DuckDB) - ALL computation happens here
  Block 2: Visualization (DuckDB → Figures) - READ ONLY

Addresses: GAP-01, CF-003 (CRITICAL-FAILURE-003-computation-decoupling-violation)

TDD: These tests define the enforcement rules.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Files that ARE allowed to import/use computation
ALLOWED_COMPUTATION_FILES = {
    "metric_registry.py",  # Defines metric functions (utility module)
    "metrics_utils.py",  # Metric utilities
    "__init__.py",  # Package init files
}

# Modules that should NOT be imported in plot code
BANNED_IMPORT_MODULES = {
    "sklearn.metrics": [
        "roc_auc_score",
        "brier_score_loss",
        "calibration_curve",
        "roc_curve",
        "precision_recall_curve",
        "f1_score",
        "accuracy_score",
        "confusion_matrix",
    ],
    "sklearn.calibration": [
        "calibration_curve",
        "CalibratedClassifierCV",
    ],
    "sklearn.linear_model": [
        "LogisticRegression",
    ],
}

# Specific function/class names that indicate computation
BANNED_NAMES = {
    # sklearn metrics
    "roc_auc_score",
    "brier_score_loss",
    "calibration_curve",
    "roc_curve",
    "precision_recall_curve",
    # sklearn models used for computation
    "LogisticRegression",
    # Our internal computation modules
    "calibration_slope_intercept",
    "scaled_brier_score",
    "net_benefit",
    "compute_dca",
    "compute_aurc",
    "decision_curve_analysis",
    "brier_decomposition",
}

# ALL src/viz/*.py files are checked (not just plot files)
# Only ALLOWED_COMPUTATION_FILES are exempted
PLOT_FILE_PATTERNS = [
    "*_plot.py",
    "*_figure.py",
    "*_viz.py",
]


class ASTImportChecker(ast.NodeVisitor):
    """AST visitor to find banned imports."""

    def __init__(self):
        self.violations = []
        self.imports = []

    def visit_Import(self, node):
        """Check regular imports like 'import sklearn.metrics'."""
        for alias in node.names:
            self.imports.append(
                {
                    "type": "import",
                    "module": alias.name,
                    "name": alias.asname or alias.name,
                    "lineno": node.lineno,
                }
            )

            # Check if it's a banned module
            for banned_module in BANNED_IMPORT_MODULES:
                if alias.name.startswith(banned_module):
                    self.violations.append(
                        {
                            "type": "banned_module_import",
                            "module": alias.name,
                            "lineno": node.lineno,
                        }
                    )

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Check from imports like 'from sklearn.metrics import roc_auc_score'."""
        module = node.module or ""

        for alias in node.names:
            self.imports.append(
                {
                    "type": "from_import",
                    "module": module,
                    "name": alias.name,
                    "lineno": node.lineno,
                }
            )

            # Check if importing from banned module
            for banned_module, banned_names in BANNED_IMPORT_MODULES.items():
                if module.startswith(banned_module):
                    if alias.name == "*" or alias.name in banned_names:
                        self.violations.append(
                            {
                                "type": "banned_from_import",
                                "module": module,
                                "name": alias.name,
                                "lineno": node.lineno,
                            }
                        )

            # Check for banned specific names from any module
            if alias.name in BANNED_NAMES:
                self.violations.append(
                    {
                        "type": "banned_name_import",
                        "name": alias.name,
                        "module": module,
                        "lineno": node.lineno,
                    }
                )

        self.generic_visit(node)

    def visit_Call(self, node):
        """Check for direct calls to banned functions."""
        # Get the function name being called
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name and func_name in BANNED_NAMES:
            self.violations.append(
                {
                    "type": "banned_function_call",
                    "name": func_name,
                    "lineno": node.lineno,
                }
            )

        self.generic_visit(node)


def check_file_for_computation(filepath: Path) -> list[dict]:
    """
    Check a Python file for computation violations.

    Supports `# noqa: computation-decoupling` to suppress individual lines.

    Returns list of violations found.
    """
    try:
        content = filepath.read_text()
        lines = content.splitlines()
        tree = ast.parse(content)

        checker = ASTImportChecker()
        checker.visit(tree)

        # Filter out violations on lines with noqa comment
        filtered = []
        for v in checker.violations:
            lineno = v.get("lineno", 0)
            line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if "noqa: computation-decoupling" not in line:
                filtered.append(v)

        return filtered

    except SyntaxError as e:
        return [{"type": "syntax_error", "message": str(e), "lineno": e.lineno}]


@pytest.fixture
def viz_plot_files():
    """Get visualization plot files that should be read-only."""
    viz_dir = PROJECT_ROOT / "src" / "viz"
    if not viz_dir.exists():
        return []

    plot_files = []

    # Get files matching plot patterns
    for pattern in PLOT_FILE_PATTERNS:
        plot_files.extend(viz_dir.glob(pattern))

    # Filter out allowed files
    return [f for f in plot_files if f.name not in ALLOWED_COMPUTATION_FILES]


@pytest.fixture
def all_viz_files():
    """Get ALL visualization files for comprehensive check."""
    viz_dir = PROJECT_ROOT / "src" / "viz"
    if not viz_dir.exists():
        return []

    return [f for f in viz_dir.glob("*.py") if f.name not in ALLOWED_COMPUTATION_FILES]


class TestNoSklearnMetricsInPlotCode:
    """Ensure sklearn.metrics is not used in plot code."""

    def test_no_sklearn_metrics_imports(self, viz_plot_files):
        """Plot files must not import sklearn.metrics."""
        all_violations = []

        for filepath in viz_plot_files:
            violations = check_file_for_computation(filepath)

            sklearn_violations = [
                v
                for v in violations
                if "sklearn" in str(v.get("module", "")).lower()
                or v.get("type") == "banned_module_import"
            ]

            if sklearn_violations:
                all_violations.append(
                    {
                        "file": filepath.name,
                        "violations": sklearn_violations,
                    }
                )

        # Format assertion message
        if all_violations:
            msg = "sklearn imports found in plot code:\n"
            for item in all_violations:
                msg += f"\n  {item['file']}:\n"
                for v in item["violations"]:
                    msg += f"    Line {v['lineno']}: {v['type']} - {v.get('module', v.get('name', ''))}\n"

            pytest.fail(msg)


class TestNoBannedComputationFunctions:
    """Ensure banned computation functions are not used in viz code."""

    def test_no_computation_function_imports(self, all_viz_files):
        """Viz files must not import computation functions — HARD ERROR."""
        all_violations = []

        for filepath in all_viz_files:
            violations = check_file_for_computation(filepath)

            banned_imports = [
                v
                for v in violations
                if v.get("type") in ["banned_name_import", "banned_from_import"]
                and v.get("name") in BANNED_NAMES
            ]

            if banned_imports:
                all_violations.append(
                    {
                        "file": filepath.name,
                        "violations": banned_imports,
                    }
                )

        if all_violations:
            msg = "Banned computation function imports found:\n"
            for item in all_violations:
                msg += f"\n  {item['file']}:\n"
                for v in item["violations"]:
                    msg += f"    Line {v['lineno']}: imports {v.get('name')}\n"

            pytest.fail(msg)


class TestNoDirectMetricComputation:
    """Ensure plot code doesn't call banned computation functions."""

    def test_no_banned_function_calls(self, viz_plot_files):
        """Plot files must not call metric computation functions — HARD ERROR."""
        all_violations = []

        for filepath in viz_plot_files:
            violations = check_file_for_computation(filepath)

            call_violations = [
                v for v in violations if v.get("type") == "banned_function_call"
            ]

            if call_violations:
                all_violations.append(
                    {
                        "file": filepath.name,
                        "violations": call_violations,
                    }
                )

        if all_violations:
            msg = "Direct computation function calls in plot code:\n"
            for item in all_violations:
                msg += f"\n  {item['file']}:\n"
                for v in item["violations"]:
                    msg += f"    Line {v['lineno']}: calls {v['name']}()\n"

            pytest.fail(msg)


class TestAllVizFilesClean:
    """Comprehensive check: ALL src/viz/*.py files must be computation-free."""

    def test_no_sklearn_in_any_viz_file(self, all_viz_files):
        """No src/viz/ file (except allowed) may import sklearn."""
        all_violations = []

        for filepath in all_viz_files:
            violations = check_file_for_computation(filepath)

            sklearn_violations = [
                v
                for v in violations
                if "sklearn" in str(v.get("module", "")).lower()
                or v.get("type") == "banned_module_import"
            ]

            if sklearn_violations:
                all_violations.append(
                    {
                        "file": filepath.name,
                        "violations": sklearn_violations,
                    }
                )

        if all_violations:
            msg = "sklearn imports found in src/viz/ files:\n"
            for item in all_violations:
                msg += f"\n  {item['file']}:\n"
                for v in item["violations"]:
                    msg += f"    Line {v['lineno']}: {v['type']} - {v.get('module', v.get('name', ''))}\n"
            msg += "\nAll metric computation must happen in extraction (Block 1), not visualization (Block 2)."

            pytest.fail(msg)

    def test_no_stats_computation_imports(self, all_viz_files):
        """No src/viz/ file may import from src.stats computation modules."""
        all_violations = []

        for filepath in all_viz_files:
            violations = check_file_for_computation(filepath)

            stats_violations = [
                v
                for v in violations
                if v.get("type") == "banned_name_import"
                and v.get("name") in BANNED_NAMES
            ]

            if stats_violations:
                all_violations.append(
                    {
                        "file": filepath.name,
                        "violations": stats_violations,
                    }
                )

        if all_violations:
            msg = "Stats computation imports found in src/viz/ files:\n"
            for item in all_violations:
                msg += f"\n  {item['file']}:\n"
                for v in item["violations"]:
                    msg += f"    Line {v['lineno']}: imports {v.get('name')} from {v.get('module', 'unknown')}\n"

            pytest.fail(msg)


class TestComputationInCorrectLayer:
    """Verify computation happens in correct files."""

    def test_stats_modules_exist(self):
        """src/stats/ modules should exist for computation."""
        stats_dir = PROJECT_ROOT / "src" / "stats"
        assert stats_dir.exists(), "src/stats/ directory should exist"

        expected_modules = [
            "calibration_extended.py",
            "clinical_utility.py",
            "scaled_brier.py",
        ]

        for module in expected_modules:
            module_path = stats_dir / module
            assert module_path.exists(), f"Expected computation module: {module_path}"

    def test_extraction_script_exists(self):
        """Extraction script should exist."""
        script = (
            PROJECT_ROOT / "scripts" / "extraction" / "extract_all_configs_to_duckdb.py"
        )
        assert script.exists(), "Extraction script should exist"


class TestDuckDBHasPrecomputedData:
    """Verify DuckDB has pre-computed metrics (if DB exists)."""

    @pytest.fixture
    def duckdb_connection(self):
        """Get DuckDB connection if database exists."""
        # Check multiple possible locations
        db_paths = [
            PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db",
            PROJECT_ROOT / "data" / "foundation_plr_results.db",
        ]
        db_path = None
        for p in db_paths:
            if p.exists():
                db_path = p
                break

        assert db_path is not None, "DuckDB not found. Run: make extract"

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        yield conn
        conn.close()

    def test_essential_metrics_has_stratos_columns(self, duckdb_connection):
        """essential_metrics table should have STRATOS columns."""
        expected_columns = [
            "auroc",
            "brier",
            "calibration_slope",
            "calibration_intercept",
            "o_e_ratio",
        ]

        # Get actual columns
        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'essential_metrics'"
        ).fetchall()
        actual_columns = {row[0] for row in result}

        for col in expected_columns:
            assert col in actual_columns, f"Missing STRATOS column: {col}"

    def test_net_benefit_columns_exist(self, duckdb_connection):
        """Net benefit columns should exist."""
        expected_columns = [
            "net_benefit_5pct",
            "net_benefit_10pct",
            "net_benefit_15pct",
            "net_benefit_20pct",
        ]

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'essential_metrics'"
        ).fetchall()
        actual_columns = {row[0] for row in result}

        for col in expected_columns:
            assert col in actual_columns, f"Missing net benefit column: {col}"


class TestDuckDBHasCurveTables:
    """Verify DuckDB has pre-computed curve data for visualization."""

    @pytest.fixture
    def duckdb_connection(self):
        """Get DuckDB connection if database exists."""
        db_paths = [
            PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db",
            PROJECT_ROOT / "data" / "foundation_plr_results.db",
        ]
        db_path = None
        for p in db_paths:
            if p.exists():
                db_path = p
                break

        assert db_path is not None, "DuckDB not found. Run: make extract"

        import duckdb

        conn = duckdb.connect(str(db_path), read_only=True)
        yield conn
        conn.close()

    def test_calibration_curves_table_exists(self, duckdb_connection):
        """calibration_curves table should exist."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "calibration_curves" in table_names, "calibration_curves table missing"

    def test_dca_curves_table_exists(self, duckdb_connection):
        """dca_curves table should exist with net benefit curves."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "dca_curves" in table_names, "dca_curves table missing"

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'dca_curves'"
        ).fetchall()
        columns = {row[0] for row in result}

        expected = {
            "config_id",
            "threshold",
            "net_benefit_model",
            "net_benefit_all",
            "net_benefit_none",
        }
        for col in expected:
            assert col in columns, f"Missing dca_curves column: {col}"

    def test_predictions_table_exists(self, duckdb_connection):
        """predictions table should store raw (y_true, y_prob) for reproducibility."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "predictions" in table_names, "predictions table missing"

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'predictions'"
        ).fetchall()
        columns = {row[0] for row in result}

        expected = {"config_id", "y_true", "y_prob"}
        for col in expected:
            assert col in columns, f"Missing predictions column: {col}"

    def test_retention_metrics_table_exists(self, duckdb_connection):
        """retention_metrics table should exist for selective classification curves."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "retention_metrics" in table_names, "retention_metrics table missing"

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'retention_metrics'"
        ).fetchall()
        columns = {row[0] for row in result}

        expected = {"config_id", "retention_rate", "metric_name", "metric_value"}
        for col in expected:
            assert col in columns, f"Missing retention_metrics column: {col}"

    def test_cohort_metrics_table_exists(self, duckdb_connection):
        """cohort_metrics table should exist for cohort-based analysis."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "cohort_metrics" in table_names, "cohort_metrics table missing"

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'cohort_metrics'"
        ).fetchall()
        columns = {row[0] for row in result}

        expected = {"config_id", "cohort_fraction", "metric_name", "metric_value"}
        for col in expected:
            assert col in columns, f"Missing cohort_metrics column: {col}"

    def test_distribution_stats_table_exists(self, duckdb_connection):
        """distribution_stats table should exist for probability distribution summaries."""
        tables = duckdb_connection.execute("SHOW TABLES").fetchall()
        table_names = {t[0] for t in tables}
        assert "distribution_stats" in table_names, "distribution_stats table missing"

        result = duckdb_connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'distribution_stats'"
        ).fetchall()
        columns = {row[0] for row in result}

        expected = {
            "config_id",
            "auroc",
            "median_cases",
            "median_controls",
            "n_cases",
            "n_controls",
        }
        for col in expected:
            assert col in columns, f"Missing distribution_stats column: {col}"
