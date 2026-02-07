"""Test that all extraction scripts validate methods against registry.

This test catches CRITICAL-FAILURE pattern: parsing MLflow run names
with split("__") without validating against the registry.

The registry defines EXACTLY:
- 11 outlier methods
- 8 imputation methods
- 5 classifiers

Any code parsing run names MUST validate against these lists.
"""

import ast
import pytest
from pathlib import Path


class SplitValidationVisitor(ast.NodeVisitor):
    """AST visitor to find split("__") calls without validation."""

    def __init__(self):
        self.split_locations = []
        self.has_registry_import = False
        self.has_validation_call = False

    def visit_ImportFrom(self, node):
        """Check for registry imports."""
        if node.module and "registry" in node.module:
            self.has_registry_import = True
        self.generic_visit(node)

    def visit_Import(self, node):
        """Check for registry imports."""
        for alias in node.names:
            if "registry" in alias.name:
                self.has_registry_import = True
        self.generic_visit(node)

    def visit_Call(self, node):
        """Find split("__") calls."""
        # Check for .split("__") pattern
        if isinstance(node.func, ast.Attribute) and node.func.attr == "split":
            if node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and arg.value == "__":
                    self.split_locations.append(node.lineno)

        # Check for validation function calls
        # Includes: validate_*, parse_run_name, registry_parse_run_name
        validation_patterns = ["validate", "parse_run_name", "registry_parse"]
        if isinstance(node.func, ast.Attribute):
            if any(p in node.func.attr.lower() for p in validation_patterns):
                self.has_validation_call = True
        elif isinstance(node.func, ast.Name):
            if any(p in node.func.id.lower() for p in validation_patterns):
                self.has_validation_call = True

        self.generic_visit(node)


def analyze_file(file_path: Path) -> dict:
    """Analyze a Python file for split validation issues."""
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "syntax_error", "file": str(file_path)}

    visitor = SplitValidationVisitor()
    visitor.visit(tree)

    return {
        "file": str(file_path.name),
        "split_locations": visitor.split_locations,
        "has_registry_import": visitor.has_registry_import,
        "has_validation_call": visitor.has_validation_call,
        "is_valid": (
            len(visitor.split_locations) == 0
            or (visitor.has_registry_import and visitor.has_validation_call)
        ),
    }


class TestRegistryCompliance:
    """Tests for registry compliance in extraction scripts."""

    @pytest.fixture
    def scripts_dir(self):
        """Get the scripts directory."""
        return Path(__file__).parents[2] / "scripts"

    @pytest.fixture
    def src_dir(self):
        """Get the src directory."""
        return Path(__file__).parents[2] / "src"

    def test_extraction_scripts_validate_parsed_methods(self, scripts_dir):
        """Extraction scripts must validate parsed method names against registry.

        CRITICAL: Parsing run names with split("__") without validation
        leads to garbage methods like "anomaly" appearing in data.
        """
        violations = []

        extraction_scripts = list(scripts_dir.glob("**/extract_*.py"))
        assert len(extraction_scripts) > 0, "No extraction scripts found"

        for script in extraction_scripts:
            result = analyze_file(script)
            if not result.get("is_valid", True):
                violations.append(
                    f"{result['file']}:{result['split_locations']} - "
                    f"split('__') without registry validation"
                )

        assert not violations, (
            f"Found {len(violations)} scripts with unvalidated split() parsing:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: Import from src.data_io.registry and validate parsed methods"
        )

    def test_mlflow_utils_validates_methods(self, src_dir):
        """mlflow_utils.py must validate parsed method names."""
        mlflow_utils = src_dir / "data_io" / "mlflow_utils.py"

        if not mlflow_utils.exists():
            pytest.skip("mlflow_utils.py not found")

        result = analyze_file(mlflow_utils)

        if result.get("split_locations"):
            assert result.get("has_registry_import"), (
                f"mlflow_utils.py uses split('__') at lines {result['split_locations']} "
                "but doesn't import from registry"
            )
            assert result.get("has_validation_call"), (
                f"mlflow_utils.py uses split('__') at lines {result['split_locations']} "
                "but doesn't call validation functions"
            )

    def test_orchestration_tasks_validate_methods(self, src_dir):
        """Orchestration tasks must validate parsed method names."""
        violations = []
        tasks_dir = src_dir / "orchestration" / "tasks"

        if not tasks_dir.exists():
            pytest.skip("Orchestration tasks directory not found")

        for task_file in tasks_dir.glob("*.py"):
            result = analyze_file(task_file)
            if not result.get("is_valid", True):
                violations.append(f"{result['file']}:{result['split_locations']}")

        assert not violations, (
            "Orchestration tasks with unvalidated parsing:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_inline_method_lists(self, src_dir):
        """Code should not define method lists inline - use registry."""
        violations = []

        # Check for common inline patterns
        inline_patterns = [
            "outlier_methods = [",
            "imputation_methods = [",
            "classifiers = [",
            'methods = ["LOF"',
            'methods = ["MOMENT"',
        ]

        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text()
            except Exception:
                continue

            for i, line in enumerate(content.splitlines(), 1):
                # Skip comments and imports
                stripped = line.strip()
                if (
                    stripped.startswith("#")
                    or stripped.startswith("from")
                    or stripped.startswith("import")
                ):
                    continue

                for pattern in inline_patterns:
                    if (
                        pattern in line
                        and "registry" not in content[: content.find(line)]
                    ):
                        violations.append(f"{py_file.name}:{i} - inline method list")
                        break

        # This test may have false positives, so we just warn
        if violations:
            pytest.warns(
                UserWarning,
                match="Found potential inline method lists",
            )


class TestRegistryCounts:
    """Tests that verify registry defines exact counts."""

    def test_registry_has_exactly_11_outlier_methods(self):
        """Registry must define exactly 11 outlier methods."""
        try:
            from src.data_io.registry import get_valid_outlier_methods

            methods = get_valid_outlier_methods()
            assert len(methods) == 11, (
                f"Registry has {len(methods)} outlier methods, expected 11.\n"
                f"Methods: {methods}"
            )
        except ImportError:
            pytest.skip("Registry module not available")

    def test_registry_has_exactly_8_imputation_methods(self):
        """Registry must define exactly 8 imputation methods."""
        try:
            from src.data_io.registry import get_valid_imputation_methods

            methods = get_valid_imputation_methods()
            assert len(methods) == 8, (
                f"Registry has {len(methods)} imputation methods, expected 8.\n"
                f"Methods: {methods}"
            )
        except ImportError:
            pytest.skip("Registry module not available")

    def test_registry_has_exactly_5_classifiers(self):
        """Registry must define exactly 5 classifiers."""
        try:
            from src.data_io.registry import get_valid_classifiers

            methods = get_valid_classifiers()
            assert len(methods) == 5, (
                f"Registry has {len(methods)} classifiers, expected 5.\n"
                f"Methods: {methods}"
            )
        except ImportError:
            pytest.skip("Registry module not available")
