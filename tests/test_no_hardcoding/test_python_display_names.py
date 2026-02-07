"""
TDD Test: Python files must NOT contain hardcoded display names.
All display names must come from YAML config.

Uses Python AST parsing - NO REGEX (per CLAUDE.md absolute ban).
"""

import ast
from pathlib import Path


def get_project_root() -> Path:
    """Find project root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


# Display names that should NOT be hardcoded in Python visualization code
BANNED_DISPLAY_NAMES = [
    "Ground Truth",
    "Ensemble",
    "Traditional",
    "Foundation Model",
    "Deep Learning",
    "MOMENT Fine-tuned",
    "MOMENT Zeroshot",
]

# Files allowed to define display names (Single Source of Truth)
ALLOWED_FILES = [
    "registry.py",
    "plot_config.py",
    "metric_registry.py",
    "__init__.py",
    "conftest.py",
    # Class definition file - type hint forward references contain "Ensemble"
    "catboost_ensemble.py",
]

# Directories to skip
SKIP_DIRS = ["__pycache__", ".pytest_cache", "test"]

# Patterns that indicate log/debug messages, not display names
LOG_MESSAGE_PATTERNS = [
    "found",
    "loading",
    "model",
    "output",
    "shape",
    "name ",  # "Ensemble name " is a log message
    "codes",
    "Confidence",
    "Generating",
    "Dashboard",  # Log message about dashboard generation
    "consists",
    ":",  # Log messages often use colons
    "{}",  # Format strings are log messages
]

# Files that are visualization code (STRICT - any violation is a fail)
VIZ_FILES = [
    "factorial_matrix.py",
    "individual_subject_traces.py",
    "plot_",  # Files starting with plot_
    "fig_",  # Files starting with fig_
    "calibration_plot",
    "cd_diagram",
]


class HardcodedStringFinder(ast.NodeVisitor):
    """AST visitor that finds hardcoded string literals.

    Skips:
    - Docstrings (first expression in module/function/class)
    - Multi-line strings (likely documentation)
    - Strings in comments (not captured by AST anyway)
    """

    def __init__(self, banned_strings: list[str]):
        self.banned_strings = banned_strings
        self.violations: list[tuple[int, str, str]] = []
        self._docstring_positions: set[int] = set()

    def _is_docstring(self, node: ast.AST) -> bool:
        """Check if this node position is a docstring."""
        return id(node) in self._docstring_positions

    def _mark_docstrings(self, body: list[ast.stmt]) -> None:
        """Mark the first Expr(Constant(str)) in a body as a docstring."""
        if body and isinstance(body[0], ast.Expr):
            expr_val = body[0].value
            if isinstance(expr_val, ast.Constant) and isinstance(expr_val.value, str):
                self._docstring_positions.add(id(expr_val))

    def visit_Module(self, node: ast.Module) -> None:
        """Mark module docstring."""
        self._mark_docstrings(node.body)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Mark function docstring."""
        self._mark_docstrings(node.body)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Mark async function docstring."""
        self._mark_docstrings(node.body)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Mark class docstring."""
        self._mark_docstrings(node.body)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Check string constants for banned values."""
        if isinstance(node.value, str):
            # Skip docstrings
            if self._is_docstring(node):
                self.generic_visit(node)
                return

            # Skip multi-line strings (likely documentation/comments)
            if "\n" in node.value:
                self.generic_visit(node)
                return

            # Skip very long strings (likely URLs, documentation)
            if len(node.value) > 100:
                self.generic_visit(node)
                return

            for banned in self.banned_strings:
                if banned in node.value:
                    # Additional check: skip if string is mostly the banned word
                    # (e.g., "Ensemble" alone is OK in logging, "Ensemble metric" is suspect)
                    value_stripped = node.value.strip()
                    if value_stripped == banned:
                        # Exact match - this is definitely a hardcoded display name
                        self.violations.append((node.lineno, node.value, "exact match"))
                    elif banned in value_stripped and len(value_stripped) < 50:
                        # Partial match in short string - likely UI text
                        self.violations.append(
                            (node.lineno, node.value, "string literal")
                        )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Check f-strings for banned values."""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                # Skip multi-line parts
                if "\n" in value.value:
                    continue
                for banned in self.banned_strings:
                    if banned in value.value:
                        self.violations.append((node.lineno, value.value, "f-string"))
        self.generic_visit(node)


class CategoryMappingFinder(ast.NodeVisitor):
    """AST visitor that finds inline category mapping dictionaries."""

    def __init__(self):
        self.violations: list[tuple[int, str]] = []
        self.method_keywords = ["moment", "lof", "pupil", "saits", "csdi", "timesnet"]
        self.category_values = [
            "Ground Truth",
            "Foundation Model",
            "Traditional",
            "Ensemble",
        ]

    def visit_Dict(self, node: ast.Dict) -> None:
        """Check dict literals for category mappings."""
        has_method_key = False
        has_category_value = False

        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if any(m in key.value.lower() for m in self.method_keywords):
                    has_method_key = True

        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                if val.value in self.category_values:
                    has_category_value = True

        if has_method_key and has_category_value:
            self.violations.append((node.lineno, "inline category mapping dict"))

        self.generic_visit(node)


def get_python_files() -> list[Path]:
    """Get Python files that should NOT have hardcoded names."""
    project_root = get_project_root()
    all_py: list[Path] = []

    for search_dir in ["src", "scripts"]:
        dir_path = project_root / search_dir
        if dir_path.exists():
            all_py.extend(dir_path.rglob("*.py"))

    return [
        f
        for f in all_py
        if f.name not in ALLOWED_FILES and not any(skip in str(f) for skip in SKIP_DIRS)
    ]


def _is_log_message(string: str) -> bool:
    """Check if string looks like a log/debug message rather than a display name."""
    string_lower = string.lower()
    return any(pattern.lower() in string_lower for pattern in LOG_MESSAGE_PATTERNS)


def _is_viz_file(filename: str) -> bool:
    """Check if file is visualization code where display names are strictly banned."""
    return any(viz_pattern in filename for viz_pattern in VIZ_FILES)


def test_no_hardcoded_display_names_python() -> None:
    """Python visualization files must not contain hardcoded display names.

    For non-viz files: log messages are allowed (filtered out)
    For viz files: ALL banned display names are violations
    """
    all_violations: list[str] = []

    for py_file in get_python_files():
        is_viz = _is_viz_file(py_file.name)

        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))

            finder = HardcodedStringFinder(BANNED_DISPLAY_NAMES)
            finder.visit(tree)

            for lineno, string, context in finder.violations:
                # For viz files: all violations count
                # For non-viz files: skip if it looks like a log message
                if is_viz or not _is_log_message(string):
                    all_violations.append(
                        f"{py_file.name}:{lineno}: '{string}' ({context})"
                    )

        except SyntaxError:
            continue

    assert not all_violations, (
        f"HARDCODED DISPLAY NAMES IN PYTHON ({len(all_violations)} instances):\n"
        + "\n".join(all_violations[:15])
        + ("\n... and more" if len(all_violations) > 15 else "")
        + "\n\nLoad display names from YAML config instead!"
    )


def test_no_hardcoded_category_mappings() -> None:
    """Python files must not define category mappings inline."""
    violations: list[str] = []

    for py_file in get_python_files():
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))

            finder = CategoryMappingFinder()
            finder.visit(tree)

            for lineno, desc in finder.violations:
                violations.append(f"{py_file.name}:{lineno}: {desc}")

        except SyntaxError:
            continue

    assert not violations, (
        f"INLINE CATEGORY MAPPINGS DETECTED ({len(violations)}):\n"
        + "\n".join(violations)
        + "\n\nUse category_mapping.yaml instead!"
    )
