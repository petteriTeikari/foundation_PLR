#!/usr/bin/env python
"""
AST-based validation for hardcoding anti-patterns in Python visualization code.

Detects:
- Hardcoded hex colors (like "#006BA2")
- Hardcoded output paths (like "figures/generated/...")
- Direct plt.savefig() instead of save_figure()

Usage:
    python scripts/validate_python_hardcoding.py

Exit codes:
    0 - All files pass
    1 - Violations found

For help fixing violations, see:
    - .claude/CLAUDE.md (Anti-Hardcoding section)
    - docs/planning/ISSUE-test-documentation-improvements.md
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Files that are ALLOWED to define colors/paths (config files)
ALLOWED_FILES = {
    "plot_config.py",
    "color_palettes.py",
    "__init__.py",
}

# Specific exceptions: (filepath_suffix, line_number, violation_type)
# Use sparingly - document WHY each exception is needed
EXCEPTIONS = [
    # config_loader.py can't import COLORS due to circular import (plot_config imports config_loader)
    ("config_loader.py", 155, "hex_color"),
    # Docstring examples are allowed to have example paths
    ("figure_data.py", 53, "hardcoded_path"),
    ("figure_data.py", 91, "hardcoded_path"),
    ("figure_data.py", 118, "hardcoded_path"),
    ("figure_data.py", 140, "hardcoded_path"),
    ("figure_data.py", 157, "hardcoded_path"),
    ("figure_data.py", 165, "hardcoded_path"),
]

# Directories to scan
SCAN_DIRS = [
    "src/viz",
    "src/stats",
]


# ==============================================================================
# Detailed Error Messages for Each Violation Type
# ==============================================================================

ERROR_MESSAGES = {
    "hex_color": """
    WHY: Hardcoded colors break when:
      - Color palette is updated project-wide
      - Accessibility requirements change
      - Journal requires different color scheme

    FIX: Use COLORS dict from plot_config.py:

      # BEFORE (wrong):
      color = "{value}"

      # AFTER (correct):
      from src.viz.plot_config import COLORS
      color = COLORS["primary"]  # or another semantic name

    Available colors are defined in:
      - configs/VISUALIZATION/figure_colors.yaml
      - src/viz/plot_config.py (COLORS dict)

    See: .claude/CLAUDE.md (Anti-Hardcoding section)
""",
    "hardcoded_path": """
    WHY: Hardcoded paths break when:
      - Figure categories are reorganized
      - Project structure changes
      - Running on different machines

    FIX: Use save_figure() from plot_config.py:

      # BEFORE (wrong):
      output_path = "figures/generated/my_figure.png"
      fig.savefig(output_path)

      # AFTER (correct):
      from src.viz.plot_config import save_figure
      save_figure(fig, "my_figure", data=data_dict)

    The figure system routes to the correct directory automatically
    based on configs/VISUALIZATION/figure_layouts.yaml

    See: .claude/CLAUDE.md (MANDATORY Pattern for Python)
""",
    "absolute_path": """
    WHY: Absolute paths with /home/ are machine-specific and will fail
    on other developers' machines or CI/CD systems.

    FIX: Use relative paths from project root or config-based paths:

      # BEFORE (wrong):
      path = "/home/user/project/figures/output.png"

      # AFTER (correct):
      from src.viz.plot_config import save_figure
      save_figure(fig, "output")

    Or use pathlib with project root detection.

    See: .claude/CLAUDE.md (Anti-Hardcoding section)
""",
    "savefig": """
    WHY: Direct savefig() bypasses the figure system, causing:
      - Figures saved to wrong directories
      - Missing JSON data export for reproducibility
      - Inconsistent DPI and format settings

    FIX: Use save_figure() from plot_config.py:

      # BEFORE (wrong):
      plt.savefig("my_figure.png", dpi=300)
      # or
      fig.savefig("my_figure.pdf")

      # AFTER (correct):
      from src.viz.plot_config import save_figure
      save_figure(fig, "my_figure", data=data_dict)

    The figure system handles:
      - PNG and PDF output
      - Correct directory routing
      - JSON data export for reproducibility

    See: .claude/CLAUDE.md (MANDATORY Pattern for Python)
""",
}


class HardcodingDetector(ast.NodeVisitor):
    """Detect hardcoded values that should come from config."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations: List[Tuple[int, str, str]] = []  # (line, message, error_type)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            value = node.value

            # Check for hex colors (6 or 8 character)
            if re.match(r"^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$", value):
                self.violations.append(
                    (node.lineno, f"Hardcoded color '{value}'", "hex_color")
                )

            # Check for hardcoded paths
            if "figures/generated" in value:
                self.violations.append(
                    (node.lineno, "Hardcoded output path", "hardcoded_path")
                )

            # Check for home directory paths
            if "/home/" in value and "figures" in value:
                self.violations.append(
                    (node.lineno, "Hardcoded absolute path", "absolute_path")
                )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for plt.savefig() or fig.savefig() calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "savefig":
                # Check if it's plt.savefig, fig.savefig, or similar
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in ("plt", "fig", "figure", "ax"):
                        self.violations.append(
                            (
                                node.lineno,
                                f"Direct {node.func.value.id}.savefig()",
                                "savefig",
                            )
                        )
                # Also catch self.fig.savefig() or ax.figure.savefig()
                elif isinstance(node.func.value, ast.Attribute):
                    if node.func.value.attr in ("fig", "figure"):
                        self.violations.append(
                            (node.lineno, "Direct .figure.savefig()", "savefig")
                        )

        self.generic_visit(node)


def is_exception(filepath: Path, line: int, error_type: str) -> bool:
    """Check if a violation is in the exceptions list."""
    for exc_file, exc_line, exc_type in EXCEPTIONS:
        if filepath.name == exc_file and line == exc_line and error_type == exc_type:
            return True
        # Also match partial path suffix (e.g., "viz/config_loader.py")
        if (
            str(filepath).endswith(exc_file)
            and line == exc_line
            and error_type == exc_type
        ):
            return True
    return False


def validate_python_file(filepath: Path) -> List[Tuple[str, str]]:
    """Validate a Python file for hardcoding violations.

    Returns list of (message, error_type) tuples.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError as e:
        return [(f"Syntax error: {e}", "syntax")]
    except UnicodeDecodeError as e:
        return [(f"Encoding error: {e}", "encoding")]

    detector = HardcodingDetector(filepath)
    detector.visit(tree)

    # Filter out exceptions
    return [
        (f"Line {line}: {msg}", error_type)
        for line, msg, error_type in detector.violations
        if not is_exception(filepath, line, error_type)
    ]


def main():
    """Run validation on all Python viz files."""
    project_root = Path(__file__).parent.parent.parent
    errors = []
    files_checked = 0
    error_types_found = set()

    for scan_dir in SCAN_DIRS:
        dir_path = project_root / scan_dir
        if not dir_path.exists():
            continue

        for py_file in dir_path.glob("**/*.py"):
            # Skip allowed files
            if py_file.name in ALLOWED_FILES:
                continue

            # Skip test files
            if py_file.name.startswith("test_") or "_test.py" in py_file.name:
                continue

            files_checked += 1
            violations = validate_python_file(py_file)

            if violations:
                rel_path = py_file.relative_to(project_root)
                errors.append((str(rel_path), violations))
                for _, error_type in violations:
                    error_types_found.add(error_type)

    print(f"Checked {files_checked} Python files")

    if errors:
        print("\n" + "=" * 70)
        print("VALIDATION FAILED - Hardcoding violations found:")
        print("=" * 70)

        for filepath, violations in errors:
            print(f"\n{filepath}:")
            for msg, _ in violations:
                print(f"  - {msg}")

        # Print detailed help for each error type found
        print("\n" + "-" * 70)
        print("HOW TO FIX THESE VIOLATIONS:")
        print("-" * 70)

        for error_type in sorted(error_types_found):
            if error_type in ERROR_MESSAGES:
                print(f"\n### {error_type.upper()} ###")
                print(ERROR_MESSAGES[error_type])

        print("-" * 70)
        print("HELP: See the following resources for more details:")
        print("  - .claude/CLAUDE.md (Anti-Hardcoding section)")
        print("  - docs/planning/ISSUE-test-documentation-improvements.md")
        print("=" * 70)

        sys.exit(1)

    print("All Python files pass validation!")
    sys.exit(0)


if __name__ == "__main__":
    main()
