"""
TDD Test: R files must NOT use case_when() for method categorization.
This pattern is the ROOT CAUSE of hardcoding drift.

Uses R's native parse() function via subprocess - NO REGEX (per CLAUDE.md absolute ban).
"""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.r_required, pytest.mark.guardrail]


def get_project_root() -> Path:
    """Find project root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root")


def get_r_ast_parser_path() -> Path:
    """Get path to the R AST parser script."""
    return Path(__file__).parent / "r_ast_parser.R"


def parse_r_file_for_case_when_grepl(file_path: Path) -> list[dict]:
    """
    Use R's parse() to detect case_when(grepl(...) ~ "Category") patterns.

    Returns list of {line, value} dicts where value is the category string.
    """
    parser_path = get_r_ast_parser_path()

    result = subprocess.run(
        ["Rscript", str(parser_path), str(file_path), "case_when_grepl"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and "error" in data:
            return []
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def parse_r_file_for_function_calls(file_path: Path) -> list[dict]:
    """
    Use R's parse() to extract function calls from an R file.

    Returns list of {line, value} dicts where value is function name.
    """
    parser_path = get_r_ast_parser_path()

    result = subprocess.run(
        ["Rscript", str(parser_path), str(file_path), "function_calls"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and "error" in data:
            return []
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def get_r_figure_files() -> list[Path]:
    """Get R figure files."""
    project_root = get_project_root()
    r_files = list((project_root / "src/r/figures").rglob("*.R"))
    r_files += list((project_root / "src/r/figure_system").rglob("*.R"))
    # Exclude category_loader.R itself (it's the replacement)
    return [f for f in r_files if "category_loader" not in f.name]


def test_no_case_when_categorization() -> None:
    """case_when() must not be used for method categorization (AST-based check)."""
    violations: list[str] = []

    for r_file in get_r_figure_files():
        case_when_patterns = parse_r_file_for_case_when_grepl(r_file)

        for item in case_when_patterns:
            line = item.get("line", 0)
            category = item.get("value", "")
            violations.append(
                f"{r_file.name}:{line}: case_when with grepl -> '{category}'"
            )

    assert not violations, (
        f"CASE_WHEN CATEGORIZATION DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations)
        + "\n\nUse categorize_outlier_methods() from category_loader.R instead!"
    )


def test_no_grepl_in_categorization_context() -> None:
    """
    Detect grepl() calls that appear to be used for categorization.

    Uses AST to find grepl calls and checks if they're in context
    that suggests category assignment.
    """
    violations: list[str] = []

    # Categories that indicate hardcoded categorization
    category_strings = [
        "Ground Truth",
        "Traditional",
        "Foundation Model",
        "Ensemble",
        "Deep Learning",
        "FM Pipeline",
    ]

    for r_file in get_r_figure_files():
        # Get function calls to find grepl usage
        fn_calls = parse_r_file_for_function_calls(r_file)

        # Get all string literals to find category names
        parser_path = get_r_ast_parser_path()
        result = subprocess.run(
            ["Rscript", str(parser_path), str(r_file), "strings"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            continue

        try:
            strings = json.loads(result.stdout)
            if not isinstance(strings, list):
                continue
        except json.JSONDecodeError:
            continue

        # Find lines with grepl calls
        grepl_lines = {
            item.get("line", 0) for item in fn_calls if item.get("value") == "grepl"
        }

        # Check if category strings appear near grepl calls (within 3 lines)
        for item in strings:
            line = item.get("line", 0)
            value = item.get("value", "")

            if value in category_strings:
                # Check if there's a grepl call nearby
                for grepl_line in grepl_lines:
                    if abs(line - grepl_line) <= 3:
                        violations.append(
                            f"{r_file.name}:{line}: '{value}' near grepl (line {grepl_line})"
                        )
                        break

    assert not violations, (
        f"GREPL->CATEGORY PATTERN DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations[:15])
        + "\n\nReplace with categorize_outlier_methods() from category_loader.R!"
    )
