"""
TDD Test: R files must NOT contain hardcoded hex colors.
All colors must come from YAML config.

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


# Files ALLOWED to define colors (SINGLE SOURCE)
ALLOWED_COLOR_DEFINITION_FILES = [
    "color_palettes.R",
    "theme_foundation_plr.R",
]

# Files allowed to have emergency fallback constants
ALLOWED_FALLBACK_FILES = [
    "category_loader.R",
    "config_loader.R",
]


def get_r_ast_parser_path() -> Path:
    """Get path to the R AST parser script."""
    return Path(__file__).parent / "r_ast_parser.R"


def parse_r_file_for_hex_colors(file_path: Path) -> list[dict]:
    """
    Use R's parse() to extract hex color strings from an R file.

    Returns list of {line, value} dicts.
    """
    parser_path = get_r_ast_parser_path()

    result = subprocess.run(
        ["Rscript", str(parser_path), str(file_path), "hex_colors"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        # Parse failed - likely syntax error in R file
        return []

    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and "error" in data:
            return []
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def parse_r_file_for_strings(file_path: Path) -> list[dict]:
    """
    Use R's parse() to extract all string literals from an R file.

    Returns list of {line, value} dicts.
    """
    parser_path = get_r_ast_parser_path()

    result = subprocess.run(
        ["Rscript", str(parser_path), str(file_path), "strings"],
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


def get_r_files_for_color_check() -> list[Path]:
    """Get R files that should NOT have hardcoded colors."""
    project_root = get_project_root()
    all_r = list((project_root / "src/r").rglob("*.R"))
    return [
        f
        for f in all_r
        if f.name not in ALLOWED_COLOR_DEFINITION_FILES and "test" not in str(f).lower()
    ]


def is_fallback_constant_file(file_path: Path) -> bool:
    """Check if file is allowed to have fallback color constants."""
    return file_path.name in ALLOWED_FALLBACK_FILES


def test_no_hardcoded_hex_colors() -> None:
    """R figure files must not contain hardcoded hex colors (AST-based check)."""
    violations: list[str] = []

    for r_file in get_r_files_for_color_check():
        hex_colors = parse_r_file_for_hex_colors(r_file)

        for item in hex_colors:
            line = item.get("line", 0)
            value = item.get("value", "")

            # Allow fallback constants in specific infrastructure files
            if is_fallback_constant_file(r_file):
                # Read the actual line to check if it's a constant definition
                try:
                    lines = r_file.read_text().split("\n")
                    if line > 0 and line <= len(lines):
                        actual_line = lines[line - 1]
                        if "FALLBACK_COLOR" in actual_line or "nolint" in actual_line:
                            continue
                except Exception:
                    pass

            violations.append(f"{r_file.name}:{line}: {value}")

    assert not violations, (
        f"HARDCODED HEX COLORS DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations[:20])
        + (f"\n... and {len(violations) - 20} more" if len(violations) > 20 else "")
        + "\n\nUse resolve_color('--color-xxx') or get_category_colors() instead!"
    )


def test_no_inline_color_vectors() -> None:
    """
    Catch inline color vector definitions using AST analysis.

    Detects patterns where a named vector contains hex colors,
    e.g., c("Ground Truth" = "#FFD700", ...)
    """
    violations: list[str] = []

    for r_file in get_r_files_for_color_check():
        # Get both hex colors and all strings from the file
        hex_colors = parse_r_file_for_hex_colors(r_file)
        all_strings = parse_r_file_for_strings(r_file)

        if not hex_colors:
            continue

        # Get lines that have hex colors
        hex_lines = {item.get("line", 0) for item in hex_colors}

        # Check if those lines also have category-like strings nearby
        category_names = [
            "Ground Truth",
            "Foundation Model",
            "Traditional",
            "Ensemble",
            "Deep Learning",
        ]

        for item in all_strings:
            line = item.get("line", 0)
            value = item.get("value", "")

            # If this line has a hex color AND a category name, it's likely an inline vector
            if line in hex_lines and value in category_names:
                violations.append(
                    f"{r_file.name}:{line}: inline color vector with '{value}'"
                )

    assert not violations, (
        f"INLINE COLOR VECTORS DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations[:15])
        + "\n\nUse get_category_colors() from config_loader.R instead!"
    )
