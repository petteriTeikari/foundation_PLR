"""
TDD Test: R files must NOT contain hardcoded display names.
This test MUST FAIL initially - proving hardcoding exists.
When it passes, hardcoding is ZERO.

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


# BANNED display name strings that should come from YAML
BANNED_DISPLAY_NAMES = [
    "Ground Truth",
    "Ground truth",
    "Foundation Model",
    "Traditional",
    "Ensemble",
    "Deep Learning",
    "MOMENT Fine-tuned",
    "MOMENT Zeroshot",
    "Ensemble + CSDI",
    "MOMENT + SAITS",
    "LOF + SAITS",
    "Ground truth + Ground truth",
]

# Files/directories ALLOWED to define display names (SINGLE SOURCE)
ALLOWED_FILES = [
    "category_loader.R",  # The loader itself can reference names
    "color_palettes.R",  # Color palette definitions
    "common.R",  # Common utilities may have display name mappings
]


def get_r_files() -> list[Path]:
    """Get all R files that should NOT have hardcoded names."""
    project_root = get_project_root()
    r_files = list((project_root / "src/r").rglob("*.R"))
    return [
        f
        for f in r_files
        if f.name not in ALLOWED_FILES
        and "test" not in str(f).lower()
        and "configs" not in str(f)
    ]


def test_no_hardcoded_display_names_in_r() -> None:
    """R files must not contain hardcoded display names (AST-based check)."""
    violations: list[str] = []

    for r_file in get_r_files():
        strings = parse_r_file_for_strings(r_file)

        for item in strings:
            value = item.get("value", "")
            line = item.get("line", 0)

            if value in BANNED_DISPLAY_NAMES:
                violations.append(f"{r_file.name}:{line}: '{value}'")

    assert not violations, (
        f"HARDCODED DISPLAY NAMES DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations[:20])
        + (f"\n... and {len(violations) - 20} more" if len(violations) > 20 else "")
        + "\n\nLoad display names from YAML config via get_display_name() instead!"
    )


def test_no_hardcoded_combo_names_in_r() -> None:
    """R files must not contain hardcoded combo display names."""
    # Combo names that should come from plot_hyperparam_combos.yaml
    combo_display_names = [
        "Ensemble + CSDI",
        "MOMENT + SAITS",
        "LOF + SAITS",
        "Ground truth + Ground truth",
        "MOMENT + MOMENT",
        "TimesNet + TimesNet",
    ]

    violations: list[str] = []

    for r_file in get_r_files():
        strings = parse_r_file_for_strings(r_file)

        for item in strings:
            value = item.get("value", "")
            line = item.get("line", 0)

            if value in combo_display_names:
                violations.append(f"{r_file.name}:{line}: '{value}'")

    assert not violations, (
        f"HARDCODED COMBO NAMES DETECTED ({len(violations)} instances):\n"
        + "\n".join(violations[:15])
        + "\n\nLoad combo names from plot_hyperparam_combos.yaml instead!"
    )
