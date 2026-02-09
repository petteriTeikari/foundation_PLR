#!/usr/bin/env python3
"""Pre-commit hook: Enforce Quarto-only notebook policy.

This script enforces the project's Quarto-only policy for notebooks:
1. No .ipynb files may be committed under notebooks/
2. All .qmd files must have required YAML header fields
3. No hardcoded paths, hex colors, or banned imports in .qmd code cells

See: docs/planning/jupyter-notebook-submission-ready.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Required YAML header fields for .qmd files
REQUIRED_YAML_FIELDS = {"title", "format"}

# Banned patterns in code cells
BANNED_PATTERNS = [
    (r"#[0-9a-fA-F]{6}\b", "Hardcoded hex color -- use COLORS dict or config"),
    (
        r"from sklearn\.metrics import",
        "sklearn.metrics import in notebook -- read from DuckDB instead",
    ),
    (r"plt\.savefig\(", "plt.savefig() -- use save_figure() or display inline"),
]


def check_no_ipynb() -> list[str]:
    """Check that no .ipynb files exist in notebooks/."""
    errors = []
    for ipynb in NOTEBOOKS_DIR.rglob("*.ipynb"):
        # Allow .ipynb in _output/ (Quarto intermediate)
        if "_output" in ipynb.parts or "_freeze" in ipynb.parts:
            continue
        errors.append(
            f"POLICY VIOLATION: {ipynb.relative_to(PROJECT_ROOT)}\n"
            f"  .ipynb files are not allowed. Use .qmd (Quarto) format.\n"
            f"  Convert with: quarto convert {ipynb.name}"
        )
    return errors


def check_qmd_headers() -> list[str]:
    """Check that .qmd files have required YAML header fields."""
    errors = []
    for qmd in NOTEBOOKS_DIR.rglob("*.qmd"):
        if qmd.name.startswith("_"):
            continue  # Skip _template.qmd and similar
        text = qmd.read_text(encoding="utf-8")

        # Check YAML frontmatter exists
        if not text.startswith("---"):
            errors.append(
                f"MISSING YAML: {qmd.relative_to(PROJECT_ROOT)}\n"
                f"  .qmd files must start with --- YAML frontmatter"
            )
            continue

        # Extract YAML block
        parts = text.split("---", 2)
        if len(parts) < 3:
            errors.append(
                f"MALFORMED YAML: {qmd.relative_to(PROJECT_ROOT)}\n"
                f"  Could not find closing --- for YAML frontmatter"
            )
            continue

        yaml_block = parts[1]
        found_fields = set()
        for line in yaml_block.splitlines():
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("#"):
                key = stripped.split(":")[0].strip()
                found_fields.add(key)

        missing = REQUIRED_YAML_FIELDS - found_fields
        if missing:
            errors.append(
                f"MISSING FIELDS: {qmd.relative_to(PROJECT_ROOT)}\n"
                f"  Required YAML fields missing: {', '.join(sorted(missing))}"
            )

    return errors


def check_qmd_banned_patterns() -> list[str]:
    """Check .qmd code cells for banned patterns."""
    errors = []
    for qmd in NOTEBOOKS_DIR.rglob("*.qmd"):
        if qmd.name.startswith("_"):
            continue
        text = qmd.read_text(encoding="utf-8")

        # Extract Python code blocks
        in_code_block = False
        code_lines: list[tuple[int, str]] = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```{python}") or line.strip().startswith(
                "```{python"
            ):
                in_code_block = True
                continue
            if in_code_block and line.strip() == "```":
                in_code_block = False
                continue
            if in_code_block:
                code_lines.append((i, line))

        for pattern, message in BANNED_PATTERNS:
            for lineno, line in code_lines:
                # Skip comments and #| directives
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(pattern, line):
                    errors.append(
                        f"BANNED PATTERN: {qmd.relative_to(PROJECT_ROOT)}:{lineno}\n"
                        f"  {message}\n"
                        f"  Line: {stripped[:80]}"
                    )

    return errors


def main() -> int:
    all_errors: list[str] = []
    all_errors.extend(check_no_ipynb())
    all_errors.extend(check_qmd_headers())
    all_errors.extend(check_qmd_banned_patterns())

    if all_errors:
        print("Notebook format check FAILED:\n")
        for err in all_errors:
            print(f"  {err}\n")
        return 1

    print("Notebook format check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
