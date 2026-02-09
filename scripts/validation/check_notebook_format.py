#!/usr/bin/env python3
"""Pre-commit hook: Enforce Quarto-only notebook policy.

This script enforces the project's Quarto-only policy for notebooks:
1. No .ipynb files may be committed ANYWHERE in the repo (not just notebooks/)
2. All .qmd files must have required YAML header fields (via YAML parser)
3. No hardcoded hex colors, banned imports, or savefig calls in .qmd code cells

Import detection uses AST parsing (not regex) for reliable detection.

See: docs/planning/jupyter-notebook-submission-ready.md
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Required YAML header fields -- format/jupyter/execute inherited from _quarto.yml
REQUIRED_YAML_FIELDS = {"title"}

# Directories to skip when scanning for .ipynb
IPYNB_SKIP_DIRS = {
    "_output",
    "_freeze",
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    "renv",
    "site",
    ".tox",
    ".mypy_cache",
}

# Banned sklearn modules (checked via AST)
BANNED_SKLEARN_MODULES = {
    "sklearn.metrics",
    "sklearn.calibration",
}

# Banned regex patterns in code cells (after AST check handles imports)
BANNED_PATTERNS = [
    # Hex colors: require non-alpha before # to avoid false positives on comments
    (
        r"(?<![a-zA-Z])#[0-9a-fA-F]{6}\b",
        "Hardcoded hex color -- use COLORS dict or config",
    ),
    # Also catch rgb()/rgba() color literals
    (
        r"rgba?\(\s*\d+\s*,",
        "Hardcoded rgb/rgba color -- use COLORS dict or config",
    ),
    # Catch any .savefig() call (plt, fig, figure, ax.get_figure())
    (
        r"\.savefig\(",
        ".savefig() -- use save_figure() or display inline with plt.show()",
    ),
]


def check_no_ipynb() -> list[str]:
    """Check that no .ipynb files exist ANYWHERE in the repo."""
    errors = []
    for ipynb in PROJECT_ROOT.rglob("*.ipynb"):
        # Skip build/cache/venv directories
        if any(skip in ipynb.parts for skip in IPYNB_SKIP_DIRS):
            continue
        errors.append(
            f"POLICY VIOLATION: {ipynb.relative_to(PROJECT_ROOT)}\n"
            f"  .ipynb files are not allowed anywhere in the repo.\n"
            f"  Convert with: quarto convert {ipynb.name}"
        )
    return errors


def _parse_yaml_header(text: str) -> tuple[dict[str, str] | None, str | None]:
    """Parse YAML frontmatter using yaml.safe_load if available, else basic parsing."""
    if not text.startswith("---"):
        return None, "File must start with --- YAML frontmatter"

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, "Could not find closing --- for YAML frontmatter"

    yaml_text = parts[1]
    try:
        import yaml

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return None, "YAML frontmatter is not a mapping"
        return data, None
    except ImportError:
        # Fallback: basic key extraction (still handles nested keys)
        found_fields: dict[str, str] = {}
        for line in yaml_text.splitlines():
            stripped = line.strip()
            if (
                ":" in stripped
                and not stripped.startswith("#")
                and not line.startswith(" ")
            ):
                key = stripped.split(":")[0].strip()
                found_fields[key] = stripped.split(":", 1)[1].strip()
        return found_fields, None
    except Exception as exc:
        return None, f"YAML parse error: {exc}"


def check_qmd_headers() -> list[str]:
    """Check that .qmd files have required YAML header fields."""
    errors = []
    for qmd in NOTEBOOKS_DIR.rglob("*.qmd"):
        if qmd.name.startswith("_"):
            continue  # Skip _template.qmd and similar

        text = qmd.read_text(encoding="utf-8")
        data, err = _parse_yaml_header(text)
        rel = qmd.relative_to(PROJECT_ROOT)

        if err:
            errors.append(f"YAML ERROR: {rel}\n  {err}")
            continue

        if data is None:
            errors.append(
                f"MISSING YAML: {rel}\n  .qmd files must have YAML frontmatter"
            )
            continue

        missing = REQUIRED_YAML_FIELDS - set(data.keys())
        if missing:
            errors.append(
                f"MISSING FIELDS: {rel}\n"
                f"  Required YAML fields missing: {', '.join(sorted(missing))}"
            )

    return errors


def _extract_code_cells(text: str) -> list[tuple[int, str, list[tuple[int, str]]]]:
    """Extract Python code cells from a .qmd file.

    Returns list of (start_line, cell_label, [(lineno, code_line), ...]).
    Also extracts {r} and {bash} blocks for limited scanning.
    """
    cells = []
    in_code_block = False
    block_lang = ""
    cell_start = 0
    cell_label = ""
    code_lines: list[tuple[int, str]] = []

    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not in_code_block and (
            stripped.startswith("```{python")
            or stripped.startswith("```{r")
            or stripped.startswith("```{bash")
        ):
            in_code_block = True
            block_lang = (
                "python"
                if "python" in stripped
                else ("r" if "{r" in stripped else "bash")
            )
            cell_start = i
            cell_label = ""
            code_lines = []
            continue
        if in_code_block and stripped == "```":
            cells.append((cell_start, cell_label, code_lines, block_lang))
            in_code_block = False
            continue
        if in_code_block:
            if stripped.startswith("#| label:"):
                cell_label = stripped.split(":", 1)[1].strip()
            code_lines.append((i, line))

    return cells


def _check_imports_ast(code: str) -> list[str]:
    """Use AST to detect banned sklearn imports in Python code."""
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Skip unparseable code (e.g., eval:false examples)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for banned in BANNED_SKLEARN_MODULES:
                    if alias.name == banned or alias.name.startswith(banned + "."):
                        violations.append(
                            f"Banned import: `import {alias.name}` -- read from DuckDB instead"
                        )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for banned in BANNED_SKLEARN_MODULES:
                    if node.module == banned or node.module.startswith(banned + "."):
                        violations.append(
                            f"Banned import: `from {node.module} import ...` -- read from DuckDB instead"
                        )
    return violations


def check_qmd_banned_patterns() -> list[str]:
    """Check .qmd code cells for banned patterns and imports."""
    errors = []
    for qmd in NOTEBOOKS_DIR.rglob("*.qmd"):
        if qmd.name.startswith("_"):
            continue
        text = qmd.read_text(encoding="utf-8")
        rel = qmd.relative_to(PROJECT_ROOT)
        cells = _extract_code_cells(text)

        for cell_start, cell_label, code_lines, lang in cells:
            # Skip eval:false cells (example code, not executed)
            is_eval_false = any(
                "#| eval: false" in line or "#| eval: False" in line
                for _, line in code_lines
            )
            if is_eval_false:
                continue

            if lang == "python":
                # AST-based import check
                full_code = "\n".join(
                    line for _, line in code_lines if not line.strip().startswith("#|")
                )
                for violation in _check_imports_ast(full_code):
                    label_hint = f" ({cell_label})" if cell_label else ""
                    errors.append(
                        f"BANNED IMPORT: {rel}:{cell_start}{label_hint}\n  {violation}"
                    )

            # Regex pattern checks (all languages)
            for pattern, message in BANNED_PATTERNS:
                for lineno, line in code_lines:
                    stripped = line.strip()
                    # Skip directive lines and comments
                    if stripped.startswith("#|") or stripped.startswith("#"):
                        continue
                    if re.search(pattern, line):
                        errors.append(
                            f"BANNED PATTERN: {rel}:{lineno}\n"
                            f"  {message}\n"
                            f"  Line: {stripped[:80]}"
                        )

    return errors


def check_no_marimo() -> list[str]:
    """Check for marimo notebooks disguised as .py files in notebooks/."""
    errors = []
    for py_file in NOTEBOOKS_DIR.rglob("*.py"):
        if py_file.name.startswith("_") or "__pycache__" in py_file.parts:
            continue
        try:
            text = py_file.read_text(encoding="utf-8")
            if "import marimo" in text or "marimo.App" in text:
                errors.append(
                    f"POLICY VIOLATION: {py_file.relative_to(PROJECT_ROOT)}\n"
                    f"  Marimo notebooks are not allowed. Use .qmd (Quarto) format."
                )
        except UnicodeDecodeError:
            pass
    return errors


def check_sensitive_patterns() -> list[str]:
    """Check for sensitive data patterns in notebook CODE CELLS only."""
    errors = []
    sensitive_patterns = [
        (r"\bPLR\d{4}\b", "Patient identifier (PLRxxxx) -- use Hxxx/Gxxx codes"),
        (r"/home/\w+/", "Absolute home path -- use relative paths"),
    ]
    for qmd in NOTEBOOKS_DIR.rglob("*.qmd"):
        if qmd.name.startswith("_"):
            continue
        text = qmd.read_text(encoding="utf-8")
        rel = qmd.relative_to(PROJECT_ROOT)
        cells = _extract_code_cells(text)

        for cell_start, cell_label, code_lines, lang in cells:
            # Skip eval:false cells
            is_eval_false = any(
                "#| eval: false" in line or "#| eval: False" in line
                for _, line in code_lines
            )
            if is_eval_false:
                continue

            for lineno, line in code_lines:
                stripped = line.strip()
                if stripped.startswith("#|") or stripped.startswith("#"):
                    continue
                for pattern, message in sensitive_patterns:
                    if re.search(pattern, line):
                        errors.append(
                            f"SENSITIVE DATA: {rel}:{lineno}\n"
                            f"  {message}\n"
                            f"  Line: {stripped[:80]}"
                        )
    return errors


def main() -> int:
    all_errors: list[str] = []
    all_errors.extend(check_no_ipynb())
    all_errors.extend(check_no_marimo())
    all_errors.extend(check_qmd_headers())
    all_errors.extend(check_qmd_banned_patterns())
    all_errors.extend(check_sensitive_patterns())

    if all_errors:
        print("Notebook format check FAILED:\n")
        for err in all_errors:
            print(f"  {err}\n")
        return 1

    print("Notebook format check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
