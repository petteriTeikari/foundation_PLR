"""
Docs Link Guardrails
====================

TDD tests that enforce documentation linking conventions:

Convention:
- Links from docs/ to files OUTSIDE docs/: use absolute GitHub URLs
  Base: https://github.com/petteriTeikari/foundation_PLR/blob/main/
- Links WITHIN docs/: use relative paths (e.g., ../tutorials/stratos-metrics.md)

These tests prevent MkDocs build warnings from broken relative links
that escape the docs/ directory.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

# Match markdown links: [text](target) — but not external URLs or anchors
# Captures the target path from markdown links
MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]*)\]"  # [text]
    r"\((?!https?://|mailto:|#)"  # (not http/https/mailto/anchor
    r"([^)#\s]+)"  # capture the path (stop at ), #, or whitespace)
    r"[^)]*\)"  # optional fragment/query, then )
)


def _get_docs_md_files() -> list[Path]:
    """Return all markdown files under docs/."""
    return sorted(DOCS_DIR.rglob("*.md"))


def _resolve_link(source_file: Path, target: str) -> Path:
    """Resolve a relative link target against its source file's directory."""
    return (source_file.parent / target).resolve()


def _lines_outside_code_fences(content: str) -> list[tuple[int, str]]:
    """Yield (line_number, line) for lines NOT inside fenced code blocks."""
    result = []
    in_fence = False
    for line_num, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            result.append((line_num, line))
    return result


class TestNoRelativeLinksOutsideDocs:
    """Links from docs/ to files outside docs/ must use GitHub URLs."""

    def test_no_relative_links_outside_docs(self):
        """Scan all docs/**/*.md for relative links that escape docs/."""
        violations = []
        docs_resolved = DOCS_DIR.resolve()

        for md_file in _get_docs_md_files():
            content = md_file.read_text(errors="replace")
            for line_num, line in _lines_outside_code_fences(content):
                for match in MARKDOWN_LINK_RE.finditer(line):
                    target = match.group(2)
                    resolved = _resolve_link(md_file, target)
                    try:
                        resolved.relative_to(docs_resolved)
                    except ValueError:
                        # Link escapes docs/
                        violations.append(
                            {
                                "file": str(md_file.relative_to(PROJECT_ROOT)),
                                "line": line_num,
                                "target": target,
                            }
                        )

        if violations:
            msg = (
                "GUARDRAIL VIOLATION: Relative links that escape docs/ directory!\n"
                "MkDocs cannot resolve these. Use absolute GitHub URLs instead.\n"
                "Base: https://github.com/petteriTeikari/foundation_PLR/blob/main/\n\n"
            )
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    Link: {v['target']}\n\n"
            msg += f"Total: {len(violations)} violations"
            pytest.fail(msg)


class TestNoDoubleDocsPrefix:
    """Files inside docs/ should not link via ../../docs/ (double-docs prefix)."""

    DOUBLE_DOCS_RE = re.compile(r"\.\./.*docs/")

    def test_no_double_docs_prefix(self):
        """Catch links like ../../docs/tutorials/ from files already in docs/."""
        violations = []

        for md_file in _get_docs_md_files():
            content = md_file.read_text(errors="replace")
            for line_num, line in _lines_outside_code_fences(content):
                for match in MARKDOWN_LINK_RE.finditer(line):
                    target = match.group(2)
                    if self.DOUBLE_DOCS_RE.match(target):
                        violations.append(
                            {
                                "file": str(md_file.relative_to(PROJECT_ROOT)),
                                "line": line_num,
                                "target": target,
                            }
                        )

        if violations:
            msg = (
                "GUARDRAIL VIOLATION: Double-docs prefix in links!\n"
                "Files inside docs/ should use relative paths without "
                "going through ../../docs/.\n\n"
            )
            for v in violations:
                msg += f"  {v['file']}:{v['line']}\n    Link: {v['target']}\n\n"
            msg += f"Total: {len(violations)} violations"
            pytest.fail(msg)


class TestNoReadmeIndexConflict:
    """No directory should have both README.md and index.md."""

    def test_no_readme_index_conflict(self):
        """A directory with index.md must not also have README.md."""
        violations = []

        for index_file in DOCS_DIR.rglob("index.md"):
            readme_file = index_file.parent / "README.md"
            if readme_file.exists():
                violations.append(str(index_file.parent.relative_to(PROJECT_ROOT)))

        if violations:
            msg = (
                "GUARDRAIL VIOLATION: README.md + index.md conflict!\n"
                "MkDocs uses index.md; README.md in the same directory "
                "creates ambiguity.\n"
                "Delete the README.md (or merge its content into index.md).\n\n"
            )
            for d in violations:
                msg += f"  {d}/  (has both README.md and index.md)\n"
            pytest.fail(msg)


class TestWithinDocsLinksExist:
    """Relative within-docs links must point to existing files."""

    def test_within_docs_links_exist(self):
        """Check that all relative links within docs/ point to real files."""
        violations = []
        docs_resolved = DOCS_DIR.resolve()

        for md_file in _get_docs_md_files():
            content = md_file.read_text(errors="replace")
            for line_num, line in _lines_outside_code_fences(content):
                for match in MARKDOWN_LINK_RE.finditer(line):
                    target = match.group(2)
                    resolved = _resolve_link(md_file, target)

                    # Only check links that stay within docs/
                    try:
                        resolved.relative_to(docs_resolved)
                    except ValueError:
                        continue  # Handled by TestNoRelativeLinksOutsideDocs

                    if not resolved.exists():
                        violations.append(
                            {
                                "file": str(md_file.relative_to(PROJECT_ROOT)),
                                "line": line_num,
                                "target": target,
                                "resolved": str(resolved),
                            }
                        )

        if violations:
            msg = (
                "GUARDRAIL VIOLATION: Broken within-docs links!\n"
                "These relative links point to non-existent files.\n\n"
            )
            for v in violations:
                msg += (
                    f"  {v['file']}:{v['line']}\n"
                    f"    Link: {v['target']}\n"
                    f"    Resolved: {v['resolved']}\n\n"
                )
            msg += f"Total: {len(violations)} violations"
            pytest.fail(msg)
