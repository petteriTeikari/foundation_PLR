"""
Guardrail tests for Makefile compliance with repo rules.

These tests verify that the Makefile follows the project's package management
and execution rules:
- No `pip install` anywhere (Rule 5: uv only, pip BANNED)
- No bare `python` calls outside Docker contexts (must use `uv run python`)
- setup target uses `uv sync`

Part of: TRIPOD-Code repo housekeeping (T1)
"""

import re
from pathlib import Path

import pytest

MAKEFILE = Path(__file__).parent.parent / "Makefile"


@pytest.fixture
def makefile_content():
    """Load Makefile content."""
    return MAKEFILE.read_text()


@pytest.fixture
def makefile_lines(makefile_content):
    """Load Makefile as lines with numbers."""
    return list(enumerate(makefile_content.splitlines(), 1))


@pytest.mark.guardrail
class TestMakefileNoPipInstall:
    """Verify pip is never used in the Makefile (Rule 5: uv only)."""

    def test_no_pip_install(self, makefile_content):
        """Makefile must not contain `pip install` anywhere."""
        matches = re.findall(r"pip install.*", makefile_content)
        assert not matches, (
            f"BANNED: Found {len(matches)} 'pip install' usage(s) in Makefile. "
            f"Use 'uv sync' or 'uv add' instead. Matches: {matches}"
        )


@pytest.mark.guardrail
class TestMakefileUvRunPython:
    """Verify all Python calls use `uv run python` (not bare `python`)."""

    def _find_bare_python_calls(self, makefile_lines):
        """Find bare `python` calls that are NOT inside Docker commands."""
        violations = []
        for lineno, line in makefile_lines:
            stripped = line.strip()
            # Skip comments, echo lines, Docker commands (they run inside container)
            if stripped.startswith("#"):
                continue
            if "docker run" in line or "docker build" in line:
                continue
            # Check for bare `python ` at start of recipe line (tab-indented)
            # or `python ` in shell commands (not `uv run python`)
            if re.search(r"(?<!\w)python\s", stripped):
                # Allow: "uv run python", "$(DOCKER_IMAGE) python"
                if "uv run python" in line:
                    continue
                if "$(DOCKER_IMAGE)" in line or "$(R_DOCKER_IMAGE)" in line:
                    continue
                violations.append((lineno, line.rstrip()))
        return violations

    def test_no_bare_python_in_recipes(self, makefile_lines):
        """All Python invocations must use `uv run python`, not bare `python`."""
        violations = self._find_bare_python_calls(makefile_lines)
        if violations:
            msg = "BANNED: Found bare 'python' calls (must use 'uv run python'):\n"
            for lineno, line in violations:
                msg += f"  Line {lineno}: {line}\n"
            pytest.fail(msg)


@pytest.mark.guardrail
class TestMakefileSetupTarget:
    """Verify setup target uses uv sync."""

    def test_setup_uses_uv_sync(self, makefile_content):
        """The setup target must use 'uv sync', not pip install."""
        # Find the setup target section
        setup_match = re.search(
            r"^setup:.*?(?=^\S|\Z)", makefile_content, re.MULTILINE | re.DOTALL
        )
        assert setup_match, "setup target not found in Makefile"
        setup_block = setup_match.group()
        assert "uv sync" in setup_block, (
            f"setup target must use 'uv sync'. Found:\n{setup_block}"
        )
