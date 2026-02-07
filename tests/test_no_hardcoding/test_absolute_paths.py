"""Test that no absolute user paths exist in source code.

This test enforces the rule that all paths must be:
1. Relative to PROJECT_ROOT
2. Loaded from environment variables
3. Loaded from configuration files

NEVER hardcode paths like /home/petteri/... which break on other machines.

See: planning/refactor-action-plan.md Phase 1.1
"""

import re
from pathlib import Path
import pytest

# Directories to scan for violations
SOURCE_DIRS = [
    Path(__file__).parent.parent.parent / "src",
    Path(__file__).parent.parent.parent / "scripts",
]

# Patterns that indicate hardcoded user paths (BANNED)
BANNED_PATTERNS = [
    r"/home/\w+/",  # Linux home directories
    r"/Users/\w+/",  # macOS home directories
    r"C:\\Users\\\w+\\",  # Windows user directories
    r"/root/",  # Root home directory
]

# Files that are allowed exceptions (with justification)
ALLOWED_EXCEPTIONS = {
    # Test files may contain example paths in docstrings
    "test_": "Test files may have example paths in documentation",
    # This paths.py is the solution - it contains defaults that are replaced by env vars
    "paths.py": "paths.py defines the centralized path resolution",
    # Validation scripts contain example paths in error messages (not actual code)
    "validate_python_hardcoding.py": "Contains example paths in error message templates",
}


def get_all_python_files() -> list[Path]:
    """Get all Python files in source directories."""
    files = []
    for src_dir in SOURCE_DIRS:
        if src_dir.exists():
            files.extend(src_dir.rglob("*.py"))
    return sorted(files)


def is_exception_file(filepath: Path) -> bool:
    """Check if file is in the allowed exceptions list."""
    for pattern, _reason in ALLOWED_EXCEPTIONS.items():
        if pattern in filepath.name:
            return True
    return False


def find_banned_paths_in_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Find banned path patterns in a file.

    Returns
    -------
    list of tuples
        Each tuple is (line_number, matched_pattern, line_content)
    """
    violations = []
    content = filepath.read_text()

    for line_num, line in enumerate(content.split("\n"), 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Skip docstrings (simple heuristic)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Check each banned pattern
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, line):
                # Extract the matched path for the error message
                match = re.search(pattern + r"[^\s\"']*", line)
                matched = match.group(0) if match else pattern
                violations.append((line_num, matched, line.strip()))

    return violations


@pytest.mark.parametrize(
    "py_file",
    [f for f in get_all_python_files() if not is_exception_file(f)],
    ids=lambda f: f.name,
)
def test_no_absolute_user_paths(py_file: Path):
    """Each Python file should not contain hardcoded user paths.

    All paths should be resolved via:
    - src.utils.paths module (environment-aware)
    - Configuration files
    - Relative to PROJECT_ROOT
    """

    violations = find_banned_paths_in_file(py_file)

    if violations:
        error_lines = [
            f"  Line {line}: {path}\n    {content}"
            for line, path, content in violations
        ]
        pytest.fail(
            f"Found {len(violations)} hardcoded user path(s) in {py_file.name}:\n"
            + "\n".join(error_lines)
            + "\n\nFix: Use src.utils.paths module or environment variables instead."
        )


def test_paths_module_exists():
    """Verify the centralized paths module exists."""
    paths_module = Path(__file__).parent.parent.parent / "src" / "utils" / "paths.py"
    assert paths_module.exists(), (
        "Missing src/utils/paths.py - this module provides centralized path resolution. "
        "See planning/refactor-action-plan.md for implementation."
    )


def test_paths_module_has_required_functions():
    """Verify paths module exports all required functions."""
    from src.utils.paths import (
        get_mlruns_dir,
        get_seri_db_path,
        get_results_db_path,
        get_figures_output_dir,
        PROJECT_ROOT,
    )

    # Verify they're callable
    assert callable(get_mlruns_dir)
    assert callable(get_seri_db_path)
    assert callable(get_results_db_path)
    assert callable(get_figures_output_dir)

    # Verify PROJECT_ROOT is a Path
    assert isinstance(PROJECT_ROOT, Path)


def test_env_example_exists():
    """Verify .env.example template exists for developers."""
    env_example = Path(__file__).parent.parent.parent / ".env.example"
    if not env_example.exists():
        pytest.skip(
            "Missing .env.example - excluded from Docker CI. "
            "See planning/refactor-action-plan.md Phase 0.1"
        )


class TestPathResolution:
    """Tests for the paths module functionality."""

    def test_mlruns_dir_returns_path(self):
        """get_mlruns_dir should return a Path object."""
        from src.utils.paths import get_mlruns_dir

        result = get_mlruns_dir()
        assert isinstance(result, Path)

    def test_seri_db_returns_path(self):
        """get_seri_db_path should return a Path object."""
        from src.utils.paths import get_seri_db_path

        result = get_seri_db_path()
        assert isinstance(result, Path)

    def test_results_db_returns_path(self):
        """get_results_db_path should return a Path object."""
        from src.utils.paths import get_results_db_path

        result = get_results_db_path()
        assert isinstance(result, Path)

    def test_figures_dir_creates_directory(self):
        """get_figures_output_dir should create the directory if needed."""
        from src.utils.paths import get_figures_output_dir

        result = get_figures_output_dir()
        assert result.exists()
        assert result.is_dir()

    def test_project_root_is_correct(self):
        """PROJECT_ROOT should point to the repository root."""
        from src.utils.paths import PROJECT_ROOT

        # Should contain key markers of the project root
        assert (PROJECT_ROOT / "pyproject.toml").exists()
        assert (PROJECT_ROOT / "src").is_dir()
        assert (PROJECT_ROOT / "configs").is_dir()
