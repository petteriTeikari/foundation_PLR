"""
Test that extraction scripts don't have unbounded memory accumulation patterns.

This test runs the AST-based pattern checker on all extraction scripts to ensure
they don't have the pattern that caused CRITICAL-FAILURE-005 (24-hour stuck extraction).
"""

import subprocess
import sys
from pathlib import Path


def test_no_unbounded_accumulation_patterns():
    """All extraction scripts should pass the pattern checker."""
    project_root = Path(__file__).parent.parent.parent
    checker_script = (
        project_root / "scripts" / "validation" / "check_extraction_patterns.py"
    )

    result = subprocess.run(
        [sys.executable, str(checker_script), "--all"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # The checker exits 0 and prints "No memory accumulation patterns detected."
    # when all scripts pass
    assert "No memory accumulation patterns detected" in result.stdout, (
        f"Pattern checker found issues:\n{result.stdout}\n{result.stderr}"
    )


def test_pattern_checker_detects_bad_patterns():
    """Pattern checker should detect accumulation patterns when present."""
    import tempfile

    # Create a bad script that has unbounded accumulation
    bad_script = """
import pickle
def process():
    all_rows = []
    for item in range(1000):
        all_rows.append({"data": item})
    return all_rows
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(bad_script)
        temp_path = f.name

    try:
        project_root = Path(__file__).parent.parent.parent
        checker_script = (
            project_root / "scripts" / "validation" / "check_extraction_patterns.py"
        )

        result = subprocess.run(
            [sys.executable, str(checker_script), temp_path],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # Should detect the all_rows.append pattern
        assert "all_rows" in result.stdout or "Unbounded" in result.stdout, (
            f"Pattern checker should have detected all_rows.append:\n{result.stdout}"
        )
    finally:
        Path(temp_path).unlink()
