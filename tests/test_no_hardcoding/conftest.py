"""Auto-apply guardrail marker and auto-skip R tests when Rscript unavailable."""

import shutil

import pytest

HAS_RSCRIPT = shutil.which("Rscript") is not None


def pytest_collection_modifyitems(items):
    """Auto-apply guardrail marker; auto-skip r_required when Rscript missing."""
    skip_r = pytest.mark.skip(reason="Rscript not available in PATH")
    for item in items:
        if "test_no_hardcoding" in str(item.fspath):
            item.add_marker(pytest.mark.guardrail)
        if not HAS_RSCRIPT and "r_required" in item.keywords:
            item.add_marker(skip_r)
