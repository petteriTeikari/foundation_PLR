"""Auto-apply guardrail marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply guardrail marker to all tests in test_no_hardcoding/."""
    for item in items:
        if "test_no_hardcoding" in str(item.fspath):
            item.add_marker(pytest.mark.guardrail)
