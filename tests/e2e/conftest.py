"""Auto-apply e2e marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply e2e marker to all tests in e2e/."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
