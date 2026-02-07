"""Auto-apply integration marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply integration marker to all tests in integration/."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
