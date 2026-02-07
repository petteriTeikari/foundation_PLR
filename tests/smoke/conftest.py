"""Auto-apply unit marker to smoke tests."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply unit marker to all tests in smoke/."""
    for item in items:
        if "/smoke/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
