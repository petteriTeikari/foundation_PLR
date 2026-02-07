"""Auto-apply unit marker to all tests in this directory."""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply unit marker to all tests in test_figure_generation/."""
    for item in items:
        if "test_figure_generation" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
