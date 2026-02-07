# Override parent conftest to avoid numpy dependency for guardrail tests
# These tests only need standard library + yaml
import sys
from pathlib import Path

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(items):
    """Auto-apply guardrail marker to all tests in this directory."""
    for item in items:
        if "test_guardrails" in str(item.fspath):
            item.add_marker(pytest.mark.guardrail)
