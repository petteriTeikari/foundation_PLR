"""
Figure QA Test Fixtures

ZERO TOLERANCE POLICY: All figure QA failures are CRITICAL.
There is no such thing as a "low priority" scientific integrity issue.
"""

import json
import pytest
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
R_DATA_DIR = PROJECT_ROOT / "data" / "r_data"
FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"
GOLDEN_DIR = Path(__file__).parent.parent / "golden_images"


@pytest.fixture
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def calibration_json_path():
    """Path to calibration data JSON."""
    path = R_DATA_DIR / "calibration_data.json"
    assert path.exists(), f"Calibration data missing: {path}. Run: make analyze"
    return path


@pytest.fixture
def calibration_data(calibration_json_path):
    """Loaded calibration data."""
    with open(calibration_json_path) as f:
        return json.load(f)


@pytest.fixture
def dca_json_path():
    """Path to DCA data JSON."""
    path = R_DATA_DIR / "dca_data.json"
    assert path.exists(), f"DCA data missing: {path}. Run: make analyze"
    return path


@pytest.fixture
def dca_data(dca_json_path):
    """Loaded DCA data."""
    with open(dca_json_path) as f:
        return json.load(f)


@pytest.fixture
def predictions_json_path():
    """Path to predictions JSON."""
    path = R_DATA_DIR / "predictions_top4.json"
    assert path.exists(), f"Predictions data missing: {path}. Run: make analyze"
    return path


@pytest.fixture
def predictions_data(predictions_json_path):
    """Loaded predictions data."""
    with open(predictions_json_path) as f:
        return json.load(f)


@pytest.fixture
def all_json_files():
    """All JSON data files in r_data directory."""
    assert R_DATA_DIR.exists(), (
        f"R data directory missing: {R_DATA_DIR}. Run: make analyze"
    )
    return list(R_DATA_DIR.glob("*.json"))


@pytest.fixture
def all_figure_files():
    """All generated figure files (PDF and PNG)."""
    assert FIGURES_DIR.exists(), (
        f"Figures directory missing: {FIGURES_DIR}. Run: make analyze"
    )
    pdfs = list(FIGURES_DIR.glob("**/*.pdf"))
    pngs = list(FIGURES_DIR.glob("**/*.png"))
    return pdfs + pngs


@pytest.fixture
def ggplot2_figures():
    """ggplot2-generated figures specifically."""
    ggplot_dir = FIGURES_DIR / "ggplot2"
    assert ggplot_dir.exists(), (
        f"ggplot2 directory missing: {ggplot_dir}. Run: make analyze"
    )
    return list(ggplot_dir.glob("*.pdf")) + list(ggplot_dir.glob("*.png"))


@pytest.fixture
def golden_dir():
    """Golden files directory for visual regression."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


def pytest_collection_modifyitems(items):
    """Auto-apply data marker to all tests in test_figure_qa/."""
    for item in items:
        if "test_figure_qa" in str(item.fspath):
            item.add_marker(pytest.mark.data)
