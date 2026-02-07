"""Shared pytest fixtures for PLR pipeline tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Disable Prefect orchestration for tests (prevents server connection attempts)
os.environ["PREFECT_DISABLED"] = "1"

import numpy as np
import pytest
from omegaconf import OmegaConf


# ============================================================================
# Canonical Data Paths (single source of truth for test paths)
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DB = PROJECT_ROOT / "data" / "public" / "foundation_plr_results.db"
CD_DIAGRAM_DB = PROJECT_ROOT / "data" / "public" / "cd_diagram_data.duckdb"
R_DATA_DIR = PROJECT_ROOT / "data" / "r_data"
FIGURES_DIR = PROJECT_ROOT / "figures" / "generated"
DEMO_DB = PROJECT_ROOT / "data" / "synthetic" / "SYNTH_PLR_DEMO.db"


# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def demo_data_path(project_root) -> Path:
    """Return path to demo DuckDB file if it exists."""
    return DEMO_DB


@pytest.fixture
def demo_data_available(demo_data_path) -> bool:
    """Check if demo data is available."""
    return demo_data_path.exists()


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def minimal_cfg():
    """Return a minimal OmegaConf configuration for testing."""
    cfg_dict = {
        "DATA": {
            "PLR_length": 1981,
            "data_path": "data",
            "filename_DuckDB": "PLR_demo_data.db",
        },
        "EXPERIMENT": {
            "debug": False,
            "use_demo_data": True,
            "hyperparam_search": False,
        },
        "DEVICE": {
            "device": "cpu",
            "use_amp": False,
        },
        "OUTLIER_MODELS": {
            "LOF": {
                "MODEL": {
                    "n_neighbors": 20,
                    "contamination": 0.05,
                }
            }
        },
        "MODELS": {
            "MissForest": {
                "MODEL": {
                    "max_iter": 3,
                    "n_estimators": 10,
                }
            }
        },
        "CLS_MODELS": {
            "XGBOOST": {
                "MODEL": {
                    "n_estimators": 100,
                    "max_depth": 6,
                }
            }
        },
        "PREFECT": {
            "FLOW_NAMES": {
                "OUTLIER_DETECTION": "PLR_OutlierDetection",
            }
        },
    }
    return OmegaConf.create(cfg_dict)


@pytest.fixture
def outlier_detection_cfg(minimal_cfg):
    """Return config suitable for outlier detection testing."""
    cfg = OmegaConf.to_container(minimal_cfg, resolve=True)
    cfg["OUTLIER_MODELS"] = {
        "LOF": {
            "MODEL": {
                "n_neighbors": 20,
                "contamination": 0.05,
            },
            "HYPERPARAMS": {
                "method": "LIST",
            },
            "SEARCH_SPACE": {
                "LIST": {
                    "n_neighbors": [10, 20, 30],
                    "contamination": [0.01, 0.05, 0.10],
                }
            },
        }
    }
    return OmegaConf.create(cfg)


@pytest.fixture
def imputation_cfg(minimal_cfg):
    """Return config suitable for imputation testing."""
    cfg = OmegaConf.to_container(minimal_cfg, resolve=True)
    cfg["MODELS"] = {
        "MissForest": {
            "MODEL": {
                "max_iter": 3,
                "n_estimators": 10,
            },
            "HYPERPARAMS": {
                "method": "LIST",
            },
            "SEARCH_SPACE": {
                "LIST": {
                    "max_iter": [3, 5, 10],
                    "n_estimators": [10, 50, 100],
                }
            },
        }
    }
    return OmegaConf.create(cfg)


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def sample_plr_array():
    """Return synthetic PLR-like numpy array (8 subjects, 1981 timepoints)."""
    np.random.seed(42)
    n_subjects = 8
    n_timepoints = 1981

    # Create synthetic pupil response data
    # Baseline around 5mm, constriction to ~3mm, then recovery
    t = np.linspace(0, 66, n_timepoints)
    baseline = 5.0

    data = np.zeros((n_subjects, n_timepoints))
    for i in range(n_subjects):
        # Add individual variation
        noise = np.random.normal(0, 0.1, n_timepoints)
        # Simulate light response (constriction around t=20-30s)
        response = baseline - 2 * np.exp(-((t - 25) ** 2) / 50)
        data[i, :] = response + noise

    return data


@pytest.fixture
def sample_plr_array_3d(sample_plr_array):
    """Return 3D PLR array (subjects, timepoints, features)."""
    return sample_plr_array[:, :, np.newaxis]


@pytest.fixture
def sample_outlier_mask(sample_plr_array):
    """Return random 5% outlier mask matching sample PLR array."""
    np.random.seed(42)
    mask = np.random.random(sample_plr_array.shape) < 0.05
    return mask.astype(int)


@pytest.fixture
def sample_outlier_mask_3d(sample_outlier_mask):
    """Return 3D outlier mask (subjects, timepoints, features)."""
    return sample_outlier_mask[:, :, np.newaxis]


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_mlflow():
    """Mock MLflow calls for testing without MLflow server."""
    with (
        patch("mlflow.set_tracking_uri") as mock_uri,
        patch("mlflow.set_experiment") as mock_exp,
        patch("mlflow.start_run") as mock_run,
        patch("mlflow.log_metric") as mock_metric,
        patch("mlflow.log_param") as mock_param,
        patch("mlflow.log_artifact") as mock_artifact,
        patch("mlflow.end_run") as mock_end,
    ):
        # Configure mock run context
        mock_run_context = MagicMock()
        mock_run_context.info.run_id = "test_run_id_12345"
        mock_run.return_value.__enter__ = MagicMock(return_value=mock_run_context)
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        yield {
            "set_tracking_uri": mock_uri,
            "set_experiment": mock_exp,
            "start_run": mock_run,
            "log_metric": mock_metric,
            "log_param": mock_param,
            "log_artifact": mock_artifact,
            "end_run": mock_end,
        }


@pytest.fixture
def force_cpu():
    """Force torch.cuda.is_available() to return False."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_artifacts_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_duckdb_path(temp_artifacts_dir):
    """Return path for temporary DuckDB file."""
    return temp_artifacts_dir / "test_data.db"


# ============================================================================
# Skip Markers
# ============================================================================


@pytest.fixture
def skip_if_no_demo_data(demo_data_available, demo_data_path):
    """Fail if demo data is not available."""
    assert (
        demo_data_available
    ), f"Demo data missing: {demo_data_path}. Run: make synthetic"
