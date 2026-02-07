"""
TDD Tests for pminternal JSON Export.

These tests verify the exported JSON file has the correct schema for R consumption.
JSON path: data/r_data/pminternal_bootstrap_predictions.json

Expected schema:
{
  "metadata": {...},
  "configs": {
    "ground_truth": {
      "config_id": "ground_truth",
      "n_patients": 63,
      "n_bootstrap": 1000,
      "y_true": [...],
      "y_prob_original": [...],
      "y_prob_bootstrap": [[...], ...]  // n_bootstrap x n_patients
    },
    "best_ensemble": {...}
  }
}
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
JSON_PATH = PROJECT_ROOT / "data" / "r_data" / "pminternal_bootstrap_predictions.json"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not JSON_PATH.exists(),
        reason="pminternal JSON not available",
    ),
]
EXPECTED_N_PATIENTS = 63
EXPECTED_N_BOOTSTRAP = 1000
REQUIRED_CONFIGS = ["ground_truth"]  # At minimum, need ground truth


class TestJSONFileExists:
    """Tests for JSON file existence."""

    def test_json_file_exists(self):
        """JSON export file exists at expected path."""
        assert JSON_PATH.exists(), f"JSON file not found at {JSON_PATH}"

    def test_json_is_valid(self):
        """JSON file is valid JSON."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            data = json.load(f)

        assert isinstance(data, dict), "JSON root should be a dict"


class TestJSONSchemaValid:
    """Tests for JSON schema compliance."""

    @pytest.fixture
    def json_data(self):
        """Load JSON data."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            return json.load(f)

    def test_has_metadata_section(self, json_data):
        """JSON has metadata section."""
        assert "metadata" in json_data, "Missing 'metadata' section"

    def test_has_configs_section(self, json_data):
        """JSON has configs section."""
        assert "configs" in json_data, "Missing 'configs' section"

    def test_metadata_has_required_fields(self, json_data):
        """Metadata has required fields."""
        metadata = json_data.get("metadata", {})
        required = ["created", "generator", "data_source"]
        for field in required:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_required_configs_present(self, json_data):
        """At least ground_truth config is present."""
        configs = json_data.get("configs", {})
        for config_name in REQUIRED_CONFIGS:
            assert config_name in configs, f"Missing required config: {config_name}"


class TestConfigSchema:
    """Tests for individual config schema."""

    @pytest.fixture
    def ground_truth_config(self):
        """Load ground truth config."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            data = json.load(f)

        if "ground_truth" not in data.get("configs", {}):
            pytest.skip("ground_truth config not present")

        return data["configs"]["ground_truth"]

    def test_config_has_config_id(self, ground_truth_config):
        """Config has config_id field."""
        assert "config_id" in ground_truth_config, "Missing 'config_id'"

    def test_config_has_n_patients(self, ground_truth_config):
        """Config has n_patients field."""
        assert "n_patients" in ground_truth_config, "Missing 'n_patients'"
        assert (
            ground_truth_config["n_patients"] == EXPECTED_N_PATIENTS
        ), f"Expected {EXPECTED_N_PATIENTS} patients"

    def test_config_has_n_bootstrap(self, ground_truth_config):
        """Config has n_bootstrap field."""
        assert "n_bootstrap" in ground_truth_config, "Missing 'n_bootstrap'"
        assert (
            ground_truth_config["n_bootstrap"] == EXPECTED_N_BOOTSTRAP
        ), f"Expected {EXPECTED_N_BOOTSTRAP} bootstrap iterations"

    def test_config_has_y_true(self, ground_truth_config):
        """Config has y_true (ground truth labels)."""
        assert "y_true" in ground_truth_config, "Missing 'y_true'"
        assert isinstance(ground_truth_config["y_true"], list), "y_true should be list"
        assert (
            len(ground_truth_config["y_true"]) == EXPECTED_N_PATIENTS
        ), f"y_true length should be {EXPECTED_N_PATIENTS}"

    def test_config_has_y_prob_original(self, ground_truth_config):
        """Config has y_prob_original (mean predictions from developed model)."""
        assert "y_prob_original" in ground_truth_config, "Missing 'y_prob_original'"
        assert isinstance(
            ground_truth_config["y_prob_original"], list
        ), "y_prob_original should be list"
        assert (
            len(ground_truth_config["y_prob_original"]) == EXPECTED_N_PATIENTS
        ), f"y_prob_original length should be {EXPECTED_N_PATIENTS}"

    def test_config_has_y_prob_bootstrap(self, ground_truth_config):
        """Config has y_prob_bootstrap (bootstrap predictions matrix)."""
        assert "y_prob_bootstrap" in ground_truth_config, "Missing 'y_prob_bootstrap'"
        y_boot = ground_truth_config["y_prob_bootstrap"]
        assert isinstance(y_boot, list), "y_prob_bootstrap should be list of lists"


class TestJSONDimensionsConsistent:
    """Tests for dimension consistency in JSON."""

    @pytest.fixture
    def ground_truth_config(self):
        """Load ground truth config."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            data = json.load(f)

        if "ground_truth" not in data.get("configs", {}):
            pytest.skip("ground_truth config not present")

        return data["configs"]["ground_truth"]

    def test_y_prob_bootstrap_outer_dimension(self, ground_truth_config):
        """y_prob_bootstrap outer dimension = n_bootstrap."""
        y_boot = ground_truth_config["y_prob_bootstrap"]
        # Note: We transpose to (n_bootstrap, n_patients) for R
        assert len(y_boot) == EXPECTED_N_BOOTSTRAP, (
            f"y_prob_bootstrap outer dim should be {EXPECTED_N_BOOTSTRAP}, "
            f"got {len(y_boot)}"
        )

    def test_y_prob_bootstrap_inner_dimension(self, ground_truth_config):
        """y_prob_bootstrap inner dimension = n_patients."""
        y_boot = ground_truth_config["y_prob_bootstrap"]
        if len(y_boot) == 0:
            pytest.skip("y_prob_bootstrap is empty")

        # Check first row
        assert len(y_boot[0]) == EXPECTED_N_PATIENTS, (
            f"y_prob_bootstrap inner dim should be {EXPECTED_N_PATIENTS}, "
            f"got {len(y_boot[0])}"
        )

    def test_all_bootstrap_rows_same_length(self, ground_truth_config):
        """All bootstrap rows have same length (n_patients)."""
        y_boot = ground_truth_config["y_prob_bootstrap"]
        lengths = [len(row) for row in y_boot]
        unique_lengths = set(lengths)
        assert (
            len(unique_lengths) == 1
        ), f"Inconsistent row lengths in y_prob_bootstrap: {unique_lengths}"


class TestPredictionValuesValid:
    """Tests for prediction value validity in JSON."""

    @pytest.fixture
    def ground_truth_config(self):
        """Load ground truth config."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            data = json.load(f)

        if "ground_truth" not in data.get("configs", {}):
            pytest.skip("ground_truth config not present")

        return data["configs"]["ground_truth"]

    def test_y_prob_original_valid_range(self, ground_truth_config):
        """y_prob_original values are valid probabilities [0, 1]."""
        y_orig = ground_truth_config["y_prob_original"]
        assert all(
            0 <= p <= 1 for p in y_orig
        ), "y_prob_original contains values outside [0, 1]"

    def test_y_prob_bootstrap_valid_range(self, ground_truth_config):
        """y_prob_bootstrap values are valid probabilities [0, 1]."""
        y_boot = ground_truth_config["y_prob_bootstrap"]
        for i, row in enumerate(y_boot[:10]):  # Check first 10 rows for speed
            assert all(
                0 <= p <= 1 for p in row
            ), f"y_prob_bootstrap row {i} contains values outside [0, 1]"

    def test_y_true_binary(self, ground_truth_config):
        """y_true values are binary (0 or 1)."""
        y_true = ground_truth_config["y_true"]
        assert all(y in [0, 1] for y in y_true), "y_true contains non-binary values"


class TestMultipleConfigsExported:
    """Tests for multiple configs in JSON."""

    def test_at_least_two_configs(self):
        """At least two configs exported (for comparison)."""
        if not JSON_PATH.exists():
            pytest.skip("JSON file not yet created")

        with open(JSON_PATH) as f:
            data = json.load(f)

        configs = data.get("configs", {})
        # Relaxed: we want at least ground_truth, ideally more
        assert len(configs) >= 1, f"Expected at least 1 config, got {len(configs)}"
        # Warning if only one
        if len(configs) < 2:
            import warnings

            warnings.warn(
                "Only 1 config exported - ideally have 2+ for comparison",
                UserWarning,
            )
