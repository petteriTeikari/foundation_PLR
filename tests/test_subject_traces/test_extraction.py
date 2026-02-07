# tests/test_subject_traces/test_extraction.py
"""
TDD tests for subject traces extraction.
Run: pytest tests/test_subject_traces/test_extraction.py -v
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.guardrail

# Project root
PROJECT_ROOT = Path(__file__).parents[2]


class TestSubjectTracesJSONSchema:
    """Test that extracted JSON has correct structure."""

    @pytest.fixture
    def json_path(self):
        return PROJECT_ROOT / "data" / "r_data" / "subject_traces.json"

    @pytest.fixture
    def json_data(self, json_path):
        if not json_path.exists():
            pytest.skip(
                f"subject_traces.json not found: {json_path}. Run: make analyze"
            )
        with open(json_path) as f:
            return json.load(f)

    def test_has_metadata(self, json_data):
        """JSON must have metadata section."""
        assert "metadata" in json_data
        assert "created" in json_data["metadata"]
        assert "n_subjects" in json_data["metadata"]
        assert "light_protocol" in json_data["metadata"]

    def test_has_subjects_list(self, json_data):
        """JSON must have subjects list."""
        assert "subjects" in json_data
        assert isinstance(json_data["subjects"], list)
        assert len(json_data["subjects"]) == 12  # 6 control + 6 glaucoma

    def test_subject_required_fields(self, json_data):
        """Each subject must have required fields."""
        required_fields = [
            "subject_id",  # Anonymized (H001, G001, etc.)
            "class_label",  # control or glaucoma
            "outlier_pct",  # percentage
            "note",  # description
            "n_timepoints",  # should be 1981
            "time",  # array
            "pupil_orig",  # array (may have NaN)
            "pupil_gt",  # array (ground truth)
            "outlier_mask",  # array (0/1)
            "blue_stimulus",  # array (light intensity)
            "red_stimulus",  # array (light intensity)
        ]
        for subject in json_data["subjects"]:
            for field in required_fields:
                assert field in subject, (
                    f"Missing field: {field} for {subject.get('subject_id', 'unknown')}"
                )

    def test_no_plr_codes_in_subject_ids(self, json_data):
        """Subject IDs must be anonymized (no PLRxxxx codes)."""
        for subject in json_data["subjects"]:
            assert not subject["subject_id"].startswith("PLR"), (
                f"PLR code found: {subject['subject_id']}"
            )
            assert subject["subject_id"][0] in [
                "H",
                "G",
            ], f"Invalid anonymized ID: {subject['subject_id']}"

    def test_arrays_same_length(self, json_data):
        """All arrays must have same length (n_timepoints)."""
        for subject in json_data["subjects"]:
            n = subject["n_timepoints"]
            assert len(subject["time"]) == n
            assert len(subject["pupil_gt"]) == n
            assert len(subject["outlier_mask"]) == n
            assert len(subject["blue_stimulus"]) == n
            assert len(subject["red_stimulus"]) == n

    def test_class_distribution(self, json_data):
        """Should have 6 control and 6 glaucoma subjects."""
        controls = [s for s in json_data["subjects"] if s["class_label"] == "control"]
        glaucomas = [s for s in json_data["subjects"] if s["class_label"] == "glaucoma"]
        assert len(controls) == 6, f"Expected 6 controls, got {len(controls)}"
        assert len(glaucomas) == 6, f"Expected 6 glaucomas, got {len(glaucomas)}"


class TestLightProtocolTiming:
    """Test that light protocol timing is correctly extracted."""

    @pytest.fixture
    def json_data(self):
        json_path = PROJECT_ROOT / "data" / "r_data" / "subject_traces.json"
        if not json_path.exists():
            pytest.skip(
                f"subject_traces.json not found: {json_path}. Run: make analyze"
            )
        with open(json_path) as f:
            return json.load(f)

    def test_light_protocol_metadata(self, json_data):
        """Metadata must include light protocol timing."""
        protocol = json_data["metadata"]["light_protocol"]
        assert "blue_start" in protocol
        assert "blue_end" in protocol
        assert "red_start" in protocol
        assert "red_end" in protocol

        # Expected timing (from actual data analysis)
        # Blue: ~15.5s - ~24.5s
        # Red: ~46.5s - ~55.5s
        assert 14 <= protocol["blue_start"] <= 17  # ~15.5s
        assert 23 <= protocol["blue_end"] <= 26  # ~24.5s
        assert 45 <= protocol["red_start"] <= 48  # ~46.5s
        assert 54 <= protocol["red_end"] <= 57  # ~55.5s

    def test_blue_before_red(self, json_data):
        """Blue stimulus must come before red stimulus."""
        protocol = json_data["metadata"]["light_protocol"]
        assert protocol["blue_start"] < protocol["red_start"]
        assert protocol["blue_end"] < protocol["red_start"]

    def test_stimulus_arrays_match_protocol(self, json_data):
        """Stimulus arrays should have non-zero values during protocol periods."""
        subject = json_data["subjects"][0]  # Test first subject
        time = subject["time"]
        blue = subject["blue_stimulus"]
        red = subject["red_stimulus"]
        protocol = json_data["metadata"]["light_protocol"]

        # Find indices where blue should be active
        blue_active_idx = [
            i
            for i, t in enumerate(time)
            if protocol["blue_start"] <= t <= protocol["blue_end"]
        ]

        # At least some blue values should be non-zero during blue period
        blue_during_blue = [blue[i] for i in blue_active_idx]
        assert max(blue_during_blue) > 0, (
            "Blue stimulus should be non-zero during blue period"
        )

        # Find indices where red should be active
        red_active_idx = [
            i
            for i, t in enumerate(time)
            if protocol["red_start"] <= t <= protocol["red_end"]
        ]

        # At least some red values should be non-zero during red period
        red_during_red = [red[i] for i in red_active_idx]
        assert max(red_during_red) > 0, (
            "Red stimulus should be non-zero during red period"
        )


class TestDataQuality:
    """Test data quality requirements."""

    @pytest.fixture
    def json_data(self):
        json_path = PROJECT_ROOT / "data" / "r_data" / "subject_traces.json"
        if not json_path.exists():
            pytest.skip(
                f"subject_traces.json not found: {json_path}. Run: make analyze"
            )
        with open(json_path) as f:
            return json.load(f)

    def test_ground_truth_has_no_missing(self, json_data):
        """pupil_gt should have no missing values (it's the denoised signal)."""
        for subject in json_data["subjects"]:
            gt = subject["pupil_gt"]
            none_count = sum(1 for v in gt if v is None)
            assert none_count == 0, (
                f"{subject['subject_id']} has {none_count} missing values in pupil_gt"
            )

    def test_outlier_mask_binary(self, json_data):
        """outlier_mask should be binary (0 or 1)."""
        for subject in json_data["subjects"]:
            mask = subject["outlier_mask"]
            unique_values = set(mask)
            assert unique_values <= {
                0,
                1,
            }, f"{subject['subject_id']} has non-binary outlier_mask: {unique_values}"

    def test_outlier_percentage_matches(self, json_data):
        """Calculated outlier percentage should match reported value."""
        for subject in json_data["subjects"]:
            mask = subject["outlier_mask"]
            calculated_pct = 100 * sum(mask) / len(mask)
            reported_pct = subject["outlier_pct"]
            # Allow 0.5% tolerance for rounding
            assert abs(calculated_pct - reported_pct) < 0.5, (
                f"{subject['subject_id']}: calculated {calculated_pct:.2f}% != reported {reported_pct:.2f}%"
            )

    def test_time_monotonic(self, json_data):
        """Time should be monotonically increasing."""
        for subject in json_data["subjects"]:
            time = subject["time"]
            for i in range(1, len(time)):
                assert time[i] > time[i - 1], (
                    f"{subject['subject_id']}: time not monotonic at index {i}"
                )
