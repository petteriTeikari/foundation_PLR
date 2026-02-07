# tests/test_data_quality/test_validation_module.py
"""
Tests for the normalization validation module.

Run: pytest tests/test_data_quality/test_validation_module.py -v
"""

import pytest

pytestmark = pytest.mark.unit


class TestValidationModule:
    """Test the validation module components."""

    def test_import_validation_module(self):
        """Validation module should be importable."""
        from src.data_io.validation import (
            NormalizationValidator,
            ScalingAnomaly,
        )

        assert NormalizationValidator is not None
        assert ScalingAnomaly is not None

    def test_known_anomalies_contains_plr4018(self):
        """PLR4018 should be in known anomalies list."""
        from src.data_io.validation import get_known_anomalies

        known = get_known_anomalies()
        assert "PLR4018" in known
        assert "description" in known["PLR4018"]

    def test_validate_subject_scaling_detects_anomaly(self):
        """validate_subject_scaling should detect large offsets."""
        from src.data_io.validation import validate_subject_scaling

        # Normal data - no anomaly
        result1 = validate_subject_scaling(
            "TEST001",
            pupil_orig=[1.0, 2.0, 3.0, 4.0],
            pupil_raw=[1.0, 2.0, 3.0, 4.0],
        )
        assert result1 is None, "Should not detect anomaly for identical values"

        # Small offset - no anomaly
        result2 = validate_subject_scaling(
            "TEST002",
            pupil_orig=[11.0, 12.0, 13.0, 14.0],
            pupil_raw=[1.0, 2.0, 3.0, 4.0],
        )
        assert result2 is None, "Should not detect anomaly for offset < 20"

        # Large offset - should detect
        result3 = validate_subject_scaling(
            "TEST003",
            pupil_orig=[101.0, 102.0, 103.0, 104.0],
            pupil_raw=[1.0, 2.0, 3.0, 4.0],
        )
        assert result3 is not None, "Should detect anomaly for offset > 50"
        assert result3.subject_code == "TEST003"
        assert result3.severity.value == "critical"

    def test_validate_subject_scaling_handles_none(self):
        """validate_subject_scaling should handle None values."""
        from src.data_io.validation import validate_subject_scaling

        _ = validate_subject_scaling(
            "TEST004",
            pupil_orig=[1.0, None, 3.0],
            pupil_raw=[1.0, 2.0, None],
        )
        # Should not crash, and should return None for this data
        # (only one valid pair at index 0)


class TestValidatorIntegration:
    """Integration tests requiring database access."""

    @pytest.fixture
    def validator(self):
        """Create validator if database available."""
        from src.data_io.validation import NormalizationValidator

        try:
            v = NormalizationValidator()
            yield v
            v.close()
        except FileNotFoundError:
            pytest.skip("Database not available (expected in CI)")

    def test_detector_finds_plr4018(self, validator):
        """Validator should detect PLR4018 as known anomaly."""
        anomalies = validator.detect_all_anomalies()

        plr4018_anomaly = next(
            (a for a in anomalies if a.subject_code == "PLR4018"), None
        )

        assert plr4018_anomaly is not None, "PLR4018 should be detected"
        assert plr4018_anomaly.is_known, "PLR4018 should be marked as known"
        assert plr4018_anomaly.severity.value == "critical"

    def test_no_new_critical_anomalies(self, validator):
        """Should be no NEW critical anomalies beyond known list."""
        _ = validator.detect_all_anomalies()
        new_critical = validator.get_new_anomalies()

        # Filter for critical only
        new_critical_severe = [
            a for a in new_critical if a.severity.value == "critical"
        ]

        assert not new_critical_severe, (
            "New critical anomalies detected:\n"
            + "\n".join(str(a) for a in new_critical_severe)
        )

    def test_validate_or_raise_passes(self, validator):
        """validate_or_raise should not raise for current data."""
        # Should not raise since PLR4018 is known
        validator.validate_or_raise()

    def test_report_generation(self, validator):
        """Should generate a report without errors."""
        report = validator.print_report()
        assert "NORMALIZATION VALIDATION REPORT" in report
        assert "PLR4018" in report
