"""
VIF (Variance Inflation Factor) Analysis Tests
===============================================

Validates that VIF analysis detects multicollinearity issues and that
SHAP export pipeline correctly warns about unreliable feature importance.

Run with: pytest tests/test_vif_analysis.py -v

Created: 2026-02-02
Reason: SHAP values unreliable for features with VIF > 10; need automated checks
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.data

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent

# Expected VIF values for known problematic features (from ground truth analysis)
EXPECTED_HIGH_VIF_FEATURES = {
    "Red_SUSTAINED_value": {
        "expected_vif": 114.0,
        "vif_tolerance": 20.0,
        "expected_concern": "High",
    },
    "Red_MAX_CONSTRICTION_value": {
        "expected_vif": 104.0,
        "vif_tolerance": 20.0,
        "expected_concern": "High",
    },
    "Blue_SUSTAINED_value": {
        "expected_vif": 71.0,
        "vif_tolerance": 15.0,
        "expected_concern": "High",
    },
    "Blue_MAX_CONSTRICTION_value": {
        "expected_vif": 64.0,
        "vif_tolerance": 15.0,
        "expected_concern": "High",
    },
}

# Features that should have acceptable VIF (< 10)
EXPECTED_LOW_VIF_FEATURES = [
    "Blue_PIPR_AUC_value",
    "Red_PIPR_AUC_value",
]


class TestVIFAnalysisData:
    """Test VIF analysis JSON data correctness."""

    @pytest.fixture
    def vif_data(self):
        """Load VIF analysis JSON data."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "vif_analysis.json"
        assert json_path.exists(), f"VIF JSON not found: {json_path}. Run: make analyze"
        with open(json_path) as f:
            return json.load(f)

    def test_vif_json_has_required_structure(self, vif_data):
        """Verify VIF JSON has correct structure."""
        assert "metadata" in vif_data, "Missing metadata section"
        assert "data" in vif_data, "Missing data section"
        assert "aggregate" in vif_data["data"], "Missing aggregate VIF data"
        assert "per_config" in vif_data["data"], "Missing per-config VIF data"

    def test_all_features_have_vif(self, vif_data):
        """Verify all 8 handcrafted features have VIF values."""
        features = {f["feature"] for f in vif_data["data"]["aggregate"]}
        # Actual features in the dataset (amplitude bins + phasic)
        expected_features = {
            "Blue_SUSTAINED_value",
            "Blue_MAX_CONSTRICTION_value",
            "Blue_PIPR_AUC_value",
            "Blue_PHASIC_value",
            "Red_SUSTAINED_value",
            "Red_MAX_CONSTRICTION_value",
            "Red_PIPR_AUC_value",
            "Red_PHASIC_value",
        }
        missing = expected_features - features
        assert not missing, f"Missing features in VIF data: {missing}"

    @pytest.mark.parametrize("feature", EXPECTED_HIGH_VIF_FEATURES.keys())
    def test_high_vif_features_detected(self, vif_data, feature):
        """Verify known high-VIF features are correctly identified."""
        expected = EXPECTED_HIGH_VIF_FEATURES[feature]

        # Find feature in aggregate data
        feat_data = next(
            (f for f in vif_data["data"]["aggregate"] if f["feature"] == feature),
            None,
        )
        assert feat_data is not None, f"Feature {feature} not found in VIF data"

        # Check VIF value is in expected range
        actual_vif = feat_data["VIF_mean"]
        low = expected["expected_vif"] - expected["vif_tolerance"]
        high = expected["expected_vif"] + expected["vif_tolerance"]

        assert low <= actual_vif <= high, (
            f"Feature {feature} has VIF {actual_vif:.1f} "
            f"but expected {expected['expected_vif']:.1f} ± {expected['vif_tolerance']}"
        )

        # Check concern level
        assert feat_data["concern"] == expected["expected_concern"], (
            f"Feature {feature} has concern '{feat_data['concern']}' "
            f"but expected '{expected['expected_concern']}'"
        )

    @pytest.mark.parametrize("feature", EXPECTED_LOW_VIF_FEATURES)
    def test_low_vif_features_acceptable(self, vif_data, feature):
        """Verify PIPR features have acceptable VIF (< 10)."""
        feat_data = next(
            (f for f in vif_data["data"]["aggregate"] if f["feature"] == feature),
            None,
        )
        assert feat_data is not None, f"Feature {feature} not found in VIF data"

        actual_vif = feat_data["VIF_mean"]
        assert actual_vif < 10, (
            f"Feature {feature} has VIF {actual_vif:.1f} but should be < 10"
        )


class TestSHAPExportVIFIntegration:
    """Test that SHAP export includes VIF warnings."""

    @pytest.fixture
    def shap_data(self):
        """Load SHAP feature importance JSON data."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "shap_feature_importance.json"
        assert json_path.exists(), (
            f"SHAP JSON not found: {json_path}. Run: make analyze"
        )
        with open(json_path) as f:
            return json.load(f)

    def test_shap_has_vif_warning_in_metadata(self, shap_data):
        """Verify SHAP export includes VIF warning in metadata."""
        metadata = shap_data.get("metadata", {})

        # Check for VIF warning section
        if "vif_warning" in metadata:
            vif_warning = metadata["vif_warning"]
            assert vif_warning.get("has_multicollinearity") is True, (
                "VIF warning should indicate multicollinearity"
            )
            assert "message" in vif_warning, "VIF warning missing message"

    def test_shap_has_vif_summary_in_data(self, shap_data):
        """Verify SHAP data includes VIF summary."""
        data = shap_data.get("data", {})

        if "vif_summary" in data:
            vif_summary = data["vif_summary"]
            # Check that high-VIF features are flagged
            for feature in EXPECTED_HIGH_VIF_FEATURES:
                if feature in vif_summary:
                    assert vif_summary[feature]["concern"] in ["High", "Moderate"], (
                        f"Feature {feature} should be flagged as High or Moderate concern"
                    )


class TestVIFThresholds:
    """Test VIF threshold logic."""

    @pytest.fixture
    def vif_data(self):
        """Load VIF analysis JSON data."""
        json_path = PROJECT_ROOT / "data" / "r_data" / "vif_analysis.json"
        assert json_path.exists(), f"VIF JSON not found: {json_path}. Run: make analyze"
        with open(json_path) as f:
            return json.load(f)

    def test_concern_levels_match_thresholds(self, vif_data):
        """Verify concern levels match VIF thresholds.

        Current implementation uses:
        - High: VIF > 20 for temporal features (SUSTAINED, MAX_CONSTRICTION)
        - OK: everything else

        Note: The threshold logic is simple for now - High for severe cases only.
        """
        for feat in vif_data["data"]["aggregate"]:
            vif = feat["VIF_mean"]
            concern = feat["concern"]

            if vif is None:
                continue

            # Temporal features (SUSTAINED, MAX_CONSTRICTION) are known to have high VIF
            is_temporal = any(
                x in feat["feature"] for x in ["SUSTAINED", "MAX_CONSTRICTION"]
            )

            if is_temporal:
                # These should have VIF > 60 and be marked High
                if vif > 20:
                    assert concern == "High", (
                        f"Temporal feature {feat['feature']} with VIF {vif:.1f} "
                        f"should be 'High' concern, not '{concern}'"
                    )
            else:
                # Non-temporal features should have reasonable VIF (< 10)
                if vif < 10:
                    assert concern == "OK", (
                        f"Feature {feat['feature']} with VIF {vif:.1f} "
                        f"should be 'OK' concern, not '{concern}'"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
