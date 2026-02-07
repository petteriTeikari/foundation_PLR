"""
P1 HIGH: Statistical and Scientific Validity Tests

Validate STRATOS compliance, metric ranges, and statistical sanity.

ZERO TOLERANCE: Invalid statistics = invalid science = scientific misconduct.
"""

import warnings

import numpy as np
import pytest


class TestSTRATOSMetricRanges:
    """
    Validate that STRATOS-compliant metrics are within valid/plausible ranges.
    """

    # Metric name -> (min, max, description)
    METRIC_RANGES = {
        "auroc": (0.5, 1.0, "AUROC must be >= 0.5 (chance) and <= 1.0"),
        "brier": (0.0, 0.25, "Brier score for binary outcomes is in [0, 0.25]"),
        "ipa": (-10.0, 1.0, "IPA (scaled Brier) has max 1.0, can be negative"),
        "calibration_slope": (-1.0, 5.0, "Slope outside this range is extreme"),
        "calibration_intercept": (-3.0, 3.0, "Intercept outside this range is extreme"),
        "o_e_ratio": (
            0.01,
            100.0,
            "O:E ratio outside this range indicates severe miscalibration",
        ),
    }

    def test_metrics_in_valid_ranges(self, calibration_data):
        """All metrics must be within theoretically valid ranges."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            name = config.get("name", "unknown")

            for metric, (lo, hi, desc) in self.METRIC_RANGES.items():
                if metric in config:
                    value = config[metric]

                    # Check for NaN/Inf
                    if np.isnan(value) or np.isinf(value):
                        pytest.fail(
                            f"CRITICAL: {name}.{metric} is {value} (NaN/Inf). "
                            f"This indicates a computation error."
                        )

                    # Check range
                    assert lo <= value <= hi, (
                        f"CRITICAL: {name}.{metric}={value:.4f} outside valid range [{lo}, {hi}]. "
                        f"{desc}"
                    )

    def test_auroc_above_chance(self, calibration_data):
        """AUROC below 0.5 means worse than random - likely a bug."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            if "auroc" in config:
                auroc = config["auroc"]
                assert auroc >= 0.5, (
                    f"CRITICAL: {config.get('name', 'unknown')} has AUROC={auroc:.4f} < 0.5. "
                    f"This is worse than chance and indicates an error "
                    f"(labels may be inverted)."
                )

    def test_brier_consistent_with_prevalence(self, calibration_data):
        """Brier score should be consistent with prevalence."""
        data = calibration_data.get("data", {})
        prevalence = data.get("sample_prevalence", 0.5)

        # Null model Brier = prevalence * (1 - prevalence)
        null_brier = prevalence * (1 - prevalence)

        configs = data.get("configs", [])
        for config in configs:
            if "brier" in config:
                brier = config["brier"]

                # Brier much worse than null model is suspicious
                if brier > null_brier * 2:
                    warnings.warn(
                        f"{config.get('name', 'unknown')} has Brier={brier:.4f} "
                        f"which is much worse than null model ({null_brier:.4f}). "
                        f"Model may be miscalibrated or predictions inverted."
                    )


class TestCalibrationCurveValidity:
    """
    Validate calibration curve structure and data.
    """

    def test_bin_midpoints_are_valid(self, calibration_data):
        """Bin midpoints should be in [0, 1] and ordered."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            curve = config.get("curve", {})
            midpoints = curve.get("bin_midpoints", [])

            if not midpoints:
                continue

            # Check range
            for mp in midpoints:
                assert 0.0 <= mp <= 1.0, (
                    f"CRITICAL: {config.get('name', 'unknown')} has bin midpoint {mp} "
                    f"outside [0, 1] range."
                )

            # Check ordering
            for i in range(len(midpoints) - 1):
                assert midpoints[i] < midpoints[i + 1], (
                    f"CRITICAL: {config.get('name', 'unknown')} has non-monotonic "
                    f"bin midpoints: {midpoints}"
                )

    def test_observed_proportions_are_valid(self, calibration_data):
        """Observed proportions must be in [0, 1] where not null."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            curve = config.get("curve", {})
            observed = curve.get("observed", [])

            for i, obs in enumerate(observed):
                if obs is not None:
                    assert 0.0 <= obs <= 1.0, (
                        f"CRITICAL: {config.get('name', 'unknown')} has observed "
                        f"proportion {obs} at bin {i} outside [0, 1] range."
                    )

    def test_counts_are_non_negative(self, calibration_data):
        """Bin counts must be non-negative integers."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            curve = config.get("curve", {})
            counts = curve.get("counts", [])

            for i, count in enumerate(counts):
                assert count >= 0, (
                    f"CRITICAL: {config.get('name', 'unknown')} has negative "
                    f"count {count} at bin {i}."
                )
                assert count == int(count), (
                    f"CRITICAL: {config.get('name', 'unknown')} has non-integer "
                    f"count {count} at bin {i}."
                )

    def test_counts_sum_to_n(self, calibration_data):
        """Sum of bin counts should equal n."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            n = config.get("n", 0)
            curve = config.get("curve", {})
            counts = curve.get("counts", [])

            if counts and n > 0:
                total = sum(counts)
                assert total == n, (
                    f"CRITICAL: {config.get('name', 'unknown')} bin counts sum to "
                    f"{total} but n={n}. Data may be corrupted or misaligned."
                )

    def test_calibration_slope_not_exactly_one(self, calibration_data):
        """
        Perfect calibration (slope=1.0, intercept=0.0) is suspicious.
        Real data almost never has perfect calibration.
        """
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            slope = config.get("calibration_slope", None)
            intercept = config.get("calibration_intercept", None)

            if slope is not None and intercept is not None:
                # Check for suspiciously perfect calibration
                if abs(slope - 1.0) < 1e-6 and abs(intercept) < 1e-6:
                    pytest.fail(
                        f"CRITICAL: {config.get('name', 'unknown')} has PERFECT "
                        f"calibration (slope=1.0, intercept=0.0). This is extremely "
                        f"unlikely with real data and suggests synthetic data or a bug."
                    )


class TestBootstrapCIValidity:
    """
    Validate bootstrap confidence interval properties.
    """

    def _find_cis(self, obj, path="", cis=None):
        """Recursively find all CI pairs in a nested structure."""
        if cis is None:
            cis = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith("_ci_lo"):
                    base = key[:-6]
                    hi_key = f"{base}_ci_hi"
                    if hi_key in obj:
                        cis.append(
                            {
                                "path": f"{path}.{base}",
                                "lo": obj[key],
                                "hi": obj[hi_key],
                                "point": obj.get(base),
                            }
                        )
                self._find_cis(value, f"{path}.{key}", cis)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._find_cis(item, f"{path}[{i}]", cis)

        return cis

    def test_cis_are_ordered(self, calibration_data):
        """CI lower bound must be <= upper bound."""
        cis = self._find_cis(calibration_data)

        for ci in cis:
            if ci["lo"] is not None and ci["hi"] is not None:
                assert ci["lo"] <= ci["hi"], (
                    f"CRITICAL: {ci['path']} has inverted CI: [{ci['lo']}, {ci['hi']}]"
                )

    def test_cis_not_zero_width(self, calibration_data):
        """Zero-width CIs indicate computation errors."""
        cis = self._find_cis(calibration_data)

        for ci in cis:
            if ci["lo"] is not None and ci["hi"] is not None:
                width = ci["hi"] - ci["lo"]
                if width == 0:
                    warnings.warn(
                        f"{ci['path']} has zero-width CI [{ci['lo']}, {ci['hi']}]. "
                        f"This may indicate insufficient bootstrap variation."
                    )

    def test_point_estimate_within_ci(self, calibration_data):
        """Point estimate should be within its CI."""
        cis = self._find_cis(calibration_data)

        for ci in cis:
            if (
                ci["point"] is not None
                and ci["lo"] is not None
                and ci["hi"] is not None
            ):
                # Allow small tolerance for floating point
                assert ci["lo"] - 1e-6 <= ci["point"] <= ci["hi"] + 1e-6, (
                    f"CRITICAL: {ci['path']} point estimate {ci['point']} "
                    f"is outside its CI [{ci['lo']}, {ci['hi']}]"
                )


class TestDCAValidity:
    """
    Validate Decision Curve Analysis data.
    """

    def test_dca_has_reference_strategies(self, dca_data):
        """DCA must include 'Treat All' and 'Treat None' references."""
        data = dca_data.get("data", {})

        # Convert entire data structure to string for comprehensive search
        data_str = str(data).lower()

        # Check for reference strategies in various formats
        # Can appear as: treat_all, nb_treat_all, treatall, etc.
        has_treat_all = any(
            pattern in data_str
            for pattern in ["treat_all", "treatall", "nb_all", "nb_treat_all"]
        )
        has_treat_none = any(
            pattern in data_str
            for pattern in ["treat_none", "treatnone", "nb_none", "nb_treat_none"]
        )

        # Also check in configs if present
        configs = data.get("configs", [])
        for config in configs:
            config_str = str(config).lower()
            if "treat_all" in config_str or "nb_treat_all" in config_str:
                has_treat_all = True
            if "treat_none" in config_str or "nb_treat_none" in config_str:
                has_treat_none = True

        assert has_treat_all, (
            "CRITICAL: DCA missing 'Treat All' reference strategy. "
            "This is required for valid DCA interpretation."
        )
        assert has_treat_none, (
            "CRITICAL: DCA missing 'Treat None' reference strategy. "
            "This is required for valid DCA interpretation."
        )

    def test_dca_thresholds_are_valid(self, dca_data):
        """DCA thresholds should be in (0, 1) and ordered."""
        data = dca_data.get("data", {})

        # Find thresholds in various possible locations
        thresholds = data.get("thresholds", [])

        if not thresholds:
            # Check nested structures
            for key, value in data.items():
                if isinstance(value, dict) and "thresholds" in value:
                    thresholds = value["thresholds"]
                    break

        if not thresholds:
            warnings.warn("No thresholds found in DCA data structure")
            return

        # Validate threshold values
        for t in thresholds:
            assert 0 < t < 1, f"CRITICAL: DCA threshold {t} outside valid range (0, 1)"

        # Check ordering
        for i in range(len(thresholds) - 1):
            assert thresholds[i] < thresholds[i + 1], (
                f"CRITICAL: DCA thresholds not monotonically increasing: {thresholds}"
            )

    def test_net_benefit_bounds(self, dca_data):
        """Net benefit has theoretical bounds based on prevalence."""
        data = dca_data.get("data", {})
        prevalence = data.get("prevalence", 0.27)  # Default from our data

        # Theoretical bounds: [-prevalence/(1-prevalence), 1]
        max_nb = prevalence  # For threshold = 0
        min_nb = -1  # Conservative lower bound

        # Find net benefit values
        for key, value in data.items():
            if "net_benefit" in key.lower() or "nb_" in key.lower():
                if isinstance(value, list):
                    for nb in value:
                        if nb is not None:
                            assert min_nb <= nb <= max_nb + 0.01, (
                                f"CRITICAL: Net benefit {nb} outside theoretical bounds "
                                f"[{min_nb}, {max_nb}]"
                            )


class TestSampleSizeConsistency:
    """
    Validate sample size consistency across data structures.
    """

    def test_n_consistent_across_configs(self, calibration_data):
        """All configs from same experiment should have same n."""
        configs = calibration_data.get("data", {}).get("configs", [])

        if len(configs) < 2:
            return

        n_values = [c.get("n") for c in configs if "n" in c]

        if len(set(n_values)) > 1:
            # Different n values - could be legitimate for different test sets
            # but worth a warning
            warnings.warn(
                f"Different sample sizes across configs: {n_values}. "
                f"Verify this is intentional."
            )

    def test_events_less_than_n(self, calibration_data):
        """Number of events must be less than total n."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            n = config.get("n", 0)
            n_events = config.get("n_events", 0)

            if n > 0 and n_events > 0:
                assert n_events <= n, (
                    f"CRITICAL: {config.get('name', 'unknown')} has n_events={n_events} "
                    f"> n={n}. This is impossible and indicates data corruption."
                )
