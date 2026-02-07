"""
P0 CRITICAL: Data Provenance Validation Tests

These tests would have caught CRITICAL-FAILURE-001 where synthetic data
was used instead of real experimental predictions.

ZERO TOLERANCE: Any failure here indicates potential scientific fraud.
"""

import hashlib
import warnings

import numpy as np
import pytest


class TestCrossModelUniqueness:
    """
    CRITICAL: Detect when multiple models have identical/near-identical predictions.

    This catches the exact failure mode from CRITICAL-FAILURE-001 where all 4 models
    had correlation = 1.0 due to shared random seed generating identical synthetic data.
    """

    CORRELATION_THRESHOLD = 0.99  # Models should NOT be this similar

    def test_calibration_curves_are_distinct(self, calibration_data):
        """Calibration curves for different models must be distinguishable."""
        configs = calibration_data.get("data", {}).get("configs", [])

        if len(configs) < 2:
            pytest.skip("Need at least 2 configs to compare")

        for i, c1 in enumerate(configs):
            for c2 in configs[i + 1 :]:
                # Compare observed calibration values
                obs1 = [
                    v for v in c1.get("curve", {}).get("observed", []) if v is not None
                ]
                obs2 = [
                    v for v in c2.get("curve", {}).get("observed", []) if v is not None
                ]

                if len(obs1) >= 3 and len(obs2) >= 3:
                    # Pad/truncate to same length for comparison
                    min_len = min(len(obs1), len(obs2))
                    obs1, obs2 = obs1[:min_len], obs2[:min_len]

                    if np.std(obs1) > 0 and np.std(obs2) > 0:
                        corr = np.corrcoef(obs1, obs2)[0, 1]
                        assert corr < self.CORRELATION_THRESHOLD, (
                            f"CRITICAL: Models '{c1['name']}' and '{c2['name']}' have correlation {corr:.4f} >= {self.CORRELATION_THRESHOLD}. Possible synthetic data with shared seed!"
                        )

    def test_calibration_metrics_are_distinct(self, calibration_data):
        """Key metrics (slope, Brier, etc.) should differ between models."""
        configs = calibration_data.get("data", {}).get("configs", [])

        if len(configs) < 2:
            pytest.skip("Need at least 2 configs to compare")

        # Collect metric vectors for each config
        metric_keys = ["calibration_slope", "brier", "ipa", "o_e_ratio"]
        metric_vectors = []

        for config in configs:
            vec = [config.get(k, 0) for k in metric_keys if k in config]
            if vec:
                metric_vectors.append((config["name"], vec))

        # Check that no two configs have identical metric vectors
        for i, (name1, vec1) in enumerate(metric_vectors):
            for name2, vec2 in metric_vectors[i + 1 :]:
                if len(vec1) == len(vec2) and len(vec1) > 0:
                    # Allow small tolerance for floating point
                    if np.allclose(vec1, vec2, rtol=1e-6):
                        pytest.fail(
                            f"CRITICAL: Models '{name1}' and '{name2}' have IDENTICAL metrics. "
                            f"Vectors: {vec1} vs {vec2}. Likely synthetic data bug!"
                        )


class TestSyntheticDataDetection:
    """
    CRITICAL: Detect keywords and patterns indicating synthetic/fake data.
    """

    SYNTHETIC_KEYWORDS = [
        "synthetic",
        "simulated",
        # "template" removed: legitimate decomposition method name (template matching)
        "fake",
        "mock",
        "dummy",
        "placeholder",
        # "generated" removed: appears in legitimate metadata (e.g., "generator" field)
        "artificial",
    ]

    def test_metadata_not_synthetic(self, calibration_data):
        """Metadata must not contain synthetic data keywords."""
        metadata = calibration_data.get("metadata", {})
        note = metadata.get("note", "").lower()

        for keyword in self.SYNTHETIC_KEYWORDS:
            if keyword in note:
                pytest.fail(
                    f"CRITICAL: Found '{keyword}' in metadata note: {metadata.get('note')}. "
                    f"This indicates synthetic data being used for scientific figures!"
                )

    def test_data_source_is_real(self, calibration_data):
        """If data_source field exists, it must indicate real data."""
        metadata = calibration_data.get("metadata", {})

        if "data_source" in metadata:
            source = metadata["data_source"]
            # data_source can be a dict (e.g., {"db_path": ..., "db_hash": ...})
            # or a string. Both indicate real data provenance.
            if isinstance(source, dict):
                # Dict with db_path/db_hash = real data from extraction pipeline
                return
            source_str = str(source).lower()
            valid_sources = ["real", "experimental", "mlflow", "actual"]
            assert any(v in source_str for v in valid_sources), (
                f"CRITICAL: data_source='{metadata['data_source']}' is not recognized as real data. "
                f"Expected one of: {valid_sources}"
            )

    def test_no_synthetic_keywords_in_any_json(self, all_json_files, project_root):
        """Scan all JSON files for synthetic data indicators."""
        for json_file in all_json_files:
            with open(json_file) as f:
                content = f.read().lower()

            for keyword in self.SYNTHETIC_KEYWORDS:
                if keyword in content:
                    # Find context around the keyword
                    idx = content.find(keyword)
                    context = content[max(0, idx - 50) : idx + 50]
                    pytest.fail(
                        f"CRITICAL: Found '{keyword}' in {json_file.name}. "
                        f"Context: '...{context}...'"
                    )


class TestDataLineage:
    """
    Validate data provenance chain - where did this data come from?
    """

    def test_has_creation_timestamp(self, calibration_data):
        """Data must have creation timestamp for audit trail."""
        metadata = calibration_data.get("metadata", {})
        assert "created" in metadata, (
            "Missing 'created' timestamp in metadata. "
            "Cannot verify data freshness without timestamp."
        )

    def test_has_generator_info(self, calibration_data):
        """Data must identify the generating script."""
        metadata = calibration_data.get("metadata", {})
        assert "generator" in metadata, (
            "Missing 'generator' in metadata. "
            "Cannot trace data provenance without knowing the source script."
        )

    def test_source_file_hash_if_present(self, calibration_data, project_root):
        """If source_file_hash is present, verify it matches."""
        metadata = calibration_data.get("metadata", {})

        if "source_file_hash" not in metadata:
            warnings.warn(
                "No source_file_hash in metadata - cannot verify data integrity. "
                "Consider adding hash for production data."
            )
            return

        source_path = project_root / metadata.get("source_file", "")
        if not source_path.exists():
            warnings.warn(f"Source file not found: {source_path}")
            return

        with open(source_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        expected_hash = metadata["source_file_hash"]
        assert actual_hash == expected_hash, (
            f"CRITICAL: Source file hash mismatch! "
            f"Expected {expected_hash}, got {actual_hash}. "
            f"Data may have been modified or regenerated from wrong source."
        )


class TestDataCompleteness:
    """
    Validate that data is complete and not truncated.
    """

    def test_sufficient_sample_size(self, calibration_data):
        """Each config must have sufficient samples for statistical validity."""
        MIN_N = 20  # Minimum for meaningful calibration

        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            n = config.get("n", 0)
            assert n >= MIN_N, (
                f"Config '{config.get('name', 'unknown')}' has only n={n} samples. "
                f"Minimum {MIN_N} required for valid calibration analysis."
            )

    def test_sufficient_events(self, calibration_data):
        """Each config must have sufficient events for calibration."""
        MIN_EVENTS = 5  # Absolute minimum for binary calibration

        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            n_events = config.get("n_events", 0)
            assert n_events >= MIN_EVENTS, (
                f"Config '{config.get('name', 'unknown')}' has only {n_events} events. "
                f"Minimum {MIN_EVENTS} required for calibration curve."
            )

    def test_calibration_curve_has_data(self, calibration_data):
        """Calibration curves must have non-trivial data points."""
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            curve = config.get("curve", {})
            observed = curve.get("observed", [])
            counts = curve.get("counts", [])

            # Count non-null observed values
            non_null_obs = [v for v in observed if v is not None]
            assert len(non_null_obs) >= 2, (
                f"Config '{config.get('name', 'unknown')}' has only {len(non_null_obs)} "
                f"non-null calibration points. Need at least 2 for a curve."
            )

            # At least some bins should have data
            bins_with_data = sum(1 for c in counts if c > 0)
            assert bins_with_data >= 2, (
                f"Config '{config.get('name', 'unknown')}' has only {bins_with_data} "
                f"bins with data. Need at least 2 for meaningful calibration."
            )


class TestEntropyAnalysis:
    """
    Detect suspiciously uniform/random data patterns.
    """

    def test_predictions_have_natural_distribution(self, calibration_data):
        """
        Real predictions should not have uniform or suspiciously regular distributions.
        Synthetic data often shows telltale patterns.
        """
        configs = calibration_data.get("data", {}).get("configs", [])

        for config in configs:
            curve = config.get("curve", {})
            counts = curve.get("counts", [])

            if not counts or sum(counts) == 0:
                continue

            # Calculate entropy of bin distribution
            total = sum(counts)
            probs = [c / total for c in counts if c > 0]

            if len(probs) < 2:
                continue

            # Shannon entropy
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(probs))  # Uniform distribution

            # Suspiciously uniform distribution (entropy near max)
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
                # Real data is rarely perfectly uniform
                if normalized_entropy > 0.98:
                    warnings.warn(
                        f"Config '{config.get('name', 'unknown')}' has suspiciously "
                        f"uniform bin distribution (entropy ratio: {normalized_entropy:.3f}). "
                        f"This could indicate synthetic data."
                    )


class TestInstabilityFigure3Groups:
    """
    TDD tests for the 3-group model instability figure.

    Required groups: Ground Truth | Best Ensemble | Traditional
    Reference: Riley RD et al. (2023) BMC Medicine 21:502
    """

    REQUIRED_CONFIGS = ["ground_truth", "best_ensemble", "traditional"]
    MIN_PATIENTS = 20
    MIN_BOOTSTRAPS = 100
    # High threshold: 0.9999 catches identical data (synthetic bug) while allowing
    # legitimately similar predictions (best_ensemble ≈ ground_truth is EXPECTED)
    CORRELATION_THRESHOLD = 0.9999

    @pytest.fixture
    def pminternal_data(self, project_root):
        """Load pminternal bootstrap predictions data."""
        import json

        json_path = (
            project_root / "data" / "r_data" / "pminternal_bootstrap_predictions.json"
        )
        assert json_path.exists(), (
            f"pminternal data not found: {json_path}. Run: make analyze"
        )

        with open(json_path) as f:
            return json.load(f)

    def test_has_all_required_configs(self, pminternal_data):
        """Data must have all 3 required configs for comparison."""
        configs = pminternal_data.get("configs", {})

        for required in self.REQUIRED_CONFIGS:
            assert required in configs, (
                f"CRITICAL: Missing required config '{required}' for 3-group instability figure. "
                f"Available: {list(configs.keys())}"
            )

    def test_sufficient_bootstrap_samples(self, pminternal_data):
        """Each config must have sufficient bootstrap samples."""
        configs = pminternal_data.get("configs", {})

        for config_id in self.REQUIRED_CONFIGS:
            if config_id not in configs:
                continue

            cfg = configs[config_id]
            n_bootstrap = cfg.get("n_bootstrap", 0)

            assert n_bootstrap >= self.MIN_BOOTSTRAPS, (
                f"Config '{config_id}' has only {n_bootstrap} bootstrap samples. "
                f"Minimum {self.MIN_BOOTSTRAPS} required for reliable instability metrics."
            )

    def test_sufficient_patients(self, pminternal_data):
        """Each config must have sufficient patients."""
        configs = pminternal_data.get("configs", {})

        for config_id in self.REQUIRED_CONFIGS:
            if config_id not in configs:
                continue

            cfg = configs[config_id]
            n_patients = cfg.get("n_patients", 0)

            assert n_patients >= self.MIN_PATIENTS, (
                f"Config '{config_id}' has only {n_patients} patients. "
                f"Minimum {self.MIN_PATIENTS} required for meaningful instability analysis."
            )

    def test_configs_are_distinct(self, pminternal_data):
        """Different configs must have distinct predictions (not synthetic).

        Note: Ground truth and best_ensemble may have high correlation (0.99+)
        because the ensemble is DESIGNED to approximate ground truth. This is
        expected and correct. We fail only on perfect correlation (0.9999+)
        which indicates synthetic data with shared random seed.
        """
        configs = pminternal_data.get("configs", {})

        available = [c for c in self.REQUIRED_CONFIGS if c in configs]
        if len(available) < 2:
            pytest.skip("Need at least 2 configs to compare")

        correlations = {}
        for i, cfg1_id in enumerate(available):
            for cfg2_id in available[i + 1 :]:
                pred1 = configs[cfg1_id].get("y_prob_original", [])
                pred2 = configs[cfg2_id].get("y_prob_original", [])

                if len(pred1) >= 5 and len(pred2) >= 5:
                    # Same length check
                    min_len = min(len(pred1), len(pred2))
                    p1, p2 = pred1[:min_len], pred2[:min_len]

                    if np.std(p1) > 0 and np.std(p2) > 0:
                        corr = np.corrcoef(p1, p2)[0, 1]
                        correlations[f"{cfg1_id} vs {cfg2_id}"] = corr

                        # Fail on near-perfect identity (synthetic data bug)
                        assert corr < self.CORRELATION_THRESHOLD, (
                            f"CRITICAL: '{cfg1_id}' and '{cfg2_id}' have correlation "
                            f"{corr:.6f} >= {self.CORRELATION_THRESHOLD}. "
                            f"This indicates identical/synthetic data!"
                        )

        # Log correlations for scientific insight
        if correlations:
            import sys

            print(
                "\n  Prediction correlations (expected: GT≈Ens, Traditional lower):",
                file=sys.stderr,
            )
            for pair, corr in sorted(correlations.items(), key=lambda x: -x[1]):
                print(f"    {pair}: {corr:.4f}", file=sys.stderr)

    def test_has_required_fields(self, pminternal_data):
        """Each config must have all required fields for plotting."""
        required_fields = [
            "y_true",
            "y_prob_original",
            "y_prob_bootstrap",
            "n_patients",
            "n_bootstrap",
        ]
        configs = pminternal_data.get("configs", {})

        for config_id in self.REQUIRED_CONFIGS:
            if config_id not in configs:
                continue

            cfg = configs[config_id]
            for field in required_fields:
                assert field in cfg, (
                    f"Config '{config_id}' missing required field '{field}'. "
                    f"Available fields: {list(cfg.keys())}"
                )

    def test_bootstrap_array_dimensions(self, pminternal_data):
        """Bootstrap predictions array must have correct dimensions."""
        configs = pminternal_data.get("configs", {})

        for config_id in self.REQUIRED_CONFIGS:
            if config_id not in configs:
                continue

            cfg = configs[config_id]
            n_patients = cfg.get("n_patients", 0)
            n_bootstrap = cfg.get("n_bootstrap", 0)
            bootstrap_preds = cfg.get("y_prob_bootstrap", [])

            # Should be list of lists: n_bootstrap x n_patients
            assert len(bootstrap_preds) == n_bootstrap, (
                f"Config '{config_id}': bootstrap array has {len(bootstrap_preds)} rows, "
                f"expected {n_bootstrap}"
            )

            if n_bootstrap > 0:
                assert len(bootstrap_preds[0]) == n_patients, (
                    f"Config '{config_id}': bootstrap array has {len(bootstrap_preds[0])} columns, "
                    f"expected {n_patients}"
                )
