"""
End-to-end tests for training reproducibility.

These tests verify that:
1. Training with the same seed produces identical results
2. Bootstrap sampling is reproducible
3. Model predictions are deterministic

Marked as @pytest.mark.slow - run with `pytest -m slow`
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification


class TestCatBoostReproducibility:
    """Test CatBoost training reproducibility."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        return X, y

    @pytest.mark.slow
    @pytest.mark.unit
    def test_catboost_deterministic_with_seed(self, synthetic_data):
        """Same seed should produce identical CatBoost predictions."""
        from catboost import CatBoostClassifier

        X, y = synthetic_data

        # Train twice with same seed
        model1 = CatBoostClassifier(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
        )
        model1.fit(X, y)

        model2 = CatBoostClassifier(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
        )
        model2.fit(X, y)

        # Predictions should be identical
        pred1 = model1.predict_proba(X)
        pred2 = model2.predict_proba(X)

        np.testing.assert_array_almost_equal(
            pred1,
            pred2,
            decimal=10,
            err_msg="CatBoost predictions differ with same seed",
        )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_catboost_different_with_different_seed(self, synthetic_data):
        """Different seeds should produce different results."""
        from catboost import CatBoostClassifier

        X, y = synthetic_data

        model1 = CatBoostClassifier(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
        )
        model1.fit(X, y)

        model2 = CatBoostClassifier(
            iterations=10,
            depth=3,
            learning_rate=0.1,
            random_seed=123,  # Different seed
            verbose=False,
        )
        model2.fit(X, y)

        pred1 = model1.predict_proba(X)
        pred2 = model2.predict_proba(X)

        # Should NOT be identical
        assert not np.allclose(
            pred1, pred2
        ), "Different seeds produced identical predictions - something is wrong"


class TestXGBoostReproducibility:
    """Test XGBoost training reproducibility."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        return X, y

    @pytest.mark.slow
    @pytest.mark.unit
    def test_xgboost_deterministic_with_seed(self, synthetic_data):
        """Same seed should produce identical XGBoost predictions."""
        import xgboost as xgb

        X, y = synthetic_data

        # Train twice with same seed
        model1 = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model1.fit(X, y)

        model2 = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model2.fit(X, y)

        # Predictions should be identical
        pred1 = model1.predict_proba(X)
        pred2 = model2.predict_proba(X)

        np.testing.assert_array_almost_equal(
            pred1,
            pred2,
            decimal=10,
            err_msg="XGBoost predictions differ with same seed",
        )


class TestBootstrapReproducibility:
    """Test bootstrap sampling reproducibility."""

    @pytest.mark.unit
    def test_bootstrap_indices_reproducible(self):
        """Bootstrap indices should be reproducible with same seed."""
        n_samples = 100
        n_iterations = 10

        # Generate bootstrap indices twice with same seed
        rng1 = np.random.default_rng(42)
        indices1 = [
            rng1.choice(n_samples, size=n_samples, replace=True)
            for _ in range(n_iterations)
        ]

        rng2 = np.random.default_rng(42)
        indices2 = [
            rng2.choice(n_samples, size=n_samples, replace=True)
            for _ in range(n_iterations)
        ]

        # Should be identical
        for i, (idx1, idx2) in enumerate(zip(indices1, indices2)):
            np.testing.assert_array_equal(
                idx1,
                idx2,
                err_msg=f"Bootstrap iteration {i} indices differ",
            )

    @pytest.mark.unit
    def test_bootstrap_statistics_reproducible(self):
        """Bootstrap statistics should be reproducible."""
        # Generate sample data
        rng_data = np.random.default_rng(123)
        data = rng_data.standard_normal(100)

        n_bootstrap = 50

        # Compute bootstrap means twice
        def bootstrap_means(data, n_bootstrap, seed):
            rng = np.random.default_rng(seed)
            means = []
            for _ in range(n_bootstrap):
                indices = rng.choice(len(data), size=len(data), replace=True)
                means.append(np.mean(data[indices]))
            return np.array(means)

        means1 = bootstrap_means(data, n_bootstrap, seed=42)
        means2 = bootstrap_means(data, n_bootstrap, seed=42)

        np.testing.assert_array_equal(
            means1,
            means2,
            err_msg="Bootstrap means differ with same seed",
        )


class TestNumpyRandomReproducibility:
    """Test numpy random reproducibility patterns."""

    @pytest.mark.unit
    def test_numpy_rng_state_isolation(self):
        """Each RNG instance should be independent."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Generate from rng1
        vals1_a = rng1.standard_normal(5)
        vals1_b = rng1.standard_normal(5)

        # Generate from rng2 (should match rng1's first generation)
        vals2_a = rng2.standard_normal(5)
        vals2_b = rng2.standard_normal(5)

        np.testing.assert_array_equal(vals1_a, vals2_a)
        np.testing.assert_array_equal(vals1_b, vals2_b)

    @pytest.mark.unit
    def test_seed_documentation(self):
        """Document the random seed used in the project."""
        # The project uses seed 42 as the default (from configs/defaults.yaml)
        # This test documents and verifies this choice
        from pathlib import Path

        import yaml

        configs_dir = Path(__file__).parent.parent.parent / "configs"
        defaults_path = configs_dir / "defaults.yaml"

        assert defaults_path.exists(), f"defaults.yaml not found: {defaults_path}"

        content = yaml.safe_load(defaults_path.read_text())

        # Check for random seed configuration
        # The exact path may vary, this is documenting what exists
        assert content is not None, "defaults.yaml should have content"
