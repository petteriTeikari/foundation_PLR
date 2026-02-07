"""Integration tests for classification functionality.

Tests classification models including XGBoost and CatBoost
for glaucoma vs control classification.
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class TestXGBoostClassification:
    """Tests for XGBoost classifier."""

    @pytest.fixture
    def xgboost_available(self):
        """Check if XGBoost is available."""
        try:
            import xgboost as xgb  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.fixture
    def binary_classification_data(self):
        """Create synthetic binary classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # Class 0: centered at -1
        X_0 = np.random.randn(n_samples // 2, n_features) - 1
        y_0 = np.zeros(n_samples // 2)

        # Class 1: centered at +1
        X_1 = np.random.randn(n_samples // 2, n_features) + 1
        y_1 = np.ones(n_samples // 2)

        X = np.vstack([X_0, X_1])
        y = np.concatenate([y_0, y_1])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.mark.integration
    def test_xgboost_trains(self, xgboost_available, binary_classification_data):
        """Test that XGBoost classifier trains without errors."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")

        import xgboost as xgb

        X_train, X_test, y_train, y_test = binary_classification_data

        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train)

        assert model is not None

    @pytest.mark.integration
    def test_xgboost_predicts(self, xgboost_available, binary_classification_data):
        """Test that XGBoost makes predictions."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")

        import xgboost as xgb

        X_train, X_test, y_train, y_test = binary_classification_data

        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})

    @pytest.mark.integration
    def test_xgboost_accuracy_above_random(
        self, xgboost_available, binary_classification_data
    ):
        """Test that XGBoost accuracy is above random chance (0.5)."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")

        import xgboost as xgb

        X_train, X_test, y_train, y_test = binary_classification_data

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.5, f"Accuracy {accuracy} should be above random (0.5)"

    @pytest.mark.integration
    def test_xgboost_probability_predictions(
        self, xgboost_available, binary_classification_data
    ):
        """Test that XGBoost provides probability predictions."""
        if not xgboost_available:
            pytest.skip("XGBoost not installed")

        import xgboost as xgb

        X_train, X_test, y_train, y_test = binary_classification_data

        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)

        assert probabilities.shape == (len(y_test), 2)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1), np.ones(len(y_test)), decimal=5
        )


class TestCatBoostClassification:
    """Tests for CatBoost classifier."""

    @pytest.fixture
    def catboost_available(self):
        """Check if CatBoost is available."""
        try:
            from catboost import CatBoostClassifier  # noqa: F401

            return True
        except ImportError:
            return False

    @pytest.fixture
    def binary_classification_data(self):
        """Create synthetic binary classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        X_0 = np.random.randn(n_samples // 2, n_features) - 1
        y_0 = np.zeros(n_samples // 2)

        X_1 = np.random.randn(n_samples // 2, n_features) + 1
        y_1 = np.ones(n_samples // 2)

        X = np.vstack([X_0, X_1])
        y = np.concatenate([y_0, y_1])

        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.mark.integration
    def test_catboost_trains(self, catboost_available, binary_classification_data):
        """Test that CatBoost classifier trains without errors."""
        if not catboost_available:
            pytest.skip("CatBoost not installed")

        from catboost import CatBoostClassifier

        X_train, X_test, y_train, y_test = binary_classification_data

        model = CatBoostClassifier(
            iterations=50,
            depth=3,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
        )

        model.fit(X_train, y_train)

        assert model is not None

    @pytest.mark.integration
    def test_catboost_accuracy_above_random(
        self, catboost_available, binary_classification_data
    ):
        """Test that CatBoost accuracy is above random chance (0.5)."""
        if not catboost_available:
            pytest.skip("CatBoost not installed")

        from catboost import CatBoostClassifier

        X_train, X_test, y_train, y_test = binary_classification_data

        model = CatBoostClassifier(
            iterations=100,
            depth=4,
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.5, f"Accuracy {accuracy} should be above random (0.5)"

    @pytest.mark.integration
    def test_catboost_roc_auc(self, catboost_available, binary_classification_data):
        """Test that CatBoost achieves reasonable ROC-AUC."""
        if not catboost_available:
            pytest.skip("CatBoost not installed")

        from catboost import CatBoostClassifier

        X_train, X_test, y_train, y_test = binary_classification_data

        model = CatBoostClassifier(
            iterations=100,
            depth=4,
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)

        assert roc_auc > 0.6, f"ROC-AUC {roc_auc} should be above 0.6"


class TestClassificationMetrics:
    """Tests for classification metrics computation."""

    @pytest.fixture
    def mock_predictions(self):
        """Create mock predictions and labels."""
        np.random.seed(42)

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.2, 0.8, 0.9, 0.7, 0.4, 0.85])

        return y_true, y_pred, y_proba

    @pytest.mark.integration
    def test_accuracy_computation(self, mock_predictions):
        """Test accuracy metric computation."""
        y_true, y_pred, _ = mock_predictions

        accuracy = accuracy_score(y_true, y_pred)

        # 8/10 correct
        assert accuracy == 0.8

    @pytest.mark.integration
    def test_roc_auc_computation(self, mock_predictions):
        """Test ROC-AUC metric computation."""
        y_true, _, y_proba = mock_predictions

        roc_auc = roc_auc_score(y_true, y_proba)

        assert 0 <= roc_auc <= 1
        assert roc_auc > 0.5  # Better than random

    @pytest.mark.integration
    def test_confusion_matrix_structure(self, mock_predictions):
        """Test confusion matrix structure."""
        from sklearn.metrics import confusion_matrix

        y_true, y_pred, _ = mock_predictions

        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)  # Binary classification
        assert cm.sum() == len(y_true)  # All samples accounted for
