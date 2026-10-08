"""
Unit Tests for the Fraud Detection System
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_transaction_data
from src.preprocessing import DataPreprocessor
from src.model import FraudDetectionModel


class TestDataGenerator:
    """Tests for the data generator module."""

    def test_generates_correct_number_of_samples(self):
        df = generate_transaction_data(n_samples=1000)
        assert len(df) == 1000

    def test_fraud_ratio_is_approximate(self):
        df = generate_transaction_data(n_samples=10000, fraud_ratio=0.05)
        actual_ratio = df["is_fraud"].mean()
        assert 0.04 <= actual_ratio <= 0.06

    def test_required_columns_present(self):
        df = generate_transaction_data(n_samples=100)
        required = [
            "transaction_id", "timestamp", "transaction_amount",
            "transaction_hour", "merchant_category", "is_fraud"
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_transaction_amounts_positive(self):
        df = generate_transaction_data(n_samples=1000)
        assert (df["transaction_amount"] > 0).all()

    def test_reproducibility_with_seed(self):
        df1 = generate_transaction_data(n_samples=100, seed=123)
        df2 = generate_transaction_data(n_samples=100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocessor:
    """Tests for the preprocessing module."""

    @pytest.fixture
    def sample_data(self):
        return generate_transaction_data(n_samples=500, fraud_ratio=0.1)

    def test_preprocess_returns_arrays(self, sample_data):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, fit=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_preprocess_correct_shape(self, sample_data):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, fit=True)
        assert X.shape[0] == len(sample_data)
        assert len(y) == len(sample_data)

    def test_no_missing_values_after_preprocessing(self, sample_data):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, fit=True)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_smote_increases_minority_class(self, sample_data):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, fit=True)
        X_resampled, y_resampled = preprocessor.handle_imbalance(X, y)
        assert y_resampled.sum() > y.sum()


class TestModel:
    """Tests for the model module."""

    @pytest.fixture
    def trained_model(self):
        df = generate_transaction_data(n_samples=2000, fraud_ratio=0.1)
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(df, fit=True)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        model = FraudDetectionModel(n_estimators=50, max_depth=10)
        model.train(X_train, y_train)

        return model, X_test, y_test, preprocessor

    def test_model_trains_successfully(self, trained_model):
        model, _, _, _ = trained_model
        assert model.is_trained

    def test_predictions_are_binary(self, trained_model):
        model, X_test, _, _ = trained_model
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1})

    def test_probabilities_in_range(self, trained_model):
        model, X_test, _, _ = trained_model
        probas = model.predict_proba(X_test)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_evaluation_returns_metrics(self, trained_model):
        model, X_test, y_test, _ = trained_model
        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

    def test_feature_importance_length(self, trained_model):
        model, X_test, _, preprocessor = trained_model
        importance = model.get_feature_importance(preprocessor.feature_columns)
        assert len(importance) == len(preprocessor.feature_columns)

    def test_model_save_and_load(self, trained_model, tmp_path):
        model, X_test, _, preprocessor = trained_model
        model.save(str(tmp_path))
        preprocessor.save(str(tmp_path))

        new_model = FraudDetectionModel()
        new_model.load(str(tmp_path))

        original_preds = model.predict(X_test)
        loaded_preds = new_model.predict(X_test)
        np.testing.assert_array_equal(original_preds, loaded_preds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
