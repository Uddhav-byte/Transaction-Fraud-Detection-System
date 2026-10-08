"""
Model Training Module

Trains a Random Forest classifier for fraud detection with hyperparameter tuning,
evaluation metrics (precision, recall, F1, AUC-ROC), and model persistence.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import cross_val_score
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any


class FraudDetectionModel:
    """
    Random Forest-based fraud detection model.

    Optimized for high recall (catching fraud) while maintaining
    acceptable precision (minimizing false positives).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of training metrics
        """
        print("Training Random Forest classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Training set metrics
        y_pred_train = self.model.predict(X_train)
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "train_precision": float(precision_score(y_train, y_pred_train)),
            "train_recall": float(recall_score(y_train, y_pred_train)),
            "train_f1": float(f1_score(y_train, y_pred_train)),
        }

        print(f"Training Accuracy:  {metrics['train_accuracy']:.4f}")
        print(f"Training Precision: {metrics['train_precision']:.4f}")
        print(f"Training Recall:    {metrics['train_recall']:.4f}")
        print(f"Training F1-Score:  {metrics['train_f1']:.4f}")

        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set with comprehensive metrics.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Core metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
        }

        # Classification report
        report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])

        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
        print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
        print(f"\n{report}")

        self.training_metrics = metrics
        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform k-fold cross validation.

        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds

        Returns:
            Dictionary with mean and std of CV scores
        """
        print(f"\nRunning {cv}-fold cross-validation...")

        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1")
        results = {
            "cv_mean_f1": float(cv_scores.mean()),
            "cv_std_f1": float(cv_scores.std()),
            "cv_scores": [float(s) for s in cv_scores],
        }

        print(f"CV F1 Scores: {cv_scores}")
        print(f"Mean F1: {results['cv_mean_f1']:.4f} ± {results['cv_std_f1']:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud labels."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance rankings."""
        if not self.is_trained:
            raise ValueError("Model must be trained first.")

        importance = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        print("\nFeature Importance:")
        print("-" * 40)
        for _, row in df.iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

        return df

    def save(self, directory: str):
        """Save trained model and metrics to disk."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, "fraud_detection_model.joblib")
        joblib.dump(self.model, model_path)

        # Save metrics
        metrics_path = os.path.join(directory, "training_metrics.json")
        metrics_to_save = {
            **self.training_metrics,
            "saved_at": datetime.now().isoformat(),
            "model_type": "RandomForestClassifier",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Metrics saved to {metrics_path}")

    def load(self, directory: str):
        """Load trained model from disk."""
        model_path = os.path.join(directory, "fraud_detection_model.joblib")
        self.model = joblib.load(model_path)
        self.is_trained = True

        metrics_path = os.path.join(directory, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.training_metrics = json.load(f)

        print(f"Model loaded from {model_path}")
