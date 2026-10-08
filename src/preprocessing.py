"""
Data Preprocessing Module

Handles data cleaning, feature engineering, encoding, scaling, and
class imbalance handling using SMOTE (Synthetic Minority Oversampling Technique).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict
import joblib
import os


class DataPreprocessor:
    """
    Preprocesses raw transaction data for model training.
    
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Train/test splitting
    - Class imbalance correction via SMOTE
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = []
        self.categorical_columns = ["merchant_category", "transaction_location"]
        self.drop_columns = ["transaction_id", "timestamp"]
        self.target_column = "is_fraud"

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Args:
            df: Raw transaction DataFrame
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        df = df.copy()

        # Step 1: Drop non-feature columns
        df = self._drop_columns(df)

        # Step 2: Handle missing values
        df = self._handle_missing_values(df)

        # Step 3: Encode categorical features
        df = self._encode_categoricals(df, fit=fit)

        # Step 4: Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].values

        if fit:
            self.feature_columns = list(X.columns)

        # Step 5: Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def preprocess_single(self, data: dict) -> np.ndarray:
        """
        Preprocess a single transaction for real-time prediction.

        Args:
            data: Dictionary with transaction features

        Returns:
            Scaled feature array ready for prediction
        """
        df = pd.DataFrame([data])

        # Drop non-feature columns if present
        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Handle missing values
        df = self._handle_missing_values(df)

        # Encode categoricals
        df = self._encode_categoricals(df, fit=False)

        # Remove target if present
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        # Ensure column order matches training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        return self.scaler.transform(df)

    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.

        Args:
            X: Feature matrix
            y: Target labels
            random_state: Random seed

        Returns:
            Resampled (X, y) with balanced classes
        """
        smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print(f"Before SMOTE: {np.bincount(y.astype(int))}")
        print(f"After SMOTE:  {np.bincount(y_resampled.astype(int))}")

        return X_resampled, y_resampled

    def split_data(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets with stratification."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def save(self, directory: str):
        """Save preprocessor state (scaler + encoders) to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(directory, "scaler.joblib"))
        joblib.dump(self.label_encoders, os.path.join(directory, "label_encoders.joblib"))
        joblib.dump(self.feature_columns, os.path.join(directory, "feature_columns.joblib"))
        print(f"Preprocessor saved to {directory}")

    def load(self, directory: str):
        """Load preprocessor state from disk."""
        self.scaler = joblib.load(os.path.join(directory, "scaler.joblib"))
        self.label_encoders = joblib.load(os.path.join(directory, "label_encoders.joblib"))
        self.feature_columns = joblib.load(os.path.join(directory, "feature_columns.joblib"))
        print(f"Preprocessor loaded from {directory}")

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-feature columns."""
        existing_drop = [col for col in self.drop_columns if col in df.columns]
        return df.drop(columns=existing_drop)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values: median for numeric, mode for categorical."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen categories
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    df[col] = 0
        return df
