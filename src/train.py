"""
Training Pipeline

End-to-end script that orchestrates data generation, preprocessing,
model training, evaluation, and artifact saving.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_transaction_data
from src.preprocessing import DataPreprocessor
from src.model import FraudDetectionModel


def main():
    """Run the complete training pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ─── Step 1: Generate Data ───────────────────────────────────
    print("=" * 60)
    print("STEP 1: Generating Synthetic Transaction Data")
    print("=" * 60)

    df = generate_transaction_data(n_samples=15000, fraud_ratio=0.03)
    data_path = os.path.join(data_dir, "transactions.csv")
    df.to_csv(data_path, index=False)

    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%")
    print(f"Saved to: {data_path}\n")

    # ─── Step 2: Preprocess ──────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)

    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, fit=True)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {dict(zip(*map(list, __import__('numpy').unique(y, return_counts=True))))}\n")

    # ─── Step 3: Split Data ──────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Splitting Data")
    print("=" * 60)

    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples\n")

    # ─── Step 4: Handle Class Imbalance ──────────────────────────
    print("=" * 60)
    print("STEP 4: Handling Class Imbalance (SMOTE)")
    print("=" * 60)

    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(X_train, y_train)
    print()

    # ─── Step 5: Train Model ─────────────────────────────────────
    print("=" * 60)
    print("STEP 5: Training Random Forest Classifier")
    print("=" * 60)

    model = FraudDetectionModel(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
    )
    model.train(X_train_balanced, y_train_balanced)
    print()

    # ─── Step 6: Evaluate ────────────────────────────────────────
    print("=" * 60)
    print("STEP 6: Evaluating Model")
    print("=" * 60)

    metrics = model.evaluate(X_test, y_test)

    # ─── Step 7: Feature Importance ──────────────────────────────
    print("=" * 60)
    print("STEP 7: Feature Importance Analysis")
    print("=" * 60)

    feature_importance = model.get_feature_importance(preprocessor.feature_columns)

    # ─── Step 8: Save Artifacts ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Saving Model & Preprocessor")
    print("=" * 60)

    model.save(model_dir)
    preprocessor.save(model_dir)

    print("\n✅ Training pipeline completed successfully!")
    print(f"\nKey Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
