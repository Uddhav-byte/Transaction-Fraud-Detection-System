"""
Synthetic Transaction Data Generator

Generates realistic synthetic financial transaction data for training
the fraud detection model. Creates a dataset with imbalanced classes
(~2% fraud rate) mimicking real-world scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_transaction_data(n_samples: int = 10000, fraud_ratio: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic patterns.

    Args:
        n_samples: Total number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions (default 2%)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with transaction features and fraud labels
    """
    np.random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Legitimate Transactions ---
    legit_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_legit)
    legit_amounts = np.clip(legit_amounts, 1, 5000)

    legit_hours = np.random.choice(range(6, 23), size=n_legit, p=_daytime_distribution())

    legit_categories = np.random.choice(
        ["grocery", "gas_station", "restaurant", "online_shopping", "entertainment", "utilities", "healthcare"],
        size=n_legit,
        p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.10, 0.05]
    )

    legit_locations = np.random.choice(
        ["domestic", "international"],
        size=n_legit,
        p=[0.92, 0.08]
    )

    # --- Fraudulent Transactions ---
    fraud_amounts = np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud)
    fraud_amounts = np.clip(fraud_amounts, 50, 25000)

    fraud_hours = np.random.choice(range(0, 24), size=n_fraud, p=_nighttime_distribution())

    fraud_categories = np.random.choice(
        ["online_shopping", "electronics", "jewelry", "wire_transfer", "cash_advance", "entertainment", "grocery"],
        size=n_fraud,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05]
    )

    fraud_locations = np.random.choice(
        ["domestic", "international"],
        size=n_fraud,
        p=[0.40, 0.60]
    )

    # Build DataFrames
    base_date = datetime(2026, 1, 1)

    legit_df = pd.DataFrame({
        "transaction_amount": legit_amounts,
        "transaction_hour": legit_hours,
        "merchant_category": legit_categories,
        "transaction_location": legit_locations,
        "is_weekend": np.random.choice([0, 1], size=n_legit, p=[0.72, 0.28]),
        "customer_age": np.random.randint(18, 80, size=n_legit),
        "account_age_days": np.random.randint(30, 3650, size=n_legit),
        "num_transactions_last_24h": np.random.poisson(lam=3, size=n_legit),
        "avg_transaction_amount_30d": np.random.lognormal(mean=3.5, sigma=0.8, size=n_legit),
        "distance_from_home": np.random.exponential(scale=15, size=n_legit),
        "is_fraud": 0
    })

    fraud_df = pd.DataFrame({
        "transaction_amount": fraud_amounts,
        "transaction_hour": fraud_hours,
        "merchant_category": fraud_categories,
        "transaction_location": fraud_locations,
        "is_weekend": np.random.choice([0, 1], size=n_fraud, p=[0.55, 0.45]),
        "customer_age": np.random.randint(18, 80, size=n_fraud),
        "account_age_days": np.random.randint(1, 365, size=n_fraud),
        "num_transactions_last_24h": np.random.poisson(lam=8, size=n_fraud),
        "avg_transaction_amount_30d": np.random.lognormal(mean=3.0, sigma=0.5, size=n_fraud),
        "distance_from_home": np.random.exponential(scale=150, size=n_fraud),
        "is_fraud": 1
    })

    # Combine and shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add transaction ID and timestamp
    df.insert(0, "transaction_id", [f"TXN_{i:06d}" for i in range(len(df))])
    df.insert(1, "timestamp", [
        base_date + timedelta(days=np.random.randint(0, 270), hours=int(h), minutes=np.random.randint(0, 60))
        for h in df["transaction_hour"]
    ])

    # Engineer additional features
    df["amount_to_avg_ratio"] = df["transaction_amount"] / (df["avg_transaction_amount_30d"] + 1)
    df["is_high_risk_category"] = df["merchant_category"].isin(
        ["wire_transfer", "cash_advance", "jewelry", "electronics"]
    ).astype(int)
    df["is_international"] = (df["transaction_location"] == "international").astype(int)

    return df


def _daytime_distribution() -> list:
    """Probability distribution biased toward daytime hours (6-22)."""
    probs = [0.02, 0.05, 0.08, 0.10, 0.12, 0.12, 0.12, 0.10, 0.08, 0.06,
             0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.02]
    return probs


def _nighttime_distribution() -> list:
    """Probability distribution biased toward nighttime hours (0-23)."""
    probs = [0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02, 0.02, 0.02, 0.02,
             0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05,
             0.05, 0.06, 0.07, 0.03]
    return probs


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating synthetic transaction data...")
    data = generate_transaction_data(n_samples=15000, fraud_ratio=0.03)

    output_path = os.path.join(output_dir, "transactions.csv")
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Total transactions: {len(data)}")
    print(f"Fraudulent: {data['is_fraud'].sum()} ({data['is_fraud'].mean()*100:.1f}%)")
    print(f"Legitimate: {(data['is_fraud'] == 0).sum()} ({(1 - data['is_fraud'].mean())*100:.1f}%)")
    print(f"\nFeatures: {list(data.columns)}")
