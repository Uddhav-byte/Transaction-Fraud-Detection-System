"""
FastAPI Application

RESTful API for real-time fraud detection. Receives transaction data
and returns fraud probability scores using the trained Random Forest model.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.model import FraudDetectionModel

# ─── App Configuration ───────────────────────────────────────

app = FastAPI(
    title="Transaction Fraud Detection API",
    description="Real-time fraud detection for financial transactions using Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model & Preprocessor ───────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

model = FraudDetectionModel()
preprocessor = DataPreprocessor()

try:
    model.load(MODEL_DIR)
    preprocessor.load(MODEL_DIR)
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Model not found. Run training pipeline first: python src/train.py")
    print(f"   Error: {e}")
    MODEL_LOADED = False


# ─── Request/Response Schemas ────────────────────────────────

class TransactionRequest(BaseModel):
    """Schema for incoming transaction prediction requests."""
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    merchant_category: str = Field(..., description="Merchant category (e.g., grocery, online_shopping)")
    transaction_location: str = Field(..., description="Location type: domestic or international")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, 0 if weekday")
    customer_age: int = Field(..., ge=18, le=120, description="Customer age")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    num_transactions_last_24h: int = Field(..., ge=0, description="Number of transactions in last 24 hours")
    avg_transaction_amount_30d: float = Field(..., ge=0, description="Average transaction amount over 30 days")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in miles")
    amount_to_avg_ratio: Optional[float] = Field(None, description="Ratio of amount to 30-day average")
    is_high_risk_category: Optional[int] = Field(None, ge=0, le=1, description="1 if high-risk merchant category")
    is_international: Optional[int] = Field(None, ge=0, le=1, description="1 if international transaction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_amount": 2500.00,
                    "transaction_hour": 2,
                    "merchant_category": "online_shopping",
                    "transaction_location": "international",
                    "is_weekend": 1,
                    "customer_age": 35,
                    "account_age_days": 45,
                    "num_transactions_last_24h": 12,
                    "avg_transaction_amount_30d": 150.00,
                    "distance_from_home": 500.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    transaction_amount: float
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    confidence: float
    message: str


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    version: str


class ModelMetricsResponse(BaseModel):
    """Schema for model metrics response."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    model_type: Optional[str] = None


# ─── API Endpoints ───────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_LOADED,
        version="1.0.0",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict whether a transaction is fraudulent.

    Returns fraud probability score and risk classification.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first: python src/train.py"
        )

    try:
        # Prepare input data
        data = transaction.model_dump()

        # Auto-compute derived features if not provided
        if data.get("amount_to_avg_ratio") is None:
            data["amount_to_avg_ratio"] = data["transaction_amount"] / (data["avg_transaction_amount_30d"] + 1)

        if data.get("is_high_risk_category") is None:
            high_risk = ["wire_transfer", "cash_advance", "jewelry", "electronics"]
            data["is_high_risk_category"] = 1 if data["merchant_category"] in high_risk else 0

        if data.get("is_international") is None:
            data["is_international"] = 1 if data["transaction_location"] == "international" else 0

        # Preprocess and predict
        X = preprocessor.preprocess_single(data)
        fraud_probability = float(model.predict_proba(X)[0])
        is_fraud = fraud_probability >= 0.5

        # Determine risk level
        if fraud_probability >= 0.8:
            risk_level = "CRITICAL"
            message = "🚨 Transaction flagged as highly suspicious. Immediate review required."
        elif fraud_probability >= 0.5:
            risk_level = "HIGH"
            message = "⚠️ Transaction appears fraudulent. Manual review recommended."
        elif fraud_probability >= 0.3:
            risk_level = "MEDIUM"
            message = "📋 Transaction shows some suspicious patterns. Monitor closely."
        else:
            risk_level = "LOW"
            message = "✅ Transaction appears legitimate."

        return PredictionResponse(
            transaction_amount=data["transaction_amount"],
            fraud_probability=round(fraud_probability, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            confidence=round(abs(fraud_probability - 0.5) * 2, 4),
            message=message,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """
    Batch prediction for multiple transactions.

    Accepts a list of transactions and returns predictions for each.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for txn in transactions:
        result = await predict_fraud(txn)
        results.append(result)

    fraud_count = sum(1 for r in results if r.is_fraud)

    return {
        "total_transactions": len(results),
        "flagged_as_fraud": fraud_count,
        "legitimate": len(results) - fraud_count,
        "predictions": results,
    }


@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get the trained model's evaluation metrics."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    metrics = model.training_metrics
    return ModelMetricsResponse(
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        roc_auc=metrics.get("roc_auc"),
        model_type="RandomForestClassifier",
    )


@app.get("/model/features")
async def get_model_features():
    """Get the list of features used by the model."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "features": preprocessor.feature_columns,
        "total_features": len(preprocessor.feature_columns),
    }
