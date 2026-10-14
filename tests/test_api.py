"""
API Integration & Route Tests for Fraud Detection System
"""

import os
import sys
import asyncio
from fastapi import Request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import (
    app,
    root,
    dashboard,
    health_check,
    predict_fraud,
    predict_batch,
    get_model_metrics,
    get_model_features,
    TransactionRequest,
)


def test_root_browser_request_returns_html_dashboard():
    """Browser request (text/html) to / should serve the interactive dashboard."""
    req_html = Request({
        "type": "http",
        "method": "GET",
        "headers": [(b"accept", b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")],
        "query_string": b"",
    })
    res_html = asyncio.run(root(req_html))
    content = res_html.body.decode("utf-8")
    assert res_html.status_code == 200
    assert "SentinelFraud" in content
    assert "Live Transaction Fraud Simulator" in content
    assert "Holdout Confusion Matrix" in content


def test_dashboard_direct_endpoint_returns_html():
    """Direct visit to /dashboard should return 200 and HTML content."""
    res_dash = asyncio.run(dashboard())
    content = res_dash.body.decode("utf-8")
    assert res_dash.status_code == 200
    assert "SentinelFraud" in content


def test_root_json_request_returns_health_json():
    """Explicit JSON request (Accept: application/json) to / should return health status JSON."""
    req_json = Request({
        "type": "http",
        "method": "GET",
        "headers": [(b"accept", b"application/json")],
        "query_string": b"",
    })
    res_json = asyncio.run(root(req_json))
    assert res_json.status == "healthy"
    assert res_json.model_loaded is True
    assert res_json.version == "1.0.0"


def test_root_query_format_json_returns_health_json():
    """Request with ?format=json should return health status JSON."""
    req_format = Request({
        "type": "http",
        "method": "GET",
        "headers": [(b"accept", b"*/*")],
        "query_string": b"format=json",
    })
    res_json = asyncio.run(root(req_format))
    assert res_json.status == "healthy"
    assert res_json.model_loaded is True


def test_health_endpoint_returns_json():
    """Dedicated /health endpoint should return JSON health info."""
    res_health = asyncio.run(health_check())
    assert res_health.status == "healthy"
    assert res_health.model_loaded is True
    assert res_health.version == "1.0.0"


def test_predict_single_legitimate_transaction():
    """Test single transaction prediction on legitimate coffee purchase."""
    txn = TransactionRequest(
        transaction_amount=4.75,
        transaction_hour=8,
        merchant_category="grocery",
        transaction_location="domestic",
        is_weekend=0,
        customer_age=32,
        account_age_days=720,
        num_transactions_last_24h=1,
        avg_transaction_amount_30d=45.0,
        distance_from_home=1.8,
    )
    res = asyncio.run(predict_fraud(txn))
    assert res.is_fraud is False
    assert 0.0 <= res.fraud_probability <= 0.3
    assert res.risk_level in ["LOW", "MEDIUM"]


def test_predict_single_fraudulent_transaction():
    """Test single transaction prediction on high-risk suspicious midnight wire."""
    txn = TransactionRequest(
        transaction_amount=9800.0,
        transaction_hour=2,
        merchant_category="wire_transfer",
        transaction_location="international",
        is_weekend=1,
        customer_age=35,
        account_age_days=12,
        num_transactions_last_24h=14,
        avg_transaction_amount_30d=220.0,
        distance_from_home=850.0,
    )
    res = asyncio.run(predict_fraud(txn))
    assert res.is_fraud is True
    assert res.fraud_probability > 0.5
    assert res.risk_level in ["HIGH", "CRITICAL"]


def test_predict_batch_transactions():
    """Test batch prediction endpoint with multiple transactions."""
    batch = [
        TransactionRequest(
            transaction_amount=10.0,
            transaction_hour=10,
            merchant_category="grocery",
            transaction_location="domestic",
            is_weekend=0,
            customer_age=30,
            account_age_days=500,
            num_transactions_last_24h=1,
            avg_transaction_amount_30d=50.0,
            distance_from_home=2.0,
        ),
        TransactionRequest(
            transaction_amount=9500.0,
            transaction_hour=3,
            merchant_category="wire_transfer",
            transaction_location="international",
            is_weekend=1,
            customer_age=28,
            account_age_days=5,
            num_transactions_last_24h=12,
            avg_transaction_amount_30d=100.0,
            distance_from_home=3000.0,
        ),
    ]
    res = asyncio.run(predict_batch(batch))
    assert res["total_transactions"] == 2
    assert res["flagged_as_fraud"] >= 1
    assert len(res["predictions"]) == 2


def test_model_metrics_endpoint():
    """Test that /model/metrics returns model performance numbers."""
    metrics = asyncio.run(get_model_metrics())
    assert metrics.accuracy is not None
    assert metrics.precision is not None
    assert metrics.recall is not None
    assert metrics.f1_score is not None
    assert metrics.roc_auc is not None
    assert metrics.model_type == "RandomForestClassifier"


def test_model_features_endpoint():
    """Test that /model/features returns feature columns."""
    features_resp = asyncio.run(get_model_features())
    assert "features" in features_resp
    assert features_resp["total_features"] > 0


if __name__ == "__main__":
    print("Running API tests...")
    test_root_browser_request_returns_html_dashboard()
    print("✅ test_root_browser_request_returns_html_dashboard passed")
    test_dashboard_direct_endpoint_returns_html()
    print("✅ test_dashboard_direct_endpoint_returns_html passed")
    test_root_json_request_returns_health_json()
    print("✅ test_root_json_request_returns_health_json passed")
    test_root_query_format_json_returns_health_json()
    print("✅ test_root_query_format_json_returns_health_json passed")
    test_health_endpoint_returns_json()
    print("✅ test_health_endpoint_returns_json passed")
    test_predict_single_legitimate_transaction()
    print("✅ test_predict_single_legitimate_transaction passed")
    test_predict_single_fraudulent_transaction()
    print("✅ test_predict_single_fraudulent_transaction passed")
    test_predict_batch_transactions()
    print("✅ test_predict_batch_transactions passed")
    test_model_metrics_endpoint()
    print("✅ test_model_metrics_endpoint passed")
    test_model_features_endpoint()
    print("✅ test_model_features_endpoint passed")
    print("\n🎉 All 10 API tests passed successfully!")
