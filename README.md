# 🛡️ Transaction Fraud Detection System

A machine learning pipeline for identifying fraudulent financial transactions in real-time using **Python**, **Scikit-Learn**, **Pandas**, and **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project implements an end-to-end fraud detection system that:

- **Cleans and preprocesses** transaction records using Pandas, with SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance between normal and fraudulent transactions
- **Trains and evaluates** a Random Forest classifier, focusing on precision and recall to accurately flag anomalies while minimizing false positives
- **Deploys** the trained model via a RESTful API using FastAPI, enabling real-time fraud probability scoring

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Preprocessing   │────▶│  Model Training │
│  (Transactions) │     │  (Pandas/SMOTE)  │     │ (Random Forest) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client App    │◀───▶│   FastAPI REST   │◀────│  Trained Model  │
│   (Frontend)    │     │   API Server     │     │   (.joblib)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
Transaction-Fraud-Detection-System/
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_generator.py      # Synthetic data generation
│   ├── preprocessing.py       # Data cleaning, encoding, scaling, SMOTE
│   ├── model.py               # Random Forest model training & evaluation
│   ├── train.py               # End-to-end training pipeline
│   └── api.py                 # FastAPI REST API
├── tests/
│   └── test_pipeline.py       # Unit tests
├── data/                      # Generated datasets (gitignored)
├── models/                    # Trained model artifacts (gitignored)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker containerization
├── Procfile                   # Deployment process file
├── render.yaml                # Render deployment config
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Uddhav-byte/Transaction-Fraud-Detection-System.git
cd Transaction-Fraud-Detection-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

This will:
1. Generate 15,000 synthetic transactions (~3% fraud rate)
2. Clean and preprocess the data
3. Apply SMOTE for class balancing
4. Train a Random Forest classifier (200 estimators)
5. Evaluate on the test set
6. Save the model and preprocessor artifacts to `models/`

### Start the API Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/predict` | Predict fraud for a single transaction |
| `POST` | `/predict/batch` | Batch prediction for multiple transactions |
| `GET` | `/model/metrics` | View model evaluation metrics |
| `GET` | `/model/features` | List features used by the model |
| `GET` | `/docs` | Interactive Swagger UI documentation |

### Example: Predict Fraud

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 2500.00,
    "transaction_hour": 2,
    "merchant_category": "online_shopping",
    "transaction_location": "international",
    "is_weekend": 1,
    "customer_age": 35,
    "account_age_days": 45,
    "num_transactions_last_24h": 12,
    "avg_transaction_amount_30d": 150.00,
    "distance_from_home": 500.0
  }'
```

### Example Response

```json
{
  "transaction_amount": 2500.0,
  "fraud_probability": 0.8723,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "confidence": 0.7446,
  "message": "🚨 Transaction flagged as highly suspicious. Immediate review required."
}
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~0.98 |
| Precision | ~0.85 |
| Recall | ~0.92 |
| F1-Score | ~0.88 |
| ROC-AUC | ~0.99 |

> Metrics are approximate and may vary with different random seeds.

### Key Features Used

| Feature | Description |
|---------|-------------|
| `transaction_amount` | Transaction value in USD |
| `transaction_hour` | Hour of day (0-23) |
| `merchant_category` | Type of merchant |
| `distance_from_home` | Distance from cardholder's home |
| `num_transactions_last_24h` | Transaction velocity |
| `amount_to_avg_ratio` | Deviation from spending pattern |
| `is_international` | Whether cross-border |
| `account_age_days` | Account maturity |

## 🐳 Docker

```bash
# Build
docker build -t fraud-detection .

# Run
docker run -p 8000:8000 fraud-detection
```

## 🌐 Deployment

This project is configured for deployment on **Render**:

1. Push code to GitHub
2. Connect the repository on [Render](https://render.com)
3. It will auto-detect `render.yaml` and deploy

The build command trains the model, and the start command serves the API.

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## 🛠️ Tech Stack

- **Python 3.11** — Core language
- **Pandas** — Data manipulation and preprocessing
- **Scikit-Learn** — Random Forest classifier, metrics, cross-validation
- **imbalanced-learn** — SMOTE for handling class imbalance
- **FastAPI** — RESTful API framework
- **Uvicorn** — ASGI server
- **Joblib** — Model serialization

## 📄 License

This project is licensed under the MIT License.
