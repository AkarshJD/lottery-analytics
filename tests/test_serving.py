"""
Tests for the FastAPI serving layer.

Models are mocked — no MLflow registry required for these tests.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import src.serving.app as serving_module
from src.serving.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_models(monkeypatch):
    """Patch _MODELS with mock detector and forecast model."""
    mock_anomaly = MagicMock()
    mock_anomaly.scaler.transform.return_value = np.zeros((1, 15))
    mock_anomaly.model.decision_function.return_value = np.array([0.05])
    mock_anomaly.threshold = 0.6

    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = np.full(7, 120_000.0)

    monkeypatch.setattr(serving_module, "_load_models", lambda: {
        "anomaly": mock_anomaly,
        "forecast": {"Powerball": mock_xgb},
    })


@pytest.fixture
def client(mock_models):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_has_models_loaded(client):
    data = client.get("/health").json()
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"].endswith("/docs")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_predict_unknown_game_returns_400(client):
    response = client.post("/predict", json={
        "game_name": "NotAGame",
        "horizon_days": 7,
        "jackpot_amount": 0,
    })
    assert response.status_code == 400


def test_predict_horizon_too_large_returns_422(client):
    response = client.post("/predict", json={
        "game_name": "Powerball",
        "horizon_days": 999,
        "jackpot_amount": 0,
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

VALID_RETAILER = {
    "location_id": "LOC_00001",
    "total_claims": 120.0,
    "total_claim_amount": 45000.0,
    "avg_claim_amount": 375.0,
    "median_claim_amount": 250.0,
    "max_claim_amount": 3000.0,
    "claim_amount_std": 400.0,
    "large_claim_ratio": 0.15,
    "avg_claim_hour": 13.5,
    "claim_hour_std": 2.8,
    "off_hours_ratio": 0.02,
    "avg_days_since_draw": 4.5,
    "min_days_since_draw": 0.0,
    "same_day_claim_ratio": 0.05,
    "claims_per_month": 6.0,
    "top10_amount_ratio": 0.42,
}


def test_score_valid_input_returns_200(client):
    response = client.post("/score", json=VALID_RETAILER)
    assert response.status_code == 200


def test_score_returns_location_id(client):
    data = client.post("/score", json=VALID_RETAILER).json()
    assert data["location_id"] == VALID_RETAILER["location_id"]


def test_score_anomaly_score_in_range(client):
    data = client.post("/score", json=VALID_RETAILER).json()
    assert 0.0 <= data["anomaly_score"] <= 1.0


def test_score_is_flagged_is_bool(client):
    data = client.post("/score", json=VALID_RETAILER).json()
    assert isinstance(data["is_flagged"], bool)


def test_score_invalid_ratio_returns_422(client):
    bad = {**VALID_RETAILER, "off_hours_ratio": 5.0}  # ratio > 1 is invalid
    response = client.post("/score", json=bad)
    assert response.status_code == 422
