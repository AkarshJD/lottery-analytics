"""
FastAPI serving layer.

Endpoints:
  GET  /health   — model load status
  POST /predict  — draw game sales forecast (XGBoost)
  POST /score    — retailer anomaly score (Isolation Forest)

Models are loaded from the MLflow registry at startup and cached in memory.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from src.features.claims import ANOMALY_FEATURE_COLUMNS
from src.features.transforms import FEATURE_COLUMNS, GAME_CFG, build_features
from src.data.load import load_game
from src.serving.schemas import (
    AnomalyResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    RetailerFeatures,
)

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlruns/mlflow.db")

# In-memory model cache populated at startup
_MODELS: dict = {}


def _load_models() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    cache: dict = {"forecast": {}, "anomaly": None}

    # Anomaly model
    try:
        cache["anomaly"] = mlflow.sklearn.load_model("models:/anomaly-retailer-claims/latest")
    except Exception as exc:
        print(f"  [warn] anomaly model not loaded: {exc}")

    # Per-game forecast models (XGBoost direct)
    for game_name in GAME_CFG:
        model_name = f"forecast-{game_name.lower().replace(' ', '-')}"
        try:
            cache["forecast"][game_name] = mlflow.xgboost.load_model(f"models:/{model_name}/latest")
        except Exception as exc:
            print(f"  [warn] forecast model for '{game_name}' not loaded: {exc}")

    return cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading models from MLflow registry...")
    _MODELS.update(_load_models())
    print(f"  anomaly model: {'✓' if _MODELS['anomaly'] else '✗'}")
    for g in GAME_CFG:
        status = "✓" if g in _MODELS["forecast"] else "✗"
        print(f"  forecast [{g}]: {status}")
    yield
    _MODELS.clear()


app = FastAPI(
    title="Retail Analytics API",
    description="Draw game sales forecasting and retailer anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    loaded = {
        "anomaly": _MODELS.get("anomaly") is not None,
        **{f"forecast_{g}": g in _MODELS.get("forecast", {}) for g in GAME_CFG},
    }
    return HealthResponse(status="ok", models_loaded=loaded)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    game_name = req.game_name

    if game_name not in GAME_CFG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown game '{game_name}'. Valid games: {sorted(GAME_CFG.keys())}",
        )

    model = _MODELS.get("forecast", {}).get(game_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Forecast model for '{game_name}' is not loaded.",
        )

    data_path = ROOT / "data" / "raw" / "daily_sales.parquet"
    if not data_path.exists():
        raise HTTPException(status_code=503, detail="Historical sales data not available.")

    # Load last 30 rows for lag feature warmup
    history = load_game(game_name).sort_values("draw_date").tail(30)
    last_date = pd.Timestamp(history["draw_date"].max())

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=req.horizon_days,
        freq="D",
    )
    future_rows = pd.DataFrame({
        "draw_date": future_dates,
        "game_name": game_name,
        "draw_sales": 0.0,
        "jackpot_amount": float(req.jackpot_amount),
    })

    combined = pd.concat([history, future_rows], ignore_index=True)
    features = build_features(combined, game_name)
    X_future = features.tail(req.horizon_days)[FEATURE_COLUMNS].fillna(0).astype("float64")

    raw_preds = model.predict(X_future)

    predictions = [
        ForecastPoint(date=str(d.date()), predicted_sales=round(float(p), 2))
        for d, p in zip(future_dates, raw_preds)
    ]

    return ForecastResponse(
        game_name=game_name,
        horizon_days=req.horizon_days,
        jackpot_amount=req.jackpot_amount,
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@app.post("/score", response_model=AnomalyResponse)
def score(req: RetailerFeatures):
    detector = _MODELS.get("anomaly")
    if detector is None:
        raise HTTPException(status_code=503, detail="Anomaly model is not loaded.")

    features = req.model_dump(exclude={"location_id"})
    X = pd.DataFrame([features])[ANOMALY_FEATURE_COLUMNS].fillna(0).astype("float64")

    # Scale and score via the detector's internal components
    X_scaled = pd.DataFrame(
        detector.scaler.transform(X),
        columns=ANOMALY_FEATURE_COLUMNS,
    )
    raw = float(detector.model.decision_function(X_scaled)[0])

    # IsolationForest decision_function: positive = inlier, negative = outlier
    # Map to [0, 1] where 1 = most anomalous using empirical range [-0.3, 0.3]
    anomaly_score = float(np.clip((-raw + 0.3) / 0.6, 0.0, 1.0))

    return AnomalyResponse(
        location_id=req.location_id,
        anomaly_score=round(anomaly_score, 4),
        is_flagged=anomaly_score >= detector.threshold,
        threshold=float(detector.threshold),
    )
