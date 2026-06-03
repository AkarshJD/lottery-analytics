"""
Pydantic request and response schemas for the serving API.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    game_name: str = Field(..., description="Draw game name (e.g. 'Powerball', 'Mega Millions')")
    horizon_days: int = Field(7, ge=1, le=30, description="Number of days to forecast")
    jackpot_amount: float = Field(0.0, ge=0.0, description="Jackpot amount for the forecast period")


class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    game_name: str
    horizon_days: int
    jackpot_amount: float
    predictions: list[ForecastPoint]


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class RetailerFeatures(BaseModel):
    location_id: str = Field(..., description="Retailer location identifier")

    # Volume
    total_claims: float = Field(..., ge=0)
    total_claim_amount: float = Field(..., ge=0)
    avg_claim_amount: float = Field(..., ge=0)
    median_claim_amount: float = Field(..., ge=0)
    max_claim_amount: float = Field(..., ge=0)
    claim_amount_std: float = Field(..., ge=0)
    large_claim_ratio: float = Field(..., ge=0, le=1)

    # Timing
    avg_claim_hour: float = Field(..., ge=0, le=23)
    claim_hour_std: float = Field(..., ge=0)
    off_hours_ratio: float = Field(..., ge=0, le=1)

    # Draw proximity
    avg_days_since_draw: float = Field(..., ge=0)
    min_days_since_draw: float = Field(..., ge=0)
    same_day_claim_ratio: float = Field(..., ge=0, le=1)

    # Concentration
    claims_per_month: float = Field(..., ge=0)
    top10_amount_ratio: float = Field(..., ge=0, le=1)


class AnomalyResponse(BaseModel):
    location_id: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="0 = normal, 1 = most anomalous")
    is_flagged: bool
    threshold: float
