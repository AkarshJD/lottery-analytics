"""
System-wide weekly demand forecaster and location allocator.

Two components, used together as a top-down hierarchical forecast:
  SystemDemandForecaster — XGBoost regressor for total weekly ticket sales
                           across all 3,000 locations.
  LocationAllocator      — disaggregates that total down to locations using
                           each location's trailing rolling share.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

from src.features.locations import (
    LOC_CFG,
    SYSTEM_FEATURE_COLUMNS,
    build_location_shares,
    build_system_features,
    current_location_shares,
)


class SystemDemandForecaster:
    """XGBoost regressor for the top level of the hierarchy."""

    def __init__(self):
        xcfg = LOC_CFG["xgboost"]
        self.model = xgb.XGBRegressor(
            n_estimators=xcfg["n_estimators"],
            max_depth=xcfg["max_depth"],
            learning_rate=xcfg["learning_rate"],
            subsample=xcfg["subsample"],
            colsample_bytree=xcfg["colsample_bytree"],
            min_child_weight=xcfg["min_child_weight"],
            objective=xcfg["objective"],
            random_state=xcfg["random_state"],
            verbosity=0,
        )
        self._fitted = False

    def fit(self, system_feat: pd.DataFrame):
        valid = system_feat[SYSTEM_FEATURE_COLUMNS].notna().all(axis=1)
        X = system_feat.loc[valid, SYSTEM_FEATURE_COLUMNS]
        y = system_feat.loc[valid, "system_sales"]
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, system_feat: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "call fit() first"
        X = system_feat[SYSTEM_FEATURE_COLUMNS].fillna(0.0)
        return self.model.predict(X)

    def evaluate(self, system_feat: pd.DataFrame) -> dict:
        valid = system_feat[SYSTEM_FEATURE_COLUMNS].notna().all(axis=1)
        preds = self.predict(system_feat.loc[valid])
        actuals = system_feat.loc[valid, "system_sales"].values
        mape = mean_absolute_percentage_error(actuals, preds)
        mae = float(np.mean(np.abs(actuals - preds)))
        return {"mape": round(float(mape), 4), "mae": round(mae, 2)}


def forecast_system_ahead(
    forecaster: SystemDemandForecaster, system: pd.DataFrame, horizon_weeks: int
) -> pd.DataFrame:
    """
    Recursive multi-step forecast for genuinely future weeks: each
    predicted week's sales feed back in as the "observed" value before
    computing the next week's lag/rolling features. (Backtesting below
    instead uses true historical lags for an apples-to-apples one-step-
    ahead walk-forward evaluation — recursion is only needed when the
    actuals don't exist yet, i.e. at serving time.)
    """
    working = system[["week_index", "fiscal_year", "fiscal_week", "system_sales"]].copy()
    last_week_index = int(working["week_index"].max())
    last_fy = int(working.iloc[-1]["fiscal_year"])
    last_fw = int(working.iloc[-1]["fiscal_week"])

    preds = []
    for step in range(1, horizon_weeks + 1):
        next_week_index = last_week_index + step
        next_fy, next_fw = last_fy, last_fw + step
        while next_fw > 52:
            next_fw -= 52
            next_fy += 1

        working = pd.concat([working, pd.DataFrame([{
            "week_index": next_week_index,
            "fiscal_year": next_fy,
            "fiscal_week": next_fw,
            "system_sales": np.nan,
        }])], ignore_index=True)

        feat = build_system_features(working)
        pred = float(forecaster.predict(feat.iloc[[-1]])[0])
        working.loc[working["week_index"] == next_week_index, "system_sales"] = pred
        preds.append({
            "week_index": next_week_index,
            "fiscal_year": next_fy,
            "fiscal_week": next_fw,
            "predicted_system_sales": pred,
        })

    return pd.DataFrame(preds)


class LocationAllocator:
    """Disaggregates a system-wide forecast down to locations via trailing
    rolling share — see src/features/locations.py for why share-based
    (rather than per-location) modeling is the right call here."""

    def __init__(self, window: int = None):
        self.window = window or LOC_CFG["share_window"]

    def compute_shares(self, loc_df: pd.DataFrame) -> pd.DataFrame:
        return build_location_shares(loc_df, window=self.window)

    def allocate(self, system_forecast: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
        """
        system_forecast: columns [week_index, predicted_system_sales]
        shares:          columns [location_id, week_index, share, ...]
        Returns one row per (location_id, week_index) with location_forecast.
        """
        merged = shares.merge(system_forecast, on="week_index", how="inner")
        merged["location_forecast"] = merged["share"] * merged["predicted_system_sales"]
        return merged

    def latest_shares(self, loc_df: pd.DataFrame) -> pd.DataFrame:
        """Share estimate per location for allocating a forecast of the
        next, not-yet-observed week (see current_location_shares)."""
        return current_location_shares(loc_df, window=self.window)
