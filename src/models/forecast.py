"""
Prophet + XGBoost ensemble forecaster.

Prophet models trend + seasonality + jackpot as external regressor.
XGBoost corrects Prophet's residuals using game-specific power-transformed
jackpot features and lag features.

Final prediction = prophet_pred + xgb_residual_correction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

from src.features.transforms import build_features, FEATURE_COLUMNS, GAME_CFG

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))["forecasting"]


class SalesForecast:
    def __init__(self, game_name: str):
        self.game_name = game_name
        gcfg = GAME_CFG.get(game_name, {})
        self.jackpot_exponent = gcfg.get("jackpot_exponent", 1.0)

        pcfg = CONFIG["prophet"]
        self.prophet = Prophet(
            seasonality_mode=pcfg["seasonality_mode"],
            yearly_seasonality=pcfg["yearly_seasonality"],
            weekly_seasonality=pcfg["weekly_seasonality"],
            daily_seasonality=pcfg["daily_seasonality"],
            changepoint_prior_scale=pcfg["changepoint_prior_scale"],
            seasonality_prior_scale=pcfg["seasonality_prior_scale"],
            holidays_prior_scale=pcfg["holidays_prior_scale"],
        )
        # Add jackpot as external regressor with game-specific power transform
        self.prophet.add_regressor("jackpot_power", standardize=True)

        xcfg = CONFIG["xgboost"]
        self.xgb = xgb.XGBRegressor(
            n_estimators=xcfg["n_estimators"],
            max_depth=xcfg["max_depth"],
            learning_rate=xcfg["learning_rate"],
            subsample=xcfg["subsample"],
            colsample_bytree=xcfg["colsample_bytree"],
            min_child_weight=xcfg["min_child_weight"],
            objective=xcfg["objective"],
            random_state=xcfg["random_state"],
        )
        self._fitted = False

    def _prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for Prophet: ds, y (if train), jackpot_power."""
        feat = build_features(df.copy(), self.game_name)
        out = feat.rename(columns={"draw_date": "ds"})
        out["jackpot_power"] = feat["jackpot_power"]
        if "draw_sales" in df.columns:
            out["y"] = df["draw_sales"].values
        return out

    def fit(self, df: pd.DataFrame):
        df = df.copy().sort_values("draw_date").reset_index(drop=True)

        # --- Prophet with jackpot regressor ---
        pdf = self._prophet_df(df)[["ds", "y", "jackpot_power"]]
        self.prophet.fit(pdf)

        prophet_pred = self.prophet.predict(pdf[["ds", "jackpot_power"]])["yhat"].values

        # --- XGBoost on residuals ---
        df_feat = build_features(df, self.game_name)
        residuals = df["draw_sales"].values - prophet_pred

        valid = df_feat[FEATURE_COLUMNS].notna().all(axis=1)
        X = df_feat.loc[valid, FEATURE_COLUMNS]
        y = residuals[valid]
        self.xgb.fit(X, y)

        self._fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "call fit() first"
        df = df.copy().sort_values("draw_date").reset_index(drop=True)

        pdf = self._prophet_df(df)[["ds", "jackpot_power"]]
        prophet_pred = self.prophet.predict(pdf)["yhat"].values

        df_feat = build_features(df, self.game_name)
        valid_mask = df_feat[FEATURE_COLUMNS].notna().all(axis=1)
        xgb_correction = np.zeros(len(df))
        if valid_mask.any():
            X = df_feat.loc[valid_mask, FEATURE_COLUMNS]
            xgb_correction[valid_mask] = self.xgb.predict(X)

        return prophet_pred + xgb_correction

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds = self.predict(df)
        actuals = df["draw_sales"].values
        valid = actuals > 0
        mape = mean_absolute_percentage_error(actuals[valid], preds[valid])
        mae = np.mean(np.abs(actuals[valid] - preds[valid]))
        return {"mape": round(mape, 4), "mae": round(mae, 2)}
