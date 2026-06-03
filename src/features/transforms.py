"""
Feature engineering for draw-level sales forecasting.

Features fall into three groups:
  - Calendar      : day-of-week indicators, draw-day flag, month, quarter
  - Jackpot       : game-specific power transform, jackpot growth, prior jackpot (MM only)
  - Lag / rolling : past sales patterns the XGBoost residual model exploits
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_cfg = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))
GAME_CFG = _cfg["forecasting"]["games"]


def add_calendar_features(df: pd.DataFrame, date_col: str = "draw_date") -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col]

    df["day_of_week"] = dt.dt.dayofweek        # 0=Mon … 6=Sun
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["fiscal_year"] = dt.apply(lambda d: d.year + 1 if d.month >= 7 else d.year)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Day-of-week one-hot (Mon=0 baseline dropped)
    for dow in range(1, 7):
        df[f"dow_{dow}"] = (df["day_of_week"] == dow).astype(int)

    return df


def add_jackpot_features(
    df: pd.DataFrame,
    game_name: str,
) -> pd.DataFrame:
    """
    Applies game-specific power transform to jackpot amount.
    Exponents come from power regression on historical data:
      PB  → jackpot^1.61
      MM  → jackpot^1.66  (+ prior jackpot^1.66)
      F5  → jackpot^1.0   (linear)
      TT  → jackpot^0.33
      TP  → jackpot^0.88
    """
    df = df.copy()
    gcfg = GAME_CFG.get(game_name, {})
    exp = gcfg.get("jackpot_exponent", 1.0)

    df["jackpot_power"] = np.power(df["jackpot_amount"].clip(lower=0), exp)

    # Jackpot growth since previous draw
    df["jackpot_growth"] = df["jackpot_amount"].diff().clip(lower=0).fillna(0)
    df["jackpot_growth_power"] = np.power(df["jackpot_growth"].clip(lower=0), exp)

    # Flag top-decile jackpots — captures the parabolic spike zone
    q90 = df["jackpot_amount"].quantile(0.90)
    df["is_large_jackpot"] = (df["jackpot_amount"] >= q90).astype(int)

    # Prior jackpot (Mega Millions only — prior draw jackpot affects current sales)
    if gcfg.get("use_prior_jackpot", False):
        df["prior_jackpot_power"] = df["jackpot_power"].shift(1).fillna(0)
    else:
        df["prior_jackpot_power"] = 0.0

    # Draw-day flag: is today an actual draw day for this game?
    draw_days = gcfg.get("draw_days", list(range(7)))
    df["is_draw_day"] = df["day_of_week"].isin(draw_days).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list = [1, 2, 3, 7, 14],
) -> pd.DataFrame:
    df = df.copy().sort_values("draw_date").reset_index(drop=True)

    for lag in lags:
        df[f"sales_lag_{lag}"] = df["draw_sales"].shift(lag)

    df["sales_roll_3"] = df["draw_sales"].shift(1).rolling(3).mean()
    df["sales_roll_7"] = df["draw_sales"].shift(1).rolling(7).mean()
    df["sales_roll_14"] = df["draw_sales"].shift(1).rolling(14).mean()
    df["sales_trend"] = df["sales_roll_7"] - df["sales_roll_14"]

    return df


def build_features(df: pd.DataFrame, game_name: str) -> pd.DataFrame:
    """Full feature pipeline for a single game's dataframe."""
    df = add_calendar_features(df)
    df = add_jackpot_features(df, game_name)
    df = add_lag_features(df)
    # Cast all feature columns to float64 so MLflow schema inference
    # handles missing values correctly at inference time
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


# Columns XGBoost trains on
FEATURE_COLUMNS = [
    # calendar
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "month", "quarter", "fiscal_year", "week_of_year",
    "is_weekend", "is_draw_day",
    # jackpot
    "jackpot_power", "jackpot_growth_power",
    "is_large_jackpot", "prior_jackpot_power",
    # lags
    "sales_lag_1", "sales_lag_2", "sales_lag_3",
    "sales_lag_7", "sales_lag_14",
    "sales_roll_3", "sales_roll_7", "sales_roll_14",
    "sales_trend",
]
