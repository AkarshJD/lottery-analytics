"""
Feature engineering for location-level weekly sales forecasting and
inventory allocation.

Grain: fiscal_year x fiscal_week x location_id (3,000 locations).

Top-down hierarchical approach:
  1. Aggregate to one system-wide weekly series and forecast it (XGBoost).
  2. Allocate the system forecast down to locations using each location's
     trailing rolling share of system sales.

Location shares are close to stationary by construction — a retailer's
share of statewide sales doesn't reshuffle week to week — so a rolling
average share is a strong, low-variance estimator. That's a safer bet than
fitting 3,000 independent per-location models on ~5 years of noisy weekly
history, and it's the standard technique (top-down / middle-out
reconciliation) when you need location-grain decisions but location-grain
series are too short or noisy to model directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
_cfg = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))
LOC_CFG = _cfg["location_forecasting"]


def load_location_sales(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Loads location_weekly_sales.parquet, sorted by location/week."""
    df = pd.read_parquet(raw_dir / "location_weekly_sales.parquet")
    return df.sort_values(["location_id", "fiscal_year", "fiscal_week"]).reset_index(drop=True)


def add_week_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a monotonic week_index so lag/rolling ops are exact across
    fiscal-year boundaries (fiscal_week resets 1-53 each year, so a plain
    groupby on fiscal_week alone would straddle years incorrectly).
    """
    weeks = (
        df[["fiscal_year", "fiscal_week"]]
        .drop_duplicates()
        .sort_values(["fiscal_year", "fiscal_week"])
        .reset_index(drop=True)
    )
    weeks["week_index"] = np.arange(len(weeks))
    return df.merge(weeks, on=["fiscal_year", "fiscal_week"], how="left")


def build_system_series(loc_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates location-level sales to one row per week."""
    df = add_week_index(loc_df)
    system = (
        df.groupby(["fiscal_year", "fiscal_week", "week_index"])["weekly_sale_amount"]
        .sum()
        .reset_index()
        .rename(columns={"weekly_sale_amount": "system_sales"})
        .sort_values("week_index")
        .reset_index(drop=True)
    )
    return system


def build_system_features(system: pd.DataFrame, lags: list = [1, 2, 4, 8, 13]) -> pd.DataFrame:
    """Calendar + lag/rolling features for the system-wide weekly series."""
    df = system.copy().sort_values("week_index").reset_index(drop=True)

    # Cyclical encoding of fiscal week — captures annual seasonality without
    # a hard discontinuity between week 52 and week 1.
    df["week_sin"] = np.sin(2 * np.pi * df["fiscal_week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["fiscal_week"] / 52)
    df["fiscal_quarter"] = (((df["fiscal_week"] - 1) // 13) + 1).clip(upper=4)

    for lag in lags:
        df[f"sales_lag_{lag}"] = df["system_sales"].shift(lag)

    df["sales_roll_4"] = df["system_sales"].shift(1).rolling(4).mean()
    df["sales_roll_8"] = df["system_sales"].shift(1).rolling(8).mean()
    df["sales_roll_13"] = df["system_sales"].shift(1).rolling(13).mean()
    df["sales_trend"] = df["sales_roll_4"] - df["sales_roll_13"]

    for col in SYSTEM_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


SYSTEM_FEATURE_COLUMNS = [
    "week_sin", "week_cos", "fiscal_quarter",
    "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8", "sales_lag_13",
    "sales_roll_4", "sales_roll_8", "sales_roll_13", "sales_trend",
]


def build_location_shares(loc_df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Trailing rolling share of system sales per location, computed using
    only data strictly BEFORE each week (shift(1)) so there's no leakage
    when this is used to allocate a forecast for that week.

    share[loc, t] = roll_mean(loc_sales, window)[t] / sum_loc(roll_mean)[t]
    """
    window = window or LOC_CFG["share_window"]
    df = add_week_index(loc_df).sort_values(["location_id", "week_index"]).reset_index(drop=True)

    df["loc_roll"] = df.groupby("location_id")["weekly_sale_amount"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    week_totals = df.groupby("week_index")["loc_roll"].transform("sum")
    df["share"] = (df["loc_roll"] / week_totals).fillna(0.0)

    return df[["location_id", "fiscal_year", "fiscal_week", "week_index", "weekly_sale_amount", "share"]]


def current_location_shares(loc_df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Share estimate for allocating a forecast of the next, not-yet-observed
    week. Unlike build_location_shares (which shifts by one week so a
    backtest never peeks at the week it's predicting), this uses the
    trailing `window` weeks of already-observed data directly — there's no
    leakage because all of it genuinely precedes "now".
    """
    window = window or LOC_CFG["share_window"]
    df = add_week_index(loc_df)
    last_week = df["week_index"].max()
    recent = df[df["week_index"] > last_week - window]
    totals = recent.groupby("location_id")["weekly_sale_amount"].mean()
    shares = (totals / totals.sum()).reset_index()
    shares.columns = ["location_id", "share"]
    return shares
