"""
Tests for feature engineering — transforms.py
"""

import numpy as np
import pandas as pd
import pytest

from src.features.transforms import build_features, FEATURE_COLUMNS


def fiscal_year(date: pd.Timestamp) -> int:
    """FY starts July 1 — July 2023 = FY2024."""
    return date.year + 1 if date.month >= 7 else date.year


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game_df(game_name: str = "Powerball", n_days: int = 40, jackpot: float = 100_000_000) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "draw_date": dates,
        "game_name": game_name,
        "draw_sales": np.random.default_rng(0).uniform(50_000, 500_000, n_days),
        "jackpot_amount": jackpot,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_feature_columns_present():
    """build_features must return every column in FEATURE_COLUMNS."""
    df = make_game_df()
    features = build_features(df, "Powerball")
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    assert not missing, f"Missing feature columns: {missing}"


def test_jackpot_power_non_negative():
    """jackpot_power must be >= 0 for any jackpot input."""
    df = make_game_df(jackpot=0.0)
    features = build_features(df, "Powerball")
    assert (features["jackpot_power"] >= 0).all()


def test_jackpot_power_scales_with_jackpot():
    """Higher jackpot → higher jackpot_power for games with exponent > 0."""
    low = build_features(make_game_df(jackpot=10_000_000), "Powerball")["jackpot_power"].iloc[0]
    high = build_features(make_game_df(jackpot=500_000_000), "Powerball")["jackpot_power"].iloc[0]
    assert high > low


def test_fiscal_year_july_start():
    """July 2023 should map to FY2024 (July 1 year start)."""
    assert fiscal_year(pd.Timestamp("2023-07-01")) == 2024


def test_fiscal_year_june_end():
    """June 2023 should remain in FY2023."""
    assert fiscal_year(pd.Timestamp("2023-06-30")) == 2023


def test_draw_day_flag_dtype():
    """is_draw_day must be numeric (0 or 1)."""
    df = make_game_df()
    features = build_features(df, "Powerball")
    assert features["is_draw_day"].isin([0, 1]).all()


def test_lag_features_first_rows_nan():
    """sales_lag_1 must be NaN for the first row (no prior data)."""
    df = make_game_df(n_days=30)
    features = build_features(df, "Powerball")
    assert pd.isna(features["sales_lag_1"].iloc[0])


def test_feature_columns_float64():
    """All FEATURE_COLUMNS must be float64 after build_features."""
    df = make_game_df()
    features = build_features(df, "Powerball")
    for col in FEATURE_COLUMNS:
        assert features[col].dtype == np.float64, f"{col} is {features[col].dtype}"
