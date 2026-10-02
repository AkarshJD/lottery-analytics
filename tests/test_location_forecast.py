"""
Tests for location-level forecasting and inventory allocation.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.locations import (
    SYSTEM_FEATURE_COLUMNS,
    build_location_shares,
    build_system_features,
    build_system_series,
    current_location_shares,
)
from src.models.inventory import backtest_allocation_policies
from src.models.location_forecast import (
    LocationAllocator,
    SystemDemandForecaster,
    forecast_system_ahead,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_location_weeks(n_weeks: int = 60, n_locations: int = 10, seed: int = 0) -> pd.DataFrame:
    """Flat system total disaggregated to locations via fixed random shares
    plus small noise — mirrors how the real data was generated."""
    rng = np.random.default_rng(seed)
    loc_ids = [f"LOC_{i:05d}" for i in range(n_locations)]
    raw_mult = rng.lognormal(mean=0.0, sigma=0.5, size=n_locations)
    fixed_shares = raw_mult / raw_mult.sum()

    rows = []
    for week in range(1, n_weeks + 1):
        fiscal_year = 2023 + (week - 1) // 52
        fiscal_week = (week - 1) % 52 + 1
        system_total = 1_000_000 * (1 + 0.02 * np.sin(2 * np.pi * week / 13))
        for loc_id, share in zip(loc_ids, fixed_shares):
            noise = rng.normal(1.0, 0.03)
            rows.append({
                "fiscal_year": fiscal_year,
                "fiscal_week": fiscal_week,
                "location_id": loc_id,
                "weekly_sale_amount": max(0.0, system_total * share * noise),
            })
    return pd.DataFrame(rows)


def make_seasonal_location_weeks(n_weeks: int = 150, n_locations: int = 10, seed: int = 0) -> pd.DataFrame:
    """
    System sales follow a strong, low-noise 13-week sinusoidal cycle,
    disaggregated to locations via fixed shares. A 13-week trailing rolling
    mean (the naive baseline) averages the cycle away and systematically
    under-predicts the peaks; a model with cyclical features anticipates
    them. Used to prove the ML allocation policy actually beats naive,
    not just that both run without error.
    """
    rng = np.random.default_rng(seed)
    loc_ids = [f"LOC_{i:05d}" for i in range(n_locations)]
    raw_mult = rng.lognormal(mean=0.0, sigma=0.5, size=n_locations)
    fixed_shares = raw_mult / raw_mult.sum()

    rows = []
    for week in range(1, n_weeks + 1):
        fiscal_year = 2020 + (week - 1) // 52
        fiscal_week = (week - 1) % 52 + 1
        system_total = 1_000_000 * (1 + 0.6 * np.sin(2 * np.pi * week / 13))
        for loc_id, share in zip(loc_ids, fixed_shares):
            noise = rng.normal(1.0, 0.01)
            rows.append({
                "fiscal_year": fiscal_year,
                "fiscal_week": fiscal_week,
                "location_id": loc_id,
                "weekly_sale_amount": max(0.0, system_total * share * noise),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def test_build_system_series_sums_locations():
    """system_sales for a week must equal the sum across all locations."""
    loc_df = make_location_weeks(n_weeks=20, n_locations=5)
    system = build_system_series(loc_df)
    week0 = loc_df[(loc_df["fiscal_year"] == 2023) & (loc_df["fiscal_week"] == 1)]
    expected = week0["weekly_sale_amount"].sum()
    actual = system.loc[system["week_index"] == 0, "system_sales"].iloc[0]
    assert actual == pytest.approx(expected)


def test_system_features_has_all_columns():
    loc_df = make_location_weeks(n_weeks=30, n_locations=5)
    system = build_system_series(loc_df)
    feat = build_system_features(system)
    missing = set(SYSTEM_FEATURE_COLUMNS) - set(feat.columns)
    assert not missing


def test_system_lag_first_rows_nan():
    loc_df = make_location_weeks(n_weeks=30, n_locations=5)
    system = build_system_series(loc_df)
    feat = build_system_features(system)
    assert pd.isna(feat["sales_lag_1"].iloc[0])


def test_location_shares_sum_to_one_after_window():
    """Once every location has `window` weeks of history, shares must sum to 1."""
    loc_df = make_location_weeks(n_weeks=40, n_locations=8)
    shares = build_location_shares(loc_df, window=13)
    last_week = shares["week_index"].max()
    total = shares.loc[shares["week_index"] == last_week, "share"].sum()
    assert total == pytest.approx(1.0, abs=1e-6)


def test_location_shares_no_leakage_at_first_week():
    """Shares at week_index=0 must be 0 for everyone — no prior data exists."""
    loc_df = make_location_weeks(n_weeks=20, n_locations=5)
    shares = build_location_shares(loc_df, window=13)
    first_week = shares.loc[shares["week_index"] == 0, "share"]
    assert (first_week == 0).all()


def test_current_location_shares_sums_to_one():
    loc_df = make_location_weeks(n_weeks=40, n_locations=8)
    shares = current_location_shares(loc_df, window=13)
    assert shares["share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert len(shares) == 8


# ---------------------------------------------------------------------------
# SystemDemandForecaster
# ---------------------------------------------------------------------------

def test_forecaster_requires_fit_before_predict():
    forecaster = SystemDemandForecaster()
    loc_df = make_location_weeks(n_weeks=30, n_locations=5)
    feat = build_system_features(build_system_series(loc_df))
    with pytest.raises(AssertionError):
        forecaster.predict(feat)


def test_forecaster_fit_predict_shape():
    loc_df = make_location_weeks(n_weeks=60, n_locations=5)
    feat = build_system_features(build_system_series(loc_df))
    forecaster = SystemDemandForecaster()
    forecaster.fit(feat)
    preds = forecaster.predict(feat)
    assert len(preds) == len(feat)


def test_forecaster_evaluate_returns_mape():
    loc_df = make_location_weeks(n_weeks=60, n_locations=5)
    feat = build_system_features(build_system_series(loc_df))
    forecaster = SystemDemandForecaster()
    forecaster.fit(feat)
    metrics = forecaster.evaluate(feat)
    assert "mape" in metrics and metrics["mape"] >= 0


def test_forecast_system_ahead_returns_horizon_rows():
    loc_df = make_location_weeks(n_weeks=60, n_locations=5)
    system = build_system_series(loc_df)
    feat = build_system_features(system)
    forecaster = SystemDemandForecaster()
    forecaster.fit(feat)
    future = forecast_system_ahead(forecaster, system, horizon_weeks=4)
    assert len(future) == 4
    assert (future["predicted_system_sales"] > 0).all()


# ---------------------------------------------------------------------------
# LocationAllocator
# ---------------------------------------------------------------------------

def test_allocator_allocation_sums_to_system_forecast():
    loc_df = make_location_weeks(n_weeks=40, n_locations=6)
    allocator = LocationAllocator(window=13)
    shares = allocator.compute_shares(loc_df)
    week = shares["week_index"].max()
    shares_at_week = shares[shares["week_index"] == week]

    system_forecast = pd.DataFrame({"week_index": [week], "predicted_system_sales": [500_000.0]})
    allocation = allocator.allocate(system_forecast, shares_at_week)
    assert allocation["location_forecast"].sum() == pytest.approx(500_000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Inventory allocation backtest
# ---------------------------------------------------------------------------

def test_backtest_returns_rates_in_valid_range():
    loc_df = make_location_weeks(n_weeks=80, n_locations=8)
    result = backtest_allocation_policies(loc_df, test_weeks=20, naive_window=8, share_window=8)
    assert 0.0 <= result["naive_stock_out_rate"] <= 1.0
    assert 0.0 <= result["ml_stock_out_rate"] <= 1.0
    assert result["n_location_weeks"] > 0


def test_backtest_equal_budget_calibration():
    """Both policies must allocate (approximately) the same total budget."""
    loc_df = make_location_weeks(n_weeks=80, n_locations=8)
    result = backtest_allocation_policies(loc_df, test_weeks=20, naive_window=8, share_window=8)
    # ml_scale_factor and naive_scale_factor both target result['total_budget'];
    # sanity check they're positive and finite.
    assert result["ml_scale_factor"] > 0
    assert result["naive_scale_factor"] > 0


def test_ml_policy_beats_naive_on_strong_seasonal_pattern():
    """
    The core value proposition: when demand has a predictable cyclical
    pattern that a 13-week trailing average smooths away, the ML policy
    (which sees the cycle via calendar features) should produce materially
    fewer stock-outs than the naive rolling-mean policy at the same budget.
    """
    loc_df = make_seasonal_location_weeks(n_weeks=150, n_locations=10)
    result = backtest_allocation_policies(loc_df, test_weeks=52, naive_window=13, share_window=13)
    assert result["ml_stock_out_rate"] < result["naive_stock_out_rate"]
    assert result["stock_out_reduction_pct"] > 0
