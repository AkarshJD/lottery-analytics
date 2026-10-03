"""
Inventory allocation policy comparison — the business-metric backtest
behind the "improved inventory allocation, reduced stock-outs" claim.

A stock-out occurs when a location's actual demand exceeds the inventory
par level allocated to it. Two policies are compared on the same held-out
weeks under an EQUAL total inventory budget, so any improvement comes from
allocation accuracy, not from simply carrying more stock:

  naive policy — par level per location = that location's own trailing
                 rolling-mean sales (the status-quo policy before any
                 system-level model existed)
  ml policy    — par level per location = XGBoost system-wide forecast x
                 trailing rolling share (top-down hierarchical allocation)
"""

import yaml
from pathlib import Path

import pandas as pd

from src.features.locations import (
    LOC_CFG,
    add_week_index,
    build_location_shares,
    build_system_features,
    build_system_series,
)
from src.models.location_forecast import SystemDemandForecaster

ROOT = Path(__file__).resolve().parents[2]
INV_CFG = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))["inventory_policy"]


def _naive_forecast(loc_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Each location's own trailing rolling-mean sales — the pre-ML baseline."""
    df = add_week_index(loc_df).sort_values(["location_id", "week_index"]).reset_index(drop=True)
    df["naive_forecast"] = df.groupby("location_id")["weekly_sale_amount"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return df[["location_id", "week_index", "weekly_sale_amount", "naive_forecast"]]


def backtest_allocation_policies(
    loc_df: pd.DataFrame,
    forecaster: SystemDemandForecaster = None,
    test_weeks: int = None,
    naive_window: int = None,
    share_window: int = None,
    budget_multiplier: float = None,
) -> dict:
    """
    Walk-forward comparison of the naive and ML allocation policies over
    the last `test_weeks` weeks. Returns stock-out rates for both and the
    relative reduction.
    """
    test_weeks = test_weeks or LOC_CFG["cv"]["test_weeks"]
    naive_window = naive_window or INV_CFG["naive_window"]
    share_window = share_window or LOC_CFG["share_window"]
    budget_multiplier = budget_multiplier or INV_CFG["budget_multiplier"]
    forecaster = forecaster or SystemDemandForecaster()

    system = build_system_series(loc_df)
    system_feat = build_system_features(system)

    cutoff = int(system_feat["week_index"].max()) - test_weeks
    train_feat = system_feat[system_feat["week_index"] <= cutoff]
    test_feat = system_feat[system_feat["week_index"] > cutoff].copy()

    forecaster.fit(train_feat)
    test_feat["predicted_system_sales"] = forecaster.predict(test_feat)

    # --- ML policy: system forecast x trailing share ---
    shares = build_location_shares(loc_df, window=share_window)
    shares_test = shares[shares["week_index"] > cutoff]
    ml = shares_test.merge(
        test_feat[["week_index", "predicted_system_sales"]], on="week_index", how="inner"
    )
    ml["raw_forecast"] = ml["share"] * ml["predicted_system_sales"]

    # --- Naive policy: location's own trailing rolling mean ---
    naive_all = _naive_forecast(loc_df, window=naive_window)
    naive = naive_all[naive_all["week_index"] > cutoff]

    merged = ml.merge(
        naive[["location_id", "week_index", "naive_forecast"]],
        on=["location_id", "week_index"],
        how="inner",
    )
    merged = merged.dropna(subset=["raw_forecast", "naive_forecast"])

    # Equal total-budget calibration — scale each policy so total allocated
    # inventory matches the SAME budget. The comparison is then purely
    # about where the (equally-sized) pool of stock gets placed.
    actual_total = merged["weekly_sale_amount"].sum()
    budget = actual_total * budget_multiplier

    ml_scale = budget / merged["raw_forecast"].sum()
    naive_scale = budget / merged["naive_forecast"].sum()

    merged["ml_par"] = merged["raw_forecast"] * ml_scale
    merged["naive_par"] = merged["naive_forecast"] * naive_scale

    merged["ml_stock_out"] = (merged["weekly_sale_amount"] > merged["ml_par"]).astype(int)
    merged["naive_stock_out"] = (merged["weekly_sale_amount"] > merged["naive_par"]).astype(int)

    ml_rate = float(merged["ml_stock_out"].mean())
    naive_rate = float(merged["naive_stock_out"].mean())
    reduction = (naive_rate - ml_rate) / naive_rate if naive_rate > 0 else 0.0

    return {
        "n_location_weeks": int(len(merged)),
        "test_weeks": test_weeks,
        "total_budget": round(float(budget), 2),
        "naive_stock_out_rate": round(naive_rate, 4),
        "ml_stock_out_rate": round(ml_rate, 4),
        "stock_out_reduction_pct": round(reduction * 100, 2),
        "ml_scale_factor": round(float(ml_scale), 4),
        "naive_scale_factor": round(float(naive_scale), 4),
        "system_mape": forecaster.evaluate(test_feat)["mape"],
    }
