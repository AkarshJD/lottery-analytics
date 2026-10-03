"""
Inventory allocation backtest — the business-metric evaluation behind the
"improved inventory allocation, reduced stock-outs" claim.

Compares the ML allocation policy (system-wide XGBoost forecast x trailing
location share) against the naive pre-ML policy (each location's own
trailing rolling-mean sales), under an EQUAL total inventory budget, over
the last `--test-weeks` weeks of the location-level sales history.

Usage:
    python experiments/inventory_allocation_backtest.py
    python experiments/inventory_allocation_backtest.py --test-weeks 26
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.locations import load_location_sales
from src.models.inventory import backtest_allocation_policies

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))


def main(args):
    loc_df = load_location_sales()
    print(f"{loc_df['location_id'].nunique():,} locations, {len(loc_df):,} location-weeks")

    result = backtest_allocation_policies(loc_df, test_weeks=args.test_weeks)

    mlflow.set_experiment("inventory-allocation")
    with mlflow.start_run(run_name=f"naive-vs-ml-{args.test_weeks}wk"):
        mlflow.log_params({
            "test_weeks": args.test_weeks,
            "policy": "naive_rolling_mean_vs_xgboost_hierarchical",
        })
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})

    print(json.dumps(result, indent=2))
    print(
        f"\nAt an equal total inventory budget ({result['total_budget']:,.0f}), "
        f"the ML allocation policy reduces the location-week stock-out rate "
        f"from {result['naive_stock_out_rate']:.1%} to {result['ml_stock_out_rate']:.1%} "
        f"— a {result['stock_out_reduction_pct']:.1f}% relative reduction."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backtest ML vs naive inventory allocation")
    parser.add_argument("--test-weeks", type=int, default=52, help="holdout weeks (default 52)")
    main(parser.parse_args())
