"""
Feature engineering for retailer claim anomaly detection.

Aggregates raw claim records to retailer-level behavioral profiles.
Isolation Forest trains on these profiles — no labels used.

Features capture three behavioral dimensions:
  - Volume    : how many claims, total amount
  - Timing    : what time of day, how soon after draw
  - Pattern   : consistency, concentration, outlier amounts
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def load_claims(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "retailer_claims.parquet"
    df = pd.read_parquet(path)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


def build_retailer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates claim records to one row per retailer.
    Returns feature matrix ready for Isolation Forest.
    """
    grp = df.groupby("location_id")

    profiles = pd.DataFrame(index=grp.groups.keys())
    profiles.index.name = "location_id"

    # --- Volume features ---
    profiles["total_claims"] = grp["claim_id"].count()
    profiles["total_claim_amount"] = grp["claim_amount"].sum()
    profiles["avg_claim_amount"] = grp["claim_amount"].mean()
    profiles["median_claim_amount"] = grp["claim_amount"].median()
    profiles["max_claim_amount"] = grp["claim_amount"].max()
    profiles["claim_amount_std"] = grp["claim_amount"].std().fillna(0)

    # Large claim ratio — fraction of claims above $1,000
    large = df[df["claim_amount"] > 1000].groupby("location_id")["claim_id"].count()
    profiles["large_claim_ratio"] = (large / profiles["total_claims"]).fillna(0)

    # --- Timing features ---
    profiles["avg_claim_hour"] = grp["claim_hour"].mean()
    profiles["claim_hour_std"] = grp["claim_hour"].std().fillna(0)

    # Off-hours ratio — claims filed between 10pm and 6am
    off_hours = df[df["claim_hour"].isin(range(22, 24)) | df["claim_hour"].isin(range(0, 6))]
    off_grp = off_hours.groupby("location_id")["claim_id"].count()
    profiles["off_hours_ratio"] = (off_grp / profiles["total_claims"]).fillna(0)

    # --- Draw proximity features ---
    profiles["avg_days_since_draw"] = grp["days_since_draw"].mean()
    profiles["min_days_since_draw"] = grp["days_since_draw"].min()

    # Same-day claim ratio — claimed on draw day itself
    same_day = df[df["days_since_draw"] == 0].groupby("location_id")["claim_id"].count()
    profiles["same_day_claim_ratio"] = (same_day / profiles["total_claims"]).fillna(0)

    # --- Concentration features ---
    # Claim frequency: claims per active month
    date_range = (df["calendar_date"].max() - df["calendar_date"].min()).days / 30
    profiles["claims_per_month"] = profiles["total_claims"] / max(date_range, 1)

    # Amount concentration: how much of total is top 10% of claims
    def top10_ratio(x):
        if len(x) < 2:
            return 0.0
        threshold = x.quantile(0.90)
        return x[x >= threshold].sum() / x.sum() if x.sum() > 0 else 0.0

    profiles["top10_amount_ratio"] = grp["claim_amount"].apply(top10_ratio)

    profiles = profiles.reset_index()
    for col in ANOMALY_FEATURE_COLUMNS:
        if col in profiles.columns:
            profiles[col] = profiles[col].astype("float64")
    return profiles


ANOMALY_FEATURE_COLUMNS = [
    "total_claims",
    "total_claim_amount",
    "avg_claim_amount",
    "median_claim_amount",
    "max_claim_amount",
    "claim_amount_std",
    "large_claim_ratio",
    "avg_claim_hour",
    "claim_hour_std",
    "off_hours_ratio",
    "avg_days_since_draw",
    "min_days_since_draw",
    "same_day_claim_ratio",
    "claims_per_month",
    "top10_amount_ratio",
]
