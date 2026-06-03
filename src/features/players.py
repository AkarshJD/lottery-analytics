"""
Feature engineering for player segmentation.

Aggregates raw transaction records to player-level RFM profiles.
K-Means and DBSCAN train on these profiles.

RFM dimensions:
  - Recency   : how recently did the player transact
  - Frequency : how often do they transact
  - Monetary  : how much do they spend

Extended with game preference and variety features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def load_transactions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / "player_transactions.parquet"
    df = pd.read_parquet(path)
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


def build_player_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction records to one row per player.
    Returns feature matrix ready for clustering.
    """
    snapshot_date = df["calendar_date"].max()
    grp = df.groupby("player_id")

    profiles = pd.DataFrame(index=grp.groups.keys())
    profiles.index.name = "player_id"

    # --- Recency ---
    profiles["days_since_last_txn"] = (
        snapshot_date - grp["calendar_date"].max()
    ).dt.days

    # --- Frequency ---
    profiles["total_transactions"] = grp["transaction_id"].count()
    profiles["active_days"] = grp["calendar_date"].nunique()

    date_span = (
        grp["calendar_date"].max() - grp["calendar_date"].min()
    ).dt.days.clip(lower=1)
    profiles["txn_per_active_day"] = (
        profiles["total_transactions"] / date_span
    )

    # --- Monetary ---
    profiles["total_spend"] = grp["transaction_amount"].sum()
    profiles["avg_spend_per_txn"] = grp["transaction_amount"].mean()
    profiles["max_single_txn"] = grp["transaction_amount"].max()
    profiles["spend_std"] = grp["transaction_amount"].std().fillna(0)

    # --- Game preference ---
    # Scratcher ratio — preference for instant vs draw games
    scratcher_txns = df[df["game_sub_type"] == "Scratcher"].groupby(
        "player_id"
    )["transaction_id"].count()
    profiles["scratcher_ratio"] = (
        scratcher_txns / profiles["total_transactions"]
    ).fillna(0)

    # Draw game ratio
    draw_types = ["Multi-State", "In-State", "Quick Draw"]
    draw_txns = df[df["game_sub_type"].isin(draw_types)].groupby(
        "player_id"
    )["transaction_id"].count()
    profiles["draw_game_ratio"] = (
        draw_txns / profiles["total_transactions"]
    ).fillna(0)

    # --- Variety ---
    profiles["unique_games"] = grp["game_name"].nunique()
    profiles["unique_game_types"] = grp["game_sub_type"].nunique()

    # --- Ticket price preference ---
    profiles["avg_ticket_price"] = grp["ticket_price"].mean()
    profiles["max_ticket_price"] = grp["ticket_price"].max()

    profiles = profiles.reset_index()
    for col in SEGMENT_FEATURE_COLUMNS:
        if col in profiles.columns:
            profiles[col] = profiles[col].astype("float64")

    return profiles


SEGMENT_FEATURE_COLUMNS = [
    "days_since_last_txn",
    "total_transactions",
    "active_days",
    "txn_per_active_day",
    "total_spend",
    "avg_spend_per_txn",
    "max_single_txn",
    "spend_std",
    "scratcher_ratio",
    "draw_game_ratio",
    "unique_games",
    "unique_game_types",
    "avg_ticket_price",
    "max_ticket_price",
]
