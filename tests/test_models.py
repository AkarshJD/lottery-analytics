"""
Tests for model fit/predict contracts — segmentation and anomaly detection.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.segmentation import PlayerSegmentation
from src.models.anomaly import RetailerAnomalyDetector
from src.features.players import SEGMENT_FEATURE_COLUMNS
from src.features.claims import ANOMALY_FEATURE_COLUMNS


EXPECTED_SEGMENT_NAMES = {"casual", "regular_scratcher", "draw_enthusiast", "high_value", "dormant"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_player_profiles(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "player_id": [f"P{i:05d}" for i in range(n)],
        "days_since_last_txn": rng.uniform(0, 1000, n),
        "total_transactions": rng.integers(1, 100, n).astype(float),
        "active_days": rng.integers(1, 80, n).astype(float),
        "txn_per_active_day": rng.uniform(0.5, 5, n),
        "total_spend": rng.uniform(10, 2000, n),
        "avg_spend_per_txn": rng.uniform(1, 50, n),
        "max_single_txn": rng.uniform(10, 200, n),
        "spend_std": rng.uniform(0, 100, n),
        "scratcher_ratio": rng.uniform(0, 1, n),
        "draw_game_ratio": rng.uniform(0, 1, n),
        "unique_games": rng.integers(1, 20, n).astype(float),
        "unique_game_types": rng.integers(1, 5, n).astype(float),
        "avg_ticket_price": rng.uniform(1, 30, n),
        "max_ticket_price": rng.uniform(5, 50, n),
    })
    return df


def make_retailer_profiles(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "location_id": [f"LOC_{i:05d}" for i in range(n)],
        "total_claims": rng.uniform(10, 500, n),
        "total_claim_amount": rng.uniform(1000, 100_000, n),
        "avg_claim_amount": rng.uniform(50, 2000, n),
        "median_claim_amount": rng.uniform(50, 1000, n),
        "max_claim_amount": rng.uniform(500, 10_000, n),
        "claim_amount_std": rng.uniform(0, 500, n),
        "large_claim_ratio": rng.uniform(0, 1, n),
        "avg_claim_hour": rng.uniform(8, 20, n),
        "claim_hour_std": rng.uniform(0, 5, n),
        "off_hours_ratio": rng.uniform(0, 0.1, n),
        "avg_days_since_draw": rng.uniform(0, 30, n),
        "min_days_since_draw": rng.uniform(0, 5, n),
        "same_day_claim_ratio": rng.uniform(0, 0.3, n),
        "claims_per_month": rng.uniform(1, 20, n),
        "top10_amount_ratio": rng.uniform(0.3, 1.0, n),
    })
    return df


# ---------------------------------------------------------------------------
# Segmentation tests
# ---------------------------------------------------------------------------

def test_segmentation_predict_output_columns():
    """predict() must return player_id, kmeans_segment, segment_name, is_outlier, dbscan_cluster."""
    profiles = make_player_profiles(50)
    seg = PlayerSegmentation()
    seg.fit(profiles)
    results = seg.predict(profiles)
    required = {"player_id", "kmeans_segment", "segment_name", "is_outlier", "dbscan_cluster"}
    assert required.issubset(results.columns)


def test_segmentation_predict_row_count():
    """predict() must return one row per player."""
    profiles = make_player_profiles(50)
    seg = PlayerSegmentation()
    seg.fit(profiles)
    results = seg.predict(profiles)
    assert len(results) == len(profiles)


def test_segmentation_segment_names_valid():
    """All segment names must be one of the 5 expected labels."""
    profiles = make_player_profiles(50)
    seg = PlayerSegmentation()
    seg.fit(profiles)
    results = seg.predict(profiles)
    unknown = set(results["segment_name"].unique()) - EXPECTED_SEGMENT_NAMES
    assert not unknown, f"Unexpected segment names: {unknown}"


def test_segmentation_is_outlier_binary():
    """is_outlier must be 0 or 1 only."""
    profiles = make_player_profiles(50)
    seg = PlayerSegmentation()
    seg.fit(profiles)
    results = seg.predict(profiles)
    assert results["is_outlier"].isin([0, 1]).all()


def test_segmentation_evaluate_returns_silhouette():
    """evaluate() must return a silhouette_score key."""
    profiles = make_player_profiles(50)
    seg = PlayerSegmentation()
    seg.fit(profiles)
    metrics = seg.evaluate(profiles)
    assert "silhouette_score" in metrics
    assert -1 <= metrics["silhouette_score"] <= 1


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------

def test_anomaly_scores_in_range():
    """Anomaly scores must be in [0, 1]."""
    profiles = make_retailer_profiles(50)
    detector = RetailerAnomalyDetector()
    detector.fit(profiles)
    scores = detector.score(profiles)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_anomaly_predict_has_required_columns():
    """predict() must return anomaly_score and is_flagged columns."""
    profiles = make_retailer_profiles(50)
    detector = RetailerAnomalyDetector()
    detector.fit(profiles)
    results = detector.predict(profiles)
    assert "anomaly_score" in results.columns
    assert "is_flagged" in results.columns


def test_anomaly_predict_row_count():
    """predict() must return one row per retailer."""
    profiles = make_retailer_profiles(50)
    detector = RetailerAnomalyDetector()
    detector.fit(profiles)
    results = detector.predict(profiles)
    assert len(results) == len(profiles)


def test_anomaly_fit_required_before_score():
    """score() must raise AssertionError if called before fit()."""
    detector = RetailerAnomalyDetector()
    profiles = make_retailer_profiles(10)
    with pytest.raises(AssertionError):
        detector.score(profiles)
