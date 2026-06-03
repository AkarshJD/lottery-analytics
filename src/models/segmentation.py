"""
Player segmentation using K-Means + DBSCAN.

K-Means: known k=5, finds compact spherical clusters.
DBSCAN:  density-based, no k needed, identifies outlier players
         that don't fit any cohort.

Both run on the same StandardScaled player profiles.
Final segments use K-Means labels (interpretable, stable).
DBSCAN flags are used to identify fringe/outlier players.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.features.players import SEGMENT_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))["segmentation"]


SEGMENT_NAMES = {
    0: "casual",
    1: "regular_scratcher",
    2: "draw_enthusiast",
    3: "high_value",
    4: "dormant",
}


class PlayerSegmentation:
    def __init__(self):
        kcfg = CFG["kmeans"]
        self.kmeans = KMeans(
            n_clusters=kcfg["n_clusters"],
            random_state=kcfg["random_state"],
            n_init=kcfg["n_init"],
            max_iter=kcfg["max_iter"],
        )

        dcfg = CFG["dbscan"]
        self.dbscan = DBSCAN(
            eps=dcfg["eps"],
            min_samples=dcfg["min_samples"],
            metric=dcfg["metric"],
        )

        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, profiles: pd.DataFrame):
        X = profiles[SEGMENT_FEATURE_COLUMNS].fillna(0)
        self.X_scaled = self.scaler.fit_transform(X)

        self.kmeans.fit(self.X_scaled)

        # DBSCAN on a 50K sample — full dataset is too slow at this scale
        sample_size = min(50_000, len(self.X_scaled))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(self.X_scaled), size=sample_size, replace=False)
        self.dbscan.fit(self.X_scaled[idx])
        self._dbscan_sample_idx = idx
        self._fitted = True

    def predict(self, profiles: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "call fit() first"
        X = profiles[SEGMENT_FEATURE_COLUMNS].fillna(0)
        X_scaled = self.scaler.transform(X)

        kmeans_labels = self.kmeans.predict(X_scaled)

        # Name clusters once for all 5, then map — not once per player
        profiles_reset = profiles.reset_index(drop=True)
        cluster_name_map = self._build_cluster_name_map(profiles_reset, kmeans_labels)

        # DBSCAN on 20K sample only
        sample_size = min(20_000, len(X_scaled))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
        dbscan_labels_sample = self.dbscan.fit_predict(X_scaled[idx])
        dbscan_labels = np.full(len(X_scaled), -2)   # -2 = unscored
        dbscan_labels[idx] = dbscan_labels_sample

        out = profiles[["player_id"]].copy().reset_index(drop=True)
        out["kmeans_segment"] = kmeans_labels
        out["segment_name"] = out["kmeans_segment"].map(cluster_name_map)
        out["is_outlier"] = (dbscan_labels == -1).astype(int)
        out["dbscan_cluster"] = dbscan_labels

        return out

    def _build_cluster_name_map(self, profiles: pd.DataFrame, labels: np.ndarray) -> dict:
        """
        Assigns unique names to each cluster based on rank of key features.
        Each name is assigned to exactly one cluster.
        """
        n_clusters = self.kmeans.n_clusters
        feat = profiles[SEGMENT_FEATURE_COLUMNS].fillna(0).copy()
        feat["_label"] = labels

        # Compute mean of key features per cluster
        stats = feat.groupby("_label").agg(
            avg_spend=("total_spend", "mean"),
            avg_recency=("days_since_last_txn", "mean"),
            avg_freq=("total_transactions", "mean"),
            avg_scratcher=("scratcher_ratio", "mean"),
            avg_draw=("draw_game_ratio", "mean"),
        )

        name_map = {}
        remaining = set(stats.index)

        # Assign in priority order — each cluster gets at most one name
        # dormant:           highest recency
        dormant = stats.loc[list(remaining), "avg_recency"].idxmax()
        name_map[dormant] = "dormant"
        remaining.remove(dormant)

        # high_value:        highest total spend
        hv = stats.loc[list(remaining), "avg_spend"].idxmax()
        name_map[hv] = "high_value"
        remaining.remove(hv)

        # draw_enthusiast:   highest draw game ratio
        de = stats.loc[list(remaining), "avg_draw"].idxmax()
        name_map[de] = "draw_enthusiast"
        remaining.remove(de)

        # regular_scratcher: highest scratcher ratio
        rs = stats.loc[list(remaining), "avg_scratcher"].idxmax()
        name_map[rs] = "regular_scratcher"
        remaining.remove(rs)

        # casual:            whatever is left
        for c in remaining:
            name_map[c] = "casual"

        return name_map

    def _name_segment(self, cluster_id: int, profiles: pd.DataFrame, labels: np.ndarray) -> str:
        """Legacy single-cluster namer — kept for compatibility."""
        return "unknown"

    def evaluate(self, profiles: pd.DataFrame) -> dict:
        assert self._fitted, "call fit() first"
        X = profiles[SEGMENT_FEATURE_COLUMNS].fillna(0)
        X_scaled = self.scaler.transform(X)
        labels = self.kmeans.predict(X_scaled)

        sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
        inertia = float(self.kmeans.inertia_)

        sample_size = min(50_000, len(X_scaled))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
        dbscan_labels = self.dbscan.fit_predict(X_scaled[idx])
        n_outliers = int((dbscan_labels == -1).sum())
        n_dbscan_clusters = len(set(dbscan_labels) - {-1})

        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        return {
            "silhouette_score": round(float(sil), 4),
            "kmeans_inertia": round(inertia, 2),
            "n_kmeans_clusters": int(self.kmeans.n_clusters),
            "n_dbscan_clusters": n_dbscan_clusters,
            "n_outliers_dbscan": n_outliers,
            "outlier_rate": round(n_outliers / len(profiles), 4),
            "cluster_sizes": cluster_sizes.to_dict(),
        }

    def profile_summary(self, profiles: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """Returns mean feature values per segment — the cohort profiles."""
        merged = profiles.merge(results[["player_id", "segment_name"]], on="player_id")
        summary = merged.groupby("segment_name")[SEGMENT_FEATURE_COLUMNS].mean().round(2)
        summary["n_players"] = merged.groupby("segment_name")["player_id"].count()
        return summary
