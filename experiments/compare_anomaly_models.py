"""
Model comparison experiment for retailer claim anomaly detection.

Trains 5 unsupervised anomaly detectors on the same retailer profiles
and evaluates against held-out ground truth labels.

Primary metric: Precision@K (top-K flagged retailers, K = true anomaly count)
Secondary: F1, Precision, Recall, runtime

Usage:
    python experiments/compare_anomaly_models.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.claims import load_claims, build_retailer_profiles, ANOMALY_FEATURE_COLUMNS

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def scale(profiles: pd.DataFrame) -> np.ndarray:
    X = profiles[ANOMALY_FEATURE_COLUMNS].fillna(0).astype("float64")
    return StandardScaler().fit_transform(X)


def normalize_scores(raw: np.ndarray) -> np.ndarray:
    """Flip and min-max normalize so higher = more anomalous."""
    flipped = -raw
    rng = flipped.max() - flipped.min()
    return (flipped - flipped.min()) / (rng + 1e-9)


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.6) -> dict:
    y_pred = (scores >= threshold).astype(int)
    y_true = labels.astype(int)

    k = int(y_true.sum())
    top_k_idx = np.argsort(scores)[::-1][:k]
    precision_at_k = float(y_true[top_k_idx].mean())

    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision_at_k": round(precision_at_k, 4),
        "n_flagged": int(y_pred.sum()),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def run_isolation_forest(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        bootstrap=False,
    )
    model.fit(X)
    return normalize_scores(model.decision_function(X))


def run_local_outlier_factor(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        metric="euclidean",
    )
    model.fit_predict(X)
    return normalize_scores(model.negative_outlier_factor_)


def run_one_class_svm(X: np.ndarray) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X)
    return normalize_scores(model.decision_function(X))


def run_elliptic_envelope(X: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    model = EllipticEnvelope(
        contamination=contamination,
        random_state=42,
        support_fraction=0.9,
    )
    model.fit(X)
    return normalize_scores(model.decision_function(X))


def run_dbscan_outlier(X: np.ndarray) -> np.ndarray:
    """
    DBSCAN noise points (-1) as anomaly scores.
    Score = 1 for noise points, 0 for cluster members.
    Within noise points, rank by distance to nearest cluster.
    """
    from sklearn.neighbors import NearestNeighbors

    model = DBSCAN(eps=0.5, min_samples=50, metric="euclidean")
    labels = model.fit_predict(X)

    scores = (labels == -1).astype(float)

    # Refine: among noise points, rank by distance to nearest non-noise neighbor
    if scores.sum() > 0 and (scores == 0).sum() > 0:
        nn = NearestNeighbors(n_neighbors=5).fit(X[labels != -1])
        dists, _ = nn.kneighbors(X)
        dist_scores = dists.mean(axis=1)
        # Combine: noise flag + normalized distance
        dist_norm = (dist_scores - dist_scores.min()) / (dist_scores.max() - dist_scores.min() + 1e-9)
        scores = scores * 0.7 + dist_norm * 0.3

    return scores


MODELS = {
    "isolation_forest": run_isolation_forest,
    "local_outlier_factor": run_local_outlier_factor,
    "one_class_svm": run_one_class_svm,
    "elliptic_envelope": run_elliptic_envelope,
    "dbscan_outlier": run_dbscan_outlier,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("loading retailer profiles...")
    claims = load_claims()
    profiles = build_retailer_profiles(claims)
    labels = claims.groupby("location_id")["is_anomalous"].max()
    labels = profiles["location_id"].map(labels).fillna(False).values.astype(int)

    print(f"  {len(profiles)} retailers, {labels.sum()} true anomalies ({labels.mean():.1%})\n")

    X = scale(profiles)

    mlflow.set_experiment("anomaly-model-comparison")

    results = []

    print(f"{'Model':<25} {'P@K':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Flagged':>8} {'Time':>6}")
    print("-" * 65)

    for model_name, model_fn in MODELS.items():
        t0 = time.perf_counter()
        try:
            scores = model_fn(X)
        except Exception as e:
            print(f"{model_name:<25} FAILED: {e}")
            continue
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(scores, labels)
        metrics["avg_time_s"] = round(elapsed, 3)

        results.append({"model": model_name, **metrics})

        print(
            f"{model_name:<25} "
            f"{metrics['precision_at_k']:>6.2%} "
            f"{metrics['f1']:>6.2%} "
            f"{metrics['precision']:>6.2%} "
            f"{metrics['recall']:>6.2%} "
            f"{metrics['n_flagged']:>8} "
            f"{elapsed:>5.2f}s"
        )

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_retailers", len(profiles))
            mlflow.log_metrics({
                "precision_at_k": metrics["precision_at_k"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "n_flagged": metrics["n_flagged"],
                "runtime_s": elapsed,
            })

    print("\n=== RANKING by Precision@K ===")
    df = pd.DataFrame(results).sort_values("precision_at_k", ascending=False)
    print(df[["model", "precision_at_k", "f1", "precision", "recall", "avg_time_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
