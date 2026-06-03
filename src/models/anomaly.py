"""
Isolation Forest anomaly detector for retailer claim patterns.

Unsupervised — no labels used during training.
`is_anomalous` column in claims data is held out for evaluation only.

Anomaly score: the higher, the more anomalous.
Flag threshold set in model_config.yaml.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from src.features.claims import (
    load_claims,
    build_retailer_profiles,
    ANOMALY_FEATURE_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))["anomaly_detection"]


class RetailerAnomalyDetector:
    def __init__(self):
        ifcfg = CFG["isolation_forest"]
        self.model = IsolationForest(
            n_estimators=ifcfg["n_estimators"],
            contamination=ifcfg["contamination"],
            max_samples=ifcfg["max_samples"],
            random_state=ifcfg["random_state"],
            bootstrap=ifcfg["bootstrap"],
        )
        self.scaler = StandardScaler()
        self.threshold = CFG["flag_threshold"]
        self._fitted = False

    def fit(self, profiles: pd.DataFrame):
        X = profiles[ANOMALY_FEATURE_COLUMNS].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=ANOMALY_FEATURE_COLUMNS,
        )
        self.model.fit(X_scaled)
        self._fitted = True

    def score(self, profiles: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1] where higher = more anomalous.
        Isolation Forest raw score is in [-1, 0] (more negative = more anomalous).
        We flip and normalise to [0, 1].
        """
        assert self._fitted, "call fit() first"
        X = profiles[ANOMALY_FEATURE_COLUMNS].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=ANOMALY_FEATURE_COLUMNS,
        )
        raw = self.model.decision_function(X_scaled)   # negative = anomalous
        # Normalise to [0, 1]: flip sign, min-max scale
        flipped = -raw
        score = (flipped - flipped.min()) / (flipped.max() - flipped.min() + 1e-9)
        return score

    def predict(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Returns profiles with anomaly_score and is_flagged columns added.
        """
        scores = self.score(profiles)
        out = profiles.copy()
        out["anomaly_score"] = scores
        out["is_flagged"] = (scores >= self.threshold).astype(int)
        return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    def evaluate(self, results: pd.DataFrame, labels: pd.Series) -> dict:
        """
        Evaluates against ground-truth labels (held out, not used in training).
        labels: boolean Series aligned with results index.
        """
        y_true = labels.astype(int).values
        y_pred = results["is_flagged"].values

        # Precision@K — what fraction of top-K flagged are truly anomalous
        k = int(y_true.sum())   # flag same number as true anomalies
        top_k = results.head(k)
        top_k_labels = labels.iloc[:k].astype(int).values
        precision_at_k = top_k_labels.mean()

        return {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precision_at_k": round(float(precision_at_k), 4),
            "n_flagged": int(y_pred.sum()),
            "n_true_anomalies": int(y_true.sum()),
        }
