"""
Training orchestration for all models.

Champion model for forecasting: XGBoost Direct
Selected via 5-fold time-series CV across 7 models and 5 games.
Full comparison in experiments/compare_models.py.

Usage:
    python src/models/train.py --model forecasting
    python src/models/train.py --model segmentation
    python src/models/train.py --model anomaly
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))


def train_forecasting():
    from src.data.load import load_all_games
    from src.features.transforms import build_features, FEATURE_COLUMNS
    import yaml

    cfg = yaml.safe_load(open(ROOT / "configs" / "model_config.yaml"))
    xcfg = cfg["forecasting"]["xgboost"]

    games = load_all_games()
    mlflow.set_experiment("sales-forecasting")

    for game in sorted(games["game_name"].unique()):
        df = games[games["game_name"] == game].copy()
        df = df.sort_values("draw_date").reset_index(drop=True)

        cutoff = df["draw_date"].max() - np.timedelta64(90, "D")
        train = df[df["draw_date"] <= cutoff]
        test = df[df["draw_date"] > cutoff]

        print(f"\ntraining {game} — {len(train)} train, {len(test)} test rows")

        # Build features
        train_feat = build_features(train, game)
        # Prepend tail of train so lag features are valid for first test rows
        test_feat = build_features(
            pd.concat([train.tail(14), test]).reset_index(drop=True), game
        ).tail(len(test))

        valid_train = train_feat[FEATURE_COLUMNS].notna().all(axis=1)
        X_train = train_feat.loc[valid_train, FEATURE_COLUMNS]
        y_train = train.loc[valid_train, "draw_sales"]

        test_feat = test_feat.reset_index(drop=True)
        test_reset = test.reset_index(drop=True)
        valid_test = test_feat[FEATURE_COLUMNS].notna().all(axis=1)
        X_test = test_feat.loc[valid_test, FEATURE_COLUMNS]
        y_test = test_reset.loc[valid_test, "draw_sales"]

        with mlflow.start_run(run_name=game):
            mlflow.log_param("model", "xgboost_direct")
            mlflow.log_param("game", game)
            mlflow.log_param("train_rows", len(train))
            mlflow.log_param("test_rows", len(test))
            mlflow.log_param("train_start", str(train["draw_date"].min().date()))
            mlflow.log_param("train_end", str(train["draw_date"].max().date()))

            model = xgb.XGBRegressor(
                n_estimators=xcfg["n_estimators"],
                max_depth=xcfg["max_depth"],
                learning_rate=xcfg["learning_rate"],
                subsample=xcfg["subsample"],
                colsample_bytree=xcfg["colsample_bytree"],
                min_child_weight=xcfg["min_child_weight"],
                objective=xcfg["objective"],
                random_state=xcfg["random_state"],
                verbosity=0,
            )
            model.fit(X_train, y_train)

            from sklearn.metrics import mean_absolute_percentage_error
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_mape = mean_absolute_percentage_error(y_train, train_preds)
            test_mape = mean_absolute_percentage_error(y_test, test_preds)
            train_mae = float(np.mean(np.abs(y_train.values - train_preds)))
            test_mae = float(np.mean(np.abs(y_test.values - test_preds)))

            mlflow.log_metrics({
                "train_mape": round(train_mape, 4),
                "test_mape": round(test_mape, 4),
                "train_mae": round(train_mae, 2),
                "test_mae": round(test_mae, 2),
            })

            model_name = f"forecast-{game.lower().replace(' ', '-')}"
            mlflow.xgboost.log_model(
                model,
                artifact_path="xgb_model",
                registered_model_name=model_name,
                model_format="json",
                input_example=X_train.head(5),
            )

            print(f"  train MAPE: {train_mape:.2%}  |  test MAPE: {test_mape:.2%}")


def train_segmentation():
    from src.features.players import load_transactions, build_player_profiles
    from src.models.segmentation import PlayerSegmentation
    import mlflow.sklearn

    print("building player profiles...")
    txns = load_transactions()
    profiles = build_player_profiles(txns)
    print(f"  {len(profiles):,} players, {len(txns):,} transactions")

    mlflow.set_experiment("player-segmentation")

    with mlflow.start_run(run_name="kmeans-dbscan"):
        mlflow.log_param("model", "kmeans+dbscan")
        mlflow.log_param("n_players", len(profiles))
        mlflow.log_param("n_transactions", len(txns))
        mlflow.log_param("n_features", 14)
        mlflow.log_param("kmeans_k", 5)

        seg = PlayerSegmentation()
        seg.fit(profiles)

        results = seg.predict(profiles)
        metrics = seg.evaluate(profiles)
        summary = seg.profile_summary(profiles, results)

        mlflow.log_metrics({
            "silhouette_score": metrics["silhouette_score"],
            "kmeans_inertia": metrics["kmeans_inertia"],
            "n_dbscan_clusters": metrics["n_dbscan_clusters"],
            "n_outliers": metrics["n_outliers_dbscan"],
            "outlier_rate": metrics["outlier_rate"],
        })

        print(f"\n  silhouette score:  {metrics['silhouette_score']:.4f}")
        print(f"  kmeans inertia:    {metrics['kmeans_inertia']:,.0f}")
        print(f"  dbscan clusters:   {metrics['n_dbscan_clusters']}")
        print(f"  dbscan outliers:   {metrics['n_outliers_dbscan']} ({metrics['outlier_rate']:.1%})")
        print(f"\n  segment distribution:")
        for seg_name, count in results["segment_name"].value_counts().items():
            pct = count / len(results)
            print(f"    {seg_name:<25} {count:>7,}  ({pct:.1%})")

        print(f"\n  cohort profiles:")
        print(summary[["n_players", "total_spend", "avg_spend_per_txn",
                        "total_transactions", "days_since_last_txn"]].to_string())

        mlflow.sklearn.log_model(
            seg.kmeans,
            artifact_path="kmeans_model",
            registered_model_name="segmentation-player-kmeans",
        )


def train_anomaly():
    from src.features.claims import load_claims, build_retailer_profiles
    from src.models.anomaly import RetailerAnomalyDetector
    import mlflow.sklearn

    print("building retailer profiles...")
    claims = load_claims()
    profiles = build_retailer_profiles(claims)

    # Ground truth labels — held out, not used in training
    labels = claims.groupby("location_id")["is_anomalous"].max().reset_index()
    labels = profiles.merge(labels, on="location_id")["is_anomalous"]

    print(f"  {len(profiles)} retailers, {labels.sum()} true anomalies ({labels.mean():.1%})")

    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run(run_name="isolation-forest"):
        mlflow.log_param("model", "isolation_forest")
        mlflow.log_param("n_retailers", len(profiles))
        mlflow.log_param("contamination", 0.05)
        mlflow.log_param("n_features", 15)

        detector = RetailerAnomalyDetector()
        detector.fit(profiles)

        results = detector.predict(profiles)

        # Re-align labels to match sorted results order
        label_map = labels.values
        loc_order = results["location_id"].values
        profiles_indexed = profiles.set_index("location_id")
        labels_series = claims.groupby("location_id")["is_anomalous"].max()
        aligned_labels = pd.Series(
            [labels_series.get(loc, False) for loc in loc_order],
            name="is_anomalous"
        )

        metrics = detector.evaluate(results, aligned_labels)

        mlflow.log_metrics(metrics)

        print(f"\n  precision:     {metrics['precision']:.2%}")
        print(f"  recall:        {metrics['recall']:.2%}")
        print(f"  f1:            {metrics['f1']:.2%}")
        print(f"  precision@k:   {metrics['precision_at_k']:.2%}")
        print(f"  flagged:       {metrics['n_flagged']} / {len(profiles)} retailers")

        from src.features.claims import ANOMALY_FEATURE_COLUMNS
        from mlflow.models.signature import infer_signature
        from sklearn.preprocessing import StandardScaler

        input_example = profiles[ANOMALY_FEATURE_COLUMNS].head(3).astype("float64")
        X_sample = pd.DataFrame(
            detector.scaler.transform(input_example),
            columns=ANOMALY_FEATURE_COLUMNS,
        ).astype("float64")
        signature = infer_signature(X_sample)

        mlflow.sklearn.log_model(
            detector.model,
            artifact_path="isolation_forest",
            registered_model_name="anomaly-retailer-claims",
            signature=signature,
            input_example=X_sample,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["forecasting", "segmentation", "anomaly"],
        required=True,
    )
    args = parser.parse_args()

    if args.model == "forecasting":
        train_forecasting()
    elif args.model == "segmentation":
        train_segmentation()
    elif args.model == "anomaly":
        train_anomaly()


if __name__ == "__main__":
    main()
