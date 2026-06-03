"""
Model comparison experiment for draw-level sales forecasting.

Trains 6 models on the same time-series CV folds for each game and
logs results to MLflow. Justifies the Prophet + XGBoost ensemble choice.

Usage:
    python experiments/compare_models.py
    python experiments/compare_models.py --game "Fantasy 5"
"""

import argparse
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_all_games
from src.features.transforms import build_features, FEATURE_COLUMNS, GAME_CFG

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CV split — time-series aware
# ---------------------------------------------------------------------------

def time_series_splits(df: pd.DataFrame, n_splits: int = 5, horizon: int = 30):
    """
    Expanding window CV splits.
    Each fold: train on everything up to cutoff, test on next `horizon` rows.
    """
    df = df.sort_values("draw_date").reset_index(drop=True)
    n = len(df)
    min_train = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        cutoff = min_train * i
        train_idx = list(range(cutoff))
        test_idx = list(range(cutoff, min(cutoff + horizon, n)))
        if test_idx:
            splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict:
    valid = actuals > 0
    mape = mean_absolute_percentage_error(actuals[valid], preds[valid])
    mae = float(np.mean(np.abs(actuals[valid] - preds[valid])))
    return {"mape": round(mape, 4), "mae": round(mae, 2)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def rolling_mean_predict(train: pd.DataFrame, test: pd.DataFrame, window: int = 7) -> np.ndarray:
    last = train["draw_sales"].tail(window).mean()
    return np.full(len(test), last)


def linear_regression_predict(
    train: pd.DataFrame, test: pd.DataFrame, game_name: str
) -> np.ndarray:
    train_feat = build_features(train, game_name)
    test_feat = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    valid_train = train_feat[FEATURE_COLUMNS].notna().all(axis=1)
    X_train = train_feat.loc[valid_train, FEATURE_COLUMNS].values
    y_train = train.loc[valid_train, "draw_sales"].values

    valid_test = test_feat[FEATURE_COLUMNS].notna().all(axis=1)
    X_test = test_feat.loc[valid_test, FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = np.zeros(len(test))
    preds[valid_test.values] = model.predict(X_test)
    return preds


def prophet_only_predict(train: pd.DataFrame, test: pd.DataFrame, game_name: str) -> np.ndarray:
    gcfg = GAME_CFG.get(game_name, {})
    exp = gcfg.get("jackpot_exponent", 1.0)

    train_f = build_features(train, game_name)
    test_f = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    pdf_train = pd.DataFrame({
        "ds": train["draw_date"].values,
        "y": train["draw_sales"].values,
        "jackpot_power": train_f["jackpot_power"].values,
    })

    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.add_regressor("jackpot_power", standardize=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pdf_train)

    pdf_test = pd.DataFrame({
        "ds": test["draw_date"].values,
        "jackpot_power": test_f["jackpot_power"].values,
    })
    return m.predict(pdf_test)["yhat"].values


def xgboost_direct_predict(train: pd.DataFrame, test: pd.DataFrame, game_name: str) -> np.ndarray:
    train_f = build_features(train, game_name)
    test_f = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    valid_train = train_f[FEATURE_COLUMNS].notna().all(axis=1)
    X_train = train_f.loc[valid_train, FEATURE_COLUMNS]
    y_train = train.loc[valid_train, "draw_sales"]

    valid_test = test_f[FEATURE_COLUMNS].notna().all(axis=1)
    X_test = test_f.loc[valid_test, FEATURE_COLUMNS]

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = np.zeros(len(test))
    preds[valid_test.values] = model.predict(X_test)
    return preds


def lightgbm_direct_predict(train: pd.DataFrame, test: pd.DataFrame, game_name: str) -> np.ndarray:
    train_f = build_features(train, game_name)
    test_f = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    valid_train = train_f[FEATURE_COLUMNS].notna().all(axis=1)
    X_train = train_f.loc[valid_train, FEATURE_COLUMNS]
    y_train = train.loc[valid_train, "draw_sales"]

    valid_test = test_f[FEATURE_COLUMNS].notna().all(axis=1)
    X_test = test_f.loc[valid_test, FEATURE_COLUMNS]

    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = np.zeros(len(test))
    preds[valid_test.values] = model.predict(X_test)
    return preds


def arimax_predict(train: pd.DataFrame, test: pd.DataFrame, game_name: str) -> np.ndarray:
    """
    ARIMAX(1,1,1) with jackpot_power as exogenous regressor.
    Classical time series baseline — captures autocorrelation + jackpot effect.
    Order (1,1,1): one lag, first-difference for stationarity, one MA term.
    """
    train_f = build_features(train, game_name)
    test_f = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    exog_train = train_f[["jackpot_power"]].values
    exog_test = test_f[["jackpot_power"]].values
    y_train = train["draw_sales"].values.astype(float)

    model = ARIMA(
        y_train,
        exog=exog_train,
        order=(1, 1, 1),
    )
    fit = model.fit(method_kwargs={"warn_convergence": False})

    forecast = fit.forecast(steps=len(test), exog=exog_test)
    return np.clip(forecast, 0, None)


def prophet_xgb_predict(train: pd.DataFrame, test: pd.DataFrame, game_name: str) -> np.ndarray:
    """Full ensemble — Prophet + XGBoost residual correction."""
    train_f = build_features(train, game_name)
    test_f = build_features(
        pd.concat([train.tail(14), test]).reset_index(drop=True), game_name
    ).tail(len(test))

    pdf_train = pd.DataFrame({
        "ds": train["draw_date"].values,
        "y": train["draw_sales"].values,
        "jackpot_power": train_f["jackpot_power"].values,
    })

    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.add_regressor("jackpot_power", standardize=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pdf_train)

    prophet_train_pred = m.predict(pdf_train[["ds", "jackpot_power"]])["yhat"].values
    residuals = train["draw_sales"].values - prophet_train_pred

    valid_train = train_f[FEATURE_COLUMNS].notna().all(axis=1)
    X_train = train_f.loc[valid_train, FEATURE_COLUMNS]
    y_res = residuals[valid_train]

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_res)

    pdf_test = pd.DataFrame({
        "ds": test["draw_date"].values,
        "jackpot_power": test_f["jackpot_power"].values,
    })
    prophet_test_pred = m.predict(pdf_test)["yhat"].values

    valid_test = test_f[FEATURE_COLUMNS].notna().all(axis=1)
    xgb_correction = np.zeros(len(test))
    if valid_test.any():
        xgb_correction[valid_test.values] = xgb_model.predict(
            test_f.loc[valid_test, FEATURE_COLUMNS]
        )

    return prophet_test_pred + xgb_correction


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

MODELS = {
    "naive_rolling_mean": lambda tr, te, g: rolling_mean_predict(tr, te),
    "arimax_1_1_1": arimax_predict,
    "linear_regression": linear_regression_predict,
    "prophet_only": prophet_only_predict,
    "xgboost_direct": xgboost_direct_predict,
    "lightgbm_direct": lightgbm_direct_predict,
    "prophet_xgboost_ensemble": prophet_xgb_predict,
}


def run_comparison(game_name: str, df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  {game_name}  ({len(df)} draws)")
    print(f"{'='*60}")

    splits = time_series_splits(df, n_splits=5, horizon=30)
    mlflow.set_experiment("model-comparison")

    results = []

    for model_name, model_fn in MODELS.items():
        fold_mapes, fold_maes, fold_times = [], [], []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            t0 = time.perf_counter()
            try:
                preds = model_fn(train, test, game_name)
            except Exception as e:
                print(f"  {model_name} fold {fold_i+1} failed: {e}")
                continue
            elapsed = time.perf_counter() - t0

            actuals = test["draw_sales"].values
            metrics = compute_metrics(actuals, preds)
            fold_mapes.append(metrics["mape"])
            fold_maes.append(metrics["mae"])
            fold_times.append(elapsed)

        if not fold_mapes:
            continue

        cv_mape = float(np.mean(fold_mapes))
        cv_mae = float(np.mean(fold_maes))
        cv_time = float(np.mean(fold_times))

        results.append({
            "model": model_name,
            "cv_mape": cv_mape,
            "cv_mae": cv_mae,
            "avg_train_time_s": round(cv_time, 3),
        })

        with mlflow.start_run(run_name=f"{game_name}__{model_name}"):
            mlflow.log_param("game", game_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_folds", len(fold_mapes))
            mlflow.log_metrics({
                "cv_mape": cv_mape,
                "cv_mae": cv_mae,
                "avg_train_time_s": cv_time,
            })

        print(f"  {model_name:<30} MAPE={cv_mape:.2%}  MAE={cv_mae:>10,.0f}  time={cv_time:.2f}s")

    # Summary table
    results_df = pd.DataFrame(results).sort_values("cv_mape")
    print(f"\n  Ranking for {game_name}:")
    print(results_df.to_string(index=False))
    return results_df


def main(args):
    games_df = load_all_games()

    if args.game:
        games = [args.game]
    else:
        games = sorted(games_df["game_name"].unique())

    all_results = []
    for game in games:
        df = games_df[games_df["game_name"] == game].copy()
        df = df.sort_values("draw_date").reset_index(drop=True)
        res = run_comparison(game, df)
        res["game"] = game
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    print("\n\n=== OVERALL RANKING (mean CV MAPE across games) ===")
    overall = (
        combined.groupby("model")["cv_mape"]
        .mean()
        .sort_values()
        .reset_index()
    )
    overall.columns = ["model", "mean_cv_mape"]
    overall["mean_cv_mape"] = overall["mean_cv_mape"].map("{:.2%}".format)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compare forecasting models across games")
    parser.add_argument("--game", type=str, default=None,
                        help="run for a single game (default: all games)")
    main(parser.parse_args())
