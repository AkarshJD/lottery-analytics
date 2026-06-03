# lottery-analytics

End-to-end retail ticket sales analytics pipeline: time-series forecasting, player segmentation, and anomaly detection on retailer claim patterns.

## Stack

| Layer | Tools |
|---|---|
| Forecasting | Prophet + XGBoost ensemble |
| Segmentation | K-Means + DBSCAN |
| Anomaly detection | Isolation Forest |
| Experiment tracking | MLflow |
| Serving | FastAPI + Pydantic |
| Tests | pytest |

## Quickstart

macOS prerequisite: `brew install libomp`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Train models, serve:

```bash
python src/models/train.py --model forecasting
python src/models/train.py --model segmentation
python src/models/train.py --model anomaly
uvicorn src.serving.app:app --reload
```

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Tests:

```bash
pytest
```

## Data schema

> Data is not included in this repository. Place parquet files in `data/raw/` before training.


### daily_sales.parquet
| Column | Type | Description |
|---|---|---|
| `calendar_date` | date | Transaction date |
| `fiscal_year` | int | FY starts July 1 (Jul 2023 = FY2024) |
| `fiscal_quarter` | int | Q1=Jul-Sep, Q2=Oct-Dec, Q3=Jan-Mar, Q4=Apr-Jun |
| `fiscal_month` | int | 1=Jan … 12=Dec |
| `fiscal_week` | int | ISO week number (1–53) |
| `game_name` | str | Game identifier |
| `game_sub_type` | str | Age Controlled / Charity / Fast Play / In-State / Instant Tabs / Multi-State / Quick Draw / Scratcher |
| `ticket_price` | float | USD ticket price |
| `jackpot_amount` | float | Current jackpot (0 for non-jackpot games) |
| `total_sale_amount` | float | Daily revenue for this game |

### location_weekly_sales.parquet
| Column | Type | Description |
|---|---|---|
| `fiscal_year` | int | |
| `fiscal_week` | int | ISO week |
| `location_id` | str | Retail location (LOC_00001 … LOC_03000) |
| `weekly_sale_amount` | float | Weekly revenue at this location |

### player_transactions.parquet
| Column | Type | Description |
|---|---|---|
| `transaction_id` | str | Unique transaction hash |
| `player_id` | str | Player identifier |
| `calendar_date` | date | Purchase date |
| `game_name` | str | |
| `game_sub_type` | str | |
| `ticket_price` | float | |
| `quantity` | int | Tickets purchased |
| `transaction_amount` | float | Total spend |

### retailer_claims.parquet
| Column | Type | Description |
|---|---|---|
| `claim_id` | str | Unique claim hash |
| `location_id` | str | Retailer |
| `calendar_date` | date | Claim date |
| `claim_hour` | int | Hour claim was filed |
| `claim_amount` | float | Prize amount claimed |
| `days_since_draw` | int | Days between draw and claim |

## Project structure

```
lottery-analytics/
  configs/
    data_config.yaml
    model_config.yaml
  src/
    data/
    features/
    models/
    serving/
  tests/
  experiments/
  data/raw/          (not included)
  data/processed/    (not included)
```
