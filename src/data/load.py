"""
Loads raw draw-level sales files and returns a unified dataframe.

Each file has columns: Jackpot Date, Draw Sales, Jackpot
Game name is inferred from the filename.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

GAME_FILE_MAP = {
    "Fantasy 5 jackpot sales by draw.xlsx": "Fantasy 5",
    "Mega Millions jackpot sales by draw.xlsx": "Mega Millions",
    "PowerBall_All.xlsx": "Powerball",
    "The Pick Jackpot sales by draw.xlsx": "The Pick",
    "Triple Twist Jackpot sales by draw.xlsx": "Triple Twist",
}


def load_raw(path: Path, game_name: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={
        "Jackpot Date": "draw_date",
        "Draw Sales": "draw_sales",
        "Jackpot": "jackpot_amount",
    })
    df["game_name"] = game_name
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df["draw_sales"] = pd.to_numeric(df["draw_sales"], errors="coerce")
    df["jackpot_amount"] = pd.to_numeric(df["jackpot_amount"], errors="coerce")
    df = df.dropna(subset=["draw_date", "draw_sales"])
    return df[["draw_date", "game_name", "draw_sales", "jackpot_amount"]]


def load_all_games(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Returns a single dataframe with all 5 games combined.
    Columns: draw_date, game_name, draw_sales, jackpot_amount
    """
    frames = []
    for filename, game_name in GAME_FILE_MAP.items():
        path = raw_dir / filename
        if not path.exists():
            print(f"warning: {filename} not found, skipping")
            continue
        df = load_raw(path, game_name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["game_name", "draw_date"]).reset_index(drop=True)
    return combined


def load_game(game_name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load a single game by name."""
    for filename, name in GAME_FILE_MAP.items():
        if name == game_name:
            return load_raw(raw_dir / filename, game_name)
    raise ValueError(f"unknown game: {game_name}. valid: {list(GAME_FILE_MAP.values())}")


if __name__ == "__main__":
    df = load_all_games()
    print(df.shape)
    print(df.groupby("game_name").agg(
        rows=("draw_date", "count"),
        date_min=("draw_date", "min"),
        date_max=("draw_date", "max"),
    ))
