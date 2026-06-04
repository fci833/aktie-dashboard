"""Persistent storage af screener-resultater for sammenligning over tid"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

HISTORY_DIR = Path("screener_history")
HISTORY_DIR.mkdir(exist_ok=True)


def _safe_filename(s: str) -> str:
    """Lav et sikkert filnavn fra universe-navn"""
    return "".join(c if c.isalnum() else "_" for c in s)[:50]


def save_snapshot(df_all: pd.DataFrame, universe_name: str) -> Path:
    """Gem screener-resultat som JSON med timestamp"""
    if df_all is None or df_all.empty:
        return None
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d_%H%M")
    filename = HISTORY_DIR / f"{date_str}__{_safe_filename(universe_name)}.json"

    # Konvertér timestamps i data til strenge
    df_clean = df_all.copy()
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(str)

    snapshot = {
        "timestamp": timestamp.isoformat(),
        "universe": universe_name,
        "data": df_clean.to_dict(orient="records"),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, default=str, ensure_ascii=False, indent=2)
    return filename


def list_snapshots(universe_name: str = None) -> list:
    """List alle tilgængelige snapshots, evt. filtreret på univers"""
    snapshots = []
    for file in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if universe_name is None or data.get("universe") == universe_name:
                snapshots.append({
                    "file": str(file),
                    "filename": file.name,
                    "timestamp": data["timestamp"],
                    "universe": data.get("universe", "?"),
                    "n_tickers": len(data.get("data", [])),
                })
        except Exception:
            continue
    return snapshots


def load_snapshot(filepath) -> tuple:
    """Indlæs et specifikt snapshot. Returnerer (df, timestamp, universe)"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["data"])
    return df, data["timestamp"], data["universe"]


def compare_snapshots(df_current: pd.DataFrame,
                      df_previous: pd.DataFrame) -> pd.DataFrame:
    """Sammenlign to snapshots og find ændringer i rating/score/pris"""
    if df_previous is None or df_previous.empty:
        return pd.DataFrame()
    if df_current is None or df_current.empty:
        return pd.DataFrame()

    cols_now = ["ticker", "name", "overall", "recommendation", "price", "f_score", "t_score"]
    cols_prev = ["ticker", "overall", "recommendation", "price"]
    cols_now = [c for c in cols_now if c in df_current.columns]
    cols_prev = [c for c in cols_prev if c in df_previous.columns]

    merged = df_current[cols_now].merge(
        df_previous[cols_prev],
        on="ticker", how="outer", suffixes=("_now", "_prev")
    )

    if "overall_now" in merged and "overall_prev" in merged:
        merged["score_change"] = merged["overall_now"] - merged["overall_prev"]
    if "price_now" in merged and "price_prev" in merged:
        merged["price_change_%"] = (merged["price_now"] / merged["price_prev"] - 1) * 100
    if "recommendation_now" in merged and "recommendation_prev" in merged:
        merged["rating_changed"] = (
            merged["recommendation_now"] != merged["recommendation_prev"]
        )

    return merged


def get_hot_stocks(universe_name: str, n_days: int = 5,
                   min_change: float = 5.0) -> pd.DataFrame:
    """Find aktier med stigende score over de sidste N snapshots"""
    snapshots = list_snapshots(universe_name)
    if len(snapshots) < 2:
        return pd.DataFrame()

    snapshots = snapshots[:n_days]
    latest_df, latest_ts, _ = load_snapshot(snapshots[0]["file"])
    oldest_df, oldest_ts, _ = load_snapshot(snapshots[-1]["file"])

    if "ticker" not in latest_df.columns or "ticker" not in oldest_df.columns:
        return pd.DataFrame()

    cols_l = [c for c in ["ticker", "name", "overall", "recommendation",
                          "price", "sector"] if c in latest_df.columns]
    cols_o = [c for c in ["ticker", "overall", "price"] if c in oldest_df.columns]

    merged = latest_df[cols_l].merge(
        oldest_df[cols_o], on="ticker", how="inner", suffixes=("_now", "_old")
    )
    if "overall_now" not in merged or "overall_old" not in merged:
        return pd.DataFrame()

    merged["score_change"] = merged["overall_now"] - merged["overall_old"]
    if "price_now" in merged and "price_old" in merged:
        merged["price_change_%"] = (merged["price_now"] / merged["price_old"] - 1) * 100

    risers = merged[merged["score_change"] >= min_change].sort_values(
        "score_change", ascending=False
    )
    fallers = merged[merged["score_change"] <= -min_change].sort_values(
        "score_change", ascending=True
    )

    return {
        "risers": risers,
        "fallers": fallers,
        "latest_ts": latest_ts,
        "oldest_ts": oldest_ts,
        "n_snapshots": len(snapshots),
    }


def get_score_history(ticker: str, n_days: int = 30) -> pd.DataFrame:
    """Hent score-historik for én ticker over de sidste N snapshots"""
    history = []
    for snap in list_snapshots()[:n_days]:
        try:
            df, ts, universe = load_snapshot(snap["file"])
            row = df[df["ticker"] == ticker]
            if not row.empty:
                history.append({
                    "timestamp": ts,
                    "universe": universe,
                    "score": row.iloc[0].get("overall"),
                    "recommendation": row.iloc[0].get("recommendation"),
                    "price": row.iloc[0].get("price"),
                })
        except Exception:
            continue
    return pd.DataFrame(history).sort_values("timestamp") if history else pd.DataFrame()


def cleanup_old_snapshots(keep_days: int = 60):
    """Slet snapshots ældre end N dage"""
    cutoff = datetime.now().timestamp() - (keep_days * 86400)
    deleted = 0
    for file in HISTORY_DIR.glob("*.json"):
        if file.stat().st_mtime < cutoff:
            file.unlink()
            deleted += 1
    return deleted
