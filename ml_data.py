"""
ml_data.py - ML Data Pipeline
==============================
Loads historical screener snapshots, computes forward returns,
and prepares feature matrices for ML training.

Main entry point: get_training_data(asset_class="stock")
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

import yfinance as yf
import streamlit as st

from history import list_snapshots, load_snapshot

# ==========================================
# CONFIG
# ==========================================

HORIZONS = [30, 90, 180]  # Forward return horizons in days

# Classification thresholds (annualized rates)
ANNUAL_BUY_THRESHOLD = 15.0   # > 15% annual = BUY
ANNUAL_SELL_THRESHOLD = -10.0  # < -10% annual = SELL

# Feature columns (must exist in snapshots)
FEATURE_COLUMNS_NUMERIC = [
    "f_score", "t_score", "overall",
    "rsi", "macd", "vs_sma200_%", "vs_52w_high_%", "atr_pct",
    "pe", "pb", "peg", "dividend_%", "profit_margin",
    "roe", "debt_equity", "dcf_upside_%",
    "change_%", "regime_confidence",
    "market_cap",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "sector", "country", "regime", "currency",
]


# ==========================================
# CLASSIFICATION HELPERS
# ==========================================

def get_class_thresholds(days: int) -> Tuple[float, float]:
    """Get BUY/SELL thresholds for given horizon (% return)."""
    factor = days / 365.25
    return ANNUAL_BUY_THRESHOLD * factor, ANNUAL_SELL_THRESHOLD * factor


# ==========================================
# PRICE FETCHING (cached)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history_for_ml(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Cached price fetch. Returns DataFrame with Close column or None."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        # Ensure timezone-naive
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist[["Close"]]
    except Exception:
        return None


def get_price_at_date(
    price_hist: pd.DataFrame,
    target_date: pd.Timestamp,
    max_lookahead: int = 7
) -> Optional[float]:
    """Find closing price on target_date or next trading day within max_lookahead."""
    if price_hist is None or price_hist.empty:
        return None

    target_ts = pd.Timestamp(target_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)

    future = price_hist[price_hist.index >= target_ts]
    if future.empty:
        return None

    days_diff = (future.index[0] - target_ts).days
    if days_diff > max_lookahead:
        return None

    return float(future["Close"].iloc[0])


# ==========================================
# SNAPSHOT LOADING
# ==========================================

def _is_crypto_universe(universe: str) -> bool:
    """Detect if universe contains crypto."""
    if not universe:
        return False
    u_lower = str(universe).lower()
    return any(k in u_lower for k in ["crypto", "krypto", "🪙", "btc", "coin"])


def load_all_snapshots(asset_class: str = "all") -> pd.DataFrame:
    """
    Load all screener snapshots into one DataFrame.

    Args:
        asset_class: "stock", "crypto", or "all"

    Returns:
        DataFrame with snapshot_ts and snapshot_universe columns added
    """
    snaps = list_snapshots()
    if not snaps:
        return pd.DataFrame()

    all_rows = []
    for snap in snaps:
        try:
            df, ts, universe = load_snapshot(snap["file"])
            if df is None or df.empty:
                continue

            # Filter by asset class
            is_crypto = _is_crypto_universe(universe)
            if asset_class == "crypto" and not is_crypto:
                continue
            if asset_class == "stock" and is_crypto:
                continue

            df = df.copy()
            df["snapshot_ts"] = pd.to_datetime(ts)
            df["snapshot_universe"] = universe
            all_rows.append(df)
        except Exception as e:
            print(f"⚠️ Skipping snapshot {snap.get('filename', '?')}: {e}")
            continue

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Only successful screenings
    if "status" in combined.columns:
        combined = combined[combined["status"] == "✅"].copy()

    return combined


# ==========================================
# FORWARD RETURNS
# ==========================================

def compute_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    progress_callback=None
) -> pd.DataFrame:
    """
    Add future_return_{h}d columns by fetching prices N days forward.

    Returns df with new columns: future_return_30d, future_return_90d, future_return_180d
    """
    if df.empty or "ticker" not in df.columns:
        return df

    df = df.copy()

    for h in horizons:
        df[f"future_return_{h}d"] = np.nan

    unique_tickers = df["ticker"].unique()
    price_cache = {}

    n_total = len(unique_tickers)
    for i, tk in enumerate(unique_tickers):
        price_cache[tk] = fetch_price_history_for_ml(tk, period="5y")
        if progress_callback:
            progress_callback(i + 1, n_total, tk)

    # Compute returns row by row
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        snap_date = row["snapshot_ts"]
        snap_price = row.get("price")

        if pd.isna(snap_price) or snap_price <= 0:
            continue

        prices = price_cache.get(ticker)
        if prices is None or prices.empty:
            continue

        for h in horizons:
            future_date = snap_date + timedelta(days=h)
            future_price = get_price_at_date(prices, future_date)

            if future_price is None or future_price <= 0:
                continue

            return_pct = (future_price / snap_price - 1) * 100
            df.at[idx, f"future_return_{h}d"] = return_pct

    return df


def add_classification_targets(
    df: pd.DataFrame,
    horizons: List[int] = HORIZONS
) -> pd.DataFrame:
    """Add target_class_{h}d columns: BUY/HOLD/SELL based on thresholds."""
    df = df.copy()
    for h in horizons:
        ret_col = f"future_return_{h}d"
        cls_col = f"target_class_{h}d"
        if ret_col not in df.columns:
            continue

        buy_th, sell_th = get_class_thresholds(h)

        def classify(r):
            if pd.isna(r):
                return None
            if r > buy_th:
                return "BUY"
            elif r < sell_th:
                return "SELL"
            return "HOLD"

        df[cls_col] = df[ret_col].apply(classify)

    return df


# ==========================================
# FEATURE PREPARATION
# ==========================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean & encode features for ML.
    Returns (prepared_df, feature_column_names).
    """
    df = df.copy()
    feat_cols = []

    # Numerical features
    for col in FEATURE_COLUMNS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            feat_cols.append(col)

    # Log-transform market_cap (it's super skewed)
    if "market_cap" in df.columns:
        df["log_market_cap"] = np.log1p(df["market_cap"].fillna(0).clip(lower=0))
        feat_cols.append("log_market_cap")

    # Derived features (signal interactions)
    if "f_score" in df.columns and "t_score" in df.columns:
        df["score_divergence"] = df["f_score"] - df["t_score"]
        feat_cols.append("score_divergence")

    if "rsi" in df.columns:
        df["rsi_extreme"] = ((df["rsi"] - 50).abs() / 50).clip(0, 1)
        feat_cols.append("rsi_extreme")

    # Categorical → one-hot
    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df, dummies], axis=1)
            feat_cols.extend(dummies.columns.tolist())

    # Keep id + features + targets
    id_cols = [c for c in ["ticker", "name", "snapshot_ts", "snapshot_universe"]
               if c in df.columns]
    target_cols = [c for c in df.columns
                   if c.startswith("future_return_") or c.startswith("target_class_")]

    out_cols = id_cols + feat_cols + target_cols
    out_cols = [c for c in out_cols if c in df.columns]

    return df[out_cols], feat_cols


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def get_training_data(
    asset_class: str = "stock",
    horizons: List[int] = HORIZONS,
    verbose: bool = True
) -> Dict:
    """
    Build complete training dataset.

    Returns dict with:
        - X_{h}d: feature matrix per horizon (rows with valid target only)
        - y_reg_{h}d: regression target (% return)
        - y_clf_{h}d: classification target (BUY/HOLD/SELL)
        - feature_columns: list of feature names
        - n_samples_{h}d: count per horizon
        - asset_class: str
    """
    if verbose:
        print(f"📊 Loading {asset_class} snapshots...")
    df = load_all_snapshots(asset_class=asset_class)

    if df.empty:
        return {"error": f"No {asset_class} snapshots found", "n_samples": 0}

    if verbose:
        print(f"  ✓ Found {len(df)} rows from {df['snapshot_ts'].nunique()} snapshots")

    if verbose:
        print("📈 Computing forward returns (this can take 1-2 min)...")
    df = compute_forward_returns(df, horizons=horizons)

    if verbose:
        print("🏷️ Adding classification targets...")
    df = add_classification_targets(df, horizons=horizons)

    if verbose:
        print("🧹 Preparing features...")
    df_prepared, feat_cols = prepare_features(df)

    result = {
        "asset_class": asset_class,
        "feature_columns": feat_cols,
        "horizons": horizons,
        "n_features": len(feat_cols),
        "total_rows_loaded": len(df_prepared),
    }

    # Per-horizon datasets (drop rows missing target)
    for h in horizons:
        ret_col = f"future_return_{h}d"
        cls_col = f"target_class_{h}d"

        if ret_col not in df_prepared.columns:
            continue

        valid = df_prepared.dropna(subset=[ret_col]).copy()
        if valid.empty:
            result[f"n_samples_{h}d"] = 0
            continue

        # Fill any remaining NaNs in features with 0
        X = valid[feat_cols].fillna(0).astype(float)

        result[f"X_{h}d"] = X
        result[f"y_reg_{h}d"] = valid[ret_col].astype(float)

        if cls_col in valid.columns:
            valid_clf = valid.dropna(subset=[cls_col])
            result[f"y_clf_{h}d"] = valid_clf[cls_col].astype(str)
            result[f"X_clf_{h}d"] = valid_clf[feat_cols].fillna(0).astype(float)

        result[f"n_samples_{h}d"] = len(valid)

        if verbose:
            class_dist = valid[cls_col].value_counts().to_dict() if cls_col in valid else {}
            print(f"  ✓ {h}d: {len(valid)} samples · classes={class_dist}")

    result["sample_data"] = df_prepared.head(5)

    return result


# ==========================================
# SUMMARY (for UI)
# ==========================================

def get_training_summary() -> dict:
    """Quick stats about available training data."""
    summary = {}
    for asset in ["stock", "crypto"]:
        try:
            df = load_all_snapshots(asset_class=asset)
            if df.empty:
                summary[asset] = {"snapshots": 0, "rows": 0, "tickers": 0}
                continue

            summary[asset] = {
                "snapshots": int(df["snapshot_ts"].nunique()) if "snapshot_ts" in df else 0,
                "rows": len(df),
                "tickers": int(df["ticker"].nunique()) if "ticker" in df else 0,
                "date_min": str(df["snapshot_ts"].min())[:10] if "snapshot_ts" in df else None,
                "date_max": str(df["snapshot_ts"].max())[:10] if "snapshot_ts" in df else None,
                "universes": list(df["snapshot_universe"].unique()) if "snapshot_universe" in df else [],
            }
        except Exception as e:
            summary[asset] = {"error": str(e)}

    return summary


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML DATA PIPELINE - TEST")
    print("=" * 70)

    # 1. Summary
    summary = get_training_summary()
    print("\n📊 Available data:")
    for asset, stats in summary.items():
        print(f"\n  {asset.upper()}:")
        for k, v in stats.items():
            if isinstance(v, list) and len(v) > 5:
                v = f"{v[:3]}... ({len(v)} total)"
            print(f"    {k}: {v}")

    # 2. Load stocks
    print("\n" + "=" * 70)
    print("LOADING STOCK TRAINING DATA")
    print("=" * 70)
    data = get_training_data(asset_class="stock")

    if "error" in data:
        print(f"\n❌ {data['error']}")
    else:
        print(f"\n✅ {data['n_features']} features ready")
        for h in HORIZONS:
            n = data.get(f"n_samples_{h}d", 0)
            print(f"   {h}d horizon: {n} valid samples")
