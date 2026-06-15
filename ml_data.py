"""
ml_data.py - ML Data Pipeline (v2 — BALANCED THRESHOLDS)
==============================
Loads historical screener snapshots, computes forward returns,
and prepares feature matrices for ML training.

🆕 v2 FIXES:
- Balanced BUY/HOLD/SELL thresholds per horizon
- Wider HOLD zone → bedre klassebalance
- Optional percentile-based dynamic thresholds

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

HORIZONS = [30, 90, 180]

# 🆕 v2: BALANCED THRESHOLDS per horizon
# Disse er bredere end før, så HOLD-klassen får ~30% af samples
# Tidligere: 30d brugte ±1% (alt for tæt!) → kun 11% HOLD
# Nu: 30d bruger +5/-3% → ~30% HOLD
HORIZON_THRESHOLDS = {
    30:  {"buy": 5.0,  "sell": -3.0},   # 🆕 ±~3-5% er meningsfuldt for 30d
    90:  {"buy": 10.0, "sell": -6.0},   # 🆕 ±~6-10% for 90d
    180: {"buy": 15.0, "sell": -10.0},  # 🆕 ±~10-15% for 180d
}

# 🆕 Dynamic mode (percentile-based) — alternativ til faste thresholds
# Hvis True: top 35% = BUY, bottom 30% = SELL, middle 35% = HOLD
# Det sikrer PERFEKT klassebalance hver gang
USE_DYNAMIC_THRESHOLDS = False  # 🔧 Sæt til True for percentile-baseret labels

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
# CLASSIFICATION HELPERS (v2)
# ==========================================

def get_class_thresholds(days: int) -> Tuple[float, float]:
    """
    Get BUY/SELL thresholds for given horizon (% return).
    
    🆕 v2: Bruger HORIZON_THRESHOLDS dict i stedet for annualized factor.
    """
    if days in HORIZON_THRESHOLDS:
        thresh = HORIZON_THRESHOLDS[days]
        return thresh["buy"], thresh["sell"]
    
    # Fallback for ukendte horisonter: lineær interpolation
    if days <= 30:
        return 5.0, -3.0
    elif days <= 90:
        return 10.0, -6.0
    elif days <= 180:
        return 15.0, -10.0
    else:
        return 20.0, -12.0


def classify_returns_dynamic(returns: pd.Series) -> pd.Series:
    """
    🆕 v2: Percentile-based classification — sikrer perfekt balance.
    
    Top 35%   → BUY
    Middle 35% → HOLD
    Bottom 30% → SELL
    """
    valid = returns.dropna()
    if len(valid) < 10:
        return pd.Series([None] * len(returns), index=returns.index)
    
    buy_threshold = valid.quantile(0.65)   # Top 35%
    sell_threshold = valid.quantile(0.30)  # Bottom 30%
    
    def classify(r):
        if pd.isna(r):
            return None
        if r > buy_threshold:
            return "BUY"
        elif r < sell_threshold:
            return "SELL"
        return "HOLD"
    
    return returns.apply(classify)


# ==========================================
# PRICE FETCHING (UNCHANGED)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history_for_ml(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Cached price fetch."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist[["Close"]]
    except Exception:
        return None


def get_price_at_date(price_hist, target_date, max_lookahead=7):
    """Find closing price on target_date or next trading day."""
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
# SNAPSHOT LOADING (UNCHANGED)
# ==========================================

def _is_crypto_universe(universe: str) -> bool:
    if not universe:
        return False
    u_lower = str(universe).lower()
    return any(k in u_lower for k in ["crypto", "krypto", "🪙", "btc", "coin"])


def load_all_snapshots(asset_class: str = "all") -> pd.DataFrame:
    """Load all screener snapshots from session + disk."""
    all_rows = []

    # Session state backfill
    try:
        backfill_df = st.session_state.get("ml_backfill_df")
        if backfill_df is not None and not backfill_df.empty:
            df_bf = backfill_df.copy()
            if asset_class == "crypto":
                df_bf = df_bf[df_bf["ticker"].str.contains("-USD", na=False)]
            elif asset_class == "stock":
                df_bf = df_bf[~df_bf["ticker"].str.contains("-USD", na=False)]
            if not df_bf.empty:
                all_rows.append(df_bf)
    except Exception:
        pass

    # Disk snapshots
    try:
        snaps = list_snapshots()
        for snap in snaps:
            try:
                df, ts, universe = load_snapshot(snap["file"])
                if df is None or df.empty:
                    continue
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
                print(f"⚠️ Skipping snapshot: {e}")
                continue
    except Exception:
        pass

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    if "status" in combined.columns:
        combined = combined[combined["status"] == "✅"].copy()

    return combined


# ==========================================
# FORWARD RETURNS (UNCHANGED)
# ==========================================

def compute_forward_returns(df, horizons=HORIZONS, progress_callback=None):
    """Add future_return_{h}d columns."""
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


def add_classification_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """
    Add target_class_{h}d columns: BUY/HOLD/SELL.
    
    🆕 v2: Bruger nu BALANCED THRESHOLDS (eller dynamic percentiles).
    """
    df = df.copy()
    
    print(f"\n🏷️ Creating classification labels (v2 — balanced thresholds)")
    print(f"   Mode: {'DYNAMIC (percentiles)' if USE_DYNAMIC_THRESHOLDS else 'FIXED thresholds'}")
    
    for h in horizons:
        ret_col = f"future_return_{h}d"
        cls_col = f"target_class_{h}d"
        if ret_col not in df.columns:
            continue

        if USE_DYNAMIC_THRESHOLDS:
            # 🆕 Percentile-based (perfekt balance)
            df[cls_col] = classify_returns_dynamic(df[ret_col])
            print(f"   {h}d: dynamic percentiles (~35/35/30 split)")
        else:
            # 🆕 Fixed thresholds (mere intuitivt)
            buy_th, sell_th = get_class_thresholds(h)
            
            def classify(r, b=buy_th, s=sell_th):
                if pd.isna(r):
                    return None
                if r > b:
                    return "BUY"
                elif r < s:
                    return "SELL"
                return "HOLD"

            df[cls_col] = df[ret_col].apply(classify)
            
            # Log distribution
            dist = df[cls_col].value_counts().to_dict()
            total = sum(dist.values())
            if total > 0:
                pct_str = " | ".join(
                    f"{k}: {v} ({v/total*100:.0f}%)" 
                    for k, v in dist.items() if k is not None
                )
                print(f"   {h}d (BUY>{buy_th:+.1f}%, SELL<{sell_th:+.1f}%): {pct_str}")

    return df


# ==========================================
# FEATURE PREPARATION (UNCHANGED)
# ==========================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Clean & encode features for ML."""
    df = df.copy()
    feat_cols = []

    for col in FEATURE_COLUMNS_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            feat_cols.append(col)

    if "market_cap" in df.columns:
        df["log_market_cap"] = np.log1p(df["market_cap"].fillna(0).clip(lower=0))
        feat_cols.append("log_market_cap")

    if "f_score" in df.columns and "t_score" in df.columns:
        df["score_divergence"] = df["f_score"] - df["t_score"]
        feat_cols.append("score_divergence")

    if "rsi" in df.columns:
        df["rsi_extreme"] = ((df["rsi"] - 50).abs() / 50).clip(0, 1)
        feat_cols.append("rsi_extreme")

    for col in FEATURE_COLUMNS_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            df = pd.concat([df, dummies], axis=1)
            feat_cols.extend(dummies.columns.tolist())

    id_cols = [c for c in ["ticker", "name", "snapshot_ts", "snapshot_universe"]
               if c in df.columns]
    target_cols = [c for c in df.columns
                   if c.startswith("future_return_") or c.startswith("target_class_")]

    out_cols = id_cols + feat_cols + target_cols
    out_cols = [c for c in out_cols if c in df.columns]

    return df[out_cols], feat_cols


# ==========================================
# MAIN ENTRY POINT (UNCHANGED)
# ==========================================

def get_training_data(asset_class="stock", horizons=HORIZONS, verbose=True) -> Dict:
    """Build complete training dataset."""
    if verbose:
        print(f"📊 Loading {asset_class} snapshots...")
    df = load_all_snapshots(asset_class=asset_class)

    if df.empty:
        return {"error": f"No {asset_class} snapshots found", "n_samples": 0}

    if verbose:
        print(f"  ✓ Found {len(df)} rows from {df['snapshot_ts'].nunique()} snapshots")
        print("📈 Computing forward returns...")
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

    for h in horizons:
        ret_col = f"future_return_{h}d"
        cls_col = f"target_class_{h}d"

        if ret_col not in df_prepared.columns:
            continue

        valid = df_prepared.dropna(subset=[ret_col]).copy()
        if valid.empty:
            result[f"n_samples_{h}d"] = 0
            continue

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
# SUMMARY (UNCHANGED)
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
    print("ML DATA PIPELINE v2 - BALANCED THRESHOLDS")
    print("=" * 70)
    
    print("\n🎯 Class thresholds:")
    for h in HORIZONS:
        b, s = get_class_thresholds(h)
        print(f"   {h}d: BUY > {b:+.1f}%, SELL < {s:+.1f}%, HOLD between")
    
    summary = get_training_summary()
    print("\n📊 Available data:")
    for asset, stats in summary.items():
        print(f"\n  {asset.upper()}:")
        for k, v in stats.items():
            if isinstance(v, list) and len(v) > 5:
                v = f"{v[:3]}... ({len(v)} total)"
            print(f"    {k}: {v}")

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
