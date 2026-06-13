"""
ml_backfill.py - Historical Training Data Generator
=====================================================
Generates synthetic 'snapshots' from historical price data so you can
train ML immediately without waiting months for forward returns.

Strategy:
  1. For each ticker, fetch 3+ years of historical data
  2. Pick N historical 'snapshot dates' (e.g. every 30 days going back 2 years)
  3. At each historical date: compute scores AS IF we had screened then
  4. Compute forward returns using known future prices
  5. Build a complete training dataset

Main entry: build_backfill_dataset(tickers, asset_class="stock")
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import ta

HORIZONS = [30, 90, 180]


# ==========================================
# DEFAULT TICKER UNIVERSES (for backfill)
# ==========================================

DEFAULT_TICKERS = {
    "us_large_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA",
        "HD", "CVX", "MRK", "LLY", "ABBV", "AVGO", "PEP", "KO",
        "COST", "WMT", "MCD", "TMO", "ADBE", "CSCO", "ACN", "DIS",
    ],
    "us_growth": [
        "TSLA", "NVDA", "META", "GOOGL", "AMZN", "AMD", "CRM", "NFLX",
        "ADBE", "INTC", "PYPL", "SHOP", "SQ", "ROKU", "ZM",
    ],
    "us_dividend": [
        "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "VZ", "T", "MO",
        "ABBV", "PFE", "MRK", "MMM", "CAT", "MCD", "WMT", "HD",
    ],
    "european": [
        "ASML.AS", "SAP.DE", "NESN.SW", "ROG.SW", "NOVN.SW",
        "MC.PA", "OR.PA", "SAN.PA", "TTE.PA", "AIR.PA",
        "ULVR.L", "AZN.L", "SHEL.L", "BP.L", "HSBA.L",
        "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
        "XRP-USD", "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD",
        "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
    ],
}


# ==========================================
# PRICE FETCHING (cached & parallel)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_history(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Fetch 5 years of price data for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 250:
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Optional[dict]:
    """Get current ticker info (sector, market cap, etc)."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or "longName" not in info:
            return None
        return info
    except Exception:
        return None


# ==========================================
# HISTORICAL INDICATOR COMPUTATION
# ==========================================

def compute_indicators_at_date(
    hist: pd.DataFrame,
    target_date: pd.Timestamp,
    lookback_days: int = 365
) -> Optional[Dict]:
    """
    Compute technical indicators using ONLY data up to target_date.
    This is the key for backfill - no lookahead bias.
    """
    if hist is None or hist.empty:
        return None

    target_ts = pd.Timestamp(target_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)

    # Slice: only data up to target_date
    hist_slice = hist[hist.index <= target_ts].tail(lookback_days)
    if len(hist_slice) < 50:
        return None

    try:
        close = hist_slice["Close"]
        high = hist_slice["High"]
        low = hist_slice["Low"]

        rsi = ta.momentum.rsi(close, window=14).iloc[-1]
        macd_obj = ta.trend.MACD(close)
        macd = macd_obj.macd_diff().iloc[-1]
        adx = ta.trend.ADXIndicator(high, low, close).adx().iloc[-1]
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]
        bb = ta.volatility.BollingerBands(close)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]

        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        current_price = float(close.iloc[-1])
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())

        # Returns
        ret_30d = ((close.iloc[-1] / close.iloc[-30] - 1) * 100) if len(close) >= 30 else 0
        ret_90d = ((close.iloc[-1] / close.iloc[-90] - 1) * 100) if len(close) >= 90 else 0

        return {
            "price": current_price,
            "rsi": float(rsi) if not pd.isna(rsi) else 50,
            "macd": float(macd) if not pd.isna(macd) else 0,
            "adx": float(adx) if not pd.isna(adx) else 25,
            "atr": float(atr) if not pd.isna(atr) else 0,
            "atr_pct": (float(atr) / current_price * 100) if current_price > 0 and not pd.isna(atr) else 2,
            "bb_width": ((bb_high - bb_low) / current_price * 100) if current_price > 0 else 5,
            "vs_sma200_%": ((current_price / sma200 - 1) * 100) if sma200 else 0,
            "vs_sma50_%": ((current_price / sma50 - 1) * 100) if sma50 else 0,
            "vs_52w_high_%": ((current_price / high_52w - 1) * 100) if high_52w > 0 else 0,
            "vs_52w_low_%": ((current_price / low_52w - 1) * 100) if low_52w > 0 else 0,
            "ret_30d": ret_30d,
            "ret_90d": ret_90d,
            "change_%": ret_30d,  # short-term momentum proxy
        }
    except Exception:
        return None


def compute_simple_score(indicators: dict) -> dict:
    """
    Simple rule-based scoring (matches your existing system).
    Returns f_score, t_score, overall.
    """
    if not indicators:
        return {"f_score": 50, "t_score": 50, "overall": 50, "regime": "UNKNOWN"}

    # Technical score (0-100)
    t = 50.0
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        t += 15  # Oversold = good buy
    elif rsi < 40:
        t += 8
    elif rsi > 70:
        t -= 15
    elif rsi > 60:
        t -= 5

    macd = indicators.get("macd", 0)
    if macd > 0:
        t += 10
    elif macd < 0:
        t -= 10

    vs_sma200 = indicators.get("vs_sma200_%", 0)
    if vs_sma200 > 5:
        t += 8
    elif vs_sma200 < -10:
        t -= 8

    vs_52w = indicators.get("vs_52w_high_%", 0)
    if vs_52w < -25:
        t += 10  # Way below highs = potential value
    elif vs_52w > -5:
        t -= 5

    # Fundamental proxy (using momentum & volatility)
    f = 50.0
    ret_90d = indicators.get("ret_90d", 0)
    if ret_90d > 10:
        f += 12
    elif ret_90d > 0:
        f += 5
    elif ret_90d < -15:
        f -= 12
    elif ret_90d < -5:
        f -= 5

    atr_pct = indicators.get("atr_pct", 2)
    if atr_pct > 5:
        f -= 8  # Too volatile
    elif atr_pct < 1.5:
        f += 5  # Stable

    # Regime detection (based on SMA200 + momentum)
    if vs_sma200 > 5 and ret_90d > 5:
        regime = "BULL"
    elif vs_sma200 < -10 and ret_90d < -10:
        regime = "BEAR"
    elif atr_pct > 4:
        regime = "VOLATILE"
    else:
        regime = "SIDEWAYS"

    f = max(0, min(100, f))
    t = max(0, min(100, t))

    # Combined (regime-weighted)
    if regime == "BULL":
        overall = f * 0.4 + t * 0.6
    elif regime in ("BEAR", "VOLATILE"):
        overall = f * 0.7 + t * 0.3
    else:
        overall = f * 0.6 + t * 0.4

    return {
        "f_score": f,
        "t_score": t,
        "overall": overall,
        "regime": regime,
    }


# ==========================================
# BACKFILL CORE
# ==========================================

def get_price_at_date(hist: pd.DataFrame, target_date: pd.Timestamp,
                     max_lookahead: int = 7) -> Optional[float]:
    """Find close price on target_date or nearest forward trading day."""
    if hist is None or hist.empty:
        return None
    target_ts = pd.Timestamp(target_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)
    future = hist[hist.index >= target_ts]
    if future.empty:
        return None
    days_diff = (future.index[0] - target_ts).days
    if days_diff > max_lookahead:
        return None
    return float(future["Close"].iloc[0])


def generate_snapshots_for_ticker(
    ticker: str,
    snapshot_dates: List[pd.Timestamp],
    horizons: List[int] = HORIZONS,
) -> List[Dict]:
    """
    Generate synthetic snapshots at multiple historical dates for one ticker.
    Returns list of dicts (one per snapshot date).
    """
    hist = fetch_ticker_history(ticker, period="5y")
    if hist is None:
        return []

    info = fetch_ticker_info(ticker) or {}
    sector = info.get("sector", "Unknown")
    country = info.get("country", "Unknown")
    currency = info.get("currency", "USD")
    market_cap = info.get("marketCap", 0) or 0
    name = info.get("longName") or info.get("shortName") or ticker

    rows = []
    for snap_date in snapshot_dates:
        snap_ts = pd.Timestamp(snap_date)
        if snap_ts.tz is not None:
            snap_ts = snap_ts.tz_localize(None)

        # Need data BEFORE this date for indicators
        if hist.index.min() > snap_ts - timedelta(days=200):
            continue

        # Compute indicators using only past data
        indicators = compute_indicators_at_date(hist, snap_ts)
        if not indicators:
            continue

        scores = compute_simple_score(indicators)

        # Get snapshot price
        snap_price = get_price_at_date(hist, snap_ts)
        if snap_price is None or snap_price <= 0:
            continue

        # Compute forward returns
        future_returns = {}
        all_horizons_ok = False
        for h in horizons:
            future_date = snap_ts + timedelta(days=h)
            future_price = get_price_at_date(hist, future_date)
            if future_price is None or future_price <= 0:
                future_returns[f"future_return_{h}d"] = None
            else:
                future_returns[f"future_return_{h}d"] = (future_price / snap_price - 1) * 100
                all_horizons_ok = True

        # Skip if no forward returns at all (snapshot too recent)
        if not all_horizons_ok:
            continue

        row = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "country": country,
            "currency": currency,
            "market_cap": market_cap,
            "snapshot_ts": snap_ts,
            "snapshot_universe": "BACKFILL",
            "price": snap_price,
            "status": "✅",
            "f_score": scores["f_score"],
            "t_score": scores["t_score"],
            "overall": scores["overall"],
            "regime": scores["regime"],
            "regime_confidence": 75,  # default
            "rsi": indicators["rsi"],
            "macd": indicators["macd"],
            "atr_pct": indicators["atr_pct"],
            "vs_sma200_%": indicators["vs_sma200_%"],
            "vs_52w_high_%": indicators["vs_52w_high_%"],
            "change_%": indicators["change_%"],
            # Fill missing fundamental fields with defaults
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "peg": info.get("pegRatio"),
            "dividend_%": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else 0,
            "profit_margin": (info.get("profitMargins") or 0) * 100 if info.get("profitMargins") else 0,
            "roe": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else 0,
            "debt_equity": info.get("debtToEquity"),
            "dcf_upside_%": 0,  # not computed in backfill
            **future_returns,
        }
        rows.append(row)

    return rows


def generate_snapshot_dates(
    months_back: int = 24,
    interval_days: int = 30,
) -> List[pd.Timestamp]:
    """Generate snapshot dates going backwards in time."""
    today = pd.Timestamp.now().normalize()
    # Start at least 200 days ago (so we have time for 180d forward return)
    end_date = today - timedelta(days=200)
    start_date = today - timedelta(days=months_back * 30)

    dates = []
    current = end_date
    while current >= start_date:
        dates.append(current)
        current -= timedelta(days=interval_days)

    return sorted(dates)


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def build_backfill_dataset(
    tickers: Optional[List[str]] = None,
    asset_class: str = "stock",
    months_back: int = 24,
    snapshot_interval_days: int = 30,
    progress_callback=None,
    max_workers: int = 4,
) -> Dict:
    """
    Build a complete backfilled training dataset.

    Args:
        tickers: List of tickers (uses defaults if None)
        asset_class: "stock" or "crypto"
        months_back: How far back to generate snapshots (months)
        snapshot_interval_days: Days between snapshots
        progress_callback: callable(current, total, ticker)

    Returns dict with same shape as ml_data.get_training_data()
    """
    # Default tickers
    if tickers is None:
        if asset_class == "crypto":
            tickers = DEFAULT_TICKERS["crypto"]
        else:
            tickers = (
                DEFAULT_TICKERS["us_large_cap"]
                + DEFAULT_TICKERS["us_growth"][:10]
                + DEFAULT_TICKERS["european"][:10]
            )
            tickers = list(set(tickers))  # dedupe

    snapshot_dates = generate_snapshot_dates(months_back, snapshot_interval_days)
    if not snapshot_dates:
        return {"error": "No valid snapshot dates", "n_samples": 0}

    print(f"📅 Generating {len(snapshot_dates)} snapshot dates × {len(tickers)} tickers")

    all_rows = []
    n_total = len(tickers)
    completed = 0

    # Sequential to avoid yfinance rate limits
    for i, ticker in enumerate(tickers):
        try:
            rows = generate_snapshots_for_ticker(ticker, snapshot_dates)
            all_rows.extend(rows)
            completed += 1
            if progress_callback:
                progress_callback(completed, n_total, ticker)
        except Exception as e:
            print(f"⚠️ {ticker}: {e}")
            completed += 1
            if progress_callback:
                progress_callback(completed, n_total, ticker)
        time.sleep(0.1)  # gentle on yfinance

    if not all_rows:
        return {"error": "No data generated", "n_samples": 0}

    df = pd.DataFrame(all_rows)
    print(f"✅ Generated {len(df)} training rows from {df['ticker'].nunique()} tickers")

    return {
        "df": df,
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "n_snapshots": df["snapshot_ts"].nunique(),
        "asset_class": asset_class,
        "date_range": (df["snapshot_ts"].min(), df["snapshot_ts"].max()),
    }


# ==========================================
# SAVE TO SNAPSHOT FORMAT
# ==========================================

def save_backfill_as_snapshots(df: pd.DataFrame, snapshots_dir: str = "screener_snapshots"):
    """
    Convert backfill DataFrame into individual snapshot CSV files.
    This makes them compatible with your existing ml_data.get_training_data()
    """
    import os
    from pathlib import Path

    Path(snapshots_dir).mkdir(exist_ok=True)
    saved = 0

    for snap_ts in df["snapshot_ts"].unique():
        snap_df = df[df["snapshot_ts"] == snap_ts].copy()
        # Drop snapshot metadata cols (we save them in filename)
        snap_df = snap_df.drop(columns=["snapshot_ts", "snapshot_universe"], errors="ignore")

        ts = pd.Timestamp(snap_ts)
        filename = f"backfill_{ts.strftime('%Y%m%d')}_BACKFILL.csv"
        filepath = os.path.join(snapshots_dir, filename)

        snap_df.to_csv(filepath, index=False)
        saved += 1

    return saved


# ==========================================
# SESSION STATE INTEGRATION (Streamlit Cloud workaround)
# ==========================================

def store_backfill_in_session(df: pd.DataFrame):
    """Store backfill DataFrame in Streamlit session state."""
    try:
        import streamlit as st
        st.session_state["ml_backfill_df"] = df.copy()
        st.session_state["ml_backfill_ts"] = pd.Timestamp.now()
        return True
    except Exception:
        return False


def get_backfill_from_session():
    """Get cached backfill DataFrame from session state."""
    try:
        import streamlit as st
        return st.session_state.get("ml_backfill_df")
    except Exception:
        return None


def has_backfill_in_session() -> bool:
    """Check if backfill data is available in session."""
    try:
        import streamlit as st
        df = st.session_state.get("ml_backfill_df")
        return df is not None and not df.empty
    except Exception:
        return False


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML BACKFILL - HISTORICAL DATA GENERATOR")
    print("=" * 70)

    def cb(c, t, tk):
        print(f"  [{c}/{t}] {tk}")

    result = build_backfill_dataset(
        tickers=["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        months_back=18,
        snapshot_interval_days=30,
        progress_callback=cb,
    )

    if "error" in result:
        print(f"\n❌ {result['error']}")
    else:
        print(f"\n✅ Generated {result['n_rows']} rows")
        print(f"   Tickers: {result['n_tickers']}")
        print(f"   Snapshots: {result['n_snapshots']}")
        print(f"   Date range: {result['date_range']}")

        df = result["df"]
        print("\nForward returns coverage:")
        for h in HORIZONS:
            col = f"future_return_{h}d"
            if col in df.columns:
                valid = df[col].notna().sum()
                print(f"  {h}d: {valid}/{len(df)} valid")
