"""
Data-fetching til metaller.
Bruger yfinance direkte da metaller/futures ikke er på Finnhub/Twelve.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


@st.cache_data(ttl=300, show_spinner=False)
def fetch_metal_data(ticker):
    """
    Henter data for et metal-instrument.
    Returnerer dict med info + hist + source, eller None ved fejl.
    """
    from metals_config import get_metal_info

    if not ticker:
        return None

    try:
        import yfinance as yf

        meta = get_metal_info(ticker)
        if not meta:
            return None

        tk = yf.Ticker(ticker)

        # Hent history (5 år for at have god base til teknisk analyse)
        hist = tk.history(period="5y", auto_adjust=True)

        if hist is None or hist.empty or len(hist) < 60:
            return None

        # Sikre alle nødvendige kolonner findes
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in hist.columns:
                if col == "Volume":
                    hist[col] = 0
                else:
                    return None

        # Hent yfinance info (best effort)
        try:
            yf_info = tk.info or {}
        except Exception:
            yf_info = {}

        current_price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price

        # 52-uger high/low
        recent_year = hist.tail(252) if len(hist) >= 252 else hist
        low_52 = float(recent_year["Low"].min())
        high_52 = float(recent_year["High"].max())

        # Momentum
        change_1d = (current_price / prev_close - 1) * 100 if prev_close > 0 else 0
        change_1w = _pct_change(hist, 5)
        change_1m = _pct_change(hist, 21)
        change_3m = _pct_change(hist, 63)
        change_6m = _pct_change(hist, 126)
        change_1y = _pct_change(hist, 252)
        change_3y = _pct_change(hist, 756)

        info = {
            "symbol": ticker,
            "longName": meta["name"],
            "shortName": meta["name"],
            "currency": "USD",
            "sector": meta["category"],
            "country": "Global",
            "category": meta["category"],
            "type": meta["type"],
            "unit": meta.get("unit", "USD"),
            "description": meta.get("description", ""),

            "currentPrice": current_price,
            "previousClose": prev_close,
            "fiftyTwoWeekHigh": high_52,
            "fiftyTwoWeekLow": low_52,

            "change_1d": change_1d,
            "change_1w": change_1w,
            "change_1m": change_1m,
            "change_3m": change_3m,
            "change_6m": change_6m,
            "change_1y": change_1y,
            "change_3y": change_3y,

            # Market cap for ETF'er/aktier
            "marketCap": yf_info.get("marketCap"),
            "totalAssets": yf_info.get("totalAssets"),
            "volume": yf_info.get("volume") or int(hist["Volume"].iloc[-1]),
            "averageVolume": yf_info.get("averageVolume") or int(hist["Volume"].tail(20).mean()),

            # Bruges IKKE for metaller (safety guards i smart_verdict)
            "trailingPE": None,
            "forwardPE": None,
            "returnOnEquity": None,
        }

        return {
            "info": info,
            "hist": hist,
            "source": "Yahoo Finance",
            "meta": meta,
        }

    except Exception as e:
        print(f"❌ fetch_metal_data({ticker}) fejlede: {e}")
        return None


def _pct_change(hist, days):
    """Beregner procent-ændring over N dage tilbage"""
    try:
        if len(hist) < days + 1:
            return None
        current = hist["Close"].iloc[-1]
        past = hist["Close"].iloc[-(days + 1)]
        if past > 0:
            return (current / past - 1) * 100
    except Exception:
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_metal_drivers():
    """
    Henter makro-drivere der påvirker metaller:
    - DXY (dollar index)
    - US 10y rente
    - VIX (fear)
    - Olie (inflation-proxy)
    - S&P 500 (risk on/off)
    """
    drivers = {}
    try:
        import yfinance as yf

        driver_tickers = {
            "DXY": ("DX-Y.NYB", "US Dollar Index"),
            "TNX": ("^TNX", "US 10-årig rente"),
            "VIX": ("^VIX", "Fear index"),
            "OIL": ("CL=F", "Crude oil"),
            "SPY": ("SPY", "S&P 500"),
        }

        for key, (ticker, name) in driver_tickers.items():
            try:
                data = yf.Ticker(ticker).history(period="3mo")
                if data.empty or len(data) < 20:
                    continue

                current = float(data["Close"].iloc[-1])
                change_1m = float((data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1) * 100)
                change_1w = float((data["Close"].iloc[-1] / data["Close"].iloc[-5] - 1) * 100) if len(data) >= 5 else 0

                drivers[key] = {
                    "name": name,
                    "ticker": ticker,
                    "current": current,
                    "change_1w": change_1w,
                    "change_1m": change_1m,
                }
            except Exception:
                continue
    except Exception:
        pass

    return drivers
