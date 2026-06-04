"""Markedsscreener - finder gode købsmuligheder"""
import time
import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_sources import fetch_data
from analysis import (
    get_indicators, fundamental_score, technical_score,
    dcf_valuation, recommendation_label, filter_by_days
)
from config import ANALYSIS_PERIODS


def analyze_ticker_for_screener(ticker):
    """Analyserer én ticker og returnerer screening-data"""
    try:
        data = fetch_data(ticker)
        if data is None:
            return {"ticker": ticker, "status": "❌ Ingen data", "overall": 0}

        info = data["info"]
        hist = data["hist"]
        if hist is None or len(hist) < 200:
            return {"ticker": ticker, "status": "❌ For lidt data", "overall": 0}

        df = get_indicators(hist)
        hist_tech = filter_by_days(hist, ANALYSIS_PERIODS["technical"])
        df_tech = df.tail(len(hist_tech))

        f_score, _ = fundamental_score(info)
        t_score, _ = technical_score(df_tech)
        overall = f_score * 0.6 + t_score * 0.4

        last = df.iloc[-1]
        pris = info.get("currentPrice") or last["Close"]
        prev = info.get("previousClose")
        change_pct = (pris / prev - 1) * 100 if prev else 0

        recent = df.tail(252) if len(df) > 252 else df
        w52_high = recent["High"].max()
        w52_low = recent["Low"].min()
        dist_high = (pris / w52_high - 1) * 100
        dist_low = (pris / w52_low - 1) * 100

        rsi = last["RSI"] if not np.isnan(last["RSI"]) else None
        sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else None
        above_sma200 = (pris / sma200 - 1) * 100 if sma200 else None

        # Quick DCF estimat (kun hvis FCF tilgængelig)
        fair_val = dcf_valuation(info, 0.10, 0.10, 0.025)
        upside = (fair_val / pris - 1) * 100 if fair_val and fair_val > 0 else None

        return {
            "ticker": ticker,
            "name": (info.get("longName") or ticker)[:30],
            "sector": info.get("sector", "?")[:20] if info.get("sector") else "?",
            "currency": info.get("currency", "USD"),
            "price": pris,
            "change_%": change_pct,
            "f_score": f_score,
            "t_score": t_score,
            "overall": overall,
            "recommendation": recommendation_label(overall),
            "rsi": rsi,
            "vs_sma200_%": above_sma200,
            "vs_52w_high_%": dist_high,
            "vs_52w_low_%": dist_low,
            "pe": info.get("trailingPE"),
            "dividend_%": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else None,
            "dcf_upside_%": upside,
            "status": "✅",
        }
    except Exception as e:
        return {"ticker": ticker, "status": f"❌ {str(e)[:50]}", "overall": 0}


def run_screener(tickers, min_score=60, max_workers=4, progress_callback=None):
    """
    Kører screener på en liste tickers parallelt.
    progress_callback(done, total, ticker) kaldes løbende.
    """
    results = []
    total = len(tickers)
    done = 0

    # Parallel fetch (forsigtigt - undgå rate limits)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(analyze_ticker_for_screener, t): t
            for t in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result(timeout=30)
            except Exception as e:
                result = {"ticker": ticker, "status": f"❌ Timeout: {e}", "overall": 0}
            results.append(result)
            done += 1
            if progress_callback:
                progress_callback(done, total, ticker)
            time.sleep(0.1)  # Lille pause mod rate limit

    df = pd.DataFrame(results)
    if df.empty:
        return df, df

    # Sortér: alle resultater + filtreret top
    df = df.sort_values("overall", ascending=False)
    df_buys = df[
        (df["overall"] >= min_score)
        & (df["status"] == "✅")
    ].copy()

    return df, df_buys


def categorize_opportunities(df_buys):
    """Kategoriserer købsmuligheder i forskellige typer"""
    if df_buys.empty:
        return {}

    categories = {}

    # 1. Stærke køb
    cat = df_buys[df_buys["overall"] >= 75]
    if not cat.empty:
        categories["🟢 Stærke køb (75+)"] = cat

    # 2. Oversolgte (RSI < 35)
    cat = df_buys[(df_buys["rsi"].notna()) & (df_buys["rsi"] < 35)]
    if not cat.empty:
        categories["📉 Oversolgte (RSI<35)"] = cat

    # 3. Tæt på 52w low (potentielle bargains)
    cat = df_buys[df_buys["vs_52w_low_%"] < 15]
    if not cat.empty:
        categories["💎 Bargain (tæt på 52w low)"] = cat

    # 4. Stærk DCF upside
    cat = df_buys[(df_buys["dcf_upside_%"].notna())
                  & (df_buys["dcf_upside_%"] > 30)]
    if not cat.empty:
        categories["💰 Undervurderet (DCF +30%)"] = cat

    # 5. Momentum (over SMA200 + høj score)
    cat = df_buys[(df_buys["vs_sma200_%"].notna())
                  & (df_buys["vs_sma200_%"] > 5)
                  & (df_buys["t_score"] >= 65)]
    if not cat.empty:
        categories["🚀 Momentum"] = cat

    # 6. Dividend plays
    cat = df_buys[(df_buys["dividend_%"].notna()) & (df_buys["dividend_%"] > 3)]
    if not cat.empty:
        categories["💸 Dividend (>3%)"] = cat

    return categories
# ... behold al eksisterende kode i screener.py ...

# Tilføj denne nye funktion til slutningen af filen:

def sector_breakdown(df_all):
    """Grupperer resultater per sektor og finder bedste pr. sektor"""
    if df_all is None or df_all.empty:
        return {}
    df = df_all[df_all["status"] == "✅"].copy()
    if df.empty or "sector" not in df.columns:
        return {}

    # Normalisér sektor-navne (fjern '?' og tomme)
    df["sector"] = df["sector"].fillna("Ukendt").replace("?", "Ukendt")

    sectors = {}
    for sector_name in df["sector"].unique():
        if not sector_name or sector_name == "Ukendt":
            continue
        sub = df[df["sector"] == sector_name].sort_values("overall", ascending=False)
        if len(sub) == 0:
            continue
        sectors[sector_name] = {
            "count": len(sub),
            "avg_score": sub["overall"].mean(),
            "best_score": sub["overall"].max(),
            "top_pick": sub.iloc[0],
            "all_stocks": sub,
        }
    # Sortér efter gennemsnitlig score (bedste sektorer først)
    return dict(sorted(sectors.items(),
                       key=lambda x: x[1]["avg_score"], reverse=True))
