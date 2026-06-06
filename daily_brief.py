"""Daily Brief - Aggregerer alt brugeren skal vide til én oversigt"""
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yfinance as yf
from data_sources import fetch_data
from analysis import fundamental_score, technical_score, recommendation, get_indicators
from history import list_snapshots, load_snapshot, compare_snapshots
from crypto_data import fetch_crypto_data, fetch_fear_greed, fetch_global_crypto_market
from crypto_analysis import crypto_overall_score, crypto_recommendation


# ============ MARKET PULSE ============

def get_market_pulse():
    """Hent overordnet markedsstatus - 30 sekunders overblik"""
    pulse = {
        "stocks": {},
        "crypto": {},
        "fx": {},
        "regime": "Ukendt",
        "regime_color": "#888",
    }

    # 1. Aktiemarked: SPY (S&P 500), QQQ (Nasdaq), VIX (frygt-indeks)
    try:
        tickers = yf.Tickers("SPY QQQ ^VIX ^GSPC ^IXIC")

        for sym, label in [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("^VIX", "VIX")]:
            try:
                t = tickers.tickers[sym]
                hist = t.history(period="5d", auto_adjust=False)
                if not hist.empty:
                    last = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
                    change = (last / prev - 1) * 100 if prev else 0
                    pulse["stocks"][label] = {
                        "price": last,
                        "change_%": change,
                    }
            except Exception:
                pass
    except Exception as e:
        print(f"[market_pulse stocks] {e}")

    # 2. Krypto-marked
    try:
        global_crypto = fetch_global_crypto_market()
        if global_crypto:
            pulse["crypto"] = {
                "total_mc_t": (global_crypto.get("total_market_cap_usd") or 0) / 1e12,
                "change_24h": global_crypto.get("market_cap_change_24h") or 0,
                "btc_dom": global_crypto.get("btc_dominance") or 0,
            }
    except Exception:
        pass

    # 3. Fear & Greed
    try:
        fg_df = fetch_fear_greed()
        if fg_df is not None and not fg_df.empty:
            pulse["fear_greed"] = {
                "value": int(fg_df["value"].iloc[-1]),
                "label": fg_df["value_classification"].iloc[-1],
            }
    except Exception:
        pass

    # 4. FX rates
    try:
        fx = yf.Ticker("DKK=X")  # USD/DKK
        h = fx.history(period="2d", auto_adjust=False)
        if not h.empty:
            pulse["fx"]["USD/DKK"] = float(h["Close"].iloc[-1])
    except Exception:
        pass

    # 5. Markedsregime (overordnet)
    spy_change = pulse["stocks"].get("S&P 500", {}).get("change_%", 0)
    vix = pulse["stocks"].get("VIX", {}).get("price", 20)

    if vix < 15 and spy_change > 0:
        pulse["regime"] = "🟢 Risk-On (lav frygt, stigende)"
        pulse["regime_color"] = "#16a34a"
    elif vix > 25:
        pulse["regime"] = "🔴 Risk-Off (høj frygt)"
        pulse["regime_color"] = "#ef4444"
    elif vix > 20:
        pulse["regime"] = "🟡 Forsigtig (moderat frygt)"
        pulse["regime_color"] = "#eab308"
    else:
        pulse["regime"] = "🟢 Neutral/Bullish"
        pulse["regime_color"] = "#22c55e"

    return pulse


# ============ ANALYSER WATCHLIST ============

def _analyze_single_ticker(ticker):
    """Hurtig analyse af én ticker - bruges parallelt"""
    try:
        # Tjek om det er krypto
        from crypto_data import is_crypto, normalize_crypto_ticker
        if is_crypto(ticker):
            symbol = normalize_crypto_ticker(ticker)
            cdata = fetch_crypto_data(symbol)
            if not cdata:
                return None
            scores = crypto_overall_score(cdata["info"], cdata["hist"])
            rec, color = crypto_recommendation(scores["overall"])
            price = cdata["info"]["currentPrice"]
            change_24h = cdata["info"].get("change_24h", 0) or 0
            return {
                "ticker": ticker,
                "name": cdata["info"].get("longName", ticker),
                "type": "🪙 Krypto",
                "price": price,
                "currency": "USD",
                "change_%": change_24h,
                "score": scores["overall"],
                "recommendation": rec,
                "color": color,
                "sector": "Cryptocurrency",
            }

        # Almindelig aktie
        data = fetch_data(ticker)
        if not data:
            return None

        info = data["info"]
        hist = data["hist"]
        price = info.get("currentPrice") or float(hist["Close"].iloc[-1])
        prev = info.get("previousClose") or (
            float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        )
        change_pct = (price / prev - 1) * 100 if prev else 0

        df = get_indicators(hist).tail(252)
        f_score, _ = fundamental_score(info)
        t_score, _ = technical_score(df)
        overall = f_score * 0.6 + t_score * 0.4
        rec, color = recommendation(overall)

        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "type": "📈 Aktie",
            "price": price,
            "currency": info.get("currency", "USD"),
            "change_%": change_pct,
            "score": overall,
            "f_score": f_score,
            "t_score": t_score,
            "recommendation": rec,
            "color": color,
            "sector": info.get("sector", "?"),
        }

    except Exception as e:
        print(f"[_analyze_single_ticker] {ticker}: {e}")
        return None


def analyze_watchlist(tickers, max_workers=4):
    """Analyser hele watchlisten parallelt"""
    if not tickers:
        return pd.DataFrame()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_analyze_single_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    return df


# ============ RATING ÆNDRINGER ============

def get_recent_rating_changes(universe_filter=None, n_max=10):
    """
    Sammenlign de 2 nyeste snapshots og find rating-ændringer.
    """
    snaps = list_snapshots()
    if len(snaps) < 2:
        return None

    # Filtrer til samme univers hvis specificeret
    if universe_filter:
        snaps = [s for s in snaps if s["universe"] == universe_filter]

    if len(snaps) < 2:
        return None

    # Hent 2 nyeste
    df_now, ts_now, _ = load_snapshot(snaps[0]["file"])
    df_prev, ts_prev, _ = load_snapshot(snaps[1]["file"])

    cmp = compare_snapshots(df_now, df_prev)
    if cmp.empty or "rating_changed" not in cmp.columns:
        return None

    changed = cmp[cmp["rating_changed"] == True].copy()
    if changed.empty:
        return {
            "ts_now": ts_now,
            "ts_prev": ts_prev,
            "upgraded": pd.DataFrame(),
            "downgraded": pd.DataFrame(),
        }

    rec_order = ["STÆRKT KØB", "KØB", "HOLD", "SÆLG", "STÆRKT SÆLG"]
    changed["was_idx"] = changed["recommendation_prev"].apply(
        lambda x: rec_order.index(x) if x in rec_order else 99
    )
    changed["now_idx"] = changed["recommendation_now"].apply(
        lambda x: rec_order.index(x) if x in rec_order else 99
    )

    upgraded = changed[changed["now_idx"] < changed["was_idx"]].head(n_max)
    downgraded = changed[changed["now_idx"] > changed["was_idx"]].head(n_max)

    return {
        "ts_now": ts_now,
        "ts_prev": ts_prev,
        "upgraded": upgraded,
        "downgraded": downgraded,
    }


# ============ TOP OPPORTUNITIES ============

def get_top_opportunities(min_score=70, n_max=5):
    """Hent top KØB-muligheder fra seneste snapshot"""
    snaps = list_snapshots()
    if not snaps:
        return None

    df, ts, universe = load_snapshot(snaps[0]["file"])
    if df is None or df.empty:
        return None

    if "overall" not in df.columns:
        return None

    buys = df[df["overall"] >= min_score].sort_values("overall", ascending=False).head(n_max)
    return {
        "snapshot_ts": ts,
        "universe": universe,
        "buys": buys,
    }


# ============ POSITION SIZING (BONUS) ============

def calculate_position_size(price, stop_loss, portfolio_value=100000, risk_pct=2.0):
    """
    Beregn anbefalet antal aktier baseret på risk management.

    risk_pct: Hvor meget % af porteføljen er du villig til at tabe (default 2%)
    """
    if not price or not stop_loss or price <= 0 or stop_loss <= 0:
        return None

    risk_amount = portfolio_value * (risk_pct / 100)
    risk_per_share = abs(price - stop_loss)

    if risk_per_share <= 0:
        return None

    shares = int(risk_amount / risk_per_share)
    position_value = shares * price
    position_pct = (position_value / portfolio_value) * 100

    return {
        "shares": shares,
        "position_value": position_value,
        "position_pct": position_pct,
        "risk_amount": risk_amount,
        "risk_per_share": risk_per_share,
    }
