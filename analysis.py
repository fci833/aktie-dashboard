"""Tekniske indikatorer, scoring, DCF, risk metrics, Monte Carlo"""
import numpy as np
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def safe(d, key, default=None):
    if d is None:
        return default
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def add_indicators(hist):
    df = hist.copy()
    df["SMA50"] = SMAIndicator(df["Close"], 50).sma_indicator()
    df["SMA200"] = SMAIndicator(df["Close"], 200).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    bb = BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch.stoch()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df


@st.cache_data(ttl=1800, show_spinner=False, max_entries=100)
def add_indicators_cached(cache_key: str, hist: pd.DataFrame):
    return add_indicators(hist)


def get_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Cached indikator-beregning baseret på data signatur"""
    if hist is None or hist.empty:
        return hist
    cache_key = f"{len(hist)}_{hist['Close'].iloc[-1]:.4f}_{hist.index[-1]}"
    return add_indicators_cached(cache_key, hist)


def fundamental_score(info):
    score, det = 50, []

    def add(s, l, v):
        nonlocal score
        score += s
        det.append({"label": l, "value": v, "impact": s})

    pe = safe(info, "trailingPE")
    if pe:
        if 0 < pe < 15: add(8, "✅ P/E lav", f"{pe:.2f}")
        elif pe < 25: add(3, "➖ P/E moderat", f"{pe:.2f}")
        elif pe < 40: add(-3, "⚠️ P/E høj", f"{pe:.2f}")
        else: add(-8, "❌ P/E meget høj", f"{pe:.2f}")
    pb = safe(info, "priceToBook")
    if pb:
        if pb < 1: add(6, "✅ P/B < 1", f"{pb:.2f}")
        elif pb < 3: add(3, "✅ P/B sund", f"{pb:.2f}")
        elif pb > 8: add(-5, "⚠️ P/B høj", f"{pb:.2f}")
    roe = safe(info, "returnOnEquity")
    if roe is not None:
        if roe > 0.20: add(10, "✅ Stærk ROE", f"{roe*100:.1f}%")
        elif roe > 0.10: add(5, "✅ God ROE", f"{roe*100:.1f}%")
        elif roe < 0: add(-10, "❌ Negativ ROE", f"{roe*100:.1f}%")
    de = safe(info, "debtToEquity")
    if de is not None:
        if de < 30: add(6, "✅ Lav gæld", f"{de:.0f}")
        elif de < 100: add(3, "✅ Moderat gæld", f"{de:.0f}")
        elif de > 200: add(-8, "❌ Høj gæld", f"{de:.0f}")
    pm = safe(info, "profitMargins")
    if pm is not None:
        if pm > 0.20: add(8, "✅ Stærk margin", f"{pm*100:.1f}%")
        elif pm > 0.05: add(3, "➖ Ok margin", f"{pm*100:.1f}%")
        elif pm < 0: add(-10, "❌ Underskud", f"{pm*100:.1f}%")
    rg = safe(info, "revenueGrowth")
    if rg is not None:
        if rg > 0.20: add(8, "✅ Eksplosiv vækst", f"{rg*100:.1f}%")
        elif rg > 0.10: add(5, "✅ Stærk vækst", f"{rg*100:.1f}%")
        elif rg > 0: add(2, "➖ Positiv vækst", f"{rg*100:.1f}%")
        else: add(-6, "⚠️ Faldende oms.", f"{rg*100:.1f}%")
    fcf = safe(info, "freeCashflow")
    if fcf:
        if fcf > 0: add(5, "✅ Positivt FCF", f"{fcf/1e9:.2f}B")
        else: add(-5, "❌ Negativt FCF", f"{fcf/1e9:.2f}B")
    cr = safe(info, "currentRatio")
    if cr:
        if cr > 1.5: add(3, "✅ Sund likviditet", f"{cr:.2f}")
        elif cr < 1: add(-4, "⚠️ Likviditetsrisiko", f"{cr:.2f}")
    return max(0, min(100, score)), det


def technical_score_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vektoriseret version - beregner score for hele serien på én gang"""
    n = len(df)
    score = np.full(n, 50.0)

    close = df["Close"].values
    sma50 = df["SMA50"].values
    sma200 = df["SMA200"].values
    rsi = df["RSI"].values
    macd = df["MACD"].values
    macd_sig = df["MACD_signal"].values
    stoch = df["STOCH_K"].values
    adx = df["ADX"].values

    cm = ~(np.isnan(sma50) | np.isnan(sma200))
    score[cm & (sma50 > sma200)] += 10
    score[cm & (sma50 <= sma200)] -= 10

    sm = ~np.isnan(sma200)
    score[sm & (close > sma200)] += 5
    score[sm & (close <= sma200)] -= 5

    rm = ~np.isnan(rsi)
    score[rm & (rsi < 30)] += 12
    score[rm & (rsi >= 30) & (rsi < 45)] += 5
    score[rm & (rsi >= 60) & (rsi < 70)] -= 5
    score[rm & (rsi >= 70)] -= 12

    mm = ~(np.isnan(macd) | np.isnan(macd_sig))
    score[mm & (macd > macd_sig)] += 8
    score[mm & (macd <= macd_sig)] -= 8

    sk = ~np.isnan(stoch)
    score[sk & (stoch < 20)] += 5
    score[sk & (stoch > 80)] -= 5

    am = ~np.isnan(adx) & (adx > 25)
    score[am] += 3

    if n > 21:
        mom = np.full(n, np.nan)
        mom[21:] = (close[21:] / close[:-21] - 1) * 100
        score[mom > 10] += 6
        score[(mom > 0) & (mom <= 10)] += 2
        score[mom < -10] -= 6

    score = np.clip(score, 0, 100)
    score[:200] = np.nan  # SMA200 ikke pålidelig før dag 200
    return pd.Series(score, index=df.index)


def technical_score(df):
    """Returnerer score + detaljer for SIDSTE dag"""
    if len(df) < 200:
        return 50, []
    scores = technical_score_vectorized(df)
    last_score = scores.iloc[-1]
    if pd.isna(last_score):
        last_score = 50

    last = df.iloc[-1]
    pris = last["Close"]
    det = []

    def add(s, l, v):
        det.append({"label": l, "value": v, "impact": s})

    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            add(10, "✅ Golden cross", "SMA50>SMA200")
        else:
            add(-10, "❌ Death cross", "SMA50<SMA200")
    if not np.isnan(last["SMA200"]):
        if pris > last["SMA200"]:
            add(5, "✅ Pris over SMA200", f"+{(pris/last['SMA200']-1)*100:.1f}%")
        else:
            add(-5, "⚠️ Pris under SMA200", f"{(pris/last['SMA200']-1)*100:.1f}%")
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < 30: add(12, "✅ RSI oversolgt", f"{rsi:.1f}")
        elif rsi < 45: add(5, "➕ RSI svag", f"{rsi:.1f}")
        elif rsi < 60: add(0, "➖ RSI neutral", f"{rsi:.1f}")
        elif rsi < 70: add(-5, "⚠️ RSI hævet", f"{rsi:.1f}")
        else: add(-12, "❌ RSI overkøbt", f"{rsi:.1f}")
    if not np.isnan(last["MACD"]):
        if last["MACD"] > last["MACD_signal"]:
            add(8, "✅ MACD bullish", f"{last['MACD']:.3f}")
        else:
            add(-8, "❌ MACD bearish", f"{last['MACD']:.3f}")
    if not np.isnan(last["STOCH_K"]):
        if last["STOCH_K"] < 20: add(5, "✅ Stoch oversold", f"{last['STOCH_K']:.1f}")
        elif last["STOCH_K"] > 80: add(-5, "⚠️ Stoch overbought", f"{last['STOCH_K']:.1f}")
    if not np.isnan(last["ADX"]) and last["ADX"] > 25:
        add(3, "✅ Stærk trend", f"ADX={last['ADX']:.1f}")
    if len(df) > 20:
        mom = (pris / df["Close"].iloc[-21] - 1) * 100
        if mom > 10: add(6, "✅ Stærkt momentum", f"+{mom:.1f}%")
        elif mom > 0: add(2, "➕ Pos. momentum", f"+{mom:.1f}%")
        elif mom < -10: add(-6, "⚠️ Neg. momentum", f"{mom:.1f}%")

    return int(last_score), det


def calculate_price_targets(df, current_price, fair_value=None):
    last = df.iloc[-1]
    atr = last["ATR"] if not np.isnan(last["ATR"]) else current_price * 0.02
    recent = df.tail(252) if len(df) > 252 else df
    week52_high = recent["High"].max()
    week52_low = recent["Low"].min()
    sma50 = last["SMA50"] if not np.isnan(last["SMA50"]) else current_price
    sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else current_price
    bb_low = last["BB_low"] if not np.isnan(last["BB_low"]) else current_price * 0.95
    bb_high = last["BB_high"] if not np.isnan(last["BB_high"]) else current_price * 1.05
    buy_zone_low = max(bb_low, sma200 - atr)
    buy_zone_high = min(sma50, current_price * 0.97)
    stop_loss = max(current_price - 2 * atr, sma200 - atr)
    if fair_value and fair_value > current_price:
        target = fair_value
    else:
        target = max(current_price * 1.20, week52_high)
    return {
        "buy_low": buy_zone_low, "buy_high": buy_zone_high,
        "stop_loss": stop_loss, "target_short": bb_high,
        "target_long": target, "week52_high": week52_high,
        "week52_low": week52_low, "atr": atr,
    }


def dcf_valuation(info, g, dr, tg, years=10):
    fcf = safe(info, "freeCashflow")
    shares = safe(info, "sharesOutstanding")
    if not fcf or not shares or fcf <= 0:
        return None
    cfs = []
    for y in range(1, years + 1):
        gy = g - (g - tg) * (y / years)
        fcf = fcf * (1 + gy)
        cfs.append(fcf / ((1 + dr) ** y))
    tv = (fcf * (1 + tg)) / (dr - tg)
    pv_tv = tv / ((1 + dr) ** years)
    ev = sum(cfs) + pv_tv
    debt = safe(info, "totalDebt", 0)
    cash = safe(info, "totalCash", 0)
    return (ev - debt + cash) / shares


def risk_metrics(hist):
    r = hist["Close"].pct_change().dropna()
    ann_r = r.mean() * 252
    ann_v = r.std() * np.sqrt(252)
    sharpe = (ann_r - 0.04) / ann_v if ann_v > 0 else 0
    ds = r[r < 0].std() * np.sqrt(252)
    sortino = (ann_r - 0.04) / ds if ds > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    var95 = np.percentile(r, 5)
    return {"ann_r": ann_r, "ann_v": ann_v, "sharpe": sharpe,
            "sortino": sortino, "max_dd": dd.min(),
            "var95": var95, "dd_series": dd}


def monte_carlo(hist, days=252, sims=300):
    """Vektoriseret Monte Carlo - 50x hurtigere end loop"""
    r = hist["Close"].pct_change().dropna()
    mu, sigma = r.mean(), r.std()
    lp = hist["Close"].iloc[-1]
    random_returns = np.random.normal(mu, sigma, size=(sims, days))
    out = lp * np.cumprod(1 + random_returns, axis=1)
    return out, lp


def recommendation(s):
    if s >= 75: return "🟢 STÆRKT KØB", "#16a34a"
    if s >= 60: return "🟢 KØB", "#22c55e"
    if s >= 45: return "🟡 HOLD", "#eab308"
    if s >= 30: return "🔴 SÆLG", "#ef4444"
    return "🔴 STÆRKT SÆLG", "#b91c1c"


def recommendation_label(s):
    if s >= 75: return "STÆRKT KØB"
    if s >= 60: return "KØB"
    if s >= 45: return "HOLD"
    if s >= 30: return "SÆLG"
    return "STÆRKT SÆLG"


def filter_by_days(hist, days):
    if hist is None or hist.empty:
        return hist
    cutoff = pd.Timestamp.now(tz=hist.index.tz) - pd.Timedelta(days=days)
    return hist[hist.index >= cutoff]


def filter_chart_period(hist, period):
    if hist is None or hist.empty or period == "max":
        return hist
    days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}.get(period, 1825)
    return filter_by_days(hist, days)
