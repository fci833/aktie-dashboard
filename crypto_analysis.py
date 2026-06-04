"""Krypto-analyse: Komplet med targets, risk, MC, backtest"""
import numpy as np
import pandas as pd
from datetime import datetime
from crypto_data import fetch_fear_greed


# ===== SCORING (4-FAKTOR MODEL) =====

def crypto_market_score(info):
    score = 50.0
    details = []

    rank = info.get("marketCapRank")
    if rank:
        if rank <= 10:
            score += 15
            details.append({"label": f"Top 10 (rank #{rank})", "impact": 15})
        elif rank <= 30:
            score += 10
            details.append({"label": f"Top 30 (rank #{rank})", "impact": 10})
        elif rank <= 100:
            score += 5
            details.append({"label": f"Top 100 (rank #{rank})", "impact": 5})
        else:
            score -= 10
            details.append({"label": f"Lavt rank (#{rank})", "impact": -10})

    ath_change = info.get("ath_change_%", 0) or 0
    if ath_change < -80:
        score += 20
        details.append({"label": "80%+ under ATH (deep value)", "impact": 20})
    elif ath_change < -60:
        score += 15
        details.append({"label": "60%+ under ATH", "impact": 15})
    elif ath_change < -40:
        score += 10
        details.append({"label": "40%+ under ATH", "impact": 10})
    elif ath_change > -10:
        score -= 15
        details.append({"label": "Tæt på ATH (overheated)", "impact": -15})
    elif ath_change > -25:
        score -= 5
        details.append({"label": "25% under ATH", "impact": -5})

    circ = info.get("circulating_supply") or 0
    max_sup = info.get("max_supply")
    if max_sup and circ > 0:
        ratio = circ / max_sup
        if ratio > 0.95:
            score += 15
            details.append({"label": f"95%+ supply mined", "impact": 15})
        elif ratio > 0.85:
            score += 10
            details.append({"label": f"85%+ supply mined", "impact": 10})
        elif ratio < 0.3:
            score -= 10
            details.append({"label": f"Højt fremtidigt udbud", "impact": -10})
    elif not max_sup:
        score -= 5
        details.append({"label": "Inflationær", "impact": -5})

    mc = info.get("marketCap", 0) or 0
    vol = info.get("totalVolume", 0) or 0
    if mc > 0 and vol > 0:
        liq_ratio = vol / mc
        if liq_ratio > 0.15:
            score += 10
            details.append({"label": "Høj likviditet", "impact": 10})
        elif liq_ratio < 0.02:
            score -= 10
            details.append({"label": "Lav likviditet", "impact": -10})

    return max(0, min(100, score)), details


def crypto_technical_score(df):
    if df is None or len(df) < 50:
        return 50.0, [{"label": "Utilstrækkelig data", "impact": 0}]

    score = 50.0
    details = []
    close = df["Close"]

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    last = close.iloc[-1]

    if sma200 and last > sma200:
        score += 10
        details.append({"label": "Pris > SMA200 (bull)", "impact": 10})
    elif sma200:
        score -= 10
        details.append({"label": "Pris < SMA200 (bear)", "impact": -10})

    if last > sma20 > sma50:
        score += 10
        details.append({"label": "Golden cross-tendens", "impact": 10})
    elif last < sma20 < sma50:
        score -= 5
        details.append({"label": "Death cross-tendens", "impact": -5})

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    if not np.isnan(rsi):
        if rsi < 30:
            score += 15
            details.append({"label": f"RSI oversold ({rsi:.0f})", "impact": 15})
        elif rsi < 40:
            score += 5
            details.append({"label": f"RSI lav ({rsi:.0f})", "impact": 5})
        elif rsi > 70:
            score -= 15
            details.append({"label": f"RSI overbought ({rsi:.0f})", "impact": -15})
        elif rsi > 60:
            score -= 5
            details.append({"label": f"RSI høj ({rsi:.0f})", "impact": -5})

    if len(close) >= 30:
        mom_30 = (close.iloc[-1] / close.iloc[-30] - 1) * 100
        if mom_30 > 30:
            score -= 5
            details.append({"label": f"+{mom_30:.0f}% (overheated)", "impact": -5})
        elif mom_30 > 10:
            score += 10
            details.append({"label": f"+{mom_30:.0f}% momentum", "impact": 10})
        elif mom_30 < -30:
            score += 10
            details.append({"label": f"{mom_30:.0f}% (bounce mulig)", "impact": 10})

    returns = close.pct_change().dropna()
    vol = returns.std() * np.sqrt(365) * 100
    if vol < 50:
        score += 5
        details.append({"label": f"Lav vol ({vol:.0f}%)", "impact": 5})
    elif vol > 100:
        score -= 5
        details.append({"label": f"Ekstrem vol ({vol:.0f}%)", "impact": -5})

    return max(0, min(100, score)), details


def crypto_sentiment_score(info):
    score = 50.0
    details = []

    fg_df = fetch_fear_greed()
    if fg_df is not None and not fg_df.empty:
        fg = int(fg_df["value"].iloc[-1])
        if fg < 25:
            score += 20
            details.append({"label": f"Extreme Fear ({fg})", "impact": 20})
        elif fg < 45:
            score += 10
            details.append({"label": f"Fear ({fg})", "impact": 10})
        elif fg > 75:
            score -= 20
            details.append({"label": f"Extreme Greed ({fg})", "impact": -20})
        elif fg > 55:
            score -= 5
            details.append({"label": f"Greed ({fg})", "impact": -5})

    twitter = info.get("twitter_followers", 0) or 0
    if twitter > 1_000_000:
        score += 10
        details.append({"label": "Stor community (>1M Twitter)", "impact": 10})
    elif twitter > 100_000:
        score += 5
        details.append({"label": ">100k Twitter", "impact": 5})

    reddit = info.get("reddit_subscribers", 0) or 0
    if reddit > 500_000:
        score += 5
        details.append({"label": f"Aktiv Reddit", "impact": 5})

    return max(0, min(100, score)), details


def crypto_developer_score(info):
    score = 50.0
    details = []

    commits = info.get("github_commits_4w", 0) or 0
    if commits > 100:
        score += 15
        details.append({"label": f"Meget aktiv ({commits} commits/4w)", "impact": 15})
    elif commits > 30:
        score += 10
        details.append({"label": f"Aktiv ({commits} commits/4w)", "impact": 10})
    elif commits > 5:
        score += 5
        details.append({"label": f"Moderat ({commits} commits/4w)", "impact": 5})
    elif commits == 0:
        score -= 15
        details.append({"label": "Ingen recent udvikling", "impact": -15})

    stars = info.get("github_stars", 0) or 0
    if stars > 50_000:
        score += 10
        details.append({"label": f"Top GitHub ({stars:,} stars)", "impact": 10})
    elif stars > 10_000:
        score += 5
        details.append({"label": f"Populært ({stars:,} stars)", "impact": 5})

    return max(0, min(100, score)), details


def crypto_overall_score(info, df):
    market_s, market_d = crypto_market_score(info)
    tech_s, tech_d = crypto_technical_score(df)
    sent_s, sent_d = crypto_sentiment_score(info)
    dev_s, dev_d = crypto_developer_score(info)

    overall = market_s * 0.35 + tech_s * 0.30 + sent_s * 0.20 + dev_s * 0.15

    return {
        "overall": overall,
        "market": market_s,
        "technical": tech_s,
        "sentiment": sent_s,
        "developer": dev_s,
        "details": {
            "market": market_d, "technical": tech_d,
            "sentiment": sent_d, "developer": dev_d,
        },
    }


def crypto_recommendation(score):
    if score >= 75:
        return "🚀 STÆRKT KØB", "#16a34a"
    elif score >= 60:
        return "✅ KØB", "#22c55e"
    elif score >= 40:
        return "⏸️ HOLD", "#eab308"
    elif score >= 25:
        return "⚠️ SÆLG", "#ef4444"
    else:
        return "🛑 STÆRKT SÆLG", "#b91c1c"


# ===== TEKNISKE INDIKATORER =====

def crypto_indicators(df):
    """Beregn alle indikatorer (samme som aktier)"""
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_mid + bb_std * 2
    df["BB_lower"] = bb_mid - bb_std * 2

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df


# ===== KURS-MÅL & STOP LOSS =====

def crypto_price_targets(df, current_price, score_data=None):
    """
    Beregn intelligente kurs-niveauer for krypto.
    Mere aggressive end aktier pga. højere volatilitet.
    """
    if df is None or len(df) < 50:
        return None

    df_ind = crypto_indicators(df.copy())
    last = df_ind.iloc[-1]

    # 90-dages high/low (mere relevant end 52-uger for krypto)
    recent_90d = df.tail(90) if len(df) >= 90 else df
    high_90d = recent_90d["High"].max()
    low_90d = recent_90d["Low"].min()

    # 1-årig
    high_365d = df["High"].max()
    low_365d = df["Low"].min()

    atr = last["ATR"] if not pd.isna(last["ATR"]) else current_price * 0.05

    # Bollinger Bands som tekniske targets
    bb_upper = last["BB_upper"] if not pd.isna(last["BB_upper"]) else current_price * 1.1
    bb_lower = last["BB_lower"] if not pd.isna(last["BB_lower"]) else current_price * 0.9

    # Score-baseret aggressivitet
    score = score_data.get("overall", 50) if score_data else 50
    aggressive = score >= 65  # Højere targets ved bullish score

    # KØB ZONE: Sværere at time bunden i krypto, så bredere zone
    if last["RSI"] and last["RSI"] < 35:
        # Allerede oversold - nuværende pris er køb
        buy_low = current_price * 0.95
        buy_high = current_price * 1.02
    else:
        # Vent på pullback
        buy_low = max(bb_lower, current_price - atr * 3)
        buy_high = current_price - atr * 1
        # Aldrig mere end 15% under nuværende
        buy_low = max(buy_low, current_price * 0.80)

    # STOP LOSS: 3x ATR (krypto er volatilt)
    stop_loss = current_price - atr * 3
    # Aldrig mere end 25% stop (krypto reality)
    stop_loss = max(stop_loss, current_price * 0.75)

    # KORT-TERM TARGET (1-3 måneder): BB upper eller +1 ATR
    target_short = max(bb_upper, current_price + atr * 2)

    # LANG-TERM TARGET (6-12 måneder): Score-baseret
    if aggressive:
        # Bullish: Target ATH eller +50%
        target_long = max(high_365d, current_price * 1.5)
    else:
        # Konservativ: +30%
        target_long = current_price * 1.3

    # MOON TARGET (12m+): For bullish scenarier
    target_moon = current_price * 2.5 if aggressive else current_price * 1.8

    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "stop_loss": stop_loss,
        "target_short": target_short,
        "target_long": target_long,
        "target_moon": target_moon,
        "high_90d": high_90d,
        "low_90d": low_90d,
        "high_365d": high_365d,
        "low_365d": low_365d,
        "atr": atr,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


# ===== RISK METRICS =====

def crypto_risk_metrics(df):
    """Risk metrics tilpasset krypto (365 dage/år, ikke 252)"""
    if df is None or len(df) < 30:
        return None

    returns = df["Close"].pct_change().dropna()

    # Annualisering: 365 dage for krypto (handles 24/7)
    ann_return = (1 + returns.mean()) ** 365 - 1
    ann_vol = returns.std() * np.sqrt(365)

    # Sharpe (risk-free = 4% USD T-bills)
    rf = 0.04
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(365) if len(downside) > 0 else ann_vol
    sortino = (ann_return - rf) / downside_std if downside_std > 0 else 0

    # Max Drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()

    # VaR 95% (1-day)
    var95 = returns.quantile(0.05)

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "ann_r": ann_return,
        "ann_v": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "var95": var95,
        "dd_series": dd,
    }


# ===== MONTE CARLO =====

def crypto_monte_carlo(df, n_sims=500, days=180):
    """Monte Carlo med fat-tail distribution (mere realistisk for krypto)"""
    if df is None or len(df) < 30:
        return None, None

    returns = df["Close"].pct_change().dropna().values
    last_price = df["Close"].iloc[-1]

    # Brug Student-t fordeling (fatter tails end normal)
    from scipy import stats
    try:
        # Fit t-distribution
        params = stats.t.fit(returns)
        df_t, loc, scale = params

        # Generer simulationer
        sims = np.zeros((n_sims, days))
        for i in range(n_sims):
            random_returns = stats.t.rvs(df_t, loc=loc, scale=scale, size=days)
            prices = last_price * np.cumprod(1 + random_returns)
            sims[i] = prices
    except Exception:
        # Fallback til normal distribution
        mu = returns.mean()
        sigma = returns.std()
        sims = np.zeros((n_sims, days))
        for i in range(n_sims):
            random_returns = np.random.normal(mu, sigma, days)
            sims[i] = last_price * np.cumprod(1 + random_returns)

    return sims, last_price


# ===== BTC HALVING-CYCLE ANALYSE =====

BTC_HALVINGS = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19"]
NEXT_HALVING = "2028-04-15"


def btc_halving_analysis(symbol):
    """Hvor er vi i halving-cyklen? Kun relevant for BTC"""
    if symbol.upper() != "BTC":
        return None

    today = pd.Timestamp.now()
    last_halving = pd.Timestamp(BTC_HALVINGS[-1])
    next_halving = pd.Timestamp(NEXT_HALVING)

    days_since = (today - last_halving).days
    days_until = (next_halving - today).days
    cycle_progress = days_since / (days_since + days_until) * 100

    # Historisk: Bull market peak ~12-18 mdr efter halving
    if days_since < 180:
        phase = "🟡 Tidlig fase (akkumulation)"
        outlook = "Historisk svag - akkumuler"
    elif days_since < 365:
        phase = "🟢 Bull market start"
        outlook = "Historisk start på rally"
    elif days_since < 540:
        phase = "🚀 Bull market peak"
        outlook = "Historisk peak-zone (12-18m post-halving)"
    elif days_since < 730:
        phase = "🔴 Distribution"
        outlook = "Historisk top-blow + correction"
    else:
        phase = "🟠 Bear market / accumulation"
        outlook = "Pre-halving - akkumuler"

    return {
        "days_since_halving": days_since,
        "days_until_halving": days_until,
        "cycle_progress": cycle_progress,
        "phase": phase,
        "outlook": outlook,
        "last_halving": BTC_HALVINGS[-1],
        "next_halving": NEXT_HALVING,
    }


# ===== KORRELATION TIL BTC =====

def calculate_btc_correlation(df_coin, df_btc):
    """Beregn korrelation til BTC"""
    if df_coin is None or df_btc is None:
        return None

    # Align på datoer
    coin_returns = df_coin["Close"].pct_change()
    btc_returns = df_btc["Close"].pct_change()

    aligned = pd.DataFrame({"coin": coin_returns, "btc": btc_returns}).dropna()
    if len(aligned) < 30:
        return None

    corr = aligned["coin"].corr(aligned["btc"])

    # Beta (sensitivity til BTC)
    cov = aligned["coin"].cov(aligned["btc"])
    var_btc = aligned["btc"].var()
    beta = cov / var_btc if var_btc > 0 else 1

    # Rolling correlation (sidste 30 dage)
    rolling_corr = aligned["coin"].rolling(30).corr(aligned["btc"])

    return {
        "correlation": corr,
        "beta": beta,
        "rolling_correlation": rolling_corr,
    }


# ===== BACKTEST =====

def crypto_backtest(df, holding_days=30, sample_freq=7):
    """Walk-forward backtest tilpasset krypto"""
    if df is None or len(df) < 200 + holding_days:
        return None

    df_ind = crypto_indicators(df.copy())
    results = []

    start_idx = 200
    end_idx = len(df_ind) - holding_days

    for i in range(start_idx, end_idx, sample_freq):
        snapshot = df_ind.iloc[:i+1]
        if len(snapshot) < 200:
            continue

        # Beregn score baseret på data op til dette punkt
        last = snapshot.iloc[-1]
        score = 50

        if not pd.isna(last["RSI"]):
            if last["RSI"] < 30:
                score += 15
            elif last["RSI"] > 70:
                score -= 15

        if not pd.isna(last["SMA50"]) and not pd.isna(last["SMA200"]):
            if last["Close"] > last["SMA200"]:
                score += 10
            else:
                score -= 10
            if last["SMA50"] > last["SMA200"]:
                score += 10

        # Recommendation
        if score >= 65:
            rec = "KØB"
        elif score >= 50:
            rec = "HOLD"
        else:
            rec = "SÆLG"

        # Faktisk afkast
        entry = last["Close"]
        exit_price = df_ind.iloc[i + holding_days]["Close"]
        ret = (exit_price / entry - 1) * 100

        results.append({
            "date": df_ind.index[i],
            "score": score,
            "recommendation": rec,
            "entry_price": entry,
            "exit_price": exit_price,
            "return_pct": ret,
        })

    if not results:
        return None

    df_res = pd.DataFrame(results)

    # Stats per recommendation
    stats = {}
    for rec in ["KØB", "HOLD", "SÆLG"]:
        subset = df_res[df_res["recommendation"] == rec]
        if len(subset) > 0:
            stats[rec] = {
                "count": len(subset),
                "avg_return": subset["return_pct"].mean(),
                "median_return": subset["return_pct"].median(),
                "win_rate": (subset["return_pct"] > 0).mean() * 100,
                "best": subset["return_pct"].max(),
                "worst": subset["return_pct"].min(),
            }
        else:
            stats[rec] = None

    # Buy & hold over samme periode
    bh_return = (df_ind.iloc[end_idx]["Close"] / df_ind.iloc[start_idx]["Close"] - 1) * 100

    return {
        "results": df_res,
        "stats": stats,
        "n_trades": len(df_res),
        "start_date": df_ind.index[start_idx],
        "end_date": df_ind.index[end_idx],
        "holding_days": holding_days,
        "buy_hold_return": bh_return,
    }
