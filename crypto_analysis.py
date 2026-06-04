"""Krypto-specifik scoring (4-faktor model)"""
import numpy as np
import pandas as pd
from crypto_data import fetch_fear_greed


def crypto_market_score(info):
    """Faktor 1: Markedsdata-baseret score (0-100)"""
    score = 50.0
    details = []

    # Market cap rang
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

    # Afstand til ATH (mean reversion)
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

    # Supply scarcity
    circ = info.get("circulating_supply") or 0
    max_sup = info.get("max_supply")
    if max_sup and circ > 0:
        ratio = circ / max_sup
        if ratio > 0.95:
            score += 15
            details.append({"label": f"95%+ supply mined ({ratio*100:.1f}%)", "impact": 15})
        elif ratio > 0.85:
            score += 10
            details.append({"label": f"85%+ supply mined", "impact": 10})
        elif ratio < 0.3:
            score -= 10
            details.append({"label": f"Højt fremtidigt udbud ({ratio*100:.0f}% mined)", "impact": -10})
    elif not max_sup:
        score -= 5
        details.append({"label": "Inflationær (ingen max supply)", "impact": -5})

    # Volume/MarketCap ratio (likviditet)
    mc = info.get("marketCap", 0) or 0
    vol = info.get("totalVolume", 0) or 0
    if mc > 0 and vol > 0:
        liq_ratio = vol / mc
        if liq_ratio > 0.15:
            score += 10
            details.append({"label": "Høj likviditet (V/MC>15%)", "impact": 10})
        elif liq_ratio < 0.02:
            score -= 10
            details.append({"label": "Lav likviditet", "impact": -10})

    return max(0, min(100, score)), details


def crypto_technical_score(df):
    """Faktor 2: Teknisk score (genbruger din eksisterende logik)"""
    if df is None or len(df) < 50:
        return 50.0, [{"label": "Utilstrækkelig data", "impact": 0}]

    score = 50.0
    details = []
    close = df["Close"]

    # SMA crossovers
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else None
    last = close.iloc[-1]

    if sma200 and last > sma200:
        score += 10
        details.append({"label": "Pris > SMA200 (bull market)", "impact": 10})
    elif sma200:
        score -= 10
        details.append({"label": "Pris < SMA200 (bear market)", "impact": -10})

    if last > sma50 > sma20:
        score -= 5
        details.append({"label": "Death cross-tendens", "impact": -5})
    elif last > sma20 > sma50:
        score += 10
        details.append({"label": "Golden cross-tendens", "impact": 10})

    # RSI
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

    # Momentum 30 dage
    if len(close) >= 30:
        mom_30 = (close.iloc[-1] / close.iloc[-30] - 1) * 100
        if mom_30 > 30:
            score -= 5
            details.append({"label": f"Stærk momentum (+{mom_30:.0f}% 30d) - korrektion mulig", "impact": -5})
        elif mom_30 > 10:
            score += 10
            details.append({"label": f"Positiv momentum (+{mom_30:.0f}% 30d)", "impact": 10})
        elif mom_30 < -30:
            score += 10
            details.append({"label": f"Stort fald ({mom_30:.0f}% 30d) - bounce mulig", "impact": 10})

    # Volatilitet (mindre er bedre for store coins)
    returns = close.pct_change().dropna()
    vol = returns.std() * np.sqrt(365) * 100
    if vol < 50:
        score += 5
        details.append({"label": f"Lav volatilitet ({vol:.0f}%)", "impact": 5})
    elif vol > 100:
        score -= 5
        details.append({"label": f"Ekstrem volatilitet ({vol:.0f}%)", "impact": -5})

    return max(0, min(100, score)), details


def crypto_sentiment_score(info):
    """Faktor 3: Sentiment & community"""
    score = 50.0
    details = []

    # Fear & Greed Index (markedssentiment)
    fg_df = fetch_fear_greed()
    if fg_df is not None and not fg_df.empty:
        fg = fg_df["value"].iloc[-1]
        if fg < 25:
            score += 20
            details.append({"label": f"Extreme Fear ({fg}) - køb-mulighed", "impact": 20})
        elif fg < 45:
            score += 10
            details.append({"label": f"Fear ({fg})", "impact": 10})
        elif fg > 75:
            score -= 20
            details.append({"label": f"Extreme Greed ({fg}) - vær forsigtig", "impact": -20})
        elif fg > 55:
            score -= 5
            details.append({"label": f"Greed ({fg})", "impact": -5})

    # Twitter followers
    twitter = info.get("twitter_followers", 0) or 0
    if twitter > 1_000_000:
        score += 10
        details.append({"label": f"Stor community (>1M Twitter)", "impact": 10})
    elif twitter > 100_000:
        score += 5
        details.append({"label": f"God community (>100k)", "impact": 5})

    # Reddit subscribers
    reddit = info.get("reddit_subscribers", 0) or 0
    if reddit > 500_000:
        score += 5
        details.append({"label": f"Aktiv Reddit ({reddit:,})", "impact": 5})

    return max(0, min(100, score)), details


def crypto_developer_score(info):
    """Faktor 4: Developer activity"""
    score = 50.0
    details = []

    commits = info.get("github_commits_4w", 0) or 0
    if commits > 100:
        score += 15
        details.append({"label": f"Meget aktiv udvikling ({commits} commits/4w)", "impact": 15})
    elif commits > 30:
        score += 10
        details.append({"label": f"Aktiv udvikling ({commits} commits/4w)", "impact": 10})
    elif commits > 5:
        score += 5
        details.append({"label": f"Moderat udvikling ({commits} commits/4w)", "impact": 5})
    elif commits == 0:
        score -= 15
        details.append({"label": "Ingen recent udvikling", "impact": -15})

    stars = info.get("github_stars", 0) or 0
    if stars > 50_000:
        score += 10
        details.append({"label": f"Top GitHub repo ({stars:,} stars)", "impact": 10})
    elif stars > 10_000:
        score += 5
        details.append({"label": f"Populært repo ({stars:,} stars)", "impact": 5})

    prs = info.get("github_pull_requests_merged", 0) or 0
    if prs > 1000:
        score += 5
        details.append({"label": f"Mange merged PRs ({prs:,})", "impact": 5})

    return max(0, min(100, score)), details


def crypto_overall_score(info, df):
    """Samlet krypto-score (4-faktor model)"""
    market_s, market_d = crypto_market_score(info)
    tech_s, tech_d = crypto_technical_score(df)
    sent_s, sent_d = crypto_sentiment_score(info)
    dev_s, dev_d = crypto_developer_score(info)

    # Vægtning: Market 35%, Tech 30%, Sentiment 20%, Dev 15%
    overall = market_s * 0.35 + tech_s * 0.30 + sent_s * 0.20 + dev_s * 0.15

    return {
        "overall": overall,
        "market": market_s,
        "technical": tech_s,
        "sentiment": sent_s,
        "developer": dev_s,
        "details": {
            "market": market_d,
            "technical": tech_d,
            "sentiment": sent_d,
            "developer": dev_d,
        },
    }


def crypto_recommendation(score):
    """Krypto-anbefaling baseret på samlet score"""
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
