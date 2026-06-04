"""Krypto-analyse: scoring, indikatorer, kursmål, risk, monte carlo, backtest"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============ TEKNISKE INDIKATORER ============

def crypto_indicators(hist):
    """Beregner tekniske indikatorer for krypto"""
    if hist is None or len(hist) < 20:
        return hist

    df = hist.copy()

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volatility (20-dages annualized)
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(365)

    return df


# ============ SCORE FUNKTIONER ============

def _market_score(info):
    """Score baseret på markedsdata: market cap, rank, volume"""
    score = 50
    details = []

    # Market cap rank (lavere er bedre)
    rank = info.get("marketCapRank")
    if rank:
        if rank <= 10:
            score += 20
            details.append({"label": f"Top 10 rank (#{rank})", "impact": 20})
        elif rank <= 30:
            score += 12
            details.append({"label": f"Top 30 rank (#{rank})", "impact": 12})
        elif rank <= 100:
            score += 5
            details.append({"label": f"Top 100 rank (#{rank})", "impact": 5})
        elif rank > 500:
            score -= 10
            details.append({"label": f"Lav rank (#{rank})", "impact": -10})

    # Market cap størrelse
    mc = info.get("marketCap") or 0
    if mc > 100e9:
        score += 15
        details.append({"label": "Mega cap (>$100B)", "impact": 15})
    elif mc > 10e9:
        score += 10
        details.append({"label": "Large cap ($10-100B)", "impact": 10})
    elif mc > 1e9:
        score += 5
        details.append({"label": "Mid cap ($1-10B)", "impact": 5})
    elif mc < 100e6:
        score -= 10
        details.append({"label": "Small cap (<$100M)", "impact": -10})

    # 24h volume / market cap ratio (likviditet)
    vol = info.get("totalVolume") or 0
    if mc > 0 and vol > 0:
        vol_ratio = vol / mc
        if vol_ratio > 0.20:
            score += 8
            details.append({"label": f"Høj likviditet ({vol_ratio*100:.1f}%)", "impact": 8})
        elif vol_ratio > 0.05:
            score += 3
            details.append({"label": f"Normal likviditet ({vol_ratio*100:.1f}%)", "impact": 3})
        elif vol_ratio < 0.01:
            score -= 5
            details.append({"label": f"Lav likviditet ({vol_ratio*100:.1f}%)", "impact": -5})

    # ATH afstand
    ath_change = info.get("ath_change_%")
    if ath_change is not None:
        if ath_change > -20:
            score += 5
            details.append({"label": f"Tæt på ATH ({ath_change:+.0f}%)", "impact": 5})
        elif ath_change < -80:
            score += 8
            details.append({"label": f"Langt fra ATH ({ath_change:+.0f}%) - billig", "impact": 8})
        elif ath_change < -50:
            score += 3
            details.append({"label": f"Et stykke fra ATH ({ath_change:+.0f}%)", "impact": 3})

    return max(0, min(100, score)), details


def _technical_score(hist):
    """Score baseret på tekniske indikatorer"""
    if hist is None or len(hist) < 50:
        return 50, []

    df = crypto_indicators(hist)
    last = df.iloc[-1]
    score = 50
    details = []

    # RSI
    rsi = last.get("RSI")
    if pd.notna(rsi):
        if 40 <= rsi <= 60:
            score += 8
            details.append({"label": f"RSI neutral ({rsi:.0f})", "impact": 8})
        elif 30 <= rsi < 40:
            score += 12
            details.append({"label": f"RSI oversolgt-zone ({rsi:.0f})", "impact": 12})
        elif rsi < 30:
            score += 15
            details.append({"label": f"RSI stærkt oversolgt ({rsi:.0f}) - køb", "impact": 15})
        elif 60 < rsi <= 70:
            score -= 3
            details.append({"label": f"RSI lidt overkøbt ({rsi:.0f})", "impact": -3})
        elif rsi > 70:
            score -= 12
            details.append({"label": f"RSI overkøbt ({rsi:.0f}) - vent", "impact": -12})

    # Pris vs SMA50
    sma50 = last.get("SMA50")
    close = last.get("Close")
    if pd.notna(sma50) and pd.notna(close):
        diff_pct = (close / sma50 - 1) * 100
        if diff_pct > 5:
            score += 5
            details.append({"label": f"Over SMA50 (+{diff_pct:.1f}%)", "impact": 5})
        elif diff_pct < -10:
            score += 8
            details.append({"label": f"Under SMA50 ({diff_pct:.1f}%) - rebound?", "impact": 8})
        elif diff_pct < -5:
            score -= 3
            details.append({"label": f"Under SMA50 ({diff_pct:.1f}%)", "impact": -3})

    # Pris vs SMA200 (Golden cross signal)
    sma200 = last.get("SMA200")
    if pd.notna(sma200) and pd.notna(sma50) and pd.notna(close):
        if close > sma200 and sma50 > sma200:
            score += 10
            details.append({"label": "Bull market (pris>SMA200, SMA50>SMA200)", "impact": 10})
        elif close < sma200 and sma50 < sma200:
            score -= 8
            details.append({"label": "Bear market (pris<SMA200, SMA50<SMA200)", "impact": -8})

    # MACD
    macd = last.get("MACD")
    macd_sig = last.get("MACD_signal")
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig and macd > 0:
            score += 7
            details.append({"label": "MACD bullish", "impact": 7})
        elif macd < macd_sig and macd < 0:
            score -= 7
            details.append({"label": "MACD bearish", "impact": -7})

    # Bollinger position
    bb_u = last.get("BB_upper")
    bb_l = last.get("BB_lower")
    if pd.notna(bb_u) and pd.notna(bb_l) and pd.notna(close):
        bb_range = bb_u - bb_l
        if bb_range > 0:
            bb_pos = (close - bb_l) / bb_range
            if bb_pos < 0.2:
                score += 8
                details.append({"label": "BB nederste zone - oversolgt", "impact": 8})
            elif bb_pos > 0.8:
                score -= 5
                details.append({"label": "BB øverste zone - overkøbt", "impact": -5})

    # Momentum 30-dages
    if len(df) >= 30:
        ret_30d = (close / df["Close"].iloc[-30] - 1) * 100
        if ret_30d > 30:
            score -= 5
            details.append({"label": f"30d momentum +{ret_30d:.0f}% - parabolic", "impact": -5})
        elif ret_30d > 10:
            score += 5
            details.append({"label": f"30d momentum +{ret_30d:.0f}%", "impact": 5})
        elif ret_30d < -30:
            score += 8
            details.append({"label": f"30d momentum {ret_30d:.0f}% - bounce?", "impact": 8})
        elif ret_30d < -10:
            score -= 3
            details.append({"label": f"30d momentum {ret_30d:.0f}%", "impact": -3})

    return max(0, min(100, score)), details


def _sentiment_score(info):
    """Score baseret på sentiment (community, social)"""
    score = 50
    details = []

    # CoinGecko sentiment votes
    sent_up = info.get("sentiment_votes_up_%")
    if sent_up is not None:
        if sent_up > 75:
            score += 15
            details.append({"label": f"Stærk positiv sentiment ({sent_up:.0f}%)", "impact": 15})
        elif sent_up > 60:
            score += 8
            details.append({"label": f"Positiv sentiment ({sent_up:.0f}%)", "impact": 8})
        elif sent_up < 40:
            score -= 10
            details.append({"label": f"Negativ sentiment ({sent_up:.0f}%)", "impact": -10})

    # Community score
    comm = info.get("community_score")
    if comm is not None:
        if comm > 60:
            score += 10
            details.append({"label": f"Stærk community ({comm:.0f})", "impact": 10})
        elif comm > 40:
            score += 5
            details.append({"label": f"OK community ({comm:.0f})", "impact": 5})
        elif comm < 20:
            score -= 5
            details.append({"label": f"Svag community ({comm:.0f})", "impact": -5})

    # Public interest score
    public = info.get("public_interest_score")
    if public is not None and public > 0:
        if public > 0.001:
            score += 8
            details.append({"label": "Høj public interest", "impact": 8})
        elif public > 0.0001:
            score += 3
            details.append({"label": "Moderate public interest", "impact": 3})

    # Twitter followers
    twitter = info.get("twitter_followers")
    if twitter:
        if twitter > 1_000_000:
            score += 8
            details.append({"label": f"Twitter: {twitter/1e6:.1f}M followers", "impact": 8})
        elif twitter > 100_000:
            score += 4
            details.append({"label": f"Twitter: {twitter/1e3:.0f}K followers", "impact": 4})

    return max(0, min(100, score)), details


def _developer_score(info):
    """Score baseret på developer aktivitet (GitHub)"""
    score = 50
    details = []

    dev = info.get("developer_score")
    if dev is not None:
        if dev > 70:
            score += 20
            details.append({"label": f"Excellent dev activity ({dev:.0f})", "impact": 20})
        elif dev > 50:
            score += 10
            details.append({"label": f"God dev activity ({dev:.0f})", "impact": 10})
        elif dev > 30:
            score += 3
            details.append({"label": f"OK dev activity ({dev:.0f})", "impact": 3})
        elif dev < 15:
            score -= 10
            details.append({"label": f"Lav dev activity ({dev:.0f})", "impact": -10})

    # GitHub stars
    stars = info.get("github_stars")
    if stars:
        if stars > 10000:
            score += 12
            details.append({"label": f"GitHub: {stars/1000:.1f}K stars", "impact": 12})
        elif stars > 1000:
            score += 6
            details.append({"label": f"GitHub: {stars} stars", "impact": 6})
        elif stars > 100:
            score += 2
            details.append({"label": f"GitHub: {stars} stars", "impact": 2})

    # Forks
    forks = info.get("github_forks")
    if forks:
        if forks > 1000:
            score += 8
            details.append({"label": f"GitHub forks: {forks}", "impact": 8})
        elif forks > 100:
            score += 3
            details.append({"label": f"GitHub forks: {forks}", "impact": 3})

    # Commits sidste 4 uger
    commits = info.get("commit_count_4_weeks")
    if commits is not None:
        if commits > 100:
            score += 10
            details.append({"label": f"{commits} commits/4 uger - meget aktiv", "impact": 10})
        elif commits > 20:
            score += 5
            details.append({"label": f"{commits} commits/4 uger", "impact": 5})
        elif commits == 0:
            score -= 8
            details.append({"label": "Ingen commits sidste 4 uger", "impact": -8})

    return max(0, min(100, score)), details


def crypto_overall_score(info, hist):
    """
    Samlet multi-faktor score for krypto:
    - 35% market data
    - 30% teknisk
    - 20% sentiment
    - 15% developer
    """
    market, market_details = _market_score(info)
    technical, technical_details = _technical_score(hist)
    sentiment, sentiment_details = _sentiment_score(info)
    developer, developer_details = _developer_score(info)

    overall = (
        market * 0.35
        + technical * 0.30
        + sentiment * 0.20
        + developer * 0.15
    )

    return {
        "overall": overall,
        "market": market,
        "technical": technical,
        "sentiment": sentiment,
        "developer": developer,
        "details": {
            "market": market_details,
            "technical": technical_details,
            "sentiment": sentiment_details,
            "developer": developer_details,
        }
    }


def crypto_recommendation(score):
    """Anbefaling baseret på samlet score"""
    if score >= 75:
        return "STÆRKT KØB", "#16a34a"
    elif score >= 60:
        return "KØB", "#22c55e"
    elif score >= 45:
        return "HOLD", "#eab308"
    elif score >= 30:
        return "SÆLG", "#ef4444"
    else:
        return "STÆRKT SÆLG", "#b91c1c"


# ============ KURSMÅL ============

def crypto_price_targets(hist, current_price, scores=None):
    """Beregn kursmål baseret på ATR, BB og historiske ranges"""
    if hist is None or len(hist) < 20:
        return None

    df = crypto_indicators(hist)
    last = df.iloc[-1]

    atr = last.get("ATR")
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.05  # 5% fallback

    bb_upper = last.get("BB_upper")
    bb_lower = last.get("BB_lower")
    if pd.isna(bb_upper) or pd.isna(bb_lower):
        bb_upper = current_price * 1.15
        bb_lower = current_price * 0.85

    # Historiske ranges
    high_90d = df["High"].tail(90).max() if len(df) >= 90 else df["High"].max()
    low_90d = df["Low"].tail(90).min() if len(df) >= 90 else df["Low"].min()
    high_365d = df["High"].tail(365).max() if len(df) >= 365 else df["High"].max()
    low_365d = df["Low"].tail(365).min() if len(df) >= 365 else df["Low"].min()

    # Køb zone (under nuværende, mod BB lower / 90d low)
    buy_high = max(bb_lower, current_price - 1.5 * atr)
    buy_low = max(low_90d * 1.02, current_price - 3 * atr)
    if buy_low > buy_high:
        buy_low, buy_high = buy_high * 0.95, buy_high

    # Stop loss (3x ATR under)
    stop_loss = current_price - 3 * atr

    # Mål
    target_short = bb_upper  # Bollinger upper (1-3 mdr)
    target_long = high_90d * 1.10  # 10% over 90d high (6-12 mdr)
    target_moon = high_365d * 1.20  # 20% over 365d high (12m+)

    # Hvis vi er nær ATH, brug mere konservative mål
    if current_price >= high_365d * 0.95:
        target_long = current_price * 1.30
        target_moon = current_price * 1.80

    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "stop_loss": max(0.01, stop_loss),
        "target_short": target_short,
        "target_long": target_long,
        "target_moon": target_moon,
        "high_90d": high_90d,
        "low_90d": low_90d,
        "high_365d": high_365d,
        "low_365d": low_365d,
        "atr": atr,
    }


# ============ RISK METRICS ============

def crypto_risk_metrics(hist):
    """Beregn risk metrics: Sharpe, Sortino, Calmar, max DD, VaR"""
    if hist is None or len(hist) < 30:
        return None

    returns = hist["Close"].pct_change().dropna()
    if len(returns) < 30:
        return None

    # Krypto: 365 trading days/year
    ann_factor = 365

    ann_r = float(returns.mean() * ann_factor)
    ann_v = float(returns.std() * np.sqrt(ann_factor))

    # Risk-free ~ 4% (US T-bill)
    rf = 0.04
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else 0

    # Sortino (downside only)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 0
    sortino = (ann_r - rf) / downside_std if downside_std > 0 else 0

    # Drawdown series
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd_series = (cum - running_max) / running_max
    max_dd = float(dd_series.min())

    # Calmar = annual return / max drawdown
    calmar = ann_r / abs(max_dd) if max_dd < 0 else 0

    # VaR 95% (1-day)
    var95 = float(np.percentile(returns, 5))

    return {
        "ann_r": ann_r,
        "ann_v": ann_v,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "var95": var95,
        "dd_series": dd_series,
    }


# ============ MONTE CARLO (uden scipy) ============

def crypto_monte_carlo(hist, n_sims=500, days=180):
    """Monte Carlo med fat tails (Student-t distribution via numpy)"""
    if hist is None or len(hist) < 30:
        return None, None

    returns = hist["Close"].pct_change().dropna().values
    if len(returns) < 30:
        return None, None

    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    last_price = float(hist["Close"].iloc[-1])

    # Student-t med df=4 for fat tails (numpy's standard_t)
    df = 4
    # Normaliser så variansen er 1 (standard_t har varians df/(df-2))
    scale_factor = np.sqrt((df - 2) / df) if df > 2 else 1.0

    sims = np.zeros((n_sims, days))
    for i in range(n_sims):
        shocks = np.random.standard_t(df, size=days) * scale_factor
        daily_returns = mu + sigma * shocks
        # Cap ekstreme afkast for stabilitet
        daily_returns = np.clip(daily_returns, -0.50, 0.50)
        price_path = last_price * np.cumprod(1 + daily_returns)
        sims[i] = price_path

    return sims, last_price


# ============ BTC HALVING ============

def btc_halving_analysis(symbol):
    """Analyse af BTC halving cycle"""
    if symbol != "BTC":
        return None

    # Historiske + næste halving
    halvings = [
        datetime(2012, 11, 28),
        datetime(2016, 7, 9),
        datetime(2020, 5, 11),
        datetime(2024, 4, 19),
        datetime(2028, 4, 15),  # forventet
    ]

    today = datetime.now()
    last_halving = None
    next_halving = None

    for h in halvings:
        if h <= today:
            last_halving = h
        else:
            next_halving = h
            break

    if last_halving is None or next_halving is None:
        return None

    days_since = (today - last_halving).days
    days_until = (next_halving - today).days
    cycle_length = (next_halving - last_halving).days
    cycle_progress = (days_since / cycle_length) * 100

    # Cycle-faser baseret på historiske mønstre
    if cycle_progress < 25:
        phase = "🟠 Bear market / accumulation"
        outlook = "Pre-halving - akkumuler"
    elif cycle_progress < 50:
        phase = "🟡 Accumulation -> early bull"
        outlook = "Bullish opbygning"
    elif cycle_progress < 75:
        phase = "🟢 Bull market"
        outlook = "Peak euphoria nærmer sig"
    elif cycle_progress < 100:
        phase = "🔴 Distribution / bear"
        outlook = "Cycle top er typisk her"
    else:
        phase = "Ukendt"
        outlook = "?"

    return {
        "last_halving": last_halving.strftime("%Y-%m-%d"),
        "next_halving": next_halving.strftime("%Y-%m-%d"),
        "days_since_halving": days_since,
        "days_until_halving": days_until,
        "cycle_progress": cycle_progress,
        "phase": phase,
        "outlook": outlook,
    }


# ============ BTC KORRELATION ============

def calculate_btc_correlation(hist, btc_hist):
    """Beregn korrelation og beta til BTC"""
    if hist is None or btc_hist is None:
        return None
    if len(hist) < 30 or len(btc_hist) < 30:
        return None

    # Align dates
    df = pd.DataFrame({
        "asset": hist["Close"],
        "btc": btc_hist["Close"]
    }).dropna()

    if len(df) < 30:
        return None

    asset_ret = df["asset"].pct_change().dropna()
    btc_ret = df["btc"].pct_change().dropna()

    # Korrelation
    correlation = float(asset_ret.corr(btc_ret))

    # Beta (asset_ret = alpha + beta * btc_ret)
    cov = float(asset_ret.cov(btc_ret))
    btc_var = float(btc_ret.var())
    beta = cov / btc_var if btc_var > 0 else 0

    # Rolling 30-dages korrelation
    rolling_correlation = asset_ret.rolling(30).corr(btc_ret).dropna()

    return {
        "correlation": correlation,
        "beta": beta,
        "rolling_correlation": rolling_correlation,
    }


# ============ BACKTEST ============

def crypto_backtest(hist, holding_days=30, sample_freq=7, min_history_days=200):
    """
    Walk-forward backtest af crypto-modellen.
    For hvert sample-tidspunkt: beregn score på data UP TO that point,
    sammenlign med faktisk afkast over de næste holding_days dage.
    """
    if hist is None or len(hist) < min_history_days + holding_days:
        return None

    # Skab fake "info" til scoring (kun det nødvendige)
    results = []
    start_idx = min_history_days
    end_idx = len(hist) - holding_days

    if end_idx <= start_idx:
        return None

    for i in range(start_idx, end_idx, sample_freq):
        # Data op til punkt i
        hist_slice = hist.iloc[:i+1]
        if len(hist_slice) < 50:
            continue

        # Minimal info-dict (vi har ikke historisk market cap data)
        info_slice = {
            "marketCapRank": 50,  # neutral
            "marketCap": 1e9,
            "totalVolume": 1e8,
        }

        # Score
        try:
            t_score, _ = _technical_score(hist_slice)
            # Brug kun teknisk score til backtest (det er det vi kan beregne historisk)
            score = t_score
        except Exception:
            continue

        # Anbefaling
        rec, _ = crypto_recommendation(score)
        # Reducér til 3 kategorier for crypto
        if rec in ("STÆRKT KØB", "KØB"):
            simple_rec = "KØB"
        elif rec in ("STÆRKT SÆLG", "SÆLG"):
            simple_rec = "SÆLG"
        else:
            simple_rec = "HOLD"

        # Faktisk afkast over de næste holding_days dage
        entry_price = float(hist["Close"].iloc[i])
        exit_idx = i + holding_days
        if exit_idx >= len(hist):
            break
        exit_price = float(hist["Close"].iloc[exit_idx])
        return_pct = (exit_price / entry_price - 1) * 100

        results.append({
            "date": hist.index[i],
            "score": score,
            "recommendation": simple_rec,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": return_pct,
        })

    if not results:
        return None

    df_results = pd.DataFrame(results)

    # Stats per anbefaling
    stats = {}
    for rec in ["KØB", "HOLD", "SÆLG"]:
        subset = df_results[df_results["recommendation"] == rec]
        if len(subset) > 0:
            stats[rec] = {
                "count": len(subset),
                "win_rate": (subset["return_pct"] > 0).mean() * 100,
                "avg_return": float(subset["return_pct"].mean()),
                "median_return": float(subset["return_pct"].median()),
                "best": float(subset["return_pct"].max()),
                "worst": float(subset["return_pct"].min()),
            }
        else:
            stats[rec] = None

    # Buy & hold over hele perioden
    bh_return = (hist["Close"].iloc[end_idx] / hist["Close"].iloc[start_idx] - 1) * 100

    return {
        "results": df_results,
        "stats": stats,
        "n_trades": len(df_results),
        "start_date": df_results["date"].iloc[0],
        "end_date": df_results["date"].iloc[-1],
        "buy_hold_return": float(bh_return),
        "holding_days": holding_days,
    }
