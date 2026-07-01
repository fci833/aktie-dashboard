"""
Metal-specifik analyse:
- 60% teknisk + 40% makro (ingen fundamentals som P/E)
- Sikker havn-detection
- USD/renter/VIX-korrelation
"""
import pandas as pd
import numpy as np
import ta


def get_metal_indicators(hist):
    """Beregner tekniske indikatorer for metal-data"""
    if hist is None or hist.empty or len(hist) < 50:
        return hist

    df = hist.copy()

    try:
        # Moving averages
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        if len(df) >= 200:
            df["SMA200"] = df["Close"].rolling(200).mean()
        else:
            df["SMA200"] = np.nan

        # RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"], window=20)
        df["BB_high"] = bb.bollinger_hband()
        df["BB_low"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()

        # ATR
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()

        # ADX
        try:
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
            df["ADX"] = adx.adx()
        except Exception:
            df["ADX"] = np.nan

    except Exception as e:
        print(f"⚠️ Indicator error: {e}")

    return df


def metals_technical_score(df):
    """Teknisk score for metaller"""
    if df is None or df.empty or len(df) < 50:
        return 50, []

    score = 50
    details = []

    try:
        last = df.iloc[-1]
        close = last["Close"]

        # SMA trend
        sma50 = last.get("SMA50")
        sma200 = last.get("SMA200")
        if pd.notna(sma50) and pd.notna(sma200):
            if close > sma50 > sma200:
                score += 15
                details.append({"label": "Pris > SMA50 > SMA200 (uptrend)", "impact": 15})
            elif close < sma50 < sma200:
                score -= 15
                details.append({"label": "Pris < SMA50 < SMA200 (downtrend)", "impact": -15})
            elif close > sma50:
                score += 5
                details.append({"label": "Pris over SMA50", "impact": 5})
            elif close < sma50:
                score -= 5
                details.append({"label": "Pris under SMA50", "impact": -5})

        # RSI
        rsi = last.get("RSI")
        if pd.notna(rsi):
            if rsi < 30:
                score += 10
                details.append({"label": f"RSI oversold ({rsi:.0f})", "impact": 10})
            elif rsi > 70:
                score -= 10
                details.append({"label": f"RSI overbought ({rsi:.0f})", "impact": -10})
            elif 40 <= rsi <= 60:
                details.append({"label": f"RSI neutral ({rsi:.0f})", "impact": 0})

        # MACD
        macd = last.get("MACD")
        macd_sig = last.get("MACD_signal")
        if pd.notna(macd) and pd.notna(macd_sig):
            if macd > macd_sig and macd > 0:
                score += 8
                details.append({"label": "MACD bullish crossover", "impact": 8})
            elif macd < macd_sig and macd < 0:
                score -= 8
                details.append({"label": "MACD bearish crossover", "impact": -8})

        # ADX (trend strength - vigtigt for metaller!)
        adx = last.get("ADX")
        if pd.notna(adx) and adx > 25:
            if score > 50:
                score += 5
                details.append({"label": f"Stærk trend (ADX={adx:.0f}) bekræfter bullish", "impact": 5})
            elif score < 50:
                score -= 5
                details.append({"label": f"Stærk trend (ADX={adx:.0f}) er negativ", "impact": -5})

        # Lang momentum (metaller har lange trends)
        if len(df) >= 126:
            close_6m = df["Close"].iloc[-126]
            if close_6m > 0:
                momentum_6m = (close / close_6m - 1) * 100
                if momentum_6m > 15:
                    score += 10
                    details.append({"label": f"6m momentum +{momentum_6m:.1f}%", "impact": 10})
                elif momentum_6m < -15:
                    score -= 10
                    details.append({"label": f"6m momentum {momentum_6m:.1f}%", "impact": -10})

        # Bollinger position
        bb_high = last.get("BB_high")
        bb_low = last.get("BB_low")
        if pd.notna(bb_high) and pd.notna(bb_low):
            bb_range = bb_high - bb_low
            if bb_range > 0:
                bb_pos = (close - bb_low) / bb_range * 100
                if bb_pos > 95:
                    score -= 5
                    details.append({"label": f"Over BB upper ({bb_pos:.0f}%)", "impact": -5})
                elif bb_pos < 5:
                    score += 5
                    details.append({"label": f"Under BB lower ({bb_pos:.0f}%)", "impact": 5})

    except Exception as e:
        details.append({"label": f"Fejl: {str(e)[:80]}", "impact": 0})

    return max(0, min(100, score)), details


def metals_macro_score(drivers):
    """Makro-score baseret på cross-asset drivere"""
    score = 50
    details = []

    if not drivers:
        return score, details

    try:
        # DXY (dollar) - invers korrelation
        dxy = drivers.get("DXY")
        if dxy:
            dxy_1m = dxy.get("change_1m", 0) or 0
            if dxy_1m < -2:
                score += 12
                details.append({
                    "label": f"USD svækket {dxy_1m:.1f}% (30d) → bullish",
                    "impact": 12
                })
            elif dxy_1m > 2:
                score -= 12
                details.append({
                    "label": f"USD styrket +{dxy_1m:.1f}% (30d) → bearish",
                    "impact": -12
                })

        # US 10y rente - invers korrelation for guld/sølv
        tnx = drivers.get("TNX")
        if tnx:
            current_rate = tnx.get("current", 4.0) or 4.0
            rate_1m = tnx.get("change_1m", 0) or 0
            if rate_1m < -5:
                score += 10
                details.append({
                    "label": f"US 10y falder ({current_rate:.2f}%) → bullish metaller",
                    "impact": 10
                })
            elif rate_1m > 5:
                score -= 10
                details.append({
                    "label": f"US 10y stiger ({current_rate:.2f}%) → bearish metaller",
                    "impact": -10
                })

        # VIX - safe haven demand
        vix = drivers.get("VIX")
        if vix:
            vix_now = vix.get("current", 15) or 15
            if vix_now > 25:
                score += 10
                details.append({
                    "label": f"VIX høj ({vix_now:.0f}) → safe haven bid",
                    "impact": 10
                })
            elif vix_now > 20:
                score += 5
                details.append({
                    "label": f"VIX forhøjet ({vix_now:.0f}) → let bullish",
                    "impact": 5
                })
            elif vix_now < 13:
                score -= 5
                details.append({
                    "label": f"VIX lav ({vix_now:.0f}) → mindre safe haven",
                    "impact": -5
                })

        # S&P 500 - hvis aktier falder, søger folk mod guld
        spy = drivers.get("SPY")
        if spy:
            spy_1m = spy.get("change_1m", 0) or 0
            if spy_1m < -5:
                score += 8
                details.append({
                    "label": f"S&P 500 ned {spy_1m:.1f}% (30d) → safe haven flow",
                    "impact": 8
                })

        # Olie - inflation-proxy
        oil = drivers.get("OIL")
        if oil:
            oil_1m = oil.get("change_1m", 0) or 0
            if oil_1m > 10:
                score += 5
                details.append({
                    "label": f"Olie stiger +{oil_1m:.1f}% (30d) → inflations-hedge",
                    "impact": 5
                })

    except Exception as e:
        details.append({"label": f"Makro-fejl: {str(e)[:80]}", "impact": 0})

    return max(0, min(100, score)), details


def metals_overall_score(ticker, info, hist, indicators_df, drivers=None):
    """
    Kombineret score for metaller.
    Vægte: 60% teknisk + 40% makro (ingen fundamentals)
    """
    from metals_data import fetch_metal_drivers

    if drivers is None:
        drivers = fetch_metal_drivers()

    t_score, t_details = metals_technical_score(indicators_df)
    m_score, m_details = metals_macro_score(drivers)

    overall = t_score * 0.6 + m_score * 0.4

    return {
        "overall": overall,
        "technical": t_score,
        "macro": m_score,
        "drivers": drivers,
        "details": {
            "technical": t_details,
            "macro": m_details,
        }
    }


def metals_recommendation(score, safe_haven_active=False):
    """Anbefaling. Bonus hvis safe-haven miljø."""
    if safe_haven_active:
        score += 3  # Small boost i risk-off perioder

    if score >= 75:
        return "STÆRKT KØB", "#16a34a"
    elif score >= 60:
        return "KØB", "#22c55e"
    elif score >= 45:
        return "HOLD", "#eab308"
    elif score >= 30:
        return "SÆLG", "#f97316"
    else:
        return "STÆRKT SÆLG", "#ef4444"


def is_safe_haven_environment(drivers):
    """Tjekker om markedet er i risk-off / safe-haven mode"""
    if not drivers:
        return False, []

    signals = []

    vix = drivers.get("VIX", {}).get("current", 15) or 15
    if vix > 25:
        signals.append(f"⚡ VIX høj ({vix:.0f})")
    elif vix > 20:
        signals.append(f"⚡ VIX forhøjet ({vix:.0f})")

    spy_1m = drivers.get("SPY", {}).get("change_1m", 0) or 0
    if spy_1m < -5:
        signals.append(f"📉 S&P 500 ned {spy_1m:.1f}% (30d)")

    dxy_1m = drivers.get("DXY", {}).get("change_1m", 0) or 0
    if dxy_1m < -2:
        signals.append(f"💵 USD svækket {dxy_1m:.1f}%")

    return len(signals) >= 1, signals


def calculate_metal_targets(hist, current_price):
    """Beregner købs-zone, stop-loss og targets baseret på ATR & BB"""
    if hist is None or hist.empty or len(hist) < 30 or not current_price:
        return None

    try:
        recent = hist.tail(126) if len(hist) >= 126 else hist  # 6 måneder

        # ATR-baseret stop
        high = recent["High"]
        low = recent["Low"]
        close = recent["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Ranges
        low_6m = recent["Low"].min()
        high_6m = recent["High"].max()

        # Bollinger for targets
        sma20 = recent["Close"].rolling(20).mean().iloc[-1]
        std20 = recent["Close"].rolling(20).std().iloc[-1]
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        return {
            "buy_low": max(low_6m * 1.02, current_price - 2 * atr),
            "buy_high": current_price + 0.5 * atr,
            "stop_loss": current_price - 2.5 * atr,
            "target_short": bb_upper,
            "target_long": high_6m * 1.05,
            "atr": atr,
            "low_6m": low_6m,
            "high_6m": high_6m,
        }
    except Exception as e:
        print(f"target error: {e}")
        return None
