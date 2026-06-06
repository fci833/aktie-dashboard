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
    score[:200] = np.nan
    return pd.Series(score, index=df.index)


def technical_score(df):
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
    """
    FIXED: Stop-loss og købszone er ALTID under current_price.
    Targets sikrer reasonable risk/reward.
    """
    last = df.iloc[-1]
    atr = last["ATR"] if not np.isnan(last["ATR"]) else current_price * 0.02
    recent = df.tail(252) if len(df) > 252 else df
    week52_high = recent["High"].max()
    week52_low = recent["Low"].min()

    sma50 = last["SMA50"] if not np.isnan(last["SMA50"]) else current_price
    sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else current_price
    bb_low = last["BB_low"] if not np.isnan(last["BB_low"]) else current_price * 0.95
    bb_high = last["BB_high"] if not np.isnan(last["BB_high"]) else current_price * 1.05

    # === STOP-LOSS: ALTID under current_price ===
    # Standard: 2 ATR under nuværende pris
    stop_loss = current_price - 2 * atr

    # Hvis SMA200 er UNDER prisen (uptrend), kan vi bruge den som tightere support
    if sma200 < current_price:
        sma_stop = sma200 - 0.5 * atr
        # Brug kun hvis det giver et tightere stop end 2 ATR
        if sma_stop > stop_loss and sma_stop < current_price:
            stop_loss = sma_stop

    # Sikkerhedsnet: stop SKAL være under prisen (max -1%)
    stop_loss = min(stop_loss, current_price * 0.99)
    # Men ikke mere end 15% under
    stop_loss = max(stop_loss, current_price * 0.85)

    # === KØB ZONE: ALTID ≤ current_price ===
    # Øverste grænse: lige under nuværende pris
    buy_zone_high = current_price * 0.99
    # Nederste grænse: 1.5 ATR under (typisk pullback)
    buy_zone_low = current_price - 1.5 * atr

    # Hvis BB_low er endnu lavere (pris allerede tæt på BB), brug det
    if bb_low < buy_zone_low and bb_low > current_price * 0.85:
        buy_zone_low = bb_low

    # Sikkerhed: buy_low < buy_high
    if buy_zone_low >= buy_zone_high:
        buy_zone_low = buy_zone_high * 0.97

    # === KORT MÅL (1-3 mdr) ===
    target_short = max(bb_high, current_price * 1.05)
    # Maks 15% over current for "kort"
    target_short = min(target_short, current_price * 1.15)

    # === LANG MÅL (6-12 mdr) ===
    if fair_value and fair_value > current_price * 1.05:
        target_long = fair_value
    else:
        target_long = max(current_price * 1.20, week52_high)

    return {
        "buy_low": buy_zone_low,
        "buy_high": buy_zone_high,
        "stop_loss": stop_loss,
        "target_short": target_short,
        "target_long": target_long,
        "week52_high": week52_high,
        "week52_low": week52_low,
        "atr": atr,
    }


def estimate_days_to_target(current_price, target_price, hist):
    """Estimer antal dage til target baseret på recent momentum + volatilitet"""
    if hist is None or len(hist) < 30:
        return None

    pct_to_target = (target_price / current_price - 1)
    if abs(pct_to_target) < 0.001:
        return 0

    # 30-dages momentum
    mom_30d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-30] - 1)
    daily_30 = mom_30d / 30

    # 90-dages momentum (mere stabilt)
    if len(hist) >= 90:
        mom_90d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-90] - 1)
        daily_90 = mom_90d / 90
        # Vægtet gennemsnit (30d får mest vægt for kort sigt)
        avg_daily = daily_30 * 0.6 + daily_90 * 0.4
    else:
        avg_daily = daily_30

    # Hvis momentum er negativ men target er positiv, brug volatilitet som fallback
    if avg_daily <= 0 and pct_to_target > 0:
        daily_vol = hist["Close"].pct_change().std()
        # Antag positiv drift baseret på volatilitet (svarer til ~10% årligt afkast)
        avg_daily = max(0.0004, daily_vol * 0.08)
    elif avg_daily >= 0 and pct_to_target < 0:
        # Reverseret case
        daily_vol = hist["Close"].pct_change().std()
        avg_daily = min(-0.0004, -daily_vol * 0.08)

    if avg_daily == 0:
        return None

    days = pct_to_target / avg_daily

    # Begræns til realistic range
    if days < 0:
        return None
    return max(7, min(int(days), 730))  # 1 uge til 2 år


def generate_action_plan(rec, score, current_price, targets, hist, currency="USD",
                        f_score=None, t_score=None, dcf_upside=None,
                        shares=None, fx_to_dkk=None):
    """
    Genererer en konkret handleplan med:
    - Klare steps
    - Datoer baseret på momentum
    - Risk/reward ratio
    - Konsistens-check (DCF vs anbefaling)
    - Total investering & forventet gevinst
    - DKK-konvertering
    """
    today = pd.Timestamp.now()

    plan = {
        "rec": rec,
        "score": score,
        "steps": [],
        "risk_reward": None,
        "warnings": [],
        "summary": "",
        "totals": None,
    }

    # Hjælpefunktion: format pris med DKK
    def fmt_price(usd_val):
        s = f"**{usd_val:,.2f} {currency}**"
        if fx_to_dkk and currency != "DKK":
            dkk_val = usd_val * fx_to_dkk
            s += f" _(≈ {dkk_val:,.0f} DKK)_"
        return s

    # === KONSISTENS-CHECK ===
    if dcf_upside is not None:
        if "KØB" in rec and dcf_upside < -10:
            plan["warnings"].append(
                f"⚠️ Modellen siger KØB pga. fundamentals/teknik, men DCF "
                f"viser overvurdering ({dcf_upside:+.1f}%). Vær forsigtig — "
                f"aktien kan være dyrt prissat."
            )
        elif "KØB" in rec and -10 <= dcf_upside < 5:
            plan["warnings"].append(
                f"ℹ️ DCF viser FAIR PRICED ({dcf_upside:+.1f}%) — "
                f"opside er begrænset. Anbefaling drevet af stærke fundamentals + teknik."
            )
        elif "SÆLG" in rec and dcf_upside > 20:
            plan["warnings"].append(
                f"⚠️ Modellen siger SÆLG, men DCF viser undervurdering "
                f"({dcf_upside:+.1f}%). Måske midlertidig svaghed?"
            )

    # === RISK/REWARD ===
    if "KØB" in rec or "HOLD" in rec:
        risk = current_price - targets["stop_loss"]
        reward_short = targets["target_short"] - current_price
        reward_long = targets["target_long"] - current_price

        if risk > 0:
            plan["risk_reward"] = {
                "risk_dkk": risk,
                "risk_pct": (risk / current_price) * 100,
                "reward_short_pct": (reward_short / current_price) * 100,
                "reward_long_pct": (reward_long / current_price) * 100,
                "ratio_short": reward_short / risk if risk > 0 else 0,
                "ratio_long": reward_long / risk if risk > 0 else 0,
            }

    # === TOTALS (hvis vi har shares) ===
    if shares and shares > 0 and ("KØB" in rec):
        total_invest_usd = shares * current_price
        # Hvis 1/3 sælges på short target, 1/3 på long, 1/3 lader vi køre (estimat: long target)
        shares_per_third = shares / 3

        # Profit ved hvert target
        profit_short = shares_per_third * (targets["target_short"] - current_price)
        profit_long = shares_per_third * (targets["target_long"] - current_price)
        profit_moon = shares_per_third * (targets["target_long"] * 1.15 - current_price)  # +15% over long
        total_profit_usd = profit_short + profit_long + profit_moon

        # Max tab (hvis stop ramt før gevinst)
        max_loss_usd = shares * (current_price - targets["stop_loss"])

        plan["totals"] = {
            "shares": shares,
            "invest_usd": total_invest_usd,
            "invest_dkk": total_invest_usd * fx_to_dkk if fx_to_dkk else None,
            "profit_short_usd": profit_short,
            "profit_long_usd": profit_long,
            "profit_moon_usd": profit_moon,
            "total_profit_usd": total_profit_usd,
            "total_profit_dkk": total_profit_usd * fx_to_dkk if fx_to_dkk else None,
            "total_profit_pct": (total_profit_usd / total_invest_usd) * 100,
            "max_loss_usd": max_loss_usd,
            "max_loss_dkk": max_loss_usd * fx_to_dkk if fx_to_dkk else None,
        }

    # === DATOER ===
    days_short = estimate_days_to_target(current_price, targets["target_short"], hist)
    days_long = estimate_days_to_target(current_price, targets["target_long"], hist)

    if days_short:
        date_short_obj = today + pd.Timedelta(days=days_short)
        date_short = date_short_obj.strftime("%d. %b %Y")
        weeks_short = days_short / 7
        time_short_str = f"{weeks_short:.0f} uger" if weeks_short < 12 else f"{days_short/30:.1f} mdr"
    else:
        date_short = "1-3 mdr (estimat)"
        time_short_str = "1-3 mdr"

    if days_long:
        date_long_obj = today + pd.Timedelta(days=days_long)
        date_long = date_long_obj.strftime("%d. %b %Y")
        time_long_str = f"{days_long/30:.0f} mdr" if days_long < 365 else f"{days_long/365:.1f} år"
    else:
        date_long = "6-12 mdr (estimat)"
        time_long_str = "6-12 mdr"

    # === STEPS PR. ANBEFALING ===
    if "STÆRKT KØB" in rec or "KØB" in rec:
        rr = plan["risk_reward"]
        if rr and rr["ratio_short"] < 1.5:
            plan["warnings"].append(
                f"⚠️ Risk/Reward er lav ({rr['ratio_short']:.1f}:1 på kort sigt). "
                f"Ideelt set bør den være min. 2:1."
            )

        plan["summary"] = (
            f"Modellen anbefaler **KØB** baseret på stærke fundamentals "
            f"(score: {f_score}/100) og teknisk billede (score: {t_score}/100)."
        )

        # Build steps med shares-info hvis tilgængelig
        shares_text = f"{shares} aktier" if shares else "din position"
        third_text = f"{int(shares/3)} aktier" if shares and shares >= 3 else "1/3 af positionen"

        plan["steps"] = [
            {
                "n": 1, "icon": "🟢",
                "title": "KØB AKTIEN",
                "main": f"Køb **{shares_text}** til markedspris omkring {fmt_price(current_price)}",
                "sub": f"Eller læg limit-ordre i KØB ZONE: {fmt_price(targets['buy_low'])} - {fmt_price(targets['buy_high'])} "
                       f"({(targets['buy_low']/current_price-1)*100:+.1f}% til {(targets['buy_high']/current_price-1)*100:+.1f}%)",
                "color": "#16a34a",
            },
            {
                "n": 2, "icon": "🛑",
                "title": "SÆT STOP-LOSS",
                "main": f"Stop-loss: {fmt_price(targets['stop_loss'])} "
                        f"({(targets['stop_loss']/current_price-1)*100:.1f}% — max tab)",
                "sub": "💡 Brug TRAILING STOP når kursen stiger — så låser du gevinst automatisk (se forklaring nedenfor)",
                "color": "#ef4444",
            },
            {
                "n": 3, "icon": "🎯",
                "title": "TAG GEVINST 1 (Sælg 1/3)",
                "main": f"Sælg **{third_text}** ved {fmt_price(targets['target_short'])} "
                        f"(+{(targets['target_short']/current_price-1)*100:.1f}%)",
                "sub": f"📅 Forventet: **{date_short}** ({time_short_str})",
                "color": "#eab308",
            },
            {
                "n": 4, "icon": "🚀",
                "title": "TAG GEVINST 2 (Sælg 1/3)",
                "main": f"Sælg **endnu {third_text}** ved {fmt_price(targets['target_long'])} "
                        f"(+{(targets['target_long']/current_price-1)*100:.1f}%)",
                "sub": f"📅 Forventet: **{date_long}** ({time_long_str})",
                "color": "#22c55e",
            },
            {
                "n": 5, "icon": "🌙",
                "title": "LAD RESTEN KØRE",
                "main": f"Behold sidste **{third_text}** med trailing stop og lad winneren løbe",
                "sub": "💎 De største gevinster kommer ofte fra de sidste 20% af en position",
                "color": "#a855f7",
            },
        ]

    elif "HOLD" in rec:
        plan["summary"] = (
            f"Modellen anbefaler **HOLD** — score er {score}/100 (ikke stærk nok til køb)."
        )
        plan["steps"] = [
            {
                "n": 1, "icon": "⏸️",
                "title": "VENT MED AT KØBE",
                "main": f"Køb **IKKE** til nuværende pris ({fmt_price(current_price)})",
                "sub": f"Vent til prisen falder til KØB ZONE: {fmt_price(targets['buy_low'])} - {fmt_price(targets['buy_high'])}",
                "color": "#eab308",
            },
            {
                "n": 2, "icon": "👁️",
                "title": "OVERVÅG",
                "main": "Tilføj til watchlist og tjek igen om en uge",
                "sub": "🔔 Sæt evt. prisalarm på din mæglerplatform ved købszonen",
                "color": "#0099ff",
            },
            {
                "n": 3, "icon": "📊",
                "title": "RE-ANALYSÉR HVIS...",
                "main": "Pris falder under købszonen, eller earnings/nyheder ændrer billedet",
                "sub": "Kør analysen igen for at se om scoren er steget over 60 (KØB-niveau)",
                "color": "#a855f7",
            },
        ]

    else:  # SÆLG
        plan["summary"] = (
            f"Modellen anbefaler **SÆLG** — score er {score}/100 (svage fundamentals/teknik)."
        )
        plan["steps"] = [
            {
                "n": 1, "icon": "🔴",
                "title": "KØB IKKE",
                "main": "Modellen advarer mod at købe denne aktie nu",
                "sub": "Der er bedre muligheder — prøv screeneren!",
                "color": "#ef4444",
            },
            {
                "n": 2, "icon": "💼",
                "title": "HVIS DU EJER AKTIEN",
                "main": "Overvej at tage profit eller skære tab",
                "sub": f"Stop-loss niveau: {fmt_price(targets['stop_loss'])} — "
                       f"hvis prisen kommer under, så ud!",
                "color": "#eab308",
            },
            {
                "n": 3, "icon": "🔄",
                "title": "RE-VURDÉR OM 1 MÅNED",
                "main": "Markedsforhold ændrer sig — analysér igen senere",
                "sub": "En aktie kan gå fra SÆLG → KØB efter en korrektion",
                "color": "#0099ff",
            },
        ]

    return plan


def dcf_valuation(info, g, dr, tg, years=10):
    """DCF med fallback: FCF → operating cashflow → net income"""
    shares = safe(info, "sharesOutstanding")
    if not shares:
        return None

    # Prøv FCF først
    fcf = safe(info, "freeCashflow")

    # Fallback 1: Operating cashflow - capex (estimat)
    if not fcf or fcf <= 0:
        ocf = safe(info, "operatingCashflow")
        if ocf and ocf > 0:
            # Antag capex ~25% af OCF (industri-gennemsnit)
            fcf = ocf * 0.75

    # Fallback 2: Net income (mindre præcist)
    if not fcf or fcf <= 0:
        ni = safe(info, "netIncomeToCommon") or safe(info, "netIncome")
        if ni and ni > 0:
            # Net income er lidt højere end FCF typisk
            fcf = ni * 0.85

    # Fallback 3: Earnings × shares
    if not fcf or fcf <= 0:
        eps = safe(info, "trailingEps") or safe(info, "earningsPerShare")
        if eps and eps > 0:
            fcf = eps * shares * 0.85

    if not fcf or fcf <= 0:
        return None

    cfs = []
    for y in range(1, years + 1):
        gy = g - (g - tg) * (y / years)
        fcf = fcf * (1 + gy)
        cfs.append(fcf / ((1 + dr) ** y))
    tv = (fcf * (1 + tg)) / (dr - tg)
    pv_tv = tv / ((1 + dr) ** years)
    ev = sum(cfs) + pv_tv
    debt = safe(info, "totalDebt", 0) or 0
    cash = safe(info, "totalCash", 0) or 0
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
