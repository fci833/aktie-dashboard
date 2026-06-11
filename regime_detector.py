"""Market regime detection - detekterer bull/bear/sideways/volatile markeder.

Integreres i scoring så vægte og tærskler justeres dynamisk baseret på markedsforhold.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st


REGIME_COLORS = {
    "BULL": "#16a34a",
    "BEAR": "#ef4444",
    "SIDEWAYS": "#eab308",
    "VOLATILE": "#a855f7",
    "UNKNOWN": "#6b7280",
}

REGIME_EMOJI = {
    "BULL": "🐂",
    "BEAR": "🐻",
    "SIDEWAYS": "➡️",
    "VOLATILE": "⚡",
    "UNKNOWN": "❓",
}

REGIME_DESCRIPTIONS = {
    "BULL": "Bull market — momentum og teknisk trend virker. Aggressive KØB acceptable.",
    "BEAR": "Bear market — kvalitet vinder. Vær konservativ, høj score-tærskel for KØB.",
    "SIDEWAYS": "Sideways market — balanceret tilgang. Range trading muligt.",
    "VOLATILE": "Højt volatilt marked — vær forsigtig, fokus på kvalitet og defensive sektorer.",
    "UNKNOWN": "Kunne ikke bestemme regime. Bruger standard-vægte.",
}


@st.cache_data(ttl=3600, show_spinner=False)
def detect_market_regime(benchmark="SPY"):
    """
    Detekterer markedsregime baseret på benchmark (default S&P 500).

    Logik:
    - VOLATILE: 30d volatilitet > 30% (annualiseret)
    - BULL: pris > SMA50 > SMA200, 6m momentum > 5%
    - BEAR: pris < SMA50, pris < SMA200, 6m momentum < -5%
    - SIDEWAYS: alt andet

    Returns:
        regime: str
        confidence: int (0-100)
        metrics: dict med detaljer
    """
    try:
        bench = yf.Ticker(benchmark).history(period="1y")
        if bench.empty or len(bench) < 200:
            return "UNKNOWN", 0, {}

        close = bench["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        price = float(close.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        sma200_val = float(sma200.iloc[-1])

        above_sma50 = price > sma50_val
        above_sma200 = price > sma200_val
        sma50_above_200 = sma50_val > sma200_val

        # Volatilitet (annualiseret)
        returns = close.pct_change().dropna()
        vol_30d = float(returns.tail(30).std() * np.sqrt(252) * 100) if len(returns) >= 30 else 0
        vol_90d = float(returns.tail(90).std() * np.sqrt(252) * 100) if len(returns) >= 90 else 0
        vol_ratio = vol_30d / vol_90d if vol_90d > 0 else 1

        # Drawdown fra peak
        peak = close.cummax()
        drawdown = float((close.iloc[-1] / peak.iloc[-1] - 1) * 100)

        # 6-måneders momentum
        if len(close) >= 126:
            mom_6m = float((close.iloc[-1] / close.iloc[-126] - 1) * 100)
        else:
            mom_6m = 0

        # 1-måneds momentum
        if len(close) >= 21:
            mom_1m = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
        else:
            mom_1m = 0

        # ===== REGIME LOGIK =====
        if vol_30d > 30:
            regime = "VOLATILE"
            confidence = min(100, int(vol_30d * 2))
        elif drawdown < -15:
            regime = "BEAR"
            confidence = min(100, int(50 + abs(drawdown) * 2))
        elif above_sma50 and above_sma200 and sma50_above_200 and mom_6m > 5:
            regime = "BULL"
            confidence = min(100, int(50 + mom_6m * 2))
        elif not above_sma50 and not above_sma200 and mom_6m < -5:
            regime = "BEAR"
            confidence = min(100, int(50 + abs(mom_6m) * 2))
        else:
            regime = "SIDEWAYS"
            confidence = 60

        return regime, confidence, {
            "benchmark": benchmark,
            "spy_price": price,
            "sma50": sma50_val,
            "sma200": sma200_val,
            "vol_30d": vol_30d,
            "vol_90d": vol_90d,
            "vol_ratio": vol_ratio,
            "drawdown_from_peak": drawdown,
            "momentum_6m": mom_6m,
            "momentum_1m": mom_1m,
            "above_sma50": above_sma50,
            "above_sma200": above_sma200,
            "sma50_above_200": sma50_above_200,
        }
    except Exception as e:
        print(f"[detect_market_regime] {e}")
        return "UNKNOWN", 0, {}


@st.cache_data(ttl=3600, show_spinner=False)
def detect_crypto_regime(benchmark="BTC-USD"):
    """
    Krypto-specifikt regime baseret på BTC.
    Krypto er mere volatilt så tærskler er anderledes.
    """
    try:
        bench = yf.Ticker(benchmark).history(period="1y")
        if bench.empty or len(bench) < 100:
            return "UNKNOWN", 0, {}

        close = bench["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(min(200, len(close))).mean()

        price = float(close.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        sma200_val = float(sma200.iloc[-1])

        # Volatilitet (annualiseret) - krypto er mere volatilt
        returns = close.pct_change().dropna()
        vol_30d = float(returns.tail(30).std() * np.sqrt(365) * 100) if len(returns) >= 30 else 0

        # Drawdown
        peak = close.cummax()
        drawdown = float((close.iloc[-1] / peak.iloc[-1] - 1) * 100)

        # Momentum
        if len(close) >= 90:
            mom_3m = float((close.iloc[-1] / close.iloc[-90] - 1) * 100)
        else:
            mom_3m = 0

        # Krypto-specifikke tærskler (mere volatilt)
        if vol_30d > 80:
            regime = "VOLATILE"
            confidence = min(100, int(vol_30d))
        elif drawdown < -25:  # Krypto kan tabe meget mere
            regime = "BEAR"
            confidence = min(100, int(50 + abs(drawdown)))
        elif price > sma50_val and price > sma200_val and mom_3m > 15:
            regime = "BULL"
            confidence = min(100, int(50 + mom_3m))
        elif price < sma50_val and price < sma200_val and mom_3m < -15:
            regime = "BEAR"
            confidence = min(100, int(50 + abs(mom_3m)))
        else:
            regime = "SIDEWAYS"
            confidence = 60

        return regime, confidence, {
            "benchmark": benchmark,
            "btc_price": price,
            "sma50": sma50_val,
            "sma200": sma200_val,
            "vol_30d": vol_30d,
            "drawdown_from_peak": drawdown,
            "momentum_3m": mom_3m,
        }
    except Exception as e:
        print(f"[detect_crypto_regime] {e}")
        return "UNKNOWN", 0, {}


def adjust_weights_for_regime(regime, asset_type="stock"):
    """
    Justér fundamental/teknisk vægte baseret på regime.

    Returns:
        (fundamental_weight, technical_weight)
    """
    if asset_type == "crypto":
        # Krypto har anderledes vægte (mere teknisk-drevet)
        weights = {
            "BULL":     (0.30, 0.70),  # Krypto bull = pure momentum
            "BEAR":     (0.55, 0.45),  # Bear: kig på fundamentals
            "VOLATILE": (0.50, 0.50),  # Vær forsigtig
            "SIDEWAYS": (0.40, 0.60),
            "UNKNOWN":  (0.40, 0.60),
        }
    else:
        weights = {
            "BULL":     (0.45, 0.55),  # Mere teknisk
            "BEAR":     (0.75, 0.25),  # Meget mere fundamental
            "VOLATILE": (0.70, 0.30),  # Kvalitet vinder
            "SIDEWAYS": (0.60, 0.40),  # Standard
            "UNKNOWN":  (0.60, 0.40),
        }
    return weights.get(regime, weights["UNKNOWN"])


def regime_recommendation(regime, score):
    """
    Justér rec-tærskler baseret på regime.
    I bear markets skal score være HØJERE for KØB.
    """
    thresholds = {
        "BULL": {
            "STÆRKT KØB": 73,
            "KØB": 58,
            "HOLD": 38,
            "SÆLG": 23,
        },
        "BEAR": {
            "STÆRKT KØB": 80,  # Skal være meget overbevisende
            "KØB": 70,
            "HOLD": 50,
            "SÆLG": 35,
        },
        "VOLATILE": {
            "STÆRKT KØB": 78,
            "KØB": 65,
            "HOLD": 45,
            "SÆLG": 30,
        },
        "SIDEWAYS": {
            "STÆRKT KØB": 75,
            "KØB": 60,
            "HOLD": 40,
            "SÆLG": 25,
        },
        "UNKNOWN": {
            "STÆRKT KØB": 75,
            "KØB": 60,
            "HOLD": 40,
            "SÆLG": 25,
        },
    }

    t = thresholds.get(regime, thresholds["UNKNOWN"])

    if score >= t["STÆRKT KØB"]:
        return "STÆRKT KØB", "#16a34a"
    elif score >= t["KØB"]:
        return "KØB", "#22c55e"
    elif score >= t["HOLD"]:
        return "HOLD", "#eab308"
    elif score >= t["SÆLG"]:
        return "SÆLG", "#f97316"
    else:
        return "STÆRKT SÆLG", "#ef4444"


def render_regime_banner(regime, confidence, metrics, asset_type="stock"):
    """
    Render et flot regime-banner i Streamlit.
    Kald øverst i analyse-views.
    """
    color = REGIME_COLORS.get(regime, "#6b7280")
    emoji = REGIME_EMOJI.get(regime, "❓")
    desc = REGIME_DESCRIPTIONS.get(regime, "")

    # Vægte for dette regime
    fund_w, tech_w = adjust_weights_for_regime(regime, asset_type)

    benchmark_label = "BTC" if asset_type == "crypto" else "S&P 500"

    banner_html = f"""
    <div style='background:linear-gradient(90deg, {color}33 0%, {color}11 100%);
                padding:1rem 1.5rem;border-radius:12px;
                border-left:5px solid {color};margin:0.5rem 0'>
        <div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap'>
            <div style='font-size:2.5rem'>{emoji}</div>
            <div style='flex:1;min-width:200px'>
                <div style='color:{color};font-weight:bold;font-size:1.3rem'>
                    {regime} MARKET ({benchmark_label})
                </div>
                <div style='color:#aaa;font-size:0.9rem;margin-top:0.2rem'>
                    {desc}
                </div>
            </div>
            <div style='text-align:right'>
                <div style='font-size:0.8rem;color:#888'>CONFIDENCE</div>
                <div style='font-size:1.5rem;font-weight:bold;color:{color}'>
                    {confidence}%
                </div>
            </div>
            <div style='text-align:right;border-left:1px solid #444;padding-left:1rem'>
                <div style='font-size:0.8rem;color:#888'>VÆGTE</div>
                <div style='font-size:0.9rem'>
                    📊 Fund: <b>{int(fund_w*100)}%</b><br>
                    🔧 Tek: <b>{int(tech_w*100)}%</b>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    # Detaljer i expander
    if metrics:
        with st.expander("📊 Regime-detaljer (klik)"):
            mcols = st.columns(4)
            if "spy_price" in metrics:
                mcols[0].metric("Benchmark pris", f"${metrics['spy_price']:.2f}")
            if "btc_price" in metrics:
                mcols[0].metric("BTC pris", f"${metrics['btc_price']:,.0f}")
            if "vol_30d" in metrics:
                mcols[1].metric("Vol 30d", f"{metrics['vol_30d']:.1f}%")
            if "drawdown_from_peak" in metrics:
                mcols[2].metric("Drawdown", f"{metrics['drawdown_from_peak']:.1f}%")
            if "momentum_6m" in metrics:
                mcols[3].metric("6m momentum", f"{metrics['momentum_6m']:+.1f}%")
            elif "momentum_3m" in metrics:
                mcols[3].metric("3m momentum", f"{metrics['momentum_3m']:+.1f}%")

            if "above_sma50" in metrics:
                tcols = st.columns(3)
                tcols[0].metric("Pris > SMA50", "✅" if metrics["above_sma50"] else "❌")
                tcols[1].metric("Pris > SMA200", "✅" if metrics["above_sma200"] else "❌")
                tcols[2].metric("SMA50 > SMA200", "✅" if metrics.get("sma50_above_200") else "❌")
