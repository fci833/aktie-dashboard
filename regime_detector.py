"""Market regime detection - NU MED smart benchmark-valg baseret på country/exchange.

Detekterer bull/bear/sideways/volatile markeder med valgbar eller automatisk benchmark.
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

# 🆕 BENCHMARK MAP: country → benchmark ticker + label
BENCHMARK_MAP = {
    # USA
    "United States": ("SPY", "S&P 500"),
    "US":            ("SPY", "S&P 500"),

    # Danmark
    "Denmark":       ("^OMXC25", "OMXC25"),
    "DK":            ("^OMXC25", "OMXC25"),

    # Sverige
    "Sweden":        ("^OMX", "OMX Stockholm"),
    "SE":            ("^OMX", "OMX Stockholm"),

    # Norge
    "Norway":        ("^OSEAX", "Oslo Børs"),
    "NO":            ("^OSEAX", "Oslo Børs"),

    # Tyskland
    "Germany":       ("^GDAXI", "DAX"),
    "DE":            ("^GDAXI", "DAX"),

    # UK
    "United Kingdom": ("^FTSE", "FTSE 100"),
    "GB":            ("^FTSE", "FTSE 100"),

    # Frankrig
    "France":        ("^FCHI", "CAC 40"),
    "FR":            ("^FCHI", "CAC 40"),

    # Holland
    "Netherlands":   ("^AEX", "AEX"),
    "NL":            ("^AEX", "AEX"),

    # Schweiz
    "Switzerland":   ("^SSMI", "SMI"),
    "CH":            ("^SSMI", "SMI"),

    # Japan
    "Japan":         ("^N225", "Nikkei 225"),
    "JP":            ("^N225", "Nikkei 225"),

    # Hong Kong / Kina
    "Hong Kong":     ("^HSI", "Hang Seng"),
    "HK":            ("^HSI", "Hang Seng"),
    "China":         ("000001.SS", "Shanghai Comp"),
    "CN":            ("000001.SS", "Shanghai Comp"),

    # Generel Europa fallback
    "Europe":        ("^STOXX50E", "Euro Stoxx 50"),
    "EU":            ("^STOXX50E", "Euro Stoxx 50"),
}

# 🆕 Suffix-baseret fallback (når country mangler)
SUFFIX_TO_BENCHMARK = {
    ".CO": ("^OMXC25", "OMXC25"),
    ".ST": ("^OMX", "OMX Stockholm"),
    ".OL": ("^OSEAX", "Oslo Børs"),
    ".DE": ("^GDAXI", "DAX"),
    ".F":  ("^GDAXI", "DAX"),
    ".L":  ("^FTSE", "FTSE 100"),
    ".PA": ("^FCHI", "CAC 40"),
    ".AS": ("^AEX", "AEX"),
    ".SW": ("^SSMI", "SMI"),
    ".T":  ("^N225", "Nikkei 225"),
    ".HK": ("^HSI", "Hang Seng"),
}


def get_benchmark_for_ticker(ticker: str = None, country: str = None):
    """
    🆕 Returnér (benchmark_ticker, label) baseret på country eller ticker-suffix.

    Prioritet:
    1. Country (mest pålidelig)
    2. Ticker suffix (.CO → OMXC25)
    3. Default: SPY
    """
    # 1. Country baseret
    if country and country in BENCHMARK_MAP:
        return BENCHMARK_MAP[country]

    # 2. Suffix baseret
    if ticker:
        ticker_upper = ticker.upper()
        for suffix, (bench, label) in SUFFIX_TO_BENCHMARK.items():
            if ticker_upper.endswith(suffix):
                return bench, label

    # 3. Default
    return "SPY", "S&P 500"


@st.cache_data(ttl=3600, show_spinner=False)
def detect_market_regime(benchmark="SPY", benchmark_label=None):
    """
    Detekterer markedsregime baseret på benchmark.

    Args:
        benchmark: ticker for benchmark (default SPY)
        benchmark_label: pænt label til UI (default = benchmark)
    """
    if benchmark_label is None:
        benchmark_label = benchmark

    try:
        bench = yf.Ticker(benchmark).history(period="1y")
        if bench.empty or len(bench) < 200:
            # Fallback til SPY hvis benchmark fejler
            if benchmark != "SPY":
                return detect_market_regime("SPY", "S&P 500 (fallback)")
            return "UNKNOWN", 0, {"benchmark_label": benchmark_label}

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
            "benchmark_label": benchmark_label,
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
        # Fallback til SPY
        if benchmark != "SPY":
            return detect_market_regime("SPY", "S&P 500 (fallback)")
        return "UNKNOWN", 0, {"benchmark_label": benchmark_label}


@st.cache_data(ttl=3600, show_spinner=False)
def detect_combined_regime(ticker: str = None, country: str = None):
    """
    🆕 Smart regime detection: kombinerer LOKAL + GLOBAL regime.

    - US-aktier (AAPL, NVO) → bare SPY
    - Danske (.CO) → OMXC25 + S&P 500 kombineret
    - Tyske (.DE) → DAX + S&P 500 kombineret
    - etc.

    Returns:
        regime: combined regime
        confidence: 0-100
        metrics: dict med både lokal og global info
    """
    # 1. Find lokalt benchmark
    local_bench, local_label = get_benchmark_for_ticker(ticker, country)

    # 2. Detect lokalt
    local_regime, local_conf, local_metrics = detect_market_regime(local_bench, local_label)

    # 3. Hvis lokal allerede er SPY, brug bare den
    if local_bench == "SPY":
        local_metrics["is_combined"] = False
        return local_regime, local_conf, local_metrics

    # 4. Detect global (SPY)
    global_regime, global_conf, global_metrics = detect_market_regime("SPY", "S&P 500")

    # 5. Kombinér: prioritér det "værste" regime hvis de er forskellige
    severity = {
        "BULL": 0,
        "SIDEWAYS": 1,
        "VOLATILE": 2,
        "BEAR": 3,
        "UNKNOWN": 1,
    }

    if severity.get(global_regime, 1) > severity.get(local_regime, 1):
        # Global er værre — flyt mod global
        combined_regime = global_regime
        combined_conf = int(local_conf * 0.5 + global_conf * 0.5)
    else:
        # Lokal er værre eller ens — brug lokal
        combined_regime = local_regime
        combined_conf = int(local_conf * 0.7 + global_conf * 0.3)

    return combined_regime, combined_conf, {
        **local_metrics,
        "is_combined": True,
        "local_regime": local_regime,
        "local_confidence": local_conf,
        "local_label": local_label,
        "global_regime": global_regime,
        "global_confidence": global_conf,
        "global_label": "S&P 500",
        "benchmark_label": f"{local_label} + S&P 500",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def detect_crypto_regime(benchmark="BTC-USD"):
    """Krypto-specifikt regime baseret på BTC."""
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

        returns = close.pct_change().dropna()
        vol_30d = float(returns.tail(30).std() * np.sqrt(365) * 100) if len(returns) >= 30 else 0

        peak = close.cummax()
        drawdown = float((close.iloc[-1] / peak.iloc[-1] - 1) * 100)

        if len(close) >= 90:
            mom_3m = float((close.iloc[-1] / close.iloc[-90] - 1) * 100)
        else:
            mom_3m = 0

        if vol_30d > 80:
            regime = "VOLATILE"
            confidence = min(100, int(vol_30d))
        elif drawdown < -25:
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
            "benchmark_label": "Bitcoin",
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
    """Justér fundamental/teknisk vægte baseret på regime."""
    if asset_type == "crypto":
        weights = {
            "BULL":     (0.30, 0.70),
            "BEAR":     (0.55, 0.45),
            "VOLATILE": (0.50, 0.50),
            "SIDEWAYS": (0.40, 0.60),
            "UNKNOWN":  (0.40, 0.60),
        }
    else:
        weights = {
            "BULL":     (0.45, 0.55),
            "BEAR":     (0.75, 0.25),
            "VOLATILE": (0.70, 0.30),
            "SIDEWAYS": (0.60, 0.40),
            "UNKNOWN":  (0.60, 0.40),
        }
    return weights.get(regime, weights["UNKNOWN"])


def regime_recommendation(regime, score):
    """Justér rec-tærskler baseret på regime."""
    thresholds = {
        "BULL": {"STÆRKT KØB": 73, "KØB": 58, "HOLD": 38, "SÆLG": 23},
        "BEAR": {"STÆRKT KØB": 80, "KØB": 70, "HOLD": 50, "SÆLG": 35},
        "VOLATILE": {"STÆRKT KØB": 78, "KØB": 65, "HOLD": 45, "SÆLG": 30},
        "SIDEWAYS": {"STÆRKT KØB": 75, "KØB": 60, "HOLD": 40, "SÆLG": 25},
        "UNKNOWN": {"STÆRKT KØB": 75, "KØB": 60, "HOLD": 40, "SÆLG": 25},
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
    """🆕 Render regime-banner — viser nu kombineret lokal+global hvis relevant."""
    color = REGIME_COLORS.get(regime, "#6b7280")
    emoji = REGIME_EMOJI.get(regime, "❓")
    desc = REGIME_DESCRIPTIONS.get(regime, "")

    fund_w, tech_w = adjust_weights_for_regime(regime, asset_type)
    benchmark_label = metrics.get("benchmark_label", "S&P 500" if asset_type == "stock" else "Bitcoin")

    # 🆕 Vis combined info hvis relevant
    is_combined = metrics.get("is_combined", False)

    if is_combined:
        local_reg = metrics.get("local_regime", regime)
        local_label = metrics.get("local_label", "Local")
        global_reg = metrics.get("global_regime", "?")
        local_emoji = REGIME_EMOJI.get(local_reg, "")
        global_emoji = REGIME_EMOJI.get(global_reg, "")

        sub_info = (
            f"<div style='font-size:0.85rem;color:#aaa;margin-top:0.4rem'>"
            f"📍 <b>{local_label}:</b> {local_emoji} {local_reg} · "
            f"🌍 <b>S&P 500:</b> {global_emoji} {global_reg}"
            f"</div>"
        )
    else:
        sub_info = ""

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
                {sub_info}
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
                mcols[0].metric(f"{benchmark_label} pris", f"{metrics['spy_price']:,.2f}")
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

            # 🆕 Vis combined-detaljer
            if is_combined:
                st.markdown("---")
                st.markdown("**🌐 Kombineret regime-analyse:**")
                ccols = st.columns(2)
                ccols[0].metric(
                    f"📍 {metrics.get('local_label')}",
                    f"{REGIME_EMOJI.get(metrics.get('local_regime'), '')} {metrics.get('local_regime')}",
                    f"{metrics.get('local_confidence')}% confidence"
                )
                ccols[1].metric(
                    f"🌍 {metrics.get('global_label')}",
                    f"{REGIME_EMOJI.get(metrics.get('global_regime'), '')} {metrics.get('global_regime')}",
                    f"{metrics.get('global_confidence')}% confidence"
                )
                st.caption(
                    "💡 Kombineret regime vægter lokal 70% og global 30%. "
                    "Hvis global er værre end lokal, bruges global (forsigtighedsprincip)."
                )
