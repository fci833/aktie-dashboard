import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# KONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Pro Aktie Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #00d4aa, #0099ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;}
    .metric-card {background: #1a1f2e; padding: 1rem; border-radius: 10px; border-left: 4px solid #00d4aa;}
    .recommendation-box {padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;}
    .stTabs [data-baseweb="tab-list"] button {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "lang" not in st.session_state:
    st.session_state.lang = "DA"

T = {
    "DA": {
        "title": "📈 Pro Aktie Analyse Dashboard",
        "subtitle": "Institutional-grade analyse · Valideret data · Multi-faktor scoring",
        "ticker": "Ticker symbol",
        "analyze": "🔍 Analysér",
        "watchlist": "⭐ Watchlist",
        "long_term": "🏛️ Langsigtet (12+ mdr)",
        "short_term": "⚡ Kortsigtet (1-3 mdr)",
        "overall": "🎯 Samlet vurdering",
    },
    "EN": {
        "title": "📈 Pro Stock Analysis Dashboard",
        "subtitle": "Institutional-grade analysis · Validated data · Multi-factor scoring",
        "ticker": "Ticker symbol",
        "analyze": "🔍 Analyze",
        "watchlist": "⭐ Watchlist",
        "long_term": "🏛️ Long-term (12+ months)",
        "short_term": "⚡ Short-term (1-3 months)",
        "overall": "🎯 Overall verdict",
    }
}

def t(key):
    return T[st.session_state.lang].get(key, key)

# =====================================================
# DATA HENTNING
# =====================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker: str, period: str = "5y"):
    """Henter alle data fra Yahoo Finance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None and "longName" not in info:
            return None
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        return {
            "info": info,
            "hist": hist,
            "financials": tk.financials,
            "balance": tk.balance_sheet,
            "cashflow": tk.cashflow,
            "earnings": getattr(tk, "earnings", pd.DataFrame()),
            "recommendations": tk.recommendations if hasattr(tk, "recommendations") else None,
            "news": tk.news if hasattr(tk, "news") else [],
            "institutional": tk.institutional_holders,
            "insider": tk.insider_transactions if hasattr(tk, "insider_transactions") else None,
            "dividends": tk.dividends,
            "splits": tk.splits,
        }
    except Exception as e:
        st.error(f"Fejl ved hentning: {e}")
        return None

def safe(d, key, default=None):
    if d is None:
        return default
    v = d.get(key, default) if isinstance(d, dict) else default
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v

# =====================================================
# FUNDAMENTAL SCORING
# =====================================================
def fundamental_score(info):
    score = 50
    detaljer = []
    weights = {}

    pe = safe(info, "trailingPE")
    fpe = safe(info, "forwardPE")
    peg = safe(info, "pegRatio")
    pb = safe(info, "priceToBook")
    ps = safe(info, "priceToSalesTrailing12Months")
    roe = safe(info, "returnOnEquity")
    roa = safe(info, "returnOnAssets")
    de = safe(info, "debtToEquity")
    pm = safe(info, "profitMargins")
    om = safe(info, "operatingMargins")
    rev_g = safe(info, "revenueGrowth")
    earn_g = safe(info, "earningsGrowth")
    eps_g = safe(info, "earningsQuarterlyGrowth")
    fcf = safe(info, "freeCashflow")
    ocf = safe(info, "operatingCashflow")
    cur = safe(info, "currentRatio")
    quick = safe(info, "quickRatio")
    div_y = safe(info, "dividendYield")
    payout = safe(info, "payoutRatio")

    def add(condition_score, label, value, weight="medium"):
        nonlocal score
        score += condition_score
        detaljer.append({"label": label, "value": value, "impact": condition_score, "weight": weight})

    if pe:
        if 0 < pe < 15: add(8, "✅ P/E lav (attraktiv)", f"{pe:.2f}", "high")
        elif pe < 25: add(3, "➖ P/E moderat", f"{pe:.2f}", "medium")
        elif pe < 40: add(-3, "⚠️ P/E høj", f"{pe:.2f}", "medium")
        else: add(-8, "❌ P/E meget høj", f"{pe:.2f}", "high")

    if pe and fpe and fpe < pe * 0.9:
        add(5, "✅ Forward P/E indikerer vækst", f"{fpe:.2f}", "medium")

    if peg:
        if 0 < peg < 1: add(8, "✅ PEG < 1 (undervurderet ift. vækst)", f"{peg:.2f}", "high")
        elif peg < 2: add(2, "➖ PEG acceptabel", f"{peg:.2f}", "low")
        elif peg > 3: add(-5, "⚠️ PEG høj", f"{peg:.2f}", "medium")

    if pb:
        if 0 < pb < 1: add(6, "✅ P/B < 1 (book-værdi)", f"{pb:.2f}", "high")
        elif pb < 3: add(3, "✅ P/B sund", f"{pb:.2f}", "medium")
        elif pb > 8: add(-5, "⚠️ P/B høj", f"{pb:.2f}", "medium")

    if ps and ps < 2:
        add(3, "✅ P/S attraktiv", f"{ps:.2f}", "low")

    if roe is not None:
        if roe > 0.20: add(10, "✅ Stærk ROE (>20%)", f"{roe*100:.1f}%", "high")
        elif roe > 0.10: add(5, "✅ God ROE", f"{roe*100:.1f}%", "medium")
        elif roe > 0: add(0, "➖ Lav ROE", f"{roe*100:.1f}%", "low")
        else: add(-10, "❌ Negativ ROE", f"{roe*100:.1f}%", "high")

    if roa and roa > 0.05:
        add(3, "✅ Sund ROA", f"{roa*100:.1f}%", "low")

    if de is not None:
        if de < 30: add(6, "✅ Meget lav gæld", f"{de:.0f}", "medium")
        elif de < 100: add(3, "✅ Moderat gæld", f"{de:.0f}", "low")
        elif de < 200: add(-3, "⚠️ Høj gæld", f"{de:.0f}", "medium")
        else: add(-8, "❌ Meget høj gæld", f"{de:.0f}", "high")

    if pm is not None:
        if pm > 0.20: add(8, "✅ Stærk profit margin", f"{pm*100:.1f}%", "high")
        elif pm > 0.05: add(3, "➖ Ok margin", f"{pm*100:.1f}%", "low")
        elif pm < 0: add(-10, "❌ Underskud", f"{pm*100:.1f}%", "high")

    if om and om > 0.15:
        add(4, "✅ Stærk operating margin", f"{om*100:.1f}%", "medium")

    if rev_g is not None:
        if rev_g > 0.20: add(8, "✅ Eksplosiv omsætningsvækst", f"{rev_g*100:.1f}%", "high")
        elif rev_g > 0.10: add(5, "✅ Stærk vækst", f"{rev_g*100:.1f}%", "medium")
        elif rev_g > 0: add(2, "➖ Positiv vækst", f"{rev_g*100:.1f}%", "low")
        else: add(-6, "⚠️ Faldende omsætning", f"{rev_g*100:.1f}%", "high")

    if earn_g and earn_g > 0.15:
        add(5, "✅ Indtjeningsvækst", f"{earn_g*100:.1f}%", "medium")

    if fcf:
        if fcf > 0:
            add(5, "✅ Positivt FCF", f"{fcf/1e9:.2f}B", "medium")
        else:
            add(-5, "❌ Negativt FCF", f"{fcf/1e9:.2f}B", "high")

    if cur:
        if cur > 1.5: add(3, "✅ Sund likviditet", f"{cur:.2f}", "low")
        elif cur < 1: add(-4, "⚠️ Likviditetsrisiko", f"{cur:.2f}", "medium")

    if div_y and 0.02 < div_y < 0.08:
        add(2, "✅ Sund dividend yield", f"{div_y*100:.2f}%", "low")
    if payout and 0 < payout < 0.6:
        add(2, "✅ Bæredygtig payout ratio", f"{payout*100:.1f}%", "low")

    return max(0, min(100, score)), detaljer

# =====================================================
# TEKNISK ANALYSE
# =====================================================
def add_indicators(hist):
    df = hist.copy()
    df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    df["SMA50"] = SMAIndicator(df["Close"], 50).sma_indicator()
    df["SMA200"] = SMAIndicator(df["Close"], 200).sma_indicator()
    df["EMA12"] = EMAIndicator(df["Close"], 12).ema_indicator()
    df["EMA26"] = EMAIndicator(df["Close"], 26).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    bb = BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_mid"] = bb.bollinger_mavg()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()
    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    return df

def technical_score(df):
    score = 50
    detaljer = []
    last = df.iloc[-1]
    pris = last["Close"]

    def add(s, label, value):
        nonlocal score
        score += s
        detaljer.append({"label": label, "value": value, "impact": s})

    # Trend
    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            add(10, "✅ Golden cross-tilstand", "SMA50 > SMA200")
        else:
            add(-10, "❌ Death cross-tilstand", "SMA50 < SMA200")

    if not np.isnan(last["SMA200"]):
        diff = (pris/last["SMA200"]-1)*100
        if pris > last["SMA200"]:
            add(5, "✅ Pris over SMA200", f"+{diff:.1f}%")
        else:
            add(-5, "⚠️ Pris under SMA200", f"{diff:.1f}%")

    # RSI
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < 30: add(12, "✅ RSI oversolgt", f"{rsi:.1f}")
        elif rsi < 45: add(5, "➕ RSI svag", f"{rsi:.1f}")
        elif rsi < 60: add(0, "➖ RSI neutral", f"{rsi:.1f}")
        elif rsi < 70: add(-5, "⚠️ RSI hævet", f"{rsi:.1f}")
        else: add(-12, "❌ RSI overkøbt", f"{rsi:.1f}")

    # MACD
    if not np.isnan(last["MACD"]) and not np.isnan(last["MACD_signal"]):
        if last["MACD"] > last["MACD_signal"]:
            add(8, "✅ MACD bullish", f"{last['MACD']:.3f}")
        else:
            add(-8, "❌ MACD bearish", f"{last['MACD']:.3f}")

    # Stochastic
    if not np.isnan(last["STOCH_K"]):
        if last["STOCH_K"] < 20: add(5, "✅ Stochastic oversold", f"{last['STOCH_K']:.1f}")
        elif last["STOCH_K"] > 80: add(-5, "⚠️ Stochastic overbought", f"{last['STOCH_K']:.1f}")

    # ADX (trendstyrke)
    if not np.isnan(last["ADX"]):
        if last["ADX"] > 25:
            add(3, "✅ Stærk trend (ADX>25)", f"{last['ADX']:.1f}")
        else:
            detaljer.append({"label": "➖ Svag trend (ADX<25)", "value": f"{last['ADX']:.1f}", "impact": 0})

    # Momentum
    if len(df) > 20:
        mom = (pris/df["Close"].iloc[-21]-1)*100
        if mom > 10: add(6, "✅ Stærkt 1M momentum", f"+{mom:.1f}%")
        elif mom > 0: add(2, "➕ Positivt momentum", f"+{mom:.1f}%")
        elif mom < -10: add(-6, "⚠️ Negativt momentum", f"{mom:.1f}%")

    # Bollinger
    if not np.isnan(last["BB_low"]):
        if pris < last["BB_low"]: add(5, "✅ Under nedre Bollinger", f"{pris:.2f}")
        elif pris > last["BB_high"]: add(-5, "⚠️ Over øvre Bollinger", f"{pris:.2f}")

    # Volume trend
    if len(df) > 20:
        vol_ratio = df["Volume"].tail(5).mean() / df["Volume"].tail(20).mean()
        if vol_ratio > 1.5:
            add(3, "✅ Stigende volumen", f"{vol_ratio:.2f}x")

    return max(0, min(100, score)), detaljer

# =====================================================
# DCF VÆRDIANSÆTTELSE
# =====================================================
def dcf_valuation(info, growth_rate=None, discount_rate=0.10, terminal_growth=0.025, years=10):
    """Forsimplet DCF baseret på FCF."""
    fcf = safe(info, "freeCashflow")
    shares = safe(info, "sharesOutstanding")
    if not fcf or not shares or fcf <= 0:
        return None

    if growth_rate is None:
        growth_rate = safe(info, "earningsGrowth") or safe(info, "revenueGrowth") or 0.05
        growth_rate = max(min(growth_rate, 0.25), 0.02)

    cash_flows = []
    for y in range(1, years+1):
        # Aftager gradvist mod terminal growth
        g = growth_rate - (growth_rate - terminal_growth) * (y/years)
        fcf = fcf * (1 + g)
        pv = fcf / ((1 + discount_rate) ** y)
        cash_flows.append(pv)

    terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    enterprise_value = sum(cash_flows) + pv_terminal
    debt = safe(info, "totalDebt", 0)
    cash = safe(info, "totalCash", 0)
    equity_value = enterprise_value - debt + cash
    fair_value = equity_value / shares

    return {
        "fair_value": fair_value,
        "enterprise_value": enterprise_value,
        "growth_used": growth_rate,
        "discount_rate": discount_rate,
    }

# =====================================================
# RISIKO METRICS
# =====================================================
def risk_metrics(hist, benchmark_hist=None):
    returns = hist["Close"].pct_change().dropna()
    daily_mean = returns.mean()
    daily_std = returns.std()
    ann_return = daily_mean * 252
    ann_vol = daily_std * np.sqrt(252)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_return - 0.04) / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    beta = None
    if benchmark_hist is not None and not benchmark_hist.empty:
        bench_ret = benchmark_hist["Close"].pct_change().dropna()
        df = pd.concat([returns, bench_ret], axis=1).dropna()
        df.columns = ["s", "b"]
        if len(df) > 30:
            cov = df["s"].cov(df["b"])
            var_b = df["b"].var()
            beta = cov / var_b if var_b > 0 else None

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
    }

# =====================================================
# MONTE CARLO
# =====================================================
def monte_carlo(hist, days=252, simulations=500):
    returns = hist["Close"].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    last_price = hist["Close"].iloc[-1]

    sims = np.zeros((simulations, days))
    for i in range(simulations):
        rand = np.random.normal(mu, sigma, days)
        path = last_price * np.cumprod(1 + rand)
        sims[i] = path
    return sims, last_price

# =====================================================
# BACKTEST
# =====================================================
def backtest_strategy(df):
    """Simpel SMA50/SMA200 + RSI strategi vs buy & hold."""
    d = df.copy().dropna(subset=["SMA50", "SMA200", "RSI"])
    if len(d) < 50:
        return None

    d["signal"] = 0
    d.loc[(d["SMA50"] > d["SMA200"]) & (d["RSI"] < 70), "signal"] = 1
    d.loc[(d["SMA50"] < d["SMA200"]) | (d["RSI"] > 80), "signal"] = 0

    d["ret"] = d["Close"].pct_change()
    d["strategy_ret"] = d["signal"].shift(1) * d["ret"]
    d["bh_cum"] = (1 + d["ret"]).cumprod()
    d["strat_cum"] = (1 + d["strategy_ret"]).cumprod()
    return d

# =====================================================
# ANBEFALING
# =====================================================
def recommendation(score):
    if score >= 75: return "🟢 STÆRKT KØB", "#16a34a", "Excellent fundamentals/teknik"
    if score >= 60: return "🟢 KØB", "#22c55e", "Solid mulighed"
    if score >= 45: return "🟡 HOLD", "#eab308", "Vent og se"
    if score >= 30: return "🔴 SÆLG", "#ef4444", "Svaghedstegn"
    return "🔴 STÆRKT SÆLG", "#b91c1c", "Undgå/exit"

# =====================================================
# UI - SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Indstillinger")
    st.session_state.lang = st.radio("Sprog / Language", ["DA", "EN"], horizontal=True, index=0 if st.session_state.lang == "DA" else 1)
    period = st.selectbox("Historisk periode", ["1y", "2y", "5y", "10y", "max"], index=2)

    st.markdown("---")
    st.markdown(f"### {t('watchlist')}")
    if st.session_state.watchlist:
        for w in st.session_state.watchlist:
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"📌 {w}")
            if col_b.button("✕", key=f"rm_{w}"):
                st.session_state.watchlist.remove(w)
                st.rerun()
    else:
        st.caption("Tilføj aktier ved at analysere dem")

    st.markdown("---")
    st.markdown("### 📋 Hurtige tickers")
    quick_tickers = {
        "🇺🇸 US": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"],
        "🇩🇰 DK": ["NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO", "CARL-B.CO"],
        "🇪🇺 EU": ["ASML.AS", "MC.PA", "SAP.DE", "NESN.SW"],
    }
    for region, tickers in quick_tickers.items():
        with st.expander(region):
            for tk in tickers:
                if st.button(tk, key=f"q_{tk}", use_container_width=True):
                    st.session_state.selected_ticker = tk

    st.markdown("---")
    st.caption("⚠️ Ikke finansiel rådgivning. Udelukkende informativ.")

# =====================================================
# UI - HOVEDDEL
# =====================================================
st.markdown(f"<h1 class='main-header'>{t('title')}</h1>", unsafe_allow_html=True)
st.caption(t("subtitle"))

col_in1, col_in2, col_in3 = st.columns([3, 1, 1])
default_ticker = st.session_state.get("selected_ticker", "AAPL")
ticker = col_in1.text_input(t("ticker"), value=default_ticker).strip().upper()
analyze_btn = col_in2.button(t("analyze"), type="primary", use_container_width=True)
compare_btn = col_in3.button("⚖️ Sammenlign", use_container_width=True)

# =====================================================
# SAMMENLIGNINGSMODE
# =====================================================
if compare_btn:
    st.session_state.compare_mode = True

if st.session_state.get("compare_mode"):
    st.markdown("### ⚖️ Sammenlign aktier")
    tickers_str = st.text_input("Indtast tickers (komma-separeret)", value=f"{ticker},MSFT,GOOGL")
    if st.button("Sammenlign nu"):
        tickers_list = [x.strip().upper() for x in tickers_str.split(",")]
        comparison_data = {}
        progress = st.progress(0)
        for i, tk in enumerate(tickers_list):
            data = fetch_data(tk, period="2y")
            if data:
                comparison_data[tk] = data
            progress.progress((i+1)/len(tickers_list))

        if comparison_data:
            # Normaliseret performance
            fig = go.Figure()
            for tk, d in comparison_data.items():
                norm = d["hist"]["Close"] / d["hist"]["Close"].iloc[0] * 100
                fig.add_trace(go.Scatter(x=norm.index, y=norm, name=tk, mode="lines"))
            fig.update_layout(title="Normaliseret performance (basis = 100)", height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Nøgletal-tabel
            rows = []
            for tk, d in comparison_data.items():
                info = d["info"]
                rows.append({
                    "Ticker": tk,
                    "Pris": safe(info, "currentPrice"),
                    "Market cap (B)": safe(info, "marketCap", 0)/1e9,
                    "P/E": safe(info, "trailingPE"),
                    "Fwd P/E": safe(info, "forwardPE"),
                    "P/B": safe(info, "priceToBook"),
                    "ROE %": safe(info, "returnOnEquity", 0)*100,
                    "Profit margin %": safe(info, "profitMargins", 0)*100,
                    "Rev. vækst %": safe(info, "revenueGrowth", 0)*100,
                    "Dividend %": safe(info, "dividendYield", 0)*100,
                    "Beta": safe(info, "beta"),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    if st.button("Luk sammenligning"):
        st.session_state.compare_mode = False
        st.rerun()

# =====================================================
# HOVED-ANALYSE
# =====================================================
elif analyze_btn or "selected_ticker" in st.session_state:
    if "selected_ticker" in st.session_state:
        ticker = st.session_state.selected_ticker
        del st.session_state.selected_ticker

    with st.spinner(f"Henter & analyserer {ticker}..."):
        data = fetch_data(ticker, period=period)

    if data is None:
        st.error(f"❌ Kunne ikke hente data for '{ticker}'. Tjek symbolet (fx AAPL, NOVO-B.CO).")
        st.stop()

    info = data["info"]
    hist = data["hist"]
    df = add_indicators(hist)

    # Tilføj til watchlist
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)

    # ---------- HEADER ----------
    navn = info.get("longName") or info.get("shortName") or ticker
    pris = info.get("currentPrice") or hist["Close"].iloc[-1]
    valuta = info.get("currency", "USD")
    prev_close = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else pris)
    change = pris - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0

    st.markdown(f"## {navn} ({ticker})")
    st.caption(f"🏢 {info.get('sector','?')} · {info.get('industry','?')} · 🌍 {info.get('country','?')} · 💱 {valuta}")

    # ---------- KEY METRICS ----------
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pris", f"{pris:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    mc = info.get("marketCap")
    k2.metric("Market cap", f"{mc/1e9:,.1f}B" if mc else "-")
    k3.metric("P/E", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "-")
    k4.metric("Fwd P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "-")
    k5.metric("Div. yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-")
    k6.metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

    # ---------- SCORES ----------
    f_score, f_detaljer = fundamental_score(info)
    t_score, t_detaljer = technical_score(df)
    overall = (f_score * 0.6 + t_score * 0.4)
    f_anb, f_color, f_desc = recommendation(f_score)
    t_anb, t_color, t_desc = recommendation(t_score)
    o_anb, o_color, o_desc = recommendation(overall)

    st.markdown("---")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""<div class='recommendation-box' style='background: {f_color}22; border: 2px solid {f_color}'>
            <div style='font-size:0.9rem'>{t('long_term')}</div>
            <div style='font-size:1.8rem; font-weight:800; color:{f_color}; margin:0.5rem 0'>{f_anb}</div>
            <div style='font-size:1.5rem; font-weight:700'>{f_score:.0f}/100</div>
            <div style='font-size:0.85rem; opacity:0.7'>{f_desc}</div>
        </div>""", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""<div class='recommendation-box' style='background: {t_color}22; border: 2px solid {t_color}'>
            <div style='font-size:0.9rem'>{t('short_term')}</div>
            <div style='font-size:1.8rem; font-weight:800; color:{t_color}; margin:0.5rem 0'>{t_anb}</div>
            <div style='font-size:1.5rem; font-weight:700'>{t_score:.0f}/100</div>
            <div style='font-size:0.85rem; opacity:0.7'>{t_desc}</div>
        </div>""", unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""<div class='recommendation-box' style='background: {o_color}22; border: 2px solid {o_color}'>
            <div style='font-size:0.9rem'>{t('overall')}</div>
            <div style='font-size:1.8rem; font-weight:800; color:{o_color}; margin:0.5rem 0'>{o_anb}</div>
            <div style='font-size:1.5rem; font-weight:700'>{overall:.0f}/100</div>
            <div style='font-size:0.85rem; opacity:0.7'>Vægtet 60/40 (fundamental/teknisk)</div>
        </div>""", unsafe_allow_html=True)

    # ---------- TABS ----------
    tabs = st.tabs([
        "📊 Charts",
        "📋 Fundamentals",
        "🔧 Teknisk",
        "💎 Værdi (DCF)",
        "📉 Risiko",
        "🎲 Monte Carlo",
        "🔁 Backtest",
        "👥 Analytikere",
        "📰 Nyheder",
        "🏛️ Ejerskab",
    ])

    # ---- CHARTS ----
    with tabs[0]:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.15, 0.175, 0.175],
                            vertical_spacing=0.03,
                            subplot_titles=("Pris + indikatorer", "Volumen", "RSI", "MACD"))

        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="Pris"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="purple", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_high"], name="BB upper", line=dict(color="rgba(150,150,150,0.5)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB lower", line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
                                fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)

        colors = ["#16a34a" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors), row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#00d4aa")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0099ff")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="orange")), row=4, col=1)
        macd_colors = ["#16a34a" if v >= 0 else "#ef4444" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=macd_colors), row=4, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # ---- FUNDAMENTALS ----
    with tabs[1]:
        st.subheader("Fundamentale faktorer (vægtet score)")

        df_fund = pd.DataFrame(f_detaljer)
        if not df_fund.empty:
            fig_fund = px.bar(df_fund, x="impact", y="label", orientation="h",
                              color="impact", color_continuous_scale="RdYlGn",
                              title="Fundamental faktor-bidrag til score")
            fig_fund.update_layout(height=500, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_fund, use_container_width=True)
            st.dataframe(df_fund[["label", "value", "impact"]], use_container_width=True, hide_index=True)

        st.subheader("Centrale nøgletal")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Værdiansættelse**")
            for k, v in [
                ("P/E (TTM)", info.get("trailingPE")),
                ("P/E (Forward)", info.get("forwardPE")),
                ("PEG", info.get("pegRatio")),
                ("P/B", info.get("priceToBook")),
                ("P/S", info.get("priceToSalesTrailing12Months")),
                ("EV/EBITDA", info.get("enterpriseToEbitda")),
            ]:
                st.write(f"- {k}: **{v:.2f}**" if isinstance(v, (int,float)) else f"- {k}: -")

        with c2:
            st.markdown("**Profitabilitet**")
            for k, v in [
                ("ROE", info.get("returnOnEquity")),
                ("ROA", info.get("returnOnAssets")),
                ("Profit margin", info.get("profitMargins")),
                ("Operating margin", info.get("operatingMargins")),
                ("Gross margin", info.get("grossMargins")),
            ]:
                st.write(f"- {k}: **{v*100:.2f}%**" if isinstance(v, (int,float)) else f"- {k}: -")

        with c3:
            st.markdown("**Vækst & balance**")
            for k, v, pct in [
                ("Revenue growth", info.get("revenueGrowth"), True),
                ("Earnings growth", info.get("earningsGrowth"), True),
                ("Debt/Equity", info.get("debtToEquity"), False),
                ("Current ratio", info.get("currentRatio"), False),
                ("Quick ratio", info.get("quickRatio"), False),
            ]:
                if isinstance(v, (int,float)):
                    st.write(f"- {k}: **{v*100:.2f}%**" if pct else f"- {k}: **{v:.2f}**")
                else:
                    st.write(f"- {k}: -")

    # ---- TEKNISK ----
    with tabs[2]:
        st.subheader("Tekniske faktorer")
        df_tech = pd.DataFrame(t_detaljer)
        if not df_tech.empty:
            fig_t = px.bar(df_tech, x="impact", y="label", orientation="h",
                           color="impact", color_continuous_scale="RdYlGn",
                           title="Teknisk faktor-bidrag til score")
            fig_t.update_layout(height=500, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)
            st.dataframe(df_tech, use_container_width=True, hide_index=True)

        last = df.iloc[-1]
        st.markdown("**Aktuelle indikatorværdier**")
        cols = st.columns(4)
        cols[0].metric("RSI(14)", f"{last['RSI']:.1f}" if not np.isnan(last['RSI']) else "-")
        cols[1].metric("MACD", f"{last['MACD']:.3f}" if not np.isnan(last['MACD']) else "-")
        cols[2].metric("ADX", f"{last['ADX']:.1f}" if not np.isnan(last['ADX']) else "-")
        cols[3].metric("ATR", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "-")

    # ---- DCF ----
    with tabs[3]:
        st.subheader("💎 DCF Værdiansættelse")
        st.caption("Forsimplet 10-årig discounted cash flow model")

        c1, c2, c3 = st.columns(3)
        custom_growth = c1.slider("Vækstrate (år 1)", 0.0, 0.30, 0.10, 0.01, format="%.2f")
        custom_discount = c2.slider("Diskonteringsrate (WACC)", 0.05, 0.20, 0.10, 0.01, format="%.2f")
        custom_terminal = c3.slider("Terminal vækst", 0.01, 0.05, 0.025, 0.005, format="%.3f")

        dcf = dcf_valuation(info, custom_growth, custom_discount, custom_terminal)
        if dcf:
            fair = dcf["fair_value"]
            upside = (fair / pris - 1) * 100
            color = "#16a34a" if upside > 0 else "#ef4444"

            d1, d2, d3 = st.columns(3)
            d1.metric("Aktuel pris", f"{pris:.2f} {valuta}")
            d2.metric("DCF fair value", f"{fair:.2f} {valuta}")
            d3.metric("Upside/Downside", f"{upside:+.1f}%", delta_color="normal")

            fig_dcf = go.Figure()
            fig_dcf.add_trace(go.Bar(x=["Aktuel pris", "DCF fair value"],
                                     y=[pris, fair],
                                     marker_color=["#0099ff", color],
                                     text=[f"{pris:.2f}", f"{fair:.2f}"],
                                     textposition="outside"))
            fig_dcf.update_layout(template="plotly_dark", height=400, title="Pris vs. DCF estimeret fair value")
            st.plotly_chart(fig_dcf, use_container_width=True)

            st.info(f"⚠️ DCF er en model — meget følsom for dine inputs. Anvendt vækst: {dcf['growth_used']*100:.1f}%, diskontering: {dcf['discount_rate']*100:.1f}%.")
        else:
            st.warning("Ikke nok data (FCF/aktier) til DCF.")

    # ---- RISIKO ----
    with tabs[4]:
        st.subheader("📉 Risikoanalyse")
        # Hent benchmark (SPY)
        bench_data = fetch_data("SPY", period=period)
        bench_hist = bench_data["hist"] if bench_data else None
        risk = risk_metrics(hist, bench_hist)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualiseret afkast", f"{risk['ann_return']*100:.2f}%")
        c2.metric("Annualiseret volatilitet", f"{risk['ann_vol']*100:.2f}%")
        c3.metric("Sharpe ratio", f"{risk['sharpe']:.2f}")
        c4.metric("Sortino ratio", f"{risk['sortino']:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max drawdown", f"{risk['max_drawdown']*100:.2f}%")
        c6.metric("VaR (95%, 1d)", f"{risk['var_95']*100:.2f}%")
        c7.metric("CVaR (95%, 1d)", f"{risk['cvar_95']*100:.2f}%")
        c8.metric("Beta vs SPY", f"{risk['beta']:.2f}" if risk['beta'] else "-")

        # Drawdown chart
        returns = hist["Close"].pct_change().dropna()
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill="tozeroy",
                                    line=dict(color="#ef4444"), name="Drawdown %"))
        fig_dd.update_layout(template="plotly_dark", height=350, title="Underwater chart (drawdowns)")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Returndistribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns*100, nbinsx=50, marker_color="#00d4aa"))
        fig_dist.add_vline(x=risk['var_95']*100, line_dash="dash", line_color="red",
                          annotation_text="VaR 95%")
        fig_dist.update_layout(template="plotly_dark", height=350, title="Daglig afkastfordeling (%)")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ---- MONTE CARLO ----
    with tabs[5]:
        st.subheader("🎲 Monte Carlo simulering")
        st.caption("Simulerer 500 mulige prisbaner over de næste 252 handelsdage")

        with st.spinner("Kører simulering..."):
            sims, last_price = monte_carlo(hist, days=252, simulations=500)

        # Statistik
        final = sims[:, -1]
        p5 = np.percentile(final, 5)
        p50 = np.percentile(final, 50)
        p95 = np.percentile(final, 95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start", f"{last_price:.2f}")
        c2.metric("5% percentil (worst case)", f"{p5:.2f}", f"{(p5/last_price-1)*100:+.1f}%")
        c3.metric("Median (50%)", f"{p50:.2f}", f"{(p50/last_price-1)*100:+.1f}%")
        c4.metric("95% percentil (best case)", f"{p95:.2f}", f"{(p95/last_price-1)*100:+.1f}%")

        fig_mc = go.Figure()
        for i in range(min(100, len(sims))):
            fig_mc.add_trace(go.Scatter(y=sims[i], mode="lines",
                                       line=dict(width=0.5, color="rgba(0,212,170,0.15)"),
                                       showlegend=False, hoverinfo="skip"))
        # Median og percentiles
        median_path = np.percentile(sims, 50, axis=0)
        p5_path = np.percentile(sims, 5, axis=0)
        p95_path = np.percentile(sims, 95, axis=0)
        fig_mc.add_trace(go.Scatter(y=median_path, mode="lines", name="Median", line=dict(color="#00d4aa", width=3)))
        fig_mc.add_trace(go.Scatter(y=p95_path, mode="lines", name="95%", line=dict(color="#0099ff", dash="dash")))
        fig_mc.add_trace(go.Scatter(y=p5_path, mode="lines", name="5%", line=dict(color="#ef4444", dash="dash")))
        fig_mc.update_layout(template="plotly_dark", height=500,
                           title=f"Monte Carlo: 1 år frem ({len(sims)} simuleringer)",
                           xaxis_title="Handelsdage", yaxis_title=f"Pris ({valuta})")
        st.plotly_chart(fig_mc, use_container_width=True)

    # ---- BACKTEST ----
    with tabs[6]:
        st.subheader("🔁 Backtest af signal-strategi")
        st.caption("Strategi: Long når SMA50>SMA200 og RSI<70, ellers cash")
        bt = backtest_strategy(df)
        if bt is not None:
            final_bh = bt["bh_cum"].iloc[-1]
            final_strat = bt["strat_cum"].iloc[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("Buy & Hold", f"{(final_bh-1)*100:+.1f}%")
            c2.metric("Strategi", f"{(final_strat-1)*100:+.1f}%")
            outperf = (final_strat - final_bh) * 100
            c3.metric("Outperformance", f"{outperf:+.1f} pp")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt.index, y=(bt["bh_cum"]-1)*100, name="Buy & Hold", line=dict(color="#0099ff")))
            fig_bt.add_trace(go.Scatter(x=bt.index, y=(bt["strat_cum"]-1)*100, name="Strategi", line=dict(color="#00d4aa")))
            fig_bt.update_layout(template="plotly_dark", height=450, title="Kumulativ afkast (%)")
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning("Ikke nok data til backtest.")

    # ---- ANALYTIKERE ----
    with tabs[7]:
        st.subheader("👥 Analytiker konsensus")
        rec = info.get("recommendationKey", "ingen data")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        target_median = info.get("targetMedianPrice")
        n_analysts = info.get("numberOfAnalystOpinions")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Konsensus", rec.upper().replace("_"," ") if rec else "-")
        c2.metric("Antal analytikere", n_analysts or "-")
        if target_mean:
            upside = (target_mean / pris - 1) * 100
            c3.metric("Mål-pris (gns.)", f"{target_mean:.2f}", f"{upside:+.1f}%")
        if target_median:
            c4.metric("Mål-pris (median)", f"{target_median:.2f}")

        if target_low and target_high:
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=["Lav", "Gns.", "Median", "Høj"],
                                   y=[target_low, target_mean or 0, target_median or 0, target_high],
                                   marker_color=["#ef4444", "#eab308", "#0099ff", "#16a34a"]))
            fig_t.add_hline(y=pris, line_dash="dash", annotation_text=f"Aktuel: {pris:.2f}")
            fig_t.update_layout(template="plotly_dark", height=400, title="Analytiker mål-priser")
            st.plotly_chart(fig_t, use_container_width=True)

        # Recommendations historik
        if data.get("recommendations") is not None and not data["recommendations"].empty:
            st.markdown("**Seneste analytiker-anbefalinger**")
            recs = data["recommendations"].tail(20)
            st.dataframe(recs, use_container_width=True)

    # ---- NYHEDER ----
    with tabs[8]:
        st.subheader("📰 Seneste nyheder")
        news = data.get("news", [])
        if news:
            for n in news[:10]:
                content = n.get("content", n)
                title = content.get("title") if isinstance(content, dict) else n.get("title", "")
                publisher = content.get("provider", {}).get("displayName", "") if isinstance(content, dict) else n.get("publisher", "")
                link = (content.get("clickThroughUrl", {}) or {}).get("url") if isinstance(content, dict) else n.get("link", "")
                pub_date = content.get("pubDate", "") if isinstance(content, dict) else ""

                if title:
                    st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
                    st.caption(f"{publisher} · {pub_date}")
                    st.markdown("---")
        else:
            st.info("Ingen nyheder tilgængelige")

    # ---- EJERSKAB ----
    with tabs[9]:
        st.subheader("🏛️ Ejerskab & insider aktivitet")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Institutionelle investorer**")
            inst = data.get("institutional")
            if inst is not None and not inst.empty:
                st.dataframe(inst.head(15), use_container_width=True, hide_index=True)
            else:
                st.info("Ingen data")

            st.markdown("**Ejerskab fordeling**")
            held_inst = info.get("heldPercentInstitutions")
            held_insider = info.get("heldPercentInsiders")
            if held_inst or held_insider:
                fig_o = go.Figure(data=[go.Pie(
                    labels=["Institutionelle", "Insiders", "Andre/retail"],
                    values=[(held_inst or 0)*100, (held_insider or 0)*100,
                            max(0, 100-((held_inst or 0)+(held_insider or 0))*100)],
                    hole=0.4,
                    marker_colors=["#0099ff", "#00d4aa", "#6b7280"]
                )])
                fig_o.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_o, use_container_width=True)

        with c2:
            st.markdown("**Insider transaktioner**")
            insider = data.get("insider")
            if insider is not None and not insider.empty:
                st.dataframe(insider.head(15), use_container_width=True, hide_index=True)
            else:
                st.info("Ingen data")

            short_pct = info.get("shortPercentOfFloat")
            if short_pct:
                st.metric("Short interest (% of float)", f"{short_pct*100:.2f}%")

    # ---- EXPORT ----
    st.markdown("---")
    st.subheader("💾 Eksport")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        csv = hist.to_csv()
        st.download_button("⬇️ Download prishistorik (CSV)", csv,
                          file_name=f"{ticker}_prices.csv", mime="text/csv", use_container_width=True)
    with col_e2:
        report = f"""# Aktie Analyse Rapport - {navn} ({ticker})
Dato: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Anbefalinger
- Langsigtet: {f_anb} ({f_score:.0f}/100)
- Kortsigtet: {t_anb} ({t_score:.0f}/100)
- Samlet: {o_anb} ({overall:.0f}/100)

## Nøgletal
- Pris: {pris:.2f} {valuta}
- Market cap: {(mc/1e9):.2f}B
- P/E: {info.get('trailingPE')}
- Forward P/E: {info.get('forwardPE')}
- ROE: {(info.get('returnOnEquity') or 0)*100:.2f}%
- Profit margin: {(info.get('profitMargins') or 0)*100:.2f}%

## Risiko
- Annualiseret volatilitet: {risk['ann_vol']*100:.2f}%
- Sharpe ratio: {risk['sharpe']:.2f}
- Max drawdown: {risk['max_drawdown']*100:.2f}%

⚠️ Dette er ikke finansiel rådgivning.
"""
        st.download_button("⬇️ Download rapport (TXT)", report,
                          file_name=f"{ticker}_report.txt", use_container_width=True)

else:
    st.info("👆 Indtast en ticker og tryk **Analysér** for at starte. Brug sidebaren til hurtig adgang.")
    st.markdown("""
    ### 🚀 Hvad kan dashboardet?
    - **Multi-faktor scoring**: Fundamental + teknisk analyse kombineret
    - **DCF værdiansættelse**: Estimer fair value baseret på fremtidige cash flows
    - **Risiko-metrics**: Sharpe, Sortino, VaR, max drawdown, beta
    - **Monte Carlo**: 500 simuleringer af mulige prisbaner
    - **Backtest**: Test signal-strategien mod buy & hold
    - **Peer-sammenligning**: Sammenlign flere aktier side om side
    - **Nyheder, insider, institutional**: Komplet billede
    - **Eksport**: Download prishistorik og rapport
    """)
