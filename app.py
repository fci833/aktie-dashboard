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

st.set_page_config(page_title="Pro Aktie Dashboard", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #00d4aa, #0099ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;}
    .recommendation-box {padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;}
    .stTabs [data-baseweb="tab-list"] button {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker, period="5y"):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None and "longName" not in info):
            return None
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        return {
            "info": info, "hist": hist,
            "recommendations": tk.recommendations if hasattr(tk, "recommendations") else None,
            "news": tk.news if hasattr(tk, "news") else [],
            "institutional": tk.institutional_holders,
            "insider": tk.insider_transactions if hasattr(tk, "insider_transactions") else None,
        }
    except Exception as e:
        st.error(f"Fejl: {e}")
        return None

def safe(d, key, default=None):
    if d is None: return default
    v = d.get(key, default) if isinstance(d, dict) else default
    if v is None or (isinstance(v, float) and np.isnan(v)): return default
    return v

def fundamental_score(info):
    score = 50
    detaljer = []
    def add(s, label, value):
        nonlocal score
        score += s
        detaljer.append({"label": label, "value": value, "impact": s})

    pe = safe(info, "trailingPE")
    fpe = safe(info, "forwardPE")
    peg = safe(info, "pegRatio")
    pb = safe(info, "priceToBook")
    roe = safe(info, "returnOnEquity")
    de = safe(info, "debtToEquity")
    pm = safe(info, "profitMargins")
    rev_g = safe(info, "revenueGrowth")
    earn_g = safe(info, "earningsGrowth")
    fcf = safe(info, "freeCashflow")
    cur = safe(info, "currentRatio")

    if pe:
        if 0 < pe < 15: add(8, "✅ P/E lav (attraktiv)", f"{pe:.2f}")
        elif pe < 25: add(3, "➖ P/E moderat", f"{pe:.2f}")
        elif pe < 40: add(-3, "⚠️ P/E høj", f"{pe:.2f}")
        else: add(-8, "❌ P/E meget høj", f"{pe:.2f}")
    if pe and fpe and fpe < pe * 0.9:
        add(5, "✅ Forward P/E indikerer vækst", f"{fpe:.2f}")
    if peg:
        if 0 < peg < 1: add(8, "✅ PEG < 1 (undervurderet)", f"{peg:.2f}")
        elif peg > 3: add(-5, "⚠️ PEG høj", f"{peg:.2f}")
    if pb:
        if 0 < pb < 1: add(6, "✅ P/B < 1", f"{pb:.2f}")
        elif pb < 3: add(3, "✅ P/B sund", f"{pb:.2f}")
        elif pb > 8: add(-5, "⚠️ P/B høj", f"{pb:.2f}")
    if roe is not None:
        if roe > 0.20: add(10, "✅ Stærk ROE (>20%)", f"{roe*100:.1f}%")
        elif roe > 0.10: add(5, "✅ God ROE", f"{roe*100:.1f}%")
        elif roe < 0: add(-10, "❌ Negativ ROE", f"{roe*100:.1f}%")
    if de is not None:
        if de < 30: add(6, "✅ Meget lav gæld", f"{de:.0f}")
        elif de < 100: add(3, "✅ Moderat gæld", f"{de:.0f}")
        elif de > 200: add(-8, "❌ Meget høj gæld", f"{de:.0f}")
    if pm is not None:
        if pm > 0.20: add(8, "✅ Stærk profit margin", f"{pm*100:.1f}%")
        elif pm > 0.05: add(3, "➖ Ok margin", f"{pm*100:.1f}%")
        elif pm < 0: add(-10, "❌ Underskud", f"{pm*100:.1f}%")
    if rev_g is not None:
        if rev_g > 0.20: add(8, "✅ Eksplosiv vækst", f"{rev_g*100:.1f}%")
        elif rev_g > 0.10: add(5, "✅ Stærk vækst", f"{rev_g*100:.1f}%")
        elif rev_g > 0: add(2, "➖ Positiv vækst", f"{rev_g*100:.1f}%")
        else: add(-6, "⚠️ Faldende omsætning", f"{rev_g*100:.1f}%")
    if earn_g and earn_g > 0.15:
        add(5, "✅ Indtjeningsvækst", f"{earn_g*100:.1f}%")
    if fcf:
        if fcf > 0: add(5, "✅ Positivt FCF", f"{fcf/1e9:.2f}B")
        else: add(-5, "❌ Negativt FCF", f"{fcf/1e9:.2f}B")
    if cur:
        if cur > 1.5: add(3, "✅ Sund likviditet", f"{cur:.2f}")
        elif cur < 1: add(-4, "⚠️ Likviditetsrisiko", f"{cur:.2f}")

    return max(0, min(100, score)), detaljer

def add_indicators(hist):
    df = hist.copy()
    df["SMA20"] = SMAIndicator(df["Close"], 20).sma_indicator()
    df["SMA50"] = SMAIndicator(df["Close"], 50).sma_indicator()
    df["SMA200"] = SMAIndicator(df["Close"], 200).sma_indicator()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    bb = BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["STOCH_K"] = stoch.stoch()
    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
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

    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]: add(10, "✅ Golden cross", "SMA50 > SMA200")
        else: add(-10, "❌ Death cross", "SMA50 < SMA200")
    if not np.isnan(last["SMA200"]):
        diff = (pris/last["SMA200"]-1)*100
        if pris > last["SMA200"]: add(5, "✅ Pris over SMA200", f"+{diff:.1f}%")
        else: add(-5, "⚠️ Pris under SMA200", f"{diff:.1f}%")
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < 30: add(12, "✅ RSI oversolgt", f"{rsi:.1f}")
        elif rsi < 45: add(5, "➕ RSI svag", f"{rsi:.1f}")
        elif rsi < 60: add(0, "➖ RSI neutral", f"{rsi:.1f}")
        elif rsi < 70: add(-5, "⚠️ RSI hævet", f"{rsi:.1f}")
        else: add(-12, "❌ RSI overkøbt", f"{rsi:.1f}")
    if not np.isnan(last["MACD"]) and not np.isnan(last["MACD_signal"]):
        if last["MACD"] > last["MACD_signal"]: add(8, "✅ MACD bullish", f"{last['MACD']:.3f}")
        else: add(-8, "❌ MACD bearish", f"{last['MACD']:.3f}")
    if not np.isnan(last["STOCH_K"]):
        if last["STOCH_K"] < 20: add(5, "✅ Stochastic oversold", f"{last['STOCH_K']:.1f}")
        elif last["STOCH_K"] > 80: add(-5, "⚠️ Stochastic overbought", f"{last['STOCH_K']:.1f}")
    if not np.isnan(last["ADX"]) and last["ADX"] > 25:
        add(3, "✅ Stærk trend (ADX>25)", f"{last['ADX']:.1f}")
    if len(df) > 20:
        mom = (pris/df["Close"].iloc[-21]-1)*100
        if mom > 10: add(6, "✅ Stærkt 1M momentum", f"+{mom:.1f}%")
        elif mom > 0: add(2, "➕ Positivt momentum", f"+{mom:.1f}%")
        elif mom < -10: add(-6, "⚠️ Negativt momentum", f"{mom:.1f}%")
    if not np.isnan(last["BB_low"]):
        if pris < last["BB_low"]: add(5, "✅ Under nedre Bollinger", f"{pris:.2f}")
        elif pris > last["BB_high"]: add(-5, "⚠️ Over øvre Bollinger", f"{pris:.2f}")

    return max(0, min(100, score)), detaljer

def dcf_valuation(info, growth_rate=None, discount_rate=0.10, terminal_growth=0.025, years=10):
    fcf = safe(info, "freeCashflow")
    shares = safe(info, "sharesOutstanding")
    if not fcf or not shares or fcf <= 0: return None
    if growth_rate is None:
        growth_rate = safe(info, "earningsGrowth") or safe(info, "revenueGrowth") or 0.05
        growth_rate = max(min(growth_rate, 0.25), 0.02)
    cash_flows = []
    for y in range(1, years+1):
        g = growth_rate - (growth_rate - terminal_growth) * (y/years)
        fcf = fcf * (1 + g)
        cash_flows.append(fcf / ((1 + discount_rate) ** y))
    terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)
    enterprise_value = sum(cash_flows) + pv_terminal
    debt = safe(info, "totalDebt", 0)
    cash = safe(info, "totalCash", 0)
    equity_value = enterprise_value - debt + cash
    return {"fair_value": equity_value / shares, "growth_used": growth_rate, "discount_rate": discount_rate}

def risk_metrics(hist, benchmark_hist=None):
    returns = hist["Close"].pct_change().dropna()
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_return - 0.04) / downside if downside > 0 else 0
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    beta = None
    if benchmark_hist is not None and not benchmark_hist.empty:
        bench_ret = benchmark_hist["Close"].pct_change().dropna()
        df = pd.concat([returns, bench_ret], axis=1).dropna()
        df.columns = ["s", "b"]
        if len(df) > 30:
            var_b = df["b"].var()
            beta = df["s"].cov(df["b"]) / var_b if var_b > 0 else None
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": drawdown.min(), "var_95": var_95, "cvar_95": cvar_95, "beta": beta}

def monte_carlo(hist, days=252, simulations=500):
    returns = hist["Close"].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    last_price = hist["Close"].iloc[-1]
    sims = np.zeros((simulations, days))
    for i in range(simulations):
        rand = np.random.normal(mu, sigma, days)
        sims[i] = last_price * np.cumprod(1 + rand)
    return sims, last_price

def backtest_strategy(df):
    d = df.copy().dropna(subset=["SMA50", "SMA200", "RSI"])
    if len(d) < 50: return None
    d["signal"] = 0
    d.loc[(d["SMA50"] > d["SMA200"]) & (d["RSI"] < 70), "signal"] = 1
    d["ret"] = d["Close"].pct_change()
    d["strategy_ret"] = d["signal"].shift(1) * d["ret"]
    d["bh_cum"] = (1 + d["ret"]).cumprod()
    d["strat_cum"] = (1 + d["strategy_ret"]).cumprod()
    return d

def recommendation(score):
    if score >= 75: return "🟢 STÆRKT KØB", "#16a34a", "Excellent"
    if score >= 60: return "🟢 KØB", "#22c55e", "Solid mulighed"
    if score >= 45: return "🟡 HOLD", "#eab308", "Vent og se"
    if score >= 30: return "🔴 SÆLG", "#ef4444", "Svaghedstegn"
    return "🔴 STÆRKT SÆLG", "#b91c1c", "Undgå/exit"

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Indstillinger")
    period = st.selectbox("Historisk periode", ["1y", "2y", "5y", "10y", "max"], index=2)
    st.markdown("---")
    st.markdown("### ⭐ Watchlist")
    if st.session_state.watchlist:
        for w in st.session_state.watchlist:
            ca, cb = st.columns([3, 1])
            ca.write(f"📌 {w}")
            if cb.button("✕", key=f"rm_{w}"):
                st.session_state.watchlist.remove(w)
                st.rerun()
    else:
        st.caption("Tilføj ved at analysere aktier")
    st.markdown("---")
    st.markdown("### 📋 Hurtige tickers")
    quick = {
        "🇺🇸 US": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"],
        "🇩🇰 DK": ["NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO", "CARL-B.CO"],
        "🇪🇺 EU": ["ASML.AS", "MC.PA", "SAP.DE", "NESN.SW"],
    }
    for region, tickers in quick.items():
        with st.expander(region):
            for tk in tickers:
                if st.button(tk, key=f"q_{tk}", use_container_width=True):
                    st.session_state.selected_ticker = tk
    st.markdown("---")
    st.caption("⚠️ Ikke finansiel rådgivning")

# HOVED
st.markdown("<h1 class='main-header'>📈 Pro Aktie Analyse Dashboard</h1>", unsafe_allow_html=True)
st.caption("Institutional-grade analyse · Valideret data · Multi-faktor scoring")

c1, c2, c3 = st.columns([3, 1, 1])
default_t = st.session_state.get("selected_ticker", "AAPL")
ticker = c1.text_input("Ticker symbol", value=default_t).strip().upper()
analyze_btn = c2.button("🔍 Analysér", type="primary", use_container_width=True)
compare_btn = c3.button("⚖️ Sammenlign", use_container_width=True)

if compare_btn:
    st.session_state.compare_mode = True

if st.session_state.get("compare_mode"):
    st.markdown("### ⚖️ Sammenlign aktier")
    tstr = st.text_input("Tickers (komma)", value=f"{ticker},MSFT,GOOGL")
    if st.button("Sammenlign nu"):
        tlist = [x.strip().upper() for x in tstr.split(",")]
        cdata = {}
        prog = st.progress(0)
        for i, tk in enumerate(tlist):
            d = fetch_data(tk, period="2y")
            if d: cdata[tk] = d
            prog.progress((i+1)/len(tlist))
        if cdata:
            fig = go.Figure()
            for tk, d in cdata.items():
                norm = d["hist"]["Close"] / d["hist"]["Close"].iloc[0] * 100
                fig.add_trace(go.Scatter(x=norm.index, y=norm, name=tk))
            fig.update_layout(title="Normaliseret performance (basis=100)", height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            rows = []
            for tk, d in cdata.items():
                i_ = d["info"]
                rows.append({"Ticker": tk, "Pris": safe(i_, "currentPrice"),
                    "Market cap (B)": safe(i_, "marketCap", 0)/1e9, "P/E": safe(i_, "trailingPE"),
                    "P/B": safe(i_, "priceToBook"), "ROE %": safe(i_, "returnOnEquity", 0)*100,
                    "Profit margin %": safe(i_, "profitMargins", 0)*100,
                    "Rev. vækst %": safe(i_, "revenueGrowth", 0)*100, "Beta": safe(i_, "beta")})
            st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
    if st.button("Luk sammenligning"):
        st.session_state.compare_mode = False
        st.rerun()

elif analyze_btn or "selected_ticker" in st.session_state:
    if "selected_ticker" in st.session_state:
        ticker = st.session_state.selected_ticker
        del st.session_state.selected_ticker

    with st.spinner(f"Analyserer {ticker}..."):
        data = fetch_data(ticker, period=period)

    if data is None:
        st.error(f"❌ Kunne ikke hente data for '{ticker}'.")
        st.stop()

    info = data["info"]
    hist = data["hist"]
    df = add_indicators(hist)

    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)

    navn = info.get("longName") or info.get("shortName") or ticker
    pris = info.get("currentPrice") or hist["Close"].iloc[-1]
    valuta = info.get("currency", "USD")
    prev = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else pris)
    change = pris - prev
    change_pct = (change/prev)*100 if prev else 0

    st.markdown(f"## {navn} ({ticker})")
    st.caption(f"🏢 {info.get('sector','?')} · {info.get('industry','?')} · 🌍 {info.get('country','?')} · 💱 {valuta}")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pris", f"{pris:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    mc = info.get("marketCap")
    k2.metric("Market cap", f"{mc/1e9:,.1f}B" if mc else "-")
    k3.metric("P/E", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "-")
    k4.metric("Fwd P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "-")
    k5.metric("Div. yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-")
    k6.metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

    f_score, f_det = fundamental_score(info)
    t_score, t_det = technical_score(df)
    overall = f_score * 0.6 + t_score * 0.4
    f_a, f_c, f_d = recommendation(f_score)
    t_a, t_c, t_d = recommendation(t_score)
    o_a, o_c, o_d = recommendation(overall)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    for col, (label, anb, color, score, desc) in zip([r1, r2, r3], [
        ("🏛️ Langsigtet (12+ mdr)", f_a, f_c, f_score, f_d),
        ("⚡ Kortsigtet (1-3 mdr)", t_a, t_c, t_score, t_d),
        ("🎯 Samlet vurdering", o_a, o_c, overall, "Vægtet 60/40")]):
        col.markdown(f"""<div class='recommendation-box' style='background: {color}22; border: 2px solid {color}'>
            <div style='font-size:0.9rem'>{label}</div>
            <div style='font-size:1.8rem; font-weight:800; color:{color}; margin:0.5rem 0'>{anb}</div>
            <div style='font-size:1.5rem; font-weight:700'>{score:.0f}/100</div>
            <div style='font-size:0.85rem; opacity:0.7'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Charts", "📋 Fundamentals", "🔧 Teknisk", "💎 DCF", "📉 Risiko",
                     "🎲 Monte Carlo", "🔁 Backtest", "👥 Analytikere", "📰 Nyheder", "🏛️ Ejerskab"])

    with tabs[0]:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.175, 0.175],
                            vertical_spacing=0.03, subplot_titles=("Pris", "Volumen", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="Pris"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="purple")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_high"], name="BB up", line=dict(color="rgba(150,150,150,0.5)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB low", line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
                                fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)
        colors = ["#16a34a" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#00d4aa")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0099ff")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="orange")), row=4, col=1)
        mc_colors = ["#16a34a" if v >= 0 else "#ef4444" for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=mc_colors), row=4, col=1)
        fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Fundamentale faktorer")
        df_f = pd.DataFrame(f_det)
        if not df_f.empty:
            fig_f = px.bar(df_f, x="impact", y="label", orientation="h",
                          color="impact", color_continuous_scale="RdYlGn")
            fig_f.update_layout(height=500, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.subheader("Centrale nøgletal")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown("**Værdiansættelse**")
            for k, v in [("P/E", info.get("trailingPE")), ("Fwd P/E", info.get("forwardPE")),
                        ("PEG", info.get("pegRatio")), ("P/B", info.get("priceToBook")),
                        ("P/S", info.get("priceToSalesTrailing12Months")),
                        ("EV/EBITDA", info.get("enterpriseToEbitda"))]:
                st.write(f"- {k}: **{v:.2f}**" if isinstance(v, (int,float)) else f"- {k}: -")
        with cc2:
            st.markdown("**Profitabilitet**")
            for k, v in [("ROE", info.get("returnOnEquity")), ("ROA", info.get("returnOnAssets")),
                        ("Profit margin", info.get("profitMargins")),
                        ("Operating margin", info.get("operatingMargins")),
                        ("Gross margin", info.get("grossMargins"))]:
                st.write(f"- {k}: **{v*100:.2f}%**" if isinstance(v, (int,float)) else f"- {k}: -")
        with cc3:
            st.markdown("**Vækst & balance**")
            for k, v, p in [("Revenue growth", info.get("revenueGrowth"), True),
                           ("Earnings growth", info.get("earningsGrowth"), True),
                           ("Debt/Equity", info.get("debtToEquity"), False),
                           ("Current ratio", info.get("currentRatio"), False),
                           ("Quick ratio", info.get("quickRatio"), False)]:
                if isinstance(v, (int,float)):
                    st.write(f"- {k}: **{v*100:.2f}%**" if p else f"- {k}: **{v:.2f}**")
                else:
                    st.write(f"- {k}: -")

    with tabs[2]:
        st.subheader("Tekniske faktorer")
        df_t = pd.DataFrame(t_det)
        if not df_t.empty:
            fig_t = px.bar(df_t, x="impact", y="label", orientation="h",
                          color="impact", color_continuous_scale="RdYlGn")
            fig_t.update_layout(height=500, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)
            st.dataframe(df_t, use_container_width=True, hide_index=True)
        last = df.iloc[-1]
        cc = st.columns(4)
        cc[0].metric("RSI(14)", f"{last['RSI']:.1f}" if not np.isnan(last['RSI']) else "-")
        cc[1].metric("MACD", f"{last['MACD']:.3f}" if not np.isnan(last['MACD']) else "-")
        cc[2].metric("ADX", f"{last['ADX']:.1f}" if not np.isnan(last['ADX']) else "-")
        cc[3].metric("ATR", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "-")

    with tabs[3]:
        st.subheader("💎 DCF Værdiansættelse")
        cd1, cd2, cd3 = st.columns(3)
        cg = cd1.slider("Vækstrate", 0.0, 0.30, 0.10, 0.01)
        cdr = cd2.slider("Diskontering", 0.05, 0.20, 0.10, 0.01)
        ct = cd3.slider("Terminal vækst", 0.01, 0.05, 0.025, 0.005)
        dcf = dcf_valuation(info, cg, cdr, ct)
        if dcf:
            fair = dcf["fair_value"]
            up = (fair/pris-1)*100
            color = "#16a34a" if up > 0 else "#ef4444"
            d1, d2, d3 = st.columns(3)
            d1.metric("Aktuel pris", f"{pris:.2f} {valuta}")
            d2.metric("DCF fair value", f"{fair:.2f} {valuta}")
            d3.metric("Upside/Downside", f"{up:+.1f}%")
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(x=["Aktuel", "DCF fair"], y=[pris, fair],
                                   marker_color=["#0099ff", color],
                                   text=[f"{pris:.2f}", f"{fair:.2f}"], textposition="outside"))
            fig_d.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.warning("Ikke nok data til DCF")

    with tabs[4]:
        st.subheader("📉 Risikoanalyse")
        bench = fetch_data("SPY", period=period)
        risk = risk_metrics(hist, bench["hist"] if bench else None)
        cr1, cr2, cr3, cr4 = st.columns(4)
        cr1.metric("Ann. afkast", f"{risk['ann_return']*100:.2f}%")
        cr2.metric("Ann. volatilitet", f"{risk['ann_vol']*100:.2f}%")
        cr3.metric("Sharpe", f"{risk['sharpe']:.2f}")
        cr4.metric("Sortino", f"{risk['sortino']:.2f}")
        cr5, cr6, cr7, cr8 = st.columns(4)
        cr5.metric("Max DD", f"{risk['max_drawdown']*100:.2f}%")
        cr6.metric("VaR 95%", f"{risk['var_95']*100:.2f}%")
        cr7.metric("CVaR 95%", f"{risk['cvar_95']*100:.2f}%")
        cr8.metric("Beta SPY", f"{risk['beta']:.2f}" if risk['beta'] else "-")
        rr = hist["Close"].pct_change().dropna()
        cum = (1+rr).cumprod()
        dd = (cum-cum.cummax())/cum.cummax()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd*100, fill="tozeroy", line=dict(color="#ef4444")))
        fig_dd.update_layout(template="plotly_dark", height=350, title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)

    with tabs[5]:
        st.subheader("🎲 Monte Carlo")
        with st.spinner("Simulerer..."):
            sims, lp = monte_carlo(hist, 252, 500)
        final = sims[:, -1]
        p5, p50, p95 = np.percentile(final, [5, 50, 95])
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Start", f"{lp:.2f}")
        cm2.metric("5% (worst)", f"{p5:.2f}", f"{(p5/lp-1)*100:+.1f}%")
        cm3.metric("Median", f"{p50:.2f}", f"{(p50/lp-1)*100:+.1f}%")
        cm4.metric("95% (best)", f"{p95:.2f}", f"{(p95/lp-1)*100:+.1f}%")
        fig_m = go.Figure()
        for i in range(min(100, len(sims))):
            fig_m.add_trace(go.Scatter(y=sims[i], line=dict(width=0.5, color="rgba(0,212,170,0.15)"), showlegend=False))
        fig_m.add_trace(go.Scatter(y=np.percentile(sims, 50, axis=0), name="Median", line=dict(color="#00d4aa", width=3)))
        fig_m.add_trace(go.Scatter(y=np.percentile(sims, 95, axis=0), name="95%", line=dict(color="#0099ff", dash="dash")))
        fig_m.add_trace(go.Scatter(y=np.percentile(sims, 5, axis=0), name="5%", line=dict(color="#ef4444", dash="dash")))
        fig_m.update_layout(template="plotly_dark", height=500, title="Monte Carlo 1 år frem")
        st.plotly_chart(fig_m, use_container_width=True)

    with tabs[6]:
        st.subheader("🔁 Backtest")
        bt = backtest_strategy(df)
        if bt is not None:
            fb = bt["bh_cum"].iloc[-1]
            fs = bt["strat_cum"].iloc[-1]
            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("Buy & Hold", f"{(fb-1)*100:+.1f}%")
            cb2.metric("Strategi", f"{(fs-1)*100:+.1f}%")
            cb3.metric("Outperf.", f"{(fs-fb)*100:+.1f} pp")
            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(x=bt.index, y=(bt["bh_cum"]-1)*100, name="B&H", line=dict(color="#0099ff")))
            fig_b.add_trace(go.Scatter(x=bt.index, y=(bt["strat_cum"]-1)*100, name="Strategi", line=dict(color="#00d4aa")))
            fig_b.update_layout(template="plotly_dark", height=450, title="Kumulativ afkast %")
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.warning("Ikke nok data")

    with tabs[7]:
        st.subheader("👥 Analytiker konsensus")
        rec = info.get("recommendationKey", "ingen data")
        tm = info.get("targetMeanPrice")
        th = info.get("targetHighPrice")
        tl = info.get("targetLowPrice")
        tmd = info.get("targetMedianPrice")
        na = info.get("numberOfAnalystOpinions")
        ca1, ca2, ca3, ca4 = st.columns(4)
        ca1.metric("Konsensus", rec.upper().replace("_"," ") if rec else "-")
        ca2.metric("Antal analytikere", na or "-")
        if tm:
            ca3.metric("Mål-pris (gns.)", f"{tm:.2f}", f"{(tm/pris-1)*100:+.1f}%")
        if tmd:
            ca4.metric("Median", f"{tmd:.2f}")
        if tl and th:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(x=["Lav", "Gns.", "Median", "Høj"],
                                   y=[tl, tm or 0, tmd or 0, th],
                                   marker_color=["#ef4444", "#eab308", "#0099ff", "#16a34a"]))
            fig_a.add_hline(y=pris, line_dash="dash", annotation_text=f"Aktuel: {pris:.2f}")
            fig_a.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_a, use_container_width=True)

    with tabs[8]:
        st.subheader("📰 Nyheder")
        news = data.get("news", [])
        if news:
            for n in news[:10]:
                content = n.get("content", n)
                title = content.get("title") if isinstance(content, dict) else n.get("title", "")
                pub = content.get("provider", {}).get("displayName", "") if isinstance(content, dict) else n.get("publisher", "")
                link = (content.get("clickThroughUrl", {}) or {}).get("url") if isinstance(content, dict) else n.get("link", "")
                if title:
                    st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
                    st.caption(pub)
                    st.markdown("---")
        else:
            st.info("Ingen nyheder")

    with tabs[9]:
        st.subheader("🏛️ Ejerskab")
        co1, co2 = st.columns(2)
        with co1:
            st.markdown("**Institutionelle**")
            inst = data.get("institutional")
            if inst is not None and not inst.empty:
                st.dataframe(inst.head(15), use_container_width=True, hide_index=True)
            hi = info.get("heldPercentInstitutions")
            hin = info.get("heldPercentInsiders")
            if hi or hin:
                fig_o = go.Figure(data=[go.Pie(labels=["Inst.", "Insiders", "Andre"],
                    values=[(hi or 0)*100, (hin or 0)*100, max(0, 100-((hi or 0)+(hin or 0))*100)],
                    hole=0.4, marker_colors=["#0099ff", "#00d4aa", "#6b7280"])])
                fig_o.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_o, use_container_width=True)
        with co2:
            st.markdown("**Insider transaktioner**")
            ins = data.get("insider")
            if ins is not None and not ins.empty:
                st.dataframe(ins.head(15), use_container_width=True, hide_index=True)
            sp = info.get("shortPercentOfFloat")
            if sp:
                st.metric("Short interest", f"{sp*100:.2f}%")

    st.markdown("---")
    st.subheader("💾 Eksport")
    ce1, ce2 = st.columns(2)
    with ce1:
        st.download_button("⬇️ Prishistorik CSV", hist.to_csv(),
                          file_name=f"{ticker}_prices.csv", use_container_width=True)
    with ce2:
        report = f"""# {navn} ({ticker})
Dato: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Anbefalinger
- Langsigtet: {f_a} ({f_score:.0f}/100)
- Kortsigtet: {t_a} ({t_score:.0f}/100)
- Samlet: {o_a} ({overall:.0f}/100)

⚠️ Ikke finansiel rådgivning.
"""
        st.download_button("⬇️ Rapport TXT", report, file_name=f"{ticker}_report.txt", use_container_width=True)

else:
    st.info("👆 Indtast en ticker og tryk Analysér")
    st.markdown("""
    ### 🚀 Hvad kan dashboardet?
    - **Multi-faktor scoring**: Fundamental + teknisk
    - **DCF værdiansættelse**
    - **Risiko-metrics**: Sharpe, Sortino, VaR, drawdown, beta
    - **Monte Carlo**: 500 simuleringer
    - **Backtest** mod buy & hold
    - **Peer-sammenligning**
    - **Nyheder, insider, institutional**
    - **Eksport**
    """)
