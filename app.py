import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
from io import StringIO
import time
import requests as plain_requests
import warnings
warnings.filterwarnings("ignore")

try:
    from curl_cffi import requests as curl_requests
    def make_session():
        return curl_requests.Session(impersonate="chrome")
except ImportError:
    def make_session():
        s = plain_requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})
        return s

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

FINNHUB_KEY = None
TWELVE_KEY = None
try:
    FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY")
    TWELVE_KEY = st.secrets.get("TWELVE_DATA_KEY")
except Exception:
    pass

st.set_page_config(page_title="Aktie Dashboard", layout="wide", page_icon="📈")
st.markdown("<h1 style='background:linear-gradient(90deg,#00d4aa,#0099ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>📈 Pro Aktie Analyse Dashboard</h1>", unsafe_allow_html=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "last_source" not in st.session_state:
    st.session_state.last_source = "?"

# ============ TICKER MAPPING ============

# Manuel mapping for kendte aktier hvor Twelve Data bruger andet symbol
MANUAL_TWELVE_MAP = {
    "NOVO-B.CO": ["NOVO-B", "NVO", "NOVOB"],  # NVO er ADR
    "MAERSK-B.CO": ["MAERSK-B", "MAERSKB", "AMKBY"],  # AMKBY er ADR
    "MAERSK-A.CO": ["MAERSK-A", "MAERSKA"],
    "DSV.CO": ["DSV", "DSDVF"],
    "ORSTED.CO": ["ORSTED", "DNNGY"],  # DNNGY er ADR
    "CARL-B.CO": ["CARL-B", "CABGY"],
    "GMAB.CO": ["GMAB", "GMAB"],
    "NDA-DK.CO": ["NDA-DK", "NDA"],
    "TRYG.CO": ["TRYG", "TGVSY"],
    "ASML.AS": ["ASML"],  # ASML er global
    "SAP.DE": ["SAP"],
    "NESN.SW": ["NESN", "NSRGY"],
}

def get_twelve_formats(ticker):
    """Returnerer liste af mulige Twelve Data symboler at prøve"""
    t = ticker.upper().strip()

    # Manuel mapping først (mest pålidelig)
    if t in MANUAL_TWELVE_MAP:
        return MANUAL_TWELVE_MAP[t]

    formats = []
    if ".CO" in t:
        base = t.replace(".CO", "")
        formats = [base, base.replace("-", ""), f"{base}:XCSE"]
    elif ".DE" in t:
        base = t.replace(".DE", "")
        formats = [base, f"{base}:XETR"]
    elif ".AS" in t:
        base = t.replace(".AS", "")
        formats = [base, f"{base}:XAMS"]
    elif ".SW" in t:
        base = t.replace(".SW", "")
        formats = [base, f"{base}:XSWX"]
    elif ".PA" in t:
        base = t.replace(".PA", "")
        formats = [base, f"{base}:XPAR"]
    elif ".L" in t:
        base = t.replace(".L", "")
        formats = [base, f"{base}:XLON"]
    else:
        formats = [t]
    return formats

# ============ DATA FETCHERS ============

def fetch_yahoo(ticker, period="5y"):
    try:
        session = make_session()
        tk = yf.Ticker(ticker, session=session)
        info = tk.info
        if not info or len(info) < 5:
            return None
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        news = []
        try:
            news = tk.news if hasattr(tk, "news") else []
        except: pass
        return {"info": info, "hist": hist, "news": news, "source": "Yahoo Finance"}
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            return "RATE_LIMIT"
        return None

def fetch_twelve_single(symbol, period="5y"):
    """Forsøger ÉN specifik Twelve Data symbol"""
    if not TWELVE_KEY:
        return None
    try:
        base = "https://api.twelvedata.com"

        # Call 1: quote
        q = plain_requests.get(f"{base}/quote", params={
            "symbol": symbol, "apikey": TWELVE_KEY
        }, timeout=15)
        quote = q.json()
        if quote.get("status") == "error" or "code" in quote:
            return None

        # Call 2: time_series
        period_days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650, "max": 5000}.get(period, 1825)
        outputsize = min(period_days, 5000)
        ts = plain_requests.get(f"{base}/time_series", params={
            "symbol": symbol, "interval": "1day",
            "outputsize": outputsize, "apikey": TWELVE_KEY
        }, timeout=20)
        ts_data = ts.json()
        if ts_data.get("status") == "error" or "values" not in ts_data:
            return None

        rows = ts_data["values"]
        hist = pd.DataFrame(rows)
        hist["datetime"] = pd.to_datetime(hist["datetime"])
        hist.set_index("datetime", inplace=True)
        hist = hist.sort_index()
        hist = hist.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
            else:
                hist[col] = 0

        info = {
            "longName": quote.get("name") or symbol,
            "sector": "?", "industry": "?", "country": "?",
            "currency": quote.get("currency", "USD"),
            "currentPrice": float(quote.get("close", 0)) if quote.get("close") else float(hist["Close"].iloc[-1]),
            "previousClose": float(quote.get("previous_close", 0)) if quote.get("previous_close") else None,
            "twelve_symbol_used": symbol,  # Husk hvilket symbol der virkede
        }
        return {"info": info, "hist": hist, "news": [], "source": f"Twelve Data ({symbol})"}
    except Exception:
        return None

def fetch_twelve(ticker, period="5y"):
    """Twelve Data med multi-format retry"""
    if not TWELVE_KEY:
        return None
    formats = get_twelve_formats(ticker)
    for symbol in formats:
        result = fetch_twelve_single(symbol, period)
        if result:
            return result
    return None

def fetch_finnhub(ticker, period="5y"):
    if not FINNHUB_AVAILABLE or not FINNHUB_KEY or "." in ticker:
        return None
    try:
        client = finnhub.Client(api_key=FINNHUB_KEY)
        profile = client.company_profile2(symbol=ticker)
        if not profile or not profile.get("name"):
            return None
        quote = client.quote(ticker)
        if not quote.get("c"):
            return None

        # Validér: pris skal være over 0.01 (filtrér nonsense)
        if quote.get("c", 0) < 0.01:
            return None

        metrics = {}
        try:
            res = client.company_basic_financials(ticker, 'all')
            metrics = res.get('metric', {}) if res else {}
        except:
            metrics = {}

        info = {
            "longName": profile.get("name", ticker),
            "sector": profile.get("finnhubIndustry", "?"),
            "industry": profile.get("finnhubIndustry", "?"),
            "country": profile.get("country", "?"),
            "currency": profile.get("currency", "USD"),
            "currentPrice": quote.get("c"),
            "previousClose": quote.get("pc"),
            "marketCap": (profile.get("marketCapitalization") or 0) * 1e6,
            "sharesOutstanding": (profile.get("shareOutstanding") or 0) * 1e6,
            "trailingPE": metrics.get("peTTM") or metrics.get("peNormalizedAnnual"),
            "priceToBook": metrics.get("pbAnnual"),
            "priceToSalesTrailing12Months": metrics.get("psAnnual"),
            "returnOnEquity": (metrics.get("roeRfy") or 0) / 100 if metrics.get("roeRfy") else None,
            "returnOnAssets": (metrics.get("roaRfy") or 0) / 100 if metrics.get("roaRfy") else None,
            "profitMargins": (metrics.get("netProfitMarginAnnual") or 0) / 100 if metrics.get("netProfitMarginAnnual") else None,
            "operatingMargins": (metrics.get("operatingMarginAnnual") or 0) / 100 if metrics.get("operatingMarginAnnual") else None,
            "debtToEquity": metrics.get("totalDebt/totalEquityAnnual"),
            "currentRatio": metrics.get("currentRatioAnnual"),
            "quickRatio": metrics.get("quickRatioAnnual"),
            "revenueGrowth": (metrics.get("revenueGrowthTTMYoy") or 0) / 100 if metrics.get("revenueGrowthTTMYoy") else None,
            "earningsGrowth": (metrics.get("epsGrowthTTMYoy") or 0) / 100 if metrics.get("epsGrowthTTMYoy") else None,
            "freeCashflow": metrics.get("freeCashFlowAnnual"),
            "totalDebt": metrics.get("totalDebt"),
            "totalCash": metrics.get("cashAndCashEquivalentsQuarterly"),
            "dividendYield": (metrics.get("dividendYieldIndicatedAnnual") or 0) / 100 if metrics.get("dividendYieldIndicatedAnnual") else None,
            "beta": metrics.get("beta"),
        }

        if TWELVE_KEY:
            tw = fetch_twelve(ticker, period)
            if tw:
                return {"info": info, "hist": tw["hist"], "news": [], "source": "Finnhub + Twelve Data"}
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker, period="5y"):
    # 1. Yahoo
    result = fetch_yahoo(ticker, period)
    if result and result != "RATE_LIMIT":
        return result
    rate_limited = result == "RATE_LIMIT"

    # 2. Finnhub + Twelve Data (US: full fundamentals)
    if FINNHUB_KEY and "." not in ticker:
        result = fetch_finnhub(ticker, period)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Finnhub + Twelve Data"
            return result

    # 3. Kun Twelve Data (international, kun pris)
    if TWELVE_KEY:
        result = fetch_twelve(ticker, period)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Twelve Data (kun pris-data)"
            else:
                result["warning"] = f"Begrænset data - kun pris-historik (symbol: {result['info'].get('twelve_symbol_used','?')})"
            return result

    return None

# ============ DIAGNOSE ============

def run_diagnostics(ticker):
    results = []

    # Yahoo
    try:
        t0 = time.time()
        r = fetch_yahoo(ticker, "1y")
        dt = time.time() - t0
        if r == "RATE_LIMIT":
            results.append(("Yahoo Finance", "❌ Rate limited", f"{dt:.1f}s", "IP blokeret af Yahoo"))
        elif r is None:
            results.append(("Yahoo Finance", "❌ Ingen data", f"{dt:.1f}s", "Ticker findes ikke eller fejl"))
        else:
            results.append(("Yahoo Finance", "✅ Virker", f"{dt:.1f}s", f"{len(r['hist'])} dage data"))
    except Exception as e:
        results.append(("Yahoo Finance", "❌ Crash", "-", str(e)[:200]))

    # Twelve Data - test alle formater
    if not TWELVE_KEY:
        results.append(("Twelve Data", "⚠️ Ingen API key", "-", "Tilføj TWELVE_DATA_KEY i secrets"))
    else:
        formats = get_twelve_formats(ticker)
        twelve_details = [f"Forsøger formater: {', '.join(formats)}"]
        success = False
        for symbol in formats:
            try:
                t0 = time.time()
                raw = plain_requests.get("https://api.twelvedata.com/quote",
                    params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=10).json()
                dt = time.time() - t0

                if raw.get("code") == 429:
                    twelve_details.append(f"  → '{symbol}': RATE LIMIT (vent 60s)")
                    results.append(("Twelve Data", "❌ Rate limit", f"{dt:.1f}s", "\n".join(twelve_details)))
                    success = True  # Skip resten
                    break
                elif raw.get("status") == "error" or "code" in raw:
                    code = raw.get('code', '?')
                    twelve_details.append(f"  → '{symbol}': fejl {code}")
                else:
                    # Test fuld fetch
                    r = fetch_twelve_single(symbol, "1y")
                    if r:
                        twelve_details.append(f"  → '{symbol}': ✅ {r['info'].get('longName')} ({len(r['hist'])} dage)")
                        results.append(("Twelve Data", "✅ Virker", f"{dt:.1f}s", "\n".join(twelve_details)))
                        success = True
                        break
                    else:
                        twelve_details.append(f"  → '{symbol}': quote OK men time_series fejl")
            except Exception as e:
                twelve_details.append(f"  → '{symbol}': crash {str(e)[:100]}")

        if not success:
            results.append(("Twelve Data", "❌ Alle formater fejler", "-", "\n".join(twelve_details)))

    # Finnhub
    if not FINNHUB_KEY:
        results.append(("Finnhub", "⚠️ Ingen API key", "-", "Tilføj FINNHUB_API_KEY"))
    elif "." in ticker:
        results.append(("Finnhub", "⚠️ Springet over", "-", "Gratis tier kun US aktier"))
    else:
        try:
            t0 = time.time()
            client = finnhub.Client(api_key=FINNHUB_KEY)
            profile = client.company_profile2(symbol=ticker)
            quote = client.quote(ticker)
            dt = time.time() - t0
            if not profile or not profile.get("name"):
                results.append(("Finnhub", "❌ Ingen profile", f"{dt:.1f}s",
                    f"Profile response: {str(profile)[:200]}"))
            elif not quote.get("c") or quote.get("c", 0) < 0.01:
                results.append(("Finnhub", "❌ Ingen gyldig quote", f"{dt:.1f}s",
                    f"Quote response: {str(quote)[:200]}"))
            else:
                results.append(("Finnhub", "✅ Virker", f"{dt:.1f}s",
                    f"{profile.get('name')}, pris: {quote.get('c')}"))
        except Exception as e:
            results.append(("Finnhub", "❌ Crash", "-", str(e)[:300]))

    return results

# ============ ANALYSE FUNKTIONER ============

def safe(d, key, default=None):
    if d is None: return default
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)): return default
    return v

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
    adx = ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx.adx()
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    return df

def technical_score(df):
    score, det = 50, []
    last = df.iloc[-1]
    pris = last["Close"]
    def add(s, l, v):
        nonlocal score
        score += s
        det.append({"label": l, "value": v, "impact": s})
    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]: add(10, "✅ Golden cross", "SMA50>SMA200")
        else: add(-10, "❌ Death cross", "SMA50<SMA200")
    if not np.isnan(last["SMA200"]):
        if pris > last["SMA200"]: add(5, "✅ Pris over SMA200", f"+{(pris/last['SMA200']-1)*100:.1f}%")
        else: add(-5, "⚠️ Pris under SMA200", f"{(pris/last['SMA200']-1)*100:.1f}%")
    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi < 30: add(12, "✅ RSI oversolgt", f"{rsi:.1f}")
        elif rsi < 45: add(5, "➕ RSI svag", f"{rsi:.1f}")
        elif rsi < 60: add(0, "➖ RSI neutral", f"{rsi:.1f}")
        elif rsi < 70: add(-5, "⚠️ RSI hævet", f"{rsi:.1f}")
        else: add(-12, "❌ RSI overkøbt", f"{rsi:.1f}")
    if not np.isnan(last["MACD"]):
        if last["MACD"] > last["MACD_signal"]: add(8, "✅ MACD bullish", f"{last['MACD']:.3f}")
        else: add(-8, "❌ MACD bearish", f"{last['MACD']:.3f}")
    if not np.isnan(last["STOCH_K"]):
        if last["STOCH_K"] < 20: add(5, "✅ Stoch oversold", f"{last['STOCH_K']:.1f}")
        elif last["STOCH_K"] > 80: add(-5, "⚠️ Stoch overbought", f"{last['STOCH_K']:.1f}")
    if not np.isnan(last["ADX"]) and last["ADX"] > 25:
        add(3, "✅ Stærk trend", f"ADX={last['ADX']:.1f}")
    if len(df) > 20:
        mom = (pris/df["Close"].iloc[-21]-1)*100
        if mom > 10: add(6, "✅ Stærkt momentum", f"+{mom:.1f}%")
        elif mom > 0: add(2, "➕ Pos. momentum", f"+{mom:.1f}%")
        elif mom < -10: add(-6, "⚠️ Neg. momentum", f"{mom:.1f}%")
    return max(0, min(100, score)), det

def dcf_valuation(info, g, dr, tg, years=10):
    fcf = safe(info, "freeCashflow")
    shares = safe(info, "sharesOutstanding")
    if not fcf or not shares or fcf <= 0: return None
    cfs = []
    for y in range(1, years+1):
        gy = g - (g - tg) * (y/years)
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
    ds = r[r<0].std() * np.sqrt(252)
    sortino = (ann_r - 0.04) / ds if ds > 0 else 0
    cum = (1+r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    var95 = np.percentile(r, 5)
    return {"ann_r": ann_r, "ann_v": ann_v, "sharpe": sharpe, "sortino": sortino,
            "max_dd": dd.min(), "var95": var95, "dd_series": dd}

def monte_carlo(hist, days=252, sims=300):
    r = hist["Close"].pct_change().dropna()
    mu, sigma = r.mean(), r.std()
    lp = hist["Close"].iloc[-1]
    out = np.zeros((sims, days))
    for i in range(sims):
        out[i] = lp * np.cumprod(1 + np.random.normal(mu, sigma, days))
    return out, lp

def recommendation(s):
    if s >= 75: return "🟢 STÆRKT KØB", "#16a34a"
    if s >= 60: return "🟢 KØB", "#22c55e"
    if s >= 45: return "🟡 HOLD", "#eab308"
    if s >= 30: return "🔴 SÆLG", "#ef4444"
    return "🔴 STÆRKT SÆLG", "#b91c1c"

# ============ SIDEBAR ============

with st.sidebar:
    st.markdown("### 📡 Datakilder")
    st.caption("✅ Yahoo Finance")
    st.caption("✅ Twelve Data" if TWELVE_KEY else "⚠️ Twelve Data (no key)")
    st.caption("✅ Finnhub" if FINNHUB_KEY else "⚠️ Finnhub (no key)")

    if st.session_state.last_source != "?":
        st.success(f"Sidst: **{st.session_state.last_source}**")

    st.markdown("---")
    period = st.selectbox("Periode", ["1y", "2y", "5y", "10y", "max"], index=2)
    if st.button("🔄 Ryd cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache ryddet!")
        time.sleep(1)
        st.rerun()
    st.markdown("---")
    st.markdown("### 📋 Hurtige tickers")
    for region, ts in {"🇺🇸 US": ["AAPL","MSFT","GOOGL","NVDA","TSLA"],
                       "🇩🇰 DK": ["NOVO-B.CO","MAERSK-B.CO","DSV.CO","ORSTED.CO"],
                       "🇪🇺 EU": ["ASML.AS","SAP.DE","NESN.SW"]}.items():
        with st.expander(region):
            for tk in ts:
                if st.button(tk, key=f"q_{tk}", use_container_width=True):
                    st.session_state.selected_ticker = tk

# ============ HOVED-UI ============

main_tab, diag_tab = st.tabs(["📊 Analyse", "🔧 Diagnose"])

with diag_tab:
    st.subheader("🔧 Diagnose - Test datakilder")
    st.caption("⏱️ Hvis Twelve Data er rate-limited: vent 60 sekunder før du prøver igen")
    diag_ticker = st.text_input("Test ticker", value="AAPL", key="diag_ticker").strip().upper()
    if st.button("🔍 Kør diagnose", type="primary"):
        with st.spinner(f"Tester alle kilder for {diag_ticker}..."):
            results = run_diagnostics(diag_ticker)
        st.markdown("### Resultater")
        for source, status, time_taken, details in results:
            with st.expander(f"{status} **{source}** ({time_taken})", expanded=True):
                st.code(details)

    st.markdown("---")
    st.markdown("### 📖 Twelve Data symbol mapping")
    st.caption("For internationale aktier prøver vi disse formater i rækkefølge:")
    sample_tickers = ["NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO", "ASML.AS", "SAP.DE", "AAPL"]
    map_data = [{"Yahoo ticker": t, "Twelve Data formater": " → ".join(get_twelve_formats(t))} for t in sample_tickers]
    st.dataframe(pd.DataFrame(map_data), use_container_width=True, hide_index=True)

with main_tab:
    c1, c2 = st.columns([4, 1])
    default_t = st.session_state.get("selected_ticker", "AAPL")
    ticker = c1.text_input("Ticker (fx AAPL, NOVO-B.CO)", value=default_t).strip().upper()
    go_btn = c2.button("🔍 Analysér", type="primary", use_container_width=True)

    if go_btn or "selected_ticker" in st.session_state:
        if "selected_ticker" in st.session_state:
            ticker = st.session_state.selected_ticker
            del st.session_state.selected_ticker

        with st.spinner(f"Henter data for {ticker}..."):
            data = fetch_data(ticker, period=period)

        if data is None:
            st.error(f"❌ Kunne ikke hente data for '{ticker}' fra nogen kilde.")
            st.info("👉 Gå til **🔧 Diagnose** fanen for at se HVAD der fejler!")
            st.stop()

        st.session_state.last_source = data["source"]
        if data.get("warning"):
            st.warning(f"⚠️ {data['warning']}")
        else:
            st.success(f"✅ Data hentet fra: **{data['source']}**")

        info = data["info"]
        hist = data["hist"]
        df = add_indicators(hist)

        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)

        navn = info.get("longName") or ticker
        pris = info.get("currentPrice") or hist["Close"].iloc[-1]
        valuta = info.get("currency", "USD")
        prev = info.get("previousClose", hist["Close"].iloc[-2] if len(hist) > 1 else pris)
        change_pct = (pris/prev-1)*100 if prev else 0

        st.markdown(f"## {navn} ({ticker})")
        st.caption(f"🏢 {info.get('sector','?')} · 🌍 {info.get('country','?')} · 💱 {valuta}")

        k = st.columns(6)
        k[0].metric("Pris", f"{pris:,.2f}", f"{change_pct:+.2f}%")
        mc = info.get("marketCap")
        try:
            k[1].metric("Market cap", f"{float(mc)/1e9:,.1f}B" if mc else "-")
        except:
            k[1].metric("Market cap", "-")
        k[2].metric("P/E", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "-")
        k[3].metric("Fwd P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "-")
        k[4].metric("Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-")
        k[5].metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

        f_score, f_det = fundamental_score(info)
        t_score, t_det = technical_score(df)
        overall = f_score * 0.6 + t_score * 0.4
        f_a, f_c = recommendation(f_score)
        t_a, t_c = recommendation(t_score)
        o_a, o_c = recommendation(overall)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        for col, label, anb, color, sc in [
            (r1, "🏛️ Langsigtet", f_a, f_c, f_score),
            (r2, "⚡ Kortsigtet", t_a, t_c, t_score),
            (r3, "🎯 Samlet", o_a, o_c, overall)]:
            col.markdown(f"<div style='padding:1.5rem;border-radius:12px;text-align:center;background:{color}22;border:2px solid {color}'><div style='font-size:0.9rem'>{label}</div><div style='font-size:1.8rem;font-weight:800;color:{color};margin:0.5rem 0'>{anb}</div><div style='font-size:1.5rem;font-weight:700'>{sc:.0f}/100</div></div>", unsafe_allow_html=True)

        sub_tabs = st.tabs(["📊 Charts", "📋 Fundamentals", "🔧 Teknisk", "💎 DCF", "📉 Risiko", "🎲 Monte Carlo"])

        with sub_tabs[0]:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Pris"), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="orange")), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="purple")), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#00d4aa")), 2, 1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#0099ff")), 3, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="orange")), 3, 1)
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with sub_tabs[1]:
            df_f = pd.DataFrame(f_det)
            if not df_f.empty:
                fig_f = px.bar(df_f, x="impact", y="label", orientation="h", color="impact", color_continuous_scale="RdYlGn")
                fig_f.update_layout(height=500, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_f, use_container_width=True)
                st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info(f"Ingen fundamentale data fra **{data['source']}**")

        with sub_tabs[2]:
            df_t = pd.DataFrame(t_det)
            if not df_t.empty:
                fig_t = px.bar(df_t, x="impact", y="label", orientation="h", color="impact", color_continuous_scale="RdYlGn")
                fig_t.update_layout(height=400, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
            last = df.iloc[-1]
            cc = st.columns(4)
            cc[0].metric("RSI", f"{last['RSI']:.1f}" if not np.isnan(last['RSI']) else "-")
            cc[1].metric("MACD", f"{last['MACD']:.3f}" if not np.isnan(last['MACD']) else "-")
            cc[2].metric("ADX", f"{last['ADX']:.1f}" if not np.isnan(last['ADX']) else "-")
            cc[3].metric("ATR", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "-")

        with sub_tabs[3]:
            c = st.columns(3)
            cg = c[0].slider("Vækstrate", 0.0, 0.30, 0.10, 0.01)
            cdr = c[1].slider("Diskontering", 0.05, 0.20, 0.10, 0.01)
            ct = c[2].slider("Terminal vækst", 0.01, 0.05, 0.025, 0.005)
            fair = dcf_valuation(info, cg, cdr, ct)
            if fair:
                up = (fair/pris-1)*100
                d = st.columns(3)
                d[0].metric("Aktuel pris", f"{pris:.2f}")
                d[1].metric("DCF fair value", f"{fair:.2f}")
                d[2].metric("Upside", f"{up:+.1f}%")
            else:
                st.warning("Ikke nok FCF-data til DCF")

        with sub_tabs[4]:
            risk = risk_metrics(hist)
            c = st.columns(4)
            c[0].metric("Ann. afkast", f"{risk['ann_r']*100:.2f}%")
            c[1].metric("Ann. volatilitet", f"{risk['ann_v']*100:.2f}%")
            c[2].metric("Sharpe", f"{risk['sharpe']:.2f}")
            c[3].metric("Sortino", f"{risk['sortino']:.2f}")
            c2 = st.columns(2)
            c2[0].metric("Max drawdown", f"{risk['max_dd']*100:.2f}%")
            c2[1].metric("VaR 95%", f"{risk['var95']*100:.2f}%")
            fig_dd = go.Figure(go.Scatter(x=risk['dd_series'].index, y=risk['dd_series']*100, fill="tozeroy", line=dict(color="#ef4444")))
            fig_dd.update_layout(template="plotly_dark", height=350, title="Drawdown %")
            st.plotly_chart(fig_dd, use_container_width=True)

        with sub_tabs[5]:
            sims, lp = monte_carlo(hist)
            final = sims[:, -1]
            p5, p50, p95 = np.percentile(final, [5, 50, 95])
            c = st.columns(4)
            c[0].metric("Start", f"{lp:.2f}")
            c[1].metric("5%", f"{p5:.2f}", f"{(p5/lp-1)*100:+.1f}%")
            c[2].metric("Median", f"{p50:.2f}", f"{(p50/lp-1)*100:+.1f}%")
            c[3].metric("95%", f"{p95:.2f}", f"{(p95/lp-1)*100:+.1f}%")
            fig_m = go.Figure()
            for i in range(min(100, len(sims))):
                fig_m.add_trace(go.Scatter(y=sims[i], line=dict(width=0.5, color="rgba(0,212,170,0.15)"), showlegend=False))
            fig_m.add_trace(go.Scatter(y=np.percentile(sims, 50, axis=0), name="Median", line=dict(color="#00d4aa", width=3)))
            fig_m.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("👆 Indtast en ticker og tryk **Analysér**")
        st.markdown("""
        ### 🔥 Funktioner
        - **Hybrid datakilde**: Yahoo → Finnhub+Twelve → Twelve Data
        - **Smart symbol mapping**: NOVO-B.CO → prøver NOVO-B, NVO, NOVOB
        - **Multi-faktor scoring**: Fundamental + teknisk
        - **DCF værdiansættelse**, Monte Carlo, Risiko-metrics

        ### ✅ Garanterede aktier (testet)
        - 🇺🇸 **US**: AAPL, MSFT, TSLA, NVDA — fuld analyse
        - 🇩🇰 **DK**: NOVO-B.CO, MAERSK-B.CO — pris + ADR fundamentals
        """)
