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
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ""

# ============ OPTION A: FASTE ANALYSE-PERIODER ============
ANALYSIS_PERIODS = {
    "technical": 365,    # Tekniske indikatorer: 12 måneder
    "targets": 180,      # Kursmål: 6 måneder (recent volatilitet)
    "risk": 365 * 3,     # Risk metrics: 3 år (statistisk signifikant)
    "monte_carlo": 730,  # Monte Carlo: 2 år (recent regime)
    "week52": 252,       # 52-uger: altid 252 handelsdage
}

# ============ FX RATES ============

@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rate(from_curr, to_curr):
    if from_curr == to_curr:
        return 1.0
    try:
        r = plain_requests.get("https://api.frankfurter.app/latest",
            params={"from": from_curr, "to": to_curr}, timeout=10)
        data = r.json()
        rate = data.get("rates", {}).get(to_curr)
        if rate:
            return float(rate)
    except Exception:
        pass
    fallback = {
        ("USD", "DKK"): 6.85, ("DKK", "USD"): 1/6.85,
        ("EUR", "DKK"): 7.46, ("DKK", "EUR"): 1/7.46,
        ("EUR", "USD"): 1.08, ("USD", "EUR"): 1/1.08,
        ("GBP", "DKK"): 8.70, ("DKK", "GBP"): 1/8.70,
        ("CHF", "DKK"): 7.75, ("DKK", "CHF"): 1/7.75,
        ("SEK", "DKK"): 0.65, ("DKK", "SEK"): 1/0.65,
    }
    return fallback.get((from_curr, to_curr), 1.0)

# ============ TICKER MAPPING ============

MANUAL_TWELVE_MAP = {
    "NOVO-B.CO": ["NOVO-B", "NVO", "NOVOB"],
    "MAERSK-B.CO": ["MAERSK-B", "MAERSKB", "AMKBY"],
    "MAERSK-A.CO": ["MAERSK-A", "MAERSKA"],
    "DSV.CO": ["DSV", "DSDVF"],
    "ORSTED.CO": ["ORSTED", "DNNGY"],
    "CARL-B.CO": ["CARL-B", "CABGY"],
    "GMAB.CO": ["GMAB"],
    "NDA-DK.CO": ["NDA-DK", "NDA"],
    "TRYG.CO": ["TRYG", "TGVSY"],
    "ASML.AS": ["ASML"],
    "SAP.DE": ["SAP"],
    "NESN.SW": ["NESN", "NSRGY"],
}

def get_twelve_formats(ticker):
    t = ticker.upper().strip()
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

# ============ TICKER SEARCH ============

@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query):
    if not query or len(query) < 2:
        return []
    results = []
    if TWELVE_KEY:
        try:
            r = plain_requests.get("https://api.twelvedata.com/symbol_search",
                params={"symbol": query, "outputsize": 20}, timeout=10)
            data = r.json()
            for item in data.get("data", []):
                results.append({
                    "symbol": item.get("symbol"),
                    "name": item.get("instrument_name"),
                    "exchange": item.get("exchange"),
                    "country": item.get("country"),
                    "type": item.get("instrument_type"),
                    "source": "Twelve Data"
                })
        except Exception:
            pass
    if FINNHUB_AVAILABLE and FINNHUB_KEY and len(results) < 10:
        try:
            client = finnhub.Client(api_key=FINNHUB_KEY)
            res = client.symbol_lookup(query)
            for item in res.get("result", [])[:10]:
                if not any(r["symbol"] == item.get("symbol") for r in results):
                    results.append({
                        "symbol": item.get("symbol"),
                        "name": item.get("description"),
                        "exchange": item.get("type"),
                        "country": "?",
                        "type": item.get("type"),
                        "source": "Finnhub"
                    })
        except Exception:
            pass
    return results

# ============ DATA FETCHERS ============

def fetch_yahoo(ticker, period="max"):
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

def fetch_twelve_single(symbol):
    if not TWELVE_KEY:
        return None
    try:
        base = "https://api.twelvedata.com"
        q = plain_requests.get(f"{base}/quote", params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=15)
        quote = q.json()
        if quote.get("status") == "error" or "code" in quote:
            return None
        # Hent altid max data så vi har til alle perioder
        ts = plain_requests.get(f"{base}/time_series", params={
            "symbol": symbol, "interval": "1day",
            "outputsize": 5000, "apikey": TWELVE_KEY
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
            "twelve_symbol_used": symbol,
        }
        return {"info": info, "hist": hist, "news": [], "source": f"Twelve Data ({symbol})"}
    except Exception:
        return None

def fetch_twelve(ticker):
    if not TWELVE_KEY:
        return None
    formats = get_twelve_formats(ticker)
    for symbol in formats:
        result = fetch_twelve_single(symbol)
        if result:
            return result
    return None

def fetch_finnhub(ticker):
    if not FINNHUB_AVAILABLE or not FINNHUB_KEY or "." in ticker:
        return None
    try:
        client = finnhub.Client(api_key=FINNHUB_KEY)
        profile = client.company_profile2(symbol=ticker)
        if not profile or not profile.get("name"):
            return None
        quote = client.quote(ticker)
        if not quote.get("c") or quote.get("c", 0) < 0.01:
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
            tw = fetch_twelve(ticker)
            if tw:
                return {"info": info, "hist": tw["hist"], "news": [], "source": "Finnhub + Twelve Data"}
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    """Henter altid MAX data - filter sker senere baseret på behov"""
    result = fetch_yahoo(ticker, "max")
    if result and result != "RATE_LIMIT":
        return result
    rate_limited = result == "RATE_LIMIT"
    if FINNHUB_KEY and "." not in ticker:
        result = fetch_finnhub(ticker)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Finnhub + Twelve Data"
            return result
    if TWELVE_KEY:
        result = fetch_twelve(ticker)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Twelve Data (kun pris-data)"
            else:
                result["warning"] = f"Begrænset data - kun pris-historik (symbol: {result['info'].get('twelve_symbol_used','?')})"
            return result
    return None

def filter_by_days(hist, days):
    """Filtrerer historik til de seneste N dage"""
    if hist is None or hist.empty:
        return hist
    cutoff = pd.Timestamp.now(tz=hist.index.tz) - pd.Timedelta(days=days)
    return hist[hist.index >= cutoff]

def filter_chart_period(hist, period):
    """Filtrerer til chart-visning"""
    if hist is None or hist.empty or period == "max":
        return hist
    days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}.get(period, 1825)
    return filter_by_days(hist, days)

# ============ DIAGNOSE ============

def run_diagnostics(ticker):
    results = []
    try:
        t0 = time.time()
        r = fetch_yahoo(ticker, "1y")
        dt = time.time() - t0
        if r == "RATE_LIMIT":
            results.append(("Yahoo Finance", "❌ Rate limited", f"{dt:.1f}s", "IP blokeret"))
        elif r is None:
            results.append(("Yahoo Finance", "❌ Ingen data", f"{dt:.1f}s", "Ticker findes ikke"))
        else:
            results.append(("Yahoo Finance", "✅ Virker", f"{dt:.1f}s", f"{len(r['hist'])} dage"))
    except Exception as e:
        results.append(("Yahoo Finance", "❌ Crash", "-", str(e)[:200]))
    if not TWELVE_KEY:
        results.append(("Twelve Data", "⚠️ Ingen API key", "-", "Tilføj TWELVE_DATA_KEY"))
    else:
        formats = get_twelve_formats(ticker)
        twelve_details = [f"Forsøger: {', '.join(formats)}"]
        success = False
        for symbol in formats:
            try:
                t0 = time.time()
                raw = plain_requests.get("https://api.twelvedata.com/quote",
                    params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=10).json()
                dt = time.time() - t0
                if raw.get("code") == 429:
                    results.append(("Twelve Data", "❌ Rate limit", f"{dt:.1f}s", "RATE LIMIT - vent 60s"))
                    success = True
                    break
                elif raw.get("status") == "error" or "code" in raw:
                    twelve_details.append(f"  → '{symbol}': fejl {raw.get('code', '?')}")
                else:
                    r = fetch_twelve_single(symbol)
                    if r:
                        twelve_details.append(f"  → '{symbol}': ✅ {r['info'].get('longName')}")
                        results.append(("Twelve Data", "✅ Virker", f"{dt:.1f}s", "\n".join(twelve_details)))
                        success = True
                        break
            except Exception as e:
                twelve_details.append(f"  → '{symbol}': crash {str(e)[:100]}")
        if not success:
            results.append(("Twelve Data", "❌ Alle formater fejler", "-", "\n".join(twelve_details)))
    if not FINNHUB_KEY:
        results.append(("Finnhub", "⚠️ Ingen API key", "-", "Tilføj FINNHUB_API_KEY"))
    elif "." in ticker:
        results.append(("Finnhub", "⚠️ Springet over", "-", "Kun US"))
    else:
        try:
            t0 = time.time()
            client = finnhub.Client(api_key=FINNHUB_KEY)
            profile = client.company_profile2(symbol=ticker)
            quote = client.quote(ticker)
            dt = time.time() - t0
            if not profile or not profile.get("name"):
                results.append(("Finnhub", "❌ Ingen profile", f"{dt:.1f}s", str(profile)[:200]))
            elif not quote.get("c") or quote.get("c", 0) < 0.01:
                results.append(("Finnhub", "❌ Ingen gyldig quote", f"{dt:.1f}s", str(quote)[:200]))
            else:
                results.append(("Finnhub", "✅ Virker", f"{dt:.1f}s", f"{profile.get('name')}, pris: {quote.get('c')}"))
        except Exception as e:
            results.append(("Finnhub", "❌ Crash", "-", str(e)[:300]))
    return results

# ============ ANALYSE ============

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

def calculate_price_targets(df, current_price, fair_value=None):
    last = df.iloc[-1]
    atr = last["ATR"] if not np.isnan(last["ATR"]) else current_price * 0.02
    recent = df.tail(252) if len(df) > 252 else df
    week52_high = recent["High"].max()
    week52_low = recent["Low"].min()
    sma50 = last["SMA50"] if not np.isnan(last["SMA50"]) else current_price
    sma200 = last["SMA200"] if not np.isnan(last["SMA200"]) else current_price
    bb_low = last["BB_low"] if not np.isnan(last["BB_low"]) else current_price * 0.95
    bb_high = last["BB_high"] if not np.isnan(last["BB_high"]) else current_price * 1.05
    buy_zone_low = max(bb_low, sma200 - atr)
    buy_zone_high = min(sma50, current_price * 0.97)
    stop_loss = max(current_price - 2*atr, sma200 - atr)
    if fair_value and fair_value > current_price:
        target = fair_value
    else:
        target = max(current_price * 1.20, week52_high)
    short_target = bb_high
    return {
        "buy_low": buy_zone_low, "buy_high": buy_zone_high,
        "stop_loss": stop_loss, "target_short": short_target,
        "target_long": target, "week52_high": week52_high,
        "week52_low": week52_low, "atr": atr,
    }

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

def make_price_box(label, value, currency, color, sublabel="", show_secondary=True):
    """HTML boks for kursniveau - INGEN INDRYKNING (Markdown problem!)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        primary_str = "-"
        secondary_div = ""
    else:
        primary_str = f"{value:,.2f} {currency}"
        if show_secondary and currency != "DKK":
            rate = get_fx_rate(currency, "DKK")
            secondary_val = value * rate
            secondary_div = f"<div style='font-size:0.7rem;opacity:0.8;color:#00d4aa'>≈ {secondary_val:,.2f} DKK</div>"
        else:
            secondary_div = ""
    # VIGTIGT: Ingen indrykning på linjerne!
    html = f"<div style='padding:0.8rem;border-radius:8px;background:{color}22;border:1px solid {color}'>"
    html += f"<div style='font-size:0.75rem;opacity:0.7'>{label}</div>"
    html += f"<div style='font-size:1.1rem;font-weight:700'>{primary_str}</div>"
    html += secondary_div
    html += f"<div style='font-size:0.7rem;opacity:0.6'>{sublabel}</div>"
    html += "</div>"
    return html

def make_range_box(label, low, high, currency, color, sublabel="", show_secondary=True):
    """HTML boks for kurs-range (køb zone)"""
    primary_str = f"{low:,.2f} - {high:,.2f} {currency}"
    if show_secondary and currency != "DKK":
        rate = get_fx_rate(currency, "DKK")
        secondary_div = f"<div style='font-size:0.7rem;opacity:0.8;color:#00d4aa'>≈ {low*rate:,.2f} - {high*rate:,.2f} DKK</div>"
    else:
        secondary_div = ""
    html = f"<div style='padding:0.8rem;border-radius:8px;background:{color}22;border:1px solid {color}'>"
    html += f"<div style='font-size:0.75rem;opacity:0.7'>{label}</div>"
    html += f"<div style='font-size:1.0rem;font-weight:700'>{primary_str}</div>"
    html += secondary_div
    html += f"<div style='font-size:0.7rem;opacity:0.6'>{sublabel}</div>"
    html += "</div>"
    return html

# ============ SIDEBAR ============

with st.sidebar:
    st.markdown("### 📡 Datakilder")
    st.caption("✅ Yahoo Finance")
    st.caption("✅ Twelve Data" if TWELVE_KEY else "⚠️ Twelve Data (no key)")
    st.caption("✅ Finnhub" if FINNHUB_KEY else "⚠️ Finnhub (no key)")
    if st.session_state.last_source != "?":
        st.success(f"Sidst: **{st.session_state.last_source}**")
    st.markdown("---")
    st.markdown("### ⚙️ Indstillinger")
    period = st.selectbox(
        "📅 Chart visningsperiode",
        ["1y", "2y", "5y", "10y", "max"], index=2,
        help="Påvirker KUN chart visning. Alle beregninger bruger faste optimerede tidsrammer."
    )
    show_secondary = st.checkbox("💱 Vis priser i DKK også", value=True)
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
                    st.session_state.current_ticker = tk
                    st.rerun()
    st.markdown("---")
    st.markdown("### 💱 Valutakurser")
    st.caption(f"USD/DKK: **{get_fx_rate('USD', 'DKK'):.2f}**")
    st.caption(f"EUR/DKK: **{get_fx_rate('EUR', 'DKK'):.2f}**")
    st.markdown("---")
    st.markdown("### ℹ️ Analyse-perioder")
    st.caption("📊 Tekniske: **12 mdr**")
    st.caption("💰 Kursmål: **6 mdr**")
    st.caption("📉 Risk: **3 år**")
    st.caption("🎲 Monte Carlo: **2 år**")

# ============ HOVED-UI ============

main_tab, search_tab, diag_tab = st.tabs(["📊 Analyse", "🔍 Søg ticker", "🔧 Diagnose"])

with search_tab:
    st.subheader("🔍 Find ticker for et firma")
    st.caption("Skriv firmanavn (fx 'novo nordisk', 'apple', 'maersk')")
    query = st.text_input("Firmanavn", value="", key="search_query", placeholder="novo nordisk")
    if query and len(query) >= 2:
        with st.spinner(f"Søger..."):
            results = search_tickers(query)
        if results:
            st.success(f"Fandt {len(results)} resultater")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.markdown("### 🎯 Vælg en ticker:")
            cols = st.columns(min(4, len(results)))
            for i, r in enumerate(results[:8]):
                if cols[i % 4].button(f"📌 {r['symbol']}\n{r['name'][:25]}", key=f"sr_{i}", use_container_width=True):
                    st.session_state.current_ticker = r['symbol']
                    st.success(f"Valgt: {r['symbol']}")
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.warning("Ingen resultater fundet")
    st.markdown("---")
    examples = pd.DataFrame([
        {"Firma": "Novo Nordisk", "Yahoo": "NOVO-B.CO", "ADR": "NVO"},
        {"Firma": "Mærsk B", "Yahoo": "MAERSK-B.CO", "ADR": "AMKBY"},
        {"Firma": "Ørsted", "Yahoo": "ORSTED.CO", "ADR": "DNNGY"},
        {"Firma": "DSV", "Yahoo": "DSV.CO", "ADR": "DSDVF"},
        {"Firma": "Apple", "Yahoo": "AAPL", "ADR": "AAPL"},
        {"Firma": "ASML", "Yahoo": "ASML.AS", "ADR": "ASML"},
    ])
    st.dataframe(examples, use_container_width=True, hide_index=True)

with diag_tab:
    st.subheader("🔧 Diagnose - Test datakilder")
    diag_ticker = st.text_input("Test ticker", value="AAPL", key="diag_ticker").strip().upper()
    if st.button("🔍 Kør diagnose", type="primary"):
        with st.spinner(f"Tester for {diag_ticker}..."):
            results = run_diagnostics(diag_ticker)
        for source, status, time_taken, details in results:
            with st.expander(f"{status} **{source}** ({time_taken})", expanded=True):
                st.code(details)

with main_tab:
    c1, c2 = st.columns([4, 1])
    default_t = st.session_state.current_ticker or "AAPL"
    ticker_input = c1.text_input("Ticker (fx AAPL, NOVO-B.CO)", value=default_t, key="ticker_input").strip().upper()
    auto_analyze = c2.button("🔍 Analysér", type="primary", use_container_width=True)

    if auto_analyze or st.session_state.current_ticker == ticker_input:
        st.session_state.current_ticker = ticker_input
    ticker = ticker_input

    if ticker:
        with st.spinner(f"Henter data for {ticker}..."):
            data = fetch_data(ticker)

        if data is None:
            st.error(f"❌ Kunne ikke hente data for '{ticker}'")
            st.info("👉 Prøv **🔍 Søg ticker** fanen")
            st.stop()

        st.session_state.last_source = data["source"]
        if data.get("warning"):
            st.warning(f"⚠️ {data['warning']}")
        else:
            st.success(f"✅ Data hentet fra: **{data['source']}**")

        # Info banner om perioder
        st.markdown(
            "<div style='background:#0099ff15;padding:0.6rem 1rem;border-radius:8px;border-left:4px solid #0099ff;margin:0.5rem 0'>"
            f"📅 <b>Chart visningsperiode:</b> {period} · "
            "ℹ️ <b>Beregninger bruger faste tidsrammer:</b> "
            "Tekniske=12mdr · Kursmål=6mdr · Risk=3år · Monte Carlo=2år"
            "</div>", unsafe_allow_html=True)

        info = data["info"]
        hist_full = data["hist"]

        # ===== OPTION A: Faste perioder for hver beregningstype =====
        hist_chart = filter_chart_period(hist_full, period)        # Brugerens valg
        hist_technical = filter_by_days(hist_full, ANALYSIS_PERIODS["technical"])  # 12 mdr
        hist_targets = filter_by_days(hist_full, ANALYSIS_PERIODS["targets"])      # 6 mdr
        hist_risk = filter_by_days(hist_full, ANALYSIS_PERIODS["risk"])            # 3 år
        hist_mc = filter_by_days(hist_full, ANALYSIS_PERIODS["monte_carlo"])       # 2 år

        if len(hist_full) < 200:
            st.warning(f"⚠️ Kun {len(hist_full)} dage total data - SMA200 og andre indikatorer kan være unøjagtige")

        # Indikatorer beregnes på FULD data (så SMA200 er korrekt) og bruges i chart
        df_chart = add_indicators(hist_full)
        df_chart_filtered = df_chart.loc[df_chart.index.isin(hist_chart.index)]
        # Tekniske signaler bruger 12-mdr periode
        df_technical = add_indicators(hist_full).tail(len(hist_technical))
        # Targets bruger 6-mdr data men med SMA200 fra fuld data
        df_targets = add_indicators(hist_full).tail(len(hist_targets))

        if ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)

        navn = info.get("longName") or ticker
        pris = info.get("currentPrice") or hist_full["Close"].iloc[-1]
        valuta = info.get("currency", "USD")
        prev = info.get("previousClose", hist_full["Close"].iloc[-2] if len(hist_full) > 1 else pris)
        change_pct = (pris/prev-1)*100 if prev else 0

        first_date = hist_full.index[0].strftime("%Y-%m-%d")
        last_date = hist_full.index[-1].strftime("%Y-%m-%d")

        st.markdown(f"## {navn} ({ticker})")
        st.caption(f"🏢 {info.get('sector','?')} · 🌍 {info.get('country','?')} · 💱 {valuta} · 📅 Total data: {first_date} → {last_date} ({len(hist_full)} dage)")

        k = st.columns(6)
        k[0].metric("Pris", f"{pris:,.2f} {valuta}", f"{change_pct:+.2f}%")
        if show_secondary and valuta != "DKK":
            k[0].caption(f"≈ {pris*get_fx_rate(valuta,'DKK'):,.2f} DKK")
        mc = info.get("marketCap")
        try:
            k[1].metric("Market cap", f"{float(mc)/1e9:,.1f}B {valuta}" if mc else "-")
        except:
            k[1].metric("Market cap", "-")
        k[2].metric("P/E", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "-")
        k[3].metric("Fwd P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "-")
        k[4].metric("Yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-")
        k[5].metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

        f_score, f_det = fundamental_score(info)
        # Brug 12-måneders data til tekniske signaler
        t_score, t_det = technical_score(df_technical)
        overall = f_score * 0.6 + t_score * 0.4
        f_a, f_c = recommendation(f_score)
        t_a, t_c = recommendation(t_score)
        o_a, o_c = recommendation(overall)

        fair_default = dcf_valuation(info, 0.10, 0.10, 0.025)
        # Brug 6-måneders data til kursmål
        targets = calculate_price_targets(df_targets, pris, fair_default)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;background:{f_c}22;border:2px solid {f_c}'>"
            f"<div style='font-size:0.85rem;opacity:0.8'>🏛️ LANGSIGTET</div>"
            f"<div style='font-size:0.75rem;opacity:0.6;margin-bottom:0.5rem'>📅 12+ måneder · Fundamentale + DCF</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{f_c}'>{f_a}</div>"
            f"<div style='font-size:1.3rem;font-weight:700;margin-top:0.3rem'>{f_score:.0f}/100</div>"
            f"</div>", unsafe_allow_html=True)
        r2.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;background:{t_c}22;border:2px solid {t_c}'>"
            f"<div style='font-size:0.85rem;opacity:0.8'>⚡ KORTSIGTET</div>"
            f"<div style='font-size:0.75rem;opacity:0.6;margin-bottom:0.5rem'>📅 1-3 måneder · 12mdr tekniske signaler</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{t_c}'>{t_a}</div>"
            f"<div style='font-size:1.3rem;font-weight:700;margin-top:0.3rem'>{t_score:.0f}/100</div>"
            f"</div>", unsafe_allow_html=True)
        r3.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;background:{o_c}22;border:2px solid {o_c}'>"
            f"<div style='font-size:0.85rem;opacity:0.8'>🎯 SAMLET</div>"
            f"<div style='font-size:0.75rem;opacity:0.6;margin-bottom:0.5rem'>⚖️ Vægtet 60% lang / 40% kort</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{o_c}'>{o_a}</div>"
            f"<div style='font-size:1.3rem;font-weight:700;margin-top:0.3rem'>{overall:.0f}/100</div>"
            f"</div>", unsafe_allow_html=True)

        # ===== KURSNIVEAUER =====
        st.markdown("### 💰 Anbefalede kursniveauer")
        st.caption(f"Baseret på 6-måneders volatilitet (ATR + Bollinger Bands) + DCF fair value · {valuta}{' + DKK' if show_secondary and valuta != 'DKK' else ''}")

        pt = st.columns(5)
        buy_low_pct = (targets["buy_low"]/pris-1)*100
        buy_high_pct = (targets["buy_high"]/pris-1)*100
        stop_pct = (targets["stop_loss"]/pris-1)*100
        target_short_pct = (targets["target_short"]/pris-1)*100
        target_long_pct = (targets["target_long"]/pris-1)*100

        pt[0].markdown(make_range_box("🟢 KØB ZONE", targets["buy_low"], targets["buy_high"], valuta, "#16a34a", f"{buy_low_pct:+.1f}% til {buy_high_pct:+.1f}%", show_secondary), unsafe_allow_html=True)
        pt[1].markdown(make_price_box("📍 AKTUEL", pris, valuta, "#0099ff", f"{change_pct:+.2f}% i dag", show_secondary), unsafe_allow_html=True)
        pt[2].markdown(make_price_box("🛑 STOP LOSS", targets["stop_loss"], valuta, "#ef4444", f"{stop_pct:+.1f}% (2x ATR)", show_secondary), unsafe_allow_html=True)
        pt[3].markdown(make_price_box("🎯 KORT MÅL (1-3m)", targets["target_short"], valuta, "#eab308", f"{target_short_pct:+.1f}% (BB upper)", show_secondary), unsafe_allow_html=True)
        pt[4].markdown(make_price_box("🚀 LANG MÅL (12m+)", targets["target_long"], valuta, "#22c55e", f"{target_long_pct:+.1f}% {'(DCF)' if fair_default else '(+20%)'}", show_secondary), unsafe_allow_html=True)

        if show_secondary and valuta != "DKK":
            rate = get_fx_rate(valuta, "DKK")
            st.caption(f"📊 52-uger: Low {targets['week52_low']:.2f} {valuta} (≈{targets['week52_low']*rate:.2f} DKK) · High {targets['week52_high']:.2f} {valuta} (≈{targets['week52_high']*rate:.2f} DKK) · Daglig ATR: {targets['atr']:.2f}")
        else:
            st.caption(f"📊 52-uger: Low {targets['week52_low']:.2f} · High {targets['week52_high']:.2f} {valuta} · Daglig ATR: {targets['atr']:.2f}")

        sub_tabs = st.tabs(["📊 Charts", "📋 Fundamentals", "🔧 Teknisk", "💎 DCF", "📉 Risiko", "🎲 Monte Carlo"])

        with sub_tabs[0]:
            df_plot = df_chart_filtered
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name="Pris"), 1, 1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA50"], name="SMA50", line=dict(color="orange")), 1, 1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA200"], name="SMA200", line=dict(color="purple")), 1, 1)
            fig.add_hline(y=targets["buy_high"], line_dash="dot", line_color="#16a34a", annotation_text="Køb", row=1, col=1)
            fig.add_hline(y=targets["stop_loss"], line_dash="dot", line_color="#ef4444", annotation_text="Stop", row=1, col=1)
            fig.add_hline(y=targets["target_long"], line_dash="dot", line_color="#22c55e", annotation_text="Mål", row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"], name="RSI", line=dict(color="#00d4aa")), 2, 1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"], name="MACD", line=dict(color="#0099ff")), 3, 1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD_signal"], name="Signal", line=dict(color="orange")), 3, 1)
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark",
                            title=f"{navn} - Visning: {period}")
            st.plotly_chart(fig, use_container_width=True)

        with sub_tabs[1]:
            st.caption("🏛️ Fundamentale data er TTM (Trailing Twelve Months) - opdateres hvert kvartal fra API")
            df_f = pd.DataFrame(f_det)
            if not df_f.empty:
                fig_f = px.bar(df_f, x="impact", y="label", orientation="h", color="impact", color_continuous_scale="RdYlGn")
                fig_f.update_layout(height=500, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_f, use_container_width=True)
                st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info(f"Ingen fundamentale data fra **{data['source']}**")

        with sub_tabs[2]:
            st.caption("⚡ Tekniske signaler beregnet på sidste 12 måneder")
            df_t = pd.DataFrame(t_det)
            if not df_t.empty:
                fig_t = px.bar(df_t, x="impact", y="label", orientation="h", color="impact", color_continuous_scale="RdYlGn")
                fig_t.update_layout(height=400, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_t, use_container_width=True)
            last = df_technical.iloc[-1]
            cc = st.columns(4)
            cc[0].metric("RSI", f"{last['RSI']:.1f}" if not np.isnan(last['RSI']) else "-")
            cc[1].metric("MACD", f"{last['MACD']:.3f}" if not np.isnan(last['MACD']) else "-")
            cc[2].metric("ADX", f"{last['ADX']:.1f}" if not np.isnan(last['ADX']) else "-")
            cc[3].metric("ATR", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "-")

        with sub_tabs[3]:
            st.caption("💎 DCF baseret på seneste FCF + dine input-antagelser")
            c = st.columns(3)
            cg = c[0].slider("Vækstrate", 0.0, 0.30, 0.10, 0.01)
            cdr = c[1].slider("Diskontering", 0.05, 0.20, 0.10, 0.01)
            ct = c[2].slider("Terminal vækst", 0.01, 0.05, 0.025, 0.005)
            fair = dcf_valuation(info, cg, cdr, ct)
            if fair:
                up = (fair/pris-1)*100
                d = st.columns(3)
                d[0].metric(f"Aktuel pris ({valuta})", f"{pris:.2f}")
                d[1].metric(f"DCF fair value ({valuta})", f"{fair:.2f}")
                d[2].metric("Upside", f"{up:+.1f}%")
                if show_secondary and valuta != "DKK":
                    rate = get_fx_rate(valuta, "DKK")
                    st.caption(f"💱 I DKK: Pris {pris*rate:.2f} → Fair value {fair*rate:.2f}")
            else:
                st.warning("Ikke nok FCF-data til DCF (kræver Yahoo eller Finnhub)")

        with sub_tabs[4]:
            st.caption("📉 Risk metrics baseret på sidste 3 års data (statistisk signifikant)")
            risk = risk_metrics(hist_risk)
            c = st.columns(4)
            c[0].metric("Ann. afkast", f"{risk['ann_r']*100:.2f}%")
            c[1].metric("Ann. volatilitet", f"{risk['ann_v']*100:.2f}%")
            c[2].metric("Sharpe", f"{risk['sharpe']:.2f}")
            c[3].metric("Sortino", f"{risk['sortino']:.2f}")
            c2 = st.columns(2)
            c2[0].metric("Max drawdown", f"{risk['max_dd']*100:.2f}%")
            c2[1].metric("VaR 95%", f"{risk['var95']*100:.2f}%")
            fig_dd = go.Figure(go.Scatter(x=risk['dd_series'].index, y=risk['dd_series']*100, fill="tozeroy", line=dict(color="#ef4444")))
            fig_dd.update_layout(template="plotly_dark", height=350, title="Drawdown % (3 år)")
            st.plotly_chart(fig_dd, use_container_width=True)

        with sub_tabs[5]:
            st.caption("🎲 Monte Carlo: 300 simulationer baseret på sidste 2 års volatilitet")
            sims, lp = monte_carlo(hist_mc)
            final = sims[:, -1]
            p5, p50, p95 = np.percentile(final, [5, 50, 95])
            c = st.columns(4)
            c[0].metric(f"Start ({valuta})", f"{lp:.2f}")
            c[1].metric("5% (worst)", f"{p5:.2f}", f"{(p5/lp-1)*100:+.1f}%")
            c[2].metric("Median (12m)", f"{p50:.2f}", f"{(p50/lp-1)*100:+.1f}%")
            c[3].metric("95% (best)", f"{p95:.2f}", f"{(p95/lp-1)*100:+.1f}%")
            if show_secondary and valuta != "DKK":
                rate = get_fx_rate(valuta, "DKK")
                st.caption(f"💱 Median i DKK: {p50*rate:.2f} (Start: {lp*rate:.2f})")
            fig_m = go.Figure()
            for i in range(min(100, len(sims))):
                fig_m.add_trace(go.Scatter(y=sims[i], line=dict(width=0.5, color="rgba(0,212,170,0.15)"), showlegend=False))
            fig_m.add_trace(go.Scatter(y=np.percentile(sims, 50, axis=0), name="Median", line=dict(color="#00d4aa", width=3)))
            fig_m.update_layout(template="plotly_dark", height=500, title="Monte Carlo - 252 dage frem")
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("👆 Indtast en ticker, eller brug **🔍 Søg ticker** fanen")
