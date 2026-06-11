"""Data fetchers fra Yahoo, Twelve Data, Finnhub - med tiered caching"""
import pandas as pd
import yfinance as yf
import requests as plain_requests
import streamlit as st
from config import MANUAL_TWELVE_MAP, FX_FALLBACK

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


# ============================================================
# 🚀 TIERED CACHE STRATEGI
# ============================================================
# CACHE_PRICE        = 600    (10 min) — pris/quote
# CACHE_INFO         = 21600  (6 t)    — company info, metadata
# CACHE_HIST         = 1800   (30 min) — daglige candles
# CACHE_FX           = 21600  (6 t)    — valutakurser (opdateres dagligt)
# CACHE_SEARCH       = 604800 (7 dage) — ticker symbols (immutable)
# CACHE_FUNDAMENTALS = 86400  (24 t)   — finnhub fundamentals
# CACHE_NEWS         = 1800   (30 min) — news feed
# ============================================================

CACHE_PRICE = 600
CACHE_INFO = 21600
CACHE_HIST = 1800
CACHE_FX = 21600
CACHE_SEARCH = 604800
CACHE_FUNDAMENTALS = 86400
CACHE_NEWS = 1800


def get_api_keys():
    """Hent API keys fra Streamlit secrets"""
    finnhub_key = None
    twelve_key = None
    try:
        finnhub_key = st.secrets.get("FINNHUB_API_KEY")
        twelve_key = st.secrets.get("TWELVE_DATA_KEY")
    except Exception:
        pass
    return finnhub_key, twelve_key


# ============================================================
# FX RATES (LANG CACHE - opdateres dagligt)
# ============================================================

@st.cache_data(ttl=CACHE_FX, show_spinner=False, max_entries=50)
def get_fx_rate(from_curr, to_curr):
    """Cache: 6 timer (FX-kurser opdateres dagligt)."""
    if from_curr == to_curr:
        return 1.0
    try:
        r = plain_requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": from_curr, "to": to_curr}, timeout=10
        )
        rate = r.json().get("rates", {}).get(to_curr)
        if rate:
            return float(rate)
    except Exception:
        pass
    return FX_FALLBACK.get((from_curr, to_curr), 1.0)


def get_twelve_formats(ticker):
    t = ticker.upper().strip()
    if t in MANUAL_TWELVE_MAP:
        return MANUAL_TWELVE_MAP[t]
    suffix_map = {
        ".CO": "XCSE", ".DE": "XETR", ".AS": "XAMS",
        ".SW": "XSWX", ".PA": "XPAR", ".L": "XLON",
    }
    for sfx, exch in suffix_map.items():
        if sfx in t:
            base = t.replace(sfx, "")
            formats = [base, f"{base}:{exch}"]
            if sfx == ".CO":
                formats.insert(1, base.replace("-", ""))
            return formats
    return [t]


# ============================================================
# YAHOO FINANCE (SPLITTET I INFO + HIST + NEWS)
# ============================================================

@st.cache_data(ttl=CACHE_INFO, show_spinner=False)
def _fetch_yahoo_info(ticker):
    """
    Hent kun company info/metadata fra Yahoo.
    Cache: 6 timer (metadata ændres sjældent).

    Returnerer: dict eller None eller "RATE_LIMIT"
    """
    try:
        session = make_session()
        tk = yf.Ticker(ticker, session=session)
        info = tk.info
        if not info or len(info) < 5:
            return None
        return dict(info)  # Konverter til ren dict for caching
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            return "RATE_LIMIT"
        return None


@st.cache_data(ttl=CACHE_HIST, show_spinner=False)
def _fetch_yahoo_hist(ticker, period="max"):
    """
    Hent kun pris-historik fra Yahoo.
    Cache: 30 min (daglige candles, intra-day relativt OK).
    """
    try:
        session = make_session()
        tk = yf.Ticker(ticker, session=session)
        hist = tk.history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            return "RATE_LIMIT"
        return None


@st.cache_data(ttl=CACHE_NEWS, show_spinner=False)
def _fetch_yahoo_news(ticker):
    """Cache: 30 min (news opdateres løbende)."""
    try:
        session = make_session()
        tk = yf.Ticker(ticker, session=session)
        return tk.news if hasattr(tk, "news") else []
    except Exception:
        return []


def fetch_yahoo(ticker, period="max"):
    """
    Wrapper der kombinerer cachet info + hist + news.
    Hver del cachet med forskellig TTL → genbrug på tværs!
    """
    try:
        # Step 1: info (6h cache)
        info = _fetch_yahoo_info(ticker)
        if info == "RATE_LIMIT":
            return "RATE_LIMIT"
        if info is None:
            return None

        # Step 2: hist (30 min cache)
        hist = _fetch_yahoo_hist(ticker, period)
        if hist == "RATE_LIMIT":
            return "RATE_LIMIT"
        if hist is None:
            return None

        # Step 3: news (30 min cache, ikke kritisk)
        news = _fetch_yahoo_news(ticker)

        return {"info": info, "hist": hist, "news": news, "source": "Yahoo Finance"}
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            return "RATE_LIMIT"
        return None


# ============================================================
# TWELVE DATA (SPLITTET I QUOTE + TIME_SERIES)
# ============================================================

@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def _fetch_twelve_quote(symbol, twelve_key):
    """Cache: 10 min."""
    if not twelve_key:
        return None
    try:
        r = plain_requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": symbol, "apikey": twelve_key}, timeout=15
        )
        quote = r.json()
        if quote.get("status") == "error" or "code" in quote:
            return None
        return quote
    except Exception:
        return None


@st.cache_data(ttl=CACHE_HIST, show_spinner=False)
def _fetch_twelve_timeseries(symbol, twelve_key):
    """Cache: 30 min (time series kan tage tid - værd at cache)."""
    if not twelve_key:
        return None
    try:
        r = plain_requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol, "interval": "1day",
                "outputsize": 5000, "apikey": twelve_key
            },
            timeout=20,
        )
        ts_data = r.json()
        if ts_data.get("status") == "error" or "values" not in ts_data:
            return None

        hist = pd.DataFrame(ts_data["values"])
        hist["datetime"] = pd.to_datetime(hist["datetime"])
        hist = hist.set_index("datetime").sort_index()
        hist = hist.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in hist.columns:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")
            else:
                hist[col] = 0
        return hist
    except Exception:
        return None


def fetch_twelve_single(symbol, twelve_key):
    """Wrapper der kombinerer quote + timeseries (begge cachet)."""
    if not twelve_key:
        return None
    try:
        quote = _fetch_twelve_quote(symbol, twelve_key)
        if quote is None:
            return None

        hist = _fetch_twelve_timeseries(symbol, twelve_key)
        if hist is None:
            return None

        info = {
            "longName": quote.get("name") or symbol,
            "sector": "?", "industry": "?", "country": "?",
            "currency": quote.get("currency", "USD"),
            "currentPrice": float(quote.get("close", 0)) if quote.get("close")
                            else float(hist["Close"].iloc[-1]),
            "previousClose": float(quote.get("previous_close", 0)) if quote.get("previous_close") else None,
            "twelve_symbol_used": symbol,
        }
        return {
            "info": info, "hist": hist, "news": [],
            "source": f"Twelve Data ({symbol})"
        }
    except Exception:
        return None


def fetch_twelve(ticker, twelve_key):
    if not twelve_key:
        return None
    for symbol in get_twelve_formats(ticker):
        result = fetch_twelve_single(symbol, twelve_key)
        if result:
            return result
    return None


# ============================================================
# FINNHUB (SPLITTET I PROFILE + QUOTE + FUNDAMENTALS)
# ============================================================

@st.cache_data(ttl=CACHE_INFO, show_spinner=False)
def _fetch_finnhub_profile(ticker, finnhub_key):
    """Cache: 6 timer (company profile ændres sjældent)."""
    if not FINNHUB_AVAILABLE or not finnhub_key:
        return None
    try:
        client = finnhub.Client(api_key=finnhub_key)
        profile = client.company_profile2(symbol=ticker)
        if not profile or not profile.get("name"):
            return None
        return profile
    except Exception:
        return None


@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def _fetch_finnhub_quote(ticker, finnhub_key):
    """Cache: 10 min."""
    if not FINNHUB_AVAILABLE or not finnhub_key:
        return None
    try:
        client = finnhub.Client(api_key=finnhub_key)
        quote = client.quote(ticker)
        if not quote.get("c") or quote.get("c", 0) < 0.01:
            return None
        return quote
    except Exception:
        return None


@st.cache_data(ttl=CACHE_FUNDAMENTALS, show_spinner=False)
def _fetch_finnhub_fundamentals(ticker, finnhub_key):
    """Cache: 24 timer (fundamentals opdateres kvartalsvis)."""
    if not FINNHUB_AVAILABLE or not finnhub_key:
        return {}
    try:
        client = finnhub.Client(api_key=finnhub_key)
        res = client.company_basic_financials(ticker, "all")
        return res.get("metric", {}) if res else {}
    except Exception:
        return {}


def fetch_finnhub(ticker, finnhub_key, twelve_key):
    """Wrapper der kombinerer profile + quote + fundamentals + twelve hist."""
    if not FINNHUB_AVAILABLE or not finnhub_key or "." in ticker:
        return None
    try:
        profile = _fetch_finnhub_profile(ticker, finnhub_key)
        if profile is None:
            return None

        quote = _fetch_finnhub_quote(ticker, finnhub_key)
        if quote is None:
            return None

        metrics = _fetch_finnhub_fundamentals(ticker, finnhub_key)

        def _pct(key):
            v = metrics.get(key)
            return (v / 100) if v else None

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
            "returnOnEquity": _pct("roeRfy"),
            "returnOnAssets": _pct("roaRfy"),
            "profitMargins": _pct("netProfitMarginAnnual"),
            "operatingMargins": _pct("operatingMarginAnnual"),
            "debtToEquity": metrics.get("totalDebt/totalEquityAnnual"),
            "currentRatio": metrics.get("currentRatioAnnual"),
            "quickRatio": metrics.get("quickRatioAnnual"),
            "revenueGrowth": _pct("revenueGrowthTTMYoy"),
            "earningsGrowth": _pct("epsGrowthTTMYoy"),
            "freeCashflow": metrics.get("freeCashFlowAnnual"),
            "totalDebt": metrics.get("totalDebt"),
            "totalCash": metrics.get("cashAndCashEquivalentsQuarterly"),
            "dividendYield": _pct("dividendYieldIndicatedAnnual"),
            "beta": metrics.get("beta"),
        }
        if twelve_key:
            tw = fetch_twelve(ticker, twelve_key)
            if tw:
                return {
                    "info": info, "hist": tw["hist"], "news": [],
                    "source": "Finnhub + Twelve Data"
                }
        return None
    except Exception:
        return None


# ============================================================
# HOVED-FETCH (USES CACHED COMPONENTS)
# ============================================================

@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def fetch_data(ticker):
    """
    Hovedfunktion - prioriterer Yahoo Finance.

    Cache: 10 min (kort cache her, men under-funktioner har egne caches!).
    Selv om denne TTL udløber, vil:
    - Yahoo info (cachet 6t) → genbruges
    - Yahoo hist (cachet 30 min) → genbruges
    - Finnhub fundamentals (cachet 24t) → genbruges
    """
    finnhub_key, twelve_key = get_api_keys()

    # Step 1: Yahoo (cached components)
    result = fetch_yahoo(ticker, "max")
    if result and result != "RATE_LIMIT":
        return result
    rate_limited = result == "RATE_LIMIT"

    # Step 2: Finnhub + Twelve Data (cached components)
    if finnhub_key and "." not in ticker:
        result = fetch_finnhub(ticker, finnhub_key, twelve_key)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Finnhub + Twelve Data"
            return result

    # Step 3: Twelve Data alene
    if twelve_key:
        result = fetch_twelve(ticker, twelve_key)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Twelve Data (kun pris-data)"
            else:
                result["warning"] = (
                    f"Begrænset data - kun pris-historik "
                    f"(symbol: {result['info'].get('twelve_symbol_used','?')})"
                )
            return result
    return None


# ============================================================
# TICKER SEARCH (LANG CACHE - symbols er immutable)
# ============================================================

@st.cache_data(ttl=CACHE_SEARCH, show_spinner=False)
def search_tickers(query):
    """Cache: 7 dage (ticker symbols ændres aldrig)."""
    if not query or len(query) < 2:
        return []
    finnhub_key, twelve_key = get_api_keys()
    results = []
    if twelve_key:
        try:
            r = plain_requests.get(
                "https://api.twelvedata.com/symbol_search",
                params={"symbol": query, "outputsize": 20}, timeout=10
            )
            for item in r.json().get("data", []):
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
    if FINNHUB_AVAILABLE and finnhub_key and len(results) < 10:
        try:
            client = finnhub.Client(api_key=finnhub_key)
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


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_stock_cache():
    """Ryd al aktie-relateret cache. Bruges fra UI hvis bruger vil tvinge refresh."""
    get_fx_rate.clear()
    _fetch_yahoo_info.clear()
    _fetch_yahoo_hist.clear()
    _fetch_yahoo_news.clear()
    _fetch_twelve_quote.clear()
    _fetch_twelve_timeseries.clear()
    _fetch_finnhub_profile.clear()
    _fetch_finnhub_quote.clear()
    _fetch_finnhub_fundamentals.clear()
    fetch_data.clear()
    search_tickers.clear()


def get_cache_info():
    """Returnér info om cache TTL'er (til Dev Mode)."""
    return {
        "Yahoo info (metadata)": f"{CACHE_INFO}s ({CACHE_INFO//3600} t)",
        "Yahoo hist (price)": f"{CACHE_HIST}s ({CACHE_HIST//60} min)",
        "Yahoo news": f"{CACHE_NEWS}s ({CACHE_NEWS//60} min)",
        "Twelve quote": f"{CACHE_PRICE}s ({CACHE_PRICE//60} min)",
        "Twelve timeseries": f"{CACHE_HIST}s ({CACHE_HIST//60} min)",
        "Finnhub profile": f"{CACHE_INFO}s ({CACHE_INFO//3600} t)",
        "Finnhub quote": f"{CACHE_PRICE}s ({CACHE_PRICE//60} min)",
        "Finnhub fundamentals": f"{CACHE_FUNDAMENTALS}s ({CACHE_FUNDAMENTALS//3600} t)",
        "FX rates": f"{CACHE_FX}s ({CACHE_FX//3600} t)",
        "Ticker search": f"{CACHE_SEARCH}s ({CACHE_SEARCH//86400} dage)",
        "fetch_data wrapper": f"{CACHE_PRICE}s ({CACHE_PRICE//60} min)",
    }
