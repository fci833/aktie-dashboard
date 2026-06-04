"""Data fetchers fra Yahoo, Twelve Data, Finnhub"""
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


@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def get_fx_rate(from_curr, to_curr):
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
        except Exception:
            pass
        return {"info": info, "hist": hist, "news": news, "source": "Yahoo Finance"}
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "429" in msg or "too many" in msg:
            return "RATE_LIMIT"
        return None


def fetch_twelve_single(symbol, twelve_key):
    if not twelve_key:
        return None
    try:
        base = "https://api.twelvedata.com"
        q = plain_requests.get(
            f"{base}/quote",
            params={"symbol": symbol, "apikey": twelve_key}, timeout=15
        )
        quote = q.json()
        if quote.get("status") == "error" or "code" in quote:
            return None
        ts = plain_requests.get(
            f"{base}/time_series",
            params={"symbol": symbol, "interval": "1day",
                    "outputsize": 5000, "apikey": twelve_key},
            timeout=20,
        )
        ts_data = ts.json()
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
        info = {
            "longName": quote.get("name") or symbol,
            "sector": "?", "industry": "?", "country": "?",
            "currency": quote.get("currency", "USD"),
            "currentPrice": float(quote.get("close", 0)) if quote.get("close")
                            else float(hist["Close"].iloc[-1]),
            "previousClose": float(quote.get("previous_close", 0)) if quote.get("previous_close") else None,
            "twelve_symbol_used": symbol,
        }
        return {"info": info, "hist": hist, "news": [],
                "source": f"Twelve Data ({symbol})"}
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


def fetch_finnhub(ticker, finnhub_key, twelve_key):
    if not FINNHUB_AVAILABLE or not finnhub_key or "." in ticker:
        return None
    try:
        client = finnhub.Client(api_key=finnhub_key)
        profile = client.company_profile2(symbol=ticker)
        if not profile or not profile.get("name"):
            return None
        quote = client.quote(ticker)
        if not quote.get("c") or quote.get("c", 0) < 0.01:
            return None
        metrics = {}
        try:
            res = client.company_basic_financials(ticker, "all")
            metrics = res.get("metric", {}) if res else {}
        except Exception:
            pass

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
                return {"info": info, "hist": tw["hist"], "news": [],
                        "source": "Finnhub + Twelve Data"}
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker):
    finnhub_key, twelve_key = get_api_keys()
    result = fetch_yahoo(ticker, "max")
    if result and result != "RATE_LIMIT":
        return result
    rate_limited = result == "RATE_LIMIT"

    if finnhub_key and "." not in ticker:
        result = fetch_finnhub(ticker, finnhub_key, twelve_key)
        if result:
            if rate_limited:
                result["warning"] = "Yahoo rate limited - bruger Finnhub + Twelve Data"
            return result

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


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query):
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
