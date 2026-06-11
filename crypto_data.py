"""Krypto-datakilder - Yahoo Finance + CoinGecko (med forbedret tiered caching)"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from crypto_config import CRYPTO_UNIVERSE


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AktieDashboard/1.0)",
    "Accept": "application/json",
}

# ============================================================
# 🚀 CACHE STRATEGI (TIERED)
# ============================================================
# CACHE_PRICE  = 600    (10 min) — pris-data, daglige candles
# CACHE_META   = 86400  (24 t)   — metadata (navn, supply, ATH, social)
# CACHE_SEARCH = 604800 (7 dage) — coin_id mapping (immutable)
# CACHE_GLOBAL = 600    (10 min) — global market data
# CACHE_FG     = 3600   (1 time) — fear & greed (opdateres dagligt)
# CACHE_TREND  = 600    (10 min) — trending/movers
# CACHE_CHAIN  = 3600   (1 time) — on-chain BTC metrics
# ============================================================

CACHE_PRICE = 600
CACHE_META = 86400
CACHE_SEARCH = 604800
CACHE_GLOBAL = 600
CACHE_FG = 3600
CACHE_TREND = 600
CACHE_CHAIN = 3600


def is_crypto(ticker):
    if not ticker:
        return False
    t = ticker.upper().replace("-USD", "").replace("USDT", "")
    return (
        t in CRYPTO_UNIVERSE
        or ticker.upper().endswith("-USD")
        or ticker.upper().endswith("USDT")
    )


def normalize_crypto_ticker(ticker):
    t = ticker.upper().strip()
    for suffix in ["-USD", "USDT", "USD"]:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


# ============================================================
# YAHOO FINANCE FOR KRYPTO (PRIMÆR)
# ============================================================

@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def fetch_yahoo_crypto(symbol, period="2y"):
    """
    Hent krypto fra Yahoo Finance med BTC-USD format.
    Cache: 10 min (daglige candles opdateres sjældent intra-day).
    """
    try:
        import yfinance as yf
        yahoo_symbol = f"{symbol.upper()}-USD"

        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period=period, auto_adjust=False)

        if hist is None or hist.empty:
            print(f"[fetch_yahoo_crypto] {yahoo_symbol}: tom historik")
            return None

        # Normaliser index
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        hist = hist[["Open", "High", "Low", "Close", "Volume"]]
        hist = hist.dropna(subset=["Close"])

        if hist.empty:
            return None

        # Prøv at hente info (kan være langsomt, men cachet sammen)
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
        change_24h = (last / prev - 1) * 100 if prev else 0

        change_7d = None
        change_30d = None
        change_1y = None
        if len(hist) >= 7:
            change_7d = (last / float(hist["Close"].iloc[-7]) - 1) * 100
        if len(hist) >= 30:
            change_30d = (last / float(hist["Close"].iloc[-30]) - 1) * 100
        if len(hist) >= 365:
            change_1y = (last / float(hist["Close"].iloc[-365]) - 1) * 100

        result_info = {
            "longName": info.get("name") or info.get("longName") or symbol,
            "symbol": symbol.upper(),
            "currency": "USD",
            "currentPrice": last,
            "previousClose": prev,
            "marketCap": info.get("marketCap"),
            "totalVolume": info.get("volume24Hr"),
            "circulating_supply": info.get("circulatingSupply"),
            "max_supply": info.get("maxSupply"),
            "change_24h": change_24h,
            "change_7d": change_7d,
            "change_30d": change_30d,
            "change_1y": change_1y,
            "sector": "Cryptocurrency",
            "country": "Global",
            "description": (info.get("description") or "")[:500],
        }

        return {"info": result_info, "hist": hist, "source": "Yahoo Finance"}

    except Exception as e:
        print(f"[fetch_yahoo_crypto] {symbol} EXCEPTION: {e}")
        return None


# ============================================================
# COINGECKO SEARCH (LANG CACHE - coin_ids er immutable)
# ============================================================

@st.cache_data(ttl=CACHE_SEARCH, show_spinner=False)
def search_coingecko_id(symbol):
    """
    Slå symbol op på CoinGecko → coin_id.
    Cache: 7 dage (coin IDs ændres aldrig).
    """
    try:
        symbol_clean = symbol.upper().strip()
        r = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": symbol_clean},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        coins = data.get("coins", [])
        if not coins:
            return None

        exact_matches = [c for c in coins if c.get("symbol", "").upper() == symbol_clean]
        if exact_matches:
            exact_matches.sort(key=lambda c: c.get("market_cap_rank") or 999999)
            return exact_matches[0].get("id")
        return coins[0].get("id")
    except Exception as e:
        print(f"[search_coingecko_id] {symbol}: {e}")
        return None


# ============================================================
# COINGECKO - SPLITTET I METADATA + CHART (forskellig TTL)
# ============================================================

@st.cache_data(ttl=CACHE_META, show_spinner=False)
def fetch_coingecko_metadata(coin_id):
    """
    Hent kun metadata (navn, supply, ATH, social, developer).
    Cache: 24 timer (metadata ændres sjældent).

    Returnerer dict eller "RATE_LIMIT" / None.
    """
    try:
        r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
                "sparkline": "false",
            },
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code == 429:
            return "RATE_LIMIT"
        if r.status_code != 200:
            return None

        data = r.json()
        md = data.get("market_data") or {}
        cd = data.get("community_data") or {}
        dd = data.get("developer_data") or {}

        def safe_get(d, *keys):
            for k in keys:
                if not isinstance(d, dict):
                    return None
                d = d.get(k)
                if d is None:
                    return None
            return d

        return {
            "longName": data.get("name"),
            "symbol": (data.get("symbol") or "").upper(),
            "marketCap": safe_get(md, "market_cap", "usd"),
            "marketCapRank": md.get("market_cap_rank"),
            "totalVolume": safe_get(md, "total_volume", "usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath": safe_get(md, "ath", "usd"),
            "ath_change_%": safe_get(md, "ath_change_percentage", "usd"),
            "ath_date": safe_get(md, "ath_date", "usd"),
            "atl": safe_get(md, "atl", "usd"),
            "atl_change_%": safe_get(md, "atl_change_percentage", "usd"),
            "currentPrice": safe_get(md, "current_price", "usd"),
            "change_1h": safe_get(md, "price_change_percentage_1h_in_currency", "usd"),
            "change_24h": md.get("price_change_percentage_24h"),
            "change_7d": md.get("price_change_percentage_7d"),
            "change_30d": md.get("price_change_percentage_30d"),
            "change_1y": md.get("price_change_percentage_1y"),
            "sentiment_votes_up_%": data.get("sentiment_votes_up_percentage"),
            "community_score": data.get("community_score"),
            "public_interest_score": data.get("public_interest_score"),
            "twitter_followers": cd.get("twitter_followers"),
            "reddit_subscribers": cd.get("reddit_subscribers"),
            "developer_score": data.get("developer_score"),
            "github_stars": dd.get("stars"),
            "github_forks": dd.get("forks"),
            "github_subscribers": dd.get("subscribers"),
            "commit_count_4_weeks": dd.get("commit_count_4_weeks"),
            "github_pull_requests_merged": dd.get("pull_requests_merged"),
            "description": ((data.get("description") or {}).get("en") or "")[:500],
        }

    except Exception as e:
        print(f"[fetch_coingecko_metadata] {coin_id} EXCEPTION: {e}")
        return None


@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def fetch_coingecko_chart(coin_id, days=365):
    """
    Hent OHLCV chart-data fra CoinGecko.
    Cache: 10 min (price data).

    Returnerer DataFrame eller "RATE_LIMIT" / None.
    """
    try:
        # Daglig pris-historik
        chart_r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
            headers=HEADERS,
            timeout=15,
        )
        if chart_r.status_code == 429:
            return "RATE_LIMIT"
        if chart_r.status_code != 200:
            return None

        chart_data = chart_r.json()
        prices = chart_data.get("prices", [])
        volumes = chart_data.get("total_volumes", [])

        if not prices:
            return None

        df_prices = pd.DataFrame(prices, columns=["ts", "Close"])
        df_prices["ts"] = pd.to_datetime(df_prices["ts"], unit="ms").dt.normalize()
        df_prices = df_prices.drop_duplicates(subset="ts").set_index("ts")

        # OHLC (best effort - ikke alle coins har det)
        try:
            ohlc_r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": str(days)},
                headers=HEADERS,
                timeout=15,
            )
            ohlc = ohlc_r.json() if ohlc_r.status_code == 200 else []
        except Exception:
            ohlc = []

        if ohlc and isinstance(ohlc, list) and len(ohlc) > 0:
            df = pd.DataFrame(ohlc, columns=["ts", "Open", "High", "Low", "Close"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
            df = df.drop_duplicates(subset="ts").set_index("ts")
        else:
            df = df_prices.copy()
            df["Open"] = df["Close"].shift(1).fillna(df["Close"])
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df = df[["Open", "High", "Low", "Close"]]

        if volumes:
            vol_df = pd.DataFrame(volumes, columns=["ts", "Volume"])
            vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms").dt.normalize()
            vol_df = vol_df.drop_duplicates(subset="ts").set_index("ts")
            df = df.join(vol_df, how="left")
            df["Volume"] = df["Volume"].fillna(0)
        else:
            df["Volume"] = 0

        df = df.dropna(subset=["Close"])
        if df.empty:
            return None

        return df

    except Exception as e:
        print(f"[fetch_coingecko_chart] {coin_id} EXCEPTION: {e}")
        return None


def fetch_coingecko(coin_id):
    """
    Wrapper der kombinerer metadata + chart.
    Hver del er cached separat med forskellig TTL.
    """
    metadata = fetch_coingecko_metadata(coin_id)
    if metadata == "RATE_LIMIT":
        return "RATE_LIMIT"
    if metadata is None:
        return None

    df = fetch_coingecko_chart(coin_id)
    if df == "RATE_LIMIT":
        return "RATE_LIMIT"
    if df is None or df.empty:
        return None

    info = dict(metadata)
    info["currency"] = "USD"
    info["sector"] = "Cryptocurrency"
    info["country"] = "Global"
    if not info.get("currentPrice"):
        info["currentPrice"] = float(df["Close"].iloc[-1])
    info["previousClose"] = (
        float(df["Close"].iloc[-2]) if len(df) >= 2 else info["currentPrice"]
    )

    return {"info": info, "hist": df, "source": "CoinGecko"}


# ============================================================
# CRYPTOCOMPARE (BACKUP)
# ============================================================

@st.cache_data(ttl=CACHE_PRICE, show_spinner=False)
def fetch_cryptocompare(symbol, days=365):
    """Backup fra CryptoCompare. Cache: 10 min."""
    try:
        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/histoday",
            params={"fsym": symbol.upper(), "tsym": "USD", "limit": days},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("Response") != "Success":
            return None

        rows = data.get("Data", {}).get("Data", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
        df = df.set_index("ts")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volumefrom": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        df = df[df["Close"] > 0]

        if df.empty:
            return None

        return df
    except Exception as e:
        print(f"[fetch_cryptocompare] {symbol}: {e}")
        return None


# ============================================================
# FEAR & GREED (LANG CACHE - opdateres dagligt)
# ============================================================

@st.cache_data(ttl=CACHE_FG, show_spinner=False)
def fetch_fear_greed():
    """Cache: 1 time (F&G opdateres dagligt)."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "data" not in data:
            return None
        df = pd.DataFrame(data["data"])
        df["value"] = df["value"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        return df.sort_values("timestamp")
    except Exception:
        return None


# ============================================================
# GLOBAL MARKET
# ============================================================

@st.cache_data(ttl=CACHE_GLOBAL, show_spinner=False)
def fetch_global_crypto_market():
    """Cache: 10 min."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/global",
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return None
        d = r.json().get("data") or {}
        return {
            "total_market_cap_usd": (d.get("total_market_cap") or {}).get("usd") or 0,
            "total_volume_usd": (d.get("total_volume") or {}).get("usd") or 0,
            "btc_dominance": (d.get("market_cap_percentage") or {}).get("btc") or 0,
            "eth_dominance": (d.get("market_cap_percentage") or {}).get("eth") or 0,
            "active_cryptos": d.get("active_cryptocurrencies") or 0,
            "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd") or 0,
        }
    except Exception:
        return None


# ============================================================
# HJÆLPER: Berig Yahoo med CoinGecko-metadata (NU CACHET!)
# ============================================================

def _enrich_with_coingecko(yahoo_data, coin_id):
    """
    Tilføj CoinGecko metadata til Yahoo data.
    Bruger nu cachet `fetch_coingecko_metadata` → spar API-kald!
    """
    metadata = fetch_coingecko_metadata(coin_id)
    if metadata in (None, "RATE_LIMIT"):
        return yahoo_data

    info = yahoo_data["info"]
    # Bevarede Yahoo-felter, men beriget med CoinGecko-detaljer
    enrichment_keys = [
        "longName", "marketCap", "marketCapRank", "totalVolume",
        "ath", "ath_change_%", "ath_date", "atl", "atl_change_%",
        "sentiment_votes_up_%", "community_score", "public_interest_score",
        "twitter_followers", "reddit_subscribers",
        "developer_score", "github_stars", "github_forks",
        "github_subscribers", "commit_count_4_weeks",
        "github_pull_requests_merged", "description",
    ]
    for key in enrichment_keys:
        val = metadata.get(key)
        if val is not None and (info.get(key) is None or info.get(key) == ""):
            info[key] = val
        elif val is not None and key in ("marketCap", "marketCapRank", "ath"):
            # Disse foretrækkes fra CoinGecko (mere pålidelige for krypto)
            info[key] = val

    yahoo_data["source"] = "Yahoo + CoinGecko"
    return yahoo_data


def _build_response_from_df(df, symbol, category=None):
    """Byg standard response fra et DataFrame"""
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
    change_24h = (last / prev - 1) * 100 if prev else 0

    change_7d = None
    change_30d = None
    change_1y = None
    if len(df) >= 7:
        change_7d = (last / float(df["Close"].iloc[-7]) - 1) * 100
    if len(df) >= 30:
        change_30d = (last / float(df["Close"].iloc[-30]) - 1) * 100
    if len(df) >= 365:
        change_1y = (last / float(df["Close"].iloc[-365]) - 1) * 100

    info = {
        "longName": symbol,
        "symbol": symbol,
        "currency": "USD",
        "currentPrice": last,
        "previousClose": prev,
        "change_24h": change_24h,
        "change_7d": change_7d,
        "change_30d": change_30d,
        "change_1y": change_1y,
        "sector": "Cryptocurrency",
        "country": "Global",
        "category": category or "Custom",
    }
    return {"info": info, "hist": df, "source": "Backup"}


# ============================================================
# HOVED-FETCH (NEGATIV-CACHE FOR FEJL)
# ============================================================

# Negativ cache: undgå retry-storm hvis ticker fejler
@st.cache_data(ttl=300, show_spinner=False)
def _is_known_bad_ticker(symbol):
    """Returnerer True hvis tickeren netop har fejlet (5 min cache)."""
    return False  # Sættes via session_state ved fejl


def fetch_crypto_data(ticker):
    """
    Hovedfunktion - PRIORITERER Yahoo Finance (virker globalt).

    Strategi:
    1. Yahoo Finance (BTC-USD format) → primær, hurtig, ingen rate limit
    2. Berig med CoinGecko metadata (cachet 24t!)
    3. Hvis Yahoo fejler → CoinGecko fuld data
    4. Sidste udvej → CryptoCompare

    Cache hits er ekstremt hurtige (< 1ms) takket være tiered TTL.
    """
    if not ticker:
        return None

    symbol = normalize_crypto_ticker(ticker)
    log = []
    log.append(f"🎯 fetch_crypto_data({ticker}) → '{symbol}'")

    # ----- 1. KENDT TICKER -----
    if symbol in CRYPTO_UNIVERSE:
        config = CRYPTO_UNIVERSE[symbol]
        log.append(f"✅ {symbol} fundet i CRYPTO_UNIVERSE")

        # Step 1a: Yahoo Finance først (cache: 10 min)
        log.append(f"🔄 Yahoo Finance: {symbol}-USD")
        ydata = fetch_yahoo_crypto(symbol)
        if ydata is not None:
            log.append(f"✅ Yahoo OK: {len(ydata['hist'])} dage")
            # Berig med CoinGecko metadata (cachet 24t)
            ydata = _enrich_with_coingecko(ydata, config["cg"])
            ydata["debug_log"] = log
            return ydata
        else:
            log.append(f"❌ Yahoo fejlede")

        # Step 1b: CoinGecko fuld
        for attempt in range(3):
            log.append(f"🔄 CoinGecko forsøg {attempt+1}/3: {config['cg']}")
            data = fetch_coingecko(config["cg"])
            if data and data != "RATE_LIMIT":
                log.append("✅ CoinGecko OK")
                data["debug_log"] = log
                return data
            if data == "RATE_LIMIT":
                wait_time = (attempt + 1) * 2
                log.append(f"⚠️ Rate limited, venter {wait_time}s...")
                time.sleep(wait_time)
            else:
                log.append("❌ CoinGecko fejlede")
                break

        # Step 1c: CryptoCompare backup
        log.append(f"🔄 CryptoCompare: {symbol}")
        df = fetch_cryptocompare(symbol)
        if df is not None and not df.empty:
            log.append(f"✅ CryptoCompare OK: {len(df)} dage")
            response = _build_response_from_df(df, symbol, config.get("category"))
            response = _enrich_with_coingecko(response, config["cg"])
            response["debug_log"] = log
            return response

        print(f"\n=== FETCH FAILED FOR {symbol} (kendt ticker) ===")
        for line in log:
            print(f"  {line}")
        return None

    # ----- 2. UKENDT TICKER -----
    log.append(f"🔍 Ukendt ticker, søger CoinGecko...")

    # Prøv Yahoo direkte først (kan virke for store coins)
    log.append(f"🔄 Yahoo Finance: {symbol}-USD")
    ydata = fetch_yahoo_crypto(symbol)
    if ydata is not None:
        log.append(f"✅ Yahoo OK: {len(ydata['hist'])} dage")
        # Find coin_id og berig (begge cachet)
        coin_id = search_coingecko_id(symbol)
        if coin_id:
            log.append(f"✅ CoinGecko ID: {coin_id}")
            ydata = _enrich_with_coingecko(ydata, coin_id)
        ydata["debug_log"] = log
        return ydata

    # Yahoo fejlede → CoinGecko search + fuld data
    coin_id = search_coingecko_id(symbol)
    if coin_id:
        log.append(f"✅ Fandt CoinGecko ID: {coin_id}")
        for attempt in range(3):
            log.append(f"🔄 CoinGecko forsøg {attempt+1}/3")
            data = fetch_coingecko(coin_id)
            if data and data != "RATE_LIMIT":
                log.append("✅ CoinGecko OK")
                data["debug_log"] = log
                return data
            if data == "RATE_LIMIT":
                time.sleep((attempt + 1) * 2)
            else:
                break
    else:
        log.append(f"❌ Ingen CoinGecko match")

    # Sidste udvej: CryptoCompare
    log.append(f"🔄 Sidste udvej: CryptoCompare")
    df = fetch_cryptocompare(symbol)
    if df is not None and not df.empty:
        log.append(f"✅ CryptoCompare OK")
        response = _build_response_from_df(df, symbol)
        response["debug_log"] = log
        return response

    print(f"\n=== FETCH FAILED FOR {symbol} ===")
    for line in log:
        print(f"  {line}")
    print("=" * 50)
    return None


# ============================================================
# ON-CHAIN BTC
# ============================================================

@st.cache_data(ttl=CACHE_CHAIN, show_spinner=False)
def fetch_btc_onchain():
    """Cache: 1 time (on-chain data opdateres ca. hver 10 min)."""
    try:
        metrics = {}
        endpoints = {
            "hash_rate": "hash-rate",
            "difficulty": "difficulty",
            "transactions": "n-transactions",
            "active_addresses": "n-unique-addresses",
            "mempool_size": "mempool-size",
            "miners_revenue": "miners-revenue",
        }
        for key, endpoint in endpoints.items():
            try:
                r = requests.get(
                    f"https://api.blockchain.info/charts/{endpoint}",
                    params={"timespan": "30days", "format": "json"},
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                if "values" in data and data["values"]:
                    metrics[key] = data["values"][-1]["y"]
                    metrics[f"{key}_history"] = data["values"]
            except Exception:
                pass
        return metrics
    except Exception:
        return {}


# ============================================================
# TRENDING / TOP MOVERS
# ============================================================

@st.cache_data(ttl=CACHE_TREND, show_spinner=False)
def fetch_trending_coins():
    """Cache: 10 min."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            headers=HEADERS,
            timeout=10
        )
        if r.status_code != 200:
            return []
        data = r.json()
        return [
            {
                "name": c["item"].get("name"),
                "symbol": (c["item"].get("symbol") or "").upper(),
                "rank": c["item"].get("market_cap_rank"),
                "price_btc": c["item"].get("price_btc"),
                "thumb": c["item"].get("thumb"),
            }
            for c in data.get("coins", [])[:7]
        ]
    except Exception:
        return []


@st.cache_data(ttl=CACHE_TREND, show_spinner=False)
def fetch_top_movers():
    """Cache: 10 min."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": 1,
                "price_change_percentage": "24h",
            },
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code != 200:
            return None, None
        data = r.json()
        df = pd.DataFrame(data)
        if df.empty:
            return None, None
        df = df[["symbol", "name", "current_price",
                 "price_change_percentage_24h", "market_cap"]].dropna()
        df["symbol"] = df["symbol"].str.upper()
        gainers = df.nlargest(10, "price_change_percentage_24h")
        losers = df.nsmallest(10, "price_change_percentage_24h")
        return gainers, losers
    except Exception:
        return None, None


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def clear_crypto_cache():
    """
    Ryd al krypto-relateret cache.
    Bruges fra UI hvis bruger vil tvinge refresh.
    """
    fetch_yahoo_crypto.clear()
    search_coingecko_id.clear()
    fetch_coingecko_metadata.clear()
    fetch_coingecko_chart.clear()
    fetch_cryptocompare.clear()
    fetch_fear_greed.clear()
    fetch_global_crypto_market.clear()
    fetch_btc_onchain.clear()
    fetch_trending_coins.clear()
    fetch_top_movers.clear()


def get_cache_info():
    """
    Returnér info om cache TTL'er (til Dev Mode).
    """
    return {
        "Yahoo crypto": f"{CACHE_PRICE}s ({CACHE_PRICE//60} min)",
        "CoinGecko metadata": f"{CACHE_META}s ({CACHE_META//3600} t)",
        "CoinGecko chart": f"{CACHE_PRICE}s ({CACHE_PRICE//60} min)",
        "Coin ID search": f"{CACHE_SEARCH}s ({CACHE_SEARCH//86400} dage)",
        "Global market": f"{CACHE_GLOBAL}s ({CACHE_GLOBAL//60} min)",
        "Fear & Greed": f"{CACHE_FG}s ({CACHE_FG//60} min)",
        "Trending/Movers": f"{CACHE_TREND}s ({CACHE_TREND//60} min)",
        "BTC on-chain": f"{CACHE_CHAIN}s ({CACHE_CHAIN//60} min)",
    }


# ============================================================
# DEPRECATED
# ============================================================

def fetch_binance(symbol, interval="1d", limit=500):
    """Deprecated: Binance er geo-blokeret på Streamlit Cloud."""
    print(f"[fetch_binance] DEPRECATED: Binance er geo-blokeret på Streamlit Cloud")
    return None
