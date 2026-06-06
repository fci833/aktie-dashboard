"""Krypto-datakilder med intelligent fallback + custom ticker support"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from crypto_config import CRYPTO_UNIVERSE


# Standard headers for at undgå CoinGecko-blokering
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AktieDashboard/1.0)",
    "Accept": "application/json",
}


def is_crypto(ticker):
    """Tjekker om ticker er krypto"""
    if not ticker:
        return False
    t = ticker.upper().replace("-USD", "").replace("USDT", "")
    return (
        t in CRYPTO_UNIVERSE
        or ticker.upper().endswith("-USD")
        or ticker.upper().endswith("USDT")
    )


def normalize_crypto_ticker(ticker):
    """Konverterer 'BTC-USD', 'BTCUSDT', 'BTC' → 'BTC'"""
    t = ticker.upper().strip()
    for suffix in ["-USD", "USDT", "USD"]:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


# ===== COINGECKO SEARCH (til custom tickers) =====

@st.cache_data(ttl=3600, show_spinner=False)
def search_coingecko_id(symbol):
    """
    Slå et symbol op på CoinGecko og få coin_id.
    Fx 'DOGE' → 'dogecoin', 'PAXG' → 'pax-gold'
    """
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": symbol},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[search_coingecko_id] {symbol}: status {r.status_code}")
            return None
        data = r.json()
        coins = data.get("coins", [])
        if not coins:
            print(f"[search_coingecko_id] {symbol}: ingen resultater")
            return None

        symbol_upper = symbol.upper()

        # Først: prøv exact symbol match, sortér efter market cap rank
        exact_matches = [c for c in coins if c.get("symbol", "").upper() == symbol_upper]
        if exact_matches:
            exact_matches.sort(key=lambda c: c.get("market_cap_rank") or 999999)
            return exact_matches[0].get("id")

        # Ellers: brug første resultat
        return coins[0].get("id")

    except Exception as e:
        print(f"[search_coingecko_id] {symbol} fejl: {e}")
        return None


# ===== COINGECKO HOVEDFETCH =====

@st.cache_data(ttl=300, show_spinner=False)
def fetch_coingecko(coin_id):
    """Hent komplet krypto-data fra CoinGecko med robust fallback"""
    try:
        # === 1. HOVED-DATA ===
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
            print(f"[fetch_coingecko] {coin_id}: RATE LIMITED (429)")
            return "RATE_LIMIT"
        if r.status_code != 200:
            print(f"[fetch_coingecko] {coin_id}: coin endpoint status {r.status_code}")
            return None
        data = r.json()

        # === 2. MARKET CHART (prices + volumes) ===
        chart_r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "365", "interval": "daily"},
            headers=HEADERS,
            timeout=15,
        )
        if chart_r.status_code == 429:
            print(f"[fetch_coingecko] {coin_id}: market_chart RATE LIMITED")
            return "RATE_LIMIT"
        if chart_r.status_code != 200:
            print(f"[fetch_coingecko] {coin_id}: market_chart status {chart_r.status_code}")
            return None

        chart_data = chart_r.json()
        prices = chart_data.get("prices", [])
        volumes = chart_data.get("total_volumes", [])

        if not prices:
            print(f"[fetch_coingecko] {coin_id}: ingen prices")
            return None

        df_prices = pd.DataFrame(prices, columns=["ts", "Close"])
        df_prices["ts"] = pd.to_datetime(df_prices["ts"], unit="ms").dt.normalize()
        df_prices = df_prices.drop_duplicates(subset="ts").set_index("ts")

        # === 3. OHLC (best effort) ===
        try:
            ohlc_r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": "365"},
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
            print(f"[fetch_coingecko] {coin_id}: bruger close-prices fallback")
            df = df_prices.copy()
            df["Open"] = df["Close"].shift(1).fillna(df["Close"])
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df = df[["Open", "High", "Low", "Close"]]

        # Tilføj volume
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
            print(f"[fetch_coingecko] {coin_id}: tom dataframe")
            return None

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

        info = {
            "longName": data.get("name"),
            "symbol": (data.get("symbol") or "").upper(),
            "currency": "USD",
            "currentPrice": safe_get(md, "current_price", "usd") or float(df["Close"].iloc[-1]),
            "marketCap": safe_get(md, "market_cap", "usd"),
            "marketCapRank": md.get("market_cap_rank"),
            "totalVolume": safe_get(md, "total_volume", "usd"),
            "sector": "Cryptocurrency",
            "country": "Global",
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath": safe_get(md, "ath", "usd"),
            "ath_change_%": safe_get(md, "ath_change_percentage", "usd"),
            "ath_date": safe_get(md, "ath_date", "usd"),
            "atl": safe_get(md, "atl", "usd"),
            "atl_change_%": safe_get(md, "atl_change_percentage", "usd"),
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
            "previousClose": float(df["Close"].iloc[-2]) if len(df) >= 2 else safe_get(md, "current_price", "usd"),
        }

        return {"info": info, "hist": df, "source": "CoinGecko"}

    except Exception as e:
        print(f"[fetch_coingecko] {coin_id} EXCEPTION: {e}")
        return None


# ===== BINANCE (primær for kendte coins - ingen rate limit) =====

@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance(symbol, interval="1d", limit=500):
    """Hent OHLCV fra Binance - hurtigt og uden rate limit"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[fetch_binance] {symbol}: status {r.status_code} - {r.text[:200]}")
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            print(f"[fetch_binance] {symbol}: tomt eller ugyldigt svar")
            return None
        df = pd.DataFrame(data, columns=[
            "ts", "Open", "High", "Low", "Close", "Volume",
            "ct", "qv", "n", "tb", "tq", "i"
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df
    except Exception as e:
        print(f"[fetch_binance] {symbol} EXCEPTION: {e}")
        return None


# ===== FEAR & GREED INDEX =====

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed():
    """Hent Crypto Fear & Greed Index"""
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
    except Exception as e:
        print(f"[fetch_fear_greed] error: {e}")
        return None


# ===== GLOBAL MARKET DATA =====

@st.cache_data(ttl=600, show_spinner=False)
def fetch_global_crypto_market():
    """Total krypto-marked statistik - robust mod manglende felter"""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/global",
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[fetch_global_crypto_market] status {r.status_code}")
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
    except Exception as e:
        print(f"[fetch_global_crypto_market] error: {e}")
        return None


# ===== HJÆLPER: Berig Binance-data med CoinGecko-metadata =====

def _enrich_binance_with_coingecko(binance_data, coin_id):
    """
    Tager Binance OHLC og prøver at berige med CoinGecko metadata.
    Hvis CoinGecko fejler, returneres binance_data uændret.
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
            timeout=10,
        )
        if r.status_code != 200:
            return binance_data

        data = r.json()
        md = data.get("market_data") or {}
        cd = data.get("community_data") or {}
        dd = data.get("developer_data") or {}

        info = binance_data["info"]
        info["longName"] = data.get("name") or info["longName"]
        info["marketCap"] = (md.get("market_cap") or {}).get("usd")
        info["marketCapRank"] = md.get("market_cap_rank")
        info["totalVolume"] = (md.get("total_volume") or {}).get("usd")
        info["ath"] = (md.get("ath") or {}).get("usd")
        info["ath_change_%"] = (md.get("ath_change_percentage") or {}).get("usd")
        info["change_24h"] = md.get("price_change_percentage_24h") or info.get("change_24h")
        info["change_7d"] = md.get("price_change_percentage_7d") or info.get("change_7d")
        info["change_30d"] = md.get("price_change_percentage_30d") or info.get("change_30d")
        info["change_1y"] = md.get("price_change_percentage_1y")
        info["sentiment_votes_up_%"] = data.get("sentiment_votes_up_percentage")
        info["community_score"] = data.get("community_score")
        info["public_interest_score"] = data.get("public_interest_score")
        info["twitter_followers"] = cd.get("twitter_followers")
        info["reddit_subscribers"] = cd.get("reddit_subscribers")
        info["developer_score"] = data.get("developer_score")
        info["github_stars"] = dd.get("stars")
        info["github_forks"] = dd.get("forks")
        info["commit_count_4_weeks"] = dd.get("commit_count_4_weeks")
        info["description"] = ((data.get("description") or {}).get("en") or "")[:500]

        binance_data["source"] = "Binance + CoinGecko"
        return binance_data

    except Exception as e:
        print(f"[_enrich_binance_with_coingecko] {coin_id}: {e}")
        return binance_data


# ===== HOVED-FETCH MED INTELLIGENT FALLBACK =====

def fetch_crypto_data(ticker):
    """
    Hovedfunktion - PRIORITERER Binance for hastighed, beriger med CoinGecko.

    Strategi:
    1. Kendt ticker → Binance + berig med CoinGecko metadata
    2. Hvis Binance fejler → CoinGecko fuld data (med retries)
    3. Ukendt ticker → CoinGecko search → Binance/CoinGecko
    4. Sidste udvej → Binance med USDT/BUSD/USD-suffix
    """
    if not ticker:
        return None

    symbol = normalize_crypto_ticker(ticker)
    log = []
    log.append(f"🎯 fetch_crypto_data({ticker}) → normaliseret til '{symbol}'")

    # ----- 1. KENDT TICKER -----
    if symbol in CRYPTO_UNIVERSE:
        config = CRYPTO_UNIVERSE[symbol]
        log.append(f"✅ {symbol} fundet i CRYPTO_UNIVERSE: binance={config['binance']}, cg={config['cg']}")

        # Step 1a: Binance først
        log.append(f"🔄 Forsøger Binance: {config['binance']}")
        df = fetch_binance(config["binance"], limit=365)
        if df is not None and not df.empty:
            log.append(f"✅ Binance OK: {len(df)} dage")
            response = _build_binance_response(df, symbol, config.get("category"))
            response = _enrich_binance_with_coingecko(response, config["cg"])
            response["debug_log"] = log
            return response
        else:
            log.append(f"❌ Binance fejlede for {config['binance']}")

        # Step 1b: CoinGecko fallback
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
                log.append("❌ CoinGecko fejlede (ingen data)")
                break

        # Print log til server
        print(f"\n=== FETCH FAILED FOR {symbol} (kendt ticker) ===")
        for line in log:
            print(f"  {line}")
        print("="*50)
        return None

    # ----- 2. UKENDT TICKER: CoinGecko search -----
    log.append(f"🔍 {symbol} ikke i CRYPTO_UNIVERSE, søger CoinGecko...")
    coin_id = search_coingecko_id(symbol)
    if coin_id:
        log.append(f"✅ Fandt CoinGecko ID: {coin_id}")

        # Prøv først Binance
        log.append(f"🔄 Forsøger Binance: {symbol}USDT")
        df = fetch_binance(f"{symbol}USDT", limit=365)
        if df is not None and not df.empty:
            log.append(f"✅ Binance OK: {len(df)} dage")
            response = _build_binance_response(df, symbol, "Custom")
            response = _enrich_binance_with_coingecko(response, coin_id)
            response["debug_log"] = log
            return response
        else:
            log.append(f"❌ {symbol}USDT ikke på Binance")

        # Ellers: CoinGecko fuld data
        for attempt in range(3):
            log.append(f"🔄 CoinGecko forsøg {attempt+1}/3: {coin_id}")
            data = fetch_coingecko(coin_id)
            if data and data != "RATE_LIMIT":
                log.append("✅ CoinGecko OK")
                data["debug_log"] = log
                return data
            if data == "RATE_LIMIT":
                log.append(f"⚠️ Rate limited, venter...")
                time.sleep((attempt + 1) * 2)
            else:
                log.append("❌ CoinGecko fejlede")
                break
    else:
        log.append(f"❌ Ingen CoinGecko match for '{symbol}'")

    # ----- 3. SIDSTE UDVEJ: Binance med suffix-varianter -----
    log.append("🔄 Sidste forsøg: Binance med USDT/BUSD/USD suffix")
    for suffix in ["USDT", "BUSD", "USD"]:
        binance_symbol = f"{symbol}{suffix}"
        log.append(f"  → Forsøger {binance_symbol}")
        df = fetch_binance(binance_symbol, limit=365)
        if df is not None and not df.empty:
            log.append(f"✅ Binance OK med {binance_symbol}")
            response = _build_binance_response(df, symbol, "Custom")
            response["debug_log"] = log
            return response

    # Print log til server
    print(f"\n=== FETCH FAILED FOR {symbol} ===")
    for line in log:
        print(f"  {line}")
    print("="*50)

    return None


def _build_binance_response(df, symbol, category=None):
    """Hjælper: bygger info-dict fra Binance-data"""
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
    change_24h = (last / prev - 1) * 100 if prev else 0

    change_7d = None
    change_30d = None
    if len(df) >= 7:
        change_7d = (last / float(df["Close"].iloc[-7]) - 1) * 100
    if len(df) >= 30:
        change_30d = (last / float(df["Close"].iloc[-30]) - 1) * 100

    info = {
        "longName": symbol,
        "symbol": symbol,
        "currency": "USD",
        "currentPrice": last,
        "previousClose": prev,
        "change_24h": change_24h,
        "change_7d": change_7d,
        "change_30d": change_30d,
        "sector": "Cryptocurrency",
        "country": "Global",
        "category": category or "Custom",
    }
    return {"info": info, "hist": df, "source": "Binance"}


# ===== ON-CHAIN DATA (BTC) =====

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_btc_onchain():
    """BTC on-chain metrics fra blockchain.com"""
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


# ===== TRENDING COINS =====

@st.cache_data(ttl=600, show_spinner=False)
def fetch_trending_coins():
    """Top 7 trending coins på CoinGecko"""
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


# ===== TOP GAINERS / LOSERS =====

@st.cache_data(ttl=600, show_spinner=False)
def fetch_top_movers():
    """Top 10 gainers + losers (24h)"""
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
