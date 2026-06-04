"""Krypto-datakilder med fallback-kæde + custom ticker support"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from crypto_config import CRYPTO_UNIVERSE


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
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        coins = data.get("coins", [])
        if not coins:
            return None

        symbol_upper = symbol.upper()

        # Først: prøv exact symbol match, sortér efter market cap rank
        exact_matches = [c for c in coins if c.get("symbol", "").upper() == symbol_upper]
        if exact_matches:
            exact_matches.sort(
                key=lambda c: c.get("market_cap_rank") or 999999
            )
            return exact_matches[0].get("id")

        # Ellers: brug første resultat
        return coins[0].get("id")

    except Exception as e:
        print(f"CoinGecko search error for {symbol}: {e}")
        return None


# ===== COINGECKO HOVEDFETCH =====

@st.cache_data(ttl=300, show_spinner=False)
def fetch_coingecko(coin_id):
    """Hent komplet krypto-data fra CoinGecko med robust fallback"""
    try:
        # === 1. HOVED-DATA (info, market_data, community, developer) ===
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
            timeout=15,
        )
        if r.status_code == 429:
            return "RATE_LIMIT"
        if r.status_code != 200:
            print(f"CoinGecko coin endpoint failed for {coin_id}: {r.status_code}")
            return None
        data = r.json()

        # === 2. MARKET CHART (prices + volumes) - mest pålidelig ===
        chart_r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "365", "interval": "daily"},
            timeout=15,
        )
        if chart_r.status_code == 429:
            return "RATE_LIMIT"
        if chart_r.status_code != 200:
            print(f"CoinGecko market_chart failed for {coin_id}: {chart_r.status_code}")
            return None

        chart_data = chart_r.json()
        prices = chart_data.get("prices", [])
        volumes = chart_data.get("total_volumes", [])

        if not prices:
            print(f"CoinGecko: ingen prices for {coin_id}")
            return None

        # Byg DataFrame fra prices (close)
        df_prices = pd.DataFrame(prices, columns=["ts", "Close"])
        df_prices["ts"] = pd.to_datetime(df_prices["ts"], unit="ms").dt.normalize()
        df_prices = df_prices.drop_duplicates(subset="ts").set_index("ts")

        # === 3. OHLC - prøv hvis muligt, ellers fake fra close ===
        try:
            ohlc_r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": "365"},
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
            # Fallback: byg OHLC fra close prices (rolling high/low)
            print(f"CoinGecko: bruger close-prices fallback for {coin_id}")
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
            return None

        md = data.get("market_data", {})
        cd = data.get("community_data", {})
        dd = data.get("developer_data", {})

        info = {
            "longName": data.get("name"),
            "symbol": data.get("symbol", "").upper(),
            "currency": "USD",
            "currentPrice": md.get("current_price", {}).get("usd") or float(df["Close"].iloc[-1]),
            "marketCap": md.get("market_cap", {}).get("usd"),
            "marketCapRank": md.get("market_cap_rank"),
            "totalVolume": md.get("total_volume", {}).get("usd"),
            "sector": "Cryptocurrency",
            "country": "Global",
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath": md.get("ath", {}).get("usd"),
            "ath_change_%": md.get("ath_change_percentage", {}).get("usd"),
            "ath_date": md.get("ath_date", {}).get("usd"),
            "atl": md.get("atl", {}).get("usd"),
            "atl_change_%": md.get("atl_change_percentage", {}).get("usd"),
            "change_1h": md.get("price_change_percentage_1h_in_currency", {}).get("usd"),
            "change_24h": md.get("price_change_percentage_24h"),
            "change_7d": md.get("price_change_percentage_7d"),
            "change_30d": md.get("price_change_percentage_30d"),
            "change_1y": md.get("price_change_percentage_1y"),
            # Sentiment
            "sentiment_votes_up_%": data.get("sentiment_votes_up_percentage"),
            "community_score": data.get("community_score"),
            "public_interest_score": data.get("public_interest_score"),
            # Community
            "twitter_followers": cd.get("twitter_followers"),
            "reddit_subscribers": cd.get("reddit_subscribers"),
            # Developer
            "developer_score": data.get("developer_score"),
            "github_stars": dd.get("stars"),
            "github_forks": dd.get("forks"),
            "github_subscribers": dd.get("subscribers"),
            "commit_count_4_weeks": dd.get("commit_count_4_weeks"),
            "github_pull_requests_merged": dd.get("pull_requests_merged"),
            "description": (data.get("description", {}).get("en", "") or "")[:500],
            "previousClose": float(df["Close"].iloc[-2]) if len(df) >= 2 else md.get("current_price", {}).get("usd"),
        }

        return {"info": info, "hist": df, "source": "CoinGecko"}

    except Exception as e:
        print(f"CoinGecko error for {coin_id}: {e}")
        return None


# ===== BINANCE =====

@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance(symbol, interval="1d", limit=500):
    """Hent OHLCV fra Binance"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data, columns=[
            "ts", "Open", "High", "Low", "Close", "Volume",
            "ct", "qv", "n", "tb", "tq", "i"
        ])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df.set_index("ts")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df
    except Exception:
        return None


# ===== FEAR & GREED INDEX =====

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed():
    """Hent Crypto Fear & Greed Index"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30", timeout=10).json()
        if "data" in r:
            df = pd.DataFrame(r["data"])
            df["value"] = df["value"].astype(int)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            return df.sort_values("timestamp")
        return None
    except Exception:
        return None


# ===== GLOBAL MARKET DATA =====

@st.cache_data(ttl=600, show_spinner=False)
def fetch_global_crypto_market():
    """Total krypto-marked statistik"""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10).json()
        d = r.get("data", {})
        return {
            "total_market_cap_usd": d.get("total_market_cap", {}).get("usd"),
            "total_volume_usd": d.get("total_volume", {}).get("usd"),
            "btc_dominance": d.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": d.get("market_cap_percentage", {}).get("eth"),
            "active_cryptos": d.get("active_cryptocurrencies"),
            "market_cap_change_24h": d.get("market_cap_change_percentage_24h_usd"),
        }
    except Exception:
        return None


# ===== HOVED-FETCH MED FALLBACK + CUSTOM TICKER SUPPORT =====

def fetch_crypto_data(ticker):
    """
    Hovedfunktion til at hente krypto-data.
    Prøver i rækkefølge:
    1. CoinGecko via CRYPTO_UNIVERSE config (hvis kendt ticker)
    2. CoinGecko via search API (custom tickers som DOGE, SHIB, PEPE, PAXG)
    3. Binance fallback (USDT-suffix)
    """
    if not ticker:
        return None

    symbol = normalize_crypto_ticker(ticker)

    # ----- 1. Kendt ticker fra CRYPTO_UNIVERSE -----
    if symbol in CRYPTO_UNIVERSE:
        config = CRYPTO_UNIVERSE[symbol]

        # Prøv CoinGecko først
        data = fetch_coingecko(config["cg"])
        if data and data != "RATE_LIMIT":
            return data

        # Fallback: Binance
        df = fetch_binance(config["binance"], limit=365)
        if df is not None and not df.empty:
            return _build_binance_response(df, symbol, config.get("category"))

        # Hvis CoinGecko er rate-limited, prøv search som sidste udvej
        if data == "RATE_LIMIT":
            time.sleep(2)
            coin_id = search_coingecko_id(symbol)
            if coin_id:
                data = fetch_coingecko(coin_id)
                if data and data != "RATE_LIMIT":
                    return data

        return None

    # ----- 2. Ukendt ticker → CoinGecko search -----
    coin_id = search_coingecko_id(symbol)
    if coin_id:
        data = fetch_coingecko(coin_id)
        if data and data != "RATE_LIMIT":
            return data
        # Hvis rate-limited, vent og prøv igen
        if data == "RATE_LIMIT":
            time.sleep(3)
            data = fetch_coingecko(coin_id)
            if data and data != "RATE_LIMIT":
                return data

    # ----- 3. Sidste forsøg: Binance med USDT-suffix -----
    binance_symbol = f"{symbol}USDT"
    df = fetch_binance(binance_symbol, limit=365)
    if df is not None and not df.empty:
        return _build_binance_response(df, symbol, "Custom")

    # ----- 4. Prøv Binance med BUSD-suffix (legacy) -----
    binance_busd = f"{symbol}BUSD"
    df = fetch_binance(binance_busd, limit=365)
    if df is not None and not df.empty:
        return _build_binance_response(df, symbol, "Custom")

    return None


def _build_binance_response(df, symbol, category=None):
    """Hjælper: bygger info-dict fra Binance-data"""
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) >= 2 else last
    change_24h = (last / prev - 1) * 100 if prev else 0

    # Beregn ændringer
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
                ).json()
                if "values" in r and r["values"]:
                    metrics[key] = r["values"][-1]["y"]
                    metrics[f"{key}_history"] = r["values"]
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
            timeout=10
        ).json()
        return [
            {
                "name": c["item"]["name"],
                "symbol": c["item"]["symbol"].upper(),
                "rank": c["item"].get("market_cap_rank"),
                "price_btc": c["item"].get("price_btc"),
                "thumb": c["item"].get("thumb"),
            }
            for c in r.get("coins", [])[:7]
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
            timeout=15,
        ).json()
        df = pd.DataFrame(r)
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
