"""Krypto-datakilder med fallback-kæde"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from crypto_config import CRYPTO_UNIVERSE


def is_crypto(ticker):
    """Tjekker om ticker er krypto"""
    t = ticker.upper().replace("-USD", "").replace("USDT", "")
    return t in CRYPTO_UNIVERSE or ticker.endswith("-USD") or ticker.endswith("USDT")


def normalize_crypto_ticker(ticker):
    """Konverterer 'BTC-USD', 'BTCUSDT', 'BTC' → 'BTC'"""
    t = ticker.upper()
    for suffix in ["-USD", "USDT", "USD"]:
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


# ===== COINGECKO =====

@st.cache_data(ttl=300, show_spinner=False)
def fetch_coingecko(coin_id):
    """Hent komplet krypto-data fra CoinGecko"""
    try:
        # 1. Coin info
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
            return None
        data = r.json()

        # 2. Historisk OHLC (365 dage daily)
        ohlc_r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": "365"},
            timeout=15,
        )
        ohlc = ohlc_r.json() if ohlc_r.status_code == 200 else []

        # 3. Volumen-data
        vol_r = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": "365", "interval": "daily"},
            timeout=15,
        )
        vol_data = vol_r.json() if vol_r.status_code == 200 else {"total_volumes": []}

        # Byg DataFrame
        if ohlc:
            df = pd.DataFrame(ohlc, columns=["ts", "Open", "High", "Low", "Close"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
            df = df.drop_duplicates(subset="ts").set_index("ts")

            # Tilføj volume
            vol_df = pd.DataFrame(vol_data["total_volumes"], columns=["ts", "Volume"])
            vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms").dt.normalize()
            vol_df = vol_df.drop_duplicates(subset="ts").set_index("ts")
            df = df.join(vol_df, how="left")
            df["Volume"] = df["Volume"].fillna(0)
        else:
            return None

        md = data.get("market_data", {})
        cd = data.get("community_data", {})
        dd = data.get("developer_data", {})

        info = {
            "longName": data.get("name"),
            "symbol": data.get("symbol", "").upper(),
            "currency": "USD",
            "currentPrice": md.get("current_price", {}).get("usd"),
            "marketCap": md.get("market_cap", {}).get("usd"),
            "marketCapRank": md.get("market_cap_rank"),
            "totalVolume": md.get("total_volume", {}).get("usd"),
            "sector": "Cryptocurrency",
            "country": "Global",
            # Krypto-specifik
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath": md.get("ath", {}).get("usd"),
            "ath_change_%": md.get("ath_change_percentage", {}).get("usd"),
            "ath_date": md.get("ath_date", {}).get("usd"),
            "atl": md.get("atl", {}).get("usd"),
            "atl_change_%": md.get("atl_change_percentage", {}).get("usd"),
            # Performance
            "change_1h": md.get("price_change_percentage_1h_in_currency", {}).get("usd"),
            "change_24h": md.get("price_change_percentage_24h"),
            "change_7d": md.get("price_change_percentage_7d"),
            "change_30d": md.get("price_change_percentage_30d"),
            "change_1y": md.get("price_change_percentage_1y"),
            # Community
            "twitter_followers": cd.get("twitter_followers"),
            "reddit_subscribers": cd.get("reddit_subscribers"),
            # Developer
            "github_stars": dd.get("stars"),
            "github_commits_4w": dd.get("commit_count_4_weeks"),
            "github_pull_requests_merged": dd.get("pull_requests_merged"),
            # Description (kort)
            "description": (data.get("description", {}).get("en", "") or "")[:300],
            "previousClose": df["Close"].iloc[-2] if len(df) >= 2 else md.get("current_price", {}).get("usd"),
        }

        return {"info": info, "hist": df.dropna(subset=["Close"]), "source": "CoinGecko"}

    except Exception as e:
        print(f"CoinGecko error for {coin_id}: {e}")
        return None


# ===== BINANCE =====

@st.cache_data(ttl=60, show_spinner=False)
def fetch_binance(symbol, interval="1d", limit=500):
    """Hent OHLCV fra Binance (real-time, gratis)"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
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
    """Hent Crypto Fear & Greed Index (alternative.me)"""
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


# ===== HOVED-FETCH MED FALLBACK =====

def fetch_crypto_data(ticker):
    """Hovedfunktion: Prøver CoinGecko → Binance → returner None"""
    symbol = normalize_crypto_ticker(ticker)

    if symbol not in CRYPTO_UNIVERSE:
        # Ukendt krypto - prøv Yahoo via eksisterende fetch
        return None

    config = CRYPTO_UNIVERSE[symbol]

    # 1. Prøv CoinGecko (bedst)
    data = fetch_coingecko(config["cg"])
    if data and data != "RATE_LIMIT":
        return data

    # 2. Fallback til Binance
    df = fetch_binance(config["binance"], limit=365)
    if df is not None and not df.empty:
        last = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df) >= 2 else last
        info = {
            "longName": symbol,
            "symbol": symbol,
            "currency": "USD",
            "currentPrice": last,
            "previousClose": prev,
            "sector": "Cryptocurrency",
            "country": "Global",
            "category": config["category"],
        }
        return {"info": info, "hist": df, "source": "Binance"}

    return None
