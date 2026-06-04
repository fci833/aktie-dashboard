"""Krypto-konfiguration og mappings"""

# Top 50 coins med CoinGecko ID + Binance symbol + Yahoo ticker
CRYPTO_UNIVERSE = {
    # Layer 1 - Store
    "BTC": {"cg": "bitcoin", "binance": "BTCUSDT", "yahoo": "BTC-USD", "category": "Layer 1"},
    "ETH": {"cg": "ethereum", "binance": "ETHUSDT", "yahoo": "ETH-USD", "category": "Layer 1"},
    "BNB": {"cg": "binancecoin", "binance": "BNBUSDT", "yahoo": "BNB-USD", "category": "Exchange"},
    "SOL": {"cg": "solana", "binance": "SOLUSDT", "yahoo": "SOL-USD", "category": "Layer 1"},
    "XRP": {"cg": "ripple", "binance": "XRPUSDT", "yahoo": "XRP-USD", "category": "Payment"},
    "ADA": {"cg": "cardano", "binance": "ADAUSDT", "yahoo": "ADA-USD", "category": "Layer 1"},
    "AVAX": {"cg": "avalanche-2", "binance": "AVAXUSDT", "yahoo": "AVAX-USD", "category": "Layer 1"},
    "DOT": {"cg": "polkadot", "binance": "DOTUSDT", "yahoo": "DOT-USD", "category": "Layer 0"},
    "LINK": {"cg": "chainlink", "binance": "LINKUSDT", "yahoo": "LINK-USD", "category": "Oracle"},
    "MATIC": {"cg": "matic-network", "binance": "MATICUSDT", "yahoo": "MATIC-USD", "category": "Layer 2"},
    "TRX": {"cg": "tron", "binance": "TRXUSDT", "yahoo": "TRX-USD", "category": "Layer 1"},
    "ATOM": {"cg": "cosmos", "binance": "ATOMUSDT", "yahoo": "ATOM-USD", "category": "Layer 0"},
    "NEAR": {"cg": "near", "binance": "NEARUSDT", "yahoo": "NEAR-USD", "category": "Layer 1"},
    "APT": {"cg": "aptos", "binance": "APTUSDT", "yahoo": "APT-USD", "category": "Layer 1"},
    "ARB": {"cg": "arbitrum", "binance": "ARBUSDT", "yahoo": "ARB-USD", "category": "Layer 2"},
    "OP": {"cg": "optimism", "binance": "OPUSDT", "yahoo": "OP-USD", "category": "Layer 2"},
    "SUI": {"cg": "sui", "binance": "SUIUSDT", "yahoo": "SUI-USD", "category": "Layer 1"},

    # DeFi
    "UNI": {"cg": "uniswap", "binance": "UNIUSDT", "yahoo": "UNI-USD", "category": "DeFi"},
    "AAVE": {"cg": "aave", "binance": "AAVEUSDT", "yahoo": "AAVE-USD", "category": "DeFi"},
    "MKR": {"cg": "maker", "binance": "MKRUSDT", "yahoo": "MKR-USD", "category": "DeFi"},
    "LDO": {"cg": "lido-dao", "binance": "LDOUSDT", "yahoo": "LDO-USD", "category": "DeFi"},
    "CRV": {"cg": "curve-dao-token", "binance": "CRVUSDT", "yahoo": "CRV-USD", "category": "DeFi"},
    "INJ": {"cg": "injective-protocol", "binance": "INJUSDT", "yahoo": "INJ-USD", "category": "DeFi"},

    # Memes (high-risk)
    "DOGE": {"cg": "dogecoin", "binance": "DOGEUSDT", "yahoo": "DOGE-USD", "category": "Meme"},
    "SHIB": {"cg": "shiba-inu", "binance": "SHIBUSDT", "yahoo": "SHIB-USD", "category": "Meme"},
    "PEPE": {"cg": "pepe", "binance": "PEPEUSDT", "yahoo": "PEPE-USD", "category": "Meme"},

    # Storage/Infrastructure
    "FIL": {"cg": "filecoin", "binance": "FILUSDT", "yahoo": "FIL-USD", "category": "Storage"},
    "RNDR": {"cg": "render-token", "binance": "RNDRUSDT", "yahoo": "RNDR-USD", "category": "AI/Compute"},

    # Klassiske
    "LTC": {"cg": "litecoin", "binance": "LTCUSDT", "yahoo": "LTC-USD", "category": "Payment"},
    "BCH": {"cg": "bitcoin-cash", "binance": "BCHUSDT", "yahoo": "BCH-USD", "category": "Payment"},
    "ETC": {"cg": "ethereum-classic", "binance": "ETCUSDT", "yahoo": "ETC-USD", "category": "Layer 1"},
    "XLM": {"cg": "stellar", "binance": "XLMUSDT", "yahoo": "XLM-USD", "category": "Payment"},
}

# Universer til screener
CRYPTO_UNIVERSES = {
    "🪙 Crypto Top 10": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC"],
    "🚀 Layer 1": ["BTC", "ETH", "SOL", "ADA", "AVAX", "NEAR", "APT", "SUI", "ATOM", "TRX"],
    "⚡ Layer 2": ["MATIC", "ARB", "OP"],
    "💎 DeFi": ["UNI", "AAVE", "MKR", "LDO", "CRV", "INJ"],
    "🎲 Memes (high risk)": ["DOGE", "SHIB", "PEPE"],
    "🏦 Store coins (>$10B)": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOGE", "TRX", "LINK"],
    "🌍 All crypto": list(CRYPTO_UNIVERSE.keys()),
}

# Krypto-specifikke konstanter
BTC_HALVING_DATES = ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19", "2028-04-15"]

CRYPTO_RECOMMENDATIONS = {
    "STRONG_BUY": {"min_score": 75, "label": "🚀 STÆRKT KØB", "color": "#16a34a"},
    "BUY": {"min_score": 60, "label": "✅ KØB", "color": "#22c55e"},
    "HOLD": {"min_score": 40, "label": "⏸️ HOLD", "color": "#eab308"},
    "SELL": {"min_score": 25, "label": "⚠️ SÆLG", "color": "#ef4444"},
    "STRONG_SELL": {"min_score": 0, "label": "🛑 STÆRKT SÆLG", "color": "#b91c1c"},
}
