"""Konstanter og konfiguration"""

ANALYSIS_PERIODS = {
    "technical": 365,
    "targets": 180,
    "risk": 365 * 3,
    "monte_carlo": 730,
    "week52": 252,
}

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

# Markeds-universer til screener
SCREENER_UNIVERSES = {
    "🇩🇰 Danske C25": [
        "NOVO-B.CO", "MAERSK-B.CO", "MAERSK-A.CO", "DSV.CO", "ORSTED.CO",
        "CARL-B.CO", "GMAB.CO", "NDA-DK.CO", "TRYG.CO", "COLO-B.CO",
        "DEMANT.CO", "PNDORA.CO", "ROCK-B.CO", "VWS.CO", "AMBU-B.CO",
        "BAVA.CO", "FLS.CO", "GN.CO", "ISS.CO", "JYSK.CO",
        "NETC.CO", "NZYM-B.CO", "SIM.CO", "TOP.CO", "DANSKE.CO",
    ],
    "🇺🇸 US Tech Giants": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "CSCO",
        "QCOM", "TXN", "INTU", "NOW", "PLTR", "SNOW",
    ],
    "🇺🇸 US Large Cap": [
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "BAC",
        "XOM", "CVX", "ABBV", "PFE", "KO", "PEP", "MRK", "TMO",
        "COST", "DIS", "MCD", "NKE",
    ],
    "🇪🇺 EU Large Cap": [
        "ASML.AS", "SAP.DE", "NESN.SW", "NOVN.SW", "ROG.SW", "MC.PA",
        "OR.PA", "SAN.PA", "AIR.PA", "SIE.DE", "ALV.DE", "DTE.DE",
        "BAS.DE", "BAYN.DE", "VOW3.DE", "BMW.DE", "ADYEN.AS", "PRX.AS",
    ],
    "🚀 Growth Stocks": [
        "NVDA", "TSLA", "PLTR", "SHOP", "SNOW", "CRWD", "DDOG",
        "NET", "MDB", "TEAM", "ZS", "ROKU", "SQ", "COIN",
    ],
    "💰 Dividend Aristocrats": [
        "JNJ", "PG", "KO", "PEP", "MMM", "CVX", "XOM", "MCD",
        "WMT", "T", "VZ", "IBM", "MO", "PM",
    ],
}

FX_FALLBACK = {
    ("USD", "DKK"): 6.85, ("DKK", "USD"): 1/6.85,
    ("EUR", "DKK"): 7.46, ("DKK", "EUR"): 1/7.46,
    ("EUR", "USD"): 1.08, ("USD", "EUR"): 1/1.08,
    ("GBP", "DKK"): 8.70, ("DKK", "GBP"): 1/8.70,
    ("CHF", "DKK"): 7.75, ("DKK", "CHF"): 1/7.75,
    ("SEK", "DKK"): 0.65, ("DKK", "SEK"): 1/0.65,
}
