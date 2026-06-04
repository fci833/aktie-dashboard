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

# ============ MARKEDS-UNIVERSER ============

# === Norden ===
DK_C25 = [
    "NOVO-B.CO", "MAERSK-B.CO", "MAERSK-A.CO", "DSV.CO", "ORSTED.CO",
    "CARL-B.CO", "GMAB.CO", "NDA-DK.CO", "TRYG.CO", "COLO-B.CO",
    "DEMANT.CO", "PNDORA.CO", "ROCK-B.CO", "VWS.CO", "AMBU-B.CO",
    "BAVA.CO", "FLS.CO", "GN.CO", "ISS.CO", "JYSK.CO",
    "NETC.CO", "NZYM-B.CO", "SIM.CO", "TOP.CO", "DANSKE.CO",
]

NORDIC_LARGE = [
    "VOLV-B.ST", "ATCO-A.ST", "ERIC-B.ST", "HM-B.ST", "INVE-B.ST",
    "SAND.ST", "SEB-A.ST", "SHB-A.ST", "SWED-A.ST", "TELIA.ST",
    "EQNR.OL", "DNB.OL", "MOWI.OL", "TEL.OL", "YAR.OL",
    "NESTE.HE", "NOKIA.HE", "FORTUM.HE", "KNEBV.HE", "SAMPO.HE",
]

# === USA ===
SP500_TOP100 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "LLY", "V", "UNH", "XOM", "JPM", "JNJ", "WMT", "MA", "PG", "HD",
    "AVGO", "CVX", "MRK", "ABBV", "KO", "PEP", "COST", "ADBE", "BAC",
    "CRM", "MCD", "TMO", "CSCO", "ACN", "ABT", "LIN", "NFLX", "AMD",
    "ORCL", "DHR", "WFC", "DIS", "CMCSA", "NKE", "VZ", "TXN", "NEE",
    "PM", "UPS", "INTU", "COP", "RTX", "QCOM", "BMY", "T", "LOW",
    "HON", "IBM", "AMGN", "INTC", "GS", "SBUX", "BA", "BLK", "AXP",
    "CAT", "GE", "ELV", "MDT", "SPGI", "NOW", "AMT", "ISRG", "GILD",
    "DE", "BKNG", "LMT", "MMC", "ADP", "SYK", "ADI", "MO", "PLD",
    "REGN", "TJX", "VRTX", "CB", "CI", "MDLZ", "C", "DUK", "ZTS",
    "SO", "EOG", "BDX", "SLB", "CL", "EQIX", "BSX", "ITW", "AON",
    "MU", "ICE", "USB",
]

US_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "CSCO",
    "QCOM", "TXN", "INTU", "NOW", "PLTR", "SNOW",
]

US_LARGE = [
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "BAC",
    "XOM", "CVX", "ABBV", "PFE", "KO", "PEP", "MRK", "TMO",
    "COST", "DIS", "MCD", "NKE",
]

# === Europa ===
DAX40 = [
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE",
    "VOW3.DE", "BMW.DE", "MBG.DE", "AIR.DE", "IFX.DE", "ADS.DE",
    "MUV2.DE", "DBK.DE", "RWE.DE", "EOAN.DE", "FRE.DE", "HEN3.DE",
    "BEI.DE", "FME.DE", "MTX.DE", "CON.DE", "DB1.DE", "PUM.DE",
    "RHM.DE", "SY1.DE", "VNA.DE", "ZAL.DE", "BNR.DE", "HFG.DE",
    "HEI.DE", "SHL.DE", "CBK.DE", "ENR.DE", "QIA.DE", "SRT3.DE",
    "TKA.DE", "MRK.DE", "P911.DE",
]

FTSE100_TOP = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "RIO.L", "BP.L",
    "GSK.L", "REL.L", "DGE.L", "BATS.L", "NWG.L", "LSEG.L",
    "AAL.L", "BARC.L", "PRU.L", "NG.L", "RR.L", "GLEN.L",
    "IMB.L", "CPG.L", "TSCO.L", "IHG.L", "FERG.L", "ABF.L",
    "AV.L", "LLOY.L", "EXPN.L", "LGEN.L", "NXT.L", "RTO.L",
    "SSE.L", "SGE.L", "WPP.L", "BNZL.L", "ITRK.L",
]

EU_LARGE = [
    "ASML.AS", "SAP.DE", "NESN.SW", "NOVN.SW", "ROG.SW", "MC.PA",
    "OR.PA", "SAN.PA", "AIR.PA", "SIE.DE", "ALV.DE", "DTE.DE",
    "BAS.DE", "BAYN.DE", "VOW3.DE", "BMW.DE", "ADYEN.AS", "PRX.AS",
]

# === Sektor-fokuserede lister ===
SECTOR_TECH = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "INTU",
    "NOW", "PLTR", "SNOW", "SAP.DE", "ASML.AS", "IFX.DE",
]

SECTOR_FINANCE = [
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SPGI",
    "USB", "PNC", "TFC", "COF", "SCHW", "BX", "KKR",
    "HSBA.L", "BARC.L", "LLOY.L", "NWG.L",
    "DBK.DE", "ALV.DE", "MUV2.DE",
    "DANSKE.CO", "NDA-DK.CO", "TRYG.CO",
]

SECTOR_HEALTHCARE = [
    "LLY", "JNJ", "UNH", "MRK", "ABBV", "TMO", "ABT", "DHR",
    "PFE", "BMY", "AMGN", "GILD", "ISRG", "VRTX", "REGN", "ZTS",
    "SYK", "BSX", "MDT", "CI", "ELV",
    "NOVO-B.CO", "GMAB.CO", "BAVA.CO", "DEMANT.CO", "AMBU-B.CO",
    "NOVN.SW", "ROG.SW", "AZN.L", "GSK.L", "BAYN.DE", "MRK.DE",
]

SECTOR_ENERGY = [
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "OXY",
    "VLO", "PXD", "WMB", "KMI", "OKE",
    "SHEL.L", "BP.L", "EQNR.OL", "TTE.PA",
]

SECTOR_CONSUMER = [
    "AMZN", "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE",
    "HD", "LOW", "TJX", "SBUX", "MDLZ", "MO", "PM", "CL",
    "ULVR.L", "DGE.L", "TSCO.L", "ABF.L", "NESN.SW",
    "MC.PA", "OR.PA",
]

SECTOR_INDUSTRIAL = [
    "GE", "CAT", "RTX", "HON", "UPS", "BA", "DE", "LMT", "MMC",
    "ADP", "ITW", "ETN", "EMR", "PH",
    "SIE.DE", "AIR.DE", "MBG.DE", "BMW.DE", "VOW3.DE",
    "AIR.PA", "SAN.PA", "MAERSK-B.CO", "DSV.CO", "VWS.CO",
]

SECTOR_SEMICONDUCTOR = [
    "NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "MU", "ADI",
    "AMAT", "LRCX", "KLAC", "MRVL", "MCHP",
    "ASML.AS", "IFX.DE",
]

SECTOR_REIT = [
    "AMT", "PLD", "EQIX", "CCI", "PSA", "WELL", "DLR", "O",
    "SPG", "VICI", "SBAC", "EXR", "AVB", "EQR", "MAA",
]

# === Strategi-lists ===
GROWTH_STOCKS = [
    "NVDA", "TSLA", "PLTR", "SHOP", "SNOW", "CRWD", "DDOG",
    "NET", "MDB", "TEAM", "ZS", "ROKU", "SQ", "COIN",
]

DIVIDEND_ARISTOCRATS = [
    "JNJ", "PG", "KO", "PEP", "MMM", "CVX", "XOM", "MCD",
    "WMT", "T", "VZ", "IBM", "MO", "PM", "ABBV", "CL",
]

# === Master dict ===
SCREENER_UNIVERSES = {
    # Norden
    "🇩🇰 Danske C25": DK_C25,
    "🇸🇪🇳🇴🇫🇮 Nordic Large Cap": NORDIC_LARGE,
    # USA
    "🇺🇸 S&P 500 Top 100": SP500_TOP100,
    "🇺🇸 US Tech Giants": US_TECH,
    "🇺🇸 US Large Cap": US_LARGE,
    # Europa
    "🇩🇪 DAX 40": DAX40,
    "🇬🇧 FTSE 100 (top 35)": FTSE100_TOP,
    "🇪🇺 EU Large Cap": EU_LARGE,
    # Sektorer
    "💻 Sektor: Technology": SECTOR_TECH,
    "🏦 Sektor: Finance & Banks": SECTOR_FINANCE,
    "💊 Sektor: Healthcare/Pharma": SECTOR_HEALTHCARE,
    "⛽ Sektor: Energy": SECTOR_ENERGY,
    "🛒 Sektor: Consumer Goods": SECTOR_CONSUMER,
    "🏭 Sektor: Industrial": SECTOR_INDUSTRIAL,
    "🔬 Sektor: Semiconductors": SECTOR_SEMICONDUCTOR,
    "🏢 Sektor: REITs (Real Estate)": SECTOR_REIT,
    # Strategi
    "🚀 Growth Stocks": GROWTH_STOCKS,
    "💰 Dividend Aristocrats": DIVIDEND_ARISTOCRATS,
}

FX_FALLBACK = {
    ("USD", "DKK"): 6.85, ("DKK", "USD"): 1/6.85,
    ("EUR", "DKK"): 7.46, ("DKK", "EUR"): 1/7.46,
    ("EUR", "USD"): 1.08, ("USD", "EUR"): 1/1.08,
    ("GBP", "DKK"): 8.70, ("DKK", "GBP"): 1/8.70,
    ("CHF", "DKK"): 7.75, ("DKK", "CHF"): 1/7.75,
    ("SEK", "DKK"): 0.65, ("DKK", "SEK"): 1/0.65,
    ("NOK", "DKK"): 0.62, ("DKK", "NOK"): 1/0.62,
    ("JPY", "DKK"): 0.045, ("DKK", "JPY"): 1/0.045,
}
