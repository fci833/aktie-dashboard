"""
Konfiguration for ædelmetaller og relaterede råvarer.
"""

METALS_UNIVERSE = {
    # ============ ÆDELMETALLER (Futures - spot pris) ============
    "GC=F": {
        "name": "Guld (Futures)",
        "category": "Ædelmetal",
        "type": "future",
        "unit": "USD/oz",
        "description": "Guld spot-pris. Den ultimative safe haven.",
    },
    "SI=F": {
        "name": "Sølv (Futures)",
        "category": "Ædelmetal",
        "type": "future",
        "unit": "USD/oz",
        "description": "Sølv - mere volatilt end guld, delvist industrielt.",
    },
    "PL=F": {
        "name": "Platin (Futures)",
        "category": "Ædelmetal",
        "type": "future",
        "unit": "USD/oz",
        "description": "Platin - industriel efterspørgsel (biler).",
    },
    "PA=F": {
        "name": "Palladium (Futures)",
        "category": "Ædelmetal",
        "type": "future",
        "unit": "USD/oz",
        "description": "Palladium - katalysatorer, meget volatilt.",
    },

    # ============ ETF'er (nemmere at handle) ============
    "GLD": {
        "name": "SPDR Gold Shares",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Størst guld-ETF. 1 andel ≈ 1/10 oz guld.",
    },
    "IAU": {
        "name": "iShares Gold Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Billigere alternativ til GLD (lavere expense ratio).",
    },
    "SLV": {
        "name": "iShares Silver Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Største sølv-ETF.",
    },
    "PPLT": {
        "name": "Aberdeen Platinum ETF",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Platinum ETF.",
    },
    "PALL": {
        "name": "Aberdeen Palladium ETF",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Palladium ETF.",
    },

    # ============ MINING-AKTIER (indirekte eksponering) ============
    "GDX": {
        "name": "VanEck Gold Miners ETF",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Store guldminer-selskaber. Gearet eksponering til guld.",
    },
    "GDXJ": {
        "name": "VanEck Junior Gold Miners",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Små/mellemstore guldminer-selskaber - HØJ risk/reward.",
    },
    "SIL": {
        "name": "Global X Silver Miners",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Sølvminer-selskaber.",
    },
    "NEM": {
        "name": "Newmont Corporation",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Verdens største guldminer-selskab.",
    },
    "GOLD": {
        "name": "Barrick Gold",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "En af verdens største guldproducenter.",
    },
    "AEM": {
        "name": "Agnico Eagle Mines",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Canadisk guldminer med lave omkostninger.",
    },

    # ============ INDUSTRIELLE METALLER (bonus) ============
    "HG=F": {
        "name": "Kobber (Futures)",
        "category": "Industri-metal",
        "type": "future",
        "unit": "USD/lb",
        "description": "Kobber - 'Dr. Copper' - økonomisk indikator.",
    },
    "CPER": {
        "name": "United States Copper Fund",
        "category": "Industri-metal",
        "type": "etf",
        "unit": "USD",
        "description": "Kobber-ETF for private investorer.",
    },
}


METALS_UNIVERSES = {
    "🥇 Kun ædelmetaller (Futures)": ["GC=F", "SI=F", "PL=F", "PA=F"],
    "📊 Metal-ETF'er": ["GLD", "IAU", "SLV", "PPLT", "PALL"],
    "⛏️ Mining-aktier & ETF'er": ["GDX", "GDXJ", "SIL", "NEM", "GOLD", "AEM"],
    "🏭 Industri-metaller": ["HG=F", "CPER"],
    "🌍 Alle metaller": list(METALS_UNIVERSE.keys()),
}


def is_metal(ticker):
    """Tjekker om ticker er et metal / relateret instrument"""
    if not ticker:
        return False
    return ticker.upper().strip() in METALS_UNIVERSE


def get_metal_info(ticker):
    """Henter metal-info fra config"""
    if not ticker:
        return None
    return METALS_UNIVERSE.get(ticker.upper().strip())
