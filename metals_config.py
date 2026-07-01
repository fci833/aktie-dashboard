"""
Konfiguration for ædelmetaller og relaterede råvarer.
Klart adskilt: FYSISK spot vs ETF vs MINING-aktier.
"""

METALS_UNIVERSE = {
    # ============ 🥇 FYSISK SPOT-PRIS (futures = spot) ============
    "GC=F": {
        "name": "🥇 GULD (fysisk spot-pris)",
        "category": "Fysisk metal",
        "type": "spot",
        "unit": "USD/oz",
        "description": "Fysisk guld spot-pris i USD per ounce. Den ægte guldpris - det man betaler for guldbarrer/-mønter. Handles via futures-kontrakter (COMEX).",
    },
    "SI=F": {
        "name": "🥈 SØLV (fysisk spot-pris)",
        "category": "Fysisk metal",
        "type": "spot",
        "unit": "USD/oz",
        "description": "Fysisk sølv spot-pris. Delvist industrielt metal (solceller, elektronik) - derfor mere volatilt end guld.",
    },
    "PL=F": {
        "name": "🔩 PLATIN (fysisk spot-pris)",
        "category": "Fysisk metal",
        "type": "spot",
        "unit": "USD/oz",
        "description": "Fysisk platin spot-pris. Bruges primært i bilkatalysatorer, smykker og medicinsk udstyr.",
    },
    "PA=F": {
        "name": "⚙️ PALLADIUM (fysisk spot-pris)",
        "category": "Fysisk metal",
        "type": "spot",
        "unit": "USD/oz",
        "description": "Fysisk palladium spot-pris. Ekstremt volatilt - bruges primært i benzinbil-katalysatorer.",
    },

    # ============ 📊 ETF'er (papir-metal, handles som aktier) ============
    "GLD": {
        "name": "📊 SPDR Gold Shares ETF",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Størst guld-ETF. 1 andel ≈ 1/10 oz guld. Handles som en aktie via din normale broker.",
    },
    "IAU": {
        "name": "📊 iShares Gold Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Billigere guld-ETF alternativ til GLD (lavere expense ratio 0.25% vs 0.40%).",
    },
    "SLV": {
        "name": "📊 iShares Silver Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Største sølv-ETF. 1 andel ≈ 1 oz sølv.",
    },
    "PPLT": {
        "name": "📊 Aberdeen Platinum ETF",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Platinum ETF - fysisk backet.",
    },
    "PALL": {
        "name": "📊 Aberdeen Palladium ETF",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "Palladium ETF - fysisk backet.",
    },
    "PHYS": {
        "name": "📊 Sprott Physical Gold Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "100% fysisk guld-ETF (indløselig mod fysiske barrer). Populær hos hardcore guld-fans.",
    },
    "PSLV": {
        "name": "📊 Sprott Physical Silver Trust",
        "category": "Metal-ETF",
        "type": "etf",
        "unit": "USD",
        "description": "100% fysisk sølv-ETF - indløselig mod barrer.",
    },

    # ============ ⛏️ MINING-AKTIER (indirekte eksponering) ============
    "GDX": {
        "name": "⛏️ VanEck Gold Miners ETF",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Store guldminer-selskaber. Gearet eksponering til guld - kan stige/falde 2-3x mere end guldprisen.",
    },
    "GDXJ": {
        "name": "⛏️ VanEck Junior Gold Miners",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Små/mellemstore guldminer-selskaber - HØJ risk/reward.",
    },
    "SIL": {
        "name": "⛏️ Global X Silver Miners",
        "category": "Mining",
        "type": "etf",
        "unit": "USD",
        "description": "Sølvminer-selskaber - endnu mere geared end guldminer.",
    },
    "NEM": {
        "name": "⛏️ Newmont Corporation",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Verdens største guldminer-selskab. Solid, diversificeret producent.",
    },
    "GOLD": {
        "name": "⛏️ Barrick Gold",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "En af verdens største guldproducenter. Global diversificering.",
    },
    "AEM": {
        "name": "⛏️ Agnico Eagle Mines",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Canadisk guldminer med meget lave produktionsomkostninger. Kvalitetsvalg.",
    },
    "FNV": {
        "name": "⛏️ Franco-Nevada (royalty)",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Royalty-selskab (køber rettigheder fra minerne). Lavere risiko end miner selv.",
    },
    "WPM": {
        "name": "⛏️ Wheaton Precious Metals",
        "category": "Mining",
        "type": "stock",
        "unit": "USD",
        "description": "Streaming-selskab (køber sølv/guld til fast pris). Lavere risiko model.",
    },

    # ============ 🏭 INDUSTRIELLE METALLER ============
    "HG=F": {
        "name": "🏭 Kobber (fysisk spot-pris)",
        "category": "Industri-metal",
        "type": "spot",
        "unit": "USD/lb",
        "description": "Kobber - 'Dr. Copper' - regnes som økonomisk indikator (stiger når økonomien vokser).",
    },
    "CPER": {
        "name": "📊 US Copper Fund ETF",
        "category": "Industri-metal",
        "type": "etf",
        "unit": "USD",
        "description": "Kobber-ETF for private investorer der ikke kan handle futures.",
    },
}


METALS_UNIVERSES = {
    "🥇 FYSISK spot-pris (guld/sølv/platin)": ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F"],
    "📊 Metal-ETF'er (nemmest at handle)": ["GLD", "IAU", "SLV", "PPLT", "PALL", "PHYS", "PSLV"],
    "⛏️ Mining-aktier & ETF'er": ["GDX", "GDXJ", "SIL", "NEM", "GOLD", "AEM", "FNV", "WPM"],
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
