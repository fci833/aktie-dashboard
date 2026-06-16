"""
ml_backfill.py - Historical Training Data Generator
=====================================================
Generates synthetic 'snapshots' from historical price data so you can
train ML immediately without waiting months for forward returns.

🚀 PHASE 1 UPDATE:
- Expanded to 250+ stocks across multiple regions/sectors
- Expanded to 75+ crypto pairs
- Default backfill period: 36 months (was 24)
- Default snapshot interval: 14 days (was 30)
- Result: ~8-10x more training samples!

🔥 PHASE 1B UPDATE (NEW):
- Added 11 POWER FEATURES for ML:
  * momentum_3m, momentum_6m, momentum_12m
  * momentum_acceleration
  * volatility_regime
  * volume_momentum
  * drawdown_depth, max_drawdown_1y
  * recovery_strength
  * trend_consistency
  * price_position_52w
- Expected F1 lift: 0.55 → 0.65+

Strategy:
  1. For each ticker, fetch 5 years of historical data
  2. Pick N historical 'snapshot dates' (e.g. every 14 days going back 3 years)
  3. At each historical date: compute scores AS IF we had screened then
  4. Compute forward returns using known future prices
  5. Build a complete training dataset

Main entry: build_backfill_dataset(tickers, asset_class="stock")
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import ta

HORIZONS = [30, 90, 180]


# ==========================================
# DEFAULT TICKER UNIVERSES (for backfill)
# 🚀 PHASE 1: Expanded to 250+ stocks, 75+ crypto
# ==========================================

DEFAULT_TICKERS = {
    "us_large_cap": [
        # ===== Big Tech =====
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        "AMD", "INTC", "ORCL", "CRM", "ADBE", "CSCO", "IBM", "QCOM",
        "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL",
        "PANW", "FTNT", "ANET", "NOW", "INTU", "WDAY",

        # ===== Finance =====
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "BLK",
        "SCHW", "USB", "PNC", "TFC", "COF", "BX", "KKR", "APO",
        "SPGI", "MCO", "ICE", "CME", "AON", "MMC", "PGR", "TRV",
        "ALL", "AIG", "MET", "PRU",

        # ===== Healthcare =====
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY", "DHR",
        "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG", "SYK", "BSX",
        "ZTS", "BDX", "EW", "DXCM", "IDXX", "VRTX", "REGN", "BIIB",
        "HUM", "CI", "ELV", "CNC", "MOH",

        # ===== Consumer =====
        "WMT", "HD", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "DIS",
        "COST", "TGT", "LOW", "TJX", "DG", "DLTR", "ROST", "ULTA",
        "BBY", "KR", "SYY", "MDLZ", "KHC", "GIS", "CL", "KMB",
        "EL", "CHD", "CLX", "MNST", "STZ",

        # ===== Industrial =====
        "BA", "CAT", "GE", "HON", "MMM", "UPS", "FDX", "RTX", "LMT",
        "DE", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "FAST",
        "NOC", "GD", "TDG", "WM", "RSG", "CSX", "UNP", "NSC",

        # ===== Energy =====
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC",
        "OXY", "DVN", "HES", "WMB", "OKE", "KMI", "ENB",

        # ===== REITs =====
        "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL",
        "DLR", "AVB", "EQR", "VICI", "EXR",

        # ===== Communication =====
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "NFLX",

        # ===== Utilities =====
        "NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "XEL",
    ],

    "us_growth": [
        # ===== Cloud / SaaS =====
        "PLTR", "SNOW", "CRWD", "ZS", "DDOG", "NET", "MDB", "OKTA",
        "TEAM", "ESTC", "GTLB", "BILL", "HUBS", "ZM", "DOCN",
        "FROG", "CFLT", "BRZE",

        # ===== FinTech =====
        "SHOP", "SQ", "PYPL", "AFRM", "SOFI", "UPST", "HOOD", "COIN",
        "MARA", "RIOT", "CLSK", "BTBT",

        # ===== AI / Robotics =====
        "AI", "BBAI", "SOUN", "SMCI", "ARM", "PATH", "ASTS", "RKLB",
        "ACHR", "JOBY", "NVTS", "POWI",

        # ===== Gaming / Media =====
        "ROKU", "TTD", "RBLX", "U", "SPOT", "WBD", "PARA", "EA",
        "TTWO", "DKNG", "PENN", "LYFT", "UBER", "ABNB", "DASH",

        # ===== Biotech =====
        "MRNA", "BNTX", "ALNY", "BMRN", "INCY", "EXEL", "NBIX",
        "RXRX", "RVMD", "KRYS", "NVAX", "ARQT", "VKTX",

        # ===== EV / Auto =====
        "RIVN", "LCID", "F", "GM", "STLA", "TM", "HMC",

        # ===== Cybersecurity =====
        "CYBR", "TENB", "QLYS", "RPD",

        # ===== Other Growth =====
        "MELI", "ENPH", "FSLR", "RUN", "PLUG", "BE",
        "CHPT", "QS", "BLNK",
    ],

    "us_dividend": [
        "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "VZ", "T", "MO",
        "ABBV", "PFE", "MRK", "MMM", "CAT", "MCD", "WMT", "HD",
        "IBM", "PM", "BMY", "PEP", "KMB", "ED", "SO", "DUK",
    ],

    "european": [
        # ===== Danish =====
        "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO", "CARL-B.CO",
        "GMAB.CO", "ROCK-B.CO", "TRYG.CO", "DANSKE.CO", "VWS.CO",
        "ISS.CO", "PNDORA.CO", "GN.CO", "DEMANT.CO", "AMBU-B.CO",
        "COLO-B.CO", "BAVA.CO", "FLS.CO", "JYSK.CO", "RBREW.CO",
        "TOP.CO", "ZEAL.CO", "NDA-DK.CO", "BO.CO",

        # ===== German =====
        "SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE",
        "MBG.DE", "DTE.DE", "MUV2.DE", "VOW3.DE", "ADS.DE", "DHL.DE",
        "DBK.DE", "IFX.DE", "RWE.DE", "BEI.DE", "HEN3.DE", "FRE.DE",
        "MRK.DE", "LIN.DE", "EOAN.DE", "PAH3.DE",

        # ===== Dutch =====
        "ASML.AS", "PHIA.AS", "INGA.AS", "AD.AS", "UNA.AS",
        "PRX.AS", "HEIA.AS", "WKL.AS", "RAND.AS", "AKZA.AS",
        "DSM.AS", "MT.AS",

        # ===== Swiss =====
        "NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW", "ABBN.SW",
        "ZURN.SW", "GIVN.SW", "LONN.SW", "SREN.SW",
        "GEBN.SW", "ALC.SW",

        # ===== French =====
        "MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "SU.PA", "BNP.PA",
        "AI.PA", "RMS.PA", "KER.PA", "CS.PA", "ENGI.PA", "VIE.PA",
        "DG.PA", "CAP.PA", "PUB.PA", "TTE.PA",

        # ===== UK =====
        "AZN.L", "HSBA.L", "BP.L", "GSK.L", "ULVR.L", "RIO.L", "VOD.L",
        "BARC.L", "LLOY.L", "REL.L", "SHEL.L", "DGE.L", "BHP.L",
        "GLEN.L", "BATS.L", "PRU.L", "TSCO.L", "AAL.L",

        # ===== Nordic (non-DK) =====
        "EQNR.OL", "DNB.OL", "TEL.OL", "MOWI.OL",
        "VOLV-B.ST", "ATCO-A.ST", "INVE-B.ST", "HEXA-B.ST",
        "ERIC-B.ST", "SEB-A.ST", "SHB-A.ST", "ASSA-B.ST",
        "NDA-FI.HE", "NESTE.HE", "KNEBV.HE",
    ],

    "emerging": [
        # ===== Latin America =====
        "MELI", "VALE", "PBR", "ITUB", "BBD", "ABEV", "NU",
        "STNE", "PAGS", "VIST",

        # ===== Asia (China/HK) =====
        "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV", "TME",
        "NTES", "TCOM", "HTHT", "BILI",

        # ===== Asia (Other) =====
        "TSM", "INFY", "WIT", "HDB", "IBN", "RDY",
        "TCEHY", "GLNG", "SE",

        # ===== Global =====
        "TM", "HMC", "SONY", "MUFG", "SMFG",
    ],

    "crypto": [
        # ===== Top 10 by market cap =====
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "TRX-USD", "TON-USD",

        # ===== Layer 1 =====
        "DOT-USD", "AVAX-USD", "ATOM-USD", "NEAR-USD", "APT-USD",
        "SUI-USD", "SEI-USD", "INJ-USD", "TIA-USD", "ALGO-USD",
        "ICP-USD", "VET-USD", "HBAR-USD", "FIL-USD", "FTM-USD",
        "RUNE-USD",

        # ===== Layer 2 / Scaling =====
        "MATIC-USD", "ARB-USD", "OP-USD", "LRC-USD", "IMX-USD",
        "MNT-USD", "STRK-USD",

        # ===== DeFi =====
        "LINK-USD", "UNI-USD", "AAVE-USD", "MKR-USD", "SNX-USD",
        "CRV-USD", "LDO-USD", "RPL-USD", "GRT-USD", "DYDX-USD",
        "1INCH-USD", "COMP-USD",

        # ===== Memecoins =====
        "SHIB-USD", "PEPE-USD", "BONK-USD", "WIF-USD", "FLOKI-USD",

        # ===== AI / Big themes =====
        "RNDR-USD", "FET-USD", "AGIX-USD", "TAO-USD", "OCEAN-USD",
        "ROSE-USD",

        # ===== Storage / Web3 =====
        "AR-USD", "STX-USD", "BLUR-USD",

        # ===== Gaming / NFT =====
        "MANA-USD", "SAND-USD", "AXS-USD", "GALA-USD", "ENJ-USD",
        "APE-USD",

        # ===== Classic =====
        "LTC-USD", "BCH-USD", "ETC-USD", "XLM-USD", "XMR-USD",
        "DASH-USD",
    ],
}


# ==========================================
# PRICE FETCHING (cached & parallel)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_history(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Fetch 5 years of price data for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 250:
            return None
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Optional[dict]:
    """Get current ticker info (sector, market cap, etc)."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or "longName" not in info:
            return None
        return info
    except Exception:
        return None


# ==========================================
# HISTORICAL INDICATOR COMPUTATION
# 🔥 PHASE 1B: Now with 11 power features!
# ==========================================

def compute_indicators_at_date(
    hist: pd.DataFrame,
    target_date: pd.Timestamp,
    lookback_days: int = 400,  # 🔥 PHASE 1B: udvidet fra 365 → 400 (til 12m momentum)
) -> Optional[Dict]:
    """
    Compute technical indicators using ONLY data up to target_date.
    🔥 PHASE 1B: Tilføjet 11 nye power features.
    
    Works for both stocks AND crypto - all features are universal.
    """
    if hist is None or hist.empty:
        return None

    target_ts = pd.Timestamp(target_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)

    # Slice: only data up to target_date
    hist_slice = hist[hist.index <= target_ts].tail(lookback_days)
    if len(hist_slice) < 50:
        return None

    try:
        close = hist_slice["Close"]
        high = hist_slice["High"]
        low = hist_slice["Low"]
        volume = hist_slice["Volume"] if "Volume" in hist_slice.columns else None

        # ===== STANDARD INDICATORS =====
        rsi = ta.momentum.rsi(close, window=14).iloc[-1]
        macd_obj = ta.trend.MACD(close)
        macd = macd_obj.macd_diff().iloc[-1]
        adx = ta.trend.ADXIndicator(high, low, close).adx().iloc[-1]
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]
        bb = ta.volatility.BollingerBands(close)
        bb_high = bb.bollinger_hband().iloc[-1]
        bb_low = bb.bollinger_lband().iloc[-1]

        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        current_price = float(close.iloc[-1])
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        low_52w = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())

        # ===== STANDARD RETURNS =====
        ret_30d = ((close.iloc[-1] / close.iloc[-30] - 1) * 100) if len(close) >= 30 else 0
        ret_90d = ((close.iloc[-1] / close.iloc[-90] - 1) * 100) if len(close) >= 90 else 0

        # ============================================================
        # 🔥 PHASE 1B POWER FEATURES (works for stocks AND crypto)
        # ============================================================

        # 1. Momentum-faktorer (3m, 6m, 12m absolute returns)
        momentum_3m = ((close.iloc[-1] / close.iloc[-63] - 1) * 100) if len(close) >= 63 else ret_90d
        momentum_6m = ((close.iloc[-1] / close.iloc[-126] - 1) * 100) if len(close) >= 126 else momentum_3m
        momentum_12m = ((close.iloc[-1] / close.iloc[-252] - 1) * 100) if len(close) >= 252 else momentum_6m

        # 2. Momentum-acceleration: stiger momentum?
        # Hvis 3m er højere end forventet ud fra 6m → accelererer
        momentum_acceleration = momentum_3m - (momentum_6m / 2)

        # 3. Volatility regime: stigende eller faldende volatilitet?
        if len(close) >= 60:
            recent_vol = close.tail(30).pct_change().std() * 100
            older_vol = close.iloc[-60:-30].pct_change().std() * 100
            volatility_regime = (recent_vol / older_vol - 1) * 100 if older_vol > 0 else 0
            volatility_regime = max(-100, min(500, volatility_regime))  # cap extreme
        else:
            volatility_regime = 0

        # 4. Volume momentum (smart money flow)
        if volume is not None and len(volume) >= 60 and volume.tail(30).mean() > 0:
            recent_vol_avg = volume.tail(30).mean()
            older_vol_avg = volume.iloc[-60:-30].mean()
            if older_vol_avg > 0:
                volume_momentum = (recent_vol_avg / older_vol_avg - 1) * 100
                volume_momentum = max(-100, min(500, volume_momentum))
            else:
                volume_momentum = 0
        else:
            volume_momentum = 0

        # 5. Drawdown depth (hvor langt fra ATH i seneste år)
        try:
            if len(close) >= 252:
                tail_close = close.tail(252)
                rolling_max = tail_close.expanding().max()
                drawdown_series = (tail_close / rolling_max - 1) * 100
            else:
                rolling_max = close.expanding().max()
                drawdown_series = (close / rolling_max - 1) * 100
            drawdown_depth = float(drawdown_series.iloc[-1])
            max_drawdown_1y = float(drawdown_series.min())
        except Exception:
            drawdown_depth = 0
            max_drawdown_1y = 0

        # 6. Recovery strength (bouncer fra recent lav?)
        if len(close) >= 60:
            recent_low = float(close.tail(60).min())
            recovery_strength = ((current_price / recent_low - 1) * 100) if recent_low > 0 else 0
        else:
            recovery_strength = 0

        # 7. Trend consistency (% af dage over SMA50)
        if sma50 and len(close) >= 50:
            try:
                sma50_series = close.rolling(50).mean()
                valid = sma50_series.dropna()
                if len(valid) > 0:
                    aligned_close = close.tail(len(valid))
                    days_above = (aligned_close > valid).sum()
                    trend_consistency = (days_above / len(valid)) * 100
                else:
                    trend_consistency = 50
            except Exception:
                trend_consistency = 50
        else:
            trend_consistency = 50

        # 8. Price position in 52w range (0=bottom, 100=top)
        if high_52w > low_52w:
            price_position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100
        else:
            price_position_52w = 50

        return {
            # ===== Standard features =====
            "price": current_price,
            "rsi": float(rsi) if not pd.isna(rsi) else 50,
            "macd": float(macd) if not pd.isna(macd) else 0,
            "adx": float(adx) if not pd.isna(adx) else 25,
            "atr": float(atr) if not pd.isna(atr) else 0,
            "atr_pct": (float(atr) / current_price * 100) if current_price > 0 and not pd.isna(atr) else 2,
            "bb_width": ((bb_high - bb_low) / current_price * 100) if current_price > 0 else 5,
            "vs_sma200_%": ((current_price / sma200 - 1) * 100) if sma200 else 0,
            "vs_sma50_%": ((current_price / sma50 - 1) * 100) if sma50 else 0,
            "vs_52w_high_%": ((current_price / high_52w - 1) * 100) if high_52w > 0 else 0,
            "vs_52w_low_%": ((current_price / low_52w - 1) * 100) if low_52w > 0 else 0,
            "ret_30d": ret_30d,
            "ret_90d": ret_90d,
            "change_%": ret_30d,
            # ===== 🔥 PHASE 1B power features =====
            "momentum_3m": float(momentum_3m),
            "momentum_6m": float(momentum_6m),
            "momentum_12m": float(momentum_12m),
            "momentum_acceleration": float(momentum_acceleration),
            "volatility_regime": float(volatility_regime),
            "volume_momentum": float(volume_momentum),
            "drawdown_depth": float(drawdown_depth),
            "max_drawdown_1y": float(max_drawdown_1y),
            "recovery_strength": float(recovery_strength),
            "trend_consistency": float(trend_consistency),
            "price_position_52w": float(price_position_52w),
        }
    except Exception as e:
        print(f"⚠️ compute_indicators_at_date error: {e}")
        return None


def compute_simple_score(indicators: dict) -> dict:
    """
    Simple rule-based scoring (matches your existing system).
    Returns f_score, t_score, overall.
    
    🔥 PHASE 1B: Nu også med momentum-aware scoring.
    """
    if not indicators:
        return {"f_score": 50, "t_score": 50, "overall": 50, "regime": "UNKNOWN"}

    # ===== Technical score (0-100) =====
    t = 50.0
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        t += 15  # Oversold = good buy
    elif rsi < 40:
        t += 8
    elif rsi > 70:
        t -= 15
    elif rsi > 60:
        t -= 5

    macd = indicators.get("macd", 0)
    if macd > 0:
        t += 10
    elif macd < 0:
        t -= 10

    vs_sma200 = indicators.get("vs_sma200_%", 0)
    if vs_sma200 > 5:
        t += 8
    elif vs_sma200 < -10:
        t -= 8

    vs_52w = indicators.get("vs_52w_high_%", 0)
    if vs_52w < -25:
        t += 10  # Way below highs = potential value
    elif vs_52w > -5:
        t -= 5

    # 🔥 PHASE 1B: trend consistency boost
    trend_cons = indicators.get("trend_consistency", 50)
    if trend_cons > 70:
        t += 5  # Konsistent uptrend
    elif trend_cons < 30:
        t -= 5  # Konsistent downtrend

    # ===== Fundamental proxy =====
    f = 50.0
    ret_90d = indicators.get("ret_90d", 0)
    if ret_90d > 10:
        f += 12
    elif ret_90d > 0:
        f += 5
    elif ret_90d < -15:
        f -= 12
    elif ret_90d < -5:
        f -= 5

    atr_pct = indicators.get("atr_pct", 2)
    if atr_pct > 5:
        f -= 8  # Too volatile
    elif atr_pct < 1.5:
        f += 5  # Stable

    # 🔥 PHASE 1B: momentum acceleration boost
    mom_accel = indicators.get("momentum_acceleration", 0)
    if mom_accel > 5:
        f += 5  # Accelererer opad
    elif mom_accel < -5:
        f -= 5  # Decelererer

    # ===== Regime detection =====
    if vs_sma200 > 5 and ret_90d > 5:
        regime = "BULL"
    elif vs_sma200 < -10 and ret_90d < -10:
        regime = "BEAR"
    elif atr_pct > 4:
        regime = "VOLATILE"
    else:
        regime = "SIDEWAYS"

    f = max(0, min(100, f))
    t = max(0, min(100, t))

    # Combined (regime-weighted)
    if regime == "BULL":
        overall = f * 0.4 + t * 0.6
    elif regime in ("BEAR", "VOLATILE"):
        overall = f * 0.7 + t * 0.3
    else:
        overall = f * 0.6 + t * 0.4

    return {
        "f_score": f,
        "t_score": t,
        "overall": overall,
        "regime": regime,
    }


# ==========================================
# BACKFILL CORE
# ==========================================

def get_price_at_date(hist: pd.DataFrame, target_date: pd.Timestamp,
                     max_lookahead: int = 7) -> Optional[float]:
    """Find close price on target_date or nearest forward trading day."""
    if hist is None or hist.empty:
        return None
    target_ts = pd.Timestamp(target_date)
    if target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)
    future = hist[hist.index >= target_ts]
    if future.empty:
        return None
    days_diff = (future.index[0] - target_ts).days
    if days_diff > max_lookahead:
        return None
    return float(future["Close"].iloc[0])


def generate_snapshots_for_ticker(
    ticker: str,
    snapshot_dates: List[pd.Timestamp],
    horizons: List[int] = HORIZONS,
) -> List[Dict]:
    """
    Generate synthetic snapshots at multiple historical dates for one ticker.
    Returns list of dicts (one per snapshot date).
    
    🔥 PHASE 1B: Now saves all 11 power features per row.
    """
    hist = fetch_ticker_history(ticker, period="5y")
    if hist is None:
        return []

    info = fetch_ticker_info(ticker) or {}
    sector = info.get("sector", "Unknown")
    country = info.get("country", "Unknown")
    currency = info.get("currency", "USD")
    market_cap = info.get("marketCap", 0) or 0
    name = info.get("longName") or info.get("shortName") or ticker

    rows = []
    for snap_date in snapshot_dates:
        snap_ts = pd.Timestamp(snap_date)
        if snap_ts.tz is not None:
            snap_ts = snap_ts.tz_localize(None)

        # Need data BEFORE this date for indicators
        if hist.index.min() > snap_ts - timedelta(days=200):
            continue

        # Compute indicators using only past data
        indicators = compute_indicators_at_date(hist, snap_ts)
        if not indicators:
            continue

        scores = compute_simple_score(indicators)

        # Get snapshot price
        snap_price = get_price_at_date(hist, snap_ts)
        if snap_price is None or snap_price <= 0:
            continue

        # Compute forward returns
        future_returns = {}
        all_horizons_ok = False
        for h in horizons:
            future_date = snap_ts + timedelta(days=h)
            future_price = get_price_at_date(hist, future_date)
            if future_price is None or future_price <= 0:
                future_returns[f"future_return_{h}d"] = None
            else:
                future_returns[f"future_return_{h}d"] = (future_price / snap_price - 1) * 100
                all_horizons_ok = True

        # Skip if no forward returns at all (snapshot too recent)
        if not all_horizons_ok:
            continue

        row = {
            # ===== Metadata =====
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "country": country,
            "currency": currency,
            "market_cap": market_cap,
            "snapshot_ts": snap_ts,
            "snapshot_universe": "BACKFILL",
            "price": snap_price,
            "status": "✅",
            # ===== Scores =====
            "f_score": scores["f_score"],
            "t_score": scores["t_score"],
            "overall": scores["overall"],
            "regime": scores["regime"],
            "regime_confidence": 75,
            # ===== Standard tech indicators =====
            "rsi": indicators["rsi"],
            "macd": indicators["macd"],
            "atr_pct": indicators["atr_pct"],
            "vs_sma200_%": indicators["vs_sma200_%"],
            "vs_52w_high_%": indicators["vs_52w_high_%"],
            "change_%": indicators["change_%"],
            # ===== 🔥 PHASE 1B power features =====
            "momentum_3m": indicators["momentum_3m"],
            "momentum_6m": indicators["momentum_6m"],
            "momentum_12m": indicators["momentum_12m"],
            "momentum_acceleration": indicators["momentum_acceleration"],
            "volatility_regime": indicators["volatility_regime"],
            "volume_momentum": indicators["volume_momentum"],
            "drawdown_depth": indicators["drawdown_depth"],
            "max_drawdown_1y": indicators["max_drawdown_1y"],
            "recovery_strength": indicators["recovery_strength"],
            "trend_consistency": indicators["trend_consistency"],
            "price_position_52w": indicators["price_position_52w"],
            # ===== Fundamental defaults =====
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "peg": info.get("pegRatio"),
            "dividend_%": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else 0,
            "profit_margin": (info.get("profitMargins") or 0) * 100 if info.get("profitMargins") else 0,
            "roe": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else 0,
            "debt_equity": info.get("debtToEquity"),
            "dcf_upside_%": 0,  # not computed in backfill
            **future_returns,
        }
        rows.append(row)

    return rows


def generate_snapshot_dates(
    months_back: int = 36,
    interval_days: int = 14,
) -> List[pd.Timestamp]:
    """Generate snapshot dates going backwards in time."""
    today = pd.Timestamp.now().normalize()
    # Start at least 200 days ago (so we have time for 180d forward return)
    end_date = today - timedelta(days=200)
    start_date = today - timedelta(days=months_back * 30)

    dates = []
    current = end_date
    while current >= start_date:
        dates.append(current)
        current -= timedelta(days=interval_days)

    return sorted(dates)


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def build_backfill_dataset(
    tickers: Optional[List[str]] = None,
    asset_class: str = "stock",
    months_back: int = 36,
    snapshot_interval_days: int = 14,
    progress_callback=None,
    max_workers: int = 4,
) -> Dict:
    """
    Build a complete backfilled training dataset.

    Args:
        tickers: List of tickers (uses defaults if None)
        asset_class: "stock" or "crypto"
        months_back: How far back to generate snapshots (months) - default 36
        snapshot_interval_days: Days between snapshots - default 14
        progress_callback: callable(current, total, ticker)

    Returns dict with same shape as ml_data.get_training_data()
    """
    # Default tickers - now uses ALL universes for max data
    if tickers is None:
        if asset_class == "crypto":
            tickers = DEFAULT_TICKERS["crypto"]
        else:
            tickers = (
                DEFAULT_TICKERS["us_large_cap"]
                + DEFAULT_TICKERS["us_growth"]
                + DEFAULT_TICKERS["european"]
                + DEFAULT_TICKERS["emerging"]
            )
            tickers = list(set(tickers))  # dedupe

    snapshot_dates = generate_snapshot_dates(months_back, snapshot_interval_days)
    if not snapshot_dates:
        return {"error": "No valid snapshot dates", "n_samples": 0}

    print(f"📅 Generating {len(snapshot_dates)} snapshot dates × {len(tickers)} tickers")
    print(f"   = ~{len(snapshot_dates) * len(tickers)} potential samples")

    all_rows = []
    n_total = len(tickers)
    completed = 0

    # Sequential to avoid yfinance rate limits
    for i, ticker in enumerate(tickers):
        try:
            rows = generate_snapshots_for_ticker(ticker, snapshot_dates)
            all_rows.extend(rows)
            completed += 1
            if progress_callback:
                progress_callback(completed, n_total, ticker)
        except Exception as e:
            print(f"⚠️ {ticker}: {e}")
            completed += 1
            if progress_callback:
                progress_callback(completed, n_total, ticker)
        time.sleep(0.1)  # gentle on yfinance

    if not all_rows:
        return {"error": "No data generated", "n_samples": 0}

    df = pd.DataFrame(all_rows)
    print(f"✅ Generated {len(df)} training rows from {df['ticker'].nunique()} tickers")
    print(f"   Features: {len(df.columns)} columns")

    return {
        "df": df,
        "n_rows": len(df),
        "n_tickers": df["ticker"].nunique(),
        "n_snapshots": df["snapshot_ts"].nunique(),
        "asset_class": asset_class,
        "date_range": (df["snapshot_ts"].min(), df["snapshot_ts"].max()),
    }


# ==========================================
# SAVE TO SNAPSHOT FORMAT
# ==========================================

def save_backfill_as_snapshots(df: pd.DataFrame, snapshots_dir: str = "screener_snapshots"):
    """
    Convert backfill DataFrame into individual snapshot CSV files.
    """
    import os
    from pathlib import Path

    Path(snapshots_dir).mkdir(exist_ok=True)
    saved = 0

    for snap_ts in df["snapshot_ts"].unique():
        snap_df = df[df["snapshot_ts"] == snap_ts].copy()
        snap_df = snap_df.drop(columns=["snapshot_ts", "snapshot_universe"], errors="ignore")

        ts = pd.Timestamp(snap_ts)
        filename = f"backfill_{ts.strftime('%Y%m%d')}_BACKFILL.csv"
        filepath = os.path.join(snapshots_dir, filename)

        snap_df.to_csv(filepath, index=False)
        saved += 1

    return saved


# ==========================================
# SESSION STATE INTEGRATION
# ==========================================

def store_backfill_in_session(df: pd.DataFrame):
    """Store backfill DataFrame in Streamlit session state."""
    try:
        import streamlit as st
        st.session_state["ml_backfill_df"] = df.copy()
        st.session_state["ml_backfill_ts"] = pd.Timestamp.now()
        return True
    except Exception:
        return False


def get_backfill_from_session():
    """Get cached backfill DataFrame from session state."""
    try:
        import streamlit as st
        return st.session_state.get("ml_backfill_df")
    except Exception:
        return None


def has_backfill_in_session() -> bool:
    """Check if backfill data is available in session."""
    try:
        import streamlit as st
        df = st.session_state.get("ml_backfill_df")
        return df is not None and not df.empty
    except Exception:
        return False


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML BACKFILL - HISTORICAL DATA GENERATOR (PHASE 1B)")
    print("=" * 70)

    total_stocks = len(set(
        DEFAULT_TICKERS["us_large_cap"]
        + DEFAULT_TICKERS["us_growth"]
        + DEFAULT_TICKERS["european"]
        + DEFAULT_TICKERS["emerging"]
    ))
    total_crypto = len(DEFAULT_TICKERS["crypto"])
    print(f"📊 Available tickers:")
    print(f"   Stocks: {total_stocks}")
    print(f"   Crypto: {total_crypto}")
    print(f"   New power features: 11")
    print()

    def cb(c, t, tk):
        print(f"  [{c}/{t}] {tk}")

    result = build_backfill_dataset(
        tickers=["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        months_back=36,
        snapshot_interval_days=14,
        progress_callback=cb,
    )

    if "error" in result:
        print(f"\n❌ {result['error']}")
    else:
        print(f"\n✅ Generated {result['n_rows']} rows")
        print(f"   Tickers: {result['n_tickers']}")
        print(f"   Snapshots: {result['n_snapshots']}")
        print(f"   Date range: {result['date_range']}")

        df = result["df"]
        print(f"\n🔥 New Phase 1B features in dataset:")
        new_features = [
            "momentum_3m", "momentum_6m", "momentum_12m",
            "momentum_acceleration", "volatility_regime", "volume_momentum",
            "drawdown_depth", "max_drawdown_1y", "recovery_strength",
            "trend_consistency", "price_position_52w"
        ]
        for f in new_features:
            if f in df.columns:
                print(f"   ✅ {f}: mean={df[f].mean():.2f}, std={df[f].std():.2f}")

        print("\nForward returns coverage:")
        for h in HORIZONS:
            col = f"future_return_{h}d"
            if col in df.columns:
                valid = df[col].notna().sum()
                print(f"  {h}d: {valid}/{len(df)} valid")
