"""
chart_patterns.py - Visuel chart-analyse + pattern-detektion
=============================================================
Detekterer almindelige tekniske mønstre og visualiserer dem på charten:
  - 🥤 Cup and Handle
  - 👤 Head and Shoulders (og inverse)
  - 🔄 Double Top / Double Bottom
  - 📐 Triangles (ascending / descending / symmetrical)
  - 🚩 Flags / Pennants
  - 📈 Trend channels
  - 🛡️ Support / Resistance
  - 💥 Breakouts

Public API:
  - detect_patterns(df) → list[Pattern]
  - render_action_plan_chart(df, price, targets, patterns, ...)
  - render_pattern_explanations(patterns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Pattern:
    """Et detekteret mønster på charten."""
    name: str                    # fx "Cup and Handle"
    emoji: str                   # fx "🥤"
    direction: str               # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float            # 0-100
    start_idx: int               # index i df hvor mønstret starter
    end_idx: int                 # index hvor det slutter
    description: str             # kort beskrivelse
    explanation: str             # længere forklaring til brugeren
    target_price: Optional[float] = None  # estimeret target
    key_levels: dict = field(default_factory=dict)  # {"resistance": 100, "support": 90}


# ============================================================
# PEAK/VALLEY DETECTION
# ============================================================

def _find_peaks_valleys(prices: np.ndarray, distance: int = 10, prominence_pct: float = 2.0):
    """Find lokale toppe og bunde i en pris-serie."""
    if len(prices) < distance * 2:
        return [], []

    avg_price = np.mean(prices)
    prominence = avg_price * prominence_pct / 100

    peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-prices, distance=distance, prominence=prominence)

    return peaks.tolist(), valleys.tolist()


# ============================================================
# PATTERN DETECTORS
# ============================================================

def detect_trend(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter overordnet trend via linear regression."""
    if len(df) < 30:
        return None

    prices = df["Close"].tail(60).values
    x = np.arange(len(prices))

    try:
        slope, intercept = np.polyfit(x, prices, 1)
        avg_price = np.mean(prices)
        slope_pct = (slope / avg_price) * 100  # % per dag

        # Trend strength
        if slope_pct > 0.15:
            direction = "BULLISH"
            emoji = "📈"
            name = "Stærk Optrend"
            desc = f"Stigende {slope_pct*30:.1f}% pr måned i gennemsnit"
            conf = min(85, 50 + abs(slope_pct) * 100)
        elif slope_pct > 0.05:
            direction = "BULLISH"
            emoji = "↗️"
            name = "Mild Optrend"
            desc = f"Lidt stigende ({slope_pct*30:.1f}% pr måned)"
            conf = 60
        elif slope_pct < -0.15:
            direction = "BEARISH"
            emoji = "📉"
            name = "Stærk Nedtrend"
            desc = f"Faldende {abs(slope_pct)*30:.1f}% pr måned"
            conf = min(85, 50 + abs(slope_pct) * 100)
        elif slope_pct < -0.05:
            direction = "BEARISH"
            emoji = "↘️"
            name = "Mild Nedtrend"
            desc = f"Lidt faldende ({slope_pct*30:.1f}% pr måned)"
            conf = 60
        else:
            direction = "NEUTRAL"
            emoji = "➡️"
            name = "Sidelæns Trend"
            desc = "Konsoliderer (ingen klar retning)"
            conf = 50

        explanation = {
            "BULLISH": (
                f"Aktien har været i en {name.lower()} de seneste 60 dage. "
                "Det betyder, at køberne har overtaget — stigende efterspørgsel presser "
                "prisen op. **Strategi:** Buy-the-dip kan være effektivt indtil trenden brydes. "
                "Stop-loss bør placeres under sidste swing-low."
            ),
            "BEARISH": (
                f"Aktien har været i en {name.lower()} de seneste 60 dage. "
                "Sælgerne dominerer — udbuddet er større end efterspørgslen. "
                "**Strategi:** Undgå at 'fange faldende knive'. Vent på klare omslagstegn "
                "(højere lav + højere høj) før du går ind."
            ),
            "NEUTRAL": (
                "Aktien handler i et range uden klar retning. Det kan være konsolidering "
                "før et nyt move (op eller ned). **Strategi:** Vent på breakout med volume, "
                "eller trade rangen (køb ved bunden, sælg ved toppen)."
            ),
        }[direction]

        return Pattern(
            name=name, emoji=emoji, direction=direction,
            confidence=conf, start_idx=len(df) - 60, end_idx=len(df) - 1,
            description=desc, explanation=explanation,
            key_levels={"slope_pct_per_day": slope_pct},
        )
    except Exception:
        return None


def detect_support_resistance(df: pd.DataFrame, window: int = 90) -> dict:
    """Find vigtige support- og resistance-niveauer."""
    if len(df) < window:
        return {}

    recent = df.tail(window)
    prices = recent["Close"].values
    highs = recent["High"].values
    lows = recent["Low"].values

    peaks, valleys = _find_peaks_valleys(prices, distance=5, prominence_pct=1.5)

    # Resistance = peaks (clusters)
    peak_prices = [highs[p] for p in peaks] if peaks else []
    valley_prices = [lows[v] for v in valleys] if valleys else []

    # Cluster nærliggende niveauer (±1%)
    def cluster_levels(levels):
        if not levels:
            return []
        levels_sorted = sorted(levels)
        clusters = [[levels_sorted[0]]]
        for lvl in levels_sorted[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] < 0.02:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # Returnér gennemsnit af hver cluster + antal touches
        return [(np.mean(c), len(c)) for c in clusters]

    resistances = cluster_levels(peak_prices)
    supports = cluster_levels(valley_prices)

    # Sortér efter "styrke" (antal touches)
    resistances = sorted(resistances, key=lambda x: -x[1])[:3]
    supports = sorted(supports, key=lambda x: -x[1])[:3]

    return {
        "resistances": resistances,
        "supports": supports,
    }


def detect_double_top_bottom(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter Double Top eller Double Bottom."""
    if len(df) < 40:
        return None

    recent = df.tail(80)
    prices = recent["Close"].values

    peaks, valleys = _find_peaks_valleys(prices, distance=8, prominence_pct=2.5)

    # === DOUBLE TOP === (bearish reversal)
    if len(peaks) >= 2:
        # Tag de 2 seneste peaks
        last_two = sorted(peaks[-2:])
        p1, p2 = last_two
        # Skal være rimeligt tæt på hinanden i pris (±3%)
        if abs(prices[p1] - prices[p2]) / prices[p1] < 0.04:
            # Skal have en valley imellem
            valleys_between = [v for v in valleys if p1 < v < p2]
            if valleys_between:
                trough = max(valleys_between, key=lambda v: prices[v])
                drop_pct = (prices[p1] - prices[trough]) / prices[p1] * 100
                if drop_pct > 3:
                    target = prices[trough] - (prices[p1] - prices[trough])  # measured move
                    return Pattern(
                        name="Double Top",
                        emoji="⛰️⛰️",
                        direction="BEARISH",
                        confidence=70,
                        start_idx=len(df) - 80 + p1,
                        end_idx=len(df) - 80 + p2,
                        description=f"2 toppe ved ~{prices[p1]:.2f}, neckline ved {prices[trough]:.2f}",
                        explanation=(
                            "🔻 **Double Top** er et klassisk bearish reversal-mønster. "
                            "Prisen testede den samme resistance 2 gange uden at bryde igennem, "
                            "hvilket betyder at sælgerne overtager. **Trigger:** Hvis prisen "
                            f"bryder under neckline ({prices[trough]:.2f}), forventes et fald "
                            f"til ca. **{target:.2f}** (measured move). "
                            "**Strategi:** Sælg eller short hvis breakdown bekræftes med volume."
                        ),
                        target_price=target,
                        key_levels={
                            "resistance": float(prices[p1]),
                            "neckline": float(prices[trough]),
                            "target": float(target),
                        },
                    )

    # === DOUBLE BOTTOM === (bullish reversal)
    if len(valleys) >= 2:
        last_two = sorted(valleys[-2:])
        v1, v2 = last_two
        if abs(prices[v1] - prices[v2]) / prices[v1] < 0.04:
            peaks_between = [p for p in peaks if v1 < p < v2]
            if peaks_between:
                peak = max(peaks_between, key=lambda p: prices[p])
                rise_pct = (prices[peak] - prices[v1]) / prices[v1] * 100
                if rise_pct > 3:
                    target = prices[peak] + (prices[peak] - prices[v1])
                    return Pattern(
                        name="Double Bottom",
                        emoji="🍶🍶",
                        direction="BULLISH",
                        confidence=70,
                        start_idx=len(df) - 80 + v1,
                        end_idx=len(df) - 80 + v2,
                        description=f"2 bunde ved ~{prices[v1]:.2f}, neckline ved {prices[peak]:.2f}",
                        explanation=(
                            "🟢 **Double Bottom** er et klassisk bullish reversal-mønster. "
                            "Prisen testede den samme support 2 gange uden at bryde igennem nedad, "
                            "hvilket betyder at køberne overtager. **Trigger:** Hvis prisen bryder "
                            f"over neckline ({prices[peak]:.2f}), forventes en stigning til ca. "
                            f"**{target:.2f}** (measured move). "
                            "**Strategi:** Køb hvis breakout bekræftes med stigende volume."
                        ),
                        target_price=target,
                        key_levels={
                            "support": float(prices[v1]),
                            "neckline": float(prices[peak]),
                            "target": float(target),
                        },
                    )

    return None


def detect_head_shoulders(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter Head and Shoulders (eller inverse)."""
    if len(df) < 60:
        return None

    recent = df.tail(120)
    prices = recent["Close"].values

    peaks, valleys = _find_peaks_valleys(prices, distance=6, prominence_pct=2.0)

    # === HEAD AND SHOULDERS (bearish) ===
    if len(peaks) >= 3:
        last_three = sorted(peaks[-3:])
        ls, h, rs = last_three
        # Head skal være højest
        if prices[h] > prices[ls] and prices[h] > prices[rs]:
            # Shoulders skal være rimeligt symmetriske (±5%)
            if abs(prices[ls] - prices[rs]) / prices[ls] < 0.06:
                # Find neckline (laveste valley mellem ls-h og h-rs)
                v_left = [v for v in valleys if ls < v < h]
                v_right = [v for v in valleys if h < v < rs]
                if v_left and v_right:
                    neckline = (prices[v_left[-1]] + prices[v_right[0]]) / 2
                    head_height = prices[h] - neckline
                    target = neckline - head_height  # measured move down
                    return Pattern(
                        name="Head and Shoulders",
                        emoji="👤",
                        direction="BEARISH",
                        confidence=75,
                        start_idx=len(df) - 120 + ls,
                        end_idx=len(df) - 120 + rs,
                        description=f"Skuldre ~{prices[ls]:.2f}, hoved {prices[h]:.2f}, neckline {neckline:.2f}",
                        explanation=(
                            "🔻 **Head and Shoulders** er et af de mest pålidelige bearish "
                            "reversal-mønstre. 3 toppe hvor det midterste (hovedet) er højest, "
                            "med to lavere skuldre på siderne. **Trigger:** Når prisen bryder "
                            f"under neckline ({neckline:.2f}), forventes et fald til ca. "
                            f"**{target:.2f}**. **Strategi:** Sælg ved breakdown med stop-loss "
                            f"over højre skulder ({prices[rs]:.2f})."
                        ),
                        target_price=target,
                        key_levels={
                            "left_shoulder": float(prices[ls]),
                            "head": float(prices[h]),
                            "right_shoulder": float(prices[rs]),
                            "neckline": float(neckline),
                            "target": float(target),
                        },
                    )

    # === INVERSE HEAD AND SHOULDERS (bullish) ===
    if len(valleys) >= 3:
        last_three = sorted(valleys[-3:])
        ls, h, rs = last_three
        if prices[h] < prices[ls] and prices[h] < prices[rs]:
            if abs(prices[ls] - prices[rs]) / prices[ls] < 0.06:
                p_left = [p for p in peaks if ls < p < h]
                p_right = [p for p in peaks if h < p < rs]
                if p_left and p_right:
                    neckline = (prices[p_left[-1]] + prices[p_right[0]]) / 2
                    head_depth = neckline - prices[h]
                    target = neckline + head_depth
                    return Pattern(
                        name="Inverse Head and Shoulders",
                        emoji="🙃",
                        direction="BULLISH",
                        confidence=75,
                        start_idx=len(df) - 120 + ls,
                        end_idx=len(df) - 120 + rs,
                        description=f"Inverteret H&S, neckline ved {neckline:.2f}",
                        explanation=(
                            "🟢 **Inverse Head and Shoulders** er det bullish modstykke. "
                            "Mønstret signalerer slutningen på en nedtrend. **Trigger:** "
                            f"Når prisen bryder over neckline ({neckline:.2f}), forventes en "
                            f"stigning til ca. **{target:.2f}**. **Strategi:** Køb ved breakout "
                            "med volume — stop-loss under højre skulder."
                        ),
                        target_price=target,
                        key_levels={
                            "left_shoulder": float(prices[ls]),
                            "head": float(prices[h]),
                            "right_shoulder": float(prices[rs]),
                            "neckline": float(neckline),
                            "target": float(target),
                        },
                    )

    return None


def detect_cup_and_handle(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter Cup and Handle (bullish continuation)."""
    if len(df) < 60:
        return None

    recent = df.tail(150)
    if len(recent) < 50:
        return None

    prices = recent["Close"].values
    n = len(prices)

    # En cup kræver:
    # - Et peak nær starten
    # - Et nedture (cup bottom) i midten
    # - Et nyt peak nær starts-niveau
    # - Et lille pullback (handle)

    peaks, valleys = _find_peaks_valleys(prices, distance=10, prominence_pct=3.0)

    if len(peaks) < 2 or len(valleys) < 1:
        return None

    # Tag det første og det sidste store peak i den seneste 60% af perioden
    relevant_peaks = [p for p in peaks if p > n * 0.1 and p < n * 0.95]
    relevant_valleys = [v for v in valleys if v > n * 0.15 and v < n * 0.85]

    if len(relevant_peaks) < 2 or not relevant_valleys:
        return None

    left_rim = relevant_peaks[0]
    right_rim = relevant_peaks[-1]

    # Skal være mindst 30 dage mellem rims
    if right_rim - left_rim < 25:
        return None

    # Find dybeste valley mellem rims
    valleys_in_cup = [v for v in relevant_valleys if left_rim < v < right_rim]
    if not valleys_in_cup:
        return None

    cup_bottom = min(valleys_in_cup, key=lambda v: prices[v])

    # Rims skal være på rimeligt samme højde (±5%)
    if abs(prices[left_rim] - prices[right_rim]) / prices[left_rim] > 0.07:
        return None

    # Cup-dybde skal være mindst 12% af rim-prisen
    cup_depth = (prices[left_rim] - prices[cup_bottom]) / prices[left_rim] * 100
    if cup_depth < 12 or cup_depth > 50:
        return None

    # Tjek for handle (lille pullback efter højre rim)
    after_right = prices[right_rim:]
    if len(after_right) < 5:
        return None

    handle_low = float(np.min(after_right))
    handle_pullback_pct = (prices[right_rim] - handle_low) / prices[right_rim] * 100

    # Handle bør være lille (5-15%)
    if handle_pullback_pct < 2 or handle_pullback_pct > 20:
        return None

    target = prices[right_rim] + (prices[right_rim] - prices[cup_bottom])  # measured move

    return Pattern(
        name="Cup and Handle",
        emoji="☕",
        direction="BULLISH",
        confidence=72,
        start_idx=len(df) - n + left_rim,
        end_idx=len(df) - 1,
        description=f"Cup-bund: {prices[cup_bottom]:.2f}, rim: {prices[right_rim]:.2f}",
        explanation=(
            "🟢 **Cup and Handle** er et klassisk bullish continuation-mønster, gjort berømt "
            "af William O'Neil. En afrundet 'kop' (gradvis nedtur og opture) efterfulgt af et "
            "lille 'håndtag' (let pullback). **Hvad det betyder:** Aktien har konsolideret efter "
            "en stigning og er klar til næste leg up. **Trigger:** Breakout over rim-niveauet "
            f"({prices[right_rim]:.2f}) med stærk volume. **Target:** Cup-dybden adderet til "
            f"breakout-prisen ≈ **{target:.2f}**. **Strategi:** Køb ved breakout, stop-loss "
            f"under handle-low ({handle_low:.2f})."
        ),
        target_price=target,
        key_levels={
            "left_rim": float(prices[left_rim]),
            "cup_bottom": float(prices[cup_bottom]),
            "right_rim": float(prices[right_rim]),
            "handle_low": handle_low,
            "target": float(target),
        },
    )


def detect_triangle(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter Triangle-mønstre (ascending / descending / symmetrical)."""
    if len(df) < 40:
        return None

    recent = df.tail(60)
    prices = recent["Close"].values
    highs = recent["High"].values
    lows = recent["Low"].values

    peaks, valleys = _find_peaks_valleys(prices, distance=4, prominence_pct=1.5)

    if len(peaks) < 3 or len(valleys) < 3:
        return None

    # Trendlines via linear regression
    peak_x = np.array(peaks[-4:]) if len(peaks) >= 4 else np.array(peaks)
    valley_x = np.array(valleys[-4:]) if len(valleys) >= 4 else np.array(valleys)
    peak_y = highs[peak_x]
    valley_y = lows[valley_x]

    if len(peak_x) < 2 or len(valley_x) < 2:
        return None

    try:
        upper_slope, _ = np.polyfit(peak_x, peak_y, 1)
        lower_slope, _ = np.polyfit(valley_x, valley_y, 1)
    except Exception:
        return None

    avg_price = np.mean(prices)
    upper_pct = upper_slope / avg_price * 100
    lower_pct = lower_slope / avg_price * 100

    # Klassificering
    flat_threshold = 0.05  # %/dag

    if abs(upper_pct) < flat_threshold and lower_pct > flat_threshold:
        # Ascending triangle (bullish)
        return Pattern(
            name="Ascending Triangle",
            emoji="📐",
            direction="BULLISH",
            confidence=68,
            start_idx=len(df) - 60,
            end_idx=len(df) - 1,
            description="Flad resistance + stigende support",
            explanation=(
                "🟢 **Ascending Triangle** er et bullish continuation-mønster. "
                "Prisen tester samme resistance flere gange (flad linje øverst), "
                "mens støtten gradvist stiger (køberne presser opad). **Trigger:** "
                "Breakout over resistance med volume → typisk fortsættelse op. "
                "**Strategi:** Køb ved breakout, stop-loss under sidste swing-low."
            ),
        )

    if upper_pct < -flat_threshold and abs(lower_pct) < flat_threshold:
        # Descending triangle (bearish)
        return Pattern(
            name="Descending Triangle",
            emoji="📐",
            direction="BEARISH",
            confidence=68,
            start_idx=len(df) - 60,
            end_idx=len(df) - 1,
            description="Faldende resistance + flad support",
            explanation=(
                "🔻 **Descending Triangle** er et bearish continuation-mønster. "
                "Sælgerne presser prisen lavere (faldende toppe) mens støtten holder "
                "konstant. **Trigger:** Breakdown under support med volume → fortsat fald. "
                "**Strategi:** Sælg eller short ved breakdown, stop-loss over sidste swing-high."
            ),
        )

    if upper_pct < -flat_threshold and lower_pct > flat_threshold:
        # Symmetrical triangle (neutral - breakout direction afgør)
        return Pattern(
            name="Symmetrical Triangle",
            emoji="🔺",
            direction="NEUTRAL",
            confidence=60,
            start_idx=len(df) - 60,
            end_idx=len(df) - 1,
            description="Konvergerende trendlinjer (lavere højder + højere lave)",
            explanation=(
                "⚖️ **Symmetrical Triangle** er et konsolideringsmønster — markedet er i "
                "balance og bygger energi op. Retningen afgøres af breakout. "
                "**Strategi:** Vent på klar breakout (op eller ned) med volume. "
                "Det indikerer hvilken retning trade'en skal tages."
            ),
        )

    return None


def detect_breakout(df: pd.DataFrame, sr_levels: dict) -> Optional[Pattern]:
    """Detekter om prisen lige har brudt et vigtigt support/resistance-niveau."""
    if len(df) < 5 or not sr_levels:
        return None

    last_close = float(df["Close"].iloc[-1])
    last_5_low = float(df["Low"].tail(5).min())
    last_5_high = float(df["High"].tail(5).max())

    # Volume-bekræftelse
    if "Volume" in df.columns and len(df) >= 20:
        recent_vol = df["Volume"].tail(3).mean()
        avg_vol = df["Volume"].tail(20).mean()
        volume_surge = recent_vol > avg_vol * 1.3
    else:
        volume_surge = False

    resistances = sr_levels.get("resistances", [])
    supports = sr_levels.get("supports", [])

    # === BREAKOUT OP ===
    for resistance, touches in resistances:
        if last_5_low < resistance < last_close and touches >= 2:
            # Brud over resistance!
            vol_note = " med stigende volume ✅" if volume_surge else " (volume bekræfter ikke endnu ⚠️)"
            return Pattern(
                name="Resistance Breakout",
                emoji="💥",
                direction="BULLISH",
                confidence=75 if volume_surge else 60,
                start_idx=len(df) - 5,
                end_idx=len(df) - 1,
                description=f"Brud over {resistance:.2f} ({touches} test'er)",
                explanation=(
                    f"🚀 **Breakout!** Prisen har lige brudt over en vigtig resistance "
                    f"({resistance:.2f}) som har holdt {touches} gange{vol_note}. "
                    "Når flere måneders modstand brydes, går trapped sælgere ofte ud, hvilket "
                    "tilføjer momentum opad. **Strategi:** Køb retest af brud-niveauet "
                    f"(nu support ved {resistance:.2f}). Stop-loss under sidste swing-low."
                ),
                key_levels={"breakout_level": resistance, "new_support": resistance},
            )

    # === BREAKDOWN NED ===
    for support, touches in supports:
        if last_5_high > support > last_close and touches >= 2:
            return Pattern(
                name="Support Breakdown",
                emoji="💥",
                direction="BEARISH",
                confidence=70,
                start_idx=len(df) - 5,
                end_idx=len(df) - 1,
                description=f"Brud under {support:.2f} ({touches} test'er)",
                explanation=(
                    f"📉 **Breakdown!** Prisen har brudt under en vigtig support ({support:.2f}) "
                    f"som har holdt {touches} gange. Tidligere support bliver nu resistance — "
                    "trapped købere går ud, hvilket forværrer faldet. **Strategi:** Sælg / short "
                    f"med stop-loss over {support:.2f}. Vent på retest fra undersiden før entry."
                ),
                key_levels={"breakdown_level": support, "new_resistance": support},
            )

    return None


def detect_consolidation(df: pd.DataFrame) -> Optional[Pattern]:
    """Detekter om prisen er i en stram konsolidering (lav volatilitet)."""
    if len(df) < 30:
        return None

    recent = df.tail(20)
    prices = recent["Close"].values
    high = float(prices.max())
    low = float(prices.min())
    avg = float(prices.mean())

    range_pct = (high - low) / avg * 100

    if range_pct < 6:  # Meget tight range
        return Pattern(
            name="Tight Konsolidering",
            emoji="🤐",
            direction="NEUTRAL",
            confidence=65,
            start_idx=len(df) - 20,
            end_idx=len(df) - 1,
            description=f"20-dages range kun {range_pct:.1f}% ({low:.2f} - {high:.2f})",
            explanation=(
                f"⚖️ **Tight konsolidering!** Prisen har handlet i et meget snævert range "
                f"({range_pct:.1f}%) de seneste 20 dage. Det er typisk 'roen før stormen' — "
                "volatiliteten skal komme tilbage, ofte med et eksplosivt move. "
                "**Strategi:** Vent på breakout enten over **{high:.2f}** (køb-signal) eller "
                "under **{low:.2f}** (sælg-signal). Ofte med stort volume."
            ),
            key_levels={"upper": high, "lower": low, "range_pct": range_pct},
        )
    return None


# ============================================================
# MAIN DETECT FUNCTION
# ============================================================

def detect_patterns(df: pd.DataFrame) -> list[Pattern]:
    """Kør alle pattern-detektorer og returnér en liste af fundne mønstre."""
    if df is None or df.empty or len(df) < 30:
        return []

    patterns = []

    # 1. Trend (altid)
    trend = detect_trend(df)
    if trend:
        patterns.append(trend)

    # 2. Support/Resistance levels (bruges af breakout)
    sr_levels = detect_support_resistance(df)

    # 3. Specifikke mønstre
    for detector in [
        detect_cup_and_handle,
        detect_head_shoulders,
        detect_double_top_bottom,
        detect_triangle,
        detect_consolidation,
    ]:
        try:
            p = detector(df)
            if p:
                patterns.append(p)
        except Exception as e:
            print(f"[chart_patterns] {detector.__name__} fejlede: {e}")

    # 4. Breakout (efter S/R er fundet)
    try:
        breakout = detect_breakout(df, sr_levels)
        if breakout:
            patterns.append(breakout)
    except Exception:
        pass

    # Gem S/R som "meta-pattern" til chart-rendering
    if sr_levels.get("resistances") or sr_levels.get("supports"):
        patterns.append(Pattern(
            name="Support/Resistance",
            emoji="🛡️",
            direction="NEUTRAL",
            confidence=80,
            start_idx=0, end_idx=len(df) - 1,
            description="Vigtige niveauer fra historikken",
            explanation="",
            key_levels=sr_levels,
        ))

    return patterns


# ============================================================
# CHART RENDERING
# ============================================================

def render_action_plan_chart(
    df: pd.DataFrame,
    current_price: float,
    targets: Optional[dict] = None,
    patterns: Optional[list[Pattern]] = None,
    currency: str = "USD",
    title: str = "Visuel Handlingsplan",
):
    """
    Tegn en samlet chart med:
      - Pris (candlesticks eller linje)
      - SMA50, SMA200
      - KØB-zone (grøn band)
      - Stop-loss linje (rød)
      - Targets (kort/lang)
      - Detekterede mønstre med annotationer
    """
    if df is None or df.empty:
        st.warning("Ingen data til chart")
        return

    # Begræns til seneste 6 måneder for læsbarhed
    df_view = df.tail(180).copy()

    fig = go.Figure()

    # ===== Candlestick =====
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view["Open"],
        high=df_view["High"],
        low=df_view["Low"],
        close=df_view["Close"],
        name="Pris",
        increasing_line_color="#16a34a",
        decreasing_line_color="#ef4444",
    ))

    # ===== SMA-overlays =====
    if "SMA50" in df_view.columns:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["SMA50"],
            name="SMA50", line=dict(color="orange", width=1.5),
            opacity=0.7,
        ))
    if "SMA200" in df_view.columns:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["SMA200"],
            name="SMA200", line=dict(color="purple", width=1.5),
            opacity=0.7,
        ))

    # ===== KØB-zone (grøn band) =====
    if targets and targets.get("buy_low") and targets.get("buy_high"):
        fig.add_hrect(
            y0=targets["buy_low"], y1=targets["buy_high"],
            fillcolor="#16a34a", opacity=0.15, line_width=0,
            annotation_text=f"🟢 KØB ZONE ({targets['buy_low']:.2f} - {targets['buy_high']:.2f})",
            annotation_position="top left",
            annotation_font_color="#16a34a",
            annotation_font_size=11,
        )

    # ===== Stop-loss =====
    if targets and targets.get("stop_loss"):
        fig.add_hline(
            y=targets["stop_loss"], line_dash="dash", line_color="#ef4444",
            line_width=2,
            annotation_text=f"🛑 STOP {targets['stop_loss']:.2f}",
            annotation_position="bottom right",
            annotation_font_color="#ef4444",
            annotation_font_size=11,
        )

    # ===== Targets =====
    if targets and targets.get("target_short"):
        fig.add_hline(
            y=targets["target_short"], line_dash="dot", line_color="#eab308",
            line_width=2,
            annotation_text=f"🎯 KORT {targets['target_short']:.2f}",
            annotation_position="top right",
            annotation_font_color="#eab308",
            annotation_font_size=11,
        )
    if targets and targets.get("target_long"):
        fig.add_hline(
            y=targets["target_long"], line_dash="dot", line_color="#22c55e",
            line_width=2,
            annotation_text=f"🚀 LANG {targets['target_long']:.2f}",
            annotation_position="top right",
            annotation_font_color="#22c55e",
            annotation_font_size=11,
        )

    # ===== Nuværende pris linje =====
    fig.add_hline(
        y=current_price, line_dash="solid", line_color="#0099ff",
        line_width=1, opacity=0.5,
        annotation_text=f"📍 {current_price:.2f}",
        annotation_position="bottom left",
        annotation_font_color="#0099ff",
    )

    # ===== Pattern key levels =====
    if patterns:
        for p in patterns:
            kl = p.key_levels or {}

            # Support/resistance levels
            if p.name == "Support/Resistance":
                for r_price, touches in kl.get("resistances", []):
                    fig.add_hline(
                        y=r_price, line_dash="dot",
                        line_color="rgba(239,68,68,0.4)", line_width=1,
                        annotation_text=f"R ({touches}x)",
                        annotation_position="right",
                        annotation_font_size=9,
                    )
                for s_price, touches in kl.get("supports", []):
                    fig.add_hline(
                        y=s_price, line_dash="dot",
                        line_color="rgba(34,197,94,0.4)", line_width=1,
                        annotation_text=f"S ({touches}x)",
                        annotation_position="right",
                        annotation_font_size=9,
                    )

            # Pattern target (hvis ikke allerede dækket)
            if p.target_price and p.name not in ["Support/Resistance"]:
                fig.add_hline(
                    y=p.target_price, line_dash="longdash",
                    line_color="#a855f7", line_width=1.5, opacity=0.7,
                    annotation_text=f"{p.emoji} target {p.target_price:.2f}",
                    annotation_position="left",
                    annotation_font_color="#a855f7",
                    annotation_font_size=10,
                )

    # ===== Layout =====
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=550,
        xaxis_rangeslider_visible=False,
        yaxis_title=f"Pris ({currency})",
        xaxis_title="Dato",
        showlegend=True,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PATTERN EXPLANATIONS UI
# ============================================================

def render_pattern_explanations(patterns: list[Pattern]):
    """Vis liste af detekterede mønstre med forklaringer."""
    if not patterns:
        st.info(
            "📊 Ingen klare mønstre detekteret. Det kan betyde at aktien handler "
            "i en uklar/sidelæns fase. Vent på et klarere setup."
        )
        return

    # Filter: kun rigtige mønstre, ikke S/R-meta
    real_patterns = [p for p in patterns if p.name != "Support/Resistance" and p.explanation]

    if not real_patterns:
        st.info("📊 Ingen specifikke mønstre detekteret — kun support/resistance niveauer.")
        return

    # Sortér: bullish først, så bearish, så neutral, og inden for hver gruppe efter confidence
    direction_order = {"BULLISH": 0, "BEARISH": 1, "NEUTRAL": 2}
    real_patterns.sort(key=lambda p: (direction_order.get(p.direction, 3), -p.confidence))

    # Top-level oversigt
    bull_count = sum(1 for p in real_patterns if p.direction == "BULLISH")
    bear_count = sum(1 for p in real_patterns if p.direction == "BEARISH")
    neutral_count = sum(1 for p in real_patterns if p.direction == "NEUTRAL")

    overview_cols = st.columns(3)
    overview_cols[0].metric("🟢 Bullish signaler", bull_count)
    overview_cols[1].metric("🔴 Bearish signaler", bear_count)
    overview_cols[2].metric("🟡 Neutrale", neutral_count)

    # Konklusion
    if bull_count > bear_count + 1:
        st.success(
            f"✅ **Samlet bias: BULLISH** ({bull_count} bullish vs {bear_count} bearish mønstre). "
            "De tekniske signaler peger overvejende op."
        )
    elif bear_count > bull_count + 1:
        st.error(
            f"🚨 **Samlet bias: BEARISH** ({bear_count} bearish vs {bull_count} bullish mønstre). "
            "De tekniske signaler peger overvejende ned."
        )
    else:
        st.info(
            f"⚖️ **Samlet bias: NEUTRAL** (mønstre peger i forskellige retninger). "
            "Vent på klarere setup eller læg vægt på fundamental analyse."
        )

    st.markdown("---")
    st.markdown("#### 📖 Detaljeret pattern-forklaring")

    # Vis hvert mønster
    for i, p in enumerate(real_patterns, 1):
        if p.direction == "BULLISH":
            color = "#16a34a"
            bg = "#16a34a15"
        elif p.direction == "BEARISH":
            color = "#ef4444"
            bg = "#ef444415"
        else:
            color = "#eab308"
            bg = "#eab30815"

        with st.expander(
            f"{p.emoji} **{p.name}** — {p.direction} ({p.confidence:.0f}% conf.)",
            expanded=(i == 1),  # Først er åben
        ):
            st.markdown(
                f"<div style='background:{bg};padding:1rem;border-radius:8px;"
                f"border-left:4px solid {color};margin:0.5rem 0'>"
                f"<small style='color:#888'>📊 BESKRIVELSE</small><br>"
                f"<b>{p.description}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(p.explanation)

            if p.key_levels:
                st.markdown("**🔑 Nøgleniveauer:**")
                level_data = []
                for k, v in p.key_levels.items():
                    if isinstance(v, (int, float)):
                        level_data.append({
                            "Niveau": k.replace("_", " ").title(),
                            "Pris": f"{v:.2f}",
                        })
                if level_data:
                    st.dataframe(pd.DataFrame(level_data), use_container_width=True, hide_index=True)


# ============================================================
# COMBINED VIEW (alt i én)
# ============================================================

def render_full_action_plan(
    df: pd.DataFrame,
    current_price: float,
    targets: Optional[dict] = None,
    currency: str = "USD",
    title_prefix: str = "",
):
    """
    Render hele pakken: chart + pattern-forklaringer.
    Kald denne fra app.py for at få det hele i én blok.
    """
    if df is None or df.empty:
        st.warning("Ingen prishistorik tilgængelig")
        return

    # 1. Detektér mønstre
    with st.spinner("🔍 Analyserer chart-mønstre..."):
        patterns = detect_patterns(df)

    # 2. Visuel chart
    st.markdown(f"### 📊 {title_prefix}Visuel Handlingsplan")
    st.caption(
        "Charten nedenfor viser alle vigtige niveauer fra handlingsplanen "
        "samt detekterede tekniske mønstre."
    )
    render_action_plan_chart(
        df, current_price, targets=targets,
        patterns=patterns, currency=currency,
        title=f"{title_prefix}Pris + handlingsplan + mønstre"
    )

    # 3. Pattern-forklaringer
    st.markdown("---")
    st.markdown("### 🔍 Hvad ser modellen i charten?")
    st.caption(
        "Algoritmen leder automatisk efter klassiske tekniske mønstre. "
        "Forklaringerne nedenfor hjælper dig med at forstå *hvorfor* mønstret betyder noget."
    )
    render_pattern_explanations(patterns)


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing chart_patterns...")

    # Lav fake data
    np.random.seed(42)
    n = 200
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n)
    prices = 100 + np.cumsum(np.random.randn(n) * 1.5)

    df = pd.DataFrame({
        "Open": prices + np.random.randn(n) * 0.5,
        "High": prices + np.abs(np.random.randn(n)) * 1,
        "Low": prices - np.abs(np.random.randn(n)) * 1,
        "Close": prices,
        "Volume": np.random.randint(1000, 5000, n),
    }, index=dates)

    patterns = detect_patterns(df)
    print(f"\n✅ Fundet {len(patterns)} mønstre:")
    for p in patterns:
        print(f"  {p.emoji} {p.name} ({p.direction}, {p.confidence}%)")
        print(f"     {p.description}")

    print("\n✅ Test passed!")
