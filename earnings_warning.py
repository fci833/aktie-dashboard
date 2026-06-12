"""
Earnings Warning System
=======================
- Henter næste earnings-dato
- Analyserer historisk earnings-volatilitet (post-earnings move)
- Tracker earnings surprises (beat/miss EPS)
- Render advarselsbannere baseret på dage-til-earnings

Krav:
- yfinance (allerede installeret)
- pandas, numpy

Cache: 1 time (earnings ændrer sig sjældent)
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ============ KONFIGURATION ============

WARNING_THRESHOLDS = {
    "critical": 3,    # < 3 dage = KRITISK ADVARSEL
    "high": 7,        # < 7 dage = HØJ ADVARSEL
    "medium": 14,     # < 14 dage = MEDIUM advarsel
    "low": 30,        # < 30 dage = info
}

# Threshold for "stor bevægelse" efter earnings
BIG_MOVE_THRESHOLD_PCT = 5.0  # ±5% = stor bevægelse


# ============ DATA-FETCHING ============

def _safe_to_datetime(val) -> Optional[datetime]:
    """Konverter forskellige date-formater til datetime"""
    if val is None or pd.isna(val):
        return None
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        if isinstance(val, datetime):
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val
        if isinstance(val, pd.Timestamp):
            if val.tz is None:
                return val.tz_localize("UTC").to_pydatetime()
            return val.to_pydatetime()
        # String
        return pd.to_datetime(val, utc=True).to_pydatetime()
    except Exception:
        return None


def _calc_days_until(target: Optional[datetime]) -> Optional[int]:
    """Beregn dage indtil target-dato (kan være negativ hvis fortid)"""
    if target is None:
        return None
    try:
        now = datetime.now(timezone.utc)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = target - now
        return delta.days
    except Exception:
        return None


def _fetch_next_earnings_date(ticker_obj) -> Optional[datetime]:
    """Hent næste earnings-dato fra yfinance (flere fallbacks)"""
    candidates = []

    # Method 1: calendar (mest pålidelig for fremtidige)
    try:
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict):
                # Nyere yfinance returnerer dict
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and ed:
                        candidates.append(_safe_to_datetime(ed[0]))
                    else:
                        candidates.append(_safe_to_datetime(ed))
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                # Ældre yfinance returnerer DataFrame
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    candidates.append(_safe_to_datetime(val))
    except Exception:
        pass

    # Method 2: get_earnings_dates() - newer API
    try:
        ed = ticker_obj.get_earnings_dates(limit=24)
        if ed is not None and not ed.empty:
            now = pd.Timestamp.now(tz="UTC")
            if ed.index.tz is None:
                ed.index = ed.index.tz_localize("UTC")
            future = ed[ed.index > now]
            if not future.empty:
                next_date = future.index.min()
                candidates.append(_safe_to_datetime(next_date))
    except Exception:
        pass

    # Method 3: earnings_dates property (fallback)
    try:
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            now = pd.Timestamp.now(tz="UTC")
            if ed.index.tz is None:
                ed.index = ed.index.tz_localize("UTC")
            future = ed[ed.index > now]
            if not future.empty:
                next_date = future.index.min()
                candidates.append(_safe_to_datetime(next_date))
    except Exception:
        pass

    # Method 4: info dict
    try:
        info = ticker_obj.info
        for key in ["earningsTimestamp", "earningsDate"]:
            val = info.get(key)
            if val:
                if isinstance(val, list) and val:
                    val = val[0]
                candidates.append(_safe_to_datetime(val))
    except Exception:
        pass

    # Filter til kun fremtidige
    now = datetime.now(timezone.utc)
    future_dates = [d for d in candidates if d is not None and d > now]

    if future_dates:
        return min(future_dates)  # tidligste fremtidige

    # Hvis ingen fremtidige, returnér tidligste (kunne være "i dag/lige rundt")
    valid = [d for d in candidates if d is not None]
    if valid:
        return min(valid, key=lambda d: abs((d - now).total_seconds()))

    return None


def _fetch_earnings_history(ticker_obj, n_quarters: int = 8) -> List[Dict]:
    """
    Hent historisk earnings (EPS estimat vs faktisk + surprise).
    Robust version med flere fallbacks for forskellige yfinance versioner.
    """
    history = []

    # Method 1: get_earnings_dates() - newer yfinance API (anbefalet)
    ed = None
    try:
        ed = ticker_obj.get_earnings_dates(limit=n_quarters * 2)
        if ed is not None and not ed.empty:
            print(f"[earnings] ✓ get_earnings_dates() returnerede {len(ed)} rows")
    except Exception as e:
        print(f"[earnings] get_earnings_dates() fejl: {e}")

    # Method 2: Fallback til earnings_dates property (ældre yfinance)
    if ed is None or (hasattr(ed, 'empty') and ed.empty):
        try:
            ed = ticker_obj.earnings_dates
            if ed is not None and not ed.empty:
                print(f"[earnings] ✓ earnings_dates property returnerede {len(ed)} rows")
        except Exception as e:
            print(f"[earnings] earnings_dates fejl: {e}")

    if ed is None or (hasattr(ed, 'empty') and ed.empty):
        print(f"[earnings] ⚠️ Ingen earnings-historik fundet")
        return []

    try:
        # Print debug-info om kolonnerne
        print(f"[earnings] Kolonner: {list(ed.columns)}")

        now = pd.Timestamp.now(tz="UTC")

        # Sørg for at index er tz-aware
        if ed.index.tz is None:
            ed.index = ed.index.tz_localize("UTC")

        # Filter: kun fortid (allerede rapporterede)
        past = ed[ed.index <= now].head(n_quarters)

        if past.empty:
            print(f"[earnings] ⚠️ Alle earnings er fremtidige - ingen historik endnu")
            return []

        for date_idx, row in past.iterrows():
            entry = {
                "date": _safe_to_datetime(date_idx),
                "eps_estimate": None,
                "eps_actual": None,
                "surprise_pct": None,
                "beat": None,
            }

            # Find EPS estimate (forskellige column-navne i versioner)
            for est_col in [
                "EPS Estimate", "Estimate", "epsEstimate", "estimate",
                "EPS_Estimate", "eps_estimate"
            ]:
                if est_col in row.index:
                    val = row.get(est_col)
                    if val is not None and not pd.isna(val):
                        try:
                            entry["eps_estimate"] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

            # Find faktisk EPS
            for act_col in [
                "Reported EPS", "Actual", "epsActual", "actual",
                "reportedEPS", "Reported_EPS", "eps_actual"
            ]:
                if act_col in row.index:
                    val = row.get(act_col)
                    if val is not None and not pd.isna(val):
                        try:
                            entry["eps_actual"] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

            # Find surprise % (direkte fra API)
            for surp_col in [
                "Surprise(%)", "Surprise %", "surprisePercent",
                "surprise(%)", "Surprise", "surprise_pct"
            ]:
                if surp_col in row.index:
                    val = row.get(surp_col)
                    if val is not None and not pd.isna(val):
                        try:
                            entry["surprise_pct"] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

            # Hvis surprise mangler men vi har estimate + actual → beregn selv
            if (entry["eps_estimate"] is not None and
                    entry["eps_actual"] is not None and
                    entry["surprise_pct"] is None):
                if entry["eps_estimate"] != 0:
                    entry["surprise_pct"] = (
                        (entry["eps_actual"] - entry["eps_estimate"]) /
                        abs(entry["eps_estimate"]) * 100
                    )

            # Beregn beat/miss
            if entry["eps_estimate"] is not None and entry["eps_actual"] is not None:
                entry["beat"] = entry["eps_actual"] >= entry["eps_estimate"]

            # Tilføj kun hvis vi minimum har en dato
            if entry["date"] is not None:
                history.append(entry)

        print(f"[earnings] ✓ Parsed {len(history)} earnings entries")

    except Exception as e:
        print(f"[earnings] Historik parsing fejl: {e}")
        import traceback
        traceback.print_exc()

    return history


def _calc_post_earnings_volatility(
    hist: pd.DataFrame, earnings_dates: List[datetime]
) -> Optional[Dict]:
    """
    Beregn historisk bevægelse på dagen efter earnings.
    Returnerer median, gennemsnit, max op/ned, % store moves.
    """
    if hist is None or hist.empty or not earnings_dates:
        return None

    moves = []
    abs_moves = []

    for ed in earnings_dates:
        if ed is None:
            continue
        try:
            # Find close-dagen før og dagen efter earnings
            ed_naive = ed.replace(tzinfo=None) if ed.tzinfo else ed
            # Hist index kan være tz-aware eller naive
            hist_idx = hist.index
            if hasattr(hist_idx, "tz") and hist_idx.tz is not None:
                hist_naive = hist.copy()
                hist_naive.index = hist_idx.tz_localize(None)
            else:
                hist_naive = hist

            # Find pris dagen før (eller tæt på)
            before_mask = hist_naive.index <= ed_naive
            after_mask = hist_naive.index > ed_naive
            if not before_mask.any() or not after_mask.any():
                continue

            price_before = float(hist_naive[before_mask]["Close"].iloc[-1])
            # Pris dagen efter (første handel efter earnings)
            price_after = float(hist_naive[after_mask]["Close"].iloc[0])

            if price_before > 0:
                pct_move = (price_after / price_before - 1) * 100
                moves.append(pct_move)
                abs_moves.append(abs(pct_move))
        except Exception:
            continue

    if not moves:
        return None

    moves_arr = np.array(moves)
    abs_arr = np.array(abs_moves)

    big_moves = (abs_arr >= BIG_MOVE_THRESHOLD_PCT).sum()
    big_move_pct = (big_moves / len(moves)) * 100 if moves else 0

    return {
        "n_observations": len(moves),
        "median_move_pct": float(np.median(moves)),
        "avg_abs_move_pct": float(np.mean(abs_arr)),
        "max_up_pct": float(max(moves)),
        "max_down_pct": float(min(moves)),
        "std_move_pct": float(np.std(moves)),
        "big_moves_count": int(big_moves),
        "big_moves_pct": float(big_move_pct),
        "all_moves": [float(m) for m in moves],
    }


# ============ HOVED-FUNKTION ============

@st.cache_data(ttl=3600, show_spinner=False)  # Cache 1 time
def get_earnings_info(ticker: str) -> Optional[Dict]:
    """
    Henter komplet earnings-info for en ticker.
    Robust version - returnerer altid noget hvis ticker er valid.
    """
    if not ticker or not YF_AVAILABLE:
        print(f"[earnings] Ticker mangler eller yfinance ikke installeret")
        return None

    try:
        tk = yf.Ticker(ticker)
        print(f"[earnings] Henter data for {ticker}...")

        # Næste earnings (hvis muligt)
        next_date = None
        try:
            next_date = _fetch_next_earnings_date(tk)
        except Exception as e:
            print(f"[earnings] _fetch_next_earnings_date fejl: {e}")

        days_until = _calc_days_until(next_date)

        # Warning level
        warning_level = "none"
        if days_until is not None and days_until >= 0:
            if days_until <= WARNING_THRESHOLDS["critical"]:
                warning_level = "critical"
            elif days_until <= WARNING_THRESHOLDS["high"]:
                warning_level = "high"
            elif days_until <= WARNING_THRESHOLDS["medium"]:
                warning_level = "medium"
            elif days_until <= WARNING_THRESHOLDS["low"]:
                warning_level = "low"

        # Historik (hvis muligt)
        history = []
        try:
            history = _fetch_earnings_history(tk, n_quarters=8)
        except Exception as e:
            print(f"[earnings] _fetch_earnings_history fejl: {e}")

        # Beat rate
        beats = [h for h in history if h.get("beat") is True]
        valid_history = [h for h in history if h.get("beat") is not None]
        beat_rate = (len(beats) / len(valid_history) * 100) if valid_history else None

        # Avg surprise
        surprises = [
            h["surprise_pct"] for h in history
            if h.get("surprise_pct") is not None
        ]
        avg_surprise = float(np.mean(surprises)) if surprises else None
        median_surprise = float(np.median(surprises)) if surprises else None

        # Post-earnings volatility (kræver hist data)
        volatility = None
        if history:
            try:
                hist_df = tk.history(period="3y", auto_adjust=False)
                if not hist_df.empty:
                    past_dates = [h["date"] for h in history if h.get("date")]
                    volatility = _calc_post_earnings_volatility(hist_df, past_dates)
            except Exception as e:
                print(f"[earnings] Volatility fejl: {e}")

        result = {
            "ticker": ticker,
            "next_date": next_date,
            "days_until": days_until,
            "warning_level": warning_level,
            "history": history,
            "beat_rate": beat_rate,
            "n_quarters": len(valid_history),
            "avg_surprise_pct": avg_surprise,
            "median_surprise_pct": median_surprise,
            "volatility": volatility,
            "fetched_at": datetime.now(),
        }

        print(f"[earnings] ✓ Result: next_date={next_date}, days={days_until}, "
              f"history={len(history)}, beat_rate={beat_rate}")

        return result

    except Exception as e:
        print(f"[earnings] ❌ KRITISK fejl for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        # Returnér tom data i stedet for None - så vi kan vise en message
        return {
            "ticker": ticker,
            "next_date": None,
            "days_until": None,
            "warning_level": "none",
            "history": [],
            "beat_rate": None,
            "n_quarters": 0,
            "avg_surprise_pct": None,
            "median_surprise_pct": None,
            "volatility": None,
            "fetched_at": datetime.now(),
            "error": str(e),
        }


# ============ HJÆLPE-FUNKTIONER ============

def _format_date_dk(dt: Optional[datetime]) -> str:
    """Format dato på dansk"""
    if dt is None:
        return "?"
    days_dk = ["Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag", "Lørdag", "Søndag"]
    months_dk = ["jan", "feb", "mar", "apr", "maj", "jun",
                 "jul", "aug", "sep", "okt", "nov", "dec"]
    try:
        return f"{days_dk[dt.weekday()]} {dt.day}. {months_dk[dt.month-1]} {dt.year}"
    except Exception:
        return dt.strftime("%Y-%m-%d")


def _warning_color(level: str) -> str:
    return {
        "critical": "#dc2626",
        "high": "#ef4444",
        "medium": "#eab308",
        "low": "#0099ff",
        "none": "#6b7280",
    }.get(level, "#6b7280")


def _warning_emoji(level: str) -> str:
    return {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "📅",
        "low": "ℹ️",
        "none": "✅",
    }.get(level, "")


def get_earnings_warning_message(data: Optional[Dict]) -> Optional[str]:
    """
    Returnerer kort tekst-warning til brug i action plan
    (uden at rendere noget).
    """
    if data is None or data.get("days_until") is None:
        return None

    days = data["days_until"]
    level = data["warning_level"]
    vol = data.get("volatility") or {}
    avg_move = vol.get("avg_abs_move_pct")

    if days < 0:
        return None  # Allerede passeret

    move_str = f" (historisk ±{avg_move:.1f}% post-earnings)" if avg_move else ""

    if level == "critical":
        return (
            f"🚨 **EARNINGS OM {days} DAGE!** Kursen kan svinge voldsomt"
            f"{move_str}. Overvej at vente med køb til efter earnings."
        )
    elif level == "high":
        return (
            f"⚠️ **EARNINGS om {days} dage**{move_str}. "
            f"Reducér evt. positionsstørrelse eller vent."
        )
    elif level == "medium":
        return (
            f"📅 Earnings om {days} dage{move_str}. Hold øje med udviklingen."
        )
    return None


# ============ UI RENDERERS ============

def render_earnings_warning(
    data: Optional[Dict],
    compact: bool = False,
) -> None:
    """
    Render earnings-warning banner.

    Args:
        data: Output fra get_earnings_info()
        compact: Hvis True, vis kompakt 4-kolonne layout
                 Hvis False, vis fuld banner
    """
    if data is None:
        # Vis fallback-besked så brugeren ved hvad der sker
        if not compact:
            st.info("📅 Kunne ikke hente earnings-data fra Yahoo Finance")
        else:
            st.markdown(
                "<div style='background:#6b728022;padding:0.6rem 1rem;border-radius:8px;"
                "border-left:4px solid #6b7280'>"
                "<small>📅 Earnings-data ikke tilgængelig for denne ticker</small>"
                "</div>",
                unsafe_allow_html=True
            )
        return

    days = data.get("days_until")
    next_date = data.get("next_date")
    level = data.get("warning_level", "none")

    if days is None or next_date is None:
        if not compact:
            st.info("📅 Ingen earnings-dato tilgængelig (yfinance returnerede ingen data)")
        else:
            st.markdown(
                "<div style='background:#6b728022;padding:0.6rem 1rem;border-radius:8px;"
                "border-left:4px solid #6b7280'>"
                "<small>📅 Ingen kommende earnings-dato fundet</small>"
                "</div>",
                unsafe_allow_html=True
            )
        return

    if days < 0:
        # Earnings allerede passeret
        if not compact:
            st.info(
                f"📅 Sidste earnings var {abs(days)} dage siden "
                f"({_format_date_dk(next_date)})"
            )
        return

    color = _warning_color(level)
    emoji = _warning_emoji(level)
    date_str = _format_date_dk(next_date)

    vol = data.get("volatility") or {}
    avg_abs = vol.get("avg_abs_move_pct")
    max_up = vol.get("max_up_pct")
    max_down = vol.get("max_down_pct")
    n_obs = vol.get("n_observations", 0)

    beat_rate = data.get("beat_rate")
    n_quarters = data.get("n_quarters", 0)
    avg_surp = data.get("avg_surprise_pct")

    if compact:
        # Tæl hvor mange felter der har data
        has_swing = avg_abs is not None
        has_beat = beat_rate is not None
        has_surp = avg_surp is not None

        n_data_fields = sum([has_swing, has_beat, has_surp])

        if n_data_fields == 0:
            # Ingen historik — vis kun næste earnings i fuld bredde
            st.markdown(
                f"<div style='background:{color}22;padding:0.8rem;border-radius:10px;"
                f"border-left:5px solid {color}'>"
                f"<small style='color:#888'>📅 NÆSTE EARNINGS</small>"
                f"<h3 style='margin:0.2rem 0;color:{color}'>{emoji} Om {days} dage · {date_str}</h3>"
                f"<small style='color:#aaa'>"
                f"⚠️ Ingen EPS-historik tilgængelig fra Yahoo Finance for denne ticker"
                f"</small>"
                f"</div>",
                unsafe_allow_html=True
            )
            return

        # Vis dynamisk antal kolonner baseret på tilgængelig data
        col_widths = [2] + [1] * n_data_fields
        cols = st.columns(col_widths)

        cols[0].markdown(
            f"<div style='background:{color}22;padding:0.8rem;border-radius:10px;"
            f"border-left:5px solid {color}'>"
            f"<small style='color:#888'>📅 NÆSTE EARNINGS</small>"
            f"<h3 style='margin:0.2rem 0;color:{color}'>{emoji} Om {days} dage</h3>"
            f"<small>{date_str}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        col_idx = 1
        if has_swing:
            cols[col_idx].metric(
                "📊 Hist. swing",
                f"±{avg_abs:.1f}%",
                f"{n_obs} kvartaler" if n_obs else None
            )
            col_idx += 1

        if has_beat:
            cols[col_idx].metric(
                "🎯 Beat rate",
                f"{beat_rate:.0f}%",
                f"{n_quarters} kvartaler"
            )
            col_idx += 1

        if has_surp:
            arrow = "📈" if avg_surp > 0 else "📉"
            cols[col_idx].metric(
                f"{arrow} Avg surprise",
                f"{avg_surp:+.1f}%"
            )

    else:
        # Fuld visning
        warning_text = ""
        if level == "critical":
            warning_text = (
                "🚨 <b>KRITISK:</b> Earnings er meget tæt på! "
                "Kursen kan svinge voldsomt - overvej at vente med nye positioner."
            )
        elif level == "high":
            warning_text = (
                "⚠️ <b>HØJ ADVARSEL:</b> Earnings inden for 1 uge. "
                "Reducér evt. positionsstørrelse, eller vent til efter rapporten."
            )
        elif level == "medium":
            warning_text = (
                "📅 <b>Medium:</b> Earnings inden for 2 uger. "
                "Hold øje med udviklingen og forventninger."
            )
        elif level == "low":
            warning_text = (
                "ℹ️ Earnings om 2-4 uger. Tid til at planlægge din strategi."
            )

        st.markdown(
            f"<div style='background:{color}15;padding:1.2rem;border-radius:12px;"
            f"border-left:5px solid {color};margin-bottom:1rem'>"
            f"<small style='color:#888'>📅 NÆSTE EARNINGS-RAPPORT</small>"
            f"<h2 style='margin:0.4rem 0;color:{color}'>{emoji} Om {days} dage</h2>"
            f"<h4 style='margin:0.3rem 0;color:#ddd'>{date_str}</h4>"
            f"<div style='margin-top:0.8rem;color:#ccc;font-size:0.95rem'>"
            f"{warning_text}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Detaljerede metrics (kun hvis vi har data)
        has_any = (avg_abs is not None or max_up is not None or
                   max_down is not None or beat_rate is not None)

        if has_any:
            det_cols = st.columns(4)

            if avg_abs is not None:
                det_cols[0].metric(
                    "📊 Avg post-earnings swing",
                    f"±{avg_abs:.1f}%",
                    f"{n_obs} kvartaler"
                )
            else:
                det_cols[0].metric("Avg swing", "N/A")

            if max_up is not None:
                det_cols[1].metric("📈 Største op-move", f"+{max_up:.1f}%")
            if max_down is not None:
                det_cols[2].metric("📉 Største ned-move", f"{max_down:.1f}%")

            if beat_rate is not None:
                beat_emoji = "🎯" if beat_rate >= 70 else "⚖️" if beat_rate >= 50 else "⚠️"
                det_cols[3].metric(
                    f"{beat_emoji} Beat rate",
                    f"{beat_rate:.0f}%",
                    f"{n_quarters} kvartaler"
                )
        else:
            st.caption("ℹ️ Ingen historisk EPS-data tilgængelig fra Yahoo Finance")


def render_earnings_history(data: Optional[Dict]) -> None:
    """Render historisk EPS surprise-tabel + chart"""
    if data is None or not data.get("history"):
        st.info("Ingen earnings-historik tilgængelig")
        return

    history = data["history"]

    # Build dataframe
    rows = []
    for h in history:
        if h.get("date") is None:
            continue
        date_str = h["date"].strftime("%Y-%m-%d")
        est = h.get("eps_estimate")
        act = h.get("eps_actual")
        surp = h.get("surprise_pct")
        beat = h.get("beat")

        rows.append({
            "Dato": date_str,
            "EPS Estimat": f"{est:.2f}" if est is not None else "-",
            "EPS Faktisk": f"{act:.2f}" if act is not None else "-",
            "Surprise %": f"{surp:+.1f}%" if surp is not None else "-",
            "Resultat": "✅ Beat" if beat is True else "❌ Miss" if beat is False else "?",
        })

    if not rows:
        st.info("Ingen historik-data")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Chart over surprises
    surprises = [
        (h["date"], h.get("surprise_pct"))
        for h in history
        if h.get("date") is not None and h.get("surprise_pct") is not None
    ]
    if surprises and len(surprises) >= 2:
        try:
            import plotly.graph_objects as go
            dates = [s[0] for s in reversed(surprises)]  # ældst først
            vals = [s[1] for s in reversed(surprises)]
            colors = ["#16a34a" if v >= 0 else "#ef4444" for v in vals]

            fig = go.Figure(data=[
                go.Bar(
                    x=dates, y=vals,
                    marker_color=colors,
                    text=[f"{v:+.1f}%" for v in vals],
                    textposition="outside",
                    textfont=dict(color="white"),
                )
            ])
            fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
            fig.update_layout(
                template="plotly_dark",
                height=350,
                title="EPS Surprise pr. kvartal (%)",
                yaxis_title="Surprise %",
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def render_post_earnings_moves(data: Optional[Dict]) -> None:
    """Render distribution af post-earnings dag-move"""
    if data is None:
        return
    vol = data.get("volatility")
    if vol is None or not vol.get("all_moves"):
        st.info("Ikke nok historik til post-earnings volatility-analyse")
        return

    moves = vol["all_moves"]

    cols = st.columns(4)
    cols[0].metric(
        "📊 Median move",
        f"{vol['median_move_pct']:+.2f}%"
    )
    cols[1].metric(
        "📏 Std. dev.",
        f"{vol['std_move_pct']:.2f}%"
    )
    cols[2].metric(
        f"⚡ Store moves (±{BIG_MOVE_THRESHOLD_PCT:.0f}%)",
        f"{vol['big_moves_count']}/{vol['n_observations']}",
        f"{vol['big_moves_pct']:.0f}%"
    )
    cols[3].metric(
        "📈 Range",
        f"{vol['max_down_pct']:+.1f}% til {vol['max_up_pct']:+.1f}%"
    )

    try:
        import plotly.graph_objects as go
        colors = ["#16a34a" if m >= 0 else "#ef4444" for m in moves]
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(1, len(moves)+1)),
                y=moves,
                marker_color=colors,
                text=[f"{m:+.1f}%" for m in moves],
                textposition="outside",
                textfont=dict(color="white", size=10),
            )
        ])
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3)
        fig.add_hline(
            y=BIG_MOVE_THRESHOLD_PCT, line_dash="dot",
            line_color="orange", opacity=0.5,
            annotation_text=f"+{BIG_MOVE_THRESHOLD_PCT:.0f}%"
        )
        fig.add_hline(
            y=-BIG_MOVE_THRESHOLD_PCT, line_dash="dot",
            line_color="orange", opacity=0.5,
            annotation_text=f"-{BIG_MOVE_THRESHOLD_PCT:.0f}%"
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            title="Post-earnings dag-move (% pris-ændring fra dag før til dag efter)",
            xaxis_title="Earnings-event (#)",
            yaxis_title="Move %",
            showlegend=False,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


# ============ WATCHLIST EARNINGS CALENDAR ============

def get_watchlist_earnings_calendar(tickers: List[str]) -> pd.DataFrame:
    """
    Hent earnings-datoer for hele watchlist og returner sorteret kalender.
    """
    rows = []
    for tk in tickers:
        try:
            data = get_earnings_info(tk)
            if data is None or data.get("days_until") is None:
                continue
            if data["days_until"] < 0:
                continue
            rows.append({
                "Ticker": tk,
                "Næste earnings": _format_date_dk(data["next_date"]),
                "Dage": data["days_until"],
                "Niveau": data["warning_level"],
                "Beat rate": (
                    f"{data['beat_rate']:.0f}%"
                    if data.get("beat_rate") is not None
                    else "-"
                ),
                "Avg swing": (
                    f"±{data['volatility']['avg_abs_move_pct']:.1f}%"
                    if data.get("volatility")
                    and data["volatility"].get("avg_abs_move_pct") is not None
                    else "-"
                ),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Dage")
    return df


def render_watchlist_earnings_calendar(tickers: List[str]) -> None:
    """Render earnings-kalender for watchlist"""
    if not tickers:
        st.info("Watchlist er tom")
        return

    with st.spinner(f"⚡ Henter earnings for {len(tickers)} tickers..."):
        df = get_watchlist_earnings_calendar(tickers)

    if df.empty:
        st.info("Ingen kommende earnings-datoer fundet for watchlist")
        return

    # Highlight critical/high
    def highlight_row(row):
        level = row.get("Niveau", "none")
        if level == "critical":
            return ["background-color: #dc262633"] * len(row)
        elif level == "high":
            return ["background-color: #ef444422"] * len(row)
        elif level == "medium":
            return ["background-color: #eab30822"] * len(row)
        return [""] * len(row)

    # Drop "Niveau" fra display, men brug til styling
    display_df = df.drop(columns=["Niveau"], errors="ignore")
    st.dataframe(
        display_df.style.apply(
            lambda r: highlight_row(df.loc[r.name]), axis=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Summary
    critical = len(df[df["Niveau"] == "critical"])
    high = len(df[df["Niveau"] == "high"])
    medium = len(df[df["Niveau"] == "medium"])

    if critical > 0 or high > 0:
        st.warning(
            f"🚨 **{critical} kritiske** + ⚠️ **{high} høje** earnings-warnings i din watchlist!"
        )
    elif medium > 0:
        st.info(f"📅 {medium} earnings inden for 2 uger")
# ============ EARNINGS SCORE BOOST ============

def calculate_earnings_score_boost(data: Optional[Dict]) -> Dict:
    """
    Beregn score-boost baseret på earnings-historik.
    Returnerer dict med:
        - boost: int (-10 til +15) - tilføjes til overall score
        - factors: List[Dict] - breakdown af hvad der påvirker
        - rating: str - "Excellent" / "God" / "OK" / "Dårlig"

    Logik:
    - Beat rate ≥ 90%      → +6 (excellent track record)
    - Beat rate ≥ 75%      → +4 (god track record)
    - Beat rate ≥ 60%      → +2 (OK)
    - Beat rate < 40%      → -3 (dårligt track record)

    - Avg surprise ≥ 5%    → +4 (markant overrasker)
    - Avg surprise ≥ 2%    → +2 (lidt over)
    - Avg surprise < -2%   → -3 (skuffer)

    - Lav post-earnings swing (<3%)  → +2 (lav risiko)
    - Høj post-earnings swing (>8%)  → -2 (høj risiko)

    - Earnings <3 dage frem  → -3 (timing-risiko)
    - Earnings <7 dage frem  → -1
    """
    factors = []
    boost = 0

    if data is None:
        return {
            "boost": 0,
            "factors": [],
            "rating": "Ukendt",
            "rating_color": "#6b7280",
            "explanation": "Ingen earnings-data tilgængelig"
        }

    # === BEAT RATE BOOST ===
    beat_rate = data.get("beat_rate")
    n_q = data.get("n_quarters", 0)

    if beat_rate is not None and n_q >= 4:  # Min. 4 kvartaler for pålidelig data
        if beat_rate >= 90:
            boost += 6
            factors.append({
                "label": f"🏆 Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +6,
                "category": "Track record"
            })
        elif beat_rate >= 75:
            boost += 4
            factors.append({
                "label": f"🎯 Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +4,
                "category": "Track record"
            })
        elif beat_rate >= 60:
            boost += 2
            factors.append({
                "label": f"✅ Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +2,
                "category": "Track record"
            })
        elif beat_rate < 40:
            boost -= 3
            factors.append({
                "label": f"⚠️ Lav beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": -3,
                "category": "Track record"
            })

    # === AVG SURPRISE BOOST ===
    avg_surp = data.get("avg_surprise_pct")

    if avg_surp is not None:
        if avg_surp >= 5:
            boost += 4
            factors.append({
                "label": f"📈 Stor positiv surprise (+{avg_surp:.1f}%)",
                "impact": +4,
                "category": "Surprise"
            })
        elif avg_surp >= 2:
            boost += 2
            factors.append({
                "label": f"📊 Positiv surprise (+{avg_surp:.1f}%)",
                "impact": +2,
                "category": "Surprise"
            })
        elif avg_surp <= -2:
            boost -= 3
            factors.append({
                "label": f"📉 Negative surprises ({avg_surp:.1f}%)",
                "impact": -3,
                "category": "Surprise"
            })

    # === VOLATILITET BOOST (risk-adjusted) ===
    vol = data.get("volatility") or {}
    avg_swing = vol.get("avg_abs_move_pct")
    n_obs = vol.get("n_observations", 0)

    if avg_swing is not None and n_obs >= 4:
        if avg_swing < 3:
            boost += 2
            factors.append({
                "label": f"🛡️ Lav post-earnings swing (±{avg_swing:.1f}%)",
                "impact": +2,
                "category": "Volatilitet"
            })
        elif avg_swing > 8:
            boost -= 2
            factors.append({
                "label": f"⚡ Høj post-earnings swing (±{avg_swing:.1f}%)",
                "impact": -2,
                "category": "Volatilitet"
            })

    # === TIMING-RISIKO ===
    days = data.get("days_until")
    if days is not None and days >= 0:
        if days <= 3:
            boost -= 3
            factors.append({
                "label": f"🚨 Earnings KRITISK tæt på ({days} dage)",
                "impact": -3,
                "category": "Timing"
            })
        elif days <= 7:
            boost -= 1
            factors.append({
                "label": f"⚠️ Earnings meget tæt på ({days} dage)",
                "impact": -1,
                "category": "Timing"
            })

    # === RATING ===
    if boost >= 8:
        rating = "Excellent"
        rating_color = "#16a34a"
        explanation = "Stærk earnings-historik booster scoren markant"
    elif boost >= 4:
        rating = "God"
        rating_color = "#22c55e"
        explanation = "Solid earnings-track record giver positivt boost"
    elif boost >= 0:
        rating = "Neutral"
        rating_color = "#eab308"
        explanation = "Earnings-data er neutral for scoren"
    elif boost >= -3:
        rating = "Svag"
        rating_color = "#f97316"
        explanation = "Earnings-historik trækker scoren let ned"
    else:
        rating = "Dårlig"
        rating_color = "#ef4444"
        explanation = "Dårlig earnings-historik trækker scoren betydeligt ned"

    return {
        "boost": boost,
        "factors": factors,
        "rating": rating,
        "rating_color": rating_color,
        "explanation": explanation,
    }


def render_earnings_score_card(data: Optional[Dict]) -> None:
    """
    Render et lille kort der viser earnings-score-bidrag.
    Kald denne i Score Breakdown sektionen.
    """
    score_info = calculate_earnings_score_boost(data)
    boost = score_info["boost"]
    rating = score_info["rating"]
    color = score_info["rating_color"]

    # Sign + farve
    if boost > 0:
        sign = "+"
        boost_color = "#16a34a"
    elif boost < 0:
        sign = ""
        boost_color = "#ef4444"
    else:
        sign = ""
        boost_color = "#6b7280"

    st.markdown(
        f"<div style='background:{color}15;padding:1rem;border-radius:10px;"
        f"border-left:5px solid {color};margin-bottom:0.6rem'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div>"
        f"<small style='color:#888'>📅 EARNINGS-SCORE BIDRAG</small>"
        f"<h3 style='margin:0.2rem 0;color:{color}'>{rating}</h3>"
        f"<small style='color:#aaa'>{score_info['explanation']}</small>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<small style='color:#888'>SCORE-EFFEKT</small>"
        f"<h1 style='margin:0.2rem 0;color:{boost_color}'>{sign}{boost}</h1>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Vis breakdown
    factors = score_info["factors"]
    if factors:
        with st.expander(f"📊 Se breakdown ({len(factors)} faktorer)"):
            for f in factors:
                imp = f["impact"]
                icon = "📈" if imp > 0 else "📉"
                color_f = "#16a34a" if imp > 0 else "#ef4444"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:0.4rem 0;border-bottom:1px solid #ffffff10'>"
                    f"<span>{f['label']} <small style='color:#666'>· {f['category']}</small></span>"
                    f"<span style='color:{color_f};font-weight:bold'>"
                    f"{icon} {imp:+d}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ============ EARNINGS MARKERS PÅ CHART ============

def add_earnings_markers_to_chart(
    fig,
    earnings_data: Optional[Dict],
    hist_df: pd.DataFrame,
    row: int = 1,
    col: int = 1,
) -> None:
    """
    Tilføj earnings-markører på et plotly chart.
    
    - Vertikale linjer på alle historiske earnings-datoer
    - Grøn farve hvis "beat", rød hvis "miss"
    - Annotation med "E" og dato
    - Special markør for NÆSTE earnings (gul/orange, fremtid)
    
    Args:
        fig: plotly figure (typisk make_subplots)
        earnings_data: Output fra get_earnings_info()
        hist_df: DataFrame med pris-historik (for x-axis range)
        row: row hvor markører skal vises (default: 1, dvs. main price chart)
        col: col (default: 1)
    """
    if earnings_data is None or hist_df is None or hist_df.empty:
        return

    history = earnings_data.get("history", [])
    next_date = earnings_data.get("next_date")

    # Find x-axis range fra hist
    try:
        hist_start = hist_df.index.min()
        hist_end = hist_df.index.max()
        # Konverter til naive datetime for sammenligning
        if hasattr(hist_start, 'tz') and hist_start.tz is not None:
            hist_start = hist_start.tz_localize(None) if hasattr(hist_start, 'tz_localize') else hist_start.replace(tzinfo=None)
            hist_end = hist_end.tz_localize(None) if hasattr(hist_end, 'tz_localize') else hist_end.replace(tzinfo=None)
    except Exception:
        return

    # Tilføj historiske earnings-markører
    for entry in history:
        ed = entry.get("date")
        if ed is None:
            continue

        try:
            # Konverter til naive
            ed_naive = ed.replace(tzinfo=None) if ed.tzinfo else ed

            # Skip hvis udenfor hist-range
            if ed_naive < hist_start or ed_naive > hist_end:
                continue

            # Bestem farve baseret på beat/miss
            beat = entry.get("beat")
            surp = entry.get("surprise_pct")

            if beat is True:
                line_color = "rgba(34, 197, 94, 0.6)"  # grøn
                marker_text = "✓"
                tooltip_color = "#16a34a"
            elif beat is False:
                line_color = "rgba(239, 68, 68, 0.6)"  # rød
                marker_text = "✗"
                tooltip_color = "#ef4444"
            else:
                line_color = "rgba(156, 163, 175, 0.5)"  # grå
                marker_text = "E"
                tooltip_color = "#9ca3af"

            # Surprise text
            surp_str = f"{surp:+.1f}%" if surp is not None else "?"

            # Tilføj vertikal linje
            fig.add_vline(
                x=ed_naive,
                line=dict(color=line_color, width=1.5, dash="dot"),
                row=row, col=col,
                annotation_text=f"{marker_text} {surp_str}",
                annotation_position="top",
                annotation_font=dict(size=10, color=tooltip_color),
                annotation_bgcolor="rgba(0,0,0,0.6)",
            )
        except Exception:
            continue

    # Tilføj NÆSTE earnings markør (fremtidig)
    if next_date is not None:
        try:
            next_naive = next_date.replace(tzinfo=None) if next_date.tzinfo else next_date
            now = datetime.now()

            # Kun hvis fremtidig
            if next_naive > now:
                # Tjek om datoen er indenfor visning (eller lige udenfor)
                # Vi tilføjer den uanset, plotly extender x-aksen
                fig.add_vline(
                    x=next_naive,
                    line=dict(color="#fbbf24", width=2.5, dash="dash"),  # gul/orange
                    row=row, col=col,
                    annotation_text=f"📅 NÆSTE EARNINGS",
                    annotation_position="top",
                    annotation_font=dict(size=11, color="#fbbf24"),
                    annotation_bgcolor="rgba(0,0,0,0.7)",
                )
        except Exception:
            pass


def add_earnings_legend_caption() -> None:
    """Vis forklaring på earnings-markører under et chart"""
    st.caption(
        "📅 **Earnings markører:** "
        "🟢 ✓ = EPS Beat · "
        "🔴 ✗ = EPS Miss · "
        "🟡 stiplet = Næste earnings"
    )
