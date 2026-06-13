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
    "critical": 3,
    "high": 7,
    "medium": 14,
    "low": 30,
}

BIG_MOVE_THRESHOLD_PCT = 5.0


# ============ DATA-FETCHING ============

def _safe_to_datetime(val) -> Optional[datetime]:
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
        return pd.to_datetime(val, utc=True).to_pydatetime()
    except Exception:
        return None


def _calc_days_until(target: Optional[datetime]) -> Optional[int]:
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
    candidates = []

    try:
        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and ed:
                        candidates.append(_safe_to_datetime(ed[0]))
                    else:
                        candidates.append(_safe_to_datetime(ed))
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    candidates.append(_safe_to_datetime(val))
    except Exception:
        pass

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

    now = datetime.now(timezone.utc)
    future_dates = [d for d in candidates if d is not None and d > now]

    if future_dates:
        return min(future_dates)

    valid = [d for d in candidates if d is not None]
    if valid:
        return min(valid, key=lambda d: abs((d - now).total_seconds()))

    return None


def _fetch_earnings_history(ticker_obj, n_quarters: int = 8) -> List[Dict]:
    """
    Hent historisk earnings med MANGE fallbacks.
    Robust mod yfinance API-ændringer og rate-limiting.
    """
    history = []
    debug_log = []  # Bruges til UI debug
    ed = None
    method_used = None

    # === Method 1: get_earnings_dates() ===
    try:
        ed = ticker_obj.get_earnings_dates(limit=n_quarters * 3)
        if ed is not None and not ed.empty:
            method_used = "get_earnings_dates()"
            debug_log.append(f"✓ Method 1 (get_earnings_dates) → {len(ed)} rows")
            debug_log.append(f"  Kolonner: {list(ed.columns)}")
        else:
            debug_log.append(f"✗ Method 1 (get_earnings_dates) returnerede tom")
            ed = None
    except Exception as e:
        debug_log.append(f"✗ Method 1 fejl: {str(e)[:100]}")
        ed = None

    # === Method 2: earnings_dates property ===
    if ed is None:
        try:
            ed = ticker_obj.earnings_dates
            if ed is not None and not ed.empty:
                method_used = "earnings_dates"
                debug_log.append(f"✓ Method 2 (earnings_dates) → {len(ed)} rows")
                debug_log.append(f"  Kolonner: {list(ed.columns)}")
            else:
                debug_log.append(f"✗ Method 2 (earnings_dates) returnerede tom")
                ed = None
        except Exception as e:
            debug_log.append(f"✗ Method 2 fejl: {str(e)[:100]}")
            ed = None

    # === Method 3: earnings_history property (NYT) ===
    if ed is None:
        try:
            eh = ticker_obj.earnings_history
            if eh is not None and not eh.empty:
                method_used = "earnings_history"
                debug_log.append(f"✓ Method 3 (earnings_history) → {len(eh)} rows")
                debug_log.append(f"  Kolonner: {list(eh.columns)}")
                ed = eh
            else:
                debug_log.append(f"✗ Method 3 (earnings_history) tom")
        except Exception as e:
            debug_log.append(f"✗ Method 3 fejl: {str(e)[:100]}")

    # === Method 4: quarterly_income_stmt fallback ===
    if ed is None:
        try:
            qis = ticker_obj.quarterly_income_stmt
            if qis is not None and not qis.empty:
                debug_log.append(f"✓ Method 4 (quarterly_income_stmt) → {qis.shape}")
                debug_log.append(f"  Index: {list(qis.index)[:5]}")
                # Try to extract EPS rows
                eps_rows = [idx for idx in qis.index if "EPS" in str(idx) or "Earnings Per Share" in str(idx)]
                debug_log.append(f"  EPS rows: {eps_rows}")
        except Exception as e:
            debug_log.append(f"✗ Method 4 fejl: {str(e)[:100]}")

    # Print all debug
    print("\n".join([f"[earnings] {line}" for line in debug_log]))

    # Save debug for UI
    try:
        st.session_state["_earnings_debug"] = debug_log
    except Exception:
        pass

    if ed is None or (hasattr(ed, 'empty') and ed.empty):
        debug_log.append(f"❌ Alle methods fejlede - ingen historik fundet")
        try:
            st.session_state["_earnings_debug"] = debug_log
        except Exception:
            pass
        return []

    try:
        now = pd.Timestamp.now(tz="UTC")

        # Sørg for tz-aware index
        if ed.index.tz is None:
            ed.index = ed.index.tz_localize("UTC")

        # Filtrér til kun fortid
        past = ed[ed.index <= now].sort_index(ascending=False).head(n_quarters)

        if past.empty:
            debug_log.append(f"⚠️ Alle earnings-datoer er fremtidige")
            try:
                st.session_state["_earnings_debug"] = debug_log
            except Exception:
                pass
            return []

        debug_log.append(f"Fandt {len(past)} historiske earnings-rows")

        EST_COLS = [
            "EPS Estimate", "Estimate", "epsEstimate", "estimate",
            "EPS_Estimate", "eps_estimate", "EPS Est.", "epsestimate"
        ]
        ACT_COLS = [
            "Reported EPS", "Actual", "epsActual", "actual",
            "reportedEPS", "Reported_EPS", "eps_actual", "EPS Reported",
            "epsactual"
        ]
        SURP_COLS = [
            "Surprise(%)", "Surprise %", "surprisePercent",
            "surprise(%)", "Surprise", "surprise_pct", "Surprise%",
            "surprisepercent"
        ]

        # Print sample row to debug
        if len(past) > 0:
            sample = past.iloc[0].to_dict()
            debug_log.append(f"Sample row: {sample}")

        for date_idx, row in past.iterrows():
            entry = {
                "date": _safe_to_datetime(date_idx),
                "eps_estimate": None,
                "eps_actual": None,
                "surprise_pct": None,
                "beat": None,
            }

            # EPS estimate - prøv også case-insensitive
            for est_col in EST_COLS:
                if est_col in row.index:
                    val = row.get(est_col)
                    if val is not None and not pd.isna(val):
                        try:
                            entry["eps_estimate"] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

            # Case-insensitive fallback
            if entry["eps_estimate"] is None:
                for col in row.index:
                    col_lower = str(col).lower().replace(" ", "").replace("_", "")
                    if "estimate" in col_lower or "epsest" in col_lower:
                        val = row.get(col)
                        if val is not None and not pd.isna(val):
                            try:
                                entry["eps_estimate"] = float(val)
                                break
                            except (ValueError, TypeError):
                                continue

            # Faktisk EPS
            for act_col in ACT_COLS:
                if act_col in row.index:
                    val = row.get(act_col)
                    if val is not None and not pd.isna(val):
                        try:
                            entry["eps_actual"] = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

            # Case-insensitive fallback for actual
            if entry["eps_actual"] is None:
                for col in row.index:
                    col_lower = str(col).lower().replace(" ", "").replace("_", "")
                    if "reported" in col_lower or "actual" in col_lower:
                        val = row.get(col)
                        if val is not None and not pd.isna(val):
                            try:
                                entry["eps_actual"] = float(val)
                                break
                            except (ValueError, TypeError):
                                continue

                        # 🆕 ALTID beregn surprise % fra estimate + actual (mest pålideligt)
            # yfinance "Surprise" er ofte EPS-forskel (0.05 = $0.05), ikke procent
            if (entry["eps_estimate"] is not None and
                    entry["eps_actual"] is not None):
                if entry["eps_estimate"] != 0:
                    entry["surprise_pct"] = (
                        (entry["eps_actual"] - entry["eps_estimate"]) /
                        abs(entry["eps_estimate"]) * 100
                    )
            else:
                # Fallback: prøv surprise-kolonner hvis vi mangler estimate/actual
                for surp_col in SURP_COLS:
                    if surp_col in row.index:
                        val = row.get(surp_col)
                        if val is not None and not pd.isna(val):
                            try:
                                surp_val = float(val)
                                # Hvis værdien er meget lille (<1), antag det er decimal-form
                                # eller EPS-difference - skip den
                                if abs(surp_val) < 1.0:
                                    continue  # Sandsynligvis EPS-difference, ikke %
                                entry["surprise_pct"] = surp_val
                                break
                            except (ValueError, TypeError):
                                continue

            # Beat/miss
            if entry["eps_estimate"] is not None and entry["eps_actual"] is not None:
                entry["beat"] = entry["eps_actual"] >= entry["eps_estimate"]

            if entry["date"] is not None:
                history.append(entry)

        debug_log.append(f"✓ Parsed {len(history)} entries")
        if history:
            for i, h in enumerate(history[:3]):
                debug_log.append(
                    f"  [{i}] {h['date']} - est:{h['eps_estimate']}, "
                    f"act:{h['eps_actual']}, surp:{h['surprise_pct']}, beat:{h['beat']}"
                )

        # Save final debug
        try:
            st.session_state["_earnings_debug"] = debug_log
            st.session_state["_earnings_method"] = method_used
        except Exception:
            pass

        # Print all
        print("\n".join([f"[earnings] {line}" for line in debug_log]))

    except Exception as e:
        debug_log.append(f"❌ Parsing fejl: {e}")
        print(f"[earnings] ❌ Historik parsing fejl: {e}")
        import traceback
        traceback.print_exc()
        try:
            st.session_state["_earnings_debug"] = debug_log
        except Exception:
            pass

    return history


def _calc_post_earnings_volatility(
    hist: pd.DataFrame, earnings_dates: List[datetime]
) -> Optional[Dict]:
    if hist is None or hist.empty or not earnings_dates:
        return None

    moves = []
    abs_moves = []

    for ed in earnings_dates:
        if ed is None:
            continue
        try:
            ed_naive = ed.replace(tzinfo=None) if ed.tzinfo else ed
            hist_idx = hist.index
            if hasattr(hist_idx, "tz") and hist_idx.tz is not None:
                hist_naive = hist.copy()
                hist_naive.index = hist_idx.tz_localize(None)
            else:
                hist_naive = hist

            before_mask = hist_naive.index <= ed_naive
            after_mask = hist_naive.index > ed_naive
            if not before_mask.any() or not after_mask.any():
                continue

            price_before = float(hist_naive[before_mask]["Close"].iloc[-1])
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

@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_info(ticker: str) -> Optional[Dict]:
    if not ticker or not YF_AVAILABLE:
        print(f"[earnings] Ticker mangler eller yfinance ikke installeret")
        return None

    try:
        tk = yf.Ticker(ticker)
        print(f"[earnings] Henter data for {ticker}...")

        next_date = None
        try:
            next_date = _fetch_next_earnings_date(tk)
        except Exception as e:
            print(f"[earnings] _fetch_next_earnings_date fejl: {e}")

        days_until = _calc_days_until(next_date)

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

        history = []
        try:
            history = _fetch_earnings_history(tk, n_quarters=8)
        except Exception as e:
            print(f"[earnings] _fetch_earnings_history fejl: {e}")

        beats = [h for h in history if h.get("beat") is True]
        valid_history = [h for h in history if h.get("beat") is not None]
        beat_rate = (len(beats) / len(valid_history) * 100) if valid_history else None

        surprises = [
            h["surprise_pct"] for h in history
            if h.get("surprise_pct") is not None
        ]
        avg_surprise = float(np.mean(surprises)) if surprises else None
        median_surprise = float(np.median(surprises)) if surprises else None

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
    if data is None or data.get("days_until") is None:
        return None

    days = data["days_until"]
    level = data["warning_level"]
    vol = data.get("volatility") or {}
    avg_move = vol.get("avg_abs_move_pct")

    if days < 0:
        return None

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

def render_earnings_warning(data: Optional[Dict], compact: bool = False) -> None:
    if data is None:
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
        has_swing = avg_abs is not None
        has_beat = beat_rate is not None
        has_surp = avg_surp is not None

        n_data_fields = sum([has_swing, has_beat, has_surp])

        if n_data_fields == 0:
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
    if data is None or not data.get("history"):
        st.info("Ingen earnings-historik tilgængelig")
        return

    history = data["history"]
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

    surprises = [
        (h["date"], h.get("surprise_pct"))
        for h in history
        if h.get("date") is not None and h.get("surprise_pct") is not None
    ]
    if surprises and len(surprises) >= 2:
        try:
            import plotly.graph_objects as go
            dates = [s[0] for s in reversed(surprises)]
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
    if data is None:
        return
    vol = data.get("volatility")
    if vol is None or not vol.get("all_moves"):
        st.info("Ikke nok historik til post-earnings volatility-analyse")
        return

    moves = vol["all_moves"]

    cols = st.columns(4)
    cols[0].metric("📊 Median move", f"{vol['median_move_pct']:+.2f}%")
    cols[1].metric("📏 Std. dev.", f"{vol['std_move_pct']:.2f}%")
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

    df = pd.DataFrame(rows).sort_values("Dage").reset_index(drop=True)
    return df


def render_watchlist_earnings_calendar(tickers: List[str]) -> None:
    """Render earnings-kalender for watchlist - FIXED styler"""
    if not tickers:
        st.info("Watchlist er tom")
        return

    with st.spinner(f"⚡ Henter earnings for {len(tickers)} tickers..."):
        df = get_watchlist_earnings_calendar(tickers)

    if df.empty:
        st.info("Ingen kommende earnings-datoer fundet for watchlist")
        return

    # 🆕 FIX: Build levels mapping FØR vi dropper kolonnen
    # Brug index for at sikre vi har 1:1 mapping
    levels_by_index = df["Niveau"].to_dict()

    # Drop "Niveau" fra display
    display_df = df.drop(columns=["Niveau"], errors="ignore")
    n_display_cols = len(display_df.columns)

    def highlight_row(row):
        """Returnerer styling for en row - matcher display_df's kolonner"""
        # row.name er index fra display_df (samme som df siden vi reset_index'ede)
        level = levels_by_index.get(row.name, "none")
        if level == "critical":
            return ["background-color: #dc262633"] * n_display_cols
        elif level == "high":
            return ["background-color: #ef444422"] * n_display_cols
        elif level == "medium":
            return ["background-color: #eab30822"] * n_display_cols
        return [""] * n_display_cols

    try:
        st.dataframe(
            display_df.style.apply(highlight_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        # Fallback hvis styling fejler - vis bare uden farver
        print(f"[earnings] Styling fejl: {e}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

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

    beat_rate = data.get("beat_rate")
    n_q = data.get("n_quarters", 0)

    if beat_rate is not None and n_q >= 4:
        if beat_rate >= 90:
            boost += 6
            factors.append({
                "label": f"🏆 Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +6, "category": "Track record"
            })
        elif beat_rate >= 75:
            boost += 4
            factors.append({
                "label": f"🎯 Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +4, "category": "Track record"
            })
        elif beat_rate >= 60:
            boost += 2
            factors.append({
                "label": f"✅ Beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": +2, "category": "Track record"
            })
        elif beat_rate < 40:
            boost -= 3
            factors.append({
                "label": f"⚠️ Lav beat rate {beat_rate:.0f}% ({n_q} kv.)",
                "impact": -3, "category": "Track record"
            })

    avg_surp = data.get("avg_surprise_pct")

    if avg_surp is not None:
        if avg_surp >= 5:
            boost += 4
            factors.append({
                "label": f"📈 Stor positiv surprise (+{avg_surp:.1f}%)",
                "impact": +4, "category": "Surprise"
            })
        elif avg_surp >= 2:
            boost += 2
            factors.append({
                "label": f"📊 Positiv surprise (+{avg_surp:.1f}%)",
                "impact": +2, "category": "Surprise"
            })
        elif avg_surp <= -2:
            boost -= 3
            factors.append({
                "label": f"📉 Negative surprises ({avg_surp:.1f}%)",
                "impact": -3, "category": "Surprise"
            })

    vol = data.get("volatility") or {}
    avg_swing = vol.get("avg_abs_move_pct")
    n_obs = vol.get("n_observations", 0)

    if avg_swing is not None and n_obs >= 4:
        if avg_swing < 3:
            boost += 2
            factors.append({
                "label": f"🛡️ Lav post-earnings swing (±{avg_swing:.1f}%)",
                "impact": +2, "category": "Volatilitet"
            })
        elif avg_swing > 8:
            boost -= 2
            factors.append({
                "label": f"⚡ Høj post-earnings swing (±{avg_swing:.1f}%)",
                "impact": -2, "category": "Volatilitet"
            })

    days = data.get("days_until")
    if days is not None and days >= 0:
        if days <= 3:
            boost -= 3
            factors.append({
                "label": f"🚨 Earnings KRITISK tæt på ({days} dage)",
                "impact": -3, "category": "Timing"
            })
        elif days <= 7:
            boost -= 1
            factors.append({
                "label": f"⚠️ Earnings meget tæt på ({days} dage)",
                "impact": -1, "category": "Timing"
            })

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
    score_info = calculate_earnings_score_boost(data)
    boost = score_info["boost"]
    rating = score_info["rating"]
    color = score_info["rating_color"]

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
    - Special markør for NÆSTE earnings (gul/orange, fremtid)
    """
    if earnings_data is None or hist_df is None or hist_df.empty:
        print("[earnings_chart] ⚠️ Ingen data eller hist_df er tom")
        return

    history = earnings_data.get("history", [])
    next_date = earnings_data.get("next_date")

    print(f"[earnings_chart] History entries: {len(history)}, next_date: {next_date}")

    # Konverter hist range til pd.Timestamp uden tz for konsistent comparison
    try:
        hist_start = pd.Timestamp(hist_df.index.min())
        hist_end = pd.Timestamp(hist_df.index.max())

        if hist_start.tz is not None:
            hist_start = hist_start.tz_localize(None)
        if hist_end.tz is not None:
            hist_end = hist_end.tz_localize(None)

        print(f"[earnings_chart] Hist range: {hist_start} → {hist_end}")
    except Exception as e:
        print(f"[earnings_chart] ❌ Fejl ved hist range: {e}")
        return

    # === HISTORISKE EARNINGS-MARKØRER ===
    n_added = 0
    n_skipped = 0

    for entry in history:
        ed = entry.get("date")
        if ed is None:
            continue

        try:
            # Konverter til pd.Timestamp uden tz
            ed_ts = pd.Timestamp(ed)
            if ed_ts.tz is not None:
                ed_ts = ed_ts.tz_localize(None)

            # Skip hvis udenfor hist-range
            if ed_ts < hist_start or ed_ts > hist_end:
                n_skipped += 1
                continue

            beat = entry.get("beat")
            surp = entry.get("surprise_pct")

            if beat is True:
                line_color = "#22c55e"
                marker_text = "✓"
                tooltip_color = "#16a34a"
            elif beat is False:
                line_color = "#ef4444"
                marker_text = "✗"
                tooltip_color = "#ef4444"
            else:
                line_color = "#9ca3af"
                marker_text = "E"
                tooltip_color = "#9ca3af"

            surp_str = f"{surp:+.1f}%" if surp is not None else "?"

            # Brug pydatetime - mest robust for plotly subplots
            x_value = ed_ts.to_pydatetime()

            fig.add_vline(
                x=x_value,
                line_width=1.5,
                line_dash="dot",
                line_color=line_color,
                opacity=0.6,
                row=row, col=col,
                annotation_text=f"{marker_text} {surp_str}",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=tooltip_color,
                annotation_bgcolor="rgba(0,0,0,0.6)",
            )
            n_added += 1

        except Exception as e:
            print(f"[earnings_chart] Fejl ved hist marker {ed}: {e}")
            continue

    print(f"[earnings_chart] ✓ Tilføjet {n_added} historiske markører, skippet {n_skipped}")

    # === NÆSTE EARNINGS MARKØR (fremtidig) ===
    if next_date is not None:
        try:
            next_ts = pd.Timestamp(next_date)
            if next_ts.tz is not None:
                next_ts = next_ts.tz_localize(None)

            now_ts = pd.Timestamp.now()

            if next_ts > now_ts:
                x_value = next_ts.to_pydatetime()

                fig.add_vline(
                    x=x_value,
                    line_width=2.5,
                    line_dash="dash",
                    line_color="#fbbf24",
                    opacity=0.9,
                    row=row, col=col,
                    annotation_text="📅 NÆSTE EARNINGS",
                    annotation_position="top",
                    annotation_font_size=11,
                    annotation_font_color="#fbbf24",
                    annotation_bgcolor="rgba(0,0,0,0.7)",
                )
                print(f"[earnings_chart] ✓ Tilføjet næste earnings markør: {next_ts}")
        except Exception as e:
            print(f"[earnings_chart] ❌ Fejl ved next-date: {e}")


def add_earnings_legend_caption() -> None:
    st.caption(
        "📅 **Earnings markører:** "
        "🟢 ✓ = EPS Beat · "
        "🔴 ✗ = EPS Miss · "
        "🟡 stiplet = Næste earnings"
    )
