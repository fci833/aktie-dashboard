"""
Track Record Integration Helper
================================
Wraps ml_predict.predict_all_horizons() så predictions automatisk
logges til prediction_logger uden at ændre eksisterende kode.

Usage i app.py:
    from track_integration import auto_log_predictions, render_track_record_view
    
    # Efter predict_all_horizons() kald:
    auto_log_predictions(
        ticker=ticker,
        ml_data=ml_predictions_data,
        asset_class="stock",
        entry_price=price,
        features={"f_score": f_score, "t_score": t_score, ...},
    )
"""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Optional

import streamlit as st

# Lazy imports så app.py ikke crasher hvis Phase 2 modules mangler
try:
    from prediction_logger import PredictionLogger, PredictionRecord
    from track_record import TrackRecord
    PHASE2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Phase 2 moduler ikke fundet: {e}")
    PHASE2_AVAILABLE = False


DB_PATH = "predictions.db"


# ============================================================
# Singleton logger (cached)
# ============================================================

@st.cache_resource
def get_logger() -> Optional["PredictionLogger"]:
    if not PHASE2_AVAILABLE:
        return None
    return PredictionLogger(DB_PATH)


# ============================================================
# Auto-logging fra ml_predict output
# ============================================================

def auto_log_predictions(
    ticker: str,
    ml_data: dict,
    asset_class: str,
    entry_price: float,
    features: Optional[dict] = None,
    deduplicate: bool = True,
) -> int:
    """
    Log alle predictions fra predict_all_horizons() output.
    
    ml_data forventes at have struktur:
      {
        "horizons": {30: {...}, 90: {...}, 180: {...}},
        ...
      }
    
    Hver horizon har:
      - "ensemble_proba": dict eller float (BUY/HOLD/SELL probs eller P(BUY))
      - "models": {"random_forest": {...}, "xgboost": {...}, "lightgbm": {...}}
      - "predicted_class": "BUY"/"HOLD"/"SELL"
      - "confidence": float
    
    Returnerer antal logged predictions.
    """
    if not PHASE2_AVAILABLE or ml_data is None:
        return 0
    
    logger = get_logger()
    if logger is None:
        return 0
    
    horizons = ml_data.get("horizons") or ml_data.get("predictions") or {}
    if not horizons:
        return 0
    
    # Dedup: spring over hvis vi allerede loggede dette ticker+dato+horizon
    if deduplicate:
        today = date.today().isoformat()
        existing = logger.get_track_record(ticker=ticker, since=today)
        existing_keys = set()
        if not existing.empty:
            for _, row in existing.iterrows():
                existing_keys.add((row["horizon_days"], row["model_name"]))
    else:
        existing_keys = set()
    
    n_logged = 0
    
    for horizon_key, h_data in horizons.items():
        if not isinstance(h_data, dict):
            continue
        
        # Parse horizon (kan være "30d", 30, "30")
        try:
            horizon_days = int(str(horizon_key).replace("d", "").strip())
        except (ValueError, AttributeError):
            continue
        
        # Extract probabilities for hver model
        model_probs = _extract_model_probs(h_data)
        
        for model_name, prob in model_probs.items():
            if (horizon_days, model_name) in existing_keys:
                continue  # already logged today
            
            try:
                rec = PredictionRecord(
                    ticker=ticker.upper(),
                    asset_type=asset_class,
                    horizon_days=horizon_days,
                    entry_price=float(entry_price),
                    predicted_prob=float(prob),
                    model_name=model_name,
                    calibrated=True,
                    raw_features=features or {},
                    notes=f"auto-logged from ml_predict",
                )
                logger.log(rec)
                n_logged += 1
            except Exception as e:
                print(f"  ⚠️ Failed to log {ticker} {model_name} {horizon_days}d: {e}")
    
    return n_logged


def _extract_model_probs(h_data: dict) -> dict[str, float]:
    """
    Extract P(up) probability for hver model fra horizon-data.
    Tolerant overfor forskellige struktur-variationer.
    """
    out = {}
    
    # Try direct model probs first
    models = h_data.get("models", {})
    for name, m_data in models.items():
        if isinstance(m_data, dict):
            # Try multiple keys
            prob = (
                m_data.get("proba_buy")
                or m_data.get("prob_buy")
                or m_data.get("p_buy")
                or m_data.get("probability")
            )
            if prob is None and "probabilities" in m_data:
                probs = m_data["probabilities"]
                if isinstance(probs, dict):
                    prob = probs.get("BUY") or probs.get("UP")
                elif isinstance(probs, (list, tuple)) and len(probs) >= 3:
                    # [SELL, HOLD, BUY] convention
                    prob = probs[-1]
            if prob is not None:
                out[name] = float(prob)
    
    # Try ensemble
    ens_prob = (
        h_data.get("ensemble_proba_buy")
        or h_data.get("ensemble_prob")
        or h_data.get("ensemble_confidence")
    )
    if ens_prob is None:
        # Try ensemble dict
        ens = h_data.get("ensemble", {})
        if isinstance(ens, dict):
            ens_prob = (
                ens.get("proba_buy")
                or ens.get("prob_buy")
                or ens.get("confidence")
            )
            if ens_prob is None and "probabilities" in ens:
                probs = ens["probabilities"]
                if isinstance(probs, dict):
                    ens_prob = probs.get("BUY") or probs.get("UP")
                elif isinstance(probs, (list, tuple)) and len(probs) >= 3:
                    ens_prob = probs[-1]
    
    if ens_prob is not None:
        out["ensemble"] = float(ens_prob)
    
    # Fallback: if no probs but we have predicted_class + confidence, infer
    if not out:
        pred_class = h_data.get("predicted_class") or h_data.get("recommendation")
        conf = h_data.get("confidence", 0.5)
        if pred_class and conf:
            if pred_class.upper() in ("BUY", "KØB"):
                out["ensemble"] = float(conf)
            elif pred_class.upper() in ("SELL", "SÆLG"):
                out["ensemble"] = 1.0 - float(conf)
            else:
                out["ensemble"] = 0.5
    
    return out


# ============================================================
# Track Record View (dropped into app.py som ny tab)
# ============================================================

def render_track_record_view():
    """
    Render Track Record view i hoveddashboardet.
    Kald denne fra app.py når active_view == "📈 Track Record"
    """
    if not PHASE2_AVAILABLE:
        st.error(
            "❌ **Phase 2 moduler ikke installeret**\n\n"
            "Tjek at følgende filer er i samme mappe som `app.py`:\n"
            "- `ml_calibration.py`\n"
            "- `prediction_logger.py`\n"
            "- `track_record.py`\n"
            "- `walk_forward.py` (valgfri)\n"
            "- `retrain_scheduler.py` (valgfri)"
        )
        return
    
    # Import dashboard tab here (lazy)
    try:
        from dashboard_track_tab import render_track_record_tab
        render_track_record_tab(db_path=DB_PATH, initial_capital=10_000.0)
    except ImportError as e:
        st.error(f"❌ Kunne ikke importere dashboard_track_tab: {e}")
    except Exception as e:
        st.error(f"❌ Fejl: {e}")
        import traceback
        with st.expander("🐛 Traceback"):
            st.code(traceback.format_exc())


# ============================================================
# Lille status-badge til sidebaren
# ============================================================

def render_track_status_badge():
    """Vis lille status i sidebar: 'X resolved, Y pending'."""
    if not PHASE2_AVAILABLE:
        return
    
    logger = get_logger()
    if logger is None:
        return
    
    try:
        stats = logger.stats_overview()
        if stats["total"] == 0:
            st.caption("📈 Track Record: ingen logs endnu")
        else:
            st.caption(
                f"📈 Track: {stats['resolved']} resolved · "
                f"{stats['pending']} pending"
            )
            if stats["pending"] > 0:
                # Auto-resolve hvis nogle er overdue
                pass
    except Exception:
        pass
