"""
ml_predict.py - ML Prediction til Analyse-fanen
================================================
Loader trænede ML-modeller og laver forudsigelser på enkelte tickers.

Main entry: predict_all_horizons(info, hist, ...) -> dict med 30/90/180d forudsigelser
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

MODELS_DIR = Path("ml_models")
HORIZONS = [30, 90, 180]
ML_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]


# ==========================================
# LOAD MODELS (cached for performance)
# ==========================================

@st.cache_resource
def load_models_for_horizon(asset_class: str, horizon: int) -> Dict:
    """Load alle tilgængelige modeller for en asset class + horisont."""
    models = {"clf": {}, "reg": {}, "feature_columns": None}

    if not MODELS_DIR.exists():
        return models

    for model_name in ML_MODEL_NAMES:
        # Classifier
        clf_path = MODELS_DIR / f"{asset_class}_{horizon}d_{model_name}_clf.joblib"
        if clf_path.exists():
            try:
                data = joblib.load(clf_path)
                models["clf"][model_name] = data
                if models["feature_columns"] is None:
                    models["feature_columns"] = data.get("feature_columns", [])
            except Exception as e:
                print(f"⚠️ Failed to load {clf_path.name}: {e}")

        # Regressor
        reg_path = MODELS_DIR / f"{asset_class}_{horizon}d_{model_name}_reg.joblib"
        if reg_path.exists():
            try:
                data = joblib.load(reg_path)
                models["reg"][model_name] = data
            except Exception as e:
                print(f"⚠️ Failed to load {reg_path.name}: {e}")

    return models


@st.cache_resource
def load_all_models(asset_class: str) -> Dict[int, Dict]:
    """Load alle horisonter for en asset class."""
    return {h: load_models_for_horizon(asset_class, h) for h in HORIZONS}


def has_trained_models(asset_class: str = "stock") -> bool:
    """Tjek om der findes trænede modeller."""
    if not MODELS_DIR.exists():
        return False
    pattern = f"{asset_class}_*_clf.joblib"
    return len(list(MODELS_DIR.glob(pattern))) > 0


def get_model_info(asset_class: str = "stock") -> Dict:
    """Hent oversigt over loadede modeller."""
    if not MODELS_DIR.exists():
        return {"n_models": 0, "horizons": []}

    all_models = load_all_models(asset_class)
    info = {
        "n_models": 0,
        "horizons": [],
        "f1_scores": {},
    }
    for h, models in all_models.items():
        clf_count = len(models.get("clf", {}))
        if clf_count > 0:
            info["horizons"].append(h)
            info["n_models"] += clf_count
            # Find best F1
            best_f1 = 0
            for name, clf_data in models["clf"].items():
                f1 = clf_data.get("metrics", {}).get("f1_macro", 0)
                if f1 > best_f1:
                    best_f1 = f1
            info["f1_scores"][h] = best_f1
    return info


# ==========================================
# FEATURE ENGINEERING (matcher ml_data.py output)
# ==========================================

def build_feature_row(
    info: dict,
    hist: pd.DataFrame,
    indicators_df: pd.DataFrame,
    f_score: float,
    t_score: float,
    overall: float,
    regime: str,
    feature_columns: List[str],
) -> Optional[pd.DataFrame]:
    """
    Bygger 1-række DataFrame med præcis de features modellen blev trænet på.
    Bruger 0.0 som default for ukendte features (sikkert for one-hot encoded).
    """
    if not feature_columns:
        return None

    # Initialiser alle features til 0
    features = {col: 0.0 for col in feature_columns}

    # Hent sidste indicator-række
    last = indicators_df.iloc[-1] if not indicators_df.empty else None
    last_price = info.get("currentPrice")
    if last_price is None and not hist.empty:
        last_price = float(hist["Close"].iloc[-1])
    prev_close = info.get("previousClose")
    if prev_close is None and len(hist) >= 2:
        prev_close = float(hist["Close"].iloc[-2])

    # ============ NUMERISKE FEATURES ============
    feature_map = {}

    # Pris-features
    if last_price:
        feature_map["price"] = last_price
        feature_map["currentPrice"] = last_price
    if last_price and prev_close:
        feature_map["change_%"] = (last_price / prev_close - 1) * 100
        feature_map["change_pct"] = (last_price / prev_close - 1) * 100

    # Score-features
    feature_map["overall"] = overall
    feature_map["f_score"] = f_score
    feature_map["t_score"] = t_score
    feature_map["fundamental_score"] = f_score
    feature_map["technical_score"] = t_score

    # Tekniske indikatorer
    if last is not None:
        for tech_key, feature_keys in [
            ("RSI", ["rsi", "RSI"]),
            ("MACD", ["macd", "MACD"]),
            ("MACD_signal", ["macd_signal", "MACD_signal"]),
            ("ATR", ["atr", "ATR"]),
            ("ADX", ["adx", "ADX"]),
            ("SMA20", ["sma20", "SMA20"]),
            ("SMA50", ["sma50", "SMA50"]),
            ("SMA200", ["sma200", "SMA200"]),
        ]:
            val = last.get(tech_key)
            if pd.notna(val):
                for fk in feature_keys:
                    feature_map[fk] = float(val)

        # vs SMA
        sma50 = last.get("SMA50")
        sma200 = last.get("SMA200")
        if pd.notna(sma50) and last_price:
            val = (last_price / sma50 - 1) * 100
            feature_map["vs_sma50_%"] = val
            feature_map["vs_sma50_pct"] = val
        if pd.notna(sma200) and last_price:
            val = (last_price / sma200 - 1) * 100
            feature_map["vs_sma200_%"] = val
            feature_map["vs_sma200_pct"] = val

        # Bollinger Bands position
        bb_high = last.get("BB_high")
        bb_low = last.get("BB_low")
        if pd.notna(bb_high) and pd.notna(bb_low) and last_price and bb_high != bb_low:
            bb_pos = (last_price - bb_low) / (bb_high - bb_low) * 100
            feature_map["bb_position_%"] = bb_pos
            feature_map["bb_position"] = bb_pos

    # 52-uger range
    if not hist.empty:
        recent_year = hist.tail(252) if len(hist) >= 252 else hist
        high_52 = float(recent_year["High"].max())
        low_52 = float(recent_year["Low"].min())
        if high_52 and last_price:
            val = (last_price / high_52 - 1) * 100
            feature_map["vs_52w_high_%"] = val
            feature_map["vs_52w_high_pct"] = val
        if low_52 and last_price and high_52 != low_52:
            pos = (last_price - low_52) / (high_52 - low_52) * 100
            feature_map["pos_in_52w_%"] = pos
            feature_map["pos_in_52w_pct"] = pos

    # Fundamentale features
    fund_map = [
        ("trailingPE", ["pe", "trailingPE", "pe_ratio"]),
        ("forwardPE", ["forward_pe", "forwardPE"]),
        ("priceToBook", ["pb", "priceToBook", "price_to_book"]),
        ("pegRatio", ["peg", "pegRatio", "peg_ratio"]),
        ("dividendYield", ["dividend_%", "dividend_yield", "dividend_pct"], lambda v: v * 100),
        ("returnOnEquity", ["roe", "returnOnEquity"], lambda v: v * 100),
        ("profitMargins", ["profit_margin", "profitMargins", "profit_margins"], lambda v: v * 100),
        ("debtToEquity", ["debt_to_equity", "debtToEquity"]),
        ("earningsGrowth", ["earnings_growth", "earningsGrowth"], lambda v: v * 100),
        ("revenueGrowth", ["revenue_growth", "revenueGrowth"], lambda v: v * 100),
        ("beta", ["beta"]),
        ("marketCap", ["market_cap", "marketCap"]),
        ("payoutRatio", ["payout_ratio", "payoutRatio"]),
    ]
    for entry in fund_map:
        if len(entry) == 2:
            info_key, feat_keys = entry
            transform = None
        else:
            info_key, feat_keys, transform = entry
        val = info.get(info_key)
        if val is not None and not pd.isna(val):
            try:
                final_val = transform(val) if transform else val
                for fk in feat_keys:
                    feature_map[fk] = float(final_val)
            except (TypeError, ValueError):
                pass

    # Apply numeric features
    for key, value in feature_map.items():
        if key in features:
            try:
                if value is not None and not pd.isna(value):
                    features[key] = float(value)
            except (ValueError, TypeError):
                pass

    # ============ KATEGORISKE FEATURES (one-hot) ============

    # Sektor
    sector = info.get("sector", "Unknown")
    if sector:
        # Prøv flere variant-formater
        for sector_col in [
            f"sector_{sector}",
            f"sector_{sector.replace(' ', '_')}",
            f"sector_{sector.replace(' ', '')}",
        ]:
            if sector_col in features:
                features[sector_col] = 1.0
                break

    # Land
    country = info.get("country", "Unknown")
    if country:
        for country_col in [
            f"country_{country}",
            f"country_{country.replace(' ', '_')}",
        ]:
            if country_col in features:
                features[country_col] = 1.0
                break

    # Valuta
    currency = info.get("currency", "USD")
    if currency:
        currency_col = f"currency_{currency}"
        if currency_col in features:
            features[currency_col] = 1.0

    # Regime
    if regime:
        regime_col = f"regime_{regime}"
        if regime_col in features:
            features[regime_col] = 1.0

    # Build DataFrame i korrekt rækkefølge
    df = pd.DataFrame([features])[feature_columns]

    # Erstat eventuelle NaN med 0
    df = df.fillna(0.0)

    return df


# ==========================================
# PREDICTION
# ==========================================

def predict_single_horizon(feature_row: pd.DataFrame, models: Dict) -> Dict:
    """Forudsig for én horisont, ensemble af alle tilgængelige modeller."""
    if feature_row is None or feature_row.empty:
        return {"error": "No features"}

    classifiers = models.get("clf", {})
    regressors = models.get("reg", {})

    if not classifiers:
        return {"error": "No classifiers loaded"}

    # ===== KLASSIFIKATION =====
    clf_predictions = {}
    proba_list = []
    class_labels = ["SELL", "HOLD", "BUY"]

    for name, clf_data in classifiers.items():
        try:
            model = clf_data["model"]
            le = clf_data.get("label_encoder")

            pred = model.predict(feature_row)[0]
            proba = model.predict_proba(feature_row)[0]

            if le is not None:
                pred_label = str(le.inverse_transform([pred])[0])
                local_labels = list(le.classes_)
            else:
                local_labels = ["SELL", "HOLD", "BUY"]
                pred_label = local_labels[pred]

            class_labels = local_labels  # Brug fra modellen

            proba_dict = {label: float(p) for label, p in zip(local_labels, proba)}

            clf_predictions[name] = {
                "label": pred_label,
                "confidence": float(max(proba)),
                "probabilities": proba_dict,
                "f1_score": clf_data.get("metrics", {}).get("f1_macro", 0),
            }
            proba_list.append(proba)
        except Exception as e:
            print(f"⚠️ Predict failed for {name}: {e}")

    if not clf_predictions:
        return {"error": "All classifiers failed"}

    # Ensemble: gennemsnit af probabilities
    avg_proba = np.mean(proba_list, axis=0)
    ensemble_idx = int(np.argmax(avg_proba))
    ensemble_label = class_labels[ensemble_idx]
    ensemble_confidence = float(avg_proba[ensemble_idx])
    ensemble_proba = {label: float(p) for label, p in zip(class_labels, avg_proba)}

    # ===== REGRESSION =====
    reg_predictions = {}
    return_list = []
    for name, reg_data in regressors.items():
        try:
            model = reg_data["model"]
            pred = float(model.predict(feature_row)[0])
            reg_predictions[name] = {
                "expected_return_%": pred,
                "mae": reg_data.get("metrics", {}).get("mae", 0),
            }
            return_list.append(pred)
        except Exception as e:
            print(f"⚠️ Regressor failed for {name}: {e}")

    avg_return = float(np.mean(return_list)) if return_list else None
    avg_mae = float(np.mean([r["mae"] for r in reg_predictions.values()])) if reg_predictions else None

    return {
        "ensemble": {
            "label": ensemble_label,
            "confidence": ensemble_confidence,
            "probabilities": ensemble_proba,
            "expected_return_%": avg_return,
            "expected_return_mae": avg_mae,
        },
        "individual_classifiers": clf_predictions,
        "individual_regressors": reg_predictions,
    }


def predict_all_horizons(
    info: dict,
    hist: pd.DataFrame,
    indicators_df: pd.DataFrame,
    f_score: float,
    t_score: float,
    overall: float,
    regime: str,
    asset_class: str = "stock",
) -> Dict:
    """Forudsig for alle 3 horisonter (30d/90d/180d)."""
    if not has_trained_models(asset_class):
        return {"error": "Ingen trænede ML-modeller fundet"}

    all_models = load_all_models(asset_class)
    predictions = {}

    for horizon, models in all_models.items():
        if not models.get("clf"):
            predictions[horizon] = {"error": f"Ingen modeller for {horizon}d"}
            continue

        feature_row = build_feature_row(
            info, hist, indicators_df,
            f_score, t_score, overall, regime,
            models["feature_columns"],
        )

        if feature_row is None:
            predictions[horizon] = {"error": "Kunne ikke bygge features"}
            continue

        try:
            predictions[horizon] = predict_single_horizon(feature_row, models)
        except Exception as e:
            predictions[horizon] = {"error": f"Forudsigelse fejlede: {str(e)[:100]}"}

    return {
        "predictions": predictions,
        "asset_class": asset_class,
        "n_models_loaded": sum(len(m.get("clf", {})) for m in all_models.values()),
    }


# ==========================================
# UI HELPERS
# ==========================================

def get_label_color(label: str) -> str:
    return {"BUY": "#16a34a", "HOLD": "#eab308", "SELL": "#ef4444"}.get(
        str(label).upper(), "#888888"
    )


def get_label_emoji(label: str) -> str:
    return {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(
        str(label).upper(), "⚪"
    )


def render_ml_summary_card(predictions_data: Dict, rule_based_rec: str = ""):
    """Kompakt summary card til top af Analyse-siden."""
    if "error" in predictions_data:
        return  # Vis ikke fejl i compact mode

    predictions = predictions_data.get("predictions", {})
    if not predictions:
        return

    # 180d er den mest pålidelige
    pred_180 = predictions.get(180, {})
    if "error" in pred_180:
        return
    best_pred = pred_180.get("ensemble", {})
    if not best_pred:
        return

    label = best_pred.get("label", "?")
    conf = best_pred.get("confidence", 0)
    exp_ret = best_pred.get("expected_return_%")
    color = get_label_color(label)
    emoji = get_label_emoji(label)

    # Sammenlign med rule-based
    rule_buy = "KØB" in rule_based_rec
    rule_sell = "SÆLG" in rule_based_rec
    ml_buy = label == "BUY"
    ml_sell = label == "SELL"

    if (rule_buy and ml_buy) or (rule_sell and ml_sell):
        agreement = "✅ Samstemmende"
        agree_color = "#16a34a"
    elif (rule_buy and ml_sell) or (rule_sell and ml_buy):
        agreement = "⚠️ Modsatte"
        agree_color = "#ef4444"
    else:
        agreement = "🟡 Delvis"
        agree_color = "#eab308"

    ret_str = f"{exp_ret:+.1f}%" if exp_ret is not None else "?"

    st.markdown(
        f"<div style='background:{color}15;padding:0.8rem 1.2rem;border-radius:10px;"
        f"border-left:5px solid {color};margin:0.5rem 0;display:flex;"
        f"justify-content:space-between;align-items:center;flex-wrap:wrap'>"
        f"<div>"
        f"<small style='color:#888'>🤖 ML FORUDSIGELSE (180 dage · ⭐ mest pålidelig)</small><br>"
        f"<b style='color:{color};font-size:1.1rem'>{emoji} {label}</b> "
        f"<span style='color:#aaa'>· {conf*100:.0f}% conf. · forventet afkast: <b>{ret_str}</b></span>"
        f"</div>"
        f"<div style='background:{agree_color}22;padding:0.4rem 0.8rem;border-radius:6px;"
        f"border-left:3px solid {agree_color}'>"
        f"<small><b>{agreement}</b></small>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )


def render_ml_full(predictions_data: Dict, rule_based_rec: str = "", rule_based_score: float = 50):
    """Fuld ML-visning til ML-tabben."""
    import plotly.graph_objects as go

    if "error" in predictions_data:
        st.warning(f"⚠️ {predictions_data['error']}")
        st.info(
            "💡 **Ingen trænede ML-modeller fundet.**\n\n"
            "👉 Gå til **🔧 Diagnose** → **🚀 Backfill** → kør backfill\n"
            "👉 Gå derefter til **🎯 Træn ML** → træn modellerne\n"
            "👉 Push til GitHub for at gemme dem permanent"
        )
        return

    predictions = predictions_data.get("predictions", {})
    n_models = predictions_data.get("n_models_loaded", 0)

    st.markdown("### 🤖 ML Forudsigelser")
    st.caption(
        f"Ensemble af {n_models} trænede modeller "
        f"(Random Forest + XGBoost + LightGBM) forudsiger over 3 tidshorisonter."
    )

    # ===== HORISONT-OVERSIGT =====
    horizon_info = {
        30: {"label": "Kort sigt", "icon": "⚡", "trust": "Lav (F1≈0.40)", "trust_color": "#ef4444"},
        90: {"label": "Mellem sigt", "icon": "📊", "trust": "OK (F1≈0.52)", "trust_color": "#eab308"},
        180: {"label": "Lang sigt", "icon": "🎯", "trust": "GOOD (F1≈0.58) ⭐", "trust_color": "#16a34a"},
    }

    cols = st.columns(3)
    for i, h in enumerate([30, 90, 180]):
        h_data = predictions.get(h, {})
        h_info = horizon_info[h]

        with cols[i]:
            if "error" in h_data:
                st.error(f"{h_info['icon']} {h}d: {h_data['error']}")
                continue

            ens = h_data.get("ensemble", {})
            label = ens.get("label", "?")
            conf = ens.get("confidence", 0)
            exp_ret = ens.get("expected_return_%")
            mae = ens.get("expected_return_mae")
            color = get_label_color(label)
            emoji = get_label_emoji(label)

            ret_str = f"{exp_ret:+.2f}%" if exp_ret is not None else "?"
            mae_str = f"± {mae:.2f}%" if mae is not None else ""

            border_size = "5px" if h == 180 else "3px"
            star = " ⭐" if h == 180 else ""

            st.markdown(
                f"<div style='background:{color}15;padding:1.2rem;border-radius:12px;"
                f"border-left:{border_size} solid {color};text-align:center'>"
                f"<small style='color:#888'>{h_info['icon']} {h_info['label']} ({h}d){star}</small>"
                f"<h2 style='color:{color};margin:0.3rem 0'>{emoji} {label}</h2>"
                f"<div style='font-size:1.3rem;margin:0.3rem 0'>{conf*100:.0f}% conf.</div>"
                f"<div style='color:#aaa;font-size:0.9rem;margin:0.5rem 0'>"
                f"Forventet: <b>{ret_str}</b><br>"
                f"<small>{mae_str}</small>"
                f"</div>"
                f"<div style='background:{h_info['trust_color']}22;padding:0.3rem;"
                f"border-radius:6px;margin-top:0.5rem'>"
                f"<small style='color:{h_info['trust_color']}'><b>{h_info['trust']}</b></small>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ===== ML VS RULE-BASED =====
    st.markdown("### 📊 ML vs Rule-based")
    st.caption("Sammenligning af de 2 systemers anbefalinger")

    pred_180 = predictions.get(180, {}).get("ensemble", {})
    if pred_180 and "error" not in predictions.get(180, {}):
        ml_label = pred_180.get("label", "?")
        ml_conf = pred_180.get("confidence", 0)

        rule_buy = "KØB" in rule_based_rec
        rule_sell = "SÆLG" in rule_based_rec
        ml_buy = ml_label == "BUY"
        ml_sell = ml_label == "SELL"

        comp_cols = st.columns(2)
        comp_cols[0].markdown(
            f"<div style='background:#0099ff15;padding:1rem;border-radius:10px;"
            f"border-left:4px solid #0099ff'>"
            f"<small style='color:#888'>📐 RULE-BASED (regler + regime)</small>"
            f"<h3 style='margin:0.3rem 0'>{rule_based_rec}</h3>"
            f"<div>Score: <b>{rule_based_score:.0f}/100</b></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        ml_color = get_label_color(ml_label)
        comp_cols[1].markdown(
            f"<div style='background:{ml_color}15;padding:1rem;border-radius:10px;"
            f"border-left:4px solid {ml_color}'>"
            f"<small style='color:#888'>🤖 ML MODEL (180d ensemble)</small>"
            f"<h3 style='margin:0.3rem 0;color:{ml_color}'>"
            f"{get_label_emoji(ml_label)} {ml_label}</h3>"
            f"<div>Confidence: <b>{ml_conf*100:.0f}%</b></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Verdict
        st.markdown("#### 🎯 Verdict")
        if rule_buy and ml_buy:
            st.success(
                f"✅ **STÆRKT KØBSSIGNAL!** Både regler og ML er enige om KØB. "
                f"Dette er en HØJ-CONFIDENCE situation. Overvej fuld position."
            )
        elif rule_sell and ml_sell:
            st.error(
                f"🚨 **STÆRKT SALGSSIGNAL!** Både regler og ML er enige om SÆLG. "
                f"Overvej at lukke positionen."
            )
        elif rule_buy and ml_sell:
            st.warning(
                f"⚠️ **MODSATTE SIGNALER!** Reglerne siger KØB, men ML siger SÆLG. "
                f"ML har set noget bekymrende i dataen. **Vær forsigtig** - "
                f"reducér position-størrelse eller vent på bedre setup."
            )
        elif rule_sell and ml_buy:
            st.info(
                f"💡 **DIVERGERENDE SIGNALER!** Reglerne siger SÆLG, men ML siger KØB. "
                f"ML har måske fanget en turnaround. Tjek nyheder og earnings før beslutning."
            )
        elif rule_buy and ml_label == "HOLD":
            st.info(
                f"🟡 **MODERATE KØBSSIGNAL.** Reglerne siger KØB, ML er neutral. "
                f"Tag mindre position end normalt (50-70% af planlagt)."
            )
        elif rule_sell and ml_label == "HOLD":
            st.info(
                f"🟡 **MODERATE SALGSSIGNAL.** Reglerne siger SÆLG, ML er neutral. "
                f"Måske bare reducér position frem for at sælge alt."
            )
        elif ml_buy and "HOLD" in rule_based_rec:
            st.info(
                f"💡 **ML SER OPTUR.** Reglerne er neutrale, men ML siger KØB. "
                f"Måske et early entry-mulighed - tag lille position."
            )
        else:
            st.info(
                f"⚖️ **NEUTRAL ZONE.** Ingen klart signal fra hverken regler eller ML. "
                f"Vent på bedre setup."
            )

    st.markdown("---")

    # ===== SANDSYNLIGHEDS-CHART =====
    st.markdown("### 📊 Sandsynlighedsfordeling per horisont")

    fig = go.Figure()
    horizons_labels = []
    buy_probs, hold_probs, sell_probs = [], [], []

    for h in [30, 90, 180]:
        h_data = predictions.get(h, {})
        if "error" in h_data:
            continue
        ens = h_data.get("ensemble", {})
        proba = ens.get("probabilities", {})
        horizons_labels.append(f"{h} dage")
        buy_probs.append(proba.get("BUY", 0) * 100)
        hold_probs.append(proba.get("HOLD", 0) * 100)
        sell_probs.append(proba.get("SELL", 0) * 100)

    if horizons_labels:
        fig.add_trace(go.Bar(
            name="🟢 BUY", x=horizons_labels, y=buy_probs,
            marker_color="#16a34a",
            text=[f"{v:.0f}%" for v in buy_probs],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="🟡 HOLD", x=horizons_labels, y=hold_probs,
            marker_color="#eab308",
            text=[f"{v:.0f}%" for v in hold_probs],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="🔴 SELL", x=horizons_labels, y=sell_probs,
            marker_color="#ef4444",
            text=[f"{v:.0f}%" for v in sell_probs],
            textposition="auto",
        ))
        fig.update_layout(
            barmode="stack", template="plotly_dark", height=400,
            yaxis_title="Sandsynlighed (%)", yaxis_range=[0, 100],
            title="Stacked sandsynligheder for BUY/HOLD/SELL",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ===== INDIVIDUELLE MODELLER (debug) =====
    with st.expander("🔍 Individuelle modeller (debug-info)"):
        for h in [30, 90, 180]:
            h_data = predictions.get(h, {})
            if "error" in h_data:
                continue

            st.markdown(f"#### 📅 {h} dages horisont")

            ind_clf = h_data.get("individual_classifiers", {})
            if ind_clf:
                st.caption("**Klassifikatorer:**")
                rows = []
                for name, p in ind_clf.items():
                    proba = p.get("probabilities", {})
                    rows.append({
                        "Model": name,
                        "Forudsigelse": f"{get_label_emoji(p.get('label', '?'))} {p.get('label', '?')}",
                        "Confidence": f"{p.get('confidence', 0)*100:.0f}%",
                        "F1 (training)": f"{p.get('f1_score', 0):.3f}",
                        "P(BUY)": f"{proba.get('BUY', 0)*100:.0f}%",
                        "P(HOLD)": f"{proba.get('HOLD', 0)*100:.0f}%",
                        "P(SELL)": f"{proba.get('SELL', 0)*100:.0f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            ind_reg = h_data.get("individual_regressors", {})
            if ind_reg:
                st.caption("**Regressorer (forventet afkast):**")
                reg_rows = []
                for name, p in ind_reg.items():
                    reg_rows.append({
                        "Model": name,
                        "Forventet afkast": f"{p.get('expected_return_%', 0):+.2f}%",
                        "MAE": f"±{p.get('mae', 0):.2f}%",
                    })
                st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # ===== DISCLAIMER =====
    st.caption(
        "⚠️ **ML er IKKE perfekt!** F1=0.55-0.60 betyder modellen har ret ~55-60% af gangene. "
        "Brug ML som ÉN af flere signaler — IKKE som eneste beslutningsgrundlag. "
        "Kombinér altid med fundamental analyse, news, earnings og rule-based score. "
        "Position sizing + stop-loss er essentielt."
    )
