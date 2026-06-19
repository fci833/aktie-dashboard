"""
Smart AI Verdict - Helhedsvurdering der ser på det BIG PICTURE
Fanger over-aggressive anbefalinger og giver personlig, disciplineret guidance
"""
import streamlit as st
import pandas as pd
import numpy as np


def _safe_get(d, key, default=None):
    """Sikker dict access der håndterer None"""
    if d is None:
        return default
    try:
        v = d.get(key, default)
        return v if v is not None else default
    except (AttributeError, TypeError):
        return default


def generate_smart_verdict(
    ticker, name, price, currency,
    score, recommendation, regime, regime_confidence,
    f_score, t_score,
    targets, hist, info,
    sentiment_data=None,
    earnings_data=None,
    pattern_bias=None,
    pattern_bullish_n=0,
    pattern_bearish_n=0,
    dcf_upside=None,
):
    """
    Genererer en smart helhedsvurdering der:
    - Identificerer røde/gule/grønne flag automatisk
    - Sammenligner alle datapunkter for konsistens
    - Nedjusterer over-aggressive anbefalinger
    - Foreslår konkrete handlinger
    """
    red_flags = []
    yellow_flags = []
    green_flags = []
    adjustments = []

    # Sikre default-værdier
    info = info or {}
    score = float(score) if score is not None else 50.0
    f_score = float(f_score) if f_score is not None else 50.0
    t_score = float(t_score) if t_score is not None else 50.0
    regime = regime or "UNKNOWN"
    regime_confidence = float(regime_confidence) if regime_confidence is not None else 50.0
    recommendation = recommendation or "HOLD"

    # ===== 1. Position i 52-uger range (KRITISK) =====
    low_52 = info.get("fiftyTwoWeekLow")
    high_52 = info.get("fiftyTwoWeekHigh")
    pos_in_range = None

    # Fallback til hist hvis info mangler
    if (low_52 is None or high_52 is None) and hist is not None and not hist.empty:
        recent = hist.tail(252) if len(hist) >= 252 else hist
        try:
            if low_52 is None:
                low_52 = float(recent["Low"].min())
            if high_52 is None:
                high_52 = float(recent["High"].max())
        except Exception:
            pass

    try:
        if low_52 and high_52 and high_52 > low_52 and price:
            pos_in_range = (price - low_52) / (high_52 - low_52) * 100

            if pos_in_range > 90:
                ath_distance = ((high_52 / price) - 1) * 100 if price > 0 else 0
                red_flags.append({
                    "icon": "🚩",
                    "title": "Du køber tæt på toppen",
                    "detail": (
                        f"Pris er på **{pos_in_range:.0f}%** i 52-uger range — "
                        f"kun **{ath_distance:.1f}%** til 52w-high. "
                        f"Aktier købt i top 10% af range har historisk lavere forward returns "
                        f"end aktier købt under medianen."
                    ),
                    "severity": "high"
                })
                adjustments.append("📉 Reducér position-størrelse med 50%")
            elif pos_in_range > 80:
                yellow_flags.append({
                    "icon": "⚠️",
                    "title": "Tæt på 52w-top",
                    "detail": (
                        f"Pris er **{pos_in_range:.0f}%** i 52w-range. "
                        f"Momentum kan fortsætte, men entry er ikke optimalt."
                    ),
                })
            elif pos_in_range < 30:
                green_flags.append({
                    "icon": "✅",
                    "title": "God entry-pris",
                    "detail": (
                        f"Pris er kun **{pos_in_range:.0f}%** i 52w-range — "
                        f"masser af opside og lavere risk fra mean reversion."
                    ),
                })
    except Exception:
        pass

    # ===== 2. Stop-loss bredde vs. daglig volatilitet =====
    try:
        if (targets and isinstance(targets, dict) and "stop_loss" in targets
                and targets["stop_loss"] is not None
                and hist is not None and not hist.empty and len(hist) >= 20
                and price and price > 0):
            stop_loss = targets["stop_loss"]
            stop_pct = abs((stop_loss / price - 1) * 100)
            recent = hist.tail(20)
            avg_daily_range_pct = ((recent["High"] - recent["Low"]) / recent["Close"]).mean() * 100

            if avg_daily_range_pct > 0 and stop_pct < avg_daily_range_pct * 2:
                wider_stop = price * (1 - avg_daily_range_pct * 3 / 100)
                yellow_flags.append({
                    "icon": "🛑",
                    "title": "Stop-loss er meget stram",
                    "detail": (
                        f"Stop på **-{stop_pct:.1f}%** er kun "
                        f"**{stop_pct/avg_daily_range_pct:.1f}x** "
                        f"daglig range ({avg_daily_range_pct:.1f}%). "
                        f"Risiko for whipsaw på normale dage."
                    ),
                })
                adjustments.append(
                    f"🛡️ Overvej bredere stop på ~**{wider_stop:.2f} {currency}** "
                    f"(-{avg_daily_range_pct*3:.1f}%)"
                )
    except Exception:
        pass

    # ===== 3. Pattern detection alignment =====
    if pattern_bias is not None:
        if "KØB" in recommendation and pattern_bias == "NEUTRAL":
            yellow_flags.append({
                "icon": "🔍",
                "title": "Chart patterns siger NEUTRAL",
                "detail": (
                    f"Modellen siger **{recommendation}**, men auto-pattern-detection "
                    f"finder kun **{pattern_bullish_n}** bullish setups. "
                    f"Ingen klassisk køb-formation (Cup & Handle, Double Bottom)."
                ),
            })
        elif "KØB" in recommendation and pattern_bias == "BEARISH":
            red_flags.append({
                "icon": "🚩",
                "title": "Chart patterns ER BEARISH",
                "detail": (
                    f"Modellen siger **{recommendation}** men "
                    f"**{pattern_bearish_n} bearish patterns** detekteret. "
                    f"STORT advarselssignal!"
                ),
                "severity": "critical"
            })
            adjustments.append("⛔ Vent på bedre setup eller skip handlen")
        elif "KØB" in recommendation and pattern_bias == "BULLISH":
            green_flags.append({
                "icon": "✅",
                "title": "Patterns bekræfter signal",
                "detail": (
                    f"**{pattern_bullish_n} bullish patterns** detekteret — "
                    f"chart understøtter {recommendation}."
                ),
            })

    # ===== 4. News sentiment alignment =====
    try:
        if sentiment_data and isinstance(sentiment_data, dict):
            article_count = _safe_get(sentiment_data, "article_count", 0)
            if article_count >= 3:
                sent_score = _safe_get(sentiment_data, "sentiment_score", 0) or 0
                sent_label = _safe_get(sentiment_data, "label", "Neutral")

                if "KØB" in recommendation and sent_score < -0.2:
                    red_flags.append({
                        "icon": "📰",
                        "title": "Negativ news flow",
                        "detail": (
                            f"Sentiment **{sent_score:+.2f}** ({sent_label}) — "
                            f"negative nyheder kan tynge prisen kortvarigt."
                        ),
                        "severity": "high"
                    })
                elif "KØB" in recommendation and sent_score > 0.3:
                    green_flags.append({
                        "icon": "📰",
                        "title": "Positiv news flow",
                        "detail": (
                            f"Sentiment **{sent_score:+.2f}** ({sent_label}) "
                            f"bekræfter momentum."
                        ),
                    })
                elif "SÆLG" in recommendation and sent_score > 0.3:
                    yellow_flags.append({
                        "icon": "📰",
                        "title": "Sentiment-divergens",
                        "detail": (
                            f"Modellen siger SÆLG men nyheder er **{sent_label}** "
                            f"({sent_score:+.2f}). Mulig turnaround?"
                        ),
                    })
    except Exception:
        pass

    # ===== 5. Earnings proximity =====
    try:
        if earnings_data and isinstance(earnings_data, dict):
            days_until = _safe_get(earnings_data, "days_until_earnings")
            if days_until is not None:
                days_until = int(days_until)
                if 0 < days_until <= 7:
                    red_flags.append({
                        "icon": "📅",
                        "title": f"Earnings om {days_until} dage",
                        "detail": (
                            "Stor risiko for **5-15% bevægelse på 1 dag**. "
                            "Undgå nye positioner lige før earnings."
                        ),
                        "severity": "critical"
                    })
                    adjustments.append(f"⏳ Vent til efter earnings ({days_until} dage)")
                elif 7 < days_until <= 14:
                    yellow_flags.append({
                        "icon": "📅",
                        "title": f"Earnings om {days_until} dage",
                        "detail": "Vær opmærksom — øget volatilitet forventes når dagen nærmer sig.",
                    })
    except Exception:
        pass

    # ===== 6. Regime check =====
    if regime in ("BEAR", "VOLATILE") and "KØB" in recommendation:
        regime_emoji = "🐻" if regime == "BEAR" else "⚡"
        yellow_flags.append({
            "icon": regime_emoji,
            "title": f"{regime} marked",
            "detail": (
                f"Selv KØB-signaler skal behandles med ekstra forsigtighed i "
                f"{regime} regime."
            ),
        })
        if "STÆRKT KØB" in recommendation:
            adjustments.append("📉 Halvér standard position-størrelse pga. regime")
    elif regime == "BULL" and "KØB" in recommendation:
        green_flags.append({
            "icon": "🐂",
            "title": "BULL regime støtter trade",
            "detail": f"Marked-momentum ({regime_confidence:.0f}% conf.) er din ven.",
        })

    # ===== 7. Risk/Reward sanity =====
    try:
        if (targets and isinstance(targets, dict)
                and targets.get("stop_loss") is not None
                and targets.get("target_long") is not None
                and price and price > 0):
            risk = price - targets["stop_loss"]
            reward = targets["target_long"] - price
            if risk > 0 and reward > 0:
                rr = reward / risk
                if rr < 2:
                    red_flags.append({
                        "icon": "⚖️",
                        "title": "Dårlig R/R ratio",
                        "detail": (
                            f"Reward {reward:.2f} / Risk {risk:.2f} = **{rr:.1f}:1**. "
                            f"Min. 2:1 anbefales for at trade kan give mening over tid."
                        ),
                        "severity": "high"
                    })
                elif rr >= 4:
                    green_flags.append({
                        "icon": "⚖️",
                        "title": "Excellent R/R",
                        "detail": (
                            f"R/R = **{rr:.1f}:1** — du risikerer 1 enhed for "
                            f"at vinde {rr:.1f}."
                        ),
                    })
    except Exception:
        pass

    # ===== 8. DCF check =====
    try:
        if dcf_upside is not None:
            dcf_upside = float(dcf_upside)
            if dcf_upside < -20 and "KØB" in recommendation:
                red_flags.append({
                    "icon": "💎",
                    "title": "DCF siger overvurderet",
                    "detail": (
                        f"DCF-upside er **{dcf_upside:+.0f}%** — modellen siger KØB, "
                        f"men fundamentals tyder på overvurdering."
                    ),
                    "severity": "high"
                })
            elif dcf_upside > 30 and "KØB" in recommendation:
                green_flags.append({
                    "icon": "💎",
                    "title": "DCF siger undervurderet",
                    "detail": f"DCF-upside er **{dcf_upside:+.0f}%** — stærkt fundament-signal.",
                })
    except Exception:
        pass

    # ===== 9. Score-konsistens (f_score vs t_score) =====
    try:
        score_gap = abs(f_score - t_score)
        if score_gap > 30:
            yellow_flags.append({
                "icon": "🔄",
                "title": "Stor uenighed F vs T",
                "detail": (
                    f"Fundamental ({f_score:.0f}) og Teknisk ({t_score:.0f}) er "
                    f"**{score_gap:.0f} point** fra hinanden. Kan signalere usikkerhed."
                ),
            })
    except Exception:
        pass

    # ===========================================
    # NEDJUSTÉR ANBEFALING BASERET PÅ FLAGS
    # ===========================================
    final_recommendation = recommendation
    confidence_modifier = 0

    n_critical = sum(1 for f in red_flags if f.get("severity") == "critical")
    n_high = sum(1 for f in red_flags if f.get("severity") == "high")
    n_yellow = len(yellow_flags)
    n_green = len(green_flags)

    if n_critical >= 1:
        if "STÆRKT KØB" in recommendation:
            final_recommendation = "KØB (med stort forbehold)"
            confidence_modifier = -35
        elif "KØB" in recommendation:
            final_recommendation = "VENT / Pas på"
            confidence_modifier = -40
    elif n_high >= 2:
        if "STÆRKT KØB" in recommendation:
            final_recommendation = "KØB (reduceret position)"
            confidence_modifier = -20
        elif "KØB" in recommendation:
            final_recommendation = "KØB (forsigtigt)"
            confidence_modifier = -15
    elif n_high == 1 or n_yellow >= 3:
        if "STÆRKT KØB" in recommendation:
            final_recommendation = "KØB (med disciplin)"
            confidence_modifier = -10
        else:
            confidence_modifier = -8
    elif n_green >= 3 and n_yellow == 0:
        confidence_modifier = +5

    final_confidence = max(0, min(100, score + confidence_modifier))

    # ===========================================
    # KONKRETE HANDLINGER
    # ===========================================
    actions = []

    if "VENT" in final_recommendation.upper() or "PAS PÅ" in final_recommendation.upper():
        actions.append({
            "icon": "⏳",
            "title": "Vent på bedre setup",
            "text": (
                "Sæt en kursalarm og vent. Du behøver ikke handle hver dag. "
                "Disciplin > FOMO. Kvalitets-setups kommer altid."
            )
        })
    elif "KØB" in final_recommendation:
        if n_critical >= 1 or n_high >= 2:
            actions.append({
                "icon": "📉",
                "title": "Reducér position kraftigt",
                "text": (
                    "Pga. flere røde flag: køb maks **25-50%** af din normale position. "
                    "Behold ammunition til pullbacks."
                )
            })
        elif n_high == 1 or n_yellow >= 2:
            actions.append({
                "icon": "🟡",
                "title": "Skaler ind gradvist",
                "text": (
                    "Køb i tre tranches: **1/3 nu**, **1/3 ved pullback** til SMA50, "
                    "**1/3 ved bekræftet breakout**."
                )
            })
        else:
            actions.append({
                "icon": "✅",
                "title": "Standard entry OK",
                "text": "Brug normal position-størrelse — alle systemer bekræfter signalet."
            })

        # Tilføj specifikke justeringer
        for adj in adjustments:
            if "stop" in adj.lower():
                actions.append({"icon": "🛡️", "title": "Stop-loss justering", "text": adj})

    elif "SÆLG" in final_recommendation:
        actions.append({
            "icon": "📤",
            "title": "Overvej at exit positionen",
            "text": (
                "Modellen ser nedside-risiko. Hvis du har positionen, så reducér "
                "eller sæt stram trailing stop. Hvis ikke, bliv væk."
            )
        })
    else:  # HOLD
        actions.append({
            "icon": "🟡",
            "title": "Hold positionen, ingen action",
            "text": (
                "Ingen klar edge i nogen retning. Hold hvis du har, "
                "vent hvis du ikke har."
            )
        })

    # ===========================================
    # GENERÉR SAMMENFATNING (natural language)
    # ===========================================
    summary = _build_summary(
        ticker, name, recommendation, final_recommendation,
        score, final_confidence, regime, pos_in_range,
        red_flags, yellow_flags, green_flags
    )

    return {
        "original_recommendation": recommendation,
        "final_recommendation": final_recommendation,
        "original_score": score,
        "final_confidence": final_confidence,
        "confidence_change": confidence_modifier,
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
        "green_flags": green_flags,
        "adjustments": adjustments,
        "actions": actions,
        "summary": summary,
        "verdict_color": _get_verdict_color(final_recommendation),
        "n_critical": n_critical,
        "n_high": n_high,
        "n_yellow": n_yellow,
        "n_green": n_green,
        "pos_in_range": pos_in_range,
    }


def _get_verdict_color(rec):
    if not rec:
        return "#888"
    rec_upper = rec.upper()
    if "VENT" in rec_upper or "PAS PÅ" in rec_upper:
        return "#f97316"
    if "FORBEHOLD" in rec_upper:
        return "#eab308"
    if "REDUCERET" in rec_upper or "FORSIGTIGT" in rec_upper or "DISCIPLIN" in rec_upper:
        return "#22c55e"
    if "STÆRKT KØB" in rec_upper:
        return "#16a34a"
    if "KØB" in rec_upper:
        return "#16a34a"
    if "HOLD" in rec_upper:
        return "#eab308"
    if "SÆLG" in rec_upper:
        return "#ef4444"
    return "#888"


def _build_summary(ticker, name, orig_rec, final_rec, score, confidence,
                   regime, pos_in_range, red_flags, yellow_flags, green_flags):
    """Bygger en menneskelig sammenfatning"""
    parts = []
    name = name or ticker

    # Opening
    if "VENT" in final_rec.upper() or "PAS PÅ" in final_rec.upper():
        parts.append(
            f"**{name}** ser umiddelbart fornuftig ud (score {score:.0f}/100), "
            f"men der er **kritiske advarselsflag** der gør at jeg ikke ville købe lige nu."
        )
    elif final_rec != orig_rec:
        parts.append(
            f"**{name}** scorer {score:.0f}/100 og modellen siger oprindeligt "
            f"**{orig_rec}**. Men når jeg ser på helheden, er der nuancer der gør "
            f"at jeg vil justere til **{final_rec}**."
        )
    elif len(green_flags) >= 3 and len(red_flags) == 0:
        parts.append(
            f"**{name}** ser stærk ud — score {score:.0f}/100, "
            f"alle systemer bekræfter **{final_rec}**, og der er ingen kritiske advarsler."
        )
    else:
        parts.append(
            f"**{name}** scorer {score:.0f}/100 i {regime} marked. "
            f"Anbefaling: **{final_rec}** — med nogle nuancer du skal være opmærksom på."
        )

    # Key concern (highest priority red flag)
    if red_flags:
        critical = [f for f in red_flags if f.get("severity") == "critical"]
        if critical:
            parts.append(
                f"\n\n🚩 **Vigtigst at vide:** {critical[0]['title']} — "
                f"{critical[0]['detail']}"
            )
        else:
            high = [f for f in red_flags if f.get("severity") == "high"]
            if high:
                parts.append(
                    f"\n\n⚠️ **Vigtigste bekymring:** {high[0]['title']} — "
                    f"{high[0]['detail']}"
                )

    # Position context
    if pos_in_range is not None:
        if pos_in_range > 90:
            parts.append(
                f"\n\n💭 **Min ærlige holdning:** Du køber sent i cyklen. "
                f"Det kan stadig gå op (momentum er momentum), men risk/reward er "
                f"dårligere end hvis du havde købt på en dyb pullback. "
                f"**Vær mere defensiv** end normalt."
            )
        elif pos_in_range < 30:
            parts.append(
                f"\n\n💎 **Min ærlige holdning:** Det er den slags setup jeg kan lide — "
                f"aktien handles tæt på 52w-low og har potentielt stort opside hvis "
                f"thesis holder. Lavere downside-risk fra mean reversion."
            )

    # Closing advice
    if "KØB" in final_rec and "VENT" not in final_rec.upper():
        if len(red_flags) == 0 and len(green_flags) >= 2:
            parts.append(
                "\n\n✅ **Konklusion:** Solidt setup. Brug standard position, "
                "respektér stop-loss, og lad winneren løbe."
            )
        else:
            parts.append(
                "\n\n🎯 **Konklusion:** Tag positionen, men **med disciplin**. "
                "Reducér størrelse, sæt rumlig stop-loss, og vær klar til at exit "
                "hvis thesis brydes."
            )
    elif "VENT" in final_rec.upper() or "PAS PÅ" in final_rec.upper():
        parts.append(
            "\n\n⏳ **Konklusion:** Skip denne. Bedre setups kommer. "
            "Sæt en alarm og vend tilbage når flagene er væk."
        )
    elif "SÆLG" in final_rec.upper():
        parts.append(
            "\n\n📤 **Konklusion:** Modellen ser klart nedside. "
            "Hvis du har positionen, beskyt din kapital."
        )

    return "".join(parts)


def render_smart_verdict(verdict, ticker, name, price, currency):
    """Render UI for smart verdict"""
    if not verdict:
        return

    color = verdict.get("verdict_color", "#888")
    final_rec = verdict.get("final_recommendation", "?")
    orig_rec = verdict.get("original_recommendation", "?")
    final_conf = verdict.get("final_confidence", 0)
    orig_score = verdict.get("original_score", 0)
    name = name or ticker

    # Hero card
    st.markdown(
        f"<div style='background: linear-gradient(135deg, {color}33, {color}11); "
        f"padding: 1.8rem; border-radius: 15px; border-left: 6px solid {color}; "
        f"margin: 1rem 0; box-shadow: 0 4px 20px {color}22'>"
        f"<div style='display: flex; justify-content: space-between; "
        f"align-items: center; flex-wrap: wrap; gap: 1rem'>"
        f"<div>"
        f"<small style='color: #888; text-transform: uppercase; "
        f"letter-spacing: 2px; font-size: 0.75rem'>"
        f"🤖 AI HELHEDSVURDERING</small>"
        f"<h2 style='margin: 0.4rem 0; color: {color}; font-size: 2rem'>{final_rec}</h2>"
        f"<small style='color: #aaa'>{ticker} · {name}</small>"
        f"</div>"
        f"<div style='text-align: right'>"
        f"<small style='color: #888; text-transform: uppercase; font-size: 0.75rem'>"
        f"CONFIDENCE</small>"
        f"<h1 style='margin: 0.3rem 0; color: {color}'>{final_conf:.0f}"
        f"<small style='font-size: 1.2rem; color: #888'>/100</small></h1>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Recommendation change banner
    if final_rec != orig_rec:
        change = verdict.get("confidence_change", 0)
        st.warning(
            f"⚠️ **Justering:** Modellen sagde oprindeligt **{orig_rec}** "
            f"({orig_score:.0f}/100). "
            f"AI har justeret til **{final_rec}** ({change:+d} confidence point) "
            f"baseret på røde/gule flag."
        )

    # Summary text
    st.markdown("#### 📝 Helhedsvurdering")
    st.markdown(verdict.get("summary", ""))

    # Stats overview
    n_critical = verdict.get("n_critical", 0)
    n_red = len(verdict.get("red_flags", []))
    n_yellow = len(verdict.get("yellow_flags", []))
    n_green = len(verdict.get("green_flags", []))

    stat_cols = st.columns(4)
    stat_cols[0].metric(
        "🚩 Røde flag",
        n_red,
        f"{n_critical} kritiske" if n_critical else "0 kritiske"
    )
    stat_cols[1].metric("⚠️ Gule flag", n_yellow)
    stat_cols[2].metric("✅ Grønne flag", n_green)

    flag_balance = n_green - n_red * 2
    stat_cols[3].metric(
        "⚖️ Flag balance",
        f"{flag_balance:+d}",
        "Positiv" if flag_balance > 0 else "Negativ" if flag_balance < 0 else "Neutral"
    )

    # Three-column flag display
    st.markdown("---")
    flag_cols = st.columns(3)

    with flag_cols[0]:
        st.markdown("##### 🚩 Røde flag")
        red_flags = verdict.get("red_flags", [])
        if red_flags:
            for flag in red_flags:
                sev = flag.get("severity", "high")
                bg = "#ef4444" if sev == "critical" else "#f97316"
                crit_marker = " 🔥" if sev == "critical" else ""
                st.markdown(
                    f"<div style='background: {bg}22; padding: 0.7rem; "
                    f"border-radius: 8px; border-left: 3px solid {bg}; "
                    f"margin-bottom: 0.5rem'>"
                    f"<b>{flag.get('icon', '')} {flag.get('title', '')}</b>"
                    f"{crit_marker}<br>"
                    f"<small style='color: #ddd'>{flag.get('detail', '')}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ Ingen røde flag")

    with flag_cols[1]:
        st.markdown("##### ⚠️ Gule flag")
        yellow_flags = verdict.get("yellow_flags", [])
        if yellow_flags:
            for flag in yellow_flags:
                st.markdown(
                    f"<div style='background: #eab30822; padding: 0.7rem; "
                    f"border-radius: 8px; border-left: 3px solid #eab308; "
                    f"margin-bottom: 0.5rem'>"
                    f"<b>{flag.get('icon', '')} {flag.get('title', '')}</b><br>"
                    f"<small style='color: #ddd'>{flag.get('detail', '')}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Ingen gule flag")

    with flag_cols[2]:
        st.markdown("##### ✅ Grønne flag")
        green_flags = verdict.get("green_flags", [])
        if green_flags:
            for flag in green_flags:
                st.markdown(
                    f"<div style='background: #16a34a22; padding: 0.7rem; "
                    f"border-radius: 8px; border-left: 3px solid #16a34a; "
                    f"margin-bottom: 0.5rem'>"
                    f"<b>{flag.get('icon', '')} {flag.get('title', '')}</b><br>"
                    f"<small style='color: #ddd'>{flag.get('detail', '')}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("Ingen særlige grønne signaler")

    # Concrete actions
    actions = verdict.get("actions", [])
    if actions:
        st.markdown("---")
        st.markdown("#### 🎯 Hvad skal jeg gøre NU?")
        for action in actions:
            st.markdown(
                f"<div style='background: {color}15; padding: 1rem; border-radius: 10px; "
                f"border-left: 4px solid {color}; margin-bottom: 0.6rem'>"
                f"<div style='font-size: 1.1rem; font-weight: bold; margin-bottom: 0.3rem'>"
                f"{action.get('icon', '')} {action.get('title', '')}</div>"
                f"<div style='color: #ddd'>{action.get('text', '')}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
