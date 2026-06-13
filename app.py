"""Aktie Dashboard - Hovedapp med Krypto + Daily Brief + News Sentiment + Earnings Warning"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============ PERFORMANCE MONITORING ============
_app_start_time = time.time()

from config import ANALYSIS_PERIODS, SCREENER_UNIVERSES
from data_sources import (
    fetch_data, search_tickers, get_fx_rate, get_api_keys,
    fetch_yahoo, fetch_twelve_single, get_twelve_formats,
    FINNHUB_AVAILABLE
)
from analysis import (
    get_indicators, fundamental_score, technical_score,
    calculate_price_targets, dcf_valuation, risk_metrics,
    monte_carlo, recommendation, filter_by_days, filter_chart_period,
    generate_action_plan
)

from backtest import run_backtest, simulate_strategy
from screener import run_screener, categorize_opportunities, sector_breakdown
from history import (
    save_snapshot, list_snapshots, load_snapshot,
    compare_snapshots, get_hot_stocks, get_score_history,
    cleanup_old_snapshots
)
from ui_helpers import make_price_box, make_range_box, make_recommendation_card

# 🪙 KRYPTO IMPORTS
from crypto_config import CRYPTO_UNIVERSE, CRYPTO_UNIVERSES
from crypto_data import (
    fetch_crypto_data, fetch_fear_greed, fetch_global_crypto_market,
    is_crypto, normalize_crypto_ticker,
    fetch_btc_onchain, fetch_trending_coins, fetch_top_movers,
)
from crypto_analysis import (
    crypto_overall_score, crypto_recommendation,
    crypto_indicators, crypto_price_targets, crypto_risk_metrics,
    crypto_monte_carlo, btc_halving_analysis, calculate_btc_correlation,
    crypto_backtest,
)

# 🏠 DAILY BRIEF
from daily_brief import (
    get_market_pulse, analyze_watchlist,
    get_recent_rating_changes, get_top_opportunities,
    calculate_position_size,
)

# 📰 NEWS SENTIMENT
from news_sentiment import (
    get_news_sentiment,
    render_sentiment_summary,
    render_news_feed,
)

# 🆕 EARNINGS WARNING - opdateret med score-boost + chart-markers
from earnings_warning import (
    get_earnings_info,
    render_earnings_warning,
    render_earnings_history,
    render_post_earnings_moves,
    render_watchlist_earnings_calendar,
    get_earnings_warning_message,
    # 🆕 NYE FUNKTIONER:
    calculate_earnings_score_boost,
    render_earnings_score_card,
    add_earnings_markers_to_chart,
    add_earnings_legend_caption,
)

import warnings
warnings.filterwarnings("ignore")
import requests as plain_requests

st.set_page_config(page_title="Aktie Dashboard", layout="wide", page_icon="📈")
st.markdown(
    "<h1 style='background:linear-gradient(90deg,#00d4aa,#0099ff);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
    "📈 Pro Aktie & Krypto Dashboard</h1>",
    unsafe_allow_html=True,
)

FINNHUB_KEY, TWELVE_KEY = get_api_keys()

# ===== SESSION STATE =====
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "last_source" not in st.session_state:
    st.session_state.last_source = "?"
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = ""
if "screener_results" not in st.session_state:
    st.session_state.screener_results = None
if "active_view" not in st.session_state:
    st.session_state.active_view = "🏠 Hjem"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False


# ============ DIAGNOSE ============

def run_diagnostics(ticker):
    results = []
    try:
        t0 = time.time()
        r = fetch_yahoo(ticker, "1y")
        dt = time.time() - t0
        if r == "RATE_LIMIT":
            results.append(("Yahoo Finance", "❌ Rate limited", f"{dt:.1f}s", "IP blokeret"))
        elif r is None:
            results.append(("Yahoo Finance", "❌ Ingen data", f"{dt:.1f}s", "Ticker findes ikke"))
        else:
            results.append(("Yahoo Finance", "✅ Virker", f"{dt:.1f}s", f"{len(r['hist'])} dage"))
    except Exception as e:
        results.append(("Yahoo Finance", "❌ Crash", "-", str(e)[:200]))

    if not TWELVE_KEY:
        results.append(("Twelve Data", "⚠️ Ingen API key", "-", "Tilføj TWELVE_DATA_KEY"))
    else:
        formats = get_twelve_formats(ticker)
        details = [f"Forsøger: {', '.join(formats)}"]
        success = False
        for symbol in formats:
            try:
                t0 = time.time()
                raw = plain_requests.get(
                    "https://api.twelvedata.com/quote",
                    params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=10
                ).json()
                dt = time.time() - t0
                if raw.get("code") == 429:
                    results.append(("Twelve Data", "❌ Rate limit", f"{dt:.1f}s",
                                    "RATE LIMIT - vent 60s"))
                    success = True
                    break
                elif raw.get("status") == "error" or "code" in raw:
                    details.append(f"  → '{symbol}': fejl {raw.get('code', '?')}")
                else:
                    r = fetch_twelve_single(symbol, TWELVE_KEY)
                    if r:
                        details.append(f"  → '{symbol}': ✅ {r['info'].get('longName')}")
                        results.append(("Twelve Data", "✅ Virker", f"{dt:.1f}s",
                                        "\n".join(details)))
                        success = True
                        break
            except Exception as e:
                details.append(f"  → '{symbol}': crash {str(e)[:100]}")
        if not success:
            results.append(("Twelve Data", "❌ Alle formater fejler", "-", "\n".join(details)))

    if not FINNHUB_KEY:
        results.append(("Finnhub", "⚠️ Ingen API key", "-", "Tilføj FINNHUB_API_KEY"))
    elif "." in ticker:
        results.append(("Finnhub", "⚠️ Springet over", "-", "Kun US"))
    else:
        try:
            import finnhub
            t0 = time.time()
            client = finnhub.Client(api_key=FINNHUB_KEY)
            profile = client.company_profile2(symbol=ticker)
            quote = client.quote(ticker)
            dt = time.time() - t0
            if not profile or not profile.get("name"):
                results.append(("Finnhub", "❌ Ingen profile", f"{dt:.1f}s", str(profile)[:200]))
            elif not quote.get("c") or quote.get("c", 0) < 0.01:
                results.append(("Finnhub", "❌ Ingen gyldig quote", f"{dt:.1f}s", str(quote)[:200]))
            else:
                results.append(("Finnhub", "✅ Virker", f"{dt:.1f}s",
                                f"{profile.get('name')}, pris: {quote.get('c')}"))
        except Exception as e:
            results.append(("Finnhub", "❌ Crash", "-", str(e)[:300]))
    return results


# ============ HJÆLPER: Skift til analyse + Search history ============

def goto_analysis(ticker):
    st.session_state.current_ticker = ticker
    st.session_state.active_view = "📊 Analyse"
    add_to_search_history(ticker)
    st.rerun()


def add_to_search_history(ticker):
    """Tilføjer ticker til søge-historik (max 10)"""
    if not ticker:
        return
    ticker_clean = ticker.strip().upper()
    history = st.session_state.search_history
    if ticker_clean in history:
        history.remove(ticker_clean)
    history.insert(0, ticker_clean)
    st.session_state.search_history = history[:10]


# ============ SIDEBAR ============

with st.sidebar:
    st.markdown("### 📡 Datakilder")
    st.caption("✅ Yahoo Finance")
    st.caption("✅ Twelve Data" if TWELVE_KEY else "⚠️ Twelve Data (no key)")
    st.caption("✅ Finnhub" if FINNHUB_KEY else "⚠️ Finnhub (no key)")
    st.caption("✅ CoinGecko (krypto)")
    st.caption("✅ Yahoo (krypto)")
    if st.session_state.last_source != "?":
        st.success(f"Sidst: **{st.session_state.last_source}**")

    st.markdown("---")
    st.markdown("### ⚙️ Indstillinger")
    period = st.selectbox(
        "📅 Chart visningsperiode",
        ["1y", "2y", "5y", "10y", "max"], index=2,
    )
    show_secondary = st.checkbox("💱 Vis priser i DKK også", value=True)

    if st.button("🔄 Ryd cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache ryddet!")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Hurtige tickers")
    quick = {
        "🇺🇸 US": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
        "🇩🇰 DK": ["NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "ORSTED.CO"],
        "🇪🇺 EU": ["ASML.AS", "SAP.DE", "NESN.SW"],
        "🪙 Crypto": ["BTC", "ETH", "SOL", "ADA"],
    }
    for region, ts in quick.items():
        with st.expander(region):
            for tk in ts:
                if st.button(tk, key=f"q_{tk}", use_container_width=True):
                    goto_analysis(tk)

    st.markdown("---")
    st.markdown("### 💱 Valutakurser")
    st.caption(f"USD/DKK: **{get_fx_rate('USD', 'DKK'):.2f}**")
    st.caption(f"EUR/DKK: **{get_fx_rate('EUR', 'DKK'):.2f}**")

    st.markdown("---")
    st.markdown("### ℹ️ Analyse-perioder")
    st.caption("📊 Tekniske: **12 mdr**")
    st.caption("💰 Kursmål: **6 mdr**")
    st.caption("📉 Risk: **3 år**")
    st.caption("🎲 Monte Carlo: **2 år**")

    st.markdown("---")
    st.session_state.dev_mode = st.checkbox(
        "🐛 Dev mode (vis perf-stats)",
        value=st.session_state.dev_mode,
        help="Viser performance-statistik nederst"
    )


# ============ NAVIGATION ============

view_options = ["🏠 Hjem", "📊 Analyse", "🔎 Screener", "🪙 Krypto", "🔍 Søg ticker", "🔧 Diagnose"]
selected_view = st.radio(
    "Navigation",
    view_options,
    index=view_options.index(st.session_state.active_view),
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)
if selected_view != st.session_state.active_view:
    st.session_state.active_view = selected_view
    st.rerun()

st.markdown("---")
# ============ HJEM (DAILY BRIEF) ============

if st.session_state.active_view == "🏠 Hjem":
    # 🆕 BRUG DANSK TID (Europe/Copenhagen) - fixer 2-timers offset
    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("Europe/Copenhagen"))
    except ImportError:
        # Fallback for ældre Python
        try:
            import pytz
            today = datetime.now(pytz.timezone("Europe/Copenhagen"))
        except ImportError:
            # Sidste fallback - manuel +2 timer (sommertid) / +1 (vintertid)
            today = datetime.utcnow() + timedelta(hours=2)

    weekday_dk = ["Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag", "Lørdag", "Søndag"][today.weekday()]
    month_dk = ["januar", "februar", "marts", "april", "maj", "juni",
                "juli", "august", "september", "oktober", "november", "december"][today.month - 1]

    # 🆕 DYNAMISK HILSEN BASERET PÅ TIDSPUNKT
    hour = today.hour
    if 5 <= hour < 10:
        greeting = "🌅 God morgen!"
    elif 10 <= hour < 12:
        greeting = "☀️ God formiddag!"
    elif 12 <= hour < 14:
        greeting = "🌞 God middag!"
    elif 14 <= hour < 18:
        greeting = "🌤️ God eftermiddag!"
    elif 18 <= hour < 22:
        greeting = "🌆 God aften!"
    else:
        greeting = "🌙 God nat!"

    st.markdown(
        f"<h2 style='margin-bottom:0'>{greeting}</h2>"
        f"<p style='color:#888;margin-top:0'>{weekday_dk} {today.day}. {month_dk} {today.year} · "
        f"Klokken {today.strftime('%H:%M')}</p>",
        unsafe_allow_html=True
    )

    refresh_col1, refresh_col2 = st.columns([5, 1])
    with refresh_col2:
        if st.button("🔄 Opdater", use_container_width=True, key="refresh_home"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Market Pulse")

    with st.spinner("Henter markedsdata..."):
        pulse = get_market_pulse()

    st.markdown(
        f"<div style='background:{pulse['regime_color']}22;padding:0.8rem 1.2rem;"
        f"border-radius:10px;border-left:5px solid {pulse['regime_color']};margin-bottom:1rem'>"
        f"<b>Markedsregime:</b> {pulse['regime']}</div>",
        unsafe_allow_html=True
    )

    pulse_cols = st.columns(5)

    spy = pulse["stocks"].get("S&P 500", {})
    if spy:
        change = spy["change_%"]
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        pulse_cols[0].metric(f"{emoji} S&P 500", f"${spy['price']:,.0f}", f"{change:+.2f}%")
    else:
        pulse_cols[0].metric("S&P 500", "N/A")

    nasdaq = pulse["stocks"].get("Nasdaq 100", {})
    if nasdaq:
        change = nasdaq["change_%"]
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        pulse_cols[1].metric(f"{emoji} Nasdaq 100", f"${nasdaq['price']:,.0f}", f"{change:+.2f}%")
    else:
        pulse_cols[1].metric("Nasdaq 100", "N/A")

    vix = pulse["stocks"].get("VIX", {})
    if vix:
        v = vix["price"]
        emoji = "🟢" if v < 15 else "🟡" if v < 25 else "🔴"
        label = "Lav frygt" if v < 15 else "Normal" if v < 25 else "HØJ FRYGT"
        pulse_cols[2].metric(f"{emoji} VIX (frygt)", f"{v:.1f}", label)
    else:
        pulse_cols[2].metric("VIX", "N/A")

    crypto = pulse.get("crypto", {})
    if crypto:
        change = crypto.get("change_24h", 0)
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        pulse_cols[3].metric(f"{emoji} Krypto MC", f"${crypto['total_mc_t']:.2f}T", f"{change:+.2f}%")
    else:
        pulse_cols[3].metric("Krypto", "N/A")

    fg = pulse.get("fear_greed", {})
    if fg:
        v = fg["value"]
        emoji = "🔴" if v < 25 else "🟢" if v > 75 else "🟡"
        pulse_cols[4].metric(f"{emoji} Krypto F&G", f"{v}/100", fg["label"])
    else:
        pulse_cols[4].metric("F&G", "N/A")

    st.markdown("---")
    st.markdown("### 🎯 Dagens Handlinger")

    # 🆕 4 TABS - inkl. earnings-kalender
    action_tabs = st.tabs([
        "🟢 KØB-muligheder",
        "👁️ Min Watchlist",
        "🔄 Rating-ændringer",
        "📅 Earnings-kalender",
    ])

    with action_tabs[0]:
        opps = get_top_opportunities(min_score=70, n_max=10)
        if opps is None:
            st.info(
                "💡 **Ingen screener-snapshots fundet endnu.**\n\n"
                "👉 Gå til **🔎 Screener** → kør en screener → så vises top-muligheder her automatisk!"
            )
        elif opps["buys"].empty:
            st.warning("Ingen aktier i seneste snapshot har score ≥ 70")
        else:
            st.caption(f"📊 Fra seneste screening: **{opps['universe']}** · {opps['snapshot_ts'][:10]}")
            for idx, (_, row) in enumerate(opps["buys"].iterrows()):
                cols = st.columns([1, 3, 1, 1, 1, 1])
                score = row.get("overall", 0)
                score_color = "#16a34a" if score >= 75 else "#22c55e" if score >= 65 else "#eab308"
                cols[0].markdown(
                    f"<div style='background:{score_color};color:white;padding:0.4rem;"
                    f"border-radius:8px;text-align:center;font-weight:bold'>{score:.0f}</div>",
                    unsafe_allow_html=True
                )
                cols[1].markdown(f"**{row['ticker']}**")
                cols[1].caption(str(row.get("name", ""))[:50])
                price = row.get("price", 0)
                change = row.get("change_%", 0)
                cols[2].metric("Pris", f"{price:.2f}", f"{change:+.2f}%", label_visibility="collapsed")
                rec = row.get("recommendation", "")
                cols[3].markdown(f"<div style='text-align:center;font-weight:bold'>{rec}</div>",
                                unsafe_allow_html=True)
                cols[4].caption(str(row.get("sector", "?"))[:15])
                if cols[5].button("📊 Åbn", key=f"home_buy_{row['ticker']}_{idx}", use_container_width=True):
                    goto_analysis(row["ticker"])
                st.markdown("<hr style='margin:0.3rem 0;opacity:0.2'>", unsafe_allow_html=True)

    with action_tabs[1]:
        if not st.session_state.watchlist:
            st.info(
                "💡 **Din watchlist er tom.**\n\n"
                "👉 Hver gang du analyserer en aktie/krypto, tilføjes den automatisk her."
            )
        else:
            st.caption(f"📋 {len(st.session_state.watchlist)} tickers i watchlist")
            with st.spinner(f"⚡ Analyserer {len(st.session_state.watchlist)} tickers parallelt..."):
                wdf = analyze_watchlist(st.session_state.watchlist, max_workers=8)
            if wdf.empty:
                st.warning("Kunne ikke analysere watchlist (data-fejl?)")
            else:
                buy_df = wdf[wdf["score"] >= 60]
                hold_df = wdf[(wdf["score"] >= 40) & (wdf["score"] < 60)]
                sell_df = wdf[wdf["score"] < 40]

                wsum_cols = st.columns(3)
                wsum_cols[0].metric("🟢 KØB-signaler", len(buy_df))
                wsum_cols[1].metric("🟡 HOLD", len(hold_df))
                wsum_cols[2].metric("🔴 SÆLG-signaler", len(sell_df))

                if not buy_df.empty:
                    st.markdown("#### 🟢 KØB i din watchlist")
                    for idx, (_, row) in enumerate(buy_df.iterrows()):
                        cols = st.columns([1, 3, 1, 1, 1])
                        cols[0].markdown(
                            f"<div style='background:#16a34a;color:white;padding:0.4rem;"
                            f"border-radius:8px;text-align:center;font-weight:bold'>{row['score']:.0f}</div>",
                            unsafe_allow_html=True
                        )
                        cols[1].markdown(f"**{row['ticker']}** {row['type']}")
                        cols[1].caption(str(row['name'])[:50])
                        cols[2].metric("Pris", f"{row['price']:.2f}", f"{row['change_%']:+.2f}%",
                                       label_visibility="collapsed")
                        cols[3].markdown(f"<div style='text-align:center'>{row['recommendation']}</div>",
                                        unsafe_allow_html=True)
                        if cols[4].button("📊", key=f"wbuy_{row['ticker']}_{idx}", use_container_width=True):
                            goto_analysis(row["ticker"])

                if not sell_df.empty:
                    with st.expander(f"🔴 SÆLG-signaler ({len(sell_df)})"):
                        for idx, (_, row) in enumerate(sell_df.iterrows()):
                            cols = st.columns([1, 3, 1, 1, 1])
                            cols[0].markdown(
                                f"<div style='background:#ef4444;color:white;padding:0.4rem;"
                                f"border-radius:8px;text-align:center;font-weight:bold'>{row['score']:.0f}</div>",
                                unsafe_allow_html=True
                            )
                            cols[1].markdown(f"**{row['ticker']}**")
                            cols[1].caption(str(row['name'])[:40])
                            cols[2].metric("Pris", f"{row['price']:.2f}", f"{row['change_%']:+.2f}%",
                                           label_visibility="collapsed")
                            cols[3].markdown(f"<div style='text-align:center'>{row['recommendation']}</div>",
                                            unsafe_allow_html=True)
                            if cols[4].button("📊", key=f"wsell_{row['ticker']}_{idx}", use_container_width=True):
                                goto_analysis(row["ticker"])

                if not hold_df.empty:
                    with st.expander(f"🟡 HOLD ({len(hold_df)})"):
                        for idx, (_, row) in enumerate(hold_df.iterrows()):
                            cols = st.columns([1, 3, 1, 1, 1])
                            cols[0].markdown(f"**{row['score']:.0f}**")
                            cols[1].markdown(f"**{row['ticker']}**")
                            cols[1].caption(str(row['name'])[:40])
                            cols[2].metric("Pris", f"{row['price']:.2f}", f"{row['change_%']:+.2f}%",
                                           label_visibility="collapsed")
                            cols[3].markdown(f"<div style='text-align:center'>{row['recommendation']}</div>",
                                            unsafe_allow_html=True)
                            if cols[4].button("📊", key=f"whold_{row['ticker']}_{idx}", use_container_width=True):
                                goto_analysis(row["ticker"])

    with action_tabs[2]:
        changes = get_recent_rating_changes()
        if changes is None:
            st.info(
                "💡 **For få snapshots til at vise ændringer.**\n\n"
                "👉 Kør screeneren minimum 2 gange (forskellige dage) for at se rating-ændringer her."
            )
        else:
            st.caption(f"🔄 Sammenligning: {changes['ts_prev'][:10]} → {changes['ts_now'][:10]}")
            up_count = len(changes["upgraded"])
            down_count = len(changes["downgraded"])
            cc = st.columns(2)
            cc[0].metric("📈 Opgraderet", up_count)
            cc[1].metric("📉 Nedgraderet", down_count)

            if up_count == 0 and down_count == 0:
                st.success("✅ Ingen rating-ændringer siden sidst")
            else:
                if up_count > 0:
                    st.markdown("#### 📈 Opgraderet (køb-vinduer åbner)")
                    up_disp = changes["upgraded"][[
                        c for c in ["ticker", "name", "recommendation_prev", "recommendation_now",
                                    "score_change", "price_change_%"]
                        if c in changes["upgraded"].columns
                    ]].rename(columns={
                        "recommendation_prev": "Var", "recommendation_now": "Nu",
                        "score_change": "Score Δ", "price_change_%": "Pris Δ%",
                    })
                    for col in ["Score Δ", "Pris Δ%"]:
                        if col in up_disp.columns:
                            up_disp[col] = pd.to_numeric(up_disp[col], errors="coerce").round(2)
                    st.dataframe(up_disp, use_container_width=True, hide_index=True)

                if down_count > 0:
                    st.markdown("#### 📉 Nedgraderet (overvej salg)")
                    down_disp = changes["downgraded"][[
                        c for c in ["ticker", "name", "recommendation_prev", "recommendation_now",
                                    "score_change", "price_change_%"]
                        if c in changes["downgraded"].columns
                    ]].rename(columns={
                        "recommendation_prev": "Var", "recommendation_now": "Nu",
                        "score_change": "Score Δ", "price_change_%": "Pris Δ%",
                    })
                    for col in ["Score Δ", "Pris Δ%"]:
                        if col in down_disp.columns:
                            down_disp[col] = pd.to_numeric(down_disp[col], errors="coerce").round(2)
                    st.dataframe(down_disp, use_container_width=True, hide_index=True)

    # 🆕 EARNINGS-KALENDER TAB
    with action_tabs[3]:
        st.markdown("### 📅 Earnings-kalender for din watchlist")
        st.caption(
            "Kommende earnings-rapporter sorteret efter dato. "
            "Kritiske advarsler er markeret med rødt."
        )

        if not st.session_state.watchlist:
            st.info(
                "💡 **Din watchlist er tom.**\n\n"
                "👉 Analyser nogle aktier først - de tilføjes automatisk."
            )
        else:
            render_watchlist_earnings_calendar(st.session_state.watchlist)

    st.markdown("---")
    st.markdown("### ⚡ Hurtige genveje")
    quick_cols = st.columns(4)
    if quick_cols[0].button("🔎 Kør screener nu", use_container_width=True, key="qg_screener"):
        st.session_state.active_view = "🔎 Screener"
        st.rerun()
    if quick_cols[1].button("🪙 Krypto-overblik", use_container_width=True, key="qg_crypto"):
        st.session_state.active_view = "🪙 Krypto"
        st.rerun()
    if quick_cols[2].button("🔍 Søg ticker", use_container_width=True, key="qg_search"):
        st.session_state.active_view = "🔍 Søg ticker"
        st.rerun()
    if quick_cols[3].button("📊 Detaljeret analyse", use_container_width=True, key="qg_analysis"):
        st.session_state.active_view = "📊 Analyse"
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ **Ikke finansiel rådgivning.** Dashboard er et analyseværktøj. "
        "Lav altid din egen research før investering. Past performance is not indicative of future results."
    )


# ============ SØGE-VIEW ============

elif st.session_state.active_view == "🔍 Søg ticker":
    st.subheader("🔍 Find ticker for et firma")
    query = st.text_input("Firmanavn", value="", key="search_query", placeholder="novo nordisk")
    if query and len(query) >= 2:
        with st.spinner("Søger..."):
            results = search_tickers(query)
        if results:
            st.success(f"Fandt {len(results)} resultater")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            cols = st.columns(min(4, len(results)))
            for i, r in enumerate(results[:8]):
                if cols[i % 4].button(f"📌 {r['symbol']}\n{r['name'][:25]}",
                                      key=f"sr_{i}", use_container_width=True):
                    goto_analysis(r["symbol"])
        else:
            st.warning("Ingen resultater fundet")
    st.markdown("---")
    examples = pd.DataFrame([
        {"Firma": "Novo Nordisk", "Yahoo": "NOVO-B.CO", "ADR": "NVO"},
        {"Firma": "Mærsk B", "Yahoo": "MAERSK-B.CO", "ADR": "AMKBY"},
        {"Firma": "Ørsted", "Yahoo": "ORSTED.CO", "ADR": "DNNGY"},
        {"Firma": "DSV", "Yahoo": "DSV.CO", "ADR": "DSDVF"},
        {"Firma": "Apple", "Yahoo": "AAPL", "ADR": "AAPL"},
        {"Firma": "ASML", "Yahoo": "ASML.AS", "ADR": "ASML"},
    ])
    st.dataframe(examples, use_container_width=True, hide_index=True)


# ============ DIAGNOSE-VIEW ============

elif st.session_state.active_view == "🔧 Diagnose":
    st.subheader("🔧 Diagnose - Test datakilder & ML")

    diag_tabs = st.tabs([
    "🌐 Data sources",
    "🤖 ML Data Pipeline",
    "🚀 Backfill (genvej)"
])

    # ===== TAB 1: Original data source diagnose =====
    with diag_tabs[0]:
        diag_ticker = st.text_input(
            "Test ticker", value="AAPL", key="diag_ticker"
        ).strip().upper()
        if st.button("🔍 Kør diagnose", type="primary", key="btn_diag_sources"):
            with st.spinner(f"Tester for {diag_ticker}..."):
                results = run_diagnostics(diag_ticker)
            for source, status, time_taken, details in results:
                with st.expander(f"{status} **{source}** ({time_taken})", expanded=True):
                    st.code(details)

    # ===== TAB 2: ML Data Pipeline test =====
    with diag_tabs[1]:
        st.markdown("### 🤖 ML Data Pipeline Test")
        st.caption(
            "Tester om ML data pipelinen kan læse dine snapshots og forberede "
            "training data til machine learning modellen."
        )
            # ===== TAB 3: ML Backfill =====
    with diag_tabs[2]:
        st.markdown("### 🚀 ML Backfill — Generer historisk training data")
        st.caption(
            "🪄 **Tidsmaskine!** Genererer 'snapshots' bagudrettet fra historisk pris-data, "
            "så du kan træne ML modellen i dag — uden at vente 6 måneder på forward returns."
        )

        st.info(
            "💡 **Sådan virker det:**\n"
            "1. Vi tager populære tickers (AAPL, MSFT, NVDA, etc.)\n"
            "2. Går tilbage i tiden (fx 24 måneder)\n"
            "3. På hver dato beregner vi scores AS IF vi havde screenet dengang\n"
            "4. Vi bruger den NUVÆRENDE pris til at beregne forward returns\n"
            "5. Bom! Hundredevis af training samples på 2-5 minutter ⚡"
        )

        bf_cols = st.columns(3)
        bf_asset = bf_cols[0].selectbox(
            "Asset class", ["stock", "crypto"], key="bf_asset"
        )
        bf_months = bf_cols[1].slider(
            "Måneder tilbage", 12, 36, 24, 6, key="bf_months",
            help="Hvor langt tilbage i tiden skal vi generere snapshots?"
        )
        bf_interval = bf_cols[2].slider(
            "Dage mellem snapshots", 14, 60, 30, key="bf_interval",
            help="Mindre = flere samples men mere overlap"
        )

        # Estimated samples
        n_dates = (bf_months * 30 - 200) // bf_interval
        n_tickers_est = 50 if bf_asset == "stock" else 15
        n_samples_est = n_dates * n_tickers_est

        st.caption(
            f"📊 **Forventet output:** ~{n_dates} snapshot-datoer × ~{n_tickers_est} tickers "
            f"= ~{n_samples_est} samples (afhængigt af data-kvalitet)"
        )

        if st.button(
            "🚀 KØR BACKFILL",
            type="primary",
            use_container_width=True,
            key="btn_run_backfill"
        ):
            try:
                from ml_backfill import build_backfill_dataset, save_backfill_as_snapshots, DEFAULT_TICKERS

                # Choose tickers based on asset class
                if bf_asset == "crypto":
                    tickers = DEFAULT_TICKERS["crypto"]
                else:
                    tickers = (
                        DEFAULT_TICKERS["us_large_cap"]
                        + DEFAULT_TICKERS["us_growth"][:10]
                        + DEFAULT_TICKERS["european"][:10]
                    )
                    tickers = list(set(tickers))

                progress = st.progress(0, text="Starter backfill...")
                status_box = st.empty()

                def update_progress(current, total, ticker):
                    progress.progress(
                        current / total,
                        text=f"📈 [{current}/{total}] Henter {ticker}..."
                    )

                status_box.info(f"⏳ Genererer historisk data for {len(tickers)} tickers...")

                with st.spinner(""):
                    result = build_backfill_dataset(
                        tickers=tickers,
                        asset_class=bf_asset,
                        months_back=bf_months,
                        snapshot_interval_days=bf_interval,
                        progress_callback=update_progress,
                    )

                progress.empty()

                if "error" in result:
                    status_box.empty()
                    st.error(f"❌ {result['error']}")
                else:
                    df = result["df"]
                    status_box.success(f"✅ Genereret {result['n_rows']} samples!")

                    # Stats
                    stats_cols = st.columns(4)
                    stats_cols[0].metric("📊 Total samples", result['n_rows'])
                    stats_cols[1].metric("🏷️ Tickers", result['n_tickers'])
                    stats_cols[2].metric("📅 Snapshots", result['n_snapshots'])

                    # Forward returns coverage
                    valid_30 = df["future_return_30d"].notna().sum() if "future_return_30d" in df.columns else 0
                    stats_cols[3].metric("✅ 30d valid", valid_30)

                    # Detail per horizon
                    st.markdown("##### 📅 Coverage per horisont")
                    cov_data = []
                    for h in [30, 90, 180]:
                        col = f"future_return_{h}d"
                        if col in df.columns:
                            valid = df[col].notna().sum()
                            avg_ret = df[col].mean() if valid > 0 else 0
                            cov_data.append({
                                "Horisont": f"{h} dage",
                                "Valid samples": valid,
                                "Coverage %": f"{valid/len(df)*100:.0f}%",
                                "Gns. afkast": f"{avg_ret:+.2f}%",
                            })
                    st.dataframe(pd.DataFrame(cov_data), use_container_width=True, hide_index=True)

                                        # 🆕 STORE IN SESSION STATE (Streamlit Cloud workaround)
                    from ml_backfill import store_backfill_in_session
                    store_backfill_in_session(df)

                    st.markdown("---")
                    st.success(
                        "✅ **Data gemt automatisk i session!** ML pipelinen kan nu læse den direkte. "
                        "Gå til **🤖 ML Data Pipeline** → klik **'Hent oversigt'** → "
                        "så kan du træne modellen 🚀"
                    )
                    st.balloons()

                    st.markdown("##### 💾 Optional: Gem også som CSV (lokalt)")
                    save_cols = st.columns([3, 1])
                    save_cols[0].caption(
                        "ℹ️ På Streamlit Cloud forsvinder CSV-filer ved rebuild. "
                        "Brug session state (allerede gemt!) til træning. "
                        "CSV-gem er kun nyttigt hvis du kører lokalt."
                    )

                    if save_cols[1].button(
                        "💾 GEM CSV",
                        use_container_width=True,
                        key="btn_save_backfill_csv"
                    ):
                        try:
                            n_saved = save_backfill_as_snapshots(df)
                            st.info(f"📁 Gemt {n_saved} CSV-filer (kun lokalt nyttigt)")
                        except Exception as e:
                            st.warning(f"⚠️ CSV-gem fejlede: {e}")

                    # Preview
                    with st.expander("👀 Preview af data"):
                        st.dataframe(df.head(20), use_container_width=True)

            except ImportError as e:
                st.error(f"❌ Kunne ikke importere ml_backfill: {e}")
                st.info("💡 Tjek at `ml_backfill.py` er gemt i samme mappe som `app.py`")
            except Exception as e:
                st.error(f"❌ Backfill fejlede: {e}")
                import traceback
                with st.expander("🐛 Full traceback"):
                    st.code(traceback.format_exc())

        # ---- Quick summary ----
        st.markdown("#### 📊 Tilgængelig data")
        if st.button("🔍 Hent oversigt", key="btn_ml_summary"):
            try:
                from ml_data import get_training_summary
                with st.spinner("Læser snapshots..."):
                    summary = get_training_summary()

                if not summary:
                    st.warning("⚠️ Ingen snapshots fundet")
                else:
                    for asset_class, stats in summary.items():
                        emoji = "📈" if asset_class == "stock" else "🪙"
                        with st.expander(f"{emoji} {asset_class.upper()}", expanded=True):
                            if "error" in stats:
                                st.error(f"Fejl: {stats['error']}")
                            elif stats.get("snapshots", 0) == 0:
                                st.info(f"Ingen {asset_class}-snapshots fundet")
                            else:
                                cols = st.columns(4)
                                cols[0].metric("📸 Snapshots", stats["snapshots"])
                                cols[1].metric("📋 Rows", stats["rows"])
                                cols[2].metric("🏷️ Tickers", stats["tickers"])
                                cols[3].metric(
                                    "📅 Date range",
                                    "OK",
                                    f"{stats.get('date_min', '?')} → {stats.get('date_max', '?')}"
                                )
                                if stats.get("universes"):
                                    st.caption(
                                        f"🌍 Universer: {', '.join(stats['universes'][:5])}"
                                        + (f" + {len(stats['universes'])-5} flere" if len(stats['universes']) > 5 else "")
                                    )
            except ImportError as e:
                st.error(f"❌ Kunne ikke importere ml_data.py: {e}")
                st.info("💡 Tjek at ml_data.py er gemt i samme mappe som app.py")
            except Exception as e:
                st.error(f"❌ Fejl: {e}")

        st.markdown("---")

        # ---- Full pipeline test ----
        st.markdown("#### 🚀 Fuld pipeline test")
        st.caption(
            "Kører hele pipelinen: snapshots → forward returns → features → training data. "
            "**Kan tage 1-3 minutter** afhængigt af antal snapshots."
        )

        ml_test_cols = st.columns(2)
        ml_asset = ml_test_cols[0].selectbox(
            "Asset class", ["stock", "crypto"], key="ml_asset_diag"
        )

        if ml_test_cols[1].button(
            "🚀 Kør ML data pipeline test",
            type="primary",
            use_container_width=True,
            key="btn_ml_pipeline"
        ):
            try:
                from ml_data import get_training_data, HORIZONS

                progress = st.progress(0, text="Starter pipeline...")
                status_box = st.empty()

                status_box.info("📊 Læser snapshots & beregner forward returns (tager længst)...")
                progress.progress(20, text="Læser snapshots...")

                with st.spinner(""):
                    data = get_training_data(asset_class=ml_asset, verbose=False)

                progress.progress(90, text="Færdiggør...")

                if "error" in data:
                    progress.empty()
                    status_box.empty()
                    st.error(f"❌ {data['error']}")
                    st.info(
                        "💡 **Mulige løsninger:**\n"
                        f"- Kør screener på **{'kryptos' if ml_asset == 'crypto' else 'aktier'}** først\n"
                        "- Gem mindst 2-3 snapshots\n"
                        "- Vent et par dage før du kører pipelinen (så forward returns kan beregnes)"
                    )
                else:
                    progress.progress(100, text="Færdig!")
                    progress.empty()
                    status_box.empty()

                    st.success(f"✅ Pipeline kørte succesfuldt for **{ml_asset}**!")

                    # ---- Top metrics ----
                    top_cols = st.columns(4)
                    top_cols[0].metric("🔢 Features", data["n_features"])
                    top_cols[1].metric("📋 Total rows", data["total_rows_loaded"])
                    top_cols[2].metric("📊 30d samples", data.get("n_samples_30d", 0))
                    top_cols[3].metric("📈 90d samples", data.get("n_samples_90d", 0))

                    # ---- Per-horizon breakdown ----
                    st.markdown("##### 📅 Per horisont")
                    hor_data = []
                    for h in HORIZONS:
                        n = data.get(f"n_samples_{h}d", 0)
                        y_clf_key = f"y_clf_{h}d"
                        if y_clf_key in data:
                            class_dist = data[y_clf_key].value_counts().to_dict()
                            buy_n = class_dist.get("BUY", 0)
                            hold_n = class_dist.get("HOLD", 0)
                            sell_n = class_dist.get("SELL", 0)
                        else:
                            buy_n = hold_n = sell_n = 0

                        if n >= 100:
                            status = "✅ Robust"
                        elif n >= 50:
                            status = "🟡 OK"
                        elif n >= 20:
                            status = "🟠 Lav"
                        else:
                            status = "🔴 For lidt"

                        hor_data.append({
                            "Horisont": f"{h} dage",
                            "Total samples": n,
                            "🟢 BUY": buy_n,
                            "🟡 HOLD": hold_n,
                            "🔴 SELL": sell_n,
                            "Status": status,
                        })

                    st.dataframe(
                        pd.DataFrame(hor_data),
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---- Feature columns preview ----
                    with st.expander("🧬 Feature columns (alle features ML modellen ser)"):
                        feat_cols = data.get("feature_columns", [])
                        st.write(f"**Antal features:** {len(feat_cols)}")
                        numeric_feats = [c for c in feat_cols if not any(
                            c.startswith(p) for p in ["sector_", "country_", "regime_", "currency_"]
                        )]
                        categorical_feats = [c for c in feat_cols if any(
                            c.startswith(p) for p in ["sector_", "country_", "regime_", "currency_"]
                        )]

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**📊 Numeriske ({len(numeric_feats)}):**")
                            for f in numeric_feats:
                                st.caption(f"• {f}")
                        with col_b:
                            st.markdown(f"**🏷️ Kategoriske ({len(categorical_feats)}):**")
                            for f in categorical_feats[:20]:
                                st.caption(f"• {f}")
                            if len(categorical_feats) > 20:
                                st.caption(f"... og {len(categorical_feats)-20} flere")

                    # ---- Sample data preview ----
                    with st.expander("👀 Sample data (første 5 rows)"):
                        sample = data.get("sample_data")
                        if sample is not None and not sample.empty:
                            st.dataframe(sample, use_container_width=True)
                        else:
                            st.info("Ingen sample data")

                    # ---- Recommendation ----
                    st.markdown("##### 🎯 Klar til ML-træning?")
                    samples_30d = data.get("n_samples_30d", 0)

                    if samples_30d >= 100:
                        st.success(
                            f"✅ **Du er klar!** {samples_30d} samples på 30d horisont "
                            f"er nok til at træne en robust ML model. "
                            f"Vi kan gå videre til **FASE 2: Træn modellen**."
                        )
                    elif samples_30d >= 50:
                        st.info(
                            f"🟡 **OK at fortsætte.** {samples_30d} samples er nok til "
                            f"en basal model, men flere data ville give bedre resultater."
                        )
                    elif samples_30d >= 20:
                        st.warning(
                            f"🟠 **For få samples.** Med kun {samples_30d} samples vil "
                            f"modellen være ustabil. Anbefalet: Kør screeneren på "
                            f"**3-5 forskellige universer** først."
                        )
                    else:
                        st.error(
                            f"🔴 **Ikke nok data!** {samples_30d} samples er for lidt. "
                            f"Du skal mindst have **20+ samples**. Kør screeneren først."
                        )

            except ImportError as e:
                st.error(f"❌ Kunne ikke importere ml_data: {e}")
                st.info(
                    "💡 **Tjek:**\n"
                    "1. `ml_data.py` er gemt i samme mappe som `app.py`\n"
                    "2. ML pakker er installeret: `scikit-learn`, `xgboost`, `lightgbm`, `joblib`\n"
                    "3. Push til GitHub og vent på rebuild"
                )
            except Exception as e:
                st.error(f"❌ Pipeline fejlede: {e}")
                import traceback
                with st.expander("🐛 Full traceback"):
                    st.code(traceback.format_exc())
                # ============ SCREENER-VIEW ============

elif st.session_state.active_view == "🔎 Screener":
    st.subheader("🔎 Markedsscreener")
    st.caption("Find gode købsmuligheder · Sammenlign over tid · Sektoranalyse · Hot stocks")

    sc_modes = st.tabs([
        "🚀 Kør screener", "📊 Sektor-breakdown", "🔔 Sammenlign",
        "🔥 Hot stocks", "📜 Historik",
    ])

    # ===== MODE 1: KØR SCREENER =====
    with sc_modes[0]:
        sc1, sc2, sc3 = st.columns([2, 1, 1])
        universe_name = sc1.selectbox(
            "🌍 Vælg univers",
            list(SCREENER_UNIVERSES.keys()) + ["✏️ Custom liste"],
            key="screener_universe",
        )
        min_score = sc2.slider("📊 Min. score", 30, 90, 60, key="sc_min_score")
        max_workers = sc3.slider("⚡ Workers", 1, 8, 4, key="sc_workers")

        if universe_name == "✏️ Custom liste":
            custom_text = st.text_area(
                "Tickers (én per linje eller komma-separeret)",
                value="AAPL\nMSFT\nGOOGL\nNVDA\nTSLA", height=150,
            )
            tickers = [t.strip().upper() for t in custom_text.replace(",", "\n").split("\n") if t.strip()]
        else:
            tickers = SCREENER_UNIVERSES[universe_name]

        st.info(f"📋 **{len(tickers)} tickers** · Estimeret tid: ~{len(tickers)*2/max_workers:.0f}s")

        col_run, col_snapshot, col_clear = st.columns([2, 1, 1])

        if col_run.button("🚀 Kør screener", type="primary", use_container_width=True):
            progress = st.progress(0, text="Starter...")

            def update_progress(done, total, ticker):
                progress.progress(done / total, text=f"Analyseret {done}/{total}: {ticker}")

            with st.spinner("Scanner marked..."):
                df_all, df_buys = run_screener(
                    tickers, min_score=min_score, max_workers=max_workers,
                    progress_callback=update_progress,
                )
            progress.empty()

            if not df_all.empty:
                save_snapshot(df_all, universe_name)

            st.session_state.screener_results = {
                "all": df_all, "buys": df_buys,
                "universe": universe_name, "min_score": min_score,
                "timestamp": pd.Timestamp.now(),
            }

        if col_snapshot.button("💾 Gem snapshot", use_container_width=True):
            if st.session_state.screener_results:
                fp = save_snapshot(
                    st.session_state.screener_results["all"],
                    st.session_state.screener_results["universe"]
                )
                st.success(f"✅ Gemt: {fp.name if fp else 'fejl'}")
            else:
                st.warning("Ingen resultater at gemme")

        if col_clear.button("🗑️ Ryd", use_container_width=True):
            st.session_state.screener_results = None
            st.rerun()

        if st.session_state.screener_results:
            res = st.session_state.screener_results
            df_all = res["all"]
            df_buys = res["buys"]

            st.markdown("---")
            st.markdown(f"### 📊 Resultater: **{res['universe']}**")
            st.caption(f"⏱️ {res['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} · Min. score: {res['min_score']}")

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("📋 Total scannet", len(df_all))
            sm2.metric("✅ Succes", (df_all["status"] == "✅").sum())
            sm3.metric("🟢 Køb-muligheder", len(df_buys))
            if not df_buys.empty:
                sm4.metric("🏆 Topscore", f"{df_buys['overall'].max():.1f}")

            if df_buys.empty:
                st.warning(f"⚠️ Ingen aktier opfylder min. score {res['min_score']}.")
            else:
                st.markdown("### 🏆 Top købsmuligheder")
                top_display = df_buys.head(15).copy()
                display_cols = {
                    "ticker": "Ticker", "name": "Navn", "sector": "Sektor",
                    "price": "Pris", "currency": "Valuta", "change_%": "Ændr %",
                    "overall": "Score", "recommendation": "Anbefaling",
                    "rsi": "RSI", "vs_sma200_%": "vs SMA200%",
                    "vs_52w_high_%": "vs 52w-H%", "pe": "P/E",
                    "dividend_%": "Div%", "dcf_upside_%": "DCF↑%",
                }
                top_display = top_display[
                    [c for c in display_cols if c in top_display.columns]
                ].rename(columns=display_cols)
                for col in ["Pris", "Ændr %", "Score", "RSI", "vs SMA200%",
                            "vs 52w-H%", "P/E", "Div%", "DCF↑%"]:
                    if col in top_display.columns:
                        top_display[col] = pd.to_numeric(top_display[col], errors="coerce").round(2)

                st.dataframe(
                    top_display, use_container_width=True, hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=100, format="%.0f"
                        ),
                    },
                )

                st.markdown("#### 👆 Klik for fuld analyse:")
                top_n = min(8, len(df_buys))
                cols = st.columns(4)
                for i, (_, row) in enumerate(df_buys.head(top_n).iterrows()):
                    if cols[i % 4].button(
                        f"📊 {row['ticker']} {row['overall']:.0f}/100",
                        key=f"sc_{row['ticker']}", use_container_width=True
                    ):
                        goto_analysis(row["ticker"])

                st.markdown("---")
                st.markdown("### 🎯 Købsmuligheder pr. kategori")
                categories = categorize_opportunities(df_buys)
                if categories:
                    for cat_name, cat_df in categories.items():
                        with st.expander(f"{cat_name} ({len(cat_df)} aktier)"):
                            cat_d = cat_df[
                                [c for c in display_cols if c in cat_df.columns]
                            ].rename(columns=display_cols)
                            for col in ["Pris", "Ændr %", "Score", "RSI"]:
                                if col in cat_d.columns:
                                    cat_d[col] = pd.to_numeric(cat_d[col], errors="coerce").round(2)
                            st.dataframe(cat_d, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### 📈 Score-distribution")
                fig = px.scatter(
                    df_all[df_all["status"] == "✅"],
                    x="f_score", y="t_score", size="overall", color="overall",
                    hover_data=["ticker", "name", "recommendation"],
                    color_continuous_scale="RdYlGn",
                    title="Fundamental vs Teknisk score",
                    labels={"f_score": "Fundamental score", "t_score": "Teknisk score"},
                )
                fig.add_hline(y=60, line_dash="dash", line_color="green", opacity=0.3)
                fig.add_vline(x=60, line_dash="dash", line_color="green", opacity=0.3)
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                csv = df_buys.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download købsmuligheder (CSV)", csv,
                    f"screener_{res['timestamp'].strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                )
        else:
            st.info("👆 Vælg et univers og tryk **🚀 Kør screener**")

    # ===== MODE 2: SEKTOR-BREAKDOWN =====
    with sc_modes[1]:
        st.markdown("### 📊 Sektor-breakdown")
        st.caption("Grupperer screening-resultater per sektor og viser bedste pr. sektor")

        if not st.session_state.screener_results:
            st.info("👈 Kør først en screener i fanen **🚀 Kør screener**")
        else:
            df_all = st.session_state.screener_results["all"]
            sectors = sector_breakdown(df_all)

            if not sectors:
                st.warning("⚠️ Ingen sektor-info tilgængelig (fungerer bedst med Yahoo+Finnhub data)")
            else:
                st.markdown("#### 🏆 Sektor-rangering")
                rank_df = pd.DataFrame([
                    {
                        "Sektor": name, "Antal aktier": s["count"],
                        "Gns. score": round(s["avg_score"], 1),
                        "Bedste score": round(s["best_score"], 1),
                        "Top pick": s["top_pick"]["ticker"],
                        "Top pick navn": str(s["top_pick"]["name"])[:30],
                    }
                    for name, s in sectors.items()
                ])

                st.dataframe(
                    rank_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Gns. score": st.column_config.ProgressColumn(
                            "Gns. score", min_value=0, max_value=100, format="%.0f"
                        ),
                    },
                )

                fig_sec = px.bar(
                    rank_df.sort_values("Gns. score", ascending=True),
                    x="Gns. score", y="Sektor", orientation="h",
                    color="Gns. score", color_continuous_scale="RdYlGn",
                    title="Gennemsnitlig score per sektor", text="Antal aktier",
                )
                fig_sec.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_sec, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🔍 Top 3 pr. sektor")
                for name, s in sectors.items():
                    with st.expander(f"**{name}** · {s['count']} aktier · gns. {s['avg_score']:.1f}"):
                        top3 = s["all_stocks"].head(3)
                        for _, row in top3.iterrows():
                            cols = st.columns([1, 2, 1, 1, 1, 1])
                            cols[0].markdown(f"**{row['ticker']}**")
                            cols[1].caption(str(row.get("name", ""))[:40])
                            cols[2].metric("Score", f"{row['overall']:.0f}")
                            cols[3].metric(
                                "RSI",
                                f"{row['rsi']:.0f}" if pd.notna(row.get('rsi')) else "-"
                            )
                            cols[4].markdown(f"_{row.get('recommendation', '')}_")
                            if cols[5].button("📊", key=f"sec_btn_{row['ticker']}"):
                                goto_analysis(row["ticker"])

    # ===== MODE 3: SAMMENLIGN =====
    with sc_modes[2]:
        st.markdown("### 🔔 Sammenlign med tidligere snapshots")
        st.caption("Find aktier der har ændret rating, fået højere/lavere score siden sidst")

        all_snaps = list_snapshots()
        if len(all_snaps) < 2:
            st.warning(f"⚠️ Du har kun {len(all_snaps)} snapshot(s). Du skal have mindst 2 for at sammenligne.")
        else:
            unique_universes = list(set(s["universe"] for s in all_snaps))
            sel_universe = st.selectbox("🌍 Univers", unique_universes, key="cmp_universe")
            uni_snaps = [s for s in all_snaps if s["universe"] == sel_universe]

            if len(uni_snaps) < 2:
                st.warning(f"Kun 1 snapshot for {sel_universe}")
            else:
                cmp1, cmp2 = st.columns(2)
                snap_now_idx = cmp1.selectbox(
                    "📅 NU (senest)", range(len(uni_snaps)),
                    format_func=lambda i: uni_snaps[i]["filename"],
                    index=0, key="cmp_now",
                )
                snap_prev_idx = cmp2.selectbox(
                    "📅 FØR (sammenlign med)", range(len(uni_snaps)),
                    format_func=lambda i: uni_snaps[i]["filename"],
                    index=min(1, len(uni_snaps)-1), key="cmp_prev",
                )

                if st.button("🔍 Sammenlign", type="primary"):
                    df_now, ts_now, _ = load_snapshot(uni_snaps[snap_now_idx]["file"])
                    df_prev, ts_prev, _ = load_snapshot(uni_snaps[snap_prev_idx]["file"])
                    cmp = compare_snapshots(df_now, df_prev)

                    if cmp.empty:
                        st.error("Kunne ikke sammenligne")
                    else:
                        st.success(f"✅ Sammenlignet {ts_prev[:10]} → {ts_now[:10]}")

                        if "rating_changed" in cmp.columns:
                            changed = cmp[cmp["rating_changed"] == True].copy()
                            st.markdown("#### 🚨 Rating-ændringer")
                            if changed.empty:
                                st.info("Ingen aktier har ændret rating")
                            else:
                                rec_order = ["STÆRKT KØB", "KØB", "HOLD", "SÆLG", "STÆRKT SÆLG"]
                                changed["was_idx"] = changed["recommendation_prev"].apply(
                                    lambda x: rec_order.index(x) if x in rec_order else 99
                                )
                                changed["now_idx"] = changed["recommendation_now"].apply(
                                    lambda x: rec_order.index(x) if x in rec_order else 99
                                )
                                upgraded = changed[changed["now_idx"] < changed["was_idx"]]
                                downgraded = changed[changed["now_idx"] > changed["was_idx"]]

                                if not upgraded.empty:
                                    st.markdown("##### 🟢 Opgraderet")
                                    st.dataframe(
                                        upgraded[[
                                            "ticker", "name", "recommendation_prev", "recommendation_now",
                                            "score_change", "price_change_%"
                                        ]].rename(columns={
                                            "recommendation_prev": "Var", "recommendation_now": "Nu",
                                            "score_change": "Score Δ", "price_change_%": "Pris Δ%",
                                        }).round(2),
                                        use_container_width=True, hide_index=True,
                                    )

                                if not downgraded.empty:
                                    st.markdown("##### 🔴 Nedgraderet")
                                    st.dataframe(
                                        downgraded[[
                                            "ticker", "name", "recommendation_prev", "recommendation_now",
                                            "score_change", "price_change_%"
                                        ]].rename(columns={
                                            "recommendation_prev": "Var", "recommendation_now": "Nu",
                                            "score_change": "Score Δ", "price_change_%": "Pris Δ%",
                                        }).round(2),
                                        use_container_width=True, hide_index=True,
                                    )

                        st.markdown("#### 📈 Største score-stigninger")
                        if "score_change" in cmp.columns:
                            risers = cmp.dropna(subset=["score_change"]).nlargest(10, "score_change")
                            st.dataframe(
                                risers[[
                                    "ticker", "name", "overall_now", "overall_prev",
                                    "score_change", "price_change_%"
                                ]].round(2),
                                use_container_width=True, hide_index=True,
                            )

                            st.markdown("#### 📉 Største score-fald")
                            fallers = cmp.dropna(subset=["score_change"]).nsmallest(10, "score_change")
                            st.dataframe(
                                fallers[[
                                    "ticker", "name", "overall_now", "overall_prev",
                                    "score_change", "price_change_%"
                                ]].round(2),
                                use_container_width=True, hide_index=True,
                            )

    # ===== MODE 4: HOT STOCKS =====
    with sc_modes[3]:
        st.markdown("### 🔥 Hot stocks - stigende score over tid")
        st.caption("Find aktier hvor scoren er steget de sidste N snapshots.")

        all_snaps = list_snapshots()
        if len(all_snaps) < 2:
            st.warning("⚠️ Du har brug for mindst 2 snapshots.")
        else:
            unique_universes = list(set(s["universe"] for s in all_snaps))
            hot_universe = st.selectbox("🌍 Univers", unique_universes, key="hot_universe")

            hcol1, hcol2 = st.columns(2)
            n_days = hcol1.slider("📅 Antal snapshots tilbage", 2, 20, 5, key="hot_n_days")
            min_change = hcol2.slider("📊 Min. score-ændring", 1, 30, 5, key="hot_min_change")

            if st.button("🔥 Find hot stocks", type="primary"):
                hot = get_hot_stocks(hot_universe, n_days=n_days, min_change=min_change)
                risers = hot.get("risers") if isinstance(hot, dict) else None
                fallers = hot.get("fallers") if isinstance(hot, dict) else None
                risers_empty = risers is None or risers.empty
                fallers_empty = fallers is None or fallers.empty

                if not isinstance(hot, dict) or (risers_empty and fallers_empty):
                    st.warning("Ingen aktier opfylder kriterierne. Prøv at sænke min. ændring.")
                else:
                    st.caption(
                        f"📅 Fra {hot['oldest_ts'][:10]} → {hot['latest_ts'][:10]} "
                        f"({hot['n_snapshots']} snapshots)"
                    )

                    if not risers_empty:
                        st.markdown("#### 🚀 Stigende score (risers)")
                        risers_disp = hot["risers"][[
                            c for c in [
                                "ticker", "name", "overall_now", "overall_old",
                                "score_change", "recommendation", "price_change_%"
                            ] if c in hot["risers"].columns
                        ]].copy()
                        for col in ["overall_now", "overall_old", "score_change", "price_change_%"]:
                            if col in risers_disp.columns:
                                risers_disp[col] = pd.to_numeric(risers_disp[col], errors="coerce").round(2)
                        st.dataframe(risers_disp, use_container_width=True, hide_index=True)

                        fig_hot = px.bar(
                            hot["risers"].head(15),
                            x="score_change", y="ticker", orientation="h",
                            color="score_change", color_continuous_scale="Greens",
                            title="Top 15 score-stigninger",
                        )
                        fig_hot.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig_hot, use_container_width=True)

                        st.markdown("##### 👆 Hurtig analyse:")
                        hot_cols = st.columns(4)
                        for i, (_, row) in enumerate(hot["risers"].head(8).iterrows()):
                            if hot_cols[i % 4].button(
                                f"📊 {row['ticker']} (+{row['score_change']:.0f})",
                                key=f"hot_{row['ticker']}", use_container_width=True,
                            ):
                                goto_analysis(row["ticker"])

                    if not fallers_empty:
                        with st.expander(f"📉 Faldende score ({len(fallers)} aktier)"):
                            fallers_disp = fallers.copy()
                            for col in ["overall_now", "overall_old", "score_change", "price_change_%"]:
                                if col in fallers_disp.columns:
                                    fallers_disp[col] = pd.to_numeric(fallers_disp[col], errors="coerce").round(2)
                            st.dataframe(fallers_disp, use_container_width=True, hide_index=True)

    # ===== MODE 5: HISTORIK =====
    with sc_modes[4]:
        st.markdown("### 📜 Snapshot-historik")
        st.caption("Alle gemte screeninger.")

        all_snaps = list_snapshots()
        if not all_snaps:
            st.info("Ingen snapshots gemt endnu.")
        else:
            st.success(f"✅ {len(all_snaps)} snapshots gemt")
            snap_df = pd.DataFrame([
                {
                    "Filnavn": s["filename"], "Univers": s["universe"],
                    "Tidspunkt": s["timestamp"][:19], "Antal tickers": s["n_tickers"],
                }
                for s in all_snaps
            ])
            st.dataframe(snap_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 📈 Score-historik for én ticker")
            track_ticker = st.text_input(
                "Ticker at tracke", value="AAPL", key="track_ticker"
            ).strip().upper()

            if track_ticker:
                hist_df = get_score_history(track_ticker, n_days=30)
                if hist_df.empty:
                    st.warning(f"Ingen historik for {track_ticker}")
                else:
                    fig_track = go.Figure()
                    fig_track.add_trace(go.Scatter(
                        x=hist_df["timestamp"], y=hist_df["score"],
                        mode="lines+markers", name="Score",
                        line=dict(color="#00d4aa", width=3),
                    ))
                    fig_track.update_layout(
                        title=f"Score-udvikling for {track_ticker}",
                        yaxis_title="Score (0-100)",
                        template="plotly_dark", height=400, yaxis_range=[0, 100],
                    )
                    fig_track.add_hline(y=60, line_dash="dash", line_color="green", opacity=0.4)
                    fig_track.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.4)
                    st.plotly_chart(fig_track, use_container_width=True)
                    st.dataframe(hist_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 🧹 Vedligeholdelse")
            cl1, cl2 = st.columns(2)
            cleanup_days = cl1.slider("Slet snapshots ældre end (dage)", 7, 365, 60, key="cleanup_days")
            if cl2.button("🗑️ Ryd op", use_container_width=True):
                deleted = cleanup_old_snapshots(cleanup_days)
                st.success(f"Slettet {deleted} gamle snapshots")
                time.sleep(1)
                st.rerun()
                # ============ KRYPTO-VIEW ============

elif st.session_state.active_view == "🪙 Krypto":
    st.subheader("🪙 Krypto Dashboard - Pro Edition")
    st.caption("Real-time data · Multi-faktor scoring · Risk management · Backtest")

    global_data = fetch_global_crypto_market()
    fg_df = fetch_fear_greed()

    if global_data:
        gc = st.columns(5)
        mc = global_data.get('total_market_cap_usd') or 0
        mc_change = global_data.get('market_cap_change_24h') or 0
        if mc > 0:
            gc[0].metric("💰 Total Market Cap", f"${mc/1e12:.2f}T", f"{mc_change:+.2f}%")
        else:
            gc[0].metric("💰 Total Market Cap", "N/A")

        vol = global_data.get('total_volume_usd') or 0
        if vol > 0:
            gc[1].metric("📊 24h Volume", f"${vol/1e9:.1f}B")
        else:
            gc[1].metric("📊 24h Volume", "N/A")

        btc_dom = global_data.get('btc_dominance') or 0
        gc[2].metric("👑 BTC Dominance", f"{btc_dom:.1f}%" if btc_dom else "N/A")

        eth_dom = global_data.get('eth_dominance') or 0
        gc[3].metric("⚡ ETH Dominance", f"{eth_dom:.1f}%" if eth_dom else "N/A")

        if fg_df is not None and not fg_df.empty:
            try:
                fg_value = int(fg_df["value"].iloc[-1])
                fg_label = fg_df["value_classification"].iloc[-1]
                fg_color = "🔴" if fg_value < 25 else "🟢" if fg_value > 75 else "🟡"
                gc[4].metric(f"{fg_color} Fear & Greed", f"{fg_value}/100", fg_label)
            except Exception:
                gc[4].metric("😱 Fear & Greed", "N/A")
        else:
            gc[4].metric("😱 Fear & Greed", "N/A")
    else:
        st.info(
            "⏳ Globale markedsdata kan ikke hentes lige nu (CoinGecko rate-limit). "
            "Prøv igen om 1-2 minutter eller fortsæt nedenfor — analysefunktionerne virker stadig."
        )

    st.markdown("---")

    crypto_tabs = st.tabs([
        "🎯 Pro Analyse", "🔎 Screener", "🔥 Trending",
        "📈 Sammenlign", "😱 Sentiment", "⛓️ On-Chain (BTC)",
    ])

    # ===== TAB 1: PRO ANALYSE =====
    with crypto_tabs[0]:
        st.markdown("### 🎯 Vælg krypto til analyse")

        input_method = st.radio(
            "Vælg metode:",
            ["📋 Vælg fra liste", "✏️ Skriv ticker selv"],
            horizontal=True, key="crypto_input_method"
        )

        if input_method == "📋 Vælg fra liste":
            ac1, ac2 = st.columns([3, 1])
            crypto_choice = ac1.selectbox(
                "Vælg krypto",
                options=list(CRYPTO_UNIVERSE.keys()),
                format_func=lambda x: f"{x} - {CRYPTO_UNIVERSE[x]['category']}",
                key="crypto_pro_select",
            )
            if ac2.button("🔍 Fuld Analyse", type="primary", use_container_width=True, key="btn_analyze_list"):
                st.session_state["crypto_analyzed"] = crypto_choice
        else:
            ac1, ac2 = st.columns([3, 1])
            custom_ticker = ac1.text_input(
                "🪙 Skriv krypto-ticker (fx BTC, ETH, DOGE, SHIB, PEPE, WIF, BONK)",
                value="", key="custom_crypto_ticker", placeholder="DOGE"
            ).strip().upper()
            st.caption(
                "💡 **Tips:** Brug standard symboler som BTC, ETH, SOL, ADA, DOGE, "
                "SHIB, PEPE, AVAX, LINK, WIF, BONK, FLOKI, TRUMP osv."
            )
            if ac2.button("🔍 Analysér", type="primary", use_container_width=True, key="btn_analyze_custom"):
                if custom_ticker:
                    norm = normalize_crypto_ticker(custom_ticker)
                    st.session_state["crypto_analyzed"] = norm
                    st.success(f"🔍 Analyserer **{norm}**...")
                else:
                    st.warning("⚠️ Indtast venligst en ticker")

        st.markdown("##### 🔥 Populære coins (klik for instant analyse):")
        popular_extra = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "AVAX"]
        pop_cols = st.columns(len(popular_extra))
        for i, sym in enumerate(popular_extra):
            if pop_cols[i].button(sym, key=f"pop_{sym}", use_container_width=True):
                st.session_state["crypto_analyzed"] = sym
                st.rerun()

        st.markdown("---")

        if st.session_state.get("crypto_analyzed"):
            symbol = st.session_state["crypto_analyzed"]
            with st.spinner(f"Henter komplet data for {symbol}..."):
                cdata = fetch_crypto_data(symbol)

            if cdata is None:
                st.error(f"❌ Kunne ikke hente data for **{symbol}**")
                col_info, col_actions = st.columns([2, 1])
                with col_info:
                    st.info(
                        "💡 **Mulige årsager:**\n"
                        "- Tickeren findes ikke (tjek stavning)\n"
                        "- CoinGecko er midlertidigt rate-limited (vent 1-2 min)\n"
                        "- Coin er for ny / ikke listet"
                    )
                with col_actions:
                    if st.button("🔄 Ryd cache & prøv igen", use_container_width=True, key="retry_crypto"):
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("🗑️ Nulstil", use_container_width=True, key="reset_crypto"):
                        st.session_state.pop("crypto_analyzed", None)
                        st.rerun()
            else:
                info = cdata["info"]
                hist = cdata["hist"]
                price = info["currentPrice"]

                st.success(f"✅ Data fra: **{cdata['source']}** · {len(hist)} dage")

                category = (
                    CRYPTO_UNIVERSE[symbol]["category"]
                    if symbol in CRYPTO_UNIVERSE
                    else info.get("category", "Cryptocurrency")
                )

                st.markdown(f"## {info['longName']} ({info['symbol']})")
                st.caption(
                    f"🏢 {category} · 📅 {hist.index[0].date()} → {hist.index[-1].date()} · 💱 USD"
                )

                change_24h = info.get("change_24h", 0) or 0
                k = st.columns(7)
                k[0].metric(
                    "Pris",
                    f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
                    f"{change_24h:+.2f}%"
                )
                if info.get("marketCap"):
                    k[1].metric("Market Cap", f"${info['marketCap']/1e9:.2f}B")
                if info.get("marketCapRank"):
                    k[2].metric("Rank", f"#{info['marketCapRank']}")
                if info.get("ath"):
                    k[3].metric("ATH", f"${info['ath']:,.2f}", f"{info.get('ath_change_%', 0):.0f}%")
                if info.get("change_7d") is not None:
                    k[4].metric("7d", f"{info['change_7d']:+.1f}%")
                if info.get("change_30d") is not None:
                    k[5].metric("30d", f"{info['change_30d']:+.1f}%")
                if info.get("change_1y") is not None:
                    k[6].metric("1y", f"{info['change_1y']:+.1f}%")

                with st.spinner("Beregner multi-faktor scores..."):
                    scores = crypto_overall_score(info, hist)

                rec, color = crypto_recommendation(scores["overall"])

                st.markdown("---")
                st.markdown("### 🎯 Multi-faktor Analyse")

                sc = st.columns(5)
                sc[0].markdown(
                    f"<div style='background:{color}22;padding:1rem;border-radius:10px;"
                    f"border-left:4px solid {color};text-align:center'>"
                    f"<h3 style='color:{color};margin:0'>{rec}</h3>"
                    f"<h1 style='margin:0.3rem 0'>{scores['overall']:.0f}/100</h1>"
                    f"<small>Samlet score</small></div>",
                    unsafe_allow_html=True,
                )
                sc[1].metric("📊 Marked", f"{scores['market']:.0f}/100", "35% vægt")
                sc[2].metric("🔧 Teknisk", f"{scores['technical']:.0f}/100", "30% vægt")
                sc[3].metric("💬 Sentiment", f"{scores['sentiment']:.0f}/100", "20% vægt")
                sc[4].metric("👨‍💻 Developer", f"{scores['developer']:.0f}/100", "15% vægt")

                st.markdown("---")
                st.markdown("### 💰 Kursniveauer & Risk Management")
                st.caption("Baseret på ATR, Bollinger Bands og 90/365 dages high/low")

                targets = crypto_price_targets(hist, price, scores)
                if targets:
                    buy_low_pct = (targets["buy_low"] / price - 1) * 100
                    buy_high_pct = (targets["buy_high"] / price - 1) * 100
                    stop_pct = (targets["stop_loss"] / price - 1) * 100
                    short_pct = (targets["target_short"] / price - 1) * 100
                    long_pct = (targets["target_long"] / price - 1) * 100
                    moon_pct = (targets["target_moon"] / price - 1) * 100

                    pt = st.columns(6)
                    pt[0].markdown(
                        f"<div style='background:#16a34a22;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #16a34a;text-align:center'>"
                        f"<small>🟢 KØB ZONE</small>"
                        f"<h4 style='margin:0.3rem 0'>"
                        f"${targets['buy_low']:,.4f}<br>${targets['buy_high']:,.4f}</h4>"
                        f"<small>{buy_low_pct:+.1f}% til {buy_high_pct:+.1f}%</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    pt[1].markdown(
                        f"<div style='background:#0099ff22;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #0099ff;text-align:center'>"
                        f"<small>📍 NUVÆRENDE</small>"
                        f"<h4 style='margin:0.3rem 0'>${price:,.4f}</h4>"
                        f"<small>{change_24h:+.2f}% (24h)</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    pt[2].markdown(
                        f"<div style='background:#ef444422;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #ef4444;text-align:center'>"
                        f"<small>🛑 STOP LOSS</small>"
                        f"<h4 style='margin:0.3rem 0'>${targets['stop_loss']:,.4f}</h4>"
                        f"<small>{stop_pct:+.1f}% (3x ATR)</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    pt[3].markdown(
                        f"<div style='background:#eab30822;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #eab308;text-align:center'>"
                        f"<small>🎯 KORT (1-3m)</small>"
                        f"<h4 style='margin:0.3rem 0'>${targets['target_short']:,.4f}</h4>"
                        f"<small>{short_pct:+.1f}% (BB)</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    pt[4].markdown(
                        f"<div style='background:#22c55e22;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #22c55e;text-align:center'>"
                        f"<small>🚀 LANG (6-12m)</small>"
                        f"<h4 style='margin:0.3rem 0'>${targets['target_long']:,.4f}</h4>"
                        f"<small>{long_pct:+.1f}%</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    pt[5].markdown(
                        f"<div style='background:#a855f722;padding:0.8rem;border-radius:10px;"
                        f"border-left:4px solid #a855f7;text-align:center'>"
                        f"<small>🌙 MOON (12m+)</small>"
                        f"<h4 style='margin:0.3rem 0'>${targets['target_moon']:,.4f}</h4>"
                        f"<small>{moon_pct:+.1f}%</small>"
                        f"</div>", unsafe_allow_html=True
                    )

                    st.caption(
                        f"📊 90d range: ${targets['low_90d']:.4f} → ${targets['high_90d']:.4f} · "
                        f"365d: ${targets['low_365d']:.4f} → ${targets['high_365d']:.4f} · "
                        f"ATR: ${targets['atr']:.4f}"
                    )

                # ============ KRYPTO ACTION PLAN ============
                from crypto_analysis import generate_crypto_action_plan

                fx_crypto = get_fx_rate("USD", "DKK")

                st.markdown("---")
                st.markdown("## 🎯 SÅDAN HANDLER DU")

                coins_for_plan = None
                invest_dkk_crypto = None
                if "KØB" in rec:
                    st.markdown("### 💼 Hvor meget vil du investere?")
                    inv_cols = st.columns([2, 1, 1, 1])
                    invest_dkk_crypto = inv_cols[0].number_input(
                        "Beløb (DKK)", min_value=500, value=5000, step=500,
                        key=f"crypto_invest_{symbol}",
                        help="Indtast hvor meget du vil investere i krypto"
                    )

                    if fx_crypto and price > 0:
                        price_dkk_crypto = price * fx_crypto
                        coins_for_plan = invest_dkk_crypto / price_dkk_crypto
                        actual_invest = coins_for_plan * price_dkk_crypto

                        inv_cols[1].metric(
                            f"🪙 Antal {symbol}",
                            f"{coins_for_plan:.6f}" if coins_for_plan < 1 else f"{coins_for_plan:.4f}"
                        )
                        inv_cols[2].metric("💰 I DKK", f"{actual_invest:,.0f} DKK")
                        inv_cols[3].metric("💵 I USD", f"${coins_for_plan * price:,.2f}")

                plan = generate_crypto_action_plan(
                    rec=rec, score=scores["overall"], current_price=price,
                    targets=targets, hist=hist, symbol=symbol,
                    fx_to_dkk=fx_crypto, investment_dkk=invest_dkk_crypto,
                    market_score=scores["market"],
                    technical_score=scores["technical"],
                    sentiment_score=scores["sentiment"],
                    dev_score=scores["developer"],
                )

                st.markdown(f"_{plan['summary']}_")

                for warn in plan["warnings"]:
                    st.warning(warn)

                if plan["totals"]:
                    t = plan["totals"]
                    st.markdown("### 💰 Din investering & forventet gevinst")
                    inv_summary = st.columns(4)

                    inv_summary[0].markdown(
                        f"<div style='background:#0099ff22;padding:1rem;border-radius:10px;"
                        f"border-left:5px solid #0099ff;text-align:center'>"
                        f"<small style='color:#888'>💵 INVESTERING</small>"
                        f"<h3 style='margin:0.3rem 0'>${t['invest_usd']:,.0f}</h3>"
                        f"<small><b>≈ {t['invest_dkk']:,.0f} DKK</b></small><br>"
                        f"<small>{t['coins']:.6f} {symbol}</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    inv_summary[1].markdown(
                        f"<div style='background:#16a34a22;padding:1rem;border-radius:10px;"
                        f"border-left:5px solid #16a34a;text-align:center'>"
                        f"<small style='color:#888'>📈 FORVENTET GEVINST</small>"
                        f"<h3 style='margin:0.3rem 0;color:#16a34a'>+${t['total_profit_usd']:,.0f}</h3>"
                        f"<small><b>≈ +{t['total_profit_dkk']:,.0f} DKK</b></small><br>"
                        f"<small>+{t['total_profit_pct']:.0f}% afkast</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    inv_summary[2].markdown(
                        f"<div style='background:#ef444422;padding:1rem;border-radius:10px;"
                        f"border-left:5px solid #ef4444;text-align:center'>"
                        f"<small style='color:#888'>⚠️ MAX TAB</small>"
                        f"<h3 style='margin:0.3rem 0;color:#ef4444'>-${t['max_loss_usd']:,.0f}</h3>"
                        f"<small><b>≈ -{t['max_loss_dkk']:,.0f} DKK</b></small><br>"
                        f"<small>Hvis stop-loss rammer</small>"
                        f"</div>", unsafe_allow_html=True
                    )
                    end_value_usd = t['invest_usd'] + t['total_profit_usd']
                    end_value_dkk = end_value_usd * fx_crypto if fx_crypto else 0
                    inv_summary[3].markdown(
                        f"<div style='background:#a855f722;padding:1rem;border-radius:10px;"
                        f"border-left:5px solid #a855f7;text-align:center'>"
                        f"<small style='color:#888'>🎯 SLUTVÆRDI</small>"
                        f"<h3 style='margin:0.3rem 0;color:#a855f7'>${end_value_usd:,.0f}</h3>"
                        f"<small><b>≈ {end_value_dkk:,.0f} DKK</b></small><br>"
                        f"<small>Efter alle 4 targets</small>"
                        f"</div>", unsafe_allow_html=True
                    )

                    with st.expander("📊 Sådan fordeler gevinsten sig (1/4 + 1/4 + 1/4 + 1/4)"):
                        quarter = t['coins'] / 4
                        breakdown = [
                            {"Salg": "🎯 Target 1 (kort sigt)", "Coins": f"{quarter:.6f}",
                             "Pris": f"${targets['target_short']:.4f}",
                             "Gevinst USD": f"+${t['profit_short_usd']:,.0f}",
                             "Gevinst DKK": f"+{t['profit_short_usd']*fx_crypto:,.0f} DKK" if fx_crypto else "-"},
                            {"Salg": "🚀 Target 2 (lang sigt)", "Coins": f"{quarter:.6f}",
                             "Pris": f"${targets['target_long']:.4f}",
                             "Gevinst USD": f"+${t['profit_long_usd']:,.0f}",
                             "Gevinst DKK": f"+{t['profit_long_usd']*fx_crypto:,.0f} DKK" if fx_crypto else "-"},
                            {"Salg": "🌙 Target 3 (moon)", "Coins": f"{quarter:.6f}",
                             "Pris": f"${targets['target_moon']:.4f}",
                             "Gevinst USD": f"+${t['profit_moon_usd']:,.0f}",
                             "Gevinst DKK": f"+{t['profit_moon_usd']*fx_crypto:,.0f} DKK" if fx_crypto else "-"},
                            {"Salg": "💎 HODL (5-10x estimat)", "Coins": f"{quarter:.6f}",
                             "Pris": f"${targets['target_moon']*1.3:.4f}",
                             "Gevinst USD": f"+${t['profit_hodl_usd']:,.0f}",
                             "Gevinst DKK": f"+{t['profit_hodl_usd']*fx_crypto:,.0f} DKK" if fx_crypto else "-"},
                        ]
                        st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)
                        st.caption(
                            "💡 **OBS:** Krypto kan tabe **50-90%** i bear markets. "
                            "Invester KUN hvad du har råd til at tabe. Brug ALTID stop-loss!"
                        )
                else:
                    if "KØB" in rec:
                        st.info("💡 Indtast et beløb ovenfor for at se forventet gevinst i DKK!")

                if plan["risk_reward"]:
                    rr = plan["risk_reward"]
                    st.markdown("### ⚖️ Risk / Reward")
                    rr_cols = st.columns(5)
                    rr_cols[0].metric("⚠️ Risk", f"-{rr['risk_pct']:.1f}%")
                    rr_cols[1].metric("🎯 Reward (kort)", f"+{rr['reward_short_pct']:.0f}%")
                    rr_cols[2].metric("🚀 Reward (lang)", f"+{rr['reward_long_pct']:.0f}%")
                    rr_cols[3].metric("🌙 Reward (moon)", f"+{rr['reward_moon_pct']:.0f}%")

                    rr_color = "#16a34a" if rr['ratio_moon'] >= 5 else "#22c55e" if rr['ratio_moon'] >= 3 else "#eab308" if rr['ratio_moon'] >= 2 else "#ef4444"
                    rr_label = "Excellent" if rr['ratio_moon'] >= 5 else "God" if rr['ratio_moon'] >= 3 else "OK" if rr['ratio_moon'] >= 2 else "Svag"
                    rr_cols[4].markdown(
                        f"<div style='background:{rr_color}22;padding:0.6rem;border-radius:8px;"
                        f"border-left:4px solid {rr_color};text-align:center'>"
                        f"<small>R/R MOON</small>"
                        f"<h3 style='margin:0.2rem 0;color:{rr_color}'>{rr['ratio_moon']:.1f}:1</h3>"
                        f"<small>{rr_label}</small></div>",
                        unsafe_allow_html=True
                    )

                st.markdown("### 📋 Trin-for-trin handleplan")
                for step in plan["steps"]:
                    st.markdown(
                        f"<div style='background:{step['color']}15;padding:1rem;border-radius:10px;"
                        f"border-left:5px solid {step['color']};margin-bottom:0.6rem'>"
                        f"<div style='display:flex;align-items:center;gap:0.8rem'>"
                        f"<div style='font-size:2rem'>{step['icon']}</div>"
                        f"<div style='flex:1'>"
                        f"<div style='color:{step['color']};font-weight:bold;font-size:0.9rem'>"
                        f"STEP {step['n']} · {step['title']}</div>"
                        f"<div style='font-size:1.1rem;margin:0.3rem 0'>{step['main']}</div>"
                        f"<div style='color:#aaa;font-size:0.9rem'>{step['sub']}</div>"
                        f"</div></div></div>",
                        unsafe_allow_html=True
                    )

                with st.expander("📚 Hvad er TRAILING STOP? (især vigtigt for krypto)"):
                    st.markdown("""
                    **Trailing stop** = "rullende stop-loss" der **følger med opad** når kursen stiger,
                    men **bevæger sig aldrig nedad**.

                    ### 📈 Eksempel med BTC:
                    ```
                    Du køber BTC @ $50,000, stop-loss = $45,000 (-10%)

                    ✅ BTC stiger til $60,000  →  trailing stop bliver $54,000 (-10%)
                    ✅ BTC stiger til $80,000  →  trailing stop bliver $72,000
                    ✅ BTC stiger til $100,000 →  trailing stop bliver $90,000
                    🛑 BTC falder til $90,000  →  SOLGT med +$40,000 profit (+80%)!
                    ```

                    ### 🎯 Hvorfor er det EKSTRA vigtigt for krypto?
                    1. **Krypto er meget volatilt** — store fald kan ske på minutter
                    2. **Markedet er åbent 24/7** — du kan ikke sidde og kigge altid
                    3. **FOMO er farligt** — trailing stop låser gevinst automatisk
                    4. **Bull → bear skift** kan være brutale (BTC -75% på 6 mdr)

                    ### 💼 Hvor sætter man det?
                    - **Coinbase Advanced** — "Stop-Limit" med trailing
                    - **Binance** — "Trailing Stop" ordre-type
                    - **Kraken** — "Trailing Stop Loss"
                    - **eToro** — "Trailing Stop Loss"

                    ### ⚠️ Krypto-tip
                    Sæt typisk **15-20% trailing** på krypto pga. volatilitet
                    (vs. 5-10% på aktier). Ellers udløses det for tidligt!
                    """)

                st.caption(
                    "⚠️ Datoer og gevinster er **estimater** baseret på historisk momentum og volatilitet. "
                    "Krypto kan tabe 50-90% i bear markets — invester KUN hvad du har råd til at tabe. "
                    "Brug ALTID stop-loss til at beskytte din kapital."
                )

                if symbol == "BTC":
                    halv = btc_halving_analysis(symbol)
                    if halv:
                        st.markdown("---")
                        st.markdown("### ⛏️ BTC Halving Cycle")
                        hc = st.columns(4)
                        hc[0].metric("📅 Sidste halving", halv["last_halving"])
                        hc[1].metric("📆 Dage siden", f"{halv['days_since_halving']}")
                        hc[2].metric("🎯 Næste halving", halv["next_halving"],
                                     f"om {halv['days_until_halving']} dage")
                        hc[3].metric("📊 Cycle progress", f"{halv['cycle_progress']:.0f}%")
                        st.info(f"**{halv['phase']}** — {halv['outlook']}")

                st.markdown("---")
                pro_tabs = st.tabs([
                    "📊 Charts", "🔧 Tekniske detaljer", "📉 Risiko",
                    "🎲 Monte Carlo", "🎯 Backtest", "🔗 BTC Korrelation",
                    "🔍 Score breakdown"
                ])

                with pro_tabs[0]:
                    df_ind = crypto_indicators(hist)
                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05
                    )
                    fig.add_trace(go.Candlestick(
                        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
                        low=df_ind["Low"], close=df_ind["Close"], name="Pris"
                    ), 1, 1)
                    if "SMA50" in df_ind.columns:
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["SMA50"], name="SMA50",
                            line=dict(color="orange")
                        ), 1, 1)
                    if len(df_ind) >= 200 and "SMA200" in df_ind.columns:
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["SMA200"], name="SMA200",
                            line=dict(color="purple")
                        ), 1, 1)
                    if "BB_upper" in df_ind.columns:
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["BB_upper"], name="BB Upper",
                            line=dict(color="rgba(255,255,255,0.3)", dash="dot")
                        ), 1, 1)
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["BB_lower"], name="BB Lower",
                            line=dict(color="rgba(255,255,255,0.3)", dash="dot"),
                            fill="tonexty", fillcolor="rgba(255,255,255,0.05)"
                        ), 1, 1)

                    if targets:
                        fig.add_hline(y=targets["buy_high"], line_dash="dot",
                                      line_color="#16a34a",
                                      annotation_text="Køb", row=1, col=1)
                        fig.add_hline(y=targets["stop_loss"], line_dash="dot",
                                      line_color="#ef4444",
                                      annotation_text="Stop", row=1, col=1)
                        fig.add_hline(y=targets["target_long"], line_dash="dot",
                                      line_color="#22c55e",
                                      annotation_text="Mål", row=1, col=1)

                    if "RSI" in df_ind.columns:
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["RSI"], name="RSI",
                            line=dict(color="#00d4aa")
                        ), 2, 1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    if "MACD" in df_ind.columns:
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["MACD"], name="MACD",
                            line=dict(color="#0099ff")
                        ), 3, 1)
                        fig.add_trace(go.Scatter(
                            x=df_ind.index, y=df_ind["MACD_signal"], name="Signal",
                            line=dict(color="orange")
                        ), 3, 1)

                    fig.update_layout(
                        height=800, xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        title=f"{info['symbol']}/USD - Komplet teknisk analyse"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with pro_tabs[1]:
                    df_ind = crypto_indicators(hist)
                    last = df_ind.iloc[-1]
                    cc = st.columns(4)
                    cc[0].metric("RSI", f"{last['RSI']:.1f}" if not pd.isna(last.get("RSI")) else "-")
                    cc[1].metric("MACD", f"{last['MACD']:.4f}" if not pd.isna(last.get("MACD")) else "-")
                    cc[2].metric("ATR", f"${last['ATR']:.4f}" if not pd.isna(last.get("ATR")) else "-")
                    if not pd.isna(last.get("BB_upper")):
                        cc[3].metric(
                            "BB Width",
                            f"{((last['BB_upper']-last['BB_lower'])/last['Close']*100):.1f}%"
                        )

                    cc2 = st.columns(3)
                    cc2[0].metric("SMA20", f"${last['SMA20']:.4f}" if not pd.isna(last.get("SMA20")) else "-")
                    cc2[1].metric("SMA50", f"${last['SMA50']:.4f}" if not pd.isna(last.get("SMA50")) else "-")
                    cc2[2].metric("SMA200", f"${last['SMA200']:.4f}" if not pd.isna(last.get("SMA200")) else "-")

                with pro_tabs[2]:
                    risk = crypto_risk_metrics(hist)
                    if risk:
                        st.caption("📉 Risk metrics (annualiseret med 365 dage)")
                        rc = st.columns(4)
                        rc[0].metric("Ann. afkast", f"{risk['ann_r']*100:.1f}%")
                        rc[1].metric("Ann. volatilitet", f"{risk['ann_v']*100:.1f}%")
                        rc[2].metric("Sharpe", f"{risk['sharpe']:.2f}")
                        rc[3].metric("Sortino", f"{risk['sortino']:.2f}")

                        rc2 = st.columns(3)
                        rc2[0].metric("Calmar", f"{risk['calmar']:.2f}")
                        rc2[1].metric("Max Drawdown", f"{risk['max_dd']*100:.1f}%")
                        rc2[2].metric("VaR 95% (1d)", f"{risk['var95']*100:.2f}%")

                        fig_dd = go.Figure(go.Scatter(
                            x=risk["dd_series"].index,
                            y=risk["dd_series"] * 100,
                            fill="tozeroy", line=dict(color="#ef4444")
                        ))
                        fig_dd.update_layout(template="plotly_dark", height=350, title="Drawdown %")
                        st.plotly_chart(fig_dd, use_container_width=True)
                    else:
                        st.warning("Ikke nok data til risk metrics")

                with pro_tabs[3]:
                    st.caption("🎲 500 simulationer · Student-t (fat tails)")
                    mc_days = st.slider("Dage frem", 30, 365, 180, key="mc_days")
                    sims, lp = crypto_monte_carlo(hist, n_sims=500, days=mc_days)

                    if sims is not None:
                        final = sims[:, -1]
                        p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])

                        mc_cols = st.columns(5)
                        mc_cols[0].metric("5% (worst)", f"${p5:,.4f}", f"{(p5/lp-1)*100:+.0f}%")
                        mc_cols[1].metric("25%", f"${p25:,.4f}", f"{(p25/lp-1)*100:+.0f}%")
                        mc_cols[2].metric(f"Median ({mc_days}d)", f"${p50:,.4f}", f"{(p50/lp-1)*100:+.0f}%")
                        mc_cols[3].metric("75%", f"${p75:,.4f}", f"{(p75/lp-1)*100:+.0f}%")
                        mc_cols[4].metric("95% (best)", f"${p95:,.4f}", f"{(p95/lp-1)*100:+.0f}%")

                        fig_m = go.Figure()
                        for i in range(min(150, len(sims))):
                            fig_m.add_trace(go.Scatter(
                                y=sims[i],
                                line=dict(width=0.5, color="rgba(0,212,170,0.1)"),
                                showlegend=False
                            ))
                        fig_m.add_trace(go.Scatter(
                            y=np.percentile(sims, 50, axis=0),
                            name="Median", line=dict(color="#00d4aa", width=3)
                        ))
                        fig_m.add_trace(go.Scatter(
                            y=np.percentile(sims, 5, axis=0),
                            name="5% (worst)",
                            line=dict(color="#ef4444", width=2, dash="dash")
                        ))
                        fig_m.add_trace(go.Scatter(
                            y=np.percentile(sims, 95, axis=0),
                            name="95% (best)",
                            line=dict(color="#22c55e", width=2, dash="dash")
                        ))
                        fig_m.update_layout(
                            template="plotly_dark", height=500,
                            title=f"Monte Carlo - {mc_days} dage frem"
                        )
                        st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("Ikke nok data til Monte Carlo")

                with pro_tabs[4]:
                    st.caption("🎯 Walk-forward backtest af model-anbefalinger")
                    bc1, bc2 = st.columns(2)
                    holding = bc1.selectbox(
                        "Holding periode (dage)",
                        [14, 30, 60, 90, 180], index=1, key="ct_hold"
                    )
                    freq = bc2.selectbox("Sample frekvens", [3, 7, 14], index=1, key="ct_freq")

                    if st.button("🚀 Kør krypto-backtest", type="primary", key="btn_crypto_bt"):
                        with st.spinner("Kører walk-forward..."):
                            bt = crypto_backtest(hist, holding_days=holding, sample_freq=freq)

                        if bt is None:
                            st.error(f"Ikke nok data ({len(hist)} dage)")
                        else:
                            st.markdown(
                                f"📊 **{bt['n_trades']} samples** · "
                                f"{bt['start_date'].date()} → {bt['end_date'].date()}"
                            )

                            rows = []
                            for rec_lbl in ["KØB", "HOLD", "SÆLG"]:
                                s = bt["stats"].get(rec_lbl)
                                if s:
                                    rows.append({
                                        "Anbefaling": rec_lbl, "Antal": s["count"],
                                        "Hit rate": f"{s['win_rate']:.1f}%",
                                        "Gns. afkast": f"{s['avg_return']:+.2f}%",
                                        "Median": f"{s['median_return']:+.2f}%",
                                        "Bedst": f"{s['best']:+.1f}%",
                                        "Værst": f"{s['worst']:+.1f}%",
                                    })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            st.markdown(f"📈 **Buy & Hold:** {bt['buy_hold_return']:+.2f}%")

                            fig_bt = px.scatter(
                                bt["results"], x="score", y="return_pct",
                                color="recommendation",
                                color_discrete_map={
                                    "KØB": "#22c55e", "HOLD": "#eab308", "SÆLG": "#ef4444"
                                },
                                title=f"Score vs {holding}-dages afkast",
                                labels={"score": "Score", "return_pct": "Afkast %"}
                            )
                            fig_bt.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                            fig_bt.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig_bt, use_container_width=True)

                            corr = bt["results"]["score"].corr(bt["results"]["return_pct"])
                            if corr > 0.3:
                                st.success(f"✅ Stærk korrelation: {corr:.3f}")
                            elif corr > 0.1:
                                st.info(f"➖ Svag korrelation: {corr:.3f}")
                            else:
                                st.warning(f"⚠️ Ingen/negativ korrelation: {corr:.3f}")

                with pro_tabs[5]:
                    if symbol == "BTC":
                        st.info("Dette er BTC — sammenligning med sig selv ikke meningsfuld")
                    else:
                        with st.spinner("Henter BTC til sammenligning..."):
                            btc_data = fetch_crypto_data("BTC")

                        if btc_data:
                            corr_data = calculate_btc_correlation(hist, btc_data["hist"])
                            if corr_data:
                                cc = st.columns(2)
                                cc[0].metric(
                                    "Korrelation til BTC",
                                    f"{corr_data['correlation']:.3f}",
                                    "Høj" if corr_data['correlation'] > 0.7 else
                                    "Medium" if corr_data['correlation'] > 0.4 else "Lav"
                                )
                                cc[1].metric(
                                    "Beta til BTC",
                                    f"{corr_data['beta']:.2f}",
                                    "Mere volatil" if corr_data['beta'] > 1.2 else
                                    "Mindre volatil" if corr_data['beta'] < 0.8 else
                                    "Ligner BTC"
                                )

                                fig_corr = go.Figure(go.Scatter(
                                    x=corr_data["rolling_correlation"].index,
                                    y=corr_data["rolling_correlation"],
                                    fill="tozeroy",
                                    line=dict(color="#00d4aa", width=2),
                                ))
                                fig_corr.add_hline(y=0.7, line_dash="dash",
                                                    line_color="green",
                                                    annotation_text="Høj")
                                fig_corr.update_layout(
                                    template="plotly_dark", height=400,
                                    title=f"30-dages rolling korrelation: {symbol} vs BTC",
                                    yaxis_range=[-1, 1]
                                )
                                st.plotly_chart(fig_corr, use_container_width=True)
                            else:
                                st.warning("Ikke nok data til korrelations-analyse")
                        else:
                            st.error("Kunne ikke hente BTC-data")

                with pro_tabs[6]:
                    detail_subtabs = st.tabs(["📊 Marked", "🔧 Teknisk", "💬 Sentiment", "👨‍💻 Developer"])
                    for tab, key in zip(detail_subtabs,
                                         ["market", "technical", "sentiment", "developer"]):
                        with tab:
                            details = scores["details"][key]
                            if details:
                                df_d = pd.DataFrame(details)
                                fig_d = px.bar(
                                    df_d, x="impact", y="label", orientation="h",
                                    color="impact", color_continuous_scale="RdYlGn"
                                )
                                fig_d.update_layout(template="plotly_dark", height=300, showlegend=False)
                                st.plotly_chart(fig_d, use_container_width=True)
                                st.dataframe(df_d, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"Ingen {key}-data tilgængelig")

                if info.get("description"):
                    with st.expander("ℹ️ Om denne krypto"):
                        st.write(info["description"])
                            # ===== TAB 2: SCREENER =====
    with crypto_tabs[1]:
        st.markdown("### 🔎 Krypto-screener")
        sc1, sc2 = st.columns([2, 1])
        sel_universe = sc1.selectbox("Univers", list(CRYPTO_UNIVERSES.keys()), key="cs_universe")
        min_score_c = sc2.slider("Min. score", 30, 90, 55, key="cs_min")

        if st.button("🚀 Kør krypto-screener", type="primary"):
            tickers_c = CRYPTO_UNIVERSES[sel_universe]
            results_c = []
            progress_c = st.progress(0, text="Starter...")

            for i, t in enumerate(tickers_c):
                progress_c.progress((i+1) / len(tickers_c), text=f"Analyserer {t}...")
                try:
                    cdata = fetch_crypto_data(t)
                    if cdata:
                        scores = crypto_overall_score(cdata["info"], cdata["hist"])
                        rec, _ = crypto_recommendation(scores["overall"])
                        results_c.append({
                            "Symbol": t, "Navn": cdata["info"]["longName"],
                            "Pris ($)": cdata["info"]["currentPrice"],
                            "MC ($B)": (cdata["info"].get("marketCap") or 0) / 1e9,
                            "24h %": cdata["info"].get("change_24h"),
                            "7d %": cdata["info"].get("change_7d"),
                            "30d %": cdata["info"].get("change_30d"),
                            "Overall": scores["overall"],
                            "Marked": scores["market"],
                            "Teknisk": scores["technical"],
                            "Sentiment": scores["sentiment"],
                            "Dev": scores["developer"],
                            "Anbefaling": rec,
                        })
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error {t}: {e}")

            progress_c.empty()

            if results_c:
                df_r = pd.DataFrame(results_c)
                df_filt = df_r[df_r["Overall"] >= min_score_c].sort_values("Overall", ascending=False)
                st.success(f"✅ {len(df_filt)} kryptos opfylder kriterierne")

                if not df_filt.empty:
                    for col in ["Pris ($)", "MC ($B)", "24h %", "7d %", "30d %",
                                "Overall", "Marked", "Teknisk", "Sentiment", "Dev"]:
                        if col in df_filt.columns:
                            df_filt[col] = pd.to_numeric(df_filt[col], errors="coerce").round(2)

                    st.dataframe(
                        df_filt, use_container_width=True, hide_index=True,
                        column_config={
                            "Overall": st.column_config.ProgressColumn(
                                "Overall", min_value=0, max_value=100, format="%.0f"
                            ),
                        }
                    )

                    csv_c = df_filt.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download CSV", csv_c,
                        f"crypto_screener_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv"
                    )

    # ===== TAB 3: TRENDING =====
    with crypto_tabs[2]:
        st.markdown("### 🔥 Trending Coins (CoinGecko)")
        trending = fetch_trending_coins()
        if trending:
            tcols = st.columns(min(7, len(trending)))
            for i, t in enumerate(trending):
                with tcols[i]:
                    st.markdown(f"**{t['symbol']}**")
                    st.caption(t['name'][:20])
                    if t.get("rank"):
                        st.caption(f"#{t['rank']}")

        st.markdown("---")
        st.markdown("### 📈 Top Gainers / Losers (24h)")
        gainers, losers = fetch_top_movers()
        if gainers is not None:
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown("#### 🚀 Top 10 Gainers")
                gainers_d = gainers.copy()
                gainers_d.columns = ["Sym", "Navn", "Pris", "24h %", "MC"]
                gainers_d["24h %"] = gainers_d["24h %"].round(2)
                gainers_d["Pris"] = gainers_d["Pris"].round(4)
                gainers_d["MC"] = (gainers_d["MC"] / 1e9).round(2)
                st.dataframe(gainers_d, use_container_width=True, hide_index=True)

            with gc2:
                st.markdown("#### 📉 Top 10 Losers")
                losers_d = losers.copy()
                losers_d.columns = ["Sym", "Navn", "Pris", "24h %", "MC"]
                losers_d["24h %"] = losers_d["24h %"].round(2)
                losers_d["Pris"] = losers_d["Pris"].round(4)
                losers_d["MC"] = (losers_d["MC"] / 1e9).round(2)
                st.dataframe(losers_d, use_container_width=True, hide_index=True)

    # ===== TAB 4: SAMMENLIGN =====
    with crypto_tabs[3]:
        st.markdown("### 📈 Sammenlign kryptos")
        selected = st.multiselect(
            "Vælg kryptos (max 6)", list(CRYPTO_UNIVERSE.keys()),
            default=["BTC", "ETH", "SOL"], max_selections=6,
        )

        if selected and st.button("📊 Sammenlign", type="primary"):
            cmp_data = {}
            for sym in selected:
                cdata = fetch_crypto_data(sym)
                if cdata:
                    cmp_data[sym] = cdata

            if cmp_data:
                fig_cmp = go.Figure()
                for sym, cdata in cmp_data.items():
                    hist = cdata["hist"]
                    norm = hist["Close"] / hist["Close"].iloc[0] * 100
                    fig_cmp.add_trace(go.Scatter(
                        x=hist.index, y=norm, name=sym, line=dict(width=2)
                    ))
                fig_cmp.update_layout(
                    template="plotly_dark", height=500,
                    title="Performance (normaliseret til 100)",
                    yaxis_title="Performance"
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                cmp_rows = []
                for sym, cdata in cmp_data.items():
                    scores = crypto_overall_score(cdata["info"], cdata["hist"])
                    cmp_rows.append({
                        "Symbol": sym,
                        "Pris ($)": round(cdata["info"]["currentPrice"], 2),
                        "MC ($B)": round((cdata["info"].get("marketCap") or 0) / 1e9, 2),
                        "Overall": round(scores["overall"], 1),
                        "Marked": round(scores["market"], 1),
                        "Teknisk": round(scores["technical"], 1),
                        "Sentiment": round(scores["sentiment"], 1),
                        "Dev": round(scores["developer"], 1),
                    })
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    # ===== TAB 5: SENTIMENT =====
    with crypto_tabs[4]:
        st.markdown("### 😱 Fear & Greed Index")
        if fg_df is not None and not fg_df.empty:
            current = int(fg_df["value"].iloc[-1])
            label = fg_df["value_classification"].iloc[-1]

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=current,
                title={"text": f"Nu: {label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "white"},
                    "steps": [
                        {"range": [0, 25], "color": "#b91c1c"},
                        {"range": [25, 45], "color": "#ef4444"},
                        {"range": [45, 55], "color": "#eab308"},
                        {"range": [55, 75], "color": "#22c55e"},
                        {"range": [75, 100], "color": "#16a34a"},
                    ],
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=fg_df["timestamp"], y=fg_df["value"],
                mode="lines+markers",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy", name="F&G"
            ))
            fig_h.add_hline(y=25, line_dash="dash", line_color="red")
            fig_h.add_hline(y=75, line_dash="dash", line_color="green")
            fig_h.update_layout(
                template="plotly_dark", height=400,
                title="Fear & Greed - sidste 30 dage", yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ===== TAB 6: ON-CHAIN =====
    with crypto_tabs[5]:
        st.markdown("### ⛓️ Bitcoin On-Chain Metrics")
        st.caption("Network health · Hash rate · Active addresses · Mempool")

        with st.spinner("Henter on-chain data..."):
            onchain = fetch_btc_onchain()

        if onchain:
            oc = st.columns(3)
            if "hash_rate" in onchain:
                oc[0].metric("⚡ Hash Rate", f"{onchain['hash_rate']/1e6:,.1f}M TH/s")
            if "difficulty" in onchain:
                oc[1].metric("⚙️ Difficulty", f"{onchain['difficulty']/1e12:,.2f}T")
            if "active_addresses" in onchain:
                oc[2].metric("👥 Active Addresses", f"{onchain['active_addresses']:,.0f}")

            oc2 = st.columns(3)
            if "transactions" in onchain:
                oc2[0].metric("📊 Daily Transactions", f"{onchain['transactions']:,.0f}")
            if "mempool_size" in onchain:
                oc2[1].metric("🔄 Mempool (bytes)", f"{onchain['mempool_size']:,.0f}")
            if "miners_revenue" in onchain:
                oc2[2].metric("⛏️ Miner Revenue", f"${onchain['miners_revenue']:,.0f}")

            if "hash_rate_history" in onchain:
                hr_df = pd.DataFrame(onchain["hash_rate_history"])
                hr_df["x"] = pd.to_datetime(hr_df["x"], unit="s")
                fig_hr = go.Figure(go.Scatter(
                    x=hr_df["x"], y=hr_df["y"] / 1e6,
                    fill="tozeroy", line=dict(color="#f59e0b")
                ))
                fig_hr.update_layout(
                    template="plotly_dark", height=400,
                    title="BTC Hash Rate (M TH/s) - 30 dage",
                    yaxis_title="Hash Rate (M TH/s)"
                )
                st.plotly_chart(fig_hr, use_container_width=True)
        else:
            st.warning("Kunne ikke hente on-chain data")
            # ============ HOVED-ANALYSE-VIEW ============

elif st.session_state.active_view == "📊 Analyse":
    c1, c2 = st.columns([4, 1])
    default_t = st.session_state.current_ticker or "AAPL"
    ticker_input = c1.text_input(
        "Ticker (fx AAPL, NOVO-B.CO)", value=default_t, key="ticker_input"
    ).strip().upper()
    auto_analyze = c2.button("🔍 Analysér", type="primary", use_container_width=True)

    # ============ SØGE-HISTORIK ============
    if st.session_state.search_history:
        st.caption("🕐 **Seneste søgninger** (klik for hurtig analyse):")
        hist_cols = st.columns(min(10, len(st.session_state.search_history)))
        for i, hist_ticker in enumerate(st.session_state.search_history):
            with hist_cols[i]:
                if st.button(
                    hist_ticker,
                    key=f"hist_{hist_ticker}_{i}",
                    use_container_width=True,
                    help=f"Analysér {hist_ticker} igen"
                ):
                    goto_analysis(hist_ticker)

        cols_clear = st.columns([5, 1])
        if cols_clear[1].button("🗑️ Ryd historik", key="clear_search_hist", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()

    if auto_analyze or st.session_state.current_ticker == ticker_input:
        st.session_state.current_ticker = ticker_input
        if auto_analyze and ticker_input:
            add_to_search_history(ticker_input)
    ticker = ticker_input

    if not ticker:
        st.info("👆 Indtast en ticker, eller brug **🔍 Søg ticker** fanen")
        st.stop()

    if is_crypto(ticker):
        norm = normalize_crypto_ticker(ticker)
        if norm in CRYPTO_UNIVERSE:
            st.info(f"🪙 **{norm}** er en kryptovaluta. Skifter til **Krypto-fanen**...")
            st.session_state["crypto_analyzed"] = norm
            st.session_state.active_view = "🪙 Krypto"
            st.rerun()

    _fetch_start = time.time()
    with st.spinner(f"Henter data for {ticker}..."):
        data = fetch_data(ticker)
    _fetch_time = time.time() - _fetch_start

    if data is None:
        st.error(f"❌ Kunne ikke hente data for '{ticker}'")
        st.info("👉 Prøv **🔍 Søg ticker** fanen")
        st.stop()

    add_to_search_history(ticker)

    st.session_state.last_source = data["source"]
    if data.get("warning"):
        st.warning(f"⚠️ {data['warning']}")
    else:
        st.success(f"✅ Data hentet fra: **{data['source']}** ({_fetch_time:.1f}s)")

    st.markdown(
        "<div style='background:#0099ff15;padding:0.6rem 1rem;border-radius:8px;"
        "border-left:4px solid #0099ff;margin:0.5rem 0'>"
        f"📅 <b>Chart:</b> {period} · "
        "ℹ️ <b>Beregninger:</b> Tekniske=12mdr · Kursmål=6mdr · Risk=3år · MC=2år"
        "</div>",
        unsafe_allow_html=True,
    )

    info = data["info"]
    hist = data["hist"]

    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)

    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"## {info.get('longName', ticker)} ({ticker})")
        st.caption(
            f"🏢 {info.get('sector', '?')} · "
            f"🌍 {info.get('country', '?')} · "
            f"💱 {info.get('currency', 'USD')}"
        )
    with h2:
        if st.button("🗑️ Fjern fra watchlist", use_container_width=True):
            if ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(ticker)
                st.success(f"Fjernet {ticker}")
                time.sleep(1)
                st.rerun()

    price = info.get("currentPrice")
    prev = info.get("previousClose")
    if price is None and not hist.empty:
        price = float(hist["Close"].iloc[-1])
    if prev is None and len(hist) >= 2:
        prev = float(hist["Close"].iloc[-2])

    change_pct = ((price / prev - 1) * 100) if (price and prev) else 0
    currency = info.get("currency", "USD")

    pcols = st.columns(4)

    with pcols[0]:
        if price is not None:
            change_color = "#16a34a" if change_pct >= 0 else "#ef4444"
            change_emoji = "🟢" if change_pct >= 0 else "🔴"
            st.markdown(
                f"<div style='background:#0099ff15;padding:0.6rem;border-radius:8px;"
                f"border-left:4px solid #0099ff'>"
                f"<small style='color:#888'>PRIS NU</small>"
                f"<div style='font-size:1.4rem;font-weight:bold;margin:0.2rem 0'>"
                f"{price:.2f} {currency}</div>"
                f"<small style='color:{change_color}'>{change_emoji} {change_pct:+.2f}%</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.metric("Pris nu", "N/A")

    with pcols[1]:
        if show_secondary and currency != "DKK" and price is not None:
            fx = get_fx_rate(currency, "DKK")
            if fx:
                price_dkk = price * fx
                st.markdown(
                    f"<div style='background:#00d4aa15;padding:0.6rem;border-radius:8px;"
                    f"border-left:4px solid #00d4aa'>"
                    f"<small style='color:#888'>PRIS (DKK)</small>"
                    f"<div style='font-size:1.4rem;font-weight:bold;margin:0.2rem 0'>"
                    f"{price_dkk:,.2f} DKK</div>"
                    f"<small>Kurs: {fx:.2f}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        elif currency == "DKK":
            st.markdown(
                f"<div style='background:#00d4aa15;padding:0.6rem;border-radius:8px;"
                f"border-left:4px solid #00d4aa'>"
                f"<small style='color:#888'>VALUTA</small>"
                f"<div style='font-size:1.4rem;font-weight:bold;margin:0.2rem 0'>"
                f"DKK ✅</div>"
                f"<small>Allerede i DKK</small>"
                f"</div>",
                unsafe_allow_html=True
            )

    with pcols[2]:
        low_52 = info.get("fiftyTwoWeekLow")
        high_52 = info.get("fiftyTwoWeekHigh")

        if (low_52 is None or high_52 is None) and not hist.empty:
            recent_year = hist.tail(252) if len(hist) >= 252 else hist
            if low_52 is None:
                low_52 = float(recent_year["Low"].min())
            if high_52 is None:
                high_52 = float(recent_year["High"].max())

        if low_52 is not None and high_52 is not None:
            if price and high_52 > low_52:
                pos_pct = ((price - low_52) / (high_52 - low_52)) * 100
            else:
                pos_pct = 50

            if pos_pct < 30:
                pos_color = "#16a34a"
                pos_label = "Tæt på lav"
            elif pos_pct > 80:
                pos_color = "#ef4444"
                pos_label = "Tæt på top"
            else:
                pos_color = "#eab308"
                pos_label = "Mellem"

            st.markdown(
                f"<div style='background:{pos_color}22;padding:0.6rem;border-radius:8px;"
                f"border-left:4px solid {pos_color}'>"
                f"<small style='color:#888'>52-UGER RANGE</small>"
                f"<div style='font-size:1.1rem;font-weight:bold;margin:0.2rem 0'>"
                f"{low_52:.2f} - {high_52:.2f}</div>"
                f"<small>{pos_pct:.0f}% i range · {pos_label}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.metric("52-uger", "N/A", "Ingen data")

    with pcols[3]:
        mc = info.get("marketCap")
        if mc:
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
                mc_label = "Mega Cap"
                mc_color = "#a855f7"
            elif mc >= 1e11:
                mc_str = f"${mc/1e9:.0f}B"
                mc_label = "Large Cap"
                mc_color = "#0099ff"
            elif mc >= 1e10:
                mc_str = f"${mc/1e9:.1f}B"
                mc_label = "Mid Cap"
                mc_color = "#00d4aa"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.2f}B"
                mc_label = "Small Cap"
                mc_color = "#eab308"
            else:
                mc_str = f"${mc/1e6:.0f}M"
                mc_label = "Micro Cap"
                mc_color = "#ef4444"

            st.markdown(
                f"<div style='background:{mc_color}22;padding:0.6rem;border-radius:8px;"
                f"border-left:4px solid {mc_color}'>"
                f"<small style='color:#888'>MARKET CAP</small>"
                f"<div style='font-size:1.4rem;font-weight:bold;margin:0.2rem 0'>"
                f"{mc_str}</div>"
                f"<small>{mc_label}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.metric("Market Cap", "N/A")

    # ============================================================
    # 🆕 REGIME DETECTION + REGIME-AWARE SCORING + EARNINGS BOOST
    # ============================================================
    from regime_detector import (
        detect_market_regime,
        detect_combined_regime,
        adjust_weights_for_regime,
        regime_recommendation,
        render_regime_banner,
    )
    from analysis import overall_score_with_regime

    _analysis_start = time.time()
    df_indicators = get_indicators(hist)
    df_technical = filter_by_days(df_indicators, ANALYSIS_PERIODS["technical"])

    f_score, f_details = fundamental_score(info)
    t_score, t_details = technical_score(df_technical)

    score_data = overall_score_with_regime(
        f_score, t_score,
        ticker=ticker,
        country=info.get("country"),
    )
    overall = score_data["overall"]
    regime = score_data["regime"]
    regime_conf = score_data["regime_confidence"]
    regime_metrics = score_data["regime_metrics"]
    fund_weight = score_data["fund_weight"]
    tech_weight = score_data["tech_weight"]

    benchmark_label = regime_metrics.get("benchmark_label", "S&P 500")
    is_combined_regime = regime_metrics.get("is_combined", False)

    # ============================================================
    # 🆕 EARNINGS-DATA HENTES TIDLIGT (bruges til score-boost + chart)
    # ============================================================
    with st.spinner("📅 Tjekker earnings-kalender..."):
        earnings_data = get_earnings_info(ticker)

    # 🆕 BEREGN EARNINGS SCORE BOOST
    earnings_boost_info = calculate_earnings_score_boost(earnings_data)
    earnings_boost = earnings_boost_info["boost"]

    # 🆕 JUSTÉR OVERALL SCORE (clamp til 0-100)
    overall_pre_earnings = overall
    overall = max(0, min(100, overall + earnings_boost))

    rec, color = recommendation(overall, regime=regime)

    _analysis_time = time.time() - _analysis_start

    st.markdown("---")

    render_regime_banner(regime, regime_conf, regime_metrics, asset_type="stock")

    st.markdown("")

    # === SCORE CARDS MED EARNINGS-JUSTERING ===
    rec_cols = st.columns([2, 1, 1, 1])
    with rec_cols[0]:
        bench_info = f"vs {benchmark_label}" if benchmark_label else ""
        # 🆕 Vis earnings-justering hvis den eksisterer
        earnings_note = ""
        if earnings_boost != 0:
            sign = "+" if earnings_boost > 0 else ""
            earnings_note = (
                f"<br><small style='color:#a855f7;font-size:0.75rem'>"
                f"📅 Earnings: {sign}{earnings_boost} (var {overall_pre_earnings:.0f})"
                f"</small>"
            )

        st.markdown(
            f"<div style='background:{color}22;padding:1.2rem;border-radius:12px;"
            f"border-left:5px solid {color};text-align:center'>"
            f"<h2 style='color:{color};margin:0.3rem 0'>{rec}</h2>"
            f"<h1 style='margin:0.3rem 0;font-size:2.5rem'>{overall:.0f}"
            f"<small style='font-size:1.2rem;color:#888'>/100</small></h1>"
            f"<small style='color:#888'>Regime + earnings-justeret</small><br>"
            f"<small style='color:#666;font-size:0.75rem'>{bench_info}</small>"
            f"{earnings_note}"
            f"</div>",
            unsafe_allow_html=True
        )
    rec_cols[1].metric(
        "📊 Fundamental",
        f"{f_score:.0f}/100",
        f"{int(fund_weight*100)}% vægt ({regime})"
    )
    rec_cols[2].metric(
        "🔧 Teknisk",
        f"{t_score:.0f}/100",
        f"{int(tech_weight*100)}% vægt ({regime})"
    )
    rec_cols[3].metric(
        "🎯 Regime",
        regime,
        f"{regime_conf}% conf."
    )

    # 🆕 EARNINGS SCORE CARD (vises kun hvis der ER en effekt)
    if earnings_boost != 0:
        st.markdown("")
        render_earnings_score_card(earnings_data)

    if regime in ("BEAR", "VOLATILE"):
        st.info(
            f"💡 **{regime} marked detected** ({benchmark_label}): "
            f"Vægtning er flyttet mod fundamentals "
            f"({int(fund_weight*100)}% vs standard 60%). "
            f"Tærskler for KØB er hævet for at være mere konservativ."
        )
    elif regime == "BULL":
        st.success(
            f"💡 **BULL marked detected** ({benchmark_label}): "
            f"Vægtning er flyttet mod teknisk "
            f"({int(tech_weight*100)}% vs standard 40%)."
        )
    elif regime == "SIDEWAYS":
        st.info(
            f"💡 **SIDEWAYS marked detected** ({benchmark_label}): "
            f"Markedet trender ikke klart — balanceret tilgang anbefales."
        )

    if is_combined_regime:
        local_reg = regime_metrics.get("local_regime", "?")
        local_label = regime_metrics.get("local_label", "Local")
        global_reg = regime_metrics.get("global_regime", "?")
        local_conf = regime_metrics.get("local_confidence", 0)
        global_conf = regime_metrics.get("global_confidence", 0)

        regime_emojis = {
            "BULL": "🐂", "BEAR": "🐻",
            "SIDEWAYS": "➡️", "VOLATILE": "⚡", "UNKNOWN": "❓"
        }
        local_emj = regime_emojis.get(local_reg, "")
        global_emj = regime_emojis.get(global_reg, "")

        if local_reg != global_reg:
            st.warning(
                f"🌐 **Divergerende markeder:** "
                f"📍 {local_label} er {local_emj} **{local_reg}** ({local_conf}% conf.) — "
                f"🌍 men globalt (S&P 500) er {global_emj} **{global_reg}** ({global_conf}% conf.). "
                f"Vi bruger **{regime}** (forsigtighedsprincip)."
            )

    # ============================================================
    # 📰 NEWS SENTIMENT - KOMPAKT OVERSIGT
    # ============================================================
    st.markdown("---")
    company_name = info.get("longName") or info.get("shortName") or ticker
    with st.spinner("📰 Henter nyheder & sentiment..."):
        sentiment_data = get_news_sentiment(ticker, company_name=company_name, limit=20)

    render_sentiment_summary(sentiment_data, compact=True)

    # ============================================================
    # 🆕 EARNINGS WARNING - lige under sentiment
    # (earnings_data er allerede hentet tidligere til score-boost)
    # ============================================================
    st.markdown("---")
    render_earnings_warning(earnings_data, compact=True)

    # ============ ACTION PLAN ============
    try:
        fv_check = dcf_valuation(info, 0.10, 0.10, 0.025)
        dcf_upside = ((fv_check / price - 1) * 100) if fv_check and price else None
    except Exception:
        dcf_upside = None
        fv_check = None

    targets_main = calculate_price_targets(
        filter_by_days(df_indicators, ANALYSIS_PERIODS["targets"]),
        price, fv_check
    )

    fx_for_plan = None
    if currency != "DKK":
        fx_for_plan = get_fx_rate(currency, "DKK")

    shares_for_plan = None
    if "KØB" in rec:
        st.markdown("### 💼 Hvor meget vil du investere?")
        inv_input_cols = st.columns([2, 1, 1, 1])
        investment_dkk = inv_input_cols[0].number_input(
            "Beløb (DKK)", min_value=1000, value=10000, step=1000,
            key=f"plan_invest_{ticker}",
            help="Indtast hvor meget du vil bruge — så beregnes alt automatisk"
        )

        if currency != "DKK" and fx_for_plan:
            price_in_dkk = price * fx_for_plan
        else:
            price_in_dkk = price

        shares_for_plan = int(investment_dkk / price_in_dkk) if price_in_dkk > 0 else 0
        actual_invest_dkk = shares_for_plan * price_in_dkk

        inv_input_cols[1].metric(
            "📦 Antal aktier",
            f"{shares_for_plan}",
            help=f"{investment_dkk:,.0f} DKK / {price_in_dkk:.2f} DKK/aktie"
        )
        inv_input_cols[2].metric(
            "💰 Faktisk køb",
            f"{actual_invest_dkk:,.0f} DKK",
            f"{actual_invest_dkk-investment_dkk:+,.0f} DKK rest"
        )
        if currency != "DKK":
            inv_input_cols[3].metric(
                f"💵 I {currency}",
                f"{shares_for_plan * price:,.2f}",
                help=f"{shares_for_plan} × {price:.2f} {currency}"
            )

    plan = generate_action_plan(
        rec=rec, score=overall, current_price=price,
        targets=targets_main, hist=hist, currency=currency,
        f_score=f_score, t_score=t_score, dcf_upside=dcf_upside,
        shares=shares_for_plan, fx_to_dkk=fx_for_plan,
        regime=regime
    )

    st.markdown("---")
    st.markdown("## 🎯 SÅDAN HANDLER DU")
    st.markdown(f"_{plan['summary']}_")

    # 🆕 EARNINGS-BASERET ADVARSEL I ACTION PLAN
    earnings_warning_msg = get_earnings_warning_message(earnings_data)
    if earnings_warning_msg:
        level = earnings_data.get("warning_level", "none") if earnings_data else "none"
        if level == "critical":
            st.error(earnings_warning_msg)
        elif level == "high":
            st.warning(earnings_warning_msg)
        else:
            st.info(earnings_warning_msg)

    # 📰 SENTIMENT-BASERET ADVARSEL I ACTION PLAN
    if sentiment_data and sentiment_data.get("article_count", 0) >= 3:
        sent_score = sentiment_data.get("sentiment_score", 0)
        sent_label = sentiment_data.get("label", "Neutral")

        if "KØB" in rec and sent_score < -0.2:
            st.warning(
                f"⚠️ **NYHEDSADVARSEL:** Modellen siger **{rec}**, men nyheds-sentiment "
                f"er **{sent_label}** ({sent_score:+.2f}). Overvej at vente på bedre "
                f"nyhedsstrøm før indgang, eller halver position-størrelsen."
            )
        elif "SÆLG" in rec and sent_score > 0.3:
            st.info(
                f"💡 **NYHEDSDIVERGENS:** Modellen siger **{rec}**, men nyhederne er "
                f"**{sent_label}** ({sent_score:+.2f}). Måske et turnaround i sigte? "
                f"Hold øje med nyhederne i Nyheder-fanen."
            )
        elif "KØB" in rec and sent_score > 0.3:
            st.success(
                f"✅ **POSITIV NYHEDSBEKRÆFTELSE:** Modellen siger **{rec}** og nyheder "
                f"er **{sent_label}** ({sent_score:+.2f}) — stærkt signal!"
            )

    if regime in ("BEAR", "VOLATILE"):
        st.markdown(
            f"<div style='background:#ef444415;padding:0.8rem 1rem;border-radius:8px;"
            f"border-left:4px solid #ef4444;margin:0.5rem 0'>"
            f"⚠️ <b>{'🐻 BEAR' if regime == 'BEAR' else '⚡ VOLATILE'} MARKED:</b> "
            f"Vær ekstra forsigtig med position-størrelse."
            f"</div>",
            unsafe_allow_html=True
        )
    elif regime == "BULL" and "KØB" in rec:
        st.markdown(
            f"<div style='background:#16a34a15;padding:0.8rem 1rem;border-radius:8px;"
            f"border-left:4px solid #16a34a;margin:0.5rem 0'>"
            f"🐂 <b>BULL MARKED:</b> Momentum er din ven."
            f"</div>",
            unsafe_allow_html=True
        )

    for warn in plan["warnings"]:
        st.warning(warn)

    if plan["totals"]:
        t = plan["totals"]
        st.markdown("### 💰 Din investering & forventet gevinst")

        inv_cols = st.columns(4)

        invest_str_dkk = f"{t['invest_dkk']:,.0f} DKK" if t['invest_dkk'] else "-"
        inv_cols[0].markdown(
            f"<div style='background:#0099ff22;padding:1rem;border-radius:10px;"
            f"border-left:5px solid #0099ff;text-align:center'>"
            f"<small style='color:#888'>💵 INVESTERING</small>"
            f"<h3 style='margin:0.3rem 0'>${t['invest_usd']:,.0f}</h3>"
            f"<small><b>≈ {invest_str_dkk}</b></small><br>"
            f"<small>{t['shares']} aktier × {price:.2f} {currency}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        profit_str_dkk = f"{t['total_profit_dkk']:,.0f} DKK" if t['total_profit_dkk'] else "-"
        inv_cols[1].markdown(
            f"<div style='background:#16a34a22;padding:1rem;border-radius:10px;"
            f"border-left:5px solid #16a34a;text-align:center'>"
            f"<small style='color:#888'>📈 FORVENTET GEVINST</small>"
            f"<h3 style='margin:0.3rem 0;color:#16a34a'>+${t['total_profit_usd']:,.0f}</h3>"
            f"<small><b>≈ +{profit_str_dkk}</b></small><br>"
            f"<small>+{t['total_profit_pct']:.1f}% afkast</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        loss_str_dkk = f"{t['max_loss_dkk']:,.0f} DKK" if t['max_loss_dkk'] else "-"
        inv_cols[2].markdown(
            f"<div style='background:#ef444422;padding:1rem;border-radius:10px;"
            f"border-left:5px solid #ef4444;text-align:center'>"
            f"<small style='color:#888'>⚠️ MAX TAB</small>"
            f"<h3 style='margin:0.3rem 0;color:#ef4444'>-${t['max_loss_usd']:,.0f}</h3>"
            f"<small><b>≈ -{loss_str_dkk}</b></small><br>"
            f"<small>Hvis stop-loss rammer</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        end_value_usd = t['invest_usd'] + t['total_profit_usd']
        end_value_dkk = end_value_usd * fx_for_plan if fx_for_plan else None
        end_str_dkk = f"{end_value_dkk:,.0f} DKK" if end_value_dkk else "-"
        inv_cols[3].markdown(
            f"<div style='background:#a855f722;padding:1rem;border-radius:10px;"
            f"border-left:5px solid #a855f7;text-align:center'>"
            f"<small style='color:#888'>🎯 SLUTVÆRDI</small>"
            f"<h3 style='margin:0.3rem 0;color:#a855f7'>${end_value_usd:,.0f}</h3>"
            f"<small><b>≈ {end_str_dkk}</b></small><br>"
            f"<small>Efter alle 3 targets</small>"
            f"</div>",
            unsafe_allow_html=True
        )

        with st.expander("📊 Sådan fordeler gevinsten sig (1/3 + 1/3 + 1/3 strategi)"):
            third_shares = t['shares'] // 3
            remaining = t['shares'] - 2 * third_shares

            breakdown_data = []
            breakdown_data.append({
                "Salg": "🎯 Target 1 (kort sigt)",
                "Antal aktier": third_shares,
                "Pris/aktie": f"{targets_main['target_short']:.2f} {currency}",
                "Gevinst (USD)": f"+${t['profit_short_usd']:,.0f}",
                "Gevinst (DKK)": f"+{t['profit_short_usd']*fx_for_plan:,.0f} DKK" if fx_for_plan else "-",
            })
            breakdown_data.append({
                "Salg": "🚀 Target 2 (lang sigt)",
                "Antal aktier": third_shares,
                "Pris/aktie": f"{targets_main['target_long']:.2f} {currency}",
                "Gevinst (USD)": f"+${t['profit_long_usd']:,.0f}",
                "Gevinst (DKK)": f"+{t['profit_long_usd']*fx_for_plan:,.0f} DKK" if fx_for_plan else "-",
            })
            breakdown_data.append({
                "Salg": "🌙 Target 3 (moon - estimat)",
                "Antal aktier": remaining,
                "Pris/aktie": f"{targets_main['target_long']*1.15:.2f} {currency}",
                "Gevinst (USD)": f"+${t['profit_moon_usd']:,.0f}",
                "Gevinst (DKK)": f"+{t['profit_moon_usd']*fx_for_plan:,.0f} DKK" if fx_for_plan else "-",
            })
            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
            st.caption(
                "💡 **OBS:** Disse tal er **forventede** gevinster hvis alle targets rammes. "
                "I virkeligheden afhænger det af markedsforhold. Brug altid stop-loss!"
            )
    else:
        if "KØB" in rec:
            st.info(
                "💡 **Tip:** Brug **Position Sizing Calculator** ovenfor til at beregne hvor mange aktier "
                "du skal købe — så får du her vist den **forventede gevinst i DKK**!"
            )

    if plan["risk_reward"]:
        rr = plan["risk_reward"]
        st.markdown("### ⚖️ Risk / Reward")
        rr_cols = st.columns(4)
        rr_cols[0].metric(
            "⚠️ Risk", f"-{rr['risk_pct']:.1f}%",
            f"-{rr['risk_dkk']:.2f} {currency}/aktie"
        )
        rr_cols[1].metric(
            "🎯 Reward (kort)", f"+{rr['reward_short_pct']:.1f}%"
        )
        rr_cols[2].metric(
            "🚀 Reward (lang)", f"+{rr['reward_long_pct']:.1f}%"
        )

        if regime in ("BEAR", "VOLATILE"):
            excellent_threshold = 4
            good_threshold = 3
            ok_threshold = 2
        else:
            excellent_threshold = 3
            good_threshold = 2
            ok_threshold = 1.5

        rr_color = (
            "#16a34a" if rr['ratio_long'] >= good_threshold
            else "#eab308" if rr['ratio_long'] >= ok_threshold
            else "#ef4444"
        )
        rr_label = (
            "Excellent" if rr['ratio_long'] >= excellent_threshold
            else "God" if rr['ratio_long'] >= good_threshold
            else "OK" if rr['ratio_long'] >= ok_threshold
            else "Svag"
        )
        rr_cols[3].markdown(
            f"<div style='background:{rr_color}22;padding:0.6rem;border-radius:8px;"
            f"border-left:4px solid {rr_color};text-align:center'>"
            f"<small>R/R RATIO (lang)</small>"
            f"<h3 style='margin:0.2rem 0;color:{rr_color}'>{rr['ratio_long']:.1f}:1</h3>"
            f"<small>{rr_label}</small></div>",
            unsafe_allow_html=True
        )

        if regime in ("BEAR", "VOLATILE") and rr['ratio_long'] < 3:
            st.warning(
                f"⚠️ I {regime} marked anbefales R/R ratio på **min. 3:1** — "
                f"din nuværende er {rr['ratio_long']:.1f}:1."
            )

    st.markdown("### 📋 Trin-for-trin handleplan")
    for step in plan["steps"]:
        st.markdown(
            f"<div style='background:{step['color']}15;padding:1rem;border-radius:10px;"
            f"border-left:5px solid {step['color']};margin-bottom:0.6rem'>"
            f"<div style='display:flex;align-items:center;gap:0.8rem'>"
            f"<div style='font-size:2rem'>{step['icon']}</div>"
            f"<div style='flex:1'>"
            f"<div style='color:{step['color']};font-weight:bold;font-size:0.9rem'>"
            f"STEP {step['n']} · {step['title']}</div>"
            f"<div style='font-size:1.1rem;margin:0.3rem 0'>{step['main']}</div>"
            f"<div style='color:#aaa;font-size:0.9rem'>{step['sub']}</div>"
            f"</div></div></div>",
            unsafe_allow_html=True
        )

    with st.expander("📚 Hvad er TRAILING STOP? (klik for forklaring)"):
        st.markdown("""
        **Trailing stop** = "rullende stop-loss" der **følger med opad** når kursen stiger.

        ### 📈 Eksempel:
        ```
        Du køber @ 120 USD, stop-loss = 114 USD (-5%)
        ✅ Kurs stiger til 130 USD  →  trailing stop bliver 123 USD
        ✅ Kurs stiger til 150 USD  →  trailing stop bliver 142 USD
        🛑 Kurs falder til 142 USD  →  SOLGT med +22 USD profit!
        ```

        ### 💼 Hvor sætter man det?
        - 🇩🇰 **Nordnet** — "Trailing stop"
        - 🇩🇰 **Saxo** — "Trailing stop loss"
        - 🌍 **eToro** — "Trailing stop loss"
        - 🌍 **Interactive Brokers** — "TRAIL"
        """)

    st.caption(
        "⚠️ Datoer og gevinster er **estimater** baseret på historisk momentum og volatilitet."
    )

    st.markdown("---")
    with st.expander("📐 Position Sizing Calculator", expanded=False):
        st.caption("Beregn hvor mange aktier du skal købe baseret på din risk tolerance")

        if regime == "BEAR":
            default_risk = 1.0
            risk_help = "🐻 Bear marked: Anbefalet 1% risk pr. trade"
        elif regime == "VOLATILE":
            default_risk = 1.5
            risk_help = "⚡ Volatile marked: Anbefalet 1.5% risk pr. trade"
        elif regime == "BULL":
            default_risk = 2.0
            risk_help = "🐂 Bull marked: 2% risk er typisk OK"
        else:
            default_risk = 2.0
            risk_help = "Standard 2% risk pr. trade"

        ps_cols = st.columns(4)
        portfolio_val = ps_cols[0].number_input(
            "💼 Din portefølje (DKK)", min_value=10000,
            value=100000, step=10000, key="ps_portfolio"
        )
        risk_pct = ps_cols[1].slider(
            "⚠️ Risk pr. trade (%)", 0.5, 5.0, default_risk, 0.5,
            key="ps_risk", help=risk_help
        )

        try:
            fv_ps = dcf_valuation(info, 0.10, 0.10, 0.025)
        except Exception:
            fv_ps = None

        targets_data = calculate_price_targets(
            filter_by_days(df_indicators, ANALYSIS_PERIODS["targets"]),
            price, fv_ps
        )

        default_stop = targets_data.get("stop_loss", price * 0.92) if targets_data else price * 0.92

        stop_loss_input = ps_cols[2].number_input(
            f"🛑 Stop-loss ({currency})",
            min_value=0.01, value=float(default_stop),
            step=0.5, key="ps_stop"
        )

        if currency != "DKK":
            fx_to_dkk = get_fx_rate(currency, "DKK")
            price_dkk = price * fx_to_dkk
            stop_dkk = stop_loss_input * fx_to_dkk
        else:
            price_dkk = price
            stop_dkk = stop_loss_input

        sizing = calculate_position_size(price_dkk, stop_dkk, portfolio_val, risk_pct)

        if sizing:
            ps_cols[3].metric(
                "📦 Antal aktier",
                f"{sizing['shares']:,}",
                f"{sizing['position_pct']:.1f}% af port."
            )

            ps_summary = st.columns(3)
            ps_summary[0].metric("💰 Position-værdi", f"{sizing['position_value']:,.0f} DKK")
            ps_summary[1].metric("⚠️ Max tab", f"{sizing['risk_amount']:,.0f} DKK", f"-{risk_pct}%")
            ps_summary[2].metric("📉 Risk pr. aktie", f"{sizing['risk_per_share']:.2f} DKK")

            if "KØB" in rec:
                regime_note = ""
                if regime == "BEAR":
                    regime_note = " 🐻 (BEAR marked — overvej halv position!)"
                elif regime == "VOLATILE":
                    regime_note = " ⚡ (VOLATILE marked — vær forsigtig)"

                st.success(
                    f"✅ **Anbefaling:** Køb **{sizing['shares']:,} aktier** "
                    f"@ {price:.2f} {currency} = {sizing['position_value']:,.0f} DKK "
                    f"({sizing['position_pct']:.1f}% af din portefølje){regime_note}"
                )
            elif "HOLD" in rec:
                st.info("ℹ️ Modellen siger HOLD - vurdér selv om du vil tage positionen")
            else:
                st.warning("⚠️ Modellen anbefaler IKKE køb lige nu")
        else:
            st.warning("Kunne ikke beregne position size (tjek input)")
                # ============================================================
    # 🆕 MAIN TABS - NU MED "📅 Earnings" TAB + earnings-markers på chart
    # ============================================================
    st.markdown("---")
    main_tabs = st.tabs([
        "📊 Charts", "🔧 Indikatorer", "💰 Kursmål",
        "📉 Risiko", "🎲 Monte Carlo", "🎯 Backtest",
        "📰 Nyheder", "📅 Earnings", "📋 Detaljer"
    ])

    # ===== CHARTS (med earnings-markører) =====
    with main_tabs[0]:
        df_chart = filter_chart_period(df_indicators, period)
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05,
            subplot_titles=("Pris + SMA + Bollinger + 📅 Earnings", "RSI", "MACD")
        )
        fig.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"], close=df_chart["Close"], name="Pris"
        ), 1, 1)
        if "SMA50" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["SMA50"],
                name="SMA50", line=dict(color="orange")
            ), 1, 1)
        if "SMA200" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["SMA200"],
                name="SMA200", line=dict(color="purple")
            ), 1, 1)
        if "BB_high" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["BB_high"],
                name="BB Upper",
                line=dict(color="rgba(255,255,255,0.3)", dash="dot")
            ), 1, 1)
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["BB_low"],
                name="BB Lower",
                line=dict(color="rgba(255,255,255,0.3)", dash="dot"),
                fill="tonexty", fillcolor="rgba(255,255,255,0.05)"
            ), 1, 1)
        if "RSI" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["RSI"],
                name="RSI", line=dict(color="#00d4aa")
            ), 2, 1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        if "MACD" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["MACD"],
                name="MACD", line=dict(color="#0099ff")
            ), 3, 1)
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["MACD_signal"],
                name="Signal", line=dict(color="orange")
            ), 3, 1)

        # 🆕 TILFØJ EARNINGS-MARKØRER PÅ MAIN PRICE CHART (row 1)
        add_earnings_markers_to_chart(
            fig=fig,
            earnings_data=earnings_data,
            hist_df=df_chart,
            row=1, col=1
        )

        fig.update_layout(
            height=800, xaxis_rangeslider_visible=False,
            template="plotly_dark",
            title=f"{ticker} - Teknisk analyse ({period})"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 🆕 VIS LEGEND under chartet
        add_earnings_legend_caption()

    # ===== INDIKATORER =====
    with main_tabs[1]:
        if not df_technical.empty:
            last = df_technical.iloc[-1]
            ic = st.columns(4)
            ic[0].metric("RSI", f"{last['RSI']:.1f}" if not pd.isna(last.get("RSI")) else "-")
            ic[1].metric("MACD", f"{last['MACD']:.2f}" if not pd.isna(last.get("MACD")) else "-")
            ic[2].metric("ATR", f"{last['ATR']:.2f}" if not pd.isna(last.get("ATR")) else "-")
            if not pd.isna(last.get("BB_high")):
                ic[3].metric(
                    "BB Width",
                    f"{((last['BB_high']-last['BB_low'])/last['Close']*100):.1f}%"
                )

            ic2 = st.columns(3)
            ic2[0].metric("SMA50", f"{last['SMA50']:.2f}" if not pd.isna(last.get("SMA50")) else "-")
            ic2[1].metric("SMA200", f"{last['SMA200']:.2f}" if not pd.isna(last.get("SMA200")) else "-")
            ic2[2].metric("ADX", f"{last['ADX']:.1f}" if not pd.isna(last.get("ADX")) else "-")

            st.markdown("---")
            st.markdown("#### 📊 Score breakdown")

            sb_tabs = st.tabs(["📊 Fundamental", "🔧 Teknisk"])
            with sb_tabs[0]:
                if f_details:
                    df_f = pd.DataFrame(f_details)
                    fig_f = px.bar(
                        df_f, x="impact", y="label", orientation="h",
                        color="impact", color_continuous_scale="RdYlGn"
                    )
                    fig_f.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.dataframe(df_f, use_container_width=True, hide_index=True)
            with sb_tabs[1]:
                if t_details:
                    df_t = pd.DataFrame(t_details)
                    fig_t = px.bar(
                        df_t, x="impact", y="label", orientation="h",
                        color="impact", color_continuous_scale="RdYlGn"
                    )
                    fig_t.update_layout(template="plotly_dark", height=400, showlegend=False)
                    st.plotly_chart(fig_t, use_container_width=True)
                    st.dataframe(df_t, use_container_width=True, hide_index=True)

    # ===== KURSMÅL =====
    with main_tabs[2]:
        df_targets = filter_by_days(df_indicators, ANALYSIS_PERIODS["targets"])

        try:
            fv_for_targets = dcf_valuation(info, 0.10, 0.10, 0.025)
        except Exception:
            fv_for_targets = None

        targets = calculate_price_targets(df_targets, price, fv_for_targets)

        if targets:
            st.markdown("### 💰 Kursniveauer (6 mdr basis)")

            buy_low_pct = (targets["buy_low"] / price - 1) * 100
            buy_high_pct = (targets["buy_high"] / price - 1) * 100
            stop_pct = (targets["stop_loss"] / price - 1) * 100
            short_pct = (targets["target_short"] / price - 1) * 100
            long_pct = (targets["target_long"] / price - 1) * 100

            fx_targets = None
            if currency != "DKK":
                fx_targets = get_fx_rate(currency, "DKK")

            def dkk_str(val):
                if fx_targets:
                    return f"≈ {val * fx_targets:,.0f} DKK"
                return ""

            tg = st.columns(5)
            tg[0].markdown(
                f"<div style='background:#16a34a22;padding:0.8rem;border-radius:10px;"
                f"border-left:4px solid #16a34a;text-align:center'>"
                f"<small>🟢 KØB ZONE</small>"
                f"<h4 style='margin:0.3rem 0'>{targets['buy_low']:.2f} - {targets['buy_high']:.2f}</h4>"
                f"<small style='color:#aaa'>{dkk_str(targets['buy_low'])}</small><br>"
                f"<small>{buy_low_pct:+.1f}% til {buy_high_pct:+.1f}%</small>"
                f"</div>", unsafe_allow_html=True
            )
            tg[1].markdown(
                f"<div style='background:#0099ff22;padding:0.8rem;border-radius:10px;"
                f"border-left:4px solid #0099ff;text-align:center'>"
                f"<small>📍 NUVÆRENDE</small>"
                f"<h4 style='margin:0.3rem 0'>{price:.2f} {currency}</h4>"
                f"<small style='color:#aaa'>{dkk_str(price)}</small><br>"
                f"<small>{change_pct:+.2f}%</small>"
                f"</div>", unsafe_allow_html=True
            )
            tg[2].markdown(
                f"<div style='background:#ef444422;padding:0.8rem;border-radius:10px;"
                f"border-left:4px solid #ef4444;text-align:center'>"
                f"<small>🛑 STOP LOSS</small>"
                f"<h4 style='margin:0.3rem 0'>{targets['stop_loss']:.2f}</h4>"
                f"<small style='color:#aaa'>{dkk_str(targets['stop_loss'])}</small><br>"
                f"<small>{stop_pct:+.1f}%</small>"
                f"</div>", unsafe_allow_html=True
            )
            tg[3].markdown(
                f"<div style='background:#eab30822;padding:0.8rem;border-radius:10px;"
                f"border-left:4px solid #eab308;text-align:center'>"
                f"<small>🎯 KORT (1-3m)</small>"
                f"<h4 style='margin:0.3rem 0'>{targets['target_short']:.2f}</h4>"
                f"<small style='color:#aaa'>{dkk_str(targets['target_short'])}</small><br>"
                f"<small>{short_pct:+.1f}%</small>"
                f"</div>", unsafe_allow_html=True
            )
            tg[4].markdown(
                f"<div style='background:#22c55e22;padding:0.8rem;border-radius:10px;"
                f"border-left:4px solid #22c55e;text-align:center'>"
                f"<small>🚀 LANG (6-12m)</small>"
                f"<h4 style='margin:0.3rem 0'>{targets['target_long']:.2f}</h4>"
                f"<small style='color:#aaa'>{dkk_str(targets['target_long'])}</small><br>"
                f"<small>{long_pct:+.1f}%</small>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### 💎 DCF Værdiansættelse")
        st.caption("Beregner fair value baseret på Discounted Cash Flow")

        dcf_cols_input = st.columns(3)
        growth_rate = dcf_cols_input[0].slider(
            "🚀 Vækstrate (år 1)", 0.02, 0.25, 0.10, 0.01,
            format="%.2f", key="dcf_growth"
        )
        discount_rate = dcf_cols_input[1].slider(
            "💸 Discount rate (WACC)", 0.05, 0.15, 0.10, 0.01,
            format="%.2f", key="dcf_discount"
        )
        terminal_growth = dcf_cols_input[2].slider(
            "🏁 Terminal vækst", 0.01, 0.05, 0.025, 0.005,
            format="%.3f", key="dcf_terminal"
        )

        try:
            fair_value = dcf_valuation(info, growth_rate, discount_rate, terminal_growth)
            if fair_value and fair_value > 0:
                dc = st.columns(4)
                dc[0].metric("💎 Fair value", f"{fair_value:.2f} {currency}")
                dc[1].metric("📍 Nuværende", f"{price:.2f} {currency}")
                upside = (fair_value / price - 1) * 100
                dc[2].metric(
                    "📊 Upside",
                    f"{upside:+.1f}%",
                    "Undervurderet" if upside > 10 else "Overvurderet" if upside < -10 else "Fair"
                )

                if upside > 30:
                    dc[3].markdown(
                        "<div style='background:#16a34a22;padding:0.6rem;border-radius:8px;"
                        "text-align:center;border-left:4px solid #16a34a'>"
                        "<small>🟢 STÆRKT UNDERVURDERET</small></div>",
                        unsafe_allow_html=True
                    )
                elif upside > 10:
                    dc[3].markdown(
                        "<div style='background:#22c55e22;padding:0.6rem;border-radius:8px;"
                        "text-align:center;border-left:4px solid #22c55e'>"
                        "<small>🟢 UNDERVURDERET</small></div>",
                        unsafe_allow_html=True
                    )
                elif upside > -10:
                    dc[3].markdown(
                        "<div style='background:#eab30822;padding:0.6rem;border-radius:8px;"
                        "text-align:center;border-left:4px solid #eab308'>"
                        "<small>🟡 FAIR PRICED</small></div>",
                        unsafe_allow_html=True
                    )
                else:
                    dc[3].markdown(
                        "<div style='background:#ef444422;padding:0.6rem;border-radius:8px;"
                        "text-align:center;border-left:4px solid #ef4444'>"
                        "<small>🔴 OVERVURDERET</small></div>",
                        unsafe_allow_html=True
                    )

                st.caption(
                    f"⚙️ Antagelser: Vækst **{growth_rate*100:.0f}%** → terminal **{terminal_growth*100:.1f}%** "
                    f"(10 år) · Discount **{discount_rate*100:.0f}%**"
                )
            else:
                st.info("ℹ️ DCF kræver positiv Free Cash Flow data (ikke tilgængelig for denne aktie)")
        except Exception as e:
            st.warning(f"⚠️ DCF kunne ikke beregnes: {str(e)[:100]}")

    # ===== RISIKO =====
    with main_tabs[3]:
        df_risk = filter_by_days(df_indicators, ANALYSIS_PERIODS["risk"])
        try:
            risk = risk_metrics(df_risk)
        except Exception as e:
            st.warning(f"⚠️ Risk metrics fejlede: {str(e)[:100]}")
            risk = None

        if risk:
            st.caption("📉 Risk metrics (3 års data)")
            rc = st.columns(4)
            rc[0].metric("Ann. afkast", f"{risk['ann_r']*100:.1f}%")
            rc[1].metric("Ann. volatilitet", f"{risk['ann_v']*100:.1f}%")
            rc[2].metric("Sharpe", f"{risk['sharpe']:.2f}")
            rc[3].metric("Sortino", f"{risk['sortino']:.2f}")

            rc2 = st.columns(2)
            rc2[0].metric("Max Drawdown", f"{risk['max_dd']*100:.1f}%")
            rc2[1].metric("VaR 95% (1d)", f"{risk['var95']*100:.2f}%")

            fig_dd = go.Figure(go.Scatter(
                x=risk["dd_series"].index,
                y=risk["dd_series"] * 100,
                fill="tozeroy", line=dict(color="#ef4444")
            ))
            fig_dd.update_layout(template="plotly_dark", height=350, title="Drawdown %")
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning("Ikke nok data til risk metrics")

    # ===== MONTE CARLO =====
    with main_tabs[4]:
        st.caption("🎲 Simulerer fremtidige prisbaner baseret på historisk afkast & volatilitet")

        df_mc = filter_by_days(df_indicators, ANALYSIS_PERIODS["monte_carlo"])

        mc_cols_input = st.columns(2)
        mc_days = mc_cols_input[0].slider("📅 Dage frem", 30, 365, 252, key="stock_mc_days")
        mc_sims = mc_cols_input[1].slider("🎲 Antal simulationer", 100, 1000, 500, 100, key="stock_mc_sims")

        try:
            sims, lp = monte_carlo(df_mc, days=mc_days, sims=mc_sims)
        except Exception as e:
            st.warning(f"⚠️ Monte Carlo fejlede: {str(e)[:100]}")
            sims, lp = None, None

        if sims is not None and lp is not None:
            final = sims[:, -1]
            p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])

            mc_cols = st.columns(5)
            mc_cols[0].metric("5% (worst)", f"{p5:.2f} {currency}", f"{(p5/lp-1)*100:+.0f}%")
            mc_cols[1].metric("25%", f"{p25:.2f} {currency}", f"{(p25/lp-1)*100:+.0f}%")
            mc_cols[2].metric(f"📊 Median ({mc_days}d)", f"{p50:.2f} {currency}", f"{(p50/lp-1)*100:+.0f}%")
            mc_cols[3].metric("75%", f"{p75:.2f} {currency}", f"{(p75/lp-1)*100:+.0f}%")
            mc_cols[4].metric("95% (best)", f"{p95:.2f} {currency}", f"{(p95/lp-1)*100:+.0f}%")

            prob_positive = (final > lp).sum() / len(final) * 100
            expected_return = (p50 / lp - 1) * 100

            prob_cols = st.columns(3)
            prob_cols[0].metric("📈 Sandsynlighed for plus", f"{prob_positive:.0f}%")
            prob_cols[1].metric("📉 Sandsynlighed for minus", f"{100-prob_positive:.0f}%")
            prob_cols[2].metric("💰 Forventet afkast", f"{expected_return:+.1f}%", f"over {mc_days} dage")

            fig_m = go.Figure()
            for i in range(min(150, len(sims))):
                fig_m.add_trace(go.Scatter(
                    y=sims[i],
                    line=dict(width=0.5, color="rgba(0,212,170,0.1)"),
                    showlegend=False, hoverinfo="skip"
                ))
            fig_m.add_trace(go.Scatter(
                y=np.percentile(sims, 95, axis=0),
                name="95% (best case)",
                line=dict(color="#22c55e", width=2, dash="dash")
            ))
            fig_m.add_trace(go.Scatter(
                y=np.percentile(sims, 50, axis=0),
                name="Median",
                line=dict(color="#00d4aa", width=3)
            ))
            fig_m.add_trace(go.Scatter(
                y=np.percentile(sims, 5, axis=0),
                name="5% (worst case)",
                line=dict(color="#ef4444", width=2, dash="dash")
            ))
            fig_m.add_hline(
                y=lp, line_dash="dot", line_color="white",
                opacity=0.5, annotation_text=f"Nu: {lp:.2f}"
            )
            fig_m.update_layout(
                template="plotly_dark", height=500,
                title=f"Monte Carlo - {mc_sims} simulationer, {mc_days} dage frem",
                yaxis_title=f"Pris ({currency})",
                xaxis_title="Dage frem"
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("ℹ️ Ikke nok data til Monte Carlo simulation")

    # ===== BACKTEST =====
    with main_tabs[5]:
        st.caption("🎯 Walk-forward backtest af model-anbefalinger")

        bc1, bc2 = st.columns(2)
        holding = bc1.selectbox(
            "Holding periode (dage)",
            [30, 60, 90, 180, 365], index=1, key="stock_hold"
        )
        freq = bc2.selectbox(
            "Sample frekvens", [7, 14, 30], index=1, key="stock_freq"
        )

        if st.button("🚀 Kør backtest", type="primary", key="btn_stock_bt"):
            with st.spinner("Kører walk-forward..."):
                try:
                    bt = run_backtest(hist, info, holding_days=holding, sample_freq=freq)
                except Exception as e:
                    st.error(f"Backtest fejlede: {str(e)[:200]}")
                    bt = None

            if bt is None:
                st.error(f"Ikke nok data ({len(hist)} dage)")
            else:
                st.markdown(
                    f"📊 **{bt['n_trades']} samples** · "
                    f"{bt['start_date'].date()} → {bt['end_date'].date()}"
                )

                rows = []
                for rec_lbl in ["KØB", "HOLD", "SÆLG"]:
                    s = bt["stats"].get(rec_lbl)
                    if s:
                        rows.append({
                            "Anbefaling": rec_lbl, "Antal": s["count"],
                            "Hit rate": f"{s['win_rate']:.1f}%",
                            "Gns. afkast": f"{s['avg_return']:+.2f}%",
                            "Median": f"{s['median_return']:+.2f}%",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown(f"📈 **Buy & Hold:** {bt['buy_hold_return']:+.2f}%")

                fig_bt = px.scatter(
                    bt["results"], x="score", y="return_pct",
                    color="recommendation",
                    color_discrete_map={"KØB": "#22c55e", "HOLD": "#eab308", "SÆLG": "#ef4444"},
                    title=f"Score vs {holding}-dages afkast"
                )
                fig_bt.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                fig_bt.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_bt, use_container_width=True)

    # ===== NYHEDER =====
    with main_tabs[6]:
        st.markdown("### 📰 Seneste nyheder & sentiment-analyse")
        st.caption(f"Henter automatisk seneste nyhedsartikler om **{company_name}** og analyserer sentiment")

        if sentiment_data is None or sentiment_data.get("article_count", 0) == 0:
            st.info(
                "💡 **Ingen nyheder fundet for denne ticker.**\n\n"
                "Mulige årsager:\n"
                "- Ticker er for niche / lille\n"
                "- News API er rate-limited\n"
                "- Ingen API-key konfigureret\n\n"
                "Tjek `news_sentiment.py` for konfiguration."
            )
        else:
            # Vis full sentiment summary (ikke compact)
            render_sentiment_summary(sentiment_data, compact=False)

            st.markdown("---")
            st.markdown("#### 📑 Nyhedsfeed")

            # Filter-controls
            filter_cols = st.columns([2, 1, 1])
            filter_sentiment = filter_cols[0].selectbox(
                "Filtrér efter sentiment",
                ["Alle", "🟢 Kun positive", "🔴 Kun negative", "🟡 Kun neutrale"],
                key="news_filter"
            )
            sort_by = filter_cols[1].selectbox(
                "Sortér efter",
                ["Nyeste først", "Mest positive", "Mest negative"],
                key="news_sort"
            )
            max_items = filter_cols[2].number_input(
                "Max artikler", min_value=5, max_value=50, value=15, step=5,
                key="news_max"
            )

            # Render feed med filtre
            render_news_feed(
                sentiment_data,
                filter_type=filter_sentiment,
                sort_by=sort_by,
                max_items=max_items
            )

            # Genopfrisk-knap
            st.markdown("---")
            if st.button("🔄 Genhent nyheder", key="refresh_news"):
                # Ryd cache for denne ticker
                get_news_sentiment.clear() if hasattr(get_news_sentiment, "clear") else None
                st.cache_data.clear()
                st.rerun()

    # ===== 🆕 EARNINGS =====
    with main_tabs[7]:
        st.markdown("### 📅 Earnings-analyse")
        st.caption(
            f"Komplet earnings-overblik for **{company_name}** — "
            "kommende rapporter, historik og post-earnings prisbevægelser."
        )

        if earnings_data is None:
            st.info(
                "💡 **Ingen earnings-data fundet for denne ticker.**\n\n"
                "Mulige årsager:\n"
                "- Ticker er for niche / lille\n"
                "- API er rate-limited\n"
                "- Ingen earnings-historik tilgængelig\n\n"
                "Tjek `earnings_warning.py` for konfiguration."
            )
        else:
            # Fuld earnings warning (ikke compact)
            render_earnings_warning(earnings_data, compact=False)

            st.markdown("---")

            # Earnings history (sub-tabs)
            earnings_subtabs = st.tabs([
                "📜 Historik (beat/miss)",
                "📊 Post-earnings bevægelser",
                "ℹ️ Hvorfor earnings betyder noget?"
            ])

            with earnings_subtabs[0]:
                st.markdown("#### 📜 Earnings-historik")
                st.caption("Sammenligning af forventet EPS vs faktisk EPS for de seneste rapporter")
                render_earnings_history(earnings_data)

            with earnings_subtabs[1]:
                st.markdown("#### 📊 Post-earnings prisbevægelser")
                st.caption(
                    "Hvor meget bevægede aktien sig dagen efter sidste earnings-rapporter? "
                    "Bruges til at estimere forventet volatilitet ved næste earnings."
                )
                render_post_earnings_moves(earnings_data)

            with earnings_subtabs[2]:
                st.markdown("#### ℹ️ Hvorfor er earnings vigtige?")
                st.markdown("""
                **Earnings-rapporter** er kvartalsvise opdateringer hvor virksomheder offentliggør:
                - 📊 **Indtjening (EPS)** — hvor meget de tjente per aktie
                - 💰 **Omsætning (Revenue)** — hvor meget de solgte for
                - 🔮 **Guidance** — deres forventninger til kommende kvartal/år

                ### 🎯 Hvorfor påvirker det aktiekursen?

                **Earnings = sandheden.** Det er det øjeblik hvor markedet får facts i hånden,
                og kursen kan svinge **5-15%** på minutter — nogle gange mere!

                ### ⚠️ Risici ved at handle ind før earnings:

                1. **Earnings surprise** — selv hvis tal er gode, kan markedet være skuffet
                2. **Guidance-cut** — virksomheden kan sænke fremtidige forventninger
                3. **Implied volatility crush** — optioner mister værdi efter earnings
                4. **Whipsaw** — kursen kan først stige, så styrtdykke (eller omvendt)

                ### ✅ Bedste praksis:

                - 🛑 **Undgå nye positioner** 1-3 dage før earnings
                - 📉 **Reducér position** hvis du allerede har en
                - 🎯 **Brug stop-loss** der ikke kan rammes af pre-earnings volatilitet
                - 📊 **Vent til efter earnings** — så er usikkerheden væk
                - 💎 **Hvis langtidsinvestor:** Kortvarige sving betyder mindre

                ### 📈 Hvornår er det OK at købe før earnings?

                - ✅ Du tror på langsigtet thesis (5+ år)
                - ✅ Du har lille position (1-3% af portefølje)
                - ✅ Stærk historik af earnings beats (track record)
                - ✅ Lav implied volatility (forventet bevægelse er lille)

                ### 🚫 Hvornår skal du ALDRIG købe før earnings?

                - ❌ Du har allerede stor position
                - ❌ Aktien er steget meget op til earnings (priced for perfection)
                - ❌ Sektoren har givet svage guidance
                - ❌ Du bruger gearing (margin/lån)
                """)

    # ===== DETALJER =====
    with main_tabs[8]:
        det_cols = st.columns(2)

        with det_cols[0]:
            st.markdown("#### 📊 Fundamentale nøgletal")
            fund_data = []
            for label, key, fmt in [
                ("Market Cap", "marketCap", "currency_b"),
                ("P/E (TTM)", "trailingPE", "ratio"),
                ("Forward P/E", "forwardPE", "ratio"),
                ("PEG", "pegRatio", "ratio"),
                ("P/B", "priceToBook", "ratio"),
                ("ROE", "returnOnEquity", "percent"),
                ("Profit margin", "profitMargins", "percent"),
                ("Debt/Equity", "debtToEquity", "ratio"),
                ("EPS Growth", "earningsGrowth", "percent"),
                ("Revenue Growth", "revenueGrowth", "percent"),
                ("Dividend %", "dividendYield", "percent"),
                ("Payout ratio", "payoutRatio", "percent"),
                ("Beta", "beta", "ratio"),
            ]:
                v = info.get(key)
                if v is None:
                    formatted = "-"
                elif fmt == "currency_b":
                    formatted = f"${v/1e9:.2f}B" if v >= 1e9 else f"${v/1e6:.0f}M"
                elif fmt == "percent":
                    formatted = f"{v*100:.2f}%" if abs(v) < 5 else f"{v:.2f}%"
                elif fmt == "ratio":
                    formatted = f"{v:.2f}"
                else:
                    formatted = str(v)
                fund_data.append({"Metric": label, "Værdi": formatted})

            st.dataframe(pd.DataFrame(fund_data), use_container_width=True, hide_index=True)

        with det_cols[1]:
            st.markdown("#### 📍 Position vs ranges")
            pos_data = []
            for label, key in [
                ("52w høj", "fiftyTwoWeekHigh"),
                ("52w lav", "fiftyTwoWeekLow"),
                ("Dagshigh", "dayHigh"),
                ("Dagslow", "dayLow"),
                ("Volume", "volume"),
                ("Avg volume", "averageVolume"),
            ]:
                v = info.get(key)
                if v is None:
                    formatted = "-"
                elif "olume" in key:
                    formatted = f"{v:,.0f}"
                else:
                    formatted = f"{v:.2f} {currency}"
                pos_data.append({"Metric": label, "Værdi": formatted})

            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

        if info.get("longBusinessSummary"):
            with st.expander("ℹ️ Om virksomheden"):
                st.write(info["longBusinessSummary"][:2000])


# ============ DEV MODE FOOTER (PERFORMANCE STATS) ============

if st.session_state.dev_mode:
    st.markdown("---")
    st.markdown("### 🐛 Dev Mode — Performance Stats")

    _total_time = time.time() - _app_start_time

    dev_cols = st.columns(4)
    dev_cols[0].metric("⏱️ Total render-tid", f"{_total_time:.2f}s")

    try:
        cache_info = "✅ Aktiv"
        dev_cols[1].metric("💾 Cache", cache_info)
    except Exception:
        dev_cols[1].metric("💾 Cache", "?")

    dev_cols[2].metric("📍 Aktiv view", st.session_state.active_view)
    dev_cols[3].metric("📋 Watchlist", f"{len(st.session_state.watchlist)} tickers")

    with st.expander("🔍 Session state (debug)"):
        debug_state = {
            "current_ticker": st.session_state.get("current_ticker", ""),
            "active_view": st.session_state.get("active_view", ""),
            "last_source": st.session_state.get("last_source", ""),
            "watchlist_count": len(st.session_state.get("watchlist", [])),
            "search_history_count": len(st.session_state.get("search_history", [])),
            "search_history": st.session_state.get("search_history", []),
            "screener_has_results": st.session_state.get("screener_results") is not None,
            "crypto_analyzed": st.session_state.get("crypto_analyzed", "ingen"),
        }
        st.json(debug_state)

    st.caption(
        "💡 **Tip:** Hvis render-tid > 5s, er der typisk ventetid på API-kald. "
        "Tryk **🔄 Ryd cache** kun hvis nødvendigt — det tvinger refetch af alt."
    )
