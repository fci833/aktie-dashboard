"""Aktie Dashboard - Hovedapp"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import ANALYSIS_PERIODS, SCREENER_UNIVERSES
from data_sources import (
    fetch_data, search_tickers, get_fx_rate, get_api_keys,
    fetch_yahoo, fetch_twelve_single, get_twelve_formats,
    FINNHUB_AVAILABLE
)
from analysis import (
    get_indicators, fundamental_score, technical_score,
    calculate_price_targets, dcf_valuation, risk_metrics,
    monte_carlo, recommendation, filter_by_days, filter_chart_period
)
from backtest import run_backtest, simulate_strategy
from screener import run_screener, categorize_opportunities, sector_breakdown
from history import (
    save_snapshot, list_snapshots, load_snapshot,
    compare_snapshots, get_hot_stocks, get_score_history,
    cleanup_old_snapshots
)
from ui_helpers import make_price_box, make_range_box, make_recommendation_card

import warnings
warnings.filterwarnings("ignore")
import requests as plain_requests

st.set_page_config(page_title="Aktie Dashboard", layout="wide", page_icon="📈")
st.markdown(
    "<h1 style='background:linear-gradient(90deg,#00d4aa,#0099ff);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
    "📈 Pro Aktie Analyse Dashboard</h1>",
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
    st.session_state.active_view = "📊 Analyse"


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


# ============ HJÆLPER: Skift til analyse ============

def goto_analysis(ticker):
    """Helper der skifter til analyse-fanen med en given ticker"""
    st.session_state.current_ticker = ticker
    st.session_state.active_view = "📊 Analyse"
    st.rerun()


# ============ SIDEBAR ============

with st.sidebar:
    st.markdown("### 📡 Datakilder")
    st.caption("✅ Yahoo Finance")
    st.caption("✅ Twelve Data" if TWELVE_KEY else "⚠️ Twelve Data (no key)")
    st.caption("✅ Finnhub" if FINNHUB_KEY else "⚠️ Finnhub (no key)")
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


# ============ NAVIGATION (radio i stedet for tabs så vi kan skifte programmatisk) ============

view_options = ["📊 Analyse", "🔎 Screener", "🔍 Søg ticker", "🔧 Diagnose"]
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


# ============ SØGE-VIEW ============

if st.session_state.active_view == "🔍 Søg ticker":
    st.subheader("🔍 Find ticker for et firma")
    query = st.text_input("Firmanavn", value="", key="search_query",
                          placeholder="novo nordisk")
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
    st.subheader("🔧 Diagnose - Test datakilder")
    diag_ticker = st.text_input("Test ticker", value="AAPL",
                                key="diag_ticker").strip().upper()
    if st.button("🔍 Kør diagnose", type="primary"):
        with st.spinner(f"Tester for {diag_ticker}..."):
            results = run_diagnostics(diag_ticker)
        for source, status, time_taken, details in results:
            with st.expander(f"{status} **{source}** ({time_taken})", expanded=True):
                st.code(details)


# ============ SCREENER-VIEW ============

elif st.session_state.active_view == "🔎 Screener":
    st.subheader("🔎 Markedsscreener")
    st.caption(
        "Find gode købsmuligheder · Sammenlign over tid · "
        "Sektoranalyse · Hot stocks"
    )

    sc_modes = st.tabs([
        "🚀 Kør screener",
        "📊 Sektor-breakdown",
        "🔔 Sammenlign",
        "🔥 Hot stocks",
        "📜 Historik",
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
                value="AAPL\nMSFT\nGOOGL\nNVDA\nTSLA",
                height=150,
            )
            tickers = [
                t.strip().upper() for t in
                custom_text.replace(",", "\n").split("\n") if t.strip()
            ]
        else:
            tickers = SCREENER_UNIVERSES[universe_name]

        st.info(
            f"📋 **{len(tickers)} tickers** · "
            f"Estimeret tid: ~{len(tickers)*2/max_workers:.0f}s"
        )

        col_run, col_snapshot, col_clear = st.columns([2, 1, 1])

        if col_run.button("🚀 Kør screener", type="primary",
                          use_container_width=True):
            progress = st.progress(0, text="Starter...")

            def update_progress(done, total, ticker):
                progress.progress(done / total,
                                  text=f"Analyseret {done}/{total}: {ticker}")

            with st.spinner("Scanner marked..."):
                df_all, df_buys = run_screener(
                    tickers, min_score=min_score,
                    max_workers=max_workers,
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

        # Vis resultater
        if st.session_state.screener_results:
            res = st.session_state.screener_results
            df_all = res["all"]
            df_buys = res["buys"]

            st.markdown("---")
            st.markdown(f"### 📊 Resultater: **{res['universe']}**")
            st.caption(
                f"⏱️ {res['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} · "
                f"Min. score: {res['min_score']}"
            )

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("📋 Total scannet", len(df_all))
            sm2.metric("✅ Succes", (df_all["status"] == "✅").sum())
            sm3.metric("🟢 Køb-muligheder", len(df_buys))
            if not df_buys.empty:
                sm4.metric("🏆 Topscore", f"{df_buys['overall'].max():.1f}")

            if df_buys.empty:
                st.warning(
                    f"⚠️ Ingen aktier opfylder min. score {res['min_score']}."
                )
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
                        top_display[col] = pd.to_numeric(
                            top_display[col], errors="coerce"
                        ).round(2)

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

                # Kategorier
                st.markdown("---")
                st.markdown("### 🎯 Købsmuligheder pr. kategori")
                categories = categorize_opportunities(df_buys)
                if categories:
                    for cat_name, cat_df in categories.items():
                        with st.expander(
                            f"{cat_name} ({len(cat_df)} aktier)"
                        ):
                            cat_d = cat_df[
                                [c for c in display_cols if c in cat_df.columns]
                            ].rename(columns=display_cols)
                            for col in ["Pris", "Ændr %", "Score", "RSI"]:
                                if col in cat_d.columns:
                                    cat_d[col] = pd.to_numeric(
                                        cat_d[col], errors="coerce"
                                    ).round(2)
                            st.dataframe(cat_d, use_container_width=True,
                                         hide_index=True)

                # Score-distribution scatter
                st.markdown("---")
                st.markdown("### 📈 Score-distribution")
                fig = px.scatter(
                    df_all[df_all["status"] == "✅"],
                    x="f_score", y="t_score",
                    size="overall", color="overall",
                    hover_data=["ticker", "name", "recommendation"],
                    color_continuous_scale="RdYlGn",
                    title="Fundamental vs Teknisk score",
                    labels={"f_score": "Fundamental score", "t_score": "Teknisk score"},
                )
                fig.add_hline(y=60, line_dash="dash", line_color="green", opacity=0.3)
                fig.add_vline(x=60, line_dash="dash", line_color="green", opacity=0.3)
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # CSV eksport
                csv = df_buys.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download købsmuligheder (CSV)", csv,
                    f"screener_{res['timestamp'].strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                )
        else:
            st.info("👆 Vælg et univers og tryk **🚀 Kør screener**")

    # ===== MODE 2: SEKTOR-BREAKDOWN =====
    with sc_modes[1]:
        st.markdown("### 📊 Sektor-breakdown")
        st.caption(
            "Grupperer screening-resultater per sektor og viser bedste pr. sektor"
        )

        if not st.session_state.screener_results:
            st.info("👈 Kør først en screener i fanen **🚀 Kør screener**")
        else:
            df_all = st.session_state.screener_results["all"]
            sectors = sector_breakdown(df_all)

            if not sectors:
                st.warning("⚠️ Ingen sektor-info tilgængelig "
                           "(fungerer bedst med Yahoo+Finnhub data)")
            else:
                st.markdown("#### 🏆 Sektor-rangering")
                rank_df = pd.DataFrame([
                    {
                        "Sektor": name,
                        "Antal aktier": s["count"],
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
                            "Gns. score", min_value=0, max_value=100,
                            format="%.0f"
                        ),
                    },
                )

                fig_sec = px.bar(
                    rank_df.sort_values("Gns. score", ascending=True),
                    x="Gns. score", y="Sektor", orientation="h",
                    color="Gns. score", color_continuous_scale="RdYlGn",
                    title="Gennemsnitlig score per sektor",
                    text="Antal aktier",
                )
                fig_sec.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_sec, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🔍 Top 3 pr. sektor")
                for name, s in sectors.items():
                    with st.expander(
                        f"**{name}** · {s['count']} aktier · "
                        f"gns. {s['avg_score']:.1f}"
                    ):
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
        st.caption(
            "Find aktier der har ændret rating, fået højere/lavere score "
            "siden sidst"
        )

        all_snaps = list_snapshots()
        if len(all_snaps) < 2:
            st.warning(
                f"⚠️ Du har kun {len(all_snaps)} snapshot(s). "
                "Du skal have mindst 2 for at sammenligne. "
                "Kør screeneren igen senere (eller på en anden dag)."
            )
        else:
            unique_universes = list(set(s["universe"] for s in all_snaps))
            sel_universe = st.selectbox(
                "🌍 Univers", unique_universes, key="cmp_universe"
            )
            uni_snaps = [s for s in all_snaps if s["universe"] == sel_universe]

            if len(uni_snaps) < 2:
                st.warning(f"Kun 1 snapshot for {sel_universe}")
            else:
                cmp1, cmp2 = st.columns(2)
                snap_now_idx = cmp1.selectbox(
                    "📅 NU (senest)",
                    range(len(uni_snaps)),
                    format_func=lambda i: uni_snaps[i]["filename"],
                    index=0, key="cmp_now",
                )
                snap_prev_idx = cmp2.selectbox(
                    "📅 FØR (sammenlign med)",
                    range(len(uni_snaps)),
                    format_func=lambda i: uni_snaps[i]["filename"],
                    index=min(1, len(uni_snaps)-1), key="cmp_prev",
                )

                if st.button("🔍 Sammenlign", type="primary"):
                    df_now, ts_now, _ = load_snapshot(
                        uni_snaps[snap_now_idx]["file"]
                    )
                    df_prev, ts_prev, _ = load_snapshot(
                        uni_snaps[snap_prev_idx]["file"]
                    )
                    cmp = compare_snapshots(df_now, df_prev)

                    if cmp.empty:
                        st.error("Kunne ikke sammenligne (manglende data)")
                    else:
                        st.success(f"✅ Sammenlignet "
                                   f"{ts_prev[:10]} → {ts_now[:10]}")

                        if "rating_changed" in cmp.columns:
                            changed = cmp[cmp["rating_changed"] == True].copy()

                            st.markdown("#### 🚨 Rating-ændringer")
                            if changed.empty:
                                st.info("Ingen aktier har ændret rating")
                            else:
                                rec_order = ["STÆRKT KØB", "KØB", "HOLD",
                                             "SÆLG", "STÆRKT SÆLG"]
                                changed["was_idx"] = changed[
                                    "recommendation_prev"
                                ].apply(
                                    lambda x: rec_order.index(x)
                                    if x in rec_order else 99
                                )
                                changed["now_idx"] = changed[
                                    "recommendation_now"
                                ].apply(
                                    lambda x: rec_order.index(x)
                                    if x in rec_order else 99
                                )

                                upgraded = changed[
                                    changed["now_idx"] < changed["was_idx"]
                                ]
                                downgraded = changed[
                                    changed["now_idx"] > changed["was_idx"]
                                ]

                                if not upgraded.empty:
                                    st.markdown("##### 🟢 Opgraderet")
                                    st.dataframe(
                                        upgraded[[
                                            "ticker", "name",
                                            "recommendation_prev",
                                            "recommendation_now",
                                            "score_change", "price_change_%"
                                        ]].rename(columns={
                                            "recommendation_prev": "Var",
                                            "recommendation_now": "Nu",
                                            "score_change": "Score Δ",
                                            "price_change_%": "Pris Δ%",
                                        }).round(2),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                if not downgraded.empty:
                                    st.markdown("##### 🔴 Nedgraderet")
                                    st.dataframe(
                                        downgraded[[
                                            "ticker", "name",
                                            "recommendation_prev",
                                            "recommendation_now",
                                            "score_change", "price_change_%"
                                        ]].rename(columns={
                                            "recommendation_prev": "Var",
                                            "recommendation_now": "Nu",
                                            "score_change": "Score Δ",
                                            "price_change_%": "Pris Δ%",
                                        }).round(2),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                        st.markdown("#### 📈 Største score-stigninger")
                        if "score_change" in cmp.columns:
                            risers = cmp.dropna(subset=["score_change"]).nlargest(
                                10, "score_change"
                            )
                            st.dataframe(
                                risers[[
                                    "ticker", "name", "overall_now",
                                    "overall_prev", "score_change",
                                    "price_change_%"
                                ]].round(2),
                                use_container_width=True, hide_index=True,
                            )

                            st.markdown("#### 📉 Største score-fald")
                            fallers = cmp.dropna(subset=["score_change"]).nsmallest(
                                10, "score_change"
                            )
                            st.dataframe(
                                fallers[[
                                    "ticker", "name", "overall_now",
                                    "overall_prev", "score_change",
                                    "price_change_%"
                                ]].round(2),
                                use_container_width=True, hide_index=True,
                            )

    # ===== MODE 4: HOT STOCKS =====
    with sc_modes[3]:
        st.markdown("### 🔥 Hot stocks - stigende score over tid")
        st.caption(
            "Find aktier hvor scoren er steget de sidste N snapshots."
        )

        all_snaps = list_snapshots()
        if len(all_snaps) < 2:
            st.warning(
                "⚠️ Du har brug for mindst 2 snapshots. "
                "Kør screeneren over flere dage."
            )
        else:
            unique_universes = list(set(s["universe"] for s in all_snaps))
            hot_universe = st.selectbox(
                "🌍 Univers", unique_universes, key="hot_universe"
            )

            hcol1, hcol2 = st.columns(2)
            n_days = hcol1.slider("📅 Antal snapshots tilbage", 2, 20, 5,
                                  key="hot_n_days")
            min_change = hcol2.slider("📊 Min. score-ændring", 1, 30, 5,
                                      key="hot_min_change")

            if st.button("🔥 Find hot stocks", type="primary"):
                hot = get_hot_stocks(
                    hot_universe, n_days=n_days, min_change=min_change
                )

                if not hot or (hot.get("risers") is not None
                               and hot["risers"].empty
                               and hot["fallers"].empty):
                    st.warning(
                        "Ingen aktier opfylder kriterierne. "
                        "Prøv at sænke min. ændring."
                    )
                else:
                    st.caption(
                        f"📅 Fra {hot['oldest_ts'][:10]} → "
                        f"{hot['latest_ts'][:10]} "
                        f"({hot['n_snapshots']} snapshots)"
                    )

                    if not hot["risers"].empty:
                        st.markdown("#### 🚀 Stigende score (risers)")
                        risers_disp = hot["risers"][[
                            c for c in [
                                "ticker", "name", "overall_now",
                                "overall_old", "score_change",
                                "recommendation", "price_change_%"
                            ] if c in hot["risers"].columns
                        ]].copy()
                        for col in ["overall_now", "overall_old",
                                    "score_change", "price_change_%"]:
                            if col in risers_disp.columns:
                                risers_disp[col] = pd.to_numeric(
                                    risers_disp[col], errors="coerce"
                                ).round(2)
                        st.dataframe(
                            risers_disp, use_container_width=True,
                            hide_index=True,
                        )

                        fig_hot = px.bar(
                            hot["risers"].head(15),
                            x="score_change", y="ticker", orientation="h",
                            color="score_change",
                            color_continuous_scale="Greens",
                            title="Top 15 score-stigninger",
                        )
                        fig_hot.update_layout(template="plotly_dark", height=500)
                        st.plotly_chart(fig_hot, use_container_width=True)

                        # Knapper til hurtig analyse
                        st.markdown("##### 👆 Hurtig analyse:")
                        hot_cols = st.columns(4)
                        for i, (_, row) in enumerate(hot["risers"].head(8).iterrows()):
                            if hot_cols[i % 4].button(
                                f"📊 {row['ticker']} (+{row['score_change']:.0f})",
                                key=f"hot_{row['ticker']}",
                                use_container_width=True,
                            ):
                                goto_analysis(row["ticker"])

                    if not hot["fallers"].empty:
                        with st.expander(
                            f"📉 Faldende score ({len(hot['fallers'])} aktier)"
                        ):
                            fallers_disp = hot["fallers"].copy()
                            for col in ["overall_now", "overall_old",
                                        "score_change", "price_change_%"]:
                                if col in fallers_disp.columns:
                                    fallers_disp[col] = pd.to_numeric(
                                        fallers_disp[col], errors="coerce"
                                    ).round(2)
                            st.dataframe(
                                fallers_disp, use_container_width=True,
                                hide_index=True,
                            )

    # ===== MODE 5: HISTORIK =====
    with sc_modes[4]:
        st.markdown("### 📜 Snapshot-historik")
        st.caption(
            "Alle gemte screeninger. Bruges til sammenligning og hot stocks."
        )

        all_snaps = list_snapshots()
        if not all_snaps:
            st.info("Ingen snapshots gemt endnu. "
                    "Kør en screener for at oprette din første!")
        else:
            st.success(f"✅ {len(all_snaps)} snapshots gemt")

            snap_df = pd.DataFrame([
                {
                    "Filnavn": s["filename"],
                    "Univers": s["universe"],
                    "Tidspunkt": s["timestamp"][:19],
                    "Antal tickers": s["n_tickers"],
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
                    st.warning(f"Ingen historik for {track_ticker} "
                               "(check at den er i et tidligere screen)")
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
                        template="plotly_dark", height=400,
                        yaxis_range=[0, 100],
                    )
                    fig_track.add_hline(y=60, line_dash="dash",
                                        line_color="green", opacity=0.4)
                    fig_track.add_hline(y=30, line_dash="dash",
                                        line_color="red", opacity=0.4)
                    st.plotly_chart(fig_track, use_container_width=True)

                    st.dataframe(hist_df, use_container_width=True,
                                 hide_index=True)

            st.markdown("---")
            st.markdown("#### 🧹 Vedligeholdelse")
            cl1, cl2 = st.columns(2)
            cleanup_days = cl1.slider("Slet snapshots ældre end (dage)",
                                       7, 365, 60, key="cleanup_days")
            if cl2.button("🗑️ Ryd op", use_container_width=True):
                deleted = cleanup_old_snapshots(cleanup_days)
                st.success(f"Slettet {deleted} gamle snapshots")
                time.sleep(1)
                st.rerun()


# ============ HOVED-ANALYSE-VIEW ============

elif st.session_state.active_view == "📊 Analyse":
    c1, c2 = st.columns([4, 1])
    default_t = st.session_state.current_ticker or "AAPL"
    ticker_input = c1.text_input(
        "Ticker (fx AAPL, NOVO-B.CO)", value=default_t,
        key="ticker_input"
    ).strip().upper()
    auto_analyze = c2.button("🔍 Analysér", type="primary", use_container_width=True)

    if auto_analyze or st.session_state.current_ticker == ticker_input:
        st.session_state.current_ticker = ticker_input
    ticker = ticker_input

    if not ticker:
        st.info("👆 Indtast en ticker, eller brug **🔍 Søg ticker** fanen")
        st.stop()

    with st.spinner(f"Henter data for {ticker}..."):
        data = fetch_data(ticker)

    if data is None:
        st.error(f"❌ Kunne ikke hente data for '{ticker}'")
        st.info("👉 Prøv **🔍 Søg ticker** fanen")
        st.stop()

    st.session_state.last_source = data["source"]
    if data.get("warning"):
        st.warning(f"⚠️ {data['warning']}")
    else:
        st.success(f"✅ Data hentet fra: **{data['source']}**")

    st.markdown(
        "<div style='background:#0099ff15;padding:0.6rem 1rem;border-radius:8px;"
        "border-left:4px solid #0099ff;margin:0.5rem 0'>"
        f"📅 <b>Chart:</b> {period} · "
        "ℹ️ <b>Beregninger:</b> Tekniske=12mdr · Kursmål=6mdr · Risk=3år · Monte Carlo=2år"
        "</div>",
        unsafe_allow_html=True,
    )

    info = data["info"]
    hist_full = data["hist"]
    hist_chart = filter_chart_period(hist_full, period)
    hist_technical = filter_by_days(hist_full, ANALYSIS_PERIODS["technical"])
    hist_targets = filter_by_days(hist_full, ANALYSIS_PERIODS["targets"])
    hist_risk = filter_by_days(hist_full, ANALYSIS_PERIODS["risk"])
    hist_mc = filter_by_days(hist_full, ANALYSIS_PERIODS["monte_carlo"])

    if len(hist_full) < 200:
        st.warning(f"⚠️ Kun {len(hist_full)} dage data - SMA200 unøjagtig")

    df_full = get_indicators(hist_full)
    df_chart_filtered = df_full.loc[df_full.index.isin(hist_chart.index)]
    df_technical = df_full.tail(len(hist_technical))
    df_targets = df_full.tail(len(hist_targets))

    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)

    navn = info.get("longName") or ticker
    pris = info.get("currentPrice") or hist_full["Close"].iloc[-1]
    valuta = info.get("currency", "USD")
    prev = info.get("previousClose",
                    hist_full["Close"].iloc[-2] if len(hist_full) > 1 else pris)
    change_pct = (pris / prev - 1) * 100 if prev else 0

    first_date = hist_full.index[0].strftime("%Y-%m-%d")
    last_date = hist_full.index[-1].strftime("%Y-%m-%d")

    st.markdown(f"## {navn} ({ticker})")
    st.caption(
        f"🏢 {info.get('sector','?')} · 🌍 {info.get('country','?')} · "
        f"💱 {valuta} · 📅 {first_date} → {last_date} ({len(hist_full)} dage)"
    )

    k = st.columns(6)
    k[0].metric("Pris", f"{pris:,.2f} {valuta}", f"{change_pct:+.2f}%")
    if show_secondary and valuta != "DKK":
        k[0].caption(f"≈ {pris*get_fx_rate(valuta,'DKK'):,.2f} DKK")
    mc = info.get("marketCap")
    try:
        k[1].metric("Market cap", f"{float(mc)/1e9:,.1f}B {valuta}" if mc else "-")
    except Exception:
        k[1].metric("Market cap", "-")
    k[2].metric("P/E", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "-")
    k[3].metric("Fwd P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "-")
    k[4].metric("Yield",
                f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-")
    k[5].metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

    f_score, f_det = fundamental_score(info)
    t_score, t_det = technical_score(df_technical)
    overall = f_score * 0.6 + t_score * 0.4
    f_a, f_c = recommendation(f_score)
    t_a, t_c = recommendation(t_score)
    o_a, o_c = recommendation(overall)

    fair_default = dcf_valuation(info, 0.10, 0.10, 0.025)
    targets = calculate_price_targets(df_targets, pris, fair_default)

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.markdown(make_recommendation_card(
        "🏛️ LANGSIGTET", "📅 12+ måneder · Fundamentale + DCF",
        f_a, f_c, f_score), unsafe_allow_html=True)
    r2.markdown(make_recommendation_card(
        "⚡ KORTSIGTET", "📅 1-3 måneder · Tekniske signaler",
        t_a, t_c, t_score), unsafe_allow_html=True)
    r3.markdown(make_recommendation_card(
        "🎯 SAMLET", "⚖️ Vægtet 60% lang / 40% kort",
        o_a, o_c, overall), unsafe_allow_html=True)

    st.markdown("### 💰 Anbefalede kursniveauer")
    st.caption(f"Baseret på 6-måneders volatilitet · {valuta}"
               f"{' + DKK' if show_secondary and valuta != 'DKK' else ''}")
    pt = st.columns(5)
    buy_low_pct = (targets["buy_low"] / pris - 1) * 100
    buy_high_pct = (targets["buy_high"] / pris - 1) * 100
    stop_pct = (targets["stop_loss"] / pris - 1) * 100
    target_short_pct = (targets["target_short"] / pris - 1) * 100
    target_long_pct = (targets["target_long"] / pris - 1) * 100
    pt[0].markdown(make_range_box("🟢 KØB ZONE", targets["buy_low"], targets["buy_high"],
                                  valuta, "#16a34a",
                                  f"{buy_low_pct:+.1f}% til {buy_high_pct:+.1f}%",
                                  show_secondary), unsafe_allow_html=True)
    pt[1].markdown(make_price_box("📍 AKTUEL", pris, valuta, "#0099ff",
                                  f"{change_pct:+.2f}% i dag",
                                  show_secondary), unsafe_allow_html=True)
    pt[2].markdown(make_price_box("🛑 STOP LOSS", targets["stop_loss"], valuta,
                                  "#ef4444", f"{stop_pct:+.1f}% (2x ATR)",
                                  show_secondary), unsafe_allow_html=True)
    pt[3].markdown(make_price_box("🎯 KORT MÅL (1-3m)", targets["target_short"], valuta,
                                  "#eab308", f"{target_short_pct:+.1f}% (BB upper)",
                                  show_secondary), unsafe_allow_html=True)
    pt[4].markdown(make_price_box("🚀 LANG MÅL (12m+)", targets["target_long"], valuta,
                                  "#22c55e",
                                  f"{target_long_pct:+.1f}% {'(DCF)' if fair_default else '(+20%)'}",
                                  show_secondary), unsafe_allow_html=True)

    if show_secondary and valuta != "DKK":
        rate = get_fx_rate(valuta, "DKK")
        st.caption(
            f"📊 52-uger: Low {targets['week52_low']:.2f} {valuta} "
            f"(≈{targets['week52_low']*rate:.2f} DKK) · "
            f"High {targets['week52_high']:.2f} {valuta} "
            f"(≈{targets['week52_high']*rate:.2f} DKK) · ATR: {targets['atr']:.2f}"
        )
    else:
        st.caption(
            f"📊 52-uger: Low {targets['week52_low']:.2f} · "
            f"High {targets['week52_high']:.2f} {valuta} · "
            f"Daglig ATR: {targets['atr']:.2f}"
        )

    sub_tabs = st.tabs([
        "📊 Charts", "📋 Fundamentals", "🔧 Teknisk",
        "💎 DCF", "📉 Risiko", "🎲 Monte Carlo", "🎯 Backtest"
    ])

    # ----- Charts -----
    with sub_tabs[0]:
        df_plot = df_chart_filtered
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot["Open"],
                                     high=df_plot["High"], low=df_plot["Low"],
                                     close=df_plot["Close"], name="Pris"), 1, 1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA50"],
                                 name="SMA50", line=dict(color="orange")), 1, 1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA200"],
                                 name="SMA200", line=dict(color="purple")), 1, 1)
        fig.add_hline(y=targets["buy_high"], line_dash="dot", line_color="#16a34a",
                      annotation_text="Køb", row=1, col=1)
        fig.add_hline(y=targets["stop_loss"], line_dash="dot", line_color="#ef4444",
                      annotation_text="Stop", row=1, col=1)
        fig.add_hline(y=targets["target_long"], line_dash="dot", line_color="#22c55e",
                      annotation_text="Mål", row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"],
                                 name="RSI", line=dict(color="#00d4aa")), 2, 1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD"],
                                 name="MACD", line=dict(color="#0099ff")), 3, 1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MACD_signal"],
                                 name="Signal", line=dict(color="orange")), 3, 1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          template="plotly_dark", title=f"{navn} - Visning: {period}")
        st.plotly_chart(fig, use_container_width=True)

    # ----- Fundamentals -----
    with sub_tabs[1]:
        st.caption("🏛️ Fundamentale data (TTM)")
        df_f = pd.DataFrame(f_det)
        if not df_f.empty:
            fig_f = px.bar(df_f, x="impact", y="label", orientation="h",
                           color="impact", color_continuous_scale="RdYlGn")
            fig_f.update_layout(height=500, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_f, use_container_width=True)
            st.dataframe(df_f, use_container_width=True, hide_index=True)
        else:
            st.info(f"Ingen fundamentale data fra **{data['source']}**")

    # ----- Teknisk -----
    with sub_tabs[2]:
        st.caption("⚡ Tekniske signaler (12 måneder)")
        df_t = pd.DataFrame(t_det)
        if not df_t.empty:
            fig_t = px.bar(df_t, x="impact", y="label", orientation="h",
                           color="impact", color_continuous_scale="RdYlGn")
            fig_t.update_layout(height=400, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)
        last = df_technical.iloc[-1]
        cc = st.columns(4)
        cc[0].metric("RSI", f"{last['RSI']:.1f}" if not np.isnan(last["RSI"]) else "-")
        cc[1].metric("MACD", f"{last['MACD']:.3f}" if not np.isnan(last["MACD"]) else "-")
        cc[2].metric("ADX", f"{last['ADX']:.1f}" if not np.isnan(last["ADX"]) else "-")
        cc[3].metric("ATR", f"{last['ATR']:.2f}" if not np.isnan(last["ATR"]) else "-")

    # ----- DCF -----
    with sub_tabs[3]:
        c = st.columns(3)
        cg = c[0].slider("Vækstrate", 0.0, 0.30, 0.10, 0.01)
        cdr = c[1].slider("Diskontering", 0.05, 0.20, 0.10, 0.01)
        ct = c[2].slider("Terminal vækst", 0.01, 0.05, 0.025, 0.005)
        fair = dcf_valuation(info, cg, cdr, ct)
        if fair:
            up = (fair / pris - 1) * 100
            d = st.columns(3)
            d[0].metric(f"Aktuel pris ({valuta})", f"{pris:.2f}")
            d[1].metric(f"DCF fair value ({valuta})", f"{fair:.2f}")
            d[2].metric("Upside", f"{up:+.1f}%")
            if show_secondary and valuta != "DKK":
                rate = get_fx_rate(valuta, "DKK")
                st.caption(f"💱 I DKK: Pris {pris*rate:.2f} → Fair value {fair*rate:.2f}")
        else:
            st.warning("Ikke nok FCF-data til DCF")

    # ----- Risiko -----
    with sub_tabs[4]:
        st.caption("📉 Risk metrics (3 år)")
        risk = risk_metrics(hist_risk)
        c = st.columns(4)
        c[0].metric("Ann. afkast", f"{risk['ann_r']*100:.2f}%")
        c[1].metric("Ann. volatilitet", f"{risk['ann_v']*100:.2f}%")
        c[2].metric("Sharpe", f"{risk['sharpe']:.2f}")
        c[3].metric("Sortino", f"{risk['sortino']:.2f}")
        c2 = st.columns(2)
        c2[0].metric("Max drawdown", f"{risk['max_dd']*100:.2f}%")
        c2[1].metric("VaR 95%", f"{risk['var95']*100:.2f}%")
        fig_dd = go.Figure(go.Scatter(x=risk["dd_series"].index,
                                      y=risk["dd_series"] * 100,
                                      fill="tozeroy", line=dict(color="#ef4444")))
        fig_dd.update_layout(template="plotly_dark", height=350,
                             title="Drawdown % (3 år)")
        st.plotly_chart(fig_dd, use_container_width=True)

    # ----- Monte Carlo -----
    with sub_tabs[5]:
        st.caption("🎲 Monte Carlo: 300 simulationer (2 års volatilitet)")
        sims, lp = monte_carlo(hist_mc)
        final = sims[:, -1]
        p5, p50, p95 = np.percentile(final, [5, 50, 95])
        c = st.columns(4)
        c[0].metric(f"Start ({valuta})", f"{lp:.2f}")
        c[1].metric("5% (worst)", f"{p5:.2f}", f"{(p5/lp-1)*100:+.1f}%")
        c[2].metric("Median (12m)", f"{p50:.2f}", f"{(p50/lp-1)*100:+.1f}%")
        c[3].metric("95% (best)", f"{p95:.2f}", f"{(p95/lp-1)*100:+.1f}%")
        fig_m = go.Figure()
        for i in range(min(100, len(sims))):
            fig_m.add_trace(go.Scatter(y=sims[i],
                                        line=dict(width=0.5, color="rgba(0,212,170,0.15)"),
                                        showlegend=False))
        fig_m.add_trace(go.Scatter(y=np.percentile(sims, 50, axis=0),
                                    name="Median",
                                    line=dict(color="#00d4aa", width=3)))
        fig_m.update_layout(template="plotly_dark", height=500,
                            title="Monte Carlo - 252 dage frem")
        st.plotly_chart(fig_m, use_container_width=True)

    # ----- Backtest -----
    with sub_tabs[6]:
        st.markdown("## 🎯 Backtest - Validerer modellens anbefalinger historisk")
        st.caption(
            "Walk-forward analyse: For hvert tidspunkt i fortiden beregnes scoren "
            "BASERET PÅ DATA OP TIL DET PUNKT, og vi sammenligner med hvad der "
            "faktisk skete bagefter."
        )

        bt_col1, bt_col2, bt_col3 = st.columns(3)
        holding_days = bt_col1.selectbox(
            "⏱️ Holding periode", [30, 60, 90, 180, 252], index=2
        )
        sample_freq = bt_col2.selectbox(
            "📊 Sample frekvens", [1, 5, 10, 20], index=1
        )
        buy_threshold = bt_col3.slider("🟢 KØB tærskel (score)", 50, 80, 60)

        if st.button("🚀 Kør backtest", type="primary"):
            with st.spinner("Kører walk-forward backtest..."):
                bt = run_backtest(hist_full, holding_days=holding_days,
                                  sample_freq=sample_freq)
                sim = simulate_strategy(hist_full, buy_threshold=buy_threshold,
                                        sell_threshold=30, sample_freq=sample_freq)

            if bt is None:
                st.error(f"❌ Ikke nok historisk data ({len(hist_full)} dage). "
                         f"Backtest kræver mindst {250 + holding_days} dage.")
            else:
                st.markdown("### 📊 Hit-rate per anbefaling")
                st.caption(
                    f"Baseret på {bt['n_trades']} samples fra "
                    f"{bt['start_date'].strftime('%Y-%m-%d')} til "
                    f"{bt['end_date'].strftime('%Y-%m-%d')} · "
                    f"Holding: {bt['holding_days']} dage"
                )

                rows = []
                for rec_label in ["STÆRKT KØB", "KØB", "HOLD", "SÆLG", "STÆRKT SÆLG"]:
                    s = bt["stats"].get(rec_label)
                    if s:
                        rows.append({
                            "Anbefaling": rec_label, "Antal signaler": s["count"],
                            "Hit rate": f"{s['win_rate']:.1f}%",
                            "Gns. afkast": f"{s['avg_return']:+.2f}%",
                            "Median": f"{s['median_return']:+.2f}%",
                            "Bedst": f"{s['best']:+.2f}%",
                            "Værst": f"{s['worst']:+.2f}%",
                        })
                    else:
                        rows.append({
                            "Anbefaling": rec_label, "Antal signaler": 0,
                            "Hit rate": "-", "Gns. afkast": "-",
                            "Median": "-", "Bedst": "-", "Værst": "-",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown(f"📈 **Buy & Hold over samme periode:** {bt['buy_hold_return']:+.2f}%")

                valid_stats = {k: v for k, v in bt["stats"].items() if v is not None}
                if valid_stats:
                    fig_hr = go.Figure()
                    fig_hr.add_trace(go.Bar(
                        x=list(valid_stats.keys()),
                        y=[v["avg_return"] for v in valid_stats.values()],
                        text=[f"n={v['count']}<br>Hit={v['win_rate']:.0f}%"
                              for v in valid_stats.values()],
                        textposition="auto",
                        marker_color=["#16a34a" if v["avg_return"] > 0 else "#ef4444"
                                      for v in valid_stats.values()],
                    ))
                    fig_hr.add_hline(y=bt["buy_hold_return"], line_dash="dash",
                                     line_color="#0099ff",
                                     annotation_text=f"Buy & Hold: {bt['buy_hold_return']:+.1f}%")
                    fig_hr.update_layout(
                        title=f"Gennemsnit afkast {bt['holding_days']} dage frem",
                        yaxis_title="Afkast %", template="plotly_dark", height=400
                    )
                    st.plotly_chart(fig_hr, use_container_width=True)

                st.markdown("---")
                st.markdown("### 💰 Strategi simulation (10.000 startkapital)")
                if sim and len(sim["equity_curve"]) > 0:
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("📊 Strategi afkast", f"{sim['strategy_return']:+.2f}%",
                               f"{sim['outperformance']:+.2f}% vs B&H")
                    sc2.metric("📈 Buy & Hold", f"{sim['buy_hold_return']:+.2f}%")
                    sc3.metric("💼 Slut værdi", f"{sim['final_value']:,.0f}")
                    sc4.metric("🔁 Antal trades", sim["n_trades"])

                    eq = sim["equity_curve"]
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=eq["date"], y=eq["strategy"],
                                                 name="Strategi",
                                                 line=dict(color="#00d4aa", width=2)))
                    fig_eq.add_trace(go.Scatter(x=eq["date"], y=eq["buy_hold"],
                                                 name="Buy & Hold",
                                                 line=dict(color="#0099ff", width=2, dash="dash")))
                    if len(sim["trades"]) > 0:
                        for _, trade in sim["trades"][sim["trades"]["action"] == "KØB"].iterrows():
                            fig_eq.add_vline(x=trade["date"], line_color="#16a34a",
                                             line_width=1, opacity=0.3)
                        for _, trade in sim["trades"][sim["trades"]["action"] == "SÆLG"].iterrows():
                            fig_eq.add_vline(x=trade["date"], line_color="#ef4444",
                                             line_width=1, opacity=0.3)
                    fig_eq.update_layout(
                        title=f"Strategi vs Buy & Hold (Køb≥{buy_threshold}, Sælg≤30)",
                        yaxis_title=f"Værdi ({valuta})",
                        template="plotly_dark", height=500
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

                    if len(sim["trades"]) > 0:
                        with st.expander(f"📋 Se alle {sim['n_trades']} trades"):
                            tl = sim["trades"].copy()
                            tl["date"] = tl["date"].dt.strftime("%Y-%m-%d")
                            tl["price"] = tl["price"].round(2)
                            st.dataframe(tl, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("### 🔍 Score vs faktisk afkast")
                fig_sc = px.scatter(
                    bt["results"], x="score", y="return_pct", color="recommendation",
                    color_discrete_map={
                        "STÆRKT KØB": "#16a34a", "KØB": "#22c55e",
                        "HOLD": "#eab308", "SÆLG": "#ef4444", "STÆRKT SÆLG": "#b91c1c"
                    },
                    hover_data=["date", "entry_price", "exit_price"],
                    title=f"Score vs {bt['holding_days']}-dages afkast",
                    labels={"score": "Score (0-100)", "return_pct": "Afkast %"}
                )
                fig_sc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
                fig_sc.add_vline(x=50, line_dash="dash", line_color="white", opacity=0.3)
                fig_sc.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_sc, use_container_width=True)

                corr = bt["results"]["score"].corr(bt["results"]["return_pct"])
                if corr > 0.3:
                    st.success(f"✅ **Stærk positiv korrelation: {corr:.3f}**")
                elif corr > 0.1:
                    st.info(f"➖ **Svag positiv korrelation: {corr:.3f}**")
                elif corr > -0.1:
                    st.warning(f"⚠️ **Ingen korrelation: {corr:.3f}**")
                else:
                    st.error(f"❌ **Negativ korrelation: {corr:.3f}**")

                st.warning("⚠️ **DISCLAIMER**: Historisk performance garanterer ikke fremtidige resultater.")
        else:
            st.info("👆 Tryk **🚀 Kør backtest** for at se hvor godt anbefalingerne har virket historisk")
