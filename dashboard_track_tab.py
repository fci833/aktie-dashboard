"""
Track Record Dashboard Tab
==========================
Streamlit-tab der visualiserer:
  - Overall performance metrics (hit rate, Brier, AUC, equity)
  - Calibration reliability diagram
  - Equity curve (paper trading)
  - Rolling hit rate over tid
  - Confusion matrix heatmap
  - Per-model / per-ticker / per-horizon breakdown
  - Walk-forward fold stability
  - Pending predictions queue
  - Quick actions: resolve, export, retrain trigger

Integration i hoved-dashboard:
    from dashboard_track_tab import render_track_record_tab
    
    if active_tab == "Track Record":
        render_track_record_tab(db_path="predictions.db")
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from prediction_logger import PredictionLogger
from track_record import TrackRecord


# ============================================================
# Theming (matches dark dashboard)
# ============================================================

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#E5E7EB"),
    margin=dict(l=10, r=10, t=40, b=10),
)

GREEN = "#10B981"
RED = "#EF4444"
YELLOW = "#F59E0B"
BLUE = "#3B82F6"
PURPLE = "#A855F7"
GRAY = "#6B7280"


# ============================================================
# Helpers
# ============================================================

def _fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_num(v: Optional[float], decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _color_for_hit_rate(hr: float) -> str:
    if hr >= 0.60:
        return GREEN
    if hr >= 0.50:
        return YELLOW
    return RED


# ============================================================
# Header / KPI cards
# ============================================================

def _render_kpi_cards(metrics: dict):
    cols = st.columns(5)

    with cols[0]:
        hr = metrics.get("hit_rate", 0)
        st.metric(
            "🎯 Hit Rate",
            _fmt_pct(hr),
            delta=f"{(hr - 0.5)*100:+.1f}pp vs random",
            delta_color="normal" if hr >= 0.5 else "inverse",
        )

    with cols[1]:
        brier = metrics.get("brier_score")
        st.metric(
            "📉 Brier Score",
            _fmt_num(brier, 4),
            delta="lower is better",
            delta_color="off",
            help="0 = perfect calibration, 0.25 = random",
        )

    with cols[2]:
        auc = metrics.get("roc_auc")
        st.metric(
            "📊 ROC-AUC",
            _fmt_num(auc, 3) if auc is not None else "—",
            delta=f"{(auc - 0.5):+.3f}" if auc is not None else "—",
            delta_color="normal" if (auc or 0) >= 0.5 else "inverse",
        )

    with cols[3]:
        n = metrics.get("n", metrics.get("n_total", 0))
        st.metric("🔢 Resolved", f"{n:,}")

    with cols[4]:
        avg_ret = metrics.get("avg_actual_return", 0)
        st.metric(
            "💰 Avg Return",
            _fmt_pct(avg_ret, 2),
            delta_color="normal" if avg_ret > 0 else "inverse",
        )


# ============================================================
# Filters
# ============================================================

def _render_filters(logger: PredictionLogger) -> dict:
    df_all = logger.get_track_record(only_resolved=True)

    with st.expander("🔍 Filtre", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            models = ["(alle)"] + (sorted(df_all["model_name"].unique().tolist())
                                    if not df_all.empty else [])
            model = st.selectbox("Model", models, key="tr_model")

        with c2:
            tickers = ["(alle)"] + (sorted(df_all["ticker"].unique().tolist())
                                     if not df_all.empty else [])
            ticker = st.selectbox("Ticker", tickers, key="tr_ticker")

        with c3:
            horizons = ["(alle)"] + (sorted(df_all["horizon_days"].unique().tolist())
                                      if not df_all.empty else [])
            horizon = st.selectbox("Horizon (dage)", horizons, key="tr_horizon")

        with c4:
            period = st.selectbox(
                "Periode",
                ["Alle", "Sidste 30 dage", "Sidste 90 dage", "Sidste 180 dage", "Sidste år"],
                key="tr_period",
            )

    period_map = {
        "Sidste 30 dage": 30, "Sidste 90 dage": 90,
        "Sidste 180 dage": 180, "Sidste år": 365,
    }
    since = None
    if period in period_map:
        since = (date.today() - timedelta(days=period_map[period])).isoformat()

    return {
        "model_name": None if model == "(alle)" else model,
        "ticker": None if ticker == "(alle)" else ticker,
        "horizon_days": None if horizon == "(alle)" else int(horizon),
        "since": since,
    }


# ============================================================
# Charts
# ============================================================

def _plot_reliability_diagram(rel_df: pd.DataFrame) -> go.Figure:
    """Calibration reliability curve."""
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color=GRAY, width=2),
        name="Perfect calibration",
        hoverinfo="skip",
    ))

    valid = rel_df.dropna(subset=["mean_predicted", "mean_actual"])
    if not valid.empty:
        fig.add_trace(go.Scatter(
            x=valid["mean_predicted"],
            y=valid["mean_actual"],
            mode="markers+lines",
            marker=dict(
                size=valid["count"] / max(valid["count"].max(), 1) * 30 + 6,
                color=GREEN,
                line=dict(color="white", width=1),
            ),
            line=dict(color=GREEN, width=2),
            name="Model",
            text=[f"n={int(c)}" for c in valid["count"]],
            hovertemplate="<b>Predicted: %{x:.1%}</b><br>Actual: %{y:.1%}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="🎯 Reliability Diagram (Calibration)",
        xaxis=dict(title="Mean predicted probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Actual hit rate", range=[0, 1], tickformat=".0%"),
        height=380,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def _plot_equity_curve(equity_result) -> go.Figure:
    """Paper trading equity curve."""
    if not equity_result.equity or equity_result.n_trades == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Ingen trades endnu",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color=GRAY),
        )
        fig.update_layout(**DARK_LAYOUT, height=380)
        return fig

    dates = pd.to_datetime(equity_result.dates)
    equity = equity_result.equity

    fig = go.Figure()

    # Equity line
    color = GREEN if equity[-1] >= equity[0] else RED
    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({16 if color==GREEN else 239},{185 if color==GREEN else 68},{129 if color==GREEN else 68},0.1)",
        name="Equity",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
    ))

    # Initial capital reference
    fig.add_hline(
        y=equity[0],
        line=dict(dash="dash", color=GRAY, width=1),
        annotation_text=f"Start: ${equity[0]:,.0f}",
        annotation_position="right",
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"📈 Equity Curve  •  Sharpe: {equity_result.sharpe:.2f}  •  "
              f"MaxDD: {equity_result.max_drawdown:.1%}  •  "
              f"Trades: {equity_result.n_trades}",
        xaxis=dict(title=""),
        yaxis=dict(title="Equity (USD)", tickformat="$,.0f"),
        height=380,
    )
    return fig


def _plot_rolling_hit_rate(rolling_df: pd.DataFrame) -> go.Figure:
    """Rolling hit rate over time → drift detection."""
    fig = go.Figure()

    if rolling_df.empty:
        fig.add_annotation(
            text="Ikke nok data til rolling metrics",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=GRAY),
        )
        fig.update_layout(**DARK_LAYOUT, height=320)
        return fig

    df = rolling_df.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df = df.sort_values("prediction_date")

    fig.add_trace(go.Scatter(
        x=df["prediction_date"], y=df["rolling_hit_rate"],
        mode="lines",
        line=dict(color=BLUE, width=2.5),
        name="Rolling hit rate",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1%}<extra></extra>",
    ))

    fig.add_hline(
        y=0.5,
        line=dict(dash="dash", color=GRAY, width=1),
        annotation_text="Random (50%)",
        annotation_position="right",
    )

    fig.add_hrect(y0=0.55, y1=1.0, fillcolor=GREEN, opacity=0.06, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.45, fillcolor=RED, opacity=0.06, line_width=0)

    fig.update_layout(
        **DARK_LAYOUT,
        title="🔄 Rolling Hit Rate (drift detection)",
        xaxis=dict(title=""),
        yaxis=dict(title="Hit rate", tickformat=".0%", range=[0, 1]),
        height=320,
    )
    return fig


def _plot_confusion_matrix(cm: pd.DataFrame) -> go.Figure:
    """Confusion matrix heatmap."""
    if cm.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=320,
                          title="🔄 Confusion Matrix")
        return fig

    z = cm.values
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, text=text,
        x=cm.columns.tolist(), y=cm.index.tolist(),
        colorscale=[[0, "#1F2937"], [0.5, BLUE], [1, GREEN]],
        showscale=True,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
        hovertemplate="Pred: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title="🔄 Confusion Matrix (BUY / HOLD / SELL)",
        height=320,
    )
    return fig


def _plot_breakdown_bar(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """Bar chart med hit rate per gruppe + count som secondary."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=320, title=title)
        return fig

    df = df.copy()
    df["color"] = df["hit_rate"].apply(_color_for_hit_rate)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df[group_col].astype(str),
        y=df["hit_rate"],
        marker=dict(color=df["color"]),
        name="Hit rate",
        text=[_fmt_pct(v) for v in df["hit_rate"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Hit rate: %{y:.1%}<br>n=%{customdata}<extra></extra>",
        customdata=df["n"],
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df[group_col].astype(str),
        y=df["n"],
        mode="markers+lines",
        line=dict(color=PURPLE, width=1.5, dash="dot"),
        marker=dict(color=PURPLE, size=8),
        name="N predictions",
        hovertemplate="<b>%{x}</b><br>N: %{y}<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=0.5, line=dict(dash="dash", color=GRAY, width=1),
                  secondary_y=False)

    fig.update_layout(
        **DARK_LAYOUT,
        title=title,
        height=320,
        showlegend=False,
        bargap=0.3,
    )
    fig.update_yaxes(title_text="Hit rate", tickformat=".0%",
                     range=[0, 1.05], secondary_y=False)
    fig.update_yaxes(title_text="N", secondary_y=True, showgrid=False)
    return fig


def _plot_walk_forward_stability(folds_df: pd.DataFrame) -> go.Figure:
    """Per-fold hit rate + Brier over time."""
    if folds_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, height=320,
                          title="🚶 Walk-forward Stability")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    folds_df = folds_df.sort_values("fold_id")

    fig.add_trace(go.Scatter(
        x=folds_df["fold_id"], y=folds_df["hit_rate"],
        mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(size=8),
        name="Hit rate",
        hovertemplate="Fold %{x}<br>Hit: %{y:.1%}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=folds_df["fold_id"], y=folds_df["brier_score"],
        mode="lines+markers",
        line=dict(color=RED, width=2, dash="dot"),
        marker=dict(size=6),
        name="Brier",
        hovertemplate="Fold %{x}<br>Brier: %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=0.5, line=dict(dash="dash", color=GRAY),
                  secondary_y=False)

    fig.update_layout(
        **DARK_LAYOUT,
        title="🚶 Walk-Forward Fold Stability",
        height=320,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
    )
    fig.update_xaxes(title="Fold")
    fig.update_yaxes(title="Hit rate", tickformat=".0%",
                     range=[0, 1], secondary_y=False)
    fig.update_yaxes(title="Brier", secondary_y=True, showgrid=False)
    return fig


# ============================================================
# Tables
# ============================================================

def _render_recent_predictions(df: pd.DataFrame, max_rows: int = 20):
    if df.empty:
        st.info("Ingen resolved predictions endnu.")
        return

    show = df.head(max_rows)[[
        "prediction_date", "ticker", "model_name", "horizon_days",
        "predicted_direction", "predicted_prob",
        "actual_direction", "actual_return", "hit",
    ]].copy()
    show["predicted_prob"] = show["predicted_prob"].apply(lambda v: f"{v*100:.1f}%")
    show["actual_return"] = show["actual_return"].apply(
        lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
    )
    show["hit"] = show["hit"].apply(lambda v: "✅" if v == 1 else "❌")
    show.columns = ["Dato", "Ticker", "Model", "Horizon",
                    "Pred", "Prob", "Actual", "Return", "Hit"]

    st.dataframe(show, use_container_width=True, hide_index=True)


def _render_pending_queue(logger: PredictionLogger):
    pending = logger.get_track_record(status="pending")
    if pending.empty:
        st.success("✅ Ingen pending predictions — alt er resolved!")
        return

    today = date.today().isoformat()
    pending["overdue"] = pending["target_date"] <= today

    overdue = pending[pending["overdue"]]
    upcoming = pending[~pending["overdue"]]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("⏰ Overdue (klar til resolve)", len(overdue))
    with c2:
        st.metric("⏳ Upcoming", len(upcoming))

    if not pending.empty:
        with st.expander(f"📋 Vis pending queue ({len(pending)} total)"):
            show = pending[["ticker", "model_name", "prediction_date",
                            "target_date", "predicted_direction",
                            "predicted_prob", "entry_price"]].copy()
            show["predicted_prob"] = show["predicted_prob"].apply(
                lambda v: f"{v*100:.1f}%"
            )
            st.dataframe(show, use_container_width=True, hide_index=True)


# ============================================================
# Main render function
# ============================================================

def render_track_record_tab(
    db_path: str | Path = "predictions.db",
    initial_capital: float = 10_000.0,
    rolling_window: int = 20,
):
    """
    Hovedfunktion til at rendere Track Record fanen.
    Kald fra dit main dashboard.
    """
    st.markdown("## 📊 Track Record & ML Performance")
    st.caption("Out-of-sample performance fra logged predictions. "
               "Alle metrics er post-resolution (faktisk markedsdata).")

    logger = PredictionLogger(db_path)
    overview = logger.stats_overview()

    # ---- Empty state ----
    if overview["total"] == 0:
        st.info("📭 Ingen predictions logget endnu.\n\n"
                "Kør en analyse i et af de andre tabs for at logge din første forudsigelse.")
        with st.expander("ℹ️ Hvordan virker track record?"):
            st.markdown("""
            **Workflow:**
            1. **Log** — Hver gang du laver en analyse, logges den til DB
            2. **Resolve** — Når horisonten udløber, hentes faktisk pris automatisk
            3. **Track** — Her ser du om dine modeller faktisk virker over tid
            
            **Vigtige metrics:**
            - **Hit rate** — % korrekte forudsigelser (>55% er godt)
            - **Brier Score** — Calibration kvalitet (<0.20 er godt)
            - **ROC-AUC** — Modellens evne til at skelne (>0.6 er godt)
            - **Sharpe** — Risk-adjusted return (>1.0 er godt)
            """)
        return

    # ---- Top action bar ----
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        st.metric("📦 Total", overview["total"])
    with c2:
        st.metric("✅ Resolved", overview["resolved"],
                  delta=f"{overview['resolved']/max(overview['total'],1)*100:.0f}%")
    with c3:
        st.metric("⏳ Pending", overview["pending"])
    with c4:
        if st.button("🔄 Resolve pending now", type="primary",
                     use_container_width=True):
            with st.spinner("Henter faktiske priser og opdaterer..."):
                stats = logger.resolve_pending(verbose=False)
            st.success(f"✅ Resolved {stats['resolved']} | "
                       f"⏳ Still pending: {stats['still_pending']} | "
                       f"⚠️ Failed: {stats['failed']}")
            st.rerun()

    st.divider()

    # ---- Filters ----
    filters = _render_filters(logger)
    tr = TrackRecord(logger)
    metrics = tr.core_metrics(**filters)

    if metrics.n == 0:
        st.warning("Ingen resolved predictions matcher disse filtre.")
        _render_pending_queue(logger)
        return

    # ---- KPI cards ----
    st.markdown("### 📈 Overall Performance")
    _render_kpi_cards(metrics.to_dict())

    # ---- Direction breakdown badges ----
    bcols = st.columns(3)
    with bcols[0]:
        wr_buy = metrics.win_rate_buys
        st.markdown(f"""<div style='padding:12px;border-radius:8px;
        background:rgba(16,185,129,0.1);border:1px solid {GREEN};'>
        <div style='color:{GREEN};font-weight:600;'>🟢 BUY signals</div>
        <div style='font-size:24px;font-weight:700;'>{_fmt_pct(wr_buy)}</div>
        <div style='color:{GRAY};font-size:12px;'>{metrics.n_buys} forudsigelser</div>
        </div>""", unsafe_allow_html=True)
    with bcols[1]:
        st.markdown(f"""<div style='padding:12px;border-radius:8px;
        background:rgba(245,158,11,0.1);border:1px solid {YELLOW};'>
        <div style='color:{YELLOW};font-weight:600;'>🟡 HOLD signals</div>
        <div style='font-size:24px;font-weight:700;'>{metrics.n_holds}</div>
        <div style='color:{GRAY};font-size:12px;'>neutrale</div>
        </div>""", unsafe_allow_html=True)
    with bcols[2]:
        wr_sell = metrics.win_rate_sells
        st.markdown(f"""<div style='padding:12px;border-radius:8px;
        background:rgba(239,68,68,0.1);border:1px solid {RED};'>
        <div style='color:{RED};font-weight:600;'>🔴 SELL signals</div>
        <div style='font-size:24px;font-weight:700;'>{_fmt_pct(wr_sell)}</div>
        <div style='color:{GRAY};font-size:12px;'>{metrics.n_sells} forudsigelser</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- Equity + Reliability ----
    st.markdown("### 💰 Paper Trading & Calibration")
    c1, c2 = st.columns(2)
    with c1:
        equity = tr.equity_curve(
            model_name=filters["model_name"],
            horizon_days=filters["horizon_days"],
            initial_capital=initial_capital,
        )
        st.plotly_chart(_plot_equity_curve(equity), use_container_width=True)
    with c2:
        rel_df = tr.reliability_data(
            n_bins=10,
            model_name=filters["model_name"],
            horizon_days=filters["horizon_days"],
        )
        st.plotly_chart(_plot_reliability_diagram(rel_df), use_container_width=True)

    # ---- Rolling hit rate ----
    st.markdown("### 🔄 Rolling Performance & Drift")
    rolling = tr.rolling_hit_rate(
        window=rolling_window,
        model_name=filters["model_name"],
        horizon_days=filters["horizon_days"],
    )
    st.plotly_chart(_plot_rolling_hit_rate(rolling), use_container_width=True)

    # ---- Breakdown by model / ticker / horizon ----
    st.markdown("### 🔬 Breakdowns")
    tab_model, tab_ticker, tab_horizon, tab_direction = st.tabs(
        ["By Model", "By Ticker", "By Horizon", "Confusion Matrix"]
    )

    with tab_model:
        df_model = tr.by_model(horizon_days=filters["horizon_days"])
        if not df_model.empty:
            st.plotly_chart(
                _plot_breakdown_bar(df_model, "model_name", "Hit rate per model"),
                use_container_width=True,
            )
            st.dataframe(df_model, use_container_width=True, hide_index=True)
        else:
            st.info("Ingen data.")

    with tab_ticker:
        df_ticker = tr.by_ticker(model_name=filters["model_name"])
        if not df_ticker.empty:
            st.plotly_chart(
                _plot_breakdown_bar(df_ticker.head(15), "ticker",
                                    "Hit rate per ticker (top 15 by N)"),
                use_container_width=True,
            )
            st.dataframe(df_ticker, use_container_width=True, hide_index=True)
        else:
            st.info("Ingen data.")

    with tab_horizon:
        df_h = tr.by_horizon(model_name=filters["model_name"])
        if not df_h.empty:
            st.plotly_chart(
                _plot_breakdown_bar(df_h, "horizon_days", "Hit rate per horisont"),
                use_container_width=True,
            )
            st.dataframe(df_h, use_container_width=True, hide_index=True)
        else:
            st.info("Ingen data.")

    with tab_direction:
        cm = tr.confusion(
            model_name=filters["model_name"],
            horizon_days=filters["horizon_days"],
        )
        st.plotly_chart(_plot_confusion_matrix(cm), use_container_width=True)
        if not cm.empty:
            st.dataframe(cm, use_container_width=True)

    # ---- Pending queue ----
    st.divider()
    st.markdown("### ⏳ Pending Queue")
    _render_pending_queue(logger)

    # ---- Recent predictions table ----
    st.divider()
    st.markdown("### 📋 Recent Resolved Predictions")
    df_recent = logger.get_track_record(only_resolved=True, **{
        k: v for k, v in filters.items() if k != "since"
    })
    if filters["since"]:
        df_recent = df_recent[df_recent["prediction_date"] >= filters["since"]]
    _render_recent_predictions(df_recent)

    # ---- Export ----
    st.divider()
    with st.expander("⚙️ Avanceret"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📤 Export CSV", use_container_width=True):
                export_path = Path(f"track_record_export_{date.today().isoformat()}.csv")
                logger.export_csv(export_path)
                st.success(f"Eksporteret: {export_path}")
        with c2:
            if st.button("🔄 Refresh data", use_container_width=True):
                st.rerun()
        with c3:
            if st.button("⚠️ Reset DB (slet alt)", type="secondary",
                         use_container_width=True):
                if st.session_state.get("confirm_reset"):
                    logger.reset_db()
                    st.session_state.pop("confirm_reset")
                    st.success("Database reset.")
                    st.rerun()
                else:
                    st.session_state["confirm_reset"] = True
                    st.warning("Klik igen for at bekræfte!")


# ============================================================
# Standalone test runner
# ============================================================

if __name__ == "__main__":
    """
    Kør standalone for at teste tabben:
        streamlit run dashboard_track_tab.py
    """
    st.set_page_config(
        page_title="Track Record Test",
        layout="wide",
        page_icon="📊",
    )

    # Hvis ingen DB, generér testdata
    test_db = Path("_test_dashboard.db")
    if not test_db.exists() or st.sidebar.button("🔄 Regenerate test data"):
        if test_db.exists():
            test_db.unlink()

        from prediction_logger import PredictionRecord, PriceFetcher
        rng = np.random.default_rng(42)

        class MockFetcher(PriceFetcher):
            def get_close_price(self, ticker, asset_type, target_date):
                base = {"ETH": 1800, "BTC": 65000, "SOL": 150,
                        "RKLB": 100, "AAPL": 200, "NVDA": 130,
                        "MSFT": 410, "AVAX": 35}.get(ticker, 100)
                return base * (1 + rng.normal(0.02, 0.10))

        logger = PredictionLogger(test_db)
        logger.price_fetcher = MockFetcher()

        tickers = ["ETH", "BTC", "SOL", "RKLB", "AAPL", "NVDA", "MSFT", "AVAX"]
        crypto = {"ETH", "BTC", "SOL", "AVAX"}
        models = ["rf", "xgb", "lgbm", "ensemble"]

        # 200 historical predictions
        for _ in range(200):
            days_ago = int(rng.integers(35, 365))
            t = str(rng.choice(tickers))
            logger.log(PredictionRecord(
                ticker=t,
                asset_type="crypto" if t in crypto else "stock",
                horizon_days=int(rng.choice([7, 30, 90])),
                entry_price=float({"ETH": 1800, "BTC": 65000, "SOL": 150,
                                    "RKLB": 100, "AAPL": 200, "NVDA": 130,
                                    "MSFT": 410, "AVAX": 35}[t]),
                predicted_prob=float(np.clip(rng.normal(0.55, 0.15), 0.05, 0.95)),
                model_name=str(rng.choice(models)),
                prediction_date=(date.today() - timedelta(days=days_ago)).isoformat(),
                calibrated=True,
            ))

        # 10 pending
        for _ in range(10):
            t = str(rng.choice(tickers))
            logger.log(PredictionRecord(
                ticker=t,
                asset_type="crypto" if t in crypto else "stock",
                horizon_days=30,
                entry_price=100.0,
                predicted_prob=float(np.clip(rng.normal(0.6, 0.1), 0.3, 0.9)),
                model_name="ensemble",
            ))

        logger.resolve_pending(verbose=False)
        st.sidebar.success("Test data genereret!")

    render_track_record_tab(db_path=test_db)
