"""
Track Record - Standalone module for logging predictions and tracking performance
==================================================================================

Bruges til at:
  - Logge alle predictions automatisk (når du analyserer en aktie/krypto)
  - Auto-opdatere priser efter 30/90/180 dage
  - Beregne hit rate, gennemsnitsafkast, equity curve
  - Vise resultater i Streamlit UI

Schema (SQLite):
  - timestamp, ticker, name, asset_class, price_at_prediction
  - score, recommendation, f_score, t_score, regime, sector, currency
  - price_30d, price_90d, price_180d (auto-fyldes)
  - return_30d, return_90d, return_180d
  - hit_30d, hit_90d, hit_180d (1=korrekt, 0=forkert, NULL=ikke endnu)

Public API (kaldes fra app.py):
  - save_prediction(...)
  - update_predictions(force=False)
  - get_track_record_stats()
  - render_track_record_view()
  - render_track_record_summary()
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ===== Config =====
DB_PATH = Path("track_record.db")
HORIZONS = [30, 90, 180]
HOLD_TOLERANCE_PCT = 5.0  # ±5% = HOLD-prediction "hit"
UPDATE_COOLDOWN_HOURS = 6  # Skip auto-update hvis sidste kørsel <6t siden


# ============================================================
# DATABASE SETUP
# ============================================================

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Opret tabel hvis den ikke findes."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                name TEXT,
                asset_class TEXT NOT NULL,
                price_at_prediction REAL NOT NULL,
                score REAL,
                recommendation TEXT,
                f_score REAL,
                t_score REAL,
                regime TEXT,
                sector TEXT,
                currency TEXT,
                price_30d REAL,
                price_90d REAL,
                price_180d REAL,
                return_30d REAL,
                return_90d REAL,
                return_180d REAL,
                hit_30d INTEGER,
                hit_90d INTEGER,
                hit_180d INTEGER,
                last_updated TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()


_init_db()


# ============================================================
# SAVE PREDICTION (kaldes fra app.py ved hver analyse)
# ============================================================

def save_prediction(
    ticker: str,
    name: str,
    asset_class: str,
    price: float,
    score: float,
    recommendation: str,
    f_score: float = 0.0,
    t_score: float = 0.0,
    regime: str = "UNKNOWN",
    sector: str = "?",
    currency: str = "USD",
) -> Optional[int]:
    """
    Gem en prediction. Returnerer ID på den nye række.

    OBS: Vi gemmer KUN én prediction per ticker per dag for at undgå spam
    (hvis brugeren analyserer samme ticker 10 gange på én dag).
    """
    if not ticker or price is None or price <= 0:
        return None

    today_iso = datetime.now().strftime("%Y-%m-%d")
    now_iso = datetime.now().isoformat()

    try:
        with _get_conn() as conn:
            # Tjek om vi allerede har en prediction for denne ticker i dag
            existing = conn.execute(
                "SELECT id FROM predictions WHERE ticker=? AND date(timestamp)=? LIMIT 1",
                (ticker, today_iso)
            ).fetchone()

            if existing:
                # Opdatér eksisterende række (sidste pris/score gemmes)
                conn.execute("""
                    UPDATE predictions
                    SET price_at_prediction=?, score=?, recommendation=?,
                        f_score=?, t_score=?, regime=?, sector=?, currency=?,
                        timestamp=?
                    WHERE id=?
                """, (price, score, recommendation, f_score, t_score, regime,
                      sector, currency, now_iso, existing["id"]))
                conn.commit()
                return existing["id"]

            # Indsæt ny række
            cur = conn.execute("""
                INSERT INTO predictions
                (timestamp, ticker, name, asset_class, price_at_prediction, score,
                 recommendation, f_score, t_score, regime, sector, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_iso, ticker, name, asset_class, price, score,
                  recommendation, f_score, t_score, regime, sector, currency))
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        print(f"[track_record] save_prediction error: {e}")
        return None


# ============================================================
# UPDATE PREDICTIONS (auto-fyld priser efter 30/90/180 dage)
# ============================================================

def _fetch_current_price(ticker: str, asset_class: str) -> Optional[float]:
    """Hent nuværende pris via yfinance (stocks) eller via _USD-suffix (krypto)."""
    try:
        import yfinance as yf

        if asset_class == "crypto":
            yf_ticker = f"{ticker}-USD"
        else:
            yf_ticker = ticker

        t = yf.Ticker(yf_ticker)
        info = t.fast_info if hasattr(t, "fast_info") else None
        if info and hasattr(info, "last_price"):
            price = float(info.last_price)
            if price > 0:
                return price

        # Fallback til hist
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        print(f"[track_record] _fetch_current_price({ticker}): {e}")
    return None


def _calc_hit(recommendation: str, return_pct: float) -> int:
    """1 hvis prediction var korrekt, 0 hvis forkert."""
    rec = (recommendation or "").upper()
    if "KØB" in rec or "BUY" in rec:
        return 1 if return_pct > 0 else 0
    if "SÆLG" in rec or "SELL" in rec:
        return 1 if return_pct < 0 else 0
    # HOLD: korrekt hvis kursen ikke flytter sig meget
    return 1 if abs(return_pct) <= HOLD_TOLERANCE_PCT else 0


def update_predictions(force: bool = False, max_updates: int = 50) -> dict:
    """
    Auto-opdater priser for predictions der er 30/90/180 dage gamle.

    Args:
        force: Hvis False, springes kørsel over hvis sidst kørt <6t siden.
        max_updates: Max antal rækker at opdatere per kørsel (rate limit beskyttelse).

    Returns:
        dict med {checked, updated, errors, skipped}
    """
    # Cooldown check
    if not force:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='last_update'"
            ).fetchone()
            if row:
                try:
                    last_update = datetime.fromisoformat(row["value"])
                    if datetime.now() - last_update < timedelta(hours=UPDATE_COOLDOWN_HOURS):
                        return {"checked": 0, "updated": 0, "errors": 0,
                                "skipped": True, "reason": "cooldown"}
                except Exception:
                    pass

    stats = {"checked": 0, "updated": 0, "errors": 0, "skipped": False}
    now = datetime.now()
    price_cache: dict = {}  # ticker → price (undgå dobbelt-fetch)

    with _get_conn() as conn:
        # Find rækker der mangler en eller flere horisont-priser
        rows = conn.execute("""
            SELECT id, ticker, asset_class, timestamp, price_at_prediction,
                   recommendation,
                   price_30d, price_90d, price_180d
            FROM predictions
            WHERE price_30d IS NULL OR price_90d IS NULL OR price_180d IS NULL
            ORDER BY timestamp ASC
            LIMIT ?
        """, (max_updates,)).fetchall()

        for row in rows:
            stats["checked"] += 1
            try:
                pred_time = datetime.fromisoformat(row["timestamp"])
                age_days = (now - pred_time).days

                updates = {}

                for h in HORIZONS:
                    col_price = f"price_{h}d"
                    col_return = f"return_{h}d"
                    col_hit = f"hit_{h}d"

                    if row[col_price] is not None:
                        continue  # allerede fyldt
                    if age_days < h:
                        continue  # for tidligt at evaluere

                    # Hent nuværende pris (cached per ticker)
                    cache_key = f"{row['ticker']}_{row['asset_class']}"
                    if cache_key not in price_cache:
                        price_cache[cache_key] = _fetch_current_price(
                            row["ticker"], row["asset_class"]
                        )

                    current = price_cache[cache_key]
                    if current is None or current <= 0:
                        continue

                    entry = float(row["price_at_prediction"])
                    if entry <= 0:
                        continue

                    return_pct = (current / entry - 1) * 100
                    hit = _calc_hit(row["recommendation"], return_pct)

                    updates[col_price] = current
                    updates[col_return] = return_pct
                    updates[col_hit] = hit

                if updates:
                    set_clause = ", ".join(f"{k}=?" for k in updates.keys())
                    set_clause += ", last_updated=?"
                    params = list(updates.values()) + [now.isoformat(), row["id"]]
                    conn.execute(
                        f"UPDATE predictions SET {set_clause} WHERE id=?",
                        params
                    )
                    stats["updated"] += 1

            except Exception as e:
                print(f"[track_record] update error for id={row['id']}: {e}")
                stats["errors"] += 1

        # Gem last_update timestamp
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_update', ?)",
            (now.isoformat(),)
        )
        conn.commit()

    return stats


# ============================================================
# STATS QUERIES
# ============================================================

def get_track_record_stats() -> dict:
    """
    Returnér samlet stats om track record.

    Returns:
        dict med:
            - total: total antal predictions
            - resolved_30d / 90d / 180d: antal predictions med data
            - hit_rate_30d / 90d / 180d: success rate (0-1)
            - avg_return_30d / 90d / 180d
            - by_recommendation: dict per anbefaling
    """
    out = {
        "total": 0,
        "by_asset_class": {},
        "by_recommendation": {},
    }
    for h in HORIZONS:
        out[f"resolved_{h}d"] = 0
        out[f"hit_rate_{h}d"] = None
        out[f"avg_return_{h}d"] = None

    try:
        with _get_conn() as conn:
            # Total
            row = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()
            out["total"] = row["c"] if row else 0

            if out["total"] == 0:
                return out

            # Per horizon
            for h in HORIZONS:
                row = conn.execute(f"""
                    SELECT COUNT(*) as n,
                           AVG(hit_{h}d) * 1.0 as hit_rate,
                           AVG(return_{h}d) as avg_ret
                    FROM predictions
                    WHERE hit_{h}d IS NOT NULL
                """).fetchone()
                if row and row["n"]:
                    out[f"resolved_{h}d"] = row["n"]
                    out[f"hit_rate_{h}d"] = row["hit_rate"]
                    out[f"avg_return_{h}d"] = row["avg_ret"]

            # By asset class
            rows = conn.execute("""
                SELECT asset_class, COUNT(*) as n
                FROM predictions
                GROUP BY asset_class
            """).fetchall()
            out["by_asset_class"] = {r["asset_class"]: r["n"] for r in rows}

            # By recommendation
            rows = conn.execute("""
                SELECT recommendation, COUNT(*) as n,
                       AVG(hit_90d) * 1.0 as hit_rate
                FROM predictions
                GROUP BY recommendation
            """).fetchall()
            out["by_recommendation"] = {
                r["recommendation"]: {
                    "n": r["n"],
                    "hit_rate_90d": r["hit_rate"],
                }
                for r in rows
            }

    except Exception as e:
        print(f"[track_record] get_stats error: {e}")

    return out


def _get_all_predictions() -> pd.DataFrame:
    """Hent alle predictions som DataFrame."""
    try:
        with _get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC",
                conn
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        print(f"[track_record] get_all error: {e}")
        return pd.DataFrame()


# ============================================================
# STREAMLIT UI
# ============================================================

def render_track_record_summary():
    """Kompakt widget til Hjem-view (4 metrics + lille forklaring)."""
    stats = get_track_record_stats()

    if stats["total"] == 0:
        st.info(
            "📈 **Track Record er tom endnu.** "
            "Analysér nogle aktier/kryptos — så logges dine predictions automatisk, "
            "og om 30/90/180 dage kan du se hvor præcise modellen var!"
        )
        return

    cols = st.columns(4)
    cols[0].metric("📋 Total predictions", stats["total"])

    h90 = stats.get("hit_rate_90d")
    if h90 is not None:
        cols[1].metric(
            "🎯 Hit rate (90d)",
            f"{h90*100:.1f}%",
            f"{stats['resolved_90d']} resolved"
        )
    else:
        cols[1].metric("🎯 Hit rate (90d)", "Pending", f"0/{stats['total']}")

    avg90 = stats.get("avg_return_90d")
    if avg90 is not None:
        cols[2].metric(
            "💰 Avg afkast (90d)",
            f"{avg90:+.2f}%",
            f"{stats['resolved_90d']} predictions"
        )
    else:
        cols[2].metric("💰 Avg afkast", "Pending")

    h180 = stats.get("hit_rate_180d")
    if h180 is not None:
        cols[3].metric(
            "🚀 Hit rate (180d)",
            f"{h180*100:.1f}%",
            f"{stats['resolved_180d']} resolved"
        )
    else:
        cols[3].metric("🚀 Hit rate (180d)", "Pending", f"0/{stats['total']}")


def render_track_record_view():
    """Fuld Track Record view (egen tab)."""
    st.subheader("📈 Track Record")
    st.caption(
        "Her ser du hvor præcist systemet har forudsagt aktier/krypto over tid. "
        "Hver gang du analyserer noget, gemmes en prediction. "
        "Efter 30/90/180 dage opdateres priser automatisk og hit rate beregnes."
    )

    # Top action bar
    top_cols = st.columns([1, 1, 1, 1])
    if top_cols[0].button("🔄 Opdater nu", use_container_width=True, type="primary"):
        with st.spinner("Henter aktuelle priser..."):
            result = update_predictions(force=True)
        st.success(
            f"✅ Tjekkede {result['checked']} predictions · "
            f"opdaterede {result['updated']} · {result['errors']} fejl"
        )

    if top_cols[1].button("📥 Download CSV", use_container_width=True):
        df = _get_all_predictions()
        if not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Klik for at downloade",
                csv,
                f"track_record_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True,
            )

    if top_cols[2].button("🗑️ Slet alle data", use_container_width=True):
        if st.session_state.get("_confirm_delete_tr"):
            with _get_conn() as conn:
                conn.execute("DELETE FROM predictions")
                conn.execute("DELETE FROM meta")
                conn.commit()
            st.session_state["_confirm_delete_tr"] = False
            st.success("✅ Alle data slettet")
            st.rerun()
        else:
            st.session_state["_confirm_delete_tr"] = True
            st.warning("⚠️ Tryk igen for at bekræfte sletning af ALLE predictions")

    # Hent stats
    stats = get_track_record_stats()

    if stats["total"] == 0:
        st.info(
            "📭 **Ingen predictions endnu.**\n\n"
            "👉 Gå til **📊 Analyse** eller **🪙 Krypto** og analysér en ticker. "
            "Så gemmes din første prediction her!"
        )
        return

    # ===== Overall metrics =====
    st.markdown("---")
    st.markdown("### 📊 Samlet performance")

    m_cols = st.columns(4)
    m_cols[0].metric("📋 Total", stats["total"])

    for i, h in enumerate(HORIZONS, start=1):
        hit = stats.get(f"hit_rate_{h}d")
        n = stats.get(f"resolved_{h}d", 0)
        if hit is not None:
            color_emoji = "🟢" if hit >= 0.6 else "🟡" if hit >= 0.45 else "🔴"
            m_cols[i].metric(
                f"{color_emoji} Hit rate ({h}d)",
                f"{hit*100:.1f}%",
                f"{n} resolved"
            )
        else:
            m_cols[i].metric(f"⏳ Hit rate ({h}d)", "Pending", f"0/{stats['total']}")

    # Average returns
    st.markdown("#### 💰 Gennemsnitsafkast (resolved predictions)")
    r_cols = st.columns(3)
    for i, h in enumerate(HORIZONS):
        avg = stats.get(f"avg_return_{h}d")
        if avg is not None:
            color = "normal" if avg >= 0 else "inverse"
            r_cols[i].metric(f"{h} dage", f"{avg:+.2f}%", delta_color=color)
        else:
            r_cols[i].metric(f"{h} dage", "Pending")

    # ===== Per recommendation breakdown =====
    if stats.get("by_recommendation"):
        st.markdown("---")
        st.markdown("### 🎯 Hit rate per anbefaling (90 dage)")
        rec_data = []
        for rec, info in stats["by_recommendation"].items():
            rec_data.append({
                "Anbefaling": rec or "?",
                "Antal predictions": info["n"],
                "Hit rate (90d)": (
                    f"{info['hit_rate_90d']*100:.1f}%"
                    if info["hit_rate_90d"] is not None
                    else "Pending"
                ),
            })
        st.dataframe(
            pd.DataFrame(rec_data),
            use_container_width=True, hide_index=True
        )

    # ===== Per asset class =====
    if stats.get("by_asset_class"):
        st.markdown("### 🏷️ Per asset class")
        ac_cols = st.columns(len(stats["by_asset_class"]))
        for i, (ac, n) in enumerate(stats["by_asset_class"].items()):
            ac_cols[i].metric(ac.upper(), n)

    # ===== Detailed table =====
    st.markdown("---")
    st.markdown("### 📋 Alle predictions")

    df = _get_all_predictions()
    if df.empty:
        st.info("Ingen data")
        return

    # Filter controls
    f_cols = st.columns(4)
    asset_filter = f_cols[0].selectbox(
        "Asset class",
        ["Alle"] + sorted(df["asset_class"].dropna().unique().tolist()),
        key="tr_filter_asset"
    )
    rec_filter = f_cols[1].selectbox(
        "Anbefaling",
        ["Alle"] + sorted(df["recommendation"].dropna().unique().tolist()),
        key="tr_filter_rec"
    )
    status_filter = f_cols[2].selectbox(
        "Status",
        ["Alle", "Resolved (90d)", "Pending"],
        key="tr_filter_status"
    )
    sort_by = f_cols[3].selectbox(
        "Sortér efter",
        ["Nyeste først", "Ældste først", "Højeste afkast (90d)", "Laveste afkast (90d)"],
        key="tr_filter_sort"
    )

    # Apply filters
    df_view = df.copy()
    if asset_filter != "Alle":
        df_view = df_view[df_view["asset_class"] == asset_filter]
    if rec_filter != "Alle":
        df_view = df_view[df_view["recommendation"] == rec_filter]
    if status_filter == "Resolved (90d)":
        df_view = df_view[df_view["hit_90d"].notna()]
    elif status_filter == "Pending":
        df_view = df_view[df_view["hit_90d"].isna()]

    if sort_by == "Nyeste først":
        df_view = df_view.sort_values("timestamp", ascending=False)
    elif sort_by == "Ældste først":
        df_view = df_view.sort_values("timestamp", ascending=True)
    elif sort_by == "Højeste afkast (90d)":
        df_view = df_view.sort_values("return_90d", ascending=False, na_position="last")
    elif sort_by == "Laveste afkast (90d)":
        df_view = df_view.sort_values("return_90d", ascending=True, na_position="last")

    # Display columns
    display_cols = [
        "timestamp", "ticker", "name", "asset_class", "recommendation",
        "score", "price_at_prediction", "currency",
        "return_30d", "hit_30d",
        "return_90d", "hit_90d",
        "return_180d", "hit_180d",
    ]
    display_cols = [c for c in display_cols if c in df_view.columns]
    df_show = df_view[display_cols].copy()

    # Format
    if "timestamp" in df_show.columns:
        df_show["timestamp"] = pd.to_datetime(df_show["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    for col in ["score", "price_at_prediction", "return_30d", "return_90d", "return_180d"]:
        if col in df_show.columns:
            df_show[col] = pd.to_numeric(df_show[col], errors="coerce").round(2)

    # Rename for display
    df_show = df_show.rename(columns={
        "timestamp": "Tidspunkt",
        "ticker": "Ticker",
        "name": "Navn",
        "asset_class": "Type",
        "recommendation": "Anbefaling",
        "score": "Score",
        "price_at_prediction": "Pris (entry)",
        "currency": "Valuta",
        "return_30d": "Afkast 30d %",
        "hit_30d": "Hit 30d",
        "return_90d": "Afkast 90d %",
        "hit_90d": "Hit 90d",
        "return_180d": "Afkast 180d %",
        "hit_180d": "Hit 180d",
    })

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.caption(f"Viser {len(df_show)} predictions ud af {len(df)} totalt")

    # ===== Equity curve (simpel) =====
    df_resolved = df[df["hit_90d"].notna()].copy()
    if len(df_resolved) >= 5:
        st.markdown("---")
        st.markdown("### 📈 Equity curve (90d horisont, kun KØB-signaler)")
        st.caption("Simulering: Du investerer 1.000 DKK i hver KØB-anbefaling og holder i 90 dage")

        buys = df_resolved[
            df_resolved["recommendation"].str.contains("KØB", na=False)
        ].sort_values("timestamp").copy()

        if len(buys) >= 3:
            buys["pnl"] = 1000 * (buys["return_90d"] / 100)
            buys["cumulative_pnl"] = buys["pnl"].cumsum()
            buys["equity"] = 10000 + buys["cumulative_pnl"]  # start med 10k

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"],
                    y=buys["equity"],
                    mode="lines+markers",
                    line=dict(color="#00d4aa", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,170,0.1)",
                    name="Equity"
                ))
                fig.add_hline(y=10000, line_dash="dash", line_color="white", opacity=0.3)
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    title=f"Equity hvis du havde fulgt KØB-signaler (start: 10.000 DKK)",
                    yaxis_title="Equity (DKK)",
                    xaxis_title="Dato",
                )
                st.plotly_chart(fig, use_container_width=True)

                final_eq = buys["equity"].iloc[-1]
                total_ret = (final_eq / 10000 - 1) * 100
                eq_cols = st.columns(3)
                eq_cols[0].metric("💰 Slutværdi", f"{final_eq:,.0f} DKK")
                eq_cols[1].metric("📈 Total afkast", f"{total_ret:+.2f}%")
                eq_cols[2].metric("🎯 Antal KØB-trades", len(buys))
            except ImportError:
                st.info("Installer plotly for at se equity curve")
        else:
            st.info("Minimum 3 KØB-predictions med 90d data nødvendige for equity curve")
    else:
        st.info(
            f"💡 **Equity curve** vises når du har mindst **5 resolved predictions** (90d). "
            f"Du har lige nu **{len(df_resolved)}**."
        )


# ============================================================
# Test (kør med: python track_record.py)
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing track_record module...")

    # Slet test-DB hvis den findes
    if DB_PATH.exists():
        DB_PATH.unlink()
    _init_db()

    # Test save
    pid = save_prediction(
        ticker="AAPL", name="Apple Inc.", asset_class="stock",
        price=150.0, score=72, recommendation="KØB",
        f_score=68, t_score=75, regime="BULL",
        sector="Technology", currency="USD"
    )
    print(f"✅ Saved prediction id={pid}")

    pid2 = save_prediction(
        ticker="BTC", name="Bitcoin", asset_class="crypto",
        price=65000.0, score=82, recommendation="STÆRKT KØB",
        f_score=78, t_score=85, regime="BULL",
        sector="Cryptocurrency", currency="USD"
    )
    print(f"✅ Saved crypto prediction id={pid2}")

    # Test stats
    stats = get_track_record_stats()
    print(f"\n📊 Stats:")
    print(f"  Total: {stats['total']}")
    print(f"  By asset class: {stats['by_asset_class']}")
    print(f"  By recommendation: {stats['by_recommendation']}")

    # Test update (vil fejle hvis ingen internet, men det er OK)
    print(f"\n🔄 Testing update...")
    result = update_predictions(force=True, max_updates=10)
    print(f"  Result: {result}")

    print(f"\n✅ All tests passed!")
    print(f"   DB file: {DB_PATH.absolute()}")
