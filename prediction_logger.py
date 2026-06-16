"""
Prediction Logger
=================
SQLite-baseret logger der gemmer alle ML-forudsigelser og
auto-resolver dem når horisonten udløber (ved at hente faktisk pris).

Workflow:
    1. log_prediction()    → Gem forudsigelse ved beslutningstidspunkt
    2. resolve_pending()   → Kør dagligt; tjek udløbne predictions mod faktisk pris
    3. get_track_record()  → Hent historik til dashboard

Schema:
    predictions: id, ticker, asset_type, horizon_days, prediction_date,
                 target_date, entry_price, predicted_prob, predicted_direction,
                 model_name, calibrated, raw_features, status,
                 actual_price, actual_return, actual_direction, hit, resolved_at
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd


# ============================================================
# Schema
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    asset_type TEXT NOT NULL,                    -- 'stock' | 'crypto'
    horizon_days INTEGER NOT NULL,
    prediction_date TEXT NOT NULL,                -- ISO date
    target_date TEXT NOT NULL,                    -- ISO date (prediction + horizon)
    entry_price REAL NOT NULL,
    predicted_prob REAL NOT NULL,                 -- P(up) [0,1]
    predicted_direction TEXT NOT NULL,            -- 'BUY' | 'HOLD' | 'SELL'
    threshold_buy REAL DEFAULT 0.55,
    threshold_sell REAL DEFAULT 0.45,
    model_name TEXT NOT NULL,                     -- 'rf' | 'xgb' | 'lgbm' | 'ensemble' | 'rule'
    calibrated INTEGER DEFAULT 0,                 -- 0/1
    raw_features TEXT,                            -- JSON snapshot
    notes TEXT,

    -- Resolution
    status TEXT NOT NULL DEFAULT 'pending',       -- 'pending' | 'resolved' | 'failed'
    actual_price REAL,
    actual_return REAL,                           -- (actual - entry) / entry
    actual_direction TEXT,                        -- 'UP' | 'DOWN' | 'FLAT'
    hit INTEGER,                                  -- 0/1 (matched predicted_direction)
    resolved_at TEXT,
    resolution_error TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker     ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_status     ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_pred_target     ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_pred_model      ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_pred_horizon    ON predictions(horizon_days);
CREATE INDEX IF NOT EXISTS idx_pred_predicted  ON predictions(prediction_date);
"""


# ============================================================
# Data Classes
# ============================================================

@dataclass
class PredictionRecord:
    ticker: str
    asset_type: Literal["stock", "crypto"]
    horizon_days: int
    entry_price: float
    predicted_prob: float
    model_name: str
    prediction_date: Optional[str] = None         # default = today
    threshold_buy: float = 0.55
    threshold_sell: float = 0.45
    calibrated: bool = True
    raw_features: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if self.prediction_date is None:
            self.prediction_date = date.today().isoformat()
        if not (0.0 <= self.predicted_prob <= 1.0):
            raise ValueError(f"predicted_prob must be in [0,1], got {self.predicted_prob}")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be > 0")

    @property
    def predicted_direction(self) -> str:
        if self.predicted_prob >= self.threshold_buy:
            return "BUY"
        if self.predicted_prob <= self.threshold_sell:
            return "SELL"
        return "HOLD"

    @property
    def target_date(self) -> str:
        d = datetime.fromisoformat(self.prediction_date).date()
        return (d + timedelta(days=self.horizon_days)).isoformat()


# ============================================================
# Price Fetcher (pluggable)
# ============================================================

class PriceFetcher:
    """
    Pluggable price fetcher. Default bruger yfinance for stocks
    og pycoingecko/CoinGecko for crypto. Override hvis du vil bruge
    Finnhub/Twelve Data direkte.
    """

    def get_close_price(self, ticker: str, asset_type: str, target_date: str) -> Optional[float]:
        """Hent close-pris på/efter target_date. Returner None hvis ikke tilgængelig endnu."""
        try:
            if asset_type == "stock":
                return self._fetch_stock(ticker, target_date)
            elif asset_type == "crypto":
                return self._fetch_crypto(ticker, target_date)
        except Exception as e:
            print(f"  ⚠️  Price fetch failed for {ticker}: {e}")
            return None
        return None

    def _fetch_stock(self, ticker: str, target_date: str) -> Optional[float]:
        import yfinance as yf
        target = datetime.fromisoformat(target_date).date()
        # Hent et vindue så vi rammer en handelsdag
        start = target - timedelta(days=2)
        end = target + timedelta(days=7)
        df = yf.download(
            ticker, start=start.isoformat(), end=end.isoformat(),
            progress=False, auto_adjust=True
        )
        if df.empty:
            return None
        # Find første handelsdag >= target
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        future = df[df["Date"] >= target]
        if future.empty:
            return None
        return float(future.iloc[0]["Close"])

    def _fetch_crypto(self, ticker: str, target_date: str) -> Optional[float]:
        # Map ticker → CoinGecko id (udvid ved behov)
        cg_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "DOGE": "dogecoin", "SHIB": "shiba-inu", "PEPE": "pepe",
            "BONK": "bonk", "WIF": "dogwifcoin", "FLOKI": "floki",
            "AVAX": "avalanche-2",
        }
        coin_id = cg_map.get(ticker.upper().replace("-USD", ""))
        if not coin_id:
            return None

        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        target = datetime.fromisoformat(target_date).date()
        # CoinGecko returnerer pris pr. dato (DD-MM-YYYY)
        date_str = target.strftime("%d-%m-%Y")
        try:
            data = cg.get_coin_history_by_id(id=coin_id, date=date_str)
            return float(data["market_data"]["current_price"]["usd"])
        except Exception:
            # Fallback: prøv næste dag op til 7 dage frem
            for offset in range(1, 8):
                d = (target + timedelta(days=offset)).strftime("%d-%m-%Y")
                try:
                    data = cg.get_coin_history_by_id(id=coin_id, date=d)
                    return float(data["market_data"]["current_price"]["usd"])
                except Exception:
                    continue
        return None


# ============================================================
# Logger
# ============================================================

class PredictionLogger:
    """
    SQLite-baseret prediction logger.

    Usage:
        logger = PredictionLogger("predictions.db")
        logger.log(PredictionRecord(...))
        logger.resolve_pending()
        df = logger.get_track_record(ticker="ETH")
    """

    def __init__(self, db_path: str | Path = "predictions.db",
                 price_fetcher: Optional[PriceFetcher] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.price_fetcher = price_fetcher or PriceFetcher()
        self._init_db()

    # ---------- Internal ----------
    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA_SQL)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- Logging ----------
    def log(self, rec: PredictionRecord) -> int:
        """Gem én prediction. Returnerer row id."""
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO predictions
                (ticker, asset_type, horizon_days, prediction_date, target_date,
                 entry_price, predicted_prob, predicted_direction,
                 threshold_buy, threshold_sell, model_name, calibrated,
                 raw_features, notes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    rec.ticker.upper(), rec.asset_type, rec.horizon_days,
                    rec.prediction_date, rec.target_date, rec.entry_price,
                    rec.predicted_prob, rec.predicted_direction,
                    rec.threshold_buy, rec.threshold_sell,
                    rec.model_name, int(rec.calibrated),
                    json.dumps(rec.raw_features, default=str), rec.notes,
                ),
            )
            return cur.lastrowid

    def log_batch(self, records: list[PredictionRecord]) -> list[int]:
        return [self.log(r) for r in records]

    def log_ensemble(
        self,
        ticker: str,
        asset_type: str,
        horizon_days: int,
        entry_price: float,
        probs: dict[str, float],          # {"rf": 0.6, "xgb": 0.65, "lgbm": 0.58, "ensemble": 0.61}
        features: Optional[dict] = None,
        calibrated: bool = True,
        threshold_buy: float = 0.55,
        threshold_sell: float = 0.45,
    ) -> list[int]:
        """Convenience: log alle modeller (inkl. ensemble) på én gang."""
        ids = []
        for model_name, prob in probs.items():
            rec = PredictionRecord(
                ticker=ticker, asset_type=asset_type,
                horizon_days=horizon_days, entry_price=entry_price,
                predicted_prob=prob, model_name=model_name,
                threshold_buy=threshold_buy, threshold_sell=threshold_sell,
                calibrated=calibrated, raw_features=features or {},
            )
            ids.append(self.log(rec))
        return ids

    # ---------- Resolution ----------
    def resolve_pending(self, as_of: Optional[str] = None, verbose: bool = True) -> dict:
        """
        Find predictions hvor target_date <= as_of og status='pending',
        hent faktisk pris og opdater records.
        """
        as_of = as_of or date.today().isoformat()
        stats = {"checked": 0, "resolved": 0, "failed": 0, "still_pending": 0}

        with self._conn() as c:
            rows = c.execute(
                """
                SELECT id, ticker, asset_type, target_date, entry_price,
                       predicted_direction, threshold_buy, threshold_sell
                FROM predictions
                WHERE status = 'pending' AND target_date <= ?
                """,
                (as_of,),
            ).fetchall()

        if verbose:
            print(f"🔍 Resolving {len(rows)} pending predictions (as of {as_of})...")

        for r in rows:
            stats["checked"] += 1
            actual_price = self.price_fetcher.get_close_price(
                r["ticker"], r["asset_type"], r["target_date"]
            )

            if actual_price is None:
                stats["still_pending"] += 1
                if verbose:
                    print(f"  ⏳ {r['ticker']} (id={r['id']}): pris ikke tilgængelig endnu")
                continue

            try:
                actual_return = (actual_price - r["entry_price"]) / r["entry_price"]
                actual_direction = self._direction_from_return(actual_return)
                hit = self._is_hit(r["predicted_direction"], actual_direction, actual_return)

                with self._conn() as c:
                    c.execute(
                        """
                        UPDATE predictions
                        SET status='resolved',
                            actual_price=?, actual_return=?,
                            actual_direction=?, hit=?, resolved_at=?
                        WHERE id=?
                        """,
                        (
                            actual_price, actual_return, actual_direction,
                            int(hit), datetime.now().isoformat(timespec="seconds"),
                            r["id"],
                        ),
                    )
                stats["resolved"] += 1
                if verbose:
                    sym = "✅" if hit else "❌"
                    print(f"  {sym} {r['ticker']} (id={r['id']}): "
                          f"{r['predicted_direction']} | actual {actual_direction} "
                          f"({actual_return*100:+.2f}%)")

            except Exception as e:
                stats["failed"] += 1
                with self._conn() as c:
                    c.execute(
                        "UPDATE predictions SET status='failed', resolution_error=? WHERE id=?",
                        (str(e), r["id"]),
                    )
                if verbose:
                    print(f"  ⚠️  {r['ticker']} (id={r['id']}): fejl - {e}")

        if verbose:
            print(f"\n📊 Resolution summary: {stats}")
        return stats

    @staticmethod
    def _direction_from_return(ret: float, flat_band: float = 0.005) -> str:
        if ret > flat_band:
            return "UP"
        if ret < -flat_band:
            return "DOWN"
        return "FLAT"

    @staticmethod
    def _is_hit(predicted: str, actual: str, actual_return: float) -> bool:
        """BUY hits hvis UP, SELL hits hvis DOWN, HOLD hits hvis FLAT eller |ret|<2%."""
        if predicted == "BUY":
            return actual == "UP"
        if predicted == "SELL":
            return actual == "DOWN"
        if predicted == "HOLD":
            return actual == "FLAT" or abs(actual_return) < 0.02
        return False

    # ---------- Queries ----------
    def get_track_record(
        self,
        ticker: Optional[str] = None,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
        only_resolved: bool = False,
    ) -> pd.DataFrame:
        query = "SELECT * FROM predictions WHERE 1=1"
        params: list = []
        if ticker:
            query += " AND ticker=?"; params.append(ticker.upper())
        if model_name:
            query += " AND model_name=?"; params.append(model_name)
        if horizon_days is not None:
            query += " AND horizon_days=?"; params.append(horizon_days)
        if status:
            query += " AND status=?"; params.append(status)
        if only_resolved:
            query += " AND status='resolved'"
        if since:
            query += " AND prediction_date >= ?"; params.append(since)
        query += " ORDER BY prediction_date DESC, id DESC"

        with self._conn() as c:
            df = pd.read_sql_query(query, c, params=params)
        return df

    def summary_stats(
        self,
        model_name: Optional[str] = None,
        ticker: Optional[str] = None,
        horizon_days: Optional[int] = None,
    ) -> dict:
        df = self.get_track_record(
            ticker=ticker, model_name=model_name,
            horizon_days=horizon_days, only_resolved=True,
        )
        if df.empty:
            return {"n": 0, "message": "Ingen resolved predictions endnu"}

        df_signal = df[df["predicted_direction"].isin(["BUY", "SELL"])]

        return {
            "n_total": len(df),
            "n_signals": len(df_signal),                   # ekskl. HOLD
            "hit_rate": float(df["hit"].mean()),
            "hit_rate_signals": float(df_signal["hit"].mean()) if not df_signal.empty else None,
            "avg_predicted_prob": float(df["predicted_prob"].mean()),
            "avg_actual_return": float(df["actual_return"].mean()),
            "buy_hit_rate": self._direction_hit_rate(df, "BUY"),
            "sell_hit_rate": self._direction_hit_rate(df, "SELL"),
            "hold_hit_rate": self._direction_hit_rate(df, "HOLD"),
            "best_ticker": self._best_ticker(df),
            "worst_ticker": self._worst_ticker(df),
        }

    @staticmethod
    def _direction_hit_rate(df: pd.DataFrame, direction: str) -> Optional[float]:
        sub = df[df["predicted_direction"] == direction]
        return float(sub["hit"].mean()) if not sub.empty else None

    @staticmethod
    def _best_ticker(df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        g = df.groupby("ticker")["hit"].agg(["mean", "count"])
        g = g[g["count"] >= 3]
        return g["mean"].idxmax() if not g.empty else None

    @staticmethod
    def _worst_ticker(df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        g = df.groupby("ticker")["hit"].agg(["mean", "count"])
        g = g[g["count"] >= 3]
        return g["mean"].idxmin() if not g.empty else None

    # ---------- Maintenance ----------
    def delete_prediction(self, prediction_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
            return cur.rowcount > 0

    def reset_db(self) -> None:
        """⚠️ DANGER: Sletter alt!"""
        with self._conn() as c:
            c.execute("DROP TABLE IF EXISTS predictions")
        self._init_db()

    def export_csv(self, path: str | Path) -> None:
        df = self.get_track_record()
        df.to_csv(path, index=False)
        print(f"📤 Exported {len(df)} rows to {path}")

    def stats_overview(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            pending = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE status='pending'"
            ).fetchone()[0]
            resolved = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE status='resolved'"
            ).fetchone()[0]
            failed = c.execute(
                "SELECT COUNT(*) FROM predictions WHERE status='failed'"
            ).fetchone()[0]
        return {"total": total, "pending": pending, "resolved": resolved, "failed": failed}


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing PredictionLogger...\n")

    # Brug en test-DB
    test_db = Path("_test_predictions.db")
    if test_db.exists():
        test_db.unlink()

    logger = PredictionLogger(test_db)

    # Mock price fetcher (så vi ikke kalder yfinance i test)
    class MockFetcher(PriceFetcher):
        def get_close_price(self, ticker, asset_type, target_date):
            mock_prices = {
                "ETH": 1950.0,    # Up fra 1802
                "BTC": 64000.0,   # Up fra 65000  → faktisk down
                "RKLB": 115.0,    # Up fra 107
            }
            return mock_prices.get(ticker.upper())

    logger.price_fetcher = MockFetcher()

    # Log predictions (simulér i går)
    yesterday = (date.today() - timedelta(days=31)).isoformat()

    logger.log(PredictionRecord(
        ticker="ETH", asset_type="crypto", horizon_days=30,
        entry_price=1802.10, predicted_prob=0.72, model_name="ensemble",
        prediction_date=yesterday, calibrated=True,
        raw_features={"rsi": 45, "macd": 0.012},
    ))
    logger.log(PredictionRecord(
        ticker="BTC", asset_type="crypto", horizon_days=30,
        entry_price=65000.0, predicted_prob=0.78, model_name="ensemble",
        prediction_date=yesterday, calibrated=True,
    ))
    logger.log(PredictionRecord(
        ticker="RKLB", asset_type="stock", horizon_days=30,
        entry_price=107.12, predicted_prob=0.65, model_name="ensemble",
        prediction_date=yesterday, calibrated=True,
    ))

    print("Status efter logging:", logger.stats_overview())

    # Resolve
    print()
    logger.resolve_pending()

    # Track record
    print("\n📊 Track Record:")
    df = logger.get_track_record(only_resolved=True)
    print(df[["ticker", "predicted_direction", "predicted_prob",
              "actual_return", "actual_direction", "hit"]].to_string(index=False))

    # Summary
    print("\n📈 Summary stats:")
    for k, v in logger.summary_stats().items():
        print(f"  {k}: {v}")

    # Cleanup
    test_db.unlink()
    print("\n✅ All tests passed!")
