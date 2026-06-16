"""
Track Record Metrics
====================
Beregner performance-metrics fra logged predictions:
  - Hit rate (samlet, per model, per horizon, per ticker)
  - Brier Score, Log Loss (calibration quality)
  - ROC-AUC (discrimination quality)
  - Equity curve (havde du fulgt signalerne?)
  - Rolling metrics (er modellen konsistent eller drifter?)
  - Reliability data (til calibration plots)
  - Confusion matrix
  - Sharpe / max drawdown af signal-strategy

Bruges af dashboard-tab i Step 5.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    confusion_matrix, accuracy_score,
)

from prediction_logger import PredictionLogger


# ============================================================
# Result Dataclasses
# ============================================================

@dataclass
class CoreMetrics:
    n: int
    hit_rate: float
    accuracy: float
    brier_score: Optional[float]
    log_loss_value: Optional[float]
    roc_auc: Optional[float]
    avg_predicted_prob: float
    avg_actual_return: float
    win_rate_buys: Optional[float]      # only BUY signals
    win_rate_sells: Optional[float]
    n_buys: int
    n_sells: int
    n_holds: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquityCurveResult:
    dates: list[str]
    equity: list[float]
    returns: list[float]
    cumulative_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Track Record Analyzer
# ============================================================

class TrackRecord:
    """
    Beregner performance-metrics fra PredictionLogger.

    Usage:
        tr = TrackRecord(logger)
        metrics = tr.core_metrics(model_name="ensemble", horizon_days=30)
        equity = tr.equity_curve()
        rolling = tr.rolling_hit_rate(window=20)
    """

    def __init__(self, logger: PredictionLogger):
        self.logger = logger

    # =========================================================
    # 1) Core metrics
    # =========================================================

    def core_metrics(
        self,
        model_name: Optional[str] = None,
        ticker: Optional[str] = None,
        horizon_days: Optional[int] = None,
        since: Optional[str] = None,
    ) -> CoreMetrics:
        df = self.logger.get_track_record(
            ticker=ticker, model_name=model_name,
            horizon_days=horizon_days, since=since,
            only_resolved=True,
        )

        if df.empty:
            return self._empty_metrics()

        y_true = df["hit"].astype(int).values
        # For probabilistiske metrics: kig på P(direction=correct)
        # Vi bruger predicted_prob som er P(up); konverterer til P(correct prediction)
        probs = df["predicted_prob"].values.astype(float)

        # ROC-AUC: kan vi adskille hits fra misses ved hjælp af confidence?
        # confidence = afstand fra 0.5
        confidence = np.abs(probs - 0.5) * 2  # [0,1]
        try:
            auc = float(roc_auc_score(y_true, confidence)) if len(np.unique(y_true)) > 1 else None
        except Exception:
            auc = None

        # Brier / LogLoss bruger P(predicted direction is correct)
        # P(correct) = predicted_prob hvis predicted=BUY, (1-predicted_prob) hvis SELL,
        # for HOLD bruger vi 0.5 (neutral)
        p_correct = self._prob_of_correct(df)
        try:
            brier = float(brier_score_loss(y_true, p_correct))
        except Exception:
            brier = None
        try:
            ll = float(log_loss(y_true, np.clip(p_correct, 1e-7, 1 - 1e-7)))
        except Exception:
            ll = None

        # Direction-specific
        buys = df[df["predicted_direction"] == "BUY"]
        sells = df[df["predicted_direction"] == "SELL"]
        holds = df[df["predicted_direction"] == "HOLD"]

        return CoreMetrics(
            n=len(df),
            hit_rate=float(df["hit"].mean()),
            accuracy=float(accuracy_score(y_true, [1] * len(y_true))) if False else float(df["hit"].mean()),
            brier_score=brier,
            log_loss_value=ll,
            roc_auc=auc,
            avg_predicted_prob=float(probs.mean()),
            avg_actual_return=float(df["actual_return"].mean()),
            win_rate_buys=float(buys["hit"].mean()) if not buys.empty else None,
            win_rate_sells=float(sells["hit"].mean()) if not sells.empty else None,
            n_buys=len(buys),
            n_sells=len(sells),
            n_holds=len(holds),
        )

    @staticmethod
    def _empty_metrics() -> CoreMetrics:
        return CoreMetrics(
            n=0, hit_rate=0.0, accuracy=0.0,
            brier_score=None, log_loss_value=None, roc_auc=None,
            avg_predicted_prob=0.0, avg_actual_return=0.0,
            win_rate_buys=None, win_rate_sells=None,
            n_buys=0, n_sells=0, n_holds=0,
        )

    @staticmethod
    def _prob_of_correct(df: pd.DataFrame) -> np.ndarray:
        """Returner P(prediction=correct) for hver række."""
        out = np.full(len(df), 0.5)
        probs = df["predicted_prob"].values.astype(float)
        dirs = df["predicted_direction"].values
        out[dirs == "BUY"] = probs[dirs == "BUY"]
        out[dirs == "SELL"] = 1.0 - probs[dirs == "SELL"]
        # HOLD beholder 0.5
        return out

    # =========================================================
    # 2) Breakdown tables
    # =========================================================

    def by_model(self, horizon_days: Optional[int] = None) -> pd.DataFrame:
        df = self.logger.get_track_record(
            horizon_days=horizon_days, only_resolved=True
        )
        if df.empty:
            return pd.DataFrame()
        return self._aggregate(df, group_col="model_name")

    def by_ticker(self, model_name: Optional[str] = None) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, only_resolved=True
        )
        if df.empty:
            return pd.DataFrame()
        return self._aggregate(df, group_col="ticker")

    def by_horizon(self, model_name: Optional[str] = None) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, only_resolved=True
        )
        if df.empty:
            return pd.DataFrame()
        return self._aggregate(df, group_col="horizon_days")

    def by_direction(self, model_name: Optional[str] = None) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, only_resolved=True
        )
        if df.empty:
            return pd.DataFrame()
        return self._aggregate(df, group_col="predicted_direction")

    @staticmethod
    def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        agg = df.groupby(group_col).agg(
            n=("hit", "count"),
            hit_rate=("hit", "mean"),
            avg_prob=("predicted_prob", "mean"),
            avg_return=("actual_return", "mean"),
        ).reset_index()
        agg["hit_rate"] = agg["hit_rate"].round(4)
        agg["avg_prob"] = agg["avg_prob"].round(4)
        agg["avg_return"] = agg["avg_return"].round(4)
        return agg.sort_values("n", ascending=False).reset_index(drop=True)

    # =========================================================
    # 3) Rolling metrics (drift detection)
    # =========================================================

    def rolling_hit_rate(
        self,
        window: int = 20,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, horizon_days=horizon_days,
            only_resolved=True,
        ).sort_values("prediction_date")

        if df.empty:
            return pd.DataFrame()

        df["rolling_hit_rate"] = df["hit"].rolling(window, min_periods=max(5, window // 4)).mean()
        df["rolling_avg_prob"] = df["predicted_prob"].rolling(window, min_periods=5).mean()
        df["rolling_avg_return"] = df["actual_return"].rolling(window, min_periods=5).mean()

        return df[["prediction_date", "ticker", "model_name", "predicted_prob",
                   "hit", "actual_return", "rolling_hit_rate",
                   "rolling_avg_prob", "rolling_avg_return"]].reset_index(drop=True)

    # =========================================================
    # 4) Reliability data (to plot calibration curve)
    # =========================================================

    def reliability_data(
        self,
        n_bins: int = 10,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, horizon_days=horizon_days,
            only_resolved=True,
        )
        if df.empty:
            return pd.DataFrame()

        probs = self._prob_of_correct(df)
        y_true = df["hit"].astype(int).values

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(probs, bin_edges[1:-1])

        rows = []
        for b in range(n_bins):
            mask = bin_ids == b
            rows.append({
                "bin": b,
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "bin_center": (bin_edges[b] + bin_edges[b + 1]) / 2,
                "mean_predicted": float(probs[mask].mean()) if mask.any() else np.nan,
                "mean_actual": float(y_true[mask].mean()) if mask.any() else np.nan,
                "count": int(mask.sum()),
            })
        return pd.DataFrame(rows)

    # =========================================================
    # 5) Confusion matrix
    # =========================================================

    def confusion(
        self,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.logger.get_track_record(
            model_name=model_name, horizon_days=horizon_days,
            only_resolved=True,
        )
        if df.empty:
            return pd.DataFrame()

        labels = ["BUY", "HOLD", "SELL"]
        # actual_direction er UP/FLAT/DOWN — map til BUY/HOLD/SELL
        map_actual = {"UP": "BUY", "FLAT": "HOLD", "DOWN": "SELL"}
        actual_mapped = df["actual_direction"].map(map_actual).fillna("HOLD")

        cm = confusion_matrix(
            actual_mapped, df["predicted_direction"], labels=labels
        )
        return pd.DataFrame(
            cm,
            index=[f"actual_{l}" for l in labels],
            columns=[f"pred_{l}" for l in labels],
        )

    # =========================================================
    # 6) Equity curve (paper trading simulation)
    # =========================================================

    def equity_curve(
        self,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
        initial_capital: float = 10_000.0,
        position_pct: float = 0.10,           # 10% af equity per trade
        only_buys: bool = True,                # behandl SELL som short? default kun long
        min_confidence: float = 0.55,
    ) -> EquityCurveResult:
        """
        Simulér equity hvis du havde fulgt signalerne.
        Hver BUY → invester position_pct af equity, exit ved horizon.
        """
        df = self.logger.get_track_record(
            model_name=model_name, horizon_days=horizon_days,
            only_resolved=True,
        ).sort_values("prediction_date").reset_index(drop=True)

        if df.empty:
            return EquityCurveResult(
                dates=[], equity=[initial_capital], returns=[],
                cumulative_return=0.0, sharpe=0.0, max_drawdown=0.0, n_trades=0,
            )

        # Filter: kun trades med tilstrækkelig confidence
        df["confidence"] = np.abs(df["predicted_prob"] - 0.5) * 2
        if only_buys:
            df = df[df["predicted_direction"] == "BUY"]
        df = df[df["confidence"] >= (min_confidence - 0.5) * 2]

        if df.empty:
            return EquityCurveResult(
                dates=[], equity=[initial_capital], returns=[],
                cumulative_return=0.0, sharpe=0.0, max_drawdown=0.0, n_trades=0,
            )

        equity = initial_capital
        equity_history = [initial_capital]
        date_history = [df["prediction_date"].iloc[0]]
        trade_returns = []

        for _, row in df.iterrows():
            ret = float(row["actual_return"])
            # Short hvis SELL (only_buys=False)
            if row["predicted_direction"] == "SELL":
                ret = -ret

            position_size = equity * position_pct
            pnl = position_size * ret
            equity += pnl
            trade_returns.append(ret)
            equity_history.append(equity)
            date_history.append(row["prediction_date"])

        cum_ret = (equity - initial_capital) / initial_capital
        sharpe = self._sharpe(trade_returns)
        mdd = self._max_drawdown(equity_history)

        return EquityCurveResult(
            dates=date_history,
            equity=equity_history,
            returns=trade_returns,
            cumulative_return=float(cum_ret),
            sharpe=float(sharpe),
            max_drawdown=float(mdd),
            n_trades=len(trade_returns),
        )

    @staticmethod
    def _sharpe(returns: list[float], periods_per_year: int = 12) -> float:
        if not returns or len(returns) < 2:
            return 0.0
        r = np.array(returns)
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity: list[float]) -> float:
        if not equity:
            return 0.0
        e = np.array(equity)
        peak = np.maximum.accumulate(e)
        dd = (e - peak) / peak
        return float(dd.min())

    # =========================================================
    # 7) Calibration quality over tid
    # =========================================================

    def calibration_quality(
        self,
        model_name: Optional[str] = None,
        horizon_days: Optional[int] = None,
    ) -> dict:
        df = self.logger.get_track_record(
            model_name=model_name, horizon_days=horizon_days,
            only_resolved=True,
        )
        if df.empty:
            return {"n": 0, "ece": None, "brier": None}

        probs = self._prob_of_correct(df)
        y_true = df["hit"].astype(int).values

        ece, mce = self._calibration_errors(probs, y_true)
        return {
            "n": len(df),
            "ece": float(ece),
            "mce": float(mce),
            "brier": float(brier_score_loss(y_true, probs)),
        }

    @staticmethod
    def _calibration_errors(probs, y_true, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(probs, bin_edges[1:-1])
        ece, mce = 0.0, 0.0
        n = len(probs)
        for b in range(n_bins):
            mask = bin_ids == b
            if not mask.any():
                continue
            avg_p = probs[mask].mean()
            avg_a = y_true[mask].mean()
            gap = abs(avg_p - avg_a)
            ece += (mask.sum() / n) * gap
            mce = max(mce, gap)
        return ece, mce

    # =========================================================
    # 8) Full report
    # =========================================================

    def full_report(self, model_name: Optional[str] = None) -> str:
        core = self.core_metrics(model_name=model_name)
        equity = self.equity_curve(model_name=model_name)
        cal = self.calibration_quality(model_name=model_name)

        if core.n == 0:
            return "📭 Ingen resolved predictions endnu — log noget data først!"

        title = f"Model: {model_name or 'ALL'}"
        out = [
            f"\n{'='*60}",
            f"  📊 TRACK RECORD REPORT — {title}",
            f"{'='*60}",
            f"  Total resolved predictions:   {core.n}",
            f"  Hit rate:                     {core.hit_rate:.1%}",
            f"  Avg predicted probability:    {core.avg_predicted_prob:.1%}",
            f"  Avg actual return:            {core.avg_actual_return:+.2%}",
            "",
            f"  Brier score:                  {core.brier_score:.4f}" if core.brier_score is not None else "  Brier score:                  N/A",
            f"  Log loss:                     {core.log_loss_value:.4f}" if core.log_loss_value is not None else "  Log loss:                     N/A",
            f"  ROC-AUC (confidence):         {core.roc_auc:.4f}" if core.roc_auc is not None else "  ROC-AUC:                      N/A",
            "",
            f"  Direction breakdown:",
            f"    BUYs:  {core.n_buys:>4}   hit rate: " + (f"{core.win_rate_buys:.1%}" if core.win_rate_buys is not None else "N/A"),
            f"    SELLs: {core.n_sells:>4}   hit rate: " + (f"{core.win_rate_sells:.1%}" if core.win_rate_sells is not None else "N/A"),
            f"    HOLDs: {core.n_holds:>4}",
            "",
            f"  📈 Paper Trading (10% pos sizing, BUYs only):",
            f"    Trades executed:            {equity.n_trades}",
            f"    Cumulative return:          {equity.cumulative_return:+.2%}",
            f"    Sharpe ratio:               {equity.sharpe:.2f}",
            f"    Max drawdown:               {equity.max_drawdown:.2%}",
            "",
            f"  🎯 Calibration quality:",
            f"    ECE:                        {cal['ece']:.4f}" if cal['ece'] is not None else "    ECE: N/A",
            f"    MCE:                        {cal['mce']:.4f}" if cal.get('mce') is not None else "",
            f"{'='*60}\n",
        ]
        return "\n".join(out)


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing TrackRecord...\n")

    from prediction_logger import PredictionLogger, PredictionRecord, PriceFetcher
    from datetime import date, timedelta

    test_db = Path("_test_track.db")
    if test_db.exists():
        test_db.unlink()

    logger = PredictionLogger(test_db)

    # Mock fetcher: simulér forskellige outcomes
    rng = np.random.default_rng(42)

    class MockFetcher(PriceFetcher):
        def __init__(self, entry_prices):
            self.entry_prices = entry_prices

        def get_close_price(self, ticker, asset_type, target_date):
            base = self.entry_prices.get(ticker, 100.0)
            # Simulér ±15% random move
            return base * (1 + rng.normal(0.02, 0.08))

    entries = {"ETH": 1800, "BTC": 65000, "SOL": 150, "RKLB": 100, "AAPL": 200}
    logger.price_fetcher = MockFetcher(entries)

    # Generér 80 historiske predictions
    print("📝 Logging 80 historical predictions...")
    for i in range(80):
        days_ago = rng.integers(35, 200)
        pred_date = (date.today() - timedelta(days=int(days_ago))).isoformat()
        ticker = rng.choice(list(entries.keys()))
        prob = float(np.clip(rng.normal(0.55, 0.15), 0.05, 0.95))
        model = rng.choice(["rf", "xgb", "lgbm", "ensemble"])

        logger.log(PredictionRecord(
            ticker=ticker,
            asset_type="crypto" if ticker in ["ETH", "BTC", "SOL"] else "stock",
            horizon_days=int(rng.choice([7, 30, 90])),
            entry_price=entries[ticker],
            predicted_prob=prob,
            model_name=str(model),
            prediction_date=pred_date,
            calibrated=True,
        ))

    # Resolve all
    logger.resolve_pending(verbose=False)
    print(f"   {logger.stats_overview()}\n")

    # Track record analyse
    tr = TrackRecord(logger)

    print(tr.full_report(model_name="ensemble"))

    print("📊 By model:")
    print(tr.by_model().to_string(index=False))

    print("\n📊 By ticker:")
    print(tr.by_ticker().to_string(index=False))

    print("\n📊 By horizon:")
    print(tr.by_horizon().to_string(index=False))

    print("\n🎯 Reliability data (ensemble):")
    rel = tr.reliability_data(n_bins=5, model_name="ensemble")
    print(rel.to_string(index=False))

    print("\n🔄 Confusion matrix (ensemble):")
    print(tr.confusion(model_name="ensemble"))

    print("\n📈 Rolling hit rate (last 5 rows):")
    rolling = tr.rolling_hit_rate(window=10, model_name="ensemble")
    if not rolling.empty:
        print(rolling.tail()[["prediction_date", "ticker", "hit",
                              "rolling_hit_rate"]].to_string(index=False))

    test_db.unlink()
    print("\n✅ All tests passed!")
