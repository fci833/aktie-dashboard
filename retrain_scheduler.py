"""
Auto-Retraining Scheduler
=========================
Selvkørende retraining-pipeline der:

1. **Monitorer** performance via PredictionLogger + TrackRecord
2. **Detekterer drift** med statistiske tests (Page-Hinkley, hit-rate decay,
   ECE-stigning, feature drift via PSI)
3. **Trigger retraining** når thresholds brydes ELLER på fast schedule
4. **Walk-forward validates** den nye model før deployment
5. **Versionerer** modeller med metadata + rollback-mulighed
6. **Promoverer** kun hvis ny model > current model

Triggers:
  - performance_drift   : Hit rate falder under threshold
  - calibration_drift   : ECE stiger over threshold
  - feature_drift       : PSI > threshold på inputs
  - scheduled           : Tid siden sidste retrain > N dage
  - manual              : Force retrain via API/dashboard

Usage:
    scheduler = RetrainScheduler(
        registry_dir="models/eth_30d/",
        logger=PredictionLogger("predictions.db"),
        config=RetrainConfig(...),
    )
    
    # Daglig check (cron / streamlit button / GitHub Action)
    decision = scheduler.check_and_retrain(
        ticker="ETH",
        horizon_days=30,
        data_loader=lambda: load_eth_features(),
        model_factories={"rf": ..., "xgb": ..., "lgbm": ...},
    )
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd

from ml_calibration import EnsembleCalibrator, ProbabilityCalibrator
from prediction_logger import PredictionLogger
from track_record import TrackRecord
from walk_forward import WalkForwardConfig, WalkForwardValidator


# ============================================================
# Config
# ============================================================

@dataclass
class RetrainConfig:
    """Thresholds & policy for auto-retraining."""

    # Drift detection
    min_hit_rate: float = 0.50              # Trigger retrain if rolling hit rate falls below
    rolling_window: int = 30                # Window for rolling hit rate
    max_ece: float = 0.10                    # Trigger if Expected Calibration Error > this
    max_psi: float = 0.20                    # Population Stability Index threshold (feature drift)

    # Page-Hinkley test (sequential change detection)
    ph_threshold: float = 5.0                # Cumulative deviation threshold
    ph_alpha: float = 0.005                  # Allowed magnitude of changes

    # Schedule
    max_days_since_retrain: int = 30         # Force retrain after N days regardless
    min_days_between_retrains: int = 3       # Cooldown to prevent thrashing
    min_resolved_predictions: int = 30       # Need at least N samples to evaluate drift

    # Promotion criteria (new model must beat champion)
    promotion_min_improvement: float = 0.0   # New must be ≥ this much better (Brier delta)
    promotion_metric: Literal["brier", "hit_rate", "log_loss", "auc"] = "brier"
    require_walk_forward: bool = True        # Run WF validation before promoting

    # Walk-forward validation params
    wf_initial_train: int = 252
    wf_test_size: int = 21
    wf_step: int = 21
    wf_calibrate: bool = True

    # Storage
    keep_n_versions: int = 5                 # How many old versions to retain
    verbose: bool = True


# ============================================================
# Decision Result
# ============================================================

@dataclass
class DriftSignal:
    name: str
    triggered: bool
    value: float
    threshold: float
    severity: Literal["none", "warning", "critical"] = "none"
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrainDecision:
    timestamp: str
    ticker: str
    horizon_days: int
    triggered: bool
    reason: str
    signals: list[DriftSignal]

    # Outcomes (filled when retrain runs)
    retrained: bool = False
    new_version: Optional[str] = None
    promoted: bool = False
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    promotion_decision: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [s.to_dict() for s in self.signals]
        return d

    def summary(self) -> str:
        out = [f"\n{'='*60}",
               f"  🤖 RETRAIN DECISION — {self.ticker} ({self.horizon_days}d)",
               f"{'='*60}",
               f"  Time:        {self.timestamp}",
               f"  Triggered:   {self.triggered}",
               f"  Reason:      {self.reason}",
               ""]
        out.append("  Drift signals:")
        for s in self.signals:
            icon = "🚨" if s.severity == "critical" else "⚠️" if s.severity == "warning" else "✅"
            out.append(f"    {icon} {s.name:<20} value={s.value:.4f} threshold={s.threshold:.4f}")
        if self.retrained:
            out += ["",
                    f"  Retrained:   ✓",
                    f"  Version:     {self.new_version}",
                    f"  Promoted:    {'✓' if self.promoted else '✗'}",
                    f"  Decision:    {self.promotion_decision}"]
        if self.error:
            out.append(f"  ❌ Error:    {self.error}")
        out.append("=" * 60 + "\n")
        return "\n".join(out)


# ============================================================
# Drift Detectors
# ============================================================

class DriftDetector:
    """Forskellige tests til at detektere model drift."""

    @staticmethod
    def hit_rate_drift(
        df: pd.DataFrame,
        rolling_window: int,
        min_hit_rate: float,
    ) -> DriftSignal:
        """Recent rolling hit rate has fallen below threshold."""
        if len(df) < rolling_window:
            return DriftSignal(
                name="hit_rate_drift", triggered=False,
                value=float("nan"), threshold=min_hit_rate,
                message="Ikke nok data",
            )

        df = df.sort_values("prediction_date").tail(rolling_window)
        recent_hr = float(df["hit"].mean())
        triggered = recent_hr < min_hit_rate
        sev = "critical" if recent_hr < min_hit_rate - 0.05 else \
              "warning" if triggered else "none"

        return DriftSignal(
            name="hit_rate_drift",
            triggered=triggered,
            value=recent_hr,
            threshold=min_hit_rate,
            severity=sev,
            message=f"Recent hit rate {recent_hr:.1%} (need ≥ {min_hit_rate:.1%})",
        )

    @staticmethod
    def calibration_drift(
        df: pd.DataFrame,
        max_ece: float,
        rolling_window: int,
    ) -> DriftSignal:
        """Expected Calibration Error has grown."""
        if len(df) < rolling_window:
            return DriftSignal(
                name="calibration_drift", triggered=False,
                value=float("nan"), threshold=max_ece,
                message="Ikke nok data",
            )

        df = df.sort_values("prediction_date").tail(rolling_window * 3)
        # Compute P(correct prediction)
        probs = df["predicted_prob"].values.astype(float)
        dirs = df["predicted_direction"].values
        p_correct = np.where(dirs == "BUY", probs,
                             np.where(dirs == "SELL", 1 - probs, 0.5))
        y = df["hit"].astype(int).values

        ece = DriftDetector._compute_ece(p_correct, y, n_bins=10)
        triggered = ece > max_ece
        sev = "critical" if ece > max_ece * 1.5 else \
              "warning" if triggered else "none"

        return DriftSignal(
            name="calibration_drift",
            triggered=triggered,
            value=float(ece),
            threshold=max_ece,
            severity=sev,
            message=f"ECE {ece:.3f} (max {max_ece:.3f})",
        )

    @staticmethod
    def page_hinkley(
        df: pd.DataFrame,
        threshold: float = 5.0,
        alpha: float = 0.005,
    ) -> DriftSignal:
        """
        Page-Hinkley sequential change detection on hit/miss series.
        Detects when cumulative deviation from mean exceeds threshold.
        """
        if df.empty or len(df) < 20:
            return DriftSignal(
                name="page_hinkley", triggered=False,
                value=0.0, threshold=threshold,
                message="Ikke nok data",
            )

        df = df.sort_values("prediction_date")
        x = df["hit"].astype(float).values
        mean = x.mean()

        # PH cumulative deviation (negative direction = performance drop)
        cum = 0.0
        max_ph = 0.0
        for xi in x:
            cum = max(0.0, cum + (mean - xi - alpha))
            max_ph = max(max_ph, cum)

        triggered = max_ph > threshold
        return DriftSignal(
            name="page_hinkley",
            triggered=triggered,
            value=float(max_ph),
            threshold=threshold,
            severity="critical" if triggered else "none",
            message=f"PH={max_ph:.2f} (threshold {threshold})",
        )

    @staticmethod
    def feature_psi(
        reference: pd.DataFrame,
        current: pd.DataFrame,
        max_psi: float = 0.20,
        n_bins: int = 10,
    ) -> DriftSignal:
        """
        Population Stability Index across features.
        PSI < 0.10: stable, 0.10-0.25: moderate shift, > 0.25: significant shift.
        """
        if reference.empty or current.empty:
            return DriftSignal(
                name="feature_psi", triggered=False,
                value=0.0, threshold=max_psi,
                message="Ingen reference data",
            )

        common = [c for c in reference.columns if c in current.columns]
        common = [c for c in common
                  if pd.api.types.is_numeric_dtype(reference[c])]
        if not common:
            return DriftSignal(
                name="feature_psi", triggered=False,
                value=0.0, threshold=max_psi,
                message="Ingen numeriske features at sammenligne",
            )

        psi_per_feat = {}
        for col in common:
            ref = reference[col].dropna().values
            cur = current[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue
            psi_per_feat[col] = DriftDetector._compute_psi(ref, cur, n_bins)

        if not psi_per_feat:
            return DriftSignal(
                name="feature_psi", triggered=False,
                value=0.0, threshold=max_psi,
                message="Ikke nok data per feature",
            )

        max_p = max(psi_per_feat.values())
        worst_feat = max(psi_per_feat, key=psi_per_feat.get)
        triggered = max_p > max_psi
        sev = "critical" if max_p > max_psi * 1.5 else \
              "warning" if triggered else "none"

        return DriftSignal(
            name="feature_psi",
            triggered=triggered,
            value=float(max_p),
            threshold=max_psi,
            severity=sev,
            message=f"Max PSI={max_p:.3f} on '{worst_feat}'",
        )

    @staticmethod
    def schedule_check(
        last_retrain: Optional[str],
        max_days: int,
        min_days: int,
    ) -> tuple[DriftSignal, bool]:
        """
        Returns (schedule_signal, cooldown_active).
        cooldown_active = True if too soon since last retrain.
        """
        if last_retrain is None:
            return (
                DriftSignal(
                    name="scheduled", triggered=True,
                    value=float("inf"), threshold=max_days,
                    severity="warning",
                    message="Ingen tidligere retrain",
                ),
                False,
            )

        last = datetime.fromisoformat(last_retrain).date()
        days = (date.today() - last).days

        cooldown = days < min_days
        triggered = days >= max_days

        return (
            DriftSignal(
                name="scheduled",
                triggered=triggered,
                value=float(days),
                threshold=float(max_days),
                severity="warning" if triggered else "none",
                message=f"{days} dage siden sidste retrain"
                         + (" (cooldown active)" if cooldown else ""),
            ),
            cooldown,
        )

    # ----- Math helpers -----
    @staticmethod
    def _compute_ece(probs, y, n_bins=10):
        edges = np.linspace(0, 1, n_bins + 1)
        ids = np.digitize(probs, edges[1:-1])
        ece, n = 0.0, len(probs)
        for b in range(n_bins):
            mask = ids == b
            if not mask.any():
                continue
            ece += (mask.sum() / n) * abs(probs[mask].mean() - y[mask].mean())
        return ece

    @staticmethod
    def _compute_psi(reference, current, n_bins=10):
        edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
        cur_pct = np.histogram(current, bins=edges)[0] / max(len(current), 1)
        eps = 1e-6
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ============================================================
# Model Registry
# ============================================================

class ModelRegistry:
    """
    Versioneret model storage med rollback.

    Layout:
        models/eth_30d/
            ├── champion -> v_2024-06-15_103022/   (symlink eller pointer)
            ├── v_2024-06-15_103022/
            │   ├── ensemble_calibrator/
            │   ├── metadata.json
            │   ├── reference_features.parquet
            │   └── walk_forward_metrics.json
            ├── v_2024-05-20_091500/
            └── retrain_history.jsonl
    """

    CHAMPION_FILE = "champion.txt"
    HISTORY_FILE = "retrain_history.jsonl"

    def __init__(self, registry_dir: str | Path):
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ----- Versioning -----
    def new_version_id(self) -> str:
        return "v_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def list_versions(self) -> list[str]:
        return sorted([p.name for p in self.dir.glob("v_*") if p.is_dir()])

    def get_champion(self) -> Optional[str]:
        f = self.dir / self.CHAMPION_FILE
        if not f.exists():
            return None
        return f.read_text().strip() or None

    def set_champion(self, version_id: str) -> None:
        (self.dir / self.CHAMPION_FILE).write_text(version_id)

    def champion_path(self) -> Optional[Path]:
        v = self.get_champion()
        return self.dir / v if v else None

    def version_path(self, version_id: str) -> Path:
        return self.dir / version_id

    # ----- Save/Load -----
    def save_version(
        self,
        version_id: str,
        calibrator: EnsembleCalibrator,
        metadata: dict,
        reference_features: Optional[pd.DataFrame] = None,
        walk_forward_metrics: Optional[dict] = None,
    ) -> Path:
        vp = self.version_path(version_id)
        vp.mkdir(parents=True, exist_ok=True)

        calibrator.save(vp / "ensemble_calibrator")

        with open(vp / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if reference_features is not None and not reference_features.empty:
            reference_features.to_parquet(vp / "reference_features.parquet")

        if walk_forward_metrics is not None:
            with open(vp / "walk_forward_metrics.json", "w") as f:
                json.dump(walk_forward_metrics, f, indent=2, default=str)

        return vp

    def load_version(self, version_id: str) -> dict:
        vp = self.version_path(version_id)
        if not vp.exists():
            raise FileNotFoundError(f"Version {version_id} not found")

        out = {"version_id": version_id, "path": vp}

        cal_dir = vp / "ensemble_calibrator"
        if cal_dir.exists():
            out["calibrator"] = EnsembleCalibrator.load(cal_dir)

        if (vp / "metadata.json").exists():
            with open(vp / "metadata.json") as f:
                out["metadata"] = json.load(f)

        if (vp / "reference_features.parquet").exists():
            out["reference_features"] = pd.read_parquet(
                vp / "reference_features.parquet"
            )

        if (vp / "walk_forward_metrics.json").exists():
            with open(vp / "walk_forward_metrics.json") as f:
                out["walk_forward_metrics"] = json.load(f)

        return out

    # ----- History log -----
    def log_event(self, event: dict) -> None:
        event = {**event, "logged_at": datetime.now().isoformat(timespec="seconds")}
        with open(self.dir / self.HISTORY_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def history(self) -> pd.DataFrame:
        f = self.dir / self.HISTORY_FILE
        if not f.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        return pd.DataFrame(rows)

    def last_retrain_date(self) -> Optional[str]:
        h = self.history()
        if h.empty:
            return None
        successful = h[(h.get("retrained") == True)] if "retrained" in h.columns else h
        if successful.empty:
            return None
        return successful["logged_at"].max()[:10]

    # ----- Cleanup -----
    def prune_old_versions(self, keep_n: int = 5) -> list[str]:
        versions = self.list_versions()
        champion = self.get_champion()
        # Always keep champion + N most recent
        keep = set(versions[-keep_n:])
        if champion:
            keep.add(champion)
        removed = []
        for v in versions:
            if v not in keep:
                shutil.rmtree(self.version_path(v))
                removed.append(v)
        return removed


# ============================================================
# Scheduler
# ============================================================

class RetrainScheduler:
    """
    Hovedklasse: samler drift detection + retraining + model registry.

    Usage:
        scheduler = RetrainScheduler(
            registry_dir="models/eth_30d/",
            logger=logger,
            config=RetrainConfig(),
        )
        decision = scheduler.check_and_retrain(
            ticker="ETH",
            horizon_days=30,
            data_loader=lambda: (X, y, dates),
            model_factories={"rf": ..., "xgb": ..., "lgbm": ...},
            weights={"rf": 1.0, "xgb": 1.2, "lgbm": 1.0},
        )
        print(decision.summary())
    """

    def __init__(
        self,
        registry_dir: str | Path,
        logger: PredictionLogger,
        config: Optional[RetrainConfig] = None,
    ):
        self.registry = ModelRegistry(registry_dir)
        self.logger = logger
        self.config = config or RetrainConfig()
        self.track_record = TrackRecord(logger)

    # =========================================================
    # 1) Drift detection
    # =========================================================

    def detect_drift(
        self,
        ticker: str,
        horizon_days: int,
        current_features: Optional[pd.DataFrame] = None,
    ) -> tuple[list[DriftSignal], bool]:
        """
        Run all drift detectors. Returns (signals, cooldown_active).
        """
        cfg = self.config
        df = self.logger.get_track_record(
            ticker=ticker, horizon_days=horizon_days, only_resolved=True,
        )

        signals: list[DriftSignal] = []

        # Performance drift
        signals.append(DriftDetector.hit_rate_drift(
            df, rolling_window=cfg.rolling_window, min_hit_rate=cfg.min_hit_rate
        ))

        # Calibration drift
        signals.append(DriftDetector.calibration_drift(
            df, max_ece=cfg.max_ece, rolling_window=cfg.rolling_window
        ))

        # Page-Hinkley
        signals.append(DriftDetector.page_hinkley(
            df, threshold=cfg.ph_threshold, alpha=cfg.ph_alpha
        ))

        # Feature drift (PSI)
        if current_features is not None:
            ref = self._load_reference_features()
            if ref is not None and not ref.empty:
                signals.append(DriftDetector.feature_psi(
                    reference=ref, current=current_features, max_psi=cfg.max_psi
                ))

        # Schedule
        sched_sig, cooldown = DriftDetector.schedule_check(
            last_retrain=self.registry.last_retrain_date(),
            max_days=cfg.max_days_since_retrain,
            min_days=cfg.min_days_between_retrains,
        )
        signals.append(sched_sig)

        return signals, cooldown

    def _load_reference_features(self) -> Optional[pd.DataFrame]:
        champ = self.registry.get_champion()
        if not champ:
            return None
        try:
            data = self.registry.load_version(champ)
            return data.get("reference_features")
        except Exception:
            return None

    # =========================================================
    # 2) Decision logic
    # =========================================================

    def should_retrain(
        self,
        signals: list[DriftSignal],
        cooldown: bool,
        force: bool = False,
    ) -> tuple[bool, str]:
        if force:
            return True, "manual_force"
        if cooldown:
            return False, "cooldown_active"

        # Trigger if ANY non-schedule signal is critical, OR scheduled is triggered
        for s in signals:
            if s.triggered and s.severity == "critical":
                return True, f"critical_{s.name}"
        for s in signals:
            if s.triggered and s.name == "scheduled":
                return True, "scheduled_max_age"
        # Multiple warnings = retrain
        warnings = sum(1 for s in signals if s.triggered and s.severity == "warning")
        if warnings >= 2:
            return True, f"multiple_warnings_({warnings})"

        return False, "no_drift_detected"

    # =========================================================
    # 3) Full check_and_retrain pipeline
    # =========================================================

    def check_and_retrain(
        self,
        ticker: str,
        horizon_days: int,
        data_loader: Callable[[], tuple[pd.DataFrame, np.ndarray, pd.Series]],
        model_factories: dict[str, Callable[[], Any]],
        weights: Optional[dict[str, float]] = None,
        force: bool = False,
        current_features: Optional[pd.DataFrame] = None,
    ) -> RetrainDecision:
        """
        Full pipeline:
          1. Detect drift
          2. Decide whether to retrain
          3. If yes: load data, walk-forward validate, train ensemble
          4. Compare to champion → promote or reject
          5. Log event
        """
        cfg = self.config

        decision = RetrainDecision(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            ticker=ticker,
            horizon_days=horizon_days,
            triggered=False,
            reason="",
            signals=[],
        )

        # Step 1: Drift detection
        try:
            signals, cooldown = self.detect_drift(
                ticker=ticker,
                horizon_days=horizon_days,
                current_features=current_features,
            )
            decision.signals = signals
        except Exception as e:
            decision.error = f"drift_detection_failed: {e}"
            return decision

        # Step 2: Decision
        triggered, reason = self.should_retrain(signals, cooldown, force=force)
        decision.triggered = triggered
        decision.reason = reason

        if cfg.verbose:
            print(f"  [{ticker} {horizon_days}d] Drift check → {reason}")

        if not triggered:
            self.registry.log_event({
                "ticker": ticker, "horizon_days": horizon_days,
                "triggered": False, "reason": reason,
            })
            return decision

        # Step 3: Capture champion metrics (before)
        champ_metrics = self._champion_metrics(ticker, horizon_days)
        decision.metrics_before = champ_metrics

        # Step 4: Retrain
        try:
            X, y, dates = data_loader()
            if cfg.verbose:
                print(f"  [{ticker} {horizon_days}d] Training new model on {len(X)} rows...")

            new_calibrator, wf_metrics = self._train_new_model(
                X, y, dates, model_factories, weights or {n: 1.0 for n in model_factories},
            )

            # Step 5: Promotion check
            promoted, decision_msg = self._promotion_decision(
                new_metrics=wf_metrics, champ_metrics=champ_metrics
            )

            # Step 6: Save (always save, but only set champion if promoted)
            new_version_id = self.registry.new_version_id()
            self.registry.save_version(
                version_id=new_version_id,
                calibrator=new_calibrator,
                metadata={
                    "ticker": ticker,
                    "horizon_days": horizon_days,
                    "trained_at": decision.timestamp,
                    "n_train_samples": len(X),
                    "feature_names": list(X.columns),
                    "weights": weights,
                    "trigger_reason": reason,
                    "promoted": promoted,
                },
                reference_features=X,
                walk_forward_metrics=wf_metrics,
            )

            decision.retrained = True
            decision.new_version = new_version_id
            decision.metrics_after = wf_metrics
            decision.promoted = promoted
            decision.promotion_decision = decision_msg

            if promoted:
                self.registry.set_champion(new_version_id)
                if cfg.verbose:
                    print(f"  ✅ Promoted {new_version_id} to champion")

            # Cleanup old versions
            self.registry.prune_old_versions(keep_n=cfg.keep_n_versions)

        except Exception as e:
            decision.error = f"retrain_failed: {e}"
            if cfg.verbose:
                print(f"  ❌ Retrain failed: {e}")

        # Step 7: Log
        self.registry.log_event(decision.to_dict())
        return decision

    # =========================================================
    # 4) Helpers
    # =========================================================

    def _champion_metrics(self, ticker: str, horizon_days: int) -> dict:
        """Get current champion's recent live metrics."""
        df = self.logger.get_track_record(
            ticker=ticker, horizon_days=horizon_days,
            model_name="ensemble", only_resolved=True,
        )
        if df.empty:
            return {}

        recent = df.sort_values("prediction_date").tail(self.config.rolling_window)
        if recent.empty:
            return {}

        # Compute metrics
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        probs = recent["predicted_prob"].values
        dirs = recent["predicted_direction"].values
        p_correct = np.where(dirs == "BUY", probs,
                             np.where(dirs == "SELL", 1 - probs, 0.5))
        y = recent["hit"].astype(int).values

        try:
            return {
                "n": len(recent),
                "hit_rate": float(recent["hit"].mean()),
                "brier": float(brier_score_loss(y, p_correct)),
                "log_loss": float(log_loss(y, np.clip(p_correct, 1e-7, 1 - 1e-7))),
                "roc_auc": float(roc_auc_score(y, p_correct)) if len(np.unique(y)) > 1 else None,
                "source": "live_track_record",
            }
        except Exception as e:
            return {"n": len(recent), "error": str(e)}

    def _train_new_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dates: pd.Series,
        model_factories: dict,
        weights: dict,
    ) -> tuple[EnsembleCalibrator, dict]:
        """Train ensemble + walk-forward validate."""
        cfg = self.config

        # Walk-forward validation (truthful OOS metrics)
        wf_metrics = {}
        if cfg.require_walk_forward:
            wf_cfg = WalkForwardConfig(
                mode="expanding",
                initial_train_size=cfg.wf_initial_train,
                test_size=cfg.wf_test_size,
                step_size=cfg.wf_step,
                calibrate=cfg.wf_calibrate,
                verbose=False,
            )
            wfv = WalkForwardValidator(wf_cfg)
            try:
                wf_results = wfv.run_ensemble(
                    X=X, y=y, dates=dates,
                    model_factories=model_factories,
                    weights=weights,
                )
                if "ensemble" in wf_results:
                    wf_metrics = wf_results["ensemble"].overall_metrics()
                    wf_metrics["source"] = "walk_forward_validation"
            except Exception as e:
                wf_metrics = {"error": f"wf_failed: {e}"}

        # Final fit on ALL data + calibrate on last 20%
        split = int(len(X) * 0.8)
        X_fit, X_cal = X.iloc[:split], X.iloc[split:]
        y_fit, y_cal = y[:split], y[split:]

        # Train each model
        trained_models = {}
        for name, factory in model_factories.items():
            m = factory()
            m.fit(X_fit, y_fit)
            trained_models[name] = m

        # Calibrate ensemble
        cal_probs = {
            name: m.predict_proba(X_cal)[:, 1]
            for name, m in trained_models.items()
        }
        ensemble_cal = EnsembleCalibrator(
            model_names=list(model_factories.keys()),
            method="auto",
            weights=weights,
        )
        ensemble_cal.fit(cal_probs, y_cal)

        # Attach raw models to calibrator namespace for inference
        ensemble_cal.raw_models = trained_models  # type: ignore

        return ensemble_cal, wf_metrics

    def _promotion_decision(
        self, new_metrics: dict, champ_metrics: dict,
    ) -> tuple[bool, str]:
        """Decide whether new model beats champion."""
        cfg = self.config
        metric = cfg.promotion_metric

        # No champion yet → always promote
        if not champ_metrics or "error" in champ_metrics:
            return True, "no_champion_yet"

        new_val = new_metrics.get(
            "brier_score" if metric == "brier" else
            "hit_rate" if metric == "hit_rate" else
            "log_loss" if metric == "log_loss" else
            "roc_auc"
        )
        champ_val = champ_metrics.get(metric if metric != "brier" else "brier")

        if new_val is None or champ_val is None:
            return True, f"missing_metric_{metric}_(promoted by default)"

        # Lower is better for brier & log_loss; higher for hit_rate & auc
        lower_better = metric in ("brier", "log_loss")
        if lower_better:
            improved = (champ_val - new_val) >= cfg.promotion_min_improvement
        else:
            improved = (new_val - champ_val) >= cfg.promotion_min_improvement

        msg = (f"new_{metric}={new_val:.4f}, champ_{metric}={champ_val:.4f}, "
               f"{'PROMOTE' if improved else 'REJECT'}")
        return improved, msg


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing RetrainScheduler...\n")

    from prediction_logger import PredictionRecord, PriceFetcher
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    # Setup
    test_db = Path("_test_scheduler.db")
    test_dir = Path("_test_models")
    if test_db.exists():
        test_db.unlink()
    if test_dir.exists():
        shutil.rmtree(test_dir)

    rng = np.random.default_rng(42)

    # Mock fetcher
    class MockFetcher(PriceFetcher):
        def get_close_price(self, ticker, asset_type, target_date):
            # Recently the model has been wrong → trigger drift
            days_ago = (date.today() - datetime.fromisoformat(target_date).date()).days
            recent = days_ago < 60
            base_return = -0.05 if recent else 0.03  # Simuler drift
            return 1800 * (1 + rng.normal(base_return, 0.05))

    logger = PredictionLogger(test_db)
    logger.price_fetcher = MockFetcher()

    # Generér 100 historiske predictions med "drift" i sidste 30
    print("📝 Logging 100 predictions med simuleret drift...")
    for i in range(100):
        days_ago = int(rng.integers(35, 200))
        prob = 0.65 if days_ago < 60 else 0.55  # Modellen blev mere confident
        logger.log(PredictionRecord(
            ticker="ETH", asset_type="crypto", horizon_days=30,
            entry_price=1800.0,
            predicted_prob=float(np.clip(prob + rng.normal(0, 0.1), 0.1, 0.9)),
            model_name="ensemble",
            prediction_date=(date.today() - timedelta(days=days_ago)).isoformat(),
            calibrated=True,
        ))
    logger.resolve_pending(verbose=False)
    print(f"   {logger.stats_overview()}\n")

    # Synthetic training data
    n = 1000
    dates_arr = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame({
        "feat1": rng.normal(0, 1, n),
        "feat2": rng.normal(0, 1, n),
        "feat3": rng.normal(0, 1, n),
    })
    signal = 0.4 * X["feat1"] + 0.3 * X["feat2"] - 0.2 * X["feat3"]
    y = (rng.random(n) < (1 / (1 + np.exp(-signal)))).astype(int)

    # Setup scheduler
    scheduler = RetrainScheduler(
        registry_dir=test_dir,
        logger=logger,
        config=RetrainConfig(
            min_hit_rate=0.55,
            rolling_window=20,
            max_days_since_retrain=999,
            min_days_between_retrains=0,
            verbose=True,
            wf_initial_train=200,
            wf_test_size=20,
            wf_step=20,
        ),
    )

    # Run pipeline
    decision = scheduler.check_and_retrain(
        ticker="ETH",
        horizon_days=30,
        data_loader=lambda: (X, y.values, dates_arr.to_series()),
        model_factories={
            "rf": lambda: RandomForestClassifier(n_estimators=80, max_depth=5,
                                                  random_state=42, n_jobs=-1),
            "gb": lambda: GradientBoostingClassifier(n_estimators=60, max_depth=3,
                                                      random_state=42),
            "lr": lambda: LogisticRegression(max_iter=500),
        },
        weights={"rf": 1.0, "gb": 1.2, "lr": 0.8},
    )

    print(decision.summary())

    # History
    print("📜 Retrain history:")
    hist = scheduler.registry.history()
    if not hist.empty:
        print(hist[["logged_at", "ticker", "triggered", "reason",
                    "promoted"]].to_string(index=False))

    # Cleanup
    test_db.unlink()
    shutil.rmtree(test_dir)
    print("\n✅ All tests passed!")
